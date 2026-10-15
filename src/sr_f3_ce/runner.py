"""Stage workers for immutable SR-F3-CE data, cached updates, and sealed evaluation."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import time
from collections import Counter
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from mm_core.vl_runtime import seed_all
from sr_f12.evaluation import durable_json, shard_lease
from sr_f12.runner import publish_adapter

from .protocol import (
    EXPERIMENT_ID,
    PARENTS,
    digest,
    file_hash,
    now,
    read_json,
    stable_seed,
    verify_run,
)


def append_record(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def accounting(root, task_id):
    def reserve(kind, count, metadata):
        if type(count) is not int or count < 0:
            raise ValueError("Nonnegative physical work count required")
        append_record(
            Path(root) / "accounting" / f"{task_id}.jsonl",
            dict(
                experiment_id=EXPERIMENT_ID,
                at=now(),
                task_id=task_id,
                kind=kind,
                count=count,
                metadata=metadata,
                pid=os.getpid(),
                job_id=os.environ.get("SLURM_JOB_ID"),
            ),
        )

    return reserve


def dataset_and_store(root):
    from .data import ModelInputStore, load_frozen_dataset

    root = Path(root)
    dataset = load_frozen_dataset(root / "plan")
    store = ModelInputStore.from_files(
        root / "plan/manifests/MODEL_INPUTS.jsonl",
        root / "data/IMAGE_MANIFEST.jsonl",
        image_root=root / "data",
    )
    return dataset, store


def load_runtime(root, parent_id, task_id):
    import torch

    from mm_core.execution import object_hash
    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend

    from .runtime import SRF3CERuntime, configure_parent_training
    from .training import fresh_optimizer

    root = Path(root)
    verify_run(root)
    if parent_id not in PARENTS or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise PermissionError("Registered parent and real Slurm GPU allocation required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PermissionError("Each worker must see one actual CUDA GPU")
    prior = read_json(root / "prior/MODEL_ENVIRONMENT_IDENTITY.json")
    audit = read_json(root / "parents/PARENT_AUDIT.json")
    if audit["status"] != "PASS":
        raise PermissionError("Parent identity audit has not passed")
    parent = next(p for p in audit["parents"] if p["parent_id"] == parent_id)
    adapter = root / parent["adapter_path"]
    files = read_json(adapter / "IDENTITY.json")["files"]
    if any(file_hash(adapter / name) != sha for name, sha in files.items()):
        raise PermissionError("Copied parent adapter bytes changed")
    backend = configure_audited_backend()
    hardware = actual_cuda_identity()
    seed_all(stable_seed("initialization", parent_id))
    runtime = SRF3CERuntime(
        prior["model_path"],
        adapter_path=str(adapter),
        output_root=root,
        account=accounting(root, task_id),
    )
    # The new registered generation contract deliberately differs from SR-F2.
    # All base bytes, processor, kernel, library and adapter identities still use
    # the inherited strict verification; generation is separately owned by F3.
    native = read_json(root / "native/PROCESSOR_RECEIPT.json")["native_processor_identity"]
    for key in ("processor_hash", "tokenizer_hash", "chat_template_hash"):
        if runtime.identity[key] != native[key] or native[key] != prior[key]:
            raise PermissionError("Loaded GPU processor differs from CPU acceptance: " + key)
    actual_freeze = dict(prior, generation_config_expanded=native["generation_config_expanded"])
    common = dict(adapter_file_hashes=files)
    common["model_hash"] = object_hash(
        dict(base_model_weights_hash=prior["model_weights_hash"], adapter_file_hashes=files)
    )
    runtime.verify_identity(actual_freeze, common)
    modules = configure_parent_training(
        runtime,
        parent_adapter_config=read_json(adapter / "adapter_config.json"),
        expected_adapter_state_hash=parent["policy_hash"],
    )
    if state_hash(trainable_state(runtime.model)) != parent["policy_hash"]:
        raise PermissionError("Loaded parent tensors differ from the committed J6 endpoint")
    optimizer, scheduler = fresh_optimizer(runtime.model)
    runtime.identity.update(
        base_model_weights_hash=prior["model_weights_hash"],
        determinism=backend,
        hardware=hardware,
        parent_id=parent_id,
    )
    durable_json(
        root / "runtime_identity" / f"{task_id}-{os.getpid()}.json",
        dict(
            at=now(),
            pid=os.getpid(),
            identity=runtime.identity,
            modules=modules,
            parent=parent,
            fresh_optimizer_state_entries=len(optimizer.state),
        ),
    )
    return runtime, optimizer, scheduler


def evaluate_panel(runtime, root, store, tasks, slots, directory, *, score):
    from .evaluation import group_slots, run_group, validate_coverage

    root, directory = Path(root), Path(directory)
    source = file_hash(root / "RUN_IDENTITY.json")
    all_rows, scored_rows, hashes = [], [], []
    rng, mode, policy = (
        capture_rng(),
        runtime.model.training,
        state_hash(trainable_state(runtime.model)),
    )
    try:
        with shard_lease(directory):
            for group in group_slots(slots):
                value = run_group(
                    runtime, group, store, tasks, root, directory, source_sha256=source, score=score
                )
                all_rows.extend(value["records"])
                hashes.append(dict(group_id=value["group_id"], sha256=value["raw_sha256"]))
                if score:
                    by_id = {s["slot_id"]: s["score"] for s in value["scores"]}
                    scored_rows.extend(
                        dict(row, score=by_id[row["slot_id"]]) for row in value["records"]
                    )
            coverage = validate_coverage(all_rows, slots)
            if state_hash(trainable_state(runtime.model)) != policy:
                raise RuntimeError("Measurement changed parent/endpoint parameters")
            summary = dict(
                coverage,
                groups=hashes,
                policy_hash=policy,
                records_sha256=digest(hashes),
                rng_isolated=True,
            )
            durable_json(directory / "COVERAGE.json", summary)
        return all_rows, scored_rows, summary
    finally:
        restore_rng(rng)
        runtime.model.train(mode)


def monitor(runtime, root, store, tasks, model_id, step):
    from .evaluation import monitor_slots, summarize_scores

    path = Path(root) / "monitor" / model_id / f"step-{step:02d}"
    _, rows, coverage = evaluate_panel(
        runtime, root, store, tasks, monitor_slots(tasks, model_id, step), path, score=True
    )
    result = dict(coverage, model_id=model_id, step=step, families=summarize_scores(rows))
    durable_json(path / "COMPLETE.json", result)
    return result


def execute_engine_panel(root, block):
    from .evaluation import engine_slots
    from .orchestration import write_task_receipt

    root = Path(root)
    parent_id, task_id = PARENTS[block], f"engine--b{block}"
    dataset, store = dataset_and_store(root)
    runtime, _, _ = load_runtime(root, parent_id, task_id)
    slots = [s for s in engine_slots(dataset.tasks) if s["model_id"] == parent_id]
    directory = root / "engine" / f"b{block}"
    _, scores, coverage = evaluate_panel(
        runtime, root, store, dataset.tasks, slots, directory, score=True
    )
    gates = {}
    for protocol in ("structured", "natural_work", "answer_only"):
        rows = [r for r in scores if r["protocol"] == protocol and r["view"] is None]
        if len(rows) != 256:
            raise RuntimeError("ENGINE protocol denominator differs")
        parseable = sum(r["score"]["value_parseable"] for r in rows)
        gates[protocol] = dict(
            records=256,
            parseable=parseable,
            fraction=parseable / 256,
            status="PASS" if parseable / 256 >= 0.9 else "PROTOCOL_FAILURE",
        )
    durable_json(directory / "PROTOCOL_GATE.json", dict(parent_id=parent_id, protocols=gates))
    if any(g["status"] != "PASS" for g in gates.values()):
        raise RuntimeError(
            "ENGINE protocol parseability below 90%; diagnose interface before science"
        )
    monitor(runtime, root, store, dataset.tasks, parent_id, 0)
    evidence = dict(parent_id=parent_id, diagnostic_rows=coverage["records"], protocol_gate="PASS")
    artifacts = immutable_files(root, directory, root / "monitor" / parent_id / "step-00")
    return write_task_receipt(root, task_id, artifacts=artifacts, evidence=evidence)


def engine_schedule(tasks):
    """Four fixed ENGINE roots per family; never consult a model score."""
    result = []
    for step in range(1, 5):
        rows = sorted(
            (
                t
                for t in tasks.values()
                if t["pool"] == "ENGINE" and t["root_id"].endswith(f"_{step - 1:04d}")
            ),
            key=lambda t: t["qid"],
        )
        if len(rows) != 16 or Counter(t["family"] for t in rows) != dict.fromkeys(
            ("CROSS", "THRESHOLD", "TOPK", "INTERVAL"), 4
        ):
            raise PermissionError("ENGINE fixed stream no longer balances four families")
        support = [t["qid"] for t in rows if t["mode"] == "repair" and t["chart"] == "grouped_bar"]
        if len(support) != 4:
            raise PermissionError("ENGINE support stream differs")
        result.append(dict(step=step, rl_qids=[t["qid"] for t in rows], support_qids=support))
    return result


def durable_training_group(
    runtime, root, store, task, directory, *, run_identity, step, slot, seeds
):
    from .contract import score

    root, directory = Path(root), Path(directory)
    qid = task["qid"]
    model_input = store.model_input(qid)
    identity = dict(
        run_identity_sha256=digest(run_identity),
        step=step,
        slot=slot,
        qid=qid,
        seeds=seeds,
        policy_hash=state_hash(trainable_state(runtime.model)),
        input_hash=digest(model_input),
        task_hash=digest(task),
    )
    path = directory / "rollouts" / f"{step:02d}-{slot:02d}.json"
    request = path.with_suffix(".request.json")
    pending = path.with_suffix(".pending.jsonl")
    durable_json(request, identity)
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity or saved["records_sha256"] != digest(saved["records"]):
            raise PermissionError("Cached rollout identity changed")
        records = saved["records"]
    else:
        if pending.exists():
            raw = [json.loads(line) for line in pending.read_text().splitlines()]
            if len(raw) < 8:
                raw = replay_partial_training(
                    runtime,
                    root,
                    store,
                    model_input,
                    pending,
                    raw,
                    identity=identity,
                    run_identity=run_identity,
                    seeds=seeds,
                )
            elif len(raw) > 8:
                raise PermissionError("Cached group has excess samples")
        else:
            raw = []

            def persist(row):
                append_record(pending, row)
                raw.append(copy.deepcopy(row))

            runtime.generate_group(
                model_input, store.image_root, seeds, protocol="structured", on_completion=persist
            )
        if len(raw) != 8:
            raise RuntimeError("All eight actual student samples are required")
        records = []
        for row in raw:
            if row.get("generation_status") != "COMPLETE" or row.get("technical_validation_errors"):
                raise RuntimeError("Invalid technical trajectory cannot enter an update")
            scored = score(
                row["full_text"],
                task["world"],
                task["query"],
                finish_reason=row["finish_reason"],
                protocol="structured",
            )
            records.append(
                dict(
                    row,
                    model_input=model_input,
                    root=str(store.image_root),
                    qid=qid,
                    score=scored,
                    reward=scored["J"],
                )
            )
        durable_json(path, dict(identity=identity, records=records, records_sha256=digest(records)))
    for index, row in enumerate(records):
        if (
            row.get("group_row_index") != index
            or row.get("group_seed") != seeds[0]
            or not row["tokens"]
            or len(row["tokens"]) != len(row["sampler_logprobs"])
        ):
            raise PermissionError("Rollout token/probability/slot alignment differs")
    return records, file_hash(path)


def _support(runtime, store, tasks, qids, arm, serialization, directory, step):
    from .contract import support_target

    records = []
    for qid in qids:
        task = tasks[qid]
        target = (
            task["teacher"][arm]
            if task["teacher"]
            else support_target(task["world"], task["query"], arm != "GOLD")
        )
        record = runtime.prepare_support(
            store.model_input(qid), store.image_root, target, arm=arm, serialization=serialization
        )
        record["qid"] = qid
        audit = dict(
            qid=qid,
            target=target,
            arm=arm,
            step=step,
            mask_audit=record["mask_audit"],
            tokens=record["tokens"],
            answer_mask=record["answer_mask"],
            work_mask=record["work_mask"],
        )
        durable_json(Path(directory) / "support" / f"{step:02d}-{qid}.json", audit)
        records.append(record)
    return records


def execute_training_segment(root, *, run=None, engine=False, segment="continuous", stop_step=None):
    from .checkpoint import checkpoint_state, commit_checkpoint, load_checkpoint, read_checkpoint
    from .training import update

    root = Path(root)
    _registered, config = verify_run(root, science=not engine)
    dataset, store = dataset_and_store(root)
    if engine:
        run = dict(
            run_id="SRF3CE_ENGINE_EXEC_b0",
            arm="EXEC",
            block=0,
            future_seed=2026101601,
            parent_id=PARENTS[0],
            updates=4,
        )
        schedule = engine_schedule(dataset.tasks)
        directory = root / "engine/resume" / segment
        cache_directory = root / "engine/resume/continuous"
    else:
        if run not in dataset.runs:
            raise PermissionError("Unregistered scientific path")
        schedule = dataset.schedules[run["block"]]
        directory = root / "runs" / run["run_id"]
        cache_directory = directory
    end = run["updates"] if stop_step is None else stop_step
    if end not in ((2, 4) if engine else (64,)):
        raise PermissionError("Unregistered segment stopping point")
    task_id = "engine--resume" if engine else "train--" + run["run_id"]
    with shard_lease(directory):
        runtime, optimizer, scheduler = load_runtime(root, run["parent_id"], task_id)
        reference = trainable_state(runtime.model)
        seed_all(run["future_seed"])
        identity = dict(
            run,
            stage="ENGINE" if engine else "SCIENCE",
            experiment_id=EXPERIMENT_ID,
            run_identity_sha256=file_hash(root / "RUN_IDENTITY.json"),
            reference_hash=state_hash(reference),
        )
        serialization = read_json(root / "native/SUPPORT_MASK_RECEIPT.json")["serialization"]
        sources = dict(
            run_identity=file_hash(root / "RUN_IDENTITY.json"),
            masks=file_hash(root / "native/SUPPORT_MASK_RECEIPT.json"),
            config=file_hash(root / "plan/config/SR_F3_CE.json"),
        )
        kwargs = dict(
            run_identity=identity,
            reference=reference,
            stream_hash=digest(schedule),
            sampling_hash=digest(config["generation"]),
            source_hashes=sources,
        )
        checkpoints = directory / "checkpoints"
        if any(checkpoints.glob("commit-*.json")):
            saved, _ = read_checkpoint(checkpoints)
            chain = verify_committed_records(cache_directory, saved, schedule, dataset.tasks, store)
            state = load_checkpoint(
                checkpoints, runtime, optimizer, scheduler, committed_records_hash=chain, **kwargs
            )
        else:
            state = checkpoint_state(
                runtime,
                optimizer,
                scheduler,
                step=0,
                **kwargs,
                used_task_qids=[],
                used_support_qids=[],
                committed_records_hash=digest([]),
                microbatch_size=4,
                step_statistics={},
            )
            commit_checkpoint(checkpoints, state)
        start = state["committed_logical_step"]
        durable_json(
            directory / f"PROCESS-{time.time_ns()}-{os.getpid()}.json",
            dict(
                pid=os.getpid(),
                restored_step=start,
                state_hash=state_hash(state),
                restored_parameters=state_hash(trainable_state(runtime.model)),
                optimizer_entries=len(optimizer.state),
                rng_hash=state_hash(capture_rng()),
            ),
        )
        mb, task_ids, support_ids = (
            state["microbatch_size"],
            list(state["used_task_qids"]),
            list(state["used_support_qids"]),
        )
        committed_chain = state["committed_records_hash"]
        if not engine:
            restore_monitor_boundary(runtime, root, store, dataset.tasks, run, start, checkpoints)
        for item in schedule[start:end]:
            step = item["step"]
            groups, group_hashes = [], []
            for slot, qid in enumerate(item["rl_qids"]):
                seeds = [
                    stable_seed("engine" if engine else "train", run["future_seed"], step, qid, i)
                    for i in range(8)
                ]
                records, group_hash = durable_training_group(
                    runtime,
                    root,
                    store,
                    dataset.tasks[qid],
                    cache_directory,
                    run_identity=identity,
                    step=step,
                    slot=slot,
                    seeds=seeds,
                )
                groups.extend(records)
                group_hashes.append(group_hash)
            support = (
                None
                if run["arm"] == "RL"
                else _support(
                    runtime,
                    store,
                    dataset.tasks,
                    item["support_qids"],
                    run["arm"],
                    serialization,
                    cache_directory,
                    step,
                )
            )
            optimizer_callback = None
            if engine:
                from .engine_recovery import reserve_engine_step, verify_engine_probes

                if segment == "continuous" and step == 1:
                    probe = verify_engine_probes(
                        root, runtime, optimizer, scheduler, groups, reference, support
                    )
                    mb = probe["selected_microbatch_size"]

                def optimizer_callback(meta):
                    return reserve_engine_step(root, dict(meta, segment=segment))

            metrics = update(
                runtime,
                optimizer,
                scheduler,
                groups,
                arm=run["arm"],
                reference=reference,
                support_records=support,
                step=step,
                microbatch_size=mb,
                save_train_logprobs=engine and step == 1,
                optimizer_step_callback=optimizer_callback,
                incident_callback=lambda attempt: append_record(
                    directory / "UPDATE_INCIDENTS.jsonl", attempt
                ),
            )
            mb = metrics["microbatch_size"]
            attempt = directory / "update_attempts" / f"{step:02d}-{time.time_ns()}.json"
            durable_json(attempt, dict(step=step, metrics=metrics))
            task_ids.extend(item["rl_qids"])
            if support is not None:
                support_ids.extend(item["support_qids"])
            # Only scientific content enters this hash; timings/replay costs differ by process.
            committed_chain = digest(
                [
                    committed_chain,
                    step,
                    group_hashes,
                    [
                        file_hash(cache_directory / "support" / f"{step:02d}-{qid}.json")
                        for qid in (item["support_qids"] if support is not None else [])
                    ],
                ]
            )
            if step % (2 if engine else 8) == 0:
                state = checkpoint_state(
                    runtime,
                    optimizer,
                    scheduler,
                    step=step,
                    **kwargs,
                    used_task_qids=task_ids,
                    used_support_qids=support_ids,
                    committed_records_hash=committed_chain,
                    microbatch_size=mb,
                    step_statistics=metrics,
                )
                commit_checkpoint(checkpoints, state)
            if not engine and step in (16, 32, 48, 64):
                publish_adapter(runtime, directory, step, identity)
                monitor(runtime, root, store, dataset.tasks, run["run_id"], step)
        if engine and end == 4:
            from .engine_recovery import record_engine_final_logprobs

            record_engine_final_logprobs(root, runtime, segment)
        if not engine:
            publish_adapter(runtime, directory, 64, identity)
            monitor(runtime, root, store, dataset.tasks, run["run_id"], 64)
        return dict(
            status="COMPLETED",
            step=end,
            model_id=run["run_id"],
            checkpoint=read_json(checkpoints / "LATEST.json"),
            directory=str(directory),
        )


def immutable_files(root, *directories):
    """Bind raw answers as well as derived summaries in task completion receipts."""
    root = Path(root).resolve(strict=True)
    files = set()
    for directory in directories:
        directory = Path(directory).resolve(strict=True)
        if not directory.is_relative_to(root):
            raise PermissionError("Output directory escapes the run")
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and not path.name.startswith(".")
                and path.suffix in {".json", ".jsonl", ".pt", ".safetensors", ".md"}
            ):
                files.add(str(path.relative_to(root)))
    return sorted(files)


def verify_committed_records(directory, state, schedule, tasks, store):
    """Rebuild the complete checkpoint chain from original cached observations."""
    directory = Path(directory)
    step = state["committed_logical_step"]
    chain, consumed_rl, consumed_support = digest([]), [], []
    identity = state["run_identity"]
    for item in schedule[:step]:
        hashes, support_hashes = [], []
        for slot, qid in enumerate(item["rl_qids"]):
            path = directory / "rollouts" / f"{item['step']:02d}-{slot:02d}.json"
            saved = read_json(path)
            request = read_json(path.with_suffix(".request.json"))
            expected = dict(
                run_identity_sha256=digest(identity),
                step=item["step"],
                slot=slot,
                qid=qid,
                input_hash=digest(store.model_input(qid)),
                task_hash=digest(tasks[qid]),
            )
            if (
                saved["identity"] != request
                or any(request.get(k) != v for k, v in expected.items())
                or len(saved["records"]) != 8
                or saved["records_sha256"] != digest(saved["records"])
                or any(row["qid"] != qid for row in saved["records"])
            ):
                raise PermissionError("Committed original rollout or input identity changed")
            hashes.append(file_hash(path))
        consumed_rl.extend(item["rl_qids"])
        if identity["arm"] != "RL":
            for qid in item["support_qids"]:
                path = directory / "support" / f"{item['step']:02d}-{qid}.json"
                saved = read_json(path)
                if (
                    saved["qid"] != qid
                    or saved["step"] != item["step"]
                    or saved["arm"] != identity["arm"]
                ):
                    raise PermissionError("Committed support identity changed")
                support_hashes.append(file_hash(path))
            consumed_support.extend(item["support_qids"])
        chain = digest([chain, item["step"], hashes, support_hashes])
    if (
        chain != state["committed_records_hash"]
        or consumed_rl != state["used_task_qids"]
        or consumed_support != state["used_support_qids"]
    ):
        raise PermissionError("Committed raw records and checkpoint cursors/hash chain differ")
    return chain


def restore_monitor_boundary(runtime, root, store, tasks, run, step, checkpoints):
    """The just-committed boundary must finish monitoring before any further Adam step."""
    from .checkpoint import read_checkpoint

    for due in (16, 32, 48, 64):
        if due > step:
            continue
        path = Path(root) / "monitor" / run["run_id"] / f"step-{due:02d}"
        receipt = path / "COMPLETE.json"
        if due == step:
            monitor(runtime, root, store, tasks, run["run_id"], due)
            continue
        if not receipt.exists():
            raise PermissionError(
                "Earlier fixed monitor is missing; cannot fill with later weights"
            )
        state, _ = read_checkpoint(checkpoints, step=due)
        if read_json(receipt)["policy_hash"] != state_hash(state["parameters"]):
            raise PermissionError("Fixed monitor does not use its corresponding checkpoint")
        from .evaluation import group_slots, monitor_slots, validate_coverage

        complete = read_json(receipt)
        coverage = read_json(path / "COVERAGE.json")
        slots = monitor_slots(tasks, run["run_id"], due)
        expected_ids = [g["group_id"] for g in group_slots(slots)]
        if (
            coverage.get("records") != 512
            or complete.get("records") != 512
            or coverage.get("groups") != complete.get("groups")
            or [g["group_id"] for g in coverage.get("groups", [])] != expected_ids
            or coverage.get("records_sha256") != digest(coverage["groups"])
            or complete.get("records_sha256") != coverage["records_sha256"]
        ):
            raise PermissionError("Fixed monitor coverage/group manifest differs")
        records = []
        for group in coverage["groups"]:
            raw = path / "groups" / (group["group_id"] + ".json")
            if not raw.is_file() or file_hash(raw) != group["sha256"]:
                raise PermissionError("Committed monitor raw bytes are missing or changed")
            saved = read_json(raw)
            if saved["group_id"] != group["group_id"] or saved["records_sha256"] != digest(
                saved["records"]
            ):
                raise PermissionError("Committed monitor trajectory hash differs")
            records.extend(saved["records"])
        actual = validate_coverage(records, slots)
        if any(coverage.get(k) != v for k, v in actual.items()):
            raise PermissionError("Committed monitor slot coverage differs")


def execute_train(root, run_id):
    from .checkpoint import read_checkpoint
    from .orchestration import write_task_receipt

    root = Path(root)
    dataset, store = dataset_and_store(root)
    run = next((r for r in dataset.runs if r["run_id"] == run_id), None)
    if run is None:
        raise PermissionError("Unknown training path")
    result = execute_training_segment(root, run=run)
    directory = root / "runs" / run_id
    state, marker = read_checkpoint(directory / "checkpoints", step=64)
    verify_committed_records(
        directory, state, dataset.schedules[run["block"]], dataset.tasks, store
    )
    for step in (16, 32, 48, 64):
        if (
            read_json(root / "monitor" / run_id / f"step-{step:02d}" / "COMPLETE.json")["records"]
            != 512
        ):
            raise PermissionError("Incomplete fixed monitor coverage")
    parent_monitor = root / "monitor" / run["parent_id"] / "step-00"
    if read_json(parent_monitor / "COMPLETE.json")["records"] != 512:
        raise PermissionError("Missing shared parent step-zero monitor")
    artifacts = immutable_files(root, directory, root / "monitor" / run_id, parent_monitor)
    evidence = dict(
        updates=64,
        rollout_rows=len(state["used_task_qids"]) * 8,
        support_examples=len(state["used_support_qids"]),
        monitor_steps=[0, 16, 32, 48, 64],
        checkpoint_sha256=marker["sha256"],
    )
    durable_json(
        directory / "COMPLETE.json",
        dict(result, evidence=evidence, policy_hash=state_hash(state["parameters"])),
    )
    artifacts.append(str((directory / "COMPLETE.json").relative_to(root)))
    return write_task_receipt(
        root, "train--" + run_id, artifacts=sorted(set(artifacts)), evidence=evidence
    )


def replay_partial_training(
    runtime, root, store, model_input, pending, original, *, identity, run_identity, seeds
):
    """One explicitly authorized, same-policy, same-seed replay of a partial group."""
    from .evaluation import _write_immutable_bytes

    if run_identity["stage"] != "SCIENCE":
        raise PermissionError("ENGINE cannot silently exceed its fixed new-generation budget")
    root = Path(root)
    entrypoint = root / "recovery/PARTIAL_REPLAY_AUTHORIZATION.json"
    authorization = read_json(entrypoint)
    source = entrypoint
    if "authorization_file" in authorization:
        from .protocol import contained

        source = contained(root, authorization["authorization_file"])
        if (
            not source.is_relative_to(root / "recovery")
            or file_hash(source) != authorization["sha256"]
        ):
            raise PermissionError("Partial replay authorization pointer differs")
        authorization = read_json(source)
    if authorization.get("experiment_id") != EXPERIMENT_ID or authorization.get(
        "run_identity_sha256"
    ) != file_hash(root / "RUN_IDENTITY.json"):
        raise PermissionError("Partial replay authorization has a different run")
    binding = dict(
        task_id="train--" + run_identity["run_id"],
        model_id=run_identity["run_id"],
        group_id=digest(identity),
        pending_sha256=file_hash(pending),
        status="AUTHORIZED",
    )
    if (
        sum(
            all(item.get(k) == v for k, v in binding.items())
            for item in authorization.get("authorizations", [])
        )
        != 1
    ):
        raise PermissionError("Explicit same-group replay authorization missing")
    recovery = pending.parent.parent / "partial_recovery" / digest(identity) / file_hash(source)
    snapshot = recovery / "ORIGINAL.partial.jsonl"
    _write_immutable_bytes(snapshot, pending.read_bytes())
    durable_json(recovery / "AUTHORIZATION.json", authorization)
    attempt = dict(
        binding,
        identity=identity,
        authorization_sha256=file_hash(source),
        policy="same seeds and complete group; exact existing trajectory prefix",
    )
    intent, replay_path = recovery / "ATTEMPT.json", recovery / "REPLAY.jsonl"
    if intent.exists():
        if read_json(intent) != attempt:
            raise PermissionError("Partial replay attempt identity changed")
        replay = (
            [json.loads(line) for line in replay_path.read_text().splitlines()]
            if replay_path.exists()
            else []
        )
        if len(replay) != 8:
            raise RuntimeError(
                "Authorized replay was interrupted; preserve it for a new technical decision"
            )
    else:
        if replay_path.exists():
            raise PermissionError("Replay evidence lacks its pre-generation intent")
        durable_json(intent, attempt)
        replay = []

        def persist(row):
            append_record(replay_path, row)
            replay.append(copy.deepcopy(row))

        runtime.generate_group(
            model_input, store.image_root, seeds, protocol="structured", on_completion=persist
        )
    if len(replay) != 8:
        raise RuntimeError("Authorized replay did not return exactly eight trajectories")
    keys = (
        "tokens",
        "raw_tokens",
        "sampler_logprobs",
        "raw_text",
        "full_text",
        "generated_text",
        "assistant_prefill",
        "finish_reason",
        "truncated",
        "completion_token_count",
        "prompt_tensor_hash",
        "format_protocol_error",
        "boundary_char",
        "boundary_token_count",
        "balanced_cut_char",
        "model_identity",
        "adapter_identity",
        "group_seed",
        "group_row_index",
        "seed",
        "protocol",
    )
    differences = [
        dict(index=i, fields=[key for key in keys if row.get(key) != replay[i].get(key)])
        for i, row in enumerate(original)
    ]
    differences = [row for row in differences if row["fields"]]
    durable_json(
        recovery / "RECEIPT.json",
        dict(
            status="FAIL" if differences else "PASS",
            original_sha256=file_hash(snapshot),
            replay_sha256=file_hash(replay_path),
            differences=differences,
            repeated_existing_sequences=len(original),
            technical_replay_generated_sequences=8,
            sampling_cost_accounting="runtime attempt ledger; explanatory counts are not additive",
            answer_selection=False,
        ),
    )
    if differences:
        raise RuntimeError(
            "Same-policy deterministic replay differs from preserved original trajectories"
        )
    return replay


def verify_resume(root, task_id, previous_job_key):
    from .recovery import verify_resume as verify

    return verify(root, task_id, previous_job_key)
