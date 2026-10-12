"""Isolated SR-F2 workers: committed updates, online measurements, and recovery."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from mm_core.training import _cpu_tree, capture_rng, state_hash, trainable_state
from mm_core.vl_runtime import seed_all
from mm_dev.runtime import atomic_json
from sr_f1.contract import digest, file_hash, stable_seed
from sr_f1.data import load_inputs, load_schedule, load_tasks
from sr_f12.evaluation import shard_lease
from sr_f12.runner import _format_rate, enable_checkpointing, publish_adapter

from . import protocol
from .training import (
    FAMILIES,
    ProbabilityMismatch,
    apply_monitor,
    checkpoint_state,
    commit_checkpoint,
    initial_adaptive_state,
    isolated_rng,
    load_checkpoint,
    update,
)

PILOT_SEED = 2026101301
ARMS = ("EARLY", "J6", "DEC_PRIME", "ADAPT")
RECOVERY_FIELDS = ("parameters", "optimizer", "scheduler", "cursor", "rng", "adaptive_state")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise PermissionError(f"Immutable SR-F2 artifact differs: {path}")
    else:
        atomic_json(path, value, exclusive=True)
    return value


def technical_matrix():
    return protocol.technical_matrix()


def technical_schedule(root):
    """One shared non-scientific ENGINE stream, never a training-seed substitute."""
    import random

    strata = {}
    for qid, task in load_tasks(root).items():
        if task["pool"] == "ENGINE":
            strata.setdefault((task["family"], task["chart"]), []).append(qid)
    if len(strata) != 8 or any(len(ids) != 16 for ids in strata.values()):
        raise PermissionError("ENGINE requires eight balanced strata of sixteen questions")
    for key, ids in strata.items():
        ids.sort()
        random.Random(stable_seed("SR-F2", "ENGINE_ORDER", *key)).shuffle(ids)
    rows = []
    for step in range(1, 9):
        chosen = [qid for key in sorted(strata) for qid in strata[key][2 * (step - 1) : 2 * step]]
        rows.extend(
            dict(
                step=step,
                slot=slot,
                qid=qid,
                paired_seed=PILOT_SEED,
                rollout_seeds=[stable_seed("SR-F2", "ENGINE", step, qid, i) for i in range(8)],
            )
            for slot, qid in enumerate(chosen)
        )
    return write_once(Path(root) / "manifests/SR_F2_TECHNICAL_SCHEDULE.json", rows)


def verify_worker_allocation():
    """A multi-GPU allocation is allowed; each independent worker sees one GPU."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise PermissionError("A running Slurm allocation is required")
    result = subprocess.run(
        ["scontrol", "show", "job", job, "--oneliner"], capture_output=True, text=True, check=True
    )
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", result.stdout))
    tres = dict(
        item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item
    )
    count = int(tres.get("gres/gpu", "0"))
    pro_count = int(tres.get("gres/gpu:pro6000", "0"))
    if fields.get("QOS") != protocol.TEACHER_QOS or fields.get("JobState") != "RUNNING":
        raise PermissionError("Worker requires the running teacher-QoS allocation")
    if not 1 <= count <= 8 or pro_count != count:
        raise PermissionError("Worker requires registered PRO6000 GPUs")
    return dict(
        job_id=job,
        qos=fields["QOS"],
        alloc_tres=fields["AllocTRES"],
        node=fields.get("NodeList"),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def load_real_runtime(plan, root, *, task_id):
    import torch

    from mm_dev.runtime import configure_audited_backend

    from .evaluation import evaluation_account
    from .runtime import SRF2Runtime, configure_training

    protocol.verify_input_files(root)
    identity = read_json(Path(root) / "MODEL_ENVIRONMENT_IDENTITY.json")
    if identity["model_revision"] != plan["model"]["revision"] or (
        identity["model_weights_hash"] != plan["model"]["prior_composite_weight_hash"]
    ):
        raise PermissionError("Base model identity differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PermissionError("Each independent SR-F2 worker must see exactly one CUDA GPU")
    hardware = verify_worker_allocation()
    determinism = configure_audited_backend()
    seed_all(plan["model"]["lora"]["init_seed"])
    runtime = SRF2Runtime(
        identity["model_path"], output_root=Path(root), account=evaluation_account(root, task_id)
    )
    runtime.verify_identity(identity)
    runtime.identity.update(
        base_model_weights_hash=identity["model_weights_hash"],
        determinism=determinism,
        hardware={**runtime.identity.get("hardware", {}), **hardware},
    )
    optimizer, scheduler, modules = configure_training(runtime, learning_rate=5e-5)
    return runtime, optimizer, scheduler, modules


def scored(record, task):
    from .scoring import score_v2

    return score_v2(
        "" if record.get("format_protocol_error") else record["raw_text"],
        task["world"],
        task["query"],
        protocol="evidence_answer",
    )


def durable_group(
    runtime,
    root,
    directory,
    run_id,
    slot,
    row,
    task,
    policy_hash,
    *,
    source_sha256,
    run_identity_sha256,
):
    """Never accept old-arm/scorer/corpus records merely because step and qid match."""
    path = Path(directory) / "rollouts" / f"{slot['step']:02d}-{slot['slot']:02d}.json"
    identity = dict(
        plan_id="SR-F2",
        run_id=run_id,
        run_identity_sha256=run_identity_sha256,
        source_sha256=source_sha256,
        scoring="SR-F2-v2",
        step=slot["step"],
        slot=slot["slot"],
        qid=slot["qid"],
        policy_hash=policy_hash,
        seeds=slot["rollout_seeds"],
        input_hash=digest(row),
        task_hash=digest(task),
    )
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity or saved["sha256"] != digest(saved["records"]):
            raise PermissionError("Durable group identity/content changed")
        records = saved["records"]
    else:
        pending = path.with_suffix(".pending.jsonl")
        request = path.with_suffix(".request.json")
        failure = path.with_suffix(".failure.json")
        write_once(request, identity)
        if failure.exists():
            raise RuntimeError("Failed raw group preserved; explicit technical recovery required")
        if pending.exists():
            try:
                raw = [json.loads(line) for line in pending.read_text().splitlines()]
            except ValueError:
                raise RuntimeError(
                    "Interrupted raw record preserved; no implicit resampling"
                ) from None
            if len(raw) != 8:
                raise RuntimeError("Incomplete durable group preserved; no implicit resampling")
        else:
            raw = []
            pending.touch(exist_ok=False)

            def persist(record):
                if len(raw) >= 8:
                    raise RuntimeError("Runtime emitted excess completion rows")
                retained = copy.deepcopy(record)
                with pending.open("a") as handle:
                    # Invalid NaN/Inf values remain in the raw technical evidence only.
                    handle.write(json.dumps(retained, sort_keys=True, allow_nan=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                raw.append(retained)

            try:
                with isolated_rng(runtime):
                    returned = runtime.generate_group(
                        row, root, seeds=slot["rollout_seeds"], on_completion=persist
                    )
                if len(raw) != 8 or raw != returned:
                    raise RuntimeError("Runtime did not durably deliver every returned completion")
            except BaseException as exc:
                write_once(
                    failure,
                    dict(
                        identity=identity,
                        error=str(exc),
                        kind=type(exc).__name__,
                        raw_path=pending.name,
                        completed_rows=len(raw),
                    ),
                )
                raise
        if any(
            r.get("technical_validation_errors")
            or (r.get("generation_status") not in (None, "COMPLETE"))
            for r in raw
        ):
            raise PermissionError("Technical-invalid raw group cannot enter the training loss")
        records = [
            {
                **r,
                "row": row,
                "root": str(Path(root).resolve()),
                "qid": slot["qid"],
                "score": scored(r, task),
            }
            for r in raw
        ]
    if len(records) != 8:
        raise PermissionError("Exactly eight complete responses required")
    for index, record in enumerate(records):
        if record.get("group_row_index") != index or (
            record.get("group_seed") != slot["rollout_seeds"][0]
        ):
            raise PermissionError("Group seed/row differs")
        if not record["tokens"] or len(record["tokens"]) != len(record["sampler_logprobs"]):
            raise PermissionError("Completion probabilities missing")
        if record["score"] != scored(record, task):
            raise PermissionError("Persisted scores differ from registered scorer v2")
    write_once(path, dict(identity=identity, records=records, sha256=digest(records)))
    return records


def complete_pending_monitor(
    runtime, root, directory, run_id, state, checkpoint, *, source_sha256, monitor_fn=None
):
    """A sidecar completes an already committed boundary without rewriting its state."""
    step = state["monitor_due_step"]
    if step is None:
        return copy.deepcopy(state["adaptive_state"]), None
    from .evaluation import run_monitor

    monitor_fn = monitor_fn or run_monitor
    path = Path(directory) / "monitor_state" / f"step-{step:02d}.json"
    identity = dict(
        model_id=run_id,
        step=step,
        checkpoint_sha256=checkpoint["sha256"],
        checkpoint_state_hash=checkpoint["state_hash"],
        source_sha256=source_sha256,
        prior_adaptive_hash=state["adaptive_state_hash"],
        monitor_definition_hash=state["monitor_definition_hash"],
    )
    policy_hash = state_hash(trainable_state(runtime.model))
    if policy_hash != state_hash(state["parameters"]):
        raise PermissionError("Monitor policy is not the committed policy")
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity or saved["measurement_hash"] != digest(
            saved["measurement"]
        ):
            raise PermissionError("Monitor sidecar identity/content differs")
        measurement = saved["measurement"]
    else:
        started = time.perf_counter()
        with isolated_rng(runtime):
            measurement = monitor_fn(
                runtime, root, directory, run_id, step, source_sha256=source_sha256
            )
        measurement = {**measurement, "monitor_seconds": time.perf_counter() - started}
        saved = None
    if (
        measurement.get("status") != "COMPLETE"
        or measurement.get("step") != step
        or (
            measurement.get("policy_hash") != policy_hash
            or measurement.get("count", measurement.get("records")) != 512
            or not measurement.get("records_sha256")
        )
    ):
        raise PermissionError("Complete fixed 512-response committed-policy monitor required")
    if state_hash(trainable_state(runtime.model)) != policy_hash:
        raise PermissionError("Monitor mutated trainable parameters")
    adaptive = apply_monitor(state["adaptive_state"], measurement)
    receipt = dict(
        identity=identity,
        measurement=measurement,
        measurement_hash=digest(measurement),
        adaptive_state=adaptive,
        adaptive_state_hash=state_hash(adaptive),
    )
    if saved is not None and saved != receipt:
        raise PermissionError("Adaptive sidecar cannot be replayed from its measurement")
    write_once(path, receipt)
    return adaptive, receipt


def _recover_metrics(directory, state):
    step = state["committed_logical_step"]
    if not step:
        return
    path = Path(directory) / "steps" / f"{step:02d}.json"
    expected = state["diagnostics_hash"]
    if path.exists() and digest(read_json(path)) == expected:
        return
    matches = [
        read_json(p)
        for p in (Path(directory) / "update_attempts").glob(f"{step:02d}-*.json")
        if digest(read_json(p)) == expected
    ]
    if len(matches) != 1:
        raise PermissionError("Committed diagnostics cannot be recovered")
    if path.exists():
        write_once(
            Path(directory) / "replayed_steps" / f"{step:02d}-{digest(read_json(path))}.json",
            read_json(path),
        )
    atomic_json(path, matches[0])


def execute_path(
    plan, root, run, *, technical=False, stop_step=None, runtime_factory=None, monitor_fn=None
):
    if run not in (technical_matrix() if technical else protocol.scientific_matrix()):
        raise PermissionError("Unregistered SR-F2 training path")
    protocol.verify_registration(root, require_technical=not technical)
    directory = Path(root) / ("technical" if technical else "runs") / run["model_id"]
    with shard_lease(directory):
        return _execute_path(
            plan,
            root,
            run,
            technical=technical,
            stop_step=stop_step,
            runtime_factory=runtime_factory,
            monitor_fn=monitor_fn,
        )


def _execute_path(
    plan, root, run, *, technical=False, stop_step=None, runtime_factory=None, monitor_fn=None
):
    import torch

    from .advantages import reward_advantages

    root = Path(root)
    plan = protocol.validate_config(plan)
    registration = protocol.verify_registration(root, require_technical=not technical)
    if registration["config_sha256"] != digest(plan):
        raise PermissionError("Worker configuration differs from registration")
    if not technical:
        validate_technical_receipt(read_json(root / "TECHNICAL_CHECK_SR_F2.json"), plan)
    permitted = technical_matrix() if technical else protocol.scientific_matrix()
    if run not in permitted:
        raise PermissionError("Unregistered SR-F2 training path")
    end = run["updates"] if stop_step is None else stop_step
    if (
        type(end) is not int
        or not 0 <= end <= run["updates"]
        or (end % (4 if technical else 8) != 0)
    ):
        raise PermissionError("Stop only at a registered durable segment boundary")
    schedule = technical_schedule(root) if technical else load_schedule(root, run["seed"])
    stream_hash = digest(schedule)
    if not technical:
        input_identity = read_json(root / "INPUT_IDENTITY.json")
        if input_identity["schedule_hashes"][str(run["seed"])] != stream_hash:
            raise PermissionError("Scientific schedule differs from the frozen inherited stream")
    directory = root / ("technical" if technical else "runs") / run["model_id"]
    directory.mkdir(parents=True, exist_ok=True)
    runtime, optimizer, scheduler, modules = (runtime_factory or load_real_runtime)(
        plan, root, task_id=run["model_id"]
    )
    enable_checkpointing(runtime, plan["training"]["gradient_checkpointing"])
    reference = trainable_state(runtime.model)
    questions, tasks = load_inputs(root), load_tasks(root)
    source = registration["source_manifest_sha256"]
    run_identity = {
        **run,
        "config_sha256": digest(plan),
        "source_manifest_sha256": source,
        "training_modules": modules,
        "reference_hash": state_hash(reference),
        "technical_only": technical,
    }
    write_once(directory / "RUN_MANIFEST.json", run_identity)
    sampling_hash = digest(
        {
            key: plan["training"][key]
            for key in ("temperature", "top_p", "top_k", "max_new_tokens", "generation_seed")
        }
    )
    monitor_hash = digest(
        dict(
            source_sha256=source,
            config_sha256=digest(plan),
            schedule="0,8,...96",
            qids="fixed128",
            K=4,
        )
    )
    kwargs = dict(
        run_identity=run_identity,
        reference=reference,
        stream_hash=stream_hash,
        sampling_hash=sampling_hash,
        monitor_definition_hash=monitor_hash,
    )
    seed_all(run["seed"])
    checkpoint_dir = directory / "checkpoints"
    if (checkpoint_dir / "LATEST.json").exists() or any(checkpoint_dir.glob("commit-*.json")):
        state = load_checkpoint(checkpoint_dir, runtime, optimizer, scheduler, **kwargs)
    else:
        state = checkpoint_state(
            runtime, optimizer, scheduler, step=0, adaptive_state=initial_adaptive_state(), **kwargs
        )
        commit_checkpoint(checkpoint_dir, state)
    start = state["committed_logical_step"]
    if hasattr(runtime, "identity"):
        runtime.identity.update(model_id=run["model_id"], step=start)
    if start > end:
        raise PermissionError("Requested segment precedes the committed cursor")
    _recover_metrics(directory, state)
    process = dict(
        pid=os.getpid(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        initial_step=start,
        restored_field_hashes={
            key: state_hash(value)
            for key, value in dict(
                parameters=trainable_state(runtime.model),
                optimizer=_cpu_tree(optimizer.state_dict()),
                scheduler=_cpu_tree(scheduler.state_dict()),
                cursor=state["cursor"],
                rng=capture_rng(),
                adaptive_state=state["adaptive_state"],
            ).items()
        },
        checkpoint_field_hashes={key: state_hash(state[key]) for key in RECOVERY_FIELDS},
        time_ns=time.time_ns(),
    )
    if process["restored_field_hashes"] != process["checkpoint_field_hashes"]:
        raise RuntimeError("Restoration hashes differ")
    write_once(directory / f"PROCESS-{process['time_ns']}-{os.getpid()}.json", process)
    checkpoint = read_json(checkpoint_dir / "LATEST.json")
    adaptive, _monitor_receipt = complete_pending_monitor(
        runtime,
        root,
        directory,
        run["model_id"],
        state,
        checkpoint,
        source_sha256=source,
        monitor_fn=monitor_fn,
    )
    for step in range(start + 1, end + 1):
        if (root / "STOP").exists() and (step - 1) % 8 == 0:
            end = step - 1
            break
        begin = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        slots = sorted((s for s in schedule if s["step"] == step), key=lambda s: s["slot"])
        if [s["slot"] for s in slots] != list(range(16)):
            raise PermissionError("Incomplete sixteen-prompt update")
        families = [tasks[s["qid"]]["family"] for s in slots]
        if any(families.count(family) != 4 for family in FAMILIES):
            raise PermissionError("Each update must contain four groups per family")
        expected_monitor = ((step - 1) // 8) * 8
        if adaptive["last_monitor_step"] != expected_monitor:
            raise PermissionError("Pending boundary monitor must complete before next update")
        policy_hash = state_hash(trainable_state(runtime.model))
        grouped = [
            durable_group(
                runtime,
                root,
                directory,
                run["model_id"],
                slot,
                questions[slot["qid"]],
                tasks[slot["qid"]],
                policy_hash,
                source_sha256=source,
                run_identity_sha256=digest(run_identity),
            )
            for slot in slots
        ]
        sampling_seconds = time.perf_counter() - begin
        advantages, audit = reward_advantages(
            run["arm"],
            [[r["score"] for r in group] for group in grouped],
            step=step,
            families=families,
            active_families=adaptive["active_families"],
        )
        records = [record for group in grouped for record in group]
        try:
            metrics = update(
                runtime,
                optimizer,
                scheduler,
                records,
                advantages.reshape(-1).tolist(),
                reference=reference,
            )
        except (ProbabilityMismatch, FloatingPointError) as exc:
            write_once(
                directory / f"FAILURE-{step:02d}-{time.time_ns()}.json",
                dict(
                    step=step,
                    uncommitted=True,
                    kind=type(exc).__name__,
                    error=str(exc),
                    diagnostics=getattr(exc, "diagnostics", None),
                ),
            )
            raise
        used = (
            sum(f in adaptive["active_families"] for f in families) if run["arm"] == "ADAPT" else 16
        )
        metrics.update(
            step=step,
            arm=run["arm"],
            format_failure_rate=_format_rate(grouped),
            sampling_seconds=sampling_seconds,
            step_seconds=time.perf_counter() - begin,
            peak_allocated_bytes=torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
            reward_advantage_audit=audit,
            advantage_energy=float((advantages**2).sum()),
            semantic_groups_used=used,
            groups_total=16,
            semantic_group_fraction=used / 16,
            active_families=adaptive["active_families"],
            adaptive_state_hash=state_hash(adaptive),
            raw_records_sha256=digest(records),
            input_slots_sha256=digest(slots),
        )
        write_once(directory / "update_attempts" / f"{step:02d}-{time.time_ns()}.json", metrics)
        state = checkpoint_state(
            runtime,
            optimizer,
            scheduler,
            step=step,
            adaptive_state=adaptive,
            diagnostics_hash=digest(metrics),
            token_path_hash=digest(records),
            **kwargs,
        )
        if hasattr(runtime, "identity"):
            runtime.identity["step"] = step
        if technical or step % 8 == 0:
            checkpoint = commit_checkpoint(checkpoint_dir, state)
        canonical = directory / "steps" / f"{step:02d}.json"
        if canonical.exists() and read_json(canonical) != metrics:
            write_once(
                directory / "replayed_steps" / f"{step:02d}-{digest(read_json(canonical))}.json",
                read_json(canonical),
            )
            atomic_json(canonical, metrics)
        else:
            write_once(canonical, metrics)
        if step % 8 == 0:
            adaptive, monitor_receipt = complete_pending_monitor(
                runtime,
                root,
                directory,
                run["model_id"],
                state,
                checkpoint,
                source_sha256=source,
                monitor_fn=monitor_fn,
            )
            write_once(
                directory / "boundary_timings" / f"{step:02d}.json",
                dict(
                    step=step,
                    update_seconds=metrics["step_seconds"],
                    monitor_seconds=monitor_receipt["measurement"]["monitor_seconds"],
                    total_seconds=metrics["step_seconds"]
                    + monitor_receipt["measurement"]["monitor_seconds"],
                ),
            )
            if not technical:
                publish_adapter(runtime, directory, step, run_identity)
    # Recovery after a monitor/adapter interruption finishes publication at the same boundary.
    if not technical and end > 0 and end % 8 == 0:
        publish_adapter(runtime, directory, end, run_identity)
    result = dict(
        status="COMPLETE" if end == run["updates"] else "CHECKPOINTED",
        model_id=run["model_id"],
        step=end,
        technical_only=technical,
        checkpoint=read_json(checkpoint_dir / "LATEST.json"),
        adaptive_state_hash=state_hash(adaptive),
        last_monitor_step=adaptive["last_monitor_step"],
    )
    write_once(directory / f"SEGMENT-{start:02d}-{end:02d}.json", result)
    if result["status"] == "COMPLETE":
        write_once(directory / "COMPLETE.json", result)
    return result


def validate_technical_receipt(receipt, plan=None):
    if receipt.get("status") != "PASS" or receipt.get("plan_id") != "SR-F2":
        raise PermissionError("SR-F2 technical qualification is incomplete")
    if plan is not None and receipt.get("config_sha256") != digest(protocol.validate_config(plan)):
        raise PermissionError("Technical configuration differs")
    pilots = receipt.get("pilots", [])
    if [pilot.get("arm") for pilot in pilots] != list(ARMS):
        raise PermissionError("All four new arms require independent technical pilots")
    for pilot in pilots:
        if (
            pilot.get("seed") != PILOT_SEED
            or pilot.get("learning_rate") != 5e-5
            or (pilot.get("microbatch_size") != 1)
        ):
            raise PermissionError("Technical pilot changed a frozen training setting")
        rows = pilot.get("metrics", [])
        if [r.get("step") for r in rows] != list(range(1, 9)):
            raise PermissionError("Eight complete technical updates required per arm")
        for row in rows:
            fields = (
                "loss",
                "kl",
                "total_gradient_norm",
                "step_seconds",
                "sampling_seconds",
                "training_seconds",
                "advantage_energy",
                "parameter_max_absolute_change",
                "sampler_train_mean_absolute_difference",
                "sampler_train_p99_absolute_difference",
                "sampler_train_max_absolute_difference",
                "tis_truncated_fraction",
            )
            if row.get("finite") is not True or any(
                type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in fields
            ):
                raise PermissionError("Measured finite update/timing/energy diagnostics required")
            if (
                row.get("optimizer_updates") != 1
                or row.get("microbatch_size") != 1
                or (row.get("clip_fraction") != 0)
            ):
                raise PermissionError("Single fixed-loss update invariant failed")
            if row.get("sampler_train_mean_absolute_difference", 1) > 0.01 or (
                row.get("tis_truncated_fraction", 1) > 0.01
            ):
                raise PermissionError("Unresolved sampler/trainer probability mismatch")
        if not any(row.get("parameters_changed") for row in rows) or (
            not rows[-1].get("adam_nonzero_moments")
        ):
            raise PermissionError("No observed trainable/Adam change")
        restore = pilot.get("restoration", {})
        if (
            restore.get("initial_step") != 4
            or restore.get("same_process") is not False
            or (
                restore.get("restored_field_hashes") != restore.get("checkpoint_field_hashes")
                or set(restore.get("restored_field_hashes", {})) != set(RECOVERY_FIELDS)
            )
        ):
            raise PermissionError("Separate-process checkpoint recovery was not verified")
        if pilot.get("monitor_steps") != [0, 8] or pilot.get("monitor_counts") != [512, 512]:
            raise PermissionError("Technical monitor measurements are incomplete")
        complete = pilot.get("completion", {})
        if (
            complete.get("status") != "COMPLETE"
            or complete.get("step") != 8
            or (
                complete.get("technical_only") is not True or complete.get("last_monitor_step") != 8
            )
        ):
            raise PermissionError("Technical path has not committed and measured its endpoint")
        if type(pilot.get("monitor_seconds")) not in (float, int) or (
            not math.isfinite(pilot["monitor_seconds"]) or pilot["monitor_seconds"] < 0
        ):
            raise PermissionError("Technical monitor timing absent")
    return True


def technical_check(plan, root, *, worker_script=None):
    """Four ENGINE pilots; restore step four in fresh processes, never tune the recipe."""
    root = Path(root).resolve()
    plan = protocol.validate_config(plan)
    registration = protocol.verify_registration(root, require_technical=False)
    worker = Path(
        worker_script or Path(__file__).resolve().parents[2] / "scripts/sr_f2/run_worker.py"
    )
    directory = root / "technical"
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "PILOT_CONFIG.json"
    write_once(config_path, plan)
    pilots = []
    for run in technical_matrix():
        path = directory / run["model_id"]
        for stop in (4, 8):
            latest = path / "checkpoints/LATEST.json"
            committed = read_json(latest)["step"] if latest.exists() else 0
            if committed > stop or (
                committed == stop and (stop == 4 or (path / "COMPLETE.json").exists())
            ):
                continue
            log = directory / f"{run['arm']}-{stop}-{time.time_ns()}.log"
            command = [
                sys.executable,
                str(worker),
                "--run-root",
                str(root),
                "--plan",
                str(config_path),
                "--phase",
                "pilot",
                "--run-id",
                run["model_id"],
                "--stop-step",
                str(stop),
            ]
            with log.open("x") as handle:
                result = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, check=False
                )
            write_once(
                log.with_suffix(".json"),
                dict(command=command, returncode=result.returncode, log_sha256=file_hash(log)),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Technical pilot failed; preserved process log contains diagnostics"
                )
        metrics = [read_json(path / "steps" / f"{step:02d}.json") for step in range(1, 9)]
        for step, metric in enumerate(metrics, 1):
            commit = read_json(path / "checkpoints" / f"commit-{step:02d}.json")
            if commit["field_hashes"]["diagnostics_hash"] != state_hash(digest(metric)):
                raise PermissionError("Technical metrics differ from the committed update")
            target = (path / "checkpoints" / commit["path"]).resolve()
            if target.parent != (path / "checkpoints").resolve() or (
                file_hash(target) != commit["sha256"]
            ):
                raise PermissionError("Technical checkpoint bytes differ")
        processes = sorted(
            (read_json(p) for p in path.glob("PROCESS-*.json")), key=lambda p: p["time_ns"]
        )
        before = [p for p in processes if p["initial_step"] == 0]
        after = [p for p in processes if p["initial_step"] == 4]
        if not before or not after:
            raise PermissionError("Technical path was not restored in a new process at step four")
        restoration = dict(after[-1], same_process=before[0]["pid"] == after[-1]["pid"])
        monitors = [read_json(path / "monitor_state" / f"step-{step:02d}.json") for step in (0, 8)]
        pilots.append(
            dict(
                arm=run["arm"],
                seed=PILOT_SEED,
                learning_rate=5e-5,
                microbatch_size=1,
                metrics=metrics,
                restoration=restoration,
                monitor_steps=[m["measurement"]["step"] for m in monitors],
                monitor_counts=[
                    m["measurement"].get("count", m["measurement"].get("records")) for m in monitors
                ],
                monitor_seconds=sum(m["measurement"]["monitor_seconds"] for m in monitors),
                completion=read_json(path / "COMPLETE.json"),
            )
        )
    receipt = dict(
        status="PASS",
        plan_id="SR-F2",
        config_sha256=digest(plan),
        pilots=pilots,
        registration_sha256=digest(registration),
        learning_rate_selection=False,
        scientific_conclusion=False,
    )
    validate_technical_receipt(receipt, plan)
    return write_once(root / "TECHNICAL_CHECK_SR_F2.json", receipt)


def train_scientific(plan, root, run_id):
    protocol.verify_registration(root, require_technical=True)
    candidates = {r["model_id"]: r for r in protocol.scientific_matrix()}
    if run_id not in candidates:
        raise PermissionError("Unregistered scientific model identifier")
    return execute_path(plan, root, candidates[run_id])
