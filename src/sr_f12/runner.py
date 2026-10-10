"""Real SR-F1.2 worker and GPU qualification, without the retired ENGINE gate.

The launcher deliberately exits at pilot update four and restores in a different
process. Scientific execution uses only the final technical recipe and the frozen
registration. No function in this module evaluates or summarizes TEST.
"""

from __future__ import annotations

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
from sr_f1.contract import digest, file_hash, reward_advantages, score, stable_seed
from sr_f1.data import load_inputs, load_schedule, load_tasks

from .protocol import AMENDMENT_ID, build_config, scientific_matrix, validate_config

PILOT_SEED = 2026101202


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise PermissionError(f"Immutable receipt differs: {path}")
    else:
        atomic_json(path, value, exclusive=True)
    return value


def technical_schedule(root):
    """Eight balanced updates covering all 128 ENGINE questions once."""
    import random

    strata = {}
    for qid, task in load_tasks(root).items():
        if task["pool"] == "ENGINE":
            strata.setdefault((task["family"], task["chart"]), []).append(qid)
    if len(strata) != 8 or any(len(qids) != 16 for qids in strata.values()):
        raise PermissionError("ENGINE pool must have eight strata of sixteen questions")
    for key, qids in strata.items():
        qids.sort()
        random.Random(stable_seed(AMENDMENT_ID, "PILOT_ORDER", *key)).shuffle(qids)
    rows = []
    for step in range(1, 9):
        chosen = [q for key in sorted(strata) for q in strata[key][2 * (step - 1) : 2 * step]]
        for slot, qid in enumerate(chosen):
            rows.append(
                dict(
                    step=step,
                    slot=slot,
                    qid=qid,
                    paired_seed=PILOT_SEED,
                    rollout_seeds=[
                        stable_seed(AMENDMENT_ID, "PILOT", step, qid, i) for i in range(8)
                    ],
                )
            )
    return write_once(Path(root) / "manifests/SRF1_2_TECHNICAL_SCHEDULE.json", rows)


def verify_worker_allocation():
    """Check the actual Slurm allocation, not an environment-provided QoS label."""
    from .protocol import TEACHER_QOS

    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise PermissionError("A Slurm GPU allocation is required")
    result = subprocess.run(
        ["scontrol", "show", "job", job, "--oneliner"], capture_output=True, text=True, check=True
    )
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", result.stdout))
    if fields.get("QOS") != TEACHER_QOS or fields.get("JobState") != "RUNNING":
        raise PermissionError("Worker requires a running teacher-QoS job")
    tres = dict(
        item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item
    )
    if tres.get("gres/gpu") != "1" or tres.get("gres/gpu:pro6000") != "1":
        raise PermissionError("Each SR-F1.2 training path must have one PRO6000")
    return dict(
        job_id=job,
        qos=fields["QOS"],
        alloc_tres=fields["AllocTRES"],
        node=fields.get("NodeList"),
        user=fields.get("UserId"),
    )


def load_real_runtime(plan, root, *, learning_rate=None, task_id="technical"):
    import torch

    from mm_dev.runtime import configure_audited_backend

    from .evaluation import evaluation_account
    from .runtime import SRF12Runtime, configure_training

    root = Path(root)
    identity = read_json(root / "MODEL_ENVIRONMENT_IDENTITY.json")
    if identity["model_revision"] != plan["model"]["revision"] or (
        identity["model_weights_hash"] != plan["model"]["prior_composite_weight_hash"]
    ):
        raise PermissionError("Base model identity differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PermissionError("SR-F1.2 worker requires exactly one visible CUDA GPU")
    hardware = verify_worker_allocation()
    determinism = configure_audited_backend()
    # Fix the adapter initialization independently of arm and seed.
    seed_all(plan["model"]["lora"]["init_seed"])
    runtime = SRF12Runtime(
        identity["model_path"], output_root=root, account=evaluation_account(root, task_id)
    )
    runtime.verify_identity(identity)
    runtime.identity.update(
        base_model_weights_hash=identity["model_weights_hash"],
        determinism=determinism,
        hardware=hardware,
    )
    optimizer, scheduler, modules = configure_training(
        runtime, learning_rate=learning_rate or plan["training"]["lr"]
    )
    return runtime, optimizer, scheduler, modules


def enable_checkpointing(runtime, enabled):
    """Only the registered last-resort case can enable HF checkpointing."""
    import torch

    if enabled:
        if any(
            isinstance(module, torch.nn.Dropout) and module.p != 0
            for module in runtime.model.modules()
        ):
            raise PermissionError("Checkpointing requires every dropout to be zero")
        runtime.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        runtime._sr_f12_checkpointing = True
        runtime.model.train()
    else:
        if hasattr(runtime.model, "gradient_checkpointing_disable"):
            runtime.model.gradient_checkpointing_disable()
        runtime._sr_f12_checkpointing = False
        runtime.model.eval()


def scored(record, task):
    result = score(
        "" if record.get("format_protocol_error") else record["raw_text"],
        task["world"],
        task["query"],
    )
    return result


def durable_group(runtime, root, directory, run_id, slot, row, task, policy_hash):
    """One atomically committed whole group prevents mixed rerolled row fragments."""
    path = Path(directory) / "rollouts" / f"{slot['step']:02d}-{slot['slot']:02d}.json"
    identity = dict(
        run_id=run_id,
        step=slot["step"],
        slot=slot["slot"],
        qid=slot["qid"],
        policy_hash=policy_hash,
        seeds=slot["rollout_seeds"],
        input_hash=digest(row),
    )
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity or saved["sha256"] != digest(saved["records"]):
            raise PermissionError("Durable rollout group identity/content changed")
        records = saved["records"]
    else:
        raw = runtime.generate_group(row, root, seeds=slot["rollout_seeds"])
        if len(raw) != 8:
            raise RuntimeError("Runtime did not return exactly eight responses")
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
        for i, record in enumerate(records):
            if (
                record.get("group_row_index") != i
                or record.get("group_seed") != slot["rollout_seeds"][0]
            ):
                raise PermissionError("Group seed/row identity differs")
            if not record["tokens"] or len(record["tokens"]) != len(record["sampler_logprobs"]):
                raise PermissionError("Missing actual completion token probabilities")
        write_once(path, dict(identity=identity, records=records, sha256=digest(records)))
    return records


def _format_rate(groups):
    scores = [r["score"] for g in groups for r in g]
    # The original strict scorer exposes format_ok; do not substitute answer correctness.
    if any(any(k not in s for k in ("L_json", "L_evidence", "L_answer")) for s in scores):
        raise ValueError("Strict scorer did not expose format_ok")
    return sum(not (s["L_json"] and s["L_evidence"] and s["L_answer"]) for s in scores) / len(
        scores
    )


def execute_path(plan, root, run, *, technical=False, stop_step=None):
    import torch

    from .training import (
        ProbabilityMismatch,
        checkpoint_state,
        commit_checkpoint,
        load_checkpoint,
        update,
    )

    root = Path(root)
    directory = root / ("technical" if technical else "runs") / run["model_id"]
    directory.mkdir(parents=True, exist_ok=True)
    runtime, optimizer, scheduler, modules = load_real_runtime(plan, root, task_id=run["model_id"])
    enable_checkpointing(runtime, plan["training"]["gradient_checkpointing"])
    reference = trainable_state(runtime.model)
    mb = plan["training"]["gradient_microbatch_sequences"]
    schedule = technical_schedule(root) if technical else load_schedule(root, run["seed"])
    questions, tasks = load_inputs(root), load_tasks(root)
    run_identity = {
        **run,
        "config_sha256": digest(plan),
        "training_modules": modules,
        "reference_hash": state_hash(reference),
        "technical_only": technical,
    }
    stream_hash, sampling_hash = (
        digest(schedule),
        digest(
            {
                k: plan["training"][k]
                for k in ("temperature", "top_p", "top_k", "max_new_tokens", "generation_seed")
            }
        ),
    )
    write_once(directory / "RUN_MANIFEST.json", run_identity)
    seed_all(run["seed"])
    checkpoint_dir = directory / "checkpoints"
    if (checkpoint_dir / "LATEST.json").exists():
        state = load_checkpoint(
            checkpoint_dir,
            runtime,
            optimizer,
            scheduler,
            run_identity=run_identity,
            reference=reference,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
            microbatch_size=mb,
            learning_rate=plan["training"]["lr"],
        )
    else:
        state = checkpoint_state(
            runtime,
            optimizer,
            scheduler,
            run_identity=run_identity,
            step=0,
            reference=reference,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
            microbatch_size=mb,
        )
        commit_checkpoint(checkpoint_dir, state)
    start = state["committed_logical_step"]
    if start:
        canonical = directory / "steps" / f"{start:02d}.json"
        if not canonical.exists() or digest(read_json(canonical)) != state["diagnostics_hash"]:
            matches = [
                read_json(p)
                for p in (directory / "update_attempts").glob(f"{start:02d}-*.json")
                if digest(read_json(p)) == state["diagnostics_hash"]
            ]
            if len(matches) != 1:
                raise PermissionError("Committed update diagnostics cannot be recovered")
            atomic_json(canonical, matches[0])
    process = dict(
        pid=os.getpid(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        initial_step=start,
        restored_field_hashes={
            key: state_hash(value)
            for key, value in {
                "parameters": trainable_state(runtime.model),
                "optimizer": _cpu_tree(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(),
                "cursor": state["cursor"],
                "rng": capture_rng(),
            }.items()
        },
        checkpoint_field_hashes={
            key: state_hash(state[key])
            for key in ("parameters", "optimizer", "scheduler", "cursor", "rng")
        },
        time_ns=time.time_ns(),
    )
    write_once(directory / f"PROCESS-{process['time_ns']}-{os.getpid()}.json", process)
    if process["restored_field_hashes"] != process["checkpoint_field_hashes"]:
        raise RuntimeError("Restoration field hashes differ")
    end = run["updates"] if stop_step is None else stop_step
    if not 0 <= start <= end <= run["updates"]:
        raise ValueError("Invalid registered segment")
    if not technical and start in (32, 64, 96):
        publish_adapter(runtime, directory, start, run_identity)
        run_train_fit(runtime, root, directory, run["model_id"], start)
    for step in range(start + 1, end + 1):
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        slots = sorted((s for s in schedule if s["step"] == step), key=lambda s: s["slot"])
        if [s["slot"] for s in slots] != list(range(16)):
            raise PermissionError("Incomplete sixteen-prompt update")
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
            )
            for slot in slots
        ]
        sampling_seconds = time.perf_counter() - begin
        advantages, audit = reward_advantages(
            run["arm"], [[r["score"] for r in g] for g in grouped]
        )
        records = [r for group in grouped for r in group]
        try:
            metrics = update(
                runtime,
                optimizer,
                scheduler,
                records,
                advantages.reshape(-1).tolist(),
                reference=reference,
                microbatch_size=mb,
                save_train_logprobs=technical and step == 1,
            )
        except ProbabilityMismatch as exc:
            write_once(
                directory / f"PROBABILITY_MISMATCH-{step:02d}-{time.time_ns()}.json",
                dict(step=step, status="FAIL", diagnostics=exc.diagnostics, uncommitted=True),
            )
            raise
        except FloatingPointError as exc:
            write_once(
                directory / f"NONFINITE-{step:02d}.json",
                dict(
                    step=step,
                    finite=False,
                    error=str(exc),
                    kind=type(exc).__name__,
                    uncommitted=True,
                ),
            )
            raise
        if technical and step == 1:
            write_once(
                directory / "FIRST_BATCH_TRAIN_LOGPROBS.json",
                dict(records_hash=digest(records), train_logprobs=metrics.pop("train_logprobs")),
            )
        metrics.update(
            step=step,
            arm=run["arm"],
            finite=True,
            format_failure_rate=_format_rate(grouped),
            sampling_seconds=sampling_seconds,
            step_seconds=time.perf_counter() - begin,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            reward_advantage_audit=audit,
            raw_records_sha256=digest(records),
        )
        write_once(directory / "update_attempts" / f"{step:02d}-{time.time_ns()}.json", metrics)
        state = checkpoint_state(
            runtime,
            optimizer,
            scheduler,
            run_identity=run_identity,
            step=step,
            reference=reference,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
            microbatch_size=mb,
            diagnostics_hash=digest(metrics),
            token_path_hash=digest(records),
        )
        # Technical steps are all committed, science every eight, and all lease boundaries.
        if technical or step % 8 == 0:
            commit_checkpoint(checkpoint_dir, state)
        metric_path = directory / "steps" / f"{step:02d}.json"
        if metric_path.exists() and read_json(metric_path) != metrics:
            # A replay after the last durable checkpoint is a technical retry, not a second update.
            metric_path = directory / "replayed_steps" / f"{step:02d}-{time.time_ns()}.json"
        write_once(metric_path, metrics)
        if not technical and step in (32, 64, 96):
            publish_adapter(runtime, directory, step, run_identity)
            run_train_fit(runtime, root, directory, run["model_id"], step)
    if technical and end == 8:
        records = [
            r
            for slot in range(16)
            for r in read_json(directory / "rollouts" / f"01-{slot:02d}.json")["records"]
        ]
        old = read_json(directory / "FIRST_BATCH_TRAIN_LOGPROBS.json")
        if old["records_hash"] != digest(records):
            raise PermissionError("Pilot first-batch records changed")
        differences = []
        for offset in range(0, 128, mb):
            now = runtime.batch_sequence_forward(
                records[offset : offset + mb], purpose="pilot_movement", grad=False
            )
            for values, previous in zip(
                now, old["train_logprobs"][offset : offset + mb], strict=True
            ):
                differences.extend((values.float().cpu() - torch.tensor(previous)).abs().tolist())
        write_once(
            directory / "MOVEMENT.json",
            dict(
                comparison="same_step1_tokens_at_step0_and_step8",
                token_count=len(differences),
                mean_absolute_logprob_movement=sum(differences) / len(differences),
            ),
        )
    status = "COMPLETE" if end == run["updates"] else "CHECKPOINTED"
    result = dict(
        status=status,
        model_id=run["model_id"],
        step=end,
        technical_only=technical,
        checkpoint=read_json(checkpoint_dir / "LATEST.json"),
    )
    write_once(directory / f"SEGMENT-{start:02d}-{end:02d}.json", result)
    if status == "COMPLETE":
        write_once(directory / "COMPLETE.json", result)
    return result


def publish_adapter(runtime, directory, step, identity):
    destination = Path(directory) / "adapters" / f"step-{step:02d}"
    marker = destination / "IDENTITY.json"
    policy_hash = state_hash(trainable_state(runtime.model))
    if marker.exists():
        old = read_json(marker)
        if old["policy_hash"] != policy_hash or any(
            file_hash(destination / name) != value for name, value in old["files"].items()
        ):
            raise PermissionError("Published adapter differs")
        return old
    destination.mkdir(parents=True, exist_ok=True)
    runtime.model.save_pretrained(destination, safe_serialization=True)
    return write_once(
        marker,
        dict(
            step=step,
            run_identity_sha256=digest(identity),
            policy_hash=policy_hash,
            files={p.name: file_hash(p) for p in destination.iterdir() if p.is_file()},
        ),
    )


def run_train_fit(runtime, root, directory, model_id, step):
    """Only the registered TRAIN_FIT panel; no TEST or MONITOR access."""
    from .protocol import iter_model_slots

    root = Path(root)
    tasks, inputs = load_tasks(root), load_inputs(root)
    fit = read_json(root / "manifests/TRAIN_FIT_QIDS.json")
    slots = list(iter_model_slots(tasks, fit, model_id, stage="train_fit", step=step))
    result = []
    for offset in range(0, len(slots), 4):
        group = slots[offset : offset + 4]
        qid = group[0]["qid"]
        path = Path(directory) / "train_fit" / f"step-{step:02d}" / f"{offset // 4:02d}.json"
        if path.exists():
            saved = read_json(path)
            if saved["slots"] != group or saved["sha256"] != digest(saved["records"]):
                raise PermissionError("TRAIN_FIT artifact differs")
        else:
            raw = runtime.generate_group(
                inputs[qid], root, seeds=[s["sampling_seed"] for s in group]
            )
            saved = dict(slots=group, records=[{**r, "score": scored(r, tasks[qid])} for r in raw])
            saved["sha256"] = digest(saved["records"])
            write_once(path, saved)
        result.extend(saved["records"])
    return write_once(
        Path(directory) / "train_fit" / f"step-{step:02d}" / "COMPLETE.json",
        dict(status="COMPLETE", count=len(result), records_sha256=digest(result)),
    )


def preflight(plan, root):
    """Real CUDA baseline, padding, sampler and memory checks; no Adam update."""
    import gc

    import torch

    from .runtime import common_zero_check
    from .training import (
        enforce_probability_gate,
        probability_diagnostics,
        select_microbatch,
        sequence_objective,
    )

    root = Path(root)
    directory = root / "technical/preflight"
    runtime, optimizer, _scheduler, modules = load_real_runtime(plan, root)
    inputs, tasks = load_inputs(root), load_tasks(root)
    rows = technical_schedule(root)[:4]
    records = [
        r
        for slot in rows
        for r in durable_group(
            runtime,
            root,
            directory,
            "PREFLIGHT",
            slot,
            inputs[slot["qid"]],
            tasks[slot["qid"]],
            state_hash(trainable_state(runtime.model)),
        )
    ]
    zero = common_zero_check(runtime, records)
    if zero["status"] != "PASS" or zero["maximum_logprob_difference"] != 0:
        raise RuntimeError("Zero LoRA differs from base")
    reference = trainable_state(runtime.model)
    original_rng = capture_rng()
    checkpoints_mode = []

    def probe(size, checkpointing):
        enable_checkpointing(runtime, checkpointing)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Longest actual completion group is used, rather than a conveniently short row.
        groups = [records[offset : offset + 8] for offset in range(0, 32, 8)]
        longest_group = max(groups, key=lambda g: max(len(r["tokens"]) for r in g))
        batch = sorted(longest_group, key=lambda r: len(r["tokens"]), reverse=True)[:size]
        if checkpointing:
            from .runtime import enable_gradient_checkpointing

            enable_gradient_checkpointing(runtime, batch)
            runtime.model.eval()
            evaluated = runtime.batch_sequence_forward(
                batch, purpose="checkpoint_eval_check", grad=False
            )
            runtime.model.train()
            trained = runtime.batch_sequence_forward(
                batch, purpose="checkpoint_train_check", grad=True
            )
            equal = all(torch.equal(a, b) for a, b in zip(evaluated, trained, strict=True))
            checkpoints_mode.append(dict(train_eval_exact=equal))
            if not equal:
                raise RuntimeError("train/eval probabilities differ for checkpointing fallback")
            del evaluated, trained
        refs = runtime.batch_reference_forward(batch, reference)
        current = runtime.batch_sequence_forward(batch, purpose="memory_probe", grad=True)
        losses = [
            sequence_objective(c, torch.tensor(r["sampler_logprobs"], device=c.device), ref, 1.0)[
                "loss"
            ]
            / 128
            for c, ref, r in zip(current, refs, batch, strict=True)
        ]
        torch.stack(losses).sum().backward()
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in runtime.model.parameters()
        ):
            raise FloatingPointError("Memory probe gradient not finite")
        evidence = dict(
            actual_sequence_lengths=[len(r["tokens"]) for r in batch],
            original_group_row_indices=[r["group_row_index"] for r in batch],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            backward_calls=1,
            optimizer_updates=0,
            train_eval_checks=list(checkpoints_mode),
        )
        optimizer.zero_grad(set_to_none=True)
        del losses, current, refs
        gc.collect()
        return evidence

    selection = select_microbatch(probe)
    if state_hash(reference) != state_hash(trainable_state(runtime.model)):
        raise RuntimeError("Memory probe changed parameters")
    # These checks do not require RNG identity between generate calls, but deterministic
    # teacher forcing must not consume random numbers when all dropout is disabled.
    if state_hash(original_rng) != state_hash(capture_rng()):
        raise RuntimeError("Deterministic training forward consumed RNG")
    differences, sampler_differences, clipped = [], [], []
    size = selection["microbatch_size"]
    for offset in range(0, 16, size):
        batch = records[offset : offset + size]
        batched = runtime.batch_sequence_forward(batch, purpose="batch_single_check", grad=False)
        for record, current in zip(batch, batched, strict=True):
            single = runtime.batch_sequence_forward(
                [record], purpose="batch_single_reference", grad=False
            )[0]
            if current.shape != single.shape:
                raise RuntimeError("Batched token positions do not align with single row")
            differences.extend((current - single).float().abs().cpu().tolist())
            delta = current.float().cpu() - torch.tensor(record["sampler_logprobs"])
            sampler_differences.append(delta.abs())
            clipped.append(delta > math.log(2))
    comparison = dict(
        answer_count=16,
        token_count=len(differences),
        mean_absolute_difference=sum(differences) / len(differences),
        maximum_absolute_difference=max(differences),
    )
    if (
        comparison["mean_absolute_difference"] > 0.002
        or comparison["maximum_absolute_difference"] > 0.05
    ):
        raise RuntimeError("Batch/single teacher-forcing exceeds registered thresholds")
    sampler = probability_diagnostics(sampler_differences, clipped)
    enforce_probability_gate(sampler)
    # Exercise the actual answer-only path and retain row-specific stop metadata.
    answer_records = runtime.generate_group(
        inputs[rows[0]["qid"]], root, seeds=rows[0]["rollout_seeds"], protocol="answer_only"
    )
    answer = dict(
        count=len(answer_records),
        protocol="answer_only",
        cross_question_batching=False,
        assistant_prefills=[
            r.get("input_routing", {}).get("assistant_prefill") for r in answer_records
        ],
        finish_reasons=[r["finish_reason"] for r in answer_records],
        records_sha256=digest(answer_records),
    )
    write_once(directory / "ANSWER_ONLY_RECORDS.json", answer_records)
    if answer["count"] != 8 or answer["assistant_prefills"] != ["{"] * 8:
        raise RuntimeError("Answer-only prefill not exercised on all rows")
    result = dict(
        status="PASS",
        config_sha256=digest(plan),
        zero_lora=zero,
        batch_single=comparison,
        sampler_training=sampler,
        microbatch_selection=selection,
        lora_modules=modules,
        evaluation_engine=answer,
        gpu_name=torch.cuda.get_device_name(0),
        cuda_count=torch.cuda.device_count(),
        pid=os.getpid(),
        job_id=os.environ.get("SLURM_JOB_ID"),
    )
    return write_once(directory / "PREFLIGHT.json", result)


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_technical_receipt(receipt, plan=None):
    """Fail closed on partial/placeholder qualification, including resume receipts."""
    if receipt.get("status") != "PASS" or receipt.get("version") != AMENDMENT_ID:
        raise PermissionError("Real SR-F1.2 technical qualification is not PASS")
    if plan is not None and receipt.get("config_sha256") != digest(validate_config(plan)):
        raise PermissionError("Technical qualification belongs to another configuration")
    pre = receipt["preflight"]
    zero, batch = pre["zero_lora"], pre["batch_single"]
    if (
        zero.get("answer_count") != 32
        or zero.get("bitwise_equal") is not True
        or zero.get("maximum_logprob_difference") != 0
    ):
        raise PermissionError("32-answer exact zero-LoRA check absent")
    if (
        batch.get("answer_count") != 16
        or not _finite_number(batch.get("mean_absolute_difference"))
        or (
            not _finite_number(batch.get("maximum_absolute_difference"))
            or batch["mean_absolute_difference"] > 0.002
            or batch["maximum_absolute_difference"] > 0.05
        )
    ):
        raise PermissionError("16-answer batch/single check absent or failed")
    if pre.get("cuda_count") != 1 or not pre.get("lora_modules", {}).get("target_modules"):
        raise PermissionError("Actual single-GPU/full-LoRA configuration missing")
    selected = pre["microbatch_selection"]
    if selected.get("microbatch_size") not in (1, 2, 4) or not selected.get("attempts"):
        raise PermissionError("Measured microbatch selection missing")
    if selected["attempts"][-1].get("status") != "PASS":
        raise PermissionError("No successful microbatch probe")
    if selected.get("gradient_checkpointing") and not selected["attempts"][-1]["evidence"].get(
        "train_eval_checks", [{}]
    )[-1].get("train_eval_exact"):
        raise PermissionError("Gradient checkpointing train/eval validation missing")
    evaluation = pre["evaluation_engine"]
    if (
        evaluation.get("count") != 8
        or evaluation.get("assistant_prefills") != ["{"] * 8
        or evaluation.get("cross_question_batching") is not False
    ):
        raise PermissionError("Answer-only/batching engine check missing")
    attempts = receipt.get("pilots", [])
    if not 1 <= len(attempts) <= 2:
        raise PermissionError("One or at most two preregistered pilots required")
    from .training import enforce_probability_gate, select_pilot_learning_rate

    enforce_probability_gate(pre["sampler_training"])
    if receipt.get("selected_microbatch") != selected["microbatch_size"] or (
        receipt.get("gradient_checkpointing") != selected["gradient_checkpointing"]
    ):
        raise PermissionError("Selected memory recipe differs from probe")
    if plan is not None and (
        receipt["selected_learning_rate"] != plan["training"]["lr"]
        or receipt["selected_microbatch"] != plan["training"]["gradient_microbatch_sequences"]
        or receipt["gradient_checkpointing"] != plan["training"]["gradient_checkpointing"]
    ):
        raise PermissionError("Technical recipe differs from final configuration")
    for index, pilot in enumerate(attempts):
        if index == 0 and pilot.get("learning_rate") != 5e-5:
            raise PermissionError("Pilot did not begin at 5e-5")
        rows = pilot["metrics"]
        expected_decision = select_pilot_learning_rate(
            rows, pilot["movement"], attempt=index, learning_rate=pilot["learning_rate"]
        )
        if pilot["decision"] != expected_decision:
            raise PermissionError("Pilot learning-rate decision differs from registered rule")
        if pilot["decision"]["status"] == "PASS":
            if len(rows) != 8 or [r["step"] for r in rows] != list(range(1, 9)):
                raise PermissionError("Eight actual committed updates required")
            if not all(
                r.get("finite") is True
                and all(
                    _finite_number(r.get(k))
                    for k in (
                        "loss",
                        "kl",
                        "total_gradient_norm",
                        "sampling_seconds",
                        "training_seconds",
                        "step_seconds",
                        "peak_allocated_bytes",
                    )
                )
                for r in rows
            ):
                raise PermissionError("Real finite metrics/timing/memory required")
            if not any(r.get("parameters_changed") for r in rows) or not rows[-1].get(
                "adam_nonzero_moments"
            ):
                raise PermissionError("No observed parameter/Adam movement")
            restoration = pilot["restoration"]
            if (
                restoration.get("initial_step") != 4
                or restoration.get("same_process") is not False
                or restoration.get("forced_termination_verified") is not True
                or (
                    restoration.get("restored_field_hashes")
                    != restoration.get("checkpoint_field_hashes")
                )
                or set(restoration.get("restored_field_hashes", {}))
                != {"parameters", "optimizer", "scheduler", "cursor", "rng"}
            ):
                raise PermissionError("Independent-process step-four restoration was not verified")
        for row in rows:
            if row.get("finite") is False:
                continue
            for key in (
                "sampler_train_mean_absolute_difference",
                "sampler_train_p99_absolute_difference",
                "sampler_train_max_absolute_difference",
                "tis_truncated_fraction",
            ):
                if not _finite_number(row.get(key)):
                    raise PermissionError("Sampler/trainer actual diagnostics missing")
            if (
                row["sampler_train_mean_absolute_difference"] > 0.01
                or row["tis_truncated_fraction"] > 0.01
            ):
                raise PermissionError("Unresolved sampler/trainer numerical mismatch")
            if row.get("clip_fraction") != 0 or row.get("optimizer_updates") != 1:
                raise PermissionError("One-update/ratio invariant failed")
    final = attempts[-1]
    if (
        final["decision"]["status"] != "PASS"
        or receipt.get("selected_learning_rate") != final["learning_rate"]
    ):
        raise PermissionError("Pilot selection incomplete or unstable")
    if len(attempts) == 2 and (
        attempts[0]["decision"]["status"] != "RETRY_ONCE"
        or final["learning_rate"] != attempts[0]["decision"]["learning_rate"]
    ):
        raise PermissionError("Learning rate changed outside preregistered rule")
    return True


def verify_forced_checkpoint(directory, returncode):
    """Only an intentional SIGKILL after a durable authenticated step four is expected."""
    directory = Path(directory)
    if returncode != -9 or not (directory / "KILL_AFTER_STEP4.json").exists():
        return False
    kill = read_json(directory / "KILL_AFTER_STEP4.json")
    latest = read_json(directory / "checkpoints/LATEST.json")
    commit = read_json(directory / "checkpoints/commit-04.json")
    if (
        kill.get("step") != 4
        or kill.get("signal") != "SIGKILL"
        or kill.get("checkpoint") != latest
        or commit != latest
    ):
        raise PermissionError("Forced termination is not bound to the committed step four")
    target = (directory / "checkpoints" / latest["path"]).resolve()
    if (
        target.parent != (directory / "checkpoints").resolve()
        or file_hash(target) != latest["sha256"]
    ):
        raise PermissionError("Forced termination checkpoint bytes differ")
    processes = [read_json(p) for p in directory.glob("PROCESS-*.json")]
    if not any(
        p.get("pid") == kill.get("pid") and p.get("initial_step", 99) < 4 for p in processes
    ):
        raise PermissionError("Forced termination process identity absent")
    segments = [read_json(p) for p in directory.glob("SEGMENT-*-04.json")]
    if not any(
        s.get("status") == "CHECKPOINTED"
        and s.get("technical_only") is True
        and s.get("model_id") == directory.name
        and s.get("step") == 4
        and s.get("checkpoint") == latest
        for s in segments
    ):
        raise PermissionError("Forced termination segment completion absent")
    return True


def technical_check(plan, root, *, worker_script=None):
    """Orchestrate fresh GPU processes; step-four checkpoint is never a same-process test."""
    from .training import select_pilot_learning_rate

    root = Path(root).resolve()
    plan = validate_config(plan)
    if plan["training"]["lr"] != 5e-5:
        raise PermissionError(
            "Technical qualification must start at registered initial learning rate"
        )
    worker_script = Path(
        worker_script or Path(__file__).resolve().parents[2] / "scripts/sr_f12/run_worker.py"
    )
    directory = root / "technical"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "TECHNICAL_CHECK_SR_F1_2.json").exists():
        existing = read_json(directory / "TECHNICAL_CHECK_SR_F1_2.json")
        selected_plan = read_json(directory / "SELECTED_CONFIG.json")
        validate_technical_receipt(existing, selected_plan)
        if existing["preflight"]["config_sha256"] != digest(plan):
            raise PermissionError("Completed qualification originated from another configuration")
        return existing
    plan_path = directory / "PREFLIGHT_PLAN.json"
    write_once(plan_path, plan)
    processes = []

    def launch(args, label):
        log = directory / f"{label}-{time.time_ns()}.log"
        command = [sys.executable, str(worker_script), "--run-root", str(root), *args]
        started = time.time()
        with log.open("x") as out:
            result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=False)
        evidence = dict(
            command=command,
            returncode=result.returncode,
            log=str(log.relative_to(root)),
            log_sha256=file_hash(log),
            seconds=time.time() - started,
        )
        processes.append(evidence)
        write_once(log.with_suffix(".json"), evidence)
        return result.returncode

    if (
        not (directory / "preflight/PREFLIGHT.json").exists()
        and launch(["--plan", str(plan_path), "--phase", "preflight"], "preflight") != 0
    ):
        raise RuntimeError("GPU preflight failed; inspect preserved technical process log")
    pre = read_json(directory / "preflight/PREFLIGHT.json")
    if pre.get("config_sha256") != digest(plan) or pre.get("status") != "PASS":
        raise PermissionError("Preflight report differs from registered candidate")
    selection = pre["microbatch_selection"]
    pilots = []
    rate = 5e-5
    for index in range(2):
        trial_plan = build_config(
            learning_rate=rate,
            microbatch_sequences=selection["microbatch_size"],
            gradient_checkpointing=selection["gradient_checkpointing"],
            optional_part=bool(plan["training"].get("optional_arms")),
        )
        trial_path = directory / f"PILOT_PLAN_{index}.json"
        write_once(trial_path, trial_plan)
        model_id = f"SRF1_2_TECHNICAL_R{index}"
        path = directory / model_id
        for stop in (4, 8):
            latest = (
                read_json(path / "checkpoints/LATEST.json")["step"]
                if (path / "checkpoints/LATEST.json").exists()
                else 0
            )
            if latest >= stop and (stop == 4 or (path / "MOVEMENT.json").exists()):
                continue
            exit_code = launch(
                [
                    "--plan",
                    str(trial_path),
                    "--phase",
                    "pilot",
                    "--run-id",
                    model_id,
                    "--stop-step",
                    str(stop),
                    *(["--kill-after-commit"] if stop == 4 else []),
                ],
                f"pilot{index}-to{stop}",
            )
            expected_kill = stop == 4 and verify_forced_checkpoint(path, exit_code)
            if exit_code and not expected_kill:
                if list(path.glob("NONFINITE-*.json")):
                    break
                raise RuntimeError(
                    "Pilot technical failure; inspect preserved logs before recovery"
                )
        rows = [read_json(p) for p in sorted((path / "steps").glob("*.json"))]
        failures = [read_json(p) for p in sorted(path.glob("NONFINITE-*.json"))]
        if failures:
            rows += failures[:1]
        movement = (
            read_json(path / "MOVEMENT.json")["mean_absolute_logprob_movement"]
            if (path / "MOVEMENT.json").exists()
            else None
        )
        decision = select_pilot_learning_rate(rows, movement, attempt=index, learning_rate=rate)
        entries = [read_json(p) for p in path.glob("PROCESS-*.json")]
        kill_marker = (
            read_json(path / "KILL_AFTER_STEP4.json")
            if (path / "KILL_AFTER_STEP4.json").exists()
            else {}
        )
        first = next((p for p in entries if p["pid"] == kill_marker.get("pid")), None)
        restored = next((p for p in entries if p["initial_step"] == 4), None)
        restore = (
            {}
            if restored is None
            else {**restored, "same_process": first is None or first["pid"] == restored["pid"]}
        )
        if restore:
            kill = (
                read_json(path / "KILL_AFTER_STEP4.json")
                if (path / "KILL_AFTER_STEP4.json").exists()
                else {}
            )
            restore["forced_termination_verified"] = (
                kill.get("step") == 4
                and kill.get("signal") == "SIGKILL"
                and first is not None
                and kill.get("pid") == first["pid"]
            )
        pilot = dict(
            learning_rate=rate,
            metrics=rows,
            movement=movement,
            decision=decision,
            restoration=restore,
            committed_step=read_json(path / "checkpoints/LATEST.json")["step"],
        )
        pilots.append(pilot)
        write_once(path / "PILOT_ASSESSMENT.json", pilot)
        if decision["status"] == "RETRY_ONCE":
            rate = decision["learning_rate"]
            continue
        if decision["status"] == "STOP_UNSTABLE":
            failure = dict(
                status="FAIL",
                version=AMENDMENT_ID,
                reason="SECOND_PILOT_UNSTABLE",
                pilots=pilots,
                preflight=pre,
            )
            write_once(directory / "TECHNICAL_CHECK_FAILED.json", failure)
            return failure
        break
    final_plan = build_config(
        learning_rate=rate,
        microbatch_sequences=selection["microbatch_size"],
        gradient_checkpointing=selection["gradient_checkpointing"],
        optional_part=bool(plan["training"].get("optional_arms")),
    )
    receipt = dict(
        status="PASS",
        version=AMENDMENT_ID,
        config_sha256=digest(final_plan),
        preflight=pre,
        pilots=pilots,
        selected_learning_rate=rate,
        selected_microbatch=selection["microbatch_size"],
        gradient_checkpointing=selection["gradient_checkpointing"],
        processes=processes,
        optional_kernel_consideration=any(
            r.get("step_seconds", 0) > 900 for p in pilots for r in p["metrics"]
        ),
        optional_fast_kernels_enabled=False,
    )
    validate_technical_receipt(receipt, final_plan)
    write_once(directory / "SELECTED_CONFIG.json", final_plan)
    write_once(directory / "TECHNICAL_CHECK_SR_F1_2.json", receipt)
    return receipt


def train_scientific(plan, root, run_id):
    root = Path(root)
    plan = validate_config(plan)
    receipt = read_json(root / "TECHNICAL_CHECK_SR_F1_2.json")
    validate_technical_receipt(receipt, plan)
    freeze = read_json(root / "SCIENCE_FREEZE.json")
    if (
        freeze.get("config_sha256") != digest(plan)
        or freeze.get("technical_check_sha256") != digest(receipt)
        or freeze.get("predictions_sha256") != digest(read_json(root / "SEMANTIC_PREDICTIONS.json"))
        or freeze.get("status") != "FROZEN_BEFORE_SCIENCE"
    ):
        raise PermissionError("Science configuration has not been frozen")
    frozen_plan = read_json(root / "config/SR_F1_2_FROZEN.json")
    if frozen_plan != plan:
        raise PermissionError("Worker configuration differs from final science freeze")
    candidates = {
        r["model_id"]: r
        for r in scientific_matrix(optional_part=bool(plan["training"].get("optional_arms")))
    }
    if run_id not in candidates:
        raise PermissionError("Unregistered scientific model identifier")
    return execute_path(plan, root, candidates[run_id])
