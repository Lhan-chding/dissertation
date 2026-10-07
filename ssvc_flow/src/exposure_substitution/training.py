"""SER-J2 explicit-slot SFT: five fixed batches, every sequence weighted 1/16."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..verified_discovery_transfer.sft_loss import encode_example, forward_batch, per_sequence_nll

MICROBATCH_SLOTS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10), (11,), (12, 13, 14, 15))
CHECKPOINT_STEPS = (0, 64, 128, 192, 256)
SCHEDULE_SEEDS = (108701, 108702, 108703)
ARMS = ("A_LOCAL_C1", "B_FORWARD_C4", "C_FORWARD_C3")


def explicit_backward(adapter, examples):
    """The donor's length never determines a common/replay padding shape."""
    if len(examples) != 16:
        raise ValueError("SER-J2 requires exactly sixteen explicit slots")
    losses, shapes, tokens, total = [], [], 0, 0.0
    for slots in MICROBATCH_SLOTS:
        batch = [examples[i] for i in slots]
        logits, labels = forward_batch(adapter, batch)
        means, _, mask = per_sequence_nll(logits, labels)
        loss = means.sum() / 16
        loss.backward()
        losses.extend(means.detach().cpu().tolist())
        tokens += int(mask.sum())
        total += float(loss.detach())
        shapes.append(list(labels.shape))
    return {
        "loss": total,
        "sequence_nll": losses,
        "target_tokens": tokens,
        "sequence_count": 16,
        "sequence_coefficient": 1 / 16,
        "microbatch_slots": [list(x) for x in MICROBATCH_SLOTS],
        "microbatch_shapes": shapes,
    }


def encode_rows(backend, rows):
    result = {}
    for task_id, row in rows.items():
        if row["task_id"] != task_id or row["role"] not in ("common", "donor", "replay"):
            raise ValueError("Training row identity/role mismatch")
        if row.get("weight", 1) != 1:
            raise ValueError("All sequence weights must remain one")
        result[task_id] = encode_example(
            backend.adapter,
            row["prompt"],
            row["target"],
            task_id=task_id,
            data_root=backend.data_root,
            max_total_tokens=1024,
        )
        example = result[task_id]
        if (
            row.get("target_token_ids") is not None
            and tuple(row["target_token_ids"]) != example.target_ids[:-1]
        ):
            raise ValueError("Target token identity differs")
        if row.get("EOS_id") is not None and row["EOS_id"] != example.target_ids[-1]:
            raise ValueError("Target EOS differs")
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def student_directory(run, parent, block, arm):
    if (
        parent not in ("S96", "REP96")
        or type(block) is not int
        or block not in range(3)
        or arm not in ARMS
    ):
        raise ValueError("Unregistered parent/block/arm")
    return Path(run).resolve() / "training" / f"{parent}_block{block}_{arm}"


def latest_checkpoint(output):
    paths = sorted(Path(output).glob("step*.pt"))
    if any(p.name not in {f"step{s:03d}.pt" for s in CHECKPOINT_STEPS} for p in paths):
        raise ValueError("Unexpected SER-J2 checkpoint filename")
    if paths:
        from .runtime import read_student_checkpoint

        state = read_student_checkpoint(paths[-1])
        if state["step"] != int(paths[-1].stem[4:]):
            raise ValueError("Latest committed checkpoint step/filename mismatch")
    return paths[-1] if paths else None


def measured_update(runtime):
    import torch

    device = torch.device(runtime.adapter.device)
    begin_event = end_event = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        begin_event, end_event = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin_event.record()
    started = time.perf_counter()
    result = runtime.update()
    if end_event is not None:
        end_event.record()
        torch.cuda.synchronize(device)
    return {
        **result,
        "update_wall_seconds": time.perf_counter() - started,
        "update_gpu_seconds": begin_event.elapsed_time(end_event) / 1000 if begin_event else None,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if begin_event
        else None,
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if begin_event
        else None,
    }


@contextmanager
def student_lock(output):
    """OS lock excludes concurrent workers and releases automatically after a crash."""
    import fcntl

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".student.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This student already has an active training process") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_training(runtime, output, *, resume_path=None, checkpoint_callback=None):
    with student_lock(output):
        return _run_training_locked(
            runtime, output, resume_path=resume_path, checkpoint_callback=checkpoint_callback
        )


def _run_training_locked(runtime, output, *, resume_path=None, checkpoint_callback=None):
    """Run the one frozen endpoint; journal replayed and uncertain physical work."""
    from ..modeling_v3.io import sha256_file
    from ..optimizer_fork import state_hash
    from .runtime import read_student_checkpoint

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = latest_checkpoint(output)
    if latest is not None and (
        resume_path is None or Path(resume_path).resolve() != latest.resolve()
    ):
        raise ValueError("Existing student requires latest complete checkpoint resume")
    if resume_path is not None:
        if latest is None:
            raise ValueError("Resume must be inside the bound student directory")
        runtime.resume(resume_path)
        if runtime.step != int(Path(resume_path).stem[4:]):
            raise ValueError("Resume checkpoint filename and completed step differ")
    attempt = output / "attempts" / f"sft_{time.time_ns()}_{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    process = {
        "attempt_id": attempt.name,
        "identity": runtime.identity,
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "start_update": runtime.step,
        "status": "RUNNING",
    }
    atomic_json(attempt / "PROCESS.json", process)
    checkpoint_records, metrics = [], []
    if runtime.step == 0 and resume_path is None:
        checkpoint_records.append(runtime.save(output / "step000.pt"))
    try:
        while runtime.step < 256:
            update = runtime.step + 1
            entry = attempt / f"update{update:03d}.json"
            atomic_json(
                entry,
                {"status": "UPDATE_STARTED", "update": update, "physical_cost_complete": False},
            )
            stats = measured_update(runtime)
            stats.update(
                status="UPDATE_APPLIED", physical_cost_complete=True, attempt_id=attempt.name
            )
            metrics.append(stats)
            atomic_json(entry, stats)
            atomic_json(output / f"update{update:03d}.json", stats)
            if update in CHECKPOINT_STEPS:
                receipt = runtime.save(output / f"step{update:03d}.pt")
                checkpoint_records.append(receipt)
                atomic_json(attempt / f"checkpoint{update:03d}.json", receipt)
                if checkpoint_callback is not None:
                    checkpoint_callback(receipt, runtime.identity)
    except BaseException as exc:
        atomic_json(
            attempt / "FAILURE.json",
            {
                "status": "FAILED_ATTEMPT",
                "error": repr(exc),
                "last_completed_runtime_step": runtime.step,
                "latest_complete_checkpoint": str(latest_checkpoint(output)),
                "physical_cost_complete": False,
            },
        )
        atomic_json(attempt / "PROCESS.json", {**process, "status": "FAILED_ATTEMPT"})
        raise
    costs = {
        "completed_updates_this_process": len(metrics),
        "processed_target_sequences": sum(x["processed_target_sequences"] for x in metrics),
        "processed_target_tokens": sum(x["processed_target_tokens"] for x in metrics),
        "update_wall_seconds": sum(x["update_wall_seconds"] for x in metrics),
        "update_gpu_seconds": sum(x["update_gpu_seconds"] or 0 for x in metrics),
    }
    atomic_json(attempt / "PROCESS.json", {**process, "status": "COMPLETE", **costs})
    cumulative_checkpoints = []
    for step in CHECKPOINT_STEPS:
        path = output / f"step{step:03d}.pt"
        saved = read_student_checkpoint(path)
        if saved["step"] != step or saved["identity"] != runtime.identity:
            raise ValueError("Completed student checkpoint set has inconsistent identity/steps")
        cumulative_checkpoints.append(
            {
                "path": str(path),
                "step": step,
                "sha256": sha256_file(path),
                "state_hash": state_hash(saved),
            }
        )
    result = {
        "status": "SFT_256_COMPLETE",
        "execution_kind": "REAL_CUDA_SFT"
        if str(runtime.adapter.device).startswith("cuda")
        else "CPU_TEST_FIXTURE",
        "identity": runtime.identity,
        "optimizer_updates": runtime.step,
        "exposures": runtime.exposures,
        "role_exposures": runtime.role_exposures,
        "checkpoints": cumulative_checkpoints,
        "checkpoints_this_process": checkpoint_records,
        "process_costs": costs,
        "all_attempts_directory": str(output / "attempts"),
        "physical_cost_scope": (
            "All attempts including interrupted updates; incomplete entries remain uncertain"
        ),
    }
    atomic_json(output / "SFT_RESULT.json", result)
    return result


def require_bridge(run):
    """Shared boundary for every real formal training/evaluation launch."""
    from ..modeling_v3.io import canonical_hash
    from .schema import verify_frozen

    run = Path(run).resolve()
    if (run / "STOP").exists():
        raise RuntimeError("SER-J2 STOP marker prohibits new formal execution")
    bridge_path = run / "bridge" / "BRIDGE_RESULT.json"
    frozen = verify_frozen(run)
    bridge = json.loads(bridge_path.read_text()) if bridge_path.is_file() else {}
    required_bridge_checks = (
        "complete_state_bitwise_equal",
        "loss_slots_counts_equal",
        "actual_parameter_updates_observed",
        "five_microbatches_observed",
        "all_sequence_weights_one_sixteenth",
        "inference_weight_restore_exact",
        "matched_donor_target_eos_equal",
    )
    if (
        bridge.get("status") != "PASS"
        or bridge.get("execution_kind") != "REAL_CUDA_BRIDGE"
        or bridge.get("identity", {}).get("frozen_plan_hash") != canonical_hash(frozen)
        or bridge.get("identity", {}).get("technical_only") is not True
        or bridge.get("physical_updates_started") != 16
        or bridge.get("physical_updates_completed") != 16
        or any(bridge.get("checks", {}).get(key) is not True for key in required_bridge_checks)
    ):
        raise ValueError(
            "A completed independent GPU restore bridge is required before formal execution"
        )
    return bridge


def train(
    run,
    parent,
    block,
    arm,
    machine=None,
    allow_gpu=False,
    *,
    resume_path=None,
    checkpoint_callback=None,
):
    from .runtime import build_runtime, load_parent_backend

    if allow_gpu is not True:
        raise PermissionError("SER-J2 training requires --allow-gpu")
    run = Path(run).resolve()
    require_bridge(run)
    output = student_directory(run, parent, block, arm)
    with student_lock(output):
        backend = load_parent_backend(run, parent, machine=machine, allow_gpu=allow_gpu)
        runtime = build_runtime(run, backend, parent, block, arm)
        return _run_training_locked(
            runtime,
            output,
            resume_path=resume_path or latest_checkpoint(output),
            checkpoint_callback=checkpoint_callback,
        )
