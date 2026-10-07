"""Bounded student training driver; all branches use the same runtime/objective."""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from pathlib import Path

from .sft_loss import encode_example
from .sft_runtime import SFTRuntime


def run_sft(
    backend,
    *,
    focus,
    replay,
    identity,
    seed,
    output_dir,
    microbatch_size=4,
    resume_path=None,
    steps=256,
    checkpoint_callback=None,
):
    from ..modeling_v3.io import canonical_hash

    if steps != 256:
        raise ValueError("Formal SFT endpoint is frozen at 256 updates")
    if seed not in (73101, 73102):
        raise ValueError("Unregistered SFT seed")
    arm = identity.get("arm")
    if not focus and arm not in (
        "SELF_O0",
        "SELF_MIX",
        "SELF_SINGLE",
        "GOLD_MATCH_MIX",
        "REPLAY_ONLY",
    ):
        raise ValueError("GOLD_ALL requires its nonempty registered training view")
    if not focus and arm != "REPLAY_ONLY":
        return {
            "status": "NO_VERIFIED_TARGETS_RETURN_PARENT",
            "identity": identity,
            "optimizer_updates": 0,
            "checkpoints": [],
            "parent": backend.receipt,
        }
    if arm == "REPLAY_ONLY" and focus:
        raise ValueError("REPLAY_ONLY may not contain focus examples")
    encoded_focus = encode_training_rows(backend, focus)
    encoded_replay = encode_training_rows(backend, replay)
    binding = {
        **identity,
        "encoded_focus_hash": canonical_hash([x.__dict__ for x in encoded_focus]),
        "encoded_replay_hash": canonical_hash([x.__dict__ for x in encoded_replay]),
        "seed": seed,
        "steps": 256,
        "loss": "completion_sequence_mean_slots16",
        "parent_checkpoint_sha256": backend.receipt["checkpoint"]["sha256"],
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    validate_resume_directory(output, resume_path)
    runtime = SFTRuntime.from_frozen_backend(backend, seed=seed, view_identity=binding)
    runtime.bind_data(encoded_focus, encoded_replay)
    # Immutable per-process journals preserve work replayed after a crash.
    if resume_path:
        runtime.resume(resume_path)
    if resume_path and runtime.step != int(Path(resume_path).stem[4:]):
        raise ValueError("Resume filename does not match checkpoint update index")
    attempt_id = f"sft_{time.time_ns()}_{uuid.uuid4().hex}"
    attempt = output / "attempts" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    process = {
        "attempt_id": attempt_id,
        "identity": binding,
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "start_update": runtime.step,
        "status": "RUNNING",
    }
    _atomic_json(attempt / "PROCESS.json", process)
    checkpoints, metrics = [], []
    metadata = {
        (role, row["task_id"]): row
        for role, rows in (("focus", focus), ("replay", replay))
        for row in rows
    }
    if runtime.step == 0 and resume_path is None:
        checkpoints.append(runtime.save(output / "step000.pt"))
    while runtime.step < steps:
        next_step = runtime.step + 1
        entry = attempt / f"update{next_step:03d}.json"
        _atomic_json(
            entry,
            {
                "status": "UPDATE_STARTED",
                "attempt_id": attempt_id,
                "update": next_step,
                "physical_cost_complete": False,
            },
        )
        started = time.perf_counter()
        try:
            stats = measured_update(runtime, microbatch_size=microbatch_size)
            stats = annotate_losses(stats, metadata)
            stats.update(
                attempt_id=attempt_id, status="UPDATE_APPLIED", physical_cost_complete=True
            )
            metrics.append(stats)
            _atomic_json(entry, stats)
            # Latest logical row may be replaced on deterministic replay; attempt row persists.
            _atomic_json(output / f"update{runtime.step:03d}.json", stats)
            if runtime.step in (32, 64, 128, 256):
                record = runtime.save(output / f"step{runtime.step:03d}.pt")
                checkpoints.append(record)
                _atomic_json(attempt / f"checkpoint{runtime.step:03d}.json", record)
                if checkpoint_callback:
                    checkpoint_callback(record, binding)
        except BaseException as exc:
            # If update already finished, preserve its measured work alongside publication failure.
            _atomic_json(
                attempt / f"failure{next_step:03d}.json",
                {
                    "status": "FAILED_ATTEMPT",
                    "update": next_step,
                    "wall_seconds": time.perf_counter() - started,
                    "error": repr(exc),
                    "physical_cost_complete": False,
                },
            )
            _atomic_json(attempt / "PROCESS.json", {**process, "status": "FAILED_ATTEMPT"})
            raise
    process_costs = {
        "processed_target_sequences": sum(x["processed_target_sequences"] for x in metrics),
        "processed_target_tokens": sum(x["processed_target_tokens"] for x in metrics),
        "update_wall_seconds": sum(x["update_wall_seconds"] for x in metrics),
        "update_gpu_seconds": sum(x["update_gpu_seconds"] or 0 for x in metrics),
        "completed_updates_this_process": len(metrics),
    }
    _atomic_json(attempt / "PROCESS.json", {**process, "status": "COMPLETE", **process_costs})
    result = {
        "status": "SFT_256_COMPLETE",
        "execution_kind": "REAL_CUDA_SFT",
        "identity": binding,
        "checkpoints": checkpoints,
        "optimizer_updates": runtime.step,
        "exposures": runtime.exposures,
        "metrics_this_process": metrics,
        "process_costs": process_costs,
        "attempt_journal": str(attempt),
        "all_attempts_directory": str(output / "attempts"),
        "physical_cost_scope": "All attempt journals, including incomplete attempts, are required",
        "trainable": [
            {"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
            for n, p in runtime.adapter.model.named_parameters()
            if p.requires_grad
        ],
    }
    (output / "SFT_RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def encode_training_rows(backend, rows):
    result = []
    for row in rows:
        example = encode_example(
            backend.adapter,
            row["prompt"],
            row["target"],
            task_id=row["task_id"],
            data_root=backend.data_root,
        )
        if (
            row.get("target_token_ids") is not None
            and tuple(row["target_token_ids"]) != example.target_ids[:-1]
        ):
            raise ValueError("Training-view target token identity differs")
        if row.get("EOS_id") is not None and row["EOS_id"] != example.target_ids[-1]:
            raise ValueError("Training-view EOS identity differs")
        if row.get("weight", 1) != 1:
            raise ValueError("Training views must have uniform per-task weight")
        result.append(example)
    if len({x.task_id for x in result}) != len(result):
        raise ValueError("Training view contains repeated tasks")
    return result


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_resume_directory(output, resume_path):
    """Reject rollback over later checkpoints before model execution or output mutation."""
    output = Path(output).resolve()
    existing = sorted(output.glob("step*.pt"))
    for path in existing:
        if path.name not in {f"step{k:03d}.pt" for k in (0, 32, 64, 128, 256)}:
            raise ValueError("Unexpected student checkpoint name")
    if existing and resume_path is None:
        raise ValueError("Existing checkpoints require explicit latest-checkpoint resume")
    if resume_path is not None:
        resume = Path(resume_path).resolve()
        if not existing or resume != existing[-1].resolve():
            raise ValueError("Resume must use the latest checkpoint in this student directory")


def annotate_losses(stats, metadata):
    groups = defaultdict(list)
    tokens = defaultdict(int)
    for task_id, role, nll, count in zip(
        stats["task_ids"],
        stats["sequence_roles"],
        stats["sequence_nll"],
        stats["sequence_target_tokens"],
        strict=True,
    ):
        family = metadata[(role, task_id)].get("family", "UNSPECIFIED")
        groups[(role, family)].append(nll)
        tokens[(role, family)] += count
    return {
        **stats,
        "loss_by_role_family": [
            {
                "role": role,
                "family": family,
                "mean_sequence_nll": sum(values) / len(values),
                "objective_contribution": sum(values) / 16,
                "processed_target_sequences": len(values),
                "processed_target_tokens": tokens[(role, family)],
            }
            for (role, family), values in sorted(groups.items())
        ],
    }


def measured_update(runtime, *, microbatch_size):
    import torch

    device = torch.device(runtime.adapter.device)
    start_event = end_event = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event, end_event = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start_event.record()
    began = time.perf_counter()
    result = runtime.update(microbatch_size=microbatch_size)
    if end_event is not None:
        end_event.record()
        torch.cuda.synchronize(device)
    return {
        **result,
        "update_wall_seconds": time.perf_counter() - began,
        "update_gpu_seconds": start_event.elapsed_time(end_event) / 1000 if start_event else None,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if start_event
        else None,
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if start_event
        else None,
        "memory_peak_scope": "process since model loader reset",
    }
