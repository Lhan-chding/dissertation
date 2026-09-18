#!/usr/bin/env python3
"""Route the frozen preview collector across two disjoint prompt workers.

This launcher is separately versioned. The imported measurement source and the
PLAN/RNG identities remain unchanged, so committed serial chunks can be resumed.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import time
from pathlib import Path


def assigned_indices(worker_id):
    if type(worker_id) is not int or worker_id not in (0, 1):
        raise ValueError("Exactly two workers, indexed 0 and 1, are supported")
    return list(range(worker_id, 6, 2))


@contextlib.contextmanager
def shared_measurement_lock(root):
    # The serial collector takes LOCK_EX on this same permanent inode.
    with (Path(root) / ".writer.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_anchor_parity(anchor, current, tolerances, parity_values):
    if anchor["runtime_identity"] != current["runtime_identity"]:
        raise ValueError("Frozen measurement runtime differs across workers")
    if len(anchor["scores"]) != len(current["scores"]):
        raise ValueError("Fixed null actions differ across workers")
    token_errors, sequence_errors = [], []
    for before, after in zip(anchor["scores"], current["scores"], strict=True):
        excluded = {"token_logprobs", "sequence_logp"}
        if {k: v for k, v in before.items() if k not in excluded} != {
            k: v for k, v in after.items() if k not in excluded
        }:
            raise ValueError("Fixed null input/policy/token identity differs")
        token_errors.append(
            [
                abs(a - b)
                for a, b in zip(before["token_logprobs"], after["token_logprobs"], strict=True)
            ]
        )
        sequence_errors.append(abs(before["sequence_logp"] - after["sequence_logp"]))
    parity = parity_values(token_errors, tolerances, sequence_errors=sequence_errors)
    if not parity["passed"]:
        raise ValueError("Cross-worker null scores exceed frozen tolerances")
    return parity


def current_null_receipt(gpu, runtime, inputs, worker):
    repeats = set(worker.glob("null_repeat_*.json"))
    resumed = (worker / "null_receipt.json").exists()
    current, _ = gpu._null_receipt(runtime, inputs, worker)
    if resumed:
        new = set(worker.glob("null_repeat_*.json")) - repeats
        if len(new) != 1:
            raise ValueError("Expected one current null repeat receipt")
        current = gpu._read(new.pop())["receipt"]
    return current


def _completed(preview, plan, index):
    gpu = preview.gpu
    path = Path(plan["root"]) / "prompts" / f"{index:02d}" / "COMPLETE.json"
    if not path.exists():
        return None
    result = gpu._read(path)
    if (result["plan_hash"], result["prompt_id"]) != (
        plan["plan_hash"],
        plan["probes"][index]["prompt_id"],
    ):
        raise ValueError("Completed prompt belongs to another frozen measurement")
    gpu.verify_artifact_bindings(result)
    return result


def publish_summary(preview, plan):
    """Short serialized view update; measurement itself never holds this lock."""
    root = Path(plan["root"])
    with (root / ".parallel-summary.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        receipts = [r for i in range(6) if (r := _completed(preview, plan, i)) is not None]
        preview._write_summary(root, plan, receipts)
        all_done = len(receipts) == 6
        if all_done:
            preview.gpu._publish(
                root / "COMPLETE.json",
                {
                    "plan_hash": plan["plan_hash"],
                    "status": "PREVIEW_MEASUREMENTS_COMPLETE",
                    "prompts": receipts,
                    "predictive_evaluation": "NOT_RUN",
                    "scientific_status": "NOT_CERTIFIED",
                    "automatic_successor": False,
                },
            )
        preview.atomic_json(
            root / "LATEST.json",
            {
                "status": "PREVIEW_MEASUREMENTS_COMPLETE" if all_done else "PARALLEL_MEASUREMENT",
                "completed_prompts": len(receipts),
                "total_prompts": 6,
                "completed_prompt_ids": [r["prompt_id"] for r in receipts],
                "workers": 2,
            },
        )
        return all_done


def run(
    plan_path, *, worker_id, execute_gpu=False, resume=False, preview=None, runtime_factory=None
):
    if execute_gpu is not True:
        raise PermissionError("Parallel preview requires --execute-gpu")
    indices = assigned_indices(worker_id)
    if preview is None:
        from src.modeling_v4 import measurement_preview as preview
    from src.modeling_v3.vlm_observation import _parity_values

    gpu = preview.gpu
    plan, tasks, banks, origin = preview._load_plan(plan_path)
    root = Path(plan["root"])
    anchor = gpu._read(root / "null_receipt.json")
    worker = root / "workers" / f"worker_{worker_id}"
    launcher_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    execution = {
        "plan_hash": plan["plan_hash"],
        "measurement_source": plan["measurement_source"],
        "launcher_sha256": launcher_hash,
        "assignments": {"0": [0, 2, 4], "1": [1, 3, 5]},
        "measurement_identity_changed": False,
        "new_adam_updates": 0,
    }
    # No old process may still hold the serial exclusive lock. Other parallel
    # workers may share it; each worker/prompt has a distinct exclusive lock.
    with shared_measurement_lock(root), preview.frozen_writer(worker):
        if (worker / "EXECUTION.json").exists() and not resume:
            raise FileExistsError("Use --resume for an existing parallel worker")
        gpu._publish(root / "PARALLEL_EXECUTION.json", execution)
        gpu._publish(worker / "EXECUTION.json", {**execution, "worker_id": worker_id})
        runtime = None
        for index in indices:
            folder = root / "prompts" / f"{index:02d}"
            with preview.frozen_writer(folder):
                receipt = _completed(preview, plan, index)
                if receipt is None:
                    if runtime is None:
                        runtime = (runtime_factory or gpu.load_runtime)(
                            tasks["config"], tasks["bindings"], device="cuda:0", execute_gpu=True
                        )
                        preview._check_runtime(runtime, origin, banks)
                        runtime["analysis_rules"] = tasks["analysis_rules"]
                        gpu._publish(worker / "RUNTIME.json", runtime["identity"])
                        current = current_null_receipt(gpu, runtime, tasks["inputs"], worker)
                        parity = check_anchor_parity(
                            anchor, current, runtime["parity_tolerances"], _parity_values
                        )
                        gpu._publish(worker / f"anchor_parity_{time.time_ns()}.json", parity)
                        preview.atomic_json(worker / "ANCHOR_PARITY.json", parity)
                        runtime["observation_score_mode"] = plan["observation_score_mode"]
                        runtime["state"].restore(runtime["checkpoint_cache"].load(origin))
                    prompt = plan["probes"][index]
                    preview.atomic_json(
                        worker / "LATEST.json",
                        {
                            "status": "MEASURING_PROMPT",
                            "worker_id": worker_id,
                            "prompt_index": index,
                            "prompt_id": prompt["prompt_id"],
                            "assigned_indices": indices,
                        },
                    )
                    publish_summary(preview, plan)
                    observation_id = f"preview:{plan['plan_hash']}:{plan['source_task_id']}"
                    started = time.perf_counter()
                    response = gpu.collect_response_map(
                        tasks["config"],
                        runtime,
                        {"origin_id": observation_id, "origin_policy": origin, "banks": banks},
                        [prompt],
                        out=folder / "response",
                        draws=plan["draws"],
                        reference_draws=plan["reference_draws"],
                        bridge=True,
                        resume=resume,
                    )
                    gpu._publish(
                        folder / "DIAGNOSTICS.json", gpu.pair_observation_diagnostics(response)
                    )
                    receipt = {
                        "plan_hash": plan["plan_hash"],
                        "prompt_id": prompt["prompt_id"],
                        "source_origin_id": plan["source_task_id"],
                        "observation_origin_id": observation_id,
                        "response": gpu._binding(folder / "response" / "COMPLETE.json"),
                        "diagnostics": gpu._binding(folder / "DIAGNOSTICS.json"),
                        "scope": plan["scope"],
                        "scientific_status": "NOT_CERTIFIED",
                        "elapsed_seconds_this_attempt": time.perf_counter() - started,
                        "new_adam_updates": 0,
                    }
                    gpu._publish(folder / "COMPLETE.json", receipt)
            publish_summary(preview, plan)
        result = {
            "status": "WORKER_ASSIGNED_PROMPTS_COMPLETE",
            "worker_id": worker_id,
            "assigned_indices": indices,
            "plan_hash": plan["plan_hash"],
            "scientific_status": "NOT_CERTIFIED",
            "automatic_successor": False,
        }
        gpu._publish(worker / "COMPLETE.json", result)
        preview.atomic_json(worker / "LATEST.json", result)
        publish_summary(preview, plan)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.execute_gpu:
        parser.error("--execute-gpu is required")
    import sys

    project = Path(args.project_root).resolve()
    sys.path.insert(0, str(project))
    from src.modeling_v4 import measurement_preview as frozen_preview

    if (
        Path(frozen_preview.__file__).resolve()
        != project / "src/modeling_v4/measurement_preview.py"
        or Path(frozen_preview.gpu.__file__).resolve() != project / "src/modeling_v4/gpu_collect.py"
    ):
        raise ValueError("Collector imports are not from the frozen project")
    print(
        run(
            args.plan,
            worker_id=args.worker_id,
            execute_gpu=True,
            resume=args.resume,
            preview=frozen_preview,
        )["status"]
    )


if __name__ == "__main__":
    main()
