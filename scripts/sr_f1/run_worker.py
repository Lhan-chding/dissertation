#!/usr/bin/env python3
"""Execute one registered SR-F1 task inside its identity-bound Slurm allocation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sr_f1.orchestration import (
    PLAN_ID,
    SCIENCE_TERMINAL,
    TEACHER_QOS,
    SlurmBackend,
    _worker_identity,
    atomic_json,
    attempt_gpu_count,
    baseline_parallel_repair,
    checkpoint_task,
    complete_task,
    compute_parallel_repair,
    digest,
    fail_task,
    file_hash,
    gpu_count,
    has_baseline_parallel_repair,
    multigpu_repair,
    read_json,
    require,
    resource_override,
    verify_multigpu_resources,
    worker_lease,
)


def current_worker_repair(root):
    """Authenticate the newest worker revision; never fall back after a failed check."""
    if (root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json").exists():
        return compute_parallel_repair(root)
    if (root / "BASELINE_PARALLEL_REPAIR.json").exists():
        return baseline_parallel_repair(root)
    if (root / "ENGINE_MULTIGPU_REPAIR.json").exists():
        return multigpu_repair(root)
    return None


def validate_allocation(root, task_id, *, backend=None):
    registration, attempt_id = _worker_identity(root, task_id)
    allocations = read_json(root / "manifests/ALLOCATIONS.json")["allocations"]
    matches = [
        row for row in allocations if row["task_id"] == task_id and row["attempt_id"] == attempt_id
    ]
    require(
        len(matches) == 1 and os.environ.get("SLURM_JOB_ID") == matches[0]["job_id"],
        "WORKER_REQUIRES_REGISTERED_SLURM_ALLOCATION",
    )
    require(
        matches[0]["status"] in {"REGISTERED", "ACTIVE", "UNKNOWN"}, "WORKER_ALLOCATION_NOT_LIVE"
    )
    # sbatch is held until ALLOCATIONS is durably published. A positive live
    # Slurm identity is still required before any model load, including when a
    # previously bound controller observation was lost over SSH.
    backend = backend or SlurmBackend()
    bound = matches[0]
    manifest = read_json(root / bound["attempt_manifest_path"])
    require(
        manifest["attempt_id"] == attempt_id
        and manifest["task_id"] == task_id
        and manifest["registration_hash"] == digest(registration)
        and bound["gpus"] == attempt_gpu_count(root, registration, task_id, manifest)
        and manifest.get("resource_override") == bound.get("resource_override"),
        "WORKER_ATTEMPT_RESOURCE_IDENTITY_MISMATCH",
    )
    expected_gpus = attempt_gpu_count(root, registration, task_id, manifest)
    repair = current_worker_repair(root)
    if repair is not None:
        require(
            file_hash(Path(__file__)) == repair["worker_source_sha256"],
            "LIVE_WORKER_SOURCE_REVISION_CHANGED",
        )
        require(
            manifest.get("resource_override") == resource_override(root, registration, task_id),
            "LIVE_WORKER_RESOURCE_REVISION_MISSING",
        )
    observation = backend._run(["scontrol", "show", "job", bound["job_id"], "--oneliner"])
    require(observation["returncode"] == 0, "LIVE_WORKER_ALLOCATION_UNKNOWN")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", observation["stdout"]))
    permission = registration["permission"]
    require(
        fields.get("JobId") == bound["job_id"]
        and fields.get("JobName") == bound["job_name"]
        and fields.get("Comment") == bound["comment"]
        and fields.get("UserId", "").split("(")[0] == permission["owner"]
        and fields.get("Account") == permission["account"]
        and fields.get("QOS") == permission["qos"]
        and fields.get("JobState") == "RUNNING"
        and "AllocTRES" in fields
        and gpu_count(fields["AllocTRES"]) == expected_gpus,
        "LIVE_WORKER_IDENTITY_MISMATCH",
    )
    if registration["tasks"][task_id]["gpus"]:
        require(
            f"gres/gpu:pro6000={expected_gpus}" in fields["AllocTRES"].split(","),
            "LIVE_WORKER_GPU_TYPE_MISMATCH",
        )
    if manifest.get("resource_override") is not None:
        require(fields.get("QOS") == TEACHER_QOS, "MULTIGPU_REQUIRES_TEACHER_QOS")
        verify_multigpu_resources(fields, manifest["resource_override"], allocated=True)
    atomic_json(
        root / f"orchestration/worker_starts/{attempt_id}.json", observation, exclusive=True
    )
    return registration["tasks"][task_id]


def dispatch(plan, root, spec):
    # Verification accepts the path; runtime consumers use the frozen mapping.
    plan = read_json(plan) if isinstance(plan, (str, Path)) else plan
    operation = spec["operation"]
    if operation == "common_start":
        from sr_f1.runtime import establish_common_start

        return establish_common_start(plan, root)
    if operation == "engine":
        from sr_f1.engine import run_engine

        return run_engine(plan, root)
    if operation == "train":
        from sr_f1.training import train_path

        return train_path(plan, root, spec["run_id"])
    if operation == "evaluate":
        if spec["stage"] == "baseline" and has_baseline_parallel_repair(root):
            from sr_f1.baseline_parallel import run_parallel_baseline
            from sr_f1.orchestration import baseline_parallel_allocation

            require(
                spec["model_id"] == "SRF1_COMMON_START",
                "PARALLEL_BASELINE_REQUIRES_COMMON_START",
            )
            registration, _ = _worker_identity(root, "BASELINE")
            allocation = baseline_parallel_allocation(root, registration)
            require(allocation is not None, "PARALLEL_BASELINE_ALLOCATION_MISSING")
            return run_parallel_baseline(plan, root, allocation["gpus"])
        if spec["stage"] == "final":
            matrix = read_json(root / "RUN_MATRIX_FINAL.json")
            require(
                matrix["all_runs_terminal"] is True
                and len(matrix["runs"]) == 15
                and all(row["status"] in SCIENCE_TERMINAL for row in matrix["runs"]),
                "TEST_SEALED_UNTIL_ALL_SCIENCE_TERMINAL",
            )
            model = next((row for row in matrix["runs"] if row["run_id"] == spec["model_id"]), None)
            if model and model["status"] == "TECHNICAL_FAILED":
                relative = "evaluation/unavailable/" + spec["model_id"] + ".json"
                receipt = {
                    "plan_id": PLAN_ID,
                    "model_id": spec["model_id"],
                    "status": "NOT_EVALUATED_TECHNICAL_FAILURE",
                    "reason": model["failure"],
                    "run_matrix_sha256": file_hash(root / "RUN_MATRIX_FINAL.json"),
                    "missing_answers_are_incorrect": False,
                }
                atomic_json(root / relative, receipt, exclusive=True)
                return {
                    "status": "COMPLETE",
                    "artifacts": [relative],
                    "metadata": {"evaluation_complete": False},
                }
        from sr_f1.evaluation import evaluate_model
        from sr_f1.runtime import load_for_evaluation

        step = 0 if spec["model_id"] == "SRF1_COMMON_START" else 96
        runtime = load_for_evaluation(plan, root, spec["model_id"], step=step)
        result = evaluate_model(runtime, root, spec["model_id"], stage=spec["stage"], step=step)
        if spec["stage"] == "final" and result["status"] == "COMPLETE":
            external_identity = read_json(root / "external/chartqa/IDENTITY.json")
            require(
                external_identity["status"] in {"AVAILABLE_VERIFIED", "UNAVAILABLE"},
                "EXTERNAL_ASSET_AVAILABILITY_NOT_REGISTERED",
            )
            if external_identity["status"] == "AVAILABLE_VERIFIED":
                from sr_f1.chartqa import evaluate_chartqa

                external_result = evaluate_chartqa(runtime, root, spec["model_id"])
                return {
                    **external_result,
                    "artifacts": [*result["artifacts"], *external_result["artifacts"]],
                    "metadata": {
                        **result.get("metadata", {}),
                        **external_result.get("metadata", {}),
                    },
                }
            result["artifacts"].append("external/chartqa/IDENTITY.json")
        return result
    matrix = read_json(root / "RUN_MATRIX_FINAL.json")
    if operation == "analyze":
        from sr_f1.analysis import analyze

        return analyze(root, matrix)
    if operation == "release":
        from sr_f1.release import release

        return release(root, matrix)
    raise ValueError("UNREGISTERED_OPERATION:" + operation)


def execute(plan, root, task_id):
    from sr_f1.freeze import verify_execution

    spec = validate_allocation(root, task_id)
    verify_execution(plan, root, require_engine=spec["phase"] in {"S3", "S4", "S5", "S6"})
    require(not (root / "STOP").exists(), "STOP_REQUESTED_NO_NEW_WORK")
    with worker_lease(root, task_id):
        try:
            result = dispatch(plan, root, spec)
            require(isinstance(result, dict), "WORKER_RESULT_REQUIRED")
            if result["status"] == "CHECKPOINTED":
                marker = checkpoint_task(
                    root, task_id, result["reason"], result["artifacts"], result.get("metadata")
                )
            else:
                require(result["status"] == "COMPLETE", "WORKER_RESULT_NOT_COMPLETE")
                marker = complete_task(root, task_id, result["artifacts"], result.get("metadata"))
            print(json.dumps(marker, sort_keys=True), flush=True)
            return 0
        except Exception as exc:
            fail_task(root, task_id, repr(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    return execute(args.plan.resolve(), args.run_root.resolve(), args.task_id)


if __name__ == "__main__":
    sys.exit(main())
