"""V8 single-attempt recovery around the unchanged b7 pilot computation.

The v7 package remains loaded at its actual frozen deployment. This independent
wrapper changes admission/provenance and observes resource use only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time

from sr_f6_memory_repair import execution as legacy

from .contract import full_contract, verify_contract, verify_validation

QUEUE = legacy.QUEUE
JOBS = legacy.JOBS
PRODUCER_REVISION = legacy.PRODUCER_REVISION
require = legacy.require
plain_path = legacy.plain_path
_now = legacy._now
_write = legacy._write
_quiescent = legacy._quiescent
_bound_model = legacy._bound_model
_storage_summary = legacy._storage_summary
_failure_summary = legacy._failure_summary


def _cuda_observation():
    """Observe lifetime allocator peaks without reset, cache eviction or kernels."""
    import torch

    require(torch.cuda.is_available(), "Actual CUDA recovery process required")
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return dict(
        device=device,
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        lifetime_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        lifetime_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        free_bytes=free,
        total_bytes=total,
        observation_only=True,
        peak_reset=False,
    )


def _safe_cuda_observation():
    try:
        return _cuda_observation()
    except BaseException as exc:
        return dict(
            status="CUDA_OBSERVATION_FAILED_PRESERVED",
            error_type=type(exc).__name__,
            error=str(exc),
        )


def run_task(contract_path, expected_contract_sha256, task_id, attempt_id):
    """Execute exactly one fixed b1 task in a fresh process, without automatic retry."""
    require(task_id in QUEUE, "Only original b1 queue is authorized")
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,40}", attempt_id) is not None,
        "Safe new repair attempt id required",
    )
    envelope = verify_contract(contract_path, expected_contract_sha256)
    contract = full_contract(envelope)
    validation_ref = envelope["full_result_ref"]
    from sr_f6 import pilot, pilot_jobs, runtime
    from sr_f6_memory_repair.offload import saved_activation_offload

    checked = verify_validation(envelope)
    require(
        checked.get("status") == "VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH"
        and checked.get("contract_sha256") == envelope["full_contract_ref"]["sha256"]
        and checked.get("producer_revision") == PRODUCER_REVISION
        and checked.get("repair_revision") == contract["repair_source"]["revision"]
        and checked.get("generated_responses") == 0
        and checked.get("optimizer_updates") == 0
        and checked.get("compared_gradient_rows") == [0]
        and checked.get("offload_rows") == 128,
        "Actual bounded storage validation receipt required; no placeholder PASS",
    )
    _quiescent()
    root = plain_path(contract["experiment_root"])
    plan, admitted = pilot_jobs.verify_plan(root, contract["plan_ref"])
    pilot_jobs._worker_environment(root)
    require(
        os.environ.get("SLURM_JOB_NAME", "").startswith("LH-SRF6-"), "Repair job prefix required"
    )
    require(not runtime._LOADED_RUNTIMES, "Repair task requires a fresh model process")
    task = next(item for item in plan["tasks"] if item["task_id"] == task_id)
    execution_id = f"LH-SRF6-A-1-{attempt_id}-{task_id}"
    folder = root / JOBS / "TASKS" / task_id
    extension = plain_path(contract_path).parent / "EXECUTIONS" / task_id
    with (
        pilot_jobs._lock(root / JOBS / "WORKERS/A/1/.worker.lock"),
        pilot_jobs._lock(folder / ".task.lock"),
    ):
        require(
            not extension.exists(), "Repair execution is single-attempt; preserve prior outcome"
        )
        for prior in QUEUE[: QUEUE.index(task_id)]:
            item = next(t for t in plan["tasks"] if t["task_id"] == prior)
            require(
                pilot_jobs._verify_task(root, contract["plan_ref"], admitted, item) is not None,
                "Original fixed b1 predecessor incomplete",
            )
        require(
            pilot_jobs._verify_task(root, contract["plan_ref"], admitted, task) is None,
            "Completed original task must not be repeated",
        )
        directory, _identity = pilot_jobs._registered(root, admitted, task)
        if task_id == QUEUE[0]:
            require(
                sorted(p.name for p in (directory / "checkpoints").iterdir()) == ["step-000"],
                "Restore is authorized only from original step zero",
            )
        else:
            require(
                not (directory / "TRAINER_ATTEMPTS").exists()
                and not (folder / "ATTEMPTS").exists(),
                "No retry of later b1 tasks authorized",
            )
        attempt = folder / "ATTEMPTS" / execution_id
        require(not attempt.exists(), "Original task attempt immutable")
        started = dict(
            schema=pilot_jobs.SCHEMA,
            status="STARTED",
            at=_now(),
            plan_ref=contract["plan_ref"],
            round="A",
            worker=1,
            attempt_id=attempt_id,
            queue=list(QUEUE),
            resume_tasks=[QUEUE[0]],
            task=task,
            execution_id=execution_id,
        )
        cuda_before = _cuda_observation()
        task_started = time.monotonic()
        extension.mkdir(parents=True)
        _write(
            extension / "STARTED.json",
            dict(
                started,
                contract_ref={"path": str(contract_path), "sha256": expected_contract_sha256},
                validation_ref=validation_ref,
                producer_revision=PRODUCER_REVISION,
                repair_source=contract["repair_source"],
                execution_source=envelope["execution_source"],
                full_contract_ref=envelope["full_contract_ref"],
                cuda_before=cuda_before,
                scientific_gate="NOT_SIGNED",
            ),
        )
        pilot_jobs._write(attempt / "STARTED.json", started)
        manager = None
        try:
            with saved_activation_offload(
                lambda: _bound_model(runtime, execution_id),
                max_host_bytes=contract["max_host_bytes"],
                max_disk_bytes=contract["max_disk_bytes"],
                scratch_directory=extension / "STORAGE",
                disk_backend=contract.get("storage_backend", "file_v2"),
                max_device_bytes=contract.get("max_device_bytes", 0),
            ) as manager:
                pilot.run(root, "b1", task["arm"], execution_id, **contract["pins"])
            storage_summary = _storage_summary(
                manager,
                contract["max_host_bytes"],
                contract["max_disk_bytes"],
                extension / "STORAGE",
                disk_backend=contract.get("storage_backend", "file_v2"),
                max_device_bytes=contract.get("max_device_bytes", 0),
            )
            _write(
                extension / "STORAGE_FACTS.json",
                dict(
                    status="ACTUAL_STORAGE_CONTEXT_COMPLETE_NOT_SCIENTIFIC_PASS",
                    at=_now(),
                    contract_ref={"path": str(contract_path), "sha256": expected_contract_sha256},
                    validation_ref=validation_ref,
                    execution_id=execution_id,
                    offload=storage_summary,
                    elapsed_seconds=time.monotonic() - task_started,
                    cuda_before=cuda_before,
                    cuda_after=_cuda_observation(),
                ),
            )
            producer = pilot_jobs._ref(
                root, directory / "TRAINER_ATTEMPTS" / execution_id / "COMPLETE.json"
            )
            receipt = dict(
                schema=pilot_jobs.SCHEMA,
                status="TASK_COMPLETE_NOT_GATE_SIGNED",
                task=task,
                plan_ref=contract["plan_ref"],
                producer=producer,
                gate_status="NOT_SIGNED",
            )
            pilot_jobs._verify_task(root, contract["plan_ref"], admitted, task, receipt=receipt)
            pilot_jobs._write(pilot_jobs._result_path(root, task, phase="PILOT_V3"), receipt)
            pilot_jobs._write(attempt / "COMPLETE.json", receipt)
            _write(
                extension / "COMPLETE.json",
                dict(
                    receipt,
                    offload=storage_summary,
                    storage_facts_ref={
                        "path": str(extension / "STORAGE_FACTS.json"),
                        "sha256": hashlib.sha256(
                            (extension / "STORAGE_FACTS.json").read_bytes()
                        ).hexdigest(),
                    },
                    full_trainer_numerical_pass=False,
                    resume_contract_ref={
                        "path": str(contract_path),
                        "sha256": expected_contract_sha256,
                    },
                    execution_source=envelope["execution_source"],
                    full_validation_ref=validation_ref,
                ),
            )
            return receipt
        except BaseException as exc:
            failure = dict(
                status="FAILED_PRESERVED",
                at=_now(),
                error_type=type(exc).__name__,
                error=str(exc),
                task=task,
                sampling_retry_authorized=False,
                peer_cancellation_requested=False,
            )
            if not (attempt / "FAILURE.json").exists():
                pilot_jobs._write(attempt / "FAILURE.json", failure)
            _write(
                extension / "FAILURE.json",
                dict(
                    failure,
                    offload=_failure_summary(manager),
                    elapsed_seconds=time.monotonic() - task_started,
                    cuda_before=cuda_before,
                    cuda_after=_safe_cuda_observation(),
                ),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--task-id", required=True, choices=QUEUE)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    run_task(args.contract, args.contract_sha256, args.task_id, args.attempt_id)


if __name__ == "__main__":
    main()
