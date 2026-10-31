"""Independent fixed Round B around the unchanged b7 pilot computation.

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
from sr_f6_memory_resume.execution import _cuda_observation, _safe_cuda_observation

from .contract import (
    QUEUES,
    ROOT,
    TASKS,
    require_untouched_primary,
    require_untouched_recovery,
    verify_contract,
)

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


def prepare_recovery(root, contract, plan, admitted, task, directory, identity):
    """Only original step004; call unchanged b7 preparation and exact finish API."""
    from sr_f6 import pilot_jobs, recovery
    from sr_f6.transactions import RunTransactions

    target = root / pilot_jobs._contract(admitted).recovery_directory
    require_untouched_recovery(target)
    primary_task = next(t for t in plan["tasks"] if t["task_id"] == "PRIMARY-b0-U11")
    primary = pilot_jobs._verify_task(root, contract["plan_ref"], admitted, primary_task)
    require(primary is not None, "Actual b0/U11 baseline required")
    report = recovery._bound(root, primary["producer"])
    with RunTransactions(directory, identity) as tx:
        markers = tx.published()
        require([m["step"] for m in markers] == list(range(13)), "Complete baseline required")
        imported = recovery.prepare_recovery(
            tx,
            experiment_root=root,
            phase_context=admitted["phase_context"],
            pilot_spec_sha256=contract["pins"]["pilot_spec_sha256"],
            source_commit_sha256=markers[4]["commit_sha256"],
            primary_execution_ref=report["execution"],
        )
    return target, imported["binding_ref"]


def run_task(contract_path, expected_contract_sha256, task_id, attempt_id):
    """Execute exactly one untouched B task; recovery is only the frozen step004 import."""
    require(task_id in TASKS, "Only original B tasks are authorized")
    require(
        os.environ.get("OMP_NUM_THREADS") == "1",
        "Original single CPU thread required before tensor imports",
    )
    worker = 0 if task_id in QUEUES[0] else 1
    queue = QUEUES[worker]
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,40}", attempt_id) is not None,
        "Safe new repair attempt id required",
    )
    from sr_f6 import pilot, pilot_jobs, runtime

    with pilot_jobs._lock(ROOT / JOBS / ".prerequisites.lock", wait=True):
        envelope, contract, validation_ref = verify_contract(
            contract_path, expected_contract_sha256
        )
    from sr_f6_memory_repair.offload import saved_activation_offload

    _quiescent()
    root = plain_path(contract["experiment_root"])
    plan, admitted = pilot_jobs.verify_plan(root, contract["plan_ref"])
    pilot_jobs._worker_environment(root)
    require(
        os.environ.get("SLURM_JOB_NAME", "").startswith("LH-SRF6-"), "Repair job prefix required"
    )
    require(not runtime._LOADED_RUNTIMES, "Repair task requires a fresh model process")
    task = next(item for item in plan["tasks"] if item["task_id"] == task_id)
    execution_id = f"LH-SRF6-B-{worker}-{attempt_id}-{task_id}"
    folder = root / JOBS / "TASKS" / task_id
    extension = plain_path(contract_path).parent / "EXECUTIONS" / task_id
    with (
        pilot_jobs._lock(root / JOBS / f"WORKERS/B/{worker}/.worker.lock"),
        pilot_jobs._lock(folder / ".task.lock"),
    ):
        require(
            not extension.exists(), "Repair execution is single-attempt; preserve prior outcome"
        )
        require(plan["queues"]["B"] == QUEUES, "Frozen B queue differs")
        with pilot_jobs._lock(root / JOBS / ".prerequisites.lock", wait=True):
            pilot_jobs._dependency_check(root, contract["plan_ref"], plan, admitted, "B", worker)
        for prior in queue[: queue.index(task_id)]:
            item = next(t for t in plan["tasks"] if t["task_id"] == prior)
            require(
                pilot_jobs._verify_task(root, contract["plan_ref"], admitted, item) is not None,
                "Original fixed B predecessor incomplete",
            )
        require(
            pilot_jobs._verify_task(root, contract["plan_ref"], admitted, task) is None,
            "Completed original task must not be repeated",
        )
        directory, identity = pilot_jobs._registered(root, admitted, task)
        require(not (folder / "ATTEMPTS").exists(), "Prior B task intent forbids retry")
        if task["kind"] == "PRIMARY":
            require_untouched_primary(directory)
        else:
            require_untouched_recovery(root / pilot_jobs._contract(admitted).recovery_directory)
        attempt = folder / "ATTEMPTS" / execution_id
        require(not attempt.exists(), "Original task attempt immutable")
        started = dict(
            schema=pilot_jobs.SCHEMA,
            status="STARTED",
            at=_now(),
            plan_ref=contract["plan_ref"],
            round="B",
            worker=worker,
            attempt_id=attempt_id,
            queue=list(queue),
            resume_tasks=[],
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
                a_predecessor_ref=envelope["a_delivery_ref"],
                cuda_before=cuda_before,
                scientific_gate="NOT_SIGNED",
            ),
        )
        pilot_jobs._write(attempt / "STARTED.json", started)
        manager = None
        try:
            binding_ref = None
            if task["kind"] == "RECOVERY":
                with pilot_jobs._lock(root / JOBS / ".prerequisites.lock", wait=True):
                    directory, binding_ref = prepare_recovery(
                        root, contract, plan, admitted, task, directory, identity
                    )
            with saved_activation_offload(
                lambda: _bound_model(runtime, execution_id),
                max_host_bytes=contract["max_host_bytes"],
                max_disk_bytes=contract["max_disk_bytes"],
                scratch_directory=extension / "STORAGE",
                disk_backend=contract.get("storage_backend", "file_v2"),
                max_device_bytes=contract.get("max_device_bytes", 0),
            ) as manager:
                pilot.run(
                    root,
                    task["parent"],
                    task["arm"],
                    execution_id,
                    **contract["pins"],
                    recovery_binding_ref=binding_ref,
                )
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
            if binding_ref is not None:
                with pilot_jobs._lock(root / JOBS / ".prerequisites.lock", wait=True):
                    producer = pilot_jobs._finish_recovery(
                        root, admitted, task, producer, binding_ref, attempt
                    )
            receipt = dict(
                schema=pilot_jobs.SCHEMA,
                status="TASK_COMPLETE_NOT_GATE_SIGNED",
                task=task,
                plan_ref=contract["plan_ref"],
                producer=producer,
                gate_status="NOT_SIGNED",
            )
            with pilot_jobs._lock(root / JOBS / ".prerequisites.lock", wait=True):
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
                    round_b_contract_ref={
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
    parser.add_argument("--task-id", required=True, choices=TASKS)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    run_task(args.contract, args.contract_sha256, args.task_id, args.attempt_id)


if __name__ == "__main__":
    main()
