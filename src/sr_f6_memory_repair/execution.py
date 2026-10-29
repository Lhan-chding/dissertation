"""Single-attempt b1 storage repair around the immutable registered pilot producer.

This module never changes a probability, optimizer, seed, budget or saved answer.
The separate actual validation receipt is necessary but is not a scientific gate.
Original producer locks, transactions and COMPLETE verification remain authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "srf6-memory-repair-contract-v1"
PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f6_semantic_factorial_20261027"
PRODUCER_REVISION = "b7e7b067f237b360b1dfb2c3075d3f14be7c281f"
PRODUCER_DEPLOYMENT = PROJECT / "sr_f6_code_b7e7b06"
JOBS = "PHASE_RUNS/PILOT_V3/JOBS"
RUN = "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/b1/r0/U10"
OLD_EXECUTION = "LH-SRF6-A-1-A-233996-PRIMARY-b1-U10"
QUEUE = ("PRIMARY-b1-U10", "PRIMARY-b1-C_SHUF", "PRIMARY-b1-U11")


def require(value, message):
    if not value:
        raise ValueError(message)


def plain_path(value):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "Absolute canonical path required")
    require(path == path.resolve(strict=False), "Symlink or noncanonical path forbidden")
    require(path.is_relative_to(PROJECT) and path != PROJECT, "Outside canonical project")
    return path


def _json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(payload, object_pairs_hook=pairs)


def strict_ref(path, sha256):
    path = plain_path(path)
    require(isinstance(sha256, str) and re.fullmatch(r"[0-9a-f]{64}", sha256), "Invalid SHA256")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), "Reference must be a regular file")
        payload = stream.read()
        after = os.fstat(stream.fileno())
    require(
        (before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns),
        "Reference changed while reading",
    )
    require(hashlib.sha256(payload).hexdigest() == sha256, "Reference hash differs: " + str(path))
    return payload


def ref_path(root, ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact reference required")
    path = Path(ref["path"])
    return plain_path(path if path.is_absolute() else plain_path(root) / path)


def bound_ref(root, ref):
    return _json(strict_ref(ref_path(root, ref), ref["sha256"]))


def read_contract(path, expected_contract_sha256):
    value = _json(strict_ref(path, expected_contract_sha256))
    require(
        isinstance(value, dict) and value.get("schema") == SCHEMA, "Repair contract schema differs"
    )
    return value


def _source(contract):
    producer = contract["producer"]
    require(
        producer["revision"] == PRODUCER_REVISION
        and plain_path(producer["deployment"]) == PRODUCER_DEPLOYMENT,
        "Original producer changed",
    )
    repair = contract["repair_source"]
    require(
        re.fullmatch(r"[0-9a-f]{40}", repair["revision"])
        and repair["revision"] != PRODUCER_REVISION,
        "Separate committed repair source required",
    )
    deployment = plain_path(repair["deployment"])
    require(deployment != PRODUCER_DEPLOYMENT, "Repair must not overwrite producer")
    files = repair["files"]
    require(isinstance(files, list) and files, "Repair source files absent")
    paths = [ref_path(deployment, ref) for ref in files]
    require(
        len(paths) == len(set(paths)) and all(p.is_relative_to(deployment) for p in paths),
        "Duplicate or escaped repair source",
    )
    required = {
        deployment / "src/sr_f6_memory_repair" / f
        for f in ("__init__.py", "execution.py", "offload.py", "validation.py")
    }
    require(required <= set(paths), "Repair implementation source closure incomplete")
    require(
        Path(__file__).resolve() == deployment / "src/sr_f6_memory_repair/execution.py",
        "Execution loaded from another repair deployment",
    )
    for path, ref in zip(paths, files, strict=True):
        strict_ref(path, ref["sha256"])
    from sr_f6 import pilot, pilot_jobs, runtime

    for module in (pilot, pilot_jobs, runtime):
        require(
            Path(module.__file__).resolve().is_relative_to(PRODUCER_DEPLOYMENT / "src"),
            "Loaded computation is not immutable b7",
        )
    plan, admitted = pilot_jobs.verify_plan(contract["experiment_root"], contract["plan_ref"])
    require(
        plan["pins"] == contract["pins"]
        and plan["phase"] == "PILOT_V3"
        and tuple(plan["queues"]["A"][1]) == QUEUE,
        "Original registration/queue changed",
    )
    require(
        admitted["phase_context"]["identity"]["source_manifest"] == producer["source_manifest_ref"],
        "Producer manifest identity differs",
    )


def _original_evidence(contract):
    root = plain_path(contract["experiment_root"])
    run = root / RUN
    for key, expected in (
        ("original_ready_ref", run / "rollouts/step-001/READY.json"),
        ("step0_commit_ref", run / "checkpoints/step-000/COMMIT.json"),
        ("original_failure_ref", run / "TRAINER_ATTEMPTS" / OLD_EXECUTION / "FAILURE.json"),
        (
            "original_task_failure_ref",
            root / JOBS / "TASKS" / QUEUE[0] / "ATTEMPTS" / OLD_EXECUTION / "FAILURE.json",
        ),
        ("original_worker_failure_ref", root / JOBS / "WORKERS/A/1/ATTEMPTS/A-233996/FAILURE.json"),
    ):
        require(
            ref_path(root, contract[key]) == expected, "Original evidence location differs: " + key
        )
    failure = bound_ref(root, contract["original_failure_ref"])
    require(
        failure.get("status") == "FAILED_PRESERVED"
        and failure.get("error_type") == "OutOfMemoryError"
        and failure.get("task_id") == OLD_EXECUTION
        and failure.get("parent") == "b1"
        and failure.get("arm") == "U10"
        and "out of memory" in failure.get("error", "").lower(),
        "Original b1 OOM not established",
    )
    for key in ("original_task_failure_ref", "original_worker_failure_ref"):
        require(
            bound_ref(root, contract[key]).get("status") == "FAILED_PRESERVED",
            "Original failed task/worker evidence absent",
        )
    commit = bound_ref(root, contract["step0_commit_ref"])
    require(commit.get("step") == 0, "Original step zero required")
    ready = bound_ref(root, contract["original_ready_ref"])
    require(
        ready.get("status") == "ROLLOUT_READY"
        and ready.get("sampler_state", {}).get("rollout_cursor")
        == {"step": 1, "groups": 16, "rows": 128},
        "Complete original READY1 required",
    )
    rollout = bound_ref(run, ready["references"]["rollout"])
    require(
        len(rollout["records"]) == 128 and len(rollout["complete_group_references"]) == 16,
        "Original 128 rows/16 groups required",
    )
    raw_count = 0
    for ref in rollout["complete_group_references"]:
        group = bound_ref(run, ref)
        raw = strict_ref(ref_path(run, group["raw_reference"]), group["raw_reference"]["sha256"])
        rows = [_json(line) for line in raw.splitlines()]
        require(
            len(rows) == 8 and all(r.get("task_id") == OLD_EXECUTION for r in rows),
            "Original physical draw provenance changed",
        )
        raw_count += len(rows)
    require(raw_count == 128, "Original draw closure incomplete")
    files = contract["original_files"]
    paths = [ref_path(root, ref) for ref in files]
    require(len(paths) == len(set(paths)), "Duplicate original file reference")
    expected = set()
    for directory in (run / "checkpoints/step-000", run / "rollouts/step-001"):
        expected.update(p for p in directory.rglob("*") if p.is_file() or p.is_symlink())
    require(set(paths) == expected, "Original checkpoint/READY tree closure differs")
    for path, ref in zip(paths, files, strict=True):
        strict_ref(path, ref["sha256"])
    audit_ref = contract["original_audit_ref"]
    snapshot = strict_ref(ref_path(root, audit_ref), audit_ref["sha256"])
    require(
        type(contract["original_audit_bytes"]) is int
        and len(snapshot) == contract["original_audit_bytes"],
        "Original audit length differs",
    )
    live = plain_path(contract["original_audit_path"])
    require(live == run / "NUMERICAL_TOKEN_AUDIT.jsonl", "Original audit path differs")
    with live.open("rb") as stream:
        require(stream.read(len(snapshot)) == snapshot, "Original audit prefix changed")
    events = [_json(line) for line in snapshot.splitlines()]
    require(len(events) == 1, "Exactly the original failed update evidence required")
    event = events[0]
    require(
        event.get("status") == "UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP"
        and event.get("task_id") == OLD_EXECUTION
        and event.get("optimizer_started") is False
        and event.get("parameter_state_changed") is False
        and event.get("committed_source_step") == 0,
        "Original optimizer state was not preserved",
    )


def verify_contract(contract_path, expected_contract_sha256):
    contract = read_contract(contract_path, expected_contract_sha256)
    require(
        plain_path(contract["experiment_root"]) == ROOT, "Only original registered root allowed"
    )
    require(
        contract["original_job_id"] == "233996"
        and contract["original_step_id"] == "233996.1"
        and contract["original_execution_id"] == OLD_EXECUTION,
        "Original execution differs",
    )
    require(
        type(contract["max_host_bytes"]) is int and 0 < contract["max_host_bytes"] <= 96 * 2**30,
        "Explicit bounded host activation memory required",
    )
    require(
        contract["plan_ref"]
        == {
            "path": JOBS + "/PLAN.json",
            "sha256": "56102cd5b10d93bde1996527e39fa1ec30943fc2618cab2838f03d5fd550d4f4",
        },
        "Original fixed jobs plan differs",
    )
    require(
        contract["pins"]
        == {
            "phase": "PILOT_V3",
            "phase_identity_sha256": (
                "d9f061d1cf08e14c485083bbed421a412a29868fd8305ad4c265fb6404562284"
            ),
            "pilot_spec_sha256": "4936b29e3a897334c1c4e67aad436bf46bdea9a105354d0155484d2acc20994c",
            "admission_ref": {
                "path": "PHASES/PILOT_V3/ADMISSION.json",
                "sha256": "b65831ad6ec57c829a69a9ad0a87a6aaf6a4a173e2d921cf953462ca7519b2a6",
            },
        },
        "Original phase, specification or admission changed",
    )
    _source(contract)
    _original_evidence(contract)
    return contract


def _quiescent():
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", "233996", "--format=JobIDRaw,State,ExitCode"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    rows = [line.split("|") for line in result.stdout.splitlines()]
    states = [
        (row[1].split()[0].rstrip("+"), row[2])
        for row in rows
        if len(row) >= 3 and row[0] == "233996.1"
    ]
    require(
        states == [("FAILED", "1:0")],
        "Original b1 Slurm step must have actual FAILED 1:0; b0 may remain running",
    )
    queue = subprocess.run(
        ["squeue", "--steps", "-h", "-u", "varun024", "-o", "%i"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    require("233996.1" not in queue.stdout.split(), "Original b1 step remains active")


def _write(path, value):
    path = plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _now():
    return datetime.now(timezone.utc).isoformat()


def _bound_model(runtime_module, execution_id):
    values = list(runtime_module._LOADED_RUNTIMES)
    require(
        len(values) == 1
        and values[0].identity.get("task_id") == execution_id
        and runtime_module._LOADED_RUNTIMES[values[0]]["model"] is values[0].model,
        "Only actual current immutable producer runtime allowed",
    )
    return values[0].model


def _storage_summary(manager, max_host_bytes):
    summary = manager.summary()
    require(
        summary["scope"] == "CUDA_SAVED_ACTIVATION_STORAGE"
        and summary["model_bound"]
        and summary["closed"]
        and summary["pack_count"] > 0
        and summary["unpack_count"] > 0
        and summary["peak_host_bytes"] <= max_host_bytes
        and summary["numerical_pass"] is False,
        "Actual bound storage context did not complete",
    )
    return summary


def _failure_summary(manager):
    try:
        return manager.summary() if manager else None
    except BaseException as exc:
        return dict(
            status="STORAGE_SUMMARY_FAILED_PRESERVED", error_type=type(exc).__name__, error=str(exc)
        )


def run_task(contract_path, expected_contract_sha256, validation_ref, task_id, attempt_id):
    """Execute exactly one fixed b1 task in a fresh process, without automatic retry."""
    require(task_id in QUEUE, "Only original b1 queue is authorized")
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,40}", attempt_id) is not None,
        "Safe new repair attempt id required",
    )
    contract = verify_contract(contract_path, expected_contract_sha256)
    from sr_f6 import pilot, pilot_jobs, runtime

    from .offload import saved_activation_offload
    from .validation import verify_validation

    checked = verify_validation(contract_path, expected_contract_sha256, validation_ref)
    require(
        checked.get("status") == "VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH"
        and checked.get("contract_sha256") == expected_contract_sha256
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
        extension.mkdir(parents=True)
        _write(
            extension / "STARTED.json",
            dict(
                started,
                contract_ref={"path": str(contract_path), "sha256": expected_contract_sha256},
                validation_ref=validation_ref,
                producer_revision=PRODUCER_REVISION,
                repair_source=contract["repair_source"],
                scientific_gate="NOT_SIGNED",
            ),
        )
        pilot_jobs._write(attempt / "STARTED.json", started)
        manager = None
        try:
            with saved_activation_offload(
                lambda: _bound_model(runtime, execution_id),
                max_host_bytes=contract["max_host_bytes"],
            ) as manager:
                pilot.run(root, "b1", task["arm"], execution_id, **contract["pins"])
            storage_summary = _storage_summary(manager, contract["max_host_bytes"])
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
                dict(receipt, offload=storage_summary, full_trainer_numerical_pass=False),
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
                dict(failure, offload=_failure_summary(manager)),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--task-id", required=True, choices=QUEUE)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    run_task(
        args.contract,
        args.contract_sha256,
        {"path": args.validation_path, "sha256": args.validation_sha256},
        args.task_id,
        args.attempt_id,
    )


if __name__ == "__main__":
    main()
