"""Bind Slurm identities and concurrency; track device time without a time cap."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .execution import (
    BudgetLedger,
    atomic_json,
    exclusive_json,
    locked,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)


def reserve_allocation(run_root, allocation_key, stage, *, seconds, gpus=1):
    root = Path(run_root)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", allocation_key):
        raise ValueError("Unsafe allocation key")
    if type(gpus) is not int or not 1 <= gpus <= 5 or type(seconds) is not int or seconds <= 0:
        raise ValueError("Invalid allocation envelope")
    folder = root / "accounting/allocations"
    with locked(root / "accounting/ALLOCATIONS.lock"):
        records = [read_json(p) for p in folder.glob("*.json")]
        if any(row["allocation_key"] == allocation_key for row in records):
            raise FileExistsError("Allocation reservation already exists")
        active = sum(r["gpus"] for r in records if r["status"] != "TERMINAL_VERIFIED")
        if active + gpus > 5:
            raise PermissionError("Five-GPU concurrent reservation ceiling exceeded")
        identity = {
            "allocation_key": allocation_key,
            "stage": stage,
            "gpus": gpus,
            "seconds": seconds,
        }
        BudgetLedger(root).reserve("allocated_gpu_hours", gpus * seconds / 3600, identity)
        record = {
            **identity,
            "time": utc_now(),
            "status": "RESERVED",
            "job_id": None,
            "pre_freeze_hash": sha256_file(root / "manifests/PRE_INFERENCE_FREEZE.json"),
        }
        exclusive_json(folder / f"{allocation_key}.json", record)
    return record


def bind_allocation(run_root, allocation_key, job_id):
    root = Path(run_root)
    if not str(job_id).isdigit():
        raise ValueError("Only an exact numeric allocation job ID can be bound")
    with locked(root / "accounting/ALLOCATIONS.lock"):
        path = root / f"accounting/allocations/{allocation_key}.json"
        record = read_json(path)
        if record["status"] != "RESERVED":
            raise PermissionError("Allocation already bound or terminal")
        record.update(status="BOUND", job_id=str(job_id), bound_at=utc_now())
        atomic_json(path, record)
    return record


def verify_live_gpu_resources(fields, registered_gpus):
    """Cross-check aggregate allocated TRES against the one-node reservation."""
    allocated = dict(
        item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item
    )
    if fields.get("NumNodes") != "1" or allocated.get("node") != "1":
        raise PermissionError("Exactly one live allocated node must be verified")
    counts = [allocated.get("gres/gpu"), allocated.get("gres/gpu:pro6000")]
    if any(value is None or not value.isdigit() for value in counts):
        raise PermissionError("Live GPU counts/type cannot be verified")
    if any(int(value) != registered_gpus for value in counts):
        raise PermissionError("Live GPU count differs from registered allocation")
    other_types = [
        key
        for key in allocated
        if key.startswith("gres/gpu:") and key != "gres/gpu:pro6000" and allocated[key] != "0"
    ]
    if other_types:
        raise PermissionError("Unregistered GPU types in live allocation")
    per_node = fields.get("TresPerNode", "").split(",")
    gpu_specs = [part for part in per_node if part.startswith("gres/gpu")]
    if gpu_specs != [f"gres/gpu:pro6000:{registered_gpus}"]:
        raise PermissionError("Per-node GPU resource identity differs from allocation")
    return dict(nodes=1, gpu_count=registered_gpus, gpu_type="pro6000")


def require_allocation(run_root, stage):
    """Verify live job identity before CUDA loading; time is accounting metadata."""
    root = Path(run_root)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise PermissionError("A registered Slurm allocation is required")
    matches = [
        read_json(p)
        for p in (root / "accounting/allocations").glob("*.json")
        if read_json(p).get("job_id") == job_id
    ]
    if len(matches) != 1:
        raise PermissionError("Slurm allocation has no unique prior reservation")
    record = matches[0]
    if record["status"] != "BOUND" or record["stage"] != stage:
        raise PermissionError("Wrong or terminated allocation stage")
    if record["pre_freeze_hash"] != sha256_file(root / "manifests/PRE_INFERENCE_FREEZE.json"):
        raise PermissionError("Allocation freeze changed")
    reservations = read_jsonl(root / "accounting/COST_LEDGER.jsonl")
    expected = record["gpus"] * record["seconds"] / 3600
    bound = [
        r
        for r in reservations
        if r["kind"] == "allocated_gpu_hours"
        and r.get("identity", {}).get("allocation_key") == record["allocation_key"]
    ]
    time_accounting_status = (
        "MATCHED"
        if len(bound) == 1 and bound[0]["amount"] == expected
        else "MISSING_OR_MISMATCHED_ESTIMATE"
    )
    result = subprocess.run(
        ["scontrol", "show", "job", job_id, "-o"],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    fields = dict(part.split("=", 1) for part in result.stdout.split() if "=" in part)
    if fields.get("JobId") != job_id or fields.get("JobState") != "RUNNING":
        raise PermissionError("Live scheduler allocation is not verified running")
    if fields.get("JobName") != "mmcore-" + record["allocation_key"]:
        raise PermissionError("Scheduler job name mismatch")
    live_resources = verify_live_gpu_resources(fields, record["gpus"])
    wall = fields.get("TimeLimit", "")
    days, sep, clock = wall.partition("-")
    if not sep:
        days, clock = "0", days
    parts = clock.split(":")
    live_seconds = None
    if len(parts) == 3 and all(part.isdigit() for part in [days, *parts]):
        live_seconds = int(days) * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    evidence = root / f"accounting/allocation_checks/{job_id}_{os.getpid()}.json"
    atomic_json(
        evidence,
        {
            "time": utc_now(),
            "job_id": job_id,
            "stage": stage,
            "scheduler_output": result.stdout,
            "live_resources": live_resources,
            "scheduler_time_limit": wall,
            "scheduler_time_limit_seconds": live_seconds,
            "gpu_hours_are_execution_gate": False,
            "time_accounting_status": time_accounting_status,
            "expected_allocation_time_estimate": expected,
            "recorded_allocation_time_estimates": bound,
            "reservation": record,
        },
    )
    return record


def mark_terminal(run_root, allocation_key, scheduler_receipt):
    """Release concurrency only after exact sacct + squeue evidence; never refund costs."""
    root = Path(run_root)
    with locked(root / "accounting/ALLOCATIONS.lock"):
        path = root / f"accounting/allocations/{allocation_key}.json"
        record = read_json(path)
        evidence = read_json(scheduler_receipt)
        if evidence.get("job_id") != record["job_id"] or evidence.get("squeue_absent") is not True:
            raise PermissionError("Scheduler terminal identity is not verified")
        if evidence.get("state") not in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
        }:
            raise PermissionError("Allocation has no known terminal state")
        if type(evidence.get("elapsed_seconds")) is not int or evidence["elapsed_seconds"] < 0:
            raise PermissionError("Missing scheduler elapsed time")
        record.update(
            status="TERMINAL_VERIFIED",
            terminal=evidence,
            terminal_receipt_sha256=sha256_file(scheduler_receipt),
        )
        atomic_json(path, record)
    return record
