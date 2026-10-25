"""Submit the same four frozen G2 shards in two dual-GPU allocations.

Unknown submission outcomes require diagnosis; no automated duplicate retry.
This is a one-shot dispatcher, not a recurring monitor or training launcher.
"""

import fcntl
import getpass
import json
import os
import subprocess
from pathlib import Path

from sr_f4.common import now, read, write
from sr_f4.pool import ACCOUNT, QOS, Slurm, gpu_count

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025")
CODE = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f5_code_26e560b")
TERMINAL_STATES = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def verify_previous_jobs(slurm, original, user):
    jobs = original.get("jobs", [])
    if len(jobs) != 4 or {job.get("shard") for job in jobs} != set(range(4)):
        raise ValueError("Original four-worker submission receipt is incomplete")
    ids = [job.get("job_id") for job in jobs]
    if len(set(ids)) != 4 or any(not isinstance(j, str) or not j.isdecimal() for j in ids):
        raise ValueError("Original job identity is ambiguous")
    active = slurm.user_job_ids(user) | set(slurm.queue())
    terminals = []
    for job in jobs:
        if job["job_id"] in active:
            raise RuntimeError("Original G2 job still active: " + job["job_id"])
        record = slurm.terminal(job["job_id"])
        if not record or record.get("state") not in TERMINAL_STATES:
            raise RuntimeError("Original G2 terminal state unknown: " + job["job_id"])
        if (record.get("user"), record.get("qos"), record.get("account")) != (user, QOS, ACCOUNT):
            raise PermissionError("Original job ownership/allocation differs")
        terminals.append(record)
    return terminals


def capacity(slurm, user, acknowledged):
    """Include pending/unknown reservations even during scheduler visibility lag."""
    user_ids = slurm.user_job_ids(user)
    queue = slurm.queue()
    for job_id in (user_ids | set(acknowledged)) - set(queue):
        queue[job_id] = slurm.show(job_id)
    if any(not row.get("ReqTRES") for row in queue.values()):
        raise RuntimeError("Unknown reservation; refuse submission")
    reserved = sum(
        max(gpu_count(row["ReqTRES"]), gpu_count(row.get("AllocTRES"))) for row in queue.values()
    )
    return len(user_ids | set(acknowledged)), reserved, queue


def main():
    if read(ROOT / "CPU_READY.json")["status"] != "CPU_READY_FOR_G2_PREFLIGHT":
        raise PermissionError("CPU preparation is not complete")
    if read(ROOT / "CPU_TESTS.json")["status"] != "PASS":
        raise PermissionError("Server CPU tests failed")
    if read(ROOT / "DATA/HISTORICAL_WORLD_AUDIT.json")["status"] != "PASS":
        raise PermissionError("Historical world exclusion failed")
    if read(ROOT / "RETENTION/FROZEN.json")["status"] != "FROZEN_BEFORE_NEW_MODEL_OUTPUTS":
        raise PermissionError("Retention subsets are not frozen")
    slurm = Slurm()
    launcher = Path(os.environ["SRF5_LAUNCHER_CODE"]).resolve(strict=True)
    if not (launcher / "src/sr_f5/dual_gpu.py").is_file():
        raise FileNotFoundError("Dual-GPU launcher source is missing")
    with (ROOT / "G2_SUBMIT.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = ROOT / "G2_SUBMISSION_DUAL_GPU.json"
        if receipt_path.exists():
            raise RuntimeError("Submission receipt already exists; inspect rather than duplicate")
        user = getpass.getuser()
        terminal = verify_previous_jobs(slurm, read(ROOT / "G2_SUBMISSION.json"), user)
        receipt = dict(
            at=now(),
            status="SUBMITTING",
            source_code=str(CODE),
            launcher_code=str(launcher),
            replaced_jobs_terminal=terminal,
            jobs=[],
            all_qos_job_limit=5,
            effective_total_gpu_limit=5,
            registered_workers=4,
            registered_jobs=2,
            gpus_per_job=2,
        )
        write(receipt_path, receipt)
        acknowledged = set()
        for pair in range(2):
            jobs, reserved, queue = capacity(slurm, user, acknowledged)
            if jobs + 1 > 5 or reserved + 2 > 5:
                receipt.update(
                    status="CAPACITY_BLOCKED",
                    observed_jobs=jobs,
                    observed_reserved_gpus=reserved,
                )
                write(receipt_path, receipt)
                raise RuntimeError("Capacity unavailable; no unrelated jobs changed")
            name = f"LH-SRF5-G2-P{pair}"
            attempt = dict(
                pair=pair,
                shards=[2 * pair, 2 * pair + 1],
                name=name,
                status="SUBMISSION_UNKNOWN",
                before=queue,
            )
            receipt["jobs"].append(attempt)
            write(receipt_path, receipt)
            command = [
                "sbatch",
                "--parsable",
                f"--job-name={name}",
                f"--output={ROOT}/slurm/G2-P{pair}-%j.log",
                str(launcher / "scripts/sr_f5/calibration.sbatch"),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=dict(
                    os.environ,
                    SRF5_CODE=str(CODE),
                    SRF5_LAUNCHER_CODE=str(launcher),
                    SRF5_PAIR=str(pair),
                ),
            )
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError("Ambiguous submission; preserve UNKNOWN and diagnose")
            attempt.update(job_id=job_id, status="SUBMITTED", scheduler_message=result.stderr)
            acknowledged.add(job_id)
            write(receipt_path, receipt)
            actual = slurm.show(job_id)
            if (
                actual.get("JobName") != name
                or actual.get("QOS") != QOS
                or actual.get("Account") != ACCOUNT
            ):
                raise RuntimeError("Submitted job identity mismatch")
            if gpu_count(actual.get("ReqTRES")) != 2:
                raise RuntimeError("Unexpected GPU allocation")
            attempt["verified"] = actual
            write(receipt_path, receipt)
        receipt["status"] = "TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED"
        write(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "jobs": [
                        {k: j[k] for k in ("job_id", "name", "shards")} for j in receipt["jobs"]
                    ],
                }
            )
        )


if __name__ == "__main__":
    main()
