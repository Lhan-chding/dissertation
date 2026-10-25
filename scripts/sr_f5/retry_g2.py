"""One explicit retry of the diagnosed G2 pre-generation launcher failure.

Original submissions, worker logs and scientific inputs remain untouched. An
existing retry receipt blocks all further attempts, including ambiguous sbatch
outcomes; diagnosis must precede any later recovery mechanism.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import re
import runpy
import subprocess
from pathlib import Path

from sr_f4.common import now, read, write
from sr_f4.pool import ACCOUNT, QOS, Slurm, gpu_count

_original = runpy.run_path(str(Path(__file__).with_name("submit_g2.py")))
capacity = _original["capacity"]
ROOT = _original["ROOT"]
CODE = _original["CODE"]
PROJECT = ROOT.parent
EXPECTED_FAILURE = "Audited CUBLAS_WORKSPACE_CONFIG=:4096:8 is required"


def verify_no_model_records(root):
    for relative in ("G2/raw", "G2/attempts", "G2/groups"):
        directory = root / relative
        if directory.is_symlink():
            raise PermissionError("Model-output path is a symlink")
        if directory.exists() and any(
            not p.is_dir() or p.is_symlink() for p in directory.rglob("*")
        ):
            raise PermissionError("Model records exist; this startup-only retry is forbidden")


def verify_failed_pairs(slurm, original, user, root):
    if original.get("status") != "TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED":
        raise PermissionError("Original dual submission was not fully verified")
    if original.get("source_code") != str(CODE):
        raise PermissionError("Unexpected original science deployment")
    old_launcher = Path(original["launcher_code"])
    if not old_launcher.is_absolute() or not old_launcher.is_relative_to(PROJECT):
        raise PermissionError("Original launcher is outside our project")
    jobs = original.get("jobs", [])
    if len(jobs) != 2 or {j.get("pair") for j in jobs} != {0, 1}:
        raise ValueError("Expected exactly the two registered shard pairs")
    ids = [j.get("job_id") for j in jobs]
    if len(set(ids)) != 2 or any(not isinstance(j, str) or not j.isdecimal() for j in ids):
        raise ValueError("Ambiguous original job IDs")
    active = slurm.user_job_ids(user) | set(slurm.queue())
    terminals, evidence = [], []
    for job in sorted(jobs, key=lambda j: j["pair"]):
        pair, job_id = job["pair"], job["job_id"]
        name = f"LH-SRF5-G2-P{pair}"
        actual = job.get("verified", {})
        if (
            job.get("name") != name
            or job.get("status") != "SUBMITTED"
            or job.get("shards") != [2 * pair, 2 * pair + 1]
            or actual.get("JobId") != job_id
            or actual.get("JobName") != name
            or actual.get("QOS") != QOS
            or actual.get("Account") != ACCOUNT
            or actual.get("WorkDir") != str(CODE)
            or actual.get("Command") != str(old_launcher / "scripts/sr_f5/calibration.sbatch")
            or actual.get("StdOut") != str(root / "slurm" / f"G2-P{pair}-{job_id}.log")
            or gpu_count(actual.get("ReqTRES")) != 2
        ):
            raise PermissionError("Original job is not bound to this root and shard pair")
        if job_id in active:
            raise RuntimeError("Original job remains queued or running")
        terminal = slurm.terminal(job_id)
        if not terminal or terminal.get("state") != "FAILED":
            raise RuntimeError("Original job is not a verified FAILED terminal")
        if (terminal.get("user"), terminal.get("account"), terminal.get("qos")) != (
            user,
            ACCOUNT,
            QOS,
        ):
            raise PermissionError("Original terminal ownership differs")
        terminals.append(terminal)
        for shard in job["shards"]:
            path = root / "slurm" / f"G2-W{shard}-{job_id}.log"
            if path.is_symlink() or path.stat().st_size > 128 * 1024:
                raise PermissionError("Unexpected startup failure log")
            text = path.read_text()
            if EXPECTED_FAILURE not in text or "PermissionError" not in text:
                raise PermissionError(
                    "Failure is not the diagnosed pre-generation environment error"
                )
            evidence.append(dict(job_id=job_id, shard=shard, path=str(path), text=text))
    return terminals, evidence


def deployment(path, required):
    path = Path(path)
    canonical = path.resolve(strict=True)
    if path != canonical or not canonical.is_relative_to(PROJECT):
        raise PermissionError("Deployment must be a canonical directory within our project")
    for relative in required:
        file = canonical / relative
        if not file.is_file() or file.resolve() != file:
            raise FileNotFoundError("Missing regular deployment source: " + str(file))
    return canonical


def retry(args, *, slurm=None, user=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.attempt_id):
        raise ValueError("Invalid attempt ID")
    if not args.reason.strip():
        raise ValueError("An explicit diagnosed retry reason is required")
    launcher = deployment(
        args.launcher_code, ("src/sr_f5/dual_gpu.py", "scripts/sr_f5/calibration.sbatch")
    )
    science = deployment(args.science_code, ("src/sr_f5/calibration.py",))
    slurm, user = slurm or Slurm(), user or getpass.getuser()
    with (ROOT / "G2_SUBMIT.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(ROOT.glob("G2_RETRY_*.json")):
            raise RuntimeError("Retry receipt already exists; no duplicate or blind retry")
        for name, expected in (
            ("CPU_READY.json", "CPU_READY_FOR_G2_PREFLIGHT"),
            ("CPU_TESTS.json", "PASS"),
            ("DATA/HISTORICAL_WORLD_AUDIT.json", "PASS"),
            ("RETENTION/FROZEN.json", "FROZEN_BEFORE_NEW_MODEL_OUTPUTS"),
        ):
            if read(ROOT / name).get("status") != expected:
                raise PermissionError("Missing registered prerequisite: " + name)
        original = read(ROOT / "G2_SUBMISSION_DUAL_GPU.json")
        terminals, evidence = verify_failed_pairs(slurm, original, user, ROOT)
        verify_no_model_records(ROOT)
        if str(launcher) == original["launcher_code"]:
            raise PermissionError("Retry requires a newly versioned launcher")
        path = ROOT / f"G2_RETRY_{args.attempt_id}.json"
        receipt = dict(
            at=now(),
            status="SUBMITTING",
            attempt_id=args.attempt_id,
            reason=args.reason,
            previous_receipt="G2_SUBMISSION_DUAL_GPU.json",
            previous_jobs_terminal=terminals,
            failure_evidence=evidence,
            original_source_code=original["source_code"],
            source_code=str(science),
            science_code_changed=str(science) != original["source_code"],
            launcher_code=str(launcher),
            model_records_before_retry="NONE",
            registered_workers=4,
            registered_jobs=2,
            gpus_per_job=2,
            all_qos_job_limit=5,
            effective_total_gpu_limit=5,
            jobs=[],
        )
        write(path, receipt)
        acknowledged = set()
        for pair in range(2):
            jobs, reserved, queue = capacity(slurm, user, acknowledged)
            if jobs + 1 > 5 or reserved + 2 > 5:
                receipt.update(
                    status="CAPACITY_BLOCKED", observed_jobs=jobs, observed_reserved_gpus=reserved
                )
                write(path, receipt)
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
            write(path, receipt)
            result = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    f"--job-name={name}",
                    f"--output={ROOT}/slurm/G2-P{pair}-%j.log",
                    str(launcher / "scripts/sr_f5/calibration.sbatch"),
                ],
                capture_output=True,
                text=True,
                check=True,
                env=dict(
                    os.environ,
                    SRF5_CODE=str(science),
                    SRF5_LAUNCHER_CODE=str(launcher),
                    SRF5_PAIR=str(pair),
                ),
                cwd=science,
            )
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdecimal() or job_id in acknowledged:
                raise RuntimeError("Ambiguous submission; preserve UNKNOWN and diagnose")
            attempt.update(job_id=job_id, status="SUBMITTED", scheduler_message=result.stderr)
            acknowledged.add(job_id)
            write(path, receipt)
            actual = slurm.show(job_id)
            if (
                actual.get("JobName") != name
                or actual.get("QOS") != QOS
                or actual.get("Account") != ACCOUNT
                or gpu_count(actual.get("ReqTRES")) != 2
                or actual.get("WorkDir") != str(science)
            ):
                raise RuntimeError("Submitted retry identity mismatch")
            attempt["verified"] = actual
            write(path, receipt)
        receipt["status"] = "TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED"
        write(path, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-code", type=Path, required=True)
    parser.add_argument("--science-code", type=Path, default=CODE)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--reason", required=True)
    receipt = retry(parser.parse_args())
    print(
        json.dumps(
            dict(
                status=receipt["status"],
                jobs=[
                    {key: row[key] for key in ("job_id", "name", "shards")}
                    for row in receipt["jobs"]
                ],
            )
        )
    )


if __name__ == "__main__":
    main()
