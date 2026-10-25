"""One-shot submission of the frozen G3 pool after actual complete G2 PASS."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import runpy
import subprocess
from pathlib import Path

from sr_f4.common import now, write
from sr_f4.pool import ACCOUNT, QOS, Slurm, gpu_count
from sr_f5.calibration import digest
from sr_f5.pilot_pool import freeze_pool, verify_code

_original = runpy.run_path(str(Path(__file__).with_name("submit_g2.py")))
capacity = _original["capacity"]
ROOT = _original["ROOT"]
PROJECT = ROOT.parent


def submit(args, *, slurm=None, user=None):
    root = args.root.resolve(strict=True)
    if args.root != root or not root.is_relative_to(PROJECT):
        raise PermissionError("G3 run root must be canonical and inside our project")
    code = verify_code(args.code)
    if not code.is_relative_to(PROJECT) or not (code / "scripts/sr_f5/pilot.sbatch").is_file():
        raise PermissionError("Scientific deployment must be inside our project")
    slurm, user = slurm or Slurm(), user or getpass.getuser()
    with (root / "G3_SUBMIT.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "G3_SUBMISSION_DUAL_GPU.json"
        if path.exists():
            raise PermissionError("G3 submission exists; no duplicate or unknown retry")
        # require_frozen inside freeze_pool rechecks the real G2 metrics and gate.
        plan = freeze_pool(root, args.revision)
        jobs, reserved, _ = capacity(slurm, user, set())
        if jobs + 2 > 5 or reserved + 4 > 5:
            # No sbatch attempt has occurred, so later capacity availability is
            # safe to revisit without introducing an ambiguous submission.
            write(
                root / "G3/CAPACITY_LAST_CHECK.json",
                dict(
                    at=now(),
                    status="WAITING_FOR_CAPACITY",
                    observed_jobs=jobs,
                    observed_reserved_gpus=reserved,
                ),
            )
            raise RuntimeError("Capacity unavailable; no submission attempted")
        receipt = dict(
            at=now(),
            status="SUBMITTING",
            source_code=str(code),
            code_revision=args.revision,
            pool_plan_digest=digest(plan),
            all_qos_job_limit=5,
            effective_total_gpu_limit=5,
            registered_workers=4,
            registered_jobs=2,
            gpus_per_job=2,
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
            name = f"LH-SRF5-G3-P{pair}"
            attempt = dict(
                pair=pair,
                workers=[2 * pair, 2 * pair + 1],
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
                    f"--output={root}/slurm/G3-P{pair}-%j.log",
                    str(code / "scripts/sr_f5/pilot.sbatch"),
                ],
                capture_output=True,
                text=True,
                check=True,
                cwd=code,
                env=dict(
                    os.environ,
                    SRF5_CODE=str(code),
                    SRF5_REVISION=args.revision,
                    SRF5_ROOT=str(root),
                    SRF5_PAIR=str(pair),
                ),
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
                or actual.get("WorkDir") != str(code)
            ):
                raise RuntimeError("Submitted G3 identity mismatch")
            attempt["verified"] = actual
            write(path, receipt)
        receipt["status"] = "TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED"
        write(path, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    receipt = submit(parser.parse_args())
    print(
        json.dumps(
            dict(
                status=receipt["status"],
                jobs=[
                    {key: row[key] for key in ("job_id", "name", "workers")}
                    for row in receipt["jobs"]
                ],
            )
        )
    )


if __name__ == "__main__":
    main()
