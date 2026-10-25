"""CPU decision job: completed G2 -> audited gate -> frozen G3 submission.

Scientific failure remains a decision for the documented diagnosis/revision
branch. This controller never changes the recipe or launches formal training.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from submit_g3 import submit

from sr_f4.common import now, read, write
from sr_f4.pool import ACCOUNT, QOS, Slurm
from sr_f5.calibration import aggregate
from sr_f5.pilot import freeze

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
TESTS = [
    "tests/sr_f5/test_pilot.py",
    "tests/sr_f5/test_pilot_pool.py",
    "tests/sr_f5/test_independent_review.py",
    "tests/sr_f5/test_training.py",
    "tests/sr_f5/test_advance_development.py",
]


def census_terminal(slurm, receipt, root, user):
    jobs = receipt.get("jobs", [])
    if (
        receipt.get("status") != "TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED"
        or len(jobs) != 2
        or {j.get("pair") for j in jobs} != {0, 1}
        or len({j.get("job_id") for j in jobs}) != 2
    ):
        raise PermissionError("G2 submission is incomplete or ambiguous")
    active = slurm.user_job_ids(user)
    states = []
    for job in jobs:
        job_id = job.get("job_id", "")
        if not job_id.isdecimal():
            raise PermissionError("Invalid G2 job ID")
        verified = job.get("verified", {})
        if (
            verified.get("JobId") != job_id
            or verified.get("JobName") != f"LH-SRF5-G2-P{job['pair']}"
            or verified.get("StdOut") != str(root / "slurm" / f"G2-P{job['pair']}-{job_id}.log")
        ):
            raise PermissionError("G2 job is not bound to this run root")
        if job_id in active:
            return "WAIT_FOR_G2", []
        state = slurm.terminal(job_id)
        if not state:
            return "WAIT_FOR_ACCOUNTING", []
        if (state.get("user"), state.get("account"), state.get("qos")) != (user, ACCOUNT, QOS):
            raise PermissionError("G2 terminal owner differs")
        states.append(state)
    if any(s.get("state") != "COMPLETED" or s.get("exit_code") != "0:0" for s in states):
        return "G2_TECHNICAL_FAILURE_REQUIRES_DIAGNOSIS", states
    return "G2_JOBS_COMPLETE", states


def advance(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("A full committed revision is required")
    root, code = args.root.resolve(strict=True), args.code.resolve(strict=True)
    if not root.is_relative_to(PROJECT) or code != Path(__file__).resolve().parents[2]:
        raise PermissionError("Controller must use its declared project deployment")
    if not code.is_relative_to(PROJECT) or Path(args.receipt).name != args.receipt:
        raise PermissionError("Receipt and code must remain inside our project")
    directory = root / "CONTINUATION" / args.revision
    directory.mkdir(parents=True, exist_ok=True)
    with (root / "DEVELOPMENT_ADVANCE.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        slurm, user = Slurm(), getpass.getuser()
        status, terminals = census_terminal(slurm, read(root / args.receipt), root, user)
        result = dict(at=now(), status=status, code_revision=args.revision, terminals=terminals)
        if status != "G2_JOBS_COMPLETE":
            write(directory / "DECISION.json", result)
            return result
        # Run affected CPU checks once in the actual server environment. No
        # generation smoke test, base-weight scan or old full-suite repetition.
        test_path = directory / "CPU_TESTS.json"
        if test_path.exists():
            tests = read(test_path)
        else:
            with (directory / "CPU_TESTS.log").open("w") as stream:
                process = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                    cwd=code,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            tests = dict(
                status="PASS" if process.returncode == 0 else "FAIL",
                code_revision=args.revision,
                tests=TESTS,
                exit_code=process.returncode,
            )
            write(test_path, tests)
        if tests.get("status") != "PASS" or tests.get("code_revision") != args.revision:
            result["status"] = "SERVER_CPU_CHECKS_FAILED"
            write(directory / "DECISION.json", result)
            return result
        report = aggregate(root)
        result["g2_gate"] = report["gate"]
        if report["gate"]["status"] != "PASS":
            result["status"] = "G2_FEASIBILITY_FAILURE_REQUIRES_DOCUMENTED_DECISION"
            write(directory / "DECISION.json", result)
            return result
        freeze(root, args.revision)
        submission = root / "G3_SUBMISSION_DUAL_GPU.json"
        if submission.exists():
            prior = read(submission)
            status = (
                "G3_ALREADY_SUBMITTED_DO_NOT_DUPLICATE"
                if prior.get("status") == "TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED"
                else "G3_SUBMISSION_REQUIRES_DIAGNOSIS_NO_RETRY"
            )
            result.update(status=status, submission=prior)
        else:
            receipt = submit(SimpleNamespace(root=root, code=code, revision=args.revision))
            result.update(status="G3_SUBMITTED_AFTER_REAL_G2_PASS", submission=receipt)
        write(directory / "DECISION.json", result)
        return result


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--receipt", required=True)
    print(json.dumps(advance(parser.parse_args()), indent=2), flush=True)
