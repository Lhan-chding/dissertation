"""Dispatch only registered confirmation parent jobs within live teacher limits."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import re
import subprocess
from pathlib import Path

from .common import now, read, write_once
from .interface_screen_dispatch import USER, capacity
from .source_binding import PROJECT

ROOT = PROJECT / "sr_f6_confirm_v3_20261028_r1"
PARENTS = ("b0", "b1", "b2")


def _canonical(path, *, exists=True):
    path = Path(path)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not path.is_relative_to(PROJECT)
        or path.resolve(strict=exists) != path
    ):
        raise PermissionError("Canonical project descendant required")
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise PermissionError("No symlinked submission paths")
        if candidate == PROJECT:
            break
    return path


def _job_states(job_ids):
    """Filter the current user queue, then resolve missing recorded IDs via sacct."""
    if not job_ids or any(re.fullmatch(r"[0-9]+", job) is None for job in job_ids):
        raise PermissionError("Exact registered job IDs required")
    active = subprocess.run(
        ["squeue", "--noheader", "--user", USER, "--format=%i|%T"],
        capture_output=True,
        text=True,
        check=False,
    )
    if active.returncode:
        raise PermissionError("Cannot determine current registered job states")
    states = {}
    for line in active.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 2:
            raise PermissionError("Unexpected scheduler identity/state response")
        # Slurm may purge terminal jobs from squeue entirely. Querying those
        # IDs directly can return Invalid job id; unrelated current jobs are
        # read only and do not become evidence for this project's submissions.
        if parts[0] not in job_ids:
            continue
        if parts[0] in states or not parts[1].strip():
            raise PermissionError("Unexpected scheduler identity/state response")
        states[parts[0]] = (parts[1].strip(), None)
    missing = set(job_ids) - states.keys()
    if missing:
        history = subprocess.run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--jobs",
                ",".join(sorted(missing)),
                "--format=JobIDRaw,State,ExitCode",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if history.returncode:
            raise PermissionError("Cannot determine terminal registered job states")
        for line in history.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 3 or parts[0] not in missing or parts[0] in states:
                raise PermissionError("Unexpected accounting identity/state response")
            if not parts[1].strip():
                raise PermissionError("Empty terminal job state")
            states[parts[0]] = (parts[1].split()[0].rstrip("+"), parts[2])
    if set(states) != set(job_ids):
        raise PermissionError("Registered job state is unknown; no further submission")
    return states


def _no_failures(root):
    failures = _canonical(root / "CONFIRM/failures", exists=False)
    if list(failures.glob("*.json")):
        raise PermissionError("Preserved technical failure blocks further submissions")
    slots = _canonical(root / "CONFIRM/execution_slots", exists=False)
    for path in slots.glob("*.json"):
        path = _canonical(path)
        slot = read(path)
        parent, shard, task = slot.get("parent"), slot.get("shard"), slot.get("task_id")
        if (
            parent not in PARENTS
            or type(shard) is not int
            or shard not in (0, 1)
            or path.name != f"{parent}-{shard}.json"
            or not isinstance(task, str)
            or re.fullmatch(rf"LH-SRF6-Confirm-[0-9]+-{parent}-{shard}", task) is None
        ):
            raise PermissionError("Invalid consumed confirmation execution slot")
        execution = _canonical(root / f"CONFIRM/executions/{task}.json", exists=False)
        if not execution.is_file() or read(execution) != slot:
            raise PermissionError("Consumed slot has no matching execution receipt")
    directory = _canonical(root / "SUBMISSIONS", exists=False)
    jobs = {}
    for parent in PARENTS:
        attempt_path = _canonical(directory / f"{parent}.attempt.json", exists=False)
        result_path = _canonical(directory / f"{parent}.result.json", exists=False)
        if result_path.exists() and not attempt_path.exists():
            raise PermissionError("Orphan submission receipt requires inspection")
        if not attempt_path.exists():
            continue
        if not result_path.exists():
            raise PermissionError("Prior submission outcome unknown; inspect manually")
        attempt, result = read(attempt_path), read(result_path)
        job = result.get("job_id")
        if (
            attempt.get("parent") != parent
            or result.get("parent") != parent
            or attempt.get("status") != "SUBMISSION_INTENT_NOT_COMPLETION"
            or result.get("status") != "SUBMITTED"
            or result.get("returncode") != 0
            or not isinstance(job, str)
            or re.fullmatch(r"[0-9]+", job) is None
            or result.get("stdout", "").strip().split(";", 1)[0] != job
        ):
            raise PermissionError("Prior submission failed or changed; no automatic retry")
        jobs[parent] = job
    if not jobs:
        return
    if len(set(jobs.values())) != len(jobs):
        raise PermissionError("Different parents cannot share one scheduler job")
    states = _job_states(set(jobs.values()))
    for parent, job in jobs.items():
        state, exit_code = states[job]
        if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}:
            continue
        if state != "COMPLETED" or exit_code != "0:0":
            raise PermissionError(f"Registered job {job} failed or is unknown: {state}")
        for shard in (0, 1):
            task = f"LH-SRF6-Confirm-{job}-{parent}-{shard}"
            worker_path = _canonical(root / f"CONFIRM/workers/{task}.json", exists=False)
            slot_path = _canonical(
                root / f"CONFIRM/execution_slots/{parent}-{shard}.json", exists=False
            )
            if not worker_path.is_file() or not slot_path.is_file():
                raise PermissionError("Terminal job lacks its complete original worker evidence")
            worker, slot = read(worker_path), read(slot_path)
            if (
                worker.get("status") != "COMPLETE"
                or worker.get("task_id") != task
                or worker.get("parent") != parent
                or worker.get("shard") != shard
                or slot.get("task_id") != task
            ):
                raise PermissionError("Terminal job worker/slot evidence differs")


def submit(root):
    from .confirm_registration import verify_registered

    root = _canonical(root)
    if root != ROOT:
        raise PermissionError("Only the registered confirmation base root is allowed")
    if getpass.getuser() != USER:
        raise PermissionError("Only the registered shared account may submit")
    _no_failures(root)
    code = Path(__file__).resolve(strict=True).parents[2]
    directory = _canonical(root / "SUBMISSIONS", exists=False)
    directory.mkdir(exist_ok=True)
    logdir = _canonical(root / "slurm", exists=False)
    logdir.mkdir(exist_ok=True)
    lock = _canonical(directory / "dispatch.lock", exists=False)
    outcomes = []
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        for parent in PARENTS:
            _no_failures(root)
            context = verify_registered(root, parent=parent, require_preflight=True)
            if (
                Path(context["root"]) != root
                or Path(context["registration_root"]) != root
                or Path(context["deployment"]) != code
            ):
                raise PermissionError("Registered execution source or root differs")
            pin = context["registration_sha256"]
            attempt_path = _canonical(directory / f"{parent}.attempt.json", exists=False)
            result_path = _canonical(directory / f"{parent}.result.json", exists=False)
            if result_path.exists() and not attempt_path.exists():
                raise PermissionError("Orphan submission receipt requires inspection")
            if attempt_path.exists():
                if not result_path.exists():
                    raise PermissionError("Prior submission outcome unknown; inspect manually")
                attempt, prior = read(attempt_path), read(result_path)
                if (
                    attempt.get("parent") != parent
                    or attempt.get("registration_sha256") != pin
                    or prior.get("parent") != parent
                    or prior.get("status") != "SUBMITTED"
                    or not isinstance(prior.get("job_id"), str)
                    or re.fullmatch(r"[0-9]+", prior["job_id"]) is None
                ):
                    raise PermissionError("Prior submission failed or changed; no automatic retry")
                outcomes.append(
                    dict(
                        parent=parent,
                        status="ALREADY_SUBMITTED",
                        job_id=prior["job_id"],
                    )
                )
                continue
            snapshot = capacity()
            if not snapshot["can_submit_two_gpu_job"]:
                outcomes.append(dict(parent=parent, status="NO_LEGAL_CAPACITY", capacity=snapshot))
                break
            argv = [
                "sbatch",
                "--parsable",
                "--no-requeue",
                f"--job-name=LH-SRF6-Confirm-{parent}",
                f"--chdir={root}",
                f"--output={logdir}/confirm-{parent}-%j.log",
                f"--export=ALL,SRF6_CODE={code},SRF6_ROOT={root},SRF6_PARENT={parent}",
                str(code / "scripts/sr_f6/confirm.sbatch"),
            ]
            write_once(
                attempt_path,
                dict(
                    at=now(),
                    parent=parent,
                    argv=argv,
                    capacity=snapshot,
                    registration_sha256=pin,
                    status="SUBMISSION_INTENT_NOT_COMPLETION",
                ),
            )
            result = subprocess.run(argv, capture_output=True, text=True, check=False)
            job_id = result.stdout.strip().split(";", 1)[0]
            valid = result.returncode == 0 and re.fullmatch(r"[0-9]+", job_id)
            receipt = dict(
                at=now(),
                parent=parent,
                status="SUBMITTED" if valid else "FAILED_OR_UNKNOWN",
                job_id=job_id if valid else None,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            write_once(result_path, receipt)
            outcomes.append(receipt)
            if not valid:
                raise RuntimeError("Submission failed or ambiguous; preserve attempt and inspect")
    return outcomes


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.root), indent=2))


if __name__ == "__main__":
    main()
