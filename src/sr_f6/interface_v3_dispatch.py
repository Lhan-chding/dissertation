"""Dispatch only registered v3 stage/parent jobs within live teacher limits."""

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

ROOT = PROJECT / "sr_f6_interface_v3_20261028"
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


def _no_failures(root):
    for number in (1, 2):
        stage_root = _canonical(root / f"STAGE{number}")
        failures = _canonical(stage_root / "SCREEN/failures", exists=False)
        if list(failures.glob("*.json")):
            raise PermissionError("Preserved technical failure blocks further submissions")


def submit(root, *, stage=1):
    from .interface_v3_registration import verify_registered

    if type(stage) is not int or stage not in (1, 2):
        raise ValueError("Only registered stage 1 or 2 may be submitted")
    root = _canonical(root)
    if root != ROOT:
        raise PermissionError("Only the registered v3 base root is allowed")
    if getpass.getuser() != USER:
        raise PermissionError("Only the registered shared account may submit")
    _no_failures(root)
    if stage == 1:
        parents = PARENTS
    else:
        from .interface_v3_screen import stage2_authorized

        parents = stage2_authorized(root)
        if (
            not isinstance(parents, list)
            or not parents
            or parents != sorted(set(parents))
            or any(parent not in PARENTS for parent in parents)
        ):
            raise PermissionError("Stage 2 requires explicitly authorized gray parents")
    stage_root = _canonical(root / f"STAGE{stage}")
    code = Path(__file__).resolve(strict=True).parents[2]
    directory = _canonical(stage_root / "SUBMISSIONS", exists=False)
    directory.mkdir(exist_ok=True)
    logdir = _canonical(stage_root / "slurm", exists=False)
    logdir.mkdir(exist_ok=True)
    lock = _canonical(directory / "dispatch.lock", exists=False)
    outcomes = []
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        for parent in parents:
            _no_failures(root)
            context = verify_registered(root, stage=stage, parent=parent, require_preflight=True)
            if (
                context["stage"] != stage
                or Path(context["root"]) != stage_root
                or Path(context["registration_root"]) != root
                or Path(context["deployment"]) != code
            ):
                raise PermissionError("Registered execution source or stage root differs")
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
                    attempt.get("stage") != stage
                    or attempt.get("parent") != parent
                    or attempt.get("registration_sha256") != pin
                    or prior.get("stage") != stage
                    or prior.get("parent") != parent
                    or prior.get("status") != "SUBMITTED"
                    or not isinstance(prior.get("job_id"), str)
                    or re.fullmatch(r"[0-9]+", prior["job_id"]) is None
                ):
                    raise PermissionError("Prior submission failed or changed; no automatic retry")
                outcomes.append(
                    dict(
                        stage=stage,
                        parent=parent,
                        status="ALREADY_SUBMITTED",
                        job_id=prior["job_id"],
                    )
                )
                continue
            snapshot = capacity()
            if not snapshot["can_submit_two_gpu_job"]:
                outcomes.append(
                    dict(stage=stage, parent=parent, status="NO_LEGAL_CAPACITY", capacity=snapshot)
                )
                break
            argv = [
                "sbatch",
                "--parsable",
                "--no-requeue",
                f"--job-name=LH-SRF6-V3-S{stage}-{parent}",
                f"--chdir={stage_root}",
                f"--output={logdir}/screen-{parent}-%j.log",
                f"--export=ALL,SRF6_CODE={code},SRF6_ROOT={root},SRF6_STAGE={stage},SRF6_PARENT={parent}",
                str(code / "scripts/sr_f6/interface_v3.sbatch"),
            ]
            write_once(
                attempt_path,
                dict(
                    at=now(),
                    stage=stage,
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
                stage=stage,
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
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    print(json.dumps(submit(args.root, stage=args.stage), indent=2))


if __name__ == "__main__":
    main()
