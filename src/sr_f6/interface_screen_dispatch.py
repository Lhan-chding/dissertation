"""Submit the fixed screen within live teacher GPU and per-user slot limits."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import re
import subprocess
from pathlib import Path

from .common import now, read, write_once
from .source_binding import PROJECT

QOS = "soujanya-poria-startfund-2026-03"
USER = "varun024"


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Scheduler query failed: {args}: {result.stderr}")
    return result.stdout.strip()


def _gpu(tres):
    values = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
    relevant = {k: v for k, v in values.items() if k.startswith("gres/gpu")}
    if not relevant or any(not v.isdecimal() for v in relevant.values()):
        raise PermissionError("Unknown GPU reservation; do not assume it is zero")
    if any(k not in ("gres/gpu", "gres/gpu:pro6000") for k in relevant):
        raise PermissionError("Unexpected GPU type in teacher reservation")
    return max(map(int, relevant.values()))


def capacity(query=command):
    """All teacher jobs reserve max(requested, allocated), including pending."""
    qos_raw = query(["sacctmgr", "-nP", "show", "qos", QOS, "format=Name,MaxSubmitPU,GrpTRES"])
    lines = [line.rstrip("|").split("|") for line in qos_raw.splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 3:
        raise PermissionError("Ambiguous live QoS definition")
    name, slots, tres = lines[0]
    limits = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
    if name != QOS or slots != "5" or limits.get("gres/gpu:pro6000") != "8":
        raise PermissionError("Registered teacher limits differ from live scheduler")
    queue_raw = query(["squeue", "-h", "-q", QOS, "-o", "%i|%u|%j|%T|%q"])
    jobs, seen = [], set()
    for line in queue_raw.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 5:
            raise PermissionError("Ambiguous teacher queue row")
        job_id, user, name, state, qos = parts
        if not re.fullmatch(r"[0-9]+", job_id) or job_id in seen or qos != QOS:
            raise PermissionError("Unknown/duplicate/array reservation requires inspection")
        seen.add(job_id)
        detail = query(["scontrol", "show", "job", "-o", job_id])
        fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)", detail))
        if fields.get("JobId") != job_id or fields.get("QOS") != QOS:
            raise PermissionError("Job disappeared or changed during capacity snapshot")
        reserved = _gpu(fields.get("ReqTRES", ""))
        allocated = fields.get("AllocTRES", "")
        if allocated and allocated != "(null)":
            reserved = max(reserved, _gpu(allocated))
        jobs.append(
            dict(
                job_id=job_id,
                user=user,
                name=name,
                state=state,
                reserved_gpus=reserved,
                detail=detail,
            )
        )
    used_gpu = sum(row["reserved_gpus"] for row in jobs)
    used_slots = sum(row["user"] == USER for row in jobs)
    return dict(
        at=now(),
        qos=QOS,
        qos_raw=qos_raw,
        queue_raw=queue_raw,
        jobs=jobs,
        gpu_limit=8,
        submit_limit=5,
        reserved_gpus=used_gpu,
        submitted_by_user=used_slots,
        can_submit_two_gpu_job=used_gpu + 2 <= 8 and used_slots + 1 <= 5,
        other_qos="EXCLUDED_FROM_TEACHER_LIMITS",
    )


def submit(root):
    from .interface_screen_registration import verify_registered

    root = Path(root)
    if (
        not root.is_absolute()
        or root.resolve(strict=True) != root
        or not root.is_relative_to(PROJECT)
        or root == PROJECT
    ):
        raise PermissionError("Canonical project descendant required")
    if getpass.getuser() != USER:
        raise PermissionError("Only the registered shared account may submit")
    for path in (root, *root.parents):
        if path.is_symlink():
            raise PermissionError("No symlinked submission paths")
        if path == PROJECT:
            break
    context = verify_registered(root, require_preflight=True)
    if list((root / "SCREEN/failures").glob("*.json")):
        raise PermissionError("Resolve preserved technical failures before further submissions")
    code = Path(__file__).resolve().parents[2]
    directory = root / "SUBMISSIONS"
    if directory.is_symlink():
        raise PermissionError("Symlinked submission directory")
    directory.mkdir(exist_ok=True)
    logdir = root / "slurm"
    if logdir.is_symlink():
        raise PermissionError("Symlinked scheduler log directory")
    logdir.mkdir(exist_ok=True)
    lock = directory / "dispatch.lock"
    if lock.is_symlink():
        raise PermissionError("Symlinked scheduler lock")
    outcomes = []
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        for parent in ("b0", "b1", "b2"):
            attempt_path = directory / f"{parent}.attempt.json"
            result_path = directory / f"{parent}.result.json"
            if attempt_path.is_symlink() or result_path.is_symlink():
                raise PermissionError("Symlinked submission evidence")
            if attempt_path.exists():
                if not result_path.exists():
                    raise PermissionError("Prior submission outcome unknown; inspect manually")
                prior = read(result_path)
                if prior.get("status") != "SUBMITTED":
                    raise PermissionError("Prior submission failed; no automatic retry")
                outcomes.append(
                    dict(parent=parent, status="ALREADY_SUBMITTED", **{"job_id": prior["job_id"]})
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
                f"--job-name=LH-SRF6-Screen-{parent}",
                f"--chdir={root}",
                f"--output={logdir}/screen-{parent}-%j.log",
                f"--export=ALL,SRF6_CODE={code},SRF6_ROOT={root},SRF6_PARENT={parent}",
                str(code / "scripts/sr_f6/interface_screen.sbatch"),
            ]
            write_once(
                attempt_path,
                dict(
                    at=now(),
                    parent=parent,
                    argv=argv,
                    capacity=snapshot,
                    registration_sha256=context["registration_sha256"],
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
