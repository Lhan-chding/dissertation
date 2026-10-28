"""Singleton two-GPU T03 dispatch after complete support and independent review."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import re
import subprocess
from pathlib import Path

from sr_f6.common import now, read, safe_evidence, write_once
from sr_f6.interface_screen_dispatch import USER, capacity
from sr_f6.transactions import _plain_path

from .source import ROOT
from .t03_access import require


def argv_for(root, context):
    code = Path(context["consumer"]["deployment"])
    producer = context["registration"]["producer"]["deployment"]
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--job-name=LH-SRF6-T03",
        f"--chdir={root}",
        f"--output={root}/slurm/t03-%j.log",
        f"--export=ALL,SRF6_CODE={code},SRF6_PRODUCER={producer},SRF6_ROOT={root}",
        str(code / "scripts/sr_f6_followup/t03.sbatch"),
    ]


def submission_receipt(root, context, spec_ref):
    directory = _plain_path(Path(root) / "T03/SUBMISSIONS")
    require(
        {p.name for p in directory.iterdir()}
        <= {"dispatch.lock", "t03.attempt.json", "t03.result.json"},
        "Unknown submission evidence",
    )
    attempt = read(safe_evidence(root, "T03/SUBMISSIONS/t03.attempt.json"))
    result = read(safe_evidence(root, "T03/SUBMISSIONS/t03.result.json"))
    pin = context["consumer"]["registration_sha256"]
    require(
        attempt.get("status") == "SUBMISSION_INTENT_NOT_COMPLETION"
        and attempt.get("consumer_registration_sha256") == pin
        and attempt.get("spec_ref") == spec_ref
        and attempt.get("argv") == argv_for(root, context),
        "Submission intent source/spec/arguments differ",
    )
    require(
        result.get("status") == "SUBMITTED"
        and result.get("returncode") == 0
        and result.get("consumer_registration_sha256") == pin
        and result.get("spec_ref") == spec_ref
        and isinstance(result.get("job_id"), str)
        and re.fullmatch(r"[0-9]+", result["job_id"])
        and result.get("stdout", "").strip().split(";", 1)[0] == result["job_id"],
        "Submission failed or unknown; no automatic retry",
    )
    return result


def submit(root):
    from . import source
    from .t03 import verified_spec

    root = _plain_path(root)
    require(root == ROOT, "Only the registered followup root is allowed")
    require(getpass.getuser() == USER, "Only the registered scheduler account may submit")
    context = source.require_support(root, purpose="t03")
    require(
        Path(context["root"]) == root
        and Path(context["consumer"]["deployment"]) == Path(__file__).resolve().parents[2],
        "Actual consumer deployment differs",
    )
    _, spec_ref = verified_spec(root, context)
    directory = _plain_path(root / "T03/SUBMISSIONS")
    directory.mkdir(parents=True, exist_ok=True)
    _plain_path(root / "slurm").mkdir(exist_ok=True)
    lock_path = _plain_path(directory / "dispatch.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(
            {p.name for p in directory.iterdir()}
            <= {"dispatch.lock", "t03.attempt.json", "t03.result.json"},
            "Unknown submission evidence",
        )
        attempt_path, result_path = directory / "t03.attempt.json", directory / "t03.result.json"
        if attempt_path.exists() or result_path.exists():
            receipt = submission_receipt(root, context, spec_ref)
            return dict(status="ALREADY_SUBMITTED", job_id=receipt["job_id"])
        for name in ("FAILURES", "RUNS", "SLOTS"):
            path = _plain_path(root / "T03" / name)
            require(
                not path.exists() or not list(path.iterdir()),
                "Preserved/orphan execution blocks submission",
            )
        snapshot = capacity()  # Frozen all-user requested/allocated teacher reservation algorithm.
        if not snapshot["can_submit_two_gpu_job"]:
            return dict(status="NO_LEGAL_CAPACITY", capacity=snapshot)
        fresh = source.require_support(root, purpose="t03")
        require(
            fresh["consumer"] == context["consumer"] and verified_spec(root, fresh)[1] == spec_ref,
            "Bindings changed before submission",
        )
        argv = argv_for(root, context)
        shared = dict(
            consumer_registration_sha256=context["consumer"]["registration_sha256"],
            spec_ref=spec_ref,
        )
        write_once(
            attempt_path,
            dict(
                at=now(),
                status="SUBMISSION_INTENT_NOT_COMPLETION",
                argv=argv,
                capacity=snapshot,
                **shared,
            ),
        )
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        job = result.stdout.strip().split(";", 1)[0]
        valid = result.returncode == 0 and re.fullmatch(r"[0-9]+", job) is not None
        receipt = dict(
            at=now(),
            status="SUBMITTED" if valid else "FAILED_OR_UNKNOWN",
            job_id=job if valid else None,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            **shared,
        )
        write_once(result_path, receipt)
        require(valid, "Submission failed or unknown; preserve intent and do not resubmit")
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    print(json.dumps(submit(parser.parse_args().root)))


if __name__ == "__main__":
    main()
