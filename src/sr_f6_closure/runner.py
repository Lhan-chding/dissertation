"""Single read-only closure over the existing evidence; write only a new result."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sr_f6.common import digest, now

from . import admission_projection, source

SCHEMA = "srf6-independent-technical-closure-v1"


def guard_write(event, args, allowed):
    """Keep Python evidence validation read-only outside the new result/cache."""
    paths = []
    if event == "open":
        path, mode, flags = args
        if isinstance(path, (str, bytes, os.PathLike)) and (
            (mode and any(char in mode for char in "wax+"))
            or (flags and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        ):
            paths = [path]
    elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate"}:
        paths = [args[0]]
        index = 2 if event in {"os.mkdir", "os.chmod"} else 1
        if event != "os.truncate":
            source.require(args[index] in (None, -1), "Descriptor-relative mutations forbidden")
    elif event in {"os.rename", "os.link", "os.symlink"}:
        paths = list(args[:2])
        source.require(
            all(fd in (None, -1) for fd in args[2:]),
            "Descriptor-relative mutations forbidden",
        )
    elif event in {"os.system", "os.fork", "os.forkpty"} or event.startswith(
        ("subprocess.", "socket.connect", "os.exec", "os.posix_spawn", "os.spawn")
    ):
        raise PermissionError("Read-only verifier cannot start external processes/connections")
    for path in paths:
        actual = Path(os.fsdecode(path)).absolute()
        source.require(
            ".." not in actual.parts
            and not any(p.is_symlink() for p in (actual, *actual.parents))
            and actual.resolve(strict=False) == actual
            and any(actual.is_relative_to(root) for root in allowed),
            "Read-only verifier attempted to modify an original or external path",
        )


def execute(revision):
    job = os.environ.get("SLURM_JOB_ID", "")
    source.require(job.isdecimal(), "An actual scheduler CPU allocation is required")
    source.require(
        os.environ.get("SLURM_JOB_NAME") == "LH-SRF6-Closure-Repair"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "Registered zero-GPU technical validation only",
    )
    cache = source.original._path(
        source.PROJECT / "sr_f6_operations_20261027/CLOSURE_REPAIR_V1_20261029" / ("cache_" + job)
    )
    source.require(Path(os.environ["TMPDIR"]) == cache / "tmp", "Private runtime cache required")
    sys.addaudithook(lambda event, args: guard_write(event, args, (source.ROOT, cache)))
    identity = source.register(revision)
    source.original._write(
        source.ROOT / "ATTEMPT.json",
        dict(schema=SCHEMA, at=now(), job_id=job, identity=identity, new_model_calls=0),
    )
    try:
        bundle = admission_projection.evaluate(source.original.ROOT)
        after = source.verify_registered()
        source.require(identity == after, "Source identity changed during technical closure")
        inner = bundle["assessment"]
        numerical = bundle["numerical"]
        if numerical is not None:
            source.require(
                digest(numerical) == inner["numerical_sha256"],
                "Assessment must bind this exact complete numerical verification",
            )
            source.original._write(source.ROOT / "NUMERICAL.json", numerical)
        source.original._write(source.ROOT / "ORIGINAL_RULE_ASSESSMENT.json", inner)
        report = dict(
            schema=SCHEMA,
            status=inner["status"],
            scope="EXISTING_CONFIRMATION_EVIDENCE_ONLY_SEPARATE_PILOT_REQUIRED",
            identity=identity,
            preserved_history=source.history(),
            original_rule_assessment=source.original._reference(
                source.ROOT / "ORIGINAL_RULE_ASSESSMENT.json"
            ),
            numerical=(
                source.original._reference(source.ROOT / "NUMERICAL.json")
                if numerical is not None
                else None
            ),
            old_admission_status="UNKNOWN_NOT_MODIFIED",
            new_model_calls=0,
            optimizer_updates=0,
            **source.PERMISSIONS,
        )
        source.original._write(source.ROOT / "ASSESSMENT.json", report)
        receipt = dict(
            schema=SCHEMA,
            status=report["status"],
            assessment=source.original._reference(source.ROOT / "ASSESSMENT.json"),
            identity=identity,
            gate_status=(
                "CORRECTED_CONFIRMATION_FACTS_SEALED"
                if report["status"] == "CONFIRMATION_GATES_PASS"
                else "NOT_SIGNED"
            ),
            **source.PERMISSIONS,
        )
        source.original._write(source.ROOT / "RECEIPT.json", receipt)
        return receipt
    except Exception as exc:
        source.original._write(
            source.ROOT / "FAILURE.json",
            dict(
                schema=SCHEMA, at=now(), error=f"{type(exc).__name__}: {exc}", **source.PERMISSIONS
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    result = execute(parser.parse_args().revision)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "CONFIRMATION_GATES_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
