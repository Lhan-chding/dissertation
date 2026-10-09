#!/usr/bin/env python3
"""Independently recount and seal the complete registered F2 experiment on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_dev.orchestration import complete_task, digest, fail_task, worker_lease
from mm_dev.release import descriptor, read_json, release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    registration = read_json(args.run_root / "orchestration/REGISTRATION.json")
    if (
        os.environ.get("MM_DEV_TASK_ID") != "RELEASE"
        or not os.environ.get("MM_DEV_ATTEMPT_ID")
        or os.environ.get("MM_DEV_REGISTRATION_HASH") != digest(registration)
        or "RELEASE" not in registration["tasks"]
    ):
        raise PermissionError("RELEASE worker identity is not registered")
    with worker_lease(args.run_root, "RELEASE"):
        try:
            receipt = release(args.plan, args.run_root)
            if (
                receipt.get("status") != "VERIFIED_RELEASE"
                or receipt.get("scientific_success_claimed") is not False
                or receipt.get("next_stage_authorized") is not False
            ):
                raise PermissionError(
                    "An incomplete or scientifically promoted release cannot complete"
                )
            paths = ["verification/INDEPENDENT_RECOUNT.json", "verification/RELEASE_MANIFEST.json"]
            marker = args.run_root / "orchestration/completions/RELEASE.json"
            if marker.exists():
                existing = read_json(marker)
                expected = {
                    name: {k: v for k, v in descriptor(args.run_root, name).items() if k != "path"}
                    for name in paths
                }
                if existing.get("status") != "COMPLETE" or existing.get("artifacts") != expected:
                    raise PermissionError(
                        "Existing RELEASE completion does not bind the verified release"
                    )
            else:
                complete_task(
                    args.run_root,
                    "RELEASE",
                    paths,
                    metadata=dict(
                        release_status=receipt["status"],
                        scientific_success_claimed=False,
                        next_stage_authorized=False,
                    ),
                )
        except Exception as exc:
            fail_task(args.run_root, "RELEASE", str(exc))
            raise
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "files": len(receipt["files"]),
                "scientific_success_claimed": False,
                "next_stage_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
