#!/usr/bin/env python3
"""Execute the separately frozen nonzero visual SFT recovery test once."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_core.execution import exclusive_json
from mm_core.nonzero_recovery import (
    NonzeroLedger,
    compare_nonzero,
    require_nonzero_allocation,
    run_nonzero_segment,
    verify_nonzero_gate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--segment", choices=("continuous", "interrupt", "resume"))
    args = parser.parse_args()
    root = args.run_root.resolve()
    verify_nonzero_gate(root)
    require_nonzero_allocation(root)
    if args.segment:
        branch, start, stop = dict(
            continuous=("continuous", 0, 4), interrupt=("resumed", 0, 2), resume=("resumed", 2, 4)
        )[args.segment]
        run_nonzero_segment(root, branch, start, stop)
        return
    exclusive_json(
        root / "engineering/NONZERO_STARTED.json", dict(status="RUNNING", pid=os.getpid())
    )
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
    began = time.monotonic()
    try:
        try:
            for segment in ("continuous", "interrupt", "resume"):
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--run-root",
                        str(root),
                        "--segment",
                        segment,
                    ],
                    check=True,
                    env=env,
                )
        finally:
            NonzeroLedger(root).reserve(
                "allocated_gpu_hours",
                (time.monotonic() - began) / 3600,
                dict(measure="wrapper_wall_time_times_one_gpu", excludes_scheduler_startup=True),
            )
        if compare_nonzero(root)["status"] != "PASS_NONZERO_SFT_RESUME":
            raise RuntimeError("NONZERO_RECOVERY_NOT_READY")
    except BaseException as exc:
        exclusive_json(
            root / "engineering/NONZERO_FAILURE.json",
            dict(
                status="NONZERO_RECOVERY_NOT_READY",
                error_type=type(exc).__name__,
                no_automatic_retry=True,
                partial_evidence_preserved=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
