#!/usr/bin/env python3
"""Run isolated native engine comparison with an actual fresh-process restart."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_core.execution import atomic_json, exclusive_json, verify_gate
from mm_core.training import compare_engine, engine_segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--segment", choices=("continuous", "interrupt", "resume"))
    args = parser.parse_args()
    root = args.run_root.resolve()
    verify_gate(root, "ENGINE")
    if args.segment:
        branch, start, stop = {
            "continuous": ("continuous", 0, 4),
            "interrupt": ("resumed", 0, 2),
            "resume": ("resumed", 2, 4),
        }[args.segment]
        engine_segment(root, branch, start, stop)
        return
    exclusive_json(
        root / "engineering/ENGINE_STARTED.json", dict(status="RUNNING", pid=os.getpid())
    )
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
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
        receipt = compare_engine(root)
        if receipt["status"] != "PASS":
            raise RuntimeError("ENGINE_REPRO_NOT_READY: exact comparison failed")
    except BaseException as exc:
        # Preserve partial checkpoints, raw rollouts, and all consumed costs.
        if not (root / "engineering/ENGINE_COMPARE.json").exists():
            atomic_json(
                root / "engineering/ENGINE_COMPARE.json",
                dict(
                    status="ENGINE_REPRO_NOT_READY",
                    executed=True,
                    error_type=type(exc).__name__,
                    exact_comparison_completed=False,
                    no_automatic_retry=True,
                ),
            )
        raise


if __name__ == "__main__":
    main()
