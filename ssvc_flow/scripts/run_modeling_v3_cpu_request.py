#!/usr/bin/env python3
"""Execute one immutable CPU CLI request inside a Slurm CPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CPU_COMMANDS = {
    "plan",
    "audit-parent",
    "prepare-historical",
    "test",
    "collect-toy",
    "observe-toy",
    "coverage-study",
    "freeze",
    "validate-cpu",
    "plan-gpu",
    "prepare-gpu",
    "merge-q4",
    "calibrate-cpu",
    "analyze-cpu",
    "analyze-cpu-generalization",
    "freeze-vlm",
    "calibrate-vlm",
    "analyze-vlm",
    "prepare-q6",
    "prepare-q6-forks",
    "measure-q6-absolute",
    "prepare-source",
    "prepare-forks",
    "prepare-observation",
    "prepare-reference-extension",
    "prepare-geometry",
    "prepare-fit",
    "freeze-predictions",
    "freeze-prediction-set",
    "prepare-evaluation",
    "finalize-stage",
    "analyze-observation",
    "analyze-coverage",
    "select",
    "fit",
    "evaluate",
    "track-offline",
    "summarize",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("production CPU requests require a server Slurm allocation")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("CPU request wrapper requires CUDA_VISIBLE_DEVICES to be empty")
    path = Path(args.request).resolve()
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.sha256:
        raise SystemExit("CPU request hash mismatch")
    request = json.loads(data)
    argv = request.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(value, str) for value in argv)
        or argv[0] not in CPU_COMMANDS
        or any(value.startswith("--allow-") for value in argv)
        or "--acknowledge-new-experiment" in argv
    ):
        raise SystemExit("request must contain one supported CPU-only CLI argv")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    from src.modeling_v3.cli import main as cli_main

    print(json.dumps({"request": str(path), "sha256": args.sha256, "argv": argv}), flush=True)
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
