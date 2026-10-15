#!/usr/bin/env python3
"""Run the SR-F3-CE parent audit inside a CPU Slurm allocation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sr_f3_ce.preflight import audit_parents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Heavy base/checkpoint hashing requires a CPU Slurm allocation")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = audit_parents(args.parent_root, args.plan_dir, args.output_root)
    print(
        json.dumps(
            {
                "status": report["status"],
                "parents": len(report["parents"]),
                "model_calls": 0,
                "receipt": str(args.output_root / "parents/PARENT_AUDIT.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
