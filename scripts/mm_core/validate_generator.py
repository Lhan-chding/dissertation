#!/usr/bin/env python3
"""Generate or validate the authorized CPU chart panels; no model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_core.generator import generate_dataset, validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report = (
        validate_dataset(args.run_root)
        if args.validate_only
        else generate_dataset(args.run_root, args.seed)
    )
    summary_keys = (
        "status",
        "questions",
        "images",
        "root_families",
        "original_geometry_status",
        "processor_status",
        "root_split_audit_status",
        "validation",
    )
    summary = report if args.validate_only else {k: report[k] for k in summary_keys}
    if not args.validate_only:
        summary["report_path"] = "manifests/GENERATOR_REPORT.json"
    print(json.dumps(summary, indent=2, sort_keys=True))
    validation = report.get("validation", report)
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
