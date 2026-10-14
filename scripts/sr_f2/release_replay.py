#!/usr/bin/env python3
"""Run the explicit three-stage SR-F2 release gate and frozen scientific reporting."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    parent = json.loads(Path(plan["parent_plan_path"]).read_text())
    extension = Path(__file__).resolve().parents[2]
    sys.path[:0] = [
        str(extension / "src"),
        str(Path(parent["extension_root"]) / "src"),
        str(Path(parent["frozen_code_root"]) / "src"),
    ]
    from sr_f2_replay.reporting import release_and_report, verify_composite_release_gate

    result = (verify_composite_release_gate if args.verify_only else release_and_report)(args.plan)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
