#!/usr/bin/env python3
"""Create or verify the independently authorized MM-DEV F2 execution freeze."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_dev.common import verify_execution
from mm_dev.freeze import freeze_run, inspect_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--code-commit")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--create", action="store_true")
    modes.add_argument("--inspect-model", action="store_true")
    parser.add_argument("--require-engine", action="store_true")
    args = parser.parse_args()
    if args.create:
        if not args.historical_root or not args.code_commit:
            parser.error("--create requires --historical-root and --code-commit")
        result = freeze_run(
            args.plan, args.run_root, args.historical_root, code_commit=args.code_commit
        )
    elif args.inspect_model:
        if not args.historical_root:
            parser.error("--inspect-model requires --historical-root")
        result = inspect_model(args.plan, args.historical_root)
    else:
        result = verify_execution(
            args.plan, args.run_root, require_engine=args.require_engine, full_hashes=True
        )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "plan_id": result["plan_id"],
                "source_files": len(result.get("source_hashes", {})),
                "input_files": len(result.get("input_hashes", {})),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
