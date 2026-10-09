#!/usr/bin/env python3
"""Run exactly one preregistered scientific F2 training path."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_dev.training import run_train_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = run_train_path(args.plan, args.run_root, args.run_id)
    except Exception as error:
        from mm_dev.orchestration import fail_task

        fail_task(args.run_root, args.run_id, repr(error))
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
