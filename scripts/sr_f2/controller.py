#!/usr/bin/env python3
"""Run SR-F2 scheduling on a CPU node under the ordinary override QoS."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f2.orchestration import Controller, run_controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    controller = Controller(args.root, Path(__file__).resolve().parents[2], python=args.python)
    result = run_controller(controller, once=args.once)
    print(json.dumps(result, sort_keys=True))
    if result["stage"].startswith("BLOCKED_"):
        raise SystemExit(2)
    if result["all_gpu_tasks_complete"] and not args.once:
        from sr_f2.reporting import release_and_report

        print(json.dumps(release_and_report(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
