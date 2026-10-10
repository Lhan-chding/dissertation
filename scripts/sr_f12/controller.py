#!/usr/bin/env python3
"""Run the registered SR-F1.2 controller; jobs are held and verified before release."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f12.orchestration import Controller, run_controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    controller = Controller(
        args.run_root,
        Path(__file__).resolve().parents[2],
        args.model_path,
        args.source_manifest,
        args.source_commit,
        python=args.python,
    )
    result = run_controller(controller, once=args.once)
    if result["stage"].startswith("BLOCKED_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
