#!/usr/bin/env python3
"""Keep one bound four-GPU pool active on the shared registered ready queue."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f2.orchestration import run_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=1)
    args = parser.parse_args()
    run_pool(args.root, args.plan, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
