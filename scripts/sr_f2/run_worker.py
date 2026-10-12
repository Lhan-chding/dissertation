#!/usr/bin/env python3
"""Run exactly one registered F2 task in one visible GPU process."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f2.workers import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "--run-root", dest="root", type=Path, required=True)
    parser.add_argument("--kind", "--phase", dest="kind", required=True)
    parser.add_argument("--model-id", "--run-id", dest="model_id", required=True)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    kind = "pilot-segment" if args.kind == "pilot" and args.stop_step is not None else args.kind
    print(json.dumps(run(kind, args.root, args.model_id, stop_step=args.stop_step), sort_keys=True))


if __name__ == "__main__":
    main()
