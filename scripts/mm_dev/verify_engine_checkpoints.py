#!/usr/bin/env python3
"""Fresh CPU-only elementwise verification of a trusted registered ENGINE run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_dev.checkpoint_recount import publish_recount, verify_engine_checkpoints
from mm_dev.common import verify_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("natural", "stress"), required=True)
    args = parser.parse_args()
    verify_execution(args.plan, args.run_root, require_engine=False)
    result = verify_engine_checkpoints(args.run_root, args.mode)
    output = publish_recount(args.run_root, result)
    print(json.dumps({"status": result["status"], "path": str(output)}))


if __name__ == "__main__":
    main()
