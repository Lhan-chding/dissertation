#!/usr/bin/env python3
"""Revalidate the frozen CPU evidence and optionally require GPU engine PASS."""

import argparse
import json
from pathlib import Path

from sr_f1.contract import load_config
from sr_f1.freeze import verify_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--require-engine", action="store_true")
    args = parser.parse_args()
    result = verify_execution(load_config(), args.root, require_engine=args.require_engine)
    print(
        json.dumps(
            {
                "status": "PASS",
                "freeze_status": result["status"],
                "gpu_engine_required": args.require_engine,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
