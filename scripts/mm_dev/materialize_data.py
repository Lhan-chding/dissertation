#!/usr/bin/env python3
"""Render registered MM-DEV F2 sources and optionally run native CPU processor QA."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_dev.data import REPORT, materialize_data, processor_preflight, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--processor-preflight", action="store_true")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    if args.processor_preflight and not args.model_path:
        parser.error("--processor-preflight requires --model-path")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        if not (args.run_root / REPORT).exists():
            report = materialize_data(
                args.plan, args.run_root, historical_root=args.historical_root
            )
        elif not args.processor_preflight:
            raise FileExistsError("Materialization exists; preserve the completed report")
        if args.processor_preflight:
            report = processor_preflight(args.plan, args.run_root, args.model_path)
    except Exception as error:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(
            args.run_root / "engineering" / f"DATA_MATERIALIZATION_FAILURE_{stamp}.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "new_model_calls": 0,
            },
        )
        raise
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "root_families",
                    "numeric_sources",
                    "images",
                    "questions",
                    "processor_status",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
