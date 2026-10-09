#!/usr/bin/env python3
"""Actual CPU S0/S1 materialization; never claims local-only inputs are GPU-ready."""

import argparse
import json
from pathlib import Path

from sr_f1.prepare import prepare_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--font", required=True, type=Path)
    parser.add_argument("--bold-font", required=True, type=Path)
    parser.add_argument("--operator", default="codex")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--defer-freeze", action="store_true")
    args = parser.parse_args()
    result = prepare_run(
        args.root,
        font_path=args.font,
        bold_path=args.bold_font,
        operator=args.operator,
        model_path=args.model_path,
        local_only=args.local_only,
        defer_freeze=args.defer_freeze,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "root": str(args.root.resolve()),
                "model_calls": 0,
                "model_weights_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
