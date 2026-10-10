#!/usr/bin/env python3
"""Execute actual single-GPU qualification and independent-process pilot recovery."""

import argparse
import json
from pathlib import Path

from sr_f12.protocol import validate_config
from sr_f12.runner import technical_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    result = technical_check(validate_config(args.plan), args.run_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
