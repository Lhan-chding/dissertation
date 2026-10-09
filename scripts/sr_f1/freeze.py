#!/usr/bin/env python3
"""Freeze already completed native CPU preparation, without additional generation."""

import argparse
import json
from pathlib import Path

from sr_f1.freeze import freeze_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--operator", default="codex")
    args = parser.parse_args()
    result = freeze_execution(args.root, operator=args.operator)
    print(json.dumps({"status": result["status"], "root": str(args.root.resolve())}))


if __name__ == "__main__":
    main()
