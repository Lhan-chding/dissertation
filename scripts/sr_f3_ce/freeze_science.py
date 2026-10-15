#!/usr/bin/env python3
"""Bind all successful engineering receipts before the fixed scientific matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f3_ce.bootstrap import freeze_science


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_science(args.root)
    print(json.dumps(dict(status=result["status"], execution_utc=result["execution_utc"])))


if __name__ == "__main__":
    main()
