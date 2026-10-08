"""Freeze verified data, environment and protocol before any GPU request."""

import argparse
from pathlib import Path

from mm_core.execution import freeze_audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    freeze_audit(args.run_root)
