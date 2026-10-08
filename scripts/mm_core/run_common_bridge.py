#!/usr/bin/env python3
"""Run only the one format-triggered common bridge registered in this audit."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_core.training import run_bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_bridge(args.run_root)


if __name__ == "__main__":
    main()
