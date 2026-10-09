#!/usr/bin/env python3
"""Run a registered starting/endpoint model on DEV_EVAL without teacher forcing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_dev.evaluation import main

if __name__ == "__main__":
    raise SystemExit(main("DEV_EVAL"))
