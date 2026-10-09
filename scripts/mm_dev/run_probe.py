#!/usr/bin/env python3
"""Run the registered PROBE state/shard with self and unique gold field scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_dev.evaluation import main

if __name__ == "__main__":
    raise SystemExit(main("PROBE"))
