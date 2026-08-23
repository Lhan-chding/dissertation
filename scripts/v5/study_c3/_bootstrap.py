from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo() -> None:
    root = Path(__file__).resolve().parents[3]
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
