from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo() -> None:
    root = Path(__file__).resolve().parents[3]
    src = root / "src"
    location = str(src)
    if location not in sys.path:
        sys.path.insert(0, location)
