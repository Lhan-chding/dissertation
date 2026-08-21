"""Shared pytest setup for repository-level verification."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repository_import_paths() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    required = (
        repository_root,
        repository_root / "src",
        repository_root / "tests/coverage_support",
    )
    for path in reversed(required):
        candidate = str(path)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _enable_subprocess_coverage_startup() -> None:
    if not (os.getenv("COVERAGE_PROCESS_CONFIG") or os.getenv("COVERAGE_PROCESS_START")):
        return
    repository_root = Path(__file__).resolve().parents[1]
    required = (
        repository_root / "tests/coverage_support",
        repository_root,
        repository_root / "src",
    )
    inherited = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    combined = [str(path) for path in required]
    combined.extend(entry for entry in inherited if entry not in combined)
    os.environ["PYTHONPATH"] = os.pathsep.join(combined)


_ensure_repository_import_paths()
_enable_subprocess_coverage_startup()
