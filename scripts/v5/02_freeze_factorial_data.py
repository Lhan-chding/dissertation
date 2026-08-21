#!/usr/bin/env python3
"""Freeze the model-independent Phase 2a factorial parent package."""

from __future__ import annotations

import argparse
from pathlib import Path

from compensability_v5.data.pre_model_freeze import freeze_pre_model_factorial

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--canonical-per-family", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v5/data/factorial_pre_model",
    )
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 2a immutable write requires explicit --execute.")
        return 2
    try:
        manifest = freeze_pre_model_factorial(
            arguments.output,
            seed=arguments.seed,
            canonical_per_family=arguments.canonical_per_family,
        )
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(
        "PHASE_2A_PRE_MODEL_FROZEN: "
        f"rows={manifest['row_count']} rows_sha256={manifest['rows_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
