#!/usr/bin/env python3
"""Build one immutable, CPU-validated protocol revision candidate; no inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f3_ce.protocol_revision import build_revision_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-plan", required=True, type=Path)
    parser.add_argument("--revision", required=True, choices=("v2", "c2"))
    args = parser.parse_args()
    result = build_revision_candidate(
        args.source_plan, args.request, args.output_plan, args.revision
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
