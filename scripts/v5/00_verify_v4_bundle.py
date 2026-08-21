#!/usr/bin/env python3
"""Verify the two frozen Qwen v4 evidence archives without extracting them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compensability_v5.audit.audit_v4_raw import verify_v4_bundles

OFFICIAL_FACT_BUNDLE_SHA256 = "6fe3d48bc1a88b35bd5804bc397ce8160534bdd7d71fa6769b755cc098a53683"
OFFICIAL_RAW_ARCHIVE_SHA256 = "f0ccb4d56415eecf90a2c456bfd7c92a33fc96a581f3603115edbcb253ba8c84"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact-bundle", required=True, type=Path)
    parser.add_argument("--raw-archive", required=True, type=Path)
    parser.add_argument(
        "--expected-fact-sha256",
        default=OFFICIAL_FACT_BUNDLE_SHA256,
        help="fixed expected SHA-256 (defaults to the 2026-08-21 frozen bundle)",
    )
    parser.add_argument(
        "--expected-raw-sha256",
        default=OFFICIAL_RAW_ARCHIVE_SHA256,
        help="fixed expected SHA-256 (defaults to the 2026-08-21 frozen archive)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v5/audit/input_verification.json"),
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verification = verify_v4_bundles(
        args.fact_bundle,
        args.raw_archive,
        expected_fact_sha256=args.expected_fact_sha256,
        expected_raw_sha256=args.expected_raw_sha256,
    )
    _write_json(args.output, verification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
