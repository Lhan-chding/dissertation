#!/usr/bin/env python3
"""Recompute and freeze the deterministic v5 post-hoc audit of v4 raw rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from compensability_v5.audit.audit_v4_raw import reproduce_phase0_analysis, sha256_file

OFFICIAL_FACT_BUNDLE_SHA256 = "6fe3d48bc1a88b35bd5804bc397ce8160534bdd7d71fa6769b755cc098a53683"
OFFICIAL_RAW_ARCHIVE_SHA256 = "f0ccb4d56415eecf90a2c456bfd7c92a33fc96a581f3603115edbcb253ba8c84"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    Path("src/compensability_v5/audit/audit_v4_raw.py"),
    Path("src/compensability_v5/audit/fiber_multiplicity.py"),
    Path("src/compensability_v5/audit/error_order.py"),
    Path("scripts/v5/00_verify_v4_bundle.py"),
    Path("scripts/v5/01_run_posthoc_audit.py"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact-bundle", required=True, type=Path)
    parser.add_argument("--raw-archive", required=True, type=Path)
    parser.add_argument("--expected-fact-sha256", default=OFFICIAL_FACT_BUNDLE_SHA256)
    parser.add_argument("--expected-raw-sha256", default=OFFICIAL_RAW_ARCHIVE_SHA256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v5/audit/derived_analysis.json"),
    )
    parser.add_argument(
        "--verification-output",
        type=Path,
        default=Path("artifacts/v5/audit/input_verification.json"),
    )
    parser.add_argument(
        "--execution-manifest",
        type=Path,
        default=Path("artifacts/v5/audit/execution_manifest.json"),
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in CODE_PATHS:
        path = REPOSITORY_ROOT / relative_path
        if path.is_file():
            hashes[relative_path.as_posix()] = sha256_file(path)
    return hashes


def _combined_code_hash(code_hashes: dict[str, str]) -> str:
    canonical = json.dumps(code_hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _execution_manifest(
    args: argparse.Namespace,
    *,
    derived_sha256: str,
    verification_sha256: str,
    completed_at: str,
) -> dict[str, object]:
    code_hashes = _code_hashes()
    return {
        "schema_version": 1,
        "artifact_type": "qwen_v5_phase0_execution_manifest",
        "completed_at_utc": completed_at,
        "git_commit": _git_commit(),
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "inputs": {
            "fact_bundle": {
                "path": str(args.fact_bundle.resolve()),
                "sha256": args.expected_fact_sha256.lower(),
            },
            "raw_archive": {
                "path": str(args.raw_archive.resolve()),
                "sha256": args.expected_raw_sha256.lower(),
            },
        },
        "code": {
            "files": code_hashes,
            "combined_sha256": _combined_code_hash(code_hashes),
        },
        "outputs": {
            "derived_analysis": {
                "path": str(args.output.resolve()),
                "sha256": derived_sha256,
            },
            "input_verification": {
                "path": str(args.verification_output.resolve()),
                "sha256": verification_sha256,
            },
        },
        "model_calls_invoked": False,
        "training_invoked": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verification, derived = reproduce_phase0_analysis(
        args.fact_bundle,
        args.raw_archive,
        expected_fact_sha256=args.expected_fact_sha256,
        expected_raw_sha256=args.expected_raw_sha256,
    )
    _write_json(args.verification_output, verification)
    _write_json(args.output, derived)
    manifest = _execution_manifest(
        args,
        derived_sha256=sha256_file(args.output),
        verification_sha256=sha256_file(args.verification_output),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(args.execution_manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
