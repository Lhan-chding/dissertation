"""Phase 0 CLI for reproducing the v4 raw-row audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from compbias.v5.audit_v4_raw import reproduce_phase0_sections, sha256_file


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-archive", required=True, type=Path)
    parser.add_argument("--fact-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    payload = {
        "inputs": {
            "raw_archive": {
                "path": str(args.raw_archive),
                "sha256": sha256_file(args.raw_archive),
            },
            "fact_file": {
                "path": str(args.fact_file),
                "sha256": sha256_file(args.fact_file),
            },
        },
        "environment": {
            "python": sys.executable,
            "git_commit": _git_commit(),
        },
        "sections": reproduce_phase0_sections(args.raw_archive),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
