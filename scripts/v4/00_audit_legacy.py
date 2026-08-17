from __future__ import annotations

import argparse
from pathlib import Path

from compensability_v4.diagnostics.legacy_audit import write_legacy_audit

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write frozen Phase-0 audit artifacts for v4.")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    arguments = parser.parse_args()
    write_legacy_audit(ROOT, arguments.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
