#!/usr/bin/env python3
"""Generate the fixed CPU-only Recoverability v1 design fixture."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from compbias.recoverability.fixture_generator import (
    audit_fixture,
    generate_fixture_50,
    serialize_fixture,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite fixture directory: {output}")
    scenes = generate_fixture_50(seed=2026081606)
    report = audit_fixture(scenes)
    if not report.audit_passed:
        raise RuntimeError("local recoverability fixture audit failed")
    output.mkdir(parents=True)
    records_path = output / "fixture_50.jsonl"
    audit_path = output / "fixture_50.audit.json"
    records_path.write_text("\n".join(serialize_fixture(scenes)) + "\n", encoding="utf-8")
    payload = asdict(report)
    payload.update(
        {
            "artifact_type": "recoverability_v1_local_fixture_audit",
            "schema_version": 1,
            "gpu_invoked": False,
            "model_loaded": False,
            "training_invoked": False,
        }
    )
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
