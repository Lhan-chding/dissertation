#!/usr/bin/env python3
"""Replay the frozen Stage-2 v1 failure without loading or calling a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_BOOTSTRAP_EXTRA_PATHS = frozenset(
    {
        "configs/recoverability/stage2_v1_failure.yaml",
        "experiments/recoverability_v1/06_diagnose_stage2_v1_failure.py",
        "src/compbias/recoverability/stage2_v1_failure.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_diagnostic_lock() -> None:
    if __name__ != "__main__" or "--help" in sys.argv or "-h" in sys.argv:
        return
    try:
        raw = sys.argv[sys.argv.index("--diagnostic-package-lock") + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical diagnostic package lock is required") from error
    root = Path(__file__).resolve().parents[2]
    path = Path(raw).resolve()
    canonical = root / "configs/recoverability/server_package_lock_stage2_v1_diagnostic.yaml"
    if path != canonical or path.is_symlink():
        raise SystemExit("BLOCKED: diagnostic package lock path is not canonical")
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths:
        raise SystemExit("BLOCKED: diagnostic package lock is malformed")
    if not _BOOTSTRAP_EXTRA_PATHS.issubset(paths) or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: diagnostic package lock closure is incomplete")
    for relative, expected in zip(paths, digests, strict=True):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: diagnostic package mismatch for {relative}")


_bootstrap_diagnostic_lock()

from compbias.recoverability.bridge import parse_stage1_evidence  # noqa: E402
from compbias.recoverability.stage2_v1 import Stage2V1Scene  # noqa: E402
from compbias.recoverability.stage2_v1_failure import (  # noqa: E402
    load_stage2_v1_failure,
    verify_stage2_v1_diagnostic_package_lock,
    verify_stage2_v1_failure_artifacts,
)


def _load_stage1_scenes(path: Path, *, expected_sha256: str) -> tuple[Stage2V1Scene, ...]:
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("frozen Stage-1 v2 records SHA-256 mismatch")
    scenes: list[Stage2V1Scene] = []
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {
                "scene_id",
                "chart_type",
                "operation",
                "raw_text",
                "parse_success",
                "exact_transcription",
                "error_code",
            }:
                raise ValueError("frozen Stage-1 v2 record schema is invalid")
            scene_id = row["scene_id"]
            operation = row["operation"]
            raw = row["raw_text"]
            if (
                not isinstance(scene_id, str)
                or scene_id in identifiers
                or not isinstance(operation, str)
                or not isinstance(raw, str)
                or row["parse_success"] is not True
                or row["error_code"] is not None
            ):
                raise ValueError("frozen Stage-1 v2 record does not contain trusted evidence")
            identifiers.add(scene_id)
            parsed = parse_stage1_evidence(raw)
            scenes.append(
                Stage2V1Scene(
                    scene_id=scene_id,
                    operation=operation,
                    evidence=parsed.target_facts,
                )
            )
    if len(scenes) != 24:
        raise ValueError("frozen Stage-1 v2 evidence must contain exactly 24 scenes")
    return tuple(sorted(scenes, key=lambda item: item.scene_id))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-config", type=Path, required=True)
    parser.add_argument("--diagnostic-package-lock", type=Path, required=True)
    parser.add_argument("--stage1-records", type=Path, required=True)
    parser.add_argument("--stage2-preflight", type=Path, required=True)
    parser.add_argument("--stage2-console", type=Path, required=True)
    parser.add_argument("--stage2-report", type=Path, required=True)
    parser.add_argument("--stage2-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    canonical_config = root / "configs/recoverability/stage2_v1_failure.yaml"
    canonical_lock = root / "configs/recoverability/server_package_lock_stage2_v1_diagnostic.yaml"
    if args.failure_config.resolve() != canonical_config or args.failure_config.is_symlink():
        raise ValueError("Stage-2 v1 failure config path is not canonical")
    verify_stage2_v1_diagnostic_package_lock(
        args.diagnostic_package_lock,
        repository_root=root,
    )
    if args.diagnostic_package_lock.resolve() != canonical_lock:
        raise ValueError("Stage-2 v1 diagnostic lock path is not canonical")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite Stage-2 v1 diagnostic")
    if args.output.parent.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("diagnostic output parent must be a regular directory")
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("Stage-2 v1 diagnostic forbids COMPBIAS path overrides")
    frozen = load_stage2_v1_failure(canonical_config)
    scenes = _load_stage1_scenes(
        args.stage1_records,
        expected_sha256=frozen.source_stage1_records_sha256,
    )
    diagnostic = verify_stage2_v1_failure_artifacts(
        frozen,
        preflight_path=args.stage2_preflight,
        console_path=args.stage2_console,
        report_path=args.stage2_report,
        records_path=args.stage2_records,
        scenes=scenes,
    )
    payload = {
        **asdict(diagnostic),
        "artifact_type": "recoverability_stage2_v1_failure_diagnostic",
        "schema_version": 1,
        "source_stage2_records_sha256": _sha256(args.stage2_records),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
