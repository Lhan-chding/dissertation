#!/usr/bin/env python3
"""Replay and capture the frozen Stage-2 v2 success without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_BOOTSTRAP_EVIDENCE_PATHS = frozenset(
    {
        "configs/data/cva_chart_pilot_v0_3.yaml",
        "configs/paths.example.yaml",
        "configs/recoverability/bridge_v1_failure.yaml",
        "configs/recoverability/power_plan_v1.json",
        "configs/recoverability/recoverability_v1.yaml",
        "configs/recoverability/server_package_lock_stage2_v2.yaml",
        "configs/recoverability/server_runtime_v1.yaml",
        "configs/recoverability/stage1_v2_frozen_result.yaml",
        "configs/recoverability/stage1_v2_probe.yaml",
        "configs/recoverability/stage2_v1_diagnostic_result.yaml",
        "configs/recoverability/stage2_v1_failure.yaml",
        "configs/recoverability/stage2_v1_probe.yaml",
        "configs/recoverability/stage2_v2_frozen_result.yaml",
        "configs/recoverability/stage2_v2_probe.yaml",
        "configs/recoverability/v0_3_negative_pilot.yaml",
        "experiments/recoverability_v1/00_preflight.py",
        "experiments/recoverability_v1/00_stage1_v2_preflight.py",
        "experiments/recoverability_v1/00_stage2_v1_preflight.py",
        "experiments/recoverability_v1/00_stage2_v2_preflight.py",
        "experiments/recoverability_v1/02_capture_v03_evidence.py",
        "experiments/recoverability_v1/03_bridge.py",
        "experiments/recoverability_v1/04_stage1_v2_probe.py",
        "experiments/recoverability_v1/05_stage2_v1_probe.py",
        "experiments/recoverability_v1/06_diagnose_stage2_v1_failure.py",
        "experiments/recoverability_v1/07_stage2_v2_probe.py",
        "experiments/recoverability_v1/08_capture_stage2_v2_evidence.py",
        "requirements-gpu.lock.txt",
        "src/compbias/gpu_pilot/chart_data.py",
        "src/compbias/gpu_pilot/collection.py",
        "src/compbias/gpu_pilot/config.py",
        "src/compbias/gpu_pilot/execution_gate.py",
        "src/compbias/gpu_pilot/preflight.py",
        "src/compbias/gpu_pilot/qwen_smoke.py",
        "src/compbias/gpu_pilot/safe_io.py",
        "src/compbias/gpu_pilot/structured_generation.py",
        "src/compbias/gpu_pilot/taxonomy.py",
        "src/compbias/io/strict_json.py",
        "src/compbias/io/yaml_config.py",
        "src/compbias/models/structured_parser.py",
        "src/compbias/recoverability/bridge.py",
        "src/compbias/recoverability/bridge_v1_failure.py",
        "src/compbias/recoverability/config.py",
        "src/compbias/recoverability/dsl/executor.py",
        "src/compbias/recoverability/dsl/parser.py",
        "src/compbias/recoverability/dsl/result_program.py",
        "src/compbias/recoverability/dsl/schema.py",
        "src/compbias/recoverability/evidence.py",
        "src/compbias/recoverability/evidence_capture.py",
        "src/compbias/recoverability/preflight.py",
        "src/compbias/recoverability/stage1_v2.py",
        "src/compbias/recoverability/stage2_v1.py",
        "src/compbias/recoverability/stage2_v1_failure.py",
        "src/compbias/recoverability/stage2_v2.py",
        "src/compbias/recoverability/stage2_v2_evidence.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_evidence_lock() -> None:
    if __name__ != "__main__" or "--help" in sys.argv or "-h" in sys.argv:
        return
    try:
        raw = sys.argv[sys.argv.index("--evidence-package-lock") + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical Stage-2 v2 evidence lock is required") from error
    root = Path(__file__).resolve().parents[2]
    lock_path = Path(raw).resolve()
    canonical = root / "configs/recoverability/server_package_lock_stage2_v2_evidence.yaml"
    if lock_path != canonical or lock_path.is_symlink():
        raise SystemExit("BLOCKED: Stage-2 v2 evidence lock path is not canonical")
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths:
        raise SystemExit("BLOCKED: Stage-2 v2 evidence lock is malformed")
    if frozenset(paths) != _BOOTSTRAP_EVIDENCE_PATHS or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: Stage-2 v2 evidence lock closure is incomplete")
    for relative, expected in zip(paths, digests, strict=True):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: Stage-2 v2 evidence mismatch for {relative}")


_bootstrap_evidence_lock()

from compbias.recoverability.stage1_v2 import validate_stage1_v2_runtime_paths  # noqa: E402
from compbias.recoverability.stage2_v2 import (  # noqa: E402
    load_stage2_v2_scenes_from_stage1_records,
)
from compbias.recoverability.stage2_v2_evidence import (  # noqa: E402
    load_stage2_v2_frozen_result,
    verify_stage2_v2_artifacts,
    verify_stage2_v2_evidence_package_lock,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-result", type=Path, required=True)
    parser.add_argument("--evidence-package-lock", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--stage1-records", type=Path, required=True)
    parser.add_argument("--stage2-v2-preflight", type=Path, required=True)
    parser.add_argument("--stage2-v2-console", type=Path, required=True)
    parser.add_argument("--stage2-v2-report", type=Path, required=True)
    parser.add_argument("--stage2-v2-records", type=Path, required=True)
    parser.add_argument("--attempt-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    canonical_frozen = root / "configs/recoverability/stage2_v2_frozen_result.yaml"
    canonical_lock = root / "configs/recoverability/server_package_lock_stage2_v2_evidence.yaml"
    canonical_paths = root / "configs/paths.yaml"
    if args.frozen_result.resolve() != canonical_frozen or args.frozen_result.is_symlink():
        raise ValueError("Stage-2 v2 frozen result path is not canonical")
    if args.evidence_package_lock.resolve() != canonical_lock:
        raise ValueError("Stage-2 v2 evidence lock path is not canonical")
    if args.paths.resolve() != canonical_paths or args.paths.is_symlink():
        raise ValueError("Stage-2 v2 paths file is not canonical")
    verify_stage2_v2_evidence_package_lock(canonical_lock, repository_root=root)
    validate_stage1_v2_runtime_paths(
        canonical_paths,
        registered_example=root / "configs/paths.example.yaml",
    )
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("Stage-2 v2 evidence capture forbids COMPBIAS path overrides")
    frozen = load_stage2_v2_frozen_result(canonical_frozen)
    canonical = {
        "stage1_records": root
        / "outputs/recoverability_v1/stage1_v2_dev_probe/probe_records.jsonl",
        "preflight": Path("/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-preflight.json"),
        "console": Path("/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-console.log"),
        "report": root / "outputs/recoverability_v1/stage2_v2_dev_probe/probe_report.json",
        "records": root / "outputs/recoverability_v1/stage2_v2_dev_probe/probe_records.jsonl",
        "attempt": root / "outputs/recoverability_v1/stage2_v2_dev_probe.attempted.json",
        "output": Path(
            "/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-external-evidence.json"
        ),
    }
    supplied = {
        "stage1_records": args.stage1_records,
        "preflight": args.stage2_v2_preflight,
        "console": args.stage2_v2_console,
        "report": args.stage2_v2_report,
        "records": args.stage2_v2_records,
        "attempt": args.attempt_marker,
        "output": args.output,
    }
    for label, expected in canonical.items():
        if supplied[label].resolve() != expected or supplied[label].is_symlink():
            raise ValueError(f"Stage-2 v2 {label} path is not canonical")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite Stage-2 v2 external evidence")
    if args.output.parent.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("Stage-2 v2 evidence parent must be a regular directory")
    scenes = load_stage2_v2_scenes_from_stage1_records(
        args.stage1_records,
        expected_sha256=frozen.source_stage1_records_sha256,
    )
    verification = verify_stage2_v2_artifacts(
        frozen,
        preflight_path=args.stage2_v2_preflight,
        console_path=args.stage2_v2_console,
        report_path=args.stage2_v2_report,
        records_path=args.stage2_v2_records,
        attempt_marker_path=args.attempt_marker,
        scenes=scenes,
    )
    payload = {
        **asdict(verification),
        "artifact_type": "recoverability_stage2_v2_external_evidence",
        "schema_version": 1,
        "source_stage2_v2_records_sha256": _sha256(args.stage2_v2_records),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
