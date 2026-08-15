#!/usr/bin/env python3
"""Run the one-shot executor-authoritative Stage-2 v2 development probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

_BOOTSTRAP_SERVER_PATHS = frozenset(
    {
        "configs/data/cva_chart_pilot_v0_3.yaml",
        "configs/paths.example.yaml",
        "configs/recoverability/bridge_v1_failure.yaml",
        "configs/recoverability/power_plan_v1.json",
        "configs/recoverability/recoverability_v1.yaml",
        "configs/recoverability/server_runtime_v1.yaml",
        "configs/recoverability/stage1_v2_frozen_result.yaml",
        "configs/recoverability/stage1_v2_probe.yaml",
        "configs/recoverability/stage2_v1_diagnostic_result.yaml",
        "configs/recoverability/stage2_v1_failure.yaml",
        "configs/recoverability/stage2_v1_probe.yaml",
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
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--execute" not in sys.argv:
        return
    try:
        raw = sys.argv[sys.argv.index("--server-package-lock") + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical Stage-2 v2 server package lock is required") from error
    root = Path(__file__).resolve().parents[2]
    lock_path = Path(raw).resolve()
    canonical = root / "configs/recoverability/server_package_lock_stage2_v2.yaml"
    if lock_path != canonical or lock_path.is_symlink():
        raise SystemExit("BLOCKED: Stage-2 v2 server package lock path is not canonical")
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths:
        raise SystemExit("BLOCKED: Stage-2 v2 server package lock is malformed")
    if frozenset(paths) != _BOOTSTRAP_SERVER_PATHS or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: Stage-2 v2 server package lock closure is incomplete")
    for relative, expected in zip(paths, digests, strict=True):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: Stage-2 v2 package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.config import load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import load_local_qwen  # noqa: E402
from compbias.recoverability.bridge import decode_text_qwen_once  # noqa: E402
from compbias.recoverability.preflight import (  # noqa: E402
    MetadataPreflightReport,
    load_runtime_spec,
    run_metadata_preflight,
)
from compbias.recoverability.stage1_v2 import validate_stage1_v2_runtime_paths  # noqa: E402
from compbias.recoverability.stage2_v1 import load_stage1_v2_frozen_result  # noqa: E402
from compbias.recoverability.stage2_v2 import (  # noqa: E402
    load_stage2_v1_diagnostic_anchor,
    load_stage2_v2_probe_config,
    load_stage2_v2_scenes_from_stage1_records,
    run_stage2_v2_probe,
    verify_stage2_v1_diagnostic,
    verify_stage2_v2_server_package_lock,
)


def _metadata_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def _pip_check() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        env=_metadata_subprocess_environment(),
    )
    return completed.returncode, completed.stdout + completed.stderr


def _pip_inventory() -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
        env=_metadata_subprocess_environment(),
    )
    return {row["name"]: row["version"] for row in json.loads(completed.stdout)}


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validate_preflight(
    path: Path,
    *,
    lock_path: Path,
    current: MetadataPreflightReport,
) -> None:
    payload = _load_json(path, label="Stage-2 v2 preflight report")
    if (
        payload.get("artifact_type") != "recoverability_stage2_v2_metadata_preflight"
        or payload.get("ready") is not current.ready
        or payload.get("requirements_lock_sha256") != current.requirements_lock_sha256
        or payload.get("installed_packages") != [list(item) for item in current.installed_packages]
        or payload.get("pip_check_passed") is not current.pip_check_passed
        or payload.get("offline_verified") is not current.offline_verified
        or payload.get("large_gpu_started") is not False
        or payload.get("model_loaded") is not False
        or payload.get("training_authorized") is not False
        or payload.get("server_package_lock_verified") is not True
        or payload.get("server_package_lock_sha256") != _sha256(lock_path)
    ):
        raise ValueError("preflight report does not authorize the Stage-2 v2 probe")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--probe-config", type=Path)
    parser.add_argument("--stage1-result", type=Path)
    parser.add_argument("--stage2-v1-diagnostic-anchor", type=Path)
    parser.add_argument("--server-package-lock", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--stage1-records", type=Path)
    parser.add_argument("--stage2-v1-diagnostic", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("BLOCKED: Stage-2 v2 probe requires explicit --execute on the reviewed GPU server")
        return 2
    required = (
        args.paths,
        args.runtime,
        args.probe_config,
        args.stage1_result,
        args.stage2_v1_diagnostic_anchor,
        args.server_package_lock,
        args.preflight_report,
        args.stage1_records,
        args.stage2_v1_diagnostic,
    )
    if any(value is None for value in required):
        print("BLOCKED: Stage-2 v2 probe requires every frozen evidence input")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = Path(__file__).resolve().parents[2]
    canonical = {
        "lock": root / "configs/recoverability/server_package_lock_stage2_v2.yaml",
        "probe": root / "configs/recoverability/stage2_v2_probe.yaml",
        "stage1": root / "configs/recoverability/stage1_v2_frozen_result.yaml",
        "diagnostic_anchor": root / "configs/recoverability/stage2_v1_diagnostic_result.yaml",
        "runtime": root / "configs/recoverability/server_runtime_v1.yaml",
        "paths": root / "configs/paths.yaml",
    }
    supplied = {
        "lock": args.server_package_lock,
        "probe": args.probe_config,
        "stage1": args.stage1_result,
        "diagnostic_anchor": args.stage2_v1_diagnostic_anchor,
        "runtime": args.runtime,
        "paths": args.paths,
    }
    for label, expected in canonical.items():
        if supplied[label].resolve() != expected or supplied[label].is_symlink():
            raise ValueError(f"Stage-2 v2 {label} path is not canonical")
    validate_stage1_v2_runtime_paths(
        canonical["paths"],
        registered_example=root / "configs/paths.example.yaml",
    )
    verify_stage2_v2_server_package_lock(canonical["lock"], repository_root=root)
    current_preflight = run_metadata_preflight(
        load_runtime_spec(canonical["runtime"]),
        repository_root=root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    _validate_preflight(
        args.preflight_report, lock_path=canonical["lock"], current=current_preflight
    )
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("Stage-2 v2 probe forbids COMPBIAS path overrides")
    paths = load_pilot_paths(canonical["paths"])
    expected_stage1_records = (
        paths.outputs / "recoverability_v1" / "stage1_v2_dev_probe" / "probe_records.jsonl"
    )
    expected_diagnostic = Path(
        "/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v1-failure-diagnostic.json"
    )
    if args.stage1_records.resolve() != expected_stage1_records:
        raise ValueError("Stage-2 v2 Stage-1 records path is not canonical")
    if args.stage2_v1_diagnostic.resolve() != expected_diagnostic:
        raise ValueError("Stage-2 v1 diagnostic path is not canonical")
    anchor = load_stage2_v1_diagnostic_anchor(canonical["diagnostic_anchor"])
    verify_stage2_v1_diagnostic(anchor, args.stage2_v1_diagnostic)
    frozen_stage1 = load_stage1_v2_frozen_result(canonical["stage1"])
    stage1_records_sha256 = dict(frozen_stage1.source_sha256)["probe_records"]
    scenes = load_stage2_v2_scenes_from_stage1_records(
        args.stage1_records,
        expected_sha256=stage1_records_sha256,
    )
    probe = load_stage2_v2_probe_config(canonical["probe"])
    if len(scenes) != probe.scenes or probe.format_retries != 0:
        raise ValueError("Stage-2 v2 scene or retry contract differs")
    output = paths.outputs / "recoverability_v1" / probe.output_subdirectory
    attempt_marker = output.parent / f"{probe.output_subdirectory}.attempted.json"
    if (
        output.exists()
        or output.is_symlink()
        or attempt_marker.exists()
        or attempt_marker.is_symlink()
    ):
        raise FileExistsError("refusing to rerun or overwrite the Stage-2 v2 probe")
    output.parent.mkdir(parents=True, exist_ok=True)
    with attempt_marker.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "status": "STAGE2_V2_DEVELOPMENT_PROBE_STARTED",
                "hypothesis_test": False,
            },
            stream,
            sort_keys=True,
        )
        stream.write("\n")
    model_hash_before = model_snapshot_sha256(paths.model_path)
    if model_hash_before != frozen_stage1.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the frozen Stage-1 v2 probe")
    model, processor = load_local_qwen(paths.model_path)
    report, records = run_stage2_v2_probe(
        scenes,
        generate=lambda _scene, messages: decode_text_qwen_once(model, processor, messages),
    )
    if model_snapshot_sha256(paths.model_path) != model_hash_before:
        raise RuntimeError("model snapshot changed during the Stage-2 v2 probe")
    payload = {
        **asdict(report),
        "error_counts": dict(report.error_counts),
        "schema_version": 1,
        "artifact_type": "recoverability_stage2_v2_development_probe",
        "dataset_id": probe.dataset_id,
        "source_dataset_id": probe.source_dataset_id,
        "source_split": probe.source_split,
        "model_snapshot_sha256": model_hash_before,
        "source_stage1_records_sha256": stage1_records_sha256,
        "source_stage2_v1_diagnostic_sha256": anchor.diagnostic_sha256,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
    }
    with tempfile.TemporaryDirectory(prefix=".stage2-v2-probe-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        with (staging / "probe_records.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n")
        with (staging / "probe_report.json").open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        staging.rename(output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.probe_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
