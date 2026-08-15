#!/usr/bin/env python3
"""Run the one-shot, development-only Stage-1 v2 probe on 24 dev scenes."""

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
        "configs/recoverability/stage1_v2_probe.yaml",
        "configs/recoverability/v0_3_negative_pilot.yaml",
        "experiments/recoverability_v1/00_preflight.py",
        "experiments/recoverability_v1/00_stage1_v2_preflight.py",
        "experiments/recoverability_v1/02_capture_v03_evidence.py",
        "experiments/recoverability_v1/03_bridge.py",
        "experiments/recoverability_v1/04_stage1_v2_probe.py",
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
        "src/compbias/recoverability/dsl/schema.py",
        "src/compbias/recoverability/evidence.py",
        "src/compbias/recoverability/evidence_capture.py",
        "src/compbias/recoverability/preflight.py",
        "src/compbias/recoverability/stage1_v2.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--execute" not in os.sys.argv:
        return
    try:
        raw = os.sys.argv[os.sys.argv.index("--server-package-lock") + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical Stage-1 v2 server package lock is required") from error
    root = Path(__file__).resolve().parents[2]
    lock_path = Path(raw).resolve()
    canonical = root / "configs/recoverability/server_package_lock_stage1_v2.yaml"
    if lock_path != canonical:
        raise SystemExit("BLOCKED: Stage-1 v2 server package lock path is not canonical")
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths:
        raise SystemExit("BLOCKED: Stage-1 v2 server package lock is malformed")
    if frozenset(paths) != _BOOTSTRAP_SERVER_PATHS or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: Stage-1 v2 server package lock closure is incomplete")
    for relative, expected in zip(paths, digests, strict=True):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: Stage-1 v2 package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.collection import calibration_gate  # noqa: E402
from compbias.gpu_pilot.config import ACTIVE_PILOT_OUTPUT_SLUG, load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.execution_gate import (  # noqa: E402
    _validate_dataset_bundle,
    _validate_manifest,
    _validate_natural_records,
)
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import decode_qwen_once, load_local_qwen  # noqa: E402
from compbias.recoverability.bridge_v1_failure import (  # noqa: E402
    load_bridge_v1_failure,
    verify_bridge_v1_failure_artifacts,
)
from compbias.recoverability.evidence import load_negative_pilot_record  # noqa: E402
from compbias.recoverability.preflight import (  # noqa: E402
    MetadataPreflightReport,
    load_runtime_spec,
    run_metadata_preflight,
)
from compbias.recoverability.stage1_v2 import (  # noqa: E402
    load_stage1_v2_probe_config,
    run_stage1_v2_probe,
    select_stage1_v2_probe_scenes,
    validate_stage1_v2_runtime_paths,
    verify_stage1_v2_server_package_lock,
)


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


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
    rows = json.loads(completed.stdout)
    return {row["name"]: row["version"] for row in rows}


def _validate_preflight(
    path: Path,
    *,
    lock_path: Path,
    current: MetadataPreflightReport,
) -> None:
    payload = _load_json(path, label="Stage-1 v2 preflight report")
    if (
        payload.get("artifact_type") != "recoverability_stage1_v2_metadata_preflight"
        or payload.get("ready") is not current.ready
        or payload.get("requirements_lock_sha256") != current.requirements_lock_sha256
        or payload.get("installed_packages") != [list(item) for item in current.installed_packages]
        or payload.get("pip_check_passed") is not current.pip_check_passed
        or payload.get("offline_verified") is not current.offline_verified
        or payload.get("large_gpu_started") is not current.large_gpu_started
        or payload.get("model_loaded") is not current.model_loaded
        or payload.get("training_authorized") is not current.training_authorized
        or payload.get("server_package_lock_verified") is not True
        or payload.get("server_package_lock_sha256") != _sha256(lock_path)
    ):
        raise ValueError("preflight report does not authorize the Stage-1 v2 probe")


def _validate_external_v03(path: Path, *, records_path: Path, negative: object) -> None:
    payload = _load_json(path, label="v0.3 external evidence")
    expected = {
        "status": "FROZEN_FAILED_NOT_TO_BE_RERUN",
        "records": negative.records,
        "gate_passed": False,
        "calibration_exit": 3,
        "model_snapshot_sha256": negative.model_snapshot_sha256,
        "dataset_manifest_sha256": negative.dataset_manifest_sha256,
        "dataset_records_sha256": negative.dataset_records_sha256,
        "dataset_images_sha256": negative.dataset_images_sha256,
        "counterfactual_sha256": negative.counterfactual_sha256,
        "calibration_exit_evidence": "replayed_raw_records_and_frozen_calibration_gate",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("v0.3 external evidence differs from the frozen negative pilot")
    failures = list(
        calibration_gate(
            {
                "answer_accuracy": negative.answer_accuracy,
                "natural_perception_error_rate": negative.natural_perception_error_rate,
                "parse_rate": negative.parse_rate,
                "error_counts": dict(negative.error_counts),
            }
        )
    )
    if payload.get("calibration_gate_failures") != failures:
        raise ValueError("v0.3 external evidence gate failures differ")
    sources = payload.get("source_files")
    if not isinstance(sources, list):
        raise ValueError("v0.3 external evidence source registry is invalid")
    source = next(
        (
            item
            for item in sources
            if isinstance(item, dict) and item.get("basename") == "calibration_records_v0_3.jsonl"
        ),
        None,
    )
    if source is None or source.get("sha256") != _sha256(records_path):
        raise ValueError("v0.3 calibration records differ from captured evidence")


def _validate_dataset(dataset: Path, *, negative: object) -> dict[str, object]:
    manifest_path = dataset / "manifest.json"
    manifest = _load_json(manifest_path, label="v0.3 dataset manifest")
    _validate_manifest(manifest)
    if (
        _sha256(manifest_path) != negative.dataset_manifest_sha256
        or manifest.get("records_sha256") != negative.dataset_records_sha256
        or manifest.get("images_sha256") != negative.dataset_images_sha256
        or manifest.get("counterfactual_sha256") != negative.counterfactual_sha256
    ):
        raise ValueError("v0.3 dataset hashes differ from the frozen negative pilot")
    return _validate_dataset_bundle(manifest_path, manifest)


def _read_dataset_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("dataset record must be a JSON object")
            rows.append(row)
    return tuple(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--probe-config", type=Path)
    parser.add_argument("--server-package-lock", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--external-evidence", type=Path)
    parser.add_argument("--v03-records", type=Path)
    parser.add_argument("--bridge-v1-records", type=Path)
    parser.add_argument("--bridge-v1-report", type=Path)
    parser.add_argument("--bridge-v1-diagnostic", type=Path)
    parser.add_argument("--bridge-v1-attempt-marker", type=Path)
    parser.add_argument("--bridge-v1-console-log", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("BLOCKED: Stage-1 v2 probe requires explicit --execute on the reviewed GPU server")
        return 2
    required = (
        args.paths,
        args.runtime,
        args.probe_config,
        args.server_package_lock,
        args.preflight_report,
        args.external_evidence,
        args.v03_records,
        args.bridge_v1_records,
        args.bridge_v1_report,
        args.bridge_v1_diagnostic,
        args.bridge_v1_attempt_marker,
        args.bridge_v1_console_log,
    )
    if any(value is None for value in required):
        print("BLOCKED: Stage-1 v2 probe requires every frozen evidence input")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    repository_root = Path(__file__).resolve().parents[2]
    canonical_lock = repository_root / "configs/recoverability/server_package_lock_stage1_v2.yaml"
    canonical_probe = repository_root / "configs/recoverability/stage1_v2_probe.yaml"
    canonical_runtime = repository_root / "configs/recoverability/server_runtime_v1.yaml"
    if args.server_package_lock.resolve() != canonical_lock:
        raise ValueError("Stage-1 v2 server package lock path is not canonical")
    if args.probe_config.resolve() != canonical_probe:
        raise ValueError("Stage-1 v2 probe configuration path is not canonical")
    if args.runtime.resolve() != canonical_runtime:
        raise ValueError("Stage-1 v2 runtime configuration path is not canonical")
    if args.paths.resolve() != repository_root / "configs/paths.yaml":
        raise ValueError("Stage-1 v2 paths configuration must use configs/paths.yaml")
    validate_stage1_v2_runtime_paths(
        args.paths,
        registered_example=repository_root / "configs/paths.example.yaml",
    )
    verify_stage1_v2_server_package_lock(canonical_lock, repository_root=repository_root)
    current_preflight = run_metadata_preflight(
        load_runtime_spec(canonical_runtime),
        repository_root=repository_root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    _validate_preflight(
        args.preflight_report,
        lock_path=canonical_lock,
        current=current_preflight,
    )
    negative = load_negative_pilot_record(
        repository_root / "configs/recoverability/v0_3_negative_pilot.yaml"
    )
    _validate_external_v03(args.external_evidence, records_path=args.v03_records, negative=negative)
    failure = load_bridge_v1_failure(
        repository_root / "configs/recoverability/bridge_v1_failure.yaml"
    )
    verify_bridge_v1_failure_artifacts(
        failure,
        records_path=args.bridge_v1_records,
        report_path=args.bridge_v1_report,
        diagnostic_path=args.bridge_v1_diagnostic,
        attempt_marker_path=args.bridge_v1_attempt_marker,
        console_log_path=args.bridge_v1_console_log,
    )
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("Stage-1 v2 probe forbids COMPBIAS path overrides")
    probe = load_stage1_v2_probe_config(canonical_probe)
    paths = load_pilot_paths(args.paths)
    dataset = paths.data / "generated" / ACTIVE_PILOT_OUTPUT_SLUG
    dataset_records = _validate_dataset(dataset, negative=negative)
    derived_v03 = _validate_natural_records(
        args.v03_records,
        split="calibration",
        dataset_records=dataset_records,
    )
    if derived_v03["error_counts"] != dict(negative.error_counts):
        raise ValueError("v0.3 calibration taxonomy does not replay")
    scenes = select_stage1_v2_probe_scenes(
        _read_dataset_rows(dataset / "records.jsonl"),
        dataset_root=dataset,
    )
    if len(scenes) != probe.scenes:
        raise ValueError("Stage-1 v2 probe scene count differs from its config")
    if any(not scene.image_path.is_file() or scene.image_path.is_symlink() for scene in scenes):
        raise ValueError("Stage-1 v2 probe image is missing or is a symlink")
    output = paths.outputs / "recoverability_v1" / probe.output_subdirectory
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Stage-1 v2 probe output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt_marker = output.parent / f"{probe.output_subdirectory}.attempted.json"
    with attempt_marker.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "status": "STAGE1_V2_DEVELOPMENT_PROBE_STARTED",
                "hypothesis_test": False,
            },
            stream,
            sort_keys=True,
        )
        stream.write("\n")
    model_hash_before = model_snapshot_sha256(paths.model_path)
    if model_hash_before != negative.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the frozen v0.3 pilot")
    if model_hash_before != failure.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the failed Bridge v1")
    model, processor = load_local_qwen(paths.model_path)
    report, records = run_stage1_v2_probe(
        scenes,
        generate=lambda scene, messages: decode_qwen_once(
            model,
            processor,
            scene.image_path,
            messages,
        ),
    )
    if model_snapshot_sha256(paths.model_path) != model_hash_before:
        raise RuntimeError("model snapshot changed during the Stage-1 v2 probe")
    payload = {
        **asdict(report),
        "error_counts": dict(report.error_counts),
        "schema_version": 1,
        "artifact_type": "recoverability_stage1_v2_development_probe",
        "dataset_id": probe.dataset_id,
        "source_dataset_id": probe.source_dataset_id,
        "source_split": probe.source_split,
        "per_stratum": probe.per_stratum,
        "model_snapshot_sha256": model_hash_before,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
    }
    with tempfile.TemporaryDirectory(prefix=".stage1-v2-probe-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        with (staging / "probe_records.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n")
        (staging / "probe_report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.probe_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
