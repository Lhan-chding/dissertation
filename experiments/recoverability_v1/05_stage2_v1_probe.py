#!/usr/bin/env python3
"""Run the one-shot, text-only Stage-2 v1 probe on frozen Stage-1 evidence."""

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
        "configs/recoverability/stage2_v1_probe.yaml",
        "configs/recoverability/v0_3_negative_pilot.yaml",
        "experiments/recoverability_v1/00_preflight.py",
        "experiments/recoverability_v1/00_stage1_v2_preflight.py",
        "experiments/recoverability_v1/00_stage2_v1_preflight.py",
        "experiments/recoverability_v1/02_capture_v03_evidence.py",
        "experiments/recoverability_v1/03_bridge.py",
        "experiments/recoverability_v1/04_stage1_v2_probe.py",
        "experiments/recoverability_v1/05_stage2_v1_probe.py",
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
        "src/compbias/recoverability/stage2_v1.py",
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
        raise SystemExit("BLOCKED: canonical Stage-2 v1 server package lock is required") from error
    root = Path(__file__).resolve().parents[2]
    lock_path = Path(raw).resolve()
    canonical = root / "configs/recoverability/server_package_lock_stage2_v1.yaml"
    if lock_path != canonical:
        raise SystemExit("BLOCKED: Stage-2 v1 server package lock path is not canonical")
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths:
        raise SystemExit("BLOCKED: Stage-2 v1 server package lock is malformed")
    if frozenset(paths) != _BOOTSTRAP_SERVER_PATHS or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: Stage-2 v1 server package lock closure is incomplete")
    for relative, expected in zip(paths, digests, strict=True):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: Stage-2 v1 package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.collection import calibration_gate  # noqa: E402
from compbias.gpu_pilot.config import ACTIVE_PILOT_OUTPUT_SLUG, load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.execution_gate import (  # noqa: E402
    _validate_dataset_bundle,
    _validate_manifest,
    _validate_natural_records,
)
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import load_local_qwen  # noqa: E402
from compbias.recoverability.bridge import decode_text_qwen_once  # noqa: E402
from compbias.recoverability.evidence import load_negative_pilot_record  # noqa: E402
from compbias.recoverability.preflight import (  # noqa: E402
    MetadataPreflightReport,
    load_runtime_spec,
    run_metadata_preflight,
)
from compbias.recoverability.stage1_v2 import (  # noqa: E402
    select_stage1_v2_probe_scenes,
    validate_stage1_v2_runtime_paths,
)
from compbias.recoverability.stage2_v1 import (  # noqa: E402
    load_stage1_v2_frozen_result,
    load_stage2_v1_probe_config,
    run_stage2_v1_probe,
    verify_stage1_v2_frozen_artifacts,
    verify_stage2_v1_server_package_lock,
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
    payload = _load_json(path, label="Stage-2 v1 preflight report")
    if (
        payload.get("artifact_type") != "recoverability_stage2_v1_metadata_preflight"
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
        raise ValueError("preflight report does not authorize the Stage-2 v1 probe")


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
    parser.add_argument("--stage1-result", type=Path)
    parser.add_argument("--server-package-lock", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--external-evidence", type=Path)
    parser.add_argument("--v03-records", type=Path)
    parser.add_argument("--stage1-preflight", type=Path)
    parser.add_argument("--stage1-console-log", type=Path)
    parser.add_argument("--stage1-report", type=Path)
    parser.add_argument("--stage1-records", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("BLOCKED: Stage-2 v1 probe requires explicit --execute on the reviewed GPU server")
        return 2
    required = (
        args.paths,
        args.runtime,
        args.probe_config,
        args.stage1_result,
        args.server_package_lock,
        args.preflight_report,
        args.external_evidence,
        args.v03_records,
        args.stage1_preflight,
        args.stage1_console_log,
        args.stage1_report,
        args.stage1_records,
    )
    if any(value is None for value in required):
        print("BLOCKED: Stage-2 v1 probe requires every frozen evidence input")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    repository_root = Path(__file__).resolve().parents[2]
    canonical = {
        "lock": repository_root / "configs/recoverability/server_package_lock_stage2_v1.yaml",
        "probe": repository_root / "configs/recoverability/stage2_v1_probe.yaml",
        "stage1": repository_root / "configs/recoverability/stage1_v2_frozen_result.yaml",
        "runtime": repository_root / "configs/recoverability/server_runtime_v1.yaml",
        "paths": repository_root / "configs/paths.yaml",
    }
    supplied = {
        "lock": args.server_package_lock,
        "probe": args.probe_config,
        "stage1": args.stage1_result,
        "runtime": args.runtime,
        "paths": args.paths,
    }
    for label, expected in canonical.items():
        if supplied[label].resolve() != expected:
            raise ValueError(f"Stage-2 v1 {label} path is not canonical")
    validate_stage1_v2_runtime_paths(
        args.paths,
        registered_example=repository_root / "configs/paths.example.yaml",
    )
    verify_stage2_v1_server_package_lock(canonical["lock"], repository_root=repository_root)
    current_preflight = run_metadata_preflight(
        load_runtime_spec(canonical["runtime"]),
        repository_root=repository_root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    _validate_preflight(
        args.preflight_report,
        lock_path=canonical["lock"],
        current=current_preflight,
    )
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("Stage-2 v1 probe forbids COMPBIAS path overrides")
    paths = load_pilot_paths(args.paths)
    expected_v03 = paths.trajectories / "natural" / "calibration_records_v0_3.jsonl"
    expected_stage1 = paths.outputs / "recoverability_v1" / "stage1_v2_dev_probe"
    if args.v03_records.resolve() != expected_v03:
        raise ValueError("Stage-2 v1 calibration evidence path is not canonical")
    if args.stage1_report.resolve() != expected_stage1 / "probe_report.json":
        raise ValueError("Stage-2 v1 Stage-1 report path is not canonical")
    if args.stage1_records.resolve() != expected_stage1 / "probe_records.jsonl":
        raise ValueError("Stage-2 v1 Stage-1 records path is not canonical")
    negative = load_negative_pilot_record(
        repository_root / "configs/recoverability/v0_3_negative_pilot.yaml"
    )
    _validate_external_v03(args.external_evidence, records_path=args.v03_records, negative=negative)
    dataset = paths.data / "generated" / ACTIVE_PILOT_OUTPUT_SLUG
    dataset_records = _validate_dataset(dataset, negative=negative)
    replay = _validate_natural_records(
        args.v03_records,
        split="calibration",
        dataset_records=dataset_records,
    )
    if replay["error_counts"] != dict(negative.error_counts):
        raise ValueError("v0.3 calibration taxonomy does not replay")
    canonical_scenes = select_stage1_v2_probe_scenes(
        _read_dataset_rows(dataset / "records.jsonl"),
        dataset_root=dataset,
    )
    frozen_stage1 = load_stage1_v2_frozen_result(canonical["stage1"])
    verified_stage1 = verify_stage1_v2_frozen_artifacts(
        frozen_stage1,
        preflight_path=args.stage1_preflight,
        console_path=args.stage1_console_log,
        report_path=args.stage1_report,
        records_path=args.stage1_records,
        canonical_scenes=canonical_scenes,
    )
    probe = load_stage2_v1_probe_config(canonical["probe"])
    if len(verified_stage1.scenes) != probe.scenes:
        raise ValueError("Stage-2 v1 scene count differs from its config")
    output = paths.outputs / "recoverability_v1" / probe.output_subdirectory
    attempt_marker = output.parent / f"{probe.output_subdirectory}.attempted.json"
    if (
        output.exists()
        or output.is_symlink()
        or attempt_marker.exists()
        or attempt_marker.is_symlink()
    ):
        raise FileExistsError("refusing to rerun or overwrite the Stage-2 v1 probe")
    output.parent.mkdir(parents=True, exist_ok=True)
    with attempt_marker.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "status": "STAGE2_V1_DEVELOPMENT_PROBE_STARTED",
                "hypothesis_test": False,
            },
            stream,
            sort_keys=True,
        )
        stream.write("\n")
    model_hash_before = model_snapshot_sha256(paths.model_path)
    if model_hash_before != negative.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the frozen v0.3 pilot")
    if model_hash_before != frozen_stage1.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the Stage-1 v2 probe")
    model, processor = load_local_qwen(paths.model_path)
    report, records = run_stage2_v1_probe(
        verified_stage1.scenes,
        generate=lambda _scene, messages: decode_text_qwen_once(model, processor, messages),
    )
    if model_snapshot_sha256(paths.model_path) != model_hash_before:
        raise RuntimeError("model snapshot changed during the Stage-2 v1 probe")
    payload = {
        **asdict(report),
        "error_counts": dict(report.error_counts),
        "schema_version": 1,
        "artifact_type": "recoverability_stage2_v1_development_probe",
        "dataset_id": probe.dataset_id,
        "source_dataset_id": probe.source_dataset_id,
        "source_split": probe.source_split,
        "model_snapshot_sha256": model_hash_before,
        "source_stage1_records_sha256": _sha256(args.stage1_records),
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
    }
    with tempfile.TemporaryDirectory(prefix=".stage2-v1-probe-", dir=output.parent) as temporary:
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
