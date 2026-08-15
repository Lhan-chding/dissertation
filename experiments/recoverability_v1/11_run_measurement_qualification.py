#!/usr/bin/env python3
"""Run the one-shot measurement-only two-stage qualification."""

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

_LOCK_RELATIVE = "configs/recoverability/server_package_lock_measurement_qualification.yaml"
_INHERITED_LOCK_RELATIVE = (
    "configs/recoverability/server_package_lock_measurement_qualification_data.yaml"
)
_BOOTSTRAP_ADDITIONS = frozenset(
    {
        "configs/recoverability/measurement_qualification_data_anchor.yaml",
        _INHERITED_LOCK_RELATIVE,
        "experiments/recoverability_v1/10_measurement_qualification_preflight.py",
        "experiments/recoverability_v1/11_run_measurement_qualification.py",
        "src/compbias/recoverability/measurement_qualification_anchor.py",
        "src/compbias/recoverability/measurement_qualification_execution.py",
    }
)
_EXPECTED_MODEL_SHA256 = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lock_rows(path: Path) -> tuple[tuple[str, str], ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: measurement qualification package lock is malformed")
    return tuple(zip(paths, digests, strict=True))


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--execute" not in sys.argv:
        return
    try:
        supplied = Path(sys.argv[sys.argv.index("--server-package-lock") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical qualification package lock is required") from error
    root = Path(__file__).resolve().parents[2]
    canonical = root / _LOCK_RELATIVE
    inherited = root / _INHERITED_LOCK_RELATIVE
    if supplied != canonical or canonical.is_symlink() or inherited.is_symlink():
        raise SystemExit("BLOCKED: qualification package lock path is not canonical")
    inherited_paths = frozenset(relative for relative, _digest in _lock_rows(inherited))
    rows = _lock_rows(canonical)
    if frozenset(relative for relative, _digest in rows) != inherited_paths | _BOOTSTRAP_ADDITIONS:
        raise SystemExit("BLOCKED: qualification package lock closure is incomplete")
    for relative, expected in rows:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: qualification package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.config import load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import decode_qwen_once, load_local_qwen  # noqa: E402
from compbias.io.strict_json import load_strict_json_mapping  # noqa: E402
from compbias.recoverability.bridge import decode_text_qwen_once  # noqa: E402
from compbias.recoverability.measurement_qualification import (  # noqa: E402
    load_measurement_qualification_config,
    load_reserved_numeric_tables,
    run_measurement_qualification,
)
from compbias.recoverability.measurement_qualification_anchor import (  # noqa: E402
    load_measurement_qualification_data_anchor,
    verify_measurement_qualification_data_evidence,
)
from compbias.recoverability.measurement_qualification_data import (  # noqa: E402
    verify_measurement_qualification_dataset,
)
from compbias.recoverability.measurement_qualification_execution import (  # noqa: E402
    verify_measurement_qualification_execution_package_lock,
)
from compbias.recoverability.preflight import (  # noqa: E402
    MetadataPreflightReport,
    load_runtime_spec,
    run_metadata_preflight,
)
from compbias.recoverability.stage1_v2 import validate_stage1_v2_runtime_paths  # noqa: E402
from compbias.recoverability.stage2_v2_anchor import (  # noqa: E402
    load_stage2_v2_external_evidence_anchor,
    verify_stage2_v2_external_evidence,
)


def _metadata_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def _pip_check() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        env=_metadata_environment(),
    )
    return completed.returncode, completed.stdout + completed.stderr


def _pip_inventory() -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
        env=_metadata_environment(),
    )
    return {row["name"]: row["version"] for row in json.loads(completed.stdout)}


def _validate_preflight(
    path: Path,
    *,
    lock_path: Path,
    current: MetadataPreflightReport,
    package_files: tuple[str, ...],
) -> None:
    payload = load_strict_json_mapping(
        path,
        label="measurement qualification preflight",
        max_bytes=512 * 1024,
    )
    expected = {
        **asdict(current),
        "server_package_lock_verified": True,
        "server_package_lock_sha256": _sha256(lock_path),
        "server_package_files": list(package_files),
        "artifact_type": "recoverability_measurement_qualification_metadata_preflight",
        "schema_version": 1,
    }
    expected["installed_packages"] = [list(item) for item in current.installed_packages]
    if payload != expected:
        raise ValueError("preflight report does not authorize measurement qualification")


def _exclusive_marker(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to rerun measurement qualification")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("qualification output parent must be a regular directory")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-anchor", type=Path)
    parser.add_argument("--server-package-lock", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-attempt-marker", type=Path)
    parser.add_argument("--dataset-console-log", type=Path)
    parser.add_argument("--source-records", type=Path)
    parser.add_argument("--stage2-v2-external-evidence", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("BLOCKED: measurement qualification requires explicit --execute")
        return 2
    required = (
        args.paths,
        args.runtime,
        args.config,
        args.data_anchor,
        args.server_package_lock,
        args.preflight_report,
        args.dataset_root,
        args.dataset_attempt_marker,
        args.dataset_console_log,
        args.source_records,
        args.stage2_v2_external_evidence,
    )
    if any(value is None for value in required):
        print("BLOCKED: measurement qualification requires every frozen evidence input")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = Path(__file__).resolve().parents[2]
    canonical = {
        "paths": root / "configs/paths.yaml",
        "runtime": root / "configs/recoverability/server_runtime_v1.yaml",
        "config": root / "configs/recoverability/measurement_qualification_v1.yaml",
        "anchor": (root / "configs/recoverability/measurement_qualification_data_anchor.yaml"),
        "lock": root / _LOCK_RELATIVE,
    }
    supplied = {
        "paths": args.paths,
        "runtime": args.runtime,
        "config": args.config,
        "anchor": args.data_anchor,
        "lock": args.server_package_lock,
    }
    for label, expected in canonical.items():
        if supplied[label].resolve() != expected or supplied[label].is_symlink():
            raise ValueError(f"qualification {label} path is not canonical")
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("measurement qualification forbids COMPBIAS path overrides")
    validate_stage1_v2_runtime_paths(
        canonical["paths"],
        registered_example=root / "configs/paths.example.yaml",
    )
    package_lock = verify_measurement_qualification_execution_package_lock(
        canonical["lock"],
        repository_root=root,
    )
    current_preflight = run_metadata_preflight(
        load_runtime_spec(canonical["runtime"]),
        repository_root=root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    paths = load_pilot_paths(canonical["paths"], environ={})
    expected_evidence_root = Path("/cloud/cloud-ssd1/recoverability-v1-evidence")
    expected = {
        "preflight": expected_evidence_root / "measurement-qualification-preflight.json",
        "dataset": paths.data / "generated/measurement_qualification_v1",
        "dataset_attempt": paths.data / "generated/measurement_qualification_v1.attempted.json",
        "dataset_console": expected_evidence_root / "measurement-qualification-data.log",
        "source_records": paths.data / "generated/cva_chart_pilot_v0_3/records.jsonl",
        "external": expected_evidence_root / "stage2-v2-external-evidence.json",
    }
    actual = {
        "preflight": args.preflight_report,
        "dataset": args.dataset_root,
        "dataset_attempt": args.dataset_attempt_marker,
        "dataset_console": args.dataset_console_log,
        "source_records": args.source_records,
        "external": args.stage2_v2_external_evidence,
    }
    for label, expected_path in expected.items():
        if actual[label].resolve() != expected_path:
            raise ValueError(f"qualification {label} evidence path is not canonical")
    _validate_preflight(
        args.preflight_report,
        lock_path=canonical["lock"],
        current=current_preflight,
        package_files=tuple(item.relative_path for item in package_lock.files),
    )
    stage2_anchor = load_stage2_v2_external_evidence_anchor(
        root / "configs/recoverability/stage2_v2_external_evidence_anchor.yaml"
    )
    verify_stage2_v2_external_evidence(stage2_anchor, args.stage2_v2_external_evidence)
    config = load_measurement_qualification_config(canonical["config"])
    if config.format_retries != 0:
        raise ValueError("measurement qualification format_retries must remain zero")
    data_anchor = load_measurement_qualification_data_anchor(canonical["anchor"])
    reserved = load_reserved_numeric_tables(
        args.source_records,
        expected_sha256=config.source_dataset_records_sha256,
    )
    verify_measurement_qualification_data_evidence(
        data_anchor,
        dataset_root=args.dataset_root,
        attempt_marker=args.dataset_attempt_marker,
        console_log=args.dataset_console_log,
        config=config,
        reserved_numeric_tables=reserved,
    )
    if data_anchor.records != 300:
        raise ValueError("measurement qualification scene count differs from anchor")
    verified_dataset = verify_measurement_qualification_dataset(
        args.dataset_root,
        config=config,
        reserved_numeric_tables=reserved,
    )
    output = paths.outputs / "recoverability_v1/measurement_qualification_v1"
    attempt = output.parent / "measurement_qualification_v1.attempted.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite measurement qualification output")
    output.parent.mkdir(parents=True, exist_ok=True)
    model_hash_before = model_snapshot_sha256(paths.model_path)
    if model_hash_before != _EXPECTED_MODEL_SHA256:
        raise RuntimeError("model snapshot differs from frozen recoverability evidence")
    _exclusive_marker(
        attempt,
        {
            "schema_version": 1,
            "status": "MEASUREMENT_QUALIFICATION_STARTED",
            "dataset_id": config.dataset_id,
            "dataset_manifest_sha256": data_anchor.manifest_sha256,
            "dataset_records_sha256": data_anchor.records_sha256,
            "dataset_images_sha256": data_anchor.images_sha256,
            "server_package_lock_sha256": _sha256(canonical["lock"]),
            "model_snapshot_sha256": model_hash_before,
            "format_retries": 0,
            "hypothesis_tested": False,
            "confirmatory_execution_authorized": False,
            "training_invoked": False,
        },
    )
    model, processor = load_local_qwen(paths.model_path)
    report, records = run_measurement_qualification(
        verified_dataset.scenes,
        config=config,
        stage1_generate=lambda scene, messages: decode_qwen_once(
            model,
            processor,
            scene.image_path,
            messages,
        ),
        stage2_generate=lambda _scene, _perceived, messages: decode_text_qwen_once(
            model,
            processor,
            messages,
        ),
    )
    if report.model_calls > 600:
        raise RuntimeError("measurement qualification exceeded the registered model-call cap")
    if model_snapshot_sha256(paths.model_path) != model_hash_before:
        raise RuntimeError("model snapshot changed during measurement qualification")
    payload = {
        **asdict(report),
        "gate_failures": list(report.gate_failures),
        "schema_version": 1,
        "artifact_type": "recoverability_measurement_qualification_report",
        "dataset_id": config.dataset_id,
        "dataset_manifest_sha256": data_anchor.manifest_sha256,
        "dataset_records_sha256": data_anchor.records_sha256,
        "dataset_images_sha256": data_anchor.images_sha256,
        "dataset_attempt_marker_sha256": data_anchor.attempt_marker_sha256,
        "dataset_console_sha256": data_anchor.console_sha256,
        "server_package_lock_sha256": _sha256(canonical["lock"]),
        "model_snapshot_sha256": model_hash_before,
        "format_retries": 0,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
        "training_invoked": False,
    }
    with tempfile.TemporaryDirectory(
        prefix=".measurement-qualification-",
        dir=output.parent,
    ) as tmp:
        staging = Path(tmp) / output.name
        staging.mkdir()
        with (staging / "qualification_records.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n")
        with (staging / "qualification_report.json").open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        staging.rename(output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.qualification_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
