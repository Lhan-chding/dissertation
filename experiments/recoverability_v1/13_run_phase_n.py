#!/usr/bin/env python3
"""Run the fixed one-shot original-protocol Phase-N prevalence screen."""

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
from functools import partial
from pathlib import Path

_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_n.yaml"
_INHERITED_LOCK_RELATIVE = (
    "configs/recoverability/server_package_lock_measurement_qualification.yaml"
)
_BOOTSTRAP_ADDITIONS = frozenset(
    {
        "configs/recoverability/measurement_qualification_frozen_result.yaml",
        _INHERITED_LOCK_RELATIVE,
        "experiments/recoverability_v1/12_phase_n_preflight.py",
        "experiments/recoverability_v1/13_run_phase_n.py",
        "src/compbias/recoverability/measurement_qualification_result.py",
        "src/compbias/recoverability/natural_inference.py",
        "src/compbias/recoverability/phase_n.py",
        "src/compbias/recoverability/phase_n_execution.py",
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
        raise SystemExit("BLOCKED: Phase N package lock is malformed")
    return tuple(zip(paths, digests, strict=True))


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--execute" not in sys.argv:
        return
    try:
        supplied = Path(sys.argv[sys.argv.index("--server-package-lock") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical Phase N package lock is required") from error
    root = Path(__file__).resolve().parents[2]
    canonical = root / _LOCK_RELATIVE
    inherited = root / _INHERITED_LOCK_RELATIVE
    if supplied != canonical or canonical.is_symlink() or inherited.is_symlink():
        raise SystemExit("BLOCKED: Phase N package lock path is not canonical")
    inherited_paths = frozenset(relative for relative, _ in _lock_rows(inherited))
    rows = _lock_rows(canonical)
    if frozenset(relative for relative, _ in rows) != inherited_paths | _BOOTSTRAP_ADDITIONS:
        raise SystemExit("BLOCKED: Phase N package lock closure is incomplete")
    for relative, expected in rows:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: Phase N package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.config import load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import decode_qwen_once, load_local_qwen  # noqa: E402
from compbias.gpu_pilot.structured_generation import generate_with_format_retries  # noqa: E402
from compbias.io.strict_json import load_strict_json_mapping  # noqa: E402
from compbias.recoverability.config import load_recoverability_protocol  # noqa: E402
from compbias.recoverability.measurement_qualification import (  # noqa: E402
    load_measurement_qualification_config,
    load_reserved_numeric_tables,
)
from compbias.recoverability.measurement_qualification_anchor import (  # noqa: E402
    load_measurement_qualification_data_anchor,
    verify_measurement_qualification_data_evidence,
)
from compbias.recoverability.measurement_qualification_data import (  # noqa: E402
    verify_measurement_qualification_dataset,
)
from compbias.recoverability.measurement_qualification_result import (  # noqa: E402
    load_measurement_qualification_frozen_result,
    verify_measurement_qualification_result_artifacts,
)
from compbias.recoverability.phase_n import (  # noqa: E402
    run_phase_n,
    verify_phase_n_dataset,
    write_phase_n_dataset,
)
from compbias.recoverability.phase_n_execution import (  # noqa: E402
    verify_phase_n_execution_package_lock,
)
from compbias.recoverability.preflight import (  # noqa: E402
    load_runtime_spec,
    run_metadata_preflight,
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


def _exclusive_marker(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to rerun Phase N")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("Phase N output parent must be a regular directory")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _validate_preflight(path: Path, *, lock: Path, package_files: list[str]) -> None:
    payload = load_strict_json_mapping(path, label="Phase N preflight", max_bytes=512 * 1024)
    if payload.get("artifact_type") != "recoverability_phase_n_metadata_preflight":
        raise ValueError("Phase N preflight artifact type is invalid")
    if payload.get("ready") is not True or payload.get("model_loaded") is not False:
        raise ValueError("Phase N preflight is not ready")
    if payload.get("training_authorized") is not False:
        raise ValueError("Phase N preflight must not authorize training")
    if payload.get("server_package_lock_sha256") != _sha256(lock):
        raise ValueError("Phase N preflight lock digest differs")
    if payload.get("server_package_files") != package_files:
        raise ValueError("Phase N preflight package closure differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--server-package-lock", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--qualification-preflight", type=Path)
    parser.add_argument("--qualification-attempt-marker", type=Path)
    parser.add_argument("--qualification-report", type=Path)
    parser.add_argument("--qualification-records", type=Path)
    parser.add_argument("--qualification-console-log", type=Path)
    parser.add_argument("--source-records", type=Path)
    parser.add_argument("--qualification-dataset-root", type=Path)
    parser.add_argument("--qualification-dataset-attempt-marker", type=Path)
    parser.add_argument("--qualification-dataset-console-log", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print("BLOCKED: Phase N requires explicit --execute")
        return 2
    required = tuple(value for key, value in vars(args).items() if key != "execute")
    if any(value is None for value in required):
        print("BLOCKED: Phase N requires every frozen evidence input")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("Phase N forbids COMPBIAS path overrides")
    root = Path(__file__).resolve().parents[2]
    canonical = {
        "paths": root / "configs/paths.yaml",
        "runtime": root / "configs/recoverability/server_runtime_v1.yaml",
        "protocol": root / "configs/recoverability/recoverability_v1.yaml",
        "lock": root / _LOCK_RELATIVE,
    }
    for key in canonical:
        supplied = getattr(args, "server_package_lock" if key == "lock" else key)
        if supplied.resolve() != canonical[key] or supplied.is_symlink():
            raise ValueError(f"Phase N {key} path is not canonical")
    package = verify_phase_n_execution_package_lock(canonical["lock"], repository_root=root)
    package_files = [item.relative_path for item in package.files]
    current = run_metadata_preflight(
        load_runtime_spec(canonical["runtime"]),
        repository_root=root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    if not current.ready or current.large_gpu_started or current.model_loaded:
        raise RuntimeError("Phase N runtime preflight failed")
    paths = load_pilot_paths(canonical["paths"], environ={})
    evidence_root = Path("/cloud/cloud-ssd1/recoverability-v1-evidence")
    qualification_output = paths.outputs / "recoverability_v1/measurement_qualification_v1"
    expected_paths = {
        "preflight_report": evidence_root / "phase-n-preflight.json",
        "qualification_preflight": evidence_root / "measurement-qualification-preflight.json",
        "qualification_attempt_marker": (
            paths.outputs / "recoverability_v1/measurement_qualification_v1.attempted.json"
        ),
        "qualification_report": qualification_output / "qualification_report.json",
        "qualification_records": qualification_output / "qualification_records.jsonl",
        "qualification_console_log": evidence_root / "measurement-qualification-console.log",
        "source_records": paths.data / "generated/cva_chart_pilot_v0_3/records.jsonl",
        "qualification_dataset_root": paths.data / "generated/measurement_qualification_v1",
        "qualification_dataset_attempt_marker": (
            paths.data / "generated/measurement_qualification_v1.attempted.json"
        ),
        "qualification_dataset_console_log": (evidence_root / "measurement-qualification-data.log"),
    }
    for label, expected in expected_paths.items():
        if getattr(args, label).resolve() != expected:
            raise ValueError(f"Phase N {label} path is not canonical")
    _validate_preflight(
        args.preflight_report,
        lock=canonical["lock"],
        package_files=package_files,
    )

    qualification_config = load_measurement_qualification_config(
        root / "configs/recoverability/measurement_qualification_v1.yaml"
    )
    source_tables = load_reserved_numeric_tables(
        args.source_records,
        expected_sha256=qualification_config.source_dataset_records_sha256,
    )
    data_anchor = load_measurement_qualification_data_anchor(
        root / "configs/recoverability/measurement_qualification_data_anchor.yaml"
    )
    verify_measurement_qualification_data_evidence(
        data_anchor,
        dataset_root=args.qualification_dataset_root,
        attempt_marker=args.qualification_dataset_attempt_marker,
        console_log=args.qualification_dataset_console_log,
        config=qualification_config,
        reserved_numeric_tables=source_tables,
    )
    qualification_dataset = verify_measurement_qualification_dataset(
        args.qualification_dataset_root,
        config=qualification_config,
        reserved_numeric_tables=source_tables,
    )
    frozen_result = load_measurement_qualification_frozen_result(
        root / "configs/recoverability/measurement_qualification_frozen_result.yaml"
    )
    verify_measurement_qualification_result_artifacts(
        frozen_result,
        preflight=args.qualification_preflight,
        attempt_marker=args.qualification_attempt_marker,
        report=args.qualification_report,
        records=args.qualification_records,
        console_log=args.qualification_console_log,
    )
    if not frozen_result.qualification_passed:
        raise RuntimeError("measurement qualification did not pass")

    protocol = load_recoverability_protocol(canonical["protocol"])
    if protocol.phase_n.scenes != 4000 or protocol.phase_n.max_format_retries != 0:
        raise RuntimeError("Phase N preregistration differs from 4,000 zero-retry calls")
    reserved_tables = source_tables | frozenset(
        scene.values for scene in qualification_dataset.scenes
    )
    output = paths.outputs / f"recoverability_v1/{protocol.phase_n.output_subdirectory}"
    attempt = output.parent / f"{protocol.phase_n.output_subdirectory}.attempted.json"
    dataset_root = paths.data / f"generated/{protocol.phase_n.output_subdirectory}"
    if output.exists() or output.is_symlink() or dataset_root.exists() or dataset_root.is_symlink():
        raise FileExistsError("refusing to overwrite Phase N evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    model_hash = model_snapshot_sha256(paths.model_path)
    if model_hash != _EXPECTED_MODEL_SHA256 or model_hash != frozen_result.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from frozen evidence")
    _exclusive_marker(
        attempt,
        {
            "schema_version": 1,
            "status": "PHASE_N_STARTED_DO_NOT_RERUN",
            "dataset_id": protocol.phase_n.dataset_id,
            "seed": protocol.phase_n.seed,
            "scenes": 4000,
            "model_call_cap": 4000,
            "format_retries": 0,
            "allow_sample_extension": False,
            "server_package_lock_sha256": _sha256(canonical["lock"]),
            "model_snapshot_sha256": model_hash,
            "measurement_qualification_report_sha256": dict(frozen_result.source_sha256)[
                "qualification_report"
            ],
            "training_invoked": False,
        },
    )
    manifest = write_phase_n_dataset(
        protocol.phase_n,
        reserved_numeric_tables=reserved_tables,
        output_dir=dataset_root,
    )
    verified_manifest, scenes = verify_phase_n_dataset(
        protocol.phase_n,
        reserved_numeric_tables=reserved_tables,
        dataset_root=dataset_root,
    )
    if manifest != verified_manifest or len(scenes) != 4000:
        raise RuntimeError("Phase N dataset replay failed")

    model, processor = load_local_qwen(paths.model_path)

    def generate(scene_record, _messages):
        scene = scenes[int(scene_record.sample_id.rsplit("-", 1)[1])]
        generation = generate_with_format_retries(
            partial(decode_qwen_once, model, processor, scene.image_path),
            question=scene_record.question,
            operation=scene_record.operation,
            sample_id=scene_record.sample_id,
            expected_value_count=4,
            max_format_retries=0,
        )
        return generation.raw_text

    report, rows = run_phase_n(
        tuple(scene.record for scene in scenes),
        phase_config=protocol.phase_n,
        analysis_config=protocol.analysis,
        generate=generate,
    )
    if report.model_calls != 4000 or len(rows) != 4000:
        raise RuntimeError("Phase N did not stop at exactly 4,000 calls")
    if model_snapshot_sha256(paths.model_path) != model_hash:
        raise RuntimeError("model snapshot changed during Phase N")
    payload = {
        **asdict(report),
        "error_counts": dict(report.error_counts),
        "strata_counts": dict(report.strata_counts),
        "schema_version": 1,
        "artifact_type": "recoverability_phase_n_report",
        "dataset_id": protocol.phase_n.dataset_id,
        "dataset_manifest_sha256": _sha256(dataset_root / "manifest.json"),
        "dataset_records_sha256": manifest["records_sha256"],
        "dataset_images_sha256": manifest["images_sha256"],
        "measurement_qualification_report_sha256": dict(frozen_result.source_sha256)[
            "qualification_report"
        ],
        "server_package_lock_sha256": _sha256(canonical["lock"]),
        "model_snapshot_sha256": model_hash,
        "training_invoked": False,
    }
    with tempfile.TemporaryDirectory(prefix=".phase-n-", dir=output.parent) as temp:
        staging = Path(temp) / output.name
        staging.mkdir()
        with (staging / "phase_n_records.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(asdict(row), sort_keys=True, allow_nan=False) + "\n")
        with (staging / "phase_n_report.json").open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        staging.rename(output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.h1_supported else 3


if __name__ == "__main__":
    raise SystemExit(main())
