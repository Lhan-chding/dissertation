#!/usr/bin/env python3
"""Run the one-shot 8,000-scene screen for amended confirmatory Phase C v2."""

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

_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_c_screen_v2.yaml"
_INHERITED_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_n.yaml"
_BOOTSTRAP_ADDITIONS = frozenset(
    {
        "configs/recoverability/phase_n_frozen_result.yaml",
        "configs/recoverability/recoverability_phase_c_v2_amendment.yaml",
        _INHERITED_LOCK_RELATIVE,
        "experiments/recoverability_v1/14_phase_c_screen_preflight.py",
        "experiments/recoverability_v1/15_run_phase_c_screen.py",
        "src/compbias/recoverability/compatibility.py",
        "src/compbias/recoverability/operators.py",
        "src/compbias/recoverability/phase_c_amendment.py",
        "src/compbias/recoverability/phase_c_screen.py",
        "src/compbias/recoverability/phase_c_screen_execution.py",
        "src/compbias/recoverability/phase_n_result.py",
        "src/compbias/recoverability/selection.py",
    }
)
_EXPECTED_MODEL_SHA256 = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
_SOURCE_RECORDS_SHA256 = "92ccdf54b11e2a6c12e12ef5273137824c6f3b94f38224abeb32d8319b83a62b"
_QUALIFICATION_RECORDS_SHA256 = "98c1ab1228480b58dc4309f7c64280c347e87ac44547d79e36ab6ceb52adff6d"


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
        raise SystemExit("BLOCKED: Phase C screen package lock is malformed")
    return tuple(zip(paths, digests, strict=True))


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--execute" not in sys.argv:
        return
    try:
        supplied = Path(sys.argv[sys.argv.index("--server-package-lock") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical Phase C screen lock is required") from error
    root = Path(__file__).resolve().parents[2]
    canonical = root / _LOCK_RELATIVE
    inherited = root / _INHERITED_LOCK_RELATIVE
    if supplied != canonical or canonical.is_symlink() or inherited.is_symlink():
        raise SystemExit("BLOCKED: Phase C screen package lock path is not canonical")
    inherited_paths = frozenset(relative for relative, _ in _lock_rows(inherited))
    rows = _lock_rows(canonical)
    if frozenset(relative for relative, _ in rows) != inherited_paths | _BOOTSTRAP_ADDITIONS:
        raise SystemExit("BLOCKED: Phase C screen package lock closure is incomplete")
    for relative, expected in rows:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: Phase C screen package mismatch for {relative}")


_bootstrap_server_lock()

from compbias.gpu_pilot.config import load_pilot_paths  # noqa: E402
from compbias.gpu_pilot.preflight import model_snapshot_sha256  # noqa: E402
from compbias.gpu_pilot.qwen_smoke import decode_qwen_once, load_local_qwen  # noqa: E402
from compbias.io.strict_json import load_strict_json_mapping  # noqa: E402
from compbias.recoverability.phase_c_amendment import load_phase_c_amendment  # noqa: E402
from compbias.recoverability.phase_c_screen import (  # noqa: E402
    evaluate_phase_c_screen,
    verify_phase_c_screen_dataset,
    write_phase_c_screen_dataset,
)
from compbias.recoverability.phase_c_screen_execution import (  # noqa: E402
    verify_phase_c_screen_execution_package_lock,
)
from compbias.recoverability.phase_n_result import (  # noqa: E402
    load_phase_n_frozen_result,
    verify_phase_n_result_artifacts,
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


def _load_tables(path: Path, *, expected_sha256: str) -> frozenset[tuple[int, int, int, int]]:
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError(f"numeric-table source differs from frozen evidence: {path.name}")
    tables: set[tuple[int, int, int, int]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            values = row.get("values") if isinstance(row, dict) else None
            if (
                not isinstance(values, list)
                or len(values) != 4
                or any(type(value) is not int for value in values)
            ):
                raise ValueError("numeric-table source row is invalid")
            tables.add(tuple(values))  # type: ignore[arg-type]
    if not tables:
        raise ValueError("numeric-table source must not be empty")
    return frozenset(tables)


def _exclusive_marker(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to rerun the Phase C screen")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("Phase C screen output parent must be a regular directory")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _validate_preflight(path: Path, *, lock: Path, package_files: list[str]) -> None:
    payload = load_strict_json_mapping(path, label="Phase C screen preflight", max_bytes=512 * 1024)
    if payload.get("artifact_type") != "recoverability_phase_c_v2_screen_metadata_preflight":
        raise ValueError("Phase C screen preflight artifact type is invalid")
    if payload.get("ready") is not True or payload.get("model_loaded") is not False:
        raise ValueError("Phase C screen preflight is not ready")
    if payload.get("training_authorized") is not False:
        raise ValueError("Phase C screen preflight must not authorize training")
    if payload.get("server_package_lock_sha256") != _sha256(lock):
        raise ValueError("Phase C screen preflight lock digest differs")
    if payload.get("server_package_files") != package_files:
        raise ValueError("Phase C screen preflight package closure differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--phase-n-result", type=Path)
    parser.add_argument("--server-package-lock", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--phase-n-preflight", type=Path)
    parser.add_argument("--phase-n-attempt-marker", type=Path)
    parser.add_argument("--phase-n-dataset-root", type=Path)
    parser.add_argument("--phase-n-output-root", type=Path)
    parser.add_argument("--phase-n-console-log", type=Path)
    parser.add_argument("--source-records", type=Path)
    parser.add_argument("--qualification-records", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print("BLOCKED: Phase C screen requires explicit --execute")
        return 2
    required = tuple(value for key, value in vars(args).items() if key != "execute")
    if any(value is None for value in required):
        print("BLOCKED: Phase C screen requires every frozen evidence input")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("Phase C screen forbids COMPBIAS path overrides")

    root = Path(__file__).resolve().parents[2]
    canonical = {
        "paths": root / "configs/paths.yaml",
        "runtime": root / "configs/recoverability/server_runtime_v1.yaml",
        "amendment": root / "configs/recoverability/recoverability_phase_c_v2_amendment.yaml",
        "phase_n_result": root / "configs/recoverability/phase_n_frozen_result.yaml",
        "lock": root / _LOCK_RELATIVE,
    }
    for key, expected in canonical.items():
        argument = "server_package_lock" if key == "lock" else key
        supplied = getattr(args, argument)
        if supplied.resolve() != expected or supplied.is_symlink():
            raise ValueError(f"Phase C screen {key} path is not canonical")
    package = verify_phase_c_screen_execution_package_lock(canonical["lock"], repository_root=root)
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
        raise RuntimeError("Phase C screen runtime preflight failed")
    paths = load_pilot_paths(canonical["paths"], environ={})
    evidence_root = Path("/cloud/cloud-ssd1/recoverability-v1-evidence")
    phase_n_output = paths.outputs / "recoverability_v1/cva_natural_prevalence_v1"
    phase_n_dataset = paths.data / "generated/cva_natural_prevalence_v1"
    expected_paths = {
        "preflight_report": evidence_root / "phase-c-screen-v2-preflight.json",
        "phase_n_preflight": evidence_root / "phase-n-preflight.json",
        "phase_n_attempt_marker": (
            paths.outputs / "recoverability_v1/cva_natural_prevalence_v1.attempted.json"
        ),
        "phase_n_dataset_root": phase_n_dataset,
        "phase_n_output_root": phase_n_output,
        "phase_n_console_log": evidence_root / "phase-n-console.log",
        "source_records": paths.data / "generated/cva_chart_pilot_v0_3/records.jsonl",
        "qualification_records": (
            paths.data / "generated/measurement_qualification_v1/records.jsonl"
        ),
    }
    for label, expected in expected_paths.items():
        if getattr(args, label).resolve() != expected:
            raise ValueError(f"Phase C screen {label} path is not canonical")
    _validate_preflight(args.preflight_report, lock=canonical["lock"], package_files=package_files)

    phase_n = load_phase_n_frozen_result(canonical["phase_n_result"])
    verify_phase_n_result_artifacts(
        phase_n,
        preflight=args.phase_n_preflight,
        attempt_marker=args.phase_n_attempt_marker,
        dataset_manifest=args.phase_n_dataset_root / "manifest.json",
        dataset_records=args.phase_n_dataset_root / "records.jsonl",
        report=args.phase_n_output_root / "phase_n_report.json",
        records=args.phase_n_output_root / "phase_n_records.jsonl",
        console_log=args.phase_n_console_log,
    )
    amendment = load_phase_c_amendment(canonical["amendment"], phase_n=phase_n)
    if not amendment.confirmatory_phase_c_authorized or amendment.training_authorized:
        raise RuntimeError("Phase C amendment does not authorize the screen")

    reserved = set(_load_tables(args.source_records, expected_sha256=_SOURCE_RECORDS_SHA256))
    reserved.update(
        _load_tables(
            args.qualification_records,
            expected_sha256=_QUALIFICATION_RECORDS_SHA256,
        )
    )
    reserved.update(
        _load_tables(
            args.phase_n_dataset_root / "records.jsonl",
            expected_sha256=dict(phase_n.source_sha256)["dataset_records"],
        )
    )
    output_parent = paths.outputs / "recoverability_v1"
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / amendment.output_subdirectory / "phase_c_screen"
    attempt = output_parent / f"{amendment.output_subdirectory}.screen.attempted.json"
    dataset_root = paths.data / "generated/cva_recoverability_causal_v2_screen"
    if output.exists() or output.is_symlink() or dataset_root.exists() or dataset_root.is_symlink():
        raise FileExistsError("refusing to overwrite Phase C screen evidence")
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    model_hash = model_snapshot_sha256(paths.model_path)
    if model_hash != _EXPECTED_MODEL_SHA256 or model_hash != phase_n.model_snapshot_sha256:
        raise RuntimeError("model snapshot differs from the frozen Phase N evidence")
    _exclusive_marker(
        attempt,
        {
            "schema_version": 1,
            "status": "PHASE_C_V2_SCREEN_STARTED_DO_NOT_RERUN",
            "amendment_id": amendment.amendment_id,
            "original_phase_n_gate_passed": False,
            "original_phase_n_inconclusive": True,
            "amended_continuation_threshold": 0.10,
            "observed_phase_n_one_sided_cp_upper": phase_n.one_sided_cp_upper,
            "dataset_id": amendment.dataset_id,
            "intake_scenes": 8000,
            "model_call_cap": 8000,
            "format_retries": 0,
            "allow_sample_extension": False,
            "allow_quota_redistribution": False,
            "server_package_lock_sha256": _sha256(canonical["lock"]),
            "model_snapshot_sha256": model_hash,
            "training_invoked": False,
        },
    )
    manifest = write_phase_c_screen_dataset(
        amendment,
        reserved_numeric_tables=reserved,
        output_dir=dataset_root,
    )
    verified_manifest, scenes = verify_phase_c_screen_dataset(
        amendment,
        reserved_numeric_tables=reserved,
        dataset_root=dataset_root,
    )
    if manifest != verified_manifest or len(scenes) != 8000:
        raise RuntimeError("Phase C screen dataset replay failed")

    max_format_retries = 0
    if max_format_retries != amendment.format_retries:
        raise RuntimeError("Phase C screen requires max_format_retries=0")
    model, processor = load_local_qwen(paths.model_path)
    scene_by_id = {scene.record.scene_id: scene for scene in scenes}

    def generate(record, messages):
        return decode_qwen_once(
            model,
            processor,
            scene_by_id[record.scene_id].image_path,
            messages,
        )

    report, rows = evaluate_phase_c_screen(
        tuple(scene.record for scene in scenes),
        amendment=amendment,
        generate=generate,
    )
    if report.model_calls != 8000 or len(rows) != 8000:
        raise RuntimeError("Phase C screen did not stop at exactly 8,000 calls")
    if model_snapshot_sha256(paths.model_path) != model_hash:
        raise RuntimeError("model snapshot changed during the Phase C screen")
    payload = {
        **asdict(report),
        "eligible_by_family": dict(report.eligible_by_family),
        "selected_by_family": dict(report.selected_by_family),
        "selected_scene_ids": list(report.selected_scene_ids),
        "failure_codes": list(report.failure_codes),
        "schema_version": 1,
        "artifact_type": "recoverability_phase_c_v2_screen_report",
        "amendment_id": amendment.amendment_id,
        "dataset_id": amendment.dataset_id,
        "dataset_manifest_sha256": _sha256(dataset_root / "manifest.json"),
        "dataset_records_sha256": manifest["records_sha256"],
        "dataset_images_sha256": manifest["images_sha256"],
        "phase_n_report_sha256": dict(phase_n.source_sha256)["phase_n_report"],
        "server_package_lock_sha256": _sha256(canonical["lock"]),
        "model_snapshot_sha256": model_hash,
        "original_phase_n_gate_passed": False,
        "original_phase_n_inconclusive": True,
        "amended_continuation_threshold": 0.10,
        "training_invoked": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phase-c-screen-", dir=output.parent) as temp:
        staging = Path(temp) / output.name
        staging.mkdir()
        with (staging / "screen_records.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(asdict(row), sort_keys=True, allow_nan=False) + "\n")
        with (staging / "screen_report.json").open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        staging.rename(output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.screen_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
