"""Fail-closed capture of the completed, failed v0.3 calibration evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from compbias.gpu_pilot.execution_gate import (
    _DATASET_RECORD_KEYS,
    _validate_natural_records,
)
from compbias.recoverability.evidence import load_negative_pilot_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str, expected_basename: str) -> Path:
    if path.name != expected_basename:
        raise ValueError(f"{label} must retain canonical basename {expected_basename}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path


@dataclass(frozen=True, slots=True)
class CapturedSourceFile:
    basename: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class V03EvidenceCaptureReport:
    verified: bool
    records: int
    gate_passed: bool
    calibration_exit: int
    source_files: tuple[CapturedSourceFile, ...]


def _load_summary(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v0.3 summary must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("v0.3 summary must be a JSON object")
    return payload


def _verify_records(path: Path, *, expected_records: int) -> dict[str, object]:
    seen: set[str] = set()
    sources: dict[str, dict[str, object]] = {}
    count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError("v0.3 records must not contain blank lines")
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) < _DATASET_RECORD_KEYS:
                    raise ValueError("every v0.3 record must contain the full dataset source")
                if not isinstance(row.get("sample_id"), str):
                    raise ValueError("every v0.3 record must contain a string sample_id")
                sample_id = row["sample_id"]
                if sample_id in seen:
                    raise ValueError("v0.3 record sample_id values must be unique")
                seen.add(sample_id)
                sources[sample_id] = {key: row[key] for key in _DATASET_RECORD_KEYS}
                count += 1
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("v0.3 records must be valid UTF-8 JSONL") from error
    if count != expected_records:
        raise ValueError(f"v0.3 evidence must contain exactly {expected_records} records")
    try:
        return _validate_natural_records(
            path,
            split="calibration",
            dataset_records=sources,
        )
    except RuntimeError as error:
        raise ValueError(f"v0.3 records fail strict semantic replay: {error}") from error


def _verify_summary(
    summary: dict[str, object],
    *,
    negative_pilot_path: Path,
    derived: dict[str, object],
) -> None:
    expected = load_negative_pilot_record(negative_pilot_path)
    fixed = {
        "answer_accuracy": expected.answer_accuracy,
        "error_counts": dict(expected.error_counts),
        "gate_passed": expected.gate_passed,
        "model_snapshot_sha256": expected.model_snapshot_sha256,
        "natural_perception_error_rate": expected.natural_perception_error_rate,
        "parse_rate": expected.parse_rate,
        "records": expected.records,
        "schema_version": 1,
        "split": "calibration",
    }
    for key, value in fixed.items():
        if summary.get(key) != value:
            raise ValueError(f"v0.3 summary {key} differs from the frozen failed pilot")
    for key in (
        "records",
        "answer_accuracy",
        "parse_rate",
        "natural_perception_error_rate",
        "error_counts",
    ):
        if summary.get(key) != derived[key]:
            raise ValueError(f"v0.3 summary {key} does not replay from raw records")


def capture_v03_evidence(
    *,
    negative_pilot_path: Path,
    records_path: Path,
    summary_path: Path,
    pilot_data_log_path: Path,
    calibration_log_path: Path,
    output_path: Path,
) -> V03EvidenceCaptureReport:
    """Validate and hash the already-completed v0.3 evidence without rerunning it."""

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence capture: {output_path}")
    if output_path.parent.is_symlink() or not output_path.parent.is_dir():
        raise ValueError("evidence output parent must be an existing regular directory")
    negative = load_negative_pilot_record(negative_pilot_path)
    expected = set(negative.required_server_artifacts)
    inputs = {
        "calibration records": (records_path, "calibration_records_v0_3.jsonl"),
        "calibration summary": (summary_path, "calibration_records_v0_3.summary.json"),
        "pilot data log": (pilot_data_log_path, "pilot-data-v0.3.log"),
        "calibration log": (calibration_log_path, "base-calibration-v0.3.log"),
    }
    if {basename for _, basename in inputs.values()} != expected:
        raise ValueError("negative-pilot artifact registry differs from the capture contract")
    paths = tuple(
        _regular_file(path, label=label, expected_basename=basename)
        for label, (path, basename) in inputs.items()
    )
    derived = _verify_records(records_path, expected_records=negative.records)
    summary = _load_summary(summary_path)
    _verify_summary(
        summary,
        negative_pilot_path=negative_pilot_path,
        derived=derived,
    )
    try:
        calibration_log = calibration_log_path.read_text(encoding="utf-8")
        pilot_log = pilot_data_log_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("v0.3 logs must be UTF-8 text") from error
    if "calibration_exit=3" not in calibration_log.splitlines():
        raise ValueError("calibration log does not preserve calibration_exit=3")
    if not pilot_log.strip():
        raise ValueError("pilot data log must not be empty")
    sources = tuple(
        sorted(
            (
                CapturedSourceFile(
                    basename=path.name,
                    bytes=path.stat().st_size,
                    sha256=_sha256(path),
                )
                for path in paths
            ),
            key=lambda item: item.basename,
        )
    )
    report = V03EvidenceCaptureReport(
        verified=True,
        records=negative.records,
        gate_passed=False,
        calibration_exit=3,
        source_files=sources,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "recoverability_v1_v0_3_external_evidence",
        "status": "FROZEN_FAILED_NOT_TO_BE_RERUN",
        "negative_pilot_sha256": _sha256(negative_pilot_path),
        "server_revision_observed": negative.server_revision_observed,
        "model_snapshot_sha256": negative.model_snapshot_sha256,
        "dataset_manifest_sha256": negative.dataset_manifest_sha256,
        "dataset_records_sha256": negative.dataset_records_sha256,
        "dataset_images_sha256": negative.dataset_images_sha256,
        "counterfactual_sha256": negative.counterfactual_sha256,
        "records": report.records,
        "answer_accuracy": negative.answer_accuracy,
        "parse_rate": negative.parse_rate,
        "natural_perception_error_rate": negative.natural_perception_error_rate,
        "error_counts": dict(negative.error_counts),
        "gate_passed": report.gate_passed,
        "calibration_exit": report.calibration_exit,
        "original_pilot_a": negative.original_pilot_a,
        "original_pilot_b": negative.original_pilot_b,
        "source_files": [asdict(item) for item in sources],
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report
