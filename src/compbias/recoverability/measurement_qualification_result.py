"""Immutable server-evidence anchor for the passed measurement qualification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from compbias.io.strict_json import load_strict_json_mapping
from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class MeasurementQualificationFrozenResult:
    schema_version: int
    status: str
    qualification_exit: int
    qualification_passed: bool
    dataset_id: str
    model_snapshot_sha256: str
    server_package_lock_sha256: str
    scenes: int
    model_calls: int
    stage1_parse_successes: int
    stage1_parse_rate: float
    stage1_parse_lower: float
    exact_transcription_rate: float
    stage2_program_parse_successes: int
    stage2_program_parse_rate: float
    stage2_program_parse_lower: float
    stage2_execution_successes: int
    stage2_execution_rate: float
    stage2_execution_lower: float
    executor_answer_successes: int
    executor_answer_accuracy: float
    executor_answer_lower: float
    gate_failures: tuple[str, ...]
    format_retries: int
    hypothesis_tested: bool
    confirmatory_execution_authorized: bool
    training_invoked: bool
    source_sha256: tuple[tuple[str, str], ...]


def _expected() -> MeasurementQualificationFrozenResult:
    return MeasurementQualificationFrozenResult(
        schema_version=1,
        status="FINAL_PASSED_MEASUREMENT_QUALIFICATION_DO_NOT_RERUN",
        qualification_exit=0,
        qualification_passed=True,
        dataset_id="CVA-Recoverability-Measurement-Qualification-v1",
        model_snapshot_sha256=("e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"),
        server_package_lock_sha256=(
            "a4179f3e4c6f90f6730ad15d3f38a4309564b1099459cfd1d3918cc7f36de691"
        ),
        scenes=300,
        model_calls=599,
        stage1_parse_successes=299,
        stage1_parse_rate=0.9966666666666667,
        stage1_parse_lower=0.9842854451084161,
        exact_transcription_rate=0.8533333333333334,
        stage2_program_parse_successes=299,
        stage2_program_parse_rate=1.0,
        stage2_program_parse_lower=0.9900308532071007,
        stage2_execution_successes=299,
        stage2_execution_rate=1.0,
        stage2_execution_lower=0.9900308532071007,
        executor_answer_successes=299,
        executor_answer_accuracy=1.0,
        executor_answer_lower=0.9900308532071007,
        gate_failures=(),
        format_retries=0,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
        source_sha256=(
            ("attempt_marker", "d8957deaa71283db638c7b644c51b69f0182843dbe447d3ba04750d5cb45e190"),
            ("console", "2513a50896e863cd2d294f5b5cb9fde06c0934dda4290cd70551dbeb131d4311"),
            ("preflight", "784149ae5482f4d1b7b31a2e22ea71cb8d0c04fc7441a5f6f07004a8e2328ffe"),
            (
                "qualification_records",
                "55d5eea889ed1733bc58d75f5783ae0a34e00e827361ac20ccfcee4351f18e4d",
            ),
            (
                "qualification_report",
                "11a2d7f44d7fdf115954baacbac1b6c53269a8e00368c2b78f27ba53c24c640e",
            ),
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_measurement_qualification_frozen_result(
    path: Path,
) -> MeasurementQualificationFrozenResult:
    """Load the exact result supplied by the completed server run."""

    mapping = load_yaml_mapping(path, label="measurement qualification frozen result")
    fields = set(MeasurementQualificationFrozenResult.__dataclass_fields__)
    reject_unknown_fields(mapping, fields, label="measurement qualification frozen result")
    if set(mapping) != fields:
        raise ValueError("measurement qualification frozen result is incomplete")
    source = mapping["source_sha256"]
    if not isinstance(source, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for key, value in source.items()
    ):
        raise ValueError("measurement qualification source digests are invalid")
    candidate = MeasurementQualificationFrozenResult(
        **{
            **mapping,
            "gate_failures": tuple(mapping["gate_failures"]),
            "source_sha256": tuple(sorted(source.items())),
        }
    )
    if candidate != _expected():
        raise ValueError("measurement qualification result differs from server evidence")
    return candidate


def _regular_file(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} is missing or differs from frozen evidence")


def verify_measurement_qualification_result_artifacts(
    frozen: MeasurementQualificationFrozenResult,
    *,
    preflight: Path,
    attempt_marker: Path,
    report: Path,
    records: Path,
    console_log: Path,
) -> MeasurementQualificationFrozenResult:
    """Bind the five immutable artifacts before any later model load."""

    if frozen != _expected():
        raise ValueError("measurement qualification frozen result is not canonical")
    paths = {
        "preflight": preflight,
        "attempt_marker": attempt_marker,
        "qualification_report": report,
        "qualification_records": records,
        "console": console_log,
    }
    digests = dict(frozen.source_sha256)
    for label, path in paths.items():
        _regular_file(path, digests[label], label)
    payload = load_strict_json_mapping(
        report,
        label="measurement qualification report",
        max_bytes=128 * 1024,
    )
    expected_report = asdict(frozen)
    for key in (
        "status",
        "qualification_exit",
        "source_sha256",
    ):
        expected_report.pop(key)
    for key, value in expected_report.items():
        if key == "gate_failures":
            value = list(value)
        if payload.get(key) != value:
            raise ValueError(f"measurement qualification report mismatch for {key}")
    marker = load_strict_json_mapping(
        attempt_marker,
        label="measurement qualification attempt marker",
        max_bytes=32 * 1024,
    )
    if marker.get("status") != "MEASUREMENT_QUALIFICATION_STARTED":
        raise ValueError("measurement qualification attempt marker status is invalid")
    if marker.get("model_snapshot_sha256") != frozen.model_snapshot_sha256:
        raise ValueError("measurement qualification model hash is invalid")
    try:
        console = console_log.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("measurement qualification console must be UTF-8") from error
    if console.count("measurement_qualification_exit=0") < 1:
        raise ValueError("measurement qualification console lacks successful exit evidence")
    line_count = 0
    with records.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("qualification record must be an object")
            line_count += 1
    if line_count != frozen.scenes:
        raise ValueError("measurement qualification record count is invalid")
    return frozen
