"""Externally anchored replay for the successful Stage-2 v2 probe."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .evidence import ProtocolLockResult, verify_protocol_lock
from .stage2_v2 import (
    STAGE2_V2_SERVER_PACKAGE_PATHS,
    Stage2V2Scene,
    run_stage2_v2_probe,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_LABELS = frozenset(
    {"attempt_marker", "console", "preflight", "probe_records", "probe_report"}
)
_RECORD_FIELDS = frozenset(
    {
        "scene_id",
        "operation",
        "raw_text",
        "program_parse_success",
        "program_execution_success",
        "executed_result",
        "final_answer",
        "executor_answer_correct",
        "error_code",
    }
)

STAGE2_V2_EVIDENCE_PACKAGE_LOCK_PATH = (
    "configs/recoverability/server_package_lock_stage2_v2_evidence.yaml"
)
STAGE2_V2_EVIDENCE_PACKAGE_PATHS = STAGE2_V2_SERVER_PACKAGE_PATHS | frozenset(
    {
        "configs/recoverability/server_package_lock_stage2_v2.yaml",
        "configs/recoverability/stage2_v2_frozen_result.yaml",
        "experiments/recoverability_v1/08_capture_stage2_v2_evidence.py",
        "src/compbias/recoverability/stage2_v2_evidence.py",
    }
)


@dataclass(frozen=True, slots=True)
class Stage2V2FrozenResult:
    schema_version: int
    status: str
    dataset_id: str
    source_dataset_id: str
    source_split: str
    model_snapshot_sha256: str
    source_stage1_records_sha256: str
    source_stage2_v1_diagnostic_sha256: str
    server_package_lock_sha256: str
    probe_exit: int
    scenes: int
    model_calls: int
    program_parse_rate: float
    execution_rate: float
    executor_answer_accuracy: float
    probe_passed: bool
    hypothesis_tested: bool
    confirmatory_execution_authorized: bool
    training_invoked: bool
    error_counts: tuple[tuple[str, int], ...]
    source_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Stage2V2EvidenceVerification:
    verified: bool
    records: int
    replayed_program_parse_successes: int
    replayed_program_execution_successes: int
    replayed_executor_answer_correct: int
    model_calls: int
    hypothesis_tested: bool
    confirmatory_execution_authorized: bool
    training_invoked: bool
    source_sha256: tuple[tuple[str, str], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage2_v2_evidence_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    """Verify the exact model-free evidence-capture code closure."""

    root = repository_root.resolve()
    canonical = root / STAGE2_V2_EVIDENCE_PACKAGE_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("Stage-2 v2 evidence package lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != STAGE2_V2_EVIDENCE_PACKAGE_PATHS:
        missing = sorted(STAGE2_V2_EVIDENCE_PACKAGE_PATHS - observed)
        extra = sorted(observed - STAGE2_V2_EVIDENCE_PACKAGE_PATHS)
        raise ValueError(
            f"Stage-2 v2 evidence package closure mismatch; missing={missing}, extra={extra}"
        )
    return result


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _rate(value: object, label: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{label} must be an exact floating-point value")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must lie in [0, 1]")
    return value


def _source_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or set(value) != _SOURCE_LABELS:
        raise ValueError("Stage-2 v2 source hash registry is incomplete")
    return tuple(sorted((key, _digest(digest, key)) for key, digest in value.items()))


def load_stage2_v2_frozen_result(path: Path) -> Stage2V2FrozenResult:
    """Load the exact externally supplied successful v2 evidence."""

    mapping = load_yaml_mapping(path, label="Stage-2 v2 frozen result")
    fields = {field.name for field in Stage2V2FrozenResult.__dataclass_fields__.values()}
    reject_unknown_fields(mapping, fields, label="Stage-2 v2 frozen result")
    if set(mapping) != fields:
        raise ValueError("Stage-2 v2 frozen result is incomplete")
    boolean_fields = (
        "probe_passed",
        "hypothesis_tested",
        "confirmatory_execution_authorized",
        "training_invoked",
    )
    if any(type(mapping[field]) is not bool for field in boolean_fields):
        raise TypeError("Stage-2 v2 flags must be exact booleans")
    raw_error_counts = mapping["error_counts"]
    if not isinstance(raw_error_counts, dict) or raw_error_counts:
        raise ValueError("Stage-2 v2 successful result must have no error counts")
    record = Stage2V2FrozenResult(
        schema_version=_exact_int(mapping["schema_version"], "schema_version", minimum=1),
        status=str(mapping["status"]),
        dataset_id=str(mapping["dataset_id"]),
        source_dataset_id=str(mapping["source_dataset_id"]),
        source_split=str(mapping["source_split"]),
        model_snapshot_sha256=_digest(mapping["model_snapshot_sha256"], "model snapshot"),
        source_stage1_records_sha256=_digest(
            mapping["source_stage1_records_sha256"], "Stage-1 records"
        ),
        source_stage2_v1_diagnostic_sha256=_digest(
            mapping["source_stage2_v1_diagnostic_sha256"], "Stage-2 v1 diagnostic"
        ),
        server_package_lock_sha256=_digest(
            mapping["server_package_lock_sha256"], "server package lock"
        ),
        probe_exit=_exact_int(mapping["probe_exit"], "probe_exit"),
        scenes=_exact_int(mapping["scenes"], "scenes", minimum=1),
        model_calls=_exact_int(mapping["model_calls"], "model_calls", minimum=1),
        program_parse_rate=_rate(mapping["program_parse_rate"], "program_parse_rate"),
        execution_rate=_rate(mapping["execution_rate"], "execution_rate"),
        executor_answer_accuracy=_rate(
            mapping["executor_answer_accuracy"], "executor_answer_accuracy"
        ),
        probe_passed=mapping["probe_passed"],
        hypothesis_tested=mapping["hypothesis_tested"],
        confirmatory_execution_authorized=mapping["confirmatory_execution_authorized"],
        training_invoked=mapping["training_invoked"],
        error_counts=(),
        source_sha256=_source_mapping(mapping["source_sha256"]),
    )
    expected = {
        "schema_version": 1,
        "status": "FINAL_PASSED_DEVELOPMENT_PROBE_DO_NOT_RERUN",
        "dataset_id": "CVA-Recoverability-Stage2-V2-Dev-Probe",
        "source_dataset_id": "CVA-Chart-Pilot-v0.3",
        "source_split": "dev",
        "model_snapshot_sha256": (
            "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
        ),
        "source_stage1_records_sha256": (
            "0b6f0518c84c83bb4b4d78c6d08db526e9449a4b022fcfc10ba607e79cfae7fa"
        ),
        "source_stage2_v1_diagnostic_sha256": (
            "d85510ea829a000bc31002f874e5a0ec795421aadec9f9042438d78337d9e7b4"
        ),
        "server_package_lock_sha256": (
            "5095b013a990c4a1d8300ba2cc8890893c3b8fe8dfad715dceeb6ade02891442"
        ),
        "probe_exit": 0,
        "scenes": 24,
        "model_calls": 24,
        "program_parse_rate": 1.0,
        "execution_rate": 1.0,
        "executor_answer_accuracy": 1.0,
        "probe_passed": True,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
        "training_invoked": False,
        "error_counts": (),
        "source_sha256": (
            (
                "attempt_marker",
                "5cf97d5cb67c5e9830ac7455b0759b47b6d1fe1e76ee5ce42f365f4059e51c7c",
            ),
            (
                "console",
                "3649a50b21a5482ab0e20bfe07ed63f4d49f72c1f1b783791b852044253eed81",
            ),
            (
                "preflight",
                "c3b8949f03ae7ba2947ad5632bfd68dc822f3aead33adc218e90619a0957fe0c",
            ),
            (
                "probe_records",
                "6b0604a08ebbf4611c62b7fe9f1d9e03954385b1bcfaedf613f0b70b32f1d2f8",
            ),
            (
                "probe_report",
                "d207cff9f6bdcb48142e3f1bb8a3d8676d7aa5abdbcd0c226666dbb58ac587b7",
            ),
        ),
    }
    if asdict(record) != expected:
        raise ValueError("Stage-2 v2 frozen result differs from external evidence")
    return record


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be one UTF-8 JSON object") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be one JSON object")
    return payload


def _read_records(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Stage-2 v2 record line must be JSON") from error
            if not isinstance(row, dict) or set(row) != _RECORD_FIELDS:
                raise ValueError("Stage-2 v2 record schema is invalid")
            rows.append(row)
    return tuple(rows)


def _validate_report(
    report: dict[str, object],
    expected: Stage2V2FrozenResult,
    replay_report: object,
) -> None:
    replay = asdict(replay_report)
    replay["error_counts"] = dict(replay["error_counts"])
    expected_fields = set(replay) | {
        "schema_version",
        "artifact_type",
        "dataset_id",
        "source_dataset_id",
        "source_split",
        "model_snapshot_sha256",
        "source_stage1_records_sha256",
        "source_stage2_v1_diagnostic_sha256",
        "hypothesis_tested",
        "confirmatory_execution_authorized",
    }
    if set(report) != expected_fields:
        raise ValueError("Stage-2 v2 report schema differs from the frozen probe")
    registered = {
        **replay,
        "schema_version": 1,
        "artifact_type": "recoverability_stage2_v2_development_probe",
        "dataset_id": expected.dataset_id,
        "source_dataset_id": expected.source_dataset_id,
        "source_split": expected.source_split,
        "model_snapshot_sha256": expected.model_snapshot_sha256,
        "source_stage1_records_sha256": expected.source_stage1_records_sha256,
        "source_stage2_v1_diagnostic_sha256": (expected.source_stage2_v1_diagnostic_sha256),
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
    }
    if report != registered:
        raise ValueError("Stage-2 v2 report does not replay from raw records")


def verify_stage2_v2_artifacts(
    expected: Stage2V2FrozenResult,
    *,
    preflight_path: Path,
    console_path: Path,
    report_path: Path,
    records_path: Path,
    attempt_marker_path: Path,
    scenes: tuple[Stage2V2Scene, ...],
) -> Stage2V2EvidenceVerification:
    """Hash-bind and replay all v2 outputs without invoking a model."""

    if not isinstance(expected, Stage2V2FrozenResult):
        raise TypeError("expected must be a Stage2V2FrozenResult")
    paths = {
        "preflight": preflight_path,
        "console": console_path,
        "probe_report": report_path,
        "probe_records": records_path,
        "attempt_marker": attempt_marker_path,
    }
    hashes = dict(expected.source_sha256)
    for label, path in paths.items():
        if path.is_symlink() or not path.is_file() or _sha256(path) != hashes[label]:
            raise ValueError(f"Stage-2 v2 {label} SHA-256 mismatch")
    preflight = _load_json_object(preflight_path, label="Stage-2 v2 preflight")
    if (
        preflight.get("artifact_type") != "recoverability_stage2_v2_metadata_preflight"
        or preflight.get("ready") is not True
        or preflight.get("large_gpu_started") is not False
        or preflight.get("model_loaded") is not False
        or preflight.get("training_authorized") is not False
        or preflight.get("server_package_lock_verified") is not True
        or preflight.get("server_package_lock_sha256") != expected.server_package_lock_sha256
    ):
        raise ValueError("Stage-2 v2 preflight differs from frozen evidence")
    if "stage2_v2_probe_exit=0" not in console_path.read_text(encoding="utf-8"):
        raise ValueError("Stage-2 v2 console does not preserve stage2_v2_probe_exit=0")
    attempt = _load_json_object(attempt_marker_path, label="Stage-2 v2 attempt marker")
    if attempt != {
        "schema_version": 1,
        "status": "STAGE2_V2_DEVELOPMENT_PROBE_STARTED",
        "hypothesis_test": False,
    }:
        raise ValueError("Stage-2 v2 attempt marker differs from the one-shot contract")
    rows = _read_records(records_path)
    if len(rows) != expected.scenes or len(scenes) != expected.scenes:
        raise ValueError("Stage-2 v2 scene count differs from frozen evidence")
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    if len(scene_by_id) != len(scenes):
        raise ValueError("Stage-2 v2 scene identifiers must be unique")
    raw_by_id: dict[str, str] = {}
    stored_by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        scene_id = row["scene_id"]
        raw = row["raw_text"]
        if (
            not isinstance(scene_id, str)
            or scene_id not in scene_by_id
            or scene_id in stored_by_id
            or not isinstance(raw, str)
        ):
            raise ValueError("Stage-2 v2 record identifier or raw output is invalid")
        stored_by_id[scene_id] = row
        raw_by_id[scene_id] = raw
    replay_report, replay_records = run_stage2_v2_probe(
        scenes,
        generate=lambda scene, _messages: raw_by_id[scene.scene_id],
    )
    if any(stored_by_id[item.scene_id] != asdict(item) for item in replay_records):
        raise ValueError("Stage-2 v2 stored record does not replay")
    report = _load_json_object(report_path, label="Stage-2 v2 report")
    _validate_report(report, expected, replay_report)
    if (
        replay_report.scenes != expected.scenes
        or replay_report.model_calls != expected.model_calls
        or replay_report.program_parse_rate != expected.program_parse_rate
        or replay_report.execution_rate != expected.execution_rate
        or replay_report.executor_answer_accuracy != expected.executor_answer_accuracy
        or replay_report.probe_passed is not expected.probe_passed
        or replay_report.error_counts != expected.error_counts
    ):
        raise ValueError("Stage-2 v2 aggregate replay differs from frozen evidence")
    return Stage2V2EvidenceVerification(
        verified=True,
        records=expected.scenes,
        replayed_program_parse_successes=sum(item.program_parse_success for item in replay_records),
        replayed_program_execution_successes=sum(
            item.program_execution_success for item in replay_records
        ),
        replayed_executor_answer_correct=sum(
            item.executor_answer_correct for item in replay_records
        ),
        model_calls=0,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
        source_sha256=expected.source_sha256,
    )
