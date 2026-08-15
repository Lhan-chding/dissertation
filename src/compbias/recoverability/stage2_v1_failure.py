"""Externally anchored replay and diagnosis for the failed Stage-2 v1 probe."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .dsl.executor import TrustedBinding, evaluate_program
from .evidence import ProtocolLockResult, verify_protocol_lock
from .stage2_v1 import (
    STAGE2_V1_SERVER_PACKAGE_PATHS,
    Stage2V1Scene,
    run_stage2_v1_probe,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_LABELS = frozenset({"console", "preflight", "probe_records", "probe_report"})
_ERROR_CODES = frozenset({"program_answer_mismatch", "program_parse_failure"})
_RECORD_FIELDS = frozenset(
    {
        "scene_id",
        "operation",
        "raw_text",
        "program_parse_success",
        "program_execution_success",
        "program_answer_match",
        "operation_result_correct",
        "error_code",
    }
)

STAGE2_V1_DIAGNOSTIC_PACKAGE_LOCK_PATH = (
    "configs/recoverability/server_package_lock_stage2_v1_diagnostic.yaml"
)
STAGE2_V1_DIAGNOSTIC_PACKAGE_PATHS = STAGE2_V1_SERVER_PACKAGE_PATHS | frozenset(
    {
        "configs/recoverability/stage2_v1_failure.yaml",
        "experiments/recoverability_v1/06_diagnose_stage2_v1_failure.py",
        "src/compbias/recoverability/stage2_v1_failure.py",
    }
)


@dataclass(frozen=True, slots=True)
class Stage2V1FailureRecord:
    schema_version: int
    status: str
    dataset_id: str
    source_dataset_id: str
    source_split: str
    model_snapshot_sha256: str
    source_stage1_records_sha256: str
    server_package_lock_sha256: str
    probe_exit: int
    scenes: int
    model_calls: int
    program_parse_rate: float
    execution_rate: float
    program_answer_consistency: float
    operation_result_accuracy: float
    probe_passed: bool
    hypothesis_tested: bool
    confirmatory_execution_authorized: bool
    training_invoked: bool
    error_counts: tuple[tuple[str, int], ...]
    source_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Stage2V1OperationDiagnostic:
    operation: str
    total: int
    parse_failures: int
    answer_mismatches: int
    passed: int


@dataclass(frozen=True, slots=True)
class Stage2V1FailureExample:
    signature: str
    scene_id: str
    operation: str
    error_code: str
    raw_signature: str
    raw_text: str
    executed_result: int | None
    final_answer: int | None


@dataclass(frozen=True, slots=True)
class Stage2V1FailureDiagnostic:
    verified: bool
    records: int
    replayed_program_parse_successes: int
    replayed_program_execution_successes: int
    replayed_program_answer_matches: int
    replayed_operation_result_correct: int
    by_operation: tuple[Stage2V1OperationDiagnostic, ...]
    failure_signatures: tuple[tuple[str, int], ...]
    representative_examples: tuple[Stage2V1FailureExample, ...]
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


def verify_stage2_v1_diagnostic_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    """Verify the canonical, model-free diagnostic code closure."""

    root = repository_root.resolve()
    canonical = root / STAGE2_V1_DIAGNOSTIC_PACKAGE_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("Stage-2 v1 diagnostic package lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != STAGE2_V1_DIAGNOSTIC_PACKAGE_PATHS:
        missing = sorted(STAGE2_V1_DIAGNOSTIC_PACKAGE_PATHS - observed)
        extra = sorted(observed - STAGE2_V1_DIAGNOSTIC_PACKAGE_PATHS)
        raise ValueError(
            f"Stage-2 v1 diagnostic package closure mismatch; missing={missing}, extra={extra}"
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


def _count_mapping(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, dict) or set(value) != _ERROR_CODES:
        raise ValueError("Stage-2 v1 error counts are incomplete")
    return tuple(
        sorted((key, _exact_int(count, f"error_counts.{key}")) for key, count in value.items())
    )


def _source_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or set(value) != _SOURCE_LABELS:
        raise ValueError("Stage-2 v1 source hash registry is incomplete")
    return tuple(sorted((key, _digest(digest, key)) for key, digest in value.items()))


def load_stage2_v1_failure(path: Path) -> Stage2V1FailureRecord:
    """Load the exact externally supplied Stage-2 failure evidence."""

    mapping = load_yaml_mapping(path, label="Stage-2 v1 failure record")
    fields = {field.name for field in Stage2V1FailureRecord.__dataclass_fields__.values()}
    reject_unknown_fields(mapping, fields, label="Stage-2 v1 failure record")
    if set(mapping) != fields:
        raise ValueError("Stage-2 v1 failure record is incomplete")
    boolean_fields = (
        "probe_passed",
        "hypothesis_tested",
        "confirmatory_execution_authorized",
        "training_invoked",
    )
    if any(type(mapping[field]) is not bool for field in boolean_fields):
        raise TypeError("Stage-2 v1 failure flags must be exact booleans")
    record = Stage2V1FailureRecord(
        schema_version=_exact_int(mapping["schema_version"], "schema_version", minimum=1),
        status=str(mapping["status"]),
        dataset_id=str(mapping["dataset_id"]),
        source_dataset_id=str(mapping["source_dataset_id"]),
        source_split=str(mapping["source_split"]),
        model_snapshot_sha256=_digest(mapping["model_snapshot_sha256"], "model snapshot"),
        source_stage1_records_sha256=_digest(
            mapping["source_stage1_records_sha256"], "Stage-1 records"
        ),
        server_package_lock_sha256=_digest(
            mapping["server_package_lock_sha256"], "server package lock"
        ),
        probe_exit=_exact_int(mapping["probe_exit"], "probe_exit"),
        scenes=_exact_int(mapping["scenes"], "scenes", minimum=1),
        model_calls=_exact_int(mapping["model_calls"], "model_calls", minimum=1),
        program_parse_rate=_rate(mapping["program_parse_rate"], "program_parse_rate"),
        execution_rate=_rate(mapping["execution_rate"], "execution_rate"),
        program_answer_consistency=_rate(
            mapping["program_answer_consistency"], "program_answer_consistency"
        ),
        operation_result_accuracy=_rate(
            mapping["operation_result_accuracy"], "operation_result_accuracy"
        ),
        probe_passed=mapping["probe_passed"],
        hypothesis_tested=mapping["hypothesis_tested"],
        confirmatory_execution_authorized=mapping["confirmatory_execution_authorized"],
        training_invoked=mapping["training_invoked"],
        error_counts=_count_mapping(mapping["error_counts"]),
        source_sha256=_source_mapping(mapping["source_sha256"]),
    )
    expected = {
        "schema_version": 1,
        "status": "FINAL_FAILED_DEVELOPMENT_PROBE_DO_NOT_RERUN",
        "dataset_id": "CVA-Recoverability-Stage2-V1-Dev-Probe",
        "source_dataset_id": "CVA-Chart-Pilot-v0.3",
        "source_split": "dev",
        "model_snapshot_sha256": (
            "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
        ),
        "source_stage1_records_sha256": (
            "0b6f0518c84c83bb4b4d78c6d08db526e9449a4b022fcfc10ba607e79cfae7fa"
        ),
        "server_package_lock_sha256": (
            "25e6a16f287502bdc12dde822f852ac13253efc2bfc30f0b19537ba4e87b5fb5"
        ),
        "probe_exit": 3,
        "scenes": 24,
        "model_calls": 24,
        "program_parse_rate": 19 / 24,
        "execution_rate": 19 / 24,
        "program_answer_consistency": 13 / 24,
        "operation_result_accuracy": 13 / 24,
        "probe_passed": False,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
        "training_invoked": False,
        "error_counts": (("program_answer_mismatch", 6), ("program_parse_failure", 5)),
        "source_sha256": (
            (
                "console",
                "9de0fbc1e610886c99d5271f08306cbf091e0c9de02eaab6762fbe52f41d1502",
            ),
            (
                "preflight",
                "0828fd63071dcb00ac97269800e565262568232064834922538c72afa6606d7e",
            ),
            (
                "probe_records",
                "b8ce766b9ba0555fe780e88195a0e4d4d3294cc8813357e7a33a5cfeea19793c",
            ),
            (
                "probe_report",
                "aca0671f5c4b5e4334544955733b7143931618f9e9a33c7cdd5a8bd09cc55548",
            ),
        ),
    }
    if asdict(record) != expected:
        raise ValueError("Stage-2 v1 failure record differs from the external evidence")
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


def _raw_signature(raw: str) -> str:
    if raw.lstrip().startswith("```"):
        return "markdown_fence"
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        return "non_json_or_trailing_text"
    if isinstance(payload, dict):
        return "json_object_keys:" + ",".join(sorted(str(key) for key in payload))
    return f"json_{type(payload).__name__}"


def _read_records(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Stage-2 v1 record line must be JSON") from error
            if not isinstance(row, dict) or set(row) != _RECORD_FIELDS:
                raise ValueError("Stage-2 v1 record schema is invalid")
            rows.append(row)
    return tuple(rows)


def _validate_report(
    report: dict[str, object],
    expected: Stage2V1FailureRecord,
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
        "hypothesis_tested",
        "confirmatory_execution_authorized",
    }
    if set(report) != expected_fields:
        raise ValueError("Stage-2 v1 report schema differs from the frozen probe")
    registered = {
        **replay,
        "schema_version": 1,
        "artifact_type": "recoverability_stage2_v1_development_probe",
        "dataset_id": expected.dataset_id,
        "source_dataset_id": expected.source_dataset_id,
        "source_split": expected.source_split,
        "model_snapshot_sha256": expected.model_snapshot_sha256,
        "source_stage1_records_sha256": expected.source_stage1_records_sha256,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
    }
    if report != registered:
        raise ValueError("Stage-2 v1 report does not replay from raw records")


def verify_stage2_v1_failure_artifacts(
    expected: Stage2V1FailureRecord,
    *,
    preflight_path: Path,
    console_path: Path,
    report_path: Path,
    records_path: Path,
    scenes: tuple[Stage2V1Scene, ...],
) -> Stage2V1FailureDiagnostic:
    """Hash-bind and replay all outputs without invoking a model."""

    if not isinstance(expected, Stage2V1FailureRecord):
        raise TypeError("expected must be a Stage2V1FailureRecord")
    paths = {
        "preflight": preflight_path,
        "console": console_path,
        "probe_report": report_path,
        "probe_records": records_path,
    }
    hashes = dict(expected.source_sha256)
    for label, path in paths.items():
        if path.is_symlink() or not path.is_file() or _sha256(path) != hashes[label]:
            raise ValueError(f"Stage-2 v1 {label} SHA-256 mismatch")
    preflight = _load_json_object(preflight_path, label="Stage-2 v1 preflight")
    if (
        preflight.get("artifact_type") != "recoverability_stage2_v1_metadata_preflight"
        or preflight.get("ready") is not True
        or preflight.get("large_gpu_started") is not False
        or preflight.get("model_loaded") is not False
        or preflight.get("training_authorized") is not False
        or preflight.get("server_package_lock_verified") is not True
        or preflight.get("server_package_lock_sha256") != expected.server_package_lock_sha256
    ):
        raise ValueError("Stage-2 v1 preflight differs from frozen evidence")
    console = console_path.read_text(encoding="utf-8")
    if "stage2_probe_exit=3" not in console:
        raise ValueError("Stage-2 v1 console does not preserve stage2_probe_exit=3")
    rows = _read_records(records_path)
    if len(rows) != expected.scenes or len(scenes) != expected.scenes:
        raise ValueError("Stage-2 v1 scene count differs from frozen evidence")
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    if len(scene_by_id) != len(scenes):
        raise ValueError("Stage-2 v1 scene identifiers must be unique")
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
            raise ValueError("Stage-2 v1 record identifier or raw output is invalid")
        stored_by_id[scene_id] = row
        raw_by_id[scene_id] = raw
    replay_report, replay_records = run_stage2_v1_probe(
        scenes,
        generate=lambda scene, _messages: raw_by_id[scene.scene_id],
    )
    if any(stored_by_id[item.scene_id] != asdict(item) for item in replay_records):
        raise ValueError("Stage-2 v1 stored record does not replay")
    report = _load_json_object(report_path, label="Stage-2 v1 report")
    _validate_report(report, expected, replay_report)
    if (
        replay_report.scenes != expected.scenes
        or replay_report.model_calls != expected.model_calls
        or replay_report.program_parse_rate != expected.program_parse_rate
        or replay_report.execution_rate != expected.execution_rate
        or replay_report.program_answer_consistency != expected.program_answer_consistency
        or replay_report.operation_result_accuracy != expected.operation_result_accuracy
        or replay_report.probe_passed is not expected.probe_passed
        or replay_report.error_counts != expected.error_counts
    ):
        raise ValueError("Stage-2 v1 aggregate replay differs from frozen evidence")
    operation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    signature_counts: Counter[str] = Counter()
    examples: dict[str, list[Stage2V1FailureExample]] = defaultdict(list)
    for record in replay_records:
        counter = operation_counts[record.operation]
        counter["total"] += 1
        if record.error_code is None:
            counter["passed"] += 1
            continue
        if record.error_code == "program_parse_failure":
            counter["parse_failures"] += 1
        elif record.error_code == "program_answer_mismatch":
            counter["answer_mismatches"] += 1
        signature = f"{record.operation}|{record.error_code}|{_raw_signature(record.raw_text)}"
        signature_counts[signature] += 1
        scene = scene_by_id[record.scene_id]
        trusted = {
            name: TrustedBinding(f"stage1_target_{name}", value)
            for name, value in zip(("a", "b", "c", "d"), scene.evidence, strict=True)
        }
        evaluation = evaluate_program(record.raw_text, constraint_bindings=trusted)
        if len(examples[signature]) < 3:
            examples[signature].append(
                Stage2V1FailureExample(
                    signature=signature,
                    scene_id=record.scene_id,
                    operation=record.operation,
                    error_code=record.error_code,
                    raw_signature=_raw_signature(record.raw_text),
                    raw_text=record.raw_text,
                    executed_result=evaluation.executed_result,
                    final_answer=evaluation.final_answer,
                )
            )
    by_operation = tuple(
        Stage2V1OperationDiagnostic(
            operation=operation,
            total=counts["total"],
            parse_failures=counts["parse_failures"],
            answer_mismatches=counts["answer_mismatches"],
            passed=counts["passed"],
        )
        for operation, counts in sorted(operation_counts.items())
    )
    return Stage2V1FailureDiagnostic(
        verified=True,
        records=expected.scenes,
        replayed_program_parse_successes=sum(item.program_parse_success for item in replay_records),
        replayed_program_execution_successes=sum(
            item.program_execution_success for item in replay_records
        ),
        replayed_program_answer_matches=sum(item.program_answer_match for item in replay_records),
        replayed_operation_result_correct=sum(
            item.operation_result_correct for item in replay_records
        ),
        by_operation=by_operation,
        failure_signatures=tuple(sorted(signature_counts.items())),
        representative_examples=tuple(
            example for signature in sorted(examples) for example in examples[signature]
        ),
        model_calls=0,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
        source_sha256=expected.source_sha256,
    )
