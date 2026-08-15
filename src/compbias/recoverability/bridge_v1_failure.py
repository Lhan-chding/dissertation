"""Externally anchored evidence contract for the failed one-shot Bridge v1."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .bridge import parse_stage1_evidence

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_LABELS = frozenset(
    {
        "attempt_marker",
        "bridge_console",
        "bridge_records",
        "bridge_report",
        "stage1_diagnostic",
    }
)
_ERROR_CODES = frozenset(
    {
        "not_exact_json_object",
        "schema_invalid",
        "target_facts_not_four_integers",
    }
)


@dataclass(frozen=True, slots=True)
class BridgeV1FailureRecord:
    schema_version: int
    status: str
    dataset_id: str
    model_snapshot_sha256: str
    bridge_exit: int
    scenes: int
    model_calls: int
    legacy_parse_rate: float
    legacy_answer_accuracy: float
    legacy_perception_error_rate: float
    stage1_parse_rate: float
    stage2_invocations: int
    hypothesis_tested: bool
    training_invoked: bool
    parse_error_counts: tuple[tuple[str, int], ...]
    source_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class BridgeV1Replay:
    records: int
    stage1_parse_successes: int
    stage2_invocations: int
    parse_error_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class BridgeV1FailureVerification:
    verified: bool
    records: int
    stage1_parse_successes: int
    stage2_invocations: int
    hypothesis_tested: bool
    source_sha256: tuple[tuple[str, str], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _rate(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _count_mapping(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, dict) or set(value) != _ERROR_CODES:
        raise ValueError("bridge v1 parse_error_counts are incomplete")
    return tuple(
        sorted(
            (key, _exact_int(count, f"parse_error_counts.{key}")) for key, count in value.items()
        )
    )


def _source_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or set(value) != _SOURCE_LABELS:
        raise ValueError("bridge v1 source SHA-256 registry is incomplete")
    return tuple(sorted((key, _digest(digest, key)) for key, digest in value.items()))


def load_bridge_v1_failure(path: Path) -> BridgeV1FailureRecord:
    """Load the closed failure record without interpreting it as a hypothesis result."""

    mapping = load_yaml_mapping(path, label="Bridge v1 failure record")
    fields = {
        "schema_version",
        "status",
        "dataset_id",
        "model_snapshot_sha256",
        "bridge_exit",
        "scenes",
        "model_calls",
        "legacy_parse_rate",
        "legacy_answer_accuracy",
        "legacy_perception_error_rate",
        "stage1_parse_rate",
        "stage2_invocations",
        "hypothesis_tested",
        "training_invoked",
        "parse_error_counts",
        "source_sha256",
    }
    reject_unknown_fields(mapping, fields, label="Bridge v1 failure record")
    if set(mapping) != fields or mapping["schema_version"] != 1:
        raise ValueError("Bridge v1 failure record schema is invalid")
    record = BridgeV1FailureRecord(
        schema_version=1,
        status=str(mapping["status"]),
        dataset_id=str(mapping["dataset_id"]),
        model_snapshot_sha256=_digest(mapping["model_snapshot_sha256"], "model_snapshot_sha256"),
        bridge_exit=_exact_int(mapping["bridge_exit"], "bridge_exit"),
        scenes=_exact_int(mapping["scenes"], "scenes", minimum=1),
        model_calls=_exact_int(mapping["model_calls"], "model_calls", minimum=1),
        legacy_parse_rate=_rate(mapping["legacy_parse_rate"], "legacy_parse_rate"),
        legacy_answer_accuracy=_rate(mapping["legacy_answer_accuracy"], "legacy_answer_accuracy"),
        legacy_perception_error_rate=_rate(
            mapping["legacy_perception_error_rate"], "legacy_perception_error_rate"
        ),
        stage1_parse_rate=_rate(mapping["stage1_parse_rate"], "stage1_parse_rate"),
        stage2_invocations=_exact_int(mapping["stage2_invocations"], "stage2_invocations"),
        hypothesis_tested=mapping["hypothesis_tested"],
        training_invoked=mapping["training_invoked"],
        parse_error_counts=_count_mapping(mapping["parse_error_counts"]),
        source_sha256=_source_mapping(mapping["source_sha256"]),
    )
    if type(record.hypothesis_tested) is not bool or type(record.training_invoked) is not bool:
        raise TypeError("Bridge v1 hypothesis/training flags must be boolean")
    registered = {
        "status": "FINAL_FAILED_STAGE1_INTERFACE",
        "dataset_id": "CVA-Recoverability-Bridge-v1",
        "bridge_exit": 3,
        "scenes": 300,
        "model_calls": 600,
        "stage1_parse_rate": 0.0,
        "stage2_invocations": 0,
        "hypothesis_tested": False,
        "training_invoked": False,
        "parse_failures": 300,
    }
    observed = {
        "status": record.status,
        "dataset_id": record.dataset_id,
        "bridge_exit": record.bridge_exit,
        "scenes": record.scenes,
        "model_calls": record.model_calls,
        "stage1_parse_rate": record.stage1_parse_rate,
        "stage2_invocations": record.stage2_invocations,
        "hypothesis_tested": record.hypothesis_tested,
        "training_invoked": record.training_invoked,
        "parse_failures": sum(dict(record.parse_error_counts).values()),
    }
    if observed != registered:
        raise ValueError("Bridge v1 failure record differs from the registered outcome")
    return record


def _stage1_error_code(error: ValueError) -> str:
    return {
        "Stage-1 output must be one exact JSON object": "not_exact_json_object",
        "Stage-1 evidence schema is invalid": "schema_invalid",
        "target_facts must contain exactly four integers": "target_facts_not_four_integers",
    }.get(str(error), "other_strict_parse_failure")


def replay_bridge_v1_records(path: Path) -> BridgeV1Replay:
    """Recompute the strict Stage-1 failure taxonomy from raw bridge records."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("Bridge v1 records must be a regular file")
    identifiers: set[str] = set()
    counts: Counter[str] = Counter()
    records = 0
    parse_successes = 0
    stage2_invocations = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Bridge v1 record must be a JSON object")
            scene_id = row.get("scene_id")
            if not isinstance(scene_id, str) or not scene_id or scene_id in identifiers:
                raise ValueError("Bridge v1 scene identifiers must be unique strings")
            identifiers.add(scene_id)
            raw = row.get("stage1_raw")
            if not isinstance(raw, str):
                raise ValueError("Bridge v1 stage1_raw must be text")
            try:
                parse_stage1_evidence(raw)
                parse_success = True
                parse_successes += 1
            except ValueError as error:
                parse_success = False
                counts[_stage1_error_code(error)] += 1
            if row.get("stage1_parse_success") is not parse_success:
                raise ValueError("Bridge v1 stored Stage-1 parse status does not replay")
            if row.get("stage2_raw") is not None:
                stage2_invocations += 1
            records += 1
    if not records:
        raise ValueError("Bridge v1 records must be non-empty")
    if stage2_invocations:
        raise ValueError("Stage 2 was invoked despite the frozen Bridge v1 failure")
    return BridgeV1Replay(
        records=records,
        stage1_parse_successes=parse_successes,
        stage2_invocations=stage2_invocations,
        parse_error_counts=tuple(sorted(counts.items())),
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def verify_bridge_v1_failure_artifacts(
    expected: BridgeV1FailureRecord,
    *,
    records_path: Path,
    report_path: Path,
    diagnostic_path: Path,
    attempt_marker_path: Path,
    console_log_path: Path,
) -> BridgeV1FailureVerification:
    """Hash-bind and replay every artifact required before a v2 development probe."""

    if not isinstance(expected, BridgeV1FailureRecord):
        raise TypeError("expected must be a BridgeV1FailureRecord")
    paths = {
        "attempt_marker": attempt_marker_path,
        "bridge_console": console_log_path,
        "bridge_records": records_path,
        "bridge_report": report_path,
        "stage1_diagnostic": diagnostic_path,
    }
    expected_hashes = dict(expected.source_sha256)
    for label, path in paths.items():
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected_hashes[label]:
            raise ValueError(f"Bridge v1 {label} SHA-256 mismatch")
    replay = replay_bridge_v1_records(records_path)
    if (
        replay.records != expected.scenes
        or replay.stage1_parse_successes != 0
        or replay.stage2_invocations != expected.stage2_invocations
        or replay.parse_error_counts != expected.parse_error_counts
    ):
        raise ValueError("Bridge v1 raw-record replay differs from the frozen failure")
    report = _load_json_object(report_path, label="Bridge v1 report")
    report_expectations = {
        "artifact_type": "recoverability_v1_bridge_report",
        "dataset_id": expected.dataset_id,
        "model_snapshot_sha256": expected.model_snapshot_sha256,
        "scenes": expected.scenes,
        "model_calls": expected.model_calls,
        "legacy_parse_rate": expected.legacy_parse_rate,
        "legacy_answer_accuracy": expected.legacy_answer_accuracy,
        "legacy_perception_error_rate": expected.legacy_perception_error_rate,
        "stage1_parse_rate": expected.stage1_parse_rate,
        "stage2_program_parse_rate": 0.0,
        "program_answer_consistency": 0.0,
        "two_stage_answer_accuracy": 0.0,
        "protocols_mergeable": False,
        "training_invoked": expected.training_invoked,
    }
    if any(report.get(key) != value for key, value in report_expectations.items()):
        raise ValueError("Bridge v1 report differs from the frozen failure")
    diagnostic = _load_json_object(diagnostic_path, label="Bridge v1 diagnostic")
    if (
        diagnostic.get("artifact_type") != "recoverability_v1_stage1_diagnostic"
        or diagnostic.get("records") != expected.scenes
        or diagnostic.get("source_records_sha256") != expected_hashes["bridge_records"]
        or diagnostic.get("replayed_stage1_parse_successes") != 0
        or diagnostic.get("stage2_invocations") != expected.stage2_invocations
    ):
        raise ValueError("Bridge v1 diagnostic differs from the frozen failure")
    attempt = _load_json_object(attempt_marker_path, label="Bridge v1 attempt marker")
    if attempt != {"schema_version": 1, "status": "BRIDGE_ATTEMPT_STARTED"}:
        raise ValueError("Bridge v1 attempt marker is invalid")
    if "bridge_exit=3" not in console_log_path.read_text(encoding="utf-8"):
        raise ValueError("Bridge v1 console log does not preserve bridge_exit=3")
    return BridgeV1FailureVerification(
        verified=True,
        records=replay.records,
        stage1_parse_successes=replay.stage1_parse_successes,
        stage2_invocations=replay.stage2_invocations,
        hypothesis_tested=False,
        source_sha256=expected.source_sha256,
    )
