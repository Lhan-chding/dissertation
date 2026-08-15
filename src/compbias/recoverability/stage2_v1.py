"""Development-only Stage-2 DSL probe using frozen Stage-1 v2 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .bridge import parse_stage1_evidence
from .dsl.executor import TrustedBinding, evaluate_program
from .evidence import ProtocolLockResult, verify_protocol_lock
from .stage1_v2 import (
    STAGE1_V2_SERVER_PACKAGE_PATHS,
    Stage1V2Scene,
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OPERATIONS = frozenset({"difference", "max_minus_min", "sum"})
_PROBE_SCENES = 24

STAGE2_V1_SERVER_PACKAGE_LOCK_PATH = "configs/recoverability/server_package_lock_stage2_v1.yaml"
STAGE2_V1_SERVER_PACKAGE_PATHS = STAGE1_V2_SERVER_PACKAGE_PATHS | frozenset(
    {
        "configs/recoverability/stage1_v2_frozen_result.yaml",
        "configs/recoverability/stage2_v1_probe.yaml",
        "experiments/recoverability_v1/00_stage2_v1_preflight.py",
        "experiments/recoverability_v1/05_stage2_v1_probe.py",
        "src/compbias/recoverability/stage2_v1.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage2_v1_server_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    """Verify the canonical closed package for the Stage-2 development probe."""

    root = repository_root.resolve()
    canonical = root / STAGE2_V1_SERVER_PACKAGE_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("Stage-2 v1 server package lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != STAGE2_V1_SERVER_PACKAGE_PATHS:
        missing = sorted(STAGE2_V1_SERVER_PACKAGE_PATHS - observed)
        extra = sorted(observed - STAGE2_V1_SERVER_PACKAGE_PATHS)
        raise ValueError(
            f"Stage-2 v1 server package lock closure mismatch; missing={missing}, extra={extra}"
        )
    return result


@dataclass(frozen=True, slots=True)
class Stage2V1ProbeConfig:
    schema_version: int
    status: str
    dataset_id: str
    output_subdirectory: str
    source_dataset_id: str
    source_split: str
    scenes: int
    format_retries: int
    required_program_parse_rate: float
    required_execution_rate: float
    required_program_answer_consistency: float
    allow_rerun: bool
    hypothesis_test: bool


def load_stage2_v1_probe_config(path: Path) -> Stage2V1ProbeConfig:
    """Load the exact one-shot development contract."""

    mapping = load_yaml_mapping(path, label="Stage-2 v1 probe config")
    expected: dict[str, object] = {
        "schema_version": 1,
        "status": "DEVELOPMENT_PROBE_NOT_RUN",
        "dataset_id": "CVA-Recoverability-Stage2-V1-Dev-Probe",
        "output_subdirectory": "stage2_v1_dev_probe",
        "source_dataset_id": "CVA-Chart-Pilot-v0.3",
        "source_split": "dev",
        "scenes": _PROBE_SCENES,
        "format_retries": 0,
        "required_program_parse_rate": 1.0,
        "required_execution_rate": 1.0,
        "required_program_answer_consistency": 1.0,
        "allow_rerun": False,
        "hypothesis_test": False,
    }
    reject_unknown_fields(mapping, set(expected), label="Stage-2 v1 probe config")
    exact_types = {
        "schema_version": int,
        "status": str,
        "dataset_id": str,
        "output_subdirectory": str,
        "source_dataset_id": str,
        "source_split": str,
        "scenes": int,
        "format_retries": int,
        "required_program_parse_rate": float,
        "required_execution_rate": float,
        "required_program_answer_consistency": float,
        "allow_rerun": bool,
        "hypothesis_test": bool,
    }
    if any(
        type(mapping.get(key)) is not expected_type for key, expected_type in exact_types.items()
    ):
        raise TypeError("Stage-2 v1 probe config field type differs from the registered contract")
    if dict(mapping) != expected:
        raise ValueError("Stage-2 v1 probe config differs from the registered contract")
    return Stage2V1ProbeConfig(**expected)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Stage1V2FrozenResult:
    schema_version: int
    status: str
    dataset_id: str
    source_dataset_id: str
    source_split: str
    model_snapshot_sha256: str
    scenes: int
    model_calls: int
    parse_rate: float
    exact_transcriptions: int
    exact_transcription_rate: float
    mismatch_scene_ids: tuple[str, ...]
    probe_passed: bool
    hypothesis_tested: bool
    confirmatory_execution_authorized: bool
    training_invoked: bool
    source_sha256: tuple[tuple[str, str], ...]


def load_stage1_v2_frozen_result(path: Path) -> Stage1V2FrozenResult:
    """Load the externally anchored successful Stage-1 v2 development result."""

    mapping = load_yaml_mapping(path, label="frozen Stage-1 v2 result")
    fields = {
        "schema_version",
        "status",
        "dataset_id",
        "source_dataset_id",
        "source_split",
        "model_snapshot_sha256",
        "scenes",
        "model_calls",
        "parse_rate",
        "exact_transcriptions",
        "exact_transcription_rate",
        "mismatch_scene_ids",
        "probe_passed",
        "hypothesis_tested",
        "confirmatory_execution_authorized",
        "training_invoked",
        "source_sha256",
    }
    reject_unknown_fields(mapping, fields, label="frozen Stage-1 v2 result")
    if set(mapping) != fields:
        raise ValueError("frozen Stage-1 v2 result is incomplete")
    fixed = {
        "schema_version": 1,
        "status": "FINAL_PASSED_DEVELOPMENT_PROBE",
        "dataset_id": "CVA-Recoverability-Stage1-V2-Dev-Probe",
        "source_dataset_id": "CVA-Chart-Pilot-v0.3",
        "source_split": "dev",
        "model_snapshot_sha256": (
            "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
        ),
        "scenes": 24,
        "model_calls": 24,
        "parse_rate": 1.0,
        "exact_transcriptions": 22,
        "exact_transcription_rate": 22 / 24,
        "mismatch_scene_ids": ["dev-000003", "dev-000019"],
        "probe_passed": True,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
        "training_invoked": False,
    }
    if any(mapping[key] != value for key, value in fixed.items()):
        raise ValueError("frozen Stage-1 v2 result differs from reviewed server evidence")
    raw_hashes = mapping["source_sha256"]
    required_hashes = {"console", "preflight", "probe_records", "probe_report"}
    if not isinstance(raw_hashes, Mapping) or set(raw_hashes) != required_hashes:
        raise ValueError("frozen Stage-1 v2 source hashes are incomplete")
    hashes = tuple(sorted((str(key), str(value)) for key, value in raw_hashes.items()))
    if any(_SHA256.fullmatch(value) is None for _key, value in hashes):
        raise ValueError("frozen Stage-1 v2 source hash is invalid")
    return Stage1V2FrozenResult(
        **{key: value for key, value in fixed.items() if key != "mismatch_scene_ids"},
        mismatch_scene_ids=tuple(fixed["mismatch_scene_ids"]),  # type: ignore[arg-type]
        source_sha256=hashes,
    )


@dataclass(frozen=True, slots=True)
class Stage2V1Scene:
    scene_id: str
    operation: str
    evidence: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or _IDENTIFIER.fullmatch(self.scene_id) is None:
            raise ValueError("scene_id must be a bounded safe identifier")
        if self.operation not in _OPERATIONS:
            raise ValueError("operation is not registered")
        if not isinstance(self.evidence, tuple) or len(self.evidence) != 4:
            raise ValueError("evidence must contain exactly four integers")
        if any(type(value) is not int for value in self.evidence):
            raise TypeError("evidence must contain exact integers")


@dataclass(frozen=True, slots=True)
class Stage1V2ArtifactVerification:
    verified: bool
    scenes: tuple[Stage2V1Scene, ...]
    exact_transcriptions: int
    mismatch_scene_ids: tuple[str, ...]


def _regular_file_digest(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return _sha256(path)


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def verify_stage1_v2_frozen_artifacts(
    frozen: Stage1V2FrozenResult,
    *,
    preflight_path: Path,
    console_path: Path,
    report_path: Path,
    records_path: Path,
    canonical_scenes: tuple[Stage1V2Scene, ...],
) -> Stage1V2ArtifactVerification:
    """Hash-bind and semantically replay all successful Stage-1 v2 outputs."""

    if not isinstance(frozen, Stage1V2FrozenResult):
        raise TypeError("frozen must be Stage1V2FrozenResult")
    expected_hashes = dict(frozen.source_sha256)
    paths = {
        "preflight": preflight_path,
        "console": console_path,
        "probe_report": report_path,
        "probe_records": records_path,
    }
    for label, candidate in paths.items():
        if _regular_file_digest(candidate, label=label) != expected_hashes[label]:
            raise ValueError(f"frozen Stage-1 v2 {label} SHA-256 mismatch")
    preflight = _json_object(preflight_path, label="Stage-1 v2 preflight")
    if (
        preflight.get("artifact_type") != "recoverability_stage1_v2_metadata_preflight"
        or preflight.get("ready") is not True
        or preflight.get("large_gpu_started") is not False
        or preflight.get("model_loaded") is not False
        or preflight.get("training_authorized") is not False
    ):
        raise ValueError("frozen Stage-1 v2 preflight semantics are invalid")
    report = _json_object(report_path, label="Stage-1 v2 report")
    expected_report = {
        "artifact_type": "recoverability_stage1_v2_development_probe",
        "dataset_id": frozen.dataset_id,
        "source_dataset_id": frozen.source_dataset_id,
        "source_split": frozen.source_split,
        "model_snapshot_sha256": frozen.model_snapshot_sha256,
        "scenes": frozen.scenes,
        "model_calls": frozen.model_calls,
        "parse_rate": frozen.parse_rate,
        "exact_transcription_rate": frozen.exact_transcription_rate,
        "probe_passed": True,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
        "training_invoked": False,
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise ValueError("frozen Stage-1 v2 report differs from its external anchor")
    if len(canonical_scenes) != frozen.scenes:
        raise ValueError("canonical Stage-1 v2 scene count differs")
    by_id = {scene.scene_id: scene for scene in canonical_scenes}
    if len(by_id) != len(canonical_scenes):
        raise ValueError("canonical Stage-1 v2 scene identifiers must be unique")
    rows: list[dict[str, object]] = []
    with records_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Stage-1 v2 record must be a JSON object")
            rows.append(row)
    if len(rows) != frozen.scenes:
        raise ValueError("frozen Stage-1 v2 record count differs")
    derived: list[Stage2V1Scene] = []
    mismatches: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if set(row) != {
            "scene_id",
            "chart_type",
            "operation",
            "raw_text",
            "parse_success",
            "exact_transcription",
            "error_code",
        }:
            raise ValueError("frozen Stage-1 v2 record schema is invalid")
        scene_id = row["scene_id"]
        if not isinstance(scene_id, str) or scene_id in seen or scene_id not in by_id:
            raise ValueError("frozen Stage-1 v2 scene identifier is invalid")
        seen.add(scene_id)
        source = by_id[scene_id]
        if row["chart_type"] != source.chart_type or row["operation"] != source.operation:
            raise ValueError("frozen Stage-1 v2 record stratum differs")
        if row["parse_success"] is not True or row["error_code"] is not None:
            raise ValueError("frozen Stage-1 v2 parse result differs")
        raw = row["raw_text"]
        if not isinstance(raw, str):
            raise ValueError("frozen Stage-1 v2 raw output must be text")
        evidence = parse_stage1_evidence(raw)
        exact = evidence.target_facts == source.values
        if row["exact_transcription"] is not exact:
            raise ValueError("frozen Stage-1 v2 exactness flag differs from replay")
        if not exact:
            mismatches.append(scene_id)
        derived.append(
            Stage2V1Scene(
                scene_id=scene_id,
                operation=source.operation,
                evidence=evidence.target_facts,
            )
        )
    if seen != set(by_id):
        raise ValueError("frozen Stage-1 v2 records do not cover the canonical probe")
    mismatch_ids = tuple(sorted(mismatches))
    if mismatch_ids != frozen.mismatch_scene_ids:
        raise ValueError("frozen Stage-1 v2 mismatch identifiers differ")
    return Stage1V2ArtifactVerification(
        verified=True,
        scenes=tuple(sorted(derived, key=lambda item: item.scene_id)),
        exact_transcriptions=frozen.scenes - len(mismatch_ids),
        mismatch_scene_ids=mismatch_ids,
    )


def _operation_result(scene: Stage2V1Scene) -> int:
    a, b, c, d = scene.evidence
    if scene.operation == "sum":
        return a + b
    if scene.operation == "difference":
        return a - b
    return max(a, b, c, d) - min(a, b, c, d)


def build_stage2_v1_messages(scene: Stage2V1Scene) -> tuple[dict[str, object], ...]:
    """Build one operation-specific exact DSL grammar without hidden gold."""

    if not isinstance(scene, Stage2V1Scene):
        raise TypeError("scene must be Stage2V1Scene")
    a, b, c, d = scene.evidence
    variables = f'"variables":{{"a":{a},"b":{b},"c":{c},"d":{d}}}'
    if scene.operation == "sum":
        steps = '"steps":[{"op":"add","inputs":["a","b"],"output":"result"}]'
    elif scene.operation == "difference":
        steps = '"steps":[{"op":"subtract","inputs":["a","b"],"output":"result"}]'
    else:
        steps = (
            '"steps":[{"op":"max","inputs":["a","b","c","d"],"output":"high"},'
            '{"op":"min","inputs":["a","b","c","d"],"output":"low"},'
            '{"op":"subtract","inputs":["high","low"],"output":"result"}]'
        )
    grammar = f'{{{variables},{steps},"answer":INTEGER}}'
    system = (
        "You are a strict deterministic integer DSL interface. Return exactly one JSON object "
        "by replacing INTEGER in this literal grammar with the result of the listed steps: "
        f"{grammar} Keep every variable and step byte-for-byte unchanged. The final step must "
        "produce result and answer must equal result. Begin with { and end with }. Do not emit "
        "code fences, prose, reasoning, extra keys, or trailing text."
    )
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": "Execute the registered integer steps exactly once."},
    )


@dataclass(frozen=True, slots=True)
class Stage2V1ProbeRecord:
    scene_id: str
    operation: str
    raw_text: str
    program_parse_success: bool
    program_execution_success: bool
    program_answer_match: bool
    operation_result_correct: bool
    error_code: str | None


@dataclass(frozen=True, slots=True)
class Stage2V1ProbeReport:
    scenes: int
    model_calls: int
    program_parse_rate: float
    execution_rate: float
    program_answer_consistency: float
    operation_result_accuracy: float
    error_counts: tuple[tuple[str, int], ...]
    format_retries: int
    training_invoked: bool
    probe_passed: bool


def run_stage2_v1_probe(
    scenes: tuple[Stage2V1Scene, ...],
    *,
    generate: Callable[[Stage2V1Scene, tuple[dict[str, object], ...]], str],
) -> tuple[Stage2V1ProbeReport, tuple[Stage2V1ProbeRecord, ...]]:
    """Run one text-only call per frozen scene; never retry or repair output."""

    if not isinstance(scenes, tuple) or len(scenes) != _PROBE_SCENES:
        raise ValueError("Stage-2 v1 probe requires exactly 24 frozen scenes")
    if any(not isinstance(scene, Stage2V1Scene) for scene in scenes):
        raise TypeError("probe scenes must contain Stage2V1Scene instances")
    identifiers = tuple(scene.scene_id for scene in scenes)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("probe scene identifiers must be unique")
    if not callable(generate):
        raise TypeError("generate must be callable")
    records: list[Stage2V1ProbeRecord] = []
    counts: Counter[str] = Counter()
    for scene in scenes:
        raw = generate(scene, build_stage2_v1_messages(scene))
        trusted = {
            name: TrustedBinding(f"stage1_target_{name}", value)
            for name, value in zip(("a", "b", "c", "d"), scene.evidence, strict=True)
        }
        evaluation = evaluate_program(raw, constraint_bindings=trusted)
        result_correct = bool(
            evaluation.program_parse_success
            and evaluation.program_execution_success
            and evaluation.program_answer_match
            and evaluation.final_answer == _operation_result(scene)
        )
        error_code = evaluation.error_code
        if error_code is None and not evaluation.program_answer_match:
            error_code = "program_answer_mismatch"
        if error_code is None and not result_correct:
            error_code = "operation_result_mismatch"
        if error_code is not None:
            counts[error_code] += 1
        records.append(
            Stage2V1ProbeRecord(
                scene_id=scene.scene_id,
                operation=scene.operation,
                raw_text=raw,
                program_parse_success=evaluation.program_parse_success,
                program_execution_success=evaluation.program_execution_success,
                program_answer_match=evaluation.program_answer_match,
                operation_result_correct=result_correct,
                error_code=error_code,
            )
        )
    total = len(records)
    parse_rate = sum(record.program_parse_success for record in records) / total
    execution_rate = sum(record.program_execution_success for record in records) / total
    consistency = sum(record.program_answer_match for record in records) / total
    accuracy = sum(record.operation_result_correct for record in records) / total
    passed = all(
        math.isclose(metric, 1.0) for metric in (parse_rate, execution_rate, consistency, accuracy)
    )
    return (
        Stage2V1ProbeReport(
            scenes=total,
            model_calls=total,
            program_parse_rate=parse_rate,
            execution_rate=execution_rate,
            program_answer_consistency=consistency,
            operation_result_accuracy=accuracy,
            error_counts=tuple(sorted(counts.items())),
            format_retries=0,
            training_invoked=False,
            probe_passed=passed,
        ),
        tuple(records),
    )
