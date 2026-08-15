"""Executor-authoritative Stage-2 v2 development interface."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .bridge import parse_stage1_evidence
from .dsl.executor import TrustedBinding
from .dsl.result_program import evaluate_result_program, parse_result_program
from .dsl.schema import ProgramOperation, ProgramStep
from .evidence import ProtocolLockResult, verify_protocol_lock
from .stage2_v1_failure import STAGE2_V1_DIAGNOSTIC_PACKAGE_PATHS

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OPERATIONS = frozenset({"difference", "max_minus_min", "sum"})
_PROBE_SCENES = 24

STAGE2_V2_SERVER_PACKAGE_LOCK_PATH = "configs/recoverability/server_package_lock_stage2_v2.yaml"
STAGE2_V2_SERVER_PACKAGE_PATHS = STAGE2_V1_DIAGNOSTIC_PACKAGE_PATHS | frozenset(
    {
        "configs/recoverability/stage2_v1_diagnostic_result.yaml",
        "configs/recoverability/stage2_v2_probe.yaml",
        "experiments/recoverability_v1/00_stage2_v2_preflight.py",
        "experiments/recoverability_v1/07_stage2_v2_probe.py",
        "src/compbias/recoverability/dsl/result_program.py",
        "src/compbias/recoverability/stage2_v2.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage2_v2_server_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    """Verify the exact closed package before any Stage-2 v2 import or model load."""

    root = repository_root.resolve()
    canonical = root / STAGE2_V2_SERVER_PACKAGE_LOCK_PATH
    if path.resolve() != canonical or path.is_symlink():
        raise ValueError("Stage-2 v2 server package lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != STAGE2_V2_SERVER_PACKAGE_PATHS:
        missing = sorted(STAGE2_V2_SERVER_PACKAGE_PATHS - observed)
        extra = sorted(observed - STAGE2_V2_SERVER_PACKAGE_PATHS)
        raise ValueError(f"Stage-2 v2 package closure mismatch; missing={missing}, extra={extra}")
    return result


@dataclass(frozen=True, slots=True)
class Stage2V2ProbeConfig:
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
    required_executor_answer_accuracy: float
    allow_rerun: bool
    hypothesis_test: bool


def load_stage2_v2_probe_config(path: Path) -> Stage2V2ProbeConfig:
    """Load the exact one-shot Stage-2 v2 development contract."""

    mapping = load_yaml_mapping(path, label="Stage-2 v2 probe config")
    expected: dict[str, object] = {
        "schema_version": 1,
        "status": "DEVELOPMENT_PROBE_NOT_RUN",
        "dataset_id": "CVA-Recoverability-Stage2-V2-Dev-Probe",
        "output_subdirectory": "stage2_v2_dev_probe",
        "source_dataset_id": "CVA-Chart-Pilot-v0.3",
        "source_split": "dev",
        "scenes": _PROBE_SCENES,
        "format_retries": 0,
        "required_program_parse_rate": 1.0,
        "required_execution_rate": 1.0,
        "required_executor_answer_accuracy": 1.0,
        "allow_rerun": False,
        "hypothesis_test": False,
    }
    reject_unknown_fields(mapping, set(expected), label="Stage-2 v2 probe config")
    if set(mapping) != set(expected):
        raise ValueError("Stage-2 v2 probe config is incomplete")
    for key, value in expected.items():
        if type(mapping[key]) is not type(value):
            raise TypeError(f"Stage-2 v2 probe config type differs for {key}")
    if dict(mapping) != expected:
        raise ValueError("Stage-2 v2 probe config differs from the registered contract")
    return Stage2V2ProbeConfig(**expected)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Stage2V1DiagnosticAnchor:
    schema_version: int
    status: str
    diagnostic_sha256: str
    source_stage2_records_sha256: str
    records: int
    program_parse_successes: int
    program_execution_successes: int
    executor_answer_matches: int
    verified: bool
    model_calls: int
    hypothesis_tested: bool
    confirmatory_execution_authorized: bool
    training_invoked: bool


def load_stage2_v1_diagnostic_anchor(path: Path) -> Stage2V1DiagnosticAnchor:
    """Load the externally supplied hash of the completed model-free v1 diagnosis."""

    mapping = load_yaml_mapping(path, label="Stage-2 v1 diagnostic anchor")
    fields = frozenset(Stage2V1DiagnosticAnchor.__dataclass_fields__)
    reject_unknown_fields(mapping, set(fields), label="Stage-2 v1 diagnostic anchor")
    if set(mapping) != fields:
        raise ValueError("Stage-2 v1 diagnostic anchor is incomplete")
    exact = {
        "schema_version": 1,
        "status": "FINAL_VERIFIED_STAGE2_V1_FAILURE_DO_NOT_RERUN",
        "diagnostic_sha256": ("d85510ea829a000bc31002f874e5a0ec795421aadec9f9042438d78337d9e7b4"),
        "source_stage2_records_sha256": (
            "b8ce766b9ba0555fe780e88195a0e4d4d3294cc8813357e7a33a5cfeea19793c"
        ),
        "records": 24,
        "program_parse_successes": 19,
        "program_execution_successes": 19,
        "executor_answer_matches": 13,
        "verified": True,
        "model_calls": 0,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
        "training_invoked": False,
    }
    for key, value in exact.items():
        if type(mapping[key]) is not type(value):
            raise TypeError(f"Stage-2 v1 diagnostic anchor type differs for {key}")
    if dict(mapping) != exact:
        raise ValueError("Stage-2 v1 diagnostic anchor differs from external evidence")
    return Stage2V1DiagnosticAnchor(**exact)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Stage2V1DiagnosticVerification:
    verified: bool
    records: int
    source_stage2_records_sha256: str


def verify_stage2_v1_diagnostic(
    anchor: Stage2V1DiagnosticAnchor,
    path: Path,
) -> Stage2V1DiagnosticVerification:
    """Hash-bind and check the minimum fail-closed semantics of the v1 diagnosis."""

    if not isinstance(anchor, Stage2V1DiagnosticAnchor):
        raise TypeError("anchor must be a Stage2V1DiagnosticAnchor")
    if path.is_symlink() or not path.is_file() or _sha256(path) != anchor.diagnostic_sha256:
        raise ValueError("Stage-2 v1 diagnostic SHA-256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage-2 v1 diagnostic must be UTF-8 JSON") from error
    required = {
        "verified": anchor.verified,
        "records": anchor.records,
        "replayed_program_parse_successes": anchor.program_parse_successes,
        "replayed_program_execution_successes": anchor.program_execution_successes,
        "replayed_program_answer_matches": anchor.executor_answer_matches,
        "replayed_operation_result_correct": anchor.executor_answer_matches,
        "model_calls": anchor.model_calls,
        "hypothesis_tested": anchor.hypothesis_tested,
        "confirmatory_execution_authorized": anchor.confirmatory_execution_authorized,
        "training_invoked": anchor.training_invoked,
        "source_stage2_records_sha256": anchor.source_stage2_records_sha256,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in required.items()
    ):
        raise ValueError("Stage-2 v1 diagnostic semantics differ from the external anchor")
    return Stage2V1DiagnosticVerification(
        verified=True,
        records=anchor.records,
        source_stage2_records_sha256=anchor.source_stage2_records_sha256,
    )


@dataclass(frozen=True, slots=True)
class Stage2V2Scene:
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


def load_stage2_v2_scenes_from_stage1_records(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[Stage2V2Scene, ...]:
    """Load only the externally hash-bound Stage-1 v2 perceived evidence."""

    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("Stage-1 v2 records digest is invalid")
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("frozen Stage-1 v2 records SHA-256 mismatch")
    rows: list[Stage2V2Scene] = []
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Stage-1 v2 record must be JSON") from error
            if not isinstance(row, dict) or set(row) != {
                "scene_id",
                "chart_type",
                "operation",
                "raw_text",
                "parse_success",
                "exact_transcription",
                "error_code",
            }:
                raise ValueError("Stage-1 v2 record schema is invalid")
            scene_id = row["scene_id"]
            operation = row["operation"]
            raw = row["raw_text"]
            if (
                not isinstance(scene_id, str)
                or scene_id in identifiers
                or not isinstance(operation, str)
                or not isinstance(raw, str)
                or row["parse_success"] is not True
                or row["error_code"] is not None
            ):
                raise ValueError("Stage-1 v2 record does not contain trusted evidence")
            identifiers.add(scene_id)
            evidence = parse_stage1_evidence(raw)
            rows.append(Stage2V2Scene(scene_id, operation, evidence.target_facts))
    if len(rows) != _PROBE_SCENES:
        raise ValueError("Stage-1 v2 evidence must contain exactly 24 scenes")
    return tuple(sorted(rows, key=lambda item: item.scene_id))


def _registered_steps(operation: str) -> tuple[ProgramStep, ...]:
    if operation == "sum":
        return (ProgramStep(ProgramOperation.ADD, ("a", "b"), "result"),)
    if operation == "difference":
        return (ProgramStep(ProgramOperation.SUBTRACT, ("a", "b"), "result"),)
    return (
        ProgramStep(ProgramOperation.MAX, ("a", "b", "c", "d"), "high"),
        ProgramStep(ProgramOperation.MIN, ("a", "b", "c", "d"), "low"),
        ProgramStep(ProgramOperation.SUBTRACT, ("high", "low"), "result"),
    )


def _operation_result(scene: Stage2V2Scene) -> int:
    a, b, c, d = scene.evidence
    if scene.operation == "sum":
        return a + b
    if scene.operation == "difference":
        return a - b
    return max(a, b, c, d) - min(a, b, c, d)


def _matches_registered_program(scene: Stage2V2Scene, raw: str) -> bool:
    try:
        program = parse_result_program(raw)
    except ValueError:
        return False
    expected_variables = tuple(zip(("a", "b", "c", "d"), scene.evidence, strict=True))
    return (
        program.variables == expected_variables
        and program.steps == _registered_steps(scene.operation)
        and program.return_variable == "result"
    )


def build_stage2_v2_messages(scene: Stage2V2Scene) -> tuple[dict[str, object], ...]:
    """Bind one literal result-pointer program without a numeric answer slot."""

    if not isinstance(scene, Stage2V2Scene):
        raise TypeError("scene must be Stage2V2Scene")
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
    grammar = f'{{{variables},{steps},"return":"result"}}'
    system = (
        "You are a strict deterministic integer DSL interface. Return exactly this one literal "
        f"JSON program byte-for-byte: {grammar} The trusted executor, not you, computes the "
        "numeric result from the returned graph. Begin with { and end with }. Do not emit code "
        "fences, prose, reasoning, extra keys, numeric results, or trailing text."
    )
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": "Return the registered result-pointer program exactly."},
    )


@dataclass(frozen=True, slots=True)
class Stage2V2ProbeRecord:
    scene_id: str
    operation: str
    raw_text: str
    program_parse_success: bool
    program_execution_success: bool
    executed_result: int | None
    final_answer: int | None
    executor_answer_correct: bool
    error_code: str | None


@dataclass(frozen=True, slots=True)
class Stage2V2ProbeReport:
    scenes: int
    model_calls: int
    program_parse_rate: float
    execution_rate: float
    executor_answer_accuracy: float
    error_counts: tuple[tuple[str, int], ...]
    format_retries: int
    training_invoked: bool
    probe_passed: bool


def run_stage2_v2_probe(
    scenes: tuple[Stage2V2Scene, ...],
    *,
    generate: Callable[[Stage2V2Scene, tuple[dict[str, object], ...]], str],
) -> tuple[Stage2V2ProbeReport, tuple[Stage2V2ProbeRecord, ...]]:
    """Run one text-only call per frozen scene without retries or output repair."""

    if not isinstance(scenes, tuple) or len(scenes) != _PROBE_SCENES:
        raise ValueError("Stage-2 v2 probe requires exactly 24 frozen scenes")
    if any(not isinstance(scene, Stage2V2Scene) for scene in scenes):
        raise TypeError("probe scenes must contain Stage2V2Scene instances")
    identifiers = tuple(scene.scene_id for scene in scenes)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("probe scene identifiers must be unique")
    if not callable(generate):
        raise TypeError("generate must be callable")
    records: list[Stage2V2ProbeRecord] = []
    counts: Counter[str] = Counter()
    for scene in scenes:
        raw = generate(scene, build_stage2_v2_messages(scene))
        contract_match = _matches_registered_program(scene, raw)
        trusted = {
            name: TrustedBinding(f"stage1_target_{name}", value)
            for name, value in zip(("a", "b", "c", "d"), scene.evidence, strict=True)
        }
        evaluation = evaluate_result_program(raw, constraint_bindings=trusted)
        correct = bool(
            contract_match
            and evaluation.program_execution_success
            and evaluation.final_answer == _operation_result(scene)
        )
        error_code = evaluation.error_code
        if error_code is None and not contract_match:
            error_code = "program_contract_mismatch"
        elif error_code is None and not correct:
            error_code = "executor_answer_mismatch"
        if error_code is not None:
            counts[error_code] += 1
        records.append(
            Stage2V2ProbeRecord(
                scene_id=scene.scene_id,
                operation=scene.operation,
                raw_text=raw,
                program_parse_success=evaluation.program_parse_success,
                program_execution_success=evaluation.program_execution_success,
                executed_result=evaluation.executed_result,
                final_answer=evaluation.final_answer,
                executor_answer_correct=correct,
                error_code=error_code,
            )
        )
    total = len(records)
    parse_rate = sum(item.program_parse_success for item in records) / total
    execution_rate = sum(item.program_execution_success for item in records) / total
    accuracy = sum(item.executor_answer_correct for item in records) / total
    passed = all(math.isclose(metric, 1.0) for metric in (parse_rate, execution_rate, accuracy))
    return (
        Stage2V2ProbeReport(
            scenes=total,
            model_calls=total,
            program_parse_rate=parse_rate,
            execution_rate=execution_rate,
            executor_answer_accuracy=accuracy,
            error_counts=tuple(sorted(counts.items())),
            format_retries=0,
            training_invoked=False,
            probe_passed=passed,
        ),
        tuple(records),
    )
