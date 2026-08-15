"""Preregistered two-stage measurement qualification on fresh controlled scenes."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Set
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import beta

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .bridge import parse_stage1_evidence
from .dsl.executor import TrustedBinding
from .dsl.result_program import evaluate_result_program, parse_result_program
from .dsl.schema import ProgramOperation, ProgramStep
from .stage1_v2 import build_stage1_v2_messages
from .stage2_v2 import Stage2V2Scene, build_stage2_v2_messages

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHART_TYPES = ("grouped_bar", "line")
_OPERATIONS = ("difference", "max_minus_min", "sum")
_VALUE_MIN = 2
_VALUE_MAX = 18
_SOURCE_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "sample_id",
        "split",
        "chart_type",
        "operation",
        "values",
        "question",
        "answer",
        "image",
        "mechanism",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MeasurementQualificationConfig:
    schema_version: int
    status: str
    dataset_id: str
    output_subdirectory: str
    seed: int
    image_size: tuple[int, int]
    source_dataset_id: str
    source_dataset_records_sha256: str
    source_stage2_v2_external_evidence_sha256: str
    scenes: int
    per_stratum: int
    format_retries: int
    confidence: float
    minimum_lower_bound: float
    allow_rerun: bool
    hypothesis_test: bool
    confirmatory_execution_authorized: bool


def load_measurement_qualification_config(path: Path) -> MeasurementQualificationConfig:
    """Load the exact one-shot measurement-only qualification contract."""

    mapping = load_yaml_mapping(path, label="measurement qualification configuration")
    expected: dict[str, object] = {
        "schema_version": 1,
        "status": "PREREGISTERED_NOT_RUN",
        "dataset_id": "CVA-Recoverability-Measurement-Qualification-v1",
        "output_subdirectory": "measurement_qualification_v1",
        "seed": 20260817,
        "image_size": [512, 384],
        "source_dataset_id": "CVA-Chart-Pilot-v0.3",
        "source_dataset_records_sha256": (
            "92ccdf54b11e2a6c12e12ef5273137824c6f3b94f38224abeb32d8319b83a62b"
        ),
        "source_stage2_v2_external_evidence_sha256": (
            "3a9e521cfe718cc3dea9aee4f1591aac761fa47f893c986eb1ba722a44374577"
        ),
        "scenes": 300,
        "per_stratum": 50,
        "format_retries": 0,
        "confidence": 0.95,
        "minimum_lower_bound": 0.98,
        "allow_rerun": False,
        "hypothesis_test": False,
        "confirmatory_execution_authorized": False,
    }
    reject_unknown_fields(mapping, set(expected), label="measurement qualification configuration")
    if set(mapping) != set(expected):
        raise ValueError("measurement qualification config is incomplete")
    for key, value in expected.items():
        if type(mapping[key]) is not type(value):
            raise TypeError(f"measurement qualification config type differs for {key}")
    if dict(mapping) != expected:
        raise ValueError("measurement qualification config differs from the registered contract")
    values = {**expected, "image_size": (512, 384)}
    return MeasurementQualificationConfig(**values)  # type: ignore[arg-type]


def _exact_values(value: object, *, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(item) is not int or not _VALUE_MIN <= item <= _VALUE_MAX for item in value)
    ):
        raise ValueError(f"{label} must contain four exact integers in [2, 18]")
    return tuple(value)  # type: ignore[return-value]


def load_reserved_numeric_tables(
    path: Path,
    *,
    expected_sha256: str,
) -> frozenset[tuple[int, int, int, int]]:
    """Read hash-bound v0.3 numeric tables used to prevent qualification overlap."""

    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("source records SHA-256 is invalid")
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("source records SHA-256 mismatch")
    tables: set[tuple[int, int, int, int]] = set()
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("source record must be JSON") from error
            if not isinstance(row, dict) or set(row) != _SOURCE_RECORD_FIELDS:
                raise ValueError("source record schema is invalid")
            identifier = row.get("sample_id")
            if (
                row.get("schema_version") != 1
                or row.get("dataset_id") != "CVA-Chart-Pilot-v0.3"
                or not isinstance(identifier, str)
                or _IDENTIFIER.fullmatch(identifier) is None
                or identifier in identifiers
            ):
                raise ValueError("source record identity is invalid")
            identifiers.add(identifier)
            tables.add(_exact_values(row.get("values"), label="source values"))
    if not identifiers:
        raise ValueError("source records must not be empty")
    return frozenset(tables)


def _answer(values: tuple[int, int, int, int], operation: str) -> int:
    if operation == "sum":
        return values[0] + values[1]
    if operation == "difference":
        return values[0] - values[1]
    if operation == "max_minus_min":
        return max(values) - min(values)
    raise ValueError("operation is not registered")


def _question(operation: str) -> str:
    return {
        "difference": "What is the first value minus the second value?",
        "max_minus_min": "What is the maximum value minus the minimum value?",
        "sum": "What is the sum of the first two values?",
    }[operation]


@dataclass(frozen=True, slots=True)
class QualificationDatasetRecord:
    schema_version: int
    dataset_id: str
    sample_id: str
    split: str
    chart_type: str
    operation: str
    values: tuple[int, int, int, int]
    question: str
    answer: int
    image: str
    mechanism: str


def build_qualification_records(
    config: MeasurementQualificationConfig,
    *,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
) -> tuple[QualificationDatasetRecord, ...]:
    """Create a fixed balanced scene plan disjoint from every prior numeric table."""

    if not isinstance(config, MeasurementQualificationConfig):
        raise TypeError("config must be a MeasurementQualificationConfig")
    if not isinstance(reserved_numeric_tables, Set):
        raise TypeError("reserved_numeric_tables must be a set-like collection")
    reserved = frozenset(reserved_numeric_tables)
    if any(
        not isinstance(values, tuple)
        or len(values) != 4
        or any(type(value) is not int for value in values)
        for values in reserved
    ):
        raise ValueError("reserved numeric tables are invalid")
    rng = random.Random(config.seed)
    chosen: set[tuple[int, int, int, int]] = set()
    records: list[QualificationDatasetRecord] = []
    for chart_type in _CHART_TYPES:
        for operation in _OPERATIONS:
            for _ in range(config.per_stratum):
                while True:
                    values = tuple(rng.randint(_VALUE_MIN, _VALUE_MAX) for _ in range(4))
                    if values not in reserved and values not in chosen:
                        break
                chosen.add(values)  # type: ignore[arg-type]
                index = len(records)
                sample_id = f"qualification-{index:06d}"
                typed_values = values  # type: ignore[assignment]
                records.append(
                    QualificationDatasetRecord(
                        schema_version=1,
                        dataset_id=config.dataset_id,
                        sample_id=sample_id,
                        split="qualification",
                        chart_type=chart_type,
                        operation=operation,
                        values=typed_values,
                        question=_question(operation),
                        answer=_answer(typed_values, operation),
                        image=f"images/{sample_id}.png",
                        mechanism="iid",
                    )
                )
    if len(records) != config.scenes:
        raise AssertionError("qualification record count drifted from the registered contract")
    return tuple(records)


@dataclass(frozen=True, slots=True)
class QualificationScene:
    scene_id: str
    image_path: Path
    chart_type: str
    operation: str
    values: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or _IDENTIFIER.fullmatch(self.scene_id) is None:
            raise ValueError("scene_id must be a bounded safe identifier")
        if not isinstance(self.image_path, Path) or not self.image_path.is_absolute():
            raise ValueError("image_path must be absolute")
        if self.chart_type not in _CHART_TYPES:
            raise ValueError("chart_type is not registered")
        if self.operation not in _OPERATIONS:
            raise ValueError("operation is not registered")
        if not isinstance(self.values, tuple) or len(self.values) != 4:
            raise ValueError("values must contain exactly four integers")
        if any(type(value) is not int for value in self.values):
            raise TypeError("values must contain exact integers")


@dataclass(frozen=True, slots=True)
class MeasurementQualificationRecord:
    scene_id: str
    chart_type: str
    operation: str
    stage1_raw: str
    stage1_parse_success: bool
    exact_transcription: bool
    perceived_values: tuple[int, int, int, int] | None
    stage1_error_code: str | None
    stage2_raw: str | None
    stage2_program_parse_success: bool
    stage2_program_execution_success: bool
    executed_result: int | None
    final_answer: int | None
    executor_answer_correct: bool
    stage2_error_code: str | None


@dataclass(frozen=True, slots=True)
class MeasurementQualificationReport:
    scenes: int
    stage1_model_calls: int
    stage2_model_calls: int
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
    qualification_passed: bool
    format_retries: int
    hypothesis_tested: bool
    confirmatory_execution_authorized: bool
    training_invoked: bool


def one_sided_binomial_lower(successes: int, total: int, *, confidence: float) -> float:
    """Return the exact one-sided Clopper-Pearson lower confidence bound."""

    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("successes and total must be exact integers with 0 <= successes <= total")
    if total < 1:
        raise ValueError("total must be positive")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    confidence_value = float(confidence)
    if not 0 < confidence_value < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence_value, successes, total - successes + 1))


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


def _program_contract_matches(
    operation: str,
    perceived: tuple[int, int, int, int],
    raw: str,
) -> bool:
    try:
        program = parse_result_program(raw)
    except ValueError:
        return False
    expected_variables = tuple(zip(("a", "b", "c", "d"), perceived, strict=True))
    return (
        program.variables == expected_variables
        and program.steps == _registered_steps(operation)
        and program.return_variable == "result"
    )


def _rate(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def run_measurement_qualification(
    scenes: tuple[QualificationScene, ...],
    *,
    config: MeasurementQualificationConfig,
    stage1_generate: Callable[[QualificationScene, tuple[dict[str, object], ...]], str],
    stage2_generate: Callable[
        [QualificationScene, tuple[int, int, int, int], tuple[dict[str, object], ...]], str
    ],
) -> tuple[MeasurementQualificationReport, tuple[MeasurementQualificationRecord, ...]]:
    """Run one Stage-1 call and at most one Stage-2 call per preregistered scene."""

    if not isinstance(config, MeasurementQualificationConfig):
        raise TypeError("config must be a MeasurementQualificationConfig")
    if not isinstance(scenes, tuple) or len(scenes) != config.scenes:
        raise ValueError("qualification requires exactly 300 preregistered scenes")
    if any(not isinstance(scene, QualificationScene) for scene in scenes):
        raise TypeError("scenes must contain QualificationScene instances")
    identifiers = tuple(scene.scene_id for scene in scenes)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("qualification scene identifiers must be unique")
    if not callable(stage1_generate) or not callable(stage2_generate):
        raise TypeError("generation callbacks must be callable")

    records: list[MeasurementQualificationRecord] = []
    stage1_messages = build_stage1_v2_messages()
    stage1_successes = 0
    exact_successes = 0
    stage2_calls = 0
    stage2_parse_successes = 0
    stage2_execution_successes = 0
    executor_answer_successes = 0

    for scene in scenes:
        stage1_raw = stage1_generate(scene, stage1_messages)
        perceived: tuple[int, int, int, int] | None = None
        stage1_error: str | None = None
        try:
            evidence = parse_stage1_evidence(stage1_raw)
            perceived = evidence.target_facts
        except ValueError:
            stage1_error = "stage1_parse_failure"
        stage1_ok = perceived is not None
        exact = stage1_ok and perceived == scene.values
        stage1_successes += int(stage1_ok)
        exact_successes += int(exact)

        stage2_raw: str | None = None
        stage2_parse = False
        stage2_execution = False
        executed_result: int | None = None
        final_answer: int | None = None
        executor_correct = False
        stage2_error: str | None = None
        if perceived is not None:
            stage2_calls += 1
            stage2_scene = Stage2V2Scene(scene.scene_id, scene.operation, perceived)
            stage2_raw = stage2_generate(
                scene,
                perceived,
                build_stage2_v2_messages(stage2_scene),
            )
            trusted = {
                name: TrustedBinding(f"stage1_target_{name}", value)
                for name, value in zip(("a", "b", "c", "d"), perceived, strict=True)
            }
            evaluation = evaluate_result_program(stage2_raw, constraint_bindings=trusted)
            contract_match = _program_contract_matches(scene.operation, perceived, stage2_raw)
            stage2_parse = evaluation.program_parse_success
            stage2_execution = evaluation.program_execution_success
            executed_result = evaluation.executed_result
            final_answer = evaluation.final_answer
            executor_correct = bool(
                contract_match
                and stage2_execution
                and final_answer == _answer(perceived, scene.operation)
            )
            stage2_error = evaluation.error_code
            if stage2_error is None and not contract_match:
                stage2_error = "program_contract_mismatch"
            elif stage2_error is None and not executor_correct:
                stage2_error = "executor_answer_mismatch"
            stage2_parse_successes += int(stage2_parse)
            stage2_execution_successes += int(stage2_execution)
            executor_answer_successes += int(executor_correct)

        records.append(
            MeasurementQualificationRecord(
                scene_id=scene.scene_id,
                chart_type=scene.chart_type,
                operation=scene.operation,
                stage1_raw=stage1_raw,
                stage1_parse_success=stage1_ok,
                exact_transcription=exact,
                perceived_values=perceived,
                stage1_error_code=stage1_error,
                stage2_raw=stage2_raw,
                stage2_program_parse_success=stage2_parse,
                stage2_program_execution_success=stage2_execution,
                executed_result=executed_result,
                final_answer=final_answer,
                executor_answer_correct=executor_correct,
                stage2_error_code=stage2_error,
            )
        )

    total = len(records)
    stage1_lower = one_sided_binomial_lower(stage1_successes, total, confidence=config.confidence)
    downstream_total = stage2_calls
    stage2_parse_lower = one_sided_binomial_lower(
        stage2_parse_successes, downstream_total, confidence=config.confidence
    )
    stage2_execution_lower = one_sided_binomial_lower(
        stage2_execution_successes, downstream_total, confidence=config.confidence
    )
    executor_lower = one_sided_binomial_lower(
        executor_answer_successes, downstream_total, confidence=config.confidence
    )
    failures: list[str] = []
    checks = (
        (stage1_lower, "stage1_parse_lower_below_0_98"),
        (stage2_parse_lower, "stage2_program_parse_lower_below_0_98"),
        (stage2_execution_lower, "stage2_execution_lower_below_0_98"),
        (executor_lower, "executor_answer_lower_below_0_98"),
    )
    for value, code in checks:
        if value < config.minimum_lower_bound or not math.isfinite(value):
            failures.append(code)
    report = MeasurementQualificationReport(
        scenes=total,
        stage1_model_calls=total,
        stage2_model_calls=stage2_calls,
        model_calls=total + stage2_calls,
        stage1_parse_successes=stage1_successes,
        stage1_parse_rate=_rate(stage1_successes, total),
        stage1_parse_lower=stage1_lower,
        exact_transcription_rate=_rate(exact_successes, total),
        stage2_program_parse_successes=stage2_parse_successes,
        stage2_program_parse_rate=_rate(stage2_parse_successes, downstream_total),
        stage2_program_parse_lower=stage2_parse_lower,
        stage2_execution_successes=stage2_execution_successes,
        stage2_execution_rate=_rate(stage2_execution_successes, downstream_total),
        stage2_execution_lower=stage2_execution_lower,
        executor_answer_successes=executor_answer_successes,
        executor_answer_accuracy=_rate(executor_answer_successes, downstream_total),
        executor_answer_lower=executor_lower,
        gate_failures=tuple(failures),
        qualification_passed=not failures,
        format_retries=config.format_retries,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
    )
    return report, tuple(records)
