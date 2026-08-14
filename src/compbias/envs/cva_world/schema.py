"""Immutable, JSON-compatible records for the CVA-World dataset."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias


class TaskFamily(str, Enum):
    """The five registered CVA-World task families."""

    DIGIT_OFFSET = "digit_offset"
    COUNT_TRANSFORM = "count_transform"
    GAUGE_CALIBRATION = "gauge_calibration"
    BAR_CHART_AGGREGATE = "bar_chart_aggregate"
    RELATION_RULE = "relation_rule"


class SemanticSplit(str, Enum):
    """Semantically distinct dataset partitions."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    VAL = "val"
    IID_TEST = "iid_test"
    OOD_TEST = "ood_test"


ImmutableJSON: TypeAlias = object

BAR_CHART_OPERATIONS = ("sum", "difference", "ratio")
BAR_CHART_INDICES = (0, 1)
BAR_CHART_QUESTION_TEXTS: Mapping[str, str] = MappingProxyType(
    {
        "sum": "Sum the first two bar heights.",
        "difference": "Subtract the second bar height from the first.",
        "ratio": "Divide the first bar height by the second.",
    }
)


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


_SAFE_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _require_safe_basename(value: object, field: str) -> str:
    result = _require_nonempty_string(value, field)
    if result in {".", ".."} or _SAFE_BASENAME.fullmatch(result) is None:
        raise ValueError(f"{field} must be a safe basename")
    return result


def _freeze(value: object, field: str) -> ImmutableJSON:
    """Detach and recursively freeze a JSON-compatible value."""

    if isinstance(value, Mapping):
        frozen = {
            _require_nonempty_string(key, f"{field} key"): _freeze(item, f"{field}.{key}")
            for key, item in value.items()
        }
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, field) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} contains a non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{field} contains unsupported value {type(value).__name__}")


def _require_closed_keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    field: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {field} fields: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {field} fields: {', '.join(sorted(unknown))}")
        raise ValueError("; ".join(details))


def _validate_bar_chart_payload(
    scene: Mapping[str, object], question: Mapping[str, object]
) -> None:
    operation = question.get("operation")
    if not isinstance(operation, str) or operation not in BAR_CHART_OPERATIONS:
        raise ValueError(f"bar chart operation is unsupported: {operation!r}")

    bars = scene.get("bars")
    if not isinstance(bars, Sequence) or isinstance(bars, (str, bytes)) or len(bars) != 4:
        raise ValueError("bar chart scene must contain exactly four bar heights")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in bars
    ):
        raise ValueError("bar chart heights must be finite non-negative numbers")

    maximum = scene.get("maximum")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or maximum < max(bars)
    ):
        raise ValueError("bar chart maximum must be finite and contain every height")

    indices = question.get("indices")
    if (
        not isinstance(indices, Sequence)
        or isinstance(indices, (str, bytes))
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or tuple(indices) != BAR_CHART_INDICES
    ):
        raise ValueError("bar chart indices must equal the registered pair [0, 1]")
    if operation == "ratio" and bars[1] == 0:
        raise ValueError("bar chart ratio denominator must be nonzero")
    if "text" in question and question["text"] != BAR_CHART_QUESTION_TEXTS[operation]:
        raise ValueError("question text does not match bar_chart_aggregate")


def _validate_scene_question(
    task_family: TaskFamily,
    scene: Mapping[str, object],
    question: Mapping[str, object],
) -> None:
    schemas = {
        TaskFamily.DIGIT_OFFSET: (
            {"value"},
            {"template", "operand"},
        ),
        TaskFamily.COUNT_TRANSFORM: (
            {"count", "shape"},
            {"template", "scale", "offset"},
        ),
        TaskFamily.GAUGE_CALIBRATION: (
            {"reading", "minimum", "maximum"},
            {"template", "scale", "offset"},
        ),
        TaskFamily.BAR_CHART_AGGREGATE: (
            {"bars", "maximum"},
            {"template", "operation", "indices"},
        ),
        TaskFamily.RELATION_RULE: (
            {"relation", "entity_pair"},
            {"template", "rule"},
        ),
    }
    scene_keys, question_keys = schemas[task_family]
    _require_closed_keys(scene, required=scene_keys, field="scene")
    _require_closed_keys(
        question,
        required=question_keys,
        optional={"text"},
        field="question",
    )
    expected_templates = {
        TaskFamily.DIGIT_OFFSET: "add_constant",
        TaskFamily.COUNT_TRANSFORM: "affine_transform",
        TaskFamily.GAUGE_CALIBRATION: "calibrate",
        TaskFamily.BAR_CHART_AGGREGATE: "aggregate",
        TaskFamily.RELATION_RULE: "relation_lookup",
    }
    if question.get("template") != expected_templates[task_family]:
        raise ValueError(f"question template does not match {task_family.value}")
    if task_family is TaskFamily.BAR_CHART_AGGREGATE:
        _validate_bar_chart_payload(scene, question)
    expected_texts = {
        TaskFamily.COUNT_TRANSFORM: "Apply the stated scale and offset to the object count.",
        TaskFamily.GAUGE_CALIBRATION: "Calibrate the gauge reading with the stated affine rule.",
        TaskFamily.RELATION_RULE: "Use the supplied relation rule to name the class.",
    }
    if "text" in question and task_family is not TaskFamily.BAR_CHART_AGGREGATE:
        expected_text = (
            f"Add {question.get('operand')} to the number shown in the image."
            if task_family is TaskFamily.DIGIT_OFFSET
            else expected_texts[task_family]
        )
        if question["text"] != expected_text:
            raise ValueError(f"question text does not match {task_family.value}")
    if task_family is TaskFamily.COUNT_TRANSFORM and scene.get("shape") != "circle":
        raise ValueError("count scene shape must be circle")
    if task_family is TaskFamily.RELATION_RULE:
        relations = {"left_of", "right_of", "above", "below", "parallel", "intersect"}
        if scene.get("relation") not in relations:
            raise ValueError("scene relation is unsupported")
        if (
            not isinstance(scene.get("entity_pair"), str)
            or _SAFE_BASENAME.fullmatch(
                scene["entity_pair"]  # type: ignore[arg-type]
            )
            is None
        ):
            raise ValueError("relation entity_pair must be a safe token")
        rule = _require_mapping(question.get("rule"), "question.rule")
        if set(rule) != relations:
            raise ValueError("relation rule keys must contain all six registered relations")
        if any(not isinstance(value, str) or not value for value in rule.values()):
            raise ValueError("relation rule values must be non-empty strings")


def _thaw(value: object) -> object:
    """Return a detached mutable representation suitable for JSON encoding."""

    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ErrorSpec:
    """An executable corruption with an auditable identifier and severity."""

    error_id: str
    family: str
    severity: float
    parameters: Mapping[str, ImmutableJSON]

    def __post_init__(self) -> None:
        object.__setattr__(self, "error_id", _require_nonempty_string(self.error_id, "error_id"))
        object.__setattr__(self, "family", _require_nonempty_string(self.family, "family"))
        if isinstance(self.severity, bool) or not isinstance(self.severity, (int, float)):
            raise TypeError("severity must be numeric")
        severity = float(self.severity)
        if not math.isfinite(severity) or severity < 0:
            raise ValueError("severity must be non-negative")
        object.__setattr__(self, "severity", severity)
        parameters = _require_mapping(self.parameters, "parameters")
        object.__setattr__(self, "parameters", _freeze(parameters, "parameters"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ErrorSpec:
        mapping = _require_mapping(value, "error")
        unknown = set(mapping) - {"error_id", "family", "severity", "parameters"}
        if unknown:
            raise ValueError(f"unknown error fields: {', '.join(sorted(unknown))}")
        try:
            return cls(
                error_id=mapping["error_id"],  # type: ignore[arg-type]
                family=mapping["family"],  # type: ignore[arg-type]
                severity=mapping["severity"],  # type: ignore[arg-type]
                parameters=mapping["parameters"],  # type: ignore[arg-type]
            )
        except KeyError as error:
            raise ValueError(f"missing error field: {error.args[0]}") from error

    def to_mapping(self) -> dict[str, object]:
        return {
            "error_id": self.error_id,
            "family": self.family,
            "severity": self.severity,
            "parameters": _thaw(self.parameters),
        }


@dataclass(frozen=True)
class SplitKeys:
    """Independent semantic, visual, and error-mechanism partition keys."""

    semantic_split: SemanticSplit
    visual_style: str
    error_mechanism: str

    def __post_init__(self) -> None:
        try:
            semantic_split = SemanticSplit(self.semantic_split)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid semantic_split: {self.semantic_split!r}") from error
        object.__setattr__(self, "semantic_split", semantic_split)
        object.__setattr__(
            self, "visual_style", _require_nonempty_string(self.visual_style, "visual_style")
        )
        object.__setattr__(
            self,
            "error_mechanism",
            _require_nonempty_string(self.error_mechanism, "error_mechanism"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SplitKeys:
        mapping = _require_mapping(value, "split_keys")
        try:
            return cls(
                semantic_split=mapping["semantic_split"],  # type: ignore[arg-type]
                visual_style=mapping["visual_style"],  # type: ignore[arg-type]
                error_mechanism=mapping["error_mechanism"],  # type: ignore[arg-type]
            )
        except KeyError as error:
            raise ValueError(f"missing split_keys field: {error.args[0]}") from error

    def to_mapping(self) -> dict[str, str]:
        return {
            "semantic_split": self.semantic_split.value,
            "visual_style": self.visual_style,
            "error_mechanism": self.error_mechanism,
        }


@dataclass(frozen=True)
class CVASample:
    """One fully specified and deeply immutable CVA-World example."""

    sample_id: str
    image_path: str
    task_family: TaskFamily
    scene: Mapping[str, ImmutableJSON]
    question: Mapping[str, ImmutableJSON]
    canonical_answer: ImmutableJSON
    canonical_reasoning: Mapping[str, ImmutableJSON]
    error_catalog: tuple[ErrorSpec, ...]
    split_keys: SplitKeys
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _require_safe_basename(self.sample_id, "sample_id"))
        if self.source_id is not None:
            object.__setattr__(
                self,
                "source_id",
                _require_safe_basename(self.source_id, "source_id"),
            )
        object.__setattr__(
            self, "image_path", _require_nonempty_string(self.image_path, "image_path")
        )
        try:
            task_family = TaskFamily(self.task_family)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid task_family: {self.task_family!r}") from error
        object.__setattr__(self, "task_family", task_family)
        scene = _require_mapping(self.scene, "scene")
        question = _require_mapping(self.question, "question")
        _validate_scene_question(task_family, scene, question)
        object.__setattr__(self, "scene", _freeze(scene, "scene"))
        object.__setattr__(self, "question", _freeze(question, "question"))
        if self.canonical_answer is None:
            raise ValueError("canonical_answer must not be None")
        object.__setattr__(
            self, "canonical_answer", _freeze(self.canonical_answer, "canonical_answer")
        )
        object.__setattr__(
            self,
            "canonical_reasoning",
            _freeze(
                _require_mapping(self.canonical_reasoning, "canonical_reasoning"),
                "canonical_reasoning",
            ),
        )

        if not isinstance(self.error_catalog, Sequence) or isinstance(
            self.error_catalog, (str, bytes)
        ):
            raise TypeError("error_catalog must be a sequence")
        errors = tuple(
            item if isinstance(item, ErrorSpec) else ErrorSpec.from_mapping(item)
            for item in self.error_catalog
        )
        from .corruptions import validate_error_spec

        for error in errors:
            validate_error_spec(error)
        if not errors:
            raise ValueError("error_catalog must not be empty")
        identifiers = tuple(item.error_id for item in errors)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("error_id values must be unique")
        truth = tuple(item for item in errors if item.error_id == "truth")
        if len(truth) != 1:
            raise ValueError("error_catalog must contain exactly one truth entry")
        if truth[0].family != "truth" or truth[0].severity != 0 or truth[0].parameters:
            raise ValueError("truth error must have family truth, severity 0, and no parameters")
        object.__setattr__(self, "error_catalog", errors)

        split_keys = (
            self.split_keys
            if isinstance(self.split_keys, SplitKeys)
            else SplitKeys.from_mapping(self.split_keys)
        )
        object.__setattr__(self, "split_keys", split_keys)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CVASample:
        mapping = _require_mapping(value, "sample")
        required = (
            "sample_id",
            "image_path",
            "task_family",
            "scene",
            "question",
            "canonical_answer",
            "canonical_reasoning",
            "error_catalog",
            "split_keys",
        )
        missing = next((field for field in required if field not in mapping), None)
        if missing is not None:
            raise ValueError(f"missing required field: {missing}")
        unknown = set(mapping) - {*required, "source_id"}
        if unknown:
            raise ValueError(f"unknown sample fields: {', '.join(sorted(unknown))}")
        return cls(
            sample_id=mapping["sample_id"],  # type: ignore[arg-type]
            image_path=mapping["image_path"],  # type: ignore[arg-type]
            task_family=mapping["task_family"],  # type: ignore[arg-type]
            scene=mapping["scene"],  # type: ignore[arg-type]
            question=mapping["question"],  # type: ignore[arg-type]
            canonical_answer=mapping["canonical_answer"],
            canonical_reasoning=mapping["canonical_reasoning"],  # type: ignore[arg-type]
            error_catalog=tuple(mapping["error_catalog"]),  # type: ignore[arg-type]
            split_keys=mapping["split_keys"],  # type: ignore[arg-type]
            source_id=mapping.get("source_id"),  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        result = {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "task_family": self.task_family.value,
            "scene": _thaw(self.scene),
            "question": _thaw(self.question),
            "canonical_answer": _thaw(self.canonical_answer),
            "canonical_reasoning": _thaw(self.canonical_reasoning),
            "error_catalog": [error.to_mapping() for error in self.error_catalog],
            "split_keys": self.split_keys.to_mapping(),
        }
        if self.source_id is not None:
            result["source_id"] = self.source_id
        return result
