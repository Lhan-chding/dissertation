"""Pure, authoritative solvers for every CVA-World task family."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

from .schema import BAR_CHART_INDICES, BAR_CHART_OPERATIONS, CVASample, TaskFamily


def _family(value: TaskFamily | str) -> TaskFamily:
    try:
        return TaskFamily(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported task_family: {value!r}") from error


def _number(mapping: Mapping[str, object], key: str, *, nonnegative: bool = False) -> Real:
    try:
        value = mapping[key]
    except KeyError as error:
        raise KeyError(f"missing numeric field: {key}") from error
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{key} must be numeric")
    if nonnegative and value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _template(question: Mapping[str, object], expected: str) -> None:
    if question.get("template") != expected:
        raise ValueError(f"question template must be {expected!r}")


def _bar_inputs(
    scene: Mapping[str, object], question: Mapping[str, object]
) -> tuple[tuple[Real, ...], tuple[int, ...], str]:
    _template(question, "aggregate")
    bars = scene.get("bars")
    if not isinstance(bars, Sequence) or isinstance(bars, (str, bytes)) or len(bars) != 4:
        raise ValueError("bars must contain exactly four numeric heights")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in bars):
        raise TypeError("bars must contain only numbers")
    if any(not math.isfinite(float(value)) or value < 0 for value in bars):
        raise ValueError("bars must contain finite non-negative heights")

    indices_value = question.get("indices")
    if not isinstance(indices_value, Sequence) or isinstance(indices_value, (str, bytes)):
        raise TypeError("indices must be a sequence")
    indices: list[int] = []
    for index in indices_value:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("indices must contain integers")
        if not 0 <= index < len(bars):
            raise ValueError("bar index out of range")
        indices.append(index)
    if tuple(indices) != BAR_CHART_INDICES:
        raise ValueError("indices must equal the registered pair [0, 1]")

    operation = question.get("operation")
    if not isinstance(operation, str) or operation not in BAR_CHART_OPERATIONS:
        raise ValueError(f"unsupported aggregate operation: {operation!r}")
    return tuple(bars), tuple(indices), operation


def solve(
    scene: Mapping[str, object],
    question: Mapping[str, object],
    task_family: TaskFamily | str,
) -> object:
    """Compute the canonical answer without changing either input mapping."""

    if not isinstance(scene, Mapping) or not isinstance(question, Mapping):
        raise TypeError("scene and question must be mappings")
    family = _family(task_family)

    if family is TaskFamily.DIGIT_OFFSET:
        _template(question, "add_constant")
        return _number(scene, "value") + _number(question, "operand")

    if family is TaskFamily.COUNT_TRANSFORM:
        _template(question, "affine_transform")
        count = _number(scene, "count", nonnegative=True)
        return count * _number(question, "scale") + _number(question, "offset")

    if family is TaskFamily.GAUGE_CALIBRATION:
        _template(question, "calibrate")
        reading = _number(scene, "reading")
        return reading * _number(question, "scale") + _number(question, "offset")

    if family is TaskFamily.BAR_CHART_AGGREGATE:
        bars, indices, operation = _bar_inputs(scene, question)
        selected = tuple(bars[index] for index in indices)
        if operation == "sum":
            return sum(selected)
        if operation == "difference":
            return selected[0] - selected[1]
        denominator = selected[1]
        if denominator == 0:
            raise ValueError("ratio denominator must be nonzero")
        return selected[0] / denominator

    _template(question, "relation_lookup")
    relation = scene.get("relation")
    if not isinstance(relation, str) or not relation:
        raise ValueError("relation must be a non-empty string")
    rule = question.get("rule")
    if not isinstance(rule, Mapping):
        raise TypeError("rule must be a mapping")
    if relation not in rule:
        raise KeyError(f"relation {relation!r} is absent from rule")
    return rule[relation]


def canonical_reasoning(
    scene: Mapping[str, object],
    question: Mapping[str, object],
    task_family: TaskFamily | str,
) -> dict[str, object]:
    """Return the registered reasoning action paired with :func:`solve`."""

    family = _family(task_family)
    answer = solve(scene, question, family)
    if family is TaskFamily.DIGIT_OFFSET:
        return {"operation": "add", "operand": question["operand"]}
    if family in {TaskFamily.COUNT_TRANSFORM, TaskFamily.GAUGE_CALIBRATION}:
        return {
            "operation": "affine",
            "scale": question["scale"],
            "offset": question["offset"],
        }
    if family is TaskFamily.BAR_CHART_AGGREGATE:
        return {"operation": question["operation"], "indices": list(question["indices"])}
    return {"operation": "lookup", "relation": scene["relation"], "result": answer}


@dataclass(frozen=True)
class SolverResult:
    """Self-check result for a stored generated sample."""

    answer: object
    reasoning: Mapping[str, object]
    is_consistent: bool


def solve_sample(sample: CVASample) -> SolverResult:
    """Recompute a sample and reject stale generated labels immediately."""

    if not isinstance(sample, CVASample):
        raise TypeError("sample must be a CVASample")
    answer = solve(sample.scene, sample.question, sample.task_family)
    reasoning = canonical_reasoning(sample.scene, sample.question, sample.task_family)
    if answer != sample.canonical_answer:
        raise ValueError(
            f"canonical_answer is stale: stored {sample.canonical_answer!r}, computed {answer!r}"
        )
    stored_reasoning = sample.to_mapping()["canonical_reasoning"]
    if reasoning != stored_reasoning:
        raise ValueError("canonical_reasoning is stale")
    return SolverResult(answer=answer, reasoning=sample.canonical_reasoning, is_consistent=True)
