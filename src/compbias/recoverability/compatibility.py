"""Finite compatibility-set recoverability using Stage-2-visible fields only."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal, Protocol

from .operators import Operation, apply_operation

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded safe identifier")
    return value


def _index(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value < 4:
        raise ValueError(f"{label} must be an integer index from 0 to 3")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


class VisibleConstraint(Protocol):
    constraint_id: str

    def accepts(self, values: tuple[int, int, int, int]) -> bool: ...


@dataclass(frozen=True, slots=True)
class PairSumConstraint:
    constraint_id: str
    left_index: int
    right_index: int
    total: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_id", _identifier(self.constraint_id, "constraint_id"))
        object.__setattr__(self, "left_index", _index(self.left_index, "left_index"))
        object.__setattr__(self, "right_index", _index(self.right_index, "right_index"))
        if self.left_index == self.right_index:
            raise ValueError("pair-sum indices must be distinct")
        object.__setattr__(self, "total", _integer(self.total, "total"))

    def accepts(self, values: tuple[int, int, int, int]) -> bool:
        return values[self.left_index] + values[self.right_index] == self.total


@dataclass(frozen=True, slots=True)
class KnownValueConstraint:
    constraint_id: str
    index: int
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_id", _identifier(self.constraint_id, "constraint_id"))
        object.__setattr__(self, "index", _index(self.index, "index"))
        object.__setattr__(self, "value", _integer(self.value, "value"))

    def accepts(self, values: tuple[int, int, int, int]) -> bool:
        return values[self.index] == self.value


@dataclass(frozen=True, slots=True)
class ArithmeticProgressionConstraint:
    constraint_id: str
    indices: tuple[int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_id", _identifier(self.constraint_id, "constraint_id"))
        if not isinstance(self.indices, tuple) or len(self.indices) != 3:
            raise ValueError("arithmetic-progression indices must contain three positions")
        indices = tuple(_index(value, "progression index") for value in self.indices)
        if len(set(indices)) != 3:
            raise ValueError("arithmetic-progression indices must be distinct")
        object.__setattr__(self, "indices", indices)

    def accepts(self, values: tuple[int, int, int, int]) -> bool:
        left, middle, right = (values[index] for index in self.indices)
        return 2 * middle == left + right


@dataclass(frozen=True, slots=True)
class CompatibilityQuery:
    """A closed operational query with no gold scene, answer, or error position."""

    observed_values: tuple[int, int, int, int]
    operation: Operation
    constraints: tuple[
        PairSumConstraint | KnownValueConstraint | ArithmeticProgressionConstraint, ...
    ]
    value_domain: tuple[int, ...]
    max_mismatches: int

    def __post_init__(self) -> None:
        if not isinstance(self.observed_values, tuple) or len(self.observed_values) != 4:
            raise ValueError("observed_values must be an exact four-integer tuple")
        if any(type(value) is not int for value in self.observed_values):
            raise TypeError("observed_values must contain exact integers")
        try:
            operation = Operation(self.operation)
        except (TypeError, ValueError) as error:
            raise ValueError("operation is not registered") from error
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.constraints, tuple) or any(
            not isinstance(
                item,
                (PairSumConstraint, KnownValueConstraint, ArithmeticProgressionConstraint),
            )
            for item in self.constraints
        ):
            raise TypeError("constraints must be a tuple of registered visible constraints")
        identifiers = tuple(item.constraint_id for item in self.constraints)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("constraint identifiers must be unique")
        if not isinstance(self.value_domain, tuple) or not self.value_domain:
            raise ValueError("value_domain must be a non-empty integer tuple")
        if any(type(value) is not int for value in self.value_domain):
            raise TypeError("value_domain must contain exact integers")
        domain = tuple(sorted(set(self.value_domain)))
        if len(domain) != len(self.value_domain):
            raise ValueError("value_domain must not contain duplicates")
        if any(value not in domain for value in self.observed_values):
            raise ValueError("observed_values must lie inside value_domain")
        object.__setattr__(self, "value_domain", domain)
        if type(self.max_mismatches) is not int or not 0 <= self.max_mismatches <= 4:
            raise ValueError("max_mismatches must be an integer from 0 to 4")


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    status: Literal["ok", "inconsistent"]
    compatible_values: tuple[tuple[int, int, int, int], ...]
    compatible_answers: tuple[int, ...]
    exactly_recoverable: bool
    bayes_ceiling: float


def _mismatch_count(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(first != second for first, second in zip(left, right, strict=True))


def analyze_compatibility(query: CompatibilityQuery) -> CompatibilityReport:
    """Enumerate worlds consistent with the declared visible-noise interface."""

    if not isinstance(query, CompatibilityQuery):
        raise TypeError("query must be a CompatibilityQuery")
    compatible: list[tuple[int, int, int, int]] = []
    for raw_values in product(query.value_domain, repeat=4):
        values = raw_values[0], raw_values[1], raw_values[2], raw_values[3]
        if _mismatch_count(values, query.observed_values) > query.max_mismatches:
            continue
        if all(constraint.accepts(values) for constraint in query.constraints):
            compatible.append(values)
    if not compatible:
        return CompatibilityReport(
            status="inconsistent",
            compatible_values=(),
            compatible_answers=(),
            exactly_recoverable=False,
            bayes_ceiling=0.0,
        )
    counts = Counter(apply_operation(values, query.operation) for values in compatible)
    answers = tuple(sorted(counts))
    return CompatibilityReport(
        status="ok",
        compatible_values=tuple(compatible),
        compatible_answers=answers,
        exactly_recoverable=len(answers) == 1,
        bayes_ceiling=max(counts.values()) / len(compatible),
    )
