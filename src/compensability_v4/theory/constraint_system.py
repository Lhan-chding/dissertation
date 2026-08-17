"""Exact integer constraint semantics used by the v4 recovery design."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

World: TypeAlias = tuple[int, int, int, int]
MatrixRow: TypeAlias = tuple[int, int, int, int]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _index(value: object, name: str) -> int:
    result = _integer(value, name)
    if not 0 <= result < 4:
        raise ValueError(f"{name} must lie in [0, 3]")
    return result


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def validate_world(value: object, name: str = "world") -> World:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{name} must contain exactly four integers")
    world = tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(value))
    return world  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class KnownValueFact:
    index: int
    value: int
    fact_id: str = "known_value"

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _index(self.index, "index"))
        object.__setattr__(self, "value", _integer(self.value, "value"))
        object.__setattr__(self, "fact_id", _identifier(self.fact_id, "fact_id"))

    def matrix_row(self) -> tuple[MatrixRow, int]:
        row = tuple(1 if position == self.index else 0 for position in range(4))
        return row, self.value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PairSumFact:
    left_index: int
    right_index: int
    total: int
    fact_id: str = "pair_sum"

    def __post_init__(self) -> None:
        left = _index(self.left_index, "left_index")
        right = _index(self.right_index, "right_index")
        if left == right:
            raise ValueError("pair-sum indices must be distinct")
        object.__setattr__(self, "left_index", left)
        object.__setattr__(self, "right_index", right)
        object.__setattr__(self, "total", _integer(self.total, "total"))
        object.__setattr__(self, "fact_id", _identifier(self.fact_id, "fact_id"))

    def matrix_row(self) -> tuple[MatrixRow, int]:
        row = tuple(
            1 if position in {self.left_index, self.right_index} else 0
            for position in range(4)
        )
        return row, self.total  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ArithmeticProgressionFact:
    indices: tuple[int, int, int]
    fact_id: str = "arithmetic_progression"

    def __post_init__(self) -> None:
        if not isinstance(self.indices, Sequence) or isinstance(self.indices, (str, bytes)):
            raise TypeError("indices must be a sequence")
        if len(self.indices) != 3:
            raise ValueError("arithmetic-progression indices must contain exactly three entries")
        indices = tuple(_index(value, f"indices[{position}]") for position, value in enumerate(self.indices))
        if len(set(indices)) != 3:
            raise ValueError("arithmetic-progression indices must be distinct")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "fact_id", _identifier(self.fact_id, "fact_id"))

    def matrix_row(self) -> tuple[MatrixRow, int]:
        left, middle, right = self.indices
        row = tuple(
            1 if position in {left, right} else -2 if position == middle else 0
            for position in range(4)
        )
        return row, 0  # type: ignore[return-value]


ConstraintFact: TypeAlias = KnownValueFact | PairSumFact | ArithmeticProgressionFact


def _closed(mapping: Mapping[str, object], required: set[str], optional: set[str]) -> None:
    missing = required - set(mapping)
    unknown = set(mapping) - required - optional
    if missing:
        raise ValueError(f"missing fact fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unknown fact fields: {', '.join(sorted(unknown))}")


def fact_from_mapping(value: Mapping[str, object]) -> ConstraintFact:
    if not isinstance(value, Mapping):
        raise TypeError("fact must be a mapping")
    fact_type = value.get("type")
    fact_id = value.get("fact_id", fact_type)
    if fact_type == "known_value":
        _closed(value, {"type", "index", "value"}, {"fact_id"})
        return KnownValueFact(value["index"], value["value"], fact_id)  # type: ignore[arg-type]
    if fact_type == "pair_sum":
        _closed(value, {"type", "left_index", "right_index", "total"}, {"fact_id"})
        return PairSumFact(
            value["left_index"], value["right_index"], value["total"], fact_id  # type: ignore[arg-type]
        )
    if fact_type == "arithmetic_progression":
        _closed(value, {"type", "indices"}, {"fact_id"})
        return ArithmeticProgressionFact(value["indices"], fact_id)  # type: ignore[arg-type]
    raise ValueError(f"unsupported fact type: {fact_type!r}")


def normalize_fact(value: object) -> ConstraintFact:
    if isinstance(value, (KnownValueFact, PairSumFact, ArithmeticProgressionFact)):
        return value
    if isinstance(value, Mapping):
        return fact_from_mapping(value)
    raise TypeError("facts must be registered fact objects or mappings")


def facts_to_matrix(facts: Iterable[object]) -> tuple[tuple[MatrixRow, ...], tuple[int, ...]]:
    normalized = tuple(normalize_fact(fact) for fact in facts)
    rows_and_targets = tuple(fact.matrix_row() for fact in normalized)
    return (
        tuple(row for row, _target in rows_and_targets),
        tuple(target for _row, target in rows_and_targets),
    )


def satisfies_all_facts(world: Sequence[int], facts: Iterable[object]) -> bool:
    canonical_world = validate_world(world)
    matrix, targets = facts_to_matrix(facts)
    return all(sum(coefficient * value for coefficient, value in zip(row, canonical_world, strict=True)) == target for row, target in zip(matrix, targets, strict=True))


def constraint_residual(world: Sequence[int], facts: Iterable[object]) -> tuple[int, ...]:
    """Return ``b - A world`` as required by the inverse-recovery identity."""

    canonical_world = validate_world(world)
    matrix, targets = facts_to_matrix(facts)
    return tuple(
        target
        - sum(coefficient * value for coefficient, value in zip(row, canonical_world, strict=True))
        for row, target in zip(matrix, targets, strict=True)
    )
