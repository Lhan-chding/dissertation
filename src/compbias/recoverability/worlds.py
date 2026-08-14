"""Immutable semantic worlds and coherent counterfactual validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .compatibility import (
    ArithmeticProgressionConstraint,
    KnownValueConstraint,
    PairSumConstraint,
)
from .operators import Operation, apply_operation

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SPLIT = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_CHART_TYPES = frozenset({"grouped_bar", "line"})

VisibleConstraint = PairSumConstraint | KnownValueConstraint | ArithmeticProgressionConstraint
_FAMILY_CONSTRAINT = {
    "cross_series": PairSumConstraint,
    "duplicate_encoding": KnownValueConstraint,
    "trend": ArithmeticProgressionConstraint,
}


def _validate_values(values: object) -> tuple[int, int, int, int]:
    if not isinstance(values, tuple) or len(values) != 4:
        raise ValueError("values must be an exact four-integer tuple")
    if any(type(value) is not int for value in values):
        raise TypeError("values must contain exact integers")
    return values


@dataclass(frozen=True, slots=True)
class SemanticWorld:
    """A complete legal world; its canonical answer is always derived."""

    scene_id: str
    chart_type: str
    operation: Operation
    question_id: str
    values: tuple[int, int, int, int]
    redundancy_family: str
    split: str
    visible_constraints: tuple[VisibleConstraint, ...]
    value_domain: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or _IDENTIFIER.fullmatch(self.scene_id) is None:
            raise ValueError("scene_id must be a bounded safe identifier")
        if self.chart_type not in _CHART_TYPES:
            raise ValueError("chart_type is not registered")
        try:
            operation = Operation(self.operation)
        except (TypeError, ValueError) as error:
            raise ValueError("operation is not registered") from error
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.question_id, str) or _IDENTIFIER.fullmatch(self.question_id) is None:
            raise ValueError("question_id must be a bounded safe identifier")
        values = _validate_values(self.values)
        if (
            not isinstance(self.redundancy_family, str)
            or _IDENTIFIER.fullmatch(self.redundancy_family) is None
        ):
            raise ValueError("redundancy_family must be a bounded safe identifier")
        if self.redundancy_family not in _FAMILY_CONSTRAINT:
            raise ValueError("redundancy_family is not registered")
        if not isinstance(self.split, str) or _SPLIT.fullmatch(self.split) is None:
            raise ValueError("split must be a bounded lowercase identifier")
        if not isinstance(self.visible_constraints, tuple) or any(
            not isinstance(
                item,
                (PairSumConstraint, KnownValueConstraint, ArithmeticProgressionConstraint),
            )
            for item in self.visible_constraints
        ):
            raise TypeError("visible_constraints must contain registered constraints")
        identifiers = tuple(item.constraint_id for item in self.visible_constraints)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("constraint identifiers must be unique")
        if any(not constraint.accepts(values) for constraint in self.visible_constraints):
            raise ValueError("every visible constraint must hold in the complete world")
        if any(
            not isinstance(item, _FAMILY_CONSTRAINT[self.redundancy_family])
            for item in self.visible_constraints
        ):
            raise ValueError("constraint type must match the registered redundancy family")
        if not isinstance(self.value_domain, tuple) or not self.value_domain:
            raise ValueError("value_domain must be a non-empty integer tuple")
        if any(type(value) is not int for value in self.value_domain):
            raise TypeError("value_domain must contain exact integers")
        domain = tuple(sorted(set(self.value_domain)))
        if len(domain) != len(self.value_domain):
            raise ValueError("value_domain must not contain duplicates")
        if any(value not in domain for value in values):
            raise ValueError("world values must lie inside value_domain")
        object.__setattr__(self, "value_domain", domain)

    @property
    def gold_answer(self) -> int:
        return apply_operation(self.values, self.operation)


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    original: SemanticWorld
    counterfactual: SemanticWorld
    changed_value_indices: tuple[int, ...]
    changed_constraint_ids: tuple[str, ...]
    original_answer: int
    counterfactual_answer: int
    answer_delta: int


def _constraint_map(world: SemanticWorld) -> dict[str, VisibleConstraint]:
    return {item.constraint_id: item for item in world.visible_constraints}


def validate_counterfactual_pair(
    original: SemanticWorld,
    counterfactual: SemanticWorld,
    *,
    changed_value_indices: tuple[int, ...],
    changed_constraint_ids: tuple[str, ...],
) -> CounterfactualPair:
    """Bind a counterfactual cue to a second complete legal semantic world."""

    if not isinstance(original, SemanticWorld) or not isinstance(counterfactual, SemanticWorld):
        raise TypeError("counterfactual members must be SemanticWorld instances")
    if original.scene_id == counterfactual.scene_id:
        raise ValueError("counterfactual members must use distinct scene identifiers")
    if original.split != counterfactual.split:
        raise ValueError("counterfactual members must remain in the same split")
    if original.operation != counterfactual.operation:
        raise ValueError("counterfactual members must use the same operation")
    if original.question_id != counterfactual.question_id:
        raise ValueError("counterfactual members must use the same question semantics")
    if original.chart_type != counterfactual.chart_type:
        raise ValueError("counterfactual members must use the same chart type")
    if original.redundancy_family != counterfactual.redundancy_family:
        raise ValueError("counterfactual members must use the same redundancy family")
    if original.value_domain != counterfactual.value_domain:
        raise ValueError("counterfactual members must use the same finite value domain")
    if (
        not isinstance(changed_value_indices, tuple)
        or not changed_value_indices
        or any(type(index) is not int or not 0 <= index < 4 for index in changed_value_indices)
        or len(set(changed_value_indices)) != len(changed_value_indices)
    ):
        raise ValueError("changed_value_indices must be unique registered indices")
    if (
        not isinstance(changed_constraint_ids, tuple)
        or not changed_constraint_ids
        or any(
            not isinstance(identifier, str) or _IDENTIFIER.fullmatch(identifier) is None
            for identifier in changed_constraint_ids
        )
        or len(set(changed_constraint_ids)) != len(changed_constraint_ids)
    ):
        raise ValueError("changed_constraint_ids must be unique safe identifiers")

    actual_value_changes = tuple(
        index
        for index, (left, right) in enumerate(
            zip(original.values, counterfactual.values, strict=True)
        )
        if left != right
    )
    if set(actual_value_changes) != set(changed_value_indices):
        raise ValueError("counterfactual contains an undeclared value change")
    original_constraints = _constraint_map(original)
    counterfactual_constraints = _constraint_map(counterfactual)
    if set(original_constraints) != set(counterfactual_constraints):
        raise ValueError("counterfactual must preserve the constraint identifier set")
    actual_constraint_changes = {
        identifier
        for identifier in original_constraints
        if original_constraints[identifier] != counterfactual_constraints[identifier]
    }
    if actual_constraint_changes != set(changed_constraint_ids):
        raise ValueError("counterfactual contains an undeclared constraint change")
    original_answer = original.gold_answer
    counterfactual_answer = counterfactual.gold_answer
    if original_answer == counterfactual_answer:
        raise ValueError("counterfactual canonical answer must change")
    return CounterfactualPair(
        original=original,
        counterfactual=counterfactual,
        changed_value_indices=tuple(sorted(changed_value_indices)),
        changed_constraint_ids=tuple(sorted(changed_constraint_ids)),
        original_answer=original_answer,
        counterfactual_answer=counterfactual_answer,
        answer_delta=counterfactual_answer - original_answer,
    )
