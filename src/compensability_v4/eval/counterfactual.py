"""Validity and compliance checks for fact counterfactuals."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from compensability_v4.theory.candidate_space import unique_constraint_projection
from compensability_v4.theory.constraint_system import validate_world


def validate_counterfactual_world(
    *,
    original_truth: Sequence[int],
    observed: Sequence[int],
    counterfactual_world: Sequence[int],
    counterfactual_facts: Iterable[object],
    value_domain: Iterable[int],
) -> None:
    original = validate_world(original_truth, "original_truth")
    counterfactual = validate_world(counterfactual_world, "counterfactual_world")
    canonical_observed = validate_world(observed, "observed")
    if counterfactual == original:
        raise ValueError("counterfactual world must be distinct from original truth")
    facts = tuple(counterfactual_facts)
    projected = unique_constraint_projection(canonical_observed, facts, value_domain)
    if projected != counterfactual:
        raise ValueError("counterfactual facts do not uniquely support counterfactual world")


def counterfactual_compliance(
    predictions: Iterable[Sequence[int]], counterfactual_worlds: Iterable[Sequence[int]]
) -> float:
    prediction_tuple = tuple(validate_world(value, "prediction") for value in predictions)
    world_tuple = tuple(
        validate_world(value, "counterfactual_world") for value in counterfactual_worlds
    )
    if not prediction_tuple or len(prediction_tuple) != len(world_tuple):
        raise ValueError("predictions and counterfactual_worlds must be non-empty and paired")
    paired = zip(prediction_tuple, world_tuple, strict=True)
    return sum(prediction == world for prediction, world in paired) / len(prediction_tuple)
