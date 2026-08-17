"""Counterfactual validation against exact fact support."""

from __future__ import annotations

from compensability_v4.theory.candidate_space import unique_constraint_projection
from compensability_v4.theory.constraint_system import KnownValueFact, PairSumFact


def validate_counterfactual_world(
    *,
    original_truth: tuple[int, int, int, int],
    observed: tuple[int, int, int, int],
    counterfactual_world: tuple[int, int, int, int],
    counterfactual_facts: list[dict[str, object]],
    value_domain: range,
) -> None:
    if counterfactual_world == original_truth:
        raise ValueError("counterfactual world must be distinct")
    facts = []
    for index, fact in enumerate(counterfactual_facts):
        if fact["type"] == "known_value":
            facts.append(KnownValueFact(index=fact["index"], value=fact["value"], fact_id=f"k{index}"))
        elif fact["type"] == "pair_sum":
            facts.append(
                PairSumFact(
                    left_index=fact["left_index"],
                    right_index=fact["right_index"],
                    total=fact["total"],
                    fact_id=f"p{index}",
                )
            )
        else:
            raise ValueError("unsupported fact type")
    if unique_constraint_projection(observed, facts, value_domain) != counterfactual_world:
        raise ValueError("counterfactual world is not uniquely fact supported")
