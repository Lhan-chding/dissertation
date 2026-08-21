"""Exact and sampled checks for correction-bearing GRPO groups."""

from __future__ import annotations

import itertools
import random

import pytest
from compensability_v5.theory.grpo_signal import (
    answer_group_signal,
    correction_bearing_answer_signal,
    state_group_signal,
)

THEORY_TOLERANCE = 1e-8


def _enumerated_probability(
    probabilities: tuple[float, float, float], group_size: int, predicate: object
) -> float:
    categories = range(3)  # X, S, F
    total = 0.0
    for group in itertools.product(categories, repeat=group_size):
        mass = 1.0
        for category in group:
            mass *= probabilities[category]
        if predicate(group):  # type: ignore[operator]
            total += mass
    return total


@pytest.mark.parametrize("group_size", [1, 2, 4, 6])
def test_group_signal_formulas_match_exhaustive_category_enumeration(group_size: int) -> None:
    p_x, p_s = 0.17, 0.31
    p_f = 1.0 - p_x - p_s
    probabilities = (p_x, p_s, p_f)

    exact_state = _enumerated_probability(
        probabilities, group_size, lambda group: 0 in group and any(item != 0 for item in group)
    )
    answer = _enumerated_probability(
        probabilities, group_size, lambda group: any(item != 2 for item in group) and 2 in group
    )
    correction_bearing = _enumerated_probability(
        probabilities, group_size, lambda group: 0 in group and 2 in group
    )

    assert abs(state_group_signal(p_x, group_size) - exact_state) < THEORY_TOLERANCE
    assert abs(answer_group_signal(p_x, p_s, group_size) - answer) < THEORY_TOLERANCE
    assert (
        abs(correction_bearing_answer_signal(p_x, p_s, group_size) - correction_bearing)
        < THEORY_TOLERANCE
    )


def test_correction_bearing_formula_matches_seeded_monte_carlo() -> None:
    p_x, p_s, group_size = 0.13, 0.42, 8
    expected = correction_bearing_answer_signal(p_x, p_s, group_size)
    rng = random.Random(20260821)
    trials = 100_000
    successes = 0

    for _ in range(trials):
        group = []
        for _ in range(group_size):
            draw = rng.random()
            group.append("X" if draw < p_x else "S" if draw < p_x + p_s else "F")
        successes += "X" in group and "F" in group

    empirical = successes / trials
    assert abs(empirical - expected) < 0.005


def test_answer_variance_does_not_imply_a_correction_bearing_group() -> None:
    p_x, p_s, group_size = 0.0, 0.5, 8

    assert answer_group_signal(p_x, p_s, group_size) > 0.99
    assert correction_bearing_answer_signal(p_x, p_s, group_size) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("p_x", "p_s", "group_size"),
    [(-0.1, 0.2, 4), (0.8, 0.3, 4), (0.2, 0.3, 0), (True, 0.2, 4)],
)
def test_group_signal_rejects_invalid_probabilities(
    p_x: object, p_s: float, group_size: int
) -> None:
    with pytest.raises((TypeError, ValueError)):
        correction_bearing_answer_signal(p_x, p_s, group_size)  # type: ignore[arg-type]
