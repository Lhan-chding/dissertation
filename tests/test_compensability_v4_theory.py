"""RED contracts for the v4 exact constraint theory."""

from __future__ import annotations

import math
import random

import pytest

from compensability_v4.theory.candidate_space import (
    AmbiguousProjectionError,
    NoProjectionError,
    enumerate_one_edit_candidates,
    unique_constraint_projection,
)
from compensability_v4.theory.constraint_system import (
    ArithmeticProgressionFact,
    KnownValueFact,
    PairSumFact,
    constraint_residual,
    facts_to_matrix,
    satisfies_all_facts,
)
from compensability_v4.theory.policy_support import (
    informative_group_probability,
    mean_informative_group_rate,
)


def test_fact_matrix_rows_match_registered_semantics() -> None:
    facts = [
        KnownValueFact(index=2, value=7, fact_id="known_c"),
        PairSumFact(left_index=0, right_index=3, total=11, fact_id="sum_ad"),
        ArithmeticProgressionFact(indices=(0, 1, 2), fact_id="trend_abc"),
    ]

    matrix, targets = facts_to_matrix(facts)

    assert matrix == ((0, 0, 1, 0), (1, 0, 0, 1), (1, -2, 1, 0))
    assert targets == (7, 11, 0)
    assert satisfies_all_facts((4, 5, 6, 7), facts[1:]) is True


def test_one_edit_candidate_space_is_unique_sorted_and_includes_observation() -> None:
    observed = (4, 5, 6, 7)

    candidates = enumerate_one_edit_candidates(observed, range(4, 9))

    assert candidates == sorted(set(candidates))
    assert observed in candidates
    assert len(candidates) == 1 + 4 * 4
    assert all(
        sum(a != b for a, b in zip(observed, world, strict=True)) <= 1 for world in candidates
    )


def test_one_edit_candidate_space_repairs_at_most_one_unbounded_observation() -> None:
    candidates = enumerate_one_edit_candidates((1000, 4, 5, 9), range(2, 19))

    assert len(candidates) == 17
    assert all(world[1:] == (4, 5, 9) for world in candidates)
    assert all(value in range(2, 19) for world in candidates for value in world)
    assert enumerate_one_edit_candidates((1000, -1000, 5, 9), range(2, 19)) == []


def test_unique_projection_and_failure_modes_are_distinct() -> None:
    observed = (7, 4, 7, 3)
    facts = [
        KnownValueFact(index=1, value=4, fact_id="b"),
        KnownValueFact(index=2, value=7, fact_id="c"),
        KnownValueFact(index=3, value=3, fact_id="d"),
        PairSumFact(left_index=0, right_index=1, total=12, fact_id="ab"),
    ]

    assert unique_constraint_projection(observed, facts, range(1, 10)) == (8, 4, 7, 3)
    with pytest.raises(AmbiguousProjectionError):
        unique_constraint_projection(observed, [], range(1, 10))
    with pytest.raises(NoProjectionError):
        unique_constraint_projection(observed, [KnownValueFact(0, 99, "impossible")], range(1, 10))


def test_constraint_residual_matches_negative_error_column_identity() -> None:
    truth = (8, 4, 7, 3)
    observed = (7, 4, 7, 3)
    facts = [
        PairSumFact(0, 1, 12, "ab"),
        PairSumFact(0, 2, 15, "ac"),
        KnownValueFact(3, 3, "d"),
    ]
    matrix, _ = facts_to_matrix(facts)
    delta = observed[0] - truth[0]

    assert constraint_residual(observed, facts) == tuple(-delta * row[0] for row in matrix)


def test_informative_group_formula_matches_monte_carlo_and_mean() -> None:
    probability = informative_group_probability(0.17, 8)
    rng = random.Random(20260817)
    trials = 30_000
    empirical = (
        sum(0 < sum(rng.random() < 0.17 for _ in range(8)) < 8 for _ in range(trials)) / trials
    )

    assert math.isclose(probability, empirical, abs_tol=0.01)
    assert mean_informative_group_rate([0.0, 0.5, 1.0], 4) == pytest.approx(0.2916666667)


@pytest.mark.parametrize(("p", "k"), [(-0.1, 2), (1.1, 2), (0.5, 0), (True, 2)])
def test_informative_group_formula_rejects_invalid_inputs(p: object, k: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        informative_group_probability(p, k)  # type: ignore[arg-type]
