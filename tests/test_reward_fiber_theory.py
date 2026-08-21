from __future__ import annotations

import math

import numpy as np
import pytest

from compbias.v5.theory import (
    correction_alignment,
    correction_bearing_group_probability,
    exact_state_update,
    is_sparse_correction_identifiable,
    orbit_risk_upper_bound,
)


def test_sparse_identifiability_detects_colliding_scaled_columns() -> None:
    identifiable = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )
    ambiguous = np.array(
        [
            [1, 2],
            [0, 0],
        ],
        dtype=np.int64,
    )

    assert is_sparse_correction_identifiable(identifiable, deltas=(-1, 1), sparsity=1) is True
    assert is_sparse_correction_identifiable(ambiguous, deltas=(1, 2), sparsity=1) is False


def test_exact_state_update_matches_closed_form_kl_projection_inside_answer_fiber() -> None:
    reference = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    rewards = np.array([1.0, 1.0, 0.0], dtype=np.float64)

    selected = exact_state_update(reference, rewards, beta=0.7)

    assert selected.sum() == pytest.approx(1.0, abs=1e-15)
    assert math.log(selected[0] / selected[1]) == pytest.approx(math.log(0.2 / 0.3), abs=1e-12)


def test_correction_bearing_group_probability_matches_inclusion_exclusion() -> None:
    observed = correction_bearing_group_probability(0.2, 0.3, 4)
    expected = 1.0 - (1.0 - 0.2) ** 4 - (0.2 + 0.3) ** 4 + 0.3**4

    assert observed == pytest.approx(expected, abs=1e-12)


def test_correction_alignment_is_positive_for_exact_state_reward() -> None:
    fisher = np.diag([2.0, 4.0])
    gradient = np.array([1.0, -2.0], dtype=np.float64)

    observed = correction_alignment(gradient, gradient, fisher)
    expected = (gradient @ np.linalg.solve(fisher, gradient)) / math.sqrt(
        (gradient @ np.linalg.solve(fisher, gradient))
        * (gradient @ np.linalg.solve(fisher, gradient))
    )

    assert observed == pytest.approx(expected, abs=1e-12)
    assert observed > 0.0


def test_orbit_risk_upper_bound_adds_equivariance_defect() -> None:
    bound = orbit_risk_upper_bound(base_risk=0.24, equivariance_defect=0.11)

    assert bound == pytest.approx(0.35)


@pytest.mark.parametrize(
    ("p_x", "p_s", "k"),
    [
        (-0.1, 0.1, 2),
        (0.6, 0.5, 2),
        (0.2, 0.1, 0),
    ],
)
def test_correction_bearing_group_probability_rejects_invalid_inputs(
    p_x: float, p_s: float, k: int
) -> None:
    with pytest.raises((TypeError, ValueError)):
        correction_bearing_group_probability(p_x, p_s, k)
