"""RED integration and plotting-data contracts for coordination dynamics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from compbias.theory.coordination import (
    CoordinationParams,
    basin_map,
    phase_portrait,
    reward,
    vector_field,
)
from scipy.integrate import solve_ivp


def _integrate(
    initial: tuple[float, float], params: CoordinationParams, horizon: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    solution = solve_ivp(
        vector_field,
        (0.0, horizon),
        np.asarray(initial, dtype=float),
        args=(params,),
        rtol=1e-10,
        atol=1e-12,
        max_step=0.02,
    )
    assert solution.success, solution.message
    return solution.t, solution.y.T


@pytest.mark.parametrize(
    ("initial", "expected"),
    [
        ((0.7, 0.4), (1.0, 1.0)),
        ((0.6, 0.3), (0.0, 0.0)),
        ((0.7, 0.3), (0.5, 0.5)),
    ],
    ids=["truthful-basin", "compensatory-basin", "separatrix"],
)
def test_symmetric_coordination_fixed_basin_fixtures(
    initial: tuple[float, float], expected: tuple[float, float]
) -> None:
    params = CoordinationParams(delta=1.0, epsilon=1.0)

    _, trajectory = _integrate(initial, params)

    np.testing.assert_allclose(trajectory[-1], expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("initial", [(0.7, 0.4), (0.6, 0.3), (0.7, 0.3)])
def test_outcome_reward_is_monotone_along_each_fixed_fixture(
    initial: tuple[float, float],
) -> None:
    params = CoordinationParams(delta=1.0, epsilon=1.0)

    _, trajectory = _integrate(initial, params)
    rewards = np.array([reward(p, q, params) for p, q in trajectory])

    assert np.min(np.diff(rewards)) >= -1e-10
    assert rewards[-1] >= rewards[0]


def test_symmetric_separatrix_is_an_invariant_manifold() -> None:
    params = CoordinationParams(delta=1.0, epsilon=1.0)

    for p in np.linspace(0.05, 0.95, 19):
        state = np.array([p, 1.0 - p])
        field = np.asarray(vector_field(0.0, state, params))
        assert field.sum() == pytest.approx(0.0, abs=1e-12, rel=0.0)


def test_phase_portrait_returns_mesh_aligned_vector_field_without_aliasing() -> None:
    params = CoordinationParams(delta=1.0, epsilon=1.0)
    p_values = np.array([0.0, 0.25, 0.75, 1.0])
    q_values = np.array([0.0, 0.4, 1.0])
    original_p = p_values.copy()
    original_q = q_values.copy()

    portrait = phase_portrait(p_values, q_values, params)
    expected_p, expected_q = np.meshgrid(p_values, q_values, indexing="xy")
    expected_dp = expected_p * (1.0 - expected_p) * (2.0 * expected_q - 1.0)
    expected_dq = expected_q * (1.0 - expected_q) * (2.0 * expected_p - 1.0)

    np.testing.assert_array_equal(portrait.p, expected_p)
    np.testing.assert_array_equal(portrait.q, expected_q)
    np.testing.assert_allclose(portrait.dp, expected_dp, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(portrait.dq, expected_dq, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(p_values, original_p)
    np.testing.assert_array_equal(q_values, original_q)
    assert not np.shares_memory(portrait.p, p_values)
    assert not np.shares_memory(portrait.q, q_values)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        portrait.dp = np.zeros_like(portrait.dp)  # type: ignore[misc]


def test_phase_portrait_supports_kl_and_is_finite_on_closed_unit_square() -> None:
    params = CoordinationParams(
        delta=1.0,
        epsilon=1.0,
        beta_p=0.3,
        beta_q=0.3,
    )

    portrait = phase_portrait(np.linspace(0.0, 1.0, 7), np.linspace(0.0, 1.0, 9), params)

    assert portrait.dp.shape == (9, 7)
    assert portrait.dq.shape == (9, 7)
    assert np.all(np.isfinite(portrait.dp))
    assert np.all(np.isfinite(portrait.dq))


def test_basin_map_matches_analytic_symmetric_boundary_and_fixed_labels() -> None:
    params = CoordinationParams(delta=1.0, epsilon=1.0)
    p_values = np.array([0.2, 0.5, 0.8])
    q_values = np.array([0.2, 0.5, 0.8])
    original_p = p_values.copy()
    original_q = q_values.copy()

    basins = basin_map(
        p_values,
        q_values,
        params,
        horizon=30.0,
        separatrix_tolerance=1e-10,
    )
    expected = np.array(
        [
            ["compensatory", "compensatory", "separatrix"],
            ["compensatory", "separatrix", "truthful"],
            ["separatrix", "truthful", "truthful"],
        ]
    )

    np.testing.assert_array_equal(basins.labels, expected)
    assert set(np.unique(basins.labels)) == {
        "truthful",
        "compensatory",
        "separatrix",
    }
    np.testing.assert_array_equal(p_values, original_p)
    np.testing.assert_array_equal(q_values, original_q)
    assert not np.shares_memory(basins.p, p_values)
    assert not np.shares_memory(basins.q, q_values)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        basins.labels = expected.copy()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("initial", "expected_label"),
    [
        ((0.7, 0.4), "truthful"),
        ((0.6, 0.3), "compensatory"),
        ((0.7, 0.3), "separatrix"),
    ],
)
def test_basin_map_contains_the_three_fixed_fixtures(
    initial: tuple[float, float], expected_label: str
) -> None:
    params = CoordinationParams(delta=1.0, epsilon=1.0)

    basins = basin_map(
        np.array([initial[0]]),
        np.array([initial[1]]),
        params,
        horizon=30.0,
        separatrix_tolerance=1e-10,
    )

    assert basins.labels.shape == (1, 1)
    assert basins.labels[0, 0] == expected_label
