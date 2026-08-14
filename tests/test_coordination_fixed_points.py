"""RED contracts for the 2x2 perception--reasoning coordination system."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from compbias.theory.coordination import (
    CoordinationParams,
    fixed_points,
    jacobian,
    reward,
    vector_field,
)


def _logit(value: float) -> float:
    return float(np.log(value / (1.0 - value)))


def test_coordination_params_are_immutable_and_validate_probabilities() -> None:
    params = CoordinationParams(delta=1.0, epsilon=0.5)

    with pytest.raises(FrozenInstanceError):
        params.delta = 2.0  # type: ignore[misc]

    for kwargs in (
        {"delta": 0.0, "epsilon": 1.0},
        {"delta": 1.0, "epsilon": -0.1},
        {"delta": 1.0, "epsilon": 1.0, "beta_p": -0.1},
        {"delta": 1.0, "epsilon": 1.0, "beta_q": -0.1},
        {"delta": 1.0, "epsilon": 1.0, "p_ref": 0.0},
        {"delta": 1.0, "epsilon": 1.0, "q_ref": 1.0},
    ):
        with pytest.raises(ValueError):
            CoordinationParams(**kwargs)


def test_reward_matches_the_reward_matrix_and_does_not_mutate_state() -> None:
    params = CoordinationParams(delta=0.2, epsilon=0.7)
    state = np.array([0.3, 0.8])
    original = state.copy()

    assert reward(1.0, 1.0, params) == pytest.approx(1.0)
    assert reward(0.0, 0.0, params) == pytest.approx(1.0)
    assert reward(1.0, 0.0, params) == pytest.approx(0.8)
    assert reward(0.0, 1.0, params) == pytest.approx(0.3)
    assert reward(state[0], state[1], params) == pytest.approx(0.596)
    np.testing.assert_array_equal(state, original)


def test_unregularized_vector_field_matches_replicator_equations() -> None:
    params = CoordinationParams(delta=0.2, epsilon=0.7)
    state = np.array([0.3, 0.8])
    original = state.copy()

    actual = vector_field(123.0, state, params)
    expected = np.array(
        [
            0.3 * 0.7 * (0.9 * 0.8 - 0.2),
            0.8 * 0.2 * (0.9 * 0.3 - 0.7),
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(state, original)
    assert not np.shares_memory(np.asarray(actual), state)


def test_kl_regularized_vector_field_uses_reference_policy_logits() -> None:
    params = CoordinationParams(
        delta=0.2,
        epsilon=0.7,
        beta_p=0.3,
        beta_q=0.4,
        p_ref=0.4,
        q_ref=0.6,
    )
    p, q = 0.3, 0.8

    actual = vector_field(0.0, np.array([p, q]), params)
    expected = np.array(
        [
            p
            * (1.0 - p)
            * (
                (params.delta + params.epsilon) * q
                - params.delta
                - params.beta_p * (_logit(p) - _logit(params.p_ref))
            ),
            q
            * (1.0 - q)
            * (
                (params.delta + params.epsilon) * p
                - params.epsilon
                - params.beta_q * (_logit(q) - _logit(params.q_ref))
            ),
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("state", [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
def test_kl_vector_field_is_finite_and_zero_at_policy_boundaries(
    state: tuple[float, float],
) -> None:
    params = CoordinationParams(
        delta=1.0,
        epsilon=1.0,
        beta_p=0.3,
        beta_q=0.4,
    )

    actual = np.asarray(vector_field(0.0, np.asarray(state), params))

    np.testing.assert_array_equal(actual, np.zeros(2))
    assert np.all(np.isfinite(actual))


def test_unregularized_jacobian_matches_the_analytic_matrix() -> None:
    params = CoordinationParams(delta=0.2, epsilon=0.7)
    state = np.array([0.3, 0.8])
    original = state.copy()

    actual = jacobian(state, params)
    expected = np.array(
        [
            [
                (1.0 - 2.0 * state[0])
                * ((params.delta + params.epsilon) * state[1] - params.delta),
                state[0] * (1.0 - state[0]) * (params.delta + params.epsilon),
            ],
            [
                state[1] * (1.0 - state[1]) * (params.delta + params.epsilon),
                (1.0 - 2.0 * state[1])
                * ((params.delta + params.epsilon) * state[0] - params.epsilon),
            ],
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(state, original)


def test_kl_jacobian_matches_a_central_finite_difference() -> None:
    params = CoordinationParams(
        delta=0.8,
        epsilon=0.6,
        beta_p=0.2,
        beta_q=0.35,
        p_ref=0.4,
        q_ref=0.7,
    )
    state = np.array([0.32, 0.71])
    step = 1e-6
    finite_difference = np.column_stack(
        [
            (
                np.asarray(vector_field(0.0, state + step * basis, params))
                - np.asarray(vector_field(0.0, state - step * basis, params))
            )
            / (2.0 * step)
            for basis in np.eye(2)
        ]
    )

    np.testing.assert_allclose(jacobian(state, params), finite_difference, atol=2e-8, rtol=2e-8)


def test_unregularized_fixed_points_and_stability_classification() -> None:
    params = CoordinationParams(delta=2.0, epsilon=1.0)
    points = fixed_points(params)

    assert isinstance(points, tuple)
    expected = (
        ((0.0, 0.0), "stable"),
        ((0.0, 1.0), "unstable"),
        ((1.0, 0.0), "unstable"),
        ((1.0, 1.0), "stable"),
        ((1.0 / 3.0, 2.0 / 3.0), "saddle"),
    )
    assert len(points) == len(expected)
    for expected_state, expected_stability in expected:
        matches = [
            point
            for point in points
            if np.allclose(point.state, expected_state, atol=1e-12, rtol=0.0)
        ]
        assert len(matches) == 1
        assert matches[0].stability == expected_stability

    for point in points:
        assert np.linalg.norm(vector_field(0.0, np.asarray(point.state), params)) < 1e-12
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            point.stability = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("beta", "expected_interior"),
    [
        (0.6, (((0.5, 0.5), "stable"),)),
        (0.5, (((0.5, 0.5), "critical"),)),
        (
            0.3,
            (
                (((1.0 - 0.9073323166453315) / 2.0,) * 2, "stable"),
                ((0.5, 0.5), "saddle"),
                (((1.0 + 0.9073323166453315) / 2.0,) * 2, "stable"),
            ),
        ),
    ],
)
def test_symmetric_kl_interior_fixed_point_classification(
    beta: float, expected_interior: tuple[tuple[tuple[float, float], str], ...]
) -> None:
    params = CoordinationParams(
        delta=1.0,
        epsilon=1.0,
        beta_p=beta,
        beta_q=beta,
    )
    interior = [
        point
        for point in fixed_points(params)
        if 0.0 < point.state[0] < 1.0 and 0.0 < point.state[1] < 1.0
    ]

    assert len(interior) == len(expected_interior)
    for expected_state, expected_stability in expected_interior:
        matches = [
            point
            for point in interior
            if np.allclose(point.state, expected_state, atol=1e-9, rtol=0.0)
        ]
        assert len(matches) == 1
        assert matches[0].stability == expected_stability


def test_fixed_point_jacobian_eigenvalue_signs_agree_with_classification() -> None:
    params = CoordinationParams(delta=2.0, epsilon=1.0)

    for point in fixed_points(params):
        real_parts = np.real(np.linalg.eigvals(jacobian(np.asarray(point.state), params)))
        if point.stability == "stable":
            assert np.all(real_parts < 0.0)
        elif point.stability == "unstable":
            assert np.all(real_parts > 0.0)
        elif point.stability == "saddle":
            assert np.min(real_parts) < 0.0 < np.max(real_parts)
        else:
            pytest.fail(f"unexpected classification: {point.stability}")


def test_outcome_reward_derivative_is_nonnegative_without_kl() -> None:
    params = CoordinationParams(delta=0.7, epsilon=1.2)
    rng = np.random.default_rng(20260814)

    for state in rng.uniform(1e-5, 1.0 - 1e-5, size=(1_000, 2)):
        p, q = state
        grad_reward = np.array(
            [
                (params.delta + params.epsilon) * q - params.delta,
                (params.delta + params.epsilon) * p - params.epsilon,
            ]
        )
        field = np.asarray(vector_field(0.0, state, params))
        derivative = float(grad_reward @ field)
        identity = p * (1.0 - p) * grad_reward[0] ** 2 + q * (1.0 - q) * grad_reward[1] ** 2

        assert derivative >= -1e-14
        assert derivative == pytest.approx(identity, abs=1e-12, rel=0.0)
