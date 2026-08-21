"""Finite-difference verification of local reward-correction alignment."""

from __future__ import annotations

import numpy as np
from compensability_v5.theory.correction_alignment import (
    alignment_coefficient,
    natural_gradient_direction,
)

THEORY_TOLERANCE = 1e-8


def _probabilities(theta: np.ndarray) -> np.ndarray:
    logits = np.asarray([theta[0], theta[1], 0.0], dtype=float)
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def _objective(theta: np.ndarray, reward: np.ndarray) -> float:
    return float(_probabilities(theta) @ reward)


def _gradient_and_fisher(theta: np.ndarray, reward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _probabilities(theta)
    scores = np.asarray(
        [
            [1.0 - probabilities[0], -probabilities[1]],
            [-probabilities[0], 1.0 - probabilities[1]],
            [-probabilities[0], -probabilities[1]],
        ]
    )
    gradient = ((probabilities * reward)[:, None] * scores).sum(axis=0)
    fisher = (probabilities[:, None, None] * scores[:, :, None] * scores[:, None, :]).sum(axis=0)
    return gradient, fisher


def test_natural_gradient_alignment_identity_matches_central_finite_difference() -> None:
    theta = np.asarray([0.2, -0.4], dtype=float)
    exact_reward = np.asarray([1.0, 0.0, 0.0], dtype=float)
    answer_reward = np.asarray([1.0, 1.0, 0.0], dtype=float)
    g_x, fisher = _gradient_and_fisher(theta, exact_reward)
    g_answer, _ = _gradient_and_fisher(theta, answer_reward)
    direction = np.asarray(natural_gradient_direction(g_answer, fisher), dtype=float)
    expected_directional_derivative = float(g_x @ direction)
    epsilon = 1e-5

    finite_difference = (
        _objective(theta + epsilon * direction, exact_reward)
        - _objective(theta - epsilon * direction, exact_reward)
    ) / (2.0 * epsilon)

    assert abs(finite_difference - expected_directional_derivative) < THEORY_TOLERANCE


def test_exact_state_reward_has_unit_self_alignment_and_nonnegative_change() -> None:
    theta = np.asarray([0.3, -0.2], dtype=float)
    exact_reward = np.asarray([1.0, 0.0, 0.0], dtype=float)
    g_x, fisher = _gradient_and_fisher(theta, exact_reward)
    direction = np.asarray(natural_gradient_direction(g_x, fisher), dtype=float)

    assert abs(alignment_coefficient(g_x, g_x, fisher) - 1.0) < THEORY_TOLERANCE
    assert float(g_x @ direction) >= 0.0


def test_answer_and_exact_rewards_can_have_negative_fisher_alignment() -> None:
    fisher = np.asarray([[2.0, 0.3], [0.3, 1.1]], dtype=float)
    g_x = np.asarray([1.0, 0.0], dtype=float)
    g_answer = np.asarray([-1.0, 0.2], dtype=float)

    coefficient = alignment_coefficient(g_x, g_answer, fisher)

    assert -1.0 <= coefficient < 0.0
