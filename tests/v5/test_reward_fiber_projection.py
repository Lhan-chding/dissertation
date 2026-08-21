"""Exact KL-projection contracts for answer-reward fibers."""

from __future__ import annotations

import numpy as np
import pytest
from compensability_v5.theory.reward_fibers import (
    conditional_distribution,
    kl_reward_projection,
)

THEORY_TOLERANCE = 1e-8


def test_kl_projection_matches_closed_form_boltzmann_tilt() -> None:
    prior = np.asarray([0.08, 0.12, 0.30, 0.50], dtype=float)
    rewards = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=float)
    beta = 0.7

    projected = np.asarray(kl_reward_projection(prior, rewards, beta), dtype=float)
    expected_unnormalized = prior * np.exp(rewards / beta)
    expected = expected_unnormalized / expected_unnormalized.sum()

    assert abs(projected.sum() - 1.0) < THEORY_TOLERANCE
    assert np.max(np.abs(projected - expected)) < THEORY_TOLERANCE


def test_answer_fiber_conditional_distribution_is_invariant() -> None:
    prior = np.asarray([0.05, 0.15, 0.30, 0.50], dtype=float)
    fiber = (0, 1)
    projected = kl_reward_projection(prior, (1.0, 1.0, 0.0, 0.0), beta=0.4)

    before = np.asarray(conditional_distribution(prior, fiber), dtype=float)
    after = np.asarray(conditional_distribution(projected, fiber), dtype=float)

    assert np.max(np.abs(before - after)) < THEORY_TOLERANCE
    assert abs(projected[0] / projected[1] - prior[0] / prior[1]) < THEORY_TOLERANCE


def test_answer_reward_can_raise_fiber_mass_without_changing_exact_state_purity() -> None:
    prior = np.asarray([0.04, 0.16, 0.30, 0.50], dtype=float)
    projected = np.asarray(
        kl_reward_projection(prior, (1.0, 1.0, 0.0, 0.0), beta=0.25), dtype=float
    )
    prior_fiber_mass = prior[:2].sum()
    projected_fiber_mass = projected[:2].sum()
    prior_purity = prior[0] / prior_fiber_mass
    projected_purity = projected[0] / projected_fiber_mass

    assert projected_fiber_mass > prior_fiber_mass
    assert abs(projected_purity - prior_purity) < THEORY_TOLERANCE


@pytest.mark.parametrize(
    ("prior", "rewards", "beta"),
    [
        ((0.5, 0.5), (1.0,), 1.0),
        ((0.5, -0.5), (1.0, 0.0), 1.0),
        ((0.0, 0.0), (1.0, 0.0), 1.0),
        ((0.5, 0.5), (1.0, 0.0), 0.0),
    ],
)
def test_kl_projection_rejects_invalid_distributions(
    prior: tuple[float, ...], rewards: tuple[float, ...], beta: float
) -> None:
    with pytest.raises((TypeError, ValueError)):
        kl_reward_projection(prior, rewards, beta)
