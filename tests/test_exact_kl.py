"""RED contracts for the Level-0 finite-distribution KL optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from compbias.rl.exact_kl import exact_kl_projection


def _direct_projection(reference: np.ndarray, rewards: np.ndarray, beta: float) -> np.ndarray:
    log_weights = np.full(reference.shape, -np.inf, dtype=np.float64)
    support = reference > 0.0
    log_weights[support] = np.log(reference[support]) + rewards[support] / beta
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / weights.sum()


def test_exact_kl_matches_joint_trajectory_formula_to_experiment_a_tolerance() -> None:
    reference = np.array(
        [[0.18, 0.12, 0.10], [0.08, 0.22, 0.30]],
        dtype=np.float64,
    )
    rewards = np.array(
        [[0.0, 0.25, 1.0], [0.7, 0.4, 0.9]],
        dtype=np.float64,
    )
    reference_before = reference.copy()
    rewards_before = rewards.copy()

    observed = exact_kl_projection(reference, rewards, beta=0.65)
    predicted = _direct_projection(reference, rewards, beta=0.65)

    assert observed.shape == reference.shape
    assert np.sum(np.abs(observed - predicted)) < 1e-10
    assert observed.sum() == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(rewards, rewards_before)


def test_exact_kl_respects_support_and_pairwise_odds_identity() -> None:
    reference = np.array([0.0, 0.2, 0.3, 0.5], dtype=np.float64)
    rewards = np.array([1.0, 0.1, 0.4, 0.9], dtype=np.float64)
    beta = 0.7

    selected = exact_kl_projection(reference, rewards, beta)

    assert selected[0] == pytest.approx(0.0, abs=0.0)
    for left, right in ((1, 2), (1, 3), (2, 3)):
        observed = np.log(selected[left] / selected[right])
        predicted = (
            np.log(reference[left] / reference[right]) + (rewards[left] - rewards[right]) / beta
        )
        assert abs(observed - predicted) < 1e-10


def test_exact_kl_is_invariant_to_common_reward_offsets() -> None:
    reference = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    rewards = np.array([-2.0, 0.0, 3.0], dtype=np.float64)

    original = exact_kl_projection(reference, rewards, beta=0.4)
    shifted = exact_kl_projection(reference, rewards + 1e9, beta=0.4)

    np.testing.assert_allclose(shifted, original, rtol=0.0, atol=1e-7)


def test_exact_kl_uses_log_space_for_extreme_valid_tilts() -> None:
    selected = exact_kl_projection(
        np.array([0.999, 0.001], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        beta=1e-8,
    )

    assert np.all(np.isfinite(selected))
    assert selected.sum() == pytest.approx(1.0, abs=1e-15)
    assert selected[1] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("beta", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_exact_kl_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match=r"(?i)beta"):
        exact_kl_projection(np.array([0.5, 0.5]), np.array([0.0, 1.0]), beta)


@pytest.mark.parametrize(
    ("reference", "rewards"),
    [
        (np.array([0.2, 0.2]), np.array([0.0, 1.0])),
        (np.array([0.5, -0.1, 0.6]), np.array([0.0, 0.5, 1.0])),
        (np.array([0.5, np.nan, 0.5]), np.array([0.0, 0.5, 1.0])),
        (np.array([0.5, 0.5]), np.array([0.0, np.inf])),
        (np.array([0.5, 0.5]), np.array([0.0, 0.5, 1.0])),
    ],
)
def test_exact_kl_rejects_invalid_finite_tables(reference: np.ndarray, rewards: np.ndarray) -> None:
    with pytest.raises(ValueError):
        exact_kl_projection(reference, rewards, beta=0.5)
