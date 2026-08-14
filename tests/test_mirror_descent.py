"""RED contracts for immutable tabular KL-regularized mirror descent."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from compbias.models.tabular_policy import CategoricalPolicy
from compbias.rl.exact_kl import exact_kl_projection
from compbias.rl.mirror_descent import mirror_descent_step, optimize_mirror_descent

REFERENCE = np.array([0.45, 0.25, 0.20, 0.10], dtype=np.float64)
REWARDS = np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float64)
BETA = 0.8


def _l1_to_exact(probabilities: np.ndarray) -> float:
    target = exact_kl_projection(REFERENCE, REWARDS, BETA)
    return float(np.sum(np.abs(probabilities - target)))


def test_single_mirror_step_returns_a_new_policy_without_mutating_inputs() -> None:
    reference = REFERENCE.copy()
    rewards = REWARDS.copy()
    policy = CategoricalPolicy.from_probabilities(reference)

    updated = mirror_descent_step(
        policy,
        rewards=rewards,
        reference_probs=reference,
        beta=BETA,
        step_size=0.25,
    )

    assert isinstance(updated, CategoricalPolicy)
    assert updated is not policy
    np.testing.assert_array_equal(policy.probabilities, REFERENCE)
    np.testing.assert_array_equal(reference, REFERENCE)
    np.testing.assert_array_equal(rewards, REWARDS)
    assert _l1_to_exact(updated.probabilities) < _l1_to_exact(policy.probabilities)


def test_optimizer_returns_probabilities_trajectory_and_metrics() -> None:
    result = optimize_mirror_descent(
        reference_probs=REFERENCE,
        rewards=REWARDS,
        beta=BETA,
        step_size=0.625,
        steps=30,
        seed=17,
    )
    trajectory = np.asarray(result.trajectory)

    assert result.probabilities.shape == REFERENCE.shape
    assert trajectory.shape == (31, REFERENCE.size)
    np.testing.assert_allclose(trajectory[0], REFERENCE, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(trajectory[-1], result.probabilities, atol=0.0)
    np.testing.assert_allclose(trajectory.sum(axis=1), 1.0, atol=1e-12)
    assert isinstance(result.metrics, Mapping)
    assert result.metrics["algorithm"] == "mirror_descent"
    assert result.metrics["final_l1_to_exact"] == pytest.approx(
        _l1_to_exact(result.probabilities), abs=1e-14
    )


def test_mirror_error_decreases_with_more_steps() -> None:
    short = optimize_mirror_descent(
        REFERENCE,
        REWARDS,
        beta=BETA,
        step_size=0.625,
        steps=2,
        seed=0,
    )
    long = optimize_mirror_descent(
        REFERENCE,
        REWARDS,
        beta=BETA,
        step_size=0.625,
        steps=30,
        seed=0,
    )

    assert _l1_to_exact(long.probabilities) < _l1_to_exact(short.probabilities)
    assert _l1_to_exact(long.probabilities) < 1e-8


def test_refining_an_oscillatory_step_size_reduces_mirror_error() -> None:
    coarse = optimize_mirror_descent(
        REFERENCE,
        REWARDS,
        beta=BETA,
        step_size=2.25,
        steps=15,
        seed=0,
    )
    refined = optimize_mirror_descent(
        REFERENCE,
        REWARDS,
        beta=BETA,
        step_size=0.625,
        steps=15,
        seed=0,
    )

    assert _l1_to_exact(refined.probabilities) < _l1_to_exact(coarse.probabilities)


def test_seed_and_equivalent_generator_are_reproducible() -> None:
    by_seed = optimize_mirror_descent(
        REFERENCE, REWARDS, beta=BETA, step_size=0.4, steps=10, seed=91
    )
    by_generator = optimize_mirror_descent(
        REFERENCE,
        REWARDS,
        beta=BETA,
        step_size=0.4,
        steps=10,
        rng=np.random.default_rng(91),
    )

    np.testing.assert_array_equal(by_seed.probabilities, by_generator.probabilities)
    np.testing.assert_array_equal(by_seed.trajectory, by_generator.trajectory)


@pytest.mark.parametrize(
    ("step_size", "steps"),
    [(0.0, 10), (-0.1, 10), (np.nan, 10), (0.1, 0), (0.1, -1), (0.1, 1.5)],
)
def test_mirror_optimizer_rejects_invalid_controls(step_size: float, steps: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        optimize_mirror_descent(
            REFERENCE,
            REWARDS,
            beta=BETA,
            step_size=step_size,
            steps=steps,  # type: ignore[arg-type]
            seed=0,
        )


def test_training_rejects_ambiguous_randomness_source() -> None:
    with pytest.raises(ValueError, match=r"(?i)(seed|rng|generator)"):
        optimize_mirror_descent(
            REFERENCE,
            REWARDS,
            beta=BETA,
            step_size=0.2,
            steps=5,
            seed=0,
            rng=np.random.default_rng(0),
        )
