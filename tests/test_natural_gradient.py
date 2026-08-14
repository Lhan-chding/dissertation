"""Categorical natural-policy-gradient contracts and mirror equivalence."""

from __future__ import annotations

import numpy as np
import pytest

from compbias.models.tabular_policy import CategoricalPolicy
from compbias.rl.mirror_descent import mirror_descent_step
from compbias.rl.natural_gradient import (
    natural_policy_gradient_step,
    optimize_natural_policy_gradient,
)


def test_categorical_natural_gradient_step_equals_entropy_mirror_step() -> None:
    reference = np.array([0.45, 0.25, 0.20, 0.10])
    rewards = np.array([0.1, 0.3, 0.7, 0.9])
    policy = CategoricalPolicy.from_probabilities([0.40, 0.30, 0.20, 0.10])

    natural = natural_policy_gradient_step(
        policy,
        rewards=rewards,
        reference_probs=reference,
        beta=0.8,
        step_size=0.25,
    )
    mirror = mirror_descent_step(
        policy,
        rewards=rewards,
        reference_probs=reference,
        beta=0.8,
        step_size=0.25,
    )

    np.testing.assert_allclose(natural.probabilities, mirror.probabilities, atol=1e-15)
    np.testing.assert_array_equal(policy.probabilities, [0.40, 0.30, 0.20, 0.10])


def test_natural_gradient_optimizer_converges_and_records_equivalence() -> None:
    result = optimize_natural_policy_gradient(
        [0.45, 0.25, 0.20, 0.10],
        [0.1, 0.3, 0.7, 0.9],
        beta=0.8,
        step_size=0.625,
        steps=30,
        seed=17,
    )

    assert result.metrics["algorithm"] == "categorical_natural_policy_gradient"
    assert result.metrics["mirror_equivalent"] is True
    assert result.metrics["final_l1_to_exact"] < 1e-8
    assert result.approximate is False


@pytest.mark.parametrize("step_size", [True, 0.0, -0.1, float("nan")])
def test_natural_gradient_rejects_invalid_step_size(step_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="step_size"):
        optimize_natural_policy_gradient(
            [0.5, 0.5],
            [0.0, 1.0],
            beta=1.0,
            step_size=step_size,  # type: ignore[arg-type]
            steps=2,
            seed=0,
        )
