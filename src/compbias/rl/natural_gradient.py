"""Natural policy gradient for a finite categorical KL objective.

For a categorical distribution, the Fisher-natural logit step is exactly the
negative-entropy mirror step after its action-independent baseline is removed.
This module exposes that Level-1 protocol explicitly rather than silently
counting mirror descent as a separate implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from compbias.models.tabular_policy import CategoricalPolicy
from compbias.rl import (
    OptimizationResult,
    _positive_float,
    _positive_int,
    _probability_array,
    _resolve_rng,
    _reward_array,
)
from compbias.rl.exact_kl import exact_kl_projection


def natural_policy_gradient_step(
    policy: CategoricalPolicy,
    *,
    rewards: ArrayLike,
    reference_probs: ArrayLike,
    beta: float,
    step_size: float,
) -> CategoricalPolicy:
    r"""Take one Fisher-natural step on ``E[r] - beta KL(p || reference)``."""

    if not isinstance(policy, CategoricalPolicy):
        raise TypeError("policy must be a CategoricalPolicy")
    current = policy.probabilities
    reference = _probability_array(reference_probs, name="reference_probs", ndim=1)
    reward_array = _reward_array(rewards, shape=current.shape)
    beta_value = _positive_float(beta, name="beta")
    step_value = _positive_float(step_size, name="step_size")
    if reference.shape != current.shape:
        raise ValueError("reference_probs must have one entry per policy action")
    if np.any((current > 0.0) & (reference == 0.0)):
        raise ValueError("policy support must be contained in reference_probs support")

    active = current > 0.0
    log_ratio = np.zeros_like(current)
    log_ratio[active] = np.log(current[active]) - np.log(reference[active])
    # The derivative of KL includes an additional constant one.  The
    # categorical Fisher inverse is defined modulo action-independent logits,
    # so that baseline cancels exactly in the softmax update.
    natural_logit_delta = np.zeros_like(current)
    natural_logit_delta[active] = step_value * (
        reward_array[active] - beta_value * log_ratio[active]
    )
    return policy.updated(natural_logit_delta)


def optimize_natural_policy_gradient(
    reference_probs: ArrayLike,
    rewards: ArrayLike,
    *,
    beta: float,
    step_size: float,
    steps: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> OptimizationResult:
    """Run deterministic categorical NPG with an explicit RNG token."""

    reference = _probability_array(reference_probs, name="reference_probs", ndim=1)
    reward_array = _reward_array(rewards, shape=reference.shape)
    beta_value = _positive_float(beta, name="beta")
    step_value = _positive_float(step_size, name="step_size")
    step_count = _positive_int(steps, name="steps")
    _resolve_rng(seed=seed, rng=rng)

    policy = CategoricalPolicy.from_probabilities(reference)
    trajectory = np.empty((step_count + 1, reference.size), dtype=np.float64)
    trajectory[0] = policy.probabilities
    for step in range(1, step_count + 1):
        policy = natural_policy_gradient_step(
            policy,
            rewards=reward_array,
            reference_probs=reference,
            beta=beta_value,
            step_size=step_value,
        )
        trajectory[step] = policy.probabilities

    final = policy.probabilities
    exact = np.asarray(exact_kl_projection(reference, reward_array, beta_value))
    return OptimizationResult(
        final,
        trajectory,
        {
            "algorithm": "categorical_natural_policy_gradient",
            "mirror_equivalent": True,
            "final_l1_to_exact": float(np.sum(np.abs(final - exact))),
            "steps": step_count,
            "step_size": step_value,
        },
    )


__all__ = ["natural_policy_gradient_step", "optimize_natural_policy_gradient"]
