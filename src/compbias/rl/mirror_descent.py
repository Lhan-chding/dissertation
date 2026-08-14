"""Exponentiated mirror descent for a finite KL-regularized policy."""

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


def mirror_descent_step(
    policy: CategoricalPolicy,
    *,
    rewards: ArrayLike,
    reference_probs: ArrayLike,
    beta: float,
    step_size: float,
) -> CategoricalPolicy:
    r"""Take one entropy-mirror step on ``E[r] - beta KL(p || reference)``."""

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
    logit_delta = np.zeros_like(current)
    logit_delta[active] = step_value * (reward_array[active] - beta_value * log_ratio[active])
    return policy.updated(logit_delta)


def optimize_mirror_descent(
    reference_probs: ArrayLike,
    rewards: ArrayLike,
    *,
    beta: float,
    step_size: float,
    steps: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> OptimizationResult:
    """Run deterministic mirror descent with an explicit reproducibility token.

    Mirror descent itself requires no samples.  Requiring exactly one of
    ``seed`` and ``rng`` keeps its experiment interface aligned with the
    stochastic optimizers without reading or advancing the supplied generator.
    """

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
        policy = mirror_descent_step(
            policy,
            rewards=reward_array,
            reference_probs=reference,
            beta=beta_value,
            step_size=step_value,
        )
        trajectory[step] = policy.probabilities

    final = policy.probabilities
    exact = exact_kl_projection(reference, reward_array, beta_value)
    metrics = {
        "algorithm": "mirror_descent",
        "final_l1_to_exact": float(np.sum(np.abs(final - exact))),
        "steps": step_count,
        "step_size": step_value,
    }
    return OptimizationResult(final, trajectory, metrics)


__all__ = ["mirror_descent_step", "optimize_mirror_descent"]
