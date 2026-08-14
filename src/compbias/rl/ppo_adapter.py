"""Clipped categorical PPO diagnostic for the finite-policy experiment.

This module intentionally implements only the tabular, outcome-only analogue
needed to compare approximate policy-gradient updates with the exact KL oracle.
It is not a neural PPO trainer and is always labelled approximate.
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
)
from compbias.rl.reinforce import (
    JointOptimizationResult,
    _finalize_result,
    _forward_kl,
    _marginal_path,
    _outcome_evidence,
    _resolve_reward_contract,
    _sample_training_values,
    _selection_diagnostics,
)


def _clip_ratio(value: float) -> float:
    ratio = _positive_float(value, name="clip_ratio")
    if ratio >= 1.0:
        raise ValueError("clip_ratio must be smaller than one")
    return ratio


def train_ppo_like(
    reference_probs: ArrayLike,
    rewards: ArrayLike | None = None,
    *,
    compensability: ArrayLike | None = None,
    unconstrained_joint_trajectory_diagnostic: bool = False,
    beta: float,
    learning_rate: float,
    steps: int,
    batch_size: int,
    clip_ratio: float,
    epochs_per_batch: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> OptimizationResult | JointOptimizationResult:
    """Run clipped importance-ratio updates on a categorical policy.

    Each outer step samples from a frozen behavior policy.  Inner epochs use
    the standard clipped-surrogate active region and an exact categorical-logit
    gradient.  The KL penalty is evaluated against the common reference policy,
    matching the other finite-policy diagnostics.
    """

    reference = _probability_array(reference_probs, name="reference_probs", ndim=1)
    beta_value = _positive_float(beta, name="beta")
    contract = _resolve_reward_contract(
        reference,
        rewards,
        compensability,
        beta_value,
        unconstrained_joint_trajectory_diagnostic=(unconstrained_joint_trajectory_diagnostic),
    )
    rate = _positive_float(learning_rate, name="learning_rate")
    step_count = _positive_int(steps, name="steps")
    batch_count = _positive_int(batch_size, name="batch_size")
    epoch_count = _positive_int(epochs_per_batch, name="epochs_per_batch")
    clip_value = _clip_ratio(clip_ratio)
    generator = _resolve_rng(seed=seed, rng=rng)

    policy_reference = contract.policy_reference
    policy = CategoricalPolicy.from_probabilities(policy_reference)
    trajectory = np.empty((step_count + 1, policy_reference.size), dtype=np.float64)
    trajectory[0] = policy.probabilities
    update_distances = np.empty(step_count, dtype=np.float64)
    clip_fractions: list[float] = []
    outcome_draws_by_error = np.zeros(contract.error_count, dtype=np.int64)
    outcome_successes_by_error = np.zeros(contract.error_count, dtype=np.int64)

    for step in range(1, step_count + 1):
        old = policy.probabilities
        actions = policy.sample(size=batch_count, rng=generator)
        support = old > 0.0
        log_ratio_to_reference = np.zeros_like(old)
        log_ratio_to_reference[support] = np.log(old[support]) - np.log(policy_reference[support])
        sampled_values, draws, successes = _sample_training_values(
            actions,
            contract,
            generator,
        )
        outcome_draws_by_error += draws
        outcome_successes_by_error += successes
        regularized_samples = sampled_values - beta_value * log_ratio_to_reference[actions]
        baseline = (
            float(np.mean(regularized_samples))
            if contract.outcome_probabilities is not None
            else float(old @ (contract.training_rewards - beta_value * log_ratio_to_reference))
        )
        advantages = regularized_samples - baseline

        for _epoch in range(epoch_count):
            current = policy.probabilities
            importance = current[actions] / old[actions]
            active = np.where(
                advantages >= 0.0,
                importance < 1.0 + clip_value,
                importance > 1.0 - clip_value,
            )
            weights = np.where(active, advantages * importance, 0.0)
            action_gradient = (
                np.bincount(
                    actions,
                    weights=weights,
                    minlength=policy_reference.size,
                )
                / batch_count
            )
            gradient = action_gradient - current * float(np.mean(weights))
            policy = policy.updated(rate * gradient)
            clip_fractions.append(float(np.mean(~active)))

        updated = policy.probabilities
        update_distances[step - 1] = float(np.sum(np.abs(updated - old)))
        trajectory[step] = updated

    final = policy.probabilities
    marginal_trajectory = _marginal_path(trajectory, contract)
    marginal_final = marginal_trajectory[-1]
    metrics: dict[str, float | bool | str | int | None] = {
        "algorithm": "ppo_like",
        "reward_mode": contract.reward_mode,
        "training_scope": contract.training_scope,
        "theory_target": contract.theory_name,
        **_selection_diagnostics(
            marginal_final,
            contract.marginal_reference,
            contract.diagnostic_signal,
            beta_value,
            theory_target=contract.marginal_target,
        ),
        "joint_kl_to_theory": (
            _forward_kl(final, contract.policy_target) if contract.joint_shape is not None else None
        ),
        "l1_to_moment_target": float(np.sum(np.abs(marginal_final - contract.marginal_target))),
        "mean_update_norm_per_learning_rate": float(np.mean(update_distances) / rate),
        **_outcome_evidence(
            contract,
            final,
            outcome_draws_by_error,
            outcome_successes_by_error,
        ),
        "mean_clip_fraction": float(np.mean(clip_fractions)),
        "clip_ratio": clip_value,
        "steps": step_count,
        "batch_size": batch_count,
        "epochs_per_batch": epoch_count,
    }
    return _finalize_result(final, trajectory, metrics, contract)


__all__ = ["train_ppo_like"]
