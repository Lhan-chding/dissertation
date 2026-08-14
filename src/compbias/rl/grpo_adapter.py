"""A deliberately approximate, finite-policy GRPO-like diagnostic.

This is not a faithful implementation of neural GRPO.  It only adapts GRPO's
within-group reward centering to a categorical policy, while applying the KL
gradient analytically.  The explicit approximation flag prevents this
diagnostic from being mistaken for the exact distribution-space theorem.
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


def train_grpo_like(
    reference_probs: ArrayLike,
    rewards: ArrayLike | None = None,
    *,
    compensability: ArrayLike | None = None,
    unconstrained_joint_trajectory_diagnostic: bool = False,
    beta: float,
    learning_rate: float,
    steps: int,
    group_size: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> OptimizationResult | JointOptimizationResult:
    """Run an approximate group-centred categorical policy update."""

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
    group_count = _positive_int(group_size, name="group_size")
    generator = _resolve_rng(seed=seed, rng=rng)

    policy_reference = contract.policy_reference
    policy = CategoricalPolicy.from_probabilities(policy_reference)
    trajectory = np.empty((step_count + 1, policy_reference.size), dtype=np.float64)
    trajectory[0] = policy.probabilities
    update_distances = np.empty(step_count, dtype=np.float64)
    outcome_draws_by_error = np.zeros(contract.error_count, dtype=np.int64)
    outcome_successes_by_error = np.zeros(contract.error_count, dtype=np.int64)

    for step in range(1, step_count + 1):
        probabilities = policy.probabilities
        actions = policy.sample(size=group_count, rng=generator)
        sampled_rewards, draws, successes = _sample_training_values(
            actions,
            contract,
            generator,
        )
        outcome_draws_by_error += draws
        outcome_successes_by_error += successes
        advantages = sampled_rewards - float(np.mean(sampled_rewards))
        reward_gradient = (
            np.bincount(
                actions,
                weights=advantages,
                minlength=policy_reference.size,
            )
            / group_count
        )
        reward_gradient -= probabilities * float(np.mean(advantages))

        support = probabilities > 0.0
        log_ratio = np.zeros_like(probabilities)
        log_ratio[support] = np.log(probabilities[support]) - np.log(policy_reference[support])
        kl_value = float(probabilities @ log_ratio)
        kl_gradient = probabilities * (log_ratio - kl_value)
        updated = policy.updated(rate * (reward_gradient - beta_value * kl_gradient))

        new_probabilities = updated.probabilities
        update_distances[step - 1] = np.sum(np.abs(new_probabilities - probabilities))
        trajectory[step] = new_probabilities
        policy = updated

    final = policy.probabilities
    marginal_trajectory = _marginal_path(trajectory, contract)
    marginal_final = marginal_trajectory[-1]
    metrics: dict[str, float | bool | str | int | None] = {
        "algorithm": "grpo_like",
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
        "steps": step_count,
        "group_size": group_count,
    }
    return _finalize_result(final, trajectory, metrics, contract)


__all__ = ["train_grpo_like"]
