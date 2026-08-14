"""Stochastic finite-policy REINFORCE with an analytic KL target."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

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


def _frozen(values: ArrayLike) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _readonly(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = values.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, init=False)
class JointOptimizationResult:
    """Immutable joint-trajectory evidence with error-marginal compatibility."""

    _joint_probabilities: NDArray[np.float64] = field(repr=False)
    _joint_trajectory: NDArray[np.float64] = field(repr=False)
    metrics: Mapping[str, Any]
    approximate: bool = True

    def __init__(
        self,
        joint_probabilities: ArrayLike,
        joint_trajectory: ArrayLike,
        metrics: Mapping[str, Any],
    ) -> None:
        final = np.array(joint_probabilities, dtype=np.float64, copy=True)
        path = np.array(joint_trajectory, dtype=np.float64, copy=True)
        if final.ndim != 2 or final.shape[1] != 2:
            raise ValueError("joint_probabilities must have shape (errors, 2 outcomes)")
        if path.ndim != 3 or path.shape[1:] != final.shape or path.shape[0] == 0:
            raise ValueError("joint_trajectory must have shape (time, errors, 2 outcomes)")
        if not np.all(np.isfinite(final)) or not np.all(np.isfinite(path)):
            raise ValueError("joint trajectory evidence must be finite")
        if np.any(final < 0.0) or np.any(path < 0.0):
            raise ValueError("joint trajectory evidence cannot be negative")
        if not np.isclose(final.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("joint_probabilities must sum to one")
        if not np.allclose(path.sum(axis=(1, 2)), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("every joint trajectory step must sum to one")
        if not np.allclose(path[-1], final, rtol=0.0, atol=1e-14):
            raise ValueError("final joint trajectory step must equal joint_probabilities")
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        object.__setattr__(self, "_joint_probabilities", _frozen(final))
        object.__setattr__(self, "_joint_trajectory", _frozen(path))
        object.__setattr__(self, "metrics", MappingProxyType(dict(metrics)))
        object.__setattr__(self, "approximate", True)

    @property
    def joint_probabilities(self) -> NDArray[np.float64]:
        return _readonly(self._joint_probabilities)

    @property
    def joint_trajectory(self) -> NDArray[np.float64]:
        return _readonly(self._joint_trajectory)

    @property
    def probabilities(self) -> NDArray[np.float64]:
        marginal = self._joint_probabilities.sum(axis=1)
        marginal.setflags(write=False)
        return marginal

    @property
    def trajectory(self) -> NDArray[np.float64]:
        marginal = self._joint_trajectory.sum(axis=2)
        marginal.setflags(write=False)
        return marginal


@dataclass(frozen=True, slots=True)
class _RewardContract:
    policy_reference: NDArray[np.float64]
    training_rewards: NDArray[np.float64]
    policy_target: NDArray[np.float64]
    marginal_reference: NDArray[np.float64]
    marginal_target: NDArray[np.float64]
    diagnostic_signal: NDArray[np.float64]
    reward_mode: str
    theory_name: str
    joint_shape: tuple[int, int] | None
    outcome_probabilities: NDArray[np.float64] | None
    training_scope: str
    kl_regularization_space: str
    reasoner_conditional_frozen: bool | None
    formal_gate_eligible: bool
    error_count: int


def _forward_kl(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    support = left > 0.0
    if np.any(right[support] == 0.0):
        return float("inf")
    divergence = float(np.sum(left[support] * (np.log(left[support]) - np.log(right[support]))))
    if divergence < -1e-12:
        raise ArithmeticError("categorical KL became negative beyond floating-point tolerance")
    return max(0.0, divergence)


def _selection_diagnostics(
    probabilities: NDArray[np.float64],
    reference: NDArray[np.float64],
    signal: NDArray[np.float64],
    beta: float,
    *,
    theory_target: NDArray[np.float64] | None = None,
) -> Mapping[str, float | bool]:
    target = (
        np.asarray(exact_kl_projection(reference, signal, beta))
        if theory_target is None
        else theory_target
    )
    predicted_shift = float(target @ signal - reference @ signal)
    observed_shift = float(probabilities @ signal - reference @ signal)
    if abs(predicted_shift) <= 1e-12:
        reward_shift_sign_correct = abs(observed_shift) <= 1e-8
    else:
        reward_shift_sign_correct = observed_shift * predicted_shift > 0.0

    support = (reference > 0.0) & (probabilities > 0.0) & (target > 0.0)
    x = np.log(target[support]) - np.log(reference[support])
    y = np.log(probabilities[support]) - np.log(reference[support])
    centered_x = x - np.mean(x)
    denominator = float(centered_x @ centered_x)
    odds_slope = 0.0 if denominator <= 1e-24 else float(centered_x @ y / denominator)
    return {
        "kl_to_theory": _forward_kl(probabilities, target),
        "reference_kl_to_theory": _forward_kl(reference, target),
        "target_kl_from_reference": _forward_kl(target, reference),
        "observed_kl_from_reference": _forward_kl(probabilities, reference),
        "reward_shift_sign_correct": bool(reward_shift_sign_correct),
        "odds_slope": odds_slope,
    }


def _resolve_reward_contract(
    reference: NDArray[np.float64],
    rewards: ArrayLike | None,
    compensability: ArrayLike | None,
    beta: float,
    *,
    unconstrained_joint_trajectory_diagnostic: bool,
) -> _RewardContract:
    """Validate and freeze one of three intentionally distinct reward views."""

    if not isinstance(unconstrained_joint_trajectory_diagnostic, bool):
        raise TypeError("unconstrained_joint_trajectory_diagnostic must be boolean")
    if (rewards is None) == (compensability is None):
        raise ValueError("provide exactly one of rewards or compensability")
    if unconstrained_joint_trajectory_diagnostic and compensability is None:
        raise ValueError(
            "unconstrained joint diagnostic requires compensability, not deterministic rewards"
        )
    if compensability is not None:
        fixed_reasoner = _reward_array(
            compensability,
            shape=reference.shape,
            name="compensability",
        )
        if np.any((fixed_reasoner < 0.0) | (fixed_reasoner > 1.0)):
            raise ValueError("compensability must lie in [0, 1]")
        from compbias.theory.selection import (
            binary_compensability_multiplier,
            selected_error_distribution,
        )

        multiplier = binary_compensability_multiplier(fixed_reasoner, beta)
        selection_target = np.asarray(selected_error_distribution(reference, multiplier))
        if not unconstrained_joint_trajectory_diagnostic:
            return _RewardContract(
                policy_reference=_frozen(reference),
                training_rewards=_frozen(fixed_reasoner),
                policy_target=_frozen(selection_target),
                marginal_reference=_frozen(reference),
                marginal_target=_frozen(selection_target),
                diagnostic_signal=_frozen(fixed_reasoner),
                reward_mode="raw_fixed_reasoner_outcome",
                theory_name="binary_selection_law_moment_target",
                joint_shape=None,
                outcome_probabilities=_frozen(fixed_reasoner),
                training_scope="error_policy_fixed_reasoner_outcomes",
                kl_regularization_space="error_marginal",
                reasoner_conditional_frozen=True,
                formal_gate_eligible=True,
                error_count=reference.size,
            )
        joint_reference = np.column_stack(
            (reference * (1.0 - fixed_reasoner), reference * fixed_reasoner)
        )
        flattened_reference = joint_reference.reshape(-1)
        binary_rewards = np.tile(np.array([0.0, 1.0]), reference.size)
        joint_target = np.asarray(exact_kl_projection(flattened_reference, binary_rewards, beta))
        marginal_target = joint_target.reshape(reference.size, 2).sum(axis=1)
        if not np.allclose(marginal_target, selection_target, rtol=0.0, atol=1e-14):
            raise AssertionError("joint binary target failed the selection-law marginal identity")
        return _RewardContract(
            policy_reference=_frozen(flattened_reference),
            training_rewards=_frozen(binary_rewards),
            policy_target=_frozen(joint_target),
            marginal_reference=_frozen(reference),
            marginal_target=_frozen(marginal_target),
            diagnostic_signal=_frozen(fixed_reasoner),
            reward_mode="unconstrained_joint_trajectory_diagnostic",
            theory_name="binary_selection_law_joint_moment",
            joint_shape=(reference.size, 2),
            outcome_probabilities=_frozen(fixed_reasoner),
            training_scope="unconstrained_joint_trajectory_diagnostic",
            kl_regularization_space="joint_error_outcome",
            reasoner_conditional_frozen=False,
            formal_gate_eligible=False,
            error_count=reference.size,
        )
    signal = _reward_array(rewards, shape=reference.shape)
    target = np.asarray(exact_kl_projection(reference, signal, beta))
    return _RewardContract(
        policy_reference=_frozen(reference),
        training_rewards=_frozen(signal),
        policy_target=_frozen(target),
        marginal_reference=_frozen(reference),
        marginal_target=_frozen(target),
        diagnostic_signal=_frozen(signal),
        reward_mode="collapsed_effective_reward_diagnostic",
        theory_name="exponential_reward_projection",
        joint_shape=None,
        outcome_probabilities=None,
        training_scope="collapsed_marginal_diagnostic",
        kl_regularization_space="error_marginal",
        reasoner_conditional_frozen=None,
        formal_gate_eligible=False,
        error_count=reference.size,
    )


def _sample_training_values(
    actions: NDArray[np.int64],
    contract: _RewardContract,
    generator: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    if contract.reward_mode == "raw_fixed_reasoner_outcome":
        if contract.outcome_probabilities is None:
            raise AssertionError("fixed-reasoner outcome contract omitted its conditional")
        outcomes = (
            generator.random(actions.size) < contract.outcome_probabilities[actions]
        ).astype(np.float64)
        return (
            outcomes,
            np.bincount(actions, minlength=contract.error_count).astype(np.int64),
            np.bincount(
                actions,
                weights=outcomes,
                minlength=contract.error_count,
            ).astype(np.int64),
        )
    values = contract.training_rewards[actions]
    if contract.reward_mode == "unconstrained_joint_trajectory_diagnostic":
        outcomes = values.astype(np.float64, copy=True)
        error_actions = actions // 2
        return (
            outcomes,
            np.bincount(error_actions, minlength=contract.error_count).astype(np.int64),
            np.bincount(
                error_actions,
                weights=outcomes,
                minlength=contract.error_count,
            ).astype(np.int64),
        )
    zeros = np.zeros(contract.error_count, dtype=np.int64)
    return values, zeros, zeros.copy()


def _outcome_evidence(
    contract: _RewardContract,
    final: NDArray[np.float64],
    outcome_draws_by_error: NDArray[np.int64],
    outcome_successes_by_error: NDArray[np.int64],
) -> dict[str, Any]:
    draws = int(np.sum(outcome_draws_by_error, dtype=np.int64))
    successes = int(np.sum(outcome_successes_by_error, dtype=np.int64))
    observed = outcome_draws_by_error > 0
    empirical = np.zeros(contract.error_count, dtype=np.float64)
    np.divide(
        outcome_successes_by_error,
        outcome_draws_by_error,
        out=empirical,
        where=observed,
    )
    if contract.reward_mode == "raw_fixed_reasoner_outcome":
        final_conditional = contract.outcome_probabilities
    elif contract.joint_shape is not None:
        joint = final.reshape(contract.joint_shape)
        marginal = joint.sum(axis=1)
        final_conditional = np.divide(
            joint[:, 1],
            marginal,
            out=np.zeros_like(marginal),
            where=marginal > 0.0,
        )
    else:
        final_conditional = None
    conditional_error = None
    final_conditional_error = None
    if contract.outcome_probabilities is not None:
        if np.any(observed):
            conditional_error = float(
                np.max(np.abs(empirical[observed] - contract.outcome_probabilities[observed]))
            )
        if final_conditional is not None:
            final_conditional_error = float(
                np.max(np.abs(final_conditional - contract.outcome_probabilities))
            )
    return {
        "policy_action_count": int(final.size),
        "error_count": contract.error_count,
        "kl_regularization_space": contract.kl_regularization_space,
        "reasoner_conditional_frozen": contract.reasoner_conditional_frozen,
        "formal_gate_eligible": contract.formal_gate_eligible,
        "sampled_outcomes_binary": draws > 0,
        "outcome_draws": draws,
        "outcome_successes": successes,
        "outcome_draws_by_error": outcome_draws_by_error.tolist(),
        "outcome_successes_by_error": outcome_successes_by_error.tolist(),
        "empirical_outcome_rate": successes / draws if draws else None,
        "empirical_outcome_rate_by_error": empirical.tolist() if draws else None,
        "final_outcome_conditional_by_error": (
            final_conditional.tolist() if final_conditional is not None else None
        ),
        "max_abs_empirical_outcome_conditional_error": conditional_error,
        "max_abs_final_reasoner_conditional_deviation": final_conditional_error,
    }


def _marginal_path(
    trajectory: NDArray[np.float64],
    contract: _RewardContract,
) -> NDArray[np.float64]:
    if contract.joint_shape is None:
        return trajectory
    return trajectory.reshape(trajectory.shape[0], *contract.joint_shape).sum(axis=2)


def _finalize_result(
    final: NDArray[np.float64],
    trajectory: NDArray[np.float64],
    metrics: Mapping[str, Any],
    contract: _RewardContract,
) -> OptimizationResult | JointOptimizationResult:
    if contract.joint_shape is None:
        return OptimizationResult(final, trajectory, metrics, approximate=True)
    return JointOptimizationResult(
        final.reshape(contract.joint_shape),
        trajectory.reshape(trajectory.shape[0], *contract.joint_shape),
        metrics,
    )


def train_reinforce(
    reference_probs: ArrayLike,
    rewards: ArrayLike | None = None,
    *,
    compensability: ArrayLike | None = None,
    unconstrained_joint_trajectory_diagnostic: bool = False,
    beta: float,
    learning_rate: float,
    steps: int,
    batch_size: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> OptimizationResult | JointOptimizationResult:
    """Estimate the regularized categorical policy gradient with REINFORCE."""

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
        support = probabilities > 0.0
        log_ratio = np.zeros_like(probabilities)
        log_ratio[support] = np.log(probabilities[support]) - np.log(policy_reference[support])
        actions = policy.sample(size=batch_count, rng=generator)
        sampled_values, draws, successes = _sample_training_values(
            actions,
            contract,
            generator,
        )
        outcome_draws_by_error += draws
        outcome_successes_by_error += successes
        regularized_samples = sampled_values - beta_value * log_ratio[actions]
        baseline = (
            float(np.mean(regularized_samples))
            if contract.outcome_probabilities is not None
            else float(probabilities @ (contract.training_rewards - beta_value * log_ratio))
        )
        advantages = regularized_samples - baseline
        action_gradient = (
            np.bincount(
                actions,
                weights=advantages,
                minlength=policy_reference.size,
            )
            / batch_count
        )
        gradient = action_gradient - probabilities * float(np.mean(advantages))
        updated = policy.updated(rate * gradient)
        new_probabilities = updated.probabilities
        update_distances[step - 1] = np.sum(np.abs(new_probabilities - probabilities))
        trajectory[step] = new_probabilities
        policy = updated

    final = policy.probabilities
    marginal_trajectory = _marginal_path(trajectory, contract)
    marginal_final = marginal_trajectory[-1]
    metrics: dict[str, Any] = {
        "algorithm": "reinforce",
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
        "batch_size": batch_count,
    }
    return _finalize_result(final, trajectory, metrics, contract)


__all__ = ["JointOptimizationResult", "train_reinforce"]
