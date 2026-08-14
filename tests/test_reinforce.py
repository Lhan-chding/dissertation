"""RED statistical contracts for approximate tabular policy-gradient updates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from compbias.rl.exact_kl import exact_kl_projection
from compbias.rl.grpo_adapter import train_grpo_like
from compbias.rl.ppo_adapter import train_ppo_like
from compbias.rl.reinforce import JointOptimizationResult, train_reinforce
from compbias.theory.selection import (
    binary_compensability_multiplier,
    selected_error_distribution,
)

REFERENCE = np.array([0.45, 0.25, 0.20, 0.10], dtype=np.float64)
SEVERITY = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
PROFILES = {
    "truth_aligned": np.array([0.90, 0.65, 0.40, 0.15]),
    "flat": np.array([0.55, 0.55, 0.55, 0.55]),
    "spurious": np.array([0.15, 0.40, 0.65, 0.90]),
}
SEEDS = tuple(range(20))


def _train_raw(
    algorithm: str,
    compensability: np.ndarray,
    seed: int,
    *,
    learning_rate: float = 0.20,
    steps: int = 400,
):
    common = {
        "reference_probs": REFERENCE,
        "compensability": compensability,
        "beta": 0.5,
        "learning_rate": learning_rate,
        "steps": steps,
        "seed": seed,
    }
    if algorithm == "reinforce":
        return train_reinforce(**common, batch_size=128)
    if algorithm == "ppo_like":
        return train_ppo_like(
            **common,
            batch_size=128,
            clip_ratio=0.2,
            epochs_per_batch=2,
        )
    if algorithm == "grpo_like":
        return train_grpo_like(**common, group_size=16)
    raise AssertionError(f"unregistered algorithm fixture: {algorithm}")


def _contract_call(algorithm: str, **signals: np.ndarray):
    common = {
        "reference_probs": REFERENCE,
        "beta": 0.5,
        "learning_rate": 0.05,
        "steps": 2,
        "seed": 1,
        **signals,
    }
    if algorithm == "reinforce":
        return train_reinforce(**common, batch_size=4)
    if algorithm == "ppo_like":
        return train_ppo_like(
            **common,
            batch_size=4,
            clip_ratio=0.2,
            epochs_per_batch=1,
        )
    if algorithm == "grpo_like":
        return train_grpo_like(**common, group_size=4)
    raise AssertionError(f"unregistered algorithm fixture: {algorithm}")


@pytest.mark.parametrize("algorithm", ("reinforce", "ppo_like", "grpo_like"))
def test_fixed_reasoner_outcome_policy_has_one_action_per_error_and_frozen_conditional(
    algorithm: str,
) -> None:
    compensability = PROFILES["spurious"].copy()
    first = _train_raw(algorithm, compensability, seed=17, steps=800)
    second = _train_raw(algorithm, compensability, seed=17, steps=800)
    target = np.asarray(
        selected_error_distribution(
            REFERENCE,
            binary_compensability_multiplier(compensability, beta=0.5),
        )
    )

    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    np.testing.assert_array_equal(first.trajectory, second.trajectory)
    np.testing.assert_array_equal(compensability, PROFILES["spurious"])
    assert not isinstance(first, JointOptimizationResult)
    assert first.probabilities.shape == (REFERENCE.size,)
    assert first.trajectory.shape == (801, REFERENCE.size)
    assert first.metrics["policy_action_count"] == REFERENCE.size
    assert first.metrics["error_count"] == REFERENCE.size
    assert first.metrics["reward_mode"] == "raw_fixed_reasoner_outcome"
    assert first.metrics["training_scope"] == "error_policy_fixed_reasoner_outcomes"
    assert first.metrics["theory_target"] == "binary_selection_law_moment_target"
    assert first.metrics["kl_regularization_space"] == "error_marginal"
    assert first.metrics["reasoner_conditional_frozen"] is True
    np.testing.assert_array_equal(
        first.metrics["final_outcome_conditional_by_error"], compensability
    )
    np.testing.assert_allclose(
        first.metrics["empirical_outcome_rate_by_error"], compensability, atol=0.08
    )
    assert first.metrics["sampled_outcomes_binary"] is True
    assert isinstance(first.metrics["outcome_draws"], int)
    assert isinstance(first.metrics["outcome_successes"], int)
    assert 0 <= first.metrics["outcome_successes"] <= first.metrics["outcome_draws"]
    assert first.metrics["empirical_outcome_rate"] == pytest.approx(
        first.metrics["outcome_successes"] / first.metrics["outcome_draws"]
    )
    assert first.metrics["kl_to_theory"] == pytest.approx(_kl(first.probabilities, target))
    assert first.metrics["observed_kl_from_reference"] == pytest.approx(
        _kl(first.probabilities, REFERENCE)
    )
    assert first.metrics["joint_kl_to_theory"] is None
    assert first.metrics["target_kl_from_reference"] == pytest.approx(_kl(target, REFERENCE))


@pytest.mark.parametrize("algorithm", ("reinforce", "ppo_like", "grpo_like"))
@pytest.mark.parametrize(("value", "expected_rate"), [(0.0, 0.0), (1.0, 1.0)])
def test_fixed_reasoner_outcome_trainers_draw_only_binary_extreme_outcomes(
    algorithm: str,
    value: float,
    expected_rate: float,
) -> None:
    result = _train_raw(
        algorithm,
        np.full(REFERENCE.shape, value),
        seed=3,
        steps=5,
    )

    assert result.metrics["empirical_outcome_rate"] == expected_rate
    assert result.metrics["outcome_successes"] == expected_rate * result.metrics["outcome_draws"]


@pytest.mark.parametrize("algorithm", ("reinforce", "ppo_like", "grpo_like"))
def test_fixed_reasoner_outcome_mean_direction_matches_moment_target_over_twenty_seeds(
    algorithm: str,
) -> None:
    mean_shifts = {
        name: float(
            np.mean(
                [
                    _severity_shift(_train_raw(algorithm, profile, seed).probabilities)
                    for seed in SEEDS
                ]
            )
        )
        for name, profile in PROFILES.items()
    }

    assert mean_shifts["truth_aligned"] < -0.05
    assert abs(mean_shifts["flat"]) < 0.03
    assert mean_shifts["spurious"] > 0.05


def test_fixed_reasoner_outcome_and_collapsed_reward_diagnostic_are_not_conflated() -> None:
    compensability = PROFILES["spurious"]
    raw = _train_raw("reinforce", compensability, seed=5, steps=80)
    multiplier = binary_compensability_multiplier(compensability, beta=0.5)
    effective_rewards = 0.5 * np.log(multiplier)
    collapsed = train_reinforce(
        REFERENCE,
        effective_rewards,
        beta=0.5,
        learning_rate=0.05,
        steps=80,
        batch_size=128,
        seed=5,
    )

    assert raw.metrics["reward_mode"] == "raw_fixed_reasoner_outcome"
    assert collapsed.metrics["reward_mode"] == "collapsed_effective_reward_diagnostic"
    assert collapsed.metrics["outcome_draws"] == 0
    assert raw.probabilities.tolist() != collapsed.probabilities.tolist()


@pytest.mark.parametrize("algorithm", ("reinforce", "ppo_like", "grpo_like"))
def test_reward_contract_requires_exactly_one_valid_signal(algorithm: str) -> None:
    profile = PROFILES["spurious"]
    with pytest.raises(ValueError, match="exactly one"):
        _contract_call(algorithm, rewards=profile, compensability=profile)
    with pytest.raises(ValueError, match="exactly one"):
        _contract_call(algorithm)
    with pytest.raises(ValueError, match=r"unconstrained|diagnostic|compensability"):
        _contract_call(
            algorithm,
            rewards=profile,
            unconstrained_joint_trajectory_diagnostic=True,
        )


@pytest.mark.parametrize(
    "invalid",
    (
        np.array([0.2, 0.3]),
        np.array([-0.1, 0.2, 0.3, 0.4]),
        np.array([0.1, 0.2, 0.3, 1.1]),
        np.array([0.1, 0.2, np.nan, 0.4]),
    ),
)
def test_fixed_reasoner_outcome_rejects_invalid_compensability(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="compensability"):
        train_reinforce(
            REFERENCE,
            compensability=invalid,
            beta=0.5,
            learning_rate=0.05,
            steps=2,
            batch_size=4,
            seed=1,
        )


@pytest.mark.parametrize("algorithm", ("reinforce", "ppo_like", "grpo_like"))
def test_unconstrained_joint_trajectory_is_explicitly_diagnostic_only(
    algorithm: str,
) -> None:
    profile = PROFILES["spurious"]
    common = {
        "reference_probs": REFERENCE,
        "compensability": profile,
        "unconstrained_joint_trajectory_diagnostic": True,
        "beta": 0.5,
        "learning_rate": 0.2,
        "steps": 80,
        "seed": 7,
    }
    if algorithm == "reinforce":
        result = train_reinforce(**common, batch_size=128)
    elif algorithm == "ppo_like":
        result = train_ppo_like(**common, batch_size=128, clip_ratio=0.2, epochs_per_batch=2)
    else:
        result = train_grpo_like(**common, group_size=16)

    assert isinstance(result, JointOptimizationResult)
    assert result.joint_probabilities.shape == (REFERENCE.size, 2)
    assert result.metrics["reward_mode"] == "unconstrained_joint_trajectory_diagnostic"
    assert result.metrics["training_scope"] == "unconstrained_joint_trajectory_diagnostic"
    assert result.metrics["policy_action_count"] == REFERENCE.size * 2
    assert result.metrics["reasoner_conditional_frozen"] is False
    assert result.metrics["kl_regularization_space"] == "joint_error_outcome"
    assert result.metrics["max_abs_final_reasoner_conditional_deviation"] > 0.0


def test_unconstrained_joint_diagnostic_target_marginal_is_binary_selection_law() -> None:
    compensability = PROFILES["spurious"]
    joint_reference = np.column_stack(
        (REFERENCE * (1.0 - compensability), REFERENCE * compensability)
    )
    joint_rewards = np.tile([0.0, 1.0], REFERENCE.size)
    joint_target = np.asarray(
        exact_kl_projection(joint_reference.reshape(-1), joint_rewards, beta=0.5)
    ).reshape(REFERENCE.size, 2)
    selection_target = np.asarray(
        selected_error_distribution(
            REFERENCE,
            binary_compensability_multiplier(compensability, beta=0.5),
        )
    )

    np.testing.assert_allclose(joint_target.sum(axis=1), selection_target, atol=1e-14)
    marginal_objective_target = np.asarray(exact_kl_projection(REFERENCE, compensability, beta=0.5))
    objective_l1 = float(np.sum(np.abs(marginal_objective_target - selection_target)))
    assert objective_l1 == pytest.approx(0.09504713922346647)
    assert objective_l1 > 0.09


def _train_reinforce(rewards: np.ndarray, seed: int):
    return train_reinforce(
        reference_probs=REFERENCE,
        rewards=rewards,
        beta=0.5,
        learning_rate=0.05,
        steps=400,
        batch_size=128,
        seed=seed,
    )


def _severity_shift(probabilities: np.ndarray) -> float:
    return float(probabilities @ SEVERITY - REFERENCE @ SEVERITY)


def _kl(left: np.ndarray, right: np.ndarray) -> float:
    support = left > 0.0
    return float(np.sum(left[support] * np.log(left[support] / right[support])))


def test_reinforce_result_contract_and_seed_reproducibility() -> None:
    rewards = PROFILES["spurious"].copy()
    first = _train_reinforce(rewards, seed=7)
    second = _train_reinforce(rewards, seed=7)
    trajectory = np.asarray(first.trajectory)

    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    np.testing.assert_array_equal(first.trajectory, second.trajectory)
    np.testing.assert_array_equal(rewards, PROFILES["spurious"])
    assert first.probabilities.shape == REFERENCE.shape
    assert trajectory.ndim == 2
    assert trajectory.shape[1] == REFERENCE.size
    np.testing.assert_allclose(trajectory.sum(axis=1), 1.0, atol=1e-12)
    assert isinstance(first.metrics, Mapping)
    assert first.metrics["algorithm"] == "reinforce"
    assert np.isfinite(first.metrics["kl_to_theory"])
    assert isinstance(first.metrics["reward_shift_sign_correct"], (bool, np.bool_))
    assert np.isfinite(first.metrics["odds_slope"])


def test_reinforce_mean_direction_matches_theory_over_twenty_seeds() -> None:
    mean_shifts = {
        name: float(
            np.mean(
                [_severity_shift(_train_reinforce(rewards, seed).probabilities) for seed in SEEDS]
            )
        )
        for name, rewards in PROFILES.items()
    }

    assert mean_shifts["truth_aligned"] < -0.05
    assert abs(mean_shifts["flat"]) < 0.03
    assert mean_shifts["spurious"] > 0.05


def test_grpo_like_is_explicitly_approximate_but_moves_toward_exact_target() -> None:
    rewards = PROFILES["spurious"]
    exact = exact_kl_projection(REFERENCE, rewards, beta=0.5)
    initial_kl = _kl(REFERENCE, exact)
    final_kls = []
    shifts = []

    for seed in SEEDS:
        result = train_grpo_like(
            reference_probs=REFERENCE,
            rewards=rewards,
            beta=0.5,
            learning_rate=0.05,
            steps=400,
            group_size=16,
            seed=seed,
        )
        assert result.approximate is True
        assert result.metrics["algorithm"] == "grpo_like"
        assert np.isfinite(result.metrics["mean_update_norm_per_learning_rate"])
        final_kls.append(_kl(result.probabilities, exact))
        shifts.append(_severity_shift(result.probabilities))

    assert float(np.mean(final_kls)) < initial_kl
    assert float(np.mean(shifts)) > 0.05


def test_ppo_like_is_clipped_approximate_and_moves_toward_exact_target() -> None:
    rewards = PROFILES["spurious"]
    exact = exact_kl_projection(REFERENCE, rewards, beta=0.5)
    initial_kl = _kl(REFERENCE, exact)
    final_kls = []
    shifts = []

    for seed in SEEDS:
        result = train_ppo_like(
            reference_probs=REFERENCE,
            rewards=rewards,
            beta=0.5,
            learning_rate=0.05,
            steps=400,
            batch_size=128,
            clip_ratio=0.2,
            epochs_per_batch=2,
            seed=seed,
        )
        assert result.approximate is True
        assert result.metrics["algorithm"] == "ppo_like"
        assert result.metrics["clip_ratio"] == pytest.approx(0.2)
        assert 0.0 <= result.metrics["mean_clip_fraction"] <= 1.0
        assert np.isfinite(result.metrics["mean_update_norm_per_learning_rate"])
        final_kls.append(_kl(result.probabilities, exact))
        shifts.append(_severity_shift(result.probabilities))

    assert float(np.mean(final_kls)) < initial_kl
    assert float(np.mean(shifts)) > 0.05


def test_reinforce_accepts_generator_and_rejects_two_randomness_sources() -> None:
    by_seed = _train_reinforce(PROFILES["truth_aligned"], seed=11)
    by_generator = train_reinforce(
        reference_probs=REFERENCE,
        rewards=PROFILES["truth_aligned"],
        beta=0.5,
        learning_rate=0.05,
        steps=400,
        batch_size=128,
        rng=np.random.default_rng(11),
    )
    np.testing.assert_array_equal(by_seed.probabilities, by_generator.probabilities)

    with pytest.raises(ValueError, match=r"(?i)(seed|rng|generator)"):
        train_reinforce(
            REFERENCE,
            PROFILES["truth_aligned"],
            beta=0.5,
            learning_rate=0.05,
            steps=2,
            batch_size=4,
            seed=11,
            rng=np.random.default_rng(11),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"learning_rate": np.nan},
        {"steps": 0},
        {"batch_size": 0},
        {"beta": 0.0},
    ],
)
def test_reinforce_rejects_invalid_training_controls(kwargs: dict[str, float]) -> None:
    controls = {
        "beta": 0.5,
        "learning_rate": 0.05,
        "steps": 2,
        "batch_size": 4,
    }
    controls.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        train_reinforce(
            REFERENCE,
            PROFILES["spurious"],
            seed=0,
            **controls,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip_ratio": 0.0},
        {"clip_ratio": 1.0},
        {"epochs_per_batch": 0},
        {"batch_size": 0},
    ],
)
def test_ppo_like_rejects_invalid_training_controls(kwargs: dict[str, float]) -> None:
    controls = {
        "beta": 0.5,
        "learning_rate": 0.05,
        "steps": 2,
        "batch_size": 16,
        "clip_ratio": 0.2,
        "epochs_per_batch": 2,
    }
    controls.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        train_ppo_like(REFERENCE, PROFILES["spurious"], seed=0, **controls)
