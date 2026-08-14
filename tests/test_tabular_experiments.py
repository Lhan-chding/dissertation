"""RED integration contracts for Experiments A--D and the Phase-B gates."""

from __future__ import annotations

import numpy as np
import pytest

from compbias.rl.exact_kl import exact_kl_projection
from compbias.rl.mirror_descent import optimize_mirror_descent
from compbias.rl.tabular_experiments import run_scaling_paths
from compbias.theory.coordination import (
    CoordinationParams,
    basin_map,
    symmetric_bifurcation_root,
)
from compbias.theory.selection import binary_compensability_multiplier

BASE = np.array([0.45, 0.25, 0.20, 0.10], dtype=np.float64)
SEVERITY = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
COMPENSABILITY_PROFILES = {
    "truth_aligned": np.array([0.90, 0.70, 0.45, 0.20]),
    "flat": np.array([0.55, 0.55, 0.55, 0.55]),
    "spurious": np.array([0.35, 0.45, 0.70, 0.90]),
}


def _equivalent_rewards_for_binary_profile(profile: np.ndarray, beta: float) -> np.ndarray:
    multiplier = binary_compensability_multiplier(profile, beta)
    return beta * np.log(multiplier)


def test_experiment_a_truth_flat_spurious_predicted_and_observed_regimes() -> None:
    beta = 0.5
    baseline_severity = float(BASE @ SEVERITY)
    shifts: dict[str, float] = {}

    for name, profile in COMPENSABILITY_PROFILES.items():
        rewards = _equivalent_rewards_for_binary_profile(profile, beta)
        predicted = exact_kl_projection(BASE, rewards, beta)
        observed = optimize_mirror_descent(
            BASE,
            rewards,
            beta=beta,
            step_size=0.5,
            steps=80,
            seed=0,
        ).probabilities

        assert np.sum(np.abs(observed - predicted)) < 1e-8
        shifts[name] = float(observed @ SEVERITY - baseline_severity)

    assert shifts["truth_aligned"] < 0.0
    assert shifts["flat"] == pytest.approx(0.0, abs=1e-10)
    assert shifts["spurious"] > 0.0


def test_experiment_a_pairwise_odds_residual_is_below_theory_gate() -> None:
    beta = 0.5
    rewards = _equivalent_rewards_for_binary_profile(COMPENSABILITY_PROFILES["spurious"], beta)
    selected = exact_kl_projection(BASE, rewards, beta)
    residuals = []

    for left in range(BASE.size):
        for right in range(left + 1, BASE.size):
            observed = np.log(selected[left] / selected[right])
            predicted = np.log(BASE[left] / BASE[right]) + (rewards[left] - rewards[right]) / beta
            residuals.append(observed - predicted)

    assert max(abs(residual) for residual in residuals) < 1e-10


def test_experiment_c_equal_average_gain_paths_have_three_scaling_directions() -> None:
    base = np.array([0.40, 0.35, 0.25], dtype=np.float64)
    severity = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    gains = {
        "truth_gain": np.array([2.0, 4.0 / 7.0, 0.0]),
        "uniform_gain": np.ones(3),
        "error_gain": np.array([0.0, 0.0, 4.0]),
    }
    kappa = 0.2
    initial_severity = float(base @ severity)
    selected_severity = {
        name: float(exact_kl_projection(base, kappa * gain, beta=1.0) @ severity)
        for name, gain in gains.items()
    }

    average_gains = [float(base @ gain) for gain in gains.values()]
    np.testing.assert_allclose(average_gains, np.ones(3), atol=1e-14, rtol=0.0)
    assert selected_severity["truth_gain"] < initial_severity
    assert selected_severity["uniform_gain"] == pytest.approx(initial_severity, abs=1e-14)
    assert selected_severity["error_gain"] > initial_severity

    paths = run_scaling_paths(base, severity, gains, kappa=kappa, beta=1.0)
    assert tuple(path.name for path in paths) == ("truth_gain", "uniform_gain", "error_gain")
    assert [path.average_gain for path in paths] == pytest.approx([1.0, 1.0, 1.0])
    assert paths[0].covariance_derivative < 0.0
    assert paths[1].covariance_derivative == pytest.approx(0.0, abs=1e-14)
    assert paths[2].covariance_derivative > 0.0


def test_experiment_d_basin_transition_repeats_across_twenty_seeded_initializations() -> None:
    params = CoordinationParams(delta=1.0, epsilon=1.0)
    labels: list[str] = []

    for seed in range(20):
        rng = np.random.default_rng(seed)
        p0 = float(rng.uniform(0.25, 0.45))
        gap = float(rng.uniform(0.08, 0.15))
        q0 = 1.0 - p0 + (gap if seed % 2 else -gap)
        expected = "truthful" if p0 + q0 > 1.0 else "compensatory"
        result = basin_map(
            np.array([p0]),
            np.array([q0]),
            params,
            horizon=30.0,
            separatrix_tolerance=1e-10,
        )
        label = str(result.labels[0, 0])
        labels.append(label)
        assert label == expected

    assert labels.count("truthful") == 10
    assert labels.count("compensatory") == 10


def test_experiment_d_symmetric_kl_transition_occurs_at_beta_over_a_half() -> None:
    assert symmetric_bifurcation_root(0.6) == pytest.approx(0.0, abs=1e-14)
    assert symmetric_bifurcation_root(0.5) == pytest.approx(0.0, abs=1e-14)

    below = symmetric_bifurcation_root(0.3)
    assert below == pytest.approx(0.9073323166453315, abs=1e-10)
    assert -below < 0.0 < below
