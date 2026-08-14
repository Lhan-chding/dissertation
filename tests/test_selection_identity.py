"""RED tests for the compensability selection identities."""

from __future__ import annotations

import numpy as np
import pytest

from compbias.theory.selection import (
    binary_compensability_multiplier,
    boltzmann_projection,
    expectation_shift,
    reward_moment_multiplier,
    selected_error_distribution,
)

PROPERTY_CASES = 1_000


def test_boltzmann_projection_matches_direct_formula_and_odds_identity() -> None:
    base = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    rewards = np.array([0.1, 0.4, 0.9], dtype=np.float64)
    base_before = base.copy()
    rewards_before = rewards.copy()

    projected = boltzmann_projection(base, rewards, beta=0.7)
    weights = base * np.exp(rewards / 0.7)
    expected = weights / weights.sum()

    np.testing.assert_allclose(projected, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(base, base_before)
    np.testing.assert_array_equal(rewards, rewards_before)
    assert np.isclose(projected.sum(), 1.0, rtol=0.0, atol=1e-15)
    assert np.all(projected >= 0.0)

    for left, right in ((0, 1), (0, 2), (1, 2)):
        actual_log_odds = np.log(projected[left] / projected[right])
        expected_log_odds = (
            np.log(base[left] / base[right]) + (rewards[left] - rewards[right]) / 0.7
        )
        assert abs(actual_log_odds - expected_log_odds) < 1e-12


def test_reward_moment_multiplier_averages_conditional_exponential_rewards() -> None:
    conditional_rewards = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.2, 0.4, 0.6, 0.8],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    actual = reward_moment_multiplier(conditional_rewards, beta=0.5)
    expected = np.mean(np.exp(conditional_rewards / 0.5), axis=-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)


def test_fixed_selection_fixture_reaches_strong_success_filtering_limit() -> None:
    mu0 = np.array([0.6, 0.4], dtype=np.float64)
    compensability = np.array([0.5, 0.9], dtype=np.float64)
    severity = np.array([0.0, 1.0], dtype=np.float64)

    multiplier = binary_compensability_multiplier(compensability, beta=0.025)
    selected = selected_error_distribution(mu0, multiplier)
    shift = expectation_shift(mu0, severity, multiplier)

    expected_error_rate = (0.4 * 0.9) / (0.6 * 0.5 + 0.4 * 0.9)
    assert selected[1] == pytest.approx(expected_error_rate, abs=1e-12)
    assert selected[1] > mu0[1]
    assert shift == pytest.approx(expected_error_rate - 0.4, abs=1e-12)


def test_expectation_shift_is_exact_covariance_identity() -> None:
    mu0 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    values = np.array([-3.0, 2.0, 0.5, 7.0], dtype=np.float64)
    multiplier = np.array([0.2, 1.5, 0.8, 4.0], dtype=np.float64)

    selected = selected_error_distribution(mu0, multiplier)
    direct_shift = np.dot(selected, values) - np.dot(mu0, values)
    mean_multiplier = np.dot(mu0, multiplier)
    covariance = np.dot(mu0, values * multiplier) - (np.dot(mu0, values) * mean_multiplier)

    actual_shift = expectation_shift(mu0, values, multiplier)

    assert actual_shift == pytest.approx(direct_shift, abs=1e-12)
    assert actual_shift == pytest.approx(covariance / mean_multiplier, abs=1e-12)


def test_selection_identity_for_1000_seeded_random_distributions() -> None:
    rng = np.random.default_rng(20260814)
    max_identity_error = 0.0
    max_odds_error = 0.0

    for _ in range(PROPERTY_CASES):
        size = int(rng.integers(2, 12))
        raw = rng.uniform(0.01, 1.0, size=size)
        mu0 = raw / raw.sum()
        values = rng.normal(size=size)
        multiplier = np.exp(rng.uniform(-4.0, 4.0, size=size))

        selected = selected_error_distribution(mu0, multiplier)
        direct_shift = float(selected @ values - mu0 @ values)
        identity_shift = float(expectation_shift(mu0, values, multiplier))
        max_identity_error = max(max_identity_error, abs(direct_shift - identity_shift))

        left, right = rng.choice(size, size=2, replace=False)
        actual_log_odds = np.log(selected[left] / selected[right])
        expected_log_odds = np.log(mu0[left] / mu0[right]) + np.log(
            multiplier[left] / multiplier[right]
        )
        max_odds_error = max(max_odds_error, abs(actual_log_odds - expected_log_odds))

        assert np.all(np.isfinite(selected))
        assert np.all(selected >= 0.0)
        assert abs(float(selected.sum()) - 1.0) < 1e-12

    assert max_identity_error < 1e-10
    assert max_odds_error < 1e-10


def test_log_space_paths_remain_finite_for_extreme_valid_inputs() -> None:
    projected = boltzmann_projection(
        np.array([0.999, 0.001], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        beta=1e-6,
    )
    assert np.all(np.isfinite(projected))
    assert projected.sum() == pytest.approx(1.0, abs=1e-15)
    assert projected[1] == pytest.approx(1.0, abs=1e-12)

    selected = selected_error_distribution(
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([1e308, 1e308], dtype=np.float64),
    )
    np.testing.assert_allclose(selected, [0.5, 0.5], rtol=0.0, atol=1e-15)
    assert np.all(np.isfinite(selected))


@pytest.mark.parametrize("beta", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_beta_raises_clear_value_error(beta: float) -> None:
    with pytest.raises(ValueError, match=r"(?i)beta"):
        boltzmann_projection(np.array([0.5, 0.5]), np.array([0.0, 1.0]), beta)
    with pytest.raises(ValueError, match=r"(?i)beta"):
        reward_moment_multiplier(np.array([[0.0, 1.0]]), beta)
    with pytest.raises(ValueError, match=r"(?i)beta"):
        binary_compensability_multiplier(np.array([0.2, 0.8]), beta)


@pytest.mark.parametrize(
    ("base", "rewards"),
    [
        (np.array([0.6, -0.1, 0.5]), np.array([0.0, 0.5, 1.0])),
        (np.array([0.2, 0.2]), np.array([0.0, 1.0])),
        (np.array([0.5, np.nan, 0.5]), np.array([0.0, 0.5, 1.0])),
        (np.array([0.5, 0.5]), np.array([0.0, np.inf])),
        (np.array([0.5, 0.5]), np.array([0.0, 0.5, 1.0])),
        (np.array([0.5, 0.5]), np.array([-0.1, 0.5])),
        (np.array([0.5, 0.5]), np.array([0.5, 1.1])),
    ],
)
def test_boltzmann_projection_rejects_invalid_distributions_and_rewards(
    base: np.ndarray, rewards: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        boltzmann_projection(base, rewards, beta=0.5)


def test_selection_helpers_reject_invalid_shapes_ranges_and_multipliers() -> None:
    with pytest.raises(ValueError):
        reward_moment_multiplier(np.array([]), beta=1.0)
    with pytest.raises(ValueError):
        reward_moment_multiplier(np.array([[0.0, 1.1]]), beta=1.0)
    with pytest.raises(ValueError):
        reward_moment_multiplier(np.array([[0.0, np.nan]]), beta=1.0)

    with pytest.raises(ValueError):
        selected_error_distribution(np.array([0.5, 0.5]), np.array([1.0]))
    with pytest.raises(ValueError):
        selected_error_distribution(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        selected_error_distribution(np.array([0.5, 0.5]), np.array([1.0, np.inf]))

    with pytest.raises(ValueError):
        expectation_shift(np.array([0.5, 0.5]), np.array([1.0]), np.ones(2))
    with pytest.raises(ValueError):
        expectation_shift(np.array([0.5, 0.5]), np.array([0.0, np.nan]), np.ones(2))

    with pytest.raises(ValueError):
        binary_compensability_multiplier(np.array([-0.1, 0.5]), beta=1.0)
    with pytest.raises(ValueError):
        binary_compensability_multiplier(np.array([0.5, 1.1]), beta=1.0)


def test_torch_selection_api_matches_numpy_and_preserves_autograd() -> None:
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    base = torch.tensor([0.2, 0.3, 0.5], dtype=dtype)
    rewards = torch.tensor([0.1, 0.4, 0.9], dtype=dtype, requires_grad=True)
    conditional = torch.tensor([[0.0, 1.0], [0.25, 0.75]], dtype=dtype)
    compensability = torch.tensor([0.2, 0.8], dtype=dtype)
    severity = torch.tensor([0.0, 1.0], dtype=dtype)

    projected = boltzmann_projection(base, rewards, beta=0.7)
    moments = reward_moment_multiplier(conditional, beta=0.5)
    binary = binary_compensability_multiplier(compensability, beta=0.5)
    selected = selected_error_distribution(base[:2] / base[:2].sum(), binary)
    shift = expectation_shift(base[:2] / base[:2].sum(), severity, binary)

    assert isinstance(projected, torch.Tensor)
    assert isinstance(moments, torch.Tensor)
    assert isinstance(binary, torch.Tensor)
    assert isinstance(selected, torch.Tensor)
    assert isinstance(shift, torch.Tensor)

    np_projected = boltzmann_projection(base.detach().numpy(), rewards.detach().numpy(), beta=0.7)
    np.testing.assert_allclose(projected.detach().numpy(), np_projected, atol=1e-12)
    np.testing.assert_allclose(
        moments.detach().numpy(),
        np.mean(np.exp(conditional.numpy() / 0.5), axis=-1),
        atol=1e-12,
    )
    assert selected.sum().item() == pytest.approx(1.0, abs=1e-15)

    projected[0].backward()
    assert rewards.grad is not None
    assert torch.all(torch.isfinite(rewards.grad))
