"""RED tests for the reasoner-scaling direction identity."""

from __future__ import annotations

import numpy as np
import pytest

from compbias.theory.scaling import (
    relative_compensability_gain,
    severity_scaling_derivative,
)

PROPERTY_CASES = 1_000


def _selected_distribution(
    mu0: np.ndarray, base_multiplier: np.ndarray, gain: np.ndarray, kappa: float
) -> np.ndarray:
    weights = mu0 * base_multiplier * np.exp(kappa * gain)
    return weights / weights.sum()


def _selected_severity(
    mu0: np.ndarray,
    base_multiplier: np.ndarray,
    severity: np.ndarray,
    gain: np.ndarray,
    kappa: float,
) -> float:
    return float(_selected_distribution(mu0, base_multiplier, gain, kappa) @ severity)


def test_relative_compensability_gain_is_log_multiplier_derivative() -> None:
    multiplier = np.array([0.25, 1.5, 8.0], dtype=np.float64)
    d_multiplier = np.array([-0.5, 0.3, 4.0], dtype=np.float64)

    actual = relative_compensability_gain(multiplier, d_multiplier)

    np.testing.assert_allclose(actual, d_multiplier / multiplier, rtol=0.0, atol=1e-15)


def test_scaling_derivative_matches_weighted_covariance_and_direction_regimes() -> None:
    mu = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    severity = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    error_favoring_gain = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    truth_favoring_gain = -error_favoring_gain
    uniform_gain = np.full(3, 7.0, dtype=np.float64)

    expected = np.sum(mu * (severity - mu @ severity) * error_favoring_gain)
    actual = severity_scaling_derivative(mu, severity, error_favoring_gain)

    assert actual == pytest.approx(expected, abs=1e-12)
    assert actual > 0.0
    assert severity_scaling_derivative(mu, severity, truth_favoring_gain) < 0.0
    assert severity_scaling_derivative(mu, severity, uniform_gain) == pytest.approx(0.0, abs=1e-14)


def test_scaling_derivative_matches_central_finite_difference() -> None:
    mu0 = np.array([0.15, 0.25, 0.60], dtype=np.float64)
    base_multiplier = np.array([0.7, 1.8, 0.4], dtype=np.float64)
    severity = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    gain = np.array([-0.8, 0.3, 1.2], dtype=np.float64)
    kappa = 0.37
    step = 1e-5

    multiplier = base_multiplier * np.exp(kappa * gain)
    d_multiplier = multiplier * gain
    mu_selected = _selected_distribution(mu0, base_multiplier, gain, kappa)
    relative_gain = relative_compensability_gain(multiplier, d_multiplier)
    analytic = severity_scaling_derivative(mu_selected, severity, relative_gain)
    finite_difference = (
        _selected_severity(mu0, base_multiplier, severity, gain, kappa + step)
        - _selected_severity(mu0, base_multiplier, severity, gain, kappa - step)
    ) / (2.0 * step)

    assert analytic == pytest.approx(finite_difference, abs=1e-8)


def test_finite_difference_identity_for_1000_seeded_random_landscapes() -> None:
    rng = np.random.default_rng(314159)
    step = 1e-5
    max_error = 0.0

    for _ in range(PROPERTY_CASES):
        size = int(rng.integers(2, 10))
        raw = rng.uniform(0.05, 1.0, size=size)
        mu0 = raw / raw.sum()
        base_multiplier = np.exp(rng.uniform(-2.0, 2.0, size=size))
        severity = rng.uniform(-3.0, 6.0, size=size)
        gain = rng.uniform(-2.0, 2.0, size=size)
        kappa = float(rng.uniform(-0.5, 0.5))

        multiplier = base_multiplier * np.exp(kappa * gain)
        d_multiplier = multiplier * gain
        mu_selected = _selected_distribution(mu0, base_multiplier, gain, kappa)
        analytic = float(
            severity_scaling_derivative(
                mu_selected,
                severity,
                relative_compensability_gain(multiplier, d_multiplier),
            )
        )
        finite_difference = (
            _selected_severity(mu0, base_multiplier, severity, gain, kappa + step)
            - _selected_severity(mu0, base_multiplier, severity, gain, kappa - step)
        ) / (2.0 * step)
        max_error = max(max_error, abs(analytic - finite_difference))

    assert max_error < 1e-8


def test_centered_covariance_is_stable_under_large_common_offsets() -> None:
    mu = np.array([0.25, 0.25, 0.5], dtype=np.float64)
    severity_offset = np.array([0.0, 2.0, 5.0], dtype=np.float64)
    gain_offset = np.array([-3.0, 1.0, 4.0], dtype=np.float64)
    severity = 1e12 + severity_offset
    gain = -1e12 + gain_offset
    expected = np.sum(
        mu * (severity_offset - mu @ severity_offset) * (gain_offset - mu @ gain_offset)
    )

    actual = severity_scaling_derivative(mu, severity, gain)

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-10)


def test_relative_gain_handles_tiny_positive_multipliers_without_underflow() -> None:
    multiplier = np.array([1e-300, 4e-250], dtype=np.float64)
    d_multiplier = np.array([2e-300, -1e-250], dtype=np.float64)

    gain = relative_compensability_gain(multiplier, d_multiplier)

    np.testing.assert_allclose(gain, [2.0, -0.25], rtol=1e-14, atol=0.0)
    assert np.all(np.isfinite(gain))


@pytest.mark.parametrize(
    ("multiplier", "d_multiplier"),
    [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([1.0, -1.0]), np.array([0.0, 1.0])),
        (np.array([1.0, np.inf]), np.array([0.0, 1.0])),
        (np.array([1.0, 2.0]), np.array([0.0, np.nan])),
        (np.array([1.0, 2.0]), np.array([0.0])),
    ],
)
def test_relative_gain_rejects_invalid_inputs(
    multiplier: np.ndarray, d_multiplier: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        relative_compensability_gain(multiplier, d_multiplier)


@pytest.mark.parametrize(
    ("mu", "severity", "gain"),
    [
        (np.array([0.2, 0.2]), np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        (np.array([0.5, -0.5, 1.0]), np.arange(3.0), np.arange(3.0)),
        (np.array([0.5, 0.5]), np.array([0.0]), np.array([0.0, 1.0])),
        (np.array([0.5, 0.5]), np.array([0.0, np.nan]), np.array([0.0, 1.0])),
        (np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([0.0, np.inf])),
    ],
)
def test_scaling_derivative_rejects_invalid_inputs(
    mu: np.ndarray, severity: np.ndarray, gain: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        severity_scaling_derivative(mu, severity, gain)


def test_torch_autograd_matches_scaling_identity_and_finite_difference() -> None:
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    mu0 = torch.tensor([0.15, 0.25, 0.60], dtype=dtype)
    base_multiplier = torch.tensor([0.7, 1.8, 0.4], dtype=dtype)
    severity = torch.tensor([0.0, 1.0, 4.0], dtype=dtype)
    direction = torch.tensor([-0.8, 0.3, 1.2], dtype=dtype)
    kappa = torch.tensor(0.37, dtype=dtype, requires_grad=True)

    multiplier = base_multiplier * torch.exp(kappa * direction)
    mu_selected = mu0 * multiplier
    mu_selected = mu_selected / mu_selected.sum()
    selected_severity = torch.sum(mu_selected * severity)
    autograd_derivative = torch.autograd.grad(selected_severity, kappa, create_graph=True)[0]
    relative_gain = relative_compensability_gain(multiplier, multiplier * direction)
    identity_derivative = severity_scaling_derivative(mu_selected, severity, relative_gain)

    assert isinstance(relative_gain, torch.Tensor)
    assert isinstance(identity_derivative, torch.Tensor)
    torch.testing.assert_close(identity_derivative, autograd_derivative, rtol=0.0, atol=1e-12)

    step = 1e-5
    with torch.no_grad():
        plus_multiplier = base_multiplier * torch.exp((kappa + step) * direction)
        minus_multiplier = base_multiplier * torch.exp((kappa - step) * direction)
        plus_mu = mu0 * plus_multiplier / torch.sum(mu0 * plus_multiplier)
        minus_mu = mu0 * minus_multiplier / torch.sum(mu0 * minus_multiplier)
        finite_difference = torch.sum((plus_mu - minus_mu) * severity) / (2.0 * step)
    torch.testing.assert_close(identity_derivative, finite_difference, rtol=0.0, atol=1e-8)
