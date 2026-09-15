import numpy as np
import pytest

from src.modeling_contrast.covariance import (
    contrast_covariance,
    covariance_of_mean,
    inverse_cdf_joint,
    lr_moments,
    multinomial_covariance,
    shrink_covariance,
    stable_exp_difference,
)


def test_shared_baseline_and_origin_cancellation():
    incidence = np.array([[0, -1, 1, 0], [0, -1, 0, 1]])
    expected = [[0.7, 0.3], [0.3, 0.8]]
    np.testing.assert_allclose(
        contrast_covariance(np.diag([0.8, 0.3, 0.4, 0.5]), incidence), expected
    )
    np.testing.assert_allclose(
        contrast_covariance(np.diag([80, 0.3, 0.4, 0.5]), incidence), expected
    )


def test_multinomial_jeffreys_is_covariance_only():
    p = np.array([0.0, 0.2, 0.6, 0.2])
    cov = multinomial_covariance(p, 64)
    np.testing.assert_allclose(cov @ np.ones(4), 0, atol=1e-16)
    np.testing.assert_array_equal(p, [0.0, 0.2, 0.6, 0.2])


def test_empirical_covariance_joint_order():
    x = np.array([[1, 3], [3, 7], [5, 11]], dtype=float)
    np.testing.assert_allclose(covariance_of_mean(x), [[4 / 3, 8 / 3], [8 / 3, 16 / 3]])


def test_zero_empirical_noise_never_gets_zero_width():
    result, metadata = shrink_covariance(np.zeros((3, 3)), eta=0.1, n=64)
    assert np.linalg.eigvalsh(result).min() > 0
    assert metadata["status"] == "EMPIRICAL_DEGENERATE"
    with pytest.raises(ValueError):
        shrink_covariance(np.zeros((3, 3)))


def test_crn_fixed_order_marginals_and_can_increase_variance():
    p = np.array([0.49, 0.01, 0.01, 0.49])
    q = p[::-1][[1, 0, 3, 2]]
    j = inverse_cdf_joint(p, q)
    np.testing.assert_allclose(j.sum(1), p)
    np.testing.assert_allclose(j.sum(0), q)
    f = np.array([1, 0, 1, 0])
    z = f[None, :] - f[:, None]
    assert (j * z**2).sum() > 0.5


def test_lr_exact_unbiased_quadratic_scaling_and_support():
    p = np.array([0.2, 0.25, 0.4, 0.15])
    h = np.array([0.001, 0, 0.001, -0.002])
    mean, cov, _ = lr_moments(p, p + h, p, np.eye(4))
    np.testing.assert_allclose(mean, h, atol=1e-16)
    _, twice, _weights = lr_moments(p, p + 2 * h, p, np.eye(4))
    np.testing.assert_allclose(twice, 4 * cov, rtol=1e-11)
    _, _, mixed = lr_moments(p, p + h, p + h / 2, np.eye(4))
    assert abs(mixed).max() <= 2
    with pytest.raises(ValueError, match="SUPPORT"):
        lr_moments([1.0, 0], [0.9, 0.1], [1.0, 0], np.eye(2))


def test_stable_difference_zero_near_equality_and_failure():
    a = np.array([0.0, -10.0, -1.0])
    b = a + 1e-12
    np.testing.assert_allclose(
        stable_exp_difference(a, b), -np.exp(a) * np.expm1(b - a), rtol=1e-12
    )
    np.testing.assert_array_equal(stable_exp_difference(a, a), 0)
    np.testing.assert_allclose(
        stable_exp_difference([-np.inf, 0, -np.inf], [0, -np.inf, -np.inf]), [-1, 1, 0]
    )
    with pytest.raises(FloatingPointError):
        stable_exp_difference([1000.0], [0.0])
