"""RED tests for the Bregman three-point decomposition."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from compbias.theory.bregman import bregman_divergence, bregman_three_point

PROPERTY_CASES = 1_000


def _quadratic_phi(value: np.ndarray) -> float:
    return 0.5 * float(value @ value)


def _quadratic_grad(value: np.ndarray) -> np.ndarray:
    return value


def _finite_difference_gradient(
    phi: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    gradient = np.empty_like(point, dtype=np.float64)
    for index in range(point.size):
        direction = np.zeros_like(point, dtype=np.float64)
        direction[index] = step
        gradient[index] = (phi(point + direction) - phi(point - direction)) / (2.0 * step)
    return gradient


def test_quadratic_bregman_divergence_matches_half_squared_distance() -> None:
    x = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    y = np.array([-3.0, 4.0, 1.5], dtype=np.float64)
    x_before = x.copy()
    y_before = y.copy()

    actual = bregman_divergence(x, y, _quadratic_phi, _quadratic_grad)
    expected = 0.5 * np.sum((x - y) ** 2)

    assert actual == pytest.approx(expected, abs=1e-12)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)


def test_bregman_three_point_identity_and_interaction_sign() -> None:
    y_true = np.array([0.0, 0.0], dtype=np.float64)
    y_perceived = np.array([1.0, -1.0], dtype=np.float64)
    y_model = np.array([0.0, 0.0], dtype=np.float64)

    lhs, perception, reasoning, interaction = bregman_three_point(
        y_true, y_perceived, y_model, _quadratic_phi, _quadratic_grad
    )

    assert lhs == pytest.approx(0.0, abs=1e-15)
    assert perception == pytest.approx(1.0, abs=1e-15)
    assert reasoning == pytest.approx(1.0, abs=1e-15)
    assert interaction == pytest.approx(-2.0, abs=1e-15)
    assert interaction < 0.0
    assert lhs == pytest.approx(perception + reasoning + interaction, abs=1e-12)


def test_fixed_coupling_fixture_reproduces_squared_error_decomposition() -> None:
    e_p = np.array([1.0, -1.0], dtype=np.float64)
    e_r = np.array([-1.0, 1.0], dtype=np.float64)
    y_true = np.zeros_like(e_p)
    y_perceived = y_true + e_p
    y_model = y_perceived + e_r

    def mean_square_phi(value: np.ndarray) -> float:
        return float(np.mean(value**2))

    def mean_square_grad(value: np.ndarray) -> np.ndarray:
        return 2.0 * value / value.size

    outcome, perception, reasoning, interaction = bregman_three_point(
        y_true, y_perceived, y_model, mean_square_phi, mean_square_grad
    )
    coupling = float(np.mean(e_p * e_r))

    assert perception == pytest.approx(1.0, abs=1e-12)
    assert reasoning == pytest.approx(1.0, abs=1e-12)
    assert coupling == pytest.approx(-1.0, abs=1e-12)
    assert interaction == pytest.approx(2.0 * coupling, abs=1e-12)
    assert outcome == pytest.approx(0.0, abs=1e-12)


def test_three_point_identity_for_1000_seeded_random_strictly_convex_quadratics() -> None:
    rng = np.random.default_rng(1618033)
    max_error = 0.0

    for _ in range(PROPERTY_CASES):
        dimension = int(rng.integers(1, 9))
        factor = rng.normal(size=(dimension, dimension))
        matrix = factor.T @ factor + 0.25 * np.eye(dimension)
        x = rng.normal(size=dimension)
        y = rng.normal(size=dimension)
        z = rng.normal(size=dimension)

        def phi(value: np.ndarray, matrix: np.ndarray = matrix) -> float:
            return 0.5 * float(value @ matrix @ value)

        def grad_phi(value: np.ndarray, matrix: np.ndarray = matrix) -> np.ndarray:
            return matrix @ value

        lhs, first, second, interaction = bregman_three_point(x, y, z, phi, grad_phi)
        independent_lhs = 0.5 * (x - z) @ matrix @ (x - z)
        max_error = max(
            max_error,
            abs(float(lhs - first - second - interaction)),
            abs(float(lhs - independent_lhs)),
        )
        assert float(first) >= -1e-12
        assert float(second) >= -1e-12

    assert max_error < 1e-8


def test_nonlinear_phi_gradient_agrees_with_central_finite_difference() -> None:
    point = np.array([-0.7, 0.2, 1.1], dtype=np.float64)

    def phi(value: np.ndarray) -> float:
        return float(np.sum(np.exp(value) + 0.25 * value**2))

    def grad_phi(value: np.ndarray) -> np.ndarray:
        return np.exp(value) + 0.5 * value

    finite_difference = _finite_difference_gradient(phi, point)
    np.testing.assert_allclose(grad_phi(point), finite_difference, rtol=0.0, atol=1e-9)

    x = np.array([-0.1, 0.5, 0.8])
    z = np.array([0.3, -0.4, 1.4])
    lhs, first, second, interaction = bregman_three_point(x, point, z, phi, grad_phi)
    assert lhs == pytest.approx(first + second + interaction, abs=1e-10)
    assert bregman_divergence(x, z, phi, grad_phi) >= 0.0


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.array([0.0, 1.0]), np.array([0.0])),
        (np.array([0.0, np.nan]), np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([0.0, np.inf])),
        (np.array([]), np.array([])),
    ],
)
def test_bregman_divergence_rejects_invalid_points(x: np.ndarray, y: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bregman_divergence(x, y, _quadratic_phi, _quadratic_grad)


def test_bregman_api_validates_phi_and_gradient_contracts() -> None:
    x = np.array([0.0, 1.0])
    y = np.array([1.0, 0.0])

    with pytest.raises(TypeError):
        bregman_divergence(x, y, None, _quadratic_grad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bregman_divergence(x, y, _quadratic_phi, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        bregman_divergence(x, y, lambda value: value, _quadratic_grad)
    with pytest.raises(ValueError):
        bregman_divergence(x, y, _quadratic_phi, lambda value: value[:1])
    with pytest.raises(ValueError):
        bregman_divergence(x, y, _quadratic_phi, lambda value: np.full_like(value, np.nan))


def test_bregman_three_point_rejects_mismatched_intermediate_shape() -> None:
    with pytest.raises(ValueError):
        bregman_three_point(np.zeros(2), np.zeros(3), np.zeros(2), _quadratic_phi, _quadratic_grad)


def test_torch_bregman_identity_matches_numpy_and_preserves_autograd() -> None:
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    x = torch.tensor([0.2, -0.8, 1.4], dtype=dtype)
    y = torch.tensor([-0.3, 0.1, 0.9], dtype=dtype)
    z = torch.tensor([0.7, -0.2, 0.4], dtype=dtype, requires_grad=True)

    def phi(value):
        return 0.5 * torch.sum(value * value)

    def grad_phi(value):
        return value

    lhs, first, second, interaction = bregman_three_point(x, y, z, phi, grad_phi)

    assert isinstance(lhs, torch.Tensor)
    torch.testing.assert_close(lhs, first + second + interaction, rtol=0.0, atol=1e-12)
    expected = 0.5 * torch.sum((x - z) ** 2)
    torch.testing.assert_close(lhs, expected, rtol=0.0, atol=1e-12)

    lhs.backward()
    assert z.grad is not None
    torch.testing.assert_close(z.grad, z.detach() - x, rtol=0.0, atol=1e-12)
