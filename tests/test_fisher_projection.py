"""RED tests for the local Fisher/KL credit-allocation projection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from compbias.theory.fisher_projection import fisher_projection, fisher_quadratic_cost

PROPERTY_CASES = 1_000


def _unpack_projection(result: Any) -> tuple[Any, Any]:
    """Accept the public pair or immutable-result-object return contracts."""
    if isinstance(result, tuple):
        if len(result) != 2:
            raise AssertionError("fisher_projection tuple must contain exactly two updates")
        return result
    if hasattr(result, "delta_p") and hasattr(result, "delta_r"):
        return result.delta_p, result.delta_r
    raise AssertionError(
        "fisher_projection must return (delta_p, delta_r) or an object with those fields"
    )


def _reference_projection(
    r0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    fisher_p: np.ndarray,
    fisher_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fisher_p_solve_at = np.linalg.solve(fisher_p, a.T)
    fisher_r_solve_bt = np.linalg.solve(fisher_r, b.T)
    schur = a @ fisher_p_solve_at + b @ fisher_r_solve_bt
    lagrange = np.linalg.solve(schur, r0)
    return -fisher_p_solve_at @ lagrange, -fisher_r_solve_bt @ lagrange


def test_scalar_fixture_allocates_more_update_to_lower_fisher_cost_block() -> None:
    r0 = np.array([2.0], dtype=np.float64)
    a = np.array([[1.0]], dtype=np.float64)
    b = np.array([[1.0]], dtype=np.float64)
    fisher_p = np.array([[1.0]], dtype=np.float64)
    fisher_r = np.array([[3.0]], dtype=np.float64)

    delta_p, delta_r = _unpack_projection(fisher_projection(r0, a, b, fisher_p, fisher_r))

    np.testing.assert_allclose(delta_p, [-1.5], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(delta_r, [-0.5], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(r0 + a @ delta_p + b @ delta_r, [0.0], atol=1e-12)
    assert fisher_quadratic_cost(delta_p, delta_r, fisher_p, fisher_r) == pytest.approx(
        1.5, abs=1e-12
    )
    assert abs(float(delta_p[0])) > abs(float(delta_r[0]))


def test_projection_satisfies_constraint_and_kkt_stationarity() -> None:
    r0 = np.array([1.2, -0.7], dtype=np.float64)
    a = np.array([[1.0, 0.2, -0.4], [0.0, 1.0, 0.3]], dtype=np.float64)
    b = np.array([[0.5, -0.2], [0.7, 1.1]], dtype=np.float64)
    fisher_p = np.array([[2.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.8]], dtype=np.float64)
    fisher_r = np.array([[1.2, 0.15], [0.15, 2.5]], dtype=np.float64)

    delta_p, delta_r = _unpack_projection(fisher_projection(r0, a, b, fisher_p, fisher_r))
    expected_p, expected_r = _reference_projection(r0, a, b, fisher_p, fisher_r)

    np.testing.assert_allclose(delta_p, expected_p, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(delta_r, expected_r, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(r0 + a @ delta_p + b @ delta_r, 0.0, atol=1e-11)

    schur = a @ np.linalg.solve(fisher_p, a.T) + b @ np.linalg.solve(fisher_r, b.T)
    lagrange = np.linalg.solve(schur, r0)
    np.testing.assert_allclose(fisher_p @ delta_p + a.T @ lagrange, 0.0, atol=1e-11)
    np.testing.assert_allclose(fisher_r @ delta_r + b.T @ lagrange, 0.0, atol=1e-11)


def test_finite_difference_cost_derivative_is_zero_along_feasible_direction() -> None:
    r0 = np.array([1.0], dtype=np.float64)
    a = np.array([[1.0, 1.0]], dtype=np.float64)
    b = np.array([[1.0]], dtype=np.float64)
    fisher_p = np.diag([2.0, 3.0])
    fisher_r = np.array([[5.0]])
    delta_p, delta_r = _unpack_projection(fisher_projection(r0, a, b, fisher_p, fisher_r))
    feasible_p_direction = np.array([1.0, -1.0])
    feasible_r_direction = np.array([0.0])
    step = 1e-5

    plus = fisher_quadratic_cost(
        delta_p + step * feasible_p_direction,
        delta_r + step * feasible_r_direction,
        fisher_p,
        fisher_r,
    )
    minus = fisher_quadratic_cost(
        delta_p - step * feasible_p_direction,
        delta_r - step * feasible_r_direction,
        fisher_p,
        fisher_r,
    )
    finite_difference = (plus - minus) / (2.0 * step)

    assert finite_difference == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(a @ feasible_p_direction + b @ feasible_r_direction, 0.0, atol=0.0)


def test_projection_formula_for_1000_seeded_random_spd_problems() -> None:
    rng = np.random.default_rng(1414213)
    max_update_error = 0.0
    max_constraint_error = 0.0

    for _ in range(PROPERTY_CASES):
        residual_dimension = int(rng.integers(1, 4))
        p_dimension = residual_dimension + int(rng.integers(0, 3))
        r_dimension = int(rng.integers(1, 5))
        a = np.concatenate(
            [
                np.eye(residual_dimension),
                rng.normal(size=(residual_dimension, p_dimension - residual_dimension)),
            ],
            axis=1,
        )
        b = rng.normal(size=(residual_dimension, r_dimension))
        p_factor = rng.normal(size=(p_dimension, p_dimension))
        r_factor = rng.normal(size=(r_dimension, r_dimension))
        fisher_p = p_factor.T @ p_factor + 0.5 * np.eye(p_dimension)
        fisher_r = r_factor.T @ r_factor + 0.5 * np.eye(r_dimension)
        r0 = rng.normal(size=residual_dimension)

        delta_p, delta_r = _unpack_projection(fisher_projection(r0, a, b, fisher_p, fisher_r))
        expected_p, expected_r = _reference_projection(r0, a, b, fisher_p, fisher_r)
        max_update_error = max(
            max_update_error,
            float(np.max(np.abs(delta_p - expected_p))),
            float(np.max(np.abs(delta_r - expected_r))),
        )
        residual = r0 + a @ delta_p + b @ delta_r
        max_constraint_error = max(max_constraint_error, float(np.max(np.abs(residual))))
        cost = fisher_quadratic_cost(delta_p, delta_r, fisher_p, fisher_r)
        assert np.isfinite(cost)
        assert float(cost) >= -1e-12

    assert max_update_error < 1e-8
    assert max_constraint_error < 1e-8


def test_projection_remains_finite_for_ill_conditioned_but_spd_metrics() -> None:
    r0 = np.array([1.0], dtype=np.float64)
    a = np.array([[1.0, 1.0]], dtype=np.float64)
    b = np.array([[1.0, -1.0]], dtype=np.float64)
    fisher_p = np.diag([1e-10, 1e10])
    fisher_r = np.diag([1e-6, 1e6])

    delta_p, delta_r = _unpack_projection(fisher_projection(r0, a, b, fisher_p, fisher_r))

    assert np.all(np.isfinite(delta_p))
    assert np.all(np.isfinite(delta_r))
    np.testing.assert_allclose(r0 + a @ delta_p + b @ delta_r, 0.0, atol=1e-10)
    assert np.isfinite(fisher_quadratic_cost(delta_p, delta_r, fisher_p, fisher_r))


@pytest.mark.parametrize(
    ("r0", "a", "b", "fisher_p", "fisher_r"),
    [
        (
            np.ones(2),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.eye(1),
            np.eye(1),
        ),
        (
            np.ones(1),
            np.ones((1, 2)),
            np.ones((1, 1)),
            np.eye(1),
            np.eye(1),
        ),
        (
            np.ones(1),
            np.ones((1, 1)),
            np.ones((1, 2)),
            np.eye(1),
            np.eye(1),
        ),
        (
            np.array([np.nan]),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.eye(1),
            np.eye(1),
        ),
        (
            np.ones(1),
            np.ones((1, 2)),
            np.ones((1, 1)),
            np.array([[1.0, 2.0], [0.0, 1.0]]),
            np.eye(1),
        ),
        (
            np.ones(1),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.array([[0.0]]),
            np.eye(1),
        ),
        (
            np.ones(1),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.eye(1),
            np.array([[-1.0]]),
        ),
    ],
)
def test_projection_rejects_invalid_shapes_values_and_fisher_matrices(
    r0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    fisher_p: np.ndarray,
    fisher_r: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        fisher_projection(r0, a, b, fisher_p, fisher_r)


def test_projection_rejects_uncontrollable_nonzero_residual() -> None:
    with pytest.raises(ValueError):
        fisher_projection(
            np.array([1.0]),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.eye(1),
            np.eye(1),
        )


def test_cost_rejects_incompatible_updates_and_metrics() -> None:
    with pytest.raises(ValueError):
        fisher_quadratic_cost(np.ones(2), np.ones(1), np.eye(1), np.eye(1))
    with pytest.raises(ValueError):
        fisher_quadratic_cost(np.ones(1), np.ones(1), np.array([[-1.0]]), np.eye(1))
    with pytest.raises(ValueError):
        fisher_quadratic_cost(np.array([np.nan]), np.ones(1), np.eye(1), np.eye(1))


def test_torch_projection_matches_numpy_and_supports_autograd() -> None:
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    r0 = torch.tensor([1.2, -0.7], dtype=dtype, requires_grad=True)
    a = torch.tensor([[1.0, 0.2], [0.0, 1.0]], dtype=dtype)
    b = torch.tensor([[0.5, -0.2], [0.7, 1.1]], dtype=dtype)
    fisher_p = torch.tensor([[2.0, 0.2], [0.2, 1.5]], dtype=dtype)
    fisher_r = torch.tensor([[1.2, 0.15], [0.15, 2.5]], dtype=dtype)

    delta_p, delta_r = _unpack_projection(fisher_projection(r0, a, b, fisher_p, fisher_r))
    cost = fisher_quadratic_cost(delta_p, delta_r, fisher_p, fisher_r)

    assert isinstance(delta_p, torch.Tensor)
    assert isinstance(delta_r, torch.Tensor)
    assert isinstance(cost, torch.Tensor)
    torch.testing.assert_close(
        r0 + a @ delta_p + b @ delta_r,
        torch.zeros(2, dtype=dtype),
        atol=1e-11,
        rtol=0.0,
    )

    expected_p, expected_r = _reference_projection(
        r0.detach().numpy(),
        a.numpy(),
        b.numpy(),
        fisher_p.numpy(),
        fisher_r.numpy(),
    )
    np.testing.assert_allclose(delta_p.detach().numpy(), expected_p, atol=1e-11)
    np.testing.assert_allclose(delta_r.detach().numpy(), expected_r, atol=1e-11)

    cost.backward()
    assert r0.grad is not None
    assert torch.all(torch.isfinite(r0.grad))
