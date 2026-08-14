"""Minimum-Fisher-cost projection for local perception/reasoning updates."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # Torch is an optional project dependency.
    import torch
except ImportError:  # pragma: no cover - exercised in NumPy-only installations
    torch = None  # type: ignore[assignment]

from .selection import (
    _all_finite,
    _as_float_array,
    _is_torch_tensor,
    _uses_torch,
    _validate_vector,
)


def _validate_matrix(value: Any, name: str, *, use_torch: bool) -> Any:
    matrix = _as_float_array(value, name, use_torch=use_torch)
    size = matrix.numel() if use_torch else matrix.size
    if matrix.ndim != 2 or size == 0 or min(matrix.shape) == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not _all_finite(matrix, use_torch=use_torch):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_torch_compatibility(*values: Any) -> None:
    if not _is_torch_tensor(values[0]):
        return
    first = values[0]
    for value in values[1:]:
        if value.device != first.device:
            raise ValueError("all Torch inputs must be on the same device")
        if value.dtype != first.dtype:
            raise ValueError("all Torch inputs must have the same dtype")


def _validate_spd(matrix: Any, name: str, *, use_torch: bool) -> Any:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if use_torch:
        tolerance = 1e-5 if matrix.element_size() <= 4 else 1e-7
        absolute_tolerance = 1e-6 if matrix.element_size() <= 4 else 1e-10
        if not torch.allclose(
            matrix, matrix.transpose(-1, -2), rtol=tolerance, atol=absolute_tolerance
        ):
            raise ValueError(f"{name} must be symmetric")
        factor, info = torch.linalg.cholesky_ex(matrix)
        if bool((info != 0).any().item()):
            raise ValueError(f"{name} must be positive definite")
        return factor

    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error


def _validate_projection_inputs(
    r0: Any,
    a: Any,
    b: Any,
    fisher_p: Any,
    fisher_r: Any,
) -> tuple[Any, Any, Any, Any, Any, bool]:
    use_torch = _uses_torch(r0, a, b, fisher_p, fisher_r)
    residual = _validate_vector(r0, "r0", use_torch=use_torch)
    jacobian_p = _validate_matrix(a, "a", use_torch=use_torch)
    jacobian_r = _validate_matrix(b, "b", use_torch=use_torch)
    metric_p = _validate_matrix(fisher_p, "fisher_p", use_torch=use_torch)
    metric_r = _validate_matrix(fisher_r, "fisher_r", use_torch=use_torch)
    _validate_torch_compatibility(residual, jacobian_p, jacobian_r, metric_p, metric_r)

    residual_dimension = residual.shape[0]
    if jacobian_p.shape[0] != residual_dimension or jacobian_r.shape[0] != residual_dimension:
        raise ValueError("a and b row counts must match the residual dimension")
    if metric_p.shape != (jacobian_p.shape[1], jacobian_p.shape[1]):
        raise ValueError("fisher_p shape must match the perception update dimension")
    if metric_r.shape != (jacobian_r.shape[1], jacobian_r.shape[1]):
        raise ValueError("fisher_r shape must match the reasoning update dimension")
    _validate_spd(metric_p, "fisher_p", use_torch=use_torch)
    _validate_spd(metric_r, "fisher_r", use_torch=use_torch)
    return residual, jacobian_p, jacobian_r, metric_p, metric_r, use_torch


def _solve_schur(schur: Any, residual: Any, *, use_torch: bool) -> Any:
    if use_torch:
        try:
            return torch.linalg.solve(schur, residual)
        except RuntimeError:
            return torch.linalg.pinv(schur, hermitian=True) @ residual
    try:
        return np.linalg.solve(schur, residual)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(schur, hermitian=True) @ residual


def _constraint_is_satisfied(value: Any, residual: Any, *, use_torch: bool) -> bool:
    if use_torch:
        absolute_tolerance = 2e-5 if value.element_size() <= 4 else 1e-9
        scale = 1.0 + float(torch.max(torch.abs(residual)).detach().item())
        return bool(
            torch.allclose(
                value,
                torch.zeros_like(value),
                rtol=1e-6,
                atol=absolute_tolerance * scale,
            )
        )
    scale = 1.0 + float(np.max(np.abs(residual)))
    return bool(np.allclose(value, 0.0, rtol=1e-7, atol=1e-9 * scale))


def fisher_projection(
    r0: Any,
    a: Any,
    b: Any,
    fisher_p: Any,
    fisher_r: Any,
) -> tuple[Any, Any]:
    """Return the minimum-cost updates satisfying ``r0 + A dp + B dr = 0``."""

    residual, jacobian_p, jacobian_r, metric_p, metric_r, use_torch = _validate_projection_inputs(
        r0, a, b, fisher_p, fisher_r
    )
    if use_torch:
        solve_p_at = torch.linalg.solve(metric_p, jacobian_p.transpose(-1, -2))
        solve_r_bt = torch.linalg.solve(metric_r, jacobian_r.transpose(-1, -2))
        schur = jacobian_p @ solve_p_at + jacobian_r @ solve_r_bt
        lagrange = _solve_schur(schur, residual, use_torch=True)
    else:
        solve_p_at = np.linalg.solve(metric_p, jacobian_p.T)
        solve_r_bt = np.linalg.solve(metric_r, jacobian_r.T)
        schur = jacobian_p @ solve_p_at + jacobian_r @ solve_r_bt
        lagrange = _solve_schur(schur, residual, use_torch=False)

    delta_p = -(solve_p_at @ lagrange)
    delta_r = -(solve_r_bt @ lagrange)
    remaining = residual + jacobian_p @ delta_p + jacobian_r @ delta_r
    if not _all_finite(delta_p, use_torch=use_torch) or not _all_finite(
        delta_r, use_torch=use_torch
    ):
        raise ValueError("projection produced non-finite updates")
    if not _constraint_is_satisfied(remaining, residual, use_torch=use_torch):
        raise ValueError("the residual is not controllable by a and b")
    return delta_p, delta_r


def fisher_quadratic_cost(
    delta_p: Any,
    delta_r: Any,
    fisher_p: Any,
    fisher_r: Any,
) -> Any:
    """Evaluate the local quadratic Fisher/KL update cost."""

    use_torch = _uses_torch(delta_p, delta_r, fisher_p, fisher_r)
    update_p = _validate_vector(delta_p, "delta_p", use_torch=use_torch)
    update_r = _validate_vector(delta_r, "delta_r", use_torch=use_torch)
    metric_p = _validate_matrix(fisher_p, "fisher_p", use_torch=use_torch)
    metric_r = _validate_matrix(fisher_r, "fisher_r", use_torch=use_torch)
    _validate_torch_compatibility(update_p, update_r, metric_p, metric_r)
    if metric_p.shape != (update_p.shape[0], update_p.shape[0]):
        raise ValueError("fisher_p shape must match delta_p")
    if metric_r.shape != (update_r.shape[0], update_r.shape[0]):
        raise ValueError("fisher_r shape must match delta_r")
    _validate_spd(metric_p, "fisher_p", use_torch=use_torch)
    _validate_spd(metric_r, "fisher_r", use_torch=use_torch)
    if use_torch:
        return 0.5 * (update_p @ metric_p @ update_p + update_r @ metric_r @ update_r)
    return 0.5 * (update_p @ metric_p @ update_p + update_r @ metric_r @ update_r)


__all__ = ["fisher_projection", "fisher_quadratic_cost"]
