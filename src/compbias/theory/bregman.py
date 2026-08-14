"""Bregman divergence and the standard three-point decomposition."""

from __future__ import annotations

import math
from collections.abc import Callable
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
    _validate_same_shape,
    _validate_vector,
)


def _validate_callbacks(phi: Any, grad_phi: Any) -> None:
    if not callable(phi):
        raise TypeError("phi must be callable")
    if not callable(grad_phi):
        raise TypeError("grad_phi must be callable")


def _evaluate_phi(phi: Callable[[Any], Any], point: Any, *, use_torch: bool) -> Any:
    result = phi(point)
    if use_torch:
        if not _is_torch_tensor(result):
            try:
                result = point.new_tensor(result)
            except (TypeError, ValueError) as error:
                raise ValueError("phi must return a scalar") from error
        if result.ndim != 0:
            raise ValueError("phi must return a scalar")
        if not bool(torch.isfinite(result).item()):
            raise ValueError("phi must return a finite scalar")
        return result

    try:
        array = np.asarray(result)
    except (TypeError, ValueError) as error:
        raise ValueError("phi must return a scalar") from error
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError("phi must return a real scalar")
    scalar = float(array)
    if not math.isfinite(scalar):
        raise ValueError("phi must return a finite scalar")
    return scalar


def _evaluate_gradient(grad_phi: Callable[[Any], Any], point: Any, *, use_torch: bool) -> Any:
    result = grad_phi(point)
    if use_torch:
        if not _is_torch_tensor(result):
            raise ValueError("grad_phi must return a Torch tensor for Torch inputs")
        if result.shape != point.shape:
            raise ValueError("grad_phi output must have the same shape as its input")
        if result.device != point.device:
            raise ValueError("grad_phi output must be on the input device")
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError("grad_phi must return only finite values")
        return result

    result_array = _as_float_array(result, "grad_phi output", use_torch=False)
    if result_array.shape != point.shape:
        raise ValueError("grad_phi output must have the same shape as its input")
    if not _all_finite(result_array, use_torch=False):
        raise ValueError("grad_phi must return only finite values")
    return result_array


def _validated_points(*points: Any) -> tuple[tuple[Any, ...], bool]:
    use_torch = _uses_torch(*points)
    validated = tuple(
        _validate_vector(point, f"point_{index}", use_torch=use_torch)
        for index, point in enumerate(points)
    )
    first = validated[0]
    for index, point in enumerate(validated[1:], start=1):
        _validate_same_shape(first, point, "point_0", f"point_{index}")
    return validated, use_torch


def _divergence_validated(
    x: Any,
    y: Any,
    phi: Callable[[Any], Any],
    grad_phi: Callable[[Any], Any],
    *,
    use_torch: bool,
) -> Any:
    phi_x = _evaluate_phi(phi, x, use_torch=use_torch)
    phi_y = _evaluate_phi(phi, y, use_torch=use_torch)
    gradient_y = _evaluate_gradient(grad_phi, y, use_torch=use_torch)
    if use_torch:
        return phi_x - phi_y - torch.sum(gradient_y * (x - y))
    return phi_x - phi_y - np.sum(gradient_y * (x - y))


def bregman_divergence(
    x: Any,
    y: Any,
    phi: Callable[[Any], Any],
    grad_phi: Callable[[Any], Any],
) -> Any:
    """Compute ``D_phi(x, y)`` without modifying either input."""

    _validate_callbacks(phi, grad_phi)
    (x_value, y_value), use_torch = _validated_points(x, y)
    return _divergence_validated(x_value, y_value, phi, grad_phi, use_torch=use_torch)


def bregman_three_point(
    x: Any,
    y: Any,
    z: Any,
    phi: Callable[[Any], Any],
    grad_phi: Callable[[Any], Any],
) -> tuple[Any, Any, Any, Any]:
    """Return outcome, first leg, second leg, and interaction terms."""

    _validate_callbacks(phi, grad_phi)
    (x_value, y_value, z_value), use_torch = _validated_points(x, y, z)
    outcome = _divergence_validated(x_value, z_value, phi, grad_phi, use_torch=use_torch)
    first = _divergence_validated(x_value, y_value, phi, grad_phi, use_torch=use_torch)
    second = _divergence_validated(y_value, z_value, phi, grad_phi, use_torch=use_torch)
    gradient_y = _evaluate_gradient(grad_phi, y_value, use_torch=use_torch)
    gradient_z = _evaluate_gradient(grad_phi, z_value, use_torch=use_torch)
    if use_torch:
        interaction = torch.sum((x_value - y_value) * (gradient_y - gradient_z))
    else:
        interaction = np.sum((x_value - y_value) * (gradient_y - gradient_z))
    return outcome, first, second, interaction


__all__ = ["bregman_divergence", "bregman_three_point"]
