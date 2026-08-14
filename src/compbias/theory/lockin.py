"""Repeated fixed-landscape selection and its closed form."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from .selection import (
    _is_torch_tensor,
    _normalise_log_weights,
    _safe_log_nonnegative,
    _uses_torch,
    _validate_positive_vector,
    _validate_probability_vector,
    _validate_same_shape,
    selected_error_distribution,
)


def _validate_inputs(mu: Any, multiplier: Any) -> tuple[Any, Any, bool]:
    use_torch = _uses_torch(mu, multiplier)
    distribution = _validate_probability_vector(mu, "mu", use_torch=use_torch)
    moments = _validate_positive_vector(multiplier, "multiplier", use_torch=use_torch)
    _validate_same_shape(distribution, moments, "mu", "multiplier")
    return distribution, moments, use_torch


def _validate_steps(steps: int) -> int:
    if isinstance(steps, bool) or not isinstance(steps, Integral):
        raise TypeError("steps must be a non-negative integer")
    steps_value = int(steps)
    if steps_value < 0:
        raise ValueError("steps must be a non-negative integer")
    return steps_value


def selection_update(mu: Any, multiplier: Any) -> Any:
    """Apply one normalized multiplicative selection update."""

    distribution, moments, _ = _validate_inputs(mu, multiplier)
    return selected_error_distribution(distribution, moments)


def repeated_selection(mu0: Any, multiplier: Any, steps: int) -> Any:
    """Apply the fixed multiplier selection update ``steps`` times."""

    steps_value = _validate_steps(steps)
    distribution, moments, _ = _validate_inputs(mu0, multiplier)
    current = distribution.clone() if _is_torch_tensor(distribution) else distribution.copy()
    for _ in range(steps_value):
        current = selected_error_distribution(current, moments)
    return current


def repeated_selection_closed_form(mu0: Any, multiplier: Any, steps: int) -> Any:
    """Evaluate ``mu0 * multiplier**steps`` using stable log weights."""

    steps_value = _validate_steps(steps)
    distribution, moments, use_torch = _validate_inputs(mu0, multiplier)
    if steps_value == 0:
        return distribution.clone() if _is_torch_tensor(distribution) else distribution.copy()

    log_moments = _safe_log_nonnegative(moments, use_torch=use_torch)
    if use_torch:
        relative_log_moments = log_moments - log_moments.max()
        log_weights = _safe_log_nonnegative(distribution, use_torch=True)
        log_weights = log_weights + steps_value * relative_log_moments
    else:
        relative_log_moments = log_moments - np.max(log_moments)
        with np.errstate(over="ignore", under="ignore"):
            log_weights = _safe_log_nonnegative(distribution, use_torch=False)
            log_weights = log_weights + steps_value * relative_log_moments
    return _normalise_log_weights(log_weights, use_torch=use_torch)


__all__ = ["repeated_selection", "repeated_selection_closed_form", "selection_update"]
