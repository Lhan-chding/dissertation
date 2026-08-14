"""Reasoner-scaling direction identities."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # Torch is an optional project dependency.
    import torch
except ImportError:  # pragma: no cover - exercised in NumPy-only installations
    torch = None  # type: ignore[assignment]

from .selection import (
    _is_torch_tensor,
    _uses_torch,
    _validate_positive_vector,
    _validate_probability_vector,
    _validate_same_shape,
    _validate_vector,
)


def relative_compensability_gain(multiplier: Any, d_multiplier: Any) -> Any:
    """Return the elementwise derivative of the log multiplier."""

    use_torch = _uses_torch(multiplier, d_multiplier)
    moments = _validate_positive_vector(multiplier, "multiplier", use_torch=use_torch)
    derivative = _validate_vector(d_multiplier, "d_multiplier", use_torch=use_torch)
    _validate_same_shape(moments, derivative, "multiplier", "d_multiplier")
    return derivative / moments


def severity_scaling_derivative(mu_selected: Any, severity: Any, gain: Any) -> Any:
    """Return the selected covariance between severity and relative gain."""

    use_torch = _uses_torch(mu_selected, severity, gain)
    distribution = _validate_probability_vector(mu_selected, "mu_selected", use_torch=use_torch)
    severity_values = _validate_vector(severity, "severity", use_torch=use_torch)
    gain_values = _validate_vector(gain, "gain", use_torch=use_torch)
    _validate_same_shape(distribution, severity_values, "mu_selected", "severity")
    _validate_same_shape(distribution, gain_values, "mu_selected", "gain")

    if _is_torch_tensor(distribution):
        centered_severity = severity_values - torch.sum(distribution * severity_values)
        centered_gain = gain_values - torch.sum(distribution * gain_values)
        return torch.sum(distribution * centered_severity * centered_gain)

    centered_severity = severity_values - np.sum(distribution * severity_values)
    centered_gain = gain_values - np.sum(distribution * gain_values)
    return np.sum(distribution * centered_severity * centered_gain)


__all__ = ["relative_compensability_gain", "severity_scaling_derivative"]
