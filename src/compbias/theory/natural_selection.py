"""Natural-trajectory selection laws for binary outcome rewards."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


def _vector(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(vector, copy=True)


def _probabilities(value: object, name: str) -> np.ndarray:
    probabilities = _vector(value, name)
    if np.any(probabilities < 0.0) or not np.isclose(
        probabilities.sum(), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{name} must be non-negative and sum to one")
    return probabilities / probabilities.sum()


def _beta(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("beta must be numeric")
    beta = float(value)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    return beta


@dataclass(frozen=True, slots=True)
class NaturalSelectionShift:
    """Direct and covariance forms of the same severity shift."""

    direct_shift: float
    covariance_shift: float
    identity_residual: float
    alpha: float


def natural_binary_selected_distribution(
    mu0: object,
    c_sel: object,
    beta: float,
) -> np.ndarray:
    """Select natural error trajectories using their conditional reward moment."""

    base = _probabilities(mu0, "mu0")
    success = _vector(c_sel, "c_sel")
    if success.shape != base.shape:
        raise ValueError("mu0 and c_sel must have the same shape")
    if np.any((success < 0.0) | (success > 1.0)):
        raise ValueError("c_sel must lie in [0, 1]")
    inverse_beta = 1.0 / _beta(beta)
    if inverse_beta < math.log(np.finfo(np.float64).max):
        multiplier = 1.0 + math.expm1(inverse_beta) * success
    else:
        multiplier = success + (1.0 - success) * math.exp(-inverse_beta)
    weights = base * multiplier
    selected = weights / weights.sum()
    selected.setflags(write=False)
    return selected


def natural_selection_shift(
    mu0: object,
    severity: object,
    c_sel: object,
    beta: float,
) -> NaturalSelectionShift:
    """Return Theorem-2 severity shift and its numerical identity residual."""

    base = _probabilities(mu0, "mu0")
    losses = _vector(severity, "severity")
    success = _vector(c_sel, "c_sel")
    if losses.shape != base.shape or success.shape != base.shape:
        raise ValueError("mu0, severity, and c_sel must have the same shape")
    if np.any((success < 0.0) | (success > 1.0)):
        raise ValueError("c_sel must lie in [0, 1]")
    beta_value = _beta(beta)
    selected = natural_binary_selected_distribution(base, success, beta_value)
    direct = float(selected @ losses - base @ losses)
    inverse_beta = 1.0 / beta_value
    if inverse_beta < 700.0:
        alpha = math.expm1(inverse_beta)
        severity_mean = float(base @ losses)
        success_mean = float(base @ success)
        covariance = float(base @ ((losses - severity_mean) * (success - success_mean)))
        covariance_shift = alpha * covariance / (1.0 + alpha * success_mean)
    else:
        # Algebraically identical scaled form, avoiding exp(1 / beta) overflow.
        scale = math.exp(-inverse_beta)
        alpha_scaled = 1.0 - scale
        severity_mean = float(base @ losses)
        success_mean = float(base @ success)
        covariance = float(base @ ((losses - severity_mean) * (success - success_mean)))
        covariance_shift = alpha_scaled * covariance / (scale + alpha_scaled * success_mean)
        alpha = math.inf
    return NaturalSelectionShift(
        direct_shift=direct,
        covariance_shift=float(covariance_shift),
        identity_residual=abs(direct - covariance_shift),
        alpha=alpha,
    )
