"""Natural-gradient reward/correction alignment in a finite parameter space."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

_SYMMETRY_TOLERANCE = 1e-12


def _gradient(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        gradient = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional numeric vector") from error
    if gradient.ndim != 1 or gradient.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(gradient)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(gradient, dtype=np.float64, copy=True)


def _fisher(values: ArrayLike, *, dimension: int) -> NDArray[np.float64]:
    try:
        fisher = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("fisher must be a finite positive-definite matrix") from error
    if fisher.shape != (dimension, dimension):
        raise ValueError(f"fisher must have shape ({dimension}, {dimension})")
    if not np.all(np.isfinite(fisher)):
        raise ValueError("fisher must contain only finite values")
    if not np.allclose(fisher, fisher.T, rtol=0.0, atol=_SYMMETRY_TOLERANCE):
        raise ValueError("fisher must be symmetric")
    canonical = np.array((fisher + fisher.T) / 2.0, dtype=np.float64, copy=True)
    try:
        np.linalg.cholesky(canonical)
    except np.linalg.LinAlgError as error:
        raise ValueError("fisher must be positive definite") from error
    return canonical


def natural_gradient_direction(gradient: ArrayLike, fisher: ArrayLike) -> tuple[float, ...]:
    r"""Return ``F^{-1} gradient`` without mutating either input."""

    normalized_gradient = _gradient(gradient, name="gradient")
    normalized_fisher = _fisher(fisher, dimension=normalized_gradient.size)
    try:
        direction = np.linalg.solve(normalized_fisher, normalized_gradient)
    except np.linalg.LinAlgError as error:
        raise ValueError("fisher could not be solved stably") from error
    if not np.all(np.isfinite(direction)):
        raise FloatingPointError("the natural-gradient direction is non-finite")
    return tuple(float(value) for value in direction)


def alignment_coefficient(
    correction_gradient: ArrayLike,
    reward_gradient: ArrayLike,
    fisher: ArrayLike,
) -> float:
    r"""Return the cosine induced by the inverse-Fisher inner product."""

    correction = _gradient(correction_gradient, name="correction_gradient")
    reward = _gradient(reward_gradient, name="reward_gradient")
    if reward.shape != correction.shape:
        raise ValueError("correction_gradient and reward_gradient must have the same shape")
    normalized_fisher = _fisher(fisher, dimension=correction.size)
    try:
        solved = np.linalg.solve(normalized_fisher, np.column_stack((correction, reward)))
    except np.linalg.LinAlgError as error:
        raise ValueError("fisher could not be solved stably") from error

    correction_norm_squared = float(correction @ solved[:, 0])
    reward_norm_squared = float(reward @ solved[:, 1])
    if correction_norm_squared <= 0.0 or reward_norm_squared <= 0.0:
        raise ValueError("both gradients must have positive inverse-Fisher norm")
    numerator = float(correction @ solved[:, 1])
    denominator = math.sqrt(correction_norm_squared * reward_norm_squared)
    coefficient = numerator / denominator
    if not math.isfinite(coefficient):
        raise FloatingPointError("the alignment coefficient is non-finite")
    return min(1.0, max(-1.0, coefficient))


__all__ = ["alignment_coefficient", "natural_gradient_direction"]
