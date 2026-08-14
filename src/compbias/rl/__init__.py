"""Shared contracts for finite-policy optimization.

The public optimizers return :class:`OptimizationResult` instances.  The
result owns defensive copies of every array and exposes read-only copies, so a
caller cannot alter a completed run after the fact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _frozen_array(values: ArrayLike) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _readonly_copy(values: NDArray[np.float64]) -> NDArray[np.float64]:
    copied = values.copy()
    copied.setflags(write=False)
    return copied


def _probability_array(
    values: ArrayLike,
    *,
    name: str = "probabilities",
    ndim: int | None = None,
) -> NDArray[np.float64]:
    try:
        probabilities = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite probability array") from error

    if probabilities.size == 0 or probabilities.ndim == 0:
        raise ValueError(f"{name} must be a non-empty probability array")
    if ndim is not None and probabilities.ndim != ndim:
        raise ValueError(f"{name} must have exactly {ndim} dimension(s)")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(probabilities < 0.0):
        raise ValueError(f"{name} cannot contain negative values")

    total = float(np.sum(probabilities, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return probabilities


def _reward_array(
    values: ArrayLike,
    *,
    shape: tuple[int, ...],
    name: str = "rewards",
) -> NDArray[np.float64]:
    try:
        rewards = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error

    if rewards.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, received {rewards.shape}")
    if not np.all(np.isfinite(rewards)):
        raise ValueError(f"{name} must contain only finite values")
    return rewards


def _positive_float(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive finite number") from error
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _resolve_rng(
    *,
    seed: int | None,
    rng: np.random.Generator | None,
) -> np.random.Generator:
    if (seed is None) == (rng is None):
        raise ValueError("provide exactly one randomness source: seed or rng generator")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return rng
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return np.random.default_rng(int(seed))


@dataclass(frozen=True, slots=True, init=False)
class OptimizationResult:
    """Immutable output shared by the tabular optimizers."""

    _probabilities: NDArray[np.float64] = field(repr=False)
    _trajectory: NDArray[np.float64] = field(repr=False)
    metrics: Mapping[str, Any]
    approximate: bool

    def __init__(
        self,
        probabilities: ArrayLike,
        trajectory: ArrayLike,
        metrics: Mapping[str, Any],
        *,
        approximate: bool = False,
    ) -> None:
        final = _probability_array(probabilities, ndim=1)
        path = np.array(trajectory, dtype=np.float64, copy=True)
        if path.ndim != 2 or path.shape[1:] != final.shape:
            raise ValueError("trajectory must have shape (time, number_of_actions)")
        if path.shape[0] == 0 or not np.all(np.isfinite(path)):
            raise ValueError("trajectory must be non-empty and finite")
        if np.any(path < 0.0) or not np.allclose(path.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("every trajectory row must be a probability vector")
        if not np.allclose(path[-1], final, rtol=0.0, atol=1e-14):
            raise ValueError("the final trajectory row must equal probabilities")
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")

        object.__setattr__(self, "_probabilities", _frozen_array(final))
        object.__setattr__(self, "_trajectory", _frozen_array(path))
        object.__setattr__(self, "metrics", MappingProxyType(dict(metrics)))
        object.__setattr__(self, "approximate", bool(approximate))

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Return a read-only defensive copy of the final probabilities."""

        return _readonly_copy(self._probabilities)

    @property
    def trajectory(self) -> NDArray[np.float64]:
        """Return a read-only defensive copy of the optimization trajectory."""

        return _readonly_copy(self._trajectory)


__all__ = ["OptimizationResult"]
