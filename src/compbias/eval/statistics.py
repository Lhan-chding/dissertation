"""Seed-local bootstrap intervals and multiplicity corrections."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A deterministic percentile-bootstrap summary."""

    mean: float
    low: float
    high: float
    confidence: float
    n_resamples: int


def _sample(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sample")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _bootstrap_parameters(confidence: float, n_resamples: int, seed: int) -> None:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a seeded percentile interval for a sample mean."""

    sample = _sample(values, "values")
    _bootstrap_parameters(confidence, n_resamples, seed)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(n_resamples, sample.size))
    bootstrap_means = sample[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        mean=float(sample.mean()),
        low=float(low),
        high=float(high),
        confidence=float(confidence),
        n_resamples=n_resamples,
    )


def paired_bootstrap_delta(
    before: Sequence[float] | np.ndarray,
    after: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap the paired ``after - before`` mean difference."""

    before_sample = _sample(before, "before")
    after_sample = _sample(after, "after")
    if before_sample.size != after_sample.size:
        raise ValueError("paired samples must have the same length")
    return bootstrap_mean_ci(
        after_sample - before_sample,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down correction while preserving input key order."""

    if not isinstance(p_values, Mapping):
        raise TypeError("p_values must be a mapping")
    validated: list[tuple[int, str, float]] = []
    for position, (key, value) in enumerate(p_values.items()):
        if not isinstance(key, str) or not key:
            raise ValueError("hypothesis keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError("p-values must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("p-values must be finite and lie in [0, 1]")
        validated.append((position, key, number))

    ordered = sorted(validated, key=lambda item: (item[2], item[0]))
    count = len(ordered)
    adjusted_by_key: dict[str, float] = {}
    running_maximum = 0.0
    for rank, (_position, key, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running_maximum = max(running_maximum, candidate)
        adjusted_by_key[key] = running_maximum
    return {key: adjusted_by_key[key] for key in p_values}
