"""Closed-form diagnostics for group-relative policy support."""

from __future__ import annotations

import math
from collections.abc import Iterable


def _probability(value: object, name: str = "p") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


def _group_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("k must be an integer")
    if value <= 0:
        raise ValueError("k must be positive")
    return value


def informative_group_probability(p: float, k: int) -> float:
    """Probability a K-rollout group contains both success and failure."""

    probability = _probability(p)
    size = _group_size(k)
    return 1.0 - probability**size - (1.0 - probability) ** size


def mean_informative_group_rate(probabilities: Iterable[float], k: int) -> float:
    values = tuple(_probability(value, "probability") for value in probabilities)
    if not values:
        raise ValueError("probabilities must not be empty")
    size = _group_size(k)
    return sum(informative_group_probability(value, size) for value in values) / len(values)


def expected_informative_groups(
    probabilities: Iterable[float], k: int, groups_per_scene: int
) -> float:
    if isinstance(groups_per_scene, bool) or not isinstance(groups_per_scene, int):
        raise TypeError("groups_per_scene must be an integer")
    if groups_per_scene < 0:
        raise ValueError("groups_per_scene must be non-negative")
    values = tuple(_probability(value, "probability") for value in probabilities)
    if not values:
        raise ValueError("probabilities must not be empty")
    return sum(informative_group_probability(value, k) for value in values) * groups_per_scene
