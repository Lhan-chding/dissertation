"""Closed-form signal probabilities for finite GRPO rollout groups."""

from __future__ import annotations

import math
from numbers import Integral, Real


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _group_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("group_size must be a positive integer")
    if value <= 0:
        raise ValueError("group_size must be a positive integer")
    return int(value)


def _category_probabilities(p_x: object, p_s: object) -> tuple[float, float, float]:
    exact = _probability(p_x, name="p_x")
    shortcut = _probability(p_s, name="p_s")
    if exact + shortcut > 1.0:
        raise ValueError("p_x + p_s must not exceed one")
    return exact, shortcut, 1.0 - exact - shortcut


def _unit_interval(value: float) -> float:
    return min(1.0, max(0.0, value))


def state_group_signal(p_x: float, group_size: int) -> float:
    """Probability that exact-state reward varies within a rollout group."""

    exact = _probability(p_x, name="p_x")
    size = _group_size(group_size)
    return _unit_interval(1.0 - exact**size - (1.0 - exact) ** size)


def answer_group_signal(p_x: float, p_s: float, group_size: int) -> float:
    """Probability that answer reward varies within a rollout group."""

    exact, shortcut, failure = _category_probabilities(p_x, p_s)
    size = _group_size(group_size)
    return _unit_interval(1.0 - (exact + shortcut) ** size - failure**size)


def correction_bearing_answer_signal(p_x: float, p_s: float, group_size: int) -> float:
    """Probability an answer-reward group contains both an exact world and a failure."""

    exact, shortcut, _failure = _category_probabilities(p_x, p_s)
    size = _group_size(group_size)
    probability = 1.0 - (1.0 - exact) ** size - (exact + shortcut) ** size + shortcut**size
    return _unit_interval(probability)


__all__ = ["answer_group_signal", "correction_bearing_answer_signal", "state_group_signal"]
