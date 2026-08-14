"""Distributional diagnostics for synthetic-to-natural mediator transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _distribution(value: object, name: str) -> np.ndarray:
    distribution = np.asarray(value, dtype=np.float64)
    if distribution.ndim != 1 or distribution.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(distribution)) or np.any(distribution < 0.0):
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    if not np.isclose(distribution.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return distribution / distribution.sum()


@dataclass(frozen=True, slots=True)
class TransportDiagnostic:
    """Observed reward transport gap and its total-variation upper bound."""

    natural_reward: float
    synthetic_reward: float
    reward_gap: float
    tv_bound: float
    bound_satisfied: bool


def total_variation(left: object, right: object) -> float:
    """Return total variation between finite distributions on shared support."""

    left_distribution = _distribution(left, "left")
    right_distribution = _distribution(right, "right")
    if left_distribution.shape != right_distribution.shape:
        raise ValueError("left and right must have the same support size")
    return float(0.5 * np.abs(left_distribution - right_distribution).sum())


def transport_diagnostic(
    natural_distribution: object,
    synthetic_distribution: object,
    continuation_value: object,
) -> TransportDiagnostic:
    """Audit the bounded-reward synthetic transport inequality."""

    natural = _distribution(natural_distribution, "natural_distribution")
    synthetic = _distribution(synthetic_distribution, "synthetic_distribution")
    value = np.asarray(continuation_value, dtype=np.float64)
    if natural.shape != synthetic.shape or value.shape != natural.shape:
        raise ValueError("distributions and continuation_value must share one shape")
    if not np.all(np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("continuation_value must be finite and lie in [0, 1]")
    natural_reward = float(natural @ value)
    synthetic_reward = float(synthetic @ value)
    gap = abs(synthetic_reward - natural_reward)
    bound = total_variation(natural, synthetic)
    return TransportDiagnostic(
        natural_reward=natural_reward,
        synthetic_reward=synthetic_reward,
        reward_gap=gap,
        tv_bound=bound,
        bound_satisfied=gap <= bound + 1e-12,
    )
