"""Optimizer-independent checkpoint error-distribution identity."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType


def _distribution(value: Mapping[str, float], name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(f"{name} probabilities must be numeric")
        probability = float(raw)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(f"{name} probabilities must be finite and non-negative")
        result[key] = probability
    if not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} probabilities must sum to one")
    return result


def _severity(value: Mapping[str, float], support: set[str]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != support:
        raise ValueError("severity must contain exactly the union distribution support")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError("severity values must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("severity values must be finite and non-negative")
        result[key] = number
    return result


@dataclass(frozen=True, slots=True)
class SelectionRatioResult:
    ratios: Mapping[str, float]
    new_support: tuple[str, ...]
    new_support_mass: float
    common_support_covariance: float
    new_support_contribution: float
    actual_shift: float
    identity_residual: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratios", MappingProxyType(dict(self.ratios)))


def checkpoint_selection_ratio(
    dist_from: Mapping[str, float],
    dist_to: Mapping[str, float],
    severity: Mapping[str, float],
) -> SelectionRatioResult:
    """Return actual ratios, singular new support, and the exact shift identity."""

    before = _distribution(dist_from, "dist_from")
    after = _distribution(dist_to, "dist_to")
    support = set(before) | set(after)
    losses = _severity(severity, support)
    before_full = {key: before.get(key, 0.0) for key in support}
    after_full = {key: after.get(key, 0.0) for key in support}
    common = tuple(sorted(key for key in support if before_full[key] > 0.0))
    new_support = tuple(
        sorted(key for key in support if before_full[key] == 0.0 and after_full[key] > 0.0)
    )
    ratios = {key: after_full[key] / before_full[key] for key in common}
    new_mass = math.fsum(after_full[key] for key in new_support)
    base_mean = math.fsum(before_full[key] * losses[key] for key in support)
    ratio_mean = math.fsum(before_full[key] * ratios[key] for key in common)
    covariance = math.fsum(
        before_full[key] * (losses[key] - base_mean) * (ratios[key] - ratio_mean) for key in common
    )
    if new_mass > 0.0:
        new_mean = math.fsum(after_full[key] * losses[key] for key in new_support) / new_mass
        new_contribution = new_mass * (new_mean - base_mean)
    else:
        new_contribution = 0.0
    actual = math.fsum((after_full[key] - before_full[key]) * losses[key] for key in support)
    identity = covariance + new_contribution
    return SelectionRatioResult(
        ratios=ratios,
        new_support=new_support,
        new_support_mass=new_mass,
        common_support_covariance=covariance,
        new_support_contribution=new_contribution,
        actual_shift=actual,
        identity_residual=abs(actual - identity),
    )
