"""Operational audits for frozen acquisition and mediator regimes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

FrozenRegime = Literal["F0", "F1", "F2"]


def _array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    return np.array(array, copy=True)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class FrozenRegimeAudit:
    regime: FrozenRegime
    acquisition_drift: float
    mediator_drift: float
    perception_deficit_shift: float
    passed: bool
    allowed_claims: tuple[str, ...]


def audit_frozen_regime(
    *,
    regime: FrozenRegime | str,
    acquisition_before: object,
    acquisition_after: object,
    mediator_before: object,
    mediator_after: object,
    perception_deficit_before: float,
    perception_deficit_after: float,
    tolerance: float = 1e-12,
) -> FrozenRegimeAudit:
    """Enforce F0/F1/F2 claim boundaries on observed representations."""

    if regime not in {"F0", "F1", "F2"}:
        raise ValueError("regime must be one of F0, F1, or F2")
    before_h = _array(acquisition_before, "acquisition_before")
    after_h = _array(acquisition_after, "acquisition_after")
    before_z = _array(mediator_before, "mediator_before")
    after_z = _array(mediator_after, "mediator_after")
    if before_h.shape != after_h.shape or before_z.shape != after_z.shape:
        raise ValueError("before and after arrays must retain their shapes")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be numeric")
    threshold = float(tolerance)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    acquisition_drift = float(np.max(np.abs(after_h - before_h)))
    mediator_drift = float(np.max(np.abs(after_z - before_z)))
    deficit_shift = _finite(perception_deficit_after, "perception_deficit_after") - _finite(
        perception_deficit_before, "perception_deficit_before"
    )

    if regime == "F0":
        if (
            acquisition_drift > threshold
            or mediator_drift > threshold
            or abs(deficit_shift) > threshold
        ):
            raise ValueError("F0 requires invariant acquisition, mediator, and perception deficit")
        claims = ("reasoning", "interaction")
    elif regime == "F1":
        if acquisition_drift > threshold:
            raise ValueError("F1 requires invariant frozen acquisition representations")
        claims = ("readout", "reasoning", "interaction")
    else:
        claims = ("operational_perception", "reasoning", "interaction")
    return FrozenRegimeAudit(
        regime=regime,
        acquisition_drift=acquisition_drift,
        mediator_drift=mediator_drift,
        perception_deficit_shift=deficit_shift,
        passed=True,
        allowed_claims=claims,
    )
