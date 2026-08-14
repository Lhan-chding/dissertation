"""Nonlinear crossed-intervention risk decomposition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal


def _bounded_loss(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    loss = float(value)
    if not math.isfinite(loss) or not 0.0 <= loss <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return loss


@dataclass(frozen=True, slots=True)
class CrossedRiskResult:
    """Four risk cells and the exact main-effect/interaction identity."""

    l_mm: float
    l_om: float
    l_mo: float
    l_oo: float
    perception_deficit: float
    reasoning_deficit: float
    interaction: float
    identity_residual: float
    interaction_label: Literal["compensation", "amplification", "additive"]


def crossed_risk_decomposition(
    *,
    l_mm: float,
    l_om: float,
    l_mo: float,
    l_oo: float,
) -> CrossedRiskResult:
    """Decompose arbitrary bounded loss without assuming additive error vectors."""

    mm = _bounded_loss(l_mm, "l_mm")
    om = _bounded_loss(l_om, "l_om")
    mo = _bounded_loss(l_mo, "l_mo")
    oo = _bounded_loss(l_oo, "l_oo")
    perception = mo - oo
    reasoning = om - oo
    interaction = mm - mo - om + oo
    residual = abs((mm - oo) - (perception + reasoning + interaction))
    tolerance = 1e-15
    label: Literal["compensation", "amplification", "additive"]
    if interaction < -tolerance:
        label = "compensation"
    elif interaction > tolerance:
        label = "amplification"
    else:
        label = "additive"
    return CrossedRiskResult(
        l_mm=mm,
        l_om=om,
        l_mo=mo,
        l_oo=oo,
        perception_deficit=perception,
        reasoning_deficit=reasoning,
        interaction=interaction,
        identity_residual=residual,
        interaction_label=label,
    )
