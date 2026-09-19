"""Explicit uncertainty contracts; mass bounds are conditional on the scorer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

EVENTS = ("X", "S", "W", "I")


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Finite values required")
    return value


def conditional_mass_bounds(
    event_mass: Mapping[str, float],
    *,
    relation_mass: float | None = None,
    single_edit_mass: float | None = None,
) -> dict:
    """Sharp marginal projections of a *joint* unknown-tail simplex.

    Input masses must represent unique, complete token actions from one policy
    and one input. Callers must deduplicate scoring computation, not samples.
    qX denotes X conditional on validity (X+S+W), not answer correctness.
    Floating/scoring error is not bounded by this function.
    """
    if set(event_mass) != set(EVENTS):
        raise ValueError("Supply exactly X/S/W/I masses")
    mass = {key: _finite(event_mass[key]) for key in EVENTS}
    if any(value < 0 for value in mass.values()):
        raise ValueError("Negative probability mass")
    total = math.fsum(mass.values())
    if total > 1:
        raise ValueError("Scored mass exceeds one; do not silently clip")
    tail = 1 - total
    valid = math.fsum(mass[key] for key in ("X", "S", "W"))
    bounds = {f"p{key}": [value, value + tail] for key, value in mass.items()}
    bounds["pA"] = [mass["X"] + mass["S"], mass["X"] + mass["S"] + tail]
    bounds["V"] = [valid, valid + tail]
    # Allocating all tail to W minimizes qX; allocating it to X maximizes qX.
    bounds["qX"] = (
        [mass["X"] / (valid + tail), (mass["X"] + tail) / (valid + tail)] if valid > 0 else None
    )
    for name, value in (("relation_score", relation_mass), ("single_edit", single_edit_mass)):
        if value is not None:
            value = _finite(value)
            if not 0 <= value <= valid:
                raise ValueError(f"{name} mass must lie in [0, known valid mass]")
            bounds[name] = [value, value + tail]
    return {
        "bounds": bounds,
        "known_event_mass": mass,
        "known_mass": total,
        "unknown_tail": tail,
        "joint_constraints": {
            "tail_event_mass_nonnegative": True,
            "tail_event_mass_sum": tail,
            "event_probability_sum": 1.0,
            "pA": "pX+pS",
            "V": "pX+pS+pW",
            "ordering": "0<=pX<=pA<=V<=1",
            "scores": "0<=relation_score,single_edit<=V",
        },
        "qX_status": "IDENTIFIED_INTERVAL"
        if valid > 0
        else "UNIDENTIFIED_ZERO_POSSIBLE_DENOMINATOR",
        "evidence_kind": "CONDITIONAL_ON_SCORER",
        "statistical_confidence_interval": False,
        "numerical_error_bounded": False,
        "certification": "NOT_CERTIFIED",
    }


def joint_mass_feasible(result: Mapping, probabilities: Mapping[str, float]) -> bool:
    """Check membership, avoiding the false rectangular marginal interpretation."""
    if set(probabilities) != set(EVENTS):
        return False
    p = {key: _finite(probabilities[key]) for key in EVENTS}
    return math.isclose(math.fsum(p.values()), 1, abs_tol=1e-12, rel_tol=0) and all(
        result["known_event_mass"][key] <= p[key] <= 1 for key in EVENTS
    )


def difference_interval(left: Sequence[float], right: Sequence[float]) -> list[float]:
    """Projection only; statistical validity requires joint error allocation."""
    a, b = map(_finite, left)
    c, d = map(_finite, right)
    if a > b or c > d:
        raise ValueError("Reversed interval")
    return [a - d, b - c]


@dataclass(frozen=True)
class FixedLookAllocation:
    """Bonferroni ledger frozen before an episode; metric includes scorer path.

    A cell is one weighted fixed-panel metric (all its prompts combined), so a
    second prompt-wise union bound is unnecessary. Register distinct methods as
    distinct metrics before taking their intersection or selecting one.
    """

    candidates: tuple[str, ...]
    metrics: tuple[str, ...]
    looks: tuple[int, ...] = (32, 128, 512)
    alpha: float = 0.05

    def __post_init__(self):
        # Freeze even when JSON config supplied mutable lists.
        for name in ("candidates", "metrics", "looks"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for values in (self.candidates, self.metrics, self.looks):
            if not values or len(set(values)) != len(values):
                raise ValueError("Nonempty unique frozen cells required")
        if any(type(n) is not int or n <= 0 for n in self.looks):
            raise ValueError("Looks must be positive integers")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")

    @property
    def cell_count(self) -> int:
        return len(self.candidates) * len(self.metrics) * len(self.looks)

    def cell_alpha(self, candidate: str, metric: str, look: int) -> float:
        if candidate not in self.candidates or metric not in self.metrics or look not in self.looks:
            raise ValueError("Unregistered decision cell")
        return self.alpha / self.cell_count


def fixed_panel_hoeffding(
    samples: Sequence[Sequence[float]],
    *,
    allocation: FixedLookAllocation,
    candidate: str,
    metric: str,
    look: int,
    value_range: tuple[float, float] = (0.0, 1.0),
    prompt_weights: Sequence[float] | None = None,
) -> dict:
    """Weighted independent bounded draws; no learned control variate implied.

    Each row is one prompt's independent RNG stream at the specified fixed look.
    Across-stream independence is required. Paired endpoints should instead be
    represented as per-draw bounded differences with the appropriate range.
    """
    alpha = allocation.cell_alpha(candidate, metric, look)
    array = np.asarray(samples, dtype=float)
    low, high = map(_finite, value_range)
    if low >= high or array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != look:
        raise ValueError("Require nonempty prompt rows exactly at the frozen look")
    if not np.isfinite(array).all() or np.any(array < low) or np.any(array > high):
        raise ValueError("Sample violates registered finite contribution range")
    weights = (
        np.ones(array.shape[0]) / array.shape[0]
        if prompt_weights is None
        else np.asarray(prompt_weights, dtype=float)
    )
    if (
        weights.shape != (array.shape[0],)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Fixed nonnegative prompt weights must sum to one")
    mean = float(weights @ array.mean(axis=1))
    radius = (high - low) * math.sqrt(math.log(2 / alpha) * float(weights @ weights) / (2 * look))
    return {
        "estimate": mean,
        "interval": [max(low, mean - radius), min(high, mean + radius)],
        "radius": radius,
        "alpha": alpha,
        "episode_alpha": allocation.alpha,
        "allocated_cells": allocation.cell_count,
        "look": look,
        "value_range": [low, high],
        "evidence_kind": "FINITE_SAMPLE_FIXED_PANEL_HOEFFDING",
        "scope": "GENERATION_RANDOMNESS_FIXED_PANEL",
        "assumptions": "independent draws; frozen weights, functions, candidates and looks",
        "training_seed_inference": False,
    }


def contribution_interval(samples, *, method: str, control_variate: bool = False, **kwargs) -> dict:
    """MIX signed RAW contributions are [-2,2]; ORIGIN is not bounded."""
    if control_variate:
        raise ValueError("Control variates require separately validated ranges and pilot/crossfit")
    method = method.upper()
    if method == "ORIGIN":
        values = np.asarray(samples, dtype=float)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Finite nonempty contributions required")
        return {
            "estimate": float(values.mean()),
            "interval": None,
            "evidence_kind": "UNBOUNDED_ORIGIN_DIAGNOSTIC",
            "status": "UNRESOLVED",
            "reason": "No population weight bound; sample maximum is not a bound",
        }
    if method not in ("MIX", "RAW"):
        raise ValueError("Unknown observation method")
    if "value_range" in kwargs:
        raise ValueError("Use registered RAW/MIX ranges, not a sample-derived range")
    return fixed_panel_hoeffding(
        samples, value_range=(-2, 2) if method == "MIX" else (0, 1), **kwargs
    )
