"""Scene-clustered paired effects for complete crossover reasoning forks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class ArmForks:
    arm: str
    faithful_outcomes: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.arm, str) or _IDENTIFIER.fullmatch(self.arm) is None:
            raise ValueError("arm must be a bounded safe identifier")
        if not isinstance(self.faithful_outcomes, tuple) or not self.faithful_outcomes:
            raise ValueError("faithful_outcomes must be a non-empty tuple")
        if any(type(value) is not bool for value in self.faithful_outcomes):
            raise TypeError("faithful_outcomes must be boolean")


@dataclass(frozen=True, slots=True)
class SceneCrossover:
    scene_id: str
    family: str
    stratum: str
    arms: tuple[ArmForks, ...]
    forks_per_arm: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.scene_id, "scene_id"),
            (self.family, "family"),
            (self.stratum, "stratum"),
        ):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{label} must be a bounded safe identifier")
        if not isinstance(self.arms, tuple) or not self.arms:
            raise ValueError("arms must be a non-empty tuple")
        if any(not isinstance(item, ArmForks) for item in self.arms):
            raise TypeError("arms must contain ArmForks instances")
        names = [item.arm for item in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique within a scene")
        if self.forks_per_arm != 8 or type(self.forks_per_arm) is not int:
            raise ValueError("forks_per_arm must equal the preregistered eight")


@dataclass(frozen=True, slots=True)
class PairedEffect:
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_independent_scenes: int
    n_forks_observed: int
    resampling_unit: str
    family_weighting: str


def _unit_probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 < result < 1:
        raise ValueError(f"{label} must lie strictly between zero and one")
    return result


def paired_scene_effect(
    scenes: tuple[SceneCrossover, ...],
    *,
    treatment_arm: str,
    control_arm: str,
    confidence: float = 0.95,
    bootstrap_resamples: int,
    seed: int,
) -> PairedEffect:
    """Aggregate forks inside scenes, then bootstrap scenes within family strata."""

    if not isinstance(scenes, tuple) or not scenes:
        raise ValueError("scenes must be a non-empty tuple")
    if any(not isinstance(item, SceneCrossover) for item in scenes):
        raise TypeError("scenes must contain SceneCrossover instances")
    identifiers = [item.scene_id for item in scenes]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("scene identifiers must be unique")
    if treatment_arm == control_arm:
        raise ValueError("treatment and control arms must differ")
    confidence_value = _unit_probability(confidence, "confidence")
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be an integer of at least 100")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    arm_sets = [{item.arm for item in scene.arms} for scene in scenes]
    if any(arms != arm_sets[0] for arms in arm_sets):
        raise ValueError("all scenes must contain the same complete arm set")
    if treatment_arm not in arm_sets[0] or control_arm not in arm_sets[0]:
        raise ValueError("requested paired arms are missing")
    by_family: dict[str, list[float]] = {}
    n_forks = 0
    for scene in scenes:
        if any(len(item.faithful_outcomes) != scene.forks_per_arm for item in scene.arms):
            raise ValueError("each arm must contain the fixed forks_per_arm outcomes")
        arm_map = {item.arm: item for item in scene.arms}
        treatment = np.mean(arm_map[treatment_arm].faithful_outcomes)
        control = np.mean(arm_map[control_arm].faithful_outcomes)
        by_family.setdefault(scene.family, []).append(float(treatment - control))
        n_forks += sum(len(item.faithful_outcomes) for item in scene.arms)
    arrays = {family: np.asarray(values, dtype=np.float64) for family, values in by_family.items()}
    estimate = float(np.mean([values.mean() for values in arrays.values()]))
    rng = np.random.default_rng(seed)
    draws = np.zeros(bootstrap_resamples, dtype=np.float64)
    for values in arrays.values():
        indices = rng.integers(0, values.size, size=(bootstrap_resamples, values.size))
        draws += values[indices].mean(axis=1)
    draws /= len(arrays)
    alpha = (1.0 - confidence_value) / 2.0
    low, high = np.quantile(draws, (alpha, 1.0 - alpha))
    return PairedEffect(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence=confidence_value,
        n_independent_scenes=len(scenes),
        n_forks_observed=n_forks,
        resampling_unit="semantic_scene_within_family",
        family_weighting="equal_preregistered_family_weight",
    )


def interval_is_equivalent(*, ci_low: float, ci_high: float, margin: float) -> bool:
    for value, label in ((ci_low, "ci_low"), (ci_high, "ci_high"), (margin, "margin")):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
    if ci_low > ci_high:
        raise ValueError("ci_low must not exceed ci_high")
    if margin <= 0:
        raise ValueError("margin must be positive")
    return float(ci_low) > -float(margin) and float(ci_high) < float(margin)
