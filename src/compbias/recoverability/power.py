"""Seeded scene-level power simulation; forks remain nested Monte Carlo repeats."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _probability(value: object, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    lower_ok = result >= 0 if allow_zero else result > 0
    if not math.isfinite(result) or not lower_ok or result >= 1:
        relation = "[0, 1)" if allow_zero else "(0, 1)"
        raise ValueError(f"{label} must lie in {relation}")
    return result


@dataclass(frozen=True, slots=True)
class PowerSimulationConfig:
    scenes: int
    forks_per_arm: int
    baseline_rate: float
    target_effect: float
    discordance: float
    scene_icc: float
    alpha: float
    repetitions: int
    seed: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.scenes, "scenes"),
            (self.forks_per_arm, "forks_per_arm"),
            (self.repetitions, "repetitions"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        baseline = _probability(self.baseline_rate, "baseline_rate", allow_zero=True)
        effect = _probability(self.target_effect, "target_effect", allow_zero=True)
        discordance = _probability(self.discordance, "discordance")
        _probability(self.scene_icc, "scene_icc", allow_zero=True)
        _probability(self.alpha, "alpha")
        if effect > discordance:
            raise ValueError("target_effect cannot exceed discordance")
        if baseline + effect >= 1:
            raise ValueError("baseline_rate plus target_effect must remain below one")


@dataclass(frozen=True, slots=True)
class PowerSimulationResult:
    estimated_power: float
    scenes: int
    forks_per_arm: int
    repetitions: int
    independent_unit: str
    seed: int


@dataclass(frozen=True, slots=True)
class PowerCurvePoint:
    scenes: int
    power: float

    def __post_init__(self) -> None:
        if type(self.scenes) is not int or self.scenes < 1:
            raise ValueError("power-curve scenes must be a positive integer")
        if (
            isinstance(self.power, bool)
            or not isinstance(self.power, (int, float))
            or not math.isfinite(float(self.power))
            or not 0 <= float(self.power) <= 1
        ):
            raise ValueError("power-curve power must lie in the closed unit interval")


@dataclass(frozen=True, slots=True)
class PowerCurve:
    scenario: str
    points: tuple[PowerCurvePoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, str) or _IDENTIFIER.fullmatch(self.scenario) is None:
            raise ValueError("scenario must be a bounded safe identifier")
        if not isinstance(self.points, tuple) or not self.points:
            raise ValueError("points must be a non-empty tuple")
        if any(not isinstance(item, PowerCurvePoint) for item in self.points):
            raise TypeError("points must contain PowerCurvePoint instances")
        scenes = [item.scenes for item in self.points]
        if scenes != sorted(scenes) or len(set(scenes)) != len(scenes):
            raise ValueError("power-curve scenes must be unique and increasing")


@dataclass(frozen=True, slots=True)
class FixedSamplePlan:
    required_eligible_scenes: int
    required_intake_scenes: int
    registered_intake_scenes: int
    family_quotas: tuple[tuple[str, int], ...]
    eligibility_rate_lower: float
    target_power: float
    feasible: bool
    independent_unit: str


def _paired_difference(uniform: np.ndarray, *, effect: float, discordance: float) -> np.ndarray:
    negative = (discordance - effect) / 2.0
    result = np.zeros(uniform.shape, dtype=np.float64)
    result[uniform < negative] = -1.0
    result[(uniform >= negative) & (uniform < discordance)] = 1.0
    return result


def simulate_paired_power(config: PowerSimulationConfig) -> PowerSimulationResult:
    """Simulate a one-sided paired scene-level test using common random numbers."""

    if not isinstance(config, PowerSimulationConfig):
        raise TypeError("config must be PowerSimulationConfig")
    rng = np.random.default_rng(config.seed)
    critical = float(norm.ppf(1.0 - config.alpha))
    rejections = 0
    for _rep in range(config.repetitions):
        shared = rng.random(config.scenes)
        independent = rng.random((config.scenes, config.forks_per_arm))
        sharing = rng.random((config.scenes, config.forks_per_arm)) < config.scene_icc
        uniform = np.where(sharing, shared[:, None], independent)
        fork_differences = _paired_difference(
            uniform,
            effect=config.target_effect,
            discordance=config.discordance,
        )
        scene_differences = fork_differences.mean(axis=1)
        estimate = float(scene_differences.mean())
        standard_deviation = float(scene_differences.std(ddof=1))
        if standard_deviation == 0:
            reject = estimate > 0
        else:
            statistic = estimate / (standard_deviation / math.sqrt(config.scenes))
            reject = statistic > critical
        rejections += int(reject)
    return PowerSimulationResult(
        estimated_power=rejections / config.repetitions,
        scenes=config.scenes,
        forks_per_arm=config.forks_per_arm,
        repetitions=config.repetitions,
        independent_unit="semantic_scene",
        seed=config.seed,
    )


def build_fixed_sample_plan(
    curves: tuple[PowerCurve, ...],
    *,
    target_power: float,
    eligibility_rate_lower: float,
    intake_scenes: int,
    family_quotas: Mapping[str, int],
) -> FixedSamplePlan:
    """Freeze the maximum requirement before outcomes; never extend after collection."""

    if not isinstance(curves, tuple) or not curves:
        raise ValueError("curves must be a non-empty tuple")
    if any(not isinstance(item, PowerCurve) for item in curves):
        raise TypeError("curves must contain PowerCurve instances")
    scenarios = [item.scenario for item in curves]
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("power scenarios must be unique")
    target = _probability(target_power, "target_power")
    rate = _probability(eligibility_rate_lower, "eligibility_rate_lower")
    if type(intake_scenes) is not int or intake_scenes < 1:
        raise ValueError("intake_scenes must be a positive integer")
    if not isinstance(family_quotas, Mapping) or not family_quotas:
        raise ValueError("family_quotas must be a non-empty mapping")
    quotas: list[tuple[str, int]] = []
    for family, quota in family_quotas.items():
        if not isinstance(family, str) or _IDENTIFIER.fullmatch(family) is None:
            raise ValueError("family quota identifier is invalid")
        if type(quota) is not int or quota < 1:
            raise ValueError("family quotas must be positive integers")
        quotas.append((family, quota))
    required_by_scenario: list[int] = []
    for curve in curves:
        passing = [point.scenes for point in curve.points if point.power >= target]
        if not passing:
            raise ValueError(f"power target is not reached for scenario {curve.scenario}")
        required_by_scenario.append(min(passing))
    required_eligible = max(max(required_by_scenario), sum(value for _key, value in quotas))
    required_intake = math.ceil(required_eligible / rate)
    return FixedSamplePlan(
        required_eligible_scenes=required_eligible,
        required_intake_scenes=required_intake,
        registered_intake_scenes=intake_scenes,
        family_quotas=tuple(sorted(quotas)),
        eligibility_rate_lower=rate,
        target_power=target,
        feasible=required_intake <= intake_scenes,
        independent_unit="semantic_scene",
    )
