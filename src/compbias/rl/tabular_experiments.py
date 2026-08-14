"""Small, auditable runners for the plan's tabular Experiments A, C, and D."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from compbias.rl import (
    _frozen_array,
    _positive_float,
    _probability_array,
    _reward_array,
)
from compbias.rl.exact_kl import exact_kl_projection
from compbias.rl.mirror_descent import optimize_mirror_descent
from compbias.theory.coordination import BasinMap, CoordinationParams, basin_map
from compbias.theory.scaling import severity_scaling_derivative
from compbias.theory.selection import binary_compensability_multiplier


def _nonnegative_float(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a nonnegative finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a nonnegative finite number") from error
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return converted


@dataclass(frozen=True, slots=True)
class SelectionProfileResult:
    """Predicted-versus-observed output for one compensability profile."""

    name: str
    predicted: NDArray[np.float64]
    observed: NDArray[np.float64]
    severity_shift: float
    pairwise_odds_residual: float
    l1_error: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicted", _frozen_array(self.predicted))
        object.__setattr__(self, "observed", _frozen_array(self.observed))


@dataclass(frozen=True, slots=True)
class ScalingPathResult:
    """Finite scaling-path selection and its local covariance direction."""

    name: str
    gain: NDArray[np.float64]
    selected: NDArray[np.float64]
    average_gain: float
    covariance_derivative: float
    severity_shift: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "gain", _frozen_array(self.gain))
        object.__setattr__(self, "selected", _frozen_array(self.selected))


def _pairwise_odds_residual(
    selected: NDArray[np.float64],
    reference: NDArray[np.float64],
    rewards: NDArray[np.float64],
    beta: float,
) -> float:
    support = np.flatnonzero((reference > 0.0) & (selected > 0.0))
    residuals: list[float] = []
    for offset, left in enumerate(support):
        for right in support[offset + 1 :]:
            observed = np.log(selected[left] / selected[right])
            predicted = (
                np.log(reference[left] / reference[right]) + (rewards[left] - rewards[right]) / beta
            )
            residuals.append(float(observed - predicted))
    return max((abs(value) for value in residuals), default=0.0)


def run_selection_profiles(
    base_probs: ArrayLike,
    severity: ArrayLike,
    profiles: Mapping[str, ArrayLike],
    *,
    beta: float,
    step_size: float = 0.5,
    steps: int = 80,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SelectionProfileResult, ...]:
    """Run exact and mirror-descent selection for named binary profiles."""

    reference = _probability_array(base_probs, name="base_probs", ndim=1)
    severity_array = _reward_array(severity, shape=reference.shape, name="severity")
    beta_value = _positive_float(beta, name="beta")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("profiles must be a non-empty mapping")

    baseline_severity = float(reference @ severity_array)
    results: list[SelectionProfileResult] = []
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError("profile names must be non-empty strings")
        multiplier = np.asarray(
            binary_compensability_multiplier(profile, beta_value),
            dtype=np.float64,
        )
        if multiplier.shape != reference.shape:
            raise ValueError(f"profile {name!r} must have shape {reference.shape}")
        equivalent_rewards = beta_value * np.log(multiplier)
        predicted = np.asarray(exact_kl_projection(reference, equivalent_rewards, beta_value))
        mirror = optimize_mirror_descent(
            reference,
            equivalent_rewards,
            beta=beta_value,
            step_size=step_size,
            steps=steps,
            seed=seed,
            rng=rng,
        )
        observed = np.asarray(mirror.probabilities)
        results.append(
            SelectionProfileResult(
                name=name,
                predicted=predicted,
                observed=observed,
                severity_shift=float(observed @ severity_array - baseline_severity),
                pairwise_odds_residual=_pairwise_odds_residual(
                    observed,
                    reference,
                    equivalent_rewards,
                    beta_value,
                ),
                l1_error=float(np.sum(np.abs(observed - predicted))),
            )
        )
    return tuple(results)


def run_scaling_paths(
    base_probs: ArrayLike,
    severity: ArrayLike,
    gains: Mapping[str, ArrayLike],
    *,
    kappa: float,
    beta: float = 1.0,
) -> tuple[ScalingPathResult, ...]:
    """Evaluate named reasoner-gain directions at a finite scaling step."""

    reference = _probability_array(base_probs, name="base_probs", ndim=1)
    severity_array = _reward_array(severity, shape=reference.shape, name="severity")
    kappa_value = _nonnegative_float(kappa, name="kappa")
    beta_value = _positive_float(beta, name="beta")
    if not isinstance(gains, Mapping) or not gains:
        raise ValueError("gains must be a non-empty mapping")

    baseline_severity = float(reference @ severity_array)
    results: list[ScalingPathResult] = []
    for name, gain in gains.items():
        if not isinstance(name, str) or not name:
            raise ValueError("gain names must be non-empty strings")
        gain_array = _reward_array(gain, shape=reference.shape, name=f"gain {name!r}")
        selected = np.asarray(exact_kl_projection(reference, kappa_value * gain_array, beta_value))
        derivative = float(
            severity_scaling_derivative(selected, severity_array, gain_array) / beta_value
        )
        results.append(
            ScalingPathResult(
                name=name,
                gain=gain_array,
                selected=selected,
                average_gain=float(reference @ gain_array),
                covariance_derivative=derivative,
                severity_shift=float(selected @ severity_array - baseline_severity),
            )
        )
    return tuple(results)


def run_coordination_grid(
    p_values: ArrayLike,
    q_values: ArrayLike,
    params: CoordinationParams,
    *,
    horizon: float = 30.0,
    separatrix_tolerance: float = 1e-8,
) -> BasinMap:
    """Return the deterministic Experiment-D basin grid."""

    return basin_map(
        p_values,
        q_values,
        params,
        horizon=horizon,
        separatrix_tolerance=separatrix_tolerance,
    )


__all__ = [
    "ScalingPathResult",
    "SelectionProfileResult",
    "run_coordination_grid",
    "run_scaling_paths",
    "run_selection_profiles",
]
