"""Simultaneous multi-interface compensation intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from .validity_gates import InterfaceValidityReport


@dataclass(frozen=True, slots=True)
class InterfaceGammaEstimate:
    interface_id: str
    cluster_gammas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.interface_id, str) or not self.interface_id:
            raise ValueError("interface_id must be a non-empty string")
        values = tuple(float(value) for value in self.cluster_gammas)
        if len(values) < 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("cluster_gammas must contain at least two finite values")
        object.__setattr__(self, "cluster_gammas", values)


@dataclass(frozen=True, slots=True)
class PartialIdentificationResult:
    valid_interfaces: tuple[str, ...]
    invalid_interfaces: tuple[str, ...]
    point_lower: float
    point_upper: float
    simultaneous_lower: float
    simultaneous_upper: float
    critical_value: float
    conclusion: Literal["robust_compensation", "robust_amplification", "not_identified"]


def robust_compensation_interval(
    interface_results: tuple[InterfaceGammaEstimate, ...],
    validity_reports: tuple[InterfaceValidityReport, ...],
    bootstrap_draws: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PartialIdentificationResult:
    """Build a max-stat simultaneous band over pre-valid interfaces only."""

    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if not 1_000 <= bootstrap_draws <= 1_000_000:
        raise ValueError("bootstrap_draws must be between 1000 and 1000000")
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise TypeError("confidence must be numeric")
    confidence_value = float(confidence)
    if not 0.5 < confidence_value < 1.0:
        raise ValueError("confidence must lie strictly between 0.5 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    estimates = {result.interface_id: result for result in interface_results}
    reports = {report.interface_id: report for report in validity_reports}
    if not estimates or set(estimates) != set(reports):
        raise ValueError("estimates and validity reports must cover identical interfaces")
    valid_ids = tuple(sorted(key for key, report in reports.items() if report.passed))
    invalid_ids = tuple(sorted(set(reports).difference(valid_ids)))
    if not valid_ids:
        raise ValueError("at least one interface must pass every validity gate")

    rng = np.random.default_rng(seed)
    points: list[float] = []
    standard_errors: list[float] = []
    bootstrap_means: list[np.ndarray] = []
    for interface_id in valid_ids:
        values = np.asarray(estimates[interface_id].cluster_gammas, dtype=np.float64)
        point = float(values.mean())
        standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
        indices = rng.integers(0, values.size, size=(bootstrap_draws, values.size))
        draws = values[indices].mean(axis=1)
        points.append(point)
        standard_errors.append(max(standard_error, np.finfo(np.float64).eps))
        bootstrap_means.append(draws)
    point_array = np.asarray(points)
    se_array = np.asarray(standard_errors)
    centered = np.column_stack(bootstrap_means) - point_array
    max_statistics = np.max(np.abs(centered / se_array), axis=1)
    critical = float(np.quantile(max_statistics, confidence_value, method="higher"))
    lowers = point_array - critical * se_array
    uppers = point_array + critical * se_array
    simultaneous_lower = float(np.min(lowers))
    simultaneous_upper = float(np.max(uppers))
    if simultaneous_upper < 0.0:
        conclusion = "robust_compensation"
    elif simultaneous_lower > 0.0:
        conclusion = "robust_amplification"
    else:
        conclusion = "not_identified"
    return PartialIdentificationResult(
        valid_interfaces=valid_ids,
        invalid_interfaces=invalid_ids,
        point_lower=float(np.min(point_array)),
        point_upper=float(np.max(point_array)),
        simultaneous_lower=simultaneous_lower,
        simultaneous_upper=simultaneous_upper,
        critical_value=critical,
        conclusion=conclusion,
    )
