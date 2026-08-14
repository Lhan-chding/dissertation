"""Frozen visual-acquisition weight, activation, and probe audits."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _hidden(value: Mapping[str, Sequence[Real]], label: str) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    result: dict[str, tuple[float, ...]] = {}
    for sample_id, vector in value.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} sample IDs must be non-empty strings")
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or not vector:
            raise ValueError(f"{label} vectors must be non-empty sequences")
        converted = tuple(float(item) for item in vector)
        if not all(math.isfinite(item) for item in converted):
            raise ValueError(f"{label} vectors must be finite")
        result[sample_id] = converted
    return result


def _unit(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return converted


@dataclass(frozen=True, slots=True)
class RepresentationInvarianceReport:
    passed: bool
    failed_gates: tuple[str, ...]
    weight_hash_equal: bool
    maximum_hidden_drift: float
    probe_drift: float
    sample_count: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failed_gates": list(self.failed_gates),
            "weight_hash_equal": self.weight_hash_equal,
            "maximum_hidden_drift": self.maximum_hidden_drift,
            "probe_drift": self.probe_drift,
            "sample_count": self.sample_count,
        }


def audit_representation_invariance(
    *,
    before_weight_sha256: str,
    after_weight_sha256: str,
    before_hidden: Mapping[str, Sequence[Real]],
    after_hidden: Mapping[str, Sequence[Real]],
    probe_before: float,
    probe_after: float,
    hidden_tolerance: float = 1e-8,
    probe_tolerance: float = 1e-8,
) -> RepresentationInvarianceReport:
    """Audit exact frozen weights plus numerical fixed-image representation drift."""

    if (
        _SHA256.fullmatch(before_weight_sha256) is None
        or _SHA256.fullmatch(after_weight_sha256) is None
    ):
        raise ValueError("vision weight hashes must be lowercase SHA-256 values")
    before = _hidden(before_hidden, "before_hidden")
    after = _hidden(after_hidden, "after_hidden")
    if set(before) != set(after):
        raise ValueError("before and after hidden mappings must cover identical sample IDs")
    for sample_id in before:
        if len(before[sample_id]) != len(after[sample_id]):
            raise ValueError("before and after hidden vectors must have identical dimensions")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (hidden_tolerance, probe_tolerance)
    ):
        raise ValueError("drift tolerances must be finite non-negative numbers")
    maximum_hidden_drift = max(
        abs(left - right)
        for sample_id in before
        for left, right in zip(before[sample_id], after[sample_id], strict=True)
    )
    probe_drift = abs(_unit(probe_before, "probe_before") - _unit(probe_after, "probe_after"))
    weight_equal = before_weight_sha256 == after_weight_sha256
    failed: list[str] = []
    if not weight_equal:
        failed.append("vision_weight_hash")
    if maximum_hidden_drift > float(hidden_tolerance):
        failed.append("hidden_representation_drift")
    if probe_drift > float(probe_tolerance):
        failed.append("probe_drift")
    return RepresentationInvarianceReport(
        passed=not failed,
        failed_gates=tuple(failed),
        weight_hash_equal=weight_equal,
        maximum_hidden_drift=maximum_hidden_drift,
        probe_drift=probe_drift,
        sample_count=len(before),
    )
