"""Pre-registered validity gates for operational mediator interfaces."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real


def _unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class InterfaceValidityThresholds:
    oracle_loss: float = 0.01
    replay_js: float = 0.05
    replay_accuracy_gap: float = 0.03
    image_exclusion_gap: float = 0.01
    parser_reliability: float = 0.95
    states_per_error: int = 200
    inputs_per_error: int = 50


_DEFAULT_THRESHOLDS = InterfaceValidityThresholds()


@dataclass(frozen=True, slots=True)
class InterfaceValidityReport:
    interface_id: str
    passed: bool
    failed_gates: tuple[str, ...]
    oracle_loss: float
    replay_js: float
    replay_accuracy_gap: float
    image_exclusion_gap: float
    parser_reliability: float
    minimum_natural_states: int
    minimum_natural_inputs: int


def _counts(value: Mapping[str, int], name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} counts must be non-negative integers")
        result[key] = count
    return result


def evaluate_interface_validity(
    *,
    interface_id: str,
    oracle_loss: float,
    replay_js: float,
    replay_accuracy_gap: float,
    image_exclusion_gap: float,
    parser_reliability: float,
    natural_state_count_by_error: Mapping[str, int],
    natural_input_count_by_error: Mapping[str, int],
    thresholds: InterfaceValidityThresholds = _DEFAULT_THRESHOLDS,
) -> InterfaceValidityReport:
    """Evaluate all gates together; no favorable interface may bypass one."""

    if not isinstance(interface_id, str) or not interface_id:
        raise ValueError("interface_id must be a non-empty string")
    metrics = {
        "oracle_consistency": _unit(oracle_loss, "oracle_loss"),
        "replay_js": _unit(replay_js, "replay_js"),
        "replay_accuracy_gap": _unit(replay_accuracy_gap, "replay_accuracy_gap"),
        "image_exclusion": _unit(image_exclusion_gap, "image_exclusion_gap"),
        "parser_reliability": _unit(parser_reliability, "parser_reliability"),
    }
    states = _counts(natural_state_count_by_error, "natural_state_count_by_error")
    inputs = _counts(natural_input_count_by_error, "natural_input_count_by_error")
    if set(states) != set(inputs):
        raise ValueError("natural support count mappings must have identical error families")
    minimum_states = min(states.values())
    minimum_inputs = min(inputs.values())
    failed: list[str] = []
    if metrics["oracle_consistency"] > thresholds.oracle_loss:
        failed.append("oracle_consistency")
    if (
        metrics["replay_js"] > thresholds.replay_js
        or metrics["replay_accuracy_gap"] > thresholds.replay_accuracy_gap
    ):
        failed.append("replay_fidelity")
    if metrics["image_exclusion"] > thresholds.image_exclusion_gap:
        failed.append("image_exclusion")
    if metrics["parser_reliability"] < thresholds.parser_reliability:
        failed.append("parser_reliability")
    if minimum_states < thresholds.states_per_error or minimum_inputs < thresholds.inputs_per_error:
        failed.append("natural_support")
    return InterfaceValidityReport(
        interface_id=interface_id,
        passed=not failed,
        failed_gates=tuple(failed),
        oracle_loss=metrics["oracle_consistency"],
        replay_js=metrics["replay_js"],
        replay_accuracy_gap=metrics["replay_accuracy_gap"],
        image_exclusion_gap=metrics["image_exclusion"],
        parser_reliability=metrics["parser_reliability"],
        minimum_natural_states=minimum_states,
        minimum_natural_inputs=minimum_inputs,
    )
