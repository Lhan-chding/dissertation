"""Frozen preferences and interval decisions; overlap never means equivalence."""

from __future__ import annotations

import math
from collections.abc import Mapping

PREFERENCES = {
    "w0": {"pX": 1.0},
    "w1": {"pX": 0.8, "V": 0.1, "relation_score": 0.1},
    "w2": {"pX": 0.6, "V": 0.2, "relation_score": 0.2},
}


def _interval(value):
    if value is None:
        return None
    low, high = map(float, value)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("Finite ordered interval required")
    return low, high


def utility_interval(bounds: Mapping, preference: str = "w0") -> list[float] | None:
    """Weighted projection. Caller must supply simultaneously valid bounds."""
    weights = PREFERENCES[preference]
    terms = [(weight, _interval(bounds.get(metric))) for metric, weight in weights.items()]
    if any(interval is None for _, interval in terms):
        return None
    return [math.fsum(weight * interval[i] for weight, interval in terms) for i in (0, 1)]


def compare_candidate(
    group_intervals: Mapping,
    target_difference,
    *,
    required_groups,
    evidence_kind: str,
    tau_x: float = 0.01,
    epsilon: float = 0.01,
) -> dict:
    """Feasibility and target equivalence are distinct conclusions."""
    if (
        not evidence_kind
        or not required_groups
        or len(set(required_groups)) != len(required_groups)
    ):
        raise ValueError("Evidence source and unique prespecified groups required")
    if not math.isfinite(tau_x) or not math.isfinite(epsilon) or tau_x < 0 or epsilon < 0:
        raise ValueError("Nonnegative finite frozen tolerances required")
    intervals = {key: _interval(group_intervals.get(key)) for key in required_groups}
    missing = [key for key, value in intervals.items() if value is None]
    harm = [key for key, value in intervals.items() if value is not None and value[1] < -tau_x]
    feasible = not missing and all(value[0] >= -tau_x for value in intervals.values())
    target = _interval(target_difference)
    equivalent = target is not None and target[0] >= -epsilon and target[1] <= epsilon
    feasibility = "HARM_DETECTED" if harm else "NONINFERIOR_AT_SCALE" if feasible else "UNRESOLVED"
    conclusions = [feasibility]
    if equivalent:
        conclusions.append("EQUIVALENT_AT_SCALE")
    return {
        "status": feasibility,
        "conclusions": conclusions,
        "target_status": "EQUIVALENT_AT_SCALE" if equivalent else "UNRESOLVED",
        "feasibility": "INFEASIBLE" if harm else "FEASIBLE" if feasible else "UNKNOWN",
        "harm_groups": harm,
        "missing_groups": missing,
        "unresolved_groups": [
            key
            for key, value in intervals.items()
            if value is None or value[0] < -tau_x <= value[1]
        ],
        "evidence_kind": evidence_kind,
        "tau_X": tau_x,
        "epsilon": epsilon,
        "certification": "NOT_CERTIFIED"
        if evidence_kind == "CONDITIONAL_ON_SCORER"
        else "NO_ADDITIONAL_CERTIFICATION_CLAIM",
    }


def select_candidate(
    intervals: Mapping,
    statuses: Mapping,
    *,
    evidence_kind: str,
    simultaneous: bool,
) -> dict:
    """Maximize L among proven feasible items, retain UNKNOWN in regret."""
    if not simultaneous:
        raise ValueError("Selection regret requires simultaneous utility intervals")
    if not intervals or set(intervals) != set(statuses) or not evidence_kind:
        raise ValueError("Complete matching candidate registry and evidence source required")
    if any(status not in {"FEASIBLE", "INFEASIBLE", "UNKNOWN"} for status in statuses.values()):
        raise ValueError("Unknown feasibility label")
    bounds = {key: _interval(value) for key, value in intervals.items()}
    possible = [key for key in bounds if statuses[key] != "INFEASIBLE"]
    feasible = [key for key in possible if statuses[key] == "FEASIBLE" and bounds[key] is not None]
    selected = min(feasible, key=lambda key: (-bounds[key][0], key)) if feasible else None
    regret = None
    if selected is not None and all(bounds[key] is not None for key in possible):
        regret = max(
            [0.0] + [bounds[key][1] - bounds[selected][0] for key in possible if key != selected]
        )
    contenders = (
        possible
        if selected is None
        else [
            key for key in possible if bounds[key] is None or bounds[key][1] >= bounds[selected][0]
        ]
    )
    return {
        "selected": selected,
        "candidate_set": contenders,
        "unknown_candidates": [key for key in possible if statuses[key] == "UNKNOWN"],
        "regret_upper": regret,
        "regret_scope": "ALL_NOT_PROVEN_INFEASIBLE_CANDIDATES",
        "status": "UNRESOLVED" if selected is None or regret is None else "BOUNDED_SELECTION",
        "evidence_kind": evidence_kind,
        "simultaneous": True,
    }
