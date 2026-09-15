"""Auditable N3 development gates and immutable selection-lock payloads.

No training or filesystem mutation occurs here. The execution guard must verify
current files against the resulting identity binding before any fresh CPU work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_selection_lock(lock):
    """Detect any post-freeze change to configuration, gates or evidence."""
    if not isinstance(lock, dict) or "lock_sha256" not in lock:
        return False
    payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    try:
        return lock["lock_sha256"] == canonical_hash(payload)
    except (TypeError, ValueError):
        return False


def _gate(reasons):
    return {"status": "FAIL" if reasons else "PASS", "reasons": reasons}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stable(value):
    if value is True:
        return True
    return (
        isinstance(value, dict)
        and value.get("cluster_count", 0) >= 2
        and _number(value.get("ci975"))
        and value["ci975"] < 0
    )


def _finite_nonzero(row, methods):
    return (
        row.get("method") in methods
        and row.get("n") in (64, 256)
        and _number(row.get("actual_rank"))
        and row["actual_rank"] >= 1
        and _number(row.get("predicted_difference_norm"))
        and row["predicted_difference_norm"] > 0
        and row.get("model") not in ("C0_ZERO", "C1_LEGACY_TOTAL_DIFFERENCE")
    )


def _cost_rows(row, frontier, allowed_ratios):
    accepted = []
    for cost in frontier:
        if cost.get("configuration_id") != row.get("configuration_id"):
            continue
        if cost.get("access_regime") != row.get("access_regime"):
            continue
        if cost.get("includes_fit_score_label") is not True:
            continue
        ratio = cost.get("score_to_generation_ratio")
        if ratio not in allowed_ratios:
            continue
        # A break-even alone is extrapolation. It must lie in a declared valid
        # reuse domain, with full fit, score and semantic-label accounting.
        if cost.get("within_domain") is not True:
            continue
        break_even = cost.get("break_even_queries")
        feasible = cost.get("cost_feasible") is True or (_number(break_even) and break_even >= 0)
        if not feasible:
            continue
        baseline = cost.get("comparison_baseline")
        if baseline not in ("D_DIRECT", "D_KNOWN_EVENT_LOGP", "D_FULL_FINITE_ENUMERATION"):
            continue
        if row.get("access_regime") == "SAMPLE_ONLY" and baseline != "D_DIRECT":
            continue
        accepted.append(copy.deepcopy(cost))
    return accepted


def _strict_cost_advantage(cost):
    """A break-even intersection alone does not establish actual cost savings."""
    fit, direct = cost.get("calibration_cost"), cost.get("direct_cost_four_queries")
    if _number(fit) and _number(direct):
        return fit < direct
    return (
        cost.get("comparison_baseline") == "D_KNOWN_EVENT_LOGP"
        and cost.get("known_event_cost_advantage") is True
    )


def freeze_selection(
    candidates,
    *,
    protocol,
    input_hashes,
    source_hashes,
    resource_forecast,
    data_gate,
    cost_frontier=None,
    packet_hashes=None,
    selector_hashes=None,
):
    """Freeze at most two eligible finite methods and retain a nonzero failure.

    Candidate fields: configuration_id/model/method/access_regime/n/actual_rank,
    nonalias_eval_units/predicted_difference_norm/pX_nrmse/v_nrmse and paired
    stable_improvement_C0/C1 (bool evidence verdict or bootstrap result). A cost
    frontier row must identify configuration/access/ratio, full cost accounting,
    a direct comparison baseline and a break-even inside a justified reuse domain.

    Resource values are total projected walltime hours, peak_ram_gib,
    added_output_gib and temporary_gib; absent estimates fail the resource gate.
    Data gate requires PASS, legacy_complete and an explicit no-collision result.
    Every mapping of input/source/packet identities must be nonempty SHA-256.
    """
    rows = copy.deepcopy(list(candidates))
    frontier = copy.deepcopy(list(cost_frontier or []))
    packet_hashes = copy.deepcopy(packet_hashes or {})
    identifiers = [row.get("configuration_id") for row in rows]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(
        set(identifiers)
    ) != len(identifiers):
        raise ValueError("unique nonempty configuration_id required")
    methods = protocol["observation"]["methods"]
    stats = protocol["statistics"]
    audits = []
    for row in rows:
        reasons = []
        if (
            row.get("target", protocol["primary_target"]["name"])
            != protocol["primary_target"]["name"]
        ):
            reasons.append("NOT_PRIMARY_CANDIDATE_CONTRAST")
        if not _finite_nonzero(row, methods):
            reasons.append("NOT_FINITE_NONZERO_N64_OR_N256_CONFIGURATION")
        if not _number(row.get("nonalias_eval_units")) or row["nonalias_eval_units"] <= 0:
            reasons.append("NO_NONALIAS_EVALUATION_COVERAGE")
        for target in ("pX", "v"):
            metric = row.get(f"{target}_nrmse")
            if not _number(metric) or not 0 <= metric < stats[f"nrmse_{target}_max"]:
                reasons.append(f"{target}_NRMSE_FAIL_OR_NO_SIGNAL")
        if not all(_stable(row.get(f"stable_improvement_{baseline}")) for baseline in ("C0", "C1")):
            reasons.append("INCONCLUSIVE_BASELINE_IMPROVEMENT")
        acceptable_costs = _cost_rows(row, frontier, protocol["cost"]["score_to_generation_ratios"])
        known_event_win = any(
            cost["comparison_baseline"] == "D_KNOWN_EVENT_LOGP" and _strict_cost_advantage(cost)
            for cost in acceptable_costs
        )
        claim = (
            any(_strict_cost_advantage(cost) for cost in acceptable_costs)
            if row.get("access_regime") == "SAMPLE_ONLY"
            else known_event_win
        )
        audits.append(
            {
                **row,
                "science_pass": not reasons,
                "cost_pass": bool(acceptable_costs),
                "actual_cost_saving_claim": claim,
                "known_event_baseline_cost_advantage": known_event_win,
                "favorable_cost_ratios": sorted(
                    {cost["score_to_generation_ratio"] for cost in acceptable_costs}
                ),
                "cost_evidence": acceptable_costs,
                "reasons": reasons,
                "cost_reasons": []
                if acceptable_costs
                else ["NO_SAME_ACCESS_FULL_COST_INTERSECTION_WITHIN_VALID_DOMAIN"],
            }
        )
    ranked = sorted(
        [row for row in audits if _finite_nonzero(row, methods)],
        key=lambda row: (
            max(
                row[f"{target}_nrmse"] if _number(row.get(f"{target}_nrmse")) else math.inf
                for target in ("pX", "v")
            ),
            row["actual_rank"],
            row["n"],
            row["configuration_id"],
        ),
    )
    accepted = [row for row in ranked if row["science_pass"] and row["cost_pass"]]
    # Unique configuration choices, frozen solely on legacy development evidence.
    selected = copy.deepcopy(accepted[:2])
    science = _gate(
        []
        if any(row["science_pass"] for row in audits)
        else ["NO_ACCEPTABLE_FINITE_CONTRAST_MODEL"]
    )
    cost = _gate([] if accepted else ["NO_COST_FEASIBLE_SCIENTIFIC_CANDIDATE"])

    identity_reasons = []
    for name, mapping in (
        ("input", input_hashes),
        ("source", source_hashes),
        ("packet", packet_hashes),
    ):
        if (
            not isinstance(mapping, dict)
            or not mapping
            or not all(
                isinstance(key, str)
                and bool(key)
                and isinstance(value, str)
                and _SHA256.fullmatch(value)
                for key, value in mapping.items()
            )
        ):
            identity_reasons.append(f"MISSING_OR_INVALID_{name.upper()}_SHA256")
    identity = _gate(identity_reasons)
    data_reasons = []
    if data_gate.get("status") != "PASS" or data_gate.get("legacy_complete") is not True:
        data_reasons.append("LEGACY_INPUT_AUDIT_NOT_COMPLETE")
    if data_gate.get("fresh_seed_collision") is not False:
        data_reasons.append("FRESH_SEED_COLLISION_OR_UNVERIFIED")
    data = {**copy.deepcopy(data_gate), **_gate(data_reasons)}

    resource_reasons = []
    requirements = {
        "total_walltime_hours": "max_total_walltime_hours",
        "peak_ram_gib": "max_ram_gib",
        "added_output_gib": "max_added_output_gib",
        "temporary_gib": "max_temporary_gib",
    }
    for actual, maximum in requirements.items():
        value = resource_forecast.get(actual)
        if not _number(value) or value < 0 or value > protocol["resources"][maximum]:
            resource_reasons.append(f"UNVERIFIED_OR_EXCEEDED_{actual.upper()}")
    resources = {**_gate(resource_reasons), "forecast": copy.deepcopy(resource_forecast)}
    allowed = all(gate["status"] == "PASS" for gate in (science, cost, identity, data, resources))
    if identity["status"] != "PASS":
        status = "BLOCKED_INPUT_IDENTITY"
    elif data_gate.get("fresh_seed_collision") is True:
        status = "STOP_FOR_REVIEW"
    elif data["status"] != "PASS":
        status = "BLOCKED_MISSING_LEGACY_INPUTS_OR_SEED_AUDIT"
    elif resources["status"] != "PASS":
        status = "RESOURCE_REVIEW_REQUIRED"
    elif science["status"] != "PASS":
        status = "NO_ACCEPTABLE_MODEL"
    elif cost["status"] != "PASS":
        status = "NO_COST_FEASIBLE_SURROGATE"
    else:
        status = "FROZEN_FOR_CONDITIONAL_FRESH_CPU"
    protocol_file = Path(__file__).resolve().parents[2] / "configs/modeling_contrast/protocol.json"
    if protocol_file.is_file() and json.loads(protocol_file.read_text()) == protocol:
        protocol_sha256 = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    else:
        protocol_sha256 = canonical_hash(protocol)
    payload = {
        "protocol_version": protocol["protocol_version"],
        "status": status,
        "allow_fresh_cpu": allowed,
        "scope": "POSTHOC_LEGACY_DEVELOPMENT_SELECTION_ONLY",
        "selected": selected,
        "best_nonzero_diagnostic": copy.deepcopy(ranked[0]) if ranked else None,
        "ablation_fallback_rank_zero": False,
        "candidate_audit": audits,
        "science_gate": science,
        "cost_gate": cost,
        "resource_gate": resources,
        "input_identity_gate": identity,
        "data_gate": data,
        "mandatory_baselines": [
            "C0_ZERO",
            "C1_LEGACY_TOTAL_DIFFERENCE",
            "C3_CONTRAST_GLS_FULL",
            "D_DIRECT",
            "D_KNOWN_EVENT_LOGP",
            "D_FULL_FINITE_ENUMERATION",
        ],
        "statistics": copy.deepcopy(stats),
        "fresh_calibration_seeds": list(protocol["data_roles"]["fresh_calibration_seeds"]),
        "fresh_locked_test_seeds": list(protocol["data_roles"]["fresh_locked_test_seeds"]),
        "protocol_sha256": protocol_sha256,
        "source_hashes": copy.deepcopy(source_hashes),
        "input_hashes": copy.deepcopy(input_hashes),
        "packet_hashes": packet_hashes,
        "selector_source_hashes": copy.deepcopy(
            selector_hashes
            or {
                str(Path(__file__).resolve()): hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest()
            }
        ),
        "cost_frontier": frontier,
        "freeze_before_fresh_data_generation": True,
        "distribution_free_95_prediction_coverage_claim": False,
        "simultaneous_safety_certification": False,
    }
    payload["binding"] = {
        "source": canonical_hash(source_hashes),
        "config": protocol_sha256,
        "data": canonical_hash(input_hashes),
        "packet": canonical_hash(packet_hashes),
        "selector": canonical_hash(selected),
    }
    payload["lock_sha256"] = canonical_hash(payload)
    return payload
