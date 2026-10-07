"""Frozen SER-J2 output semantics; public projection never reads audit truth."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any


def parse(text: str) -> list[int] | None:
    """Whole-string JSON4, accepting exact integers only (not bool/float)."""
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return None
    return (
        value
        if (isinstance(value, list) and len(value) == 4 and all(type(v) is int for v in value))
        else None
    )


def valid(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(type(v) is int and 0 <= v <= 99 for v in value)
    )


def _public_contract(task: Mapping[str, Any]) -> None:
    if not valid(task["observed"]):
        raise ValueError("Public observation must be legal JSON4")
    if task.get("legal_domain", [0, 99]) != [0, 99]:
        raise ValueError("SER-J2 domain must remain 0..99")
    H, b = task["H_original"], task["b_original"]
    if (
        not H
        or len(H) != len(b)
        or any(len(row) != 4 or any(type(v) is not int for v in row) for row in H)
        or any(type(v) is not int for v in b)
    ):
        raise ValueError("Malformed public linear relationships")


def satisfy(value: Sequence[int], H: Sequence[Sequence[int]], b: Sequence[int]) -> bool:
    return all(
        sum(v * c for v, c in zip(value, row, strict=True)) == rhs
        for row, rhs in zip(H, b, strict=True)
    )


def verify(task: Mapping[str, Any], value: Any) -> bool:
    """Public legal-domain, all-relationship, exactly-one-edit verifier."""
    _public_contract(task)
    return bool(
        valid(value)
        and sum(v != o for v, o in zip(value, task["observed"], strict=True)) == 1
        and satisfy(value, task["H_original"], task["b_original"])
    )


def deletion_projection(task: Mapping[str, Any], value: Any) -> dict[str, Any]:
    """At most four original proposed values, no truth and no value search."""
    _public_contract(task)
    started = time.process_time()
    found: set[tuple[int, ...]] = set()
    calls = 0
    parsed = (
        isinstance(value, (list, tuple)) and len(value) == 4 and all(type(v) is int for v in value)
    )
    if parsed:
        for k, v in enumerate(value):
            candidate = list(task["observed"])
            candidate[k] = v
            calls += 1
            if verify(task, candidate):
                found.add(tuple(candidate))
    if len(found) > 1:
        raise ValueError("Unique single-repair task contract violated")
    return {
        "projection_success": bool(found),
        "projected_vector": list(next(iter(found))) if found else None,
        "projection_verifier_calls": calls,
        "projection_cpu_seconds": time.process_time() - started,
    }


def public_center(task: Mapping[str, Any]) -> int | None:
    """Recover the cross-star hub from public relationships alone."""
    if task["family"] != "cross_series":
        return None
    rows = task["H_original"]
    if len(rows) != 3 or any(
        sum(v == 1 for v in row) != 2 or any(v not in (0, 1) for v in row) for row in rows
    ):
        raise ValueError("Cross task is not a pair-sum star")
    centers = [k for k in range(4) if all(row[k] == 1 for row in rows)]
    if len(centers) != 1 or any(
        sum(row[k] for row in rows) != 1 for k in range(4) if k not in centers
    ):
        raise ValueError("Cross relationships must form one complete star")
    return centers[0]


def mixed_anchor_public(task: Mapping[str, Any], value: Any) -> list[bool]:
    """Per-coordinate public incompatible-old-hub anchor signatures."""
    bits = [False] * 4
    if value is None:
        return bits
    h = public_center(task)
    observed = task["observed"]
    if h is None or value[h] == observed[h]:
        return bits
    for row, rhs in zip(task["H_original"], task["b_original"], strict=True):
        k = next(k for k in range(4) if k != h and row[k] == 1)
        bits[k] = value[k] != observed[k] and value[k] == rhs - observed[h]
    return bits


def operand_signatures(
    task: Mapping[str, Any], audit: Mapping[str, Any], value: Any
) -> list[dict[str, Any]]:
    """Overlapping arithmetic coincidences, never inferred internal execution.

    All wrong proposed leaf values are retained, with right-location-single-edit
    and forward-target flags available for reproducing narrower historical views.
    """
    h = public_center(task)
    if h is None or value is None:
        return []
    observed, truth = task["observed"], audit["true_world"]
    edit = [k for k in range(4) if value[k] != observed[k]]
    signatures = []
    for row_index, (row, rhs) in enumerate(
        zip(task["H_original"], task["b_original"], strict=True)
    ):
        k = next(k for k in range(4) if k != h and row[k] == 1)
        wrong = value[k] != truth[k]
        partners = [
            q for q in range(4) if q not in (h, k) and wrong and value[k] == rhs - observed[q]
        ]
        other_constants = [
            q
            for q, b in enumerate(task["b_original"])
            if q != row_index and wrong and value[k] == b - observed[h]
        ]
        signatures.append(
            {
                "coordinate": k,
                "proposed_value": value[k],
                "wrong_value": wrong,
                "edited": k in edit,
                "target_leaf": k == audit["corrupted_index"],
                "forward_target": k == audit["corrupted_index"] and h > k,
                "right_location_single_edit": edit == [audit["corrupted_index"]],
                "wrong_partner": bool(partners),
                "wrong_partner_indices": partners,
                "other_relation_constant": bool(other_constants),
                "other_relation_indices": other_constants,
                "off_by_2": wrong and abs(value[k] - truth[k]) <= 2,
                "equals_hub": wrong and value[k] == observed[h],
            }
        )
    return signatures


def downstream(value: Sequence[int], operation: str) -> int:
    if operation == "sum4":
        return sum(value)
    if operation == "difference_pairs":
        return value[0] + value[1] - value[2] - value[3]
    if operation == "range4":
        return max(value) - min(value)
    raise ValueError("Unregistered downstream operation")


def score_output(
    task: Mapping[str, Any], audit: Mapping[str, Any], raw_text: str, stop_reason: str | None = None
) -> dict[str, Any]:
    _public_contract(task)
    audit_root = audit.get("root_id", audit.get("base_instance_id"))
    if task["task_id"] != audit["task_id"] or task["root_id"] != audit_root:
        raise ValueError("Public/audit identity mismatch")
    truth, observed, j = audit["true_world"], task["observed"], audit["corrupted_index"]
    if (
        not valid(truth)
        or type(j) is not int
        or j not in range(4)
        or [k for k in range(4) if truth[k] != observed[k]] != [j]
        or not verify(task, truth)
    ):
        raise ValueError("Invalid single-corruption audit contract")
    h = public_center(task)
    if h != audit.get("center"):
        raise ValueError("Audit/public center mismatch")
    y = parse(raw_text)
    legal = valid(y)
    event = "I"
    if legal:
        event = (
            "X"
            if y == truth
            else (
                "S"
                if downstream(y, task["operation"]) == downstream(truth, task["operation"])
                else "W"
            )
        )
    mask = [v != o for v, o in zip(y, observed, strict=True)] if y is not None else None
    value_present = y is not None and y[j] == truth[j]
    extra = value_present and event != "X"
    M = sum(mask) if legal else None
    L = int(mask[j]) if legal else None
    B = M - L if legal else None
    F = int(value_present) if legal else None
    if legal and not (M == B + L and F <= L):
        raise AssertionError("M=B+L and F<=L must hold for legal outputs")
    if int(event == "X") != int(value_present) - int(extra):
        raise AssertionError("Exact value/extra decomposition failed")
    public_bits = mixed_anchor_public(task, y)
    audit_bits = [v and h == j and value_present for v in public_bits]
    center_correct = y is not None and h is not None and y[h] == truth[h]
    projection = deletion_projection(task, y)
    if projection["projection_success"] != value_present:
        raise ValueError("Projection equivalence failed; check unique-repair task contract")
    result = {
        "task_id": task["task_id"],
        "root_id": task["root_id"],
        "family": task["family"],
        "root_cohort": audit.get("root_cohort"),
        "center": h,
        "corrupted_index": j,
        "parsed_vector": y,
        "parse_ok": y is not None,
        "domain_ok": legal,
        "out_of_domain": y is not None and not legal,
        "event": event,
        **{name: event == name for name in "XSWI"},
        "verifier_pass": verify(task, y),
        "edit_mask": mask,
        "M": M,
        "B": B,
        "L": L,
        "F": F,
        "value_present": value_present,
        "value_present_but_not_X": extra,
        "mixed_anchor_public": any(public_bits),
        "mixed_anchor_public_bits": public_bits,
        "mixed_anchor_audit": any(audit_bits),
        "mixed_anchor_audit_bits": audit_bits,
        "center_correct": center_correct,
        "center_correct_but_not_X": center_correct and event != "X",
        "leaf_edited_and_center_correct": [
            bool(center_correct and k != h and mask[k]) if mask is not None else False
            for k in range(4)
        ],
        "operand_signatures": operand_signatures(task, audit, y),
        "truncated": stop_reason in ("length", "max_tokens", "max_new_tokens"),
        **projection,
    }
    return result


parse_json4 = parse
