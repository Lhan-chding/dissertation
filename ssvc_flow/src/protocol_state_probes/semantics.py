"""Raw-only strict parse, fixed inverse permutation, and separate audit features."""

import json
from fractions import Fraction

from .ptlc import anchors
from .transforms import FWD, emit, restore, valid_domain


def parse_four_ints(raw):
    if not isinstance(raw, str):
        return None
    try:
        values = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(values, list) or len(values) != 4 or any(type(v) is not int for v in values):
        return None
    return values


def value(world, operation):
    a, b, c, d = world
    if operation == "sum4":
        return a + b + c + d
    if operation == "difference_pairs":
        return a + b - c - d
    if operation == "range4":
        return max(world) - min(world)
    raise ValueError("Unknown downstream operation")


def event_label(raw, order, truth, operation):
    emitted = parse_four_ints(raw)
    if emitted is None:
        return "I", None
    canonical = restore(emitted, order)
    if not valid_domain(canonical):
        return "I", canonical
    if canonical == truth:
        return "X", canonical
    return ("S" if value(canonical, operation) == value(truth, operation) else "W"), canonical


def relation_values(H, b, world):
    if world is None:
        return None
    return [
        sum(Fraction(a) * x for a, x in zip(row, world, strict=True)) == Fraction(rhs)
        for row, rhs in zip(H, b, strict=True)
    ]


def _score(flags):
    return sum(flags) / len(flags) if flags else None


def _matches(world, program):
    return bool(
        world is not None and program["integer_prediction"] and world == program["canonical"]
    )


def score_raw(raw_text, case, audit, prediction=None):
    """Score entire text without repair; out-of-domain integers remain diagnostics.

    Controls have is_control=True. Copy/clean tasks have event=None and only an
    execution success target; duplicate controls retain repair events separately.
    Stop reasons intentionally do not affect parsing; the runner records them.
    """
    if audit["case_id"] != case["case_id"] or (
        prediction is not None and prediction["case_id"] != case["case_id"]
    ):
        raise ValueError("Mismatched case/audit/prediction identity")
    emitted = parse_four_ints(raw_text)
    y = restore(emitted, case["output_order"]) if emitted is not None else None
    parsed, legal = y is not None, valid_domain(y)
    is_control = case["panel"].startswith("CONTROL_")
    repair = case["kind"] == "repair"
    event = (
        event_label(raw_text, case["output_order"], audit["truth_world"], case["operation"])[0]
        if repair
        else None
    )
    observed = case["observed_world_public"]
    anchor_coordinates = anchors(case["H_display"], case["output_order"])
    flags_orig = relation_values(case["H_original"], case["b_original"], y)
    flags_display = relation_values(case["H_display"], case["b_display"], y)
    copy = bool(parsed and y == observed)
    programs = prediction["programs"] if prediction is not None else None
    matches = {
        variant: _matches(y, programs[variant]) if programs else None for variant in ("V1", "V2")
    }
    mask = (
        int(copy) + 2 * int(matches["V1"]) + 4 * int(matches["V2"]) if parsed and programs else None
    )
    anchor_mask = (
        [y[k] == observed[k] if k in anchor_coordinates else None for k in FWD] if parsed else None
    )
    anchor_all = all(y[k] == observed[k] for k in anchor_coordinates) if parsed else None
    positions = {k: index for index, k in enumerate(case["output_order"])}
    last_coordinates = [
        max((k for k, a in enumerate(row) if a), key=positions.get) for row in case["H_display"]
    ]
    local_groups = (
        {
            str(k): [flags_display[i] for i, last in enumerate(last_coordinates) if last == k]
            for k in case["output_order"]
        }
        if parsed
        else None
    )
    expected = audit["control_expected_canonical"]
    if is_control and expected is None:
        expected = audit["truth_world"]
    features = {
        "event": event,
        "E": event,
        "is_control": is_control,
        "control_kind": case["panel"] if is_control else None,
        "X": bool(event == "X") if repair else None,
        "A": bool(event in ("X", "S")) if repair else None,
        "V": legal,
        "parse_four_ints": parsed,
        "domain_valid": legal,
        "in_domain": legal,
        "emitted_world": emitted,
        "canonical_world": y,
        "copy_parsed": copy,
        "copy": copy,
        "match_V1": matches["V1"],
        "match_V2": matches["V2"],
        "match_V1_in_domain": bool(legal and matches["V1"]) if programs else None,
        "match_V2_in_domain": bool(legal and matches["V2"]) if programs else None,
        "match_mask": mask,
        "outside_copy_ptlc_union": bool(parsed and mask == 0) if programs else None,
        "other": bool(parsed and mask == 0) if programs else None,
        "anchor_coordinates": anchor_coordinates,
        "anchor_mask": anchor_mask,
        "anchor_all": anchor_all,
        "anchor_all_parsed": bool(parsed and anchor_all),
        "C_orig": _score(flags_orig) if legal else 0.0,
        "C_display": _score(flags_display) if legal else 0.0,
        "C_alg_orig": _score(flags_orig) if parsed else None,
        "C_alg_display": _score(flags_display) if parsed else None,
        "relation_flags_orig": flags_orig,
        "relation_flags_display": flags_display,
        "relation_last_coordinates": last_coordinates,
        "local_relation_flags_by_coordinate": local_groups,
        "edit_mask": sum((1 << k) for k in FWD if y[k] != observed[k]) if parsed else None,
        "F": None,
        "B": None,
        "M": None,
        "control_success": bool(legal and y == expected) if is_control else None,
        "control_default_forward_output": bool(parsed and emitted == expected)
        if is_control
        else None,
        "control_expected_emitted": emit(expected, case["output_order"]) if is_control else None,
    }
    features["C"] = features["C_orig"]
    if repair and legal:
        truth, j = audit["truth_world"], audit["corrupted_coordinate"]
        if [k for k in FWD if truth[k] != observed[k]] != [j]:
            raise ValueError("Repair audit requires exactly its registered single corruption")
        features.update(
            F=int(y[j] == truth[j]),
            B=sum(y[k] != truth[k] for k in FWD if k != j),
            M=sum(y[k] != observed[k] for k in FWD),
        )
    if prediction is not None and "b1_diagnostic" in prediction:
        diagnostic = prediction["b1_diagnostic"]
        features.update(
            b1_risk_category=diagnostic["change_category"],
            first_leaf_risk=diagnostic["first_leaf_risk"],
        )
        for variant in ("V1", "V2"):
            features[f"b1_predictions_changed_{variant}"] = diagnostic["changed"][variant]
            for label in ("old", "new"):
                features[f"match_{variant}_{label}"] = _matches(y, diagnostic[label][variant])
                features[f"match_{variant}_{label}_in_domain"] = bool(
                    legal and _matches(y, diagnostic[label][variant])
                )
    return features
