"""SER-J23 answer events, retaining the native SER-J2 arithmetic scorer.

Only analysis/auditor callers may provide ``audit``.  The public deletion
projection uses at most the four values already proposed in the answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from ..exposure_substitution import semantics as native

# These public operations are unchanged from the historical runner.
parse = native.parse
parse_json4 = native.parse_json4
valid = native.valid
verify = native.verify
satisfy = native.satisfy
downstream = native.downstream
deletion_projection = native.deletion_projection
public_center = native.public_center
mixed_anchor_public = native.mixed_anchor_public
operand_signatures = native.operand_signatures


@lru_cache(maxsize=4096)
def _unique_repair(observed, relationships, rhs):
    """Validate the public task once; this is not answer projection/search."""
    solutions = []
    for k in range(4):
        for value in range(100):
            if value == observed[k]:
                continue
            candidate = list(observed)
            candidate[k] = value
            if native.satisfy(candidate, relationships, rhs):
                solutions.append(tuple(candidate))
    if len(solutions) != 1:
        raise ValueError("Unique public single-repair task contract violated")
    return solutions[0]


def score_output(
    task: Mapping[str, Any],
    audit: Mapping[str, Any],
    raw_text: str,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Score one complete response without discarding malformed/out-of-domain answers.

    ``task`` uses historical public fields (observed, H_original, b_original,
    operation, family, task_id, root_id). ``audit`` supplies true_world, center,
    corrupted_index (all coordinates zero-based), and optional root_cohort.
    ``F_legal`` retains the native nullable legal-answer field; ``F_val`` and
    ``R`` are events over ALL answers, including out-of-domain parsed answers.
    ``q_events``/``r_events`` are four Boolean numerator indicators. Conditional
    edit probabilities are formed by aggregate_events, never per-answer division.
    """
    native._public_contract(task)
    truth = audit["true_world"]
    if not native.valid(truth):
        raise ValueError("Audit truth must be four legal exact integers")
    # Validate the operation even when the answer is malformed or exactly correct.
    native.downstream(truth, task["operation"])
    unique = _unique_repair(
        tuple(task["observed"]),
        tuple(tuple(row) for row in task["H_original"]),
        tuple(task["b_original"]),
    )
    if unique != tuple(truth):
        raise ValueError("Audit truth differs from the unique public repair")
    # Native scoring compares list values; copy caller-owned audit data rather
    # than mutating it or letting a tuple truth turn a correct answer into S.
    result = native.score_output(task, {**audit, "true_world": list(truth)}, raw_text, stop_reason)
    y, x = result["parsed_vector"], result["X"]
    edits = result["edit_mask"] or [False] * 4
    result.update(
        F_val=result.pop("value_present"),
        F_legal=result.pop("F"),
        R=result.pop("value_present_but_not_X"),
        q_events=list(edits),
        r_events=[bool(edited and not x) for edited in edits],
        multi_edit_parse=bool(y is not None and sum(edits) > 1),
        multi_edit_legal=bool(result["domain_ok"] and sum(edits) > 1),
        copy_observed=bool(y is not None and y == task["observed"]),
        out_of_domain_bits=[bool(y is not None and not 0 <= y[k] <= 99) for k in range(4)],
        mixed_anchor_interpretation="output_match_only_not_identified_internal_rule",
    )
    validate_events(result)
    return result


EVENT_FIELDS = (
    "X",
    "S",
    "W",
    "I",
    "F_val",
    "R",
    "parse_ok",
    "domain_ok",
    "out_of_domain",
    "truncated",
    "multi_edit_parse",
    "multi_edit_legal",
    "copy_observed",
)
VECTOR_FIELDS = ("q_events", "r_events", "out_of_domain_bits")


def _boolean(value):
    return type(value) is bool or (type(value) is int and value in (0, 1))


def validate_events(row: Mapping[str, Any]) -> None:
    """Reject corrupted score records before aggregation; invalid answers remain valid rows."""
    if any(not _boolean(row.get(field)) for field in EVENT_FIELDS):
        raise ValueError("Scored events must be exact Boolean/0/1 indicators")
    if any(
        not isinstance(row.get(field), (list, tuple))
        or len(row[field]) != 4
        or any(not _boolean(value) for value in row[field])
        for field in VECTOR_FIELDS
    ):
        raise ValueError("Four exact event indicators required per coordinate")
    j = row.get("corrupted_index")
    if type(j) is not int or j not in range(4):
        raise ValueError("Zero-based corrupted_index must be an exact integer")
    x, parsed, domain = bool(row["X"]), bool(row["parse_ok"]), bool(row["domain_ok"])
    if sum(row[name] for name in "XSWI") != 1 or bool(row["I"]) == domain:
        raise ValueError("X/S/W/I must partition all answers with I = not domain_ok")
    if row.get("event", next(name for name in "XSWI" if row[name])) not in tuple("XSWI"):
        raise ValueError("Unknown answer event")
    if "event" in row and not row[row["event"]]:
        raise ValueError("Answer event label disagrees with indicators")
    if int(x) != int(row["F_val"]) - int(row["R"]):
        raise ValueError("X = F_val - R identity violated")
    if domain and not parsed:
        raise ValueError("Legal-domain answers must parse")
    if bool(row["out_of_domain"]) != (parsed and not domain):
        raise ValueError("Out-of-domain answers must be retained explicitly")
    if bool(row["F_val"]) and not parsed:
        raise ValueError("F_val requires a parsed answer")
    for k in range(4):
        q, r = int(row["q_events"][k]), int(row["r_events"][k])
        if q != r + int(x and k == j) or r != int(bool(q) and not x):
            raise ValueError("q/r/X single-corruption identity violated")
        if (q or row["out_of_domain_bits"][k]) and not parsed:
            raise ValueError("Coordinate events require a parsed answer")
    if any(row["out_of_domain_bits"]) != bool(row["out_of_domain"]):
        raise ValueError("Coordinate out-of-domain indicators disagree")
    multi = sum(row["q_events"]) > 1
    if bool(row["multi_edit_parse"]) != multi or bool(row["multi_edit_legal"]) != (
        domain and multi
    ):
        raise ValueError("Multi-edit indicators disagree with parse/domain/edit events")
    if bool(row["copy_observed"]) != (parsed and not any(row["q_events"])):
        raise ValueError("Copy-observed indicator disagrees with edit events")
    expected_legal = int(bool(row["F_val"])) if domain else None
    if row.get("F_legal") != expected_legal:
        raise ValueError("Native F_legal must remain nullable outside the legal domain")
    if "projection_success" in row and bool(row["projection_success"]) != bool(row["F_val"]):
        raise ValueError("Public deletion projection must equal F_val")


def aggregate_events(rows) -> dict[str, Any]:
    """All-answer event rates and exact edit denominators for a nonempty group.

    Each entry has equal weight here. Formal inference first calls this within
    each task (K=8), then uses fixed root/cell/path weights in statistics.py.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot summarize an empty answer group")
    for row in rows:
        validate_events(row)
    n = len(rows)
    counts = {field: sum(int(row[field]) for row in rows) for field in EVENT_FIELDS}
    result = {field: count / n for field, count in counts.items()}
    result.update({field + "_count": count for field, count in counts.items()})
    result["F_legal_count"] = sum(row["F_legal"] == 1 for row in rows)
    result["F_legal"] = result["F_legal_count"] / n
    result["n_draws"] = result["denominator_all_answers"] = n
    non_x = n - counts["X"]
    result["denominator_not_X"] = non_x
    for name, field in (
        ("q", "q_events"),
        ("r", "r_events"),
        ("out_of_domain_by_position", "out_of_domain_bits"),
    ):
        vector = [sum(int(row[field][k]) for row in rows) for k in range(4)]
        result[name + "_counts"] = vector
        result[name] = [count / n for count in vector]
    result["c"] = [count / non_x if non_x else None for count in result["r_counts"]]
    result["c_undefined_reason"] = "no_non_X_answers" if non_x == 0 else None
    result["F_legal_denominator"] = "all_answers; native invalid values remain null per answer"
    if abs(result["X"] - result["F_val"] + result["R"]) > 1e-12:
        raise AssertionError("Aggregate value/extra decomposition failed")
    return result
