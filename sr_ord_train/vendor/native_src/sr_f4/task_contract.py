"""Explicit SR-F4 query extensions, preserving the native binary J definition.

The old scorer is untouched. ``selection_series`` extends TOPK's ranking column;
``include_end`` extends INTERVAL's endpoint. Legacy queries use the original
scorer verbatim. Extensions reuse its parsing/presentation/finish rules and only
replace the query-dependent answer and sufficient-evidence calculations.
"""

from __future__ import annotations

import copy
from fractions import Fraction

from sr_f3_ce import contract as native

VERSION = "srf4-query-v1"
operand = native.operand
aggregate = native.aggregate
rational = native.rational
encode = native.encode
parse_facts = native.parse_facts


def validate_query(world, query):
    native.validate_world(world)
    family = query["family"]
    if family not in native.FAMILIES:
        raise ValueError("Unknown family")
    if query.get("aggregate") not in {"count", "sum", "mean", "range", "min", "max"}:
        raise ValueError("Unknown aggregate")
    if query.get("value", "Alpha") not in {"Alpha", "Beta", "diff"}:
        raise ValueError("Unknown operand")
    n = len(world["categories"])
    if family == "TOPK":
        if query.get("selection_series", "Alpha") not in {"Alpha", "Beta"}:
            raise ValueError("Unknown ranking series")
        if type(query.get("k", 2)) is not int or not 0 <= query.get("k", 2) <= n:
            raise ValueError("Invalid top-k")
        if query.get("direction", "largest") not in {"largest", "smallest"}:
            raise ValueError("Unknown ranking direction")
    if family == "INTERVAL":
        if not all(type(query[k]) is int for k in ("start", "end")):
            raise ValueError("Interval positions must be integers")
        if not 0 <= query["start"] <= query["end"] < n:
            raise ValueError("Invalid interval")
        for key in ("include_start", "include_end", "positive_only"):
            if key in query and type(query[key]) is not bool:
                raise ValueError("Interval flags must be booleans")


def normalized_interval(query):
    result = copy.deepcopy(query)
    if result["family"] == "INTERVAL":
        result["end"] -= int(not result.pop("include_end", True))
    return result


def selected(world, query):
    validate_query(world, query)
    q = normalized_interval(query)
    if q["family"] == "TOPK" and q.get("selection_series", "Alpha") == "Beta":
        column = world["series"]["Beta"]
        sign = -1 if q.get("direction", "largest") == "largest" else 1
        return sorted(range(len(column)), key=lambda i: (sign * column[i], i))[: q.get("k", 2)]
    return native.selected(world, q)


def execute(world, query):
    return aggregate([operand(world, query, i) for i in selected(world, query)], query["aggregate"])


def independently_answer(world, query):
    """Separate table evaluator, without selection/operand/aggregate helpers."""
    validate_query(world, query)
    rows = list(
        zip(
            range(len(world["categories"])),
            world["series"]["Alpha"],
            world["series"]["Beta"],
            strict=True,
        )
    )
    family = query["family"]
    if family == "INTERVAL":
        lo = query["start"] + int(not query.get("include_start", True))
        hi = query["end"] - int(not query.get("include_end", True))
        rows = [r for r in rows if lo <= r[0] <= hi]
        if query.get("positive_only", False):
            rows = [r for r in rows if r[1] > r[2]]
    elif family == "TOPK":
        column = 1 if query.get("selection_series", "Alpha") == "Alpha" else 2
        sign = -1 if query.get("direction", "largest") == "largest" else 1
        rows.sort(key=lambda r: (sign * r[column], r[0]))
        rows = rows[: query.get("k", 2)]
    elif family == "CROSS":
        rows = [r for r in rows if r[1] >= r[2] + int(query.get("comparator", "gt") == "gt")]
    else:
        rows = [
            r
            for r in rows
            if r[1] >= query["threshold"] + int(query.get("comparator", "ge") == "gt")
        ]
    mode = query.get(
        "value", "Beta" if family == "TOPK" else "diff" if family == "INTERVAL" else "Alpha"
    )
    values = [
        Fraction(r[1] if mode == "Alpha" else r[2] if mode == "Beta" else r[1] - r[2]) for r in rows
    ]
    op = query["aggregate"]
    if op == "count":
        return Fraction(len(values))
    if op == "sum":
        return sum(values, Fraction())
    if not values:
        return None
    if op == "mean":
        return sum(values, Fraction()) / len(values)
    ordered = sorted(values)
    return {"range": ordered[-1] - ordered[0], "min": ordered[0], "max": ordered[-1]}[op]


def full_keys(world, query):
    return native.full_keys(world, normalized_interval(query))


def sufficient_template(world, query, facts):
    q = normalized_interval(query)
    if q["family"] != "TOPK" or q.get("selection_series", "Alpha") == "Alpha":
        return native.sufficient_template(world, q, facts)
    # Ranking must be justified by the reported Beta facts, never hidden values.
    n = len(world["categories"])
    if any(("Beta", i) not in facts for i in range(n)):
        return False
    sign = -1 if q.get("direction", "largest") == "largest" else 1
    chosen = sorted(range(n), key=lambda i: (sign * facts["Beta", i], i))[: q.get("k", 2)]
    return native._value_keys(q, chosen) <= set(facts)


def canonical_certificate(world, query):
    keys = full_keys(world, query)
    for key in sorted(keys):
        candidate = keys - {key}
        if sufficient_template(world, query, native.oracle_facts(world, candidate)):
            keys = candidate
    return [
        dict(series=s, category=world["categories"][i], value=world["series"][s][i])
        for s, i in sorted(keys)
    ]


def score(text, world, query, *, finish_reason="balanced", protocol="structured"):
    """Native J = value_correct * certificate * presentation * finish."""
    validate_query(world, query)
    if "include_end" not in query and "selection_series" not in query:
        return native.score(text, world, query, finish_reason=finish_reason, protocol=protocol)
    result = native.score(
        text, world, normalized_interval(query), finish_reason=finish_reason, protocol=protocol
    )
    truth_answer = execute(world, query)
    if protocol == "natural_work":
        if result["value_parseable"]:
            end = text.index("</final>") + len("</final>")
            result["value_correct"] = int(native.extract_final_tag(text[:end]) == truth_answer)
        return result
    end = native.first_object_end(text)
    if end is None:
        return result
    try:
        obj = native.loads(text[:end])
        answer = None if obj["answer"] is None else rational(obj["answer"])
        result["value_correct"] = int(answer == truth_answer)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, RecursionError):
        return result
    if protocol == "answer_only":
        return result
    try:
        facts = parse_facts(obj["evidence"], world)
        truth = all(v == world["series"][s][i] for (s, i), v in facts.items())
        suff = sufficient_template(world, query, facts)
        result.update(
            certificate_truth=int(truth),
            certificate_sufficient=int(suff),
            certificate=int(truth and suff),
            legacy_full=int(truth and full_keys(world, query) <= set(facts)),
        )
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        pass
    result["J"] = int(
        result["value_correct"]
        and result["certificate"]
        and result["presentation"]
        and result["finish"]
    )
    result["reason"] = (
        "ok"
        if result["J"]
        else "not_proved_by_registered_template"
        if result["certificate_truth"] and not result["certificate_sufficient"]
        else "contract_not_met"
    )
    return result
