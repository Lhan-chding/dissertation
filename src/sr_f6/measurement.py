"""SR-F6 field-local measurements and a lossless adapter to the native F5 J.

Extraction never receives gold. A readable field can remain measurable when a
different field invalidates the common protocol; credit eligibility still
requires the complete protocol and generation. No invalid response earns J.
"""

from __future__ import annotations

import json
from decimal import Decimal

from sr_f4 import task_contract
from sr_f5.measurement import score_response as native_score
from srf5_ref.oracle import evaluate_a
from srf6_ref.core import enc, exact, oracle_a, reduce_values

VERSION = "srf6-native-measurement-v1"
REQUIRED = {"evidence", "selected", "bindings", "aggregate", "answer"}
MAX_TEXT = 100_000


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("nonfinite JSON")


def _object(text):
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise ValueError("text bound")
    obj = json.loads(
        text, object_pairs_hook=_unique, parse_float=Decimal, parse_constant=_nonfinite
    )
    if not isinstance(obj, dict):
        raise ValueError("one JSON object required")
    return obj


def _selected(value):
    if not isinstance(value, list) or not all(type(x) is str for x in value):
        raise ValueError("selected shape")
    return value


def _bindings(value):
    if not isinstance(value, list):
        raise ValueError("bindings shape")
    for entry in value:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"category", "operand"}
            or not isinstance(entry["category"], str)
        ):
            raise ValueError("binding entry")
        exact(entry["operand"])
    return value


def _valid_protocol(obj):
    if not obj.keys() >= REQUIRED or set(obj) - REQUIRED - {"work"}:
        raise ValueError("top-level schema")
    _selected(obj["selected"])
    _bindings(obj["bindings"])
    if obj["aggregate"] not in ("sum", "mean"):
        raise ValueError("aggregate shape")
    if "work" in obj and not isinstance(obj["work"], str):
        raise ValueError("work shape")
    if not isinstance(obj["evidence"], list):
        raise ValueError("evidence shape")
    for entry in obj["evidence"]:
        if not isinstance(entry, dict) or set(entry) != {"series", "category", "value"}:
            raise ValueError("evidence entry")
        if not isinstance(entry["series"], str) or not isinstance(entry["category"], str):
            raise ValueError("evidence identifiers")
        exact(entry["value"])
    if obj["answer"] is not None:
        exact(obj["answer"])


def parse(text):
    """Return a gold-blind object independently of common-protocol validity."""
    obj = None
    try:
        obj = _object(text)
        _valid_protocol(obj)
        return {"valid": True, "object": obj, "json_valid": True, "reason": None}
    except (ValueError, TypeError, ArithmeticError, RecursionError) as exc:
        return {
            "valid": False,
            "object": obj,
            "json_valid": obj is not None,
            "reason": str(exc),
        }


def _delete_member(text, target):
    """Delete one validated raw member and separator; never serialize values."""
    decoder = json.JSONDecoder(parse_float=Decimal)
    pos = len(text) - len(text.lstrip()) + 1
    previous_comma = None
    while True:
        while text[pos].isspace():
            pos += 1
        member_start = pos
        key, pos = decoder.raw_decode(text, pos)
        while text[pos].isspace():
            pos += 1
        pos += 1  # Already validated colon.
        while text[pos].isspace():
            pos += 1
        _, value_end = decoder.raw_decode(text, pos)
        pos = value_end
        while text[pos].isspace():
            pos += 1
        if key == target:
            if text[pos] == ",":
                return text[:member_start] + text[pos + 1 :]
            if previous_comma is None:
                return text[:member_start] + text[value_end:]
            return text[:previous_comma] + text[value_end:]
        if text[pos] != ",":
            raise ValueError("projection member missing: " + target)
        previous_comma = pos
        pos += 1


def lossless_legacy_projection(text):
    """Remove only bindings/aggregate after validating the complete F6 object."""
    parsed = parse(text)
    if not parsed["valid"]:
        raise ValueError("Invalid SR-F6 protocol: " + parsed["reason"])
    return _delete_member(_delete_member(text, "bindings"), "aggregate")


def answer_only(text, expected):
    """Exact A from one unambiguous JSON object, independent of added fields."""
    try:
        return int(exact(_object(text)["answer"]) == expected)
    except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError):
        return 0


def score_response(text, world, query, finished=True, *, finish_reason=None):
    """Measure observable claims; native J is old F5 J intersected with F6 V.

    S/B/P/E are diagnostics even when V=0. Downstream credit must also require
    V and finished; ``credit_eligible`` and ``eligible_components`` expose that
    boundary. Incomplete generation leaves semantic labels UNKNOWN, even when
    a syntactically complete prefix happens to be present. A is still separate.
    """
    checked = evaluate_a(world, query)  # Reject unknown task fields at the boundary.
    gold = oracle_a(world, query)
    if checked["answer"] != gold["answer"] or checked["selected"] != gold["selected"]:
        raise ValueError("native and SR-F6 task oracle disagree")
    parsed = parse(text)
    answer = answer_only(text, gold["answer"])
    reason = (finish_reason or "balanced") if finished else "length"
    completed = bool(finished) and reason in ("balanced", "eos")
    result = {
        "A": answer,
        "V": int(parsed["valid"]),
        "J": 0,
        "J_M3": 0,
        "J_legacy_projection": None,
        "finished": int(completed),
        "finish": int(completed),
        "finish_reason": reason,
        "S": None,
        "B": None,
        "P": None,
        "E_raw": None,
        "E_anchored": None,
        "consistency": None,
        "reason": parsed["reason"],
        "measurement_version": VERSION,
        "primary_answer_parse_status": "INVALID_OR_MISSING",
        "unknown_reasons": {},
        "witness": {"gold_selected": gold["selected"], "gold_answer": enc(gold["answer"])},
    }
    obj = parsed["object"]
    if obj is not None:
        try:
            exact(obj["answer"])
            result["primary_answer_parse_status"] = "KNOWN"
        except (ValueError, TypeError, KeyError, ArithmeticError):
            pass
    if parsed["valid"]:
        old = native_score(
            lossless_legacy_projection(text),
            world,
            query,
            finished=finished,
            finish_reason=finish_reason,
        )
        for key in (
            "certificate",
            "certificate_truth",
            "certificate_sufficient",
            "legacy_full",
            "value_parseable",
            "presentation",
            "reported_facts",
        ):
            if key in old:
                result[key] = old[key]
        result["J_legacy_projection"] = old["J"]
        result["J"] = result["J_M3"] = int(parsed["valid"] and old["J"])
        result["finish"] = old["finish"]
        result["finished"] = old["finish"]
        result["native_reason"] = old["reason"]
    if obj is not None and result["finished"]:
        _measure_fields(result, obj, world, query, gold)
    else:
        why = "INCOMPLETE_GENERATION" if not result["finished"] else parsed["reason"]
        result["unknown_reasons"].update(dict.fromkeys(("S", "B", "P", "E_raw", "E_anchored"), why))
    result["known_mask"] = {key: result[key] is not None for key in ("S", "B")}
    result["credit_eligible"] = bool(result["V"] and result["finished"] and result["finish"])
    result["eligible_components"] = {
        key: result["credit_eligible"] and result["known_mask"][key] for key in ("S", "B")
    }
    return result


def _measure_fields(result, obj, world, query, gold):
    selected = bindings = None
    try:
        selected = _selected(obj["selected"])
        result["S"] = int(
            len(selected) == len(set(selected)) and set(selected) == set(gold["selected"])
        )
        result["witness"]["selected"] = selected
    except (ValueError, TypeError, KeyError) as exc:
        result["unknown_reasons"]["S"] = str(exc)
    try:
        bindings = _bindings(obj["bindings"])
        names = [entry["category"] for entry in bindings]
        target = dict(zip(gold["selected"], gold["operands"], strict=True))
        result["B"] = int(
            len(names) == len(set(names))
            and set(names) == set(target)
            and all(exact(entry["operand"]) == target[entry["category"]] for entry in bindings)
        )
        result["witness"]["gold_bindings"] = {key: enc(value) for key, value in target.items()}
        result["witness"]["bindings"] = [
            {"category": entry["category"], "operand": enc(exact(entry["operand"]))}
            for entry in bindings
        ]
        if selected is not None:
            result["consistency"] = int(
                len(names) == len(set(names))
                and len(selected) == len(set(selected))
                and set(names) == set(selected)
            )
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        result["unknown_reasons"]["B"] = str(exc)
    try:
        facts = task_contract.parse_facts(obj["evidence"], world)
        if not facts:
            raise ValueError("no submitted facts")
        result["P"] = int(all(value == world["series"][s][i] for (s, i), value in facts.items()))
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        result["unknown_reasons"]["P"] = str(exc)
    try:
        if bindings is None:
            raise ValueError("bindings unavailable")
        if obj.get("aggregate") not in ("sum", "mean"):
            raise ValueError("aggregate unavailable")
        value = exact(obj["answer"])
        computed = reduce_values([exact(entry["operand"]) for entry in bindings], obj["aggregate"])
        result["E_raw"] = int(computed is not None and computed == value)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        result["unknown_reasons"]["E_raw"] = str(exc)
    if result["B"] is not None and obj.get("aggregate") in ("sum", "mean"):
        result["E_anchored"] = int(
            result["B"] == 1 and obj["aggregate"] == query["aggregate"] and result["A"] == 1
        )
    else:
        result["unknown_reasons"]["E_anchored"] = "binding or aggregate unavailable"


labels = score_response
legacy_projection = lossless_legacy_projection
