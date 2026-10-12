"""Versioned exact scoring; the authenticated SR-F1 scorer stays unchanged.

The only answer-language extension is ``-?digits.digits`` inside a JSON string.
The blind syntax classifier accepts no world, query, or gold-answer arguments.
"""

from __future__ import annotations

import re
from decimal import Decimal

from sr_f1.contract import (
    execute,
    fraction_of,
    gold_evidence,
    load_json_strict,
    reference_module,
)

VERSION = "SR-F2-scorer-v2"
SYNTAX_CLASSIFIER_VERSION = "SR-F2-blind-syntax-v2"
DECIMAL_STRING_PATTERN = r"-?[0-9]+\.[0-9]+"
_DECIMAL_STRING = re.compile(DECIMAL_STRING_PATTERN)
_INTEGER_STRING = re.compile(r"-?\d+")
_FRACTION_STRING = re.compile(r"-?\d+/[1-9]\d*")
_EXPONENT_STRING = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)[eE][+-]?[0-9]+")
_NONFINITE_STRING = re.compile(r"[+-]?(?:nan|inf|infinity)", flags=re.IGNORECASE | re.ASCII)
_EXPRESSION_CHARACTERS = re.compile(r"[0-9.\s+*/()\u2212\u00d7\u00f7-]+")
_EXPRESSION_STRUCTURE = re.compile(r"[()]|(?<=[0-9.)])\s*[+*/\u2212\u00d7\u00f7-]")
_PROTOCOLS = {"evidence_answer", "answer_only", "free", "list"}


def answer_fraction(value, *, text=False):
    """Extend only answer strings, retaining the existing exact numeric bounds."""
    if text and isinstance(value, str) and _DECIMAL_STRING.fullmatch(value):
        return fraction_of(Decimal(value))
    return fraction_of(value, text=text)


# A separate module instance prevents v2 from mutating SR-F1 or SR-F1.2 scoring.
_v2 = reference_module("semantic_contract")
_v2.fraction_of = answer_fraction
_v2.load_json_strict = load_json_strict


def _protocol(protocol):
    if not isinstance(protocol, str) or protocol.lower() not in _PROTOCOLS:
        raise ValueError("Unsupported SR-F2 answer protocol")
    return protocol.lower()


def last_balanced_json(text):
    """Return the last complete top-level brace object, without gold-based repair.

    Quoted braces and escaped quotes inside objects do not affect nesting. A
    balanced but invalid last object is returned and subsequently fails parsing;
    it never falls back to an earlier answer that happens to be valid.
    """
    if not isinstance(text, str) or len(text.encode()) > 65536:
        return None
    start, depth, quoted, escaped, last = None, 0, False, False, None
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start, depth = index, 1
            continue
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                last, start = text[start : index + 1], None
    return last


def _payload(text, protocol):
    return last_balanced_json(text) if protocol in ("free", "list") else text


def _answer_kind(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer_number"
    if isinstance(value, Decimal):
        return "decimal_number"
    if isinstance(value, str):
        if _INTEGER_STRING.fullmatch(value):
            return "integer_string"
        if _FRACTION_STRING.fullmatch(value):
            return "fraction_string"
        if _DECIMAL_STRING.fullmatch(value):
            return "decimal_string"
        if _EXPONENT_STRING.fullmatch(value):
            return "exponent_string"
        if value != value.strip():
            return "whitespace_string"
        if _NONFINITE_STRING.fullmatch(value):
            return "nonfinite_string"
        if (
            _EXPRESSION_CHARACTERS.fullmatch(value)
            and re.search(r"[0-9]", value)
            and _EXPRESSION_STRUCTURE.search(value)
        ):
            return "expression_string"
        return "other_string"
    return "array" if isinstance(value, list) else "object"


def classify_answer_syntax(text, protocol="evidence_answer"):
    """Blind syntax strata, including parseability under each frozen grammar.

    Expression/nonfinite labels describe otherwise rejected strings; they never
    evaluate an expression or expand the scorer's accepted answer language.
    Numeric grammars and leading/trailing whitespace retain their own strata.
    """
    protocol = _protocol(protocol)
    result = dict(L_json=0, answer_syntax="invalid_json", v1_parseable=0, v2_parseable=0)
    try:
        obj = load_json_strict(_payload(text, protocol))
        allowed = {"evidence", "answer"} if protocol == "evidence_answer" else {"answer"}
        if not isinstance(obj, dict) or not set(obj).issubset(allowed):
            result["answer_syntax"] = "invalid_root"
            return result
        result["L_json"] = 1
        if "answer" not in obj:
            result["answer_syntax"] = "missing_answer"
            return result
        value = obj["answer"]
        result["answer_syntax"] = _answer_kind(value)
        for label, parser in (("v1_parseable", fraction_of), ("v2_parseable", answer_fraction)):
            try:
                if value is not None:
                    parser(value, text=True)
                result[label] = 1
            except (ValueError, TypeError, ZeroDivisionError, OverflowError):
                pass
    except (ValueError, TypeError, OverflowError, RecursionError):
        pass
    return result


def sufficient_keys(world, query):
    """The task-side certificate, using gold Alpha rank with printed tie order."""
    full = _v2.required_keys(world, query)
    if query["family"] != "TOPK":
        return full
    selected = _v2.selected_indices(world, query)
    return [("Alpha", category) for category in world["categories"]] + [
        ("Beta", world["categories"][index]) for index in selected
    ]


def _answer_only_score(text, world, query):
    result = dict(L_json=0, L_answer=0, A=0, reason="invalid_json")
    try:
        obj = load_json_strict(text)
        if not isinstance(obj, dict) or not set(obj).issubset({"answer"}):
            return result
        result.update(L_json=1, reason="missing_answer")
        if "answer" in obj:
            answer = None if obj["answer"] is None else answer_fraction(obj["answer"], text=True)
            result.update(L_answer=1, A=int(answer == execute(world, query)), reason="scored")
    except (ValueError, TypeError, OverflowError, RecursionError, ZeroDivisionError):
        pass
    return result


def score_v2(text, world, query, protocol="evidence_answer"):
    """Preserve original E/P/J and add TOPK's six-value certificate scores.

    E6 checks coverage of the correct certificate and rejects duplicates or any
    key outside the full eight-value contract. P6 additionally checks every
    supplied value, including optional Beta values in a certificate superset.
    Answer-only protocols retain their original answer-only result schema.
    """
    protocol = _protocol(protocol)
    payload = _payload(text, protocol)
    if protocol != "evidence_answer":
        return _answer_only_score(payload, world, query)
    result = _v2.score(payload, world, query)
    result.update(E6=result["E"], P6=result["P"], J6=result["J"])
    result["A_minus_P"] = result["A"] - result["P"]
    result["contract_relaxation"] = 6 if query["family"] == "TOPK" else 0
    if query["family"] != "TOPK" or not result["L_evidence"]:
        return result
    obj = load_json_strict(payload)
    supplied = {(item["series"], item["category"]): item["value"] for item in obj["evidence"]}
    truth = {
        (item["series"], item["category"]): item["value"] for item in gold_evidence(world, query)
    }
    result["E6"] = int(set(sufficient_keys(world, query)) <= set(supplied) <= set(truth))
    result["P6"] = int(result["E6"] and all(supplied[key] == truth[key] for key in supplied))
    result["J6"] = int(result["P6"] and result["A"])
    return result


def score_record_v2(record, task):
    """Respect the stored generation-format failure, as SR-F1.2 does.

    Generation provenance must be validated separately before this scorer is
    called; this boundary does not reinterpret a technical failure as an answer.
    """
    error = record.get("format_protocol_error")
    result = score_v2(
        "" if error else record["raw_text"],
        task["world"],
        task["query"],
        record.get("protocol", "evidence_answer"),
    )
    if error:
        result["reason"] = error
    return result
