"""Read-only interface diagnostics beside the unchanged SR-F6 wire scorer.

The raw generation reason is retained separately from the original census's
balanced-only completion policy. No diagnostic repairs text, grants credit,
changes a reward, or signs a gate. A core task projection removes only complete
raw members from an already validated single JSON object; it never serializes
Decimal values or selects an answer from work.
"""

from __future__ import annotations

from sr_f4 import task_contract
from srf6_ref.core import exact

from . import measurement

VERSION = "srf6-interface-diagnostics-v1"
COMPLETION_POLICY = "original-census-balanced-only"
_FIELDS = ("evidence", "selected", "bindings", "aggregate", "work", "answer")
_ERRORS = (ValueError, TypeError, KeyError, ArithmeticError, RecursionError)


def core_task_projection(text):
    """Remove auxiliary members without accepting invalid/ambiguous JSON.

    Auxiliary *shapes* need not be valid for this diagnostic. The original text
    must nevertheless be one strict whole object, without duplicate keys, a
    trailing second object, nonfinite values, or control-character repairs.
    """
    obj = measurement._object(text)
    projected = text
    for field in ("selected", "bindings", "aggregate"):
        if field in obj:
            projected = measurement._delete_member(projected, field)
    return projected


def _number_error(value, path):
    try:
        exact(value)
    except _ERRORS as exc:
        return [{"path": path, "reason": str(exc)}]
    return []


def _field_errors(field, value):
    """Collect all independent shape/number errors in a readable field."""
    errors = []

    def add(path, reason):
        errors.append({"path": path, "reason": reason})

    if field == "selected":
        if not isinstance(value, list):
            add(field, "selected shape: array of strings required")
        else:
            for index, item in enumerate(value):
                if type(item) is not str:
                    add(f"{field}[{index}]", "selected element must be a string")
    elif field in ("bindings", "evidence"):
        if not isinstance(value, list):
            add(field, "array required")
        else:
            keys = (
                {"category", "operand"} if field == "bindings" else {"series", "category", "value"}
            )
            numeric = "operand" if field == "bindings" else "value"
            for index, item in enumerate(value):
                path = f"{field}[{index}]"
                if not isinstance(item, dict):
                    add(path, "object required")
                    continue
                for key in sorted(keys - item.keys()):
                    add(f"{path}.{key}", "missing required member")
                for key in sorted(item.keys() - keys):
                    add(f"{path}.{key}", "unexpected member")
                for key in sorted(keys - {numeric}):
                    if key in item and not isinstance(item[key], str):
                        add(f"{path}.{key}", "string identifier required")
                if numeric in item:
                    errors.extend(_number_error(item[numeric], f"{path}.{numeric}"))
    elif field == "aggregate":
        if value not in ("sum", "mean"):
            add(field, "aggregate must be sum or mean")
    elif field == "work":
        if not isinstance(value, str):
            add(field, "work must be a string")
    elif field == "answer" and value is not None:
        # The unchanged V grammar allows null; numeric A still treats it as
        # unknown. Shape validity and numeric observability remain distinct.
        errors.extend(_number_error(value, field))
    return errors


def _certificate(obj, world, query):
    result = {"known": False, "truth": None, "sufficient": None, "errors": []}
    try:
        facts = task_contract.parse_facts(obj["evidence"], world)
        result.update(
            known=True,
            truth=int(all(v == world["series"][s][i] for (s, i), v in facts.items())),
            sufficient=int(task_contract.sufficient_template(world, query, facts)),
        )
    except _ERRORS as exc:
        result["errors"].append({"path": "evidence", "reason": str(exc)})
    return result


def diagnose_response(text, world, query, raw_finish_reason):
    """Return additive diagnostics, preserving the original wire score exactly.

    ``fields`` uses syntax VALID/INVALID/ABSENT/UNAVAILABLE. ``known`` and
    ``truth`` for selection, binding and evidence retain the wire scorer's
    completion requirements; content-only certificate diagnostics have their
    own explicit boundary. Work has no task-truth judgment. A readable but
    incorrect answer is known and false; unreadable answers remain unknown.

    This function is not a replacement training scorer. In particular EOS is
    not silently upgraded to balanced, and J_core_diag is never J_wire.
    """
    if raw_finish_reason not in ("balanced", "eos", "length"):
        raise ValueError("unregistered raw generation finish reason")
    wire = measurement.score_response(
        text,
        world,
        query,
        finished=raw_finish_reason == "balanced",
        finish_reason=raw_finish_reason,
    )
    result = {
        "diagnostic_version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "gate_status": "NOT_SIGNED",
        "training_authorized": False,
        "completion_policy": COMPLETION_POLICY,
        "generation_finish_reason": raw_finish_reason,
        "scored_finish_reason": wire["finish_reason"],
        "finished": wire["finished"],
        "wire_score": wire,
        "A": wire["A"],
        "J_wire": wire["J"],
        "V_aux": wire["V"],
        "S": wire["S"],
        "B": wire["B"],
        "E_raw": wire["E_raw"],
        "consistency": wire["consistency"],
        "J_core_diag": None,
        "core_status": "UNAVAILABLE",
        "core_score": None,
        "certificate": {
            "known": False,
            "truth": None,
            "sufficient": None,
            "errors": [],
            "scope": "content_only_not_credit_eligibility",
        },
        "fields": {},
        "errors": [],
        "json_valid": False,
        "field_order": None,
        "work_before_answer": None,
        "answer_last": None,
    }
    try:
        obj = measurement._object(text)
    except _ERRORS as exc:
        error = {"path": "$", "reason": str(exc)}
        result["errors"].append(error)
        for field in _FIELDS:
            result["fields"][field] = {
                "syntax": "UNAVAILABLE",
                "known": False,
                "truth": None,
                "errors": [dict(error)],
            }
        result["certificate"]["errors"] = [dict(error)]
        return result

    result["json_valid"] = True
    order = list(obj)
    result["field_order"] = order
    if "answer" in obj:
        result["answer_last"] = order[-1] == "answer"
        if "work" in obj:
            result["work_before_answer"] = order.index("work") < order.index("answer")
    for key in sorted(obj.keys() - measurement.REQUIRED - {"work"}):
        result["errors"].append({"path": key, "reason": "unexpected top-level member"})
    for field in _FIELDS:
        present = field in obj
        errors = _field_errors(field, obj[field]) if present else []
        if not present and field in measurement.REQUIRED:
            errors.append({"path": field, "reason": "missing required member"})
        state = {
            "syntax": "ABSENT" if not present else "INVALID" if errors else "VALID",
            "known": False,
            "truth": None,
            "errors": errors,
        }
        if field in ("selected", "bindings", "evidence"):
            label = {"selected": "S", "bindings": "B", "evidence": "P"}[field]
            state.update(known=wire[label] is not None, truth=wire[label])
            if wire[label] is None:
                why = wire["unknown_reasons"].get(label)
                if why is not None:
                    state["unknown_reason"] = why
        elif field == "answer":
            state["known"] = wire["primary_answer_parse_status"] == "KNOWN"
            state["truth"] = wire["A"] if state["known"] else None
        elif field == "aggregate" and present and not errors and wire["finished"]:
            state.update(known=True, truth=int(obj[field] == query["aggregate"]))
        elif field == "work" and present:
            state["known"] = not errors
        result["fields"][field] = state
        result["errors"].extend(dict(error) for error in errors)
    result["certificate"].update(_certificate(obj, world, query))
    projection = core_task_projection(text)
    core = task_contract.score(
        projection, world, query, finish_reason=wire["finish_reason"], protocol="structured"
    )
    result.update(J_core_diag=core["J"], core_status="EVALUATED", core_score=core)
    return result
