"""One public-only prompt candidate; the v2 wire grammar and scorer stay fixed.

The checklist names evidence requirements from the public query, never from a
world, model answer, gold selection, or candidate certificate. It changes no
parser, duplicate-key policy, exact-number bound, or reward.
"""

from __future__ import annotations

import json

from srf5_ref.oracle import validate
from srf6_ref.design import visible_prompt

from . import interface_protocol as v2

VERSION = "srf6-interface-v3"
WIRE_PROTOCOL = "structured_v4"


def response_shape():
    """Return the unchanged shape as a fresh object, not a permissive decoder."""
    return v2.response_shape()


def format_contract():
    """Add explicit declaration rules without changing the shared v2 fields."""
    contract = v2.format_contract()
    opening = contract.splitlines()[0]
    contract = contract.replace(
        opening,
        opening + "\nWrite every key exactly once in each object.",
        1,
    )
    old_evidence = v2.FIELDS["evidence"][1]
    contract = contract.replace(
        '"evidence": ' + old_evidence,
        '"evidence": '
        + old_evidence
        + " Each object reports exactly ONE original chart cell: one series, "
        "one category, and one value. Never merge Alpha and Beta into the same "
        "object. Never repeat the same (series, category) cell in evidence.",
        1,
    )
    old_selected = v2.FIELDS["selected"][1]
    selected = old_selected.replace(
        'A TYPE example is ["Example_A", "Example_B"].',
        'TYPE examples are ["Example_A", "Example_B"], or ["Example_A"] '
        "when only one category is adopted.",
    )
    selected += (
        " Write the JSON array itself, not a single string containing an array "
        "and not a bare category string."
    )
    return contract.replace('"selected": ' + old_selected, '"selected": ' + selected, 1)


def _evidence_requirements(categories, query):
    """Public requirements as (scope, labels or None, series) tuples.

    None is symbolic: the model's finally selected categories. No selection is
    computed here. The only explicit subset is the public interval after its
    endpoint rules, before any value-dependent filter.
    """
    operand_series = ("Alpha", "Beta") if query["value"] == "diff" else (query["value"],)
    if query["family"] == "TOPK":
        return (
            ("ranking", tuple(categories), (query["selection_series"],)),
            ("selected_operand", None, operand_series),
        )
    start = query["start"] + int(not query["include_start"])
    end = query["end"] - int(not query["include_end"])
    interval = tuple(categories[start : end + 1]) if start <= end else ()
    if query["positive_only"]:
        return (("prefilter_interval", interval, ("Alpha", "Beta")),)
    return (("interval", interval, operand_series),)


def _evidence_checklist(categories, query):
    lines = ["EVIDENCE CHECKLIST FOR THIS QUESTION:"]
    for scope, labels, series in _evidence_requirements(categories, query):
        columns = "BOTH Alpha and Beta" if len(series) == 2 else series[0]
        if scope == "ranking":
            lines.append(
                f"- Ranking: report the original {columns} value for EVERY category "
                f"in this public list: {json.dumps(labels, ensure_ascii=False)}."
            )
        elif scope == "selected_operand":
            lines.append(
                f"- Operand: report the original {columns} "
                f"{'values' if len(series) == 2 else 'value'} of each finally "
                "selected category in evidence."
            )
        else:
            lines.append(
                "- The public interval after applying both endpoint rules is "
                f"{json.dumps(labels, ensure_ascii=False)}. Report the original "
                f"{columns} {'values' if len(series) == 2 else 'value'} for EVERY "
                "category in this list"
                + (
                    " BEFORE the positive-only filter, including categories "
                    "that the filter then excludes."
                    if scope == "prefilter_interval"
                    else "."
                )
            )
    if query["value"] == "diff":
        lines.append(
            "- Keep each Alpha value and Beta value in separate original-cell "
            "evidence objects; put the computed signed Alpha minus Beta only "
            "in bindings.operand."
        )
    lines.append(
        "- A cell already reported for ranking or filtering is reported only "
        "once. Values written only in bindings or work do not count as evidence."
    )
    return "\n".join(lines)


def public_prompt(categories, query):
    """Accept the same public labels/query grammar as v2, with no hidden inputs."""
    if not isinstance(categories, list) or not isinstance(query, dict):
        raise ValueError("Public category list and query object required")
    if not all(type(category) is str for category in categories):
        raise ValueError("Public category strings required")
    labels, public_query = list(categories), dict(query)
    # These dummy values validate the public schema; they do not select a set.
    validate(
        {
            "categories": labels,
            "domain": [10, 90],
            "series": {name: [50] * len(labels) for name in ("Alpha", "Beta")},
        },
        public_query,
    )
    safe = {"world": {"categories": labels}, "query": public_query}
    question = visible_prompt(safe).rsplit("Return exactly one JSON object", 1)[0]
    return question + "\n" + format_contract() + "\n" + _evidence_checklist(labels, public_query)
