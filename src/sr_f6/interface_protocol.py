"""One explicit, public-only interface revision; the frozen scorer is unchanged.

This is a prompt revision, not a new JSON decoder or reward. Shape documentation
never replaces measurement.parse (duplicate keys and exact-number bounds).
"""

from __future__ import annotations

import copy

from srf5_ref.oracle import validate
from srf6_ref.design import visible_prompt

VERSION = "srf6-interface-v2"
WIRE_PROTOCOL = "structured_v4"

# The same field specification supplies both the prompt and the shape document.
# No task values, category enum, gold cardinality or gold answer enter the schema.
NUMBER = {
    "anyOf": [{"type": "number"}, {"type": "string"}],
    "description": "Must also pass the unchanged exact-number grammar and bounds.",
}


def _entry(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


FIELDS = {
    "evidence": (
        {
            "type": "array",
            "items": _entry(
                {
                    "series": {"type": "string"},
                    "category": {"type": "string"},
                    "value": NUMBER,
                }
            ),
        },
        'an array of objects with exactly "series", "category", and "value". '
        "series and category are strings; value is an original chart cell NUMBER "
        "or exact numeric string, never an expression. A TYPE example is "
        '{"series":"Alpha","category":"Example_A","value":42}.',
    ),
    "work": (
        {"type": "string"},
        "an OPTIONAL concise working string. If used, put it BEFORE selected, "
        "bindings, aggregate and answer. Complete calculations and corrections "
        "before writing the final declarations. Escape newlines inside JSON strings.",
    ),
    "selected": (
        {"type": "array", "items": {"type": "string"}},
        "an array of STRINGS naming your FINAL adopted categories after every "
        'condition and filter. A TYPE example is ["Example_A", "Example_B"]. '
        'Do NOT use objects such as [{"category":"Example_A"}] or report a '
        "preliminary candidate pool. Do not repeat labels. Order is irrelevant.",
    ),
    "bindings": (
        {
            "type": "array",
            "items": _entry(
                {
                    "category": {"type": "string"},
                    "operand": NUMBER,
                }
            ),
        },
        "an array with one object per finally selected category, with exactly "
        '"category" (a string) and "operand" (its computed NUMERIC VALUE). '
        'A TYPE example is {"category":"Example_A","operand":-18}. '
        "Use the requested Alpha value, Beta value, or already computed signed "
        "Alpha minus Beta for that category. Negative numbers are allowed. "
        '"Beta", "Alpha - Beta", and "60-48" are NOT numeric operands. '
        "Do not write an unquoted expression either. Order is irrelevant.",
    ),
    "aggregate": (
        {"enum": ["sum", "mean"]},
        'the string "sum" or "mean", as requested. Mean means arithmetic mean '
        "over your final selected categories.",
    ),
    "answer": (
        {"anyOf": [NUMBER, {"type": "null"}]},
        "the final exact result: an integer, an exact finite decimal, or a quoted "
        'fraction such as "17/3". Do not round repeating decimals or write '
        "unquoted fractions. Put answer last. It is the sole final answer; "
        "numbers in work do not replace it.",
    ),
}


def response_shape():
    """Return fresh documentation; exact syntax remains the production parser's job."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$comment": (
            "Shape only; not constrained decoding. Duplicate keys, exact numbers, "
            "completion and truth require the unchanged production measurement. "
            "Null answer retains the old syntax, not an automatically correct empty answer."
        ),
        **_entry({key: copy.deepcopy(shape) for key, (shape, _) in FIELDS.items()}),
        "required": [key for key in FIELDS if key != "work"],
    }


def format_contract():
    instructions = [
        "Return exactly one JSON object, with no Markdown fences or text outside it.",
        "Use these fields in the following suggested order; work is optional:",
    ]
    instructions.extend(f'"{key}": {description}' for key, (_, description) in FIELDS.items())
    instructions.append(
        "The examples illustrate types only. Use actual labels and values from THIS "
        "chart, never the Example_A/Example_B placeholders. Do not add extra fields. "
        "Evidence and bindings serve different roles: a number written only in "
        "bindings does not count as an evidence entry. Do not use external tools."
    )
    return "\n".join(instructions)


def public_prompt(categories, query):
    """Accept only public labels and the exact public query AST, never a world."""
    if not isinstance(categories, list) or not isinstance(query, dict):
        raise ValueError("Public category list and query object required")
    if not all(type(category) is str for category in categories):
        raise ValueError("Public category strings required")
    # Reuse the task grammar with fixed dummy values, never hidden chart values.
    # validate only checks structure and bounds; it does not compute an answer.
    validate(
        {
            "categories": categories,
            "domain": [10, 90],
            "series": {name: [50] * len(categories) for name in ("Alpha", "Beta")},
        },
        query,
    )
    safe = {"world": {"categories": list(categories)}, "query": dict(query)}
    question = visible_prompt(safe).split("Return exactly one JSON object", 1)[0]
    if query["family"] == "TOPK":
        evidence = (
            f"For ranking, report {query['selection_series']} for EVERY category "
            "to establish the selection. Also report the original chart values "
            "needed for the requested operand of every finally selected category."
        )
    elif query["positive_only"]:
        evidence = (
            "Apply the stated endpoint rules first. For the positive-only filter, "
            "report BOTH Alpha and Beta for EVERY category in that interval BEFORE "
            "filtering, including categories you then exclude."
        )
    else:
        evidence = (
            "Report the original chart values needed for every category inside "
            "the specified interval after applying the endpoint rules."
        )
    if query["value"] == "diff":
        evidence += (
            " For each required difference, evidence must contain the original "
            "Alpha and Beta values separately; bindings.operand contains their "
            "computed signed Alpha minus Beta, including its sign."
        )
    return question + "\n" + format_contract() + "\nEVIDENCE REQUIREMENT: " + evidence
