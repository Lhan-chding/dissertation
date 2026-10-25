"""SR-F5 selected measurement around the unchanged historical task scorer.

The extraction boundary is gold-blind. Projection deletes one raw member rather
than reserializing JSON, so Decimal literals and work bytes remain unchanged.
The reference certificate implementation is never used for production labels.
"""

from __future__ import annotations

import json
from decimal import Decimal

from sr_f4 import task_contract as legacy
from srf5_ref.measurement import map_token_mask, parse_response, verify_selection
from srf5_ref.oracle import evaluate_a

VERSION = "srf5-native-measurement-v1"

__all__ = [
    "VERSION",
    "lossless_legacy_projection",
    "map_token_mask",
    "parse_response",
    "score_response",
    "verify_selection",
]


def _project_validated(text):
    """Called only after strict whole-response parsing, including duplicate keys."""
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
        pos += 1  # Colon, already checked by strict parsing.
        while text[pos].isspace():
            pos += 1
        _, value_end = decoder.raw_decode(text, pos)
        pos = value_end
        while text[pos].isspace():
            pos += 1
        if key == "selected":
            if text[pos] == ",":
                return text[:member_start] + text[pos + 1 :]
            if previous_comma is None:
                raise ValueError("selected cannot be the only v3 member")
            return text[:previous_comma] + text[value_end:]
        if text[pos] != ",":
            raise ValueError("selected field missing")
        previous_comma = pos
        pos += 1


def lossless_legacy_projection(text):
    """Remove selected and one separator, rejecting invalid originals first."""
    extraction = parse_response(text)
    if extraction["status"] != "KNOWN":
        raise ValueError("Invalid v3 response: " + extraction["reason"])
    return _project_validated(text)


def _perception(obj, world):
    """Native fact grammar and truth; absent/invalid facts remain UNKNOWN."""
    try:
        facts = legacy.parse_facts(obj["evidence"], world)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError):
        return "UNKNOWN"
    if not facts:
        return "UNKNOWN"
    truth = all(value == world["series"][series][index] for (series, index), value in facts.items())
    return "PASS" if truth else "FAIL"


def score_response(
    text,
    world,
    query,
    finished=True,
    *,
    finish_reason=None,
    protocol="structured_v3",
):
    """Keep native A/certificate/finish and add only v3 structure and selection.

    ``J_legacy_projection`` is None when a valid projection cannot be made.
    A malformed v3 response can retain native diagnostic A/P, but always has
    V=0, S=UNKNOWN and J_M3=0. No malformed original is repaired to earn reward.
    Natural/legacy protocols retain their native score and have S=NOT_APPLICABLE.
    """
    reason = finish_reason or (
        ("final_tag" if protocol == "natural_work" else "balanced") if finished else "length"
    )
    if not finished:
        reason = "length"
    if protocol in {"natural_work", "structured_v2", "structured"}:
        native_protocol = "natural_work" if protocol == "natural_work" else "structured"
        score = legacy.score(text, world, query, finish_reason=reason, protocol=native_protocol)
        score.update(
            V=score["presentation"],
            A=score["value_correct"],
            P="NOT_APPLICABLE" if protocol == "natural_work" else "UNKNOWN",
            S="NOT_APPLICABLE",
            J_M3=None,
            J_legacy_projection=None,
            selected_span=None,
            measurement_version=VERSION,
            protocol=protocol,
        )
        return score
    if protocol != "structured_v3":
        raise ValueError("Unregistered SR-F5 measurement protocol")

    # Also rejects unknown query fields before the more permissive legacy AST.
    gold = evaluate_a(world, query)
    extraction = parse_response(text)
    known = extraction["status"] == "KNOWN"
    projection = _project_validated(text) if known else None
    score = legacy.score(
        projection if known else text,
        world,
        query,
        finish_reason=reason,
        protocol="structured",
    )
    legacy_j = score["J"] if known else None
    obj = extraction["object"]
    if not known:
        try:
            obj = legacy.native.loads(text)
        except (ValueError, TypeError, RecursionError, OverflowError):
            obj = None
    score.update(
        V=int(known),
        A=score["value_correct"],
        P=_perception(obj, world),
        S=verify_selection(extraction, gold["selected"]),
        J_legacy_projection=legacy_j,
        selected_span=extraction["selected_span"],
        extraction_reason=extraction["reason"],
        legacy_reason=score["reason"],
        measurement_version=VERSION,
        protocol=protocol,
        gold_source="srf5_ref.oracle.evaluate_a; native exact answer and certificate",
    )
    score["J_M3"] = int(
        score["V"] and score["finish"] and score["value_correct"] and score["certificate"]
    )
    score["J"] = score["J_M3"]
    score["reason"] = extraction["reason"]
    return score
