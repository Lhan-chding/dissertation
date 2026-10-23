"""Bounded, programmatic-only development extension of selection measurement.

Adds bounded natural declarations, month aliases, and Sum -> Mean provenance.
It does not implement model consensus, change native scoring, or open C gates.
Extraction uses response evidence; task targets are consulted only afterward.
"""

from __future__ import annotations

import ast
import json
import re

from sr_f3_ce import contract as native

from . import verification as v1
from . import verification_v2 as v2

VERIFIER_VERSION = "srf4-observable-v3-programmatic-dev-r4"
_NATURAL_MEAN = re.compile(
    r"^(?:(?:the|final|their)\s+)?(?:(?P<mode>Alpha|Beta|diff(?:erences?)?)\s+)?"
    r"(?:arithmetic\s+)?(?P<op>mean|average)\s*(?:=|:|\bis\b)\s*(?P<expression>.*)$",
    re.I,
)
_NATURAL_SUM = re.compile(
    r"^(?:the\s+)?(?P<op>sum)\s+of\s+(?P<mode>Alpha|Beta)\s+values\s+"
    r"for\s+these\s+categories\s*:\s*(?P<expression>.*)$", re.I
)
_COUNT_ONLY = re.compile(
    r"(?:count|number\s+of\s+categories)\s*=\s*\d+"
    r"(?:\s+(?:months|categories))?", re.I
)
_MONTHS = (
    ("January", "Jan"),
    ("February", "Feb"),
    ("March", "Mar"),
    ("April", "Apr"),
    ("May",),
    ("June", "Jun"),
    ("July", "Jul"),
    ("August", "Aug"),
    ("September", "Sep", "Sept"),
    ("October", "Oct"),
    ("November", "Nov"),
    ("December", "Dec"),
)


def _normalize_months(work, categories):
    aliases = {}
    for month in _MONTHS:
        targets = [c for c in categories if c.casefold() in {a.casefold() for a in month}]
        if len(targets) == 1:
            aliases.update({a.casefold(): targets[0] for a in month})
    pieces, boundaries, changes = [], [0], []
    cursor = 0
    for match in re.finditer(r"\b[A-Za-z]+\b", work):
        replacement = aliases.get(match.group().casefold(), match.group())
        prefix = work[cursor : match.start()]
        pieces.append(prefix)
        boundaries.extend(range(cursor + 1, match.start() + 1))
        pieces.append(replacement)
        boundaries.extend(
            match.start() + i * len(match.group()) // len(replacement)
            for i in range(1, len(replacement) + 1)
        )
        if replacement != match.group():
            changes.append(
                dict(
                    original=match.group(),
                    canonical=replacement,
                    decoded_active_work_span=list(match.span()),
                )
            )
        cursor = match.end()
    pieces.append(work[cursor:])
    boundaries.extend(range(cursor + 1, len(work) + 1))
    return "".join(pieces), boundaries, changes


def _source_spans(value, original_work, boundaries):
    """Retain normalized offsets and cite corresponding original decoded work."""
    if isinstance(value, list):
        for item in value:
            _source_spans(item, original_work, boundaries)
    elif isinstance(value, dict):
        for item in list(value.values()):
            _source_spans(item, original_work, boundaries)
        if "decoded_work_span" in value:
            start, end = value.pop("decoded_work_span")
            value["normalized_work_span"] = [start, end]
            lo, hi = boundaries[start], boundaries[end]
            value["decoded_active_work_span"] = [lo, hi]
            value["source_text"] = original_work[lo:hi]


def _explicit_claims(work, categories):
    """Only complete lists can supply a set; annotations must agree with it."""
    claims, unsupported, provisional = v2._explicit_claims(work, categories)
    names = {c.casefold(): c for c in categories}
    category = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    prefix = r"(?:the\s+)?(?:selected\s+(?:categories|months)|categories\s+selected)"
    annotated = re.compile(
        r"^" + prefix + r"\s*(?:=|:|\bare\b)\s*(?P<items>.+?)\s*"
        r"\((?P<annotations>.+)\)$", re.I
    )
    membership = re.compile(
        r"(?P<action>including|excluding|include|exclude)\s+(?P<category>" + category + r")",
        re.I,
    )
    ranged = re.compile(
        r"^" + prefix + r"\s+from\s+(?P<start>" + category + r")\s+to\s+"
        r"(?P<end>" + category + r"),?\s+(?P<start_action>including|excluding)\s+"
        r"(?:the\s+)?start(?:\s*\((?P<start_name>" + category + r")\))?\s+and\s+"
        r"(?P<end_action>including|excluding)\s+(?:the\s+)?end"
        r"(?:\s*\((?P<end_name>" + category + r")\))?\s*:\s*(?P<items>.+)$", re.I
    )
    for clause, span in v2._clauses(work):
        match = annotated.fullmatch(clause) or ranged.fullmatch(clause)
        if match is None:
            continue
        selected = v2._parse_category_list(match.group("items"), categories)
        if selected is None:
            continue
        details = []
        if "annotations" in match.groupdict():
            for annotation in re.split(r"\s*,\s*|\s+and\s+", match.group("annotations"),
                                       flags=re.I):
                member = membership.fullmatch(annotation.strip())
                if member is None:
                    break
                name = names[member.group("category").casefold()]
                included = member.group("action").lower().startswith("includ")
                details.append(dict(category=name, included=included))
            else:
                if all((x["category"] in selected) == x["included"] for x in details):
                    unsupported[:] = [x for x in unsupported
                                      if x["decoded_work_span"] != list(span)]
                    claims.append(dict(text=clause, decoded_work_span=list(span),
                                       categories=selected, membership_annotations=details,
                                       rule="V3_COMPLETE_LIST_WITH_MEMBERSHIP_ANNOTATIONS"))
            continue
        start, end = (names[match.group(k).casefold()] for k in ("start", "end"))
        start_index, end_index = categories.index(start), categories.index(end)
        endpoint_names_agree = all(
            match.group(k + "_name") is None
            or names[match.group(k + "_name").casefold()] == endpoint
            for k, endpoint in (("start", start), ("end", end))
        )
        lo = start_index + (match.group("start_action").lower() == "excluding")
        hi = end_index + (match.group("end_action").lower() == "including")
        if (endpoint_names_agree and start_index <= end_index
                and selected == sorted(categories[lo:hi])):
            unsupported[:] = [x for x in unsupported if x["decoded_work_span"] != list(span)]
            claims.append(dict(text=clause, decoded_work_span=list(span), categories=selected,
                               rule="V3_COMPLETE_LIST_WITH_RANGE_ANNOTATION"))
    return claims, unsupported, provisional


def _mean_expression(match, work, span):
    """Keep literal RHS stages as execution claims, never as selected operands."""
    parts = match.group("expression").replace("\N{MINUS SIGN}", "-").split("=")
    if len(parts) < 2:
        raise ValueError("NO_COMPLETE_FINAL_EQUALITY")
    tree = ast.parse(parts[0].strip(), mode="eval").body
    if not isinstance(tree, ast.BinOp) or not isinstance(tree.op, ast.Div):
        raise ValueError("MEAN_REQUIRES_EXPLICIT_SUM_AND_DIVISOR")
    divisor = v2._literal(tree.right)
    if not divisor:
        raise ValueError("ZERO_DIVISOR")
    values = v2._sum_leaves(tree.left)
    displays = []
    for i, part in enumerate(parts[1:], 1):
        rhs = part.strip()
        approximate = (
            i == len(parts) - 1 and rhs.endswith(".")
            and re.fullmatch(v2._NUMBER, rhs[:-1]) is not None
            and work[span[1] : span[1] + 2] == ".."
            and work[span[1] + 2 : span[1] + 3] != "."
        )
        fraction = re.fullmatch(r"(?P<a>[+-]?\d+)\s*/\s*(?P<b>[+-]?\d+)", rhs)
        if approximate:
            displays.append(dict(claimed_result=rhs[:-1] + "...",
                                 result_display_kind="ELLIPSIS_NOT_EXACT"))
        elif fraction and int(fraction.group("b")):
            value = v2.Fraction(int(fraction.group("a")), int(fraction.group("b")))
            displays.append(dict(claimed_result=rhs, result_display_kind="EXACT_FRACTION",
                                 exact_display_value=str(value)))
        elif re.fullmatch(v2._NUMBER, rhs):
            displays.append(dict(claimed_result=rhs, result_display_kind="FINITE_LITERAL",
                                 exact_display_value=str(v2.Fraction(rhs))))
        else:
            raise ValueError("UNSUPPORTED_EQUALITY_STAGE")
    return parts[0].strip(), tree.left, values, divisor, displays


def _display_metadata(displays, numerator, divisor):
    stages = [dict(stage) for stage in displays]
    for stage in stages:
        if "exact_display_value" in stage:
            stage["display_matches_computation"] = (
                v2.Fraction(stage["exact_display_value"]) == numerator / divisor
            )
    return dict(**stages[-1], equality_stages=stages)


def _trace_reductions(work, facts, categories):
    traces, ambiguous, unsupported = v2._trace_reductions(work, facts, categories)
    reductions = []
    for clause, span in v2._clauses(work):
        natural = _NATURAL_SUM.fullmatch(clause)
        match = (v2._REDUCTION.fullmatch(clause) or v2._VERBAL_SUM.fullmatch(clause)
                 or _NATURAL_MEAN.fullmatch(clause) or natural)
        if natural:
            # This bounded rewrite supplies an observed series, never the
            # task's desired one. Every operand still needs evidence binding.
            parsed = v2._trace_reductions(
                natural.group("mode") + " sum=" + natural.group("expression"), facts, categories
            )
            for destination, extracted in zip(
                (traces, ambiguous, unsupported), parsed, strict=True
            ):
                for item in extracted:
                    item.update(text=clause, decoded_work_span=list(span),
                                rule="V3_EXPLICIT_SERIES_NATURAL_SUM")
                    destination.append(item)
        if match:
            reductions.append((match, span, clause))
    for match, span, clause in reductions:
        operation = match.groupdict().get("op", "sum").lower()
        if operation not in {"mean", "average"}:
            continue
        earlier = [
            (m, s)
            for m, s, _ in reductions
            if s[0] < span[0] and m.groupdict().get("op", "sum").lower() in {"sum", "total"}
        ]
        current = dict(text=clause, decoded_work_span=list(span))
        try:
            expression, numerator_tree, values, divisor, displays = _mean_expression(
                match, work, span
            )
        except (ValueError, TypeError, SyntaxError, OverflowError):
            for collection in (traces, ambiguous, unsupported):
                collection[:] = [x for x in collection if x["decoded_work_span"] != list(span)]
            unsupported.append(dict(current, reason="UNSUPPORTED_MEAN_EQUALITY"))
            continue
        for collection in (traces, ambiguous, unsupported):
            collection[:] = [x for x in collection if x["decoded_work_span"] != list(span)]
        if len(values) > 1 or not earlier:
            # Reuse v2's unique response-evidence binding, keeping preceding
            # category-level differences available at their original offsets.
            canonical = (match.group("mode") or "") + " mean=" + expression + "=0"
            parsed = v2._trace_reductions(work[:span[0]] + canonical, facts, categories)
            for destination, extracted in zip((traces, ambiguous, unsupported), parsed,
                                               strict=True):
                for item in extracted:
                    if item["decoded_work_span"][0] < span[0]:
                        continue
                    numerator = sum(values, v2.Fraction())
                    item.update(current, rule="V3_EXPLICIT_MEAN_EQUALITY_CHAIN",
                                claimed_numerator=str(numerator), divisor=str(divisor),
                                **_display_metadata(displays, numerator, divisor))
                    item.pop("exact_equality", None)  # synthetic RHS is not a result claim
                    destination.append(item)
            continue
        numerator = v2._literal(numerator_tree)
        # A literal numerator following an explicit Sum may be an intermediate.
        # Never instead map that Sum's RHS to a coincidentally equal world datum.
        candidates = [
            t for t in traces if len(earlier) == 1 and t["decoded_work_span"] == list(earlier[0][1])
        ]
        current["rule"] = "V3_LITERAL_SUM_TO_MEAN_CHAIN"
        if len(candidates) != 1 or divisor == 0:
            ambiguous.append(dict(current, reason="NO_UNIQUE_COMPLETE_PRECEDING_SUM"))
            continue
        source = candidates[0]
        intervening = [
            text
            for text, position in v2._clauses(work)
            if source["decoded_work_span"][1] <= position[0] < span[0]
        ]
        if any(not _COUNT_ONLY.fullmatch(text) for text in intervening):
            ambiguous.append(dict(current, reason="UNSUPPORTED_INTERVENING_CHAIN_TEXT"))
            continue
        mode = match.group("mode")
        observed_mode = source["mapping_candidates"][0]["mode"]
        normalized_mode = (
            None if mode is None else ("diff" if mode.lower().startswith("diff") else mode.title())
        )
        if numerator != v2.Fraction(source["claimed_result"]) or (
            normalized_mode is not None and normalized_mode != observed_mode
        ):
            ambiguous.append(dict(current, reason="SUM_REFERENCE_OR_SERIES_CONFLICT"))
            continue
        current.update(
            categories=source["categories"],
            intermediate_source=source.copy(),
            claimed_numerator=str(numerator),
            divisor=str(divisor),
            observed_mode=observed_mode,
            **_display_metadata(displays, numerator, divisor),
        )
        traces.append(current)
    return traces, ambiguous, unsupported


def _unrecognized_tail(work, explicit, traces, categories):
    """A known earlier calculation cannot hide a later unsupported replacement."""
    anchors = [*explicit, *traces]
    if not anchors:
        return []
    final = max(anchors, key=lambda item: item["decoded_work_span"][1])
    end = final["decoded_work_span"][1]
    suspicious = []
    for clause, span in v2._clauses(work):
        if span[0] < end:
            continue
        # A scalar count does not change a previously observable category set.
        # It is never used here to infer, complete, or verify that set.
        if _COUNT_ONLY.fullmatch(clause):
            continue
        # Permit only the specific exact-fraction annotation accompanying an
        # already traced repeating-decimal Mean, without judging its E label.
        continuation = re.fullmatch(
            r"(?:which\s+is(?:\s+exactly)?|\N{RIGHTWARDS ARROW})\s*"
            r"(?P<a>[+-]?\d+)\s*/\s*(?P<b>[+-]?\d+)", clause, re.I
        )
        if continuation and final.get("result_display_kind") == "ELLIPSIS_NOT_EXACT":
            denominator = int(continuation.group("b"))
            if denominator and v2.Fraction(int(continuation.group("a")), denominator) == (
                v2.Fraction(final["claimed_numerator"]) / v2.Fraction(final["divisor"])
            ):
                continue
        adoption = re.search(
            r"\b(?:final|answer|result|average|mean|sum|total|take|use|used|"
            r"select(?:ed|ion)?|choose|chosen|include[ds]?|exclude[ds]?|"
            r"instead|discard|ignore|correction|revised)\b", clause, re.I
        )
        named = any(re.search(r"\b" + re.escape(c) + r"\b", clause, re.I)
                    for c in categories)
        calculation = re.search(r"\d", clause) and re.search(r"[=+*/\-]", clause)
        if adoption or named or calculation:
            suspicious.append(dict(text=clause, decoded_work_span=list(span),
                                   reason="UNRECOGNIZED_TRAILING_ADOPTION_OR_CALCULATION"))
    return suspicious


def verify_semantics(task, raw_record, native_score):
    result = v2.verify_semantics(task, raw_record, native_score)
    result.update(
        verifier_version=VERIFIER_VERSION,
        measurement_mode="PROGRAMMATIC_ONLY",
        model_consensus_implemented=False,
        development_only=True,
        frozen_gate_eligible=False,
        human_audited=False,
    )
    result["dimensions_source_versions"]["S"] = VERIFIER_VERSION
    if not result["normalization"]["finished"] or not result["normalization"]["protocol_valid"]:
        return result
    if task["query"]["family"] not in {"INTERVAL", "TOPK"}:
        return result
    try:
        obj, spans = v1._top_fields(v1.raw_text(raw_record))
    except (ValueError, TypeError, RecursionError, json.JSONDecodeError, IndexError):
        return result
    work = obj.get("work")
    if not isinstance(work, str) or not work.strip():
        return result
    active, withdrawn = v1._active_work(work)
    categories = task["world"]["categories"]
    normalized, boundaries, aliases = _normalize_months(active, categories)
    explicit, rejected, provisional = _explicit_claims(normalized, categories)
    try:
        facts = native.parse_facts(obj.get("evidence", []), task["world"])
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        facts = {}
    traces, ambiguity, unsupported = _trace_reductions(normalized, facts, categories)
    unsupported_tail = _unrecognized_tail(normalized, explicit, traces, categories)
    observed_sets = {tuple(c["categories"]) for c in [*explicit, *traces]}
    observations = dict(
        explicit_claims=explicit,
        final_reduction_claims=traces,
        unsupported_claims=rejected,
        ambiguous_traces=ambiguity,
        unsupported_reductions=unsupported,
        unsupported_tail=unsupported_tail,
        provisional_claims=provisional,
        withdrawn_work=withdrawn,
        month_aliases=aliases,
        decoded_work_span_origin="ACTIVE_WORK_AFTER_LAST_EXPLICIT_CORRECTION",
    )
    _source_spans(observations, active, boundaries)
    target = dict(
        categories=sorted(
            categories[i] for i in v1._query_api().selected(task["world"], task["query"])
        ),
        selection_series=task["query"].get("selection_series", "Alpha"),
    )
    common = dict(spans=[spans["work"]], observed=observations, target=target)
    if rejected or ambiguity or unsupported or unsupported_tail or len(observed_sets) > 1:
        result["dimensions"]["S"] = v1._dimension(
            reason="V3_CONFLICTING_UNSUPPORTED_OR_AMBIGUOUS_TRACE", **common
        )
    elif not observed_sets:
        result["dimensions"]["S"] = v1._dimension(
            reason="V3_NO_OBSERVABLE_FINAL_SELECTION", **common
        )
    else:
        observed = list(next(iter(observed_sets)))
        series = [c["selection_series"] for c in explicit if "selection_series" in c]
        series += [
            s.title()
            for s in re.findall(
                r"\b(?:selection_series|rank_series)\s*=\s*(Alpha|Beta)\b", normalized, re.I
            )
        ]
        observations.update(categories=observed, selection_series=series)
        okay = observed == target["categories"] and all(
            s == target["selection_series"] for s in series
        )
        result["dimensions"]["S"] = v1._dimension(
            "PASS" if okay else "FAIL",
            "V3_ADOPTED_SET_COMPARED_TO_TARGET",
            rule="S_V3_PROGRAMMATIC_MONTH_ALIASES_AND_SUM_MEAN",
            **common,
        )
    return result
