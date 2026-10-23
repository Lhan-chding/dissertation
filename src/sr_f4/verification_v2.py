"""Development-only selection measurement; never imported by the frozen C gate.

This is a bounded grammar, not a general reader of hidden reasoning. A category
set must be explicitly adopted, or traced from every operand of an explicitly
named final sum/mean to response evidence. Query targets and answers are used
only after extraction, never to choose a candidate interpretation. P, O, E and
native normalization retain v1 semantics. All labels remain machine judgments.
"""
# ruff: noqa: RUF001

from __future__ import annotations

import ast
import json
import re
from fractions import Fraction

from sr_f3_ce import contract as native
from srf4_ref.parsing import arithmetic

from . import verification as v1

VERIFIER_VERSION = "srf4-observable-v2-dev"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_SPLIT = re.compile(r"[;\n]+|(?<!\d)\.(?!\d)|(?<=\d)\.(?=\s|$)")
_PROVISIONAL = re.compile(
    r"^\s*(?:initial(?:ly)?|hypothetical(?:ly)?|consider(?:ing)?|"
    r"discarded|rejected|not\s+selected|incorrect|wrong)\b", re.I
)
_PREFIX = re.compile(
    r"^(?:the\s+)?(?P<key>"
    r"(?:final[ _])?(?:selection|selected(?:[ _](?:categories|months))?)|"
    r"(?:ranked|interval)[ _]selection|chosen(?:[ _](?:categories|months))?"
    r")\s*(?:=|:|\bare\b|\bis\b)\s*(?P<items>.*)$", re.I
)
_POSTFIX = re.compile(
    r"^(?P<items>.+?)\s+(?:are|were)\s+(?:the\s+)?"
    r"(?:final\s+)?(?:selected|chosen)(?:\s+(?:categories|months))?\s*$", re.I
)
_RANK = re.compile(
    r"^(?:the\s+)?(?:(?:top|bottom)\s+(?:\d+|one|two|three|four|five|six)|"
    r"(?:\d+|one|two|three|four|five|six)\s+(?:highest|lowest|largest|smallest))"
    r"\s+(?P<series>Alpha|Beta)\s+values\s+(?:are|is)\s*:?[ \t]*(?P<items>.*)$",
    re.I,
)
_RANK_CATEGORIES = re.compile(
    r"^(?:the\s+)?(?:\d+|one|two|three|four|five|six)\s+"
    r"(?:categories|months)\s+with\s+the\s+(?:highest|lowest|largest|smallest)"
    r"\s+(?P<series>Alpha|Beta)\s+values\s+(?:are|is)\s*:?[ \t]*(?P<items>.*)$",
    re.I,
)
_REDUCTION = re.compile(
    r"^(?:(?:the|final|their)\s+)?(?:(?P<mode>Alpha|Beta|diff(?:erences?)?)\s+)?"
    r"(?P<op>sum|total|mean|average)\s*(?:=|:|\bis\b)\s*(?P<expression>.*)$", re.I
)
_VERBAL_SUM = re.compile(
    r"^summing\s+(?:their|the)\s+(?P<mode>Alpha|Beta)\s+values\s*:\s*"
    r"(?P<expression>.*)$", re.I
)
_PAIR = re.compile(
    r"(?<![\w.])(?P<a>" + _NUMBER + r")\s*[-−]\s*(?P<b>" + _NUMBER
    + r")\s*=\s*(?P<value>" + _NUMBER + r")(?![\d.])"
)


def _clauses(work):
    start = 0
    for separator in _SPLIT.finditer(work):
        text = work[start : separator.start()].strip()
        if text:
            yield text, (start, separator.start())
        start = separator.end()
    if work[start:].strip():
        yield work[start:].strip(), (start, len(work))


def _parse_category_list(text, categories):
    """Exact category tokens, optional numeric annotations, and separators only."""
    text = text.strip().rstrip(".").strip()
    if re.fullmatch(r"\[\s*\]|none|empty(?:\s+set)?", text, re.I):
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    pattern = re.compile(
        r"(?:" + "|".join(re.escape(c) for c in sorted(categories, key=len, reverse=True))
        + r")(?![\w])", re.I
    )
    names = {c.casefold(): c for c in categories}
    found = []
    while text:
        match = pattern.match(text)
        if match is None:
            return None
        category = names[match.group(0).casefold()]
        if category in found:
            return None
        found.append(category)
        text = text[match.end() :].lstrip()
        annotation = re.match(r"\(\s*" + _NUMBER + r"\s*\)", text)
        if annotation:
            text = text[annotation.end() :].lstrip()
        if not text:
            break
        separator = re.match(r"(?:,\s*(?:and\b\s*)?|\band\b\s*)", text, re.I)
        if separator is None:
            return None
        text = text[separator.end() :].lstrip()
        if not text:
            return None
    return sorted(found) if found else None


def _explicit_claims(work, categories):
    claims, unsupported, provisional = [], [], []
    for clause, span in _clauses(work):
        if _PROVISIONAL.search(clause):
            provisional.append(dict(text=clause, decoded_work_span=list(span)))
            continue
        match = next(
            (m for pattern in (_PREFIX, _POSTFIX, _RANK, _RANK_CATEGORIES)
             if (m := pattern.fullmatch(clause))), None
        )
        if match is None:
            # Do not accept one understood declaration while silently ignoring
            # another unsupported assertion about what was selected or used.
            if re.search(r"\b(?:selection|selected|chosen)\b", clause, re.I) or (
                re.search(r"\b(?:used|using|included)\b", clause, re.I)
                and any(re.search(r"\b" + re.escape(c) + r"\b", clause, re.I)
                        for c in categories)
            ):
                unsupported.append(dict(text=clause, decoded_work_span=list(span)))
            continue
        selected = _parse_category_list(match.group("items"), categories)
        item = dict(text=clause, decoded_work_span=list(span), categories=selected)
        if "series" in match.groupdict():
            item["selection_series"] = match.group("series").title()
        (claims if selected is not None else unsupported).append(item)
    return claims, unsupported, provisional


def _literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("NON_LITERAL_OPERAND")


def _sum_leaves(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sum_leaves(node.left) + _sum_leaves(node.right)
    # A printed negative term is one data operand, not a separate constant.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return [*_sum_leaves(node.left), -_literal(node.right)]
    return [_literal(node)]


def _reduction_operands(expression, operation):
    # RHS is a claimed result; neither it nor a mean divisor is a selected value.
    parts = expression.replace("−", "-").split("=")
    if len(parts) != 2 or not re.fullmatch(r"\s*" + _NUMBER + r"\s*", parts[1]):
        raise ValueError("NO_COMPLETE_FINAL_EQUALITY")
    tree = ast.parse(parts[0].strip(), mode="eval").body
    if operation in {"mean", "average"}:
        if not isinstance(tree, ast.BinOp) or not isinstance(tree.op, ast.Div):
            raise ValueError("MEAN_REQUIRES_EXPLICIT_SUM_AND_DIVISOR")
        divisor = _literal(tree.right)
        if divisor == 0:
            raise ValueError("ZERO_DIVISOR")
        tree = tree.left
    values = _sum_leaves(tree)
    return values, dict(
        expression=parts[0].strip(), claimed_result=parts[1].strip(),
        exact_equality=arithmetic(parts[0].strip()) == arithmetic(parts[1].strip()),
    )


def _pair_sources(work, facts, categories):
    """Keep data-pair identities separate from whether a final reducer uses them."""
    sources = []
    for clause, span in _clauses(work):
        if _PROVISIONAL.search(clause):
            continue
        local = re.sub(
            r"^(?:calculate|compute)\s+(?:Alpha\s*[-−]\s*Beta|"
            r"\(\s*Alpha\s*[-−]\s*Beta\s*\))\s+for\s+each"
            r"(?:\s+(?:month|category))?\s*:\s*", "", clause, flags=re.I,
        )
        named = []
        for i, category in enumerate(categories):
            prefix = re.match(
                r"^\s*(?:[-*]\s*)?(?:For\s+)?" + re.escape(category) + r"\s*:\s*", local, re.I
            )
            if prefix:
                named = [i]
                local = local[prefix.end():]
                break
        local = re.sub(r"^Alpha\s*[-−]\s*Beta\s*=\s*", "", local, flags=re.I)
        # Range/extrema, examples, or unrelated prose are not category-level
        # differences. In particular, never collect arbitrary subtractions.
        match = _PAIR.fullmatch(local.strip())
        if match is None:
            continue
        a, b, value = (Fraction(match.group(k)) for k in ("a", "b", "value"))
        candidates = [
            i for i in range(len(categories))
            if facts.get(("Alpha", i)) == a and facts.get(("Beta", i)) == b
        ]
        if named:
            candidates = [i for i in candidates if i in named]
        sources.append(dict(
            value=value, index_candidates=candidates, text=clause,
            decoded_work_span=list(span), named=bool(named),
        ))
    return sources


def _trace_reductions(work, facts, categories):
    pair_sources = _pair_sources(work, facts, categories)
    claims, ambiguous, unsupported = [], [], []
    for clause, span in _clauses(work):
        match = _REDUCTION.fullmatch(clause) or _VERBAL_SUM.fullmatch(clause)
        if match is None:
            continue
        item = dict(text=clause, decoded_work_span=list(span))
        try:
            values, calculation = _reduction_operands(
                match.group("expression"), match.groupdict().get("op", "sum").lower()
            )
        except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError):
            unsupported.append(item)
            continue
        item.update(calculation, operand_values=list(map(str, values)))
        mode = match.group("mode")
        modes = ["Alpha", "Beta", "diff"] if mode is None else [
            "diff" if mode.lower().startswith("diff") else mode.title()
        ]
        candidates = []
        for candidate_mode in modes:
            mappings, provenance = [], []
            for value in values:
                if candidate_mode == "diff":
                    sources = [source for source in pair_sources
                               if source["value"] == value
                               and source["decoded_work_span"][0] < span[0]]
                    indices = sorted({i for source in sources for i in source["index_candidates"]})
                    provenance.append([
                        {k: v for k, v in source.items() if k != "value"} for source in sources
                    ])
                else:
                    indices = [i for i in range(len(categories))
                               if facts.get((candidate_mode, i)) == value]
                    provenance.append(dict(series=candidate_mode, value=str(value)))
                mappings.append(indices)
            if all(mappings):
                candidates.append(dict(mode=candidate_mode, mappings=mappings, sources=provenance))
        item["mapping_candidates"] = candidates
        if len(candidates) != 1 or any(len(m) != 1 for m in candidates[0]["mappings"]):
            ambiguous.append(item)
            continue
        indices = [m[0] for m in candidates[0]["mappings"]]
        if len(indices) != len(set(indices)):
            ambiguous.append(item)
            continue
        item["categories"] = sorted(categories[i] for i in indices)
        claims.append(item)
    return claims, ambiguous, unsupported


def verify_semantics(task, raw_record, native_score):
    """v1-compatible label envelope, with explicitly developmental S provenance."""
    result = v1.verify_semantics(task, raw_record, native_score)
    result.update(
        verifier_version=VERIFIER_VERSION, development_only=True,
        frozen_gate_eligible=False, annotation_source="DEVELOPMENT_PROGRAMMATIC_VERIFIER",
        dimensions_source_versions={d: v1.VERIFIER_VERSION for d in "POE"},
    )
    result["dimensions_source_versions"]["S"] = VERIFIER_VERSION
    if not result["normalization"]["finished"]:
        return result
    text = v1.raw_text(raw_record)
    try:
        obj, spans = v1._top_fields(text)
    except (ValueError, TypeError, RecursionError, json.JSONDecodeError, IndexError):
        return result
    if not result["normalization"]["protocol_valid"]:
        result["dimensions"]["S"] = v1._dimension(reason="INVALID_NATIVE_PROTOCOL")
        return result
    if task["query"]["family"] not in {"INTERVAL", "TOPK"}:
        result["dimensions"]["S"] = v1._dimension(reason="V2_UNSUPPORTED_FAMILY")
        return result
    work = obj.get("work")
    if not isinstance(work, str) or not work.strip():
        return result
    active, withdrawn = v1._active_work(work)
    categories = task["world"]["categories"]
    explicit, rejected, provisional = _explicit_claims(active, categories)
    try:
        facts = native.parse_facts(obj.get("evidence", []), task["world"])
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        facts = {}
    traces, ambiguity, unsupported = _trace_reductions(active, facts, categories)
    observations = dict(
        explicit_claims=explicit, final_reduction_claims=traces,
        unsupported_claims=rejected, ambiguous_traces=ambiguity,
        unsupported_reductions=unsupported, provisional_claims=provisional,
        withdrawn_work=withdrawn,
        decoded_work_span_origin="ACTIVE_WORK_AFTER_LAST_EXPLICIT_CORRECTION",
    )
    observed_sets = {tuple(c["categories"]) for c in [*explicit, *traces]}
    target = dict(
        categories=sorted(
            categories[i] for i in v1._query_api().selected(task["world"], task["query"])
        ),
        selection_series=task["query"].get("selection_series", "Alpha"),
    )
    common = dict(spans=[spans["work"]], observed=observations, target=target)
    if rejected or ambiguity or len(observed_sets) > 1:
        result["dimensions"]["S"] = v1._dimension(
            reason="V2_CONFLICTING_UNSUPPORTED_OR_AMBIGUOUS_TRACE",
            ambiguous=[*rejected, *ambiguity, *[list(s) for s in sorted(observed_sets)]], **common,
        )
    elif not observed_sets:
        result["dimensions"]["S"] = v1._dimension(
            reason="V2_NO_OBSERVABLE_FINAL_SELECTION", **common
        )
    else:
        observed = list(next(iter(observed_sets)))
        declared_series = [c["selection_series"] for c in explicit if "selection_series" in c]
        declared_series += [s.title() for s in re.findall(
            r"\b(?:selection_series|rank_series)\s*=\s*(Alpha|Beta)\b", active, re.I
        )]
        observations.update(categories=observed, selection_series=declared_series)
        okay = observed == target["categories"] and all(
            s == target["selection_series"] for s in declared_series
        )
        result["dimensions"]["S"] = v1._dimension(
            "PASS" if okay else "FAIL", "V2_ADOPTED_SET_COMPARED_TO_TARGET",
            rule="S_V2_EXPLICIT_SET" if explicit else "S_V2_EVIDENCE_BOUND_FINAL_REDUCTION",
            **common,
        )
    return result
