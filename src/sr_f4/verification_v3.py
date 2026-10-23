"""Bounded, programmatic-only development extension of selection measurement.

Adds bounded natural declarations, month aliases, and Sum -> Mean provenance.
It does not implement model consensus, change native scoring, or open C gates.
Extraction uses response evidence; task targets are consulted only afterward.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter

from sr_f3_ce import contract as native

from . import verification as v1
from . import verification_v2 as v2

VERIFIER_VERSION = "srf4-observable-v3-programmatic-dev-r6"
_NATURAL_MEAN = re.compile(
    r"^(?:(?:the|final|their)\s+)?(?:(?P<mode>Alpha|Beta|diff(?:erences?)?)\s+)?"
    r"(?:arithmetic\s+)?(?P<op>mean|average)(?:\s+of\s+(?P<listed>\[[^\]]+\]))?"
    r"\s*(?:=|:|\bis\b)\s*(?P<expression>.*)$",
    re.I,
)
_ANAPHOR_MEAN = re.compile(
    r"^(?:(?:the|their)\s+)?(?:arithmetic\s+)?(?P<op>mean|average)\s+of\s+these"
    r"\s+(?P<mode>Alpha|Beta)\s+values\s*(?:=|:|\bis\b)\s*(?P<expression>.*)$", re.I
)
_NATURAL_SUM = re.compile(
    r"^(?:the\s+)?(?P<op>sum)\s+(?:of\s+)?(?:corresponding\s+)?"
    r"(?P<mode>Alpha|Beta|diff(?:s|erences?)?)(?:\s+values)?"
    r"(?:\s+for\s+(?:these|those)(?:\s+(?:categories|months))?)?"
    r"\s*(?:=|:|\bis\b)\s*(?P<expression>.*)$", re.I
)
_ANAPHOR_SUM = re.compile(
    r"^(?:(?P<mode>Alpha|Beta)\s+)?summing\s+(?:them|these)"
    r"\s*(?:gives|:)\s*(?P<expression>.*)$", re.I
)
_COUNT_ONLY = re.compile(
    r"(?:count|number\s+of\s+categories)\s*(?:=|:)\s*\d+"
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

_CARDINAL = r"(?:\d+|one|two|three|four|five|six|seven|eight)"
_CARDINAL_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight")
_RANK_NATURAL = re.compile(
    r"^(?:(?:the|so|identify)\s+)?(?:"
    r"(?:top|bottom)\s+" + _CARDINAL + r"(?:\s+(?:largest|smallest))?"
    r"(?:\s+(?:categories|months)\s+with\s+(?:the\s+)?(?:largest|smallest))?|"
    + _CARDINAL + r"(?:\s+(?:categories|months)\s+with\s+(?:the\s+)?)?"
    r"\s*(?:highest|lowest|largest|smallest)|"
    r"(?:highest|lowest|largest|smallest)\s+" + _CARDINAL + r"|"
    r"selected\s+(?:(?:categories|months)\s+(?:with|by)\s+(?:the\s+)?"
    r"(?:highest|lowest|largest|smallest)|top\s+" + _CARDINAL + r")|"
    r"(?:highest|lowest|largest|smallest))"
    r"\s+(?P<series>Alpha|Beta)(?:\s+(?:values|categories|months))?"
    r"(?:\s+\(in\s+(?:printed\s+)?chart\s+order\s+for\s+ties\))?"
    r"\s*(?:(?:are|is)\s*(?:in\s+|at\s+)?|:)\s*:?[ \t]*(?P<items>.+)$", re.I
)
_RANK_POSTMODIFIER = re.compile(
    r"^(?:the\s+)?" + _CARDINAL + r"\s+(?P<series>Alpha|Beta)\s+values\s+"
    r"are\s+(?:highest|lowest|largest|smallest)\s+(?:in|at)\s+(?P<items>.+)$", re.I
)


def _named_values(text, categories):
    """A complete named list with literal values, including value (name)."""
    names = {c.casefold(): c for c in categories}
    category = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    token = re.compile(
        r"(?:(?P<name>" + category + r")\s*(?:\(\s*(?P<a>" + v2._NUMBER
        + r")\s*\)|(?:[=:]|\bis\b)\s*(?P<b>" + v2._NUMBER + r"))|(?P<c>"
        + v2._NUMBER + r")\s*\(\s*(?P<reverse>" + category + r")\s*\))", re.I
    )
    result = []
    rest = text.strip()
    while rest:
        match = token.match(rest)
        if match is None:
            return None
        name = names[(match.group("name") or match.group("reverse")).casefold()]
        if any(item[0] == name for item in result):
            return None
        result.append((name, v2.Fraction(next(match.group(k) for k in "abc"
                                            if match.group(k) is not None))))
        rest = rest[match.end():].strip()
        if not rest:
            return result
        separator = re.match(r",\s*(?:and\s+)?|and\s+", rest, re.I)
        if separator is None:
            return None
        rest = rest[separator.end():]
    return None


def _complete_list(text, categories):
    parsed = v2._parse_category_list(text, categories)
    if parsed is not None:
        return parsed
    named = _named_values(text, categories)
    return sorted(name for name, _ in named) if named is not None else None


def _local_series_context(clause, categories):
    """A complete immediately preceding data list can name a rank's series."""
    match = re.fullmatch(
        r"^(?:from\s+the\s+chart,\s*)?(?:the\s+)?(?P<series>Alpha|Beta)\s+values"
        r"(?:\s*\((?P<note1>[^()]*)\))?"
        r"(?:\s+(?:for\s+(?:each\s+|all\s+)?|per\s+|by\s+|across\s+)"
        r"(?:category|categories|month|months))?"
        r"(?:\s*\((?P<note2>[^()]*)\))?\s*(?:are\s*)?:\s*(?P<items>.+)$", clause, re.I
    )
    if not match or _named_values(match.group("items"), categories) is None:
        return None
    for note in (match.group("note1"), match.group("note2")):
        if note and (re.search(
            r"\b(?:Alpha|Beta|not|except|only|wrong|use|take|selected|remove|replace|"
            r"include|exclude)\b|\d|[=+*/]", note, re.I
        ) or any(re.search(r"\b" + re.escape(c) + r"\b", note, re.I) for c in categories)):
            return None
    return match


def _quoted_spans(text):
    return [match.span() for match in re.finditer(
        r'''"(?:\\.|[^"\\])*"|(?<!\w)'(?:\\.|[^'\\])*' ''', text, re.X
    )]


def _without_quotes(text):
    characters = list(text)
    for lo, hi in _quoted_spans(text):
        characters[lo:hi] = " " * (hi - lo)
    return "".join(characters)


def _active_work(work, categories):
    """Extend withdrawal only when earlier work is explicitly repudiated."""
    corrections = list(v1._CORRECTION.finditer(_without_quotes(work)))
    point = corrections[-1].end() if corrections else 0
    active, withdrawn = work[point:], work[:point]
    withdrawal = re.compile(
        r"(?:^|(?<=[.;\n]))\s*(?:discard|ignore|withdraw)\s+(?:the\s+)?"
        r"(?:previous|earlier|initial)"
        r"\s+(?:selection|calculation|work)\s*[.;:]\s*", re.I
    )
    matches = list(withdrawal.finditer(_without_quotes(active)))
    if matches:
        point = matches[-1].end()
        withdrawn += active[:point]
        active = active[point:]
    normalized, boundaries, _ = _normalize_months(active, categories)
    category = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    commitment = re.compile(
        r"\b(?P<prefix>so\s+(?:selected(?:\s+categories)?|categories)\s*:|"
        r"so\s+I\s+will\s+assume:\s*categories\s+selected\s+are\s+|"
        + _CARDINAL + r"\s+categories\s*:|that\s+means\s+we\s+include\s+)"
        r"\s*(?P<items>" + category + r"(?:\s*,\s*(?:and\s+)?" + category
        + r"|\s+and\s+" + category + r")*)(?=\s*(?:[.;\n—)]|only\b|$))", re.I
    )
    quoted = _quoted_spans(normalized)
    markers = [m for m in re.finditer(r"\bwait\b|\bre[- ]read(?:ing)?\b|\bno\s*[—,-]",
                                     normalized, re.I)
               if not any(lo <= m.start() < hi for lo, hi in quoted)]
    for candidate in commitment.finditer(normalized):
        if any(lo <= candidate.start() < hi for lo, hi in quoted):
            continue
        lead = re.split(r"[.;\n]", normalized[:candidate.start()])[-1]
        if lead.strip(" '\u2019\""):
            continue
        selected = _complete_list(candidate.group("items"), categories)
        if selected is None:
            continue
        prior = [m for m in markers if m.end() < candidate.start()]
        assumption = "assume" in candidate.group("prefix").lower()
        if assumption:
            if not any(not any(lo <= m.start() < hi for lo, hi in quoted)
                       for m in re.finditer(r"\blet(?:'s|\u2019s|\s+us)\s+proceed\b",
                                            normalized, re.I) if m.start() >= candidate.end()):
                continue
        elif not prior:
            continue
        else:
            reconsidered = normalized[prior[-1].start():candidate.start()]
            reconsidered = _without_quotes(reconsidered)
            if re.search(r"\b(?:keep|retain|unchanged)\b|\balready\s+satisfied\b|"
                         r"\b(?:do\s+not|don't)\s+(?:change|revise)\b|"
                         r"\b(?:wrong|incorrect|never)\b|\bnot\s+(?:include|exclude|remove)\b",
                         reconsidered, re.I):
                continue
            if not re.search(r"\b(?:instruction|prompt|interpretation|start|end|"
                             r"select|selected|category|categories|excluding|including|"
                             r"exclude|include|ambiguous)\b", reconsidered, re.I):
                continue
            # A final statement cannot silently retain a member that the same
            # correction explicitly excludes (or omit one it includes).
            membership_pattern = re.compile(
                r"\b(?P<action>exclud(?:e|ing)|remove|includ(?:e|ing))\s+"
                r"(?:the\s+(?:very\s+)?(?:first|last)"
                r"(?:\s+category)?\s*\()?\s*(?P<items>" + category
                + r"(?:\s*,\s*(?:and\s+)?" + category + r"|\s+and\s+" + category + r")*)",
                re.I
            )
            # Resolve only this explicit named-role pronoun, never a free "it".
            reconsidered = re.sub(
                r"(?:start|end)\s+is\s+(?P<name>" + category + r"),\s*"
                r"(?P<action>exclude|include)\s+it\b",
                lambda m: m.group("action") + " " + m.group("name"), reconsidered, flags=re.I
            )
            memberships = list(membership_pattern.finditer(reconsidered))
            unparsed = membership_pattern.sub("", reconsidered)
            if re.search(r"\b(?:includ\w*|exclud\w*|remov\w*)\b", unparsed, re.I):
                continue
            if any((name in selected) != member.group("action").lower().startswith("includ")
                   for member in memberships
                   for name in _complete_list(member.group("items"), categories) or []):
                continue
        point = boundaries[candidate.start()]
        scope = dict(rule="V3_R6_EXPLICIT_FINAL_COMMITMENT_AFTER_RECONSIDERATION",
                     source_text=active[:point], final_commitment=candidate.group(),
                     adopted_categories=selected, explicit_assumption=assumption)
        return active[point:], withdrawn + active[:point], scope
    return active, withdrawn, None


def _r6_surface(work):
    """Mask only explicitly quoted instructions; retain offsets and quotations."""
    characters, annotations = list(work), []
    quote = re.compile(
        r"\b(?:the\s+)?(?:instruction|prompt)\s+says\s*:\s*"
        r"(?P<quote>['\"])(?P<body>Select\b[^'\"]+)(?P=quote)", re.I
    )
    for match in quote.finditer(work):
        characters[match.start():match.end()] = " " * len(match.group())
        characters[match.end() - 1] = "."
        annotations.append(dict(text=match.group(), decoded_work_span=list(match.span()),
                                rule="V3_R6_QUOTED_INSTRUCTION_IS_NOT_ADOPTION"))
    for match in re.finditer(r"\bStep\s+\d+\s*:\s*", work, re.I):
        characters[match.start():match.end()] = " " * len(match.group())
        annotations.append(dict(text=match.group(), decoded_work_span=list(match.span()),
                                rule="V3_R6_STEP_LABEL"))
    for match in re.finditer(r"\bi\.e\.", work, re.I):
        characters[match.start():match.end()] = "viz "
    return "".join(characters), annotations


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


def _r6_range_claim(clause, categories):
    category = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    names = {c.casefold(): c for c in categories}
    match = re.fullmatch(
        r"^(?:the\s+)?(?:selected\s+(?:categories|range)|selection|range)"
        r"\s*(?:is\s+|:\s*)?(?:from\s+)?(?P<a>" + category + r")\s+to\s+"
        r"(?P<b>" + category + r")(?P<tail>.*)$", clause, re.I
    )
    if match is None:
        return None
    a, b = [names[match.group(k).casefold()] for k in ("a", "b")]
    lo, hi = categories.index(a), categories.index(b)
    if lo > hi:
        return None
    tail = match.group("tail").strip().lstrip(",").strip()
    declared = None
    separator = re.search(r"\s*(?:→|->|:)\s*", tail)
    if separator:
        declared = _complete_list(re.sub(r"^only\s+", "", tail[separator.end():], flags=re.I),
                                  categories)
        if declared is None:
            return None
        tail = tail[:separator.start()].strip()
    if tail.startswith("(") and tail.endswith(")"):
        tail = tail[1:-1].strip()
    memberships = []
    if tail.lower() not in {"", "inclusive"}:
        grouped = re.fullmatch(r"(?P<action>including|excluding)\s+(?P<items>.+)", tail, re.I)
        grouped_names = _complete_list(grouped.group("items"), categories) if grouped else None
        if grouped_names is not None:
            memberships = [(name, grouped.group("action").lower() == "including")
                           for name in grouped_names]
        else:
            annotation = re.compile(
                r"(?P<action>including|excluding|inclusive\s+of|exclusive\s+of)\s+"
                r"(?:the\s+)?(?:(?P<role>start|end)\s*)?"
                r"(?:[('\"]?(?P<name>" + category + r")[)'\"]?)?", re.I
            )
            matches = list(annotation.finditer(tail))
            rest = annotation.sub("", tail)
            if not matches or re.sub(r"\band\b|[\s,]", "", rest, flags=re.I):
                return None
            for item in matches:
                role = item.group("role")
                if item.group("name"):
                    name = names[item.group("name").casefold()]
                elif role:
                    name = a if role.lower() == "start" else b
                else:
                    return None
                included = item.group("action").lower().startswith("inclus") or (
                    item.group("action").lower() == "including"
                )
                memberships.append((name, included))
        lo += any(name == a and not included for name, included in memberships)
        hi -= any(name == b and not included for name, included in memberships)
    selected = sorted(categories[lo:hi + 1])
    if any((name in selected) != included for name, included in memberships):
        return None
    if declared is not None and declared != selected:
        return None
    return dict(categories=selected, range_endpoints=[a, b],
                membership_annotations=[dict(category=name, included=value)
                                        for name, value in memberships],
                rule="V3_R6_RESPONSE_DECLARED_PRINTED_ORDER_RANGE")


def _r6_named_declaration(clause, categories):
    category = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    text = clause
    excluded = None
    exclusion = re.fullmatch(r"excluding\s+(?P<excluded>.+?),\s*we\s+(?:select|consider)"
                             r"\s+(?P<items>.+)", text, re.I)
    if exclusion:
        exclusions = re.sub(r"\s*\((?:start|end)\)", "", exclusion.group("excluded"), flags=re.I)
        excluded = _complete_list(exclusions, categories)
        text = "So we select " + exclusion.group("items")
        if excluded is None:
            return None
    pattern = re.compile(
        r"^(?:that\s+means\s+we\s+include\s+|(?:so\s+)?we\s+(?:select|adopt)\s+|"
        r"that\s+(?:leaves|gives)\s+|so\s+categories\s*:\s*|"
        + _CARDINAL + r"\s+categories\s*:\s*|"
        r"so\s+I\s+will\s+assume:\s*categories\s+selected\s+are\s+)(?P<items>.+)$", re.I
    )
    match = pattern.fullmatch(text)
    if match:
        items = re.sub(r"\s+only$", "", match.group("items"), flags=re.I)
        not_included = re.search(r"\s*—\s*but\s+NOT\s+(?P<items>.+)$", items, re.I)
        if not_included:
            excluded = _complete_list(not_included.group("items"), categories)
            items = items[:not_included.start()]
            if excluded is None:
                return None
        items = re.sub(r"\s*—\s*viz\s*,?\s*all\s+from\s+start\s+to\s+end,\s*"
                       r"despite\s+the\s+confusing\s+wording$", "", items, flags=re.I)
        selected = _complete_list(items.strip(), categories)
        if selected is not None and not any(c in selected for c in excluded or []):
            return dict(categories=selected, excluded_categories=excluded,
                        rule="V3_R6_EXPLICIT_ACTIVE_NAMED_ADOPTION")
    annotated = re.fullmatch(r"selected\s+categories:\s*(?P<items>.+)", text, re.I)
    if annotated:
        parts = re.split(r",\s*|\s+and\s+", annotated.group("items"))
        parsed = [re.fullmatch(r"(?P<name>" + category + r")\s*\((?:start|end)-included\)",
                               part, re.I) for part in parts]
        if parsed and all(parsed):
            selected = _complete_list(", ".join(m.group("name") for m in parsed), categories)
            if selected is not None:
                return dict(categories=selected, rule="V3_R6_NAMED_ITEM_MEMBERSHIP_ANNOTATIONS")
    return None


def _r6_rank_claims(work, categories):
    category = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    numbers = dict(zip(_CARDINAL_WORDS, range(1, 9), strict=True))
    clauses, claims = list(v2._clauses(work)), []
    for index, (clause, span) in enumerate(clauses):
        enumerated = re.fullmatch(
            r"(?:the\s+)?(?P<count>" + _CARDINAL + r")\s+(?P<series>Alpha|Beta)"
            r"\s+values\s+are\s*:\s*(?P<items>.+)", clause, re.I
        )
        if enumerated:
            selected = _complete_list(enumerated.group("items"), categories)
            count_text = enumerated.group("count").lower()
            count = int(count_text) if count_text.isdigit() else numbers[count_text]
            if selected is not None and len(selected) == count:
                claims.append(dict(text=clause, decoded_work_span=list(span), categories=selected,
                                   requires_linked_reduction=True,
                                   rule="V3_R6_ENUMERATED_DOMAIN_REQUIRES_COMPLETE_LINK"))
        ordinal = re.fullmatch(r"from\s+the\s+chart,\s*(?P<series>Alpha|Beta)\s+values:"
                               r"\s*(?P<items>.+)", clause, re.I)
        if ordinal:
            tokens = [re.fullmatch(r"(?P<name>" + category + r")\s*=\s*" + v2._NUMBER
                                   + r"\s*\((?P<order>largest|highest|second|third|fourth|"
                                   r"fifth|sixth|seventh|eighth)\)", item, re.I)
                      for item in re.split(r",\s*(?:and\s+)?", ordinal.group("items"))]
            orders = dict(largest=1, highest=1, second=2, third=3, fourth=4,
                          fifth=5, sixth=6, seventh=7, eighth=8)
            if tokens and all(tokens) and [orders[t.group("order").lower()] for t in tokens] == (
                list(range(1, len(tokens) + 1))
            ):
                selected = _complete_list(", ".join(t.group("name") for t in tokens), categories)
                if selected is not None:
                    claims.append(dict(text=clause, decoded_work_span=list(span),
                                       categories=selected,
                                       selection_series=ordinal.group("series").title(),
                                       rule="V3_R6_CONTIGUOUS_NAMED_ORDINAL_RANK"))
        top = re.fullmatch(r"so\s+top\s+" + _CARDINAL + r"\s+are\s+(?P<items>.+)", clause, re.I)
        if top is None:
            continue
        preceding = clauses[max(0, index - 3):index]
        for offset, (intro, intro_span) in enumerate(preceding):
            intent = re.fullmatch(
                r"identify\s+the\s+" + _CARDINAL + r"\s+categories\s+with\s+the\s+"
                r"(?:largest|highest|smallest|lowest)\s+(?P<series>Alpha|Beta)\s+values"
                r"\s+from\s+the\s+chart", intro, re.I
            )
            if intent is None:
                continue
            local = " ".join(text for text, _ in preceding[offset:])
            modes = {s.title() for s in re.findall(r"\b(?:Alpha|Beta)\b", local, re.I)}
            selected = _complete_list(top.group("items"), categories)
            if modes == {intent.group("series").title()} and selected is not None:
                claims.append(dict(text=clause, decoded_work_span=list(span), categories=selected,
                                   selection_series=intent.group("series").title(),
                                   rank_context=dict(text=local,
                                                     decoded_work_span=[intro_span[0], span[0]]),
                                   rule="V3_R6_LOCAL_SINGLE_SERIES_RANK_CONTEXT"))
    return claims


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
    endpoint_membership = re.compile(
        r"(?:start|end)\s*=\s*(?P<category>" + category + r")\s+"
        r"(?P<action>included|excluded)", re.I
    )
    ranged = re.compile(
        r"^" + prefix + r"\s*(?::\s*)?from\s+(?P<start>" + category + r")\s+to\s+"
        r"(?P<end>" + category + r"),?\s+(?P<start_action>including|excluding)\s+"
        r"(?:the\s+)?start(?:\s*\((?P<start_name>" + category + r")\))?\s+and\s+"
        r"(?P<end_action>including|excluding)\s+(?:the\s+)?end"
        r"(?:\s*\((?P<end_name>" + category + r")\))?"
        r"\s*(?::|,?\s+are\s*:|→)\s*(?P<items>.+)$", re.I
    )
    previous_clause = None
    for clause, span in v2._clauses(work):
        context = previous_clause
        previous_clause = clause, span
        r6_claim = _r6_range_claim(clause, categories) or _r6_named_declaration(clause, categories)
        if r6_claim:
            for collection in (claims, unsupported):
                collection[:] = [x for x in collection if x["decoded_work_span"] != list(span)]
            claims.append(dict(text=clause, decoded_work_span=list(span), **r6_claim))
            continue
        rank_text = re.sub(r"\s*—\s*using\s+printed\s+(?:category\s+)?order\s+for\s+ties$",
                           "", clause, flags=re.I)
        rank = _RANK_NATURAL.fullmatch(rank_text) or _RANK_POSTMODIFIER.fullmatch(rank_text)
        if rank is not None:
            selected = _complete_list(rank.group("items"), categories)
            if selected is not None:
                for collection in (claims, unsupported):
                    collection[:] = [x for x in collection
                                     if x["decoded_work_span"] != list(span)]
                claims.append(dict(text=clause, decoded_work_span=list(span),
                                   categories=selected,
                                   selection_series=rank.group("series").title(),
                                   rule="V3_R5_NATURAL_NAMED_RANK_DECLARATION"))
                continue
        contextual = re.fullmatch(
            r"^(?:the\s+|so\s+)?(?:(?:top|bottom)\s+" + _CARDINAL + r"|"
            + _CARDINAL + r"\s+(?:largest|smallest|highest|lowest))"
            r"\s*(?:are\s*:?[ \t]*|:)\s*(?P<items>.+)$", clause, re.I
        )
        if contextual and context:
            antecedent = _local_series_context(context[0], categories)
            antecedent = antecedent or re.fullmatch(
                r"^sorting\s+(?P<series>Alpha|Beta)\s+values\s+in\s+"
                r"(?:ascending|descending)\s+order:\s*(?P<items>.+)$", context[0], re.I
            )
            antecedent_values = _named_values(antecedent.group("items"), categories) \
                if antecedent else None
            selected = _complete_list(contextual.group("items"), categories)
            if antecedent_values and selected is not None:
                for collection in (claims, unsupported):
                    collection[:] = [x for x in collection
                                     if x["decoded_work_span"] != list(span)]
                claims.append(dict(text=clause, decoded_work_span=list(span), categories=selected,
                                   selection_series=antecedent.group("series").title(),
                                   rank_context=dict(text=context[0],
                                                     decoded_work_span=list(context[1])),
                                   rule="V3_R5_NATURAL_RANK_WITH_LOCAL_SERIES_CONTEXT"))
                continue
        plain = re.fullmatch(
            r"^(?:so\s+)?(?:the\s+)?(?:selected(?:\s+(?:categories|months))?|"
            r"categories\s+selected)\s*(?:are\s*:?[ \t]*|:|=)\s*(?P<items>.+)",
            clause, re.I
        )
        excluded = re.fullmatch(
            r"^(?:the\s+)?selected\s+categories\s*\(excluding\s+(?P<excluded>.+)\)"
            r"\s*:\s*(?P<items>.+)$", clause, re.I
        )
        excluded = excluded or re.fullmatch(
            r"^(?:the\s+)?selected\s+categories\s+excluding\s+(?P<excluded>.+?)"
            r"\s*(?:are|is|:)\s*(?:only\s+)?(?P<items>.+)$", clause, re.I
        )
        candidate = plain or excluded
        selected = _complete_list(candidate.group("items"), categories) if candidate else None
        if selected is not None:
            excluded_names = None
            if excluded:
                excluded_text = re.sub(r"\b(?:start|end)\s+", "", excluded.group("excluded"),
                                       flags=re.I)
                excluded_text = re.sub(r"\s*\((?:start|end)\)", "", excluded_text, flags=re.I)
                excluded_text = re.sub(r"\((" + category + r")\)", r"\1", excluded_text,
                                       flags=re.I)
                excluded_names = _complete_list(excluded_text, categories)
                if excluded_names is None or any(c in selected for c in excluded_names):
                    continue
            for collection in (claims, unsupported):
                collection[:] = [x for x in collection if x["decoded_work_span"] != list(span)]
            claims.append(dict(text=clause, decoded_work_span=list(span), categories=selected,
                               excluded_categories=excluded_names,
                               rule="V3_R5_COMPLETE_NATURAL_ADOPTION"))
            continue
        # An unsupported ranking declaration is a conflict even before a
        # comprehensible final reduction; do not silently lose its series/set.
        if (re.search(r"\b(?:top|bottom|highest|lowest|largest|smallest)\b", clause, re.I)
                and re.search(r"\b(?:are|is)\b|:", clause)
                and any(re.search(r"\b" + re.escape(c) + r"\b", clause, re.I)
                        for c in categories)
                and not any(x["decoded_work_span"] == list(span) for x in claims)
                and not v2._PROVISIONAL.search(clause)
                and not any(x["decoded_work_span"] == list(span) for x in unsupported)):
            unsupported.append(dict(text=clause, decoded_work_span=list(span),
                                    reason="UNSUPPORTED_RANK_DECLARATION"))
        match = annotated.fullmatch(clause) or ranged.fullmatch(clause)
        if match is None:
            continue
        selected = v2._parse_category_list(match.group("items"), categories)
        if selected is None:
            continue
        details = []
        if "annotations" in match.groupdict():
            annotations = match.group("annotations")
            grouped = re.fullmatch(r"(?P<action>including|excluding)\s+(?P<items>.+)",
                                   annotations, re.I)
            grouped_names = _complete_list(grouped.group("items"), categories) \
                if grouped else None
            inclusive = re.fullmatch(r"from\s+(?P<start>" + category + r")\s+to\s+"
                                     r"(?P<end>" + category + r")\s+inclusive", annotations, re.I)
            valid = False
            if grouped_names is not None:
                included = grouped.group("action").lower() == "including"
                details = [dict(category=c, included=included) for c in grouped_names]
                valid = all((x["category"] in selected) == x["included"] for x in details)
            elif inclusive:
                start, end = [names[inclusive.group(k).casefold()] for k in ("start", "end")]
                valid = selected == sorted(categories[categories.index(start):
                                                       categories.index(end) + 1])
            if valid:
                unsupported[:] = [x for x in unsupported if x["decoded_work_span"] != list(span)]
                claims.append(dict(text=clause, decoded_work_span=list(span), categories=selected,
                                   membership_annotations=details,
                                   rule="V3_R5_COMPLETE_LIST_WITH_CONSISTENT_ANNOTATION"))
                continue
            for annotation in re.split(r"\s*,\s*|\s+and\s+", match.group("annotations"),
                                       flags=re.I):
                member = (membership.fullmatch(annotation.strip())
                          or endpoint_membership.fullmatch(annotation.strip()))
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
    clauses = list(v2._clauses(work))
    for i in range(1, len(clauses) - 1):
        clause, span = clauses[i]
        sorting = re.fullmatch(r"sorting\s+(?:in\s+)?(?:ascending|descending)"
                              r"(?:\s+order)?\s*:\s*(?P<items>.+)", clause, re.I)
        conclusion, conclusion_span = clauses[i + 1]
        if not sorting or not re.fullmatch(r"these\s+are\s+the\s+(?:top|bottom)\s+"
                                           + _CARDINAL, conclusion, re.I):
            continue
        context, context_span = clauses[i - 1]
        antecedent = _local_series_context(context, categories)
        selected = _complete_list(sorting.group("items"), categories)
        if (antecedent and _named_values(antecedent.group("items"), categories)
                and selected is not None):
            claims.append(dict(text=work[span[0]:conclusion_span[1]],
                               decoded_work_span=[span[0], conclusion_span[1]],
                               categories=selected,
                               selection_series=antecedent.group("series").title(),
                               rank_context=dict(text=context,
                                                 decoded_work_span=list(context_span)),
                               rule="V3_R5_NAMED_SORT_LIST_EXPLICITLY_ADOPTED_AS_TOPK"))
    for claim in _r6_rank_claims(work, categories):
        for collection in (claims, unsupported):
            collection[:] = [x for x in collection
                             if x["decoded_work_span"] != claim["decoded_work_span"]]
        claims.append(claim)
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
    listed = match.groupdict().get("listed")
    if listed:
        literals = ast.parse(listed, mode="eval").body
        if not isinstance(literals, ast.List) or Counter(v2._literal(n) for n in literals.elts) != (
            Counter(values)
        ):
            raise ValueError("MEAN_OPERANDS_CONFLICT_WITH_LIST")
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


def _difference_sources(work, facts, categories):
    sources = v2._pair_sources(work, facts, categories)
    category = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    number = "(" + v2._NUMBER + ")"
    prefix = re.compile(
        r"^(?:(?:compute|calculate)\s+\(?Alpha\s*[-\N{MINUS SIGN}]\s*Beta\)?\s+for\s+each"
        r"(?:\s+(?:month|category))?(?:\s+from\s+" + category + r"\s+to\s+"
        + category + r")?|(?:Alpha\s*[-\N{MINUS SIGN}]\s*Beta\s+for\s+each)|"
        r"(?:calculating\s+)?differences|for\s+each\s+of\s+these\s+months,\s*"
        r"compute\s+Alpha\s*-\s*Beta)\s*:\s*", re.I
    )
    named = re.compile(
        r"^(?:(?:for|in)\s+|Alpha\s*[-\N{MINUS SIGN}]\s*Beta\s+in\s+|Alpha-)?"
        r"(?P<name>" + category + r")\s*:\s*(?P<expression>.+)$", re.I
    )
    parenthetical = re.compile(
        r"(?P<name>" + category + r")\s*\((?P<expression>[^()]*)\)", re.I
    )
    assignments = re.compile(
        r"^Alpha\s*=\s*" + number + r"\s*,\s*Beta\s*=\s*" + number
        + r"\s*(?:→|=>)\s*diff(?:erence)?\s*=\s*(?P<calculation>.+)$", re.I
    )
    hyphenated = re.compile(
        r"^Alpha-(?P<name>" + category + r")\s*:\s*" + number + r"\s*,\s*"
        r"Beta-(?P=name)\s*:\s*" + number + r"\s*(?:→|=>)\s*diff\s*=\s*"
        r"(?P<value>" + v2._NUMBER + r")$", re.I
    )
    for clause, span in v2._clauses(work):
        if v2._PROVISIONAL.search(clause):
            continue
        hyphen = hyphenated.fullmatch(clause)
        if hyphen:
            index = next(i for i, c in enumerate(categories)
                         if c.casefold() == hyphen.group("name").casefold())
            sources.append(dict(value=v2.Fraction(hyphen.group("value")),
                                index_candidates=[index], text=clause,
                                decoded_work_span=list(span), named=True,
                                rule="V3_R5_NAMED_WORK_DIFFERENCE"))
            continue
        local = prefix.sub("", clause)
        local = re.sub(
            r"^(?:for\s+each\s+selected\s+category,\s*compute\s+Alpha\s*-\s*Beta\s*:"
            r"|sum\s+of\s+Alpha\s*-\s*Beta\s+for\s+|from\s+the\s+chart\s*:)\s*",
            "", local, flags=re.I
        )
        fragments = [local]
        # Only a complete list of category-level equations may be split.
        candidates = re.split(r",\s*", local)
        if len(candidates) > 1 and all(parenthetical.fullmatch(x) or named.fullmatch(x)
                                       for x in candidates):
            fragments = candidates
        for fragment in fragments:
            match = named.fullmatch(fragment) or parenthetical.fullmatch(fragment)
            if match is None:
                continue
            expression = re.sub(r"^Alpha\s*[-\N{MINUS SIGN}]\s*Beta\s*=\s*", "",
                                match.group("expression"), flags=re.I)
            assigned = assignments.fullmatch(expression)
            if assigned:
                expression = assigned.group("calculation")
            pair = v2._PAIR.fullmatch(expression.strip())
            if pair is None and assigned and re.fullmatch(v2._NUMBER, expression.strip()):
                value = v2.Fraction(expression.strip())
            elif pair is not None:
                if assigned and tuple(v2.Fraction(assigned.group(k)) for k in (1, 2)) != (
                    v2.Fraction(pair.group("a")), v2.Fraction(pair.group("b"))
                ):
                    continue
                value = v2.Fraction(pair.group("value"))
            else:
                continue
            index = next(i for i, c in enumerate(categories)
                         if c.casefold() == match.group("name").casefold())
            source = dict(value=value, index_candidates=[index], text=clause,
                          decoded_work_span=list(span), named=True,
                          rule="V3_R5_NAMED_WORK_DIFFERENCE")
            # Prefer the explicit name over reverse lookup in structured facts.
            sources = [s for s in sources if s["decoded_work_span"] != list(span)
                       or s.get("rule") == "V3_R5_NAMED_WORK_DIFFERENCE"]
            sources.append(source)
    return sources


def _r6_single_differences(work, categories):
    category = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    assigned = re.compile(
        r"Alpha\s+in\s+(?P<name>" + category + r")\s*=\s*(?P<a>" + v2._NUMBER
        + r")\s*,\s*Beta\s+in\s+(?P=name)\s*=\s*(?P<b>" + v2._NUMBER + r")", re.I
    )
    previous, traces = None, []
    for clause, span in v2._clauses(work):
        difference = re.fullmatch(r"difference\s*=\s*(?P<expression>.+)", clause, re.I)
        match = assigned.fullmatch(previous[0]) if previous else None
        pair = v2._PAIR.fullmatch(difference.group("expression")) if difference else None
        if match and pair and all(v2.Fraction(match.group(k)) == v2.Fraction(pair.group(k))
                                  for k in ("a", "b")):
            name = next(c for c in categories if c.casefold() == match.group("name").casefold())
            traces.append(dict(text=clause, decoded_work_span=list(span), categories=[name],
                               named_pair_source=dict(text=previous[0],
                                                      decoded_work_span=list(previous[1])),
                               claimed_result=pair.group("value"),
                               rule="V3_R6_SINGLE_NAMED_PAIR_DIFFERENCE"))
        previous = clause, span
    return traces


def _base_reductions(work, facts, categories):
    traces, ambiguous, unsupported = v2._trace_reductions(work, facts, categories)
    sources = _difference_sources(work, facts, categories)
    for item in list(ambiguous):
        reduction = v2._REDUCTION.fullmatch(item["text"]) or v2._VERBAL_SUM.fullmatch(item["text"])
        if reduction and (reduction.group("mode") or "").lower() in {"alpha", "beta"}:
            continue
        if any(c["mode"] != "diff" for c in item.get("mapping_candidates", [])):
            continue
        candidates = []
        for value in item.get("operand_values", []):
            matching = [s for s in sources if str(s["value"]) == value
                        and s["decoded_work_span"][0] < item["decoded_work_span"][0]]
            indices = {i for s in matching for i in s["index_candidates"]}
            if len(indices) != 1:
                break
            candidates.append((next(iter(indices)), matching))
        else:
            indices = [i for i, _ in candidates]
            if not indices or len(indices) != len(set(indices)):
                continue
            item["categories"] = sorted(categories[i] for i in indices)
            item["mapping_candidates"] = [dict(mode="diff", mappings=[[i] for i in indices],
                                                sources=[[{k: v for k, v in s.items()
                                                           if k != "value"} for s in matches]
                                                         for _, matches in candidates])]
            item["rule"] = "V3_R5_NAMED_DIFFERENCES_TO_REDUCTION"
            ambiguous.remove(item)
            traces.append(item)
    return traces, ambiguous, unsupported


def _linked_values(work, explicit, categories):
    """Read complete response-local correspondences, never missing world facts.

    An anaphor supplies provenance only after a complete named adoption. A
    numeric list must have exactly one value per adopted category. A named list
    must name exactly that set. Subsequent reductions must use the whole value
    multiset, so equal-valued categories require no numeric reverse lookup.
    """
    bindings, conflicts = [], []
    header = re.compile(
        r"^(?:(?:for\s+(?:these|those)\s+(?:categories|months),\s*)?"
        r"(?:the\s+|their\s+)?)?(?:corresponding\s+)?(?P<mode>Alpha|Beta)"
        r"(?:\s+values)?(?:\s+for\s+(?:these|those)"
        r"(?:\s+" + _CARDINAL + r")?(?:\s+(?:categories|months))?)?"
        r"\s*(?:are|is|:)\s*:?[ \t]*(?P<items>.+)$", re.I
    )
    for clause, span in v2._clauses(work):
        local = re.sub(r"^(?:now,?\s*)?(?:sum|get)\s+(?:(?:the|their|corresponding)\s+)*"
                       r"(?=(?:Alpha|Beta)\b)", "", clause, flags=re.I)
        match = header.fullmatch(local)
        if match is None or not re.search(r"\b(?:corresponding|their|these|those)\b",
                                          clause, re.I):
            continue
        previous = [x for x in explicit if x["decoded_work_span"][1] <= span[0]]
        if not previous:
            continue
        adoption = max(previous, key=lambda x: x["decoded_work_span"][1])
        mode = match.group("mode").title()
        items = re.sub(r",?\s+respectively$", "", match.group("items"), flags=re.I)
        approximate = bool(re.match(r"approximately\s+", items, re.I))
        items = re.sub(r"^approximately\s+", "", items, flags=re.I)
        # Named work values are observed claims, independently of P truth.
        items = re.sub(r"\b" + mode + r"\s+in\s+", "", items, flags=re.I)
        items = re.sub(r"\b" + mode + r"\s*(?==|:)", "", items, flags=re.I)
        named = _named_values(items, categories)
        conflict = dict(text=clause, decoded_work_span=list(span),
                        reason="INCOMPLETE_OR_CONFLICTING_ADOPTION_CORRESPONDENCE")
        if named is not None:
            values = [value for _, value in named]
            if sorted(name for name, _ in named) != adoption["categories"]:
                conflicts.append(conflict)
                continue
        else:
            parts = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", items)
            if not parts or not all(re.fullmatch(v2._NUMBER, p) for p in parts):
                conflicts.append(conflict)
                continue
            values = [v2.Fraction(p) for p in parts]
            if len(values) != len(adoption["categories"]):
                conflicts.append(conflict)
                continue
        bindings.append(dict(text=clause, decoded_work_span=list(span),
                             categories=adoption["categories"], mode=mode,
                             values=list(map(str, values)), adoption_source=adoption.copy(),
                             value_qualification="APPROXIMATELY" if approximate else "LITERAL",
                             binding_kind="NAMED" if named is not None else "COMPLETE_ANAPHOR"))
    return bindings, conflicts


def _linked_reduction(work, span, match, values, bindings, explicit):
    previous = [b for b in bindings if b["decoded_work_span"][1] <= span[0]]
    if not previous:
        # An explicitly linked complete sum may carry its own correspondence.
        # Count alone is never sufficient without both the named adoption and
        # the response's own anaphoric link to it.
        previous_adoptions = [x for x in explicit if x["decoded_work_span"][1] <= span[0]]
        clause = work[span[0]:span[1]].strip()
        if (not previous_adoptions or not _NATURAL_SUM.fullmatch(clause)
                or not re.search(r"\b(?:corresponding|these|those|their)\b", clause, re.I)):
            return None
        adoption = max(previous_adoptions, key=lambda x: x["decoded_work_span"][1])
        if len(values) != len(adoption["categories"]):
            return None
        previous = [dict(text=clause, decoded_work_span=list(span),
                         categories=adoption["categories"], mode=match.group("mode").title(),
                         values=list(map(str, values)), adoption_source=adoption.copy(),
                         binding_kind="COMPLETE_ANAPHOR_INLINE_SUM")]
    binding = max(previous, key=lambda x: x["decoded_work_span"][1])
    mode = match.group("mode")
    if mode is not None and mode.title() != binding["mode"]:
        return None
    if Counter(values) != Counter(v2.Fraction(x) for x in binding["values"]):
        return None
    # Any intervening change of categories remains visible to the conflict
    # checks; this link itself carries no authority to withdraw earlier work.
    return dict(categories=binding["categories"], correspondence_source=binding.copy(),
                mapping_candidates=[dict(mode=binding["mode"],
                                         sources=[binding.copy()])],
                rule="V3_R5_COMPLETE_ADOPTION_LINKED_WORK_VALUES")


def _binding_disagrees(span, match, values, bindings):
    previous = [b for b in bindings if b["decoded_work_span"][1] <= span[0]]
    if not previous:
        return False
    binding = max(previous, key=lambda x: x["decoded_work_span"][1])
    mode = match.group("mode")
    return (mode is not None and mode.title() != binding["mode"]) or (
        Counter(values) != Counter(v2.Fraction(x) for x in binding["values"])
    )


def _aggregate_annotations(work, explicit, bindings, facts, categories):
    """Only complete adopted-set provenance makes scalar summaries neutral."""
    results = []
    pattern = re.compile(
        r"^(?:(?:the|their)\s+)?(?:(?P<mode>Alpha|Beta)\s+)?"
        r"(?P<op>range|sum|total|min|max|minimum|maximum)\s*(?:=|:|\bis\b)"
        r"\s*(?P<expression>.+)$", re.I
    )
    for clause, span in v2._clauses(work):
        match = pattern.fullmatch(clause)
        preceding = [x for x in explicit if x["decoded_work_span"][1] <= span[0]]
        if match is None or not preceding:
            continue
        adoption = max(preceding, key=lambda x: x["decoded_work_span"][1])
        available = [b for b in bindings if b["decoded_work_span"][1] <= span[0]
                     and b["categories"] == adoption["categories"]]
        mode = match.group("mode")
        if available:
            source = max(available, key=lambda x: x["decoded_work_span"][1])
            if mode and mode.title() != source["mode"]:
                continue
            values = [v2.Fraction(x) for x in source["values"]]
            mode = source["mode"]
        elif mode:
            mode = mode.title()
            indices = [categories.index(c) for c in adoption["categories"]]
            if any((mode, i) not in facts for i in indices):
                continue
            values = [facts[(mode, i)] for i in indices]
            source = dict(source="STRUCTURED_RESPONSE_FACTS", series=mode,
                          values=list(map(str, values)))
        else:
            continue
        if not values:
            continue
        op, expression = match.group("op").lower(), match.group("expression")
        scalar = re.fullmatch(v2._NUMBER, expression.strip())
        if scalar:
            expected = {"sum": sum(values), "total": sum(values), "min": min(values),
                        "minimum": min(values), "max": max(values), "maximum": max(values),
                        "range": max(values) - min(values)}[op]
            if v2.Fraction(expression.strip()) != expected:
                continue
            result = expression.strip()
        elif op == "range":
            pair = v2._PAIR.fullmatch(expression.strip())
            if pair is None or (v2.Fraction(pair.group("a")), v2.Fraction(pair.group("b")),
                                v2.Fraction(pair.group("value"))) != (
                                    max(values), min(values), max(values) - min(values)):
                continue
            result = pair.group("value")
        else:
            continue
        results.append(dict(text=clause, decoded_work_span=list(span),
                            categories=adoption["categories"], claimed_result=result,
                            correspondence_source=source,
                            mapping_candidates=[dict(mode=mode, sources=[source])],
                            rule="V3_R5_CONSISTENT_FULL_SET_AGGREGATE_ANNOTATION"))
    return results


def _trace_reductions(work, facts, categories, explicit=()):
    traces, ambiguous, unsupported = _base_reductions(work, facts, categories)
    traces.extend(_r6_single_differences(work, categories))
    bindings, conflicts = _linked_values(work, explicit, categories)
    ambiguous.extend(conflicts)
    annotations = _aggregate_annotations(work, explicit, bindings, facts, categories)
    for annotation in annotations:
        for collection in (traces, ambiguous, unsupported):
            collection[:] = [x for x in collection
                             if x["decoded_work_span"] != annotation["decoded_work_span"]]
        traces.append(annotation)
    reductions = []
    for clause, span in v2._clauses(work):
        natural = _NATURAL_SUM.fullmatch(clause) or _ANAPHOR_SUM.fullmatch(clause)
        match = (v2._REDUCTION.fullmatch(clause) or v2._VERBAL_SUM.fullmatch(clause)
                 or _NATURAL_MEAN.fullmatch(clause) or _ANAPHOR_MEAN.fullmatch(clause) or natural)
        if natural:
            # This bounded rewrite supplies an observed series, never the
            # task's desired one. Every operand still needs evidence binding.
            parsed = _base_reductions(
                work[:span[0]] + " " + (
                    "diff" if (natural.group("mode") or "").lower().startswith("diff")
                    else (natural.group("mode") or "")
                ) + " sum="
                + natural.group("expression"),
                facts, categories
            )
            for destination, extracted in zip(
                (traces, ambiguous, unsupported), parsed, strict=True
            ):
                for item in extracted:
                    if item["decoded_work_span"][0] < span[0]:
                        continue
                    item.update(text=clause, decoded_work_span=list(span),
                                rule="V3_EXPLICIT_SERIES_NATURAL_SUM")
                    destination.append(item)
        if match:
            reductions.append((match, span, clause))
            if match.groupdict().get("op", "sum").lower() in {"sum", "total"}:
                try:
                    values, calculation = v2._reduction_operands(match.group("expression"), "sum")
                except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError):
                    continue
                if _binding_disagrees(span, match, values, bindings):
                    ambiguous.append(dict(text=clause, decoded_work_span=list(span),
                                          reason="REDUCTION_CONFLICTS_WITH_LINKED_VALUES"))
                    continue
                linked = _linked_reduction(work, span, match, values, bindings, explicit)
                if linked is not None:
                    if (linked["correspondence_source"]["binding_kind"]
                            == "COMPLETE_ANAPHOR_INLINE_SUM" and any(
                                t["decoded_work_span"] == list(span)
                                and t["categories"] != linked["categories"] for t in traces)):
                        # A pronoun/count must not erase an independently
                        # observable contrary operand set.
                        continue
                    for collection in (traces, ambiguous, unsupported):
                        collection[:] = [x for x in collection
                                         if x["decoded_work_span"] != list(span)]
                    traces.append(dict(text=clause, decoded_work_span=list(span),
                                       operand_values=list(map(str, values)),
                                       **calculation, **linked))
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
        if len(values) > 1 and _binding_disagrees(span, match, values, bindings):
            ambiguous.append(dict(current, reason="REDUCTION_CONFLICTS_WITH_LINKED_VALUES"))
            continue
        linked = _linked_reduction(work, span, match, values, bindings, explicit)
        if linked is not None:
            numerator = sum(values, v2.Fraction())
            traces.append(dict(current, **linked, claimed_numerator=str(numerator),
                               divisor=str(divisor),
                               **_display_metadata(displays, numerator, divisor)))
            continue
        if len(values) > 1 or not earlier:
            # Reuse v2's unique response-evidence binding, keeping preceding
            # category-level differences available at their original offsets.
            canonical = (match.group("mode") or "") + " mean=" + expression + "=0"
            parsed = _base_reductions(work[:span[0]] + " " + canonical, facts, categories)
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
            r",?\s*(?:which\s+(?:is(?:\s+exactly)?|as\s+(?:an?\s+)?"
            r"(?:exact\s+|quoted\s+)?fraction\s+is)|\N{RIGHTWARDS ARROW}|->|=)\s*"
            r"(?P<a>[+-]?\d+)\s*/\s*(?P<b>[+-]?\d+)(?:\s+as\s+a\s+fraction)?",
            clause, re.I
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


def _unrecognized_adoptions(work, explicit, categories):
    """A later numeric reduction cannot conceal an earlier replacement cue."""
    recognized = {tuple(x["decoded_work_span"]) for x in explicit}
    rejected = []
    for clause, span in v2._clauses(work):
        if span in recognized or v2._PROVISIONAL.search(clause):
            continue
        if (re.search(r"\b(?:use|take|choose|replace|discard|ignore|withdraw|remove|omit|drop|"
                      r"select(?:ed|ion)?|adopt(?:ed)?|include[ds]?|exclude[ds]?|instead)\b",
                      clause, re.I)
                and any(re.search(r"\b" + re.escape(c) + r"\b", clause, re.I)
                        for c in categories)):
            rejected.append(dict(text=clause, decoded_work_span=list(span),
                                 reason="UNRESOLVED_REPLACEMENT_OR_ADOPTION"))
    return rejected


def _r6_explanation_annotations(work, explicit, categories):
    """Validate bounded restatements against a complete adopted set.

    These clauses cannot supply a selection alone. Every mentioned endpoint,
    membership operation and resulting list must agree with the adoption.
    """
    cat = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    names = {c.casefold(): c for c in categories}
    patterns = [
        ("removed", r"so we interpret as: range \[(?P<a>" + cat + r"),\s*(?P<b>" + cat
         + r")\], but remove (?P<removed>" + cat + r") → so \[(?P<items>.+)\]"),
        ("reread", r"let(?:'s|\u2019s) re-read carefully: ['\"]from (?P<a>" + cat + r") to "
         r"(?P<b>" + cat + r"), excluding the start and including the end['\"] → "
         r"exclude (?P<removed>" + cat + r"), include (?P<included>" + cat
         + r") → so (?P<items>.+)"),
        ("because", r"because ['\"]including the start \((?P<a>" + cat
         + r")\) and excluding the end \((?P<b>" + cat + r")\)['\"] implies range "
         r"\[(?P<c>" + cat + r"),\s*(?P<d>" + cat + r")\]"),
        ("modified", r"however, the phrase ['\"]from (?P<a>" + cat + r") to (?P<b>"
         + cat + r")[ '\"]* suggests the range includes both endpoints, but then "
         r"['\"]excluding the start['\"] modifies it"),
        ("note", r"but note: ['\"]excluding the start and including the end['\"] of "
         r"the range (?P<a>" + cat + r")[\u2013-](?P<b>" + cat + r") means we exclude "
         r"(?P<removed>" + cat + r"), include (?P<included>" + cat + r")"),
        ("proceed", r"let(?:'s|\u2019s) proceed with (?P<a>" + cat + r")[\u2013-](?P<b>" + cat
         + r") as requested in natural language interpretation"),
    ]
    annotations = []
    for clause, span in v2._clauses(work):
        preceding = [x for x in explicit if x["decoded_work_span"][1] <= span[0]]
        if not preceding:
            continue
        adopted = max(preceding, key=lambda x: x["decoded_work_span"][1])["categories"]
        expected = None
        if re.fullmatch(r"but then it asks for sum of \(Alpha - Beta\) over selected categories",
                        clause, re.I):
            expected = adopted
        for kind, pattern in patterns:
            match = re.fullmatch(pattern, clause, re.I)
            if match is None:
                continue
            groups = match.groupdict()
            a, b = [names[groups[k].casefold()] for k in ("a", "b")]
            lo, hi = categories.index(a), categories.index(b)
            if lo > hi:
                break
            expected = list(categories[lo:hi + 1])
            if kind == "because":
                expected = expected[:-1]
                c, d = [names[groups[k].casefold()] for k in ("c", "d")]
                if expected != categories[categories.index(c):categories.index(d) + 1]:
                    expected = None
            elif kind in {"modified", "note", "reread"}:
                expected = expected[1:]
                if groups.get("removed") and names[groups["removed"].casefold()] != a:
                    expected = None
                if groups.get("included") and names[groups["included"].casefold()] != b:
                    expected = None
            elif kind == "removed":
                removed = names[groups["removed"].casefold()]
                expected = [c for c in expected if c != removed]
            if groups.get("items") and _complete_list(groups["items"], categories) != (
                sorted(expected) if expected is not None else None
            ):
                expected = None
            break
        if expected is not None and sorted(expected) == adopted:
            annotations.append(dict(text=clause, decoded_work_span=list(span),
                                    categories=adopted,
                                    rule="V3_R6_CONSISTENT_ADOPTED_SET_EXPLANATION"))
    return annotations


def _r6_calculation_annotations(work, explicit, traces, facts, categories):
    annotations = []
    for source in _difference_sources(work, facts, categories):
        if not source.get("named"):
            continue
        preceding = [x for x in explicit
                     if x["decoded_work_span"][1] <= source["decoded_work_span"][0]]
        if preceding and all(categories[i] in preceding[-1]["categories"]
                             for i in source["index_candidates"]):
            annotations.append(dict(text=source["text"],
                                    decoded_work_span=source["decoded_work_span"],
                                    rule="V3_R6_CATEGORY_CALCULATION_WITHIN_ADOPTED_SET"))
    for clause, span in v2._clauses(work):
        preceding = [t for t in traces if t["decoded_work_span"][1] <= span[0]]
        if not preceding:
            continue
        trace = max(preceding, key=lambda x: x["decoded_work_span"][1])
        dividing = re.fullmatch(r"dividing by (?P<d>\d+) gives (?P<n>[+-]?\d+)/(?P<d2>\d+)",
                                clause, re.I)
        rounded = re.fullmatch(
            r"which as a finite decimal is (?P<decimal>" + v2._NUMBER + r") "
            r"\(rounded to (?P<places>one|two|three|four|five|six|seven|eight|[1-8]) "
            r"decimals?\) or exact fraction (?P<n>[+-]?\d+)/(?P<d>\d+)", clause, re.I
        )
        valid = False
        if dividing and int(dividing.group("d")) and dividing.group("d") == dividing.group("d2"):
            intervening = work[trace["decoded_work_span"][1]:span[0]].strip(" .;\n")
            valid = (not intervening and trace.get("claimed_result") is not None
                     and v2.Fraction(trace["claimed_result"]) == int(dividing.group("n")))
        if rounded and int(rounded.group("d")) and trace.get("result_display_kind") == (
            "ELLIPSIS_NOT_EXACT"
        ):
            value = v2.Fraction(int(rounded.group("n")), int(rounded.group("d")))
            places = rounded.group("places")
            places = int(places) if places.isdigit() else (
                _CARDINAL_WORDS.index(places.lower()) + 1)
            valid = (value == v2.Fraction(trace["claimed_numerator"])
                     / v2.Fraction(trace["divisor"])
                     and abs(v2.Fraction(rounded.group("decimal")) - value)
                     < v2.Fraction(1, 2 * 10 ** places))
        if valid:
            annotations.append(dict(text=clause, decoded_work_span=list(span),
                                    rule="V3_R6_SCALAR_DISPLAY_CONTINUATION"))
    return annotations


def _order_annotations(work, explicit, categories):
    """Recognize only ordering statements inside an already named full set."""
    category = "(?:" + "|".join(re.escape(c) for c in categories) + ")"
    names = {c.casefold(): c for c in categories}
    pair = re.compile(
        r"^using\s+the\s+printed\s+category\s+order\s+for\s+ties,\s*"
        r"(?P<a>" + category + r")\s+comes\s+before\s+(?P<b>" + category + r")$", re.I
    )
    ordered = re.compile(
        r"^since\s+(?P<a>" + category + r")\s+and\s+(?P<b>" + category + r")"
        r"\s+are\s+tied\s+for\s+largest,\s*we\s+use\s+printed\s+category\s+order"
        r"\s*—\s*so\s+we\s+take\s+(?P<items>.+)$", re.I
    )
    annotations = []
    for clause, span in v2._clauses(work):
        tied = re.fullmatch(
            r"since (?P<a>" + category + r") and (?P<b>" + category + r") are tied at "
            r"(?P<value>" + v2._NUMBER + r"), we use printed order: (?P=a) before (?P=b)",
            clause, re.I
        )
        match = pair.fullmatch(clause) or ordered.fullmatch(clause) or tied
        preceding = [x for x in explicit if x["decoded_work_span"][1] <= span[0]]
        if tied and not preceding:
            # A tie-order preface may refer to the immediately following rank.
            following = [x for x in explicit if x["decoded_work_span"][0] >= span[1]]
            if following:
                following = min(following, key=lambda x: x["decoded_work_span"][0])
                if not work[span[1]:following["decoded_work_span"][0]].strip(" .;\n"):
                    preceding = [following]
        if not match or not preceding:
            continue
        adoption = max(preceding, key=lambda x: x["decoded_work_span"][1])
        a, b = [names[match.group(k).casefold()] for k in ("a", "b")]
        if a not in adoption["categories"] or b not in adoption["categories"]:
            continue
        if categories.index(a) >= categories.index(b):
            continue
        if tied:
            rank = _RANK_NATURAL.fullmatch(adoption["text"])
            ranked = dict(_named_values(rank.group("items"), categories) or []) if rank else {}
            if not all(ranked.get(name) == v2.Fraction(tied.group("value")) for name in (a, b)):
                continue
        if "items" in match.groupdict():
            items = re.sub(r"\s+first\b|\bthen\s+", "", match.group("items"), flags=re.I)
            if _complete_list(items, categories) != adoption["categories"]:
                continue
            rank = _RANK_NATURAL.fullmatch(adoption["text"])
            ranked = _named_values(rank.group("items"), categories) if rank else None
            values = dict(ranked or [])
            if a not in values or b not in values or values[a] != values[b]:
                continue
        annotations.append(dict(text=clause, decoded_work_span=list(span),
                                categories=adoption["categories"],
                                rule="V3_R5_SAME_SET_PRINTED_ORDER_ANNOTATION"))
    return annotations


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
    categories = task["world"]["categories"]
    active, withdrawn, reconsideration = _active_work(work, categories)
    normalized, boundaries, aliases = _normalize_months(active, categories)
    normalized, surface_annotations = _r6_surface(normalized)
    explicit, rejected, provisional = _explicit_claims(normalized, categories)
    rejected.extend(_unrecognized_adoptions(normalized, explicit, categories))
    try:
        facts = native.parse_facts(obj.get("evidence", []), task["world"])
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        facts = {}
    traces, ambiguity, unsupported = _trace_reductions(normalized, facts, categories, explicit)
    # An enumerated data domain becomes an adopted set only through a complete
    # response-local correspondence and its final reduction, never by itself.
    linked_spans = {
        tuple(t["correspondence_source"]["adoption_source"]["decoded_work_span"])
        for t in traces if "adoption_source" in t.get("correspondence_source", {})
    }
    explicit[:] = [c for c in explicit if not c.get("requires_linked_reduction")
                   or tuple(c["decoded_work_span"]) in linked_spans]
    named_difference_spans = {tuple(x["decoded_work_span"])
                              for x in _difference_sources(normalized, facts, categories)
                              if x.get("rule") == "V3_R5_NAMED_WORK_DIFFERENCE"}
    rejected[:] = [x for x in rejected if not (
        tuple(x["decoded_work_span"]) in named_difference_spans
        and re.match(r"^for\s+each\s+selected\s+category,\s*compute\b", x["text"], re.I)
    )]
    order_annotations = _order_annotations(normalized, explicit, categories)
    explanation_annotations = _r6_explanation_annotations(normalized, explicit, categories)
    calculation_annotations = _r6_calculation_annotations(normalized, explicit, traces,
                                                         facts, categories)
    neutral_spans = {tuple(x["decoded_work_span"]) for x in
                     [*order_annotations, *explanation_annotations, *calculation_annotations]}
    rejected[:] = [x for x in rejected if tuple(x["decoded_work_span"]) not in neutral_spans]
    unsupported_tail = _unrecognized_tail(normalized, explicit, traces, categories)
    unsupported_tail[:] = [x for x in unsupported_tail
                           if tuple(x["decoded_work_span"]) not in neutral_spans]
    observed_sets = {tuple(c["categories"]) for c in [*explicit, *traces]}
    series = [c["selection_series"] for c in explicit if "selection_series" in c]
    series += [s.title() for s in re.findall(
        r"\b(?:selection_series|rank_series)\s*=\s*(Alpha|Beta)\b", normalized, re.I
    )]
    observations = dict(
        explicit_claims=explicit,
        final_reduction_claims=traces,
        unsupported_claims=rejected,
        ambiguous_traces=ambiguity,
        unsupported_reductions=unsupported,
        unsupported_tail=unsupported_tail,
        provisional_claims=provisional,
        withdrawn_work=withdrawn,
        reconsideration=reconsideration,
        month_aliases=aliases,
        surface_annotations=surface_annotations,
        neutral_order_annotations=order_annotations,
        neutral_explanation_annotations=explanation_annotations,
        neutral_calculation_annotations=calculation_annotations,
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
    if (rejected or ambiguity or unsupported or unsupported_tail or len(observed_sets) > 1
            or len(set(series)) > 1):
        result["dimensions"]["S"] = v1._dimension(
            reason="V3_CONFLICTING_UNSUPPORTED_OR_AMBIGUOUS_TRACE", **common
        )
    elif not observed_sets:
        result["dimensions"]["S"] = v1._dimension(
            reason="V3_NO_OBSERVABLE_FINAL_SELECTION", **common
        )
    else:
        observed = list(next(iter(observed_sets)))
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
