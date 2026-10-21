"""Conservative, evidence-bearing labels for observable SR-F4 responses.

This parser does not infer a selection from its answer. Unsupported or ambiguous
work stays UNKNOWN; labels are machine measurements, never human audit results.
"""
# ruff: noqa: RUF001

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from decimal import Decimal
from fractions import Fraction

from sr_f3_ce import contract as native
from srf4_ref.parsing import arithmetic

VERIFIER_VERSION = "srf4-observable-v1"
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?\d+)?"
_CORRECTION = re.compile(
    r"\b(?:correction|corrected final|revised final)\s*:"
    r"|\b(?:instead|actually)\s*[:,]|更正[:：]|修正[:：]",
    re.I,
)
_SELECTION = re.compile(
    r"\b(?P<key>final[ _]selection|final[ _]selected(?:[ _]categories)?|"
    r"ranked_selection|interval_selection|selected[ _](?:categories|months)|"
    r"selection|selected)\s*(?:=|:|\bare\b|\bis\b)\s*(?P<values>[^;\n.]+)",
    re.I,
)
_OPERANDS = re.compile(
    r"\b(?:(?P<mode>Alpha|Beta|diff(?:erences?)?)[ _-]+)?operands\s*=\s*\[(?P<values>[^\]]*)\]",
    re.I,
)
_REDUCER = re.compile(r"\breduce\s*=\s*(sum|mean|count|range|max|min)\s*\(\s*operands\s*\)", re.I)
_EQUATION = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<lhs>[+\-−]?(?:\d|\()[\d.\s()+\-−*/]*?)\s*=\s*"
    r"(?P<rhs>[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+\-−]?\d+)?)"
    r"(?![\d.])"
)


def raw_text(record):
    if isinstance(record, str):
        return record
    for key in ("full_text", "raw_text", "text", "completion", "response", "generated_text"):
        if isinstance(record.get(key), str):
            return record[key]
    return ""


def _dimension(
    status="UNKNOWN",
    reason="UNSUPPORTED_OR_ABSENT",
    *,
    rule=None,
    spans=(),
    observed=None,
    target=None,
    ambiguous=(),
):
    return dict(
        status=status,
        reason=reason,
        rule_id=rule,
        source_spans=[list(x) for x in spans],
        observed=observed,
        target=target,
        ambiguous_candidates=list(ambiguous),
    )


def _plain(value):
    if isinstance(value, (Decimal, Fraction)):
        return native.encode(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _top_fields(text):
    """Parse one initial object and retain top-level raw-document field spans."""
    end = native.first_object_end(text)
    if end is None:
        raise ValueError("NO_COMPLETE_INITIAL_OBJECT")
    obj = native.loads(text[:end])  # Rejects duplicates, nonfinite and oversized data.
    if not isinstance(obj, dict):
        raise ValueError("ROOT_NOT_OBJECT")
    decoder = json.JSONDecoder()
    pos = text.index("{") + 1
    spans = {}
    while pos < end:
        while text[pos].isspace() or text[pos] == ",":
            pos += 1
        if text[pos] == "}":
            break
        key, pos = decoder.raw_decode(text, pos)
        while text[pos].isspace():
            pos += 1
        if text[pos] != ":":
            raise ValueError("FIELD_SEPARATOR")
        pos += 1
        while text[pos].isspace():
            pos += 1
        start = pos
        _, pos = decoder.raw_decode(text, pos)
        spans[key] = (start, pos)
    return obj, spans


def _query_api():
    # Separate module supplies the explicitly registered extended query fields.
    from . import task_contract

    return task_contract


def _active_work(work):
    corrections = list(_CORRECTION.finditer(work))
    if not corrections:
        return work, ""
    point = corrections[-1].end()
    return work[point:], work[:point]


def _category_claims(work, categories):
    claims, rejected = [], []
    for match in _SELECTION.finditer(work):
        segment = match.group("values").strip()
        preceding = re.split(r"[;\n.]", work[: match.start()])[-1]
        if re.search(
            r"\b(?:not|incorrect|wrong|initial|initially|consider|hypothetical|discarded|rejected)\b",
            preceding,
            re.I,
        ):
            rejected.append(match.group(0))
            continue
        found = []
        rest = segment
        for category in sorted(categories, key=len, reverse=True):
            pattern = r"(?<![\w])" + re.escape(category) + r"(?![\w])"
            found.extend([category] * len(re.findall(pattern, rest, re.I)))
            rest = re.sub(pattern, " ", rest, flags=re.I)
        rest = re.sub(r"\b(?:and|only|none|empty)\b", " ", rest, flags=re.I)
        rest = re.sub(r"[\s,\[\]{}'\"()]", "", rest)
        if rest or len(found) != len(set(found)):
            rejected.append(segment)
            continue
        if found or re.search(r"\[\s*\]|\b(?:none|empty)\b", segment, re.I):
            claims.append(dict(categories=sorted(found), key=match.group("key"), text=segment))
    return claims, rejected


def _operand_claim(work, world):
    """Only an explicit operand list is mapped; no answer-candidate matching."""
    matches = list(_OPERANDS.finditer(work))
    if len(matches) != 1:
        return None, []
    match = matches[0]
    try:
        vals = [
            arithmetic(x.strip().replace("−", "-"))
            for x in match.group("values").split(",")
            if x.strip()
        ]
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None, []
    mode_text = match.group("mode")
    mode = (
        None
        if mode_text is None
        else ("diff" if mode_text.lower().startswith("diff") else mode_text.title())
    )
    candidates = []
    for candidate_mode in [mode] if mode else ["Alpha", "Beta", "diff"]:
        values = (
            world["series"][candidate_mode]
            if candidate_mode != "diff"
            else [
                a - b
                for a, b in zip(world["series"]["Alpha"], world["series"]["Beta"], strict=False)
            ]
        )
        mapping = [[i for i, value in enumerate(values) if Fraction(value) == x] for x in vals]
        if all(mapping):
            candidates.append(dict(mode=candidate_mode, index_candidates=mapping))
    if len(candidates) != 1 or any(len(x) != 1 for x in candidates[0]["index_candidates"]):
        return dict(values=vals, mode=mode, text=match.group(0), indices=None), candidates
    indices = [x[0] for x in candidates[0]["index_candidates"]]
    if len(indices) != len(set(indices)):
        return dict(values=vals, mode=mode, text=match.group(0), indices=None), candidates
    return dict(
        values=vals, mode=candidates[0]["mode"], text=match.group(0), indices=indices
    ), candidates


def _equations(work):
    results = []
    for match in _EQUATION.finditer(work):
        lhs, rhs = match.group("lhs").strip(), match.group("rhs").strip()
        try:
            left, right = arithmetic(lhs.replace("−", "-")), arithmetic(rhs.replace("−", "-"))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        results.append(
            dict(
                text=match.group(0),
                lhs=lhs,
                rhs=rhs,
                correct=left == right,
                computed=str(left),
                claimed=str(right),
            )
        )
    return results


def verify_semantics(task, raw_record, native_score):
    """Return labels plus an explicit normalization of untouched native fields.

    ``source_spans`` use Unicode-codepoint offsets in the complete raw response.
    Work judgments cite the whole serialized work field, so JSON escapes cannot
    silently corrupt an offset. They also retain the exact decoded claim.
    """
    record = raw_record if isinstance(raw_record, dict) else {"full_text": raw_record}
    world, query = task["world"], task["query"]
    api = _query_api()
    text = raw_text(record)
    target_indices = api.selected(world, query)
    target_categories = sorted(world["categories"][i] for i in target_indices)
    target_value = query.get(
        "value",
        "Beta"
        if query["family"] == "TOPK"
        else "diff"
        if query["family"] == "INTERVAL"
        else "Alpha",
    )
    defined = (
        task.get(
            "definition_status", "DEFINED" if api.execute(world, query) is not None else "UNDEFINED"
        )
        == "DEFINED"
    )
    dimensions = {key: _dimension() for key in "PSOE"}
    qid = str(task.get("qid", task.get("question_id", record.get("qid", "UNKNOWN_QID"))))
    block = record.get("block", "b0")
    block = f"b{block}" if isinstance(block, int) else block
    row_id = (
        record.get("row_id")
        or "/".join(
            str(record.get(k, "unknown"))
            for k in ("model_id", "block", "repeat", "step", "group_id", "draw")
        )
        + "/"
        + qid
    )
    finished = (
        bool(native_score.get("finish"))
        and record.get("generation_status", "COMPLETE") == "COMPLETE"
        and not record.get("truncated", False)
    )
    output = dict(
        row_id=row_id,
        qid=qid,
        world_id=task.get("world_id", task.get("root_id", world.get("world_id", qid))),
        model_id=str(record.get("model_id", "UNKNOWN_MODEL")),
        block=block,
        step=int(record.get("step", 0)),
        verifier_version=VERIFIER_VERSION,
        native_score=copy.deepcopy(native_score),
        dimensions=dimensions,
        normalization=dict(
            j=native_score.get("J"),
            protocol_valid=bool(native_score.get("presentation")),
            finished=finished,
            sufficient=bool(native_score.get("certificate_sufficient")),
            defined=defined,
        ),
        span_coordinate_system="UNICODE_CODEPOINT_RAW_RESPONSE",
        human_audited=False,
    )
    if not finished:
        for key in dimensions:
            dimensions[key] = _dimension(reason="UNFINISHED_OR_TECHNICAL_GENERATION")
        return output
    try:
        obj, spans = _top_fields(text)
    except (ValueError, TypeError, RecursionError, json.JSONDecodeError, IndexError):
        for key in dimensions:
            dimensions[key] = _dimension(reason="UNSAFE_OR_UNPARSEABLE_DOCUMENT")
        return output
    evidence = obj.get("evidence")
    if isinstance(evidence, list) and evidence:
        try:
            facts = native.parse_facts(evidence, world)
            targets = [
                dict(series=s, category=world["categories"][i], value=world["series"][s][i])
                for s, i in facts
            ]
            truth = all(v == world["series"][s][i] for (s, i), v in facts.items())
            dimensions["P"] = _dimension(
                "PASS" if truth else "FAIL",
                "NONEMPTY_FACTS_COMPARED_TO_WORLD",
                rule="P_NONEMPTY_NATIVE_FACT_TRUTH",
                spans=[spans["evidence"]],
                observed=_plain(evidence),
                target=targets,
            )
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            dimensions["P"] = _dimension(reason="INVALID_OR_UNMAPPABLE_EVIDENCE")
    else:
        dimensions["P"] = _dimension(reason="EMPTY_OR_ABSENT_EVIDENCE")
    work = obj.get("work")
    if not isinstance(work, str) or not work.strip():
        for key in "SOE":
            dimensions[key] = _dimension(reason="EMPTY_OR_ABSENT_WORK")
        return output
    active, withdrawn = _active_work(work)
    work_span = [spans["work"]]
    claims, rejected = _category_claims(active, world["categories"])
    operands, candidates = _operand_claim(active, world)
    observed_set = None
    rule = None
    distinct = {tuple(claim["categories"]) for claim in claims}
    if rejected or len(distinct) > 1:
        dimensions["S"] = _dimension(
            reason="CONFLICTING_OR_UNSUPPORTED_SELECTION_CLAIMS",
            spans=work_span,
            observed=claims,
            target=target_categories,
            ambiguous=rejected or list(map(list, distinct)),
        )
    elif len(distinct) == 1:
        observed_set = list(next(iter(distinct)))
        rule = "S_EXPLICIT_FINAL_CATEGORY_SET"
        if operands is not None and operands["indices"] is not None:
            mapped = sorted(world["categories"][i] for i in operands["indices"])
            if mapped != observed_set:
                observed_set = None
                dimensions["S"] = _dimension(
                    reason="DECLARED_SET_CONFLICTS_WITH_ADOPTED_OPERANDS",
                    spans=work_span,
                    observed=claims,
                    target=target_categories,
                    ambiguous=[mapped],
                )
    elif operands is not None and operands["indices"] is not None:
        observed_set = sorted(world["categories"][i] for i in operands["indices"])
        rule = "S_UNIQUE_EXPLICIT_OPERAND_MAPPING"
    else:
        dimensions["S"] = _dimension(
            reason="NO_UNIQUE_FINAL_SELECTION",
            spans=work_span,
            observed=active,
            target=target_categories,
            ambiguous=candidates,
        )
    if observed_set is not None:
        series_claims = re.findall(
            r"\b(?:selection_series|rank_series)\s*=\s*(Alpha|Beta)\b", active, re.I
        )
        wrong_series = series_claims and {x.title() for x in series_claims} != {
            query.get("selection_series", "Alpha")
        }
        dimensions["S"] = _dimension(
            "PASS" if observed_set == target_categories and not wrong_series else "FAIL",
            "ADOPTED_SET_COMPARED_TO_PROGRAM_TARGET",
            rule=rule,
            spans=work_span,
            observed=dict(categories=observed_set, selection_series=series_claims),
            target=dict(
                categories=target_categories,
                selection_series=query.get("selection_series", "Alpha"),
            ),
        )
    reducers = _REDUCER.findall(active)
    reducer = reducers[0].lower() if len(reducers) == 1 else None
    if (
        reducer
        and operands is not None
        and observed_set is not None
        and operands["indices"] is not None
    ):
        expected_values = [api.operand(world, query, i) for i in target_indices]
        matches = (
            reducer == query["aggregate"]
            and operands["mode"] == target_value
            and Counter(operands["values"]) == Counter(expected_values)
        )
        dimensions["O"] = _dimension(
            "PASS" if matches else "FAIL",
            "EXPLICIT_REDUCER_AND_OPERANDS",
            rule="O_EXPLICIT_DATA_DEPENDENT_REDUCE",
            spans=work_span,
            observed=dict(
                reducer=reducer, values=list(map(str, operands["values"])), mode=operands["mode"]
            ),
            target=dict(
                reducer=query["aggregate"],
                values=list(map(str, expected_values)),
                mode=target_value,
            ),
        )
    equations = _equations(active)
    errors = [equation for equation in equations if not equation["correct"]]
    execution_observed = dict(equations=equations, withdrawn_equations=_equations(withdrawn))
    if errors:
        dimensions["E"] = _dimension(
            "FAIL",
            "SURVIVING_ARITHMETIC_EQUALITY_IS_FALSE",
            rule="E_EXACT_EQUALITY",
            spans=work_span,
            observed=execution_observed,
            target="Every surviving equality is exact",
        )
    elif reducer and operands is not None:
        computed = native.aggregate(operands["values"], reducer)
        try:
            answer = None if obj.get("answer") is None else native.rational(obj["answer"])
            declared_checks = []
            for key, expected in (
                ("sum", sum(operands["values"], Fraction())),
                ("count", Fraction(len(operands["values"]))),
            ):
                for match in re.finditer(
                    r"\b" + key + r"\s*=\s*(" + _NUM + r")(?![\d.])", active, re.I
                ):
                    declared_checks.append(
                        dict(
                            name=key,
                            claimed=match.group(1),
                            correct=native.rational(match.group(1)) == expected,
                        )
                    )
            okay = computed == answer and all(x["correct"] for x in declared_checks)
            execution_observed.update(
                computed=native.encode(computed),
                answer=native.encode(answer),
                declared_checks=declared_checks,
            )
            dimensions["E"] = _dimension(
                "PASS" if okay else "FAIL",
                "EXPLICIT_COMPUTATION_AND_ANSWER_COMPARED",
                rule="E_EXACT_EXPLICIT_REDUCTION",
                spans=[*work_span, spans["answer"]],
                observed=execution_observed,
                target="All explicit equations and final reduction agree with answer",
            )
        except (ValueError, TypeError, ZeroDivisionError):
            dimensions["E"] = _dimension(
                reason="ANSWER_OR_DECLARED_ARITHMETIC_UNSUPPORTED", observed=execution_observed
            )
    else:
        dimensions["E"] = _dimension(
            reason="NO_VERIFIABLE_FINAL_COMPUTATION", spans=work_span, observed=execution_observed
        )
    return output


def observation(label):
    from srf4_ref.credit import Observation

    values = label["normalization"]
    return Observation(
        row_id=label["row_id"],
        j=float(values["j"]),
        protocol_valid=values["protocol_valid"],
        finished=values["finished"],
        sufficient=values["sufficient"],
        defined=values["defined"],
        perception=label["dimensions"]["P"]["status"],
        selection=label["dimensions"]["S"]["status"],
    )
