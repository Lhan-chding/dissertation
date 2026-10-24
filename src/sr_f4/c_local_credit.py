"""Frozen, explicit selection measurement and bounded segment-local credit.

This is a new protocol, not a replacement for any historical verifier. Labels
describe observable declarations; they are not human audits or hidden reasoning.
"""

from __future__ import annotations

import json
import random

from sr_f1.json_protocol import decode_generated
from sr_f3_ce import contract as native

from . import task_contract as contract
from .training import native_advantages
from .verification import _top_fields

VERIFIER_VERSION = "srf4-c-local-explicit-v1"
CREDIT_SCHEMA = "srf4-c-local-credit-v1"
ARMS = ("J", "EVID", "SELECT", "SHUF")
BETA_EVIDENCE = 0.25
BETA_SELECTION = 0.25
KNOWN_FRACTION = 0.75
PROTOCOL_INSTRUCTION = (
    'For this task, "work" is mandatory despite the generic optional-work contract. '
    'Its entire string must be SELECT <JSON object> COMPUTE <JSON object>, in that order, '
    'without other prose. SELECT has exactly "categories" (the adopted category-name list) '
    'and "rule". For TOPK, rule has exactly "family":"TOPK", "selection_series" '
    '("Alpha" or "Beta"), "direction" ("largest" or "smallest"), and integer "k". '
    'For INTERVAL, rule has exactly "family":"INTERVAL", "start" and "end" '
    '(category names), and booleans "include_start", "include_end", "positive_only". '
    'COMPUTE has exactly "categories", "value" ("Alpha", "Beta", or "diff"), '
    '"aggregate" ("count", "sum", "mean", "range", "min", or "max"), and '
    '"operands" (numeric values in the same order as COMPUTE categories). Its categories '
    'must equal the adopted SELECT set. Declare SELECT before computing. Keep evidence '
    'as chart facts and place the final scalar only in the outer "answer" field. '
    'Escape the internal JSON quotes correctly inside the work string.'
)


def expected_rule(task):
    """Only the query's selection fields, with visible names for interval ends."""
    q, cats = task["query"], task["world"]["categories"]
    if q["family"] == "TOPK":
        return dict(family="TOPK", selection_series=q.get("selection_series", "Alpha"),
                    direction=q.get("direction", "largest"), k=q.get("k", 2))
    if q["family"] == "INTERVAL":
        return dict(family="INTERVAL", start=cats[q["start"]], end=cats[q["end"]],
                    include_start=q.get("include_start", True),
                    include_end=q.get("include_end", True),
                    positive_only=q.get("positive_only", False))
    raise ValueError("Local selection measurement supports TOPK and INTERVAL only")


def _category_list(value, categories):
    return (isinstance(value, list) and len(value) <= len(categories)
            and all(isinstance(v, str) and v in categories for v in value)
            and len(set(value)) == len(value))


def _rule_schema(value, categories):
    if not isinstance(value, dict):
        return False
    if value.get("family") == "TOPK":
        return (set(value) == {"family", "selection_series", "direction", "k"}
                and value["selection_series"] in ("Alpha", "Beta")
                and value["direction"] in ("largest", "smallest")
                and type(value["k"]) is int and 0 <= value["k"] <= len(categories))
    if value.get("family") == "INTERVAL":
        return (set(value) == {"family", "start", "end", "include_start", "include_end",
                               "positive_only"}
                and value["start"] in categories and value["end"] in categories
                and all(type(value[k]) is bool
                        for k in ("include_start", "include_end", "positive_only")))
    return False


def _work_objects(work):
    if not isinstance(work, str) or not work.startswith("SELECT "):
        raise ValueError("ABSENT_EXACT_SELECT_COMPUTE_PROTOCOL")
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(work, 7)
    selection = native.loads(work[7:end])
    if not work[end:].startswith(" COMPUTE "):
        raise ValueError("MISSING_ORDERED_COMPUTE_DECLARATION")
    compute_start = end + len(" COMPUTE ")
    _, compute_end = decoder.raw_decode(work, compute_start)
    if work[compute_end:].strip():
        raise ValueError("EXTRA_WORK_PROSE_OR_DECLARATIONS")
    computation = native.loads(work[compute_start:compute_end])
    return selection, computation, (7, end)


def _decoded_string_spans(raw, lo, hi):
    """Map each decoded JSON-string character to its exact raw source interval."""
    if raw[lo:lo + 1] != '"' or raw[hi - 1:hi] != '"':
        raise ValueError("WORK_NOT_SERIALIZED_STRING")
    pos, chars, spans = lo + 1, [], []
    while pos < hi - 1:
        start = pos
        if raw[pos] != "\\":
            char = raw[pos]
            pos += 1
        else:
            pos += 6 if raw[pos + 1:pos + 2] == "u" else 2
            # A surrogate pair is one decoded Unicode code point.
            if (raw[start + 1:start + 2] == "u"
                    and 0xD800 <= int(raw[start + 2:start + 6], 16) <= 0xDBFF
                    and raw[pos:pos + 2] == "\\u"
                    and 0xDC00 <= int(raw[pos + 2:pos + 6], 16) <= 0xDFFF):
                pos += 6
            char = json.loads('"' + raw[start:pos] + '"')
        if len(char) != 1 or pos > hi - 1:
            raise ValueError("UNMAPPABLE_JSON_STRING_ESCAPE")
        chars.append(char)
        spans.append((start, pos))
    if "".join(chars) != json.loads(raw[lo:hi]):
        raise ValueError("JSON_STRING_MAP_MISMATCH")
    return spans


def measure(task, record):
    """Measure only explicit claims; retain native J and all UNKNOWN cases."""
    world, query = task["world"], task["query"]
    contract.validate_query(world, query)
    score = record["score"]
    if score.get("J") not in (0, 1) or type(score.get("J")) is not int:
        raise ValueError("Native J must be a binary integer")
    text = record.get("full_text")
    if not isinstance(text, str):
        raise ValueError("Raw full_text is required")
    rule = expected_rule(task)
    selected = [world["categories"][i] for i in contract.selected(world, query)]
    dimensions = {k: dict(status="UNKNOWN", reason="ABSENT_OR_UNSUPPORTED") for k in "PSOE"}
    out = dict(
        row_id=record["row_id"], qid=task["qid"], world_id=task.get("world_id"),
        verifier_version=VERIFIER_VERSION, dimensions=dimensions,
        normalization=dict(j=score["J"], protocol_valid=bool(score.get("presentation")),
                           finished=bool(score.get("finish"))
                           and record.get("generation_status", "COMPLETE") == "COMPLETE"
                           and not record.get("truncated", False),
                           defined=contract.execute(world, query) is not None,
                           sufficient=bool(score.get("certificate_sufficient"))),
        evidence_span=None, selection_span=None, reasons=[], selected_categories=None,
        expected_categories=selected, rule=None, expected_rule=rule,
        span_coordinate_system="UNICODE_CODEPOINT_RAW_FULL_TEXT", human_audited=False,
    )

    def status(key, value, reason):
        dimensions[key] = dict(status=value, reason=reason)

    try:
        obj, spans = _top_fields(text)
        end = native.first_object_end(text)
        if end is None or text[end:].strip():
            raise ValueError("NONWHITESPACE_SUFFIX")
    except (ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
        out["reasons"].append("OUTER_JSON:" + str(exc))
        return dict(out, **{k: dimensions[k]["status"] for k in "PSOE"})
    try:
        facts = contract.parse_facts(obj.get("evidence"), world)
        if not facts:
            raise ValueError("EMPTY_EVIDENCE")
        truth = all(v == world["series"][s][i] for (s, i), v in facts.items())
        status("P", "PASS" if truth else "FAIL", "SUBMITTED_NONEMPTY_CHART_FACTS")
        out["evidence_span"] = list(spans["evidence"])
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        out["reasons"].append("EVIDENCE:" + str(exc))
    try:
        selection, computation, work_span = _work_objects(obj.get("work"))
        cats = world["categories"]
        if (not isinstance(selection, dict) or set(selection) != {"categories", "rule"}
                or not _category_list(selection["categories"], cats)
                or not _rule_schema(selection["rule"], cats)):
            raise ValueError("SELECT_SCHEMA")
        out["selected_categories"] = selection["categories"]
        out["rule"] = selection["rule"]
        if (not isinstance(computation, dict)
                or not _category_list(computation.get("categories"), cats)):
            raise ValueError("COMPUTE_CATEGORIES_UNOBSERVABLE")
        if set(computation["categories"]) != set(selection["categories"]):
            raise ValueError("CONFLICTING_SELECT_COMPUTE_CATEGORIES")
        if (set(computation) != {"categories", "value", "aggregate", "operands"}
                or computation["value"] not in ("Alpha", "Beta", "diff")
                or computation["aggregate"] not in ("count", "sum", "mean", "range", "min", "max")
                or not isinstance(computation["operands"], list)
                or len(computation["operands"]) != len(computation["categories"])):
            raise ValueError("COMPUTE_SCHEMA")
        correct = selection["rule"] == rule and set(selection["categories"]) == set(selected)
        status("S", "PASS" if correct else "FAIL", "EXPLICIT_RULE_AND_ADOPTED_SET")
        mapping = _decoded_string_spans(text, *spans["work"])
        out["selection_span"] = [mapping[work_span[0]][0], mapping[work_span[1] - 1][1]]
        operands = [contract.rational(v) for v in computation["operands"]]
        target_mode = query.get("value", "Beta" if query["family"] == "TOPK" else "diff")
        expected = [contract.operand(world, query, cats.index(c))
                    for c in computation["categories"]]
        status("O", "PASS" if computation["value"] == target_mode and operands == expected
               else "FAIL", "DECLARED_VALUE_AND_CATEGORY_ALIGNED_OPERANDS")
        answer = None if obj["answer"] is None else contract.rational(obj["answer"])
        computed = contract.aggregate(operands, computation["aggregate"])
        status("E", "PASS" if computation["aggregate"] == query["aggregate"]
               and answer == computed else "FAIL", "DECLARED_REDUCTION_AND_OUTER_RESULT")
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, RecursionError) as exc:
        out["reasons"].append("WORK:" + str(exc))
    return dict(out, **{k: dimensions[k]["status"] for k in "PSOE"})


def group_credit(task, records, labels, *, arm, seed):
    """Shared reliable cohort over successes AND failures; no second whitening."""
    if arm not in ARMS or type(seed) is not int:
        raise ValueError("Registered arm and integer shuffle seed required")
    if len(records) != 8 or len(labels) != 8:
        raise ValueError("Exactly eight records and labels required")
    row_ids = [r["row_id"] for r in records]
    if (len(set(row_ids)) != 8 or len({r["group_id"] for r in records}) != 1
            or any(r["qid"] != task["qid"] for r in records)):
        raise ValueError("Group identities differ")
    js, candidates, known = [], [], []
    for i, (record, label) in enumerate(zip(records, labels, strict=True)):
        n = label["normalization"]
        if (label.get("row_id") != record["row_id"]
                or label.get("verifier_version") != VERIFIER_VERSION
                or n["j"] != record["score"]["J"] or type(n["j"]) is not int
                or n["j"] not in (0, 1)):
            raise ValueError("Semantic/native identity mismatch")
        if any(type(n[k]) is not bool for k in ("protocol_valid", "finished", "defined")):
            raise ValueError("Eligibility flags must be booleans")
        if any(label[k] not in ("PASS", "FAIL", "UNKNOWN") for k in ("P", "S")):
            raise ValueError("Invalid semantic status")
        js.append(n["j"])
        if n["protocol_valid"] and n["finished"] and n["defined"]:
            candidates.append(i)
            if label["P"] != "UNKNOWN" and label["S"] != "UNKNOWN":
                known.append(i)
    reason = ("CANDIDATE_LT2" if len(candidates) < 2 else "KNOWN_LT2" if len(known) < 2
              else "KNOWN_COVERAGE_BELOW_75_PERCENT"
              if len(known) < KNOWN_FRACTION * len(candidates) else "ACTIVE")
    active = reason == "ACTIVE"
    permutation, strata = list(range(8)), []
    rng = random.Random(seed)
    # Draw once even for non-SHUF receipts, so comparisons share the exact draw.
    if active:
        for j in (0, 1):
            for p in ("FAIL", "PASS"):
                dst = [i for i in known if js[i] == j and labels[i]["P"] == p]
                src = list(dst)
                rng.shuffle(src)
                for d, s in zip(dst, src, strict=True):
                    permutation[d] = s
                strata.append(dict(j=j, p=p, destination_indices=dst, source_indices=src))
    evidence, selection, shuffled = [0.0] * 8, [0.0] * 8, [0.0] * 8
    if active:
        p = [float(labels[i]["P"] == "PASS") for i in known]
        ps = [float(labels[i]["P"] == "PASS" and labels[i]["S"] == "PASS") for i in known]
        shuffled_ps = [float(labels[i]["P"] == "PASS"
                             and labels[permutation[i]]["S"] == "PASS") for i in known]
        for slot, i in enumerate(known):
            evidence[i] = BETA_EVIDENCE * (p[slot] - sum(p) / len(p))
            selection[i] = BETA_SELECTION * (ps[slot] - sum(ps) / len(ps))
            shuffled[i] = BETA_SELECTION * (shuffled_ps[slot] - sum(shuffled_ps) / len(ps))
    actual_e = evidence if arm != "J" else [0.0] * 8
    actual_s = selection if arm == "SELECT" else shuffled if arm == "SHUF" else [0.0] * 8
    selection_signal = active and {
        labels[i]["S"] for i in known if labels[i]["P"] == "PASS"
    } == {"PASS", "FAIL"}
    native_a = native_advantages(js)
    return dict(
        schema=CREDIT_SCHEMA, row_ids=row_ids, qid=task["qid"], world_id=task.get("world_id"),
        root_id=task.get("root_id", task.get("world_id")), group_id=records[0]["group_id"],
        arm=arm, seed=seed, labels=labels, native_advantages=native_a,
        evidence_advantages=actual_e, selection_advantages=actual_s,
        candidate_indices=candidates, known_indices=known, eligibility_reason=reason,
        active=active, signal=any(v != 0 for v in actual_e + actual_s),
        selection_signal=selection_signal, native_J_all_zero=not any(js),
        zero_native_J_selection_signal=not any(js) and selection_signal,
        permutation=permutation,
        shuffle=dict(seed=seed, applied=active and arm == "SHUF", within_j_p_strata=strata,
                     permutation_source_indices=permutation,
                     changed_count=sum(labels[i]["S"] != labels[permutation[i]]["S"]
                                       for i in range(8))),
        evidence_l1=sum(map(abs, actual_e)), selection_l1=sum(map(abs, actual_s)),
        native_l1=sum(map(abs, native_a)),
        select_minus_shuffle_l1=sum(abs(a - b) for a, b in zip(selection, shuffled, strict=True)),
        counterfactual=dict(evidence=evidence, selection=selection, shuffled_selection=shuffled),
        counterfactual_note="same_own_rollouts_diagnostic_not_causal",
    )


def token_masks(record, label, tokenizer):
    """Map recorded IDs only, by exact stable prefix decoding; never re-encode."""
    tokens, text = record["tokens"], record["full_text"]
    masks = {k + "_mask": [False] * len(tokens) for k in ("evidence", "selection")}
    diagnostics = dict(reason="EXACT_PREFIX_MAPPING", excluded_boundary_tokens={},
                       span_token_counts={}, channel_reasons={})

    def abstain(reason):
        diagnostics["reason"] = reason
        return dict(masks, diagnostics=diagnostics)

    if (label.get("row_id") != record["row_id"]
            or label.get("verifier_version") != VERIFIER_VERSION):
        raise ValueError("Token mask label identity mismatch")
    prefill = record.get("assistant_prefill", record.get("prefill"))
    if (not isinstance(prefill, str) or not isinstance(tokens, list) or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)):
        return abstain("MISSING_RECORDED_PREFILL_OR_TOKEN_IDS")
    decoded = decode_generated(tokenizer, tokens)
    if prefill + decoded != text:
        return abstain("FULL_DECODE_MISMATCH")
    previous, offsets = "", []
    for i in range(len(tokens)):
        prefix = decode_generated(tokenizer, tokens[:i + 1])
        if not decoded.startswith(prefix) or not prefix.startswith(previous):
            return abstain("UNSTABLE_TOKEN_PREFIX_DECODE")
        offsets.append((len(prefill) + len(previous), len(prefill) + len(prefix)))
        previous = prefix
    spans = [label.get(k + "_span") for k in ("evidence", "selection")]
    for span in spans:
        if span is not None and (not isinstance(span, (list, tuple)) or len(span) != 2
                                 or any(type(x) is not int for x in span)
                                 or not 0 <= span[0] < span[1] <= len(text)):
            return abstain("INVALID_RAW_CHARACTER_SPAN")
    if all(s is not None for s in spans) and max(spans[0][0], spans[1][0]) < min(
        spans[0][1], spans[1][1]
    ):
        return abstain("OVERLAPPING_AUXILIARY_SPANS")
    for name, span, dimension in zip(("evidence", "selection"), spans, ("P", "S"), strict=True):
        if span is None or label.get(dimension) == "UNKNOWN":
            diagnostics["channel_reasons"][name] = "UNKNOWN_OR_ABSENT_SPAN"
            diagnostics["span_token_counts"][name] = 0
            continue
        lo, hi = span
        mask, crossing = [], []
        for i, (a, b) in enumerate(offsets):
            inside = b > a and lo <= a and b <= hi
            mask.append(inside)
            if b > a and max(a, lo) < min(b, hi) and not inside:
                crossing.append(i)
        masks[name + "_mask"] = mask
        diagnostics["excluded_boundary_tokens"][name] = crossing
        diagnostics["span_token_counts"][name] = sum(mask)
        diagnostics["channel_reasons"][name] = "MAPPED" if any(mask) else "NO_CONTAINED_TOKENS"
    return dict(masks, diagnostics=diagnostics)


def sequence_objective(current, sampler, reference, native_advantage, *,
                       evidence_advantage=0.0, selection_advantage=0.0,
                       evidence_mask=None, selection_mask=None, kl_coefficient=0.02):
    """Native full-sequence objective plus two separate masked policy means.

    KL occurs exactly once. Each auxiliary channel averages only its own mapped
    tokens; an empty mask abstains without changing the native loss or denominator.
    """
    import torch

    from sr_f12.training import sequence_objective as native_objective

    result = native_objective(current, sampler, reference, native_advantage,
                              kl_coefficient=kl_coefficient)
    advantages, masks = [], []
    for value, mask in ((evidence_advantage, evidence_mask), (selection_advantage, selection_mask)):
        a = torch.as_tensor(value, device=current.device, dtype=torch.float32)
        if a.numel() != 1 or a.requires_grad or not bool(torch.isfinite(a)) or abs(float(a)) > .25:
            raise ValueError("One finite detached auxiliary coefficient bounded by .25 required")
        a = a.reshape(())
        if mask is None:
            m = torch.zeros_like(current, dtype=torch.bool)
        else:
            m = torch.as_tensor(mask, device=current.device)
            if m.requires_grad or m.dtype != torch.bool or m.shape != current.shape:
                raise ValueError("Detached boolean masks must align with completion tokens")
        advantages.append(a)
        masks.append(m)
    if bool((masks[0] & masks[1]).any()):
        raise ValueError("Evidence and selection masks cannot overlap")
    auxiliaries = []
    for a, mask in zip(advantages, masks, strict=True):
        policy = -result["weights"] * torch.minimum(
            a * result["ratio"], a * result["ratio"].clamp(.8, 1.2))
        auxiliary = policy[mask].mean() if bool(mask.any()) else current.sum() * 0.0
        if not bool(torch.isfinite(auxiliary)):
            raise FloatingPointError("Nonfinite local auxiliary objective")
        auxiliaries.append(auxiliary)
    result = dict(result, native_loss=result["loss"], native_policy=result["policy"],
                  evidence_policy=auxiliaries[0], selection_policy=auxiliaries[1],
                  evidence_token_count=int(masks[0].sum()),
                  selection_token_count=int(masks[1].sum()))
    result["loss"] = result["loss"] + sum(auxiliaries)
    result["policy"] = result["policy"] + sum(auxiliaries)
    return result
