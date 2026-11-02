#!/usr/bin/env python3
"""Report frozen scores and saved token costs; select, but never perform, audit.

This adds descriptive tables only. It does not re-score responses, alter paired
root inference, estimate training effects, or grant follow-up authorization.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from order_probe.common import file_sha, read_jsonl, write_json, write_jsonl
from order_probe.inputs import NATURAL, PREFIX

AUDIT_SEED = 2026110203
PARENTS = ("b0", "b1", "b2")
CONDITIONS = tuple(NATURAL) + tuple(PREFIX)
EXCEPTION_PRIORITY = (
    "SELECTED_UNKNOWN", "ANSWER_UNPARSEABLE", "PREFIX_DUPLICATE_KEY",
    "ORDER_COMPLETE_VALUES_WRONG",
)


def index_unique(rows, label):
    indexed = {row["request_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate request IDs in " + label)
    return indexed


def exception_reasons(score):
    reasons = []
    if score["S"] is None:
        reasons.append("SELECTED_UNKNOWN")
    if not score["A_parseable"]:
        reasons.append("ANSWER_UNPARSEABLE")
    if score["condition"] in PREFIX and "DUPLICATE" in str(score.get("parse_error", "")).upper():
        reasons.append("PREFIX_DUPLICATE_KEY")
    if score["rank_order_known"] == 1 and score["rank_values_true"] == 0:
        reasons.append("ORDER_COMPLETE_VALUES_WRONG")
    return reasons


def select_audit(scores):
    """54 condition-stratified selections, then 18 fixed-priority exceptions.

    Sort IDs first so physical JSONL order cannot change the seeded sample.
    Rows selected in one category cannot be selected again in another category.
    """
    indexed = index_unique(scores, "scores")
    rng = random.Random(AUDIT_SEED)
    selected, selected_ids = [], set()
    for parent in PARENTS:
        for condition in CONDITIONS:
            pool = sorted(row["request_id"] for row in scores
                          if row["parent"] == parent and row["condition"] == condition)
            if len(pool) < 2:
                raise ValueError("Audit needs two rows per parent and condition")
            for rid in rng.sample(pool, 2):
                selected.append(dict(request_id=rid, selection_stage="STRATIFIED_54",
                                     selection_reason="TWO_PER_PARENT_CONDITION"))
                selected_ids.add(rid)
    if len(selected) != 54:
        raise ValueError("Unexpected audit strata")
    for reason in EXCEPTION_PRIORITY:
        pool = sorted(rid for rid, row in indexed.items()
                      if rid not in selected_ids and reason in exception_reasons(row))
        for rid in rng.sample(pool, min(72 - len(selected), len(pool))):
            selected.append(dict(request_id=rid, selection_stage="EXCEPTIONS_18",
                                 selection_reason=reason))
            selected_ids.add(rid)
    remaining = sorted(set(indexed) - selected_ids)
    if len(remaining) < 72 - len(selected):
        raise ValueError("Fewer than 72 distinct audit candidates")
    for rid in rng.sample(remaining, 72 - len(selected)):
        selected.append(dict(request_id=rid, selection_stage="EXCEPTIONS_18",
                             selection_reason="RANDOM_REMAINDER_FILL"))
        selected_ids.add(rid)
    if len(selected) != 72 or len(selected_ids) != 72:
        raise AssertionError("Audit sample must contain 72 distinct outputs")
    return selected


def ternary_counts(rows, field):
    values = [r[field] for r in rows]
    if any(value not in (None, 0, 1) for value in values):
        raise ValueError("Expected frozen ternary metric " + field)
    passed, failed, unknown = values.count(1), values.count(0), values.count(None)
    known = passed + failed
    return {field + "_pass": passed, field + "_fail": failed, field + "_unknown": unknown,
            field + "_known": known, field + "_confirmed_rate": passed / len(rows),
            field + "_pass_given_known": passed / known if known else None}


def number_summary(values):
    known = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) or v < 0 for v in known):
        raise ValueError("Nonfinite or negative token/cost observation")
    return dict(known=len(known), missing=len(values) - len(known), total=sum(known) if known else None,
                mean=statistics.mean(known) if known else None,
                median=statistics.median(known) if known else None,
                min=min(known) if known else None, max=max(known) if known else None)


def token_cost(row):
    # Never retokenize the answer. These are actual inputs and sampled IDs.
    ids = row["prompt_input_ids"]
    suffix = row["suffix_token_ids"]
    boundary = row["prefill_boundary"]
    prefix = boundary["prefill_token_ids"]
    base = boundary["base_prompt_token_count"]
    if not all(isinstance(seq, list) and all(type(x) is int for x in seq) for seq in (ids, suffix, prefix)):
        raise ValueError("Actual integer input/suffix/prefill IDs required")
    if type(base) is not int or base < 0 or ids[base:] != prefix or len(ids) != base + len(prefix):
        raise ValueError("Input/prefill token boundary inconsistent")
    if row["generated_tokens"] != len(suffix):
        raise ValueError("Sampled-token count inconsistent")
    return dict(input_tokens=len(ids), forced_prefix_tokens=len(prefix),
                context_tokens_before_prefix=base, sampled_tokens=len(suffix),
                generation_seconds=row.get("generation_seconds"))


def descriptive(scores, raw):
    raw_by_id = index_unique(raw, "raw")
    groups = defaultdict(list)
    for score in scores:
        groups[(score["parent"], score["condition"], score["rename"])].append(score)
    out = []
    for (parent, condition, rename), group in sorted(groups.items()):
        row = dict(parent=parent, condition=condition, rename=rename, N=len(group))
        for field in ("B", "rank_values_true", "order_compliance", "rank_order_print",
                      "rank_order_alphabetical", "rank_order_numeric_world", "boundary_order_print",
                      "boundary_first_r_gold", "own_order_predicts_selected"):
            row.update(ternary_counts(group, field))
        for field in ("A_parseable", "complete", "rank_order_known", "forced_evidence"):
            row[field + "_count"] = sum(r[field] for r in group)
            row[field + "_rate"] = row[field + "_count"] / len(group)
        forced = condition in PREFIX
        if any(bool(r["forced_evidence"]) != forced for r in group):
            raise ValueError("Forced-evidence flag disagrees with frozen condition")
        row["evidence_origin"] = "FORCED_INPUT" if forced else "MODEL_GENERATED"
        row["spontaneous_rank_truth_known"] = 0 if forced else row["rank_values_true_known"]
        row["spontaneous_rank_truth_rate_given_known"] = None if forced else row["rank_values_true_pass_given_known"]
        costs = [token_cost(raw_by_id[r["request_id"]]) for r in group]
        for metric in costs[0]:
            for stat, value in number_summary([c[metric] for c in costs]).items():
                row[metric + "_" + stat] = value
        out.append(row)
    return out


def validate_complete(inputs, raw, scores):
    expected = index_unique(read_jsonl(Path(inputs) / "MAIN_REQUESTS_PUBLIC.jsonl"), "requests")
    actual = index_unique(raw, "raw")
    measured = index_unique(scores, "scores")
    if len(expected) != 4320 or set(actual) != set(expected) or set(measured) != set(expected):
        raise ValueError("Need all and only 4320 MAIN raw outputs and scores; no imputation")
    tasks = {t["qid"]: t for t in read_jsonl(Path(inputs) / "MAIN_TASKS_PRIVATE.jsonl")}
    for rid, req in expected.items():
        saved, score = actual[rid], measured[rid]
        for key in ("qid", "root_id", "split", "parent", "condition", "branch", "draw"):
            if saved[key] != req[key] or score[key] != req[key]:
                raise ValueError("Row identity differs from frozen request")
        if saved["seed"] != req["seed"] or saved["assistant_prefill"] != req["assistant_prefill"]:
            raise ValueError("Seed/prefill differs from frozen request")
        if saved["full_text"] != req["assistant_prefill"] + saved["suffix_text"]:
            raise ValueError("Saved full text is not literal prefix plus suffix")
        if saved["finish_reason"] != score["finish_reason"] or score["rename"] != tasks[req["qid"]]["rename"]:
            raise ValueError("Scored finish/naming differs from saved source")
        token_cost(saved)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw, scores = read_jsonl(args.raw), read_jsonl(args.scores)
    tasks = validate_complete(args.inputs, raw, scores)
    selected = select_audit(scores)
    table = descriptive(scores, raw)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "SUPPLEMENTARY_DESCRIPTIVE.json", table)
    with (args.out / "SUPPLEMENTARY_DESCRIPTIVE.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    costs = [token_cost(row) for row in raw]
    write_json(args.out / "TOKEN_COSTS.json", dict(
        status="MEASURED_MAIN_OUTPUTS_ONLY", N=len(raw),
        metrics={key: number_summary([r[key] for r in costs]) for key in costs[0]},
        note="Input tokens include the forced prefix and expanded image tokens. Generation seconds sum may overlap across parallel workers; it is not GPU allocation wall time. Smoke and infrastructure retries are excluded and must be reported separately.",
        gpu_allocation_seconds=None, gpu_allocation_status="REQUIRES_SCHEDULER_ACCOUNTING"))
    by_id = index_unique(raw, "raw")
    blind, private = [], []
    for item in selected:
        row = by_id[item["request_id"]]
        blind.append(dict(request_id=row["request_id"], full_text=row["full_text"], finish_reason=row["finish_reason"]))
        private.append(dict(request_id=row["request_id"], task=tasks[row["qid"]]))
    audit = args.out / "AUDIT"
    write_jsonl(audit / "72_SELECTED_BLIND.jsonl", blind)
    write_jsonl(audit / "72_PRIVATE_TASKS.jsonl", private)
    write_json(audit / "SELECTION_MANIFEST.json", dict(
        seed=AUDIT_SEED, audit_status="NOT_RUN", selected=selected,
        method="54: two per parent/condition, sorted-ID seeded random sample. 18: fixed priority exceptional categories, same RNG, deduplicate; fill any remainder from all unused requests.",
        exception_priority=list(EXCEPTION_PRIORITY),
        reviewer_order="Extract fields from 72_SELECTED_BLIND before opening PRIVATE_TASKS or score/selection reasons. This is not a claim of human double-blind review; original request IDs retain their registered condition names.",
        raw_sha256=file_sha(args.raw), scores_sha256=file_sha(args.scores),
        blind_sha256=file_sha(audit / "72_SELECTED_BLIND.jsonl"),
        private_sha256=file_sha(audit / "72_PRIVATE_TASKS.jsonl")))
    write_json(args.out / "SUPPLEMENTARY_SCOPE.json", dict(
        main_comparisons_changed=False, scores_recomputed=False, model_calls=0, optimizer_updates=0,
        primary_inference="UNCHANGED_FROZEN_ANALYSIS", audit_status="NOT_RUN", audit_seed=AUDIT_SEED))
    print("Descriptive costs and 72-row review selection written; independent audit NOT_RUN")


if __name__ == "__main__":
    main()
