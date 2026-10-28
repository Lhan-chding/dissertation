"""Read-only interface-repair diagnostics; never select data or issue a gate.

The finite-library certificates, independent-draw 8x8 summaries, and matched
oracle-only control answer different questions. No weighted combined score is
formed. All original scoring, sampling, candidate rules and datasets remain
unchanged. Sixty-four cross products are not sixty-four independent samples.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from srf5_ref.oracle import validate
from srf6_ref.core import FAMILIES, enc, exact, oracle_a
from srf6_ref.design import wrong_rules, wrong_value

from .group_diagnostics import _parsed_answer

VERSION = "srf6-group-interface-audit-v1"
PARENTS = ("b0", "b1", "b2")
FACTORS = ("P0", "P1")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _task_index(private_tasks):
    tasks, roots = {}, defaultdict(lambda: defaultdict(dict))
    for task in private_tasks:
        root, family = task["root_id"], task["family"]
        factor, variant, qid = task["factor"], task["variant"], task["qid"]
        _require(
            task["split"] == "DEV_CAL"
            and family in FAMILIES
            and isinstance(root, str)
            and re.fullmatch(rf"DEV_CAL_{family}_[0-9]{{4}}", root)
            and factor in FACTORS
            and type(variant) is int
            and variant in range(4)
            and qid == f"{root}_{factor}_v{variant}",
            "Only explicit DEV_CAL four-view task identities are accepted",
        )
        _require(qid not in tasks, "Duplicate task qid")
        validate(task["world"], task["query"])
        gold = oracle_a(task["world"], task["query"])
        _require(gold["answer"] is not None, "Registered task must have a defined answer")
        _require(
            task["gold_answer"] == enc(gold["answer"])
            and task["gold_selected"] == gold["selected"]
            and task["gold_bindings"]
            == dict(zip(gold["selected"], map(enc, gold["operands"]), strict=True)),
            "Supplied task truth differs from its world/query",
        )
        tasks[qid] = task
        roots[root][factor][variant] = task
    _require(bool(tasks), "Nonempty development task set required")
    ordered = {}
    for root, factors in sorted(roots.items()):
        _require(set(factors) == set(FACTORS), "Both factors required per root")
        _require(all(set(v) == set(range(4)) for v in factors.values()), "Four views required")
        ordered[root] = {f: [factors[f][v] for v in range(4)] for f in FACTORS}
        anchor = ordered[root]["P0"][0]
        _require(ordered[root]["P1"][0]["world"] == anchor["world"], "Shared anchor differs")
        for factor in FACTORS:
            for task in ordered[root][factor]:
                _require(task["query"] == anchor["query"], "Query differs between root views")
                _require(
                    task["world"]["categories"] == anchor["world"]["categories"],
                    "Printed categories differ between root views",
                )
    return tasks, ordered


def candidate_certificates(view_tasks):
    """All-registered-rule denominators, separate from original active coverage.

    C_resp follows the original finite-pair response mask, with its undefined
    pairs counted explicitly. C_abs and alias-break abstain if an unobserved
    candidate value could change an otherwise negative conclusion.
    """
    _require(
        len(view_tasks) == 4 and [t["variant"] for t in view_tasks] == [0, 1, 2, 3],
        "Four ordered views required",
    )
    _require(
        all(
            all(t[k] == view_tasks[0][k] for k in ("root_id", "factor", "family"))
            for t in view_tasks
        ),
        "Shared root/factor/family required",
    )
    query = view_tasks[0]["query"]
    _require(all(t["query"] == query for t in view_tasks), "Shared query required")
    gold = [oracle_a(t["world"], query) for t in view_tasks]
    _require(all(g["answer"] is not None for g in gold), "Defined task answers required")
    rules = []
    for rule in wrong_rules(query):
        values = [wrong_value(t["world"], query, rule) for t in view_tasks]
        wrong = [None if v is None else v != g["answer"] for v, g in zip(values, gold, strict=True)]
        pairs = [
            None
            if values[0] is None or values[i] is None
            else values[i] - gold[i]["answer"] != values[0] - gold[0]["answer"]
            for i in (1, 2, 3)
        ]
        anchor_alias = values[0] == gold[0]["answer"] if values[0] is not None else None
        broken = (
            None
            if anchor_alias is None
            else False
            if not anchor_alias
            else True
            if any(x is True for x in wrong[1:])
            else None
            if any(x is None for x in wrong[1:])
            else False
        )
        rules.append(
            {
                "rule": rule,
                "candidate_answers": list(map(enc, values)),
                "C_resp_registered": any(x is True for x in pairs),
                "response_undefined_pairs": sum(x is None for x in pairs),
                "C_abs": True
                if any(x is True for x in wrong)
                else None
                if any(x is None for x in wrong)
                else False,
                "anchor_alias": anchor_alias,
                "C_alias_break": broken,
                "all_four_constant_nonzero_offset": (
                    None
                    if any(v is None for v in values)
                    else len({v - g["answer"] for v, g in zip(values, gold, strict=True)}) == 1
                    and values[0] != gold[0]["answer"]
                ),
            }
        )
    base = dict(zip(gold[0]["selected"], gold[0]["operands"], strict=True))
    boundaries = []
    for index in (1, 2, 3):
        current = dict(zip(gold[index]["selected"], gold[index]["operands"], strict=True))
        common = base.keys() & current.keys()
        boundaries.append(
            {
                "qid": view_tasks[index]["qid"],
                "variant": index,
                "membership_changed": set(base) != set(current),
                "binding_map_changed": base != current,
                "operand_value_changed_common_categories": sorted(
                    k for k in common if base[k] != current[k]
                ),
                "operand_sign_changed_common_categories": sorted(
                    k
                    for k in common
                    if (base[k] > 0) - (base[k] < 0) != (current[k] > 0) - (current[k] < 0)
                ),
                "selected_count_changed": len(base) != len(current),
                "mean_denominator_changed": query["aggregate"] == "mean"
                and len(base) != len(current),
                "endpoint_query_changed": False,
            }
        )
    return {
        "root_id": view_tasks[0]["root_id"],
        "factor": view_tasks[0]["factor"],
        "family": view_tasks[0]["family"],
        "candidate_denominator": len(rules),
        "candidate_denominator_kind": "ALL_REGISTERED_NOT_ACTIVE_POOL",
        "rules": rules,
        "C_semantic_boundary": boundaries,
        "original_active_C_resp_certificate": "UNCHANGED_NOT_REPLACED",
    }


def cross_product_pair(anchor_answers, edit_answers, anchor_gold, edit_gold):
    """Permutation-invariant full 8x8 answer summary; unknowns stay in 64."""
    _require(len(anchor_answers) == len(edit_answers) == 8, "Complete G8 pairs required")
    _require(anchor_gold is not None and edit_gold is not None, "Defined gold required")
    counts = Counter()
    for left in anchor_answers:
        for right in edit_answers:
            a = left is not None and left == anchor_gold
            b = right is not None and right == edit_gold
            counts["both_A_correct"] += a and b
            counts["anchor_A_correct_edit_A_zero"] += a and not b
            counts["anchor_correct_edit_known_wrong"] += a and right is not None and not b
            counts["anchor_correct_edit_unknown"] += a and right is None
            if left is None or right is None:
                counts["relative_unknown"] += 1
            else:
                counts["relative_known"] += 1
                matched = right - left == edit_gold - anchor_gold
                counts["relative_correct"] += matched
                counts["both_known_wrong_relative_correct"] += not a and not b and matched
    names = (
        "both_A_correct",
        "anchor_A_correct_edit_A_zero",
        "anchor_correct_edit_known_wrong",
        "anchor_correct_edit_unknown",
        "relative_unknown",
        "relative_known",
        "relative_correct",
        "both_known_wrong_relative_correct",
    )
    return {
        "cross_product_pairs": 64,
        "counts": {k: counts[k] for k in names},
        "rates_all_64": {k: counts[k] / 64 for k in names},
        "relative_accuracy_known": _rate(counts["relative_correct"], counts["relative_known"]),
        "independent_sample_count": "NOT_64; descriptive product of two empirical G8 distributions",
    }


def matched_oracle_control(signatures, task_index, *, answers=None):
    """Compare saved predictions with oracle on exactly the same eligible subset.

    This does not refit signatures. Optionally bind every saved observed answer
    to the corresponding raw record to reject stale projections.
    """
    counts, seen = defaultdict(Counter), set()
    for row in signatures:
        parent, factor, root, draw = (row[k] for k in ("parent", "factor", "root_id", "draw"))
        _require(parent in PARENTS and factor in FACTORS, "Signature parent/factor")
        _require(type(draw) is int and draw in range(8), "Signature draw")
        key = parent, factor, root, draw
        _require(key not in seen, "Duplicate signature identity")
        seen.add(key)
        _require(row["fit_variants"] == [0, 1, 2] and row["heldout_variant"] == 3, "Fit leakage")
        observed = [None if v is None else exact(v) for v in row["observed_answers"]]
        _require(len(observed) == 4, "Four observed answers required")
        qids = [f"{root}_{factor}_v{v}" for v in range(4)]
        _require(all(qid in task_index for qid in qids), "Signature task unavailable")
        if answers is not None:
            _require(
                all(observed[v] == answers[(parent, qids[v], draw)] for v in range(4)),
                "Saved signature observations differ from raw answers",
            )
        gold = exact(task_index[qids[3]]["gold_answer"])
        for mode in ("absolute", "response_delta"):
            cell = counts[f"{factor}/{mode}"]
            cell["all_quartets"] += 1
            prediction = row[mode]["prediction"]
            reported = row[mode]["heldout_prediction_correct"]
            if prediction is None or observed[3] is None:
                _require(reported is None, "Unevaluable signature claims correctness")
                cell["unevaluable"] += 1
                continue
            prediction = exact(prediction)
            correct, oracle_correct = prediction == observed[3], gold == observed[3]
            _require(type(reported) is int and reported == int(correct), "Stale signature result")
            _require(row[mode]["reason"] is None, "Evaluated signature has unresolved reason")
            cell["evaluable"] += 1
            cell["candidate_correct"] += correct
            cell["oracle_correct_same_subset"] += oracle_correct
            cell["candidate_only_correct"] += correct and not oracle_correct
            cell["oracle_only_correct"] += oracle_correct and not correct
    expected = {
        (parent, t["factor"], t["root_id"], draw)
        for parent in PARENTS
        for t in task_index.values()
        if t["variant"] == 0
        for draw in range(8)
    }
    _require(seen == expected, "Full supplied-task signature matrix required")
    return {
        key: {
            **dict(value),
            "net_correct_gain": value["candidate_correct"] - value["oracle_correct_same_subset"],
        }
        for key, value in sorted(counts.items())
    }


def audit_groups(records, private_tasks, *, signatures=None):
    """Audit complete G8s for all three parents on a supplied DEV_CAL task set.

    A is checked against strict original-text extraction. Other stored labels
    are not rescored here. The caller retains original provenance references.
    """
    tasks, roots = _task_index(private_tasks)
    answers, identities = {}, set()
    for row in records:
        parent, qid, draw = row["parent"], row["qid"], row["draw"]
        _require(parent in PARENTS and qid in tasks, "Unregistered answer identity")
        _require(
            type(draw) is int and draw in range(8), "Complete original G8 draw indices required"
        )
        key = parent, qid, draw
        _require(key not in identities, "Duplicate answer identity")
        identities.add(key)
        task = tasks[qid]
        _require(
            all(row[k] == task[k] for k in ("root_id", "factor", "family")),
            "Answer/task identity differs",
        )
        answer = _parsed_answer(row["raw_text"])
        expected_a = int(answer is not None and answer == exact(task["gold_answer"]))
        _require(type(row["A"]) is int and row["A"] == expected_a, "Stored A differs from raw text")
        answers[key] = answer
    expected = {(p, q, d) for p in PARENTS for q in tasks for d in range(8)}
    _require(identities == expected, "All three parents and complete G8s required")
    certificates, pairs = [], []
    for root, factors in roots.items():
        for factor, views in factors.items():
            certificates.append(candidate_certificates(views))
            for parent in PARENTS:
                anchor = views[0]
                left = [answers[(parent, anchor["qid"], d)] for d in range(8)]
                for edit in views[1:]:
                    right = [answers[(parent, edit["qid"], d)] for d in range(8)]
                    pairs.append(
                        {
                            "parent": parent,
                            "factor": factor,
                            "root_id": root,
                            "family": anchor["family"],
                            "variant": edit["variant"],
                            **cross_product_pair(
                                left,
                                right,
                                exact(anchor["gold_answer"]),
                                exact(edit["gold_answer"]),
                            ),
                        }
                    )
    strata = defaultdict(list)
    for pair in pairs:
        for family in ("ALL", pair["family"]):
            strata[f"{pair['parent']}/{pair['factor']}/{family}"].append(pair)
    summaries = {}
    for key, items in sorted(strata.items()):
        summed = {name: sum(i["counts"][name] for i in items) for name in items[0]["counts"]}
        denominator = 64 * len(items)
        summaries[key] = {
            "roots": len({i["root_id"] for i in items}),
            "view_pairs": len(items),
            "cross_product_denominator": denominator,
            "counts": summed,
            "rates_all_pairs": {k: v / denominator for k, v in summed.items()},
        }
    complete = (
        len(tasks) == 384
        and Counter(t["family"] for t in tasks.values()) == Counter({f: 64 for f in FAMILIES})
        and set(roots) == {f"DEV_CAL_{f}_{i:04d}" for f in FAMILIES for i in range(8)}
    )
    return {
        "version": VERSION,
        "status": "DERIVED_COMPLETE_DEV_CAL" if complete else "DERIVED_PARTIAL_TASK_MATRIX",
        "scope": "POSTHOC_DESCRIPTIVE_NOT_NEW_EXPERIMENT",
        "model_calls": 0,
        "optimizer_updates": 0,
        "gate_status": "NOT_SIGNED",
        "training_authorized": False,
        "source_rows": len(answers),
        "tasks": len(tasks),
        "root_certificates": certificates,
        "all_draw_pairs": pairs,
        "pair_summaries": summaries,
        "matched_oracle_control": None
        if signatures is None
        else matched_oracle_control(signatures, tasks, answers=answers),
        "claim_limit": (
            "Finite-library mathematics and empirical response distributions only; "
            "no inferred hidden trajectory, causal learning effect, or 15-percent admission rule. "
            "Cross products retain unknown answers; any inference must cluster by root, "
            "and this module performs no inferential test."
        ),
    }


def _rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--signatures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "Preserve existing diagnostic output")
    result = audit_groups(
        [row for path in args.answers for row in _rows(path)],
        _rows(args.tasks),
        signatures=None if args.signatures is None else _rows(args.signatures),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
