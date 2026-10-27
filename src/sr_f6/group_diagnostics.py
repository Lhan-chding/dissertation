"""Descriptive audits of the fixed finite-library manipulation and saved census.

No function samples a model, selects new data, changes a gate, or claims that a
compatible candidate describes a model's internal program. Unmatched/invalid
responses remain in their registered denominators.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from srf6_ref.core import enc, exact, oracle_a, seed_for
from srf6_ref.design import edit_pool, wrong_rules, wrong_value

from .credit import build_credit_group
from .data import canonical, digest
from .measurement import parse, score_response

VERSION = "srf6-fixed-group-descriptive-v1"


def _mean(values):
    return sum(values) / len(values) if values else None


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _correlation(pairs):
    if len(pairs) < 2:
        return None
    left, right = zip(*pairs, strict=True)
    ml, mr = _mean(left), _mean(right)
    covariance = sum((x - ml) * (y - mr) for x, y in pairs)
    scale = math.sqrt(sum((x - ml) ** 2 for x in left) * sum((y - mr) ** 2 for y in right))
    return covariance / scale if scale else None


def root_diagnostic(root):
    """Recompute certificates and candidate redundancy from actual edited worlds."""
    if set(root["tasks"]) != {"P0", "P1"}:
        raise ValueError("Paired four-view roots are required")
    anchor = root["tasks"]["P0"][0]
    world, query = anchor["world"], anchor["query"]
    if any(len(root["tasks"][factor]) != 4 for factor in ("P0", "P1")):
        raise ValueError("Four actual views per factor required")
    if root["tasks"]["P1"][0]["world"] != world:
        raise ValueError("Paired anchor worlds differ")
    pool = edit_pool(world, query, False, anchor["family"])
    by_id = {edit["edit_id"]: edit for edit in pool}
    rules = wrong_rules(query)
    gold = oracle_a(world, query)
    base_predictions = {rule: wrong_value(world, query, rule) for rule in rules}
    incidence = {
        rule: tuple(bool(edit["response_mask"] >> index & 1) for edit in pool)
        for index, rule in enumerate(rules)
    }
    active = [rule for rule in rules if any(incidence[rule])]
    aliases = [rule for rule in active if base_predictions[rule] == gold["answer"]]
    patterns = defaultdict(list)
    for rule in active:
        patterns[incidence[rule]].append(rule)
    patterns = list(patterns.values())
    representatives = {names[0] for names in patterns}
    certificates, descriptors = {}, {}
    for factor in ("P0", "P1"):
        tasks = root["tasks"][factor]
        coverage = set()
        edits = []
        for task in tasks[1:]:
            edit = by_id.get(task["edit_id"])
            if edit is None or edit["world"] != task["world"]:
                raise ValueError("Actual view is not its registered pool edit")
            if task["query"] != query:
                raise ValueError("A view changed the query")
            target = oracle_a(task["world"], query)
            exposed = [
                rule for index, rule in enumerate(rules) if edit["response_mask"] >> index & 1
            ]
            coverage.update(exposed)
            old_set, new_set = set(gold["selected"]), set(target["selected"])
            ranking = query.get("selection_series")
            rank_gap = None
            tie = None
            if ranking is not None:
                values = sorted(
                    task["world"]["series"][ranking], reverse=query["direction"] == "largest"
                )
                rank_gap = abs(values[query["k"] - 1] - values[query["k"]])
                tie = rank_gap == 0
            predictions = {rule: wrong_value(task["world"], query, rule) for rule in rules}
            edits.append(
                {
                    "qid": task["qid"],
                    "edit_id": edit["edit_id"],
                    "descriptor": list(edit["descriptor"]),
                    "gold_answer_delta": edit["answer_delta"],
                    "gold_selected_same": old_set == new_set,
                    "gold_selected_size_changed": len(old_set) != len(new_set),
                    "gold_membership_symmetric_difference": len(old_set ^ new_set),
                    "gold_members_replaced_same_size": old_set != new_set
                    and len(old_set) == len(new_set),
                    "edited_category": world["categories"][edit["index"]],
                    "edited_category_selected_at_anchor": world["categories"][edit["index"]]
                    in old_set,
                    "edited_category_selected_after_edit": world["categories"][edit["index"]]
                    in new_set,
                    "rank_boundary_gap": rank_gap,
                    "rank_boundary_tie": tie,
                    "operand_abs_mean": _mean([float(abs(value)) for value in target["operands"]]),
                    "absolute_answer": float(abs(target["answer"])),
                    "response_distinguished_rules": exposed,
                    "anchor_aliases_exposed": sorted(set(exposed) & set(aliases)),
                    "candidate_answers": {rule: enc(value) for rule, value in predictions.items()},
                }
            )
        descriptors[factor] = [edit["descriptor"] for edit in edits]
        certificates[factor] = {
            "coverage_count": len(coverage),
            "coverage_active": _rate(len(coverage), len(active)),
            "coverage_all_registered": _rate(len(coverage), len(rules)),
            "coverage_unique_incidence": _rate(len(coverage & representatives), len(patterns)),
            "covered_rules": sorted(coverage),
            "alias_exposed_count": len(coverage & set(aliases)),
            "alias_exposed_rules": sorted(coverage & set(aliases)),
            "boundary_swap_exposed": "boundary_swap" in coverage,
            "edits": edits,
        }
    if descriptors["P0"] != descriptors["P1"]:
        raise ValueError("P0/P1 edit cost bins differ")
    # A sensitivity audit only: do not replace the registered weighted objective.
    chosen_bins = [
        [edit for edit in pool if list(edit["descriptor"]) == description]
        for description in descriptors["P0"]
    ]
    raw_best, unique_best = 0, 0
    representative_mask = sum(1 << rules.index(rule) for rule in representatives)
    for edits in itertools.product(*chosen_bins):
        mask = 0
        for edit in edits:
            mask |= edit["response_mask"]
        raw_best = max(raw_best, mask.bit_count())
        unique_best = max(unique_best, (mask & representative_mask).bit_count())
    if raw_best != certificates["P1"]["coverage_count"]:
        raise ValueError("P1 failed its registered exact coverage maximization")
    same_edits = sum(
        left["world"] == right["world"]
        for left, right in zip(root["tasks"]["P0"][1:], root["tasks"]["P1"][1:], strict=True)
    )
    return {
        "root_id": root["root_id"],
        "family": anchor["family"],
        "split": anchor["split"],
        "forced": bool(anchor["engineered_collision_anchor"]),
        "root_payload_sha256": digest(root),
        "registered_rules": rules,
        "active_rules": active,
        "pool_blind_rules": sorted(set(rules) - set(active)),
        "active_anchor_aliases": aliases,
        "anchor_gold_answer": enc(gold["answer"]),
        "anchor_candidate_answers": {rule: enc(value) for rule, value in base_predictions.items()},
        "pool_size": len(pool),
        "pool_edit_ids": [edit["edit_id"] for edit in pool],
        "incidence_signature_classes": patterns,
        "candidate_pool_incidence": {
            rule: [int(value) for value in pattern] for rule, pattern in incidence.items()
        },
        "P0": certificates["P0"],
        "P1": certificates["P1"],
        "cost_bins_match": True,
        "gold_answer_deltas_match": True,
        "same_edit_views": same_edits,
        "identical_question_group": same_edits == 3,
        "raw_coverage_optimum": raw_best,
        "unique_incidence_coverage_optimum": unique_best,
        "P1_unique_incidence_count": len(
            set(certificates["P1"]["covered_rules"]) & representatives
        ),
    }


def _math_summary(rows):
    output = {
        "roots": len(rows),
        "same_groups": sum(row["identical_question_group"] for row in rows),
        "same_edit_views": sum(row["same_edit_views"] for row in rows),
        "edit_view_denominator": len(rows) * 3,
        "cost_bin_mismatches": sum(not row["cost_bins_match"] for row in rows),
        "mean_registered_candidates": _mean([len(row["registered_rules"]) for row in rows]),
        "mean_active_candidates": _mean([len(row["active_rules"]) for row in rows]),
        "mean_unique_incidence_patterns": _mean(
            [len(row["incidence_signature_classes"]) for row in rows]
        ),
        "pool_blind_counts": dict(
            Counter(rule for row in rows for rule in row["pool_blind_rules"])
        ),
        "candidate_active_counts": dict(
            Counter(rule for row in rows for rule in row["active_rules"])
        ),
        "active_anchor_aliases": sum(len(row["active_anchor_aliases"]) for row in rows),
        "P1_alias_count_below_P0_roots": [
            row["root_id"]
            for row in rows
            if row["P1"]["alias_exposed_count"] < row["P0"]["alias_exposed_count"]
        ],
        "forced_boundary_unexposed_P1_roots": [
            row["root_id"]
            for row in rows
            if row["forced"] and not row["P1"]["boundary_swap_exposed"]
        ],
        "P1_below_unique_incidence_optimum_roots": [
            row["root_id"]
            for row in rows
            if row["P1_unique_incidence_count"] < row["unique_incidence_coverage_optimum"]
        ],
    }
    for factor in ("P0", "P1"):
        edits = [edit for row in rows for edit in row[factor]["edits"]]
        output[factor] = {
            "candidate_coverage_count": sum(row[factor]["coverage_count"] for row in rows),
            "mean_coverage_active": _mean([row[factor]["coverage_active"] for row in rows]),
            "mean_coverage_all_registered": _mean(
                [row[factor]["coverage_all_registered"] for row in rows]
            ),
            "mean_coverage_unique_incidence": _mean(
                [row[factor]["coverage_unique_incidence"] for row in rows]
            ),
            "alias_exposed_count": sum(row[factor]["alias_exposed_count"] for row in rows),
            "gold_membership_change_rate": _mean(
                [not edit["gold_selected_same"] for edit in edits]
            ),
            "gold_same_size_replacement_rate": _mean(
                [edit["gold_members_replaced_same_size"] for edit in edits]
            ),
            "edited_anchor_selected_category_rate": _mean(
                [edit["edited_category_selected_at_anchor"] for edit in edits]
            ),
            "mean_rank_boundary_gap": _mean(
                [
                    edit["rank_boundary_gap"]
                    for edit in edits
                    if edit["rank_boundary_gap"] is not None
                ]
            ),
            "rank_boundary_tie_rate": _mean(
                [
                    edit["rank_boundary_tie"]
                    for edit in edits
                    if edit["rank_boundary_tie"] is not None
                ]
            ),
            "mean_operand_absolute_value": _mean([edit["operand_abs_mean"] for edit in edits]),
            "mean_absolute_answer": _mean([edit["absolute_answer"] for edit in edits]),
        }
    output["gap"] = {
        key: output["P1"][key] - output["P0"][key]
        for key in (
            "candidate_coverage_count",
            "mean_coverage_active",
            "mean_coverage_all_registered",
            "mean_coverage_unique_incidence",
            "alias_exposed_count",
        )
    }
    return output


def mathematical_diagnostics(roots):
    if not roots:
        raise ValueError("No actual paired roots")
    rows = [root_diagnostic(root) for root in roots]
    if len({row["root_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate root identity")
    summaries = {}
    for family in ("ALL", *sorted({row["family"] for row in rows})):
        for stratum in ("ALL", "FORCED", "NONFORCED"):
            selected = [
                row
                for row in rows
                if (family == "ALL" or row["family"] == family)
                and (stratum == "ALL" or row["forced"] == (stratum == "FORCED"))
            ]
            if selected:
                summaries[f"{family}/{stratum}"] = _math_summary(selected)
    return {
        "status": "DESCRIPTIVE_MATHEMATICAL_AUDIT",
        "version": VERSION,
        "model_calls": 0,
        "summaries": summaries,
        "roots": rows,
        "limits": [
            "Finite candidate-library manipulation only; no measured model benefit.",
            "Duplicate incidence classes are a sensitivity analysis, not new selection weights.",
            "D0 estimates the whole P1 construction policy, "
            "including unmatched difficulty attributes.",
        ],
    }


def _parsed_answer(text):
    parsed = parse(text)
    if not parsed["json_valid"]:
        return None
    try:
        return exact(parsed["object"]["answer"])
    except (KeyError, ValueError, TypeError, ArithmeticError):
        return None


def signature_prediction(tasks, answers):
    """Fit only anchor/v1/v2; retain v3 strictly as held-out response prediction.

    Absolute matching and additive-offset response matching are reported
    separately. Neither is a causal attribution to an internal program.
    """
    if len(tasks) != 4 or len(answers) != 4:
        raise ValueError("Four views are required")
    rules = ["oracle", *wrong_rules(tasks[0]["query"])]
    candidate = {
        rule: [
            oracle_a(task["world"], task["query"])["answer"]
            if rule == "oracle"
            else wrong_value(task["world"], task["query"], rule)
            for task in tasks
        ]
        for rule in rules
    }
    output = {
        "fit_variants": [0, 1, 2],
        "heldout_variant": 3,
        "observed_answers": [enc(value) for value in answers],
        "scope": "behavioral compatibility; not an inferred internal program",
    }
    for mode in ("absolute", "response_delta"):
        compatible = []
        if all(value is not None for value in answers[:3]):
            for rule, values in candidate.items():
                if any(value is None for value in values[:3]):
                    continue
                match = (
                    all(answers[index] == values[index] for index in range(3))
                    if mode == "absolute"
                    else all(
                        answers[index] - answers[0] == values[index] - values[0] for index in (1, 2)
                    )
                )
                if match:
                    compatible.append(rule)
        predictions = []
        for rule in compatible:
            values = candidate[rule]
            if values[3] is not None:
                prediction = values[3] if mode == "absolute" else answers[0] + values[3] - values[0]
                predictions.append(prediction)
        unique = sorted(set(predictions))
        if any(value is None for value in answers[:3]):
            reason = "FIT_ANSWER_UNKNOWN"
        elif not compatible:
            reason = "NO_COMPATIBLE_REGISTERED_CANDIDATE"
        elif len(unique) != 1 or len(predictions) != len(compatible):
            reason = "AMBIGUOUS_OR_UNDEFINED_PREDICTION"
        else:
            reason = None
        predicted = unique[0] if reason is None else None
        output[mode] = {
            "compatible_rules": compatible,
            "compatible_count": len(compatible),
            "prediction": enc(predicted),
            "reason": reason,
            "heldout_observed_known": answers[3] is not None,
            "heldout_prediction_correct": None
            if predicted is None or answers[3] is None
            else int(predicted == answers[3]),
            "predicted_answer_is_oracle_correct": None
            if predicted is None
            else int(predicted == candidate["oracle"][3]),
        }
    return output


def census_diagnostics(records, private_tasks, *, master=2026102701):
    """Read saved DEV_CAL records only; missing rows cannot become a signed PASS."""
    if not records or not private_tasks:
        raise ValueError("Existing records and private tasks are required")
    tasks = {task["qid"]: task for task in private_tasks}
    if len(tasks) != len(private_tasks) or any(
        task["split"] != "DEV_CAL" for task in tasks.values()
    ):
        raise ValueError("Unique DEV_CAL private task manifest required")
    roots = defaultdict(lambda: {"tasks": defaultdict(list)})
    for task in tasks.values():
        root = roots[task["root_id"]]
        root["root_id"] = task["root_id"]
        root["tasks"][task["factor"]].append(task)
    for root in roots.values():
        for factor in root["tasks"]:
            root["tasks"][factor].sort(key=lambda task: task["variant"])
    math_rows = {root_id: root_diagnostic(root) for root_id, root in roots.items()}
    grouped, lookup, policies = defaultdict(list), {}, defaultdict(set)
    row_ids = set()
    strata = defaultdict(list)
    for row in records:
        task = tasks.get(row["qid"])
        if task is None or row["parent"] not in {"b0", "b1", "b2"}:
            raise ValueError("Response is outside the supplied registered DEV_CAL manifest")
        if any(row[key] != task[key] for key in ("root_id", "factor", "family")):
            raise ValueError("Response/task identity differs")
        if type(row["draw"]) is not int or not 0 <= row["draw"] < 8:
            raise ValueError("Draw must be 0 through 7")
        key = (row["parent"], row["qid"], row["draw"])
        if key in lookup or row["row_id"] in row_ids:
            raise ValueError("Duplicate original response identity")
        row_ids.add(row["row_id"])
        text = row.get("raw_text", row.get("full_text"))
        scored = score_response(
            text,
            task["world"],
            task["query"],
            finished=bool(row["finished"]),
            finish_reason=row.get("finish_reason"),
        )
        if any(row[field] != scored[field] for field in ("A", "J", "V", "finished", "S", "B")):
            raise ValueError("Saved label differs from the frozen scorer")
        lookup[key] = (row, _parsed_answer(text))
        policies[row["parent"]].add(row["policy_hash"])
        grouped[(row["parent"], row["qid"])].append(row)
        for family in ("ALL", task["family"]):
            for population in (
                "ALL",
                "FORCED" if task["engineered_collision_anchor"] else "NONFORCED",
            ):
                strata[(row["parent"], task["factor"], family, population)].append(row)
    if any(len(values) != 1 for values in policies.values()):
        raise ValueError("A census parent changed policy content identity")
    groups = []
    for (parent, qid), rows in grouped.items():
        task = tasks[qid]
        item = {
            "parent": parent,
            "qid": qid,
            "root_id": task["root_id"],
            "family": task["family"],
            "factor": task["factor"],
            "variant": task["variant"],
            "forced": task["engineered_collision_anchor"],
            "rows": len(rows),
            "coverage_active": math_rows[task["root_id"]][task["factor"]]["coverage_active"],
        }
        if len(rows) != 8:
            item.update(
                credit_status="INCOMPLETE_G8", nonzero=None, exchangeable=None, changed=None
            )
        else:
            gid = rows[0]["group_id"]
            shuffle_seed = seed_for(master, "census_shuffle", parent, task["factor"], gid)
            credit = build_credit_group(
                rows, "U01" if task["factor"] == "P0" else "U11", shuffle_seed
            )
            cond, shuf = credit["counterfactuals"]["COND"], credit["counterfactuals"]["SHUF"]
            item.update(
                credit_status="COMPLETE_G8",
                nonzero=cond["nonzero"],
                exchangeable=cond["exchangeable"],
                changed=shuf["permutation_changed"],
                eligible_rows=cond["eligible_count"],
                credit_L1=sum(map(abs, cond["candidate"])),
            )
        groups.append(item)
    summaries = {}
    for key, rows in strata.items():
        parent, factor, family, population = key
        gg = [
            group
            for group in groups
            if group["parent"] == parent
            and group["factor"] == factor
            and (family == "ALL" or group["family"] == family)
            and (population == "ALL" or group["forced"] == (population == "FORCED"))
        ]
        result = {
            "observed_rows": len(rows),
            "A_rate": _mean([row["A"] for row in rows]),
            "J_rate": _mean([row["J"] for row in rows]),
            "V_rate": _mean([row["V"] for row in rows]),
            "S_known_rate": _mean([row["S"] is not None for row in rows]),
            "B_known_rate": _mean([row["B"] is not None for row in rows]),
            "groups_observed": len(gg),
            "groups_with_complete_G8": sum(group["credit_status"] == "COMPLETE_G8" for group in gg),
            "nonzero_groups": sum(group["nonzero"] is True for group in gg),
            "actually_changed_groups": sum(group["changed"] is True for group in gg),
        }
        for view_type in ("anchor", "edited"):
            selected = [
                row
                for row in rows
                if (tasks[row["qid"]]["variant"] == 0) == (view_type == "anchor")
            ]
            result[view_type] = {
                "rows": len(selected),
                "A_rate": _mean([row["A"] for row in selected]),
                "J_rate": _mean([row["J"] for row in selected]),
            }
        by_root = defaultdict(list)
        for group in gg:
            if group["nonzero"] is not None:
                by_root[group["root_id"]].append(group)
        pairs = [
            (items[0]["coverage_active"], _mean([int(group["nonzero"]) for group in items]))
            for items in by_root.values()
            if len(items) == 4
        ]
        result["coverage_nonzero_group_fraction_pearson"] = _correlation(pairs)
        result["correlation_root_count"] = len(pairs)
        result["correlation_incomplete_roots_excluded"] = sum(
            len(items) != 4 for items in by_root.values()
        )
        result["correlation_inference"] = "descriptive_only; no p value or causal claim"
        summaries["/".join(key)] = result
    pairs, signatures = [], []
    for parent in ("b0", "b1", "b2"):
        for root_id, root in roots.items():
            for factor, view_tasks in root["tasks"].items():
                for draw in range(8):
                    observed = [lookup.get((parent, task["qid"], draw)) for task in view_tasks]
                    if all(item is not None for item in observed):
                        signature = signature_prediction(view_tasks, [item[1] for item in observed])
                        signatures.append(
                            {
                                "parent": parent,
                                "root_id": root_id,
                                "factor": factor,
                                "family": view_tasks[0]["family"],
                                "forced": view_tasks[0]["engineered_collision_anchor"],
                                "draw": draw,
                                **signature,
                            }
                        )
                    anchor = observed[0]
                    for variant in range(1, 4):
                        edited = observed[variant]
                        if anchor is None or edited is None:
                            continue
                        gold_delta = exact(view_tasks[variant]["gold_answer"]) - exact(
                            view_tasks[0]["gold_answer"]
                        )
                        relative = (
                            None
                            if anchor[1] is None or edited[1] is None
                            else int(edited[1] - anchor[1] == gold_delta)
                        )
                        pairs.append(
                            {
                                "parent": parent,
                                "root_id": root_id,
                                "family": view_tasks[0]["family"],
                                "factor": factor,
                                "forced": view_tasks[0]["engineered_collision_anchor"],
                                "variant": variant,
                                "draw": draw,
                                "anchor_A": anchor[0]["A"],
                                "view_A": edited[0]["A"],
                                "anchor_J": anchor[0]["J"],
                                "view_J": edited[0]["J"],
                                "fragile_response": bool(anchor[0]["A"] and not edited[0]["A"]),
                                "both_answers_correct": bool(anchor[0]["A"] and edited[0]["A"]),
                                "relative_correct": relative,
                                "both_wrong_but_relative_correct": bool(
                                    not anchor[0]["A"] and not edited[0]["A"] and relative == 1
                                ),
                            }
                        )
    for key, summary in summaries.items():
        parent, factor, family, population = key.split("/")
        selected = [
            pair
            for pair in pairs
            if pair["parent"] == parent
            and pair["factor"] == factor
            and (family == "ALL" or pair["family"] == family)
            and (population == "ALL" or pair["forced"] == (population == "FORCED"))
        ]
        known = [pair for pair in selected if pair["relative_correct"] is not None]
        summary["anchor_view_pairs"] = {
            "observed_pair_denominator": len(selected),
            "fragile_response_rate": _mean([pair["fragile_response"] for pair in selected]),
            "both_answers_correct_rate": _mean([pair["both_answers_correct"] for pair in selected]),
            "relative_known_pairs": len(known),
            "relative_correct_count": sum(pair["relative_correct"] for pair in known),
            "relative_correct_known_rate": _mean([pair["relative_correct"] for pair in known]),
            "both_wrong_relative_correct_count": sum(
                pair["both_wrong_but_relative_correct"] for pair in selected
            ),
        }
        matched = [
            item
            for item in signatures
            if item["parent"] == parent
            and item["factor"] == factor
            and (family == "ALL" or item["family"] == family)
            and (population == "ALL" or item["forced"] == (population == "FORCED"))
        ]
        summary["heldout_signature_prediction"] = {}
        for mode in ("absolute", "response_delta"):
            evaluated = [
                item[mode]["heldout_prediction_correct"]
                for item in matched
                if item[mode]["heldout_prediction_correct"] is not None
            ]
            summary["heldout_signature_prediction"][mode] = {
                "four_view_observed_denominator": len(matched),
                "no_compatible_candidate_count": sum(
                    item[mode]["reason"] == "NO_COMPATIBLE_REGISTERED_CANDIDATE" for item in matched
                ),
                "multiple_compatible_count": sum(
                    item[mode]["compatible_count"] > 1 for item in matched
                ),
                "evaluable_predictions": len(evaluated),
                "prediction_correct_count": sum(evaluated),
                "prediction_accuracy_evaluable": _mean(evaluated),
                "unknown_reasons": dict(
                    Counter(
                        item[mode]["reason"] for item in matched if item[mode]["reason"] is not None
                    )
                ),
            }
    factor_pairs = []
    for parent in ("b0", "b1", "b2"):
        for root_id, root in roots.items():
            for variant in range(4):
                left_task, right_task = (root["tasks"][factor][variant] for factor in ("P0", "P1"))
                for draw in range(8):
                    left = lookup.get((parent, left_task["qid"], draw))
                    right = lookup.get((parent, right_task["qid"], draw))
                    if left is None or right is None:
                        continue
                    factor_pairs.append(
                        {
                            "parent": parent,
                            "root_id": root_id,
                            "family": left_task["family"],
                            "forced": left_task["engineered_collision_anchor"],
                            "variant": variant,
                            "draw": draw,
                            "P1_minus_P0_A": right[0]["A"] - left[0]["A"],
                            "P1_minus_P0_J": right[0]["J"] - left[0]["J"],
                            "identical_input_world": left_task["world"] == right_task["world"],
                            "same_exact_answer": None
                            if left[1] is None or right[1] is None
                            else left[1] == right[1],
                        }
                    )
    expected = len(tasks) * 3 * 8
    return {
        "status": "DERIVED_COMPLETE_SUPPLIED_MATRIX"
        if len(records) == expected
        else "DERIVED_PARTIAL_MATRIX",
        "version": VERSION,
        "source_kind": "PROVIDED_EXISTING_RECORDS",
        "model_calls": 0,
        "stage_gate_decision": "NOT_ISSUED",
        "observed_rows": len(records),
        "expected_rows_for_supplied_tasks": expected,
        "missing_rows_for_supplied_tasks": expected - len(records),
        "full_registered_task_manifest": len(tasks) == 384 and len(roots) == 48,
        "summaries": summaries,
        "groups": groups,
        "anchor_view_pairs": pairs,
        "signature_predictions": signatures,
        "factor_pairs": factor_pairs,
        "limits": [
            "Same draw index across independently sampled views is a descriptive pairing, "
            "not a shared hidden trajectory.",
            "Factor-identical anchors with common seeds are correlated "
            "and not new independent samples.",
            "Coverage and credit association is observational; "
            "only registered formal contrasts assess learning value.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        type=Path,
        required=True,
        help="Existing ROOT_CERTIFICATES_PRIVATE.jsonl; only TRAIN is diagnosed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--tasks", type=Path)
    args = parser.parse_args()
    roots = [json.loads(line) for line in args.bank.read_text().splitlines() if line.strip()]
    result = mathematical_diagnostics(
        [root for root in roots if root["tasks"]["P0"][0]["split"] == "TRAIN"]
    )
    args.output.mkdir(parents=True, exist_ok=False)
    details = result.pop("roots")
    (args.output / "MATH_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "MATH_ROOTS.jsonl").write_text("".join(canonical(row) + "\n" for row in details))
    if (args.records is None) != (args.tasks is None):
        raise ValueError("Existing records and tasks must be supplied together")
    if args.records is not None:
        records = [
            json.loads(line) for line in args.records.read_text().splitlines() if line.strip()
        ]
        tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
        result = census_diagnostics(records, tasks)
        for field, filename in (
            ("groups", "CENSUS_GROUPS.jsonl"),
            ("anchor_view_pairs", "CENSUS_PAIRS.jsonl"),
            ("signature_predictions", "CENSUS_SIGNATURES.jsonl"),
            ("factor_pairs", "CENSUS_FACTOR_PAIRS.jsonl"),
        ):
            (args.output / filename).write_text(
                "".join(canonical(row) + "\n" for row in result.pop(field))
            )
        (args.output / "CENSUS_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        canonical(
            {"status": "DESCRIPTIVE_AUDIT_WRITTEN", "output": str(args.output), "model_calls": 0}
        )
    )


if __name__ == "__main__":
    main()
