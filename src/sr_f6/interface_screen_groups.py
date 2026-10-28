"""Descriptive finite-library and two-draw group checks for the fixed screen.

The source is the complete registered 576-response screen. This module issues
no gate, estimates no learning effect, and never computes or approximates G8
credit. Its 2x2 products are descriptive pairs, not independent observations.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from srf6_ref.core import FAMILIES, exact, seed_for

from .group_diagnostics import _parsed_answer
from .group_interface_audit import _task_index, candidate_certificates
from .interface_protocol import VERSION as PROMPT_VERSION
from .interface_repair import FACTORS, MASTER_SEED, PARENTS
from .measurement import score_response

VERSION = "srf6-interface-screen-group-descriptive-v1"
PHASE = "INTERFACE_SCREEN_V2"
_PAIR_COUNTS = (
    "both_A_correct",
    "anchor_A_correct_edit_A_zero",
    "anchor_correct_edit_known_wrong",
    "anchor_correct_edit_unknown",
    "relative_known",
    "relative_unknown",
    "relative_correct",
    "both_known_wrong_relative_correct",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def screen_cross_product(anchor_answers, edit_answers, anchor_gold, edit_gold):
    """Keep UNKNOWN in the complete four-pair denominator; no fitted trajectory."""
    _require(
        len(anchor_answers) == len(edit_answers) == 2,
        "Exactly two original draws per question required",
    )
    _require(anchor_gold is not None and edit_gold is not None, "Defined gold answers required")
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
    counts = {key: counts[key] for key in _PAIR_COUNTS}
    return {
        "cross_product_pairs": 4,
        "counts": counts,
        "rates_all_four": {key: value / 4 for key, value in counts.items()},
        "relative_accuracy_known": (
            counts["relative_correct"] / counts["relative_known"]
            if counts["relative_known"]
            else None
        ),
        "independent_sample_count": "NOT_4; product of two empirical two-draw distributions",
    }


def _fixed_tasks(private_tasks):
    if isinstance(private_tasks, dict):
        _require(
            all(key == value.get("qid") for key, value in private_tasks.items()),
            "Private task mapping keys differ from qids",
        )
        private_tasks = list(private_tasks.values())
    _require(isinstance(private_tasks, list), "Private task list or qid mapping required")
    tasks, roots = _task_index(private_tasks)
    expected = {
        f"DEV_CAL_{family}_{index:04d}_{factor}_v{variant}"
        for family in FAMILIES
        for index in (0, 1)
        for factor in FACTORS
        for variant in range(4)
    }
    _require(set(tasks) == expected, "Exactly the fixed 96 screen tasks at indices 0/1 required")
    forced = {}
    for root, factors in roots.items():
        flags = [
            task.get("engineered_collision_anchor") for views in factors.values() for task in views
        ]
        expected_flag = int(root.rsplit("_", 1)[1]) == 0
        _require(
            all(type(flag) is bool and flag == expected_flag for flag in flags),
            "Fixed screen collision strata differ from registered indices",
        )
        forced[root] = expected_flag
    return tasks, roots, forced


def _validated_answers(records, tasks):
    _require(isinstance(records, list) and len(records) == 576, "Complete 576 screen rows required")
    answers, seen, common, policies = {}, set(), {}, {}
    finishes = {}
    for row in records:
        parent, qid, draw = row.get("parent"), row.get("qid"), row.get("draw")
        _require(parent in PARENTS and qid in tasks, "Unregistered screen answer identity")
        _require(type(draw) is int and draw in (0, 1), "Original draw 0/1 required")
        key = parent, qid, draw
        _require(key not in seen, "Duplicate screen answer identity")
        seen.add(key)
        task = tasks[qid]
        _require(
            type(row.get("variant")) is int
            and all(
                row.get(name) == task[name] for name in ("root_id", "family", "factor", "variant")
            )
            and row.get("row_id") == f"{parent}/{qid}/draw{draw}",
            "Screen answer/task lineage differs",
        )
        _require(
            row.get("phase") == PHASE
            and row.get("prompt_version") == PROMPT_VERSION
            and row.get("execution_mode") == "READONLY_INTERFACE_SCREEN"
            and type(row.get("optimizer_updates")) is int
            and row["optimizer_updates"] == 0
            and row.get("generation_status") == "COMPLETE"
            and row.get("technical_validation_errors") == [],
            "Screen execution identity or technical validity differs",
        )
        _require(
            type(row.get("seed")) is int
            and row["seed"]
            == seed_for(
                MASTER_SEED,
                PROMPT_VERSION,
                "screen",
                parent,
                task["root_id"],
                task["variant"],
                draw,
            ),
            "Screen answer seed differs from fixed plan",
        )
        for name in ("registration_sha256", "source_sha256", "data_hash"):
            value = row.get(name)
            _require(
                isinstance(value, str)
                and re.fullmatch(r"[a-f0-9]{64}", value)
                and common.setdefault(name, value) == value,
                "Screen rows mix registration, source, or data identities",
            )
        for name in ("parent_content_hash", "policy_tensor_hash"):
            value = row.get(name)
            _require(
                isinstance(value, str)
                and re.fullmatch(r"[a-f0-9]{64}", value)
                and policies.setdefault((parent, name), value) == value,
                "Screen rows mix parent or policy identities",
            )
        raw, finish = row.get("full_text"), row.get("finish_reason")
        _require(
            isinstance(raw, str) and finish in ("balanced", "eos", "length"),
            "Original screen text and raw termination required",
        )
        measured = score_response(
            raw, task["world"], task["query"], finished=finish == "balanced", finish_reason=finish
        )
        supplied = row.get("wire_score")
        _require(
            isinstance(supplied, dict)
            and all(type(supplied.get(name)) is int for name in ("A", "V", "J", "finished"))
            and all(
                supplied.get(name) is None or type(supplied[name]) is int for name in ("S", "B")
            )
            and supplied == measured,
            "Stored wire labels or completion differ from strict original-text scoring",
        )
        answers[key] = _parsed_answer(raw)
        finishes[key] = measured["finished"]
    _require(
        seen == {(parent, qid, draw) for parent in PARENTS for qid in tasks for draw in (0, 1)},
        "Complete fixed parent/question/draw matrix required",
    )
    return answers, finishes, common


def _tri_counts(values):
    return {
        "true": sum(value is True for value in values),
        "false": sum(value is False for value in values),
        "unknown": sum(value is None for value in values),
        "denominator": len(values),
    }


def _certificate_summary(items):
    rules = [rule for item in items for rule in item["rules"]]
    boundaries = [row for item in items for row in item["C_semantic_boundary"]]
    aliases = [row for row in rules if row["anchor_alias"] is True]
    return {
        "roots": len(items),
        "candidate_denominator": len(rules),
        "candidate_denominator_kind": "ALL_REGISTERED_NOT_ACTIVE_POOL",
        "C_resp_registered": _tri_counts([row["C_resp_registered"] for row in rules]),
        "response_undefined_pairs": sum(row["response_undefined_pairs"] for row in rules),
        "response_pair_denominator": 3 * len(rules),
        "C_abs": _tri_counts([row["C_abs"] for row in rules]),
        "anchor_alias": _tri_counts([row["anchor_alias"] for row in rules]),
        "C_alias_break_among_anchor_aliases": _tri_counts(
            [row["C_alias_break"] for row in aliases]
        ),
        "semantic_boundary_denominator": len(boundaries),
        "semantic_boundaries": {
            key: sum(row[key] for row in boundaries)
            for key in (
                "membership_changed",
                "binding_map_changed",
                "selected_count_changed",
                "mean_denominator_changed",
                "endpoint_query_changed",
            )
        },
        "views_with_operand_sign_change": sum(
            bool(row["operand_sign_changed_common_categories"]) for row in boundaries
        ),
    }


def audit_screen_groups(records, private_tasks):
    """Audit already source-validated screen rows; not a replacement admission check."""
    tasks, roots, forced = _fixed_tasks(private_tasks)
    answers, finishes, source = _validated_answers(records, tasks)
    certificates, pairs = [], []
    for root, factors in roots.items():
        for factor, views in factors.items():
            certificates.append({**candidate_certificates(views), "forced": forced[root]})
            for parent in PARENTS:
                anchor = views[0]
                left = [answers[parent, anchor["qid"], draw] for draw in (0, 1)]
                for edit in views[1:]:
                    right = [answers[parent, edit["qid"], draw] for draw in (0, 1)]
                    finished_pairs = sum(
                        finishes[parent, anchor["qid"], a] and finishes[parent, edit["qid"], b]
                        for a in (0, 1)
                        for b in (0, 1)
                    )
                    pairs.append(
                        {
                            "parent": parent,
                            "factor": factor,
                            "root_id": root,
                            "family": anchor["family"],
                            "forced": forced[root],
                            "variant": edit["variant"],
                            "both_finished_pairs": finished_pairs,
                            "at_least_one_unfinished_pairs": 4 - finished_pairs,
                            **screen_cross_product(
                                left,
                                right,
                                exact(anchor["gold_answer"]),
                                exact(edit["gold_answer"]),
                            ),
                        }
                    )
    certificate_groups, pair_groups = defaultdict(list), defaultdict(list)
    for row in certificates:
        for stratum in ("ALL", row["family"], "FORCED" if row["forced"] else "UNFORCED"):
            certificate_groups[f"{row['factor']}/{stratum}"].append(row)
    for row in pairs:
        for stratum in ("ALL", row["family"], "FORCED" if row["forced"] else "UNFORCED"):
            pair_groups[f"{row['parent']}/{row['factor']}/{stratum}"].append(row)
    summaries = {}
    for key, items in sorted(pair_groups.items()):
        counts = {name: sum(item["counts"][name] for item in items) for name in _PAIR_COUNTS}
        denominator = 4 * len(items)
        summaries[key] = {
            "roots": len({item["root_id"] for item in items}),
            "view_pairs": len(items),
            "cross_product_denominator": denominator,
            "counts": counts,
            "rates_all_pairs": {name: count / denominator for name, count in counts.items()},
            "both_finished_pairs": sum(item["both_finished_pairs"] for item in items),
            "at_least_one_unfinished_pairs": sum(
                item["at_least_one_unfinished_pairs"] for item in items
            ),
        }
    return {
        "version": VERSION,
        "status": "DESCRIPTIVE_COMPLETE_SCREEN",
        "source_identity": source,
        "source_responses": len(records),
        "tasks": len(tasks),
        "roots": len(roots),
        "forced_roots": sum(forced.values()),
        "forced_fraction": sum(forced.values()) / len(roots),
        "root_indices": [0, 1],
        "new_model_calls": 0,
        "optimizer_updates": 0,
        "gate_status": "NOT_SIGNED",
        "training_allowed": False,
        "complete_G8_credit": "NOT_APPLICABLE",
        "effect_estimated": "NOT_RUN",
        "signature_prediction": "NOT_PERFORMED_TWO_DRAW_SCREEN",
        "root_certificates": certificates,
        "candidate_summaries": {
            key: _certificate_summary(items) for key, items in sorted(certificate_groups.items())
        },
        "all_draw_pairs": pairs,
        "pair_summaries": summaries,
        "claim_limit": (
            "Fixed 12-root engineering sample with six forced-collision roots. "
            "Finite-library coverage and observed response fragility are separate facts. "
            "The four cross-products are not four independent observations; paired P0/P1 "
            "inputs and seeds are correlated. A follows the unchanged answer endpoint; "
            "unfinished and UNKNOWN rows remain in all denominators. This report gives "
            "no G8 credit support, hidden-program attribution, causal learning effect, "
            "or additional P1-better-than-P0 gate."
        ),
    }
