"""Synthetic CPU fixtures only; no actual screen or scientific PASS is produced."""

import copy
import json
from fractions import Fraction

import pytest

from sr_f6 import interface_v3_groups as audit
from sr_f6.interface_v3_plan import FACTORS, MASTER_SEED, PARENTS
from sr_f6.interface_v3_protocol import VERSION
from sr_f6.measurement import score_response
from srf6_ref.core import enc, exact, seed_for
from srf6_ref.design import build_bank
from srf6_ref.measurement import full_truth_response


@pytest.fixture(scope="module")
def synthetic_screen():
    tasks = [
        task
        for root in build_bank("DEV_CAL", 2, 2026102711)
        for factor in FACTORS
        for task in root["tasks"][factor]
    ]
    records = []
    for parent in PARENTS:
        for task in tasks:
            for draw in (0, 1):
                obj = json.loads(full_truth_response(task["world"], task["query"]))
                finish = "balanced"
                if draw == 1 and task["variant"] in (0, 2):
                    obj["answer"] = enc(exact(obj["answer"]) + 1)
                raw = json.dumps(obj)
                if draw == 1 and task["variant"] == 1:
                    raw, finish = "{truncated", "length"
                elif draw == 1 and task["variant"] == 3:
                    finish = "eos"
                records.append(
                    {
                        **{
                            key: task[key]
                            for key in ("qid", "root_id", "family", "factor", "variant")
                        },
                        "parent": parent,
                        "draw": draw,
                        "row_id": f"{parent}/{task['qid']}/draw{draw}",
                        "phase": "INTERFACE_SCREEN_V3",
                        "screen_stage": 1,
                        "prompt_version": VERSION,
                        "execution_mode": "READONLY_INTERFACE_SCREEN",
                        "optimizer_updates": 0,
                        "generation_status": "COMPLETE",
                        "technical_validation_errors": [],
                        "seed": seed_for(
                            MASTER_SEED,
                            VERSION,
                            "screen",
                            1,
                            parent,
                            task["root_id"],
                            task["variant"],
                            draw,
                        ),
                        "registration_sha256": "a" * 64,
                        "source_sha256": "b" * 64,
                        "data_hash": "c" * 64,
                        "parent_content_hash": "d" * 64,
                        "policy_tensor_hash": str(PARENTS.index(parent)) * 64,
                        "full_text": raw,
                        "finish_reason": finish,
                        "wire_score": score_response(
                            raw,
                            task["world"],
                            task["query"],
                            finished=finish == "balanced",
                            finish_reason=finish,
                        ),
                    }
                )
    return records, tasks


def test_four_product_pairs_retain_unknown_and_exact_differences():
    result = audit.screen_cross_product(
        [Fraction(1, 3), Fraction(4, 3)],
        [Fraction(2, 3), None],
        Fraction(1, 3),
        Fraction(2, 3),
    )
    assert result["cross_product_pairs"] == 4
    assert result["counts"] == {
        "both_A_correct": 1,
        "anchor_A_correct_edit_A_zero": 1,
        "anchor_correct_edit_known_wrong": 0,
        "anchor_correct_edit_unknown": 1,
        "relative_known": 2,
        "relative_unknown": 2,
        "relative_correct": 1,
        "both_known_wrong_relative_correct": 0,
    }
    assert result["relative_accuracy_known"] == 0.5
    assert "NOT_4" in result["independent_sample_count"]
    assert audit.screen_cross_product([None, None], [1, 2], 3, 4)["relative_accuracy_known"] is None
    assert result == audit.screen_cross_product(
        [Fraction(4, 3), Fraction(1, 3)],
        [None, Fraction(2, 3)],
        Fraction(1, 3),
        Fraction(2, 3),
    )


def test_wrong_pairs_can_have_correct_response_delta_without_correct_answers():
    result = audit.screen_cross_product([11, 11], [13, 13], 10, 12)
    assert result["counts"]["both_known_wrong_relative_correct"] == 4
    assert result["counts"]["relative_correct"] == 4
    assert result["counts"]["both_A_correct"] == 0
    with pytest.raises(ValueError, match="Exactly two"):
        audit.screen_cross_product([11] * 8, [13] * 8, 10, 12)


def test_complete_report_preserves_denominators_strata_and_readonly_scope(synthetic_screen):
    records, tasks = synthetic_screen
    before = copy.deepcopy(synthetic_screen)
    result = audit.audit_screen_groups(records, {task["qid"]: task for task in tasks})
    assert synthetic_screen == before
    assert (
        result["source_responses"],
        result["tasks"],
        result["roots"],
        result["forced_roots"],
    ) == (
        576,
        96,
        12,
        6,
    )
    assert len(result["root_certificates"]) == 24
    assert len(result["all_draw_pairs"]) == 216
    assert result["forced_fraction"] == 0.5
    assert result["gate_status"] == "NOT_SIGNED"
    assert not result["training_allowed"]
    assert result["effect_estimated"] == "NOT_RUN"
    assert result["complete_G8_credit"] == "NOT_APPLICABLE"
    assert result["new_model_calls"] == result["optimizer_updates"] == 0
    assert "NOT_PERFORMED" in result["signature_prediction"]
    for parent in PARENTS:
        for factor in FACTORS:
            cell = result["pair_summaries"][f"{parent}/{factor}/ALL"]
            assert cell["roots"] == 12
            assert cell["view_pairs"] == 36
            assert cell["cross_product_denominator"] == 144
            assert cell["counts"]["relative_unknown"] == 24
            assert cell["counts"]["both_A_correct"] == 48
            assert cell["counts"]["both_known_wrong_relative_correct"] == 12
            assert cell["at_least_one_unfinished_pairs"] == 48
            assert result["pair_summaries"][f"{parent}/{factor}/FORCED"]["roots"] == 6
            assert result["pair_summaries"][f"{parent}/{factor}/BINDING"]["roots"] == 2
    for factor in FACTORS:
        math = result["candidate_summaries"][f"{factor}/ALL"]
        assert math["roots"] == 12
        assert math["candidate_denominator_kind"] == "ALL_REGISTERED_NOT_ACTIVE_POOL"
        assert math["semantic_boundary_denominator"] == 36
        assert math["semantic_boundaries"]["endpoint_query_changed"] == 0
        for name in ("C_resp_registered", "C_abs", "anchor_alias"):
            value = math[name]
            assert value["true"] + value["false"] + value["unknown"] == value["denominator"]
            assert value["denominator"] == math["candidate_denominator"]
        assert (
            math["C_alias_break_among_anchor_aliases"]["denominator"]
            == math["anchor_alias"]["true"]
        )


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing", "authorized stage"),
        ("duplicate", "Duplicate"),
        ("draw", "draw 0/1"),
        ("lineage", "lineage"),
        ("seed", "seed differs"),
        ("provenance", "mix registration"),
        ("policy", "mix parent"),
        ("phase", "execution identity"),
        ("technical", "technical validity"),
        ("score", "wire labels"),
        ("bool", "wire labels"),
        ("finish", "completion differ"),
    ],
)
def test_corrupt_or_partial_screen_is_not_a_descriptive_complete_result(
    synthetic_screen, mutation, message
):
    records, tasks = copy.deepcopy(synthetic_screen)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records[-1] = copy.deepcopy(records[0])
    elif mutation == "draw":
        records[0]["draw"] = True
    elif mutation == "lineage":
        records[0]["factor"] = "P1"
    elif mutation == "seed":
        records[0]["seed"] += 1
    elif mutation == "provenance":
        records[-1]["source_sha256"] = "d" * 64
    elif mutation == "policy":
        records[1]["policy_tensor_hash"] = "f" * 64
    elif mutation == "phase":
        records[0]["phase"] = "CENSUS_S0"
    elif mutation == "technical":
        records[0]["technical_validation_errors"] = ["unresolved"]
    elif mutation == "score":
        records[0]["wire_score"]["A"] = 0
    elif mutation == "bool":
        records[0]["wire_score"]["V"] = True
    elif mutation == "finish":
        records[0]["finish_reason"] = "eos"
    with pytest.raises(ValueError, match=message):
        audit.audit_screen_groups(records, tasks)


@pytest.mark.parametrize("mutation", ["split", "truth", "forced", "missing", "index"])
def test_only_original_fixed_screen_task_geometry_is_accepted(synthetic_screen, mutation):
    records, tasks = copy.deepcopy(synthetic_screen)
    if mutation == "split":
        tasks[0]["split"] = "TEST_COMPOSITION"
    elif mutation == "truth":
        tasks[0]["gold_answer"] = "999999"
    elif mutation == "forced":
        tasks[0]["engineered_collision_anchor"] = False
    elif mutation == "missing":
        tasks.pop()
    else:
        for task in tasks:
            if task["root_id"].endswith("_0001"):
                task["root_id"] = task["root_id"].replace("_0001", "_0002")
                task["qid"] = task["qid"].replace("_0001_", "_0002_")
    with pytest.raises(ValueError):
        audit.audit_screen_groups(records, tasks)


@pytest.fixture(scope="module", params=[("b1",), ("b0", "b2")], ids=["one_parent", "two_parents"])
def synthetic_stage2(request):
    parents = request.param
    tasks = [
        task
        for root in build_bank("DEV_CAL", 4, 2026102711)
        if int(root["root_id"].rsplit("_", 1)[1]) in (2, 3)
        for factor in FACTORS
        for task in root["tasks"][factor]
    ]
    by_parent = {}
    for parent in PARENTS:
        records = []
        for task in tasks:
            raw = full_truth_response(task["world"], task["query"])
            for draw in (0, 1):
                records.append(
                    {
                        **{
                            key: task[key]
                            for key in ("qid", "root_id", "family", "factor", "variant")
                        },
                        "parent": parent,
                        "draw": draw,
                        "row_id": f"{parent}/{task['qid']}/draw{draw}",
                        "phase": "INTERFACE_SCREEN_V3",
                        "screen_stage": 2,
                        "prompt_version": VERSION,
                        "execution_mode": "READONLY_INTERFACE_SCREEN",
                        "optimizer_updates": 0,
                        "generation_status": "COMPLETE",
                        "technical_validation_errors": [],
                        "seed": seed_for(
                            MASTER_SEED,
                            VERSION,
                            "screen",
                            2,
                            parent,
                            task["root_id"],
                            task["variant"],
                            draw,
                        ),
                        "registration_sha256": "a" * 64,
                        "source_sha256": "b" * 64,
                        "data_hash": "c" * 64,
                        "parent_content_hash": "d" * 64,
                        "policy_tensor_hash": str(PARENTS.index(parent)) * 64,
                        "full_text": raw,
                        "finish_reason": "balanced",
                        "wire_score": score_response(
                            raw,
                            task["world"],
                            task["query"],
                            finished=True,
                            finish_reason="balanced",
                        ),
                    }
                )
        by_parent[parent] = records
    records = [row for parent in parents for row in by_parent[parent]]
    excluded = next(parent for parent in PARENTS if parent not in parents)
    return records, tasks, parents, by_parent[excluded]


def test_stage2_real_root_geometry_and_exact_authorized_parent_subset(synthetic_stage2):
    records, tasks, parents, _ = synthetic_stage2
    before = copy.deepcopy(synthetic_stage2)
    result = audit.audit_screen_groups(records, tasks, stage=2, parents=parents)
    assert synthetic_stage2 == before
    assert result["screen_stage"] == 2 and result["parents"] == list(parents)
    assert result["root_indices"] == [2, 3]
    assert (result["tasks"], result["roots"], result["forced_roots"]) == (96, 12, 6)
    assert result["source_responses"] == 192 * len(parents)
    assert len(result["root_certificates"]) == 24
    assert len(result["all_draw_pairs"]) == 72 * len(parents)
    assert {key.split("/")[0] for key in result["pair_summaries"]} == set(parents)
    assert all(
        row["forced"] == row["root_id"].endswith("_0003") for row in result["root_certificates"]
    )
    for parent in parents:
        for factor in FACTORS:
            cell = result["pair_summaries"][f"{parent}/{factor}/ALL"]
            assert cell["view_pairs"] == 36 and cell["cross_product_denominator"] == 144
            assert cell["counts"]["both_A_correct"] == 144
            assert cell["counts"]["relative_unknown"] == 0
            assert result["pair_summaries"][f"{parent}/{factor}/FORCED"]["roots"] == 6
            assert result["pair_summaries"][f"{parent}/{factor}/UNFORCED"]["roots"] == 6
    assert result["new_model_calls"] == result["optimizer_updates"] == 0
    assert result["complete_G8_credit"] == "NOT_APPLICABLE"
    assert result["effect_estimated"] == "NOT_RUN" and not result["training_allowed"]


@pytest.mark.parametrize("mutation", ["stage_argument", "row_stage", "stage1_seed"])
def test_stage2_rejects_wrong_stage_and_stage1_seed_domain(synthetic_stage2, mutation):
    records, tasks, parents, _ = copy.deepcopy(synthetic_stage2)
    stage = 2
    if mutation == "stage_argument":
        stage = 1
    elif mutation == "row_stage":
        records[0]["screen_stage"] = 1
    else:
        row = records[0]
        row["seed"] = seed_for(
            MASTER_SEED,
            VERSION,
            "screen",
            1,
            row["parent"],
            row["root_id"],
            row["variant"],
            row["draw"],
        )
    with pytest.raises(ValueError):
        audit.audit_screen_groups(records, tasks, stage=stage, parents=parents)


def test_stage2_rejects_extra_parent_rows_and_unfulfilled_parent_claim(synthetic_stage2):
    records, tasks, parents, extra = synthetic_stage2
    with pytest.raises(ValueError, match="Complete authorized stage rows"):
        audit.audit_screen_groups(records + extra, tasks, stage=2, parents=parents)
    with pytest.raises(ValueError, match="Complete authorized stage rows"):
        audit.audit_screen_groups(records, tasks, stage=2, parents=PARENTS)
    altered = copy.deepcopy(records)
    altered[0]["parent"] = extra[0]["parent"]
    with pytest.raises(ValueError, match="Unregistered screen answer identity"):
        audit.audit_screen_groups(altered, tasks, stage=2, parents=parents)
