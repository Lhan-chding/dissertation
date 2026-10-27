import copy
import json

import pytest

from sr_f6.group_diagnostics import (
    census_diagnostics,
    mathematical_diagnostics,
    signature_prediction,
)
from sr_f6.measurement import score_response
from srf6_ref.core import enc, oracle_a
from srf6_ref.design import build_bank, make_root
from srf6_ref.measurement import full_truth_response


@pytest.fixture(scope="module")
def train():
    return build_bank("TRAIN", 48, 2026102714)


def test_math_audit_checks_nonvacuous_gain_and_retains_adverse_roots(train):
    result = mathematical_diagnostics(train)
    overall = result["summaries"]["ALL/ALL"]
    assert result["model_calls"] == 0
    assert overall["roots"] == 288
    assert overall["same_groups"] == 20
    assert overall["same_edit_views"] == 321
    assert overall["P0"]["candidate_coverage_count"] == 1665
    assert overall["P1"]["candidate_coverage_count"] == 1945
    assert overall["P0"]["alias_exposed_count"] == 145
    assert overall["P1"]["alias_exposed_count"] == 188
    assert len(overall["forced_boundary_unexposed_P1_roots"]) == 3
    assert len(overall["P1_alias_count_below_P0_roots"]) == 2
    assert overall["P1_below_unique_incidence_optimum_roots"] == []
    assert overall["gap"]["mean_coverage_all_registered"] > 0.12
    assert overall["gap"]["mean_coverage_unique_incidence"] > 0.14
    assert overall["cost_bin_mismatches"] == 0
    assert len(result["roots"]) == 288


def test_signature_prediction_does_not_fit_on_heldout_view():
    root = make_root("DEV_CAL", "BINDING", 0, 2026102711)
    tasks = root["tasks"]["P1"]
    answers = [oracle_a(task["world"], task["query"])["answer"] for task in tasks]
    before = signature_prediction(tasks, answers)
    after = signature_prediction(tasks, [*answers[:3], answers[3] + 90])
    for mode in ("absolute", "response_delta"):
        assert before[mode]["compatible_rules"] == after[mode]["compatible_rules"]
        assert before[mode]["prediction"] == after[mode]["prediction"]


def test_consistent_wrong_answers_are_not_absolute_program_evidence():
    root = make_root("DEV_CAL", "BINDING", 0, 2026102711)
    tasks = root["tasks"]["P1"]
    answers = [oracle_a(task["world"], task["query"])["answer"] + 90 for task in tasks]
    result = signature_prediction(tasks, answers)
    assert "oracle" not in result["absolute"]["compatible_rules"]
    assert "oracle" in result["response_delta"]["compatible_rules"]
    assert "internal program" in result["scope"]


def _records():
    root = make_root("DEV_CAL", "BINDING", 0, 2026102711)
    tasks = [task for factor in ("P0", "P1") for task in root["tasks"][factor]]
    records = []
    for task in tasks:
        for draw in range(8):
            obj = json.loads(full_truth_response(task["world"], task["query"]))
            if draw == 0:
                obj["answer"] = enc(oracle_a(task["world"], task["query"])["answer"] + 90)
            if draw == 1:
                obj["selected"] = []
            text = json.dumps(obj)
            if draw == 2:
                text = "truncated incomplete JSON"
            labels = score_response(text, task["world"], task["query"], finished=True)
            records.append(
                {
                    **labels,
                    "parent": "b0",
                    "factor": task["factor"],
                    "family": task["family"],
                    "root_id": task["root_id"],
                    "qid": task["qid"],
                    "policy_hash": "a" * 64,
                    "group_id": f"b0/{task['qid']}",
                    "row_id": f"b0/{task['qid']}/{draw}",
                    "draw": draw,
                    "raw_text": text,
                }
            )
    return records, tasks


def test_saved_census_diagnostics_keep_invalid_and_consistently_wrong_denominators():
    records, tasks = _records()
    original = copy.deepcopy(records)
    result = census_diagnostics(records, tasks)
    assert records == original
    assert result["status"] == "DERIVED_PARTIAL_MATRIX"
    assert result["observed_rows"] == 64
    assert result["missing_rows_for_supplied_tasks"] == 128
    assert result["stage_gate_decision"] == "NOT_ISSUED"
    assert result["full_registered_task_manifest"] is False
    assert len(result["groups"]) == 8
    assert len(result["anchor_view_pairs"]) == 48
    assert len(result["factor_pairs"]) == 32
    for factor in ("P0", "P1"):
        summary = result["summaries"][f"b0/{factor}/ALL/ALL"]
        assert summary["observed_rows"] == 32
        assert summary["anchor_view_pairs"]["observed_pair_denominator"] == 24
        assert summary["anchor_view_pairs"]["relative_known_pairs"] == 21
        assert summary["anchor_view_pairs"]["both_wrong_relative_correct_count"] == 3
        assert summary["nonzero_groups"] == 4
        assert (
            summary["heldout_signature_prediction"]["absolute"]["four_view_observed_denominator"]
            == 8
        )
        assert (
            summary["heldout_signature_prediction"]["absolute"]["unknown_reasons"][
                "FIT_ANSWER_UNKNOWN"
            ]
            == 1
        )


def test_partial_credit_group_cannot_appear_as_zero_credit():
    records, tasks = _records()
    result = census_diagnostics(records[1:], tasks)
    partial = [group for group in result["groups"] if group["credit_status"] == "INCOMPLETE_G8"]
    assert len(partial) == 1
    assert partial[0]["nonzero"] is None
    assert result["summaries"]["b0/P0/ALL/ALL"]["correlation_root_count"] == 0


def test_stale_score_and_duplicate_rows_are_rejected():
    records, tasks = _records()
    bad = copy.deepcopy(records)
    bad[0]["A"] = 1 - bad[0]["A"]
    with pytest.raises(ValueError, match="label differs"):
        census_diagnostics(bad, tasks)
    with pytest.raises(ValueError, match="Duplicate"):
        census_diagnostics([*records, records[0]], tasks)
