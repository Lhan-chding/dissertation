import copy
import json
from fractions import Fraction

import pytest

from sr_f6 import group_interface_audit as audit
from sr_f6.group_diagnostics import signature_prediction
from srf6_ref.core import enc, oracle_a
from srf6_ref.design import make_root


@pytest.fixture(scope="module")
def fixture():
    root = make_root("DEV_CAL", "BINDING", 0, 2026102711)
    tasks = [t for factor in audit.FACTORS for t in root["tasks"][factor]]
    records = []
    for parent in audit.PARENTS:
        for task in tasks:
            answer = oracle_a(task["world"], task["query"])["answer"]
            for draw in range(8):
                raw = '{"answer":' + json.dumps(enc(answer + (1 if draw == 1 else 0))) + "}"
                if draw == 0:
                    raw = "{truncated"
                records.append(
                    {
                        "parent": parent,
                        "qid": task["qid"],
                        "root_id": task["root_id"],
                        "family": task["family"],
                        "factor": task["factor"],
                        "draw": draw,
                        "raw_text": raw,
                        "A": int(draw not in (0, 1)),
                    }
                )
    signatures = []
    for parent in audit.PARENTS:
        for factor in audit.FACTORS:
            views = root["tasks"][factor]
            for draw in range(8):
                answers = [
                    None
                    if draw == 0
                    else oracle_a(t["world"], t["query"])["answer"] + (1 if draw == 1 else 0)
                    for t in views
                ]
                signatures.append(
                    {
                        "parent": parent,
                        "factor": factor,
                        "root_id": root["root_id"],
                        "draw": draw,
                        **signature_prediction(views, answers),
                    }
                )
    return records, tasks, signatures


def test_full_cross_product_is_permutation_invariant_and_keeps_unknown():
    left = [None, 10, 10, 10, 11, 11, 11, 11]
    right = [None, 12, 12, 13, 13, 13, 13, 13]
    result = audit.cross_product_pair(left, right, 10, 12)
    assert result == audit.cross_product_pair(left[::-1], right[3:] + right[:3], 10, 12)
    assert result["counts"]["anchor_A_correct_edit_A_zero"] == 18
    assert result["counts"]["anchor_correct_edit_known_wrong"] == 15
    assert result["counts"]["anchor_correct_edit_unknown"] == 3
    assert result["counts"]["relative_known"] == 49
    assert result["counts"]["relative_unknown"] == 15
    assert result["counts"]["both_known_wrong_relative_correct"] == 20
    assert result["cross_product_pairs"] == 64
    assert "NOT_64" in result["independent_sample_count"]


def test_short_group_rejected():
    with pytest.raises(ValueError, match="Complete G8"):
        audit.cross_product_pair([1], [2] * 8, 1, 2)


def test_wrong_error_magnitude_changes_are_not_alias_breaks(monkeypatch, fixture):
    _, tasks, _ = fixture
    views = [t for t in tasks if t["factor"] == "P0"]
    worlds = [t["world"] for t in views]
    monkeypatch.setattr(audit, "wrong_rules", lambda q: ["shift", "vary", "alias", "undefined"])

    def wrong(world, query, rule):
        index = worlds.index(world)
        gold = oracle_a(world, query)["answer"]
        if rule == "undefined":
            return None
        return gold + {"shift": 7, "vary": index + 1, "alias": index}[rule]

    monkeypatch.setattr(audit, "wrong_value", wrong)
    result = audit.candidate_certificates(views)
    rows = {r["rule"]: r for r in result["rules"]}
    assert rows["shift"]["C_abs"] is True
    assert rows["shift"]["C_resp_registered"] is False
    assert rows["shift"]["all_four_constant_nonzero_offset"] is True
    assert rows["vary"]["C_resp_registered"] is True
    assert rows["vary"]["C_alias_break"] is False
    assert rows["alias"]["C_alias_break"] is True
    assert rows["undefined"]["C_abs"] is None
    assert rows["undefined"]["C_alias_break"] is None
    assert rows["undefined"]["response_undefined_pairs"] == 3


def test_all_supplied_rows_retained_and_inputs_immutable(fixture):
    records, tasks, signatures = copy.deepcopy(fixture)
    before = copy.deepcopy((records, tasks, signatures))
    result = audit.audit_groups(records, tasks, signatures=signatures)
    assert (records, tasks, signatures) == before
    assert result["source_rows"] == 192
    assert result["status"] == "DERIVED_PARTIAL_TASK_MATRIX"
    assert len(result["all_draw_pairs"]) == 18
    assert result["gate_status"] == "NOT_SIGNED"
    assert result["training_authorized"] is False
    summary = result["pair_summaries"]["b0/P0/ALL"]
    assert summary["cross_product_denominator"] == 192
    assert summary["counts"]["relative_unknown"] == 45
    assert result["matched_oracle_control"]["P0/absolute"]["all_quartets"] == 24
    assert result["matched_oracle_control"]["P0/absolute"]["evaluable"] == 18
    assert result["matched_oracle_control"]["P0/absolute"]["net_correct_gain"] == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "complete G8"),
        ("duplicate", "Duplicate answer"),
        ("truth", "Stored A"),
        ("identity", "identity differs"),
    ],
)
def test_corrupt_answer_evidence_rejected(fixture, mutation, match):
    records, tasks, _ = copy.deepcopy(fixture)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "truth":
        records[0]["A"] = 1
    else:
        records[0]["family"] = "TIES"
    with pytest.raises(ValueError, match=match):
        audit.audit_groups(records, tasks)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("heldout", "DEV_CAL"),
        ("gold", "truth differs"),
        ("duplicate", "Duplicate task"),
        ("missing", "Four views"),
    ],
)
def test_task_boundary_and_world_truth_rejected(fixture, mutation, match):
    records, tasks, _ = copy.deepcopy(fixture)
    if mutation == "heldout":
        tasks[0]["split"] = "TEST_COMPOSITION"
    elif mutation == "gold":
        tasks[0]["gold_answer"] = "999"
    elif mutation == "duplicate":
        tasks.append(copy.deepcopy(tasks[0]))
    else:
        tasks.pop()
    with pytest.raises(ValueError, match=match):
        audit.audit_groups(records, tasks)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("observed", "observations differ"),
        ("correctness", "Stale signature"),
        ("fit", "Fit leakage"),
        ("missing", "signature matrix"),
    ],
)
def test_saved_signature_binding_and_same_subset(fixture, mutation, match):
    records, tasks, signatures = copy.deepcopy(fixture)
    if mutation == "observed":
        signatures[2]["observed_answers"][3] = "999"
    elif mutation == "correctness":
        signatures[2]["absolute"]["heldout_prediction_correct"] = 0
    elif mutation == "fit":
        signatures[2]["fit_variants"] = [0, 1, 3]
    else:
        signatures.pop()
    with pytest.raises(ValueError, match=match):
        audit.audit_groups(records, tasks, signatures=signatures)


def test_semantic_boundaries_are_separate_not_weighted_score(fixture):
    _, tasks, _ = fixture
    result = audit.candidate_certificates([t for t in tasks if t["factor"] == "P1"])
    assert "combined_score" not in result
    assert result["candidate_denominator_kind"] == "ALL_REGISTERED_NOT_ACTIVE_POOL"
    assert result["original_active_C_resp_certificate"] == "UNCHANGED_NOT_REPLACED"
    for item in result["C_semantic_boundary"]:
        assert type(item["membership_changed"]) is bool
        assert type(item["binding_map_changed"]) is bool
        assert item["mean_denominator_changed"] is False
        assert item["endpoint_query_changed"] is False


def test_exact_fraction_delta_does_not_round():
    result = audit.cross_product_pair(
        [Fraction(1, 3)] * 8, [Fraction(2, 3)] * 8, Fraction(1, 3), Fraction(2, 3)
    )
    assert result["counts"]["relative_correct"] == result["counts"]["both_A_correct"] == 64


def test_unordered_candidate_views_rejected(fixture):
    _, tasks, _ = fixture
    views = [t for t in tasks if t["factor"] == "P0"]
    with pytest.raises(ValueError, match="ordered views"):
        audit.candidate_certificates(views[::-1])
