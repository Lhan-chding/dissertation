"""Deterministic final-audit fixtures; none are model-generated evidence."""

import copy
import json

import pytest

from src.verified_discovery_transfer.config import TEMPLATE_VERSION
from src.verified_discovery_transfer.evaluate import audit_endpoint
from src.verified_discovery_transfer.public_tasks import verify_raw


def fixture():
    task = {
        "task_id": "fixture-E",
        "base_instance_id": "fixture-E",
        "split": "E_test",
        "family": "cross_series",
        "observed": [99, 20, 30, 40],
        "H_original": [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
        "b_original": [50, 60, 70],
        "legal_domain": [0, 99],
        "operation": "sum4",
        "template_version": TEMPLATE_VERSION,
        "chart_type": "line",
        "interface": "SYMBOLIC_FRESH",
        "image_path": None,
        "image_sha256": None,
    }
    audit = {
        "task_id": task["task_id"],
        "base_instance_id": task["base_instance_id"],
        "split": "E_test",
        "true_world": [10, 20, 30, 40],
        "corrupted_index": 0,
        "DPE1": {"O0": False, "L11": True, "B1": False},
        "truth_all_in_0_49": True,
    }
    return task, audit


def record(task, raw, draw=0, protocol="O0"):
    return {
        "request_id": f"fixture-{protocol}-{draw}",
        "task_id": task["task_id"],
        "base_instance_id": task["base_instance_id"],
        "split": task["split"],
        "parent_id": "S96",
        "model_revision": "CPU_FIXTURE",
        "pipeline_repeat": 0,
        "source_job": "fixture-student",
        "step": 256,
        "protocol_id": protocol,
        "draw_index": draw,
        "raw_completion": raw,
        **verify_raw(task, protocol, raw),
    }


def test_exact_categories_aliases_partial_repair_and_k8():
    task, audit = fixture()
    vectors = [
        [10, 20, 30, 40],
        [10, 20, 30, 40],
        [99, 20, 30, 40],
        [11, 19, 30, 40],
        [10, 21, 30, 40],
        [True, 2, 3, 4],
        [99, 109, 119, -49],
        [11, 21, 31, 39],
    ]
    rows = [record(task, json.dumps(world), i) for i, world in enumerate(vectors)]
    result = audit_endpoint(rows, [task], [audit], pre_zero16_ids={task["task_id"]})
    metric = result["tasks"][0]
    assert [metric["nX"], metric["nS"], metric["nW"], metric["nI"]] == [2, 1, 3, 2]
    assert metric["pX"] == 0.25
    assert metric["K8_contrast"]["state"] == "mixed-X"
    assert metric["K8_contrast"]["has_partial_repair"]
    assert metric["K8_contrast"]["C_orig_varies"]
    assert metric["partial_repair_count"] == 1  # Correct corrupted value + another damage.
    assert metric["all_original_relations_wrong_count"] == 1
    assert metric["PRE_ZERO16"] and metric["DPE1_O0"] is False
    assert metric["F_count"] == 6 and metric["F_mean"] == pytest.approx(3 / 6)
    copy_row = result["records"][2]
    assert copy_row["alias_flags"] == {"truth": False, "copy": True, "V1": True, "V2": True}
    assert copy_row["alias_mask"] == 14
    assert not copy_row["noncopy_nontruth_ptlc_match"]
    assert result["records"][5]["event"] == "I"
    assert result["records"][5]["F"] is None


def test_out_of_domain_diagnostics_and_original_anchors_separate():
    task, audit = fixture()
    # B1 V1 satisfies every relation but is illegal; it remains I with no pX credit.
    raw = "[99,109,119,-49]"
    result = audit_endpoint([record(task, raw, protocol="B1")], [task], [audit])
    scored = result["records"][0]
    assert scored["event"] == "I" and scored["C_orig"] == 0
    assert scored["C_alg_orig"] == scored["C_alg_display"] == 1
    assert scored["match_V1"] and not scored["match_V1_in_domain"]
    assert scored["noncopy_nontruth_ptlc_match"] and not scored["noncopy_nontruth_ptlc_match_legal"]
    assert scored["all_original_relations_wrong_alg"] and not scored["all_original_relations_wrong"]
    assert scored["anchor_all"] and not scored["original_anchor_all"]
    assert scored["original_anchor_coordinates"] == [0, 1, 2]
    assert scored["anchor_coordinates"] == [0]
    assert scored["F"] is None and scored["M_alg"] == 3
    assert result["tasks"][0]["K8_contrast"] is None


def test_fixed_inverse_mapping_and_truth_program_overlap():
    task, audit = fixture()
    result = audit_endpoint([record(task, "[40,30,20,10]", protocol="L11")], [task], [audit])
    row = result["records"][0]
    assert row["canonical_world"] == audit["true_world"]
    assert row["event"] == "X" and row["DPE1_protocol"]
    assert row["alias_flags"]["truth"] and row["alias_flags"]["V1"] and row["alias_flags"]["V2"]
    assert not row["noncopy_nontruth_ptlc_match"]


def test_audit_rejects_training_mixed_endpoints_and_invalid_draws():
    task, audit = fixture()
    row = record(task, "[10,20,30,40]")
    bad_task, bad_audit = copy.deepcopy(task), copy.deepcopy(audit)
    bad_task["split"] = bad_audit["split"] = "T_train"
    with pytest.raises(PermissionError):
        audit_endpoint([row], [bad_task], [bad_audit])
    changed = {**record(task, "[10,20,30,40]", 1), "step": 128}
    with pytest.raises(ValueError, match="pool"):
        audit_endpoint([row, changed], [task], [audit])
    with pytest.raises(ValueError, match="duplicate"):
        audit_endpoint([row, row], [task], [audit])
    with pytest.raises(ValueError, match="contiguous"):
        audit_endpoint([record(task, "[10,20,30,40]", 1)], [task], [audit])
    with pytest.raises(ValueError, match="disagree"):
        audit_endpoint([{**row, "public_verifier_pass": False}], [task], [audit])
    with pytest.raises(ValueError, match="DPE1"):
        audit_endpoint([row], [task], [{**audit, "DPE1": {"O0": True}}])
