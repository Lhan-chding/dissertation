"""Tiny frozen hypothesis contracts: no campaign or model execution."""

import copy
import json
from pathlib import Path

import pytest

from src.modeling_v3.cpu_campaign import _digest, development_designs
from src.modeling_v3.frozen_comparisons import (
    COMMON_TEST_SPEC,
    RQ4_TEMPLATE,
    paired_seed_inference,
    summarize_primary_family,
    validate_primary_comparison_family,
)


def frozen_fixture():
    config = json.loads(Path("configs/modeling_v3/protocol.json").read_text())
    selected = {
        "observation_methods": ["PRESERVE_XI", "PILOT_SHRINK_ZERO_SUM"],
        "selection_rules": ["STRATIFIED_RANDOM", "BLOCK_PIVOT_QR"],
        "models": [
            "ZERO",
            "FULL_RIDGE",
            "FULL_GLS",
            "PCA",
            "RANDOM_Q",
            "RESPONSE_SVD",
            "GROUP_WEIGHTED_RESPONSE",
            "RBF_RIDGE",
            "DIRECT_MEASURE",
            "KNOWN_EVENT_SCORE",
        ],
        "rank_caps": [2, "FULL"],
        "alpha": 1e-5,
        "rho_threshold": 0.05,
        "leverage_threshold": 10,
    }
    designs = development_designs(config, selected=selected)

    def model(method, selector="BLOCK_PIVOT_QR", rank="FULL"):
        d = next(
            d
            for d in designs
            if d["model"] == method
            and d["selector"] == selector
            and d["rank_cap"] == rank
            and d["n"] == 1024
            and d["estimator"] == "PRESERVE_XI"
        )
        return {"kind": "MODEL", "design_id": _digest(d)[:20]}

    specs = [
        {
            **COMMON_TEST_SPEC,
            "id": "RQ1",
            "stage": "CPU",
            "scope": "ALL_CASES",
            "left": {"kind": "DIRECT_MEASURE", "estimator": "PILOT_SHRINK_ZERO_SUM", "n": 1024},
            "right": {"kind": "DIRECT_MEASURE", "estimator": "PRESERVE_XI", "n": 1024},
        },
        {
            **COMMON_TEST_SPEC,
            "id": "RQ2",
            "stage": "CPU",
            "scope": "ALL_CASES",
            "left": model("FULL_RIDGE"),
            "right": model("FULL_RIDGE", "STRATIFIED_RANDOM"),
        },
        {
            **COMMON_TEST_SPEC,
            "id": "RQ3",
            "stage": "CPU",
            "scope": "COMMON_R_LESS_K_ORIGINS",
            "left": model("RESPONSE_SVD", rank=2),
            "right": model("FULL_RIDGE"),
        },
        copy.deepcopy(RQ4_TEMPLATE),
    ]
    selected["primary_comparison_family"] = {
        "schema": "ssvc-v3-primary-comparison-family-1",
        "family_id": "SSVC_V3_RQ1_RQ4",
        "family_size": 4,
        "alpha": 0.05,
        "correction": "HOLM",
        "hypotheses": specs,
    }
    return config, selected


def totals(n=6):
    return [
        {
            "seed": 100 + s,
            "left_loss_sum": 10,
            "right_loss_sum": 2,
            "count": 4,
            "eligible_origin_count": 1,
            "total_origin_count": 2,
            "channel_totals": {
                "group_pX": {"left_loss_sum": 6, "right_loss_sum": 2, "count": 2},
                "group_v": {"left_loss_sum": 4, "right_loss_sum": 0, "count": 2},
            },
        }
        for s in range(n)
    ]


def test_family_validates_real_matrix_and_rq4_immutable_relation():
    config, selected = frozen_fixture()
    family = validate_primary_comparison_family(config, selected)
    assert len(development_designs(config, selected=selected)) == 192
    assert family == selected["primary_comparison_family"]
    changed = copy.deepcopy(selected)
    changed["primary_comparison_family"]["hypotheses"][3]["left"]["estimator"] = (
        "PILOT_SHRINK_ZERO_SUM"
    )
    with pytest.raises(ValueError, match="RQ4"):
        validate_primary_comparison_family(config, changed)
    changed = copy.deepcopy(selected)
    changed["primary_comparison_family"]["hypotheses"][1]["right"] = copy.deepcopy(
        changed["primary_comparison_family"]["hypotheses"][1]["left"]
    )
    with pytest.raises(ValueError, match="RQ2"):
        validate_primary_comparison_family(config, changed)


def test_equal_channel_seed_metric_is_not_count_weighted_pool():
    _, selected = frozen_fixture()
    spec = selected["primary_comparison_family"]["hypotheses"][0]
    data = totals()
    for row in data:
        row["channel_totals"]["group_v"]["count"] = 100
        row["count"] = 102
    result = paired_seed_inference(spec, data, bootstrap_seed=1)
    assert result["status"] == "AVAILABLE"
    assert result["estimate"] == pytest.approx((2 + 0.04) / 2)
    assert result["reps"] == 5000 and result["independent_seeds"] == 6
    assert result["channel_effects"]["group_pX"]["estimate"] == 2
    assert result["paired_seed_totals"] == data
    assert result["null_sign_symmetry_assumed"] is True


def test_no_empty_seed_or_unresolved_reference_can_disappear():
    _, selected = frozen_fixture()
    spec = selected["primary_comparison_family"]["hypotheses"][2]
    data = totals()
    data[0]["status"] = "UNKNOWN_REFERENCE"
    result = paired_seed_inference(spec, data, bootstrap_seed=1)
    assert result["status"] == "UNKNOWN" and result["raw_pvalue"] is None
    assert len(result["paired_seed_totals"]) == 6
    assert paired_seed_inference(spec, totals(5), bootstrap_seed=1)["status"] == "UNKNOWN"


def test_family_holm_waits_for_actual_fourth_test():
    config, selected = frozen_fixture()
    family = validate_primary_comparison_family(config, selected)
    results = [paired_seed_inference(s, totals(), bootstrap_seed=1) for s in family["hypotheses"]]
    pending = summarize_primary_family(family, results[:3])
    assert pending["status"] == "PENDING" and pending["holm_applied"] is False
    complete = summarize_primary_family(family, results)
    assert complete["status"] == "COMPLETE" and complete["holm_applied"] is True
    changed = copy.deepcopy(results)
    changed[0]["spec_hash"] = "tampered"
    with pytest.raises(ValueError, match="spec"):
        summarize_primary_family(family, changed)
