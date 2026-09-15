"""N3 must fail closed and preserve a nonzero diagnostic on rejection."""

import copy
import json
from pathlib import Path

from src.modeling_contrast.selection import freeze_selection, verify_selection_lock

PROTOCOL = json.loads(
    (Path(__file__).parents[2] / "configs/modeling_contrast/protocol.json").read_text()
)


def candidate(name="a", **overrides):
    return {
        "configuration_id": name,
        "model": "C3_CONTRAST_GLS_FULL",
        "method": "O_CRN",
        "access_regime": "SAMPLE_ONLY",
        "n": 64,
        "actual_rank": 2,
        "nonalias_eval_units": 12,
        "predicted_difference_norm": 0.001,
        "pX_nrmse": 0.5,
        "v_nrmse": 0.6,
        "stable_improvement_C0": True,
        "stable_improvement_C1": True,
        **overrides,
    }


def freeze(rows, **overrides):
    kwargs = {
        "protocol": PROTOCOL,
        "input_hashes": {"raw.npz": "a" * 64},
        "source_hashes": {"evaluation.py": "b" * 64},
        "packet_hashes": {"packets.npz": "c" * 64},
        "resource_forecast": {
            "total_walltime_hours": 1,
            "peak_ram_gib": 2,
            "added_output_gib": 0.5,
            "temporary_gib": 0.2,
        },
        "data_gate": {
            "status": "PASS",
            "legacy_complete": True,
            "fresh_seed_collision": False,
        },
        "cost_frontier": [
            {
                "configuration_id": row["configuration_id"],
                "access_regime": row["access_regime"],
                "score_to_generation_ratio": 0.25,
                "break_even_queries": 20,
                "within_domain": True,
                "includes_fit_score_label": True,
                "comparison_baseline": "D_DIRECT",
            }
            for row in rows
        ],
    }
    kwargs.update(overrides)
    return freeze_selection(rows, **kwargs)


def test_n3_freezes_at_most_two_with_content_hash_and_fixed_fresh_seeds():
    lock = freeze([candidate("a"), candidate("b"), candidate("c")])
    assert lock["allow_fresh_cpu"] is True
    assert len(lock["selected"]) == 2
    assert lock["fresh_locked_test_seeds"] == list(range(601, 611))
    assert verify_selection_lock(lock) is True
    changed = copy.deepcopy(lock)
    changed["selected"][0]["n"] = 999
    assert verify_selection_lock(changed) is False


def test_zero_and_exact_cannot_qualify_and_failure_keeps_nonzero_diagnostic():
    lock = freeze(
        [
            candidate("zero", actual_rank=0, predicted_difference_norm=0),
            candidate("bad", pX_nrmse=1.2),
            candidate("oracle", n="exact", pX_nrmse=0.01, v_nrmse=0.01),
        ]
    )
    assert lock["allow_fresh_cpu"] is False
    assert lock["selected"] == []
    assert lock["best_nonzero_diagnostic"]["configuration_id"] == "bad"
    assert lock["status"] == "NO_ACCEPTABLE_MODEL"


def test_nrmse_equality_at_n3_gate_does_not_pass_strict_threshold():
    assert freeze([candidate(pX_nrmse=0.75)])["allow_fresh_cpu"] is False


def test_unknown_baseline_improvement_is_inconclusive():
    lock = freeze([candidate(stable_improvement_C1=None)])
    assert lock["allow_fresh_cpu"] is False
    assert "INCONCLUSIVE_BASELINE_IMPROVEMENT" in lock["candidate_audit"][0]["reasons"]


def test_resource_and_data_hash_gates_fail_independently():
    lock = freeze([candidate()], resource_forecast={"peak_ram_gib": 9})
    assert lock["status"] == "RESOURCE_REVIEW_REQUIRED"
    assert lock["allow_fresh_cpu"] is False
    lock = freeze([candidate()], input_hashes={"raw.npz": "unverified"})
    assert lock["status"] == "BLOCKED_INPUT_IDENTITY"
    assert lock["allow_fresh_cpu"] is False


def test_same_access_cost_and_full_accounting_required():
    row = candidate()
    wrong_access = [
        {
            "configuration_id": "a",
            "access_regime": "SAMPLE_AND_LOGP",
            "cost_feasible": True,
            "includes_fit_score_label": True,
        }
    ]
    lock = freeze([row], cost_frontier=wrong_access)
    assert lock["status"] == "NO_COST_FEASIBLE_SURROGATE"
    assert lock["best_nonzero_diagnostic"]["actual_rank"] == 2


def test_logp_claim_requires_known_event_baseline_and_reports_conditional_cost():
    row = candidate(method="O_LR_MIX", access_regime="SAMPLE_AND_LOGP")
    lock = freeze([row])
    assert lock["candidate_audit"][0]["actual_cost_saving_claim"] is False
    frontier = [
        {
            "configuration_id": "a",
            "access_regime": "SAMPLE_AND_LOGP",
            "score_to_generation_ratio": 0.05,
            "break_even_queries": 10,
            "within_domain": True,
            "includes_fit_score_label": True,
            "comparison_baseline": "D_KNOWN_EVENT_LOGP",
            "cost_feasible": True,
            "known_event_cost_advantage": True,
        }
    ]
    lock = freeze([row], cost_frontier=frontier)
    assert lock["candidate_audit"][0]["actual_cost_saving_claim"] is True
    assert lock["selected"][0]["favorable_cost_ratios"] == [0.05]


def test_known_event_cost_tie_is_a_crossing_but_not_actual_savings():
    row = candidate(method="O_LR_MIX", access_regime="SAMPLE_AND_LOGP")
    frontier = [
        {
            "configuration_id": "a",
            "access_regime": "SAMPLE_AND_LOGP",
            "score_to_generation_ratio": 0.25,
            "break_even_queries": 4,
            "within_domain": True,
            "includes_fit_score_label": True,
            "comparison_baseline": "D_KNOWN_EVENT_LOGP",
            "cost_feasible": True,
            "calibration_cost": 100,
            "direct_cost_four_queries": 100,
        }
    ]
    lock = freeze([row], cost_frontier=frontier)
    assert lock["candidate_audit"][0]["cost_pass"] is True
    assert lock["candidate_audit"][0]["actual_cost_saving_claim"] is False


def test_missing_legacy_or_seed_collision_blocks_new_cpu():
    lock = freeze([candidate()], data_gate={"status": "PASS", "legacy_complete": False})
    assert lock["allow_fresh_cpu"] is False
    lock = freeze(
        [candidate()],
        data_gate={
            "status": "PASS",
            "legacy_complete": True,
            "fresh_seed_collision": True,
        },
    )
    assert lock["status"] == "STOP_FOR_REVIEW"
