import pytest

from src.r1_parity import parity_comparison, reference_panel


def thresholds():
    return {
        "parity_alarm_mean_abs_token_logp": 0.02,
        "parity_alarm_p99_abs_token_logp": 0.1,
        "parity_alarm_min_top1_agreement": 0.999,
    }


def test_selected_scores_do_not_invent_top1_evidence():
    result = parity_comparison(([1, 2], [-1.0, -2.0]), (None, [-1.0, -2.0]), thresholds())
    assert result["passed"]
    assert result["top1_agreement"] is None
    assert result["top1_measured"] is False


def test_production_gate_retains_rejected_optimizations_and_rejects_missing_reference():
    from src.r1_parity import production_gate

    checks = {
        k: {"passed": True, "sequences": 120, "top1_measured": True}
        for k in ("reference_repeat", "behavior", "evaluation_likelihood", "training_likelihood")
    }
    result = production_gate(checks, {"token": {"passed": False}})
    assert result["status"] == "PASS"
    assert result["selected_path"] == "uncached_prefix_recompute"
    assert result["rejected_optimization_paths"] == ["token"]
    assert result["adopted_optimization_paths"] == []
    assert production_gate({**checks, "behavior": {"passed": False}}, {})["status"] == "FAIL"
    assert (
        production_gate({k: v for k, v in checks.items() if k != "behavior"}, {})["status"]
        == "FAIL"
    )
    assert (
        production_gate({**checks, "behavior": {"passed": True, "sequences": 1}}, {})["status"]
        == "FAIL"
    )


def test_reference_audit_detects_behavior_drift_and_training_mode_drift(monkeypatch):
    import torch

    from src import r1_parity

    monkeypatch.setattr(r1_parity, "prefix_scores", lambda *args: ([1, 2], [-1.0, -2.0]))

    class Adapter:
        def logprobs(self, prepared, completion, require_grad=False):
            return torch.tensor([-1.0, -2.5 if require_grad else -2.0], requires_grad=require_grad)

    rows = r1_parity.reference_execution_audit(
        Adapter(),
        {},
        {
            "token_ids": [1, 2],
            "behavior_token_logprobs": [-1.0, -2.0],
            "behavior_top1_token_ids": [1, 3],
        },
        thresholds(),
    )
    assert not rows["behavior"]["comparison"]["passed"]
    assert not rows["training_likelihood"]["comparison"]["passed"]
    assert rows["reference_repeat"]["comparison"]["passed"]


def test_reference_panel_hash_selects_two_per_family_chart():
    rows = [
        {
            "split": "calibration",
            "constraint_family": family,
            "chart_type": chart,
            "base_scene_id": f"{family}-{chart}-{i}",
        }
        for family in ("duplicate_encoding", "cross_series", "trend")
        for chart in ("line", "grouped_bar")
        for i in range(3)
    ]
    chosen = reference_panel(rows)
    assert len(chosen) == 12
    assert all(
        sum(r["constraint_family"] == family and r["chart_type"] == chart for r in chosen) == 2
        for family in ("duplicate_encoding", "cross_series", "trend")
        for chart in ("line", "grouped_bar")
    )


def test_parity_reports_p99_and_sequence_ratio():
    config = {
        "parity_alarm_mean_abs_token_logp": 0.02,
        "parity_alarm_p99_abs_token_logp": 0.1,
        "parity_alarm_min_top1_agreement": 0.999,
    }
    result = parity_comparison(
        ([1, 2, 3], [-1.0, -2.0, -3.0]), ([1, 2, 4], [-1.0, -1.9, -3.0]), config
    )
    assert result["sequence_log_ratio_candidate_minus_reference"] == pytest.approx(0.1)
    assert result["top1_agreement"] == pytest.approx(2 / 3)
    assert result["passed"] is False
