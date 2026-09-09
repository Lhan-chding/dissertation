import pytest

from src.r1_parity import parity_comparison, reference_panel


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
