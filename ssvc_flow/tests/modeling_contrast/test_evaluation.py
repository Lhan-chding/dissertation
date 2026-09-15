"""Regression contracts for contrast scoring and seed-level uncertainty."""

import numpy as np
import pytest

from src.modeling_contrast.evaluation import (
    calibration_resolution,
    classify_interval,
    contrast_metrics,
    direction_grid,
    paired_seed_bootstrap,
    resolution_grid,
)

GROUPS = np.repeat(np.arange(6), 12)


def events(values):
    x = np.broadcast_to(np.asarray(values, dtype=float)[:, None], (len(values), 72))
    return np.stack((x, np.zeros_like(x), np.zeros_like(x), -x), axis=-1)


def test_pooled_nrmse_keeps_original_tiny_energy_and_separate_targets():
    truth = events([1e-100, 2e-100])
    pred = truth.copy()
    pred[..., 0] *= 0.5
    pred[..., 3] *= 2
    score = contrast_metrics(pred, truth, GROUPS, active=[True, False])
    assert score["all"]["pX"]["pooled_nrmse"] == pytest.approx(0.5)
    assert score["all"]["v"]["pooled_nrmse"] == pytest.approx(1.0)
    assert score["active_only"]["case_count"] == 1
    assert score["all"]["case_count"] == 2


def test_zero_truth_reports_no_signal_even_for_false_nonzero_prediction():
    score = contrast_metrics(events([1]), events([0]), GROUPS)
    assert score["all"]["pX"]["status"] == "NO_SIGNAL"
    assert score["all"]["pX"]["pooled_nrmse"] is None
    assert score["all"]["pX"]["mae"] == 1
    assert score["active_only"]["status"] == "ACTIVE_MASK_NOT_SUPPLIED"


def test_groups_average_prompts_before_error_and_pooled_is_not_mean_nrmse():
    truth = events([1, 10])
    pred = events([0, 10])
    pred[:, :12, 0] = truth[:, :12, 0]
    score = contrast_metrics(pred, truth, GROUPS)["all"]["pX"]
    assert score["pooled_nrmse"] == pytest.approx(np.sqrt(5 / (6 * 101)))
    assert score["worst_group"]["group"] == 1
    assert len(score["by_group"]) == 6


def test_direction_replicas_do_not_manufacture_seed_or_unit_support():
    truth = events([0.001] * 80)
    result = direction_grid(
        truth,
        truth,
        GROUPS,
        training_seeds=[601] * 80,
        anchor_bank_units=[(8, i % 4) for i in range(80)],
    )
    primary = next(row for row in result["pX"] if row["delta"] == 0.00025)
    assert primary["effective_anchor_bank_units"] == 4
    assert primary["training_seed_count"] == 1
    assert primary["status"] == "INSUFFICIENT_EFFECTIVE_CASES"
    assert primary["accuracy"] is None


def test_direction_requires_both_support_thresholds():
    truth = events([0.001] * 20)
    seeds = np.repeat(np.arange(601, 606), 4)
    result = direction_grid(truth, truth, GROUPS, seeds, [(8, i % 4) for i in range(20)])
    assert all(row["accuracy"] == 1 for row in result["pX"])
    assert len(result["v"]) == 4


def test_seed_bootstrap_pairs_whole_clusters_and_is_reproducible():
    seeds = [1, 1, 1, 2, 2, 2]
    first = paired_seed_bootstrap([0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7], seeds)
    second = paired_seed_bootstrap([0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7], seeds)
    assert first == second
    assert first["bootstrap_replicates"] == 5000
    assert first["cluster_count"] == 2
    assert first["paired_mean"] == first["ci025"] == first["ci975"] == -2
    assert len(first["per_seed"]) == 2


def test_nrmse_bootstrap_uses_energy_sufficient_statistics():
    score = paired_seed_bootstrap([[1, 4], [4, 9]], [[4, 4], [9, 9]], [1, 2], statistic="nrmse")
    assert score["paired_mean"] == pytest.approx(np.sqrt(5 / 13) - 1)
    assert score["statistic"] == "nrmse"


@pytest.mark.parametrize(
    "lower,upper,result",
    [
        (0.0003, 0.0004, "MEANINGFUL_POSITIVE"),
        (-0.0004, -0.0003, "MEANINGFUL_NEGATIVE"),
        (-0.0001, 0.0001, "WITHIN_TOLERANCE"),
        (-0.001, 0.001, "UNKNOWN"),
        (None, None, "UNKNOWN"),
    ],
)
def test_interval_resolution_requires_an_interval(lower, upper, result):
    assert classify_interval(lower, upper) == result


def test_six_seed_calibration_cannot_claim_distribution_free_95_percent():
    result = calibration_resolution(6)
    assert result["status"] == "INSUFFICIENT_CALIBRATION_FOR_DISTRIBUTION_FREE_95"
    assert result["distribution_free_95_claim"] is False
    assert result["maximum_supported_nominal_coverage"] == pytest.approx(6 / 7)


def test_invalid_nonfinite_predictions_are_rejected():
    pred = events([1])
    pred[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        contrast_metrics(pred, events([1]), GROUPS)


def test_good_prediction_does_not_make_an_unknown_interval_resolved():
    panel = contrast_metrics(events([0.001]), events([0.001]), GROUPS)["all"]
    rows = resolution_grid(panel, intervals={"pX": [-0.01, 0.01]})
    assert len(rows) == 8
    assert all(row["prediction_status"] == "PREDICTION_PASS" for row in rows)
    assert all(row["interval_resolution_status"] == "RESOLUTION_FAIL" for row in rows)
    assert all(row["safety_certified"] is False for row in rows)
