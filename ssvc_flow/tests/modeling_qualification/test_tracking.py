import numpy as np

from src.modeling_qualification.tracking import (
    cost_amortization,
    empirical_envelope,
    fixed_window_prediction,
)


def test_fixed_window_uses_net_sum_and_keeps_intermediate():
    p = np.zeros((2, 3))
    d = np.array([[1.0, 0.0], [-1.0, 0.0]])
    pred, telemetry = fixed_window_prediction(p, d, np.eye(2), np.ones((6, 2)))
    assert pred.shape == (2, 2, 3)
    assert np.allclose(pred[-1], p)
    assert telemetry["path_length"][-1] == 2
    assert telemetry["net_displacement"][-1] == 0


def test_envelope_calibrates_complete_window_maxima_by_seed():
    e = empirical_envelope({1: [0.1, 0.2], 2: [0.3, 0.4]}, 0.9)
    assert e["width"] == 0.4
    assert e["label"] == "EMPIRICAL_CALIBRATED_ENVELOPE"
    assert e["calibration_clusters"] == 2


def test_twenty_four_panels_not_amortized_in_sixteen_steps():
    c = cost_amortization(8, 16, panel_cost=1.0, telemetry_cost=0.0, refresh_cost=1.0)
    assert c["status"] == "NOT_COST_FEASIBLE_AT_TESTED_HORIZON"
    assert c["break_even_horizon"] == 25


def test_step_tolerance_includes_relative_response_failure():
    from src.modeling_qualification.tracking import _failure_series

    p0 = np.tile([0.2, 0.1, 0.3, 0.4], (2, 1))
    truth = (p0 + np.array([0.0001, 0.0, 0.0, -0.0001]))[None]
    rules = {
        "group_delta_pX_abs_error_q95_max": 0.01,
        "group_delta_v_abs_error_q95_max": 0.01,
        "per_prompt_event_mae_max": 0.02,
        "raw_invalid_simplex_fraction_max": 0.05,
        "response_nrmse_max": 0.75,
    }
    assert _failure_series(
        p0[None], np.zeros_like(truth), truth, p0, np.array([0, 1]), rules
    ).tolist() == [True]


def test_projected_rmse_aggregates_squared_errors_before_root():
    from src.modeling_qualification.tracking import _aggregate_extended, _score_extended

    p0 = np.tile([0.2, 0.1, 0.3, 0.4], (2, 1))
    truth = (p0 + np.array([0.0001, 0.0, 0.0, -0.0001]))[None]
    scores = [
        _score_extended(truth, truth - p0, truth, p0, np.array([0, 1])),
        _score_extended(p0[None], np.zeros_like(truth), truth, p0, np.array([0, 1])),
    ]
    assert np.isclose(_aggregate_extended(scores)["projected_delta_rmse"], 0.00005)


def test_envelope_weights_seed_maxima_not_window_count():
    result = empirical_envelope({1: [0.1] * 100, 2: [0.2, 0.9], 3: [0.3]}, 0.5)
    assert result["width"] == 0.3
    assert result["per_seed_complete_window_maximum"] == {"1": 0.1, "2": 0.9, "3": 0.3}


def test_frozen_basis_validation_rejects_nonorthogonal_basis():
    import pytest

    with pytest.raises(ValueError, match="orthonormal"):
        fixed_window_prediction(np.zeros((2, 3)), np.ones((3, 2)), 2 * np.eye(2), np.ones((6, 2)))


def test_tracking_freezes_predictions_and_envelopes_before_locked_labels(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from src.modeling_qualification import tracking
    from src.modeling_qualification.evaluation import file_hash
    from src.modeling_qualification.math_contracts import helmert

    source, model_dir, out = tmp_path / "input", tmp_path / "models", tmp_path / "output"
    source.mkdir()
    model_dir.mkdir()
    config = json.loads(Path("configs/modeling_qualification/protocol.json").read_text())
    config["profiles"]["smoke"]["anchors"] = [0]
    config["measurement"]["samples_per_prompt_grid"] = [64]
    config["measurement"]["noise_replicas"] = 1
    entries = [
        {"id": "cal", "seed": 251, "arm": "X_BASE", "split": "interval_calibration"},
        {"id": "test", "seed": 301, "arm": "X_BASE", "split": "locked_test"},
    ]
    manifest = {
        "profile": "smoke",
        "anchors": [0],
        "trajectories": entries,
        "parameter_count": 2,
        "bank_roles": {"fit": [0], "diagnostic": [1], "evaluation": [2]},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "probe_metadata.json").write_text(json.dumps([{"group": i} for i in range(6)]))
    (source / "summary.json").write_text(
        json.dumps({"exact_panel_calls": 100, "exact_seconds": 0.5, "branch_seconds": 1.0})
    )
    selected = {
        "method": "B5_WORST_GROUP_RANK",
        "rank_cap": 1,
        "alpha": 0.0,
        "qualified": False,
        "status": "SMOKE_ONLY",
    }
    selection = {
        "selected": selected,
        "frozen_grid": {
            "B0_PERSISTENCE|0": {"method": "B0_PERSISTENCE", "rank_cap": 0, "alpha": 0.0},
            "B6_FULL_Q_RIDGE|FULL": {"method": "B6_FULL_Q_RIDGE", "rank_cap": "FULL", "alpha": 0.0},
            "B5_WORST_GROUP_RANK|1": selected,
        },
        "input_manifest_sha256": file_hash(source / "manifest.json"),
    }
    (model_dir / "selection.json").write_text(json.dumps(selection))
    p0 = np.tile([0.2, 0.1, 0.3, 0.4], (6, 1))
    dp = np.tile([0.01, 0.0, 0.0, -0.01], (6, 1))
    updates = np.tile([0.1, 0.0], (16, 1))
    dfit = np.array([[0.1, 0.0], [0.2, 0.0], [0.1, 0.1]])
    fit = {
        "p0": p0,
        "p1": p0 + dfit[:, :1, None] * dp,
        "d": dfit,
        "anchor_measurement_id": "fixed_observed_anchor",
    }
    view = SimpleNamespace(updates=updates, anchor=lambda index, role: fit)
    monkeypatch.setattr(tracking, "_view", lambda *args: view)
    fulltheta = np.vstack((np.zeros(2), np.cumsum(updates, axis=0)))
    truep = p0 + fulltheta[:, :1, None] * dp
    seen = []

    def oracle_loader(root, entry):
        assert (out / "prediction_freeze.json").is_file()
        if entry["split"] == "locked_test":
            assert (out / "envelope_freeze.json").is_file()
            envelope = json.loads((out / "envelope_freeze.json").read_text())
            assert envelope["envelopes"]
        seen.append(entry["split"])
        return SimpleNamespace(p=truep, theta=fulltheta)

    monkeypatch.setattr(tracking, "_oracle", oracle_loader)
    jac = np.column_stack(((dp @ helmert()).reshape(-1), np.zeros(18)))
    monkeypatch.setattr(tracking, "_oracle_j", lambda *args: jac)
    result = tracking.run_track(config, source, model_dir, out)
    assert result["status"] == "COMPLETED"
    assert seen == ["interval_calibration", "locked_test"]
    assert result["empirical_coverage_status"] == "EMPIRICAL_CALIBRATED_ENVELOPE"
    assert result["T2_T3_point_predictions_identical_by_construction"]
    assert {row["H"] for row in result["aggregate_metrics"]} == {1, 2, 4, 8, 16}
    costs = json.loads((out / "cost_audit.json").read_text())
    assert costs["T3_additional_storage_bytes_float64"] == 16


def test_absolute_failure_takes_precedence_over_zero_response_energy():
    from src.modeling_qualification.tracking import _failure_series, step_tolerance_status

    p0 = np.tile([0.2, 0.1, 0.3, 0.4], (2, 1))
    truth = p0[None]
    delta = np.broadcast_to(np.array([0.1, 0.0, 0.0, -0.1]), truth.shape)
    rules = {
        "group_delta_pX_abs_error_q95_max": 0.01,
        "group_delta_v_abs_error_q95_max": 0.01,
        "per_prompt_event_mae_max": 0.02,
        "raw_invalid_simplex_fraction_max": 0.05,
        "response_nrmse_max": 0.75,
    }
    fail = _failure_series(truth + delta, delta, truth, p0, np.array([0, 1]), rules)
    assert step_tolerance_status(fail[0], True) == "EXCEEDS_MODEL_TOLERANCE"
    assert step_tolerance_status(False, True) == "UNKNOWN"
