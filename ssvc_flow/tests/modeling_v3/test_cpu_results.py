"""Tiny completed-original fixtures exercise Q3 analysis without a CPU campaign."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v3.cpu_campaign import _digest, _finish, _json, _npz, development_designs
from src.modeling_v3.cpu_results import analyze_calibration, analyze_test
from src.modeling_v3.io import canonical_hash, sha256_file, source_identity


def config_lock():
    config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v3/protocol.json").read_text()
    )
    selected = {
        "observation_methods": ["RAW4"],
        "selection_rules": ["FIRST"],
        "models": ["FULL_RIDGE"],
        "rank_caps": ["FULL"],
        "alpha": 0,
        "rho_threshold": 0.05,
        "leverage_threshold": 10,
        "n_grid": [1024],
    }
    source = source_identity()
    lock = {
        "selected": selected,
        "selection_hash": canonical_hash(selected),
        "config_sha256": canonical_hash(config),
        "source_sha256": source["sha256"],
        "development_evidence": {},
    }
    return config, lock


def response_fixture(root, config, lock, seeds, role, *, unknown=False, errors=None):
    root.mkdir()
    rows = []
    designs = development_designs(config, selected=lock["selected"])
    for seed_index, seed in enumerate(seeds):
        unit = root / "units" / f"seed{seed}_a8_repeat00"
        unit.mkdir(parents=True)
        truth = np.ones((3, 2, 4)) * np.array([1.0, 2, 3, -6])
        _npz(unit / "QUERY_REFERENCE.npz", truth=truth)
        entries = []
        for design in designs:
            design_id = _digest(design)[:20]
            fit = unit / "fits" / design_id
            fit.mkdir(parents=True)
            error = np.full_like(truth, (seed_index + 1) * 1e-6)
            if errors is not None:
                error = np.asarray(errors[seed_index]) + np.zeros_like(truth)
            prediction = truth + error
            if unknown and seed_index == 0:
                prediction[0, 0, 0] = np.nan
            labels = np.array(
                ["HIGH_LEVERAGE", "PREDICTABLE_AT_VALIDATED_TOLERANCE", "IDENTICAL_POLICY"]
            )
            _npz(
                fit / "PREDICTIONS.npz",
                prediction=prediction,
                classifications=labels,
                rho=np.zeros(3),
                leverage=np.array([20.0, 1, 0]),
            )
            row = {
                "origin_id": unit.name,
                "design_id": design_id,
                "design": design,
                "seed": seed,
                "arm": "X_BASE",
                "initialization_seed": 7001,
                "anchor": 8,
                "repeat": 0,
                "k": 2,
                "r": 1,
                "rank_comparison_eligible": True,
                "predictions_relative": str((fit / "PREDICTIONS.npz").relative_to(unit)),
                "prediction_sha256": sha256_file(fit / "PREDICTIONS.npz"),
            }
            _finish(fit, {"design": design}, row)
            entries.append(row)
        for n in {design["n"] for design in designs}:
            direct = unit / f"direct_n{n}"
            direct.mkdir()
            base = np.stack((truth[0], truth[1]), axis=0)[None]
            _npz(direct / "estimates.npz", RAW4=base)
            _finish(direct, {"n": n}, {"n": n})
        body = {"origin": unit.name, "results": entries, "direct_baselines": []}
        _json(unit / "RESULTS.json", body)
        _finish(unit, {"role": role}, body)
        rows.extend(entries)
    summary = {"stage": "Q3", "role": role, "pilot": False, "probe_groups": [0, 1], "results": rows}
    _finish(
        root,
        {
            "config_sha256": canonical_hash(config),
            "designs": designs,
            "source_hashes": source_identity()["files"],
        },
        summary,
    )
    return root


def test_calibration_twenty_seed_max_order_statistic_and_original_hashes(tmp_path):
    config, lock = config_lock()
    root = response_fixture(
        tmp_path / "responses", config, lock, range(21001, 21021), "interval_calibration"
    )
    report = analyze_calibration(config, lock, [root], tmp_path / "analysis", fixture=True)
    assert report["fixture"] is True
    assert report["status"] == "TEST_FIXTURE_NOT_AUTHORIZATION"
    for row in report["designs"].values():
        assert row["order_statistic"] == 20
        assert row["radius"] == pytest.approx(20e-6)
        assert row["tolerance"] == config["statistics"]["old_toy_q95_threshold"]
        assert row["tolerance_applied_to"] == "MAX_ABSOLUTE_FOUR_CHANNEL_ERROR"
    assert report["source_manifest_hashes"][str(root / "COMPLETE.json")] == sha256_file(
        root / "COMPLETE.json"
    )


def test_unknown_calibration_predictions_have_no_finite_radius(tmp_path):
    config, lock = config_lock()
    root = response_fixture(
        tmp_path / "responses",
        config,
        lock,
        range(21001, 21021),
        "interval_calibration",
        unknown=True,
    )
    report = analyze_calibration(config, lock, [root], tmp_path / "analysis", fixture=True)
    assert all(row["radius"] is None for row in report["designs"].values())
    assert all(
        row["status"] == "UNRESOLVED_CALIBRATION_PREDICTION" for row in report["designs"].values()
    )


def test_test_analysis_pools_raw_residuals_and_keeps_acceptance_separate(tmp_path):
    config, lock = config_lock()
    cal = response_fixture(
        tmp_path / "cal", config, lock, range(21001, 21021), "interval_calibration"
    )
    receipt = analyze_calibration(config, lock, [cal], tmp_path / "cal_analysis", fixture=True)
    err0 = np.zeros((3, 2, 4))
    err0[0, 0, 0] = 10
    err1 = np.ones((3, 2, 4))
    test = response_fixture(
        tmp_path / "test", config, lock, [22001, 22002], "locked_test", errors=[err0, err1]
    )
    report = analyze_test(
        config, lock, [test], tmp_path / "test_analysis", calibration_receipt=receipt, fixture=True
    )
    primary = next(
        row
        for row in report["metrics"]
        if row["kind"] == "MODEL" and row["target_index"] == 0 and row["channel"] == "X"
    )
    assert primary["q95_absolute_residual"] == pytest.approx(np.quantile([10, 0, 1, 1], 0.95))
    assert primary["nrmse"] == pytest.approx(np.sqrt(102 / 4))
    assert primary["accepted_cases"] == 0
    assert primary["coverage"] == 0
    assert primary["unknown_cases"] == 4
    assert any(row["kind"] == "KNOWN_EVENT_SCORE" for row in report["metrics"])
    assert report["formal_tests"]["status"] == "NOT_RUN_NO_COMPLETE_FROZEN_FOUR_TEST_SPECIFICATIONS"
    assert (tmp_path / "test_analysis/METRICS.csv").is_file()
    assert all(row["reps"] == 5000 for row in report["paired_comparisons"])
    assert all(row["independent_seeds"] == 2 for row in report["paired_comparisons"])
    assert all(row["empirical_seed_coverage"] == 0 for row in report["interval_coverage"])
    assert all(row["interval_width"] == pytest.approx(40e-6) for row in report["interval_coverage"])
    assert {row["target_index"] for row in report["metrics"]} == {0, 1, 2}
    curve = [
        row
        for row in report["risk_coverage_curves"]
        if row["target_index"] == 0 and row["channel"] == "group_pX"
    ]
    assert next(row for row in curve if row["geometry_threshold_multiplier"] == 1)["coverage"] == 0
    assert next(row for row in curve if row["geometry_threshold_multiplier"] == 2)["coverage"] == 1
    assert all(row["thresholds_chosen_from_test_errors"] is False for row in curve)


def test_missing_matrix_and_tampered_original_rejected(tmp_path):
    config, lock = config_lock()
    root = response_fixture(tmp_path / "responses", config, lock, [21001], "interval_calibration")
    with pytest.raises(ValueError, match=r"twenty|matrix|seed"):
        analyze_calibration(config, lock, [root], tmp_path / "full")
    prediction = next(root.rglob("PREDICTIONS.npz"))
    prediction.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash"):
        analyze_calibration(config, lock, [root], tmp_path / "tampered", fixture=True)


def test_too_few_calibration_seeds_and_fixture_cannot_authorize_test(tmp_path):
    config, lock = config_lock()
    root = response_fixture(
        tmp_path / "responses", config, lock, [21001, 21002], "interval_calibration"
    )
    report = analyze_calibration(config, lock, [root], tmp_path / "analysis", fixture=True)
    assert all(row["radius"] is None for row in report["designs"].values())
    assert all(
        row["status"] == "INSUFFICIENT_CALIBRATION_SEEDS" for row in report["designs"].values()
    )
    with pytest.raises(ValueError, match="fixture"):
        analyze_test(config, lock, [], tmp_path / "test", calibration_receipt=report)


def test_unknown_test_predictions_keep_all_case_denominator_and_no_paired_inference(tmp_path):
    config, lock = config_lock()
    cal = response_fixture(
        tmp_path / "cal", config, lock, range(21001, 21021), "interval_calibration"
    )
    receipt = analyze_calibration(config, lock, [cal], tmp_path / "cal_analysis", fixture=True)
    test = response_fixture(
        tmp_path / "test", config, lock, [22001, 22002], "locked_test", unknown=True
    )
    report = analyze_test(
        config, lock, [test], tmp_path / "analysis", calibration_receipt=receipt, fixture=True
    )
    row = next(
        row
        for row in report["metrics"]
        if row["kind"] == "MODEL" and row["target_index"] == 0 and row["channel"] == "X"
    )
    assert row["all_cases"] == 4
    assert row["finite_cases"] == 2
    assert row["unknown_cases"] == 4
    assert row["all_case_mse"] is None and row["nrmse"] is None
    pairs = [row for row in report["paired_comparisons"] if row["target_index"] == 0]
    assert all(row["status"] == "UNKNOWN_INCOMPLETE_PAIRED_PREDICTIONS" for row in pairs)
