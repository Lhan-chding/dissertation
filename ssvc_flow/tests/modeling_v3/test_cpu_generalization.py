"""Small immutable originals; no trajectories are trained by these tests."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v3.cpu_campaign import _digest, _finish, _json, _npz, development_designs
from src.modeling_v3.cpu_results import analyze_calibration
from src.modeling_v3.io import canonical_hash, sha256_file, source_identity, verify_manifest


def setup_config():
    config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v3/protocol.json").read_text()
    )
    config["observation"]["total_draws_grid"] = [64]
    config["cpu"]["anchors"] = [8]
    config["cpu"]["repeat_measurements_test"] = 2
    config["cpu"]["orthogonal_generalization"]["train_rng_seeds"] = [23001, 23002]
    selected = {
        "observation_methods": ["RAW4"],
        "selection_rules": ["FIRST"],
        "models": ["FULL_RIDGE", "ZERO"],
        "rank_caps": ["FULL"],
        "alpha": 0,
        "rho_threshold": 0.05,
        "leverage_threshold": 10,
    }
    lock = {
        "selected": selected,
        "selection_hash": canonical_hash(selected),
        "config_sha256": canonical_hash(config),
        "source_sha256": source_identity()["sha256"],
        "development_evidence": {},
    }
    return config, lock


def originals(
    root, config, lock, *, calibration=None, omit=None, unknown=False, role=None, bad_freeze=False
):
    is_cal = calibration is None
    role = role or ("interval_calibration" if is_cal else "orthogonal_generalization")
    axes = (
        [config["cpu"]["interval_calibration_seeds"], ["X_BASE"], [8], [0], [7001]]
        if is_cal
        else [
            config["cpu"]["orthogonal_generalization"]["train_rng_seeds"],
            config["cpu"]["arms"],
            config["cpu"]["anchors"],
            range(config["cpu"]["repeat_measurements_test"]),
            config["cpu"]["orthogonal_generalization"]["init_seeds"],
        ]
    )
    designs = development_designs(config, selected=lock["selected"])
    rows = []
    root.mkdir()
    for seed, arm, anchor, repeat, init in itertools.product(*axes):
        identity = (seed, arm, anchor, repeat, init)
        if omit and omit(identity):
            continue
        unit = root / "units" / f"seed{seed}_{arm}_init{init}_a{anchor}_repeat{repeat:02d}"
        unit.mkdir(parents=True)
        first = np.tile([0.01, 0.02, -0.01, -0.02], (2, 1))
        second = first * 0.5
        truth = np.stack((first, second, first - second))
        _npz(unit / "QUERY_REFERENCE.npz", truth=truth)
        entries = []
        for design in designs:
            design_id = _digest(design)[:20]
            fit = unit / "fits" / design_id
            fit.mkdir(parents=True)
            error = (seed - 21000) * 1e-6 if is_cal else (1e-5 if init == 7101 else 1e-3)
            prediction = truth + error
            if design["model"] == "ZERO" and not is_cal:
                prediction = np.zeros_like(truth)
            if unknown and not is_cal and seed == 23001 and init == 7101:
                prediction[0, 0, 0] = np.nan
            labels = np.full(3, "MEASUREMENT_UNRESOLVED")
            if design["model"] == "ZERO":
                labels[:] = "STATISTICAL_BASELINE_ONLY"
            _npz(
                fit / "PREDICTIONS.npz",
                prediction=prediction,
                classifications=labels,
                rho=np.zeros(3),
                leverage=np.zeros(3),
            )
            row = {
                "origin_id": unit.name,
                "design_id": design_id,
                "design": design,
                "seed": seed,
                "arm": arm,
                "initialization_seed": init,
                "anchor": anchor,
                "repeat": repeat,
                "k": 2,
                "r": 1,
                "predictions_relative": str((fit / "PREDICTIONS.npz").relative_to(unit)),
                "prediction_sha256": sha256_file(fit / "PREDICTIONS.npz"),
            }
            _finish(fit, {"design": design}, row)
            entries.append(row)
        frozen = {row["design_id"]: row["prediction_sha256"] for row in entries}
        if bad_freeze:
            frozen[next(iter(frozen))] = "not-the-frozen-prediction"
        _json(unit / "PREDICTION_LOCK.json", frozen)
        _json(
            unit / "REFERENCE_ACCESS_RECEIPT.json",
            {
                "prediction_lock_sha256": sha256_file(unit / "PREDICTION_LOCK.json"),
                "heldout_read_after_prediction_freeze": True,
            },
        )
        direct = unit / "direct_n64"
        direct.mkdir()
        _npz(direct / "estimates.npz", RAW4=np.stack((first, second))[None])
        _finish(direct, {"n": 64}, {"n": 64})
        _finish(unit, {"role": role}, {"origin": unit.name, "results": entries})
        rows.extend(entries)
    _finish(
        root,
        {
            "config_sha256": canonical_hash(config),
            "designs": designs,
            "source_hashes": source_identity()["files"],
            "calibration_receipt": calibration,
        },
        {"role": role, "pilot": False, "probe_groups": [0, 1], "results": rows},
    )
    return root


def prepared(tmp_path, **kwargs):
    config, lock = setup_config()
    cal = originals(tmp_path / "cal", config, lock)
    receipt = analyze_calibration(config, lock, [cal], tmp_path / "cal-analysis", fixture=True)
    root = originals(tmp_path / "generalization", config, lock, calibration=receipt, **kwargs)
    return config, lock, receipt, root


def analyze(tmp_path, **kwargs):
    from src.modeling_v3.cpu_generalization import analyze_generalization

    config, lock, receipt, root = prepared(tmp_path, **kwargs)
    return analyze_generalization(
        config, lock, [root], tmp_path / "analysis", calibration_receipt=receipt, fixture=True
    )


def test_initializations_remain_separate_with_seed_bootstrap_and_frozen_risk(tmp_path):
    report = analyze(tmp_path)
    assert report["schema"] == "ssvc-v3-cpu-generalization-analysis-1"
    assert report["stage"] == "Q3_ORTHOGONAL_GENERALIZATION_ANALYSIS"
    assert report["status"] == "TEST_FIXTURE_NOT_AUTHORIZATION"
    assert report["distribution_shift_included"] is True
    assert report["calibration_guarantee_transfers"] is False
    assert report["pooled_initialization_inference"] is False
    strata = {row["initialization_seed"]: row for row in report["initialization_strata"]}
    assert set(strata) == {7101, 7201}
    for init, error in ((7101, 1e-5), (7201, 1e-3)):
        layer = strata[init]
        row = next(
            r
            for r in layer["metrics"]
            if r["kind"] == "MODEL" and r["target_index"] == 0 and r["channel"] == "X"
        )
        assert row["all_case_mse"] == pytest.approx(error**2)
        assert row["initialization_seed"] == init
        assert row["all_cases"] == 16  # 2 seeds, 2 arms, 2 repetitions, 2 probes
        assert {r["target_index"] for r in layer["metrics"]} == {0, 1, 2}
        assert all(r["independent_seeds"] == 2 for r in layer["paired_comparisons"])
        assert all(r["reps"] == 5000 for r in layer["paired_comparisons"])
        assert any(r["kind"] == "DIRECT_MEASURE" for r in layer["metrics"])
        assert any(r["kind"] == "KNOWN_EVENT_SCORE" for r in layer["metrics"])
        model_ids = {r["design_id"] for r in layer["metrics"] if r["kind"] == "MODEL"}
        assert {r["design_id"] for r in layer["risk_coverage_curves"]} == model_ids
        cov = next(r for r in layer["interval_coverage"] if r["design_id"] in model_ids)
        assert cov["empirical_seed_coverage"] == (1 if init == 7101 else 0)
        assert cov["distribution_shift_covered"] is False
        assert all(
            r["thresholds_chosen_from_generalization_errors"] is False
            for r in layer["risk_coverage_curves"]
        )
        assert all(
            r["seed_bootstrap"]["independent_seeds"] == 2 for r in layer["risk_coverage_curves"]
        )
        assert all(r["seed_bootstrap"]["reps"] == 5000 for r in layer["risk_coverage_curves"])
    manifest = verify_manifest(tmp_path / "analysis")
    assert manifest["metadata"]["stage"] == report["stage"]
    assert (tmp_path / "analysis/SEED_METRICS.csv").is_file()


@pytest.mark.parametrize("axis,value", [(0, 23001), (1, "X_BASE"), (3, 1), (4, 7101)])
def test_missing_entire_frozen_axis_is_rejected_even_in_fixture(tmp_path, axis, value):
    with pytest.raises(ValueError, match=r"orthogonal.*matrix"):
        analyze(tmp_path, omit=lambda identity: identity[axis] == value)
    assert not (tmp_path / "analysis").exists()


def test_unknown_predictions_retain_denominators_and_block_paired_inference(tmp_path):
    report = analyze(tmp_path, unknown=True)
    layer = report["initialization_strata"][0]
    row = next(
        r
        for r in layer["metrics"]
        if r["kind"] == "MODEL" and r["target_index"] == 0 and r["channel"] == "X"
    )
    assert row["all_cases"] == 16 and row["finite_cases"] == 8
    assert row["all_case_mse"] is None
    model_ids = {r["design_id"] for r in layer["metrics"] if r["kind"] == "MODEL"}
    pairs = [
        r
        for r in layer["paired_comparisons"]
        if r["design_id"] in model_ids and r["target_index"] == 0
    ]
    assert all(r["status"] == "UNKNOWN_INCOMPLETE_PAIRED_PREDICTIONS" for r in pairs)
    cov = next(r for r in layer["interval_coverage"] if r["design_id"] in model_ids)
    assert cov["unresolved_seeds"] == [23001]
    assert cov["empirical_seed_coverage"] == 0.5


def test_full_matrix_contract_requires_40_trajectories_and_2400_units():
    from src.modeling_v3.cpu_generalization import _validate_matrix

    config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v3/protocol.json").read_text()
    )
    cpu = config["cpu"]
    axes = [
        cpu["orthogonal_generalization"]["train_rng_seeds"],
        cpu["arms"],
        cpu["anchors"],
        range(cpu["repeat_measurements_test"]),
        cpu["orthogonal_generalization"]["init_seeds"],
    ]
    keys = list(itertools.product(*axes))
    rows = [
        {
            "row": dict(
                zip(("seed", "arm", "anchor", "repeat", "initialization_seed"), key, strict=True)
            )
        }
        for key in keys
    ]
    result = _validate_matrix(config, {"records": {"design": rows}}, fixture=False)
    assert result["trajectory_count"] == 40
    assert result["unit_count_per_design"] == 2400
    with pytest.raises(ValueError, match=r"orthogonal.*matrix"):
        _validate_matrix(config, {"records": {"design": rows[:-1]}}, fixture=False)


def test_fixture_calibration_cannot_authorize_production_and_payload_tamper_fails(tmp_path):
    from src.modeling_v3.cpu_generalization import analyze_generalization

    config, lock, receipt, root = prepared(tmp_path)
    path = tmp_path / "cal-analysis/CALIBRATION_RECEIPT.json"
    with pytest.raises(ValueError, match="fixture"):
        analyze_generalization(
            config, lock, [root], tmp_path / "production", calibration_receipt=path
        )
    next(root.rglob("PREDICTIONS.npz")).write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash"):
        analyze_generalization(
            config, lock, [root], tmp_path / "tampered", calibration_receipt=receipt, fixture=True
        )


def test_wrong_seed_role_and_calibration_chain_are_rejected(tmp_path):
    from src.modeling_v3.cpu_generalization import analyze_generalization

    config, lock, receipt, root = prepared(tmp_path, role="locked_test")
    with pytest.raises(ValueError, match="role"):
        analyze_generalization(
            config, lock, [root], tmp_path / "analysis", calibration_receipt=receipt, fixture=True
        )
    receipt["source_manifest_hashes"] = {}
    with pytest.raises(ValueError, match="originals"):
        analyze_generalization(
            config,
            lock,
            [root],
            tmp_path / "bad-calibration",
            calibration_receipt=receipt,
            fixture=True,
        )


def test_internally_bound_bad_prediction_freeze_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="prediction lock"):
        analyze(tmp_path, bad_freeze=True)


def test_receipt_file_manifest_and_prediction_receipt_binding_are_checked(tmp_path):
    from src.modeling_v3.cpu_generalization import analyze_generalization

    config, lock, receipt, root = prepared(tmp_path)
    receipt["designs"][next(iter(receipt["designs"]))]["radius"] = 0.01
    with pytest.raises(ValueError, match="different calibration receipt"):
        analyze_generalization(
            config,
            lock,
            [root],
            tmp_path / "changed-binding",
            calibration_receipt=receipt,
            fixture=True,
        )
    path = tmp_path / "cal-analysis/CALIBRATION_RECEIPT.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="hash"):
        analyze_generalization(
            config, lock, [root], tmp_path / "changed-file", calibration_receipt=path, fixture=True
        )


def test_zero_acceptance_seed_risk_bootstrap_keeps_all_seeds():
    from src.modeling_v3.cpu_generalization import _risk_bootstrap

    config, _ = setup_config()
    result = _risk_bootstrap(
        [
            {"all_cases": 10, "accepted_cases": 0, "accepted_error_ss": 0},
            {"all_cases": 10, "accepted_cases": 0, "accepted_error_ss": 0},
        ],
        config,
    )
    assert result["independent_seeds"] == 2
    assert result["coverage_interval"] == [0, 0]
    assert result["accepted_mse_interval"] is None
    assert result["replicates_with_no_accepted_cases"] == 5000
