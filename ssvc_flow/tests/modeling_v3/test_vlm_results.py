"""Small, immutable originals; no model execution and no server authorization."""

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from src.modeling_v3 import vlm_results as v
from src.modeling_v3.io import atomic_json, atomic_npz, canonical_hash, finalize_run, sha256_file
from src.modeling_v3.vlm_response import _identity


@pytest.fixture(autouse=True)
def stable_fixture_source(monkeypatch):
    # Other agents may edit source while this isolated fixture is running.
    # These receipts are explicitly CPU fixtures and can never authorize a run.
    monkeypatch.setattr(
        "src.modeling_v3.vlm_response.source_identity",
        lambda: {"sha256": "fixture-source", "files": {}},
    )


def bound(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def publish(root, config, name, doc, arrays=None):
    root.mkdir(parents=True)
    if arrays is not None:
        atomic_npz(root / "ARRAYS.npz", arrays)
        doc["arrays"] = bound(root / "ARRAYS.npz")
    atomic_json(root / name, doc)
    finalize_run(root, _identity(config))
    return bound(root / name)


def protocol(tmp_path, *, two_designs=False):
    config = json.loads(Path("configs/modeling_v3/protocol.json").read_text())
    config["qwen"]["probe_panel"]["prompts"] = 1
    config["qwen"]["training_bank_partition"]["heldout_banks"] = 1
    design = {
        "design_id": "ridge",
        "method": "FULL_RIDGE",
        "rank_cap": "FULL",
        "alpha": 0.0,
        "output_policy": "RAW4",
        "regression": "RIDGE",
        "observation_method": "RAW4",
        "selector": "FIRST",
        "n_banks": 1,
        "selection_seed": 7,
    }
    selected = {
        "designs": [design],
        "rho_threshold": 0.05,
        "leverage_threshold": 10.0,
        "pointwise_criteria": {
            "primary_tolerance": 0.001,
            "reference_half_width_max": 0.00025,
            "minimum_geometry_coverage": 0.5,
            "maximum_q95_absolute_residual": 0.001,
            "maximum_nrmse": 0.5,
        },
    }
    if two_designs:
        selected["designs"].append({**design, "design_id": "pca", "method": "PCA", "rank_cap": 1})
    lock = {
        "kind": "V3_VLM_SELECTION_LOCK",
        **_identity(config),
        "selected": selected,
        "selection_hash": canonical_hash(selected),
        "fixture": True,
    }
    lock_binding = publish(tmp_path / "selection", config, "SELECTION_LOCK.json", lock)
    return config, lock_binding, design


def origin(
    tmp_path,
    config,
    lock,
    design,
    seed,
    arm,
    step,
    *,
    calibration=None,
    error=1e-5,
    unresolved=False,
    mask_override=False,
    role_override=None,
    omit_variance=False,
    false_variance=False,
):
    role = (
        "interval_calibration"
        if seed in config["qwen"]["seed_roles"]["interval_calibration"]
        else "locked_test"
    )
    role = role_override or role
    origin_id = f"{seed}_{arm}_{step}"
    root = tmp_path / origin_id
    if design["design_id"] != "ridge":
        root = root / design["design_id"]
    identity = {**_identity(config), "fixture": True, "origin_id": origin_id, "role": role}
    units = [{"bank_id": "H0", "contrast_id": name} for name in v.TARGETS]
    truth = np.broadcast_to(np.array([0.01, 0.02, -0.01, -0.02]), (3, 1, 4)).copy()
    predictions = truth + error
    geometry = publish(
        root / "geometry",
        config,
        "GEOMETRY_SPEC.json",
        {
            "kind": "V3_Q5_CALIBRATION_GEOMETRY",
            **identity,
            "method": design["selector"],
            "n_banks": design["n_banks"],
            "seed": design["selection_seed"],
        },
    )
    fit = publish(
        root / "fit_input",
        config,
        "FIT_SPEC.json",
        {
            "kind": "V3_Q5_RESPONSE_FIT",
            **identity,
            **design,
            "selection_lock": lock,
            "geometry": geometry,
            "probe_ids": ["p0"],
            "query_units": units,
            "reference_labels_read": False,
            "heldout_labels_read": False,
        },
        {"updates": np.eye(2), "query_updates": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])},
    )
    model_root = root / "model"
    model_root.mkdir()
    atomic_npz(model_root / "MODEL_ARRAYS.npz", {"predictions": predictions, "Q": np.eye(2)})
    atomic_json(
        model_root / "MODEL.json",
        {"k": 2, "r": 2 if design["rank_cap"] == "FULL" else design["rank_cap"]},
    )
    atomic_json(
        model_root / "RECEIPT.json",
        {"command": "fit", "input_binding": {"spec_sha256": fit["sha256"]}},
    )
    finalize_run(model_root, _identity(config))
    model = bound(model_root / "MODEL_ARRAYS.npz")
    accepted = np.zeros((3, 1), dtype=bool)
    if calibration is not None:
        accepted = v.derive_vlm_acceptance(config, lock, calibration, fit, model, fixture=True)[
            "accepted"
        ]
    if mask_override:
        accepted[0, 0] = ~accepted[0, 0]
    prediction = publish(
        root / "prediction",
        config,
        "PREDICTION_LOCK.json",
        {
            "kind": "V3_FROZEN_PREDICTIONS",
            **identity,
            "query_units": units,
            "probe_ids": ["p0"],
            "fit_spec": fit,
            "model_arrays": model,
            "selection_lock": lock,
            "reference_labels_read": False,
            "heldout_labels_read": False,
            "accepted_count": int(accepted.sum()),
            "calibration_receipt": calibration,
        },
        {"predictions": predictions, "accepted": accepted},
    )
    eval_root = root / "evaluation"
    eval_root.mkdir()
    raw_offset = np.tile([-1e-6, 1e-6], 4)
    raw = truth[0, 0] + raw_offset[:, None]
    variance = np.var(raw, axis=0, ddof=1) / len(raw)
    ref = truth.copy()
    if unresolved:
        ref[:] = np.nan
    arrays = {
        "predictions": predictions,
        "accepted": accepted,
        "reference": ref,
        "reference_estimate_unmasked": truth,
        "reference_variance": np.broadcast_to(variance, truth.shape),
        "query_direct_measurement": truth + 2e-5,
        "direct_count_measurement": truth + 3e-5,
    }
    if omit_variance:
        arrays.pop("reference_variance")
    elif false_variance:
        arrays["reference_variance"] = np.zeros_like(truth)
    reports = []
    for i, unit in enumerate(units):
        arrays[f"reference_origin_{i}_0"] = raw
        arrays[f"reference_origin_{i}_0_covariance"] = np.cov(raw, rowvar=False, ddof=1) / len(raw)
        half = NormalDist().inv_cdf(1 - 0.01 / 2) * np.sqrt(variance)
        reports.append(
            {
                **unit,
                "prompt_id": "p0",
                "selected_proposal": "UNRESOLVED_ORIGIN_DIAGNOSTIC" if unresolved else "ORIGIN",
                "crosscheck_consistent": True,
                "reference_is_exact_truth": False,
                "mix": None,
                "origin": {
                    "n": len(raw),
                    "proposal": "ORIGIN",
                    "standard_error": np.sqrt(variance).tolist(),
                    "empirical_normal_half_width": half.tolist(),
                    "formal_hoeffding_half_width": None,
                    "alpha_per_interval": 0.01,
                    "zero_variance_event_indices": [],
                    "empirical_precision_met": not unresolved,
                },
            }
        )
    atomic_npz(eval_root / "EVALUATION_ARRAYS.npz", arrays)
    spec = {
        "kind": "V3_Q5_REFERENCE_EVALUATION_INPUT",
        **identity,
        "arrays": bound(eval_root / "EVALUATION_ARRAYS.npz"),
        "prediction_lock": prediction,
        "query_units": units,
        "probe_ids": ["p0"],
        "reference_kind": "INDEPENDENT_NOISY_PRECISION_MASKED",
        "primary_delta_v": "-delta_pI",
    }
    atomic_json(eval_root / "EVALUATION_INPUT.json", spec)
    atomic_json(
        eval_root / "REFERENCE_PRECISION_DIAGNOSTICS.json",
        {"unit_reports": reports, "precision_only": True, "predictor_rankings_used": False},
    )
    atomic_json(
        eval_root / "REFERENCE_PRECISION_RECEIPT.json",
        {
            "kind": "V3_REFERENCE_PRECISION",
            **identity,
            "prediction_binding": prediction,
            "precision_only": True,
            "predictor_rankings_used": False,
            "decision_evidence": bound(eval_root / "REFERENCE_PRECISION_DIAGNOSTICS.json"),
        },
    )
    atomic_json(
        eval_root / "LABEL_ACCESS.json",
        {
            "prediction_lock": prediction,
            "prediction_validated_before_any_heldout_reference_labels": True,
        },
    )
    atomic_json(
        eval_root / "RESPONSE_EVALUATION_RECEIPT.json",
        {
            "kind": "V3_RESPONSE_EVALUATION",
            **identity,
            "evaluation_input": bound(eval_root / "EVALUATION_INPUT.json"),
            "prediction_lock": prediction,
            "fit_spec": fit,
            "reference_precision": bound(eval_root / "REFERENCE_PRECISION_RECEIPT.json"),
            "measurement_bundle": {"fixture": True},
            "reference_bundle": {"fixture": True},
        },
    )
    finalize_run(eval_root, _identity(config))
    return bound(eval_root / "EVALUATION_INPUT.json")


def stage(tmp_path, config, inputs, role):
    tasks = {}
    for item in inputs:
        spec = json.loads(Path(item["path"]).read_text())
        pred = json.loads(Path(spec["prediction_lock"]["path"]).read_text())
        fit = json.loads(Path(pred["fit_spec"]["path"]).read_text())
        tasks["evaluation_" + spec["origin_id"] + "_" + fit["design_id"]] = {
            "binding": bound(Path(item["path"]).parent / "RESPONSE_EVALUATION_RECEIPT.json")
        }
    return publish(
        tmp_path / (role + "_completion"),
        config,
        "Q5_STAGE_COMPLETION.json",
        {
            "kind": "V3_Q5_STAGE_COMPLETION",
            **_identity(config),
            "fixture": True,
            "role": role,
            "status": "COMPLETE",
            "verified_tasks": tasks,
            "technical_failures": [],
        },
    )


def campaign(tmp_path, config, lock, design, role, *, calibration=None, **kwargs):
    inputs = []
    for seed in config["qwen"]["seed_roles"][role]:
        origins = [("X_BASE", 32), ("X_BASE", 96)]
        if role == "locked_test":
            origins.append(("X_VALID", 96))
        for arm, step in origins:
            inputs.append(
                origin(
                    tmp_path,
                    config,
                    lock,
                    design,
                    seed,
                    arm,
                    step,
                    calibration=calibration,
                    **kwargs,
                )
            )
    return inputs, stage(tmp_path, config, inputs, role)


def calibrated(tmp_path, *, unresolved=False):
    config, lock, design = protocol(tmp_path)
    inputs, completion = campaign(
        tmp_path, config, lock, design, "interval_calibration", unresolved=unresolved
    )
    report = v.analyze_vlm_calibration(
        config,
        lock,
        stage_completion=completion,
        evaluation_inputs=inputs,
        out=tmp_path / "cal",
        fixture=True,
    )
    return config, lock, design, report, inputs, completion


def test_three_seed_empirical_calibration_is_not_95_percent_certification(tmp_path):
    config, lock, _, report, _, _ = calibrated(tmp_path)
    assert report["calibration_seeds"] == [42001, 42002, 42003]
    assert report["conformal_95"]["radius"] is None
    assert report["conformal_95"]["order_statistic"] == 4
    assert report["cross_seed_95"] == "NOT_CERTIFIED"
    assert report["designs"]["ridge"]["pointwise_calibration_eligible"] is True
    assert (
        v.verify_vlm_calibration(config, lock, report["receipt"], fixture=True)["selection_lock"]
        == lock
    )
    with pytest.raises((ValueError, PermissionError), match=r"fixture|server|Slurm"):
        v.verify_vlm_calibration(config, lock, report["receipt"])


def test_independent_test_keeps_shift_targets_seed_pairing_and_uncertainty(tmp_path):
    config, lock, design, cal, _, _ = calibrated(tmp_path)
    inputs, completion = campaign(
        tmp_path, config, lock, design, "locked_test", calibration=cal["receipt"]
    )
    result = v.analyze_vlm_test(
        config,
        lock,
        calibration_receipt=cal["receipt"],
        stage_completion=completion,
        evaluation_inputs=inputs,
        out=tmp_path / "test",
        fixture=True,
    )
    row = result["designs"]["ridge"]
    assert row["pointwise_qualified"]
    assert row["primary"]["origins"] == 12
    assert row["shift"]["origins"] == 6
    assert set(row["primary"]["targets"]) == set(v.TARGETS)
    target = row["primary"]["targets"][v.TARGETS[0]]
    assert target["MODEL"]["all_case_count"] == 12
    assert target["MODEL"]["accepted_case_count"] == 12
    assert target["MODEL"]["aggregate_reference_noise_corrected_mse"] < target["MODEL"]["mse"]
    assert target["MODEL"]["pX_v"]["v_definition"] == "-delta_pI"
    assert target["paired_model_minus_direct"]["reps"] == 5000
    assert target["paired_model_minus_direct"]["independent_seeds"] == 6
    assert "DIRECT_COUNTS" in target
    receipt = v.verify_vlm_qualification(config, result["qualification_receipt"], fixture=True)
    assert receipt["qualified_design_ids"] == ["ridge"]
    assert receipt["tracking_eligible_design_ids"] == ["ridge"]
    assert receipt["cross_seed_95"] == "NOT_CERTIFIED"
    with pytest.raises(ValueError, match=r"path|binding|original"):
        v.verify_vlm_qualification(config, {"pointwise_qualified": True}, fixture=True)
    # Qualification above exists before any step-64 fit. Creating a new window
    # fit afterwards must not require step-64 measurement to justify Q5 itself.
    future = origin(tmp_path, config, lock, design, 43001, "X_BASE", 64, calibration=cal["receipt"])
    evaluation = json.loads(Path(future["path"]).read_text())
    prediction = json.loads(Path(evaluation["prediction_lock"]["path"]).read_text())
    tracked = v.verify_vlm_qualification(
        config, result["qualification_receipt"], fit_binding=prediction["fit_spec"], fixture=True
    )
    assert tracked["fit_hash"] == prediction["fit_spec"]["sha256"]
    assert tracked["tracking_qualified"] is True


def test_unresolved_reference_cannot_create_calibration_eligibility(tmp_path):
    config, lock, design, report, _, _ = calibrated(tmp_path, unresolved=True)
    assert not report["designs"]["ridge"]["pointwise_calibration_eligible"]
    inputs, _ = campaign(
        tmp_path, config, lock, design, "locked_test", calibration=report["receipt"]
    )
    with np.load(json.loads(Path(inputs[0]["path"]).read_text())["arrays"]["path"]) as data:
        assert not data["accepted"].any()


def test_completed_matrix_rejects_duplicates_missing_origin_and_role_drift(tmp_path):
    config, lock, design, _, inputs, completion = calibrated(tmp_path)
    for bad in [inputs + inputs[:1], inputs[:-1]]:
        with pytest.raises(ValueError, match=r"(?i)duplicate|matrix|complete"):
            v.analyze_vlm_calibration(
                config,
                lock,
                stage_completion=completion,
                evaluation_inputs=bad,
                out=tmp_path / "bad",
                fixture=True,
            )
    wrong = origin(tmp_path, config, lock, design, 43001, "X_BASE", 32, role_override="development")
    with pytest.raises(ValueError, match="role"):
        v.analyze_vlm_calibration(
            config,
            lock,
            stage_completion=completion,
            evaluation_inputs=[wrong, *inputs[1:]],
            out=tmp_path / "wrong",
            fixture=True,
        )


def test_test_errors_never_select_acceptance_mask(tmp_path):
    config, lock, design, cal, _, _ = calibrated(tmp_path)
    inputs, completion = campaign(
        tmp_path, config, lock, design, "locked_test", calibration=cal["receipt"], error=0.02
    )
    result = v.analyze_vlm_test(
        config,
        lock,
        calibration_receipt=cal["receipt"],
        stage_completion=completion,
        evaluation_inputs=inputs,
        out=tmp_path / "bad_test",
        fixture=True,
    )
    row = result["designs"]["ridge"]
    assert not row["pointwise_qualified"]
    assert row["primary"]["targets"][v.TARGETS[0]]["MODEL"]["accepted_case_count"] == 12
    assert row["primary"]["targets"][v.TARGETS[0]]["MODEL"]["geometry_coverage"] == 1.0


def test_tampered_frozen_mask_and_original_bytes_are_rejected(tmp_path):
    config, lock, design, cal, inputs, _ = calibrated(tmp_path)
    tests, completion = campaign(
        tmp_path,
        config,
        lock,
        design,
        "locked_test",
        calibration=cal["receipt"],
        mask_override=True,
    )
    with pytest.raises(ValueError, match=r"acceptance|mask"):
        v.analyze_vlm_test(
            config,
            lock,
            calibration_receipt=cal["receipt"],
            stage_completion=completion,
            evaluation_inputs=tests,
            out=tmp_path / "bad",
            fixture=True,
        )
    raw_path = Path(json.loads(Path(inputs[0]["path"]).read_text())["arrays"]["path"])
    raw_path.write_bytes(raw_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=r"hash|manifest"):
        v.verify_vlm_calibration(config, lock, cal["receipt"], fixture=True)


@pytest.mark.parametrize("option", ["omit_variance", "false_variance"])
def test_reference_uncertainty_is_required_and_recomputed(tmp_path, option):
    config, lock, design = protocol(tmp_path)
    inputs, completion = campaign(
        tmp_path, config, lock, design, "interval_calibration", **{option: True}
    )
    with pytest.raises(ValueError, match="variance"):
        v.analyze_vlm_calibration(
            config,
            lock,
            stage_completion=completion,
            evaluation_inputs=inputs,
            out=tmp_path / "cal",
            fixture=True,
        )


def test_frozen_design_matrix_and_real_non_degenerate_rank_comparison(tmp_path):
    config, lock, _ = protocol(tmp_path, two_designs=True)
    designs = json.loads(Path(lock["path"]).read_text())["selected"]["designs"]
    calibration_inputs = []
    for seed in config["qwen"]["seed_roles"]["interval_calibration"]:
        for step in (32, 96):
            for design in designs:
                calibration_inputs.append(
                    origin(tmp_path, config, lock, design, seed, "X_BASE", step)
                )
    completion = stage(tmp_path, config, calibration_inputs, "interval_calibration")
    with pytest.raises(ValueError, match="matrix"):
        v.analyze_vlm_calibration(
            config,
            lock,
            stage_completion=completion,
            evaluation_inputs=calibration_inputs[:-1],
            out=tmp_path / "incomplete",
            fixture=True,
        )
    cal = v.analyze_vlm_calibration(
        config,
        lock,
        stage_completion=completion,
        evaluation_inputs=calibration_inputs,
        out=tmp_path / "cal",
        fixture=True,
    )
    test_inputs = []
    for seed in config["qwen"]["seed_roles"]["locked_test"]:
        for arm, step in (("X_BASE", 32), ("X_BASE", 96), ("X_VALID", 96)):
            for design in designs:
                test_inputs.append(
                    origin(
                        tmp_path, config, lock, design, seed, arm, step, calibration=cal["receipt"]
                    )
                )
    completion = stage(tmp_path, config, test_inputs, "locked_test")
    report = v.analyze_vlm_test(
        config,
        lock,
        calibration_receipt=cal["receipt"],
        stage_completion=completion,
        evaluation_inputs=test_inputs,
        out=tmp_path / "test",
        fixture=True,
    )
    assert set(report["designs"]) == {"ridge", "pca"}
    (comparison,) = report["dimension_comparisons"]
    assert comparison["r_less_k_every_origin"] is True
    assert comparison["targets"][v.TARGETS[0]]["estimate"] == 0
    assert comparison["targets"][v.TARGETS[0]]["independent_seeds"] == 6


def test_seed_bootstrap_nrmse_uses_total_energy_and_q95_uses_raw_residuals(tmp_path):
    config, _, _ = protocol(tmp_path)
    reference = np.array([[1.0, 1.0, 1.0, 1.0], [100.0, 100.0, 100.0, 100.0]])
    prediction = reference + np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    result = v._cluster_error_intervals(
        np.array([1, 2]), prediction, reference, np.array([True, True]), config
    )
    assert result["nrmse"]["estimate"] == pytest.approx(np.sqrt(1 / 10001))
    metric = v._metric(
        prediction,
        reference,
        np.zeros_like(reference),
        np.ones_like(reference) * 1e-6,
        np.array([True, True]),
    )
    assert metric["q95_absolute_residual"] == 1.0
    assert metric["q95_absolute_residual"] != np.mean([1.0, 0.0])
