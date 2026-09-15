"""Independent raw-array recomputation catches both score and hash tampering."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_contrast.recompute import (
    independent_model_statistics,
    independent_observation_statistics,
    independent_raw_covariance,
    project_contrast_diagnostic,
    run_recompute,
)


def test_target_specific_active_population_uses_fit_excitation_only():
    metadata = [{"prompt_id": "a", "group": 0, "weight": 1.0}]
    pred = np.zeros((4, 2, 1, 4))
    truth = pred.copy()
    truth[:, 1, 0] = [0.2, 0, 0, -0.2]  # Test target is active; fit target is not.
    fit = np.array([[[1.0, 0], [0, 0]]])
    rows = independent_model_statistics(pred, truth, metadata, fit, {})
    assert len(rows) == 5
    assert not any(
        r["target"] == "no_x_off_1_minus_joint_0" and r["population"] == "active" for r in rows
    )
    target = next(r for r in rows if r["target"] == "no_x_off_1_minus_joint_0")
    assert np.isclose(target["pX_error_ss"], 0.16)
    assert target["pX_nrmse"] == 1.0


def test_raw_event_validity_uses_sum_xsw_for_nonzero_mass_estimators():
    pred = np.array([[[0.2, 0.3, 0.1, -0.1]], [[0.4, 0.3, 0.1, -0.1]]])
    truth = np.array([[0.1, 0.2, 0.1, -0.4]])
    metadata = [{"prompt_id": "a", "group": 0, "weight": 1.0}]
    raw = independent_observation_statistics(pred, pred, truth, metadata, "raw_event")
    projected = independent_observation_statistics(pred, pred, truth, metadata, "helmert")
    assert np.isclose(raw["v_error_ss"], 0.2)
    assert np.isclose(projected["v_error_ss"], 0.18)
    assert np.isclose(raw["mse"], np.mean((pred - truth) ** 2))


def test_posthoc_projection_is_sparse_recoverable_and_does_not_modify_raw():
    baseline = np.full((1, 1, 4), 0.25)
    raw = np.array([[[[-0.35, 0.25, 0.05, 0.05]], [[0.0, 0.0, 0.0, 0.0]]]])
    original = raw.copy()
    result = project_contrast_diagnostic(raw, baseline)
    assert np.array_equal(raw, original)
    assert result["changed_indices"].tolist() == [[0, 0, 0]]
    restored = raw.copy()
    restored[tuple(result["changed_indices"].T)] = result["projected_contrast_events"]
    assert np.array_equal(restored, result["projected"])
    assert np.allclose(result["projected"][0, 0, 0] + baseline[0, 0], [0, 7 / 15, 4 / 15, 4 / 15])
    assert result["negative_component_count"] == 1
    assert result["main_ranking_eligible"] is False


@pytest.mark.parametrize("method", ["O_IND", "O_CRN", "O_LR_ORIGIN", "O_LR_MIX"])
@pytest.mark.parametrize("alias", [False, True])
def test_independent_joint_raw_covariance_matches_exact_action_moments(method, alias):
    from src.modeling_contrast.observations import OracleAudit

    p = np.array([[0.2, 0.15, 0.2, 0.1, 0.2, 0.15]])
    q = p + np.array([[0.001, -0.002, 0.002, 0, 0, -0.001]])
    r = p + np.array([[-0.003, 0.001, 0.001, 0.001, 0, 0]])
    tables = np.stack((p, q, q if alias else r))
    labels = np.array([[2, 3, 0, 1, 2, 3]])
    identities = ("b", "u", "u" if alias else "v")
    actual = independent_raw_covariance(tables, labels, method, 64, policy_ids=identities, origin=p)
    oracle = OracleAudit.from_action_probabilities(
        {"b": p, "u": q, "v": q if alias else r, "o": p}, labels
    )
    expected = oracle.moments([("b", "u"), ("b", "v")], method=method, n=64, origin_id="o")[
        "raw_covariance"
    ]
    assert np.allclose(actual, expected, atol=1e-15, rtol=1e-10)


def test_independent_lr_covariance_rejects_missing_proposal_support():
    tables = np.array([[[0.5, 0.5]], [[0.6, 0.4]], [[0.5, 0.5]]])
    with pytest.raises(ValueError, match="SUPPORT"):
        independent_raw_covariance(
            tables, np.array([[0, 3]]), "O_LR_ORIGIN", 64, origin=np.array([[0.0, 1.0]])
        )


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def fixture(tmp_path):
    parent, run = tmp_path / "parent", tmp_path / "run"
    parent.mkdir()
    run.mkdir()
    metadata = [{"prompt_id": str(i), "group": i, "weight": 0.5} for i in range(2)]
    _json(parent / "probe_metadata.json", metadata)
    theta = np.zeros((3, 14, 3, 2))
    theta[0, :, 1, 0] = 1.0
    probability = np.full((3, 14, 3, 2, 4), 0.25)
    probability[0, :, 1, :, 0] += 0.01
    probability[0, :, 1, :, 3] -= 0.01
    np.savez_compressed(parent / "obs.npz", branch_theta=theta)
    np.savez_compressed(parent / "oracle.npz", branch_p=probability)
    entry = {
        "id": "seed101_X_BASE",
        "seed": 101,
        "arm": "X_BASE",
        "split": "development",
        "observations_file": "obs.npz",
        "oracle_file": "oracle.npz",
        "sha256": {
            "observations_file": _sha(parent / "obs.npz"),
            "oracle_file": _sha(parent / "oracle.npz"),
        },
    }
    _json(parent / "manifest.json", {"anchors": [8, 24, 40], "trajectories": [entry]})
    unit = run / "N2/N2A/seed101_X_BASE/a0"
    unit.mkdir(parents=True)
    predictions = np.zeros((1, 4, 2, 2, 3))
    np.savez_compressed(unit / "predictions.npz", prediction=predictions)
    ident = {
        "stage": "N2A",
        "method": "C2",
        "observation": "EXACT",
        "n": 0,
        "rank": "FULL",
        "alpha": 1e-5,
        "eta": 0.01,
        "fit_banks": 1,
        "configuration_id": "fixture-C2",
        "noise_replica": 0,
        "seed": 101,
        "arm": "X_BASE",
        "anchor": 8,
        "actual_rank": 1,
        "k": 1,
        "prediction_hash": _array_sha(predictions[0]),
    }
    freeze = {
        "models": [ident],
        "prediction_sha256": _sha(unit / "predictions.npz"),
        "parameters_sha256": entry["sha256"]["observations_file"],
    }
    with gzip.open(unit / "freeze_metadata.json.gz", "wt") as f:
        json.dump(freeze, f)
    _json(
        unit / "freeze.json",
        {
            "metadata_file": "freeze_metadata.json.gz",
            "metadata_sha256": _sha(unit / "freeze_metadata.json.gz"),
        },
    )
    rows = []
    for ti, target in enumerate(
        ["joint_1_minus_joint_0", "no_x_off_1_minus_joint_0", "joint_1_minus_no_x_off_1"]
    ):
        for population in ["all"] if ti == 1 else ["all", "active"]:
            scale = 0 if ti == 1 else 1
            row = dict(
                ident,
                target=target,
                population=population,
                case_count=4,
                event_error_ss=0.0016 * scale,
                event_truth_ss=0.0016 * scale,
                event_mae=0.005 * scale,
                predicted_difference_norm=0.0,
                pX_error_ss=0.0008 * scale,
                pX_truth_ss=0.0008 * scale,
                v_error_ss=0.0008 * scale,
                v_truth_ss=0.0008 * scale,
                pX_abs_errors=[0.01 * scale] * 8,
                v_abs_errors=[0.01 * scale] * 8,
                event_nrmse=1.0 if scale else None,
                pX_nrmse=1.0 if scale else None,
                v_nrmse=1.0 if scale else None,
            )
            rows.append(row)
    _jsonl(unit / "metrics.jsonl.gz", rows)
    _jsonl(run / "N2/METRICS.jsonl.gz", rows)
    return parent, run, unit, rows


def test_complete_small_raw_recomputation_and_explicit_not_run(tmp_path):
    parent, run, _unit, _rows = fixture(tmp_path)
    result = run_recompute(run, parent, tmp_path / "verification")
    assert result["status"] == "PASS"
    assert result["checks"]["N2A"]["matched_rows"] == 5
    assert result["checks"]["N2_final_metrics"]["matched_rows"] == 5
    assert result["checks"]["N1"]["status"] == "NOT_RUN"
    assert result["checks"]["N2D"]["status"] == "NOT_RUN"
    assert result["model_forward_calls"] == result["optimizer_updates"] == 0
    assert (tmp_path / "verification/COVERAGE_MANIFEST.json").exists()


def test_changed_metric_is_detected_without_rewriting_predictions(tmp_path):
    parent, run, unit, rows = fixture(tmp_path)
    before = _sha(unit / "predictions.npz")
    rows[0]["pX_error_ss"] += 0.001
    _jsonl(unit / "metrics.jsonl.gz", rows)
    result = run_recompute(run, parent, tmp_path / "verification")
    assert result["status"] == "FAIL"
    assert any("pX_error_ss" in error["detail"] for error in result["errors"])
    assert _sha(unit / "predictions.npz") == before


def test_changed_prediction_archive_hash_is_detected(tmp_path):
    parent, run, unit, _rows = fixture(tmp_path)
    np.savez_compressed(unit / "predictions.npz", prediction=np.ones((1, 4, 2, 2, 3)))
    result = run_recompute(run, parent, tmp_path / "verification")
    assert result["status"] == "FAIL"
    assert any("hash" in error["detail"].lower() for error in result["errors"])


def test_all_absent_phases_are_not_successful_experiments(tmp_path):
    parent = tmp_path / "parent"
    run = tmp_path / "run"
    parent.mkdir()
    run.mkdir()
    _json(parent / "manifest.json", {"anchors": [8, 24, 40], "trajectories": []})
    _json(parent / "probe_metadata.json", [{"prompt_id": "a", "group": 0, "weight": 1.0}])
    result = run_recompute(run, parent, tmp_path / "verification")
    assert result["status"] == "NOT_RUN"
    assert result["not_run"]


def test_final_aggregate_mae_is_checked_against_raw_errors(tmp_path):
    parent, run, _unit, rows = fixture(tmp_path)
    aggregate = []
    for row in rows:
        result = {k: row[k] for k in ("configuration_id", "target", "population", "seed")}
        result.update(
            seed_count=1,
            actual_rank_min=1,
            actual_rank_max=1,
            k_min=1,
            k_max=1,
            predicted_difference_norm=0.0,
        )
        for metric in ("event", "pX", "v"):
            for suffix in ("error_ss", "truth_ss", "nrmse"):
                result[f"{metric}_{suffix}"] = row[f"{metric}_{suffix}"]
            if metric != "event":
                result[f"{metric}_error_q95"] = row[f"{metric}_abs_errors"][0]
                result[f"{metric}_mae"] = row[f"{metric}_abs_errors"][0]
        aggregate.append(result)
    for filename in ("AGGREGATE.csv", "CONTRAST_PREDICTION_BY_SEED.csv"):
        with (run / "N2" / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
            writer.writeheader()
            writer.writerows(aggregate)
    good = run_recompute(run, parent, tmp_path / "good")
    assert good["status"] == "PASS"
    assert good["checks"]["N2_aggregate"]["matched_rows"] == 5
    aggregate[0]["pX_mae"] += 0.1
    with (run / "N2/AGGREGATE.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    bad = run_recompute(run, parent, tmp_path / "bad")
    assert bad["status"] == "FAIL"
    assert any("pX_mae" in error["detail"] for error in bad["errors"])


def test_n2d_errors_are_recovered_and_kept_out_of_main_ranking(tmp_path):
    parent, run, _unit, _rows = fixture(tmp_path)
    with np.load(parent / "oracle.npz") as data:
        truth = data["branch_p"][0, 10:14, 1:] - data["branch_p"][0, 10:14, :1]
    d = run / "N2/N2D/seed101_X_BASE/a0/O_IND"
    d.mkdir(parents=True)
    np.savez_compressed(d / "predictions.npz", prediction=np.zeros((1, 8, 6)), truth=truth)
    record = {
        "array_index": 0,
        "method": "C5",
        "rank_cap": 1,
        "diagnostic": "FINITE_U_FINITE_COEFFICIENTS",
        "covariance_source": "FINITE_PAID",
        "seed": 101,
        "arm": "X_BASE",
        "anchor": 8,
        "observation": "O_IND",
        "main_ranking_eligible": False,
    }
    _json(
        d / "diagnostics.json",
        {
            "records": [record],
            "main_ranking_eligible": False,
            "prediction_sha256": _sha(d / "predictions.npz"),
        },
    )
    result = run_recompute(run, parent, tmp_path / "verification")
    assert result["status"] == "PASS"
    assert result["checks"]["N2D"]["matched_rows"] == 1
    with (tmp_path / "verification/N2D_RECOMPUTED_POOLED.csv").open() as f:
        recovered = list(csv.DictReader(f))
    primary = next(
        r for r in recovered if r["target"] == "joint_1_minus_joint_0" and r["population"] == "all"
    )
    assert float(primary["pX_nrmse"]) == 1.0
    assert primary["main_ranking_eligible"] == "False"


def test_n1_paid_counts_replay_and_parent_endpoint_truth_binding(tmp_path):
    from src.modeling_contrast.packet_codec import encode_arrays

    parent, run, _, _ = fixture(tmp_path)
    directory = run / "N1/packets/seed101_X_BASE/a0/O_IND_n100"
    directory.mkdir(parents=True)
    primitives, packets = {}, []
    for noise in range(2):
        prefix = f"r{noise}b0_"
        primitives[prefix + "sample_0_counts"] = np.tile([25, 25, 25, 25], (2, 1))
        primitives[prefix + "sample_1_counts"] = np.tile(
            [26 + 2 * noise, 25, 25, 24 - 2 * noise], (2, 1)
        )
        packets.append(
            {
                "method": "O_IND",
                "n": 100,
                "bank": 0,
                "noise_replica": noise,
                "array_prefix": prefix,
                "policy_fingerprints": [["b", "u"], ["b", "b"]],
                "prompt_ids": ["0", "1"],
                "sample_names": ["b", "u"],
                "sample_layout": [
                    {"name": "b", "fields": ["counts"]},
                    {"name": "u", "fields": ["counts"]},
                ],
            }
        )
    np.savez_compressed(directory / "packet_arrays.npz", **encode_arrays(primitives))
    meta = {
        "method": "O_IND",
        "n": 100,
        "banks": [0],
        "replicas": 2,
        "packets": packets,
        "packet_arrays_sha256": _sha(directory / "packet_arrays.npz"),
    }
    with gzip.open(directory / "packet_metadata.json.gz", "wt") as f:
        json.dump(meta, f)
    _json(
        directory / "packets.json", {"metadata_sha256": _sha(directory / "packet_metadata.json.gz")}
    )
    oracle = run / "N1/oracle/seed101_X_BASE/a0"
    oracle.mkdir(parents=True)
    truth = np.zeros((2, 2, 4))
    truth[0, :, 0] = 0.01
    truth[0, :, 3] = -0.01
    h = np.array(
        [
            [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
            [-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
            [0, -2 / np.sqrt(6), 1 / np.sqrt(12)],
            [0, 0, -3 / np.sqrt(12)],
        ]
    )
    b = np.array([0.25, 0.25, 0.25, 0.25])
    u = np.array([0.26, 0.25, 0.25, 0.24])
    raw_cov = (np.diag(b) - np.outer(b, b) + np.diag(u) - np.outer(u, u)) / 100
    covariance = np.zeros((2, 6, 6))
    covariance[:, :3, :3] = h.T @ raw_cov @ h
    np.savez_compressed(
        oracle / "oracle_audit.npz", O_IND_n100_b0_truth=truth, O_IND_n100_b0_covariance=covariance
    )
    _json(oracle / "covariance_audit.json", {"oracle_sha256": _sha(oracle / "oracle_audit.npz")})
    comparison = []
    for target_index, target in enumerate(["joint_1_minus_joint_0", "no_x_off_1_minus_joint_0"]):
        factor = int(target_index == 0)
        for view in ["raw_event", "helmert"]:
            comparison.append(
                {
                    "trajectory_id": "seed101_X_BASE",
                    "seed": 101,
                    "arm": "X_BASE",
                    "anchor": 8,
                    "bank": 0,
                    "method": "O_IND",
                    "n": 100,
                    "target": target,
                    "view": view,
                    "replicas": 2,
                    "bias": 0.0,
                    "rms_bias": np.sqrt(0.00005) * factor,
                    "rmse": 0.01 * factor,
                    "empirical_variance": 0.0001 * factor,
                    "pX_error_ss": 0.0008 * factor,
                    "pX_truth_ss": 0.0004 * factor,
                    "v_error_ss": 0.0008 * factor,
                    "v_truth_ss": 0.0004 * factor,
                    "mass_residual_rms": 0.0,
                    "theoretical_variance": 0.0037495 * factor,
                    "empirical_to_theoretical_variance": 0.0001 / 0.0037495 if factor else None,
                    "packet_sha256": meta["packet_arrays_sha256"],
                }
            )
    for path in [oracle / "comparison.csv", run / "N1/OBSERVATION_COMPARISON.csv"]:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
    good = run_recompute(run, parent, tmp_path / "good")
    assert good["status"] == "PASS"
    assert good["checks"]["N1"]["matched_rows"] == 4
    truth[0, :, 0] += 0.01
    np.savez_compressed(
        oracle / "oracle_audit.npz", O_IND_n100_b0_truth=truth, O_IND_n100_b0_covariance=covariance
    )
    _json(oracle / "covariance_audit.json", {"oracle_sha256": _sha(oracle / "oracle_audit.npz")})
    bad = run_recompute(run, parent, tmp_path / "bad")
    assert bad["status"] == "FAIL"
    assert any("parent endpoints" in error["detail"] for error in bad["errors"])
