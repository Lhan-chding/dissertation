import numpy as np

from src.modeling_qualification.evaluation import (
    choose_smallest_rank,
    cluster_bootstrap,
    freeze_predictions,
    score_predictions,
)


def test_level_and_response_error_are_separate():
    p = np.tile([0.2, 0.1, 0.3, 0.4], (2, 4, 1))
    truth0 = p[0]
    anchor = truth0 + np.array([0.05, -0.05, 0.0, 0.0])
    result = score_predictions(
        np.broadcast_to(anchor, p.shape), np.zeros_like(p), p, truth0, np.array([0, 0, 1, 1])
    )
    assert result["level_mae"] > 0
    assert result["delta_mae"] == 0
    assert result["response_nrmse"] is None
    assert result["response_scale_status"] == "UNINFORMATIVE_RESPONSE_SCALE"


def test_selection_allows_failure_and_favors_smallest_rank():
    assert choose_smallest_rank([], {})["status"] == "NO_ACCEPTABLE_R"
    row = {
        "rank_cap": 2,
        "group_delta_pX_q95_max": 0.001,
        "group_delta_v_q95_max": 0.001,
        "delta_mae": 0.001,
        "response_nrmse": 0.1,
        "raw_invalid_fraction": 0.0,
        "alpha": 0.0,
    }
    assert choose_smallest_rank([row, dict(row, rank_cap=1)], {})["rank_cap"] == 1


def test_cluster_bootstrap_is_seed_clustered_and_reproducible():
    a = cluster_bootstrap({1: 1.0, 2: 2.0, 3: 3.0}, {1: 0.0, 2: 0.0, 3: 0.0}, 5000, 9)
    assert a == cluster_bootstrap({1: 1.0, 2: 2.0, 3: 3.0}, {1: 0.0, 2: 0.0, 3: 0.0}, 5000, 9)
    assert a["cluster_count"] == 3 and a["paired_mean"] == 2.0


def test_freeze_prediction_is_no_clobber(tmp_path):
    p = freeze_predictions(tmp_path / "frozen", {"p": np.ones((2, 4))}, {"selection": "abc"})
    assert p["prediction_sha256"]
    import pytest

    with pytest.raises(FileExistsError):
        freeze_predictions(tmp_path / "frozen", {"p": np.zeros((2, 4))}, {})


def test_alpha_with_all_gates_beats_lower_group_error_that_fails_nrmse():
    fail = {
        "rank_cap": 1,
        "group_delta_pX_q95_max": 0.001,
        "group_delta_v_q95_max": 0.001,
        "delta_mae": 0.001,
        "response_nrmse": 1.0,
        "raw_invalid_fraction": 0.0,
        "alpha": 0.0,
    }
    good = dict(fail, alpha=0.01, response_nrmse=0.5, group_delta_pX_q95_max=0.002)
    selected = choose_smallest_rank([fail, good], {})
    assert selected["rank_cap"] == 1
    assert selected["alpha"] == 0.01


def test_aggregate_rmse_uses_squared_errors():
    from src.modeling_qualification.evaluation import aggregate_scores

    truth = np.full((1, 4, 4), 0.25)
    a = score_predictions(truth, np.zeros_like(truth), truth, truth[0], np.array([0, 0, 1, 1]))
    p = truth + np.array([0.1, -0.1, 0.0, 0.0])
    b = score_predictions(p, p - truth, truth, truth[0], np.array([0, 0, 1, 1]))
    combined = aggregate_scores([a, b])
    assert np.isclose(combined["level_rmse"], 0.05)
    assert "mean_block_prompt_abs_error_q95" in combined


def test_formal_runner_never_loads_locked_test_labels_before_freeze(tmp_path, monkeypatch):
    """Small independent fixture tests orchestration, not scientific qualification."""
    import json
    import types

    from src.modeling_qualification import evaluation as ev

    config = {
        "profiles": {"core": {"anchors": [0]}},
        "measurement": {
            "primary_noise_replicas": 1,
            "samples_per_prompt_grid": [64],
            "noise_replicas": 1,
        },
        "response": {
            "models": [
                "B0_PERSISTENCE",
                "B1_LOG_RIDGE",
                "B2_RANDOM_Q",
                "B3_UPDATE_PCA",
                "B4_RESPONSE_SVD",
                "B5_WORST_GROUP_RANK",
                "B6_FULL_Q_RIDGE",
                "O0_FULL_J_ORACLE",
                "O1_JQ_SVD_ORACLE",
            ],
            "rank_caps": [0, 1],
            "ridge_alpha_grid": [0.0, 0.01],
        },
        "model_selection": {},
        "statistics": {"bootstrap_replicates": 5000, "bootstrap_seed": 99001},
    }
    source = tmp_path / "source"
    source.mkdir()
    entries = [
        {"id": role, "seed": seed, "arm": "X_BASE", "split": role}
        for role, seed in [("development_fit", 101), ("model_selection", 201), ("locked_test", 301)]
    ]
    (source / "manifest.json").write_text(json.dumps({"profile": "core", "trajectories": entries}))
    (source / "probe_metadata.json").write_text(json.dumps([{"group": i // 2} for i in range(4)]))
    p0 = np.full((4, 4), 0.25)
    d = np.array([[0.1, 0], [0.2, 0], [0.3, 0.0]])
    change = np.array([0.1, -0.1, 0, 0])
    p1 = p0 + d[:, 0, None, None] * change
    fit = {
        "p0": p0,
        "p1": p1,
        "d": d,
        "g": d,
        "logs": np.ones((3, 11)),
        "operation_indices": np.arange(3),
        "anchor_measurement_id": "shared-anchor",
    }
    view = types.SimpleNamespace(anchor=lambda *a, **kw: fit, anchor_inputs=lambda *a, **kw: fit)
    monkeypatch.setattr(ev, "_view", lambda *args: view)
    monkeypatch.setattr(ev, "_checkpoint", lambda *args: np.zeros((1, 2)))
    monkeypatch.setattr(ev, "_oracle_j", lambda *args: np.zeros((12, 2)))
    monkeypatch.setattr(ev, "run_oracle_diagnostics", lambda *args: {"status": "TEST_STUB"})
    target = tmp_path / "result"
    calls = []

    def checked_oracle(root, entry):
        if entry["split"] == "locked_test":
            assert (target / "prediction_freeze.json").exists()
            assert (target / "selection.json").exists()
            calls.append(entry["id"])
        return types.SimpleNamespace(
            p=np.array([p0]), theta=np.zeros((1, 2)), anchor=lambda *a, **kw: {"p1": p1}
        )

    monkeypatch.setattr(ev, "_oracle", checked_oracle)
    result = ev.run_fit_evaluate(config, source, target)
    assert result["status"] == "COMPLETED"
    assert calls
    assert any(
        row["method"] == "B5_EXACT_SELECTED_DIAGNOSTIC" and row["n"] == 64
        for row in result["aggregate_metrics"]
    )


def test_shared_noise_postprocess_pairs_three_and_five_replicas_without_predicting(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from src.modeling_qualification import evaluation as ev
    from src.modeling_qualification.models import array_hash

    source, models, target = tmp_path / "input", tmp_path / "M3", tmp_path / "post"
    source.mkdir()
    models.mkdir()
    entries = [
        {"id": f"seed{seed}", "seed": seed, "arm": "X_BASE", "split": "locked_test"}
        for seed in (301, 302)
    ]
    ev.write_json(source / "manifest.json", {"trajectories": entries})
    ev.write_json(source / "probe_metadata.json", [{"group": 0}, {"group": 1}])
    selection = {
        "selected": {"method": "B5_WORST_GROUP_RANK", "rank_cap": 1},
        "input_manifest_sha256": ev.file_hash(source / "manifest.json"),
    }
    ev.write_json(models / "selection.json", selection)
    truth0 = np.full((2, 4), 0.25)
    truth = np.array([truth0 + np.array([0.01, 0.0, 0.0, -0.01])] * 3)
    monkeypatch.setattr(
        ev,
        "_oracle",
        lambda *args: SimpleNamespace(p=np.array([truth0]), anchor=lambda *a, **k: {"p1": truth}),
    )

    def forbid(*args, **kwargs):
        raise AssertionError("a postprocessor must not call a predictor")

    monkeypatch.setattr(ev, "_predict_one", forbid)
    records, rows = [], []
    for seed in (301, 302):
        for method, cap, alpha in [("B0_PERSISTENCE", 0, 0.0), ("B5_WORST_GROUP_RANK", 1, 0.01)]:
            for noise in range(5):
                delta = (
                    np.zeros_like(truth)
                    if cap == 0
                    else truth - truth0 + noise * np.array([0.001, 0.0, 0.0, -0.001])
                )
                raw = truth0 + delta
                name = f"{len(records)}.npz"
                np.savez(models / name, raw=raw, delta=delta)
                rec = {
                    "trajectory_id": f"seed{seed}",
                    "seed": seed,
                    "arm": "X_BASE",
                    "anchor": 0,
                    "anchor_index": 0,
                    "method": method,
                    "rank_cap": cap,
                    "alpha": alpha,
                    "n": 64,
                    "noise": noise,
                    "fit_budget": 8,
                    "representation": "per_prompt",
                    "input_mode": "actual_d",
                    "r": cap,
                    "k": 1,
                    "record": len(records),
                    "prediction_file": name,
                    "raw_prediction_hash": array_hash(raw),
                    "delta_prediction_hash": array_hash(delta),
                }
                records.append(rec)
                score = score_predictions(raw, delta, truth, truth0, np.array([0, 1]))
                rows.append(
                    {
                        **rec,
                        **{k: v for k, v in score.items() if k not in ("group_rows", "direction")},
                    }
                )
    ev.write_json(
        models / "prediction_freeze.json",
        {"records": records, "selection_sha256": ev.file_hash(models / "selection.json")},
    )
    ev.write_csv(models / "prediction_by_anchor.csv", rows)
    ev.write_csv(models / "aggregate_metrics.csv", [{"status": "original retained"}])
    config = {
        "measurement": {
            "primary_noise_replicas": 3,
            "noise_replicas": 5,
            "samples_per_prompt_grid": [64],
        },
        "statistics": {"bootstrap_replicates": 5000, "bootstrap_seed": 99001},
    }
    result = ev.export_shared_noise_tables(config, source, models, target)
    assert result["status"] == "PASS" and result["new_predictions"] == 0
    import csv

    with (target / "primary_aggregate_metrics.csv").open() as stream:
        primary = list(csv.DictReader(stream))
    with (target / "robust_aggregate_metrics.csv").open() as stream:
        robust = list(csv.DictReader(stream))
    assert all(row["sample_count"] == "18" and row["noise_replicas"] == "3" for row in primary)
    assert all(row["sample_count"] == "30" and row["noise_replicas"] == "5" for row in robust)
    p = next(row for row in primary if row["rank_cap"] == "1")
    r = next(row for row in robust if row["rank_cap"] == "1")
    assert np.isclose(float(p["delta_mae"]), 0.0005)
    assert np.isclose(float(r["delta_mae"]), 0.001)
    assert float(p["group_delta_pX_q95_max"]) < float(r["group_delta_pX_q95_max"])
    assert result["source_hashes_before"] == result["source_hashes_after"]


def test_four_level_vectors_keep_cancelling_components_and_triangle_bound():
    from src.modeling_qualification.evaluation import four_level_residuals

    full = np.array([[1.0, 0.0]])
    low = np.array([[0.0, 0.0]])
    fitted = np.array([[0.25, 0.0]])
    truth = np.array([[0.25, 0.0]])
    result = four_level_residuals(full, low, fitted, truth)
    assert np.allclose(result["total_residual"], 0)
    assert np.allclose(result["direction_omission"], [[-1.0, 0.0]])
    assert np.allclose(result["fit_sampling_error"], [[0.25, 0.0]])
    assert np.allclose(result["nonlinearity"], [[0.75, 0.0]])
    assert result["triangle_upper_bound"][0] == 2.0
