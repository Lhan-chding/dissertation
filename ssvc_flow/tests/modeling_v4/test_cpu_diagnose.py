import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v4 import cpu_diagnose as cpu


def _config():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/modeling_v4/protocol.json").read_text())
    config["cpu_fixture"] = {
        "fit_banks": [2, 3],
        "heldout_banks": 2,
        "probe_count": 2,
        "fisher_draws": 8,
        "measurement_draws": 8,
        "measurement_repeats": 2,
        "observation_n": [8],
    }
    return config


@pytest.fixture
def collection(tmp_path):
    from src.modeling_qualification.toy import ToyDataset
    from src.modeling_v3 import cpu_campaign as v3

    config = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/modeling_v3/protocol.json").read_text()
    )
    rng = np.random.default_rng(42)
    metadata = [{"prompt_id": f"train{i}", "group": i // 12} for i in range(72)]
    probes = [{"prompt_id": f"probe{i}", "group": i // 12} for i in range(72)]
    categories = np.tile(np.arange(16) % 4, (72, 1)).astype(np.int8)
    data = ToyDataset(
        rng.normal(size=(72, 16, 44)),
        categories,
        rng.normal(size=(72, 16, 44)),
        categories.copy(),
        metadata,
        probes,
        {},
    )
    root = tmp_path / "collection"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    v3._npz(
        dataset / "dataset.npz",
        train_features=data.train_features,
        train_categories=categories,
        probe_features=data.probe_features,
        probe_categories=categories,
    )
    v3._json(dataset / "train_metadata.json", metadata)
    v3._json(dataset / "probe_metadata.json", probes)
    v3._finish(dataset, {}, {"fixture": True})
    spec = v3.trajectory_specs(config, pilot=True)[0]
    name = f"seed{spec['seed']}_{spec['arm']}_init{spec['initialization_seed']}"
    unit = root / "trajectories" / name
    unit.mkdir(parents=True)
    row = v3._collect_trajectory(data, spec, config, unit, pilot=True)
    v3._finish(unit, {}, row)
    v3._finish(
        root,
        {},
        {
            "role": "timing_pilot",
            "pilot": True,
            "trajectories": [{"id": name, "path": f"trajectories/{name}", **row}],
        },
    )
    return root


def test_server_gate_precedes_any_reuse_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="server CPU"):
        cpu.run_cpu_diagnose(_config(), {"collection_root": "/missing"}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_fixture_cannot_be_used_for_a_full_size_local_campaign(tmp_path):
    config = _config()
    config["cpu_fixture"]["fit_banks"] = [96]
    with pytest.raises(ValueError, match="fixture"):
        cpu.run_cpu_diagnose(
            config, {"collection_root": "/missing"}, tmp_path / "out", fixture=True
        )


def test_reuse_pin_and_output_overlap_rejected_before_work(collection, tmp_path):
    with pytest.raises(ValueError, match=r"SHA|hash"):
        cpu.run_cpu_diagnose(
            _config(),
            {"collection_root": str(collection), "collection_complete_sha256": "0" * 64},
            tmp_path / "out",
            fixture=True,
        )
    with pytest.raises(ValueError, match="overlap"):
        cpu.run_cpu_diagnose(
            _config(), {"collection_root": str(collection)}, collection / "overwrite", fixture=True
        )


def test_actual_saved_adam_extension_only_missing_banks_and_resume(collection, tmp_path):
    import pyarrow.parquet as pq

    config = _config()
    reuse = {"collection_root": str(collection)}
    original = {
        str(p.relative_to(collection)): cpu.sha256_file(p)
        for p in collection.rglob("*")
        if p.is_file()
    }
    out = tmp_path / "diagnosis"
    result = cpu.run_cpu_diagnose(config, reuse, out, fixture=True)
    assert result["fixture"] is True
    assert result["new_source_training_steps"] == 0
    assert result["origins_completed"] == 1
    assert result["new_candidate_optimizer_steps"] == 3
    assert result["restoration_checks"] == 3
    assert result["A1"]["status"] == "MISSING_INPUTS"
    assert (out / "CPU_FAILURE_DECOMPOSITION.md").is_file()
    table = pq.read_table(out / "QUERY_RESULTS.parquet")
    assert table.num_rows > 0
    assert set(
        [
            "prediction",
            "reference",
            "origin_jvp",
            "base_jvp",
            "midpoint_jvp",
            "nonlinearity",
            "omitted_semantic",
            "fit_error",
            "cross_terms",
        ]
    ).issubset(table.column_names)
    rows = table.to_pylist()
    assert all(
        r["reference_kind"] == "EXACT" and r["reference_estimator"] == "EXACT_FINITE_ACTION"
        for r in rows
    )
    for r in rows:
        assert r["k"] == r["model_k"] and r["r"] == r["model_r"]
        assert r["geometry_k"] >= 0
        if r["rank_label"] == "FULL":
            assert r["model_k"] == r["model_r"]
        if r["prediction_status"] == "PREDICTED":
            np.testing.assert_allclose(
                np.asarray(r["nonlinearity"]) + r["omitted_semantic"] + r["fit_error"],
                np.asarray(r["reference"]) - r["prediction"],
                atol=2e-12,
            )
    assert original == {
        str(p.relative_to(collection)): cpu.sha256_file(p)
        for p in collection.rglob("*")
        if p.is_file()
    }
    assert cpu.run_cpu_diagnose(config, reuse, out, fixture=True, resume=True) == result
    with pytest.raises(FileExistsError):
        cpu.run_cpu_diagnose(config, reuse, out, fixture=True)
    changed = copy.deepcopy(config)
    changed["models"]["ridge_alpha"][0] *= 2
    with pytest.raises(ValueError, match="binding"):
        cpu.run_cpu_diagnose(changed, reuse, out, fixture=True, resume=True)


def test_a1_uses_only_first_frozen_repeats_and_never_relabels_alias(tmp_path):
    from src.modeling_v3 import cpu_campaign as v3

    root = tmp_path / "q1"
    unit = root / "units" / "pair"
    for repeat in range(3):
        rep = unit / f"repeat{repeat:03d}"
        rep.mkdir(parents=True)
        truth = np.array([[0.01, -0.01, 0.0, 0.0]])
        values = {
            f"estimate_{method}": truth + repeat * 0.001
            for method in ("RAW4", "PRESERVE_XI", "CROSSFIT_COV_ZERO_SUM")
        }
        v3._npz(rep / "statistics.npz", truth=truth, **values)
        v3._json(
            rep / "sampling_identity.json",
            {"repeat": repeat, "seed": repeat + 7, "source_policy_sha256": "a" * 64},
        )
        v3._finish(rep, {}, {"repeat": repeat})
    row = {
        "id": "pair",
        "n": 8,
        "proposal": "origin",
        "case": "IDENTITY_HASH_SELECTED",
        "seed": 1999001,
        "anchor": 8,
        "bank": 0,
        "observed_case_inventory": {"identical_policy": False},
    }
    v3._finish(unit, {}, row)
    v3._finish(root, {}, {"units": [row]})
    result = cpu.recheck_observation_records(
        root, tmp_path / "a1", n_grid=[8], repeats=2, fixture=True
    )
    assert result["status"] == "FIXTURE_REUSED"
    assert result["repeat_records_read"] == 2
    with np.load(tmp_path / "a1" / "OBSERVATION_RECHECK.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["errors"][0, :, 0], [[0.0] * 4, [0.001] * 4])


def test_focused_gls_uses_the_same_absolute_parameter_penalty():
    rng = np.random.default_rng(9)
    e = rng.normal(size=(4, 3))
    y = rng.normal(size=(4, 2, 4))
    queries = rng.normal(size=(2, 3))
    blocks = np.stack([np.eye(16)] * 2)
    result = cpu.fit_focused_gls(e, y, queries, blocks, ridge=0.17)
    beta = np.linalg.solve(e.T @ e + 0.17 * np.eye(3), e.T @ y.reshape(4, -1))
    np.testing.assert_allclose(result["prediction"], (queries @ beta).reshape(2, 2, 4), atol=1e-10)
    assert result["metadata"]["ridge_lambda"] == 0.17
    assert result["metadata"]["covariance_scope"] == "FULL_CROSS_CALIBRATION_EVENT_BLOCK"
    bad = blocks.copy()
    bad[0, 0, 0] = -1
    with pytest.raises(ValueError, match="positive semidefinite"):
        cpu.fit_focused_gls(e, y, queries, bad, ridge=0.17)


def test_resume_recomputes_unfinished_origin_without_overwriting_evidence(
    collection, tmp_path, monkeypatch
):
    config = _config()
    out = tmp_path / "interrupted"
    original = cpu._write_parquet
    called = False

    def interrupt_once(path, records):
        nonlocal called
        if not called:
            called = True
            raise RuntimeError("fixture interruption before parquet publish")
        return original(path, records)

    monkeypatch.setattr(cpu, "_write_parquet", interrupt_once)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        cpu.run_cpu_diagnose(config, {"collection_root": str(collection)}, out, fixture=True)
    saved = {str(p.relative_to(out)): cpu.sha256_file(p) for p in out.rglob("*") if p.is_file()}
    result = cpu.run_cpu_diagnose(
        config, {"collection_root": str(collection)}, out, fixture=True, resume=True
    )
    assert result["origins_completed"] == 1
    assert all(cpu.sha256_file(out / p) == digest for p, digest in saved.items())
    (out / "unregistered.txt").write_text("altered inventory")
    with pytest.raises(ValueError, match="inventory"):
        cpu.run_cpu_diagnose(
            config, {"collection_root": str(collection)}, out, fixture=True, resume=True
        )


def test_finite_measurement_packet_preserves_primary_targets_and_joint_covariance():
    origin = np.full((1, 16), 1 / 16)
    probabilities = np.tile(origin, (2, 3, 1, 1))
    probabilities[0, 1, 0, 0] += 0.01
    probabilities[0, 1, 0, 1] -= 0.01
    probabilities[0, 2, 0, 2] += 0.02
    probabilities[0, 2, 0, 3] -= 0.02
    pool = {"origin_action_p": origin, "heldout_action_p": probabilities}
    packet = cpu._finite_packet(
        pool,
        np.array([np.arange(16) % 4]),
        np.arange(2),
        8,
        "tiny",
        role="heldout",
        all_targets=True,
    )
    np.testing.assert_array_equal(packet["raw"][:, :, [0, 3]], packet["preserve_xi"][:, :, [0, 3]])
    np.testing.assert_allclose(packet["preserve_xi"].sum(-1), 0, atol=1e-15)
    np.testing.assert_allclose(packet["crossfit"].sum(-1), 0, atol=1e-15)
    np.testing.assert_array_equal(packet["raw"][3:], 0)
    np.testing.assert_array_equal(packet["crossfit"][3:], 0)
    assert packet["finite_covariance"].shape == (1, 24, 24)
    np.testing.assert_allclose(packet["raw"][2], packet["raw"][0] - packet["raw"][1])
    assert np.isfinite(packet["crossfit_fold_coefficients"]).all()


def test_resume_after_consolidation_preserves_published_parquet(collection, tmp_path, monkeypatch):
    config, out = _config(), tmp_path / "merged_interruption"
    original = cpu._evaluate_parquet

    def fail(path):
        raise RuntimeError("interrupted after consolidation")

    monkeypatch.setattr(cpu, "_evaluate_parquet", fail)
    with pytest.raises(RuntimeError, match="after consolidation"):
        cpu.run_cpu_diagnose(config, {"collection_root": str(collection)}, out, fixture=True)
    digest = cpu.sha256_file(out / "QUERY_RESULTS.parquet")
    monkeypatch.setattr(cpu, "_evaluate_parquet", original)
    result = cpu.run_cpu_diagnose(
        config, {"collection_root": str(collection)}, out, fixture=True, resume=True
    )
    assert cpu.sha256_file(out / "QUERY_RESULTS.parquet") == digest
    assert result["origins_completed"] == 1
    summary = json.loads((out / "SUMMARY.json").read_text())
    assert summary
    assert (out / "BOOTSTRAP.json").is_file()


def test_full_gls_keeps_weak_directions_under_absolute_parameter_penalty():
    e = np.diag([1.0, 1e-11])
    y = np.zeros((2, 1, 4))
    y[1, 0, 0] = 1.0
    covariance = np.eye(8)[None] * 1e-16
    result = cpu.fit_focused_gls(e, y, e, covariance, ridge=1e-8)
    expected = (1e-22 / 1e-16) / (1e-22 / 1e-16 + 1e-8)
    np.testing.assert_allclose(result["prediction"][1, 0, 0], expected, atol=1e-13)
    assert result["Q"].shape == (2, 2)
    assert result["metadata"]["positive_modes_truncated"] is False
    assert result["metadata"]["rank_label"] == "FULL"


def test_model_and_euclidean_geometry_rank_labels_are_never_mixed():
    from src.modeling_v3.coverage import coverage_diagnostics
    from src.modeling_v4.full_kernels import fit_full_response

    e = np.diag([1.0, 1e-6])
    y = np.zeros((2, 1, 4))
    y[1, 0, 0] = 1.0
    model = fit_full_response(e, y, alpha=1e-8)
    geometry = coverage_diagnostics(e, e)
    assert geometry["k"] == 2 and model["k"] == 1
    fields = cpu._rank_fields(geometry["k"], model["metadata"], model["r"])
    assert fields["geometry_k"] == 2
    assert fields["model_k"] == fields["model_r"] == 1
    assert fields["k"] == fields["r"] == 1
    assert fields["rank_label"] == "FULL"
    assert model["predict"](e)[1, 0, 0] > 0
    baseline = cpu._rank_fields(2, {"information": "STATISTICAL_BASELINE"}, 0)
    assert baseline["k"] is None and baseline["r"] is None
    assert baseline["rank_label"] == "NOT_APPLICABLE_BASELINE"
