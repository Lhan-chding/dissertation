import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v3 import cpu_campaign as campaign


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "configs/modeling_v3/protocol.json").read_text())


def _fixture_dataset():
    """Small deterministic engineering fixture, never scientific evidence."""
    from src.modeling_qualification.toy import ToyDataset

    rng = np.random.default_rng(912)
    metadata = [{"prompt_id": f"training-{i}", "group": i // 12} for i in range(72)]
    probe = [{"prompt_id": f"control-{i}", "group": i // 12} for i in range(72)]
    categories = np.tile(np.array([0, 1, *([2] * 12), 3, 3], dtype=np.int8), (72, 1))
    return ToyDataset(
        rng.normal(size=(72, 16, 44)),
        categories,
        rng.normal(size=(72, 16, 44)),
        categories.copy(),
        metadata,
        probe,
        {},
    )


def test_frozen_cpu_matrix_and_disjoint_partition(config):
    specs = campaign.trajectory_specs(config, "interval_calibration")
    specs += campaign.trajectory_specs(config, "locked_test")
    assert len(specs) == 100
    assert len({(r["seed"], r["arm"], r["initialization_seed"]) for r in specs}) == 100
    assert len(campaign.trajectory_specs(config, "orthogonal_generalization")) == 40
    assert campaign.trajectory_specs(config, pilot=True)[0]["seed"] == 1999001
    data = _fixture_dataset()
    partition = campaign.partition_training_prompts(data.train_metadata)
    assert len(partition["calibration"]) == 48
    assert len(partition["heldout"]) == 24
    assert set(partition["calibration"]).isdisjoint(partition["heldout"])
    for role in partition:
        for bank in range(36):
            selected = campaign.bank_prompt_indices(
                data.train_metadata, partition[role], bank, np.random.default_rng(bank)
            )
            assert len(set(selected)) == 4
            assert set(selected).issubset(partition[role])


def test_actual_adam_forks_preserve_full_origin_and_safe_replay(config, tmp_path):
    import torch

    from src.modeling_qualification.toy import (
        flatten_parameters,
        make_model,
        perform_step,
    )

    dataset = _fixture_dataset()
    spec = campaign.trajectory_specs(config, pilot=True)[0]
    summary = campaign._collect_trajectory(dataset, spec, config, tmp_path, pilot=True)
    assert summary["source_updates"] == 8
    assert summary["fork_updates"] == summary["full_origin_restoration_checks"] == 12
    with np.load(tmp_path / "observations.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["d"], np.diff(data["theta"], axis=0))
        np.testing.assert_array_equal(data["branch_d"], data["branch_theta"] - data["theta"][8])
        np.testing.assert_array_equal(data["adam_step"], np.arange(9))
        expected_theta = data["branch_theta"][0, 0, 1].copy()
        indices = data["branch_prompt_indices"][0, 0].copy()
        bank = {
            "features": dataset.train_features[indices],
            "prompt_indices": indices,
            "actions": data["branch_actions"][0, 0].copy(),
            "categories": data["branch_categories"][0, 0].copy(),
            "old_logp": data["branch_old_logp"][0, 0].copy(),
        }
    trees = json.loads((tmp_path / "restoration_state_trees.json").read_text())
    with np.load(tmp_path / "restoration_state_tensors.npz", allow_pickle=False) as tensors:
        state = campaign.unpack_state(trees["source_step8"], tensors)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    rng = np.random.default_rng(1)
    campaign._full_restore(model, optimizer, rng, state)
    perform_step(model, optimizer, bank, lam=1, policy="joint", operation_index=1)
    np.testing.assert_array_equal(flatten_parameters(model), expected_theta)


def test_immutable_resume_rejects_source_and_payload_changes(tmp_path):
    root = tmp_path / "unit"
    binding = {"source": "original"}
    assert campaign._prepare(root, binding, False) is None
    campaign._npz(root / "raw.npz", x=np.arange(4))
    campaign._finish(root, binding, {"status": "COLLECTED"})
    assert campaign._prepare(root, binding, True)["status"] == "COLLECTED"
    with pytest.raises(ValueError, match="binding"):
        campaign._prepare(root, {"source": "changed"}, True)
    (root / "raw.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        campaign._verify_complete(root)


def test_nested_manifest_tampering_and_unregistered_files_rejected(tmp_path):
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    campaign._npz(child / "raw.npz", a=np.ones(4))
    campaign._finish(child, {}, {"value": 1})
    campaign._finish(root, {}, {"value": 2})
    assert "child/COMPLETE.json" in campaign._verify_complete(root)["files"]
    (child / "COMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        campaign._verify_complete(root)
    other = tmp_path / "other"
    other.mkdir()
    campaign._finish(other, {}, {})
    (other / "unregistered.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        campaign._verify_complete(other)


def test_v3_seed_inventory_uses_directory_identity_without_opening_raw(tmp_path):
    unit = tmp_path / "trajectories" / "seed22001_X_BASE_init7001"
    unit.mkdir(parents=True)
    (unit / "observations.npz").write_bytes(b"opaque originals never opened by inventory")
    campaign._json(unit / "identity.json", {"seed": 22001, "steps": 64})
    rows = campaign.scan_cpu_seed_inventory([tmp_path], {22001, 22002})
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {22001}


def test_large_campaign_refuses_local_mac_cpu(monkeypatch):
    monkeypatch.setattr(campaign.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="server CPU"):
        campaign._require_server(False)
    campaign._require_server(True)


def test_actual_observation_sampler_and_covariance(config):
    rng = np.random.default_rng(21)
    origin = rng.dirichlet(np.ones(16), size=6)
    baseline = rng.dirichlet(np.ones(16), size=6)
    candidate = rng.dirichlet(np.ones(16), size=6)
    categories = np.tile(np.arange(16) % 4, (6, 1))
    packet = campaign.draw_packet(origin, baseline, candidate, categories, 64, 777)
    same = campaign.draw_packet(origin, baseline, candidate, categories, 64, 777)
    np.testing.assert_array_equal(packet["z_raw"], same["z_raw"])
    outputs, adjusted, _ = campaign.estimate_packet(
        packet,
        config["observation"]["methods"],
        config,
        known_covariance=campaign._exact_covariance(
            origin, baseline, candidate, categories, "origin"
        ),
    )
    np.testing.assert_array_equal(
        outputs["estimate_RAW4"][:, (0, 3)], outputs["estimate_PRESERVE_XI"][:, (0, 3)]
    )
    assert len(adjusted["PILOT_COV_ZERO_SUM"]) == 48
    assert len(adjusted["RAW4"]) == 64
    assert not np.isfinite(outputs["b_CROSSFIT_COV_ZERO_SUM"]).any()
    np.testing.assert_allclose(
        outputs["raw_valid_event_sum"] - outputs["raw_delta_v_minus_I"],
        outputs["raw_mass_residual"],
        atol=1e-14,
    )
    # All bank contrasts share draw identity; cross-bank covariance must survive.
    contributions = np.stack([adjusted["RAW4"], adjusted["RAW4"]])[None]
    covariance = campaign._joint_covariance(contributions)
    np.testing.assert_allclose(covariance[:, :4, :4], covariance[:, :4, 4:])


def test_shrinkage_grid_uses_one_packet_and_prespecified_coefficients(config):
    rng = np.random.default_rng(215)
    origin = rng.dirichlet(np.ones(16), size=2)
    candidate = rng.dirichlet(np.ones(16), size=2)
    categories = np.tile(np.arange(16) % 4, (2, 1))
    packet = campaign.draw_packet(origin, origin, candidate, categories, 64, 218)
    before = packet["z_raw"].copy()
    methods = [
        f"PILOT_SHRINK_ZERO_SUM_ETA_{eta:g}" for eta in config["observation"]["cov_shrink_grid"]
    ]
    outputs, adjusted, diagnostics = campaign.estimate_packet(packet, methods, config)
    for method, eta in zip(methods, config["observation"]["cov_shrink_grid"], strict=True):
        assert all(row["shrink"] == eta for row in diagnostics[method])
        assert adjusted[method].shape == (48, 2, 4)
        assert outputs[f"estimate_{method}"].shape == (2, 4)
    np.testing.assert_array_equal(packet["z_raw"], before)


def test_alias_independent_counts_use_no_new_draws():
    probabilities = np.full((2, 16), 1 / 16)
    categories = np.tile(np.arange(16) % 4, (2, 1))
    estimate, counts = campaign._independent_counts(
        probabilities, probabilities.copy(), categories, 64, 991
    )
    np.testing.assert_array_equal(estimate, 0)
    assert counts.sum() == 0


def test_random_response_basis_matches_estimators_and_nests_rank():
    from src.modeling_v3.response_models import fit_response_model

    design = {
        "m": 8,
        "selector": "STRATIFIED_RANDOM",
        "design_seed": 8,
        "estimator": "PRESERVE_XI",
        "rank_cap": 1,
        "n": 64,
        "alpha": 1e-5,
    }
    changed = {
        **design,
        "estimator": "PILOT_SHRINK_ZERO_SUM",
        "rank_cap": 4,
        "n": 4096,
        "alpha": 0.01,
    }
    seed = campaign.random_direction_seed("seed101_X_BASE", 8, design)
    assert seed == campaign.random_direction_seed("seed101_X_BASE", 8, changed)
    rng = np.random.default_rng(12)
    updates, responses = rng.normal(size=(8, 20)), rng.normal(size=(8, 2, 4))
    rank1 = fit_response_model(updates, responses, "RANDOM_Q", 1, random_seed=seed)
    rank4 = fit_response_model(updates, responses * 2, "RANDOM_Q", 4, alpha=0.01, random_seed=seed)
    # BLAS uses vector/matrix kernels for different r; only roundoff may differ.
    np.testing.assert_allclose(rank1["directions"], rank4["directions"][:, :1], atol=2e-15, rtol=0)


def test_crossfit_gls_rejected_before_measurement(config):
    with pytest.raises(ValueError, match="full-refit covariance"):
        campaign.development_designs(
            config,
            selected={"observation_methods": ["CROSSFIT_COV_ZERO_SUM"], "models": ["FULL_GLS"]},
        )


def test_analytic_cases_are_explicitly_separate_from_source_data(config, tmp_path):
    result = campaign.analytic_observation_witnesses(config, tmp_path / "witnesses")
    assert len(result["cases"]) == 7
    assert result["scientific_confirmation"] is False
    assert all(row["source_training_steps"] == 0 for row in result["cases"])
    assert all(
        record["exact_expectation_residual"] < 1e-14
        for row in result["cases"]
        for record in row["records"]
    )


def test_pilot_coverage_freezes_predictions_before_heldout(config, tmp_path, monkeypatch):
    dataset = _fixture_dataset()
    root = tmp_path / "collection"
    data_root = root / "dataset"
    data_root.mkdir(parents=True)
    campaign._npz(
        data_root / "dataset.npz",
        probe_categories=dataset.probe_categories,
        probe_features=dataset.probe_features,
    )
    campaign._json(data_root / "train_metadata.json", dataset.train_metadata)
    campaign._json(data_root / "probe_metadata.json", dataset.probe_metadata)
    spec = campaign.trajectory_specs(config, pilot=True)[0]
    unit = root / "trajectories" / "pilot"
    unit.mkdir(parents=True)
    row = campaign._collect_trajectory(dataset, spec, config, unit, pilot=True)
    campaign._finish(unit, {}, row)
    campaign._finish(
        root,
        {},
        {
            "role": "development",
            "trajectories": [{**row, "id": "pilot", "path": "trajectories/pilot"}],
        },
    )
    out = tmp_path / "coverage"
    original_load = np.load
    access = []

    def checked_load(file, *args, **kwargs):
        if Path(file).name == "heldout_action_p.npz":
            locks = list(out.rglob("PREDICTION_LOCK.json"))
            assert locks
            access.append(str(file))
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(campaign.np, "load", checked_load)
    original_designs = campaign.development_designs

    def designs_with_zero(*args, **kwargs):
        designs = original_designs(*args, **kwargs)
        return [*designs, {**designs[0], "model": "ZERO"}]

    monkeypatch.setattr(campaign, "development_designs", designs_with_zero)
    summary = campaign.coverage_study(config, root, out, pilot=True)
    assert summary["fit_count"] == 3
    assert summary["scientific_status"] == "PILOT_NOT_CONFIRMATION"
    assert len(access) == 1
    decomposition = json.loads(next(out.rglob("DECOMPOSITION_SUMMARY.json")).read_text())
    assert decomposition["fit_count"] == 3
    assert all(abs(row["reconstruction_residual"]) < 1e-12 for row in decomposition["diagnostics"])
    with np.load(
        next(out.rglob("ORIGIN_JACOBIAN_AND_BASELINES.npz")), allow_pickle=False
    ) as arrays:
        assert arrays["jacobian"].shape == (72, 4, 737)
        assert arrays["all_bank_baseline_d0"].shape == (4, 737)
    zero_row = next(row for row in summary["results"] if row["design"]["model"] == "ZERO")
    with np.load(
        out / "units" / zero_row["origin_id"] / zero_row["predictions_relative"], allow_pickle=False
    ) as arrays:
        assert set(arrays["classifications"]) == {"STATISTICAL_BASELINE_ONLY"}
        np.testing.assert_array_equal(arrays["prediction"], 0)
    campaign._verify_complete(out)


def test_fixture_calibration_cannot_authorize_real_collection(config):
    with pytest.raises(ValueError, match="fixture"):
        campaign.verify_cpu_calibration_receipt(
            config,
            {"selection_hash": "frozen"},
            {
                "fixture": True,
                "status": "TEST_FIXTURE_NOT_AUTHORIZATION",
                "calibration_seeds": config["cpu"]["interval_calibration_seeds"],
            },
        )
    with pytest.raises(PermissionError, match="calibration"):
        campaign.verify_cpu_calibration_receipt(config, {"selection_hash": "frozen"}, None)


def test_historical_q1_keeps_original_bank_and_provenance(config, tmp_path):
    data = _fixture_dataset()
    root = tmp_path / "historical"
    dataset_root = root / "dataset"
    dataset_root.mkdir(parents=True)
    campaign._npz(dataset_root / "dataset.npz", probe_categories=data.probe_categories)
    campaign._json(dataset_root / "probe_metadata.json", data.probe_metadata)
    unit = root / "trajectories/historical_seed101_X_BASE_a8_b17"
    unit.mkdir(parents=True)
    probabilities = np.full((1, 72, 16), 1 / 16)
    campaign._npz(unit / "source_probabilities.npz", origin_action_p=probabilities)
    campaign._npz(
        unit / "calibration_action_p.npz",
        action_p=np.tile(probabilities[:, None, None], (1, 1, 3, 1, 1)),
    )
    campaign._json(
        unit / "identity.json",
        {
            "anchors": [8],
            "bank_counts": {"calibration": 1},
            "bank_original_ids": [[17]],
            "original_trajectory_id": "seed101_X_BASE",
            "historical_reused_fixed_policies": True,
        },
    )
    campaign._finish(unit, {}, {})
    campaign._finish(
        root,
        {},
        {
            "role": "development",
            "trajectories": [
                {"id": unit.name, "path": str(unit.relative_to(root)), "seed": 101, "arm": "X_BASE"}
            ],
        },
    )
    result = campaign.observe_toy(config, root, tmp_path / "observation", pilot=True)
    assert all(row["bank"] == 17 and row["local_bank_index"] == 0 for row in result["units"])
    assert all(row["historical_reused_fixed_policies"] for row in result["units"])
    assert all(
        row["observed_case_inventory"]["source"] == "REUSED_HISTORICAL_FIXED_POLICIES_NEW_SCORING"
        for row in result["units"]
    )
    receipt = json.loads(
        next((tmp_path / "observation").glob("units/*/repeat000/COMPLETE.json")).read_text()
    )
    times = receipt["summary"]["timings"]
    blocks = (
        "packet_sampling_cached_scoring_label_seconds",
        "estimator_seconds",
        "independent_count_seconds",
        "sufficient_statistics_seconds",
        "artifact_io_seconds",
    )
    assert all(np.isfinite(times[key]) and times[key] >= 0 for key in blocks)
    assert (
        sum(times[key] for key in blocks) <= times["repeat_wall_seconds_before_completion_manifest"]
    )
    assert receipt["summary"]["timing_scope"]["sampling_scoring_label_split_measured"] is False
    assert receipt["summary"]["actual_additional_model_forward_calls"] == 0


def test_cpu_dependency_binding_excludes_unrelated_gpu_modules():
    sources = campaign._sources()
    assert "src/modeling_v3/cpu_results.py" in sources
    assert "src/modeling_v3/observation_geometry.py" in sources
    assert "src/prompts.py" in sources
    assert "src/modeling_v3/vlm_campaign.py" not in sources
    assert "src/modeling_v3/cli.py" not in sources
