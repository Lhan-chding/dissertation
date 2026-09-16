import copy

import numpy as np
import pytest

from src.modeling_v4 import tracking


def test_windows_keep_endpoint_and_intermediate_max_separate_and_unknown():
    reference = np.zeros((17, 2, 4))
    prediction = np.zeros_like(reference)
    prediction[2, 0, 0] = 0.3
    prediction[4, 0, 0] = 0.02
    prediction[7, 1, 2] = np.nan
    result = tracking.evaluate_tracking_paths(
        {"fixed": prediction},
        reference,
        np.zeros_like(reference),
        np.ones_like(reference, dtype=bool),
        groups=["g0", "g1"],
        methods={"fixed": {"uses_intermediate_measurements": False}},
    )
    row4 = next(r for r in result["windows"] if r["horizon"] == 4)
    assert row4["endpoint_max_absolute_error"] == 0.02
    assert row4["window_max_absolute_error"] == 0.3
    row8 = next(r for r in result["windows"] if r["horizon"] == 8)
    assert row8["unknown_prediction_cells"] == 1
    assert row8["window_max_absolute_error"] is None
    assert row8["finite_subset_window_max_absolute_error"] == 0.3
    assert result["status"] != "COMPLETE_E"
    assert result["online_control"] is False


def test_refreshed_derivatives_only_use_current_block_start_not_future_gradient():
    # Each J_s is applied to the actual cumulative parameter update coordinate.
    products = {
        s: np.arange(17)[:, None, None] * np.ones((1, 2, 4)) * (s - 63)
        for s in (64, 68, 72, 76, 80)
    }
    result = tracking.accumulate_refreshed_derivatives(products)
    assert result[0, 0, 0] == 0
    assert result[4, 0, 0] == 4
    assert result[5, 0, 0] == 9
    assert result[16, 0, 0] == 4 * (1 + 5 + 9 + 13)
    mutated = copy.deepcopy(products)
    mutated[80][:] = 1e9
    np.testing.assert_array_equal(result, tracking.accumulate_refreshed_derivatives(mutated))
    with pytest.raises(ValueError, match="refresh"):
        tracking.accumulate_refreshed_derivatives({64: products[64]})


def test_tracking_cannot_start_without_actual_resolved_gate(tmp_path):
    with pytest.raises(ValueError, match="tracking extension"):
        tracking.run_tracking_task({}, {}, {}, {"id": "track"}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_raw_data_or_missing_nonlinear_never_claim_complete_e():
    reference = np.zeros((17, 1, 4))
    result = tracking.evaluate_tracking_paths(
        {"ZERO": reference.copy()},
        reference,
        reference.copy(),
        np.ones_like(reference, dtype=bool),
        groups=["group"],
        methods={"ZERO": {"uses_intermediate_measurements": False}},
        unavailable={"NONLINEAR": "NO_ACTUAL_FROZEN_MODEL_ARTIFACT"},
    )
    assert result["status"] == "EVALUATED_INCOMPLETE_METHODS"
    assert result["unavailable_methods"]["NONLINEAR"]
    with pytest.raises(ValueError, match="17"):
        tracking.evaluate_tracking_paths(
            {},
            reference[:16],
            reference[:16],
            np.ones_like(reference[:16], dtype=bool),
            groups=["g"],
            methods={},
        )


def test_actual_timepoint_packet_keeps_labels_and_refuses_token_substitution():
    raw, scores, policies = [], {}, {}
    for i in range(4):
        raw.append(
            {
                "sample_id": f"s{i}",
                "prompt_id": "p",
                "rng_namespace": "independent",
                "token_ids": [7],
                "input_hash": "input",
                "runtime_identity": "runtime",
                "role": "reference",
                "category": ["X", "S", "W", "I"][i],
                "generation_sequence_logp": -1.0,
            }
        )
    for step in tracking.STEPS:
        policies[step] = {"inference_fingerprint": f"policy{step}"}
        scores[step] = [
            {
                **r,
                "inference_fingerprint": f"policy{step}",
                "token_logprobs": [-1.0 + (step - 64) * 0.001],
                "sequence_logp": -1.0 + (step - 64) * 0.001,
            }
            for r in raw
        ]
    packet = tracking._trajectory_batch(raw, scores, policies)
    assert packet.contributions.shape == (4, 17, 4)
    assert packet.contrast_ids[1] == "step_065_minus_step_064"
    np.testing.assert_array_equal(packet.contributions[:, 0], 0)
    changed = copy.deepcopy(scores)
    changed[65][0]["token_ids"] = [8]
    with pytest.raises(ValueError, match="identity"):
        tracking._trajectory_batch(raw, changed, policies)


def test_refresh_full_gradient_products_are_complete_coordinate_contractions():
    data = np.arange(6 * 2 * 3, dtype=float).reshape(6, 2, 3)
    layout = {
        "a": {"start": 0, "stop": 2, "shape": (2,)},
        "b": {"start": 2, "stop": 3, "shape": (1,)},
    }
    a = np.arange(8, dtype=float).reshape(1, 4, 2)
    b = np.arange(4, dtype=float).reshape(1, 4, 1)
    gradient = {
        "parameter_order": ["a", "b"],
        "reference_labels_read": False,
        "expansion_point": "ORIGIN",
        "groups": [["family", "interface"]],
        "grouped_gradients": {"a": a, "b": b},
    }
    result = tracking._gradient_products(data, layout, gradient, 2)
    expected = (data[2:, 0] - data[2:, 1]) @ np.concatenate([a, b], axis=-1).reshape(4, 3).T
    np.testing.assert_array_equal(result[:, 0], expected)
    bad = copy.deepcopy(gradient)
    bad["reference_labels_read"] = True
    with pytest.raises(ValueError, match="independent"):
        tracking._gradient_products(data, layout, bad, 2)


def test_unresolved_references_keep_all_finite_observed_path_errors():
    reference = np.zeros((17, 1, 4))
    prediction = reference.copy()
    prediction[3, 0, 0] = 0.6
    resolved = np.ones_like(reference, dtype=bool)
    resolved[3, 0, 0] = False
    result = tracking.evaluate_tracking_paths(
        {"fixed": prediction},
        reference,
        reference.copy(),
        resolved,
        groups=["g"],
        methods={"fixed": {"uses_intermediate_measurements": False}},
    )
    row = next(v for v in result["windows"] if v["horizon"] == 4)
    assert row["window_max_absolute_error"] is None
    assert row["finite_subset_window_max_absolute_error"] == 0.6
    assert row["all_finite_observed_cells"] == 16
    assert row["finite_resolved_cells"] == 15


def test_tiny_actual_coordinate_fit_freezes_models_before_reference_and_resumes(
    tmp_path, monkeypatch
):
    import json

    import torch

    from src.modeling_v4 import gpu_collect as collect
    from src.modeling_v4 import response_fit
    from src.modeling_v4.analysis_rules import load_analysis_rules

    groups = [[f"family{i // 2}", f"interface{i % 2}"] for i in range(6)]
    probes = [
        {"prompt_id": f"p{i}", "family": groups[i // 12][0], "interface": groups[i // 12][1]}
        for i in range(72)
    ]
    names = ["m.lora_A.default.weight", "m.lora_B.default.weight"]
    states = {}

    def policy(label, value):
        state = {
            "parameters": {
                names[0]: torch.tensor([[1.0]], dtype=torch.float64),
                names[1]: torch.tensor([[value]], dtype=torch.float64),
            },
            "fp": label,
        }
        states[label] = state
        return {
            "candidate_id": label,
            "inference_fingerprint": label,
            "checkpoint": {"key": label},
            "module_scalings": {"m": 1.0},
            "parameter_layout": [{"name": n, "shape": [1, 1]} for n in names],
        }

    policies = {step: policy(str(step), (step - 64) * 0.01) for step in tracking.STEPS}
    banks = []
    for i in range(2):
        banks.append(
            {
                "bank_id": str(i),
                "policies": {
                    "joint_0": policy(f"b{i}", 0.001 * i),
                    "joint_1": policy(f"u{i}", 0.001 * i + 0.01 * (i + 1)),
                    "no_x_off_1": policy(f"n{i}", 0.001 * i + 0.015 * (i + 1)),
                },
            }
        )

    class Cache:
        def load(self, spec):
            return states[spec["checkpoint"]["key"]]

    runtime = {
        "checkpoint_cache": Cache(),
        "identity": {"model_hash": "tiny", "execution_kind": "CPU_FIXTURE"},
        "analysis_rules": load_analysis_rules(),
    }
    monkeypatch.setattr(collect, "_fingerprint", lambda state, identity: state["fp"])
    monkeypatch.setattr(collect, "_diagnostic_prompt_ids", lambda p: [])
    monkeypatch.setattr(
        response_fit, "_action_rows", lambda *a, **kw: {p["prompt_id"]: [] for p in probes}
    )
    direction = np.array([1.0, -1.0, 0.5, -0.5])

    def observed(bank, *args, **kwargs):
        b = states[bank["policies"]["joint_0"]["checkpoint"]["key"]]["parameters"][names[1]].item()
        u = states[bank["policies"]["joint_1"]["checkpoint"]["key"]]["parameters"][names[1]].item()
        n = states[bank["policies"]["no_x_off_1"]["checkpoint"]["key"]]["parameters"][
            names[1]
        ].item()
        values = np.array([u - b, n - b, u - n])[:, None] * direction
        return {"PRESERVE_XI": np.tile(values, (72, 1, 1))}

    monkeypatch.setattr(response_fit, "_observations", observed)
    measured_steps = []

    def derivative(runtime, forks, probes, response, *, out, **kwargs):
        step = int(Path(out).name[-3:])
        measured_steps.append(step)
        return {"step": step}

    from pathlib import Path

    monkeypatch.setattr(collect, "_score_derivative", derivative)

    def gradient(receipt, *args):
        step = receipt["step"]
        return {
            "parameter_order": names,
            "groups": groups,
            "prompt_subset": [],
            "expansion_point": "ORIGIN",
            "reference_labels_read": False,
            "grouped_gradients": {
                names[0]: torch.zeros((6, 4, 1, 1), dtype=torch.float64),
                names[1]: torch.tensor(np.tile(direction[None, :, None, None], (6, 1, 1, 1)))
                * (step - 63),
            },
            "prompt_gradients": {n: torch.zeros((0, 4, 1, 1), dtype=torch.float64) for n in names},
        }

    monkeypatch.setattr(response_fit, "_read_gradient", gradient)
    monkeypatch.setattr(collect, "collect_actions", lambda *a, **kw: {"fixture": True})
    target = tmp_path / "E"
    target.mkdir()
    (target / "DATA_COMPLETED.json").write_text('{"status":"DATA_COMPLETED","fixture":true}')
    frozen_checked = []
    truth = np.arange(17)[:, None, None] * 0.01 * np.tile(direction[None, :], (72, 1))

    def path_observations(*args, role, methods=(), **kwargs):
        if role == "direct_work":
            return {"PRESERVE_XI": truth.copy()}
        assert (target / "PREDICTIONS_FROZEN.json").exists()
        assert (target / "models/fixed_full/MODEL.json").exists()
        assert (target / "models/nonlinear/MODEL.json").exists()
        frozen_checked.append(True)
        se = np.full_like(truth, 1e-5)
        se[0] = 0
        return {
            "estimate": truth,
            "se": se,
            "resolved": np.ones_like(truth, bool),
            "is_alias": np.arange(17) == 0,
            "all_prompt_overlap_usable": np.ones(17, bool),
        }

    monkeypatch.setattr(tracking, "_path_observations", path_observations)
    task = {"id": "track", "seed": 41001, "arm": "X_BASE"}
    selection = {
        "primary_models": [
            {
                "id": "rbf",
                "model": "FULL_RBF_RAW",
                "representation": "R2",
                "settings": {"alpha": 1e-5, "bandwidth_multiplier": 1.0},
            }
        ]
    }
    response = {"work": {}, "direct_work": {}, "reference": {}}
    scores = {"direct_work": {}, "reference": {}}
    result = tracking._fit_and_evaluate(
        {},
        runtime,
        task,
        target,
        probes,
        policies,
        {"banks": banks},
        response,
        scores,
        selection,
        False,
    )
    assert result["status"] == "FIXTURE_EVALUATED"
    assert measured_steps == list(tracking.REFRESH)
    assert frozen_checked == [True]
    with np.load(target / "PREDICTIONS_FROZEN.npz") as predictions:
        np.testing.assert_allclose(predictions["ORIGIN_SCORE_CUMULATIVE"][:, 0], truth[:, 0])
        expected = 0.01 * 4 * (1 + 5 + 9 + 13) * direction
        np.testing.assert_allclose(predictions["REFRESHED_SCORE_EVERY4"][-1, 0], expected)
    before = {
        str(p.relative_to(target)): tracking.sha256_file(p)
        for p in target.rglob("*")
        if p.is_file()
    }
    repeated = tracking._fit_and_evaluate(
        {},
        runtime,
        task,
        target,
        probes,
        policies,
        {"banks": banks},
        response,
        scores,
        selection,
        True,
    )
    assert repeated == result
    assert before == {
        str(p.relative_to(target)): tracking.sha256_file(p)
        for p in target.rglob("*")
        if p.is_file()
    }
    assert json.loads((target / "EVALUATION.json").read_text())["fixture"] is True


def test_float32_endpoints_are_promoted_before_subtraction():
    data = np.zeros((2, 2, 1), dtype=np.float32)
    data[1, 0, 0] = 16777216
    data[1, 1, 0] = -1
    layout = {"a": {"start": 0, "stop": 1, "shape": (1,)}}
    gradient = {
        "parameter_order": ["a"],
        "expansion_point": "ORIGIN",
        "reference_labels_read": False,
        "groups": [["f", "i"]],
        "grouped_gradients": {"a": np.ones((1, 4, 1))},
    }
    result = tracking._gradient_products(data, layout, gradient, 1)
    np.testing.assert_array_equal(result, np.full((1, 1, 4), 16777217, dtype=np.float64))


def test_endpoint_view_checks_quota_before_allocation_and_preserves_dtype(tmp_path, monkeypatch):
    import torch

    from src.modeling_v4 import gpu_collect as collect
    from src.modeling_v4 import storage

    parameter = torch.tensor([1.25, 2.5], dtype=torch.float32)
    state = {"parameters": {"a": parameter}}
    policy = {
        "inference_fingerprint": "fp",
        "module_scalings": {"m": 1.0},
        "parameter_layout": [{"name": "a", "shape": [2]}],
    }

    class Cache:
        def load(self, policy):
            return state

    monkeypatch.setattr(collect, "_fingerprint", lambda *args: "fp")
    calls = []

    def blocked(path, size):
        calls.append(size)
        raise OSError("quota fixture")

    monkeypatch.setattr(storage, "require_space", blocked)
    out = tmp_path / "mapped.npy"
    with pytest.raises(OSError, match="quota"):
        tracking._endpoint_view([(policy, policy)], Cache(), {}, out)
    assert calls == [4112]
    assert not out.exists()
    monkeypatch.setattr(storage, "require_space", lambda *a: {})
    data, _, _ = tracking._endpoint_view([(policy, policy)], Cache(), {}, out)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data[0, 0], parameter.numpy())


def test_any_prompt_overlap_failure_marks_all_groups_of_contrast_unresolved():
    groups = [["f0", "i0"], ["f1", "i1"]]
    probes = [{"family": "f0", "interface": "i0"}, {"family": "f1", "interface": "i1"}]
    se = np.full((17, 2, 4), 1e-5)
    se[0] = 0
    usable = np.ones(17, dtype=bool)
    usable[3] = False
    reference = {
        "estimate": np.zeros_like(se),
        "se": se,
        "is_alias": np.arange(17) == 0,
        "all_prompt_overlap_usable": usable,
    }
    grouped = tracking._group_reference(reference, probes, groups)
    assert grouped["resolved"][1].all()
    assert not grouped["resolved"][3].any()
    assert grouped["resolved"][0].all()
    assert set(grouped["resolved_at_scale"]) == {"0.00025", "0.001", "0.005"}


def test_interrupted_model_publication_recovers_without_replacing_actual_fit(tmp_path):
    from src.modeling_v4.full_kernels import fit_precomputed_response, save_model

    model = fit_precomputed_response(
        np.eye(2), np.ones((2, 1, 4)), query_cross_gram=np.ones((1, 2)), query_norms=np.array([2.0])
    )
    path = tmp_path / "model"
    save_model(model, path)
    original = tracking.sha256_file(path / "FIT.npz")
    (path / "MODEL.json").unlink()  # Fixture simulates a crash before manifest publication.
    receipt = tracking._freeze_model(model, path)
    assert receipt["fit"]["sha256"] == original
    assert (path / "MODEL.json").is_file()
    (path / "MODEL.json").unlink()
    changed = fit_precomputed_response(
        np.eye(2),
        np.full((2, 1, 4), 2.0),
        query_cross_gram=np.ones((1, 2)),
        query_norms=np.array([2.0]),
    )
    with pytest.raises(ValueError, match="differs"):
        tracking._freeze_model(changed, path)
    assert tracking.sha256_file(path / "FIT.npz") == original


def test_collection_failure_restores_entry_state_and_keeps_source_originals(tmp_path, monkeypatch):
    import json

    from src.modeling_v4 import data_adapter
    from src.modeling_v4 import gpu_collect as collect

    task = {"id": "E_test", "source_task": "source_test", "seed": 41001, "arm": "X_BASE"}
    source_root = tmp_path / "tasks/source_test"
    (source_root / "tracking").mkdir(parents=True)
    source = {
        "status": "COMPLETED",
        "steps": 128,
        "seed": 41001,
        "arm": "X_BASE",
        "execution_kind": "REAL_CUDA_MODEL",
        "origins": {"64": {"fixture": True}},
    }
    (source_root / "COMPLETE.json").write_text(json.dumps(source))
    for step in tracking.STEPS:
        policy = {
            "inference_fingerprint": "same64" if step == 64 else str(step),
            "checkpoint": {"identity": {"step": step, "seed": 41001, "arm": "X_BASE"}},
        }
        (source_root / "tracking" / f"step_{step:03d}.json").write_text(json.dumps(policy))
    original = {str(p): tracking.sha256_file(p) for p in source_root.rglob("*") if p.is_file()}
    saved = {
        "parameters": {},
        "optimizer": {},
        "sampler": {},
        "rng": dict.fromkeys(["python", "numpy", "torch", "cuda"]),
    }

    class Cache:
        def load(self, spec):
            return saved

    restored = []
    runtime = {
        "identity": {"execution_kind": "CPU_FIXTURE"},
        "checkpoint_cache": Cache(),
        "tracking_extension": {"extension_hash": "fixture"},
        "analysis_rules": {},
        "capture_complete": lambda: {"entry": 1},
        "restore_complete": restored.append,
    }
    monkeypatch.setattr(
        tracking, "_validate_gate", lambda *a: {"m": 24, "selection_hash": "fixture"}
    )
    monkeypatch.setattr(collect, "verify_artifact_bindings", lambda value: None)
    monkeypatch.setattr(collect, "_fingerprint", lambda *a: "same64")
    monkeypatch.setattr(data_adapter, "build_bank_plan", lambda *a, **kw: {"fixture": True})

    def failure(*args, **kwargs):
        raise RuntimeError("fixture failed fork")

    monkeypatch.setattr(collect, "run_forks", failure)
    inputs = {
        "panels": {"observation": [{"prompt_id": str(i)} for i in range(72)]},
        "train_prompts": [],
    }
    with pytest.raises(RuntimeError, match="failed fork"):
        tracking.run_tracking_task({}, runtime, inputs, task, tmp_path)
    assert restored == [{"entry": 1}]
    assert original == {
        str(p): tracking.sha256_file(p) for p in source_root.rglob("*") if p.is_file()
    }
    assert not (tmp_path / "tasks/E_test/COMPLETE.json").exists()


def test_frozen_observation_setting_precedence_and_explicit_default():
    value = tracking._frozen_observation(
        {"settings": {"observation_method": "RAW4", "observation": "PRESERVE_XI"}}
    )
    assert value["observation_method"] == "RAW4"
    assert value["observation_setting_source"] == "FROZEN_observation_method"
    assert (
        tracking._frozen_observation(None)["observation_setting_source"]
        == "EXPLICIT_BASELINE_DEFAULT_PRESERVE_XI"
    )
    assert (
        tracking._frozen_observation({"settings": {"observation": "CROSSFIT_COV_ZERO_SUM"}})[
            "observation_method"
        ]
        == "CROSSFIT_COV_ZERO_SUM"
    )
    with pytest.raises(ValueError, match="Unsupported"):
        tracking._frozen_observation({"settings": {"observation_method": "UNKNOWN"}})


def test_tracking_execution_never_substitutes_same_named_representations():
    raw = {
        "id": "raw-rbf",
        "model": "FULL_RBF_RAW",
        "representation": "R2",
        "settings": {"alpha": 1e-4, "observation_method": "RAW4"},
    }
    fixed, nonlinear, missing = tracking._tracking_models({"primary_models": [raw]})
    assert fixed is None and nonlinear == raw and missing == {}
    for representation, settings, expected in (
        ("R4", {"signature_model_selection": {}}, "MISSING_PAID_TRACK_SIGNATURES"),
        ("R2", {"standardize_blocks": True}, "UNSUPPORTED_TRACK_INPUT_METRIC"),
        ("R3", {}, "UNSUPPORTED_TRACK_REPRESENTATION"),
        ("R2", {"rank_cap": 8}, "UNSUPPORTED_TRACK_RANK"),
        ("R2", {"extra_setting": 1}, "UNSUPPORTED_TRACK_SETTINGS"),
    ):
        spec = {**raw, "representation": representation, "settings": settings}
        fixed, nonlinear, missing = tracking._tracking_models({"primary_models": [spec]})
        assert fixed is None and nonlinear is None
        assert missing["SELECTED_MODEL:raw-rbf"] == "NOT_AVAILABLE_" + expected
        assert missing["NONLINEAR"] == "NOT_AVAILABLE_" + expected
    signature_linear = {**raw, "model": "FULL_DUAL_RIDGE", "representation": "R4"}
    fixed, nonlinear, missing = tracking._tracking_models({"primary_models": [signature_linear]})
    assert fixed is None and nonlinear is None
    assert missing["SELECTED_MODEL:raw-rbf"] == "NOT_AVAILABLE_MISSING_PAID_TRACK_SIGNATURES"
