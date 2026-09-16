"""Tiny real collection and safe inference, never actual Qwen evidence."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.modeling_v3.io import atomic_json, canonical_hash, finalize_run
from src.modeling_v4 import gpu_collect as gpu
from src.modeling_v4.full_kernels import fit_full_response
from src.modeling_v4.signature_fit import save_signature_model
from src.modeling_v4.tracking_signature import (
    collect_tracking_signatures,
    load_tracking_predictions,
)


def originals(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "signature_collect_fixture", Path(__file__).with_name("test_signature_collect.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    runtime, _, adapter_module = helper.setup(tmp_path)
    prompts = adapter_module.fake_prompts(36)
    panels = helper.splits(prompts)
    task = {"id": "track_41001", "seed": 41001, "arm": "X_BASE", "kind": "offline_track"}
    policies = {}
    entry = runtime["capture_complete"]()
    for step in range(64, 81):
        runtime["restore_complete"](entry)
        with torch.no_grad():
            runtime["adapter"].model.logits[0] += (0 if step == 65 else step - 64) * 0.002
        policies[step] = gpu._save_policy(
            runtime,
            runtime["capture_complete"](),
            tmp_path / f"step{step}.pt",
            {"candidate_id": f"step{step}", "step": step, "seed": 41001, "arm": "X_BASE"},
        )
    runtime["restore_complete"](entry)
    e = np.random.default_rng(4).normal(size=(3, 576))
    y = np.random.default_rng(5).normal(size=(3, 72, 4)) * 0.001
    model = fit_full_response(e, y, method="FULL_DUAL_RIDGE", alpha=1e-5)
    binding = save_signature_model(model, tmp_path / "linear", output_shape=(72, 4))
    width, output, state = 256, 72 * 4, 36 * 4
    weights = {
        "linear.weight": np.full((output, 576), 0.00001),
        "residual.0.weight": np.full((width, state + 576), 0.0001),
        "residual.0.bias": np.zeros(width),
        "residual.2.weight": np.eye(width) * 0.01,
        "residual.2.bias": np.zeros(width),
        "residual.4.weight": np.full((output, width), 0.0001),
        "residual.4.bias": np.zeros(output),
    }
    nonlinear = {
        "metadata": {
            "method": "NONLINEAR_SIGNATURE_RESIDUAL",
            "hidden_width": width,
            "state_feature_width": state,
        },
        "state_dicts": [weights] * 3,
        "normalization": {
            "signature_scale": np.ones(576),
            "state_mean": np.zeros(state),
            "state_scale": np.ones(state),
        },
    }
    nb = save_signature_model(nonlinear, tmp_path / "network", output_shape=(72, 4))
    selected = [
        {
            "id": "RAW4/linear",
            "spec": {"model": "FULL_DUAL_RIDGE", "alpha": 1e-5},
            "observation_method": "RAW4",
            "model": binding,
        },
        {
            "id": "RAW4/mlp",
            "spec": {
                "model": "NONLINEAR_SIGNATURE_RESIDUAL",
                "hidden_width": 256,
                "state_mode": "EVENT_ONLY",
            },
            "observation_method": "RAW4",
            "model": nb,
        },
    ]
    lock = {
        "kind": "V4_SIGNATURE_SELECTION",
        "status": "FROZEN",
        "fixture": True,
        "config_hash": "tiny",
        "source_hash": "tiny",
        "models": selected,
        "prompt_ids": [f"obs{i}" for i in range(72)],
        "probe_groups": [f"g{i % 6}|SYMBOLIC" for i in range(72)],
        "fit_scope": "ALL_REGISTERED_DEVELOPMENT_ORIGINS",
        "development_seeds": [41001, 41002, 41003],
        "query_labels_read": False,
        "reference_labels_read": False,
    }
    lock["selection_hash"] = canonical_hash(lock)
    folder = tmp_path / "selection"
    folder.mkdir()
    atomic_json(folder / "SIGNATURE_SELECTION.json", lock)
    finalize_run(folder, {"fixture": True})
    primary = {
        "primary_models": [
            {
                "id": m["id"],
                "model": m["spec"]["model"],
                "representation": "R4",
                "settings": {
                    "signature_model_selection": gpu._binding(folder / "SIGNATURE_SELECTION.json"),
                    "signature_model_id": m["id"],
                    "observation_method": "RAW4",
                },
            }
            for m in selected
        ]
    }
    return runtime, task, policies, prompts, panels, primary


def test_actual_trajectory_shared_anchor_safe_models_and_immutable_resume(tmp_path):
    from src.optimizer_fork import state_hash

    runtime, task, policies, prompts, panels, selection = originals(tmp_path)
    before = state_hash(runtime["capture_complete"]())
    result = collect_tracking_signatures(
        {},
        runtime,
        task,
        policies,
        prompts,
        panel_splits=panels,
        selection=selection,
        out=tmp_path / "track",
        fixture=True,
    )
    with np.load(result["arrays"]["path"], allow_pickle=False) as arrays:
        assert arrays["native_signatures"].shape == (17, 576)
        assert arrays["predictions"].shape == (2, 17, 72, 4)
        np.testing.assert_array_equal(arrays["native_signatures"][:2], 0)
        np.testing.assert_array_equal(arrays["predictions"][:, :2], 0)
        assert np.any(arrays["predictions"][1, 2:] != 0)
    assert result["completed_unique_generation_sequences"] == 576
    assert result["completed_unique_scoring_sequences"] == 16 * 576
    assert result["physical_scored_policies"] == 16
    assert result["query_labels_read"] is result["reference_labels_read"] is False
    assert result["fit_or_refit_performed"] is False
    assert result["intermediate_paid_features"] is True
    assert [u["step"] for u in result["units"]] == list(range(64, 81))
    assert state_hash(runtime["capture_complete"]()) == before
    probes = [
        {"prompt_id": f"obs{i}", "family": f"g{i % 6}", "interface": "SYMBOLIC"} for i in range(72)
    ]
    grouped, methods, artifacts = load_tracking_predictions(
        gpu._binding(tmp_path / "track" / "COMPLETE.json"),
        selection=selection,
        origin_id=task["id"],
        policies=policies,
        probes=probes,
        groups=[(f"g{i}", "SYMBOLIC") for i in range(6)],
    )
    assert grouped["SELECTED_MODEL:RAW4/mlp"].shape == (17, 6, 4)
    assert all(m["uses_intermediate_measurements"] for m in methods.values())
    assert artifacts["SELECTED_MODEL:RAW4/mlp"]["frozen_model"]
    wrong = copy.deepcopy(selection)
    wrong["primary_models"][0]["model"] = "NONLINEAR_SIGNATURE_RESIDUAL"
    with pytest.raises(ValueError, match="model label"):
        load_tracking_predictions(
            gpu._binding(tmp_path / "track" / "COMPLETE.json"),
            selection=wrong,
            origin_id=task["id"],
            policies=policies,
            probes=probes,
            groups=[(f"g{i}", "SYMBOLIC") for i in range(6)],
        )
    calls = runtime["adapter"].forward_calls
    assert (
        collect_tracking_signatures(
            {},
            runtime,
            task,
            policies,
            prompts,
            panel_splits=panels,
            selection=selection,
            out=tmp_path / "track",
            fixture=True,
            resume=True,
        )
        == result
    )
    assert runtime["adapter"].forward_calls == calls
    with open(result["arrays"]["path"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        collect_tracking_signatures(
            {},
            runtime,
            task,
            policies,
            prompts,
            panel_splits=panels,
            selection=selection,
            out=tmp_path / "track",
            fixture=True,
            resume=True,
        )


def test_trajectory_boundary_rejects_wrong_steps_and_outcome_inputs_before_generation(tmp_path):
    runtime, task, policies, prompts, panels, selection = originals(tmp_path)
    before = runtime["adapter"].generation_calls
    invalid = copy.deepcopy(policies)
    invalid[70]["checkpoint"]["identity"]["step"] = 69
    with pytest.raises(ValueError, match="trajectory"):
        collect_tracking_signatures(
            {},
            runtime,
            task,
            invalid,
            prompts,
            panel_splits=panels,
            selection=selection,
            out=tmp_path / "bad",
            fixture=True,
        )
    panels["reference"]["labels"] = [1]
    with pytest.raises(ValueError, match="metadata"):
        collect_tracking_signatures(
            {},
            runtime,
            task,
            policies,
            prompts,
            panel_splits=panels,
            selection=selection,
            out=tmp_path / "bad",
            fixture=True,
        )
    assert runtime["adapter"].generation_calls == before
    with pytest.raises(ValueError, match="fixture"):
        collect_tracking_signatures(
            {},
            runtime,
            task,
            policies,
            prompts,
            panel_splits=panels,
            selection=selection,
            out=tmp_path / "bad",
        )


def test_interrupted_physical_score_restores_state_and_reuses_committed_anchor(
    tmp_path, monkeypatch
):
    from src.optimizer_fork import state_hash

    runtime, task, policies, prompts, panels, selection = originals(tmp_path)
    before = state_hash(runtime["capture_complete"]())
    original = gpu.collect_scores

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected interruption after durable physical scores")

    monkeypatch.setattr(gpu, "collect_scores", interrupted)
    with pytest.raises(RuntimeError, match="injected"):
        collect_tracking_signatures(
            {},
            runtime,
            task,
            policies,
            prompts,
            panel_splits=panels,
            selection=selection,
            out=tmp_path / "track",
            fixture=True,
        )
    assert state_hash(runtime["capture_complete"]()) == before
    generated = runtime["adapter"].generation_calls
    failed = json.loads((tmp_path / "track" / "COST_ATTEMPT_0000.json").read_text())
    assert failed["status"] == "FAILED" and failed["forward_calls"] > 0
    monkeypatch.setattr(gpu, "collect_scores", original)
    result = collect_tracking_signatures(
        {},
        runtime,
        task,
        policies,
        prompts,
        panel_splits=panels,
        selection=selection,
        out=tmp_path / "track",
        fixture=True,
        resume=True,
    )
    assert runtime["adapter"].generation_calls == generated
    assert state_hash(runtime["capture_complete"]()) == before
    assert "COST_ATTEMPT_0000.json" in result["artifact_hashes"]
    assert "COST_ATTEMPT_0001.json" in result["artifact_hashes"]
