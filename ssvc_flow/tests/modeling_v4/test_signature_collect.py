"""Small differentiable CPU producers; these are not Qwen experiment evidence."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.modeling_v4 import gpu_collect as gpu
from src.modeling_v4.signature_collect import (
    collect_development_directional,
    collect_signature_features,
)


def setup(tmp_path):
    path = Path(__file__).parents[1] / "followup_test_adapter.py"
    spec = importlib.util.spec_from_file_location("signature_fixture_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.make_fake_adapter(17)
    adapter.pad_id = 4
    adapter.audit["probability_execution"] = "uncached_prefix_recompute"
    optimizer = torch.optim.AdamW([p for p in adapter.model.parameters() if p.requires_grad])
    runtime = gpu.attach_runtime(
        adapter,
        optimizer,
        identity={
            "fixture": True,
            "model_hash": "tiny",
            "config_hash": "tiny",
            "source_hash": "tiny",
        },
        parity_tolerances={
            "mean_abs_token_logp": 1e-5,
            "max_abs_token_logp": 1e-5,
            "max_abs_sequence_logp": 1e-4,
        },
    )
    original = runtime["capture_complete"]()
    origin = gpu._save_policy(runtime, original, tmp_path / "origin.pt", {"candidate_id": "origin"})
    with torch.no_grad():
        adapter.model.logits[0] += 0.08
    changed = runtime["capture_complete"]()
    candidate = gpu._save_policy(
        runtime, changed, tmp_path / "candidate.pt", {"candidate_id": "candidate"}
    )
    runtime["restore_complete"](original)
    banks = [
        {
            "bank_id": f"b{i}",
            "role": "calibration",
            "policies": {"joint_0": origin, "joint_1": candidate, "no_x_off_1": origin},
        }
        for i in range(4)
    ]
    forks = {"origin_id": "origin0", "origin_policy": origin, "banks": banks}
    return runtime, forks, module


def splits(prompts):
    result = {
        r: {"base_scene_ids": [r], "prompt_ids": [r], "sample_ids": [], "rng_namespaces": [r]}
        for r in ("train", "observation", "reference")
    }
    result["signature"] = {
        "base_scene_ids": sorted({p["base_scene_id"] for p in prompts}),
        "prompt_ids": [p["prompt_id"] for p in prompts],
        "sample_ids": [],
        "rng_namespaces": ["frozen-signature"],
    }
    return result


def test_actual_producer_native_signature_alias_dedup_and_origin_only_features(tmp_path):
    runtime, forks, module = setup(tmp_path)
    prompts = module.fake_prompts(36)
    result = collect_signature_features(
        runtime, forks, prompts, panel_splits=splits(prompts), out=tmp_path / "sig"
    )
    arrays = np.load(result["arrays"]["path"])
    assert arrays["native_signatures"].shape == (12, 576)
    assert arrays["EVENT_ONLY"].shape == (36, 4)
    assert arrays["EVENT_PLUS_DIAGNOSTICS"].shape == (36, 11)
    assert result["physical_scored_policies"] == 2
    assert result["raw_draws"] == 576
    assert result["query_endpoint_target_labels_read"] is False
    np.testing.assert_array_equal(arrays["native_signatures"][1], 0)
    before = runtime["adapter"].forward_calls
    resumed = collect_signature_features(
        runtime, forks, prompts, panel_splits=splits(prompts), out=tmp_path / "sig", resume=True
    )
    assert resumed == result and runtime["adapter"].forward_calls == before
    with open(result["arrays"]["path"], "ab") as f:
        f.write(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        collect_signature_features(
            runtime, forks, prompts, panel_splits=splits(prompts), out=tmp_path / "sig", resume=True
        )


def test_signature_split_labels_or_overlapping_namespace_fail_before_generation(tmp_path):
    runtime, forks, module = setup(tmp_path)
    prompts = module.fake_prompts(36)
    panel = splits(prompts)
    panel["reference"]["outcomes"] = ["forbidden"]
    with pytest.raises(ValueError, match="metadata"):
        collect_signature_features(
            runtime, forks, prompts, panel_splits=panel, out=tmp_path / "bad"
        )
    panel = splits(prompts)
    panel["reference"]["rng_namespaces"] = ["frozen-signature"]
    with pytest.raises(ValueError, match="overlap"):
        collect_signature_features(
            runtime, forks, prompts, panel_splits=panel, out=tmp_path / "bad"
        )
    assert runtime["adapter"].generation_calls == 0


def test_real_full_gradient_base_midpoint_and_fd_restore_complete_state(tmp_path, monkeypatch):
    from src.modeling_v4 import signature_collect as collector
    from src.optimizer_fork import state_hash
    from src.r4_inputs import STRATA

    runtime, forks, module = setup(tmp_path)
    prompts = module.fake_prompts(6)
    for p, group in zip(prompts, STRATA, strict=True):
        p["family"], p["interface"] = group
    receipt = gpu.collect_actions(
        runtime,
        forks["origin_policy"],
        prompts,
        origin_id="origin0",
        role="work",
        draws=2,
        out=tmp_path / "anchors",
    )
    samples = list(gpu._rows(receipt))
    assert all(s["proposal"] == "ORIGIN" for s in samples)
    original = runtime["capture_complete"]()
    query = collector._anchor_event_means

    def interrupted(*args):
        raise RuntimeError("injected numeric interruption")

    monkeypatch.setattr(collector, "_anchor_event_means", interrupted)
    options = dict(
        bank_ids=[b["bank_id"] for b in forks["banks"]],
        steps=[0.1, 0.01],
        sample_limit=2,
        prompt_subset=[prompts[0]["prompt_id"]],
        out=tmp_path / "directions",
    )
    with pytest.raises(RuntimeError, match="injected"):
        collect_development_directional(runtime, forks, prompts, samples, **options)
    assert state_hash(runtime["capture_complete"]()) == state_hash(original)
    assert len(list((tmp_path / "directions").glob("*/BASE_AD.json"))) == 4
    assert len(list((tmp_path / "directions").glob("*/MIDPOINT_AD.json"))) == 4
    backwards = len(runtime["score_cost_ledger"])
    monkeypatch.setattr(collector, "_anchor_event_means", query)
    result = collect_development_directional(
        runtime,
        forks,
        prompts,
        samples,
        bank_ids=[b["bank_id"] for b in forks["banks"]],
        steps=[0.1, 0.01],
        sample_limit=2,
        prompt_subset=[prompts[0]["prompt_id"]],
        out=tmp_path / "directions",
        resume=True,
    )
    assert len(runtime["score_cost_ledger"]) == backwards
    assert state_hash(runtime["capture_complete"]()) == state_hash(original)
    assert len(result["banks"]) == 4
    first = result["banks"][0]
    for point in ("BASE", "MIDPOINT"):
        assert first[point]["derivative_method"] == "FULL_REVERSE_SCORE_GRADIENT"
        assert first[point]["finite_difference_is_derivative"] is False
        assert len(first[point]["numerical_checks"]) == 2
        assert first[point]["numerical_checks"][-1]["max_abs_error"] < 1e-5
        raw = np.load(first[point]["numerical_checks"][-1]["arrays"]["path"])
        assert raw["plus_token_logprobs"].shape == (12, 2)
        assert "effective_direction_relative_error" in first[point]["numerical_checks"][-1]
    assert first["MIDPOINT"]["reachable_training_policy"] is False
    assert first["MIDPOINT"]["parameter_rounding"]["dtype"]
    bad = copy.deepcopy(samples)
    bad[0]["role"] = "reference"
    before = runtime["adapter"].forward_calls
    with pytest.raises(ValueError, match="reference"):
        collect_development_directional(
            runtime,
            forks,
            prompts,
            bad,
            bank_ids=[b["bank_id"] for b in forks["banks"]],
            steps=[0.01],
            sample_limit=2,
            out=tmp_path / "forbidden",
        )
    assert runtime["adapter"].forward_calls == before
