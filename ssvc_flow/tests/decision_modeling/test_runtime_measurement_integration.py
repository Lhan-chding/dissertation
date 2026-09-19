"""Production measurement wiring on explicit CPU fake Torch policies, never Qwen."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.decision_modeling import measurement
from src.decision_modeling.runtime import runtime_metadata
from src.decision_modeling.score_cache import ScoreCache
from src.modeling_v3.io import canonical_hash, sha256_file
from src.modeling_v4 import gpu_collect as gpu


@pytest.fixture
def inputs(tmp_path):
    helper_path = Path(__file__).parents[1] / "modeling_v4/test_gpu_collect.py"
    spec = importlib.util.spec_from_file_location("decision_tiny_gpu_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime, adapter_module = module.tiny_runtime()
    prompts = adapter_module.fake_prompts(2)
    origin = runtime["state"].capture({"origin": "fixture"})
    origin_policy = gpu._save_policy(
        runtime, origin, tmp_path / "origin.pt", {"candidate_id": "origin"}
    )
    left = gpu._save_policy(runtime, origin, tmp_path / "left.pt", {"candidate_id": "left"})
    with torch.no_grad():
        runtime["adapter"].model.logits[0].add_(0.75)
    right_state = runtime["state"].capture({"origin": "fixture", "candidate": "right"})
    right = gpu._save_policy(runtime, right_state, tmp_path / "right.pt", {"candidate_id": "right"})
    assert left["inference_fingerprint"] != right["inference_fingerprint"]
    return runtime, prompts, origin_policy, [left, right]


def test_cached_backend_retains_duplicate_draws_and_parity(inputs, tmp_path):
    runtime, prompts, _, policies = inputs
    with ScoreCache(tmp_path / "scores.sqlite", namespace={"fixture": "integration"}) as cache:
        backend = measurement.make_backend(runtime, cache)
        backend.activate(policies[0])
        first = backend.generate(prompts[0], seed=7)
        second = backend.generate(prompts[0], seed=7)
        assert first["token_ids"] == second["token_ids"]
        assert first["scoring_usable"] and second["scoring_usable"]
        assert cache.counters["logical_scores"] == 2
        assert cache.counters["physical_scoring_keys"] == 1
        assert cache.counters["memory_hits"] == 1
        assert backend.counters["generated_sequences"] == 2
        assert backend.counters["scored_sequences"] == 1
        backend.activate(policies[1])
        backend.score(prompts[0], {**first, "sample_id": "first"})
        assert cache.counters["physical_scoring_keys"] == 2


def test_observe_32_to_128_nested_resume_without_resampling(inputs, tmp_path):
    runtime, prompts, _, policies = inputs
    out = tmp_path / "observe"
    first, rows32 = measurement.observe(
        runtime, policies[:1], prompts, out=out, look=32, stream_id="nested"
    )
    assert first["execution_kind"] == "CPU_FAKE_TORCH"
    assert runtime["adapter"].generation_calls == 64
    second, rows128 = measurement.observe(
        runtime, policies[:1], prompts, out=out, look=128, stream_id="nested"
    )
    assert second["cost_this_invocation"]["generated_sequences"] == 192
    assert runtime["adapter"].generation_calls == 256
    before = {row["sample_id"]: row for row in rows32}
    after = {row["sample_id"]: row for row in rows128}
    assert all(after[key] == row for key, row in before.items())
    third, replay = measurement.observe(
        runtime, policies[:1], prompts, out=out, look=128, stream_id="nested"
    )
    assert replay == rows128
    assert third["cost_this_invocation"]["generated_sequences"] == 0
    assert runtime["adapter"].generation_calls == 256
    assert len(list((out / "costs").glob("invocation_*.json"))) == 3


def test_finite_parity_failure_preserves_endpoint_samples(inputs, tmp_path, monkeypatch):
    runtime, prompts, _, policies = inputs
    generate = runtime["adapter"].generate

    def mismatched_generate(*args, **kwargs):
        row = generate(*args, **kwargs)
        row["behavior_token_logprobs"][0] -= 0.25
        return row

    monkeypatch.setattr(runtime["adapter"], "generate", mismatched_generate)
    result, rows = measurement.observe(
        runtime, policies[:1], prompts[:1], out=tmp_path / "mismatch", stream_id="mismatch"
    )
    assert result["status"] == "OBSERVED"
    assert not result["scoring_usable"]
    assert len(rows) == 32
    assert all(not row["scoring_usable"] for row in rows)
    assert all(row["event"] in "XSWI" for row in rows)


def test_roles_and_policies_have_disjoint_rng_even_with_same_stream_id(inputs, tmp_path):
    runtime, prompts, _, policies = inputs
    keys, rng_ids = [], []
    for name, role, policy in (
        ("endpoint", "endpoint", policies[0]),
        ("reference", "reference", policies[0]),
        ("discovery", "discovery", policies[0]),
        ("other_policy", "endpoint", policies[1]),
    ):
        _, rows = measurement.observe(
            runtime,
            [policy],
            prompts[:1],
            out=tmp_path / name,
            look=32,
            role=role,
            stream_id="intentionally-repeated",
        )
        keys.append({row["sample_id"] for row in rows})
        rng_ids.append({row["rng_stream_id"] for row in rows})
    assert sum(map(len, keys)) == len(set().union(*keys))
    assert len(set().union(*rng_ids)) == 4


def test_bridge_origin_mix_support_projection_and_idempotent_replay(inputs, tmp_path):
    runtime, prompts, origin, policies = inputs
    out = tmp_path / "bridge"
    result = measurement.bridge(
        runtime, policies, prompts, out=out, stream_id="bridge", origin_policy=origin
    )
    assert result["status"] == "D1_MEASURED"
    assert result["lr_mass_enabled"]
    assert result["scoring_paths"]["tested_sequences"] == 24
    assert {row["proposal"] for row in result["proposal_comparisons"]} == {"ORIGIN", "MIX"}
    for comparison in result["proposal_comparisons"]:
        assert comparison["status"] == "MEASURED"
        assert len(comparison["units"]) == len(prompts)
        for unit in comparison["units"]:
            assert unit["n"] == 32
            raw, projected = np.array(unit["RAW4"]), np.array(unit["PRESERVE_XI"])
            np.testing.assert_allclose(projected[[0, 3]], raw[[0, 3]])
            assert abs(projected.sum()) < 1e-12
            assert len(unit["mass_bounds"]) == 2
            for bounds in unit["mass_bounds"]:
                assert bounds["evidence_kind"] == "CONDITIONAL_ON_SCORER"
                assert 0 <= bounds["known_mass"] <= 1
                assert bounds["known_mass"] + bounds["unknown_tail"] == 1
            if comparison["proposal"] == "MIX":
                assert max(unit["max_abs_contribution"]) <= 2
    generation_calls = runtime["adapter"].generation_calls
    receipt = (out / "COMPLETE.json").read_bytes()
    replay = measurement.bridge(
        runtime, policies, prompts, out=out, stream_id="bridge", origin_policy=origin
    )
    assert runtime["adapter"].generation_calls == generation_calls
    assert (out / "COMPLETE.json").read_bytes() == receipt
    assert replay["proposal_comparisons"] == result["proposal_comparisons"]


def test_runtime_metadata_rejects_corrupt_image_before_gpu(tmp_path):
    config = {"model": {"id": "fixture"}}
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.jsonl", "control.jsonl", "image.png"):
        (data / name).write_bytes(name.encode())
    panels = {
        "config_hash": canonical_hash(config),
        "status": "READY",
        "input_bindings": {
            split: {"path": f"{split}.jsonl", "sha256": sha256_file(data / f"{split}.jsonl")}
            for split in ("train", "control")
        },
        "image_verification": {
            "sha256_by_relative_path": {"image.png": sha256_file(data / "image.png")}
        },
    }
    panels["inputs_hash"] = canonical_hash(panels)
    panel_path, task_path = tmp_path / "panels.json", tmp_path / "tasks.json"
    panel_path.write_text(json.dumps(panels))
    task_path.write_text(json.dumps({"config": {"qwen": config["model"]}}))
    private = {
        "config_hash": canonical_hash(config),
        "source_root": str(tmp_path / "original"),
        "data_root": str(data),
        "tasks": gpu._binding(task_path),
        "panels": gpu._binding(panel_path),
    }
    private_path = tmp_path / "private.json"
    private_path.write_text(json.dumps(private))
    runtime_metadata(config, private_path, out=tmp_path / "new")
    (data / "image.png").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match=r"Server input changed.*image\.png"):
        runtime_metadata(config, private_path, out=tmp_path / "new")
