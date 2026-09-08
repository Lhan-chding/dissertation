"""CPU fake-model orchestration evidence; never real model/GPU compatibility."""

import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.core import load_config
from src.smoke_runtime import run_smoke


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Parameter(torch.zeros(4))
        self.frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)


class FakeAdapter:
    def __init__(self, key, spec, **kwargs):
        self.model = ToyPolicy()
        self.model_id = spec["id"]
        self.revision = "a" * 40
        self.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(eos_token_id=2))
        self.audit = {"processor_hash": "fixture-processor", "tokenizer_hash": "fixture-tokenizer"}
        self.forward_calls = self.generation_calls = 0
        self.last_vision_hash = None

    def prepare(self, prompt, root):
        image = prompt.get("image_path")
        return {
            "inputs": {},
            "audit": {
                "final_prompt": prompt["user"],
                "final_prompt_hash": prompt["prompt_hash"],
                "prompt_token_count": 4,
                "image_token_count": int(bool(image)),
                "pixel_values_hash": image,
            },
            "image": image,
        }

    def generate(self, prepared, *, seed, max_new_tokens):
        self.generation_calls += 1
        good = seed % 2 == 0
        token = 1 if good else 3
        logprobs = self.model.lora.log_softmax(-1)
        return {
            "raw_completion": "[1,2,3,4]" if good else "invalid",
            "token_ids": [token, 2],
            "completion_length": 2,
            "stop_reason": "eos",
            "elapsed_seconds": 0.001,
            "behavior_token_logprobs": logprobs[[token, 2]].detach().tolist(),
        }

    def logprobs(self, prepared, completion, require_grad=False):
        self.forward_calls += 1
        self.last_vision_hash = prepared["image"]
        result = self.model.lora.log_softmax(-1)[completion]
        return result if require_grad else result.detach()

    def reference_logprobs(self, prepared, completion):
        return torch.zeros(4).log_softmax(-1)[completion]

    def continuation_scores(self, prepared, completion, mode):
        return [0] * len(completion), self.logprobs(prepared, completion).tolist()

    def make_prefix_state(self, prepared):
        return prepared

    def continue_from_state(self, state, completion):
        return self.continuation_scores(state, completion, "full")


def panel(root=None):
    from src.core import file_hash

    if root is not None:
        (Path(root) / "images").mkdir(parents=True, exist_ok=True)
        for index in range(18):
            (Path(root) / f"images/scene-{index}.png").write_text(f"image{index}")
    return [
        {
            "base_scene_id": f"scene-{index}",
            "split": "calibration",
            "truth_world": [1, 2, 3, 4],
            "observed_world": [9, 2, 3, 4],
            "changed_index": 0,
            "cue": {"family": "duplicate_encoding", "known_index": 0, "known_value": 1},
            "constraint_family": "duplicate_encoding",
            "operation": "sum4",
            "chart_type": "line",
            "image_path": f"images/scene-{index}.png",
            "image_hash": file_hash(Path(root) / f"images/scene-{index}.png")
            if root
            else f"image{index}",
        }
        for index in range(18)
    ]


def test_smoke_fake_adapter_real_cpu_updates_resume_and_identity(tmp_path):
    config = load_config()
    config["data_root"] = str(tmp_path)
    scenes = panel(tmp_path)
    first = run_smoke(config, scenes, tmp_path / "P1", "qwen35_9b", _adapter_factory=FakeAdapter)
    assert first["passed"], first["checks"]
    assert first["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
    assert first["raw_sample_count"] == 352
    assert first["initial_prompt_count"] == 36
    assert first["backward_calls_including_replays"] == 128
    assert first["fork_resume_audit"]["optimizer_state_equal"]
    assert first["fork_resume_audit"]["comparisons_to_uninterrupted"][0]["bitwise_equal"]
    second = run_smoke(
        config, scenes, tmp_path / "P1", "qwen35_9b", resume=True, _adapter_factory=FakeAdapter
    )
    assert second["passed"]
    assert second["generation_calls_this_invocation"] == 0
    assert second["raw_sample_count"] == first["raw_sample_count"]
    changed = {**config, "sample_seed": 3}
    with pytest.raises(ValueError, match="Resume data/config"):
        run_smoke(
            changed, scenes, tmp_path / "P1", "qwen35_9b", resume=True, _adapter_factory=FakeAdapter
        )
    (tmp_path / scenes[0]["image_path"]).write_text("changed")
    with pytest.raises(ValueError, match="image content hash"):
        run_smoke(
            config, scenes, tmp_path / "P1", "qwen35_9b", resume=True, _adapter_factory=FakeAdapter
        )


def test_runtime_stops_before_model_download_without_cuda(tmp_path):
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="no model download"),
    ):
        run_smoke(load_config(), panel(), tmp_path, "qwen35_9b")
    with pytest.raises(ValueError, match="exactly 18 calibration"):
        run_smoke(load_config(), panel()[:1], tmp_path, "qwen35_9b", _adapter_factory=FakeAdapter)


def test_update_counts_measured_forwards_separately_from_sequences():
    from src.grpo_update import perform_update

    class PrefixAdapter(FakeAdapter):
        def logprobs(self, prepared, completion, require_grad=False):
            self.forward_calls += len(completion) - 1
            return super().logprobs(prepared, completion, require_grad=require_grad)

    adapter = PrefixAdapter("test", load_config()["models"]["qwen35_9b"])
    groups = [
        [
            {
                "prepared": {"image": None},
                "token_ids": [token, 2],
                "old_logprobs": [-1.38629436] * 2,
                "reward_sum": reward,
            }
            for token, reward in ((1, 1), (3, 0))
        ]
    ]
    optimizer = torch.optim.AdamW([adapter.model.lora], lr=1e-5)
    result = perform_update(adapter, optimizer, groups)
    assert result["post_update_likelihood_forwards"] == 4
    assert result["post_update_likelihood_sequences"] == 2


def test_incompatible_bank_and_no_signal_are_not_success(tmp_path):
    class NoSignal(FakeAdapter):
        def generate(self, prepared, **kwargs):
            result = super().generate(prepared, **kwargs)
            return {**result, "raw_completion": "invalid"}

    config = {**load_config(), "data_root": str(tmp_path)}
    result = run_smoke(
        config, panel(tmp_path), tmp_path / "P1", "qwen35_9b", _adapter_factory=NoSignal
    )
    assert not result["passed"]
    assert not result["checks"]["nonempty_gradients_and_measured_step"]
    assert all(step["actual_step_norm"] == 0 for step in result["training_steps"])


def test_group_bank_rejects_checkpoint_policy_drift(tmp_path):
    from src.core import RunStore
    from src.smoke_runtime import Telemetry, _collect_group

    config = load_config()
    adapter = FakeAdapter("test", config["models"]["qwen35_9b"])
    store = RunStore(tmp_path, {"model_hash": "m", "data_hash": "d", "config_hash": "c"})
    prepared = adapter.prepare({"user": "test", "prompt_hash": "p"}, tmp_path)
    scene = panel()[0]
    _collect_group(adapter, store, scene, "SYMBOLIC_FRESH", prepared, 0, "g", config, Telemetry())
    with torch.no_grad():
        adapter.model.lora[0] = 1
    with pytest.raises(ValueError, match="policy differs"):
        _collect_group(
            adapter, store, scene, "SYMBOLIC_FRESH", prepared, 0, "g", config, Telemetry()
        )


def test_resume_preserves_original_runtime_lock_on_dependency_drift(tmp_path):
    from src.smoke_runtime import _runtime_lock

    config = {**load_config(), "_model_key": "qwen35_9b"}
    adapter = FakeAdapter("test", config["models"]["qwen35_9b"])
    identity = {"model_hash": "m", "data_hash": "d", "config_hash": "c"}
    _runtime_lock(config, adapter, tmp_path, identity)
    original = (tmp_path / "runtime_lock.json").read_bytes()
    freeze = (tmp_path / "pip-freeze.txt").read_bytes()
    with (
        patch(
            "src.smoke_runtime.subprocess.run",
            return_value=types.SimpleNamespace(stdout="changed dependency", returncode=0),
        ),
        pytest.raises(ValueError, match="dependency/source drift"),
    ):
        _runtime_lock(config, adapter, tmp_path, identity)
    assert (tmp_path / "runtime_lock.json").read_bytes() == original
    assert (tmp_path / "pip-freeze.txt").read_bytes() == freeze


def test_final_update_kl_alarm_cannot_be_bypassed_by_resume(tmp_path):
    from src.smoke_runtime import _real_update

    config = {**load_config(), "data_root": str(tmp_path)}
    scenes = panel(tmp_path)
    calls = 0

    def audited_update(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = _real_update(*args, **kwargs)
        return {**result, "empirical_same_training_bank_kl": 2.0 if calls == 2 else 0.0}

    with (
        patch("src.smoke_runtime._real_update", side_effect=audited_update),
        pytest.raises(RuntimeError, match="terminal alarm"),
    ):
        run_smoke(config, scenes, tmp_path / "P1", "qwen35_9b", _adapter_factory=FakeAdapter)
    checkpoint = (tmp_path / "P1/checkpoint.pt").read_bytes()
    with (
        patch("src.smoke_runtime._real_update", side_effect=AssertionError("must not update")),
        pytest.raises(RuntimeError, match="terminal alarm"),
    ):
        run_smoke(
            config,
            scenes,
            tmp_path / "P1",
            "qwen35_9b",
            resume=True,
            _adapter_factory=FakeAdapter,
        )
    assert (tmp_path / "P1/checkpoint.pt").read_bytes() == checkpoint


def test_new_on_policy_mismatch_stops_before_next_update(tmp_path):
    from src.smoke_runtime import _real_update

    class MismatchAfterFirstUpdate(FakeAdapter):
        def logprobs(self, prepared, completion, require_grad=False):
            result = super().logprobs(prepared, completion, require_grad=require_grad)
            drifted = bool(self.model.lora.detach().abs().sum()) and not require_grad
            return result + 0.2 if drifted else result

    config = {**load_config(), "data_root": str(tmp_path)}
    with (
        patch("src.smoke_runtime._real_update", wraps=_real_update) as update,
        pytest.raises(RuntimeError, match="current update bank failed"),
    ):
        run_smoke(
            config,
            panel(tmp_path),
            tmp_path / "P1",
            "qwen35_9b",
            _adapter_factory=MismatchAfterFirstUpdate,
        )
    assert update.call_count == 1
