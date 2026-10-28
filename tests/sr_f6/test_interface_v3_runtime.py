"""Actual sampler adapter permissions, read-only tensors, and two-draw bounds."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from sr_f6.interface_v3_runtime import ScreenRuntime


@pytest.fixture
def runtime(tmp_path):
    runtime = ScreenRuntime.__new__(ScreenRuntime)
    runtime.torch = torch
    runtime.model = torch.nn.Linear(2, 2)
    for p in runtime.model.parameters():
        p.requires_grad_(False)
    runtime.identity = dict(
        step=0, phase="INTERFACE_SCREEN_V3", prompt_version="srf6-interface-v3", screen_stage=1
    )
    runtime._stage = 1
    runtime._bound_identity = copy.deepcopy(runtime.identity)
    runtime._policy_hash = "a" * 64
    runtime._parameter_versions = {
        n: (id(p), p._version) for n, p in runtime.model.named_parameters()
    }
    runtime._source_root = tmp_path
    runtime._permit = None
    runtime._attempted_qids = set()
    runtime._public = {"q": {"qid": "q", "screen_stage": 1, "model_input": {"prompt": "q"}}}
    runtime._seeds = {"q": [123, 456]}
    runtime.account = lambda *args: None
    return runtime


def test_direct_native_sampling_or_unregistered_input_is_closed(runtime, tmp_path):
    with pytest.raises(PermissionError, match="active"):
        runtime._generate_native_group({"prompt": "q"}, tmp_path, [123, 456])
    with pytest.raises(PermissionError, match="registered screen"):
        runtime.generate_group({"prompt": "q"}, tmp_path, [123, 456])
    with pytest.raises(PermissionError, match="active"):
        runtime.prepare_input({"prompt": "q"}, tmp_path)
    with pytest.raises(PermissionError):
        runtime._native_forward(None, grad=True)


@pytest.mark.parametrize("mutation", ["value", "gradient", "replacement", "identity"])
def test_frozen_parameters_and_identity_are_checked_without_rehash(runtime, mutation):
    assert runtime.policy_tensor_identity() == "a" * 64
    if mutation == "value":
        runtime.model.weight.add_(1)
    elif mutation == "gradient":
        runtime.model.weight.requires_grad_(True)
    elif mutation == "replacement":
        runtime.model.weight = torch.nn.Parameter(runtime.model.weight.clone(), requires_grad=False)
    else:
        runtime.identity["step"] = 1
    with pytest.raises(PermissionError, match="changed"):
        runtime.policy_tensor_identity()


def test_only_two_original_completion_reservations_and_no_extra_forward(runtime):
    runtime._permit = dict(attempted=False, token_rows=0)
    with pytest.raises(PermissionError):
        runtime.reserve("completion_attempts", 1)
    runtime.reserve("completion_attempts", 2)
    with pytest.raises(PermissionError):
        runtime.reserve("completion_attempts", 2)
    runtime.reserve("generated_tokens", 30)
    runtime.reserve("generated_tokens", 1536)
    with pytest.raises(PermissionError):
        runtime.reserve("generated_tokens", 1)
    with pytest.raises(PermissionError):
        runtime.reserve("extra_forward_sequences", 1)


def test_failed_attempt_cannot_resample_same_unit_and_clears_permit(runtime):
    def fail(*args, **kwargs):
        runtime.reserve("completion_attempts", 2)
        raise RuntimeError("retained technical failure")

    runtime._generate_native_group = fail
    with pytest.raises(RuntimeError, match="retained"):
        runtime.generate_registered("q", on_completion=lambda row: None)
    assert runtime._permit is None
    with pytest.raises(PermissionError, match="repeated"):
        runtime.generate_registered("q", on_completion=lambda row: None)


def test_exact_input_and_seed_guard_before_native_call(runtime, tmp_path):
    runtime._permit = dict(item=runtime._public["q"], seeds=[123, 456])
    with pytest.raises(PermissionError, match="frozen screen draws"):
        runtime._generate_native_group({"prompt": "q"}, tmp_path, [123, 789])
    with pytest.raises(PermissionError, match="registered screen unit"):
        runtime.prepare_input({"prompt": "private"}, tmp_path)


def test_actual_borrowed_native_body_retains_two_draws_and_fp32_recipe(
    runtime, tmp_path, monkeypatch
):
    from sr_f6 import runtime as native

    class Config(SimpleNamespace):
        def to_dict(self):
            return vars(self)

    def generate(input_ids, generation_config, **kwargs):
        assert input_ids.shape[0] == 2
        assert generation_config.max_new_tokens == 1536
        assert generation_config.temperature == 1.0
        logits = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        return SimpleNamespace(
            sequences=torch.cat((input_ids, torch.ones((2, 1), dtype=torch.long)), 1),
            logits=[logits],
            scores=[logits],
        )

    runtime.model.generate = generate
    runtime.generation_config = Config()
    runtime.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(
            pad_token_id=0,
            decode=lambda *args, **kwargs: '"answer":1}',
        )
    )
    runtime.protocol_revision = "v2"
    runtime.eos_ids = []
    runtime.image_calls = 0
    runtime.adapter_path = "registered-parent"
    runtime.identity.update(
        parent_id="b0",
        run_id="task",
        task_id="task",
        parent_content_hash="p",
        source_sha256="s",
        data_hash="d",
        registration_sha256="r",
    )
    runtime._bound_identity = copy.deepcopy(runtime.identity)
    runtime._public["q"]["model_input"].update(image_file=None, image_sha256=None)
    prepared = dict(inputs={"input_ids": torch.tensor([[0]])}, routing={"input_tensor_hash": "h"})
    runtime._srf5_prepared = prepared
    runtime.prepare_input = lambda *args, **kwargs: prepared
    monkeypatch.setattr(
        native,
        "BatchedProtocolStop",
        lambda *args, **kwargs: SimpleNamespace(counts=[1, 1], cuts=[12, 12]),
    )
    monkeypatch.setattr(native, "first_boundary_token", lambda *args, **kwargs: (1, 12))
    monkeypatch.setattr(native, "decode_generated", lambda *args: '"answer":1}')
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(StoppingCriteriaList=list))
    captured = []
    result = runtime.generate_registered("q", on_completion=captured.append)
    assert result == captured
    assert [row["seed"] for row in result] == [123, 456]
    assert all(row["phase"] == "INTERFACE_SCREEN_V3" for row in result)
    assert all(
        row["screen_stage"] == 1 and row["prompt_version"] == "srf6-interface-v3" for row in result
    )
    assert all(row["tokens"] == [1] and row["finish_reason"] == "balanced" for row in result)
    assert all(row["optimizer_updates"] == 0 and row["group_size"] == 2 for row in result)
    assert runtime._permit is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "INTERFACE_SCREEN_V2"),
        ("prompt_version", "srf6-interface-v2"),
        ("screen_stage", 2),
    ],
)
def test_wrong_scientific_identity_rejected_even_if_caller_copies_its_bound_snapshot(
    runtime, field, value
):
    runtime.identity[field] = value
    runtime._bound_identity = copy.deepcopy(runtime.identity)
    with pytest.raises(PermissionError, match="changed"):
        runtime.policy_tensor_identity()


def test_question_cannot_escape_its_stage(runtime):
    runtime._public["q"]["screen_stage"] = 2
    with pytest.raises(PermissionError, match="different registered screen stage"):
        runtime.generate_registered("q", on_completion=lambda row: None)
    assert runtime._permit is None


@pytest.mark.parametrize("stage", [True, 0, 3, "1", None])
def test_model_load_rejects_unregistered_stage_before_cuda(stage):
    from sr_f6.interface_v3_runtime import load_screen

    with pytest.raises(PermissionError, match="stage"):
        load_screen({"stage": stage}, "b0", "LH-SRF6-V3-S1-100-b0-0")


@pytest.mark.parametrize(
    "task",
    [
        "old-task",
        "LH-SRF6-V3-S2-100-b0-0",
        "LH-SRF6-V3-S1-100-b1-0",
        "LH-SRF6-V3-S1-100-b0-2",
        "../escape",
    ],
)
def test_model_load_task_identity_is_stage_parent_and_shard_bound(task):
    from sr_f6.interface_v3_runtime import load_screen

    with pytest.raises(ValueError, match="identifier"):
        load_screen({"stage": 1}, "b0", task)


def test_stage2_authorization_is_checked_before_model_construction(tmp_path, monkeypatch):
    from sr_f6 import interface_v3_registration as registration
    from sr_f6.interface_v3_runtime import load_screen

    calls = []

    def deny(base, **kwargs):
        calls.append((base, kwargs))
        raise PermissionError("No gray parent authorized")

    monkeypatch.setattr(registration, "verify_registered", deny)
    with pytest.raises(PermissionError, match="gray"):
        load_screen({"stage": 2, "registration_root": tmp_path}, "b1", "LH-SRF6-V3-S2-200-b1-0")
    assert calls == [(tmp_path, {"stage": 2, "parent": "b1", "require_preflight": True})]


def test_runtime_refuses_caller_stage_root_substitution(tmp_path, monkeypatch):
    from sr_f6 import interface_v3_registration as registration
    from sr_f6.interface_v3_runtime import load_screen

    context = dict(
        stage=1, registration_root=tmp_path, root=tmp_path / "STAGE2", registration_sha256="p"
    )
    verified = dict(context, root=tmp_path / "STAGE1")
    monkeypatch.setattr(registration, "verify_registered", lambda *args, **kwargs: verified)
    with pytest.raises(PermissionError, match="stage output root"):
        load_screen(context, "b0", "LH-SRF6-V3-S1-100-b0-0")
