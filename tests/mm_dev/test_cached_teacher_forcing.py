"""Native tiny Qwen checks; these cannot certify the real 9B CUDA ENGINE gate."""

from types import SimpleNamespace

import pytest
import torch
import transformers
from peft import LoraConfig, get_peft_model

from mm_core.training import trainable_state
from mm_core.vl_runtime import QwenRuntime
from mm_dev.runtime import F2Runtime
from mm_dev.training import sequence_objective


@pytest.fixture
def native_runtime():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(61009)
    config = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            layer_types=[
                "full_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            rope_parameters=dict(
                rope_type="default",
                mrope_section=[1, 1, 2],
                mrope_interleaved=True,
                partial_rotary_factor=1.0,
            ),
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
    )
    config._attn_implementation = "eager"
    runtime = F2Runtime.__new__(F2Runtime)
    runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
    runtime.model = get_peft_model(
        transformers.Qwen3_5ForConditionalGeneration(config),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj", "v_proj"]),
    ).eval()
    # Nonzero B exercises the whole LoRA history, including gradients to A.
    with torch.no_grad():
        for name, parameter in runtime.model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.03)
    events = []
    runtime.account = lambda *args: events.append(args)
    runtime.eos_ids = [2]
    runtime.generation_config = transformers.GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=192,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )
    generation = runtime.training_generation_config()
    generation.max_new_tokens = 7  # CPU architecture fixture only.
    runtime.training_generation_config = lambda: generation
    runtime.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda tokens, **_kwargs: str(tokens))
    )
    hook = runtime._visual_module().register_forward_pre_hook(runtime._mark_vision)
    ids = torch.tensor([[3, 58, 60, 59, 4, 5]])
    inputs = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        pixel_values=torch.full((4, 24), 0.5),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]]),
    )
    prepared = dict(inputs=inputs, routing={})
    runtime.prepare = lambda *_args: prepared
    yield runtime, prepared, events
    hook.remove()
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_training_scores_actual_raw_generation_with_same_native_path(native_runtime, dtype):
    runtime, prepared, _events = native_runtime
    runtime.model.to(dtype=dtype)
    for parameter in runtime.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    record = runtime.generate_training(dict(question_id="cached-native-fixture"), None, 19)
    assert record["generation_status"] == "COMPLETE"
    assert record["old_logprobs"] == record["sampler_logprobs"]
    for purpose, grad in [("training_reference", False), ("training_gradient", True)]:
        scored = runtime.sequence_forward(prepared, record["tokens"], purpose=purpose, grad=grad)
        torch.testing.assert_close(
            scored["logprobs"].detach(), torch.tensor(record["old_logprobs"]), atol=0, rtol=0
        )
        assert scored["logprobs"].requires_grad is grad
        assert scored["vision_forward_calls"] == 1
        assert scored["cached_model_forward_calls"] == len(record["tokens"])
        if grad:
            scored["logprobs"].sum().backward()
            for parameter in runtime.model.parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_generated_eos_is_scored_without_an_extra_decode(native_runtime):
    runtime, prepared, _events = native_runtime
    # Every vocabulary token is EOS in this fixture, so the true sampler ends at
    # its first raw draw without forcing/filtering any token's logits.
    runtime.generation_config.eos_token_id = list(range(64))
    runtime.eos_ids = list(range(64))
    runtime.training_generation_config().eos_token_id = runtime.eos_ids
    record = runtime.generate_training(dict(question_id="cached-eos-fixture"), None, 29)
    assert record["finish_reason"] == "eos" and len(record["tokens"]) == 1
    scored = runtime.sequence_forward(
        prepared, record["tokens"], purpose="training_gradient", grad=True
    )
    assert scored["cached_model_forward_calls"] == 1
    torch.testing.assert_close(
        scored["logprobs"].detach(), torch.tensor(record["old_logprobs"]), atol=0, rtol=0
    )


@pytest.mark.parametrize("advantage", [-1.0, 1.0])
@pytest.mark.parametrize("prompt_padding", [0, 64])
def test_cached_lora_gradients_match_full_prefix_and_include_eos(
    native_runtime, advantage, prompt_padding
):
    runtime, prepared, _events = native_runtime
    if prompt_padding:
        prepared["inputs"]["input_ids"] = torch.cat(
            [prepared["inputs"]["input_ids"], torch.full((1, prompt_padding), 5)], dim=1
        )
        prepared["inputs"]["mm_token_type_ids"] = torch.cat(
            [
                prepared["inputs"]["mm_token_type_ids"],
                torch.zeros((1, prompt_padding), dtype=torch.long),
            ],
            dim=1,
        )
        prepared["inputs"]["attention_mask"] = torch.ones_like(prepared["inputs"]["input_ids"])
    tokens = [6, 7, 8, 9, 2]
    parameters = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    full = QwenRuntime.sequence_forward(
        runtime, prepared, tokens, purpose="fixture_full", grad=True
    )
    old = full["logprobs"].detach()
    full_loss = sequence_objective(full["logprobs"], old, old, advantage)["loss"]
    full_grads = torch.autograd.grad(full_loss, parameters)
    cached = runtime.sequence_forward(prepared, tokens, purpose="training_gradient", grad=True)
    cached_loss = sequence_objective(cached["logprobs"], old, old, advantage)["loss"]
    cached_grads = torch.autograd.grad(cached_loss, parameters)
    assert len(cached["logprobs"]) == len(tokens)
    assert cached["cached_model_forward_calls"] == len(tokens)
    torch.testing.assert_close(cached["logprobs"], old, atol=2e-6, rtol=2e-6)
    for full_grad, cached_grad in zip(full_grads, cached_grads, strict=True):
        assert full_grad.norm() > 0
        torch.testing.assert_close(cached_grad, full_grad, atol=2e-6, rtol=2e-4)


def test_final_token_gradient_reaches_prompt_and_matches_full_prefix(native_runtime, monkeypatch):
    runtime, prepared, _events = native_runtime
    base = runtime.model.get_base_model()
    prompt_length = prepared["inputs"]["input_ids"].shape[1]
    tokens = [6, 7, 8, 9, 2]
    embeddings = []

    def retain_embeddings(_module, _args, output):
        output.requires_grad_(True)
        output.retain_grad()
        embeddings.append(output)

    hook = base.get_input_embeddings().register_forward_hook(retain_embeddings)
    try:
        full = QwenRuntime.sequence_forward(
            runtime, prepared, tokens, purpose="fixture_full", grad=True
        )
        full["logprobs"][-1].backward()
        expected = embeddings[0].grad[:, :prompt_length].clone()
        assert expected.norm() > 0
        embeddings.clear()
        runtime.model.zero_grad(set_to_none=True)
        cached = runtime.sequence_forward(prepared, tokens, purpose="training_gradient", grad=True)
        cached["logprobs"][-1].backward()
        assert embeddings[0].grad is not None and embeddings[0].grad.norm() > 0
        torch.testing.assert_close(embeddings[0].grad, expected, atol=2e-6, rtol=2e-4)

        # A deliberately detached-history control must fail this property.
        original_prepare = base.prepare_inputs_for_generation

        def detached_prepare(*args, **kwargs):
            cache = kwargs.get("past_key_values")
            if cache is not None and cache.get_seq_length() > 0:
                for layer in cache.layers:
                    for name in ("keys", "values"):
                        value = getattr(layer, name, None)
                        if value is not None:
                            setattr(layer, name, value.detach())
                    for name in ("conv_states", "recurrent_states"):
                        for key, value in getattr(layer, name, {}).items():
                            if value is not None:
                                getattr(layer, name)[key] = value.detach()
            return original_prepare(*args, **kwargs)

        monkeypatch.setattr(base, "prepare_inputs_for_generation", detached_prepare)
        embeddings.clear()
        runtime.model.zero_grad(set_to_none=True)
        detached = runtime.sequence_forward(
            prepared, tokens, purpose="training_gradient", grad=True
        )
        detached["logprobs"][-1].backward()
        assert embeddings[0].grad is None or embeddings[0].grad.count_nonzero() == 0
    finally:
        hook.remove()


def test_cached_training_restores_checkpoint_flags_and_probe_uses_core_path(
    native_runtime, monkeypatch
):
    runtime, prepared, _events = native_runtime
    base = runtime.model.get_base_model()
    base.gradient_checkpointing_enable()
    original_flags = {
        name: module.gradient_checkpointing
        for name, module in runtime.model.named_modules()
        if hasattr(module, "gradient_checkpointing")
    }
    assert any(original_flags.values())
    reference = trainable_state(runtime.model)
    scored = runtime.reference_forward(prepared, [6, 7, 2], reference)
    current = runtime.sequence_forward(prepared, [6, 7, 2], purpose="training_gradient", grad=True)
    current["logprobs"].sum().backward()
    assert torch.equal(scored["logprobs"], current["logprobs"].detach())
    assert original_flags == {
        name: module.gradient_checkpointing
        for name, module in runtime.model.named_modules()
        if hasattr(module, "gradient_checkpointing")
    }
    sentinel = object()
    monkeypatch.setattr(QwenRuntime, "sequence_forward", lambda *_args, **_kwargs: sentinel)
    assert runtime.sequence_forward(prepared, [6, 2], purpose="probe_self") is sentinel


def test_cached_training_restores_checkpoint_flags_on_exception(native_runtime, monkeypatch):
    runtime, prepared, _events = native_runtime
    base = runtime.model.get_base_model()
    base.gradient_checkpointing_enable()
    modules = [
        module for module in runtime.model.modules() if hasattr(module, "gradient_checkpointing")
    ]
    original = [module.gradient_checkpointing for module in modules]

    def fail(*_args, **_kwargs):
        assert not any(module.gradient_checkpointing for module in modules)
        raise RuntimeError("fixture forward failure")

    monkeypatch.setattr(base, "forward", fail)
    with pytest.raises(RuntimeError, match="fixture forward failure"):
        runtime.sequence_forward(prepared, [6, 2], purpose="training_gradient", grad=True)
    assert [module.gradient_checkpointing for module in modules] == original


@pytest.mark.parametrize("fail_on_call", [None, 1, 2])
def test_cached_forward_accounting_retains_actual_calls_on_success_and_failure(
    native_runtime, monkeypatch, fail_on_call
):
    runtime, prepared, events = native_runtime
    base = runtime.model.get_base_model()
    original_forward = base.forward
    calls = 0

    def tracked_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            raise RuntimeError("fixture counted forward failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(base, "forward", tracked_forward)
    if fail_on_call:
        with pytest.raises(RuntimeError, match="fixture counted forward failure"):
            runtime.sequence_forward(prepared, [6, 7, 2], purpose="training_gradient", grad=True)
    else:
        runtime.sequence_forward(prepared, [6, 7, 2], purpose="training_gradient", grad=True)
    assert events[0][0:2] == ("gradient_forward_sequences", 1)
    rows = [event for event in events if event[0] == "cached_training_model_forward_calls"]
    assert len(rows) == 1
    _kind, count, metadata = rows[0]
    attempted = fail_on_call or 3
    completed = fail_on_call - 1 if fail_on_call else 3
    assert count == attempted
    assert metadata["prefill_model_forward_calls"] == 1
    assert metadata["decode_model_forward_calls"] == attempted - 1
    assert metadata["prefill_input_tokens"] == prepared["inputs"]["input_ids"].numel()
    assert metadata["decode_input_tokens"] == attempted - 1
    assert metadata["completed_model_forward_calls"] == completed
    assert metadata["scored_completion_tokens"] == completed
    assert metadata["requested_completion_tokens"] == 3
    assert metadata["vision_forward_calls"] == int(fail_on_call != 1)
    assert metadata["status"] == ("TECHNICAL_FAILED" if fail_on_call else "COMPLETE")
