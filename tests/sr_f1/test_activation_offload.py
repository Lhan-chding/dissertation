"""Exact CPU offload checks; actual 9B CUDA ENGINE qualification is separate."""

import pytest
import torch
import transformers
from peft import LoraConfig, get_peft_model

from mm_dev.runtime import F2Runtime
from sr_f1.runtime import SavedActivationOffload, SRRuntime
from sr_f1.training import sequence_objective


@pytest.fixture
def hybrid_runtime():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(71010)
    config = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=96,
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
    runtime = SRRuntime.__new__(SRRuntime)
    runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
    runtime.model = get_peft_model(
        transformers.Qwen3_5ForConditionalGeneration(config),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj", "v_proj"]),
    ).eval()
    with torch.no_grad():
        for name, parameter in runtime.model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.03)
    runtime.events = []
    runtime.account = lambda *args: runtime.events.append(args)
    hook = runtime._visual_module().register_forward_pre_hook(runtime._mark_vision)
    ids = torch.tensor([[3, 58, 60, 59, 4, 90]])
    prepared = dict(
        inputs=dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=torch.full((4, 24), 0.5),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]]),
        ),
        routing=dict(image_token_count=1),
    )
    yield runtime, prepared
    hook.remove()
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prompt_padding", [0, 64])
def test_offload_keeps_exact_logprobs_and_dual_backward_lora_gradients(
    hybrid_runtime, dtype, prompt_padding
):
    runtime, prepared = hybrid_runtime
    runtime.model.to(dtype=dtype)
    if prompt_padding:
        inputs = prepared["inputs"]
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.full((1, prompt_padding), 5)], dim=1
        )
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros((1, prompt_padding), dtype=torch.long)],
            dim=1,
        )
    for parameter in runtime.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    original_values = [p.detach().clone() for p in parameters]
    tokens = [6, 7, 8, 9, 2]

    def measure(forward):
        runtime.model.zero_grad(set_to_none=True)
        result = forward(prepared, tokens, purpose="offload_cpu_fixture", grad=True)
        values = result["logprobs"]
        # Nonzero KL plus policy terms exercise the actual two-backward pattern.
        objective = sequence_objective(values, values.detach(), values.detach() + 0.03, -1.0)
        policy_grads = torch.autograd.grad(
            objective["policy"] / 128, parameters, retain_graph=True, allow_unused=False
        )
        (objective["loss"] / 128).backward()
        return result, policy_grads, [p.grad.detach().clone() for p in parameters]

    baseline, original_pg, original_grad = measure(
        lambda *args, **kwargs: F2Runtime.cached_training_forward(runtime, *args, **kwargs)
    )
    offloaded, copied_pg, copied_grad = measure(runtime.cached_training_forward)
    assert torch.equal(baseline["logprobs"], offloaded["logprobs"])
    assert offloaded["cached_model_forward_calls"] == len(tokens)
    assert offloaded["vision_forward_calls"] == 1
    for actual, expected, total, expected_total, parameter, original in zip(
        copied_pg, original_pg, copied_grad, original_grad, parameters, original_values, strict=True
    ):
        assert torch.equal(actual, expected)
        assert torch.equal(total, expected_total)
        assert torch.isfinite(total).all() and total.norm() > 0
        assert torch.equal(parameter, original)
    evidence = offloaded["activation_offload"]
    assert evidence["saved_activation_tensors"] > 0
    assert evidence["saved_activation_bytes"] > 0
    assert evidence["resident_parameter_views"] > 0
    assert evidence["unpacked_activation_tensors"] > evidence["saved_activation_tensors"]
    assert evidence["unpacked_activation_bytes"] > evidence["saved_activation_bytes"]
    rows = [args[2] for args in runtime.events if args[0] == "cached_training_model_forward_calls"]
    assert (
        rows[-1]["activation_offload"]["saved_activation_bytes"]
        == evidence["saved_activation_bytes"]
    )
    # Durable forward accounting is a snapshot, not mutated by later backward.
    assert rows[-1]["activation_offload"]["unpacked_activation_bytes"] == 0


def test_last_token_loss_retains_prompt_and_recurrent_history(hybrid_runtime):
    runtime, prepared = hybrid_runtime
    embedding = runtime.model.get_input_embeddings().weight
    embedding.requires_grad_(True)
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    tokens = [6, 7, 8, 9, 2]
    baseline = F2Runtime.cached_training_forward(
        runtime, prepared, tokens, purpose="history_baseline", grad=True
    )
    baseline_grad = torch.autograd.grad(baseline["logprobs"][-1], parameters)
    offloaded = runtime.cached_training_forward(
        prepared, tokens, purpose="history_offload", grad=True
    )
    actual_grad = torch.autograd.grad(offloaded["logprobs"][-1], parameters)
    for actual, expected in zip(actual_grad, baseline_grad, strict=True):
        assert torch.equal(actual, expected)
    embedding_index = next(i for i, parameter in enumerate(parameters) if parameter is embedding)
    # IDs 3 and 90 occur only in the prompt; 6 only in an earlier decoded step.
    for token in (3, 90, 6):
        assert actual_grad[embedding_index][token].norm() > 0


def test_parameter_views_stay_resident_and_mutation_is_rejected():
    model = torch.nn.Linear(3, 2, bias=False)
    offload = SavedActivationOffload(model)
    packed = offload.pack(model.weight.T)
    assert packed[0] == "parameter"
    assert packed[1].untyped_storage().data_ptr() == model.weight.untyped_storage().data_ptr()
    assert offload.statistics["saved_activation_bytes"] == 0
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(RuntimeError, match="parameter changed"):
        offload.unpack(packed)


def test_activation_snapshot_preserves_dtype_shape_stride_values_and_grad_edges():
    model = torch.nn.Linear(2, 2)
    source = torch.randn(3, 7, requires_grad=True)
    value = source.T
    offload = SavedActivationOffload(model)
    packed = offload.pack(value)
    restored = offload.unpack(packed)
    assert packed[0] == "activation"
    assert restored.data_ptr() != value.data_ptr()
    assert restored.dtype == value.dtype and restored.shape == value.shape
    assert restored.stride() == value.stride() and torch.equal(restored, value)
    with offload:
        recurrent = source.square()
        for _ in range(4):
            recurrent = (recurrent + source).tanh()
        loss = recurrent.sum()
    actual = torch.autograd.grad(loss, source)[0]
    expected_state = source.square()
    for _ in range(4):
        expected_state = (expected_state + source).tanh()
    expected = torch.autograd.grad(expected_state.sum(), source)[0]
    assert torch.equal(actual, expected)


def test_no_grad_reference_path_does_not_offload(hybrid_runtime):
    runtime, prepared = hybrid_runtime
    result = runtime.cached_training_forward(
        prepared, [6, 7, 2], purpose="reference_no_grad", grad=False
    )
    assert result["activation_offload"] is None
    assert not result["logprobs"].requires_grad
    assert result["entropy"] is not None


def test_partial_forward_failure_retains_offload_accounting(hybrid_runtime, monkeypatch):
    runtime, prepared = hybrid_runtime
    original = runtime.model.forward
    count = 0

    def failing_forward(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("simulated allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.model, "forward", failing_forward)
    with pytest.raises(RuntimeError, match="simulated allocation"):
        runtime.cached_training_forward(prepared, [6, 7, 2], purpose="failure_fixture", grad=True)
    final = runtime.events[-1][2]
    assert final["status"] == "TECHNICAL_FAILED"
    assert final["scored_completion_tokens"] == 1
    assert final["activation_offload"]["saved_activation_bytes"] > 0
