from types import SimpleNamespace

import pytest
import torch

from sr_f12.runtime import (
    BatchedBalancedJSONStop,
    SRF12Runtime,
    _repeat_prompt,
    common_zero_check,
    configure_training,
    language_linear_modules,
)


@pytest.fixture
def native_runtime():
    import transformers

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
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
    runtime = SRF12Runtime.__new__(SRF12Runtime)
    runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
    runtime.model = transformers.Qwen3_5ForConditionalGeneration(config).eval()
    runtime.identity, runtime.account = {}, lambda *args: None
    runtime.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    runtime.eos_ids = [2]
    hook = runtime._visual_module().register_forward_pre_hook(runtime._mark_vision)
    ids = torch.tensor([[3, 58, 60, 59, 4, 90]])
    prepared = dict(
        inputs=dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]]),
            pixel_values=torch.full((4, 24), 0.5),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        ),
        routing=dict(image_token_count=1, input_tensor_hash="same"),
    )
    yield runtime, prepared
    hook.remove()
    torch.set_num_threads(old_threads)


def test_native_whole_sequence_right_padding_matches_single_and_gradients(native_runtime):
    runtime, prepared = native_runtime
    configure_training(runtime)
    records = [dict(prepared=prepared, tokens=tokens) for tokens in ([5, 6, 7], [9], [8, 12])]
    batch = runtime.batch_sequence_forward(records)
    single = [
        runtime.sequence_forward(prepared, item["tokens"], purpose="single", grad=True)["logprobs"]
        for item in records
    ]
    for actual, expected in zip(batch, single, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    sum(value.mean() for value in batch).backward()
    assert any(
        p.grad is not None and bool(p.grad.any())
        for p in runtime.model.parameters()
        if p.requires_grad
    )
    assert runtime.model.get_base_model().model.rope_deltas is None


def test_lora_all_decoder_linears_and_zero_base_exact32(native_runtime):
    runtime, prepared = native_runtime
    expected = language_linear_modules(runtime.model)
    optimizer, _scheduler, receipt = configure_training(runtime)
    assert receipt["target_modules"] == expected
    assert any("linear_attn" in name for name in expected)
    assert any("mlp" in name for name in expected)
    assert any("k_proj" in name for name in expected)
    assert all("language_model.layers." in name for name in receipt["trainable_names"])
    assert optimizer.param_groups[0]["lr"] == 5e-5
    check = common_zero_check(runtime, [dict(prepared=prepared, tokens=[5, 6])] * 32)
    assert check["status"] == "PASS" and check["maximum_logprob_difference"] == 0
    # Existing full-range adapter can be restored without accidentally narrowing it.
    assert configure_training(runtime, learning_rate=2e-5)[2]["target_modules"] == expected


def test_reject_narrow_old_adapter(native_runtime):
    from peft import LoraConfig, get_peft_model

    runtime, _ = native_runtime
    runtime.model = get_peft_model(
        runtime.model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    )
    with pytest.raises(PermissionError, match="omits"):
        configure_training(runtime)


def test_microbatch_prompt_mismatch_rejected(native_runtime):
    runtime, prepared = native_runtime
    different = dict(prepared, inputs=dict(prepared["inputs"]))
    different["inputs"]["input_ids"] = prepared["inputs"]["input_ids"] + 1
    with pytest.raises(PermissionError, match="identical"):
        runtime.batch_sequence_forward(
            [dict(prepared=prepared, tokens=[5]), dict(prepared=different, tokens=[6])]
        )


def test_rowwise_json_stop_retains_first_boundary():
    tokenizer = SimpleNamespace(
        decode=lambda tokens, **kwargs: "".join(
            {5: '"a":', 6: "1", 7: "}", 0: "PAD"}[x] for x in tokens
        )
    )
    stop = BatchedBalancedJSONStop(tokenizer, 1, 2)
    assert stop(torch.tensor([[90, 7], [90, 5]]), None).tolist() == [True, False]
    assert stop(torch.tensor([[90, 7, 0], [90, 5, 7]]), None).tolist() == [True, True]
    assert stop.counts == [1, 2]


def test_repeat_image_patches_and_grid_preserves_row_order(native_runtime):
    _, prepared = native_runtime
    inputs = _repeat_prompt(prepared["inputs"], 8)
    assert inputs["pixel_values"].shape == (32, 24)
    assert inputs["image_grid_thw"].shape == (8, 3)
    assert inputs["input_ids"].shape == (8, 6)
    for row in range(8):
        assert torch.equal(
            inputs["pixel_values"][row * 4 : (row + 1) * 4], prepared["inputs"]["pixel_values"]
        )


def test_generation_rows_cut_at_own_stop_and_save_actual_sampler_probabilities(
    native_runtime, monkeypatch
):
    from transformers import GenerationConfig

    from sr_f1.runtime import generation_recipe

    runtime, prepared = native_runtime
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        decode=lambda tokens, **kwargs: "".join(
            {5: '"a":', 6: "1", 7: "}", 0: "PAD"}[x] for x in tokens
        ),
    )
    runtime.processor.tokenizer = tokenizer
    runtime.generation_config = GenerationConfig(
        **generation_recipe(), eos_token_id=2, pad_token_id=0
    )
    monkeypatch.setattr(runtime, "prepare", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(runtime, "current_adapter_identity", lambda: {})
    monkeypatch.setattr(runtime, "stable_model_identity", lambda: {})

    def generate(**kwargs):
        inputs = kwargs["input_ids"]
        assert inputs.shape == (2, 6)
        seq = inputs
        for pair in ([7, 5], [0, 6], [0, 7]):
            seq = torch.cat([seq, torch.tensor(pair)[:, None]], -1)
            kwargs["stopping_criteria"](seq, None)
        runtime.image_calls += 1
        logits = tuple(torch.arange(96, dtype=torch.float32).repeat(2, 1) for _ in range(3))
        return SimpleNamespace(sequences=seq, logits=logits, scores=logits)

    monkeypatch.setattr(runtime.model, "generate", generate)
    initial_rng = torch.get_rng_state().clone()
    records = runtime.generate_group({"qid": "q"}, "/unused", [100, 101])
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert [r["tokens"] for r in records] == [[7], [5, 6, 7]]
    assert [len(r["sampler_logprobs"]) for r in records] == [1, 3]
    assert all(r["finish_reason"] == "balanced" and not r["truncated"] for r in records)
    assert all(r["group_seed"] == 100 and "old_logprobs" not in r for r in records)
    assert records[0]["prompt_tensor_hash"] == records[1]["prompt_tensor_hash"]


def test_real_native_generate_eight_rows_with_image_expansion(native_runtime, monkeypatch):
    from transformers import GenerationConfig

    from sr_f1.runtime import generation_recipe

    runtime, prepared = native_runtime
    configure_training(runtime)
    runtime.processor.tokenizer = SimpleNamespace(
        pad_token_id=0, decode=lambda tokens, **kwargs: "}" * len(tokens)
    )
    runtime.generation_config = GenerationConfig(
        **generation_recipe(), eos_token_id=None, pad_token_id=0
    )
    runtime.eos_ids = []
    monkeypatch.setattr(runtime, "prepare", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(runtime, "current_adapter_identity", lambda: {})
    monkeypatch.setattr(runtime, "stable_model_identity", lambda: {})
    records = runtime.generate_group({"qid": "q"}, "/unused", list(range(8)))
    assert len(records) == 8
    assert all(
        record["finish_reason"] == "balanced" and len(record["tokens"]) == 1 for record in records
    )
    assert all(len(record["sampler_logprobs"]) == 1 for record in records)
    assert runtime.image_calls > 0


def test_answer_protocol_passes_same_prefill_boundary_but_preserves_actual_prompt(monkeypatch):
    from sr_f1.runtime import SRRuntime

    seen = []

    def parent(self, text, image_path, run_root, **kwargs):
        seen.append((text, kwargs["protocol"]))
        return {"routing": {"assistant_prefill": "{", "assistant_prefill_token_ids": [90]}}

    monkeypatch.setattr(SRRuntime, "prepare_text", parent)
    runtime = SRF12Runtime.__new__(SRF12Runtime)
    prepared = runtime.prepare_text(
        "answer-only actual prompt", None, "/unused", protocol="answer_only"
    )
    assert seen == [("answer-only actual prompt", "evidence_answer")]
    assert prepared["routing"]["protocol"] == "answer_only"
    assert prepared["routing"]["assistant_prefill_token_ids"] == [90]


def test_checkpoint_fallback_preserves_native_train_eval_probabilities(native_runtime):
    from sr_f12.runtime import enable_gradient_checkpointing

    runtime, prepared = native_runtime
    configure_training(runtime)
    cases = [dict(prepared=prepared, tokens=[5, 6, 7])]
    assert enable_gradient_checkpointing(runtime, cases)["train_eval_bitwise_equal"]
    runtime.batch_sequence_forward(cases)[0].mean().backward()
    assert any(
        p.grad is not None and bool(p.grad.any())
        for p in runtime.model.parameters()
        if p.requires_grad
    )


def test_chartqa_single_greedy_128_and_eos_boundary(native_runtime, monkeypatch):
    from transformers import GenerationConfig

    from sr_f1.runtime import generation_recipe

    runtime, prepared = native_runtime
    runtime.processor.tokenizer = SimpleNamespace(
        pad_token_id=0, decode=lambda tokens, **kwargs: "42"
    )
    runtime.generation_config = GenerationConfig(
        **generation_recipe(), eos_token_id=2, pad_token_id=0
    )
    monkeypatch.setattr(runtime, "prepare_text", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(runtime, "current_adapter_identity", lambda: {})
    monkeypatch.setattr(runtime, "stable_model_identity", lambda: {})

    def generate(**kwargs):
        cfg = kwargs["generation_config"]
        assert cfg.max_new_tokens == 128 and cfg.do_sample is False
        runtime.image_calls += 1
        raw = (torch.zeros(1, 96),)
        return SimpleNamespace(
            sequences=torch.cat([kwargs["input_ids"], torch.tensor([[2]])], 1),
            logits=raw,
            scores=raw,
        )

    monkeypatch.setattr(runtime.model, "generate", generate)
    record = runtime.generate(None, "/unused", 17, text="question", protocol="plain_answer")
    assert record["finish_reason"] == "eos" and record["tokens"] == [2]
    assert record["raw_text"] == "42" and "assistant_prefill" not in record
    with pytest.raises(PermissionError, match="single-row greedy"):
        runtime.generate_group(None, "/unused", [17, 18], text="question", protocol="plain_answer")


def test_evidence_cannot_reduce_registered_token_cap(native_runtime, monkeypatch):
    from transformers import GenerationConfig

    from sr_f1.runtime import generation_recipe

    runtime, prepared = native_runtime
    runtime.generation_config = GenerationConfig(
        **generation_recipe(), eos_token_id=2, pad_token_id=0
    )
    monkeypatch.setattr(runtime, "prepare", lambda *args, **kwargs: prepared)
    with pytest.raises(PermissionError, match="token cap"):
        runtime.generate_group({"qid": "q"}, "/unused", [1], generation={"max_new_tokens": 128})
