import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import state_hash
from sr_f1.json_protocol import AMENDMENT_ID
from sr_f3_ce.masks import target_spans
from sr_f3_ce.runtime import (
    BatchedProtocolStop,
    SRF3CERuntime,
    configure_parent_training,
    decoded_protocol,
    enable_gradient_checkpointing,
    first_boundary_token,
    generation_recipe,
    native_readability_measurements,
    validate_model_input,
)


@pytest.fixture
def native_runtime():
    import transformers

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=128,
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
    runtime = SRF3CERuntime.__new__(SRF3CERuntime)
    runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
    runtime.model = transformers.Qwen3_5ForConditionalGeneration(config).eval()
    runtime.identity, runtime.account, runtime.adapter_path = {}, lambda *args: None, None
    runtime.protocol_amendment = {"id": AMENDMENT_ID}
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


def safe_input():
    return {"prompt": "frozen question", "image_file": None, "image_sha256": None}


def test_only_frozen_three_fields_enter_runtime():
    assert validate_model_input(safe_input()) == safe_input()
    for extra in ("qid", "pool", "answer", "gold", "world", "certificate"):
        with pytest.raises(PermissionError, match="only"):
            validate_model_input({**safe_input(), extra: "private"})
    with pytest.raises(ValueError, match="SHA"):
        validate_model_input({**safe_input(), "image_file": "image.png"})
    runtime = SRF3CERuntime.__new__(SRF3CERuntime)
    with pytest.raises(PermissionError, match="ModelInputStore"):
        runtime.prepare_text("arbitrary", None, "/unused")


def test_protocol_recipes_are_fixed_and_external_greedy():
    for protocol in ("structured", "answer_only", "natural_work"):
        recipe = generation_recipe(protocol=protocol)
        assert recipe["max_new_tokens"] == 1024 and recipe["do_sample"]
        assert (recipe["temperature"], recipe["top_p"], recipe["top_k"]) == (1, 1, 0)
        assert recipe["forced_eos_token_id"] is None
    assert generation_recipe(protocol="chartqa_plain")["max_new_tokens"] == 256
    assert not generation_recipe(protocol="chartqapro_official")["do_sample"]


def test_token_attached_suffix_preserved_and_json_strings_do_not_stop_early():
    tokenizer = SimpleNamespace(
        decode=lambda ids, **kw: "".join({1: '"work":"}\\"",', 2: '"answer":1}BAD'}[i] for i in ids)
    )
    assert first_boundary_token(tokenizer, [1, 2], "structured")[0] == 2
    result = decoded_protocol(tokenizer.decode([1, 2]), "structured")
    assert result["raw_text"].endswith("}BAD")
    assert result["format_protocol_error"] == "trailing_after_close"
    assert result["full_text"] == result["raw_text"]
    assert result["assistant_prefill"] == "{"


def test_natural_stop_is_first_closed_tag_and_keeps_invalid_suffix():
    tokenizer = SimpleNamespace(
        decode=lambda ids, **kw: "".join(
            {5: "work <final>3", 6: "</final>suffix", 0: "PAD"}[i] for i in ids
        )
    )
    stop = BatchedProtocolStop(tokenizer, 1, 2, "natural_work")
    assert stop(torch.tensor([[90, 6], [90, 5]]), None).tolist() == [True, False]
    assert stop(torch.tensor([[90, 6, 0], [90, 5, 6]]), None).tolist() == [True, True]
    assert stop.counts == [1, 2]
    output = decoded_protocol("work <final>3</final>suffix", "natural_work")
    assert output["prefill"] == "" and output["format_protocol_error"] == "trailing_after_close"
    assert (
        decoded_protocol("work <final>3</final>", "natural_work")["format_protocol_error"] is None
    )
    assert (
        decoded_protocol("work </final>", "natural_work")["format_protocol_error"]
        == "invalid_final_tag_count_or_order"
    )


def attach_parent(runtime):
    from peft import LoraConfig, get_peft_model

    targets = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in runtime.model.named_modules()
            if isinstance(module, torch.nn.Linear) and "language_model.layers." in name
        }
    )
    runtime.model = get_peft_model(
        runtime.model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    runtime.adapter_path = "/verified/j6-parent"
    config = dict(r=8, lora_alpha=16, lora_dropout=0, bias="none", target_modules=targets)
    parameter_hash = state_hash(
        {
            name: value.detach()
            for name, value in runtime.model.named_parameters()
            if ".lora_" in name
        }
    )
    return config, parameter_hash


def test_exact_loaded_suffix_parent_expands_to_all_actual_modules(native_runtime):
    runtime, _ = native_runtime
    config, digest = attach_parent(runtime)
    receipt = configure_parent_training(
        runtime, parent_adapter_config=config, expected_adapter_state_hash=digest
    )
    assert receipt["parent_parameter_hash"] == digest
    assert len(receipt["expanded_target_modules"]) >= len(config["target_modules"])
    assert not receipt["new_adapter_initialized"]
    assert any("linear_attn" in name for name in receipt["expanded_target_modules"])
    assert all("language_model.layers." in name for name in receipt["trainable_names"])
    with pytest.raises(PermissionError, match="identity"):
        configure_parent_training(
            runtime, parent_adapter_config=config, expected_adapter_state_hash="0" * 64
        )
    with pytest.raises(PermissionError, match="target module"):
        configure_parent_training(
            runtime, parent_adapter_config={**config, "target_modules": ["q_proj"]}
        )


def test_no_random_lora_or_narrow_parent_is_permitted(native_runtime):
    runtime, _ = native_runtime
    with pytest.raises(PermissionError, match="already be loaded"):
        configure_parent_training(runtime, parent_adapter_config={})
    from peft import LoraConfig, get_peft_model

    runtime.model = get_peft_model(
        runtime.model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    )
    runtime.adapter_path = "/verified/wrong-parent"
    with pytest.raises(PermissionError, match="omits"):
        configure_parent_training(
            runtime,
            parent_adapter_config=dict(
                r=8, lora_alpha=16, lora_dropout=0, bias="none", target_modules=["q_proj", "v_proj"]
            ),
        )


def test_native_multimodal_batch_probabilities_gradients_and_long_completion(native_runtime):
    runtime, prepared = native_runtime
    config, digest = attach_parent(runtime)
    configure_parent_training(
        runtime, parent_adapter_config=config, expected_adapter_state_hash=digest
    )
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
    # A real native full forward beyond the old 768-token boundary.
    result = runtime.sequence_forward(prepared, [5] * 769, purpose="long_native", grad=False)
    assert result["logprobs"].shape == (769,)
    with pytest.raises(ValueError, match="1024"):
        runtime.sequence_forward(prepared, [5] * 1025, purpose="invalid", grad=False)


def test_native_support_forward_uses_full_sequence_and_nonzero_field_ce(native_runtime):
    runtime, prepared = native_runtime
    config, digest = attach_parent(runtime)
    configure_parent_training(
        runtime, parent_adapter_config=config, expected_adapter_state_hash=digest
    )
    tokens = [5, 6, 7, 8]
    inputs = dict(prepared["inputs"])
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([tokens])], -1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    inputs["mm_token_type_ids"] = torch.cat(
        [inputs["mm_token_type_ids"], torch.zeros(1, len(tokens), dtype=torch.long)], -1
    )
    record = dict(
        full_inputs=inputs,
        tokens=tokens,
        completion_start_index=6,
        answer_mask=[0, 0, 1, 1],
        work_mask=[1, 1, 0, 0],
    )
    support = runtime.batch_support_forward([record])[0]
    task = runtime.sequence_forward(prepared, tokens, purpose="comparison", grad=False)["logprobs"]
    torch.testing.assert_close(support.detach(), task, rtol=0, atol=0)
    loss = -0.5 * support[:2].mean() - 0.5 * support[2:].mean()
    loss.backward()
    assert any(
        p.grad is not None and bool(p.grad.any())
        for p in runtime.model.parameters()
        if p.requires_grad
    )


def generation_setup(runtime, prepared, monkeypatch, pieces):
    from transformers import GenerationConfig

    runtime.processor.tokenizer = SimpleNamespace(
        pad_token_id=0, decode=lambda ids, **kw: "".join(pieces.get(i, "") for i in ids)
    )
    runtime.generation_config = GenerationConfig(
        **generation_recipe(), eos_token_id=2, pad_token_id=0
    )
    monkeypatch.setattr(runtime, "prepare_input", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(runtime, "current_adapter_identity", lambda: {})
    monkeypatch.setattr(runtime, "stable_model_identity", lambda: {})


def test_generation_stops_each_row_and_preserves_actual_logp_rng_and_suffix(
    native_runtime, monkeypatch
):
    runtime, prepared = native_runtime
    generation_setup(runtime, prepared, monkeypatch, {5: '"answer":', 6: "1", 7: "}BAD", 8: "}"})

    def generate(**kwargs):
        assert kwargs["generation_config"].max_new_tokens == 1024
        seq = kwargs["input_ids"]
        for pair in ([7, 5], [0, 6], [0, 8]):
            seq = torch.cat([seq, torch.tensor(pair)[:, None]], -1)
            kwargs["stopping_criteria"](seq, None)
        runtime.image_calls += 1
        logits = tuple(torch.arange(128, dtype=torch.float32).repeat(2, 1) for _ in range(3))
        return SimpleNamespace(sequences=seq, logits=logits, scores=logits)

    monkeypatch.setattr(runtime.model, "generate", generate)
    initial_rng, written = torch.get_rng_state().clone(), []
    records = runtime.generate_group(
        safe_input(), "/unused", [100, 101], on_completion=written.append
    )
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert [record["tokens"] for record in records] == [[7], [5, 6, 8]]
    assert records[0]["raw_text"] == "{}BAD"
    assert records[0]["format_protocol_error"] == "trailing_after_close"
    assert records[0]["backend_sequence_tokens"] == [7, 0, 0]
    assert [len(record["sampler_logprobs"]) for record in records] == [1, 3]
    assert all(
        record["finish_reason"] == "balanced" and not record["truncated"] for record in records
    )
    assert written == records


def test_technical_generation_error_is_persisted_before_raising(native_runtime, monkeypatch):
    runtime, prepared = native_runtime
    generation_setup(runtime, prepared, monkeypatch, {7: "}"})

    def generate(**kwargs):
        seq = torch.cat([kwargs["input_ids"], torch.tensor([[7]])], -1)
        kwargs["stopping_criteria"](seq, None)
        runtime.image_calls += 1
        return SimpleNamespace(
            sequences=seq, logits=(torch.zeros(1, 128),), scores=(torch.ones(1, 128),)
        )

    monkeypatch.setattr(runtime.model, "generate", generate)
    written = []
    with pytest.raises(RuntimeError, match="preserved"):
        runtime.generate_group(safe_input(), "/unused", [1], on_completion=written.append)
    assert written[0]["generation_status"] == "TECHNICAL_INVALID"
    assert "UNREGISTERED_LOGITS_TRANSFORM" in written[0]["technical_validation_errors"]


def test_first_close_at_exact_1024_boundary_is_balanced_not_length(native_runtime, monkeypatch):
    runtime, prepared = native_runtime
    generation_setup(runtime, prepared, monkeypatch, {5: " ", 7: "}"})

    def generate(**kwargs):
        completion = torch.tensor([[*([5] * 1023), 7]])
        seq = torch.cat([kwargs["input_ids"], completion], -1)
        assert kwargs["stopping_criteria"](seq, None).tolist() == [True]
        runtime.image_calls += 1
        logits = (torch.zeros(1, 128),) * 1024
        return SimpleNamespace(sequences=seq, logits=logits, scores=logits)

    monkeypatch.setattr(runtime.model, "generate", generate)
    result = runtime.generate(safe_input(), "/unused", 1)
    assert result["completion_token_count"] == 1024
    assert result["finish_reason"] == "balanced" and result["truncated"] is False
    assert result["decoded_prefix_views"]["128"] == "{" + " " * 128
    assert result["decoded_prefix_views"]["1024"].endswith("}")


@pytest.mark.parametrize("protocol,cap", [("chartqa_plain", 256), ("chartqapro_official", 1024)])
def test_external_native_generation_is_greedy_without_prefill(
    native_runtime, monkeypatch, protocol, cap
):
    runtime, prepared = native_runtime
    generation_setup(runtime, prepared, monkeypatch, {2: "42"})

    def generate(**kwargs):
        config = kwargs["generation_config"]
        assert config.max_new_tokens == cap and config.do_sample is False
        assert "stopping_criteria" not in kwargs
        runtime.image_calls += 1
        logits = (torch.zeros(1, 128),)
        return SimpleNamespace(
            sequences=torch.cat([kwargs["input_ids"], torch.tensor([[2]])], -1),
            logits=logits,
            scores=logits,
        )

    monkeypatch.setattr(runtime.model, "generate", generate)
    record = runtime.generate(safe_input(), "/unused", 1, protocol=protocol)
    assert record["prefill"] == "" and record["raw_text"] == "42"
    assert record["tokens"] == [2] and record["finish_reason"] == "eos"
    with pytest.raises(PermissionError, match="recipe"):
        runtime.generate(
            safe_input(), "/unused", 1, protocol=protocol, generation={"do_sample": True}
        )


def test_model_input_records_restore_without_cached_prepared_tensor(native_runtime, monkeypatch):
    runtime, prepared = native_runtime
    calls = []

    def prepare(model_input, root, **kwargs):
        calls.append((model_input, root, kwargs))
        return prepared

    monkeypatch.setattr(runtime, "prepare_input", prepare)
    records = [dict(model_input=safe_input(), root="/registered", tokens=[5, 6]) for _ in range(2)]
    output = runtime.batch_sequence_forward(records, grad=False)
    assert len(output) == 2 and len(calls) == 1
    assert calls[0][1] == "/registered"


def test_real_native_generation_accepts_eight_independent_stop_flags(native_runtime, monkeypatch):
    runtime, prepared = native_runtime
    generation_setup(runtime, prepared, monkeypatch, {})
    runtime.processor.tokenizer.decode = lambda ids, **kwargs: "}" * len(ids)
    runtime.generation_config.eos_token_id = None
    runtime.eos_ids = []
    records = runtime.generate_group(safe_input(), "/unused", list(range(8)))
    assert len(records) == 8 and runtime.image_calls > 0
    assert all(record["tokens"] and len(record["tokens"]) == 1 for record in records)
    assert all(record["finish_reason"] == "balanced" for record in records)


def test_checkpoint_forward_mode_restore_preserves_native_gradient_and_autograd_grad(
    native_runtime,
):
    runtime, prepared = native_runtime
    config, digest = attach_parent(runtime)
    configure_parent_training(
        runtime, parent_adapter_config=config, expected_adapter_state_hash=digest
    )
    record = {"prepared": prepared, "tokens": [5, 6, 7]}
    parameters = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    baseline = runtime.batch_sequence_forward([record])[0]
    expected = torch.autograd.grad(baseline.mean(), parameters, allow_unused=True)
    del baseline
    assert enable_gradient_checkpointing(runtime, [record])["status"] == "PASS"
    current = runtime.batch_sequence_forward([record])[0]
    actual = torch.autograd.grad(current.mean(), parameters, allow_unused=True, retain_graph=True)
    for a, b in zip(actual, expected, strict=True):
        assert (a is None) is (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    current.mean().backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.any()) for parameter in parameters
    )


def test_processor_readability_uses_actual_resize_and_measured_glyphs():
    prepared = {
        "routing": dict(
            source_size=[1024, 768],
            processed_size=[1024, 768],
            processed_pixel_sha256="pixel",
            processor_hash="processor",
        )
    }
    receipt = native_readability_measurements(
        prepared, source_pixels_per_count=5.15, source_text_heights=[12, 14, 9]
    )
    assert receipt["status"] == "PASS" and receipt["minimum_processed_text_height"] == 9
    prepared["routing"]["processed_size"] = [512, 384]
    assert (
        native_readability_measurements(
            prepared, source_pixels_per_count=5.15, source_text_heights=[12]
        )["status"]
        == "FAIL"
    )


def test_real_tokenizer_full_processor_support_masks_and_restore(monkeypatch):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import BatchFeature, PreTrainedTokenizerFast

    text = '{"evidence":[],"work":"sum=5;count=2;reduce=mean(operands)","answer":"5/2"}'
    _, spans = target_spans(text)
    target = {"text": text, "spans": spans}
    prefix = "native image expanded assistant\n"
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer, backend.decoder = (
        pre_tokenizers.ByteLevel(add_prefix_space=False),
        decoders.ByteLevel(),
    )
    backend.train_from_iterator(
        [prefix + text],
        trainers.BpeTrainer(vocab_size=256, min_frequency=100, special_tokens=["[UNK]"]),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")

    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer

        def __call__(self, **kwargs):
            encoded = tokenizer(
                kwargs["text"],
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            encoded["mm_token_type_ids"] = torch.zeros_like(encoded["input_ids"])
            return BatchFeature(dict(encoded))

    runtime = SRF3CERuntime.__new__(SRF3CERuntime)
    runtime.torch, runtime.device, runtime.processor = torch, "cpu", Processor()
    prepared = {"chat_text": prefix + "{", "inputs": {}, "routing": {}}
    monkeypatch.setattr(runtime, "prepare_input", lambda *args, **kwargs: prepared)
    a = runtime.prepare_support(safe_input(), "/unused", target, "PREFIX")
    b = runtime.prepare_support(safe_input(), "/unused", target, "EXEC")
    assert a["tokens"] == b["tokens"] and a["answer_mask"] == b["answer_mask"]
    assert a["mask_audit"] == b["mask_audit"]
    assert sum(a["answer_mask"]) > 0 and sum(a["work_mask"]) > 0
    assert a["mask_audit"]["source_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert all(not (x and y) for x, y in zip(a["answer_mask"], a["work_mask"], strict=True))
    captured = []
    monkeypatch.setattr(
        runtime,
        "_native_forward",
        lambda inputs, **kw: captured.append(kw) or [torch.ones(len(a["tokens"]))],
    )
    durable = {
        key: copy.deepcopy(value)
        for key, value in a.items()
        if key not in ("prepared", "full_inputs")
    }
    assert len(runtime.support_forward(durable)) == len(a["tokens"])
    assert captured[0]["completion_start"] == a["completion_start_index"]
    durable["answer_mask"][0] = 1
    with pytest.raises(PermissionError, match="changed"):
        runtime.support_forward(durable)
    with pytest.raises(PermissionError, match="no teacher"):
        runtime.prepare_support(safe_input(), "/unused", target, "RL")
