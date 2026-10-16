import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import state_hash
from sr_f1.json_protocol import AMENDMENT_ID
from sr_f3_ce.masks import target_spans
from sr_f3_ce.prompts import C2_PREFILL, prefill_for
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


def test_revision_constructor_validates_before_loading_and_records_explicit_identity(monkeypatch):
    from transformers import GenerationConfig

    from sr_f12.runtime import SRF12Runtime

    loaded = []

    def parent_init(runtime, *args, **kwargs):
        loaded.append((args, kwargs))
        runtime.identity = {}
        runtime.generation_config = GenerationConfig()

    monkeypatch.setattr(SRF12Runtime, "__init__", parent_init)
    with pytest.raises(ValueError, match="revision"):
        SRF3CERuntime("/no-weights", protocol_revision="unregistered")
    assert not loaded
    for revision in ("v1", "v2", "c2"):
        runtime = SRF3CERuntime("/no-weights", protocol_revision=revision)
        assert runtime.identity["protocol_revision"] == revision
        assert runtime.identity["assistant_prefills"]["structured"] == prefill_for(
            "structured", revision
        )
        assert runtime.identity["assistant_prefills"]["answer_only"] == "{"
        assert runtime.identity["assistant_prefills"]["chartqa_plain"] == ""
    assert all("protocol_revision" not in kwargs for _, kwargs in loaded)


def test_c2_multitoken_prefill_closes_only_the_root_and_keeps_token_suffix():
    pieces = {5: '{"series":"A","category":"Jan","value":42}', 6: '],"answer":42}BAD'}
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "".join(pieces[i] for i in ids))
    stop = BatchedProtocolStop(tokenizer, 8, 1, "structured", revision="c2")
    context = [90, 198, 220, 328, 68, 26590, 763, 498]
    assert not stop(torch.tensor([[*context, 5]]), None).item()
    assert stop(torch.tensor([[*context, 5, 6]]), None).item()
    full = C2_PREFILL + pieces[5] + pieces[6]
    assert first_boundary_token(tokenizer, [5, 6], "structured", revision="c2") == (
        2,
        len(full) - 3,
    )
    decoded = decoded_protocol(pieces[5] + pieces[6], "structured", revision="c2")
    assert decoded["assistant_prefill"] == C2_PREFILL
    assert decoded["full_text"] == decoded["raw_text"] == full
    assert decoded["format_protocol_error"] == "trailing_after_close"
    assert stop.counts == [2] and stop.cuts == [len(full) - 3]


def test_c2_generate_keeps_registered_prefill_out_of_recorded_completion_tokens(
    native_runtime, monkeypatch
):
    runtime, prepared = native_runtime
    runtime.protocol_revision = "c2"
    prepared["routing"].update(
        protocol="structured", protocol_revision="c2", assistant_prefill=C2_PREFILL
    )
    for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
        old = prepared["inputs"][name]
        extension = (
            old.new_tensor([[198, 220, 328, 68, 26590, 763, 498]])
            if name == "input_ids"
            else (old.new_ones((1, 7)) if name == "attention_mask" else old.new_zeros((1, 7)))
        )
        prepared["inputs"][name] = torch.cat([old, extension], -1)
    pieces = {5: '{"series":"A","category":"Jan","value":42}', 6: '],"answer":42}BAD'}
    generation_setup(runtime, prepared, monkeypatch, pieces)

    def generate(**kwargs):
        seq = torch.cat([kwargs["input_ids"], torch.tensor([[5]])], -1)
        assert not kwargs["stopping_criteria"](seq, None).item()
        seq = torch.cat([seq, torch.tensor([[6]])], -1)
        assert kwargs["stopping_criteria"](seq, None).item()
        runtime.image_calls += 1
        logits = (torch.zeros(1, 128),) * 2
        return SimpleNamespace(sequences=seq, logits=logits, scores=logits)

    monkeypatch.setattr(runtime.model, "generate", generate)
    record = runtime.generate(safe_input(), "/unused", 101)
    full = C2_PREFILL + pieces[5] + pieces[6]
    assert record["protocol_revision"] == "c2"
    assert record["assistant_prefill"] == C2_PREFILL
    assert record["tokens"] == [5, 6] and record["completion_token_count"] == 2
    assert record["prompt_token_count"] == prepared["inputs"]["input_ids"].shape[-1]
    assert record["generated_text"] == pieces[5] + pieces[6]
    assert record["full_text"] == record["raw_text"] == full
    assert record["finish_reason"] == "balanced" and record["boundary_char"] == len(full) - 3
    assert record["token_prefix_character_ends"] == [len(C2_PREFILL + pieces[5]), len(full)]


@pytest.mark.parametrize(
    "protocol,text",
    [
        ("answer_only", '"answer":null}'),
        ("natural_work", "working <final>98/3</final>"),
        ("chartqa_plain", "42"),
        ("chartqapro_official", "42"),
    ],
)
def test_other_protocol_decoding_is_byte_identical_for_all_revisions(protocol, text):
    baseline = decoded_protocol(text, protocol)
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: text)
    for revision in ("v2", "c2"):
        assert decoded_protocol(text, protocol, revision=revision) == baseline
        assert first_boundary_token(tokenizer, [1], protocol, revision=revision) == (
            first_boundary_token(tokenizer, [1], protocol)
        )


def test_replay_rejects_mixed_revision_or_prefill_before_any_forward():
    runtime = SRF3CERuntime.__new__(SRF3CERuntime)
    runtime.protocol_revision = "c2"
    with pytest.raises(PermissionError, match="revision"):
        runtime._prepared_records([dict(protocol_revision="v1")])
    with pytest.raises(PermissionError, match="prefill"):
        runtime._prepared_records([dict(protocol_revision="c2", assistant_prefill="{")])
    with pytest.raises(PermissionError, match="context protocol revision"):
        runtime._prepared_records([dict(protocol_revision="c2", prepared={"routing": {}})])


@pytest.mark.parametrize("revision", ["v2", "c2"])
def test_revised_runtime_refuses_nonprescribed_teacher_serialization(revision):
    runtime = SRF3CERuntime.__new__(SRF3CERuntime)
    runtime.protocol_revision = revision
    for serialization in ("canonical", "field_newlines_v1"):
        with pytest.raises(PermissionError, match="serialization"):
            runtime.prepare_support(
                safe_input(), "/unused", {}, "GOLD", serialization=serialization
            )


@pytest.fixture(scope="module")
def frozen_native_processor():
    """Optional local integration: authentic released tokenizer/processor, zero weights."""
    supplement = (
        Path(__file__).resolve().parents[2]
        / "artifacts/sr_f3_ce_20261016/engine_gate_audit_20261016/supplement"
    )
    path = supplement / "frozen_tokenizer"
    if not path.is_dir():
        pytest.skip("Frozen tokenizer supplement is an offline integration-test input")
    prior = json.loads((supplement / "prior/MODEL_ENVIRONMENT_IDENTITY.json").read_text())
    expected = {row["name"]: row["sha256"] for row in prior["model_files"]}
    for file in path.iterdir():
        assert hashlib.sha256(file.read_bytes()).hexdigest() == expected[file.name]
    runtime = SRF3CERuntime.processor_only(path)
    assert not hasattr(runtime.model, "parameters")
    return runtime


def _revision_processor(base, revision):
    runtime = copy.copy(base)
    runtime.identity, runtime.generation_config = (
        copy.deepcopy(base.identity),
        copy.deepcopy(base.generation_config),
    )
    runtime.protocol_revision = revision
    runtime._set_recipe()
    return runtime


def test_frozen_native_c2_prefill_route_keeps_image_and_actual_base_prefix(
    frozen_native_processor, tmp_path
):
    from PIL import Image

    from sr_f3_ce.native_audit import _audit_protocol_context

    path = tmp_path / "image.png"
    Image.new("RGB", (1024, 768), "white").save(path)
    safe = dict(
        prompt="Which displayed count is greatest?",
        image_file=path.name,
        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    v1 = _revision_processor(frozen_native_processor, "v1")
    v2 = _revision_processor(frozen_native_processor, "v2")
    c2 = _revision_processor(frozen_native_processor, "c2")
    original = v1.prepare_input(safe, tmp_path)
    unchanged = v2.prepare_input(safe, tmp_path)
    revised = c2.prepare_input(safe, tmp_path)
    route = revised["routing"]
    assert route["assistant_prefill_token_ids"] == [90, 198, 220, 328, 68, 26590, 763, 498]
    assert route["assistant_prefill"] == C2_PREFILL and route["protocol_revision"] == "c2"
    assert revised["chat_text"] == original["chat_text"][:-1] + C2_PREFILL
    assert original["routing"]["input_tensor_hash"] == unchanged["routing"]["input_tensor_hash"]
    assert torch.equal(revised["inputs"]["input_ids"][:, :-7], original["inputs"]["input_ids"])
    for name in ("pixel_values", "image_grid_thw"):
        assert torch.equal(original["inputs"][name], revised["inputs"][name])
    assert route["image_token_count"] == 768
    for runtime in (v1, v2, c2):
        receipt = _audit_protocol_context(runtime, safe, tmp_path)
        assert (
            receipt["input_ids"]
            == runtime.prepare_input(safe, tmp_path)["inputs"]["input_ids"][0].tolist()
        )
        assert len(receipt["offsets"]) == len(receipt["input_ids"])
        assert receipt["native_expanded_chat_text"].count("<|image_pad|>") == 768
        assert receipt["prefill_token_ids"] == (
            [90] if runtime.protocol_revision != "c2" else route["assistant_prefill_token_ids"]
        )
    for protocol in ("answer_only", "natural_work", "chartqa_plain", "chartqapro_official"):
        a = v1.prepare_input(safe, tmp_path, protocol=protocol)
        b = c2.prepare_input(safe, tmp_path, protocol=protocol)
        assert a["chat_text"] == b["chat_text"]
        assert a["routing"]["input_tensor_hash"] == b["routing"]["input_tensor_hash"]


@pytest.mark.parametrize("corruption", ["offset", "input_id"])
def test_native_context_audit_rejects_offset_or_tensor_change(frozen_native_processor, corruption):
    from sr_f3_ce.native_audit import _audit_protocol_context

    runtime = _revision_processor(frozen_native_processor, "v1")
    original = runtime.processor

    class CorruptedProcessor:
        def __getattr__(self, key):
            return getattr(original, key)

        def __call__(self, **kwargs):
            encoded = original(**kwargs)
            if kwargs.get("return_offsets_mapping"):
                if corruption == "offset":
                    encoded["offset_mapping"][0, 0, 0] += 1
                else:
                    encoded["input_ids"][0, -1] = 1
            return encoded

    runtime.processor = CorruptedProcessor()
    with pytest.raises(PermissionError, match=r"identity differs|changed model input tensors"):
        _audit_protocol_context(runtime, safe_input(), "/unused")


@pytest.mark.parametrize("revision", ["v2", "c2"])
@pytest.mark.parametrize("empty_evidence,answer", [(False, "98/3"), (True, None)])
def test_frozen_native_indent_targets_masks_and_prefill_context(
    frozen_native_processor, monkeypatch, revision, empty_evidence, answer
):
    runtime = _revision_processor(frozen_native_processor, revision)
    obj = dict(
        evidence=[] if empty_evidence else [dict(series="Alpha", category="January", value=42)],
        work='first line\nsecond line; "answer"; 中文',
        answer=answer,
    )
    original = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    _, spans = target_spans(original)
    target = dict(text=original, spans=spans, object=obj)
    prefix = runtime.prepare_support(safe_input(), "/unused", target, "PREFIX")
    execution = runtime.prepare_support(safe_input(), "/unused", target, "EXEC")
    for key in ("tokens", "answer_mask", "work_mask", "serialized_teacher_target", "mask_audit"):
        assert prefix[key] == execution[key]
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    assert prefix["serialized_teacher_target"]["text"] == text
    assert prefix["serialization"] == "indent_2"
    assert prefix["protocol_revision"] == revision
    audit = prefix["mask_audit"]
    for index in audit["prefill_token_indices"]:
        assert audit["answer_mask"][index] == audit["work_mask"][index] == 0
    if revision == "c2" and empty_evidence:
        assert audit["prefill_cross_boundary_tokens"]
        assert audit["prefill_cross_boundary_tokens"][0]["token_id"] == 9774
    assert sum(prefix["answer_mask"]) and sum(prefix["work_mask"])
    tokenizer = runtime.processor.tokenizer
    native_ids = prefix["full_inputs"]["input_ids"][0].tolist()
    assert native_ids == audit["token_ids"]
    full_text = prefix["prepared"]["chat_text"][: -len(prefill_for("structured", revision))] + text
    encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    assert encoded["input_ids"] == native_ids
    assert [list(pair) for pair in encoded["offset_mapping"]] == audit["offsets"]
    assert tokenizer.decode(native_ids, skip_special_tokens=False).endswith(text)
    gold_obj = {k: v for k, v in obj.items() if k != "work"}
    gold_text = json.dumps(gold_obj, indent=2, ensure_ascii=False)
    _, gold_spans = target_spans(gold_text)
    gold = runtime.prepare_support(
        safe_input(), "/unused", dict(text=gold_text, spans=gold_spans, object=gold_obj), "GOLD"
    )
    assert not any(gold["work_mask"]) and any(gold["answer_mask"])
    captured = []
    monkeypatch.setattr(
        runtime,
        "_native_forward",
        lambda inputs, **kw: (
            captured.append((inputs["input_ids"].tolist(), kw))
            or [torch.zeros(len(prefix["tokens"]))]
        ),
    )
    durable = {k: v for k, v in prefix.items() if k not in ("prepared", "full_inputs")}
    runtime.support_forward(durable)
    assert captured[0][0][0] == native_ids
    assert captured[0][1]["completion_start"] == prefix["completion_start_index"]


def test_c2_likelihood_replay_keeps_all_prefill_tokens_in_context(
    frozen_native_processor, monkeypatch
):
    runtime = _revision_processor(frozen_native_processor, "c2")
    prepared = runtime.prepare_input(safe_input(), "/unused")
    context = prepared["inputs"]["input_ids"].tolist()
    captured = []
    monkeypatch.setattr(
        runtime,
        "_native_forward",
        lambda inputs, **kw: (
            captured.append((inputs["input_ids"].tolist(), kw)) or [torch.zeros(2)]
        ),
    )
    record = dict(
        model_input=safe_input(),
        root="/unused",
        protocol_revision="c2",
        assistant_prefill=C2_PREFILL,
        tokens=[198, 60],
        prompt_tensor_hash=prepared["routing"]["input_tensor_hash"],
    )
    runtime.batch_sequence_forward([record], grad=False)
    assert captured[0][0] == [context[0] + [198, 60]]
    assert captured[0][1]["completion_start"] == len(context[0])
    assert captured[0][1]["lengths"] == [2]
