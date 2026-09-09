"""Optional actual official 69,936-parameter CPU model, initialized at random.

No from_pretrained model downloads. Run with the optional GPU bootstrap libraries
installed; these tests are not evidence about 9B/BF16/CUDA compatibility.
"""

import types
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from src.likelihood import audit_model_cache
from src.model_adapters import load_adapter
from src.model_adapters.qwen35 import Qwen35Adapter
from src.optimizer_fork import parameter_hash

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def tiny_model():
    config = transformers.Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
    )
    return transformers.Qwen3_5ForConditionalGeneration(config).eval()


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = 1

    def get_vocab(self):
        return {str(i): i for i in range(64)}

    def encode(self, text, **kwargs):
        return [3] * len(text)

    def decode(self, ids, **kwargs):
        return ",".join(map(str, ids))


class TinyProcessor:
    tokenizer = TinyTokenizer()
    image_processor = types.SimpleNamespace(
        patch_size=2, merge_size=2, max_pixels=4096, size={"longest_edge": 4096}
    )
    chat_template = "fixture template with actual switch"

    def to_dict(self):
        return {"processor": "cpu fixture"}

    def apply_chat_template(self, messages, **kwargs):
        text = "\n".join(
            item["text"]
            for msg in messages
            for item in msg["content"]
            if isinstance(msg["content"], list) and item["type"] == "text"
        )
        return text + ("<think>\n\n</think>\n\n" if not kwargs["enable_thinking"] else "<think>\n")

    def __call__(self, **kwargs):
        image = kwargs.get("images")
        tokens = [3, 58, 60, 59, 4, 5] if image else [3, 4, 5, 6]
        ids = torch.tensor([tokens])
        result = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == 60).long(),
        }
        if image:
            intensity = sum(image[0].getpixel((0, 0))) / 765.0
            result = {
                **result,
                "pixel_values": torch.full((4, 12), intensity),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        return result


@pytest.fixture
def adapter():
    torch.manual_seed(17)
    result = Qwen35Adapter(
        tiny_model(), TinyProcessor(), "cpu-random-fixture", "a" * 40, device="cpu"
    )
    result._install_hooks()
    return result


def prepared(adapter, root, image=False):
    prompt = {"system": "system", "user": "observed [1,2,3,4]", "prompt_hash": "fixture"}
    if image:
        Image.new("RGB", (4, 4), "red").save(root / "image.png")
        prompt["image_path"] = "image.png"
    return adapter.prepare(prompt, root)


@pytest.mark.parametrize("image", [False, True])
def test_actual_hybrid_cache_and_gradient(adapter, tmp_path, image):
    item = prepared(adapter, tmp_path, image)
    assert item["audit"]["image_token_count"] == int(image)
    assert item["audit"]["processor_width"] == (4 if image else None)
    before = adapter.vision_forward_calls
    scores = adapter.logprobs(item, [7, 8, 9, 2], require_grad=True)
    assert len(scores) == 4
    scores.sum().backward()
    adapter.model.eval()
    audit = audit_model_cache(adapter, item, [7, 8, 9, 2])
    assert audit["passed"], audit
    assert audit["I4"]["status"] == "supported"
    assert audit["token"]["mean_selected_logprob_error"] < 1e-5
    if image:
        assert adapter.vision_forward_calls > before
        assert adapter.last_vision_hash is not None
    with pytest.raises(ValueError, match="Unknown cache"):
        adapter.continuation_scores(item, [2], mode="bad")


def test_actual_generation_raw_scores_no_hidden_transform(adapter, tmp_path):
    item = prepared(adapter, tmp_path)
    sample = adapter.generate(item, seed=101, max_new_tokens=4)
    likelihood = adapter.logprobs(item, sample["token_ids"])
    torch.testing.assert_close(
        likelihood, torch.tensor(sample["behavior_token_logprobs"]), atol=1e-5, rtol=1e-5
    )
    assert sample["completion_length"] <= 4
    assert len(sample["behavior_top1_token_ids"]) == len(sample["token_ids"])
    assert sample["raw_completion"] == adapter.processor.tokenizer.decode(
        sample["token_ids"][:-1] if sample["stop_reason"] == "eos" else sample["token_ids"]
    )
    assert adapter.generation_calls == 1


@pytest.mark.parametrize("image", [False, True])
def test_reference_certificate_measures_actual_generation_and_grad_path(adapter, tmp_path, image):
    from src.r1_parity import reference_execution_audit

    item = prepared(adapter, tmp_path, image)
    sample = adapter.generate(item, seed=101, max_new_tokens=4)
    checks = reference_execution_audit(
        adapter,
        item,
        sample,
        {
            "parity_alarm_mean_abs_token_logp": 0.02,
            "parity_alarm_p99_abs_token_logp": 0.1,
            "parity_alarm_min_top1_agreement": 0.999,
        },
    )
    assert all(x["comparison"]["passed"] for x in checks.values()), checks
    assert checks["behavior"]["candidate_top1"] == sample["behavior_top1_token_ids"]
    assert checks["training_likelihood"]["comparison"]["top1_agreement"] is None


def test_r1_reference_runner_writes_full_token_records_with_cpu_tiny_model(adapter, tmp_path):
    import json

    from src.audit_r1_runtime import _run_parity

    # Random weights otherwise sample image placeholders as text, which Qwen
    # correctly rejects on the next multimodal forward. Limit this CPU fixture's
    # output vocabulary consistently for generation and every scoring path.
    def fixture_text_vocabulary(_module, _inputs, logits):
        placeholder_mask = torch.isin(
            torch.arange(logits.shape[-1], device=logits.device),
            torch.tensor([60, 61], device=logits.device),
        )
        return logits.masked_fill(placeholder_mask, -1e9)

    adapter.model.lm_head.register_forward_hook(fixture_text_vocabulary)
    scenes = []
    for index in range(12):
        Image.new("RGB", (4, 4), (index * 20, 0, 255 - index * 20)).save(tmp_path / f"{index}.png")
        scenes.append(
            {
                "base_scene_id": f"scene{index}",
                "split": "calibration",
                "observed_world": [9, 2, 3, 4],
                "operation": "sum4",
                "cue": {"family": "duplicate_encoding", "known_index": 0, "known_value": 1},
                "image_path": f"{index}.png",
                "image_hash": f"fixture{index}",
            }
        )
    certificate = _run_parity(
        adapter,
        scenes,
        {
            "data_root": str(tmp_path),
            "max_new_tokens": 2,
            "parity_alarm_mean_abs_token_logp": 0.02,
            "parity_alarm_p99_abs_token_logp": 0.1,
            "parity_alarm_min_top1_agreement": 0.999,
        },
        tmp_path,
    )
    assert certificate["status"] == "PASS"
    assert certificate["fixed_sequence_boundaries"]["checks_completed"] == 27
    assert len(json.loads((tmp_path / "fixed_reference_rollouts.json").read_text())) == 120
    rows = [
        json.loads(line) for line in (tmp_path / "parity_records.jsonl").read_text().splitlines()
    ]
    assert {"batch2_reversed_images", "batch2_repeated_image", "training_likelihood"} <= {
        r["path"] for r in rows
    }
    assert all(len(r["reference_logp"]) == len(r["candidate_logp"]) for r in rows)


@pytest.mark.parametrize("side", ["left", "right"])
def test_multimodal_batch_padding_retains_image_order(adapter, tmp_path, side):
    from src.r1_parity import batch_teacher_scores

    red = prepared(adapter, tmp_path, True)
    Image.new("RGB", (4, 4), "blue").save(tmp_path / "blue.png")
    blue = adapter.prepare(
        {
            "system": "system",
            "user": "observed [1,2,3,4]",
            "prompt_hash": "fixture",
            "image_path": "blue.png",
        },
        tmp_path,
    )
    completions = [[7, 2], [8, 9, 2]]
    with torch.no_grad():
        reference = [
            adapter.continuation_scores(item, tokens, mode="full")
            for item, tokens in zip([red, blue], completions, strict=True)
        ]
    for items, tokens, expected in (
        ([red, blue], completions, reference),
        ([blue, red], completions[::-1], reference[::-1]),
    ):
        observed = batch_teacher_scores(adapter, items, tokens, side)
        for a, b in zip(observed, expected, strict=True):
            assert a[0] == b[0]
            torch.testing.assert_close(torch.tensor(a[1]), torch.tensor(b[1]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("image", [False, True])
def test_bf16_prefix_generation_scoring_and_gradient_agree(adapter, tmp_path, image):
    adapter.model.to(dtype=torch.bfloat16)
    item = prepared(adapter, tmp_path, image)
    sample = adapter.generate(item, seed=101, max_new_tokens=4)
    expected = torch.tensor(sample["behavior_token_logprobs"])
    observed = adapter.logprobs(item, sample["token_ids"])
    torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        differentiable = adapter.logprobs(item, sample["token_ids"], require_grad=True)
    torch.testing.assert_close(differentiable.detach(), expected, atol=1e-5, rtol=1e-5)
    differentiable.sum().backward()
    gradients = [p.grad for p in adapter.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)


@pytest.mark.parametrize("image", [False, True])
def test_peft_forward_counter_includes_scoring_but_excludes_checkpoint_replays(tmp_path, image):
    from src.model_adapters.base import select_language_mlp_modules

    torch.manual_seed(17)
    base = tiny_model()
    targets = select_language_mlp_modules((name for name, _ in base.named_modules()), 4)
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0, target_modules=targets, task_type="CAUSAL_LM"
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    measured = Qwen35Adapter(model, TinyProcessor(), "cpu-peft-fixture", "a" * 40, device="cpu")
    measured._install_hooks()
    item = prepared(measured, tmp_path, image)

    sample = measured.generate(item, seed=101, max_new_tokens=4)
    assert measured.forward_calls == len(sample["token_ids"])
    completion = [7, 8, 9, 2]
    for score in (measured.logprobs, measured.reference_logprobs):
        before = measured.forward_calls
        assert len(score(item, completion)) == len(completion)
        assert measured.forward_calls - before == len(completion)

    layer_calls = []
    handle = base.model.language_model.layers[0].register_forward_pre_hook(
        lambda *args: layer_calls.append(True)
    )
    try:
        before = measured.forward_calls
        scores = measured.logprobs(item, completion, require_grad=True)
        assert measured.forward_calls - before == len(completion)
        before_backward = measured.forward_calls
        layer_calls_before_backward = len(layer_calls)
        scores.sum().backward()
        # Verify checkpoint recomputation really happened, without counting it
        # again as a new outer model forward.
        assert len(layer_calls) > layer_calls_before_backward
        assert measured.forward_calls == before_backward
    finally:
        handle.remove()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_official_loader_lora_selection_with_mocked_download_and_cuda(tmp_path):
    import huggingface_hub

    model = tiny_model()
    processor = TinyProcessor()
    with (
        patch.object(
            huggingface_hub,
            "HfApi",
            return_value=types.SimpleNamespace(
                model_info=lambda *a, **k: types.SimpleNamespace(sha="a" * 40)
            ),
        ),
        patch.object(
            transformers.Qwen3_5ForConditionalGeneration, "from_pretrained", return_value=model
        ) as load,
        patch.object(transformers.AutoProcessor, "from_pretrained", return_value=processor),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.reset_peak_memory_stats"),
        patch("torch.cuda.synchronize"),
        patch("torch.cuda.max_memory_allocated", return_value=0),
    ):
        result = load_adapter(
            "qwen35_9b", {"id": "Qwen/Qwen3.5-9B", "revision": None, "expected_layers": 4}
        )
    assert load.call_args.kwargs["revision"] == "a" * 40
    assert len(result.audit["lora_modules"]) == 12
    assert result.audit["trainable_dtypes"] == ["torch.float32"]
    assert all("lora_" in name for name, p in result.model.named_parameters() if p.requires_grad)
    result.device = "cpu"
    item = prepared(result, tmp_path)
    before = parameter_hash(result.model, trainable=False)
    output = result.logprobs(item, [7, 2], require_grad=True)
    output.sum().backward()
    assert all(p.grad is None for p in result.model.parameters() if not p.requires_grad)
    assert parameter_hash(result.model, trainable=False) == before
    torch.testing.assert_close(
        result.reference_logprobs(item, [7, 2]), result.logprobs(item, [7, 2])
    )


def test_processor_rejects_path_escape_and_broken_thinking(adapter, tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        adapter.prepare({"system": "s", "user": "u", "image_path": "../escape.png"}, tmp_path)
    with (
        patch.object(adapter.processor, "apply_chat_template", return_value="unchanged"),
        pytest.raises(RuntimeError, match="enable_thinking"),
    ):
        prepared(adapter, tmp_path)
    with pytest.raises(ValueError, match="Unsupported model"):
        load_adapter("unknown", {})
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="no weights"),
    ):
        Qwen35Adapter.load({})
