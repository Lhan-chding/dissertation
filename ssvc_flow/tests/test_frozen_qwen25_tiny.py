"""Actual random tiny Qwen2.5-VL + PEFT on CPU, never pretrained GPU evidence.

The processor deliberately supplies small deterministic inputs. These tests cover
our adapter's call shapes, model kernels and hooks, not released processor assets,
3B/7B pretrained behavior, CUDA, or the model-specific real P1 gate.
"""

import types

import pytest
import torch
from PIL import Image
from test_model_official_tiny import TinyTokenizer

from src.model_adapters.base import select_language_mlp_modules
from src.model_adapters.qwen25vl import Qwen25VLAdapter
from src.optimizer_fork import parameter_hash

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def tiny_qwen25():
    config = transformers.Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "window_size": 4,
            "fullatt_block_indexes": [0],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
    )
    return transformers.Qwen2_5_VLForConditionalGeneration(config).eval()


class Qwen25FixtureProcessor:
    tokenizer = TinyTokenizer()
    image_processor = types.SimpleNamespace(patch_size=2, merge_size=2)
    chat_template = "local deterministic Qwen2.5 test template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"]
        self.messages = messages
        # Qwen2.5 does not use the Qwen3.5 thinking switch.
        return "fixture user\nassistant:"

    def __call__(self, **kwargs):
        image = kwargs.get("images")
        ids = torch.tensor([[3, 58, 60, 59, 4, 5] if image else [3, 4, 5, 6]])
        result = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if image:
            intensity = sum(image[0].getpixel((0, 0))) / 765
            result.update(
                # 3 channels * 2 temporal frames * 2x2 spatial patch = 24.
                pixel_values=torch.full((4, 24), intensity),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
            )
        return result


@pytest.fixture
def measured_adapter():
    torch.manual_seed(17)
    base = tiny_qwen25()
    targets = select_language_mlp_modules((name for name, _ in base.named_modules()), 2)
    assert len(targets) == 6
    assert all("visual" not in name for name in targets)
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    # Nonzero adapters make disabling meaningful, unlike freshly initialized B=0.
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_A" in name:
                value.fill_(0.05)
            elif "lora_B" in name:
                value.fill_(0.1)
    model.requires_grad_(False)
    result = Qwen25VLAdapter(
        model,
        Qwen25FixtureProcessor(),
        "cpu-random-qwen25-fixture",
        "a" * 40,
        device="cpu",
    )
    result._install_hooks()
    return result


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("do_sample", [True, False])
def test_frozen_qwen25_generation_and_prefix_scores(measured_adapter, tmp_path, image, do_sample):
    adapter = measured_adapter
    prompt = {"system": "Return four integers.", "user": "observed [1,2,3,4]"}
    if image:
        Image.new("RGB", (4, 4), "red").save(tmp_path / "chart.png")
        prompt["image_path"] = "chart.png"
    item = adapter.prepare(prompt, tmp_path)
    assert item["audit"]["image_token_count"] == int(image)
    assert item["audit"]["processor_width"] == (4 if image else None)
    if image:
        assert item["inputs"]["pixel_values"].shape == (4, 24)
    before_hash = parameter_hash(adapter.model, trainable=False)
    forward_flags = []

    def inspect_prefix(*args):
        forward_flags.append((torch.is_grad_enabled(), adapter.model._adapters_disabled))
        assert not any(p.requires_grad for p in adapter.model.parameters())

    handle = adapter.model.get_base_model().model.register_forward_pre_hook(inspect_prefix)
    try:
        with adapter.model.disable_adapter():
            output = adapter.generate(item, seed=101, max_new_tokens=4, do_sample=do_sample)
            tokens = output["token_ids"]
            assert 1 <= len(tokens) <= 4
            assert adapter.forward_calls == len(tokens)
            if image:
                assert adapter.vision_forward_calls == len(tokens)
                assert adapter.last_vision_hash is not None
            else:
                assert adapter.vision_forward_calls == 0
            observed = adapter.logprobs(item, tokens)
            assert not observed.requires_grad
            torch.testing.assert_close(
                observed,
                torch.tensor(output["behavior_token_logprobs"]),
                atol=1e-5,
                rtol=1e-5,
            )
            assert adapter.forward_calls == 2 * len(tokens)
            assert all(not enabled and disabled for enabled, disabled in forward_flags)
            # Independent direct official forward verifies argmax and raw softmax scores.
            with torch.no_grad():
                for index, token in enumerate(tokens):
                    adapter._reset_positions()
                    logits = (
                        adapter.model(
                            **adapter.full_inputs(item, tokens[:index]),
                            use_cache=False,
                            logits_to_keep=1,
                        )
                        .logits[0, -1]
                        .float()
                    )
                    assert output["behavior_token_logprobs"][index] == pytest.approx(
                        logits.log_softmax(-1)[token].item(),
                        abs=1e-5,
                    )
                    if not do_sample:
                        assert token == logits.argmax().item()
            assert not any(p.grad is not None for p in adapter.model.parameters())
            assert parameter_hash(adapter.model, trainable=False) == before_hash
    finally:
        handle.remove()


def test_qwen25_legacy_single_user_prepare_and_48_token_setting(measured_adapter, tmp_path):
    from src.model_adapters.base import pure_generation_options

    adapter = measured_adapter
    raw = "Recover the state: 3,8,5,18"
    item = adapter.prepare(
        {
            "system": None,
            "user": raw,
            "messages": [{"role": "user", "content": raw}],
        },
        tmp_path,
    )
    assert adapter.processor.messages == [{"role": "user", "content": raw}]
    assert item["audit"]["image_token_count"] == 0
    assert pure_generation_options(48)["max_new_tokens"] == 48
