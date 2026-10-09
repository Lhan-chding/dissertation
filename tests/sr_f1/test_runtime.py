from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sr_f1.runtime import TARGET_PIXELS, SRRuntime, generation_recipe


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    image_processor = SimpleNamespace(patch_size=16, merge_size=2)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        self.messages = messages
        content = messages[0]["content"]
        return "<image>" + content[-1]["text"] + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __call__(self, *, text, images=None, **kwargs):
        if images is None:
            return Inputs(
                input_ids=torch.tensor([[2, 3]]), attention_mask=torch.ones(1, 2, dtype=torch.long)
            )
        im = images[0]
        pixels = torch.tensor(np.asarray(im).copy(), dtype=torch.float32)
        ids = torch.tensor([[99] * 768 + [2, 3]])
        return Inputs(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=(ids == 99).long(),
            pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 48, 64]]),
        )


def runtime():
    obj = SRRuntime.__new__(SRRuntime)
    obj.processor = Processor()
    obj.device = "cpu"
    obj.model = SimpleNamespace(config=SimpleNamespace(image_token_id=99))
    obj.identity = dict(processor_hash="p", chat_template_hash="c")
    return obj


def test_native_input_ignores_gold_sidecar_and_ids_and_paths_do_not_enter_prompt(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (1024, 768), "white").save(tmp_path / "images/chart.png")
    obj = runtime()
    row = dict(
        qid="secretqid",
        root_id="secretroot",
        image_file="images/chart.png",
        text="What is visible?",
        answer=32,
    )
    a = obj.prepare(row, tmp_path)
    b = obj.prepare(
        {
            **row,
            "qid": "changed",
            "root_id": "other",
            "answer": 99,
            "required_evidence": ["private"],
        },
        tmp_path,
    )
    assert a["routing"]["input_tensor_hash"] == b["routing"]["input_tensor_hash"]
    assert a["routing"]["processed_size"] == [1024, 768]
    assert a["routing"]["processor_target_pixels"] == TARGET_PIXELS
    assert "secretqid" not in a["chat_text"] and "images/" not in a["chat_text"]
    assert not isinstance(obj.processor.messages[0]["content"][0]["image"], str)
    Image.new("RGB", (1024, 768), "black").save(tmp_path / "images/chart.png")
    c = obj.prepare(row, tmp_path)
    assert a["routing"]["input_tensor_hash"] != c["routing"]["input_tensor_hash"]


def test_text_only_view_has_zero_visual_tokens_and_cannot_escape_image_root(tmp_path):
    obj = runtime()
    p = obj.prepare_text("explicit text values", None, tmp_path)
    assert p["routing"]["image_token_count"] == 0
    assert p["inputs"]["mm_token_type_ids"].count_nonzero() == 0
    with pytest.raises((PermissionError, ValueError)):
        obj.prepare_text("query", "../outside.png", tmp_path)


def test_registered_sampling_recipe_has_no_warpers_and_exact_lengths():
    for limit in (128, 768):
        r = generation_recipe(max_new_tokens=limit)
        assert (r["temperature"], r["top_p"], r["top_k"], r["repetition_penalty"]) == (
            1.0,
            1.0,
            0,
            1.0,
        )
        assert r["max_new_tokens"] == limit and r["output_logits"] and r["output_scores"]
    with pytest.raises(ValueError):
        generation_recipe(max_new_tokens=192)
