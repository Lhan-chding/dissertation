"""Actual tiny official models: greedy decoding and L single-user prompts."""

from unittest.mock import patch

import torch
from test_model_official_tiny import adapter as adapter
from test_model_official_tiny import prepared


def test_greedy_selects_untransformed_argmax_and_scores_match(adapter, tmp_path):
    p = prepared(adapter, tmp_path)
    first = adapter.generate(p, seed=17, max_new_tokens=4, do_sample=False)
    second = adapter.generate(p, seed=29, max_new_tokens=4, do_sample=False)
    assert first["token_ids"] == second["token_ids"]
    for i, token in enumerate(first["token_ids"]):
        adapter._reset_positions()
        output = adapter.model(
            **adapter.full_inputs(p, first["token_ids"][:i]), use_cache=False, logits_to_keep=1
        )
        assert token == output.logits[0, -1].argmax().item()
    scored = adapter.logprobs(p, first["token_ids"]).detach()
    torch.testing.assert_close(scored, torch.tensor(first["behavior_token_logprobs"]))


def test_legacy_prompt_does_not_gain_system_or_multimodal_wrapper(adapter, tmp_path):
    messages = [{"role": "user", "content": "original prompt"}]
    prompt = {
        "system": None,
        "user": "original prompt",
        "messages": messages,
        "template_version": "L-C3-user-only",
        "prompt_hash": "fixture",
    }
    # Controlled output to inspect exact chat-template inputs with a real processor wrapper.
    with patch.object(
        adapter.processor,
        "apply_chat_template",
        side_effect=["original prompt</think>", "original prompt<think>"],
    ) as render:
        result = adapter.prepare(prompt, tmp_path)
    assert render.call_args_list[0].args[0] == messages
    assert result["audit"]["image_token_count"] == 0
