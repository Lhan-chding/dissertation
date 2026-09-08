"""Chat EOS must not be hidden by a model's corpus-end token configuration."""

from types import SimpleNamespace

import pytest
import torch

from src.model_adapters.base import HuggingFaceAdapter
from src.verifiers import strict_parse


@pytest.mark.parametrize(
    ("model_eos", "tokenizer_eos", "expected"),
    [
        (248044, 248046, {248044, 248046}),
        ([248044, 248046], 248046, {248044, 248046}),
        ((248044, None), [248046, None], {248044, 248046}),
        (None, 248046, {248046}),
        (248044, None, {248044}),
        (None, None, set()),
    ],
)
def test_chat_and_model_end_tokens_are_both_terminal(model_eos, tokenizer_eos, expected):
    adapter = HuggingFaceAdapter(
        SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=model_eos)),
        SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=tokenizer_eos)),
        "fixture",
        "a" * 40,
        device="cpu",
    )
    assert adapter.eos_ids == expected


class ChatTokenizer:
    eos_token_id = 248046
    pad_token_id = 248044
    bos_token_id = None

    def __init__(self):
        self.decoded_ids = None

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        self.decoded_ids = list(ids)
        text = {
            17: "[1,2,3,4]",
            31: "<think>",
            32: "</think>",
            33: "<|im_start|>",
            198: "\n",
            248044: "<|endoftext|>",
            248046: "<|im_end|>",
        }
        return "".join(text[token] for token in ids)


class ScriptedChatModel(torch.nn.Module):
    """Emit a fixed chat response, obeying the supplied stopping configuration."""

    def __init__(self, response):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=248044)
        self.model = SimpleNamespace(rope_deltas=None)
        self.response = response
        self.emitted = []

    def generate(self, input_ids, attention_mask, generation_config):
        scores = []
        for token in self.response[: generation_config.max_new_tokens]:
            self.emitted.append(token)
            logits = torch.zeros((1, 248047))
            logits[0, token] = 20.0
            scores.append(logits)
            if token in generation_config.eos_token_id:
                break
        return SimpleNamespace(
            sequences=torch.cat((input_ids, torch.tensor([self.emitted])), dim=1),
            scores=tuple(scores),
        )


@pytest.mark.parametrize(
    ("body", "raw", "valid"),
    [
        ([17], "[1,2,3,4]", True),
        ([31, 17, 32], "<think>[1,2,3,4]</think>", False),
        ([33, 17], "<|im_start|>[1,2,3,4]", False),
    ],
)
def test_generation_stops_at_chat_eos_without_repairing_body(body, raw, valid):
    pytest.importorskip("transformers")
    # The recorded failure had <|im_end|>\n<|endoftext|> after an otherwise valid array.
    model = ScriptedChatModel([*body, 248046, 198, 248044])
    tokenizer = ChatTokenizer()
    adapter = HuggingFaceAdapter(
        model, SimpleNamespace(tokenizer=tokenizer), "fixture", "a" * 40, device="cpu"
    )
    prepared = {
        "inputs": {"input_ids": torch.tensor([[5]]), "attention_mask": torch.ones((1, 1))},
        "audit": {"image_token_count": 0},
    }
    result = adapter.generate(prepared, seed=17, max_new_tokens=8)
    assert model.emitted == [*body, 248046]
    assert result["token_ids"] == [*body, 248046]
    assert len(result["behavior_token_logprobs"]) == len(body) + 1
    assert result["stop_reason"] == "eos"
    assert tokenizer.decoded_ids == body
    assert result["raw_completion"] == raw
    assert (strict_parse(raw) is not None) is valid
