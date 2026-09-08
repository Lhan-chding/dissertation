"""Generation and gradients must evaluate the same finite-precision prefixes."""

import types

import pytest
import torch

from src.model_adapters.base import HuggingFaceAdapter, pure_generation_options


class ShapeSensitivePolicy(torch.nn.Module):
    """Model fixture with a sequence-shape-dependent numerical discrepancy."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.3, 0.1]))
        self.generation_config = types.SimpleNamespace(eos_token_id=2)
        self.calls = []

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0, **kwargs):
        self.calls.append((input_ids.clone(), use_cache, self.training, kwargs))
        # BF16 kernels can produce different values for the same causal position
        # when given different total sequence shapes. A large fixture discrepancy
        # makes accidental full-sequence scoring observable even on CPU/FP32.
        offset = input_ids.shape[1] * torch.tensor([0.1, -0.1, 0.2, -0.2])
        logits = (self.weight + offset).expand(1, input_ids.shape[1], 4)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return types.SimpleNamespace(logits=logits)


@pytest.mark.parametrize("require_grad", [False, True])
def test_likelihood_matches_each_sampling_prefix_and_preserves_gradient(require_grad):
    model = ShapeSensitivePolicy()
    tokenizer = types.SimpleNamespace(eos_token_id=2, pad_token_id=0)
    adapter = HuggingFaceAdapter(
        model, types.SimpleNamespace(tokenizer=tokenizer), "fixture", "a" * 40, device="cpu"
    )
    prepared = {
        "inputs": {
            "input_ids": torch.tensor([[1, 3]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, 2, dtype=torch.long),
        },
        "audit": {"prompt_token_count": 2},
    }
    completion = [0, 3, 2, 1]  # Sampled PAD is real; tokens after EOS are not.
    expected = torch.stack(
        [
            (model.weight + length * torch.tensor([0.1, -0.1, 0.2, -0.2])).log_softmax(-1)[token]
            for length, token in zip((2, 3, 4), completion[:3], strict=True)
        ]
    )
    scores = adapter.logprobs(prepared, completion, require_grad=require_grad)
    torch.testing.assert_close(scores, expected)
    assert scores.requires_grad == require_grad
    assert [call[0].tolist() for call in model.calls] == [[[1, 3]], [[1, 3, 0]], [[1, 3, 0, 3]]]
    assert all(not call[1] for call in model.calls)
    assert prepared["inputs"]["input_ids"].tolist() == [[1, 3]]
    assert [call[3]["mm_token_type_ids"].shape[1] for call in model.calls] == [2, 3, 4]
    if require_grad:
        expected_grad = torch.autograd.grad(expected.sum(), model.weight)[0]
        scores.sum().backward()
        torch.testing.assert_close(model.weight.grad, expected_grad)


def test_sampling_uses_the_same_uncached_forward_family_as_scoring():
    assert pure_generation_options()["use_cache"] is False
