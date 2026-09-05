"""Actual FP32 CPU gradients and exact Adam/RNG checkpoint tests."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.grpo_update import torch_ppo_loss
from src.likelihood import selected_token_logprobs
from src.optimizer_fork import (
    capture_state,
    load_checkpoint,
    restore_state,
    save_checkpoint,
    state_hash,
)


class TorchBoundaryTests(unittest.TestCase):
    def test_selected_logprob_shift_and_generated_mask(self):
        logits = torch.zeros(1, 4, 8, requires_grad=True)
        ids = torch.tensor([[3, 4, 5, 2]])
        result = selected_token_logprobs(logits, ids, prompt_length=2, eos_ids={2}, pad_id=0)
        self.assertEqual(tuple(result.shape), (2,))
        result.sum().backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 1].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 3].abs().sum().item(), 0)

    def test_fixed_denominator_gradient_is_sequence_score_over_64(self):
        new = torch.tensor([[-1.0, -2.0, 0.0], [-1.0, 0.0, 0.0]], requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        loss, audit = torch_ppo_loss(new, new.detach().clone(), torch.tensor([1.0, -1.0]), mask)
        loss.backward()
        torch.testing.assert_close(
            new.grad, torch.tensor([[-1 / 128, -1 / 128, 0], [1 / 128, 0, 0]])
        )
        self.assertEqual(audit["clip_fraction"], 0)

    def test_adam_rng_forks_and_disk_resume_match_uninterrupted(self):
        torch.manual_seed(17)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def update():
            optimizer.zero_grad()
            model(torch.randn(3, 2)).square().mean().backward()
            optimizer.step()

        update()
        origin = capture_state(model, optimizer, {"step": 1, "sample_keys": ["a"]})
        origin_hash = state_hash(origin)
        update()
        expected = capture_state(model, optimizer, {"step": 2})
        restore_state(model, optimizer, origin)
        self.assertEqual(
            state_hash(capture_state(model, optimizer, origin["metadata"])), origin_hash
        )
        update()
        self.assertEqual(
            state_hash(capture_state(model, optimizer, {"step": 2})), state_hash(expected)
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "resume.pt"
            save_checkpoint(path, origin, {"config": "abc", "model": "def", "data": "ghi"})
            with self.assertRaises(ValueError):
                load_checkpoint(path, {"config": "changed", "model": "def", "data": "ghi"})
            disk = load_checkpoint(path, {"config": "abc", "model": "def", "data": "ghi"})
            restore_state(model, optimizer, disk)
            update()
            self.assertEqual(
                state_hash(capture_state(model, optimizer, {"step": 2})), state_hash(expected)
            )


if __name__ == "__main__":
    unittest.main()
