"""CPU execution contracts; these are not real Qwen bridge evidence."""

import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.optimizer_fork import state_hash
from src.verified_discovery_transfer.sft_loss import (
    EncodedExample,
    accumulated_backward,
    collate_examples,
    encode_example,
    forward_batch,
    per_sequence_nll,
)
from src.verified_discovery_transfer.sft_runtime import (
    SFTRuntime,
    ShuffledCycle,
    reopen_original_lora,
)


class TinyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(31, 8)
        self.embedding.weight.requires_grad_(False)
        self.lora_A = torch.nn.Linear(8, 8, bias=False)
        self.lora_B = torch.nn.Linear(8, 31, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False, **kwargs):
        x = self.embedding(input_ids)
        scores = x @ x.transpose(1, 2) / 8**0.5
        future = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        scores = scores.masked_fill(~attention_mask[:, None, :].bool(), float("-inf"))
        return types.SimpleNamespace(
            logits=self.lora_B(torch.tanh(self.lora_A(scores.softmax(-1) @ x)))
        )


class TinyAdapter:
    def __init__(self):
        self.model, self.device, self.pad_id = TinyCausal(), "cpu", 30
        self.eos_ids = {29, 30}
        self.processor = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(
                eos_token_id=30, encode=lambda text, add_special_tokens=False: [7, 8, 9]
            )
        )

    def _reset_positions(self):
        pass

    def prepare(self, prompt, data_root):
        return {
            "inputs": {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }
        }


def examples():
    return [
        EncodedExample(str(i), tuple([1, 2] + [3] * (i % 3)), tuple([5, 6] + [7] * (i % 2) + [30]))
        for i in range(8)
    ]


def runtime():
    torch.manual_seed(7)
    adapter = TinyAdapter()
    value = SFTRuntime(
        adapter,
        seed=73101,
        view_identity={"view": "fixture"},
        parameter_names=[n for n, p in adapter.model.named_parameters() if p.requires_grad],
    )
    value.bind_data(examples(), examples()[::-1])
    return value


class SFTLossTests(unittest.TestCase):
    def test_prompt_shift_eos_pad_same_id(self):
        _inputs, labels = collate_examples(
            [EncodedExample("a", (1, 2), (5, 30)), EncodedExample("b", (1,), (30,))], pad_id=30
        )
        self.assertEqual(labels.tolist(), [[-100, -100, 5, 30], [-100, 30, -100, -100]])
        logits = torch.zeros(2, 4, 31, requires_grad=True)
        means, _, mask = per_sequence_nll(logits, labels)
        self.assertEqual(mask.tolist(), [[False, True, True], [True, False, False]])
        means.sum().backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0, 1, 5].abs()), 0)
        self.assertGreater(float(logits.grad[1, 0, 30].abs()), 0)
        self.assertEqual(float(logits.grad[:, -1].abs().sum()), 0)

    def test_sequence_mean_not_token_mean(self):
        labels = torch.tensor([[-100, 0, -100], [-100, 0, 1]])
        logits = torch.tensor(
            [[[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]]
        )
        means, nll, mask = per_sequence_nll(logits, labels)
        torch.testing.assert_close(means, torch.stack([nll[0, 0], nll[1].mean()]))
        self.assertNotAlmostEqual(float(means.mean()), float(nll.sum() / mask.sum()))

    def test_encoding_no_truncation(self):
        adapter, prompt = TinyAdapter(), {"system": "x", "user": "y"}
        value = encode_example(adapter, prompt, "[1, 2, 3, 4]")
        self.assertEqual(value.target_ids, (7, 8, 9, 30))
        with self.assertRaisesRegex(ValueError, "truncation"):
            encode_example(adapter, prompt, [1, 2, 3, 4], max_total_tokens=6)
        for target in ([True, 2, 3, 4], [1.0, 2, 3, 4], [100, 2, 3, 4]):
            with self.assertRaises(ValueError):
                encode_example(adapter, prompt, target)

    def test_tokenized_view_payload_excludes_supervised_eos(self):
        from src.verified_discovery_transfer.sft_runner import encode_training_rows

        backend = types.SimpleNamespace(adapter=TinyAdapter(), data_root=None)
        row = {
            "task_id": "t0",
            "prompt": {"system": "s", "user": "u"},
            "target": "[1,2,3,4]",
            "target_token_ids": [7, 8, 9],
            "EOS_id": 30,
            "weight": 1,
        }
        self.assertEqual(encode_training_rows(backend, [row])[0].target_ids, (7, 8, 9, 30))
        with self.assertRaises(ValueError):
            encode_training_rows(backend, [{**row, "target_token_ids": [7, 8, 9, 30]}])
        with self.assertRaises(ValueError):
            encode_training_rows(backend, [{**row, "EOS_id": 29}])

    def test_future_and_cross_sample_isolation(self):
        adapter, first = TinyAdapter(), examples()[0]
        original, _ = forward_batch(adapter, [first])
        later, _ = forward_batch(
            adapter, [EncodedExample("changed", first.prompt_ids, (5, 20, 30))]
        )
        torch.testing.assert_close(original[:, :3], later[:, :3], rtol=0, atol=0)
        batch, _ = forward_batch(adapter, [first, examples()[2]])
        torch.testing.assert_close(
            original[0], batch[0, : len(first.input_ids)], rtol=1e-6, atol=1e-7
        )

    def test_accumulation_and_replay_quarter(self):
        adapter = TinyAdapter()
        for data, factor in ((examples() * 2, 1.0), (examples()[:4], 0.25)):
            adapter.model.zero_grad(set_to_none=True)
            logits, labels = forward_batch(adapter, data)
            want = per_sequence_nll(logits, labels)[0].mean() * factor
            want.backward()
            params = [p for p in adapter.model.parameters() if p.requires_grad]
            expected = [p.grad.clone() for p in params]
            for micro in (1, 2, 4):
                adapter.model.zero_grad(set_to_none=True)
                stats = accumulated_backward(adapter, data, microbatch_size=micro)
                self.assertAlmostEqual(stats["loss"], float(want.detach()), places=6)
                for p, value in zip(params, expected, strict=True):
                    torch.testing.assert_close(p.grad, value, rtol=2e-6, atol=1e-7)


class RuntimeTests(unittest.TestCase):
    def test_fresh_adam_exact_resume_scheduler_rng_sampler(self):
        r = runtime()
        self.assertEqual(len(r.optimizer.state), 0)
        self.assertEqual(r.update()["lr"], 1e-5 / 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            r.save(path)
            second = r.update()
            expected = r.capture()
            draws = (random.random(), float(np.random.rand()), float(torch.rand(())))
            r.resume(path)
            self.assertEqual(second, r.update())
            self.assertEqual(state_hash(expected), state_hash(r.capture()))
            self.assertEqual(
                draws, (random.random(), float(np.random.rand()), float(torch.rand(())))
            )
            with self.assertRaises(FileExistsError):
                r.save(path)
        for _ in range(6):
            stats = r.update()
        self.assertEqual(stats["update"], 8)
        self.assertEqual(stats["lr"], 1e-5)
        self.assertIsNone(r.adapter.model.embedding.weight.grad)

    def test_identity_sampler_mismatch_rejected(self):
        r = runtime()
        for key in ("identity", "sampler"):
            state = r.capture()
            if key == "identity":
                state[key] = {"view": "wrong"}
            else:
                state[key]["replay"]["size"] = 9
            with self.assertRaises(ValueError):
                r.restore(state)

    def test_replay_schedule_independent_focus(self):
        a, b = runtime(), runtime()
        b.bind_data(examples()[:1], examples()[::-1])
        for _ in range(3):
            self.assertEqual(a.update()["task_ids"][-4:], b.update()["task_ids"][-4:])
        b.bind_data([], examples())
        self.assertEqual(b.update()["sequence_count"], 4)

    def test_sampler_cycles_resume(self):
        s = ShuffledCycle(3, 77)
        self.assertEqual(len(set(s.take(3))), 3)
        state = s.state_dict()
        expected = s.take(19)
        s.load_state_dict(state)
        self.assertEqual(expected, s.take(19))

    def test_exact_lora_whitelist(self):
        class Pair(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 8, bias=False)})
                self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 2, bias=False)})
                self.base = torch.nn.Linear(2, 2)

        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList()
        targets = []
        for i in range(32):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(layer.mlp, name, Pair())
                targets.append(f"layers.{i}.mlp.{name}")
            model.layers.append(layer)
        adapter = types.SimpleNamespace(
            model=model,
            audit={"lora_modules": targets, "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0},
            _reset_positions=lambda: None,
        )
        names = reopen_original_lora(adapter)
        self.assertEqual(len(names), 192)
        self.assertEqual({n for n, p in model.named_parameters() if p.requires_grad}, set(names))
        model.extra_lora_A = torch.nn.Linear(2, 8)
        with self.assertRaises(ValueError):
            reopen_original_lora(adapter)


class RunnerRecoveryTests(unittest.TestCase):
    def test_rollback_checkpoint_directory_rejected(self):
        from src.verified_discovery_transfer.sft_runner import validate_resume_directory

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            zero, later = output / "step000.pt", output / "step032.pt"
            zero.touch()
            later.touch()
            with self.assertRaises(ValueError):
                validate_resume_directory(output, zero)
            with self.assertRaises(ValueError):
                validate_resume_directory(output, None)
            validate_resume_directory(output, later)

    def test_attempt_journal_preserves_replayed_work_and_final_state(self):
        import json

        from src.verified_discovery_transfer.sft_runner import run_sft
        from src.verified_discovery_transfer.sft_runtime import read_student_checkpoint

        focus = [
            {
                "task_id": f"f{i}",
                "prompt": {"system": "s", "user": str(i)},
                "target": "[1,2,3,4]",
                "family": "cross_series" if i % 2 else "trend",
            }
            for i in range(8)
        ]
        replay = [{**row, "task_id": "r" + row["task_id"]} for row in focus]
        backend = types.SimpleNamespace(
            adapter=TinyAdapter(),
            data_root=None,
            receipt={"checkpoint": {"sha256": "parent-fixture"}},
        )

        def factory(backend, *, seed, view_identity):
            value = runtime()
            value.identity = view_identity
            return value

        with tempfile.TemporaryDirectory() as temporary:
            base, recovered = Path(temporary) / "baseline", Path(temporary) / "recovered"
            options = {
                "focus": focus,
                "replay": replay,
                "identity": {"arm": "SELF_MIX"},
                "seed": 73101,
            }
            with patch.object(SFTRuntime, "from_frozen_backend", side_effect=factory):
                expected_result = run_sft(backend, output_dir=base, **options)
                original_update = SFTRuntime.update

                def interrupted_update(value, **kwargs):
                    result = original_update(value, **kwargs)
                    if value.step == 50:
                        raise RuntimeError("injected update50 crash")
                    return result

                with (
                    patch.object(SFTRuntime, "update", interrupted_update),
                    self.assertRaisesRegex(RuntimeError, "update50"),
                ):
                    run_sft(backend, output_dir=recovered, **options)
                result = run_sft(
                    backend, output_dir=recovered, resume_path=recovered / "step032.pt", **options
                )
            self.assertEqual(
                state_hash(read_student_checkpoint(base / "step256.pt")),
                state_hash(read_student_checkpoint(recovered / "step256.pt")),
            )
            journals = list((recovered / "attempts").glob("sft_*/update*.json"))
            complete = [json.loads(p.read_text()) for p in journals]
            self.assertEqual(sum(r["status"] == "UPDATE_APPLIED" for r in complete), 49 + 224)
            self.assertEqual(sum(r["status"] == "UPDATE_STARTED" for r in complete), 1)
            self.assertEqual(result["process_costs"]["completed_updates_this_process"], 224)
            self.assertEqual(
                expected_result["process_costs"]["processed_target_sequences"], 256 * 16
            )
            stats = result["metrics_this_process"][0]
            self.assertAlmostEqual(
                stats["loss"], stats["focus_loss_contribution"] + stats["replay_loss_contribution"]
            )
            self.assertAlmostEqual(
                stats["loss"],
                sum(r["objective_contribution"] for r in stats["loss_by_role_family"]),
            )
            self.assertEqual(
                sum(r["processed_target_sequences"] for r in stats["loss_by_role_family"]), 16
            )


if __name__ == "__main__":
    unittest.main()
