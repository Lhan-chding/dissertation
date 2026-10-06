"""R0 crash recovery with actual Torch Adam/state and a tiny online policy."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.optimizer_fork import state_hash
from src.verified_discovery_transfer import r0_reference
from src.verified_discovery_transfer.queue import digest, read_json, write_json
from src.verified_discovery_transfer.sft_runtime import SFTRuntime, read_student_checkpoint


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_logits = torch.nn.Parameter(torch.tensor([-0.3, 0.4]))
        self.register_buffer("forward_counter", torch.tensor(0))
        self.rope_deltas = None


class OnlineAdapter:
    def __init__(self, calls):
        self.model, self.device = TinyPolicy(), "cpu"
        self.eos_ids = {2}
        self.calls = calls
        self.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(decode=self.decode))

    @staticmethod
    def decode(ids, **kwargs):
        return "[1,2,3,4]" if ids == [1] else "[4,3,2,1]"

    def _reset_positions(self):
        self.model.rope_deltas = None

    def prepare(self, prompt, data_root):
        return prompt

    def generate(self, prepared, *, seed, max_new_tokens, do_sample):
        self.calls.append(seed)
        self.model.eval()
        self._reset_positions()
        torch.manual_seed(seed)
        self.model.forward_counter.add_(1)
        self.model.rope_deltas = torch.ones(1)
        scores = self.model.lora_logits.log_softmax(0)
        token = int(torch.multinomial(scores.exp(), 1))
        return {
            "token_ids": [token, 2],
            "completion_length": 2,
            "stop_reason": "eos",
            "raw_completion": self.decode([token]),
            "behavior_token_logprobs": [float(scores[token]), -0.2],
        }

    def logprobs(self, prepared, completion, *, require_grad):
        self.model.train(require_grad)
        self._reset_positions()
        return torch.stack(
            [
                self.model.lora_logits.log_softmax(0)[completion[0]],
                self.model.lora_logits.sum() * 0 - 0.2,
            ]
        )


TASKS = [{"task_id": f"t{i:03d}", "split": "T_train"} for i in range(128)]


class FakeRoleDataset:
    def __init__(self, *args):
        self.manifest = {"manifest_digest": "fixed_fixture"}

    def public(self, split):
        assert split == "T_train"
        return TASKS


def make_runtime(backend, *, seed, view_identity):
    return SFTRuntime(
        backend.adapter, seed=seed, view_identity=view_identity, parameter_names=["lora_logits"]
    )


def make_run(root):
    root.mkdir()
    schedule = {
        "cohort_manifest_digest": "fixed_fixture",
        "repeats": {"0": [t["task_id"] for t in TASKS]},
    }
    schedule["digest"] = digest(schedule)
    write_json(root / "R0_PROMPT_SCHEDULES.json", schedule)


def execute(root, calls):
    backend = types.SimpleNamespace(
        adapter=OnlineAdapter(calls),
        data_root=None,
        receipt={"checkpoint": {"sha256": "parent-fixture"}},
    )
    job = {"id": "r0-fixture", "parent": "S96", "repeat": 0, "seed": 73101}
    with (
        patch("src.verified_discovery_transfer.public_tasks.RoleDataset", FakeRoleDataset),
        patch(
            "src.verified_discovery_transfer.public_tasks.compile_case",
            side_effect=lambda t, p: {"prompt": {"task": t["task_id"]}},
        ),
        patch(
            "src.verified_discovery_transfer.public_tasks.verify_raw",
            side_effect=lambda t, p, s: {"public_verifier_pass": s == "[1,2,3,4]"},
        ),
        patch.object(SFTRuntime, "from_frozen_backend", side_effect=make_runtime),
    ):
        return r0_reference.run_r0(root, job, backend)


class R0ResumeTests(unittest.TestCase):
    def test_partial_rollout_and_checkpoint_crashes_do_not_repeat_committed_draws(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            make_run(baseline)
            baseline_calls = []
            execute(baseline, baseline_calls)
            expected = read_student_checkpoint(baseline / "evidence/r0-fixture/step032.pt")
            self.assertEqual(len(baseline_calls), 32 * 4 * 8)
            self.assertEqual(len(set(baseline_calls)), len(baseline_calls))
            for mode in ("after_atomic", "before_checkpoint", "after_checkpoint"):
                run = root / mode
                make_run(run)
                calls = []
                tripped = []
                actual_write, actual_save = r0_reference.write_json, SFTRuntime.save

                def interrupted_write(
                    path,
                    value,
                    *,
                    actual_write=actual_write,
                    mode=mode,
                    calls=calls,
                    tripped=tripped,
                ):
                    actual_write(path, value)
                    if (
                        mode == "after_atomic"
                        and "/rollouts/" in str(path)
                        and len(calls) == 3
                        and not tripped
                    ):
                        tripped.append(True)
                        raise RuntimeError("injected after atomic publication")

                def interrupted_save(
                    runtime, path, *, mode=mode, tripped=tripped, actual_save=actual_save
                ):
                    eligible = (
                        mode in ("before_checkpoint", "after_checkpoint")
                        and runtime.step == 1
                        and not tripped
                    )
                    if eligible and mode == "before_checkpoint":
                        tripped.append(True)
                        raise RuntimeError("injected before checkpoint publication")
                    result = actual_save(runtime, path)
                    if eligible:
                        tripped.append(True)
                        raise RuntimeError("injected after checkpoint publication")
                    return result

                with (
                    patch.object(r0_reference, "write_json", side_effect=interrupted_write),
                    patch.object(SFTRuntime, "save", interrupted_save),
                    self.assertRaisesRegex(RuntimeError, "injected"),
                ):
                    execute(run, calls)
                self.assertTrue(tripped)
                execute(run, calls)
                resumed = read_student_checkpoint(run / "evidence/r0-fixture/step032.pt")
                self.assertEqual(calls, baseline_calls, mode)
                self.assertEqual(state_hash(resumed), state_hash(expected), mode)
                rows = list((run / "evidence/r0-fixture/rollouts").glob("*/*.json"))
                self.assertEqual(len(rows), 1024)
                self.assertEqual(len({read_json(p)["identity"] for p in rows}), 1024)

    def test_zero_advantage_advances_momentum_and_explicit_zero_matches(self):
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([1.0]))
        a = torch.optim.AdamW([first], lr=1e-5, weight_decay=0.0)
        b = torch.optim.AdamW([second], lr=1e-5, weight_decay=0.0)
        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second)
        a.step()
        b.step()
        r0_reference.zero_gradient_adam_step(a, [first], 2e-5)
        second.grad = torch.zeros_like(second)
        b.param_groups[0]["lr"] = 2e-5
        b.step()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(state_hash(a.state_dict()), state_hash(b.state_dict()))
        self.assertEqual(int(a.state[first]["step"]), 2)


if __name__ == "__main__":
    unittest.main()
