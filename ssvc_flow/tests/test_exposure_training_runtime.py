"""CPU integration evidence only; these fixtures never claim a Qwen/GPU bridge."""

import json
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.exposure_substitution.bridge import bridge_runtime, run_bridge
from src.exposure_substitution.runtime import SERRuntime, read_student_checkpoint
from src.exposure_substitution.schedule import ExplicitScheduleSampler
from src.exposure_substitution.training import (
    atomic_json,
    explicit_backward,
    latest_checkpoint,
    require_bridge,
    run_training,
    student_lock,
)
from src.optimizer_fork import state_hash
from src.verified_discovery_transfer.sft_loss import (
    EncodedExample,
    collate_examples,
    encode_example,
    forward_batch,
    per_sequence_nll,
)

torch.set_num_threads(1)
PLAN = (
    Path(__file__).resolve().parents[1]
    / "docs/exposure_substitution/design/manifests/schedule_108701.jsonl"
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 4)
        self.embedding.weight.requires_grad_(False)
        self.lora_A = torch.nn.Linear(4, 4, bias=False)
        self.lora_B = torch.nn.Linear(4, 17, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert use_cache is False
        hidden = (self.embedding(input_ids) * attention_mask[..., None]).cumsum(1)
        return types.SimpleNamespace(logits=self.lora_B(torch.tanh(self.lora_A(hidden))))


class TinyAdapter:
    def __init__(self):
        self.model, self.device, self.pad_id, self.eos_ids = TinyModel(), "cpu", 16, {16}
        self.processor = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(
                eos_token_id=16, encode=lambda text, add_special_tokens=False: [5, 6]
            )
        )

    def _reset_positions(self):
        pass

    def prepare(self, prompt, data_root=None):
        return {
            "inputs": {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            }
        }


def make_runtime():
    torch.manual_seed(79)
    adapter = TinyAdapter()
    sampler = ExplicitScheduleSampler(PLAN, "A_LOCAL_C1")
    ids = sorted(
        {
            row["task_id"]
            for line in PLAN.read_text().splitlines()
            if (row := json.loads(line))["arm"] == "A_LOCAL_C1"
        }
    )
    encoded = {
        key: EncodedExample(key, (1, 2) + (3,) * (i % 3), (4, 5) + (6,) * (i % 2) + (16,))
        for i, key in enumerate(ids)
    }
    return SERRuntime(
        adapter,
        seed=108701,
        identity={"cpu_fixture": True, "schedule": sampler.schedule_id},
        parameter_names=[n for n, p in adapter.model.named_parameters() if p.requires_grad],
        sampler=sampler,
        encoded=encoded,
    )


class LossTests(unittest.TestCase):
    def test_shift_unique_eos_and_pad_mask(self):
        adapter = TinyAdapter()
        example = encode_example(adapter, {"system": "s", "user": "u"}, "[1,2,3,4]")
        self.assertEqual(example.target_ids, (5, 6, 16))
        with self.assertRaisesRegex(ValueError, "truncation"):
            encode_example(adapter, {"system": "s", "user": "u"}, "[1,2,3,4]", max_total_tokens=4)
        _, labels = collate_examples([example, EncodedExample("short", (1,), (16,))], pad_id=16)
        self.assertEqual(labels[1].tolist(), [-100, 16, -100, -100, -100])
        logits = torch.zeros(2, 5, 17, requires_grad=True)
        means, _, mask = per_sequence_nll(logits, labels)
        means.sum().backward()
        self.assertEqual(mask.sum(1).tolist(), [3, 1])
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0, 1].abs().sum()), 0)
        self.assertEqual(float(logits.grad[1, 1:].abs().sum()), 0)

    def test_five_batches_equal_sequence_weight_not_batch_weight(self):
        runtime = make_runtime()
        data = [runtime.encoded[r["task_id"]] for r in runtime.sampler.peek()]
        params = [p for p in runtime.adapter.model.parameters() if p.requires_grad]
        logits, labels = forward_batch(runtime.adapter, data)
        means = per_sequence_nll(logits, labels)[0]
        means.mean().backward()
        expected = [p.grad.clone() for p in params]
        runtime.adapter.model.zero_grad(set_to_none=True)
        actual = explicit_backward(runtime.adapter, data)
        self.assertAlmostEqual(actual["loss"], float(means.mean().detach()), places=6)
        self.assertEqual([x[0] for x in actual["microbatch_shapes"]], [4, 4, 3, 1, 4])
        for parameter, gradient in zip(params, expected, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, atol=1e-7, rtol=1e-6)

    def test_donor_length_cannot_change_common_or_replay_padding(self):
        runtime = make_runtime()
        data = [runtime.encoded[r["task_id"]] for r in runtime.sampler.peek()]
        before = explicit_backward(runtime.adapter, data)["microbatch_shapes"]
        donor = data[11]
        data[11] = EncodedExample(donor.task_id, donor.prompt_ids + (7,) * 30, donor.target_ids)
        after = explicit_backward(runtime.adapter, data)["microbatch_shapes"]
        self.assertNotEqual(before[3], after[3])
        self.assertEqual(before[:3] + before[4:], after[:3] + after[4:])


class RuntimeTests(unittest.TestCase):
    def test_resume_complete_adam_rng_cursor_and_forward(self):
        runtime = make_runtime()
        self.assertFalse(runtime.optimizer.state)
        runtime.update()
        saved = runtime.capture()
        expected_update = runtime.update()
        expected_state = runtime.capture()
        expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
        runtime.restore(saved)
        self.assertEqual(runtime.update(), expected_update)
        self.assertEqual(state_hash(runtime.capture()), state_hash(expected_state))
        self.assertEqual(
            expected_rng, (random.random(), float(np.random.rand()), float(torch.rand(())))
        )
        self.assertEqual(runtime.role_exposures, {"common": 22, "donor": 2, "replay": 8})
        self.assertEqual(runtime.sampler.step, 2)

    def test_zero_finite_gradients_preserve_adam_history_and_step(self):
        runtime = make_runtime()
        runtime.update()
        original = explicit_backward

        def zeros(adapter, examples):
            stats = original(adapter, examples)
            for p in adapter.model.parameters():
                if p.requires_grad:
                    p.grad.zero_()
            return stats

        with patch("src.exposure_substitution.runtime.explicit_backward", side_effect=zeros):
            result = runtime.update()
        self.assertTrue(result["zero_finite_gradient"])
        self.assertEqual(result["update"], 2)
        self.assertGreater(result["parameter_delta_norm"], 0)
        self.assertTrue(all(int(v["step"]) == 2 for v in runtime.optimizer.state.values()))

    def test_resume_rejects_wrong_identity_cursor_and_fresh_adam(self):
        runtime = make_runtime()
        runtime.update()
        for mutate in (
            lambda s: s["identity"].update(wrong=True),
            lambda s: s["sampler"].update(committed_step=0),
            lambda s: s["optimizer"].update(state={}),
            lambda s: s["role_exposures"].update(donor=0),
        ):
            bad = runtime.capture()
            mutate(bad)
            with self.assertRaises(ValueError):
                runtime.restore(bad)

    def test_missing_gradient_poison_requires_restore(self):
        runtime = make_runtime()
        origin = runtime.capture()
        original = explicit_backward

        def disconnect(adapter, examples):
            result = original(adapter, examples)
            next(p for p in adapter.model.parameters() if p.requires_grad).grad = None
            return result

        with (
            patch("src.exposure_substitution.runtime.explicit_backward", side_effect=disconnect),
            self.assertRaises(FloatingPointError),
        ):
            runtime.update()
        with self.assertRaises(RuntimeError):
            runtime.update()
        with self.assertRaises(RuntimeError):
            runtime.capture()
        runtime.restore(origin)
        self.assertEqual(runtime.update()["update"], 1)

    def test_bridge_exact_sixteen_physical_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bridge"
            result = bridge_runtime(make_runtime(), path)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["execution_kind"], "CPU_TEST_FIXTURE")
            self.assertEqual(result["physical_updates_started"], 16)
            self.assertEqual(result["processed_target_sequences"], 256)
            self.assertEqual(result["generations"], 0)
            self.assertTrue(result["checks"]["complete_state_bitwise_equal"])
            with self.assertRaises(FileExistsError):
                bridge_runtime(make_runtime(), path)

    def test_bridge_inference_failure_cannot_leave_pass(self):
        runtime = make_runtime()
        backend = types.SimpleNamespace(adapter=runtime.adapter, data_root=None)
        donors = {
            f"d{i}": {
                "task_id": f"d{i}",
                "role": "donor",
                "root_id": f"root{i}",
                "prompt": {"system": "s", "user": "u"},
                "target": "[1,2,3,4]",
            }
            for i in range(32)
        }

        def compared(value, output):
            output.mkdir()
            result = {
                "status": "RESTORE_COMPARISON_PASS",
                "checks": {},
                "checkpoints": {
                    "resumed": {"path": str(output / "mock.pt")},
                    "origin": {"path": str(output / "origin.pt")},
                },
            }
            atomic_json(output / "BRIDGE_RESULT.json", result)
            return result

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("src.exposure_substitution.bridge.load_parent_backend", return_value=backend),
            patch("src.exposure_substitution.schema.training_rows", return_value=donors),
            patch("src.exposure_substitution.bridge.build_runtime", return_value=runtime),
            patch.object(runtime, "resume"),
            patch("src.exposure_substitution.bridge.bridge_runtime", side_effect=compared),
            patch(
                "src.exposure_substitution.runtime.read_student_checkpoint",
                return_value={"parameters": {}},
            ),
            patch(
                "src.exposure_substitution.bridge.load_student_into_backend",
                side_effect=RuntimeError("restore failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "restore failure"):
                run_bridge(temporary, allow_gpu=True)
            receipt = json.loads((Path(temporary) / "bridge/BRIDGE_RESULT.json").read_text())
            self.assertEqual(receipt["status"], "BLOCKED_TECHNICAL")
            self.assertEqual(receipt["failed_phase"], "inference_restore")

    def test_bridge_detects_missing_parameter_overwrite_on_resume(self):
        from src.optimizer_fork import restore_state

        def skip_weight_restore(model, optimizer, state, **kwargs):
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            result = restore_state(model, optimizer, state, **kwargs)
            if state["step"] == 4:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.copy_(before[name])
            return result

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("src.optimizer_fork.restore_state", side_effect=skip_weight_restore),
        ):
            result = bridge_runtime(make_runtime(), Path(temporary) / "bridge")
            self.assertEqual(result["status"], "BLOCKED_TECHNICAL")
            self.assertFalse(result["checks"]["parameters_bitwise_equal"])
            self.assertEqual(result["physical_updates_started"], 16)

    def test_shared_bridge_guard_rejects_incomplete_receipt_and_stop(self):
        from src.modeling_v3.io import canonical_hash

        frozen = {"cpu_fixture": True}
        checks = {
            key: True
            for key in (
                "complete_state_bitwise_equal",
                "loss_slots_counts_equal",
                "actual_parameter_updates_observed",
                "five_microbatches_observed",
                "all_sequence_weights_one_sixteenth",
                "inference_weight_restore_exact",
                "matched_donor_target_eos_equal",
            )
        }
        good = {
            "status": "PASS",
            "execution_kind": "REAL_CUDA_BRIDGE",
            "identity": {"frozen_plan_hash": canonical_hash(frozen), "technical_only": True},
            "physical_updates_started": 16,
            "physical_updates_completed": 16,
            "checks": checks,
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("src.exposure_substitution.schema.verify_frozen", return_value=frozen),
        ):
            run = Path(temporary)
            path = run / "bridge/BRIDGE_RESULT.json"
            for changed in (
                {"status": "RESTORE_COMPARISON_PASS"},
                {"physical_updates_completed": 15},
                {"execution_kind": "CPU_TEST_FIXTURE"},
                {"checks": {}},
                {"identity": {"frozen_plan_hash": "wrong", "technical_only": True}},
            ):
                atomic_json(path, {**good, **changed})
                with self.assertRaises(ValueError):
                    require_bridge(run)
            atomic_json(path, good)
            self.assertEqual(require_bridge(run), good)
            (run / "STOP").touch()
            with self.assertRaisesRegex(RuntimeError, "STOP"):
                require_bridge(run)

    def test_student_lock_excludes_second_writer(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            student_lock(temporary),
            self.assertRaisesRegex(RuntimeError, "active training process"),
            student_lock(temporary),
        ):
            self.fail("second writer acquired active lock")

    def test_checkpoints_192_and_resume_preserve_attempt_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            baseline, interrupted = directory / "baseline", directory / "interrupted"
            run_training(make_runtime(), baseline)
            runtime = make_runtime()
            update = runtime.update

            def fail():
                result = update()
                if runtime.step == 70:
                    raise RuntimeError("injected interruption")
                return result

            with (
                patch.object(runtime, "update", side_effect=fail),
                self.assertRaisesRegex(RuntimeError, "interruption"),
            ):
                run_training(runtime, interrupted)
            self.assertEqual(latest_checkpoint(interrupted).name, "step064.pt")
            result = run_training(
                make_runtime(), interrupted, resume_path=latest_checkpoint(interrupted)
            )
            self.assertEqual(result["process_costs"]["completed_updates_this_process"], 192)
            self.assertEqual(
                [record["step"] for record in result["checkpoints"]], [0, 64, 128, 192, 256]
            )
            self.assertEqual(
                {p.name for p in interrupted.glob("step*.pt")},
                {"step000.pt", "step064.pt", "step128.pt", "step192.pt", "step256.pt"},
            )
            self.assertEqual(
                state_hash(read_student_checkpoint(baseline / "step256.pt")),
                state_hash(read_student_checkpoint(interrupted / "step256.pt")),
            )
            rows = [
                json.loads(p.read_text()) for p in (interrupted / "attempts").glob("*/update*.json")
            ]
            self.assertEqual(sum(r["status"] == "UPDATE_APPLIED" for r in rows), 69 + 192)
            self.assertEqual(sum(r["status"] == "UPDATE_STARTED" for r in rows), 1)
            with self.assertRaises(ValueError):
                run_training(make_runtime(), interrupted, resume_path=interrupted / "step192.pt")


if __name__ == "__main__":
    unittest.main()
