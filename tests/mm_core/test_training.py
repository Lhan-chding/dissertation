"""CPU contract tests; these never claim CUDA or optimizer-resume certification."""

import importlib.util
import unittest

from mm_core.training import (
    BRIDGE_RECIPE,
    ENGINE_RECIPE,
    answer_reward,
    deterministic_order,
    group_advantages,
    language_qv_modules,
    state_hash,
)
from mm_core.vl_runtime import GENERATION, assign_field_tokens, field_value_spans, gold_completion


class TrainingContracts(unittest.TestCase):
    def test_hf_snapshot_identity_accepts_blob_links_and_rejects_escape(self):
        import hashlib
        import tempfile
        from pathlib import Path

        from mm_core.execution import object_hash
        from mm_core.vl_runtime import QwenRuntime, model_file_path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "models--Qwen--Qwen2.5-VL-3B-Instruct"
            snapshot = repo / "snapshots" / ("a" * 40)
            blobs = repo / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            files = []
            for name, value in (("config.json", b"{}"), ("model.safetensors", b"weights")):
                digest = hashlib.sha256(value).hexdigest()
                (blobs / digest).write_bytes(value)
                (snapshot / name).symlink_to(Path("../../blobs") / digest)
                self.assertEqual(model_file_path(snapshot, name), (blobs / digest).resolve())
                files.append(dict(name=name, bytes=len(value), sha256=digest))
            runtime = QwenRuntime.__new__(QwenRuntime)
            runtime.model_path = str(snapshot)
            runtime.adapter_path = None
            runtime.identity = dict(
                processor_hash="processor",
                chat_template_hash="chat",
                generation_config_expanded={"do_sample": True},
            )
            freeze = {
                **runtime.identity,
                "model_files": files,
                "model_weights_hash": object_hash({f["name"]: f["sha256"] for f in files}),
            }
            self.assertTrue(runtime.verify_identity(freeze)["verified"])
            outside = root / "private-file"
            outside.write_text("outside")
            (snapshot / "escape.bin").symlink_to(outside)
            with self.assertRaises(ValueError):
                model_file_path(snapshot, "escape.bin")
            for bad in ("../blobs/anything", str(outside), "", "."):
                with self.assertRaises(ValueError):
                    model_file_path(snapshot, bad)
            direct = root / "direct-model"
            direct.mkdir()
            (direct / "escape.bin").symlink_to(outside)
            with self.assertRaises(ValueError):
                model_file_path(direct, "escape.bin")
            # Hash/size checks continue to apply to the resolved official blob.
            (blobs / files[0]["sha256"]).write_bytes(b"mutated")
            with self.assertRaises(PermissionError):
                runtime.verify_identity(freeze)

    def test_exact_language_enumeration_excludes_visual(self):
        names = [
            f"model.language_model.layers.{i}.self_attn.{kind}"
            for i in range(36)
            for kind in ("q_proj", "v_proj")
        ]
        visual = [f"model.visual.layers.{i}.self_attn.q_proj" for i in range(32)]
        self.assertEqual(language_qv_modules(names + visual), sorted(names))
        with self.assertRaises(ValueError):
            language_qv_modules(names[:-1] + visual)

    def test_group_advantage_sign_and_zero_contrast(self):
        self.assertEqual(group_advantages([0] * 8), [0] * 8)
        self.assertEqual(group_advantages([1] * 8), [0] * 8)
        result = group_advantages([0] * 4 + [1] * 4)
        self.assertTrue(all(x < 0 for x in result[:4]))
        self.assertTrue(all(x > 0 for x in result[4:]))
        self.assertAlmostEqual(sum(result), 0)

    def test_reward_only_answer_and_full_parse(self):
        row = dict(true_values=[31, 12], operation="difference")
        self.assertEqual(answer_reward('{"readings": [0, 0], "answer": 19}', row), 1)
        self.assertEqual(answer_reward('{"answer": 19}', row), 1)
        for bad in (
            '{"answer": true}',
            '{"answer": 18}',
            '{"answer": 19,"answer":18}',
            'Here {"answer":19}',
            '{"readings":[31,12]}',
            '{"answer":NaN}',
        ):
            self.assertEqual(answer_reward(bad, row), 0)

    def test_gold_fields_and_operation_are_fixed(self):
        self.assertEqual(
            gold_completion(dict(true_values_decimal=["31", "12"], operation="sum")),
            '{"readings": [31, 12], "answer": 43}',
        )
        self.assertEqual(
            gold_completion(dict(true_values=[31, 12, 44], operation="range")),
            '{"readings": [31, 12, 44], "answer": 32}',
        )

    def test_value_boundaries_preserve_overlap_rule(self):
        text = '{"readings":[1,2],"answer":3}'
        spans = field_value_spans(text)
        self.assertEqual(text[slice(*spans["readings"])], "[1,2]")
        assignment, overlaps = assign_field_tokens(text, [(0, len(text))])
        self.assertEqual(assignment["readings"], [0])
        self.assertEqual(assignment["answer"], [])
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(field_value_spans("prefix " + text), {})
        self.assertEqual(field_value_spans('{"answer":3,"answer":4}'), {})

    def test_deterministic_stream_without_mutating_pool(self):
        rows = [dict(question_id=str(i)) for i in range(192)]
        before = list(rows)
        self.assertEqual(deterministic_order(rows, 17, 128), deterministic_order(rows, 17, 128))
        self.assertEqual(rows, before)
        self.assertEqual(len({r["question_id"] for r in deterministic_order(rows, 17, 128)}), 128)

    def test_expanded_recipe_cannot_silently_expand_budget(self):
        self.assertEqual(
            BRIDGE_RECIPE["optimizer_updates"] * BRIDGE_RECIPE["effective_batch_sequences"], 128
        )
        self.assertEqual(
            ENGINE_RECIPE["physical_updates"]
            * ENGINE_RECIPE["prompts_per_step"]
            * ENGINE_RECIPE["group_size"],
            128,
        )
        self.assertEqual(ENGINE_RECIPE["generation_reuse"], 1)
        self.assertEqual(GENERATION["max_new_tokens"], 192)
        self.assertEqual(GENERATION["top_k"], 0)

    def test_hash_preserves_rng_types_order_and_values(self):
        self.assertEqual(state_hash({"b": [1, 2], "a": 3}), state_hash({"a": 3, "b": [1, 2]}))
        self.assertNotEqual(state_hash([1, 2]), state_hash((1, 2)))
        self.assertNotEqual(state_hash([1, 2]), state_hash([2, 1]))


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
    "Local CPU torch/numpy are unavailable; tensor recovery test not run",
)
class TensorRecovery(unittest.TestCase):
    def test_rng_capture_restore_exact(self):
        import random

        import numpy as np
        import torch

        from mm_core.training import capture_rng, restore_rng

        state = capture_rng()
        first = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(state)
        second = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_actual_optimizer_checkpoint_roundtrip_is_exact(self):
        import random
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        import numpy as np
        import torch

        from mm_core.training import load_checkpoint, save_checkpoint, trainable_state
        from mm_core.vl_runtime import seed_all

        def setup():
            seed_all(2718)
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            return SimpleNamespace(model=model), optimizer, scheduler

        def step(runtime, optimizer, scheduler):
            optimizer.zero_grad()
            inputs = torch.randn(2, 3) * (1 + np.random.rand()) + random.random()
            runtime.model(inputs).square().mean().backward()
            optimizer.step()
            scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous, optimizer, scheduler = setup()
            reference = trainable_state(continuous.model)
            for _ in range(4):
                step(continuous, optimizer, scheduler)
            expected = save_checkpoint(
                root / "continuous.pt",
                continuous,
                optimizer,
                scheduler,
                4,
                reference,
                stream_hash="fixed-input-slots",
            )
            interrupted, optimizer, scheduler = setup()
            for _ in range(2):
                step(interrupted, optimizer, scheduler)
            save_checkpoint(
                root / "checkpoint.pt",
                interrupted,
                optimizer,
                scheduler,
                2,
                reference,
                stream_hash="fixed-input-slots",
            )
            resumed, optimizer, scheduler = setup()
            loaded = load_checkpoint(
                root / "checkpoint.pt", resumed, optimizer, scheduler, "fixed-input-slots"
            )
            for _ in range(2):
                step(resumed, optimizer, scheduler)
            actual = save_checkpoint(
                root / "resumed.pt",
                resumed,
                optimizer,
                scheduler,
                4,
                loaded["reference"],
                stream_hash="fixed-input-slots",
            )
            self.assertEqual(state_hash(expected), state_hash(actual))

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers unavailable")
    def test_native_tiny_qwen_image_forward_matches_actual_prefix(self):
        import torch
        import transformers

        from mm_core.vl_runtime import QwenRuntime

        config = transformers.Qwen2_5_VLConfig(
            text_config=dict(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]},
            ),
            vision_config=dict(
                depth=1,
                hidden_size=32,
                intermediate_size=48,
                num_heads=4,
                patch_size=2,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=32,
                window_size=4,
                fullatt_block_indexes=[0],
            ),
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=58,
            vision_end_token_id=59,
        )
        runtime = QwenRuntime.__new__(QwenRuntime)
        runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
        runtime.model = transformers.Qwen2_5_VLForConditionalGeneration(config).eval()
        reservations = []
        runtime.account = lambda kind, count, meta: reservations.append((kind, count, meta))
        hook = runtime._visual_module().register_forward_pre_hook(runtime._mark_vision)
        ids = torch.tensor([[3, 58, 60, 59, 4, 5]])
        inputs = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=torch.full((4, 24), 0.5),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )
        tokens = [10, 11, 2]
        result = runtime.sequence_forward({"inputs": inputs}, tokens, purpose="CPU_RANDOM_FIXTURE")
        actual_inputs = {
            **inputs,
            "input_ids": torch.cat([ids, torch.tensor([tokens])], dim=1),
            "attention_mask": torch.ones(1, 9, dtype=torch.long),
            "use_cache": False,
        }
        runtime.model.model.rope_deltas = None
        with torch.no_grad():
            logits = runtime.model(**actual_inputs).logits[0, 5:8].float().log_softmax(-1)
            expected = logits.gather(-1, torch.tensor(tokens)[:, None]).squeeze(-1)
        self.assertTrue(torch.allclose(result["logprobs"], expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(result["vision_forward_calls"], 1)
        self.assertEqual(reservations[0][:2], ("extra_forward_sequences", 1))
        self.assertEqual(tuple(result["entropy"].shape), (3,))
        hook.remove()


if __name__ == "__main__":
    unittest.main()
