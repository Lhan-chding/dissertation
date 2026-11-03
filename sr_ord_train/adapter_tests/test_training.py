"""CPU synthetic checks; these never certify a real model/GPU trajectory."""
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("SR_ORD_NATIVE_SRC", ROOT / "execution/historical_readonly/native_source/src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NATIVE))
from adapter.training import canonical_hash, native_j_advantages, trajectory_hash, update, validate_records

HAS_TORCH = importlib.util.find_spec("torch") is not None


class CreditTests(unittest.TestCase):
    def test_all_binary_g8_match_native_j_only(self):
        # These are historical-shaped scalar fixtures, never new training rows.
        from sr_f6.credit import build_credit_group
        for rewards in itertools.product((0, 1), repeat=8):
            rows = [dict(group_id="fixture", qid="fixture", root_id="fixture", policy_hash="a" * 64,
                         parent="fixture", factor="P1", family="TIES", row_id=f"fixture-{i}",
                         draw=i, J=j, V=1, finished=1, S=0, B=0) for i, j in enumerate(rewards)]
            native = build_credit_group(rows, "U10", 0)
            self.assertEqual(native_j_advantages(rewards), native["native"])
            self.assertEqual(native["applied"], [0.] * 8)

    def test_constant_groups_keep_all_zero_advantages(self):
        for value in (0, 1):
            self.assertEqual(native_j_advantages([value] * 8), [0.] * 8)

    def test_invalid_credit_does_not_become_zero_reward(self):
        for value in ([0] * 7, [0] * 7 + [None], [0] * 7 + [0.5]):
            with self.assertRaises(ValueError):
                native_j_advantages(value)


def synthetic_fixture(directory):
    import torch
    from mm_core.training import state_hash, trainable_state
    from sr_f3_ce.training import fresh_optimizer

    root = Path(directory)
    (root / "audit").mkdir()
    (root / "image.png").write_bytes(b"synthetic processor fixture; not a real image")
    image_hash = hashlib.sha256((root / "image.png").read_bytes()).hexdigest()
    plan = copy.deepcopy(json.loads((ROOT / "frozen/receipt_plans/paths/b0-11031-R.json").read_text())[0])

    class SyntheticRuntime:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1, bias=False).float()
            with torch.no_grad():
                self.model.weight.zero_()
            self.kl_displacement = 0.
            self.gradient_scale = 1.
            self.sampler_mismatch = 0.
            self.parameter_factors = {}
            self.identity = dict(experiment_id="SR-ORD-TRAIN-1", execution_mode="TRAINING", arm="R",
                                 parent_id="b0", train_seed=11031, step=0, run_id="synthetic-b0-R-11031",
                                 run_root=str(root), audit_root=str(root / "audit"), source_sha256="b" * 64,
                                 manifest_sha256="c" * 64, plan_step_sha256=canonical_hash(plan))
            self.reference = trainable_state(self.model)
            self.identity["reference_hash"] = state_hash(self.reference)

        def prepare_input(self, model_input, run_root, *, protocol):
            return dict(routing={"input_tensor_hash": "d" * 64},
                        inputs={"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.tensor([[1.]])})

        def batch_reference_forward(self, items, reference):
            return [torch.full((len(row["tokens"]),), -2. + self.kl_displacement, dtype=torch.float32)
                    for row in items]

        def batch_sequence_forward(self, items, *, purpose, grad):
            return [self.model.weight.flatten()[0] * self.parameter_factors[row["row_id"]] * self.gradient_scale
                    + torch.full((len(row["tokens"]),), -2. + self.sampler_mismatch, dtype=torch.float32)
                    for row in items]

    runtime = SyntheticRuntime()
    policy_hash = state_hash(trainable_state(runtime.model))
    rows = []
    for group in plan["groups"]:
        for draw, request_id in enumerate(group["request_ids"]):
            model_input = dict(prompt=f"synthetic {group['qid']}", image_file="image.png", image_sha256=image_hash)
            # Variable lengths detect accidental token weighting instead of /128.
            tokens = [3] * (draw + 1)
            row = dict(group_id=group["group_id"], qid=group["qid"], root_id=group["root_id"],
                       input_id=group["input_id"], case_id=group["case_id"], request_id=request_id, row_id=request_id,
                       seed=group["seeds"][draw], draw=draw, step=1, parent="b0", arm="R", train_seed=11031,
                       run_id=runtime.identity["run_id"], policy_hash=policy_hash, reference_hash=policy_hash,
                       source_sha256="b" * 64, protocol="structured_v4", assistant_prefill="{",
                       model_input=model_input, prompt_sha256=hashlib.sha256(model_input["prompt"].encode()).hexdigest(),
                       prompt_tensor_hash="d" * 64, native_prompt_inputs={"input_ids": [[1, 2]]}, prompt_token_count=2,
                       tokens=tokens, raw_tokens=tokens[:], completion_token_count=len(tokens),
                       sampler_logprobs=[-2.] * len(tokens),
                       sampler_logprob_source="actual_generate_raw_logits_selected_token",
                       sampling_parameters=dict(max_new_tokens=1536, temperature=1., top_p=1., top_k=0, do_sample=True),
                       score=dict(J=draw % 2, V=1, finished=1))
            row["token_path_sha256"] = trajectory_hash(row)
            runtime.parameter_factors[request_id] = 1. if draw % 2 else -1.
            rows.append(row)
    optimizer, scheduler = fresh_optimizer(runtime.model)
    return runtime, optimizer, scheduler, rows, runtime.reference, plan


@unittest.skipUnless(HAS_TORCH, "Requires existing torch environment; not a passing GPU check")
class SyntheticUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = synthetic_fixture(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_equal_sequence_weight_scalar_gradient_and_update(self):
        import torch
        from sr_f6.objective import sequence_loss
        runtime, optimizer, scheduler, records, reference, plan = self.fixture
        parameter = torch.zeros((), requires_grad=True)
        native_loss = 0.
        for row in records:
            factor = runtime.parameter_factors[row["row_id"]]
            current = parameter * factor + torch.full((len(row["tokens"]),), -2.)
            native_loss = native_loss + sequence_loss(current, current.detach(), current.detach(), factor, 0.)["loss"] / 128
        native_loss.backward()
        report = update(*self.fixture)
        self.assertEqual(report["loss"], float(native_loss.detach()))
        self.assertAlmostEqual(report["numerics"]["gradient_norm_before_clip"], abs(float(parameter.grad)), places=6)
        self.assertEqual(report["status"], "UPDATED_UNCOMMITTED")
        self.assertEqual(runtime.identity["step"], 0)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(report["responses"], 128)
        self.assertEqual(len(report["probability_vectors"]), 128)
        self.assertGreater(float(runtime.model.weight), 0.)
        events = [json.loads(s) for s in (Path(self.temp.name) / "audit/NUMERICAL_TOKEN_AUDIT.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["status"], "PREUPDATE_PASS")

    def assert_blocked_without_adam(self):
        from mm_core.training import state_hash, trainable_state
        runtime, optimizer, scheduler, *_ = self.fixture
        before = state_hash(trainable_state(runtime.model))
        with self.assertRaises((ValueError, PermissionError, FloatingPointError)):
            update(*self.fixture)
        self.assertEqual(before, state_hash(trainable_state(runtime.model)))
        self.assertEqual(optimizer.state, {})
        self.assertEqual(scheduler.last_epoch, 0)

    def test_kl_gate_holds_before_adam(self):
        self.fixture[0].kl_displacement = 2.
        self.assert_blocked_without_adam()

    def test_gradient_gate_holds_before_adam(self):
        self.fixture[0].gradient_scale = 2000.
        self.assert_blocked_without_adam()

    def test_sampler_probability_gate_holds_before_adam(self):
        self.fixture[0].sampler_mismatch = .02
        self.assert_blocked_without_adam()

    def test_other_arm_rollout_is_rejected(self):
        self.fixture[3][0]["arm"] = "D"
        self.assert_blocked_without_adam()

    def test_unknown_seed_is_rejected(self):
        self.fixture[3][0]["seed"] += 1
        self.assert_blocked_without_adam()

    def test_token_mutation_is_rejected(self):
        self.fixture[3][0]["tokens"][0] = 9
        self.assert_blocked_without_adam()

    def test_prompt_tensor_mismatch_is_rejected(self):
        self.fixture[3][0]["native_prompt_inputs"] = {"input_ids": [[8, 2]]}
        self.fixture[3][0]["token_path_sha256"] = trajectory_hash(self.fixture[3][0])
        self.assert_blocked_without_adam()


if __name__ == "__main__":
    unittest.main()
