"""CPU fixtures prove the original update aborts before Adam without mutation."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get('SR_ORD_NATIVE_SRC', ROOT/'execution/historical_readonly/native_source/src'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NATIVE))
sys.path.insert(0, str(Path(__file__).parent))
from diagnostics.full_batch_memory_probe import (StopBeforeAdam, collect_baseline,
    compare_probability_vectors, immutable_state, probe_update, require_output, step_memory_budget)

HAS_TORCH = importlib.util.find_spec('torch') is not None


class FullBatchGuardTests(unittest.TestCase):
    def test_stop_is_outside_native_exception_handler(self):
        self.assertTrue(issubclass(StopBeforeAdam, BaseException))
        self.assertFalse(issubclass(StopBeforeAdam, Exception))

    def test_worker_grant_can_only_reduce_memory_budget(self):
        host = {'effective_mib':67584, 'status':'PASS'}
        self.assertEqual(step_memory_budget(host, 33792)['effective_mib'], 33792)
        self.assertEqual(step_memory_budget(host, 24000)['status'], 'BLOCKED_HOST_MEMORY')
        self.assertEqual(step_memory_budget({'effective_mib':20000}, 33792)['status'], 'BLOCKED_HOST_MEMORY')
        for invalid in (0, -1, '33792', True):
            with self.assertRaises(PermissionError):
                step_memory_budget(host, invalid)

    def test_output_rejects_original_paths(self):
        with self.assertRaises(PermissionError):
            require_output(ROOT/'run/paths/b0-11031-R')

    def test_probability_comparison_requires_all_128_original_paths(self):
        rows = [dict(row_id=str(i), tokens=[1], token_path_sha256='a'*64,
                     train_logprobs=[-2.], reference_logprobs=[-3.]) for i in range(128)]
        self.assertEqual(compare_probability_vectors(rows, rows)['status'], 'EXACT_ALL_128')
        changed = [dict(row) for row in rows]
        changed[3]['train_logprobs'] = [-2.0001]
        self.assertEqual(compare_probability_vectors(rows, changed)['status'], 'PROBABILITY_DIFFERENCE')
        with self.assertRaises(PermissionError):
            compare_probability_vectors(rows, changed[:-1])
        changed[3]['row_id'] = 'different'
        with self.assertRaises(PermissionError):
            compare_probability_vectors(rows, changed)


@unittest.skipUnless(HAS_TORCH, 'Requires actual Torch/native source CPU fixture environment')
class FullBatchSyntheticTests(unittest.TestCase):
    def setUp(self):
        import torch
        from test_training import synthetic_fixture
        from adapter.training import update, canonical_hash, trajectory_hash
        from mm_core.training import state_hash, trainable_state
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = synthetic_fixture(self.root)
        runtime, optimizer, scheduler, rows, reference, plan = self.fixture
        # Prime real AdamW state using the existing native synthetic fixture only.
        update(*self.fixture)
        self.assertTrue(optimizer.state)
        self.assertEqual(scheduler.last_epoch, 1)
        audit = self.root/'fullbatch_audit'
        audit.mkdir()
        runtime.identity.update(step=1, audit_root=str(audit))
        plan['step'] = 2
        runtime.identity['plan_step_sha256'] = canonical_hash(plan)
        policy = state_hash(trainable_state(runtime.model))
        for row in rows:
            row.update(step=2, policy_hash=policy)
            row['sampler_logprobs'] = runtime.batch_sequence_forward([row], purpose='training_policy', grad=False)[0].detach().tolist()
            row['token_path_sha256'] = trajectory_hash(row)
        # A value-preserving fixture-only operation forces saved tensors on CPU.
        original = runtime.batch_sequence_forward
        def saves_tensor(items, *, purpose, grad):
            return [value * torch.ones_like(value) for value in original(items, purpose=purpose, grad=grad)]
        runtime.batch_sequence_forward = saves_tensor
        self.output = self.root/'probe_output'
        self.output.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def before(self):
        from mm_core.training import capture_rng, state_hash
        runtime, optimizer, scheduler, _, reference, _ = self.fixture
        return immutable_state(runtime, optimizer, scheduler, reference), state_hash(capture_rng())

    def assert_unchanged(self, before):
        from mm_core.training import capture_rng, state_hash
        runtime, optimizer, scheduler, _, reference, _ = self.fixture
        self.assertEqual(immutable_state(runtime, optimizer, scheduler, reference), before[0])
        self.assertEqual(state_hash(capture_rng()), before[1])
        self.assertTrue(all(p.grad is None for p in runtime.model.parameters()))
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertTrue(all(float(state['step']) == 1 for state in optimizer.state.values()))
        self.assertEqual(len(optimizer._optimizer_step_pre_hooks), 0)

    def test_full128_native_gates_gradients_and_pre_hook_stop_before_adam(self):
        import torch
        runtime, optimizer, scheduler, rows, reference, plan = self.fixture
        before = self.before()
        baseline = collect_baseline(runtime, rows, reference, self.output/'baseline.jsonl')
        result = probe_update(*self.fixture, self.output, force_cpu_copy=True)
        self.assertEqual(result['status'], 'FULL_BATCH_PRE_ADAM_PASS_NO_UPDATE')
        self.assertEqual(result['gradient_forwards'], 128)
        self.assertEqual(result['optimizer_body_calls'], 0)
        self.assertEqual(result['scheduler_steps'], 0)
        self.assertEqual(result['capture']['pre_hook_calls'], 1)
        self.assertGreater(result['saved_tensor_copied_bytes_total'], 0)
        self.assertLessEqual(result['saved_tensor_max_bytes_per_forward'], 4*1024**3)
        self.assertEqual(compare_probability_vectors(baseline, result['native_audit']['probability_vectors'])['status'],
                         'EXACT_ALL_128')
        gradients = torch.load(result['capture']['gradient_file'], weights_only=True)
        self.assertEqual(set(gradients), {'weight'})
        self.assertTrue(torch.isfinite(gradients['weight']).all())
        self.assertLessEqual(float(gradients['weight'].norm()), 1+1e-5)
        events = [json.loads(line) for line in (Path(runtime.identity['audit_root'])/'NUMERICAL_TOKEN_AUDIT.jsonl').read_text().splitlines()]
        self.assertEqual([event['status'] for event in events], ['PREUPDATE_PASS'])
        self.assertEqual(len(events[0]['probability_vectors']), 128)
        self.assert_unchanged(before)

    def test_original_kl_gate_prevents_pre_adam_capture_and_state_change(self):
        self.fixture[0].kl_displacement = 2.
        before = self.before()
        with self.assertRaises((PermissionError, ValueError, FloatingPointError)):
            probe_update(*self.fixture, self.output, force_cpu_copy=True)
        capture = json.loads((self.output/'PRE_ADAM_CAPTURE.json').read_text())
        self.assertEqual(capture['pre_hook_calls'], 0)
        self.assert_unchanged(before)

    def test_capture_failure_still_stops_before_adam_and_preserves_native_finally(self):
        before = self.before()
        with patch('torch.save', side_effect=OSError('Synthetic snapshot failure')):
            with self.assertRaisesRegex(PermissionError, 'verified pre-Adam stop'):
                probe_update(*self.fixture, self.output, force_cpu_copy=True)
        capture = json.loads((self.output/'PRE_ADAM_CAPTURE.json').read_text())
        self.assertEqual(capture['status'], 'PRE_ADAM_CAPTURE_FAILED')
        self.assertEqual(capture['pre_hook_calls'], 1)
        events = [json.loads(line) for line in (Path(self.fixture[0].identity['audit_root'])/'NUMERICAL_TOKEN_AUDIT.jsonl').read_text().splitlines()]
        self.assertEqual([event['status'] for event in events], ['PREUPDATE_PASS'])
        self.assert_unchanged(before)


if __name__ == '__main__':
    unittest.main()
