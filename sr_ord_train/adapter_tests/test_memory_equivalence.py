"""Small deterministic CPU fixture and fail-closed diagnostic input tests."""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from diagnostics.memory_equivalence import (BoundedSavedTensors, OFFLOAD_CAP_BYTES,
    OFFLOAD_MODE, compare_results, memory_budget, readonly_path, record_final_audit,
    require_output, select_probes, objective_evidence)
from adapter.transactions import canonical, digest

try:
    import torch
except ImportError:
    torch = None


class DiagnosticSelectionTests(unittest.TestCase):
    def rows(self):
        return [{'request_id': 'row-'+str(i), 'tokens': [1]*(2 if i != 7 else 9)} for i in range(128)]

    def test_first_and_longest_are_predeclared_and_ties_use_frozen_order(self):
        rows = self.rows()
        rows[8]['tokens'] = [2]*9
        selected = select_probes(list(reversed(rows)), [r['request_id'] for r in rows])
        self.assertEqual([p['request_id'] for p in selected], ['row-0', 'row-7'])
        rows[0]['tokens'] = [3]*9
        selected = select_probes(rows, [r['request_id'] for r in rows])
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(selected[0]['roles']), 2)

    def test_missing_duplicate_and_invalid_tokens_are_rejected(self):
        rows = self.rows()
        ids = [r['request_id'] for r in rows]
        for invalid in (rows[:-1], rows[:-1]+[rows[0]], [dict(rows[0], tokens=[])]+rows[1:]):
            with self.assertRaises(PermissionError):
                select_probes(invalid, ids)

    def test_memory_floor_uses_per_worker_grant_cgroup_and_available(self):
        self.assertEqual(memory_budget({'SLURM_MEM_PER_NODE':'33792','SLURM_NTASKS':'1'})['status'], 'PASS')
        self.assertEqual(memory_budget({'SLURM_MEM_PER_NODE':'33792','SLURM_NTASKS':'2'})['status'], 'BLOCKED_HOST_MEMORY')
        self.assertEqual(memory_budget({'SLURM_MEM_PER_NODE':'67584','SLURM_NTASKS':'1'},
                         cgroup_limit_bytes=27*1024**3)['status'], 'BLOCKED_HOST_MEMORY')
        with self.assertRaises(PermissionError):
            memory_budget({})

    def test_output_cannot_be_original_path(self):
        with self.assertRaises(PermissionError):
            require_output(Path(__file__).parents[1]/'run/paths/b0-11031-R')

    def test_unknown_or_staging_attempts_and_missing_identity_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder).resolve()
            with self.assertRaises(FileNotFoundError):
                with readonly_path(path):
                    pass
            identity = dict(experiment_id='SR-ORD-TRAIN-1', parent='b0', arm='R', train_seed=11031,
                            run_id='SR-ORD-TRAIN-1/b0-11031-R', manifest_sha256='a'*64,
                            plan_sha256='b'*64, source_sha256='c'*64)
            (path/'IDENTITY.json').write_bytes(canonical(identity))
            (path/'LOCK').touch()
            (path/'checkpoints').mkdir()
            (path/'checkpoints/.staging-interrupted').mkdir()
            before = sorted(str(p.relative_to(path)) for p in path.rglob('*'))
            with self.assertRaisesRegex(PermissionError, 'staging'):
                with readonly_path(path):
                    pass
            self.assertEqual(before, sorted(str(p.relative_to(path)) for p in path.rglob('*')))
            (path/'checkpoints/.staging-interrupted').rmdir()
            context = dict(identity_sha256=digest(identity), request={'request_id':'x'},
                           policy_sha256='d'*64, step=1)
            begin = dict(event='BEGIN', attempt_id=digest(context), previous_sha256='0'*64, **context)
            (path/'ATTEMPTS.jsonl').write_bytes(canonical(begin)+b'\n')
            before_bytes = (path/'ATTEMPTS.jsonl').read_bytes()
            with self.assertRaisesRegex(PermissionError, 'UNKNOWN'):
                with readonly_path(path):
                    pass
            self.assertEqual((path/'ATTEMPTS.jsonl').read_bytes(), before_bytes)

    def test_baseline_oom_cannot_be_called_equivalent(self):
        result = compare_results({'status':'OOM'}, {'status':'COMPLETE'})
        self.assertEqual(result['status'], 'EQUIVALENCE_UNVERIFIED')
        self.assertFalse(result['production_recovery_authorized'])

    def test_comparing_two_baselines_is_rejected(self):
        with self.assertRaisesRegex(PermissionError, 'Comparison requires'):
            compare_results({'status':'COMPLETE', 'mode':'baseline'},
                            {'status':'COMPLETE', 'mode':'baseline'})

    def test_unexercised_offload_or_missing_final_audit_cannot_pass(self):
        baseline = {'status':'COMPLETE', 'mode':'baseline', 'state_unchanged_after_attempt':True,
                    'rng_restored_after_attempt':True}
        offload = dict(baseline, mode=OFFLOAD_MODE, saved_tensor_offload={'copied_bytes':0})
        self.assertEqual(compare_results(baseline, offload)['status'], 'EQUIVALENCE_UNVERIFIED')
        offload['saved_tensor_offload']['copied_bytes'] = 32
        for key in ('state_unchanged_after_attempt', 'rng_restored_after_attempt'):
            with self.subTest(key=key):
                invalid = dict(offload, **{key:False})
                self.assertEqual(compare_results(baseline, invalid)['status'], 'EQUIVALENCE_UNVERIFIED')

    def test_failed_final_audit_retains_original_oom(self):
        def fail():
            raise RuntimeError('Audit unavailable')
        result = {'status':'OOM', 'error':'Original OOM', 'traceback':'Original failure trace'}
        record_final_audit(result, fail)
        self.assertEqual(result['status'], 'OOM')
        self.assertEqual(result['error'], 'Original OOM')
        self.assertEqual(result['traceback'], 'Original failure trace')
        self.assertEqual(result['final_audit_error']['error'], 'Audit unavailable')
        complete = {'status':'COMPLETE'}
        record_final_audit(complete, fail)
        self.assertEqual(complete['status'], 'FAILED_FINAL_AUDIT')


@unittest.skipIf(torch is None, 'PyTorch unavailable: CPU equivalence fixture not executed')
class BoundedHookTests(unittest.TestCase):
    def test_objective_evidence_preserves_vectors_booleans_and_metadata(self):
        import json
        terms = {'loss': torch.tensor(1.25, requires_grad=True),
                 'ratio': torch.tensor([1., 2.]), 'singleton': torch.tensor([3.]),
                 'tis_truncated': torch.tensor([False, True]),
                 'token_count': 2, 'kl_coefficient': .02}
        scalars, tensors = objective_evidence(terms)
        self.assertEqual(set(scalars) | set(tensors), set(terms))
        self.assertEqual(scalars, {'loss': 1.25, 'token_count': 2, 'kl_coefficient': .02})
        self.assertEqual(tensors['ratio']['values'], [1., 2.])
        self.assertEqual(tensors['singleton']['shape'], [1])
        self.assertEqual(tensors['tis_truncated']['values'], [False, True])
        json.dumps({'scalars': scalars, 'tensors': tensors}, allow_nan=False)
        with self.assertRaisesRegex(PermissionError, 'Nonfinite'):
            objective_evidence({'loss': torch.tensor(1.), 'ratio': torch.tensor([float('nan')])})

    def test_real_native_objective_evidence_retains_all_fields(self):
        import importlib.util
        from adapter.transactions import file_hash
        from runtime_contract import PINS
        source = Path('/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_b7e7b06/src/sr_f6/objective.py')
        if not source.exists():
            self.skipTest('Pinned native source is only available on the execution server')
        self.assertEqual(file_hash(source), PINS['sr_f6/objective.py'])
        spec = importlib.util.spec_from_file_location('native_objective_regression', source)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        current = torch.tensor([-1.0, -2.0], requires_grad=True)
        terms = native.sequence_loss(current, torch.tensor([-1.1, -2.1]),
                                     torch.tensor([-1.05, -2.05]), 1., 0.)
        scalars, tensors = objective_evidence(terms)
        self.assertEqual(set(scalars) | set(tensors), set(terms))
        self.assertEqual(scalars['token_count'], 2)
        self.assertEqual(tensors['ratio']['values'], terms['ratio'].detach().tolist())
        self.assertIn('tis_truncated', tensors)

    def test_cumulative_cap_does_not_reset_on_unpack_and_snapshot_owns_bytes(self):
        source = torch.arange(8, dtype=torch.float32, requires_grad=True)
        hook = BoundedSavedTensors(cap_bytes=32, force_cpu_copy=True)
        saved = hook.pack(source)
        with torch.no_grad():
            source.add_(1)
        restored = hook.unpack(saved)
        self.assertTrue(torch.equal(restored, torch.arange(8, dtype=torch.float32)))
        self.assertEqual(restored.dtype, source.dtype)
        self.assertEqual(restored.device, source.device)
        self.assertFalse(saved[1].requires_grad)
        kept = hook.pack(source)
        self.assertEqual(kept[0], 'kept')
        self.assertEqual(hook.stats['copied_bytes'], 32)
        self.assertFalse(kept[1].requires_grad)
        with self.assertRaises(PermissionError):
            BoundedSavedTensors(cap_bytes=OFFLOAD_CAP_BYTES+1)

    def test_parameter_storage_aliases_noncontiguous_and_cpu_default_are_kept(self):
        parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3,4))
        hook = BoundedSavedTensors([parameter], cap_bytes=1024, force_cpu_copy=True)
        self.assertEqual(hook.pack(parameter[:, 1:])[0], 'kept')
        self.assertEqual(hook.stats['persistent_storage_skips'], 1)
        self.assertEqual(hook.pack(torch.ones(2,3).t())[0], 'kept')
        default = BoundedSavedTensors(cap_bytes=1024)
        self.assertEqual(default.pack(torch.ones(4))[0], 'kept')
        self.assertEqual(hook.stats['copied_bytes'], 0)

    def test_scalar_outputs_and_every_adapter_gradient_are_exact_on_cpu_fixture(self):
        def execute(offload):
            torch.manual_seed(94031)
            adapter_a = torch.nn.Parameter(torch.randn(4,3))
            adapter_b = torch.nn.Parameter(torch.randn(3,2))
            frozen = torch.randn(4,2)
            x = torch.randn(5,4)
            hook = BoundedSavedTensors([adapter_a, adapter_b, frozen], cap_bytes=4096,
                                      force_cpu_copy=True)
            with hook.context() if offload else nullcontext():
                logits = x @ frozen + (x @ adapter_a).tanh() @ adapter_b
                selected = logits.log_softmax(-1)[:, 0]
                loss = (selected.square().mean() + selected.exp().mean()) / 128
            loss.backward()
            return selected.detach(), loss.detach(), [adapter_a.grad, adapter_b.grad], hook.stats
        baseline, offload = execute(False), execute(True)
        self.assertTrue(torch.equal(baseline[0], offload[0]))
        self.assertTrue(torch.equal(baseline[1], offload[1]))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(baseline[2],offload[2])))
        self.assertGreater(offload[3]['copied_bytes'], 0)
        self.assertLessEqual(offload[3]['copied_bytes'], 4096)


if __name__ == '__main__':
    unittest.main()
