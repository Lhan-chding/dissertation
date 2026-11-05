"""CPU unit/real native tiny-fixture regression; not a GPU equivalence claim."""
import copy
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, os.environ.get('SR_ORD_NATIVE_SRC', str(ROOT/'execution/historical_readonly/native_source/src')))
from gpu_production.update import compare_audit, verify_storage, production_update


class AuditTests(unittest.TestCase):
    def test_only_gpu_rng_hash_is_excluded_from_full_audit_comparison(self):
        a = dict(rng_before_sha256='a', totals={'loss':1.}, probability_vectors=[{'tokens':[1]}])
        b = dict(a, rng_before_sha256='b')
        compare_audit(a, b)
        b['totals'] = {'loss':1.00001}
        with self.assertRaises(PermissionError): compare_audit(a, b)
        b = copy.deepcopy(a); b['probability_vectors'][0]['tokens'] = [2]
        with self.assertRaises(PermissionError): compare_audit(a, b)

    def test_transfer_gates_reject_missing_unpack_cpu_and_reorder(self):
        rows = [dict(row_id=str(i)) for i in range(128)]
        good = [dict(row_id=str(i), forward_returned=True, stats=dict(copied_bytes=1, unpacked_copies=1,
            version_failures=0, copy_failures=0, unpack_failures=0, cpu_offload=False)) for i in range(128)]
        verify_storage(good, rows)
        for key, value in [('copied_bytes',0),('unpacked_copies',0),('version_failures',1),
                           ('copy_failures',1),('unpack_failures',1),('cpu_offload',True)]:
            changed = copy.deepcopy(good); changed[4]['stats'][key] = value
            with self.assertRaises(PermissionError): verify_storage(changed, rows)
        with self.assertRaises(PermissionError): verify_storage(list(reversed(good)), rows)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires server Torch/native CPU fixture')
class NativeUpdateTests(unittest.TestCase):
    def setUp(self):
        import torch
        from test_training import synthetic_fixture
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'base').mkdir(); (self.root/'wrapped').mkdir()
        self.base = synthetic_fixture(self.root/'base')
        self.wrapped = synthetic_fixture(self.root/'wrapped')
        class StorageFixture:
            def __init__(self, persistent_tensors, *, row_id):
                self.stats=dict(row_id=row_id,copied_bytes=1,unpacked_copies=1,version_failures=0,
                                copy_failures=0,unpack_failures=0,cpu_offload=False)
            def context(self): return nullcontext()
        for mocked in [patch('gpu_production.update.GPUSavedTensors', StorageFixture),
                       patch('gpu_production.update.validate_compute_placement'),
                       patch('gpu_production.update.gpu_memory_snapshot', return_value={}),
                       patch('torch.cuda.get_rng_state', return_value=torch.zeros(8,dtype=torch.uint8))]:
            mocked.start(); self.addCleanup(mocked.stop)

    def test_real_native_one_adam_update_matches_unwrapped_cpu_fixture(self):
        from adapter.training import update
        from mm_core.training import state_hash, trainable_state
        baseline = update(*self.base)
        runtime,opt,scheduler,rows,reference,plan = self.wrapped
        result = production_update(*self.wrapped, Path(runtime.identity['audit_root']), {})
        for key in ['loss','native','aux','kl','kl_unweighted','numerics','parameter_update_norm']:
            self.assertEqual(result[key],baseline[key])
        self.assertEqual(state_hash(trainable_state(runtime.model)),state_hash(trainable_state(self.base[0].model)))
        self.assertEqual(state_hash(opt.state_dict()),state_hash(self.base[1].state_dict()))
        self.assertEqual(scheduler.state_dict(),self.base[2].state_dict())
        accounting=json.loads(Path(result['gpu_update_accounting']['path']).read_text())
        self.assertTrue(accounting['adam_permitted'])
        self.assertEqual(accounting['optimizer_returned_calls'],1)
        self.assertEqual(accounting['scheduler_returned_calls'],1)
        self.assertEqual(len(accounting['forwards']),128)
        self.assertFalse(opt._optimizer_step_pre_hooks or opt._optimizer_step_post_hooks)

    def test_denied_pre_adam_gate_records_zero_actual_update_and_restores_methods(self):
        from mm_core.training import state_hash,trainable_state
        runtime,opt,scheduler,rows,reference,plan=self.wrapped
        before=state_hash(trainable_state(runtime.model));original=runtime.batch_sequence_forward
        with patch('gpu_production.update.verify_storage', side_effect=PermissionError('fixture denial')):
            with self.assertRaisesRegex(PermissionError,'fixture denial'):
                production_update(*self.wrapped,Path(runtime.identity['audit_root']),{})
        self.assertEqual(state_hash(trainable_state(runtime.model)),before)
        self.assertFalse(opt.state);self.assertEqual(scheduler.last_epoch,0)
        self.assertEqual(runtime.batch_sequence_forward,original)
        a=json.loads((Path(runtime.identity['audit_root'])/'GPU_UPDATE_ACCOUNTING_step-001.json').read_text())
        self.assertFalse(a['adam_permitted']);self.assertEqual(a['optimizer_returned_calls'],0)
        self.assertEqual(a['scheduler_entered_calls'],0);self.assertEqual(a['status'],'FAILED')
        self.assertFalse(opt._optimizer_step_pre_hooks or opt._optimizer_step_post_hooks)

    def test_telemetry_failure_preserves_completed_adam_accounting_and_blocks_commit(self):
        runtime,opt,scheduler,*unused=self.wrapped
        with patch('gpu_production.update.gpu_memory_snapshot', side_effect=[{}, RuntimeError('telemetry failed')]):
            with self.assertRaisesRegex(PermissionError,'GPU telemetry failed'):
                production_update(*self.wrapped,Path(runtime.identity['audit_root']),{})
        a=json.loads((Path(runtime.identity['audit_root'])/'GPU_UPDATE_ACCOUNTING_step-001.json').read_text())
        self.assertEqual(a['optimizer_returned_calls'],1)
        self.assertEqual(a['scheduler_returned_calls'],1)
        self.assertEqual(a['status'],'FAILED')
        self.assertEqual(a['gpu_memory_error']['error'],'telemetry failed')


if __name__ == '__main__': unittest.main()
