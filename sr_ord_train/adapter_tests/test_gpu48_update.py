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
from gpu_production_v2.update import compare_audit, verify_storage, production_update, first_recovery_evidence
from adapter.transactions import file_hash
from types import SimpleNamespace


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
            version_failures=0, copy_failures=0, unpack_failures=0, cpu_offload=False,cap_bytes=48*1024**3,mode='bounded_gpu_saved_tensors_capacity48_v2')) for i in range(128)]
        verify_storage(good, rows)
        for key, value in [('copied_bytes',0),('unpacked_copies',0),('version_failures',1),
                           ('copy_failures',1),('unpack_failures',1),('cpu_offload',True),('cap_bytes',16*1024**3),('mode','bounded_gpu_saved_tensors')]:
            changed = copy.deepcopy(good); changed[4]['stats'][key] = value
            with self.assertRaises((PermissionError, RuntimeError)): verify_storage(changed, rows)
        with self.assertRaises(PermissionError): verify_storage(list(reversed(good)), rows)


class FirstRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.runtime=SimpleNamespace(identity={'run_id':'SR-ORD-TRAIN-1/b0-11031-G'},model=object())
        self.audit={'run_id':self.runtime.identity['run_id'],'step':9,'totals':{'loss':1.},'probability_vectors':[{'tokens':[1]}]}
        self.summary={'path':'b0-11031-G','status':'FULL_BATCH_PRE_ADAM_PASS_NO_UPDATE','native_audit':copy.deepcopy(self.audit),
                      'capture':{'gradient_file':str(self.root/'grads.pt'),'gradient_file_sha256':'a'*64}}
        self.binding={};self.write_evidence()

    def write_evidence(self):
        summary=self.root/'SUMMARY.json';summary.write_text(json.dumps(self.summary))
        acceptance=self.root/'ACCEPTANCE.json';acceptance.write_text(json.dumps({'diagnostic_summary':{'path':str(summary),'sha256':file_hash(summary)}}))
        self.binding={'gpu_production':{'acceptance':{'path':str(acceptance),'sha256':file_hash(acceptance)}}}

    def test_b0g_step9_requires_exact_audit_and496_gradients(self):
        with patch('gpu_production_v2.update.compare_gradients',return_value=496) as compare:
            result=first_recovery_evidence(self.runtime,9,self.binding,self.audit)
            self.assertEqual(result['compared_gradients'],496)
            self.assertEqual(result['first_recovery_comparison'],'EXACT_DIAGNOSTIC_128_AND_CLIPPED_GRADIENTS')
            compare.assert_called_once_with(self.runtime.model,self.root/'grads.pt','a'*64)
        with patch('gpu_production_v2.update.compare_gradients',return_value=495):
            with self.assertRaises(PermissionError):first_recovery_evidence(self.runtime,9,self.binding,self.audit)

    def test_changed_probabilities_deny_before_gradient_comparison(self):
        changed=copy.deepcopy(self.audit);changed['probability_vectors'][0]['tokens']=[2]
        with patch('gpu_production_v2.update.compare_gradients') as compare:
            with self.assertRaises(PermissionError):first_recovery_evidence(self.runtime,9,self.binding,changed)
            compare.assert_not_called()

    def test_wrong_diagnostic_path_and_step_deny(self):
        for key,value in [('path','b0-11031-R'),('step',5)]:
            saved=copy.deepcopy(self.summary)
            if key=='step':self.summary['native_audit']['step']=value
            else:self.summary[key]=value
            self.write_evidence()
            with self.assertRaises(PermissionError):first_recovery_evidence(self.runtime,9,self.binding,self.audit)
            self.summary=saved

    def test_changed_evidence_bytes_deny(self):
        p=self.root/'SUMMARY.json';p.write_text(p.read_text()+'\n')
        with self.assertRaises(PermissionError):first_recovery_evidence(self.runtime,9,self.binding,self.audit)

    def test_other_paths_or_steps_do_not_reuse_b0g_diagnostic(self):
        for run_id,step in [('SR-ORD-TRAIN-1/b0-11031-R',9),('SR-ORD-TRAIN-1/b0-11031-G',10),('SR-ORD-TRAIN-1/b1-11031-G',3)]:
            self.runtime.identity['run_id']=run_id
            self.assertEqual(first_recovery_evidence(self.runtime,step,{},{}),
                             {'first_recovery_comparison':'NOT_THIS_REGISTERED_B0G_STEP9'})


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
                                copy_failures=0,unpack_failures=0,cpu_offload=False,cap_bytes=48*1024**3,
                                mode='bounded_gpu_saved_tensors_capacity48_v2')
            def context(self): return nullcontext()
        for mocked in [patch('gpu_production_v2.update.GPUSavedTensors', StorageFixture),
                       patch('gpu_production_v2.update.validate_compute_placement'),
                       patch('gpu_production_v2.update.gpu_memory_snapshot', return_value={}),
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
        with patch('gpu_production_v2.update.verify_storage', side_effect=PermissionError('fixture denial')):
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
        with patch('gpu_production_v2.update.gpu_memory_snapshot', side_effect=[{}, RuntimeError('telemetry failed')]):
            with self.assertRaisesRegex(PermissionError,'GPU telemetry failed'):
                production_update(*self.wrapped,Path(runtime.identity['audit_root']),{})
        a=json.loads((Path(runtime.identity['audit_root'])/'GPU_UPDATE_ACCOUNTING_step-001.json').read_text())
        self.assertEqual(a['optimizer_returned_calls'],1)
        self.assertEqual(a['scheduler_returned_calls'],1)
        self.assertEqual(a['status'],'FAILED')
        self.assertEqual(a['gpu_memory_error']['error'],'telemetry failed')


if __name__ == '__main__': unittest.main()
