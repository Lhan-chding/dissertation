"""CPU control-flow tests: real 496-tensor native Adam/state/transactions.

Sampler, model forwards, CUDA fingerprints, storage observations and canary
teacher kernels are explicitly faked. These tests do not certify actual GPU,
quota, process exit or production admission. No server/model is contacted.
"""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import first_two_control as c
sys.path.pop(0)
spec=importlib.util.spec_from_file_location('_first_two_step_fixture',Path(__file__).with_name('test_training_step.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
t=c.tx


@unittest.skipUnless((f.f.fixtures.NATIVE/'sr_f3_ce/training.py').exists(),'Pinned actual native CPU optimizer required')
class ActualTransactionCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(f.TrainingStepTests.doClassCleanups);f.TrainingStepTests.setUpClass()
    def setUp(self):
        fixture=f.TrainingStepTests('test_real_transaction_adam_codec_and_idempotent_composition')
        fixture.setUp();self.addCleanup(fixture.doCleanups);self.f=fixture
        self.root=fixture.root;self.handle=fixture.handle;self.native=fixture.tiny.ops
        sources=fixture.sources
        for key,module in [('first_two_control',c),('canary_replay',c.canary_replay),
                           ('frozen_fingerprint',c.canary_audit),('storage_ready',c.storage_ready),
                           ('evaluation_analysis',c.evaluation),('canary_controller',c)]:
            sources[key]=dict(path=str(Path(module.__file__).resolve()),sha256=t.file_sha(module.__file__))
        identity=copy.deepcopy(self.handle.path_identity);identity['execution_sources_sha256']=c.object_sha(sources)
        self.handle.path_identity=copy.deepcopy(identity);fixture.fixture.identity=identity
        genesis=fixture.fixture.genesis(identity);self.handle.genesis_reference=genesis
        fixture.store=t.TransactionStore(self.root,fixture.fixture.freeze,fixture.fixture.freeze_sha,identity,
            native=self.native,genesis_reference=genesis);self.store=fixture.store
        self.slot=self.root/'first_two_control/start-11031/T0/baseline';self.slot.mkdir(parents=True)
        self.ctx=dict(slot=self.slot,sources=sources,authority=fixture.authority,release_ref={'CPU':'FAKE'},
            release=dict(storage_budget={'CPU':'FAKE'},admission=fixture.authority.refs['admission'],risk_clarification={'CPU':'FAKE'}),dev={'CPU':'FAKE'})
        self.fingerprints=[];self.teacher_calls=[];self.bad_frozen=False;self.fail_teacher=False
        def fingerprint(handle,*,phase,**kwargs):
            self.fingerprints.append((phase,handle.endpoint()['step']))
            return dict(parameters=[{'name':'CPU_NON_LORA','tensor_sha256':('changed' if self.bad_frozen and phase=='AFTER_STEP2' else 'unchanged')}],
                        gpu_inventory=['CPU_FAKE_NOT_GPU'],phase=phase)
        def baseline(handle,store,*,spec_path,expected_spec_sha256,execution_sources):
            self.teacher_calls.append(dict(step=handle.endpoint()['step'],spec=t.read_json(spec_path)))
            self.assertEqual(t.file_sha(spec_path),expected_spec_sha256)
            if self.fail_teacher:raise RuntimeError('CPU synthetic teacher failure')
            slot=spec_path.parent/'baseline';slot.mkdir()
            result=dict(status=c.canary_replay.BASELINE,request_count=128,CPU_SYNTHETIC=True)
            t.write_once(slot/'COMPLETE.json',result);return result
        for p in (patch.object(c,'ROOT',self.root),
                  patch.object(c.canary_audit,'fingerprint_first_two',side_effect=fingerprint),
                  patch.object(c.storage_ready,'make_storage_ready',return_value=fixture.storage_ready),
                  patch.object(c.canary_replay,'baseline',side_effect=baseline),
                  patch.object(c.canary_replay,'_process_identity',return_value={'host':'CPU','boot_id':'CPU','pid':101,'start_ticks':1})):
            p.start();self.addCleanup(p.stop)
    def run_baseline(self):return c._baseline(self.handle,self.store,self.ctx)
    def test_exact_two_native_updates_256_responses_and_readonly_baseline(self):
        result=self.run_baseline()
        self.assertEqual((len(self.f.generated),self.f.adam_calls,self.f.scheduler_calls),(256,2,2))
        self.assertEqual(self.fingerprints,[('BEFORE_FIRST_STEP',0),('AFTER_STEP2',2)])
        self.assertEqual(len(self.teacher_calls),1);self.assertEqual(self.teacher_calls[0]['step'],2)
        self.assertEqual(result['status'],c.BASELINE_STATUS);self.assertFalse(result['process_exit_observed'])
        self.assertEqual(set(result['commits']),{'1','2'});self.assertFalse((self.store.path/'step-003').exists())
        self.assertEqual(self.native.state_hash(self.handle.fork.reference),self.handle.path_identity['source_binding']['reference_sha256'])
        self.assertEqual(self.handle.fork.scheduler.last_epoch,2)
        for state in self.handle.fork.optimizer.state.values():self.assertEqual(float(state['step']),2)
    def test_duplicate_baseline_cannot_repeat_adam_or_generation(self):
        self.run_baseline();counts=(len(self.f.generated),self.f.adam_calls)
        with self.assertRaises(FileExistsError):self.run_baseline()
        self.assertEqual(counts,(len(self.f.generated),self.f.adam_calls))
    def test_incomplete_generation_aborts_before_adam_without_commit(self):
        self.f.fail_runner=True
        with self.assertRaisesRegex(RuntimeError,'missing completion'):self.run_baseline()
        self.assertEqual(self.f.adam_calls,0);self.assertFalse((self.store.path/'step-001/COMMIT.json').exists())
        self.assertTrue(self.handle._closed)
        before=len(self.f.generated)
        with self.assertRaises((PermissionError,FileExistsError)):self.run_baseline()
        self.assertEqual(len(self.f.generated),before)
    def test_actual_adam_return_failure_keeps_unknown_and_never_retries(self):
        self.f.fail_scheduler=True
        with self.assertRaisesRegex(RuntimeError,'after Adam'):self.run_baseline()
        self.assertEqual(self.f.adam_calls,1);self.assertFalse((self.store.path/'step-001/COMMIT.json').exists())
        self.assertTrue(any(p.name.startswith('.staging-') for p in self.store.path.iterdir()))
        with self.assertRaises((PermissionError,FileExistsError)):self.run_baseline()
        self.assertEqual(self.f.adam_calls,1)
    def test_frozen_parameter_change_stops_before_canary(self):
        self.bad_frozen=True
        with self.assertRaisesRegex(PermissionError,'parameters changed'):self.run_baseline()
        self.assertEqual(self.f.adam_calls,2);self.assertFalse(self.teacher_calls)
    def test_teacher_failure_does_not_reexecute_training(self):
        self.fail_teacher=True
        with self.assertRaisesRegex(RuntimeError,'teacher failure'):self.run_baseline()
        self.assertEqual(self.f.adam_calls,2)
        with self.assertRaises((PermissionError,FileExistsError)):self.run_baseline()
        self.assertEqual(self.f.adam_calls,2)


class ControlBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        p=patch.object(c,'ROOT',self.root);p.start();self.addCleanup(p.stop)
    def test_one_global_lock_excludes_second_worker_and_releases(self):
        with c._lease():
            with self.assertRaises(BlockingIOError):
                with c._lease():pass
        with c._lease():pass
    def test_unknown_baseline_or_reload_prevents_new_work(self):
        base=c._paths(11031,'T0')/'baseline';base.mkdir(parents=True)
        with self.assertRaisesRegex(PermissionError,'incomplete'):c._no_unknown()
        t.write_once(base/'RESULT.json',dict(status=c.BASELINE_STATUS,permission='NO_STEP3_OR_CANARY_ADMISSION'))
        c._no_unknown()
        (base.parent/'reload').mkdir()
        with self.assertRaisesRegex(PermissionError,'incomplete'):c._no_unknown()
    def test_global_hold_and_uncommitted_transaction_block(self):
        t.write_once(self.root/'HOLD.json',{})
        with self.assertRaises(PermissionError):c._no_unknown()
        (self.root/'HOLD.json').unlink()
        p=self.root/'training_transactions/start-11031/T0/.staging-step-001';p.mkdir(parents=True)
        with self.assertRaisesRegex(PermissionError,'uncommitted'):c._no_unknown()
    def test_outer_launcher_hold_blocks_before_model(self):
        directory=self.root/'first_two_launcher';directory.mkdir();t.write_once(directory/'HOLD.json',{})
        with self.assertRaisesRegex(PermissionError,'Global training HOLD'):c._no_unknown()
    def test_unknown_phase_path_rejected(self):
        (c._paths(11031,'T0')/'step3').mkdir(parents=True)
        with self.assertRaises(PermissionError):c._no_unknown()
    def test_published_step3_on_any_path_is_outside_first_two_scope(self):
        step=self.root/'training_transactions/start-11032/T1/step-003';step.mkdir(parents=True)
        t.write_once(step/'COMMIT.json',{'status':'COMMITTED'})
        with self.assertRaisesRegex(PermissionError,'beyond first two'):c._no_unknown()
    def test_extra_canary_slot_is_not_ignored(self):
        (self.root/'canary_replay/start-11032/T1/retry').mkdir(parents=True)
        with self.assertRaisesRegex(PermissionError,'canary slot'):c._no_unknown()
    def test_wrong_path_identity_rejected(self):
        for start,arm in ((11033,'T0'),(True,'T0'),(11031,'MAIN')):
            with self.subTest(start=start,arm=arm),self.assertRaises(PermissionError):c._paths(start,arm)
    def test_actual_slurm_qos_verified_instead_of_env_label(self):
        env={'SLURM_JOB_ID':'123','SLURM_NTASKS':'1','SLURM_NNODES':'1','SLURM_RESTART_COUNT':'0','SLURM_JOB_QOS':'fake'}
        fields='JobId=123 JobState=RUNNING QOS=soujanya-poria-startfund-2026-03 Account=rose NumNodes=1 AllocTRES=cpu=8,gres/gpu:pro6000=2'
        with patch.dict(os.environ,env,clear=True),patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=fields)):
            self.assertEqual(c._allocation()['qos'],'soujanya-poria-startfund-2026-03')
        with patch.dict(os.environ,env,clear=True),patch.object(c.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=fields.replace('QOS=soujanya-poria-startfund-2026-03','QOS=rose'))):
            with self.assertRaises(PermissionError):c._allocation()
    def test_requeue_or_missing_scheduler_cannot_execute(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaises(PermissionError):c._allocation()
        env={'SLURM_JOB_ID':'123','SLURM_NTASKS':'1','SLURM_NNODES':'1','SLURM_RESTART_COUNT':'1'}
        with patch.dict(os.environ,env,clear=True),self.assertRaises(PermissionError):c._allocation()
    def test_release_source_or_hash_failure_precedes_models(self):
        p=self.root/'release.json';t.write_once(p,{})
        with self.assertRaises(PermissionError):c.verify_release(p,'0'*64)
    def test_reload_rejects_reused_process_before_model_or_store(self):
        base=c._paths(11031,'T0')/'baseline';base.mkdir(parents=True)
        process={'host':'CPU','boot_id':'CPU','pid':101,'start_ticks':1}
        result=dict(status=c.BASELINE_STATUS,release={'CPU':'fake'},process=process)
        t.write_once(base/'RESULT.json',result)
        ctx=dict(start=11031,arm='T0',release_ref={'CPU':'fake'},baseline_ref=c._ref(base/'RESULT.json'))
        with patch.object(c.canary_replay,'_process_identity',return_value=process),patch.object(c.tx,'TransactionStore',side_effect=AssertionError('store constructed')):
            with self.assertRaisesRegex(PermissionError,'another process'):c._reload(ctx)
    def test_prior_reload_process_cannot_start_another_worker(self):
        slot=c._paths(11031,'T0')/'reload';slot.mkdir(parents=True)
        process={'host':'CPU','boot_id':'CPU','pid':101,'start_ticks':1}
        t.write_once(slot/'RESULT.json',dict(process=process))
        with patch.object(c,'verify_release',return_value={}),patch.object(c,'_no_unknown'),\
             patch.object(c.canary_replay,'_process_identity',return_value=process),\
             patch.object(c,'_execution_sources',side_effect=AssertionError('continued past process gate')):
            with self.assertRaisesRegex(PermissionError,'fresh process'):
                c.preflight(self.root/'release','a'*64,11032,'T0','baseline')
    def test_execute_failure_leaves_durable_hold_and_occupied_slot(self):
        slot=c._paths(11031,'T0')/'baseline'
        ctx=dict(slot=slot,release_ref={'CPU':'fake'},release={'storage_budget':{'CPU':'fake'}},registry=None,source_lock=None,sources={})
        process={'host':'CPU','boot_id':'CPU','pid':101,'start_ticks':1}
        with patch.object(c,'_allocation',return_value={'CPU':'fake'}),patch.object(c,'preflight',return_value=ctx),\
             patch.object(c,'_pre_model_storage',return_value={'CPU':'fake'}),\
             patch.object(c.canary_replay,'_process_identity',return_value=process),\
             patch.object(c.training_runtime,'build_training_runtime',side_effect=RuntimeError('CPU constructor failed')):
            with self.assertRaisesRegex(RuntimeError,'constructor failed'):c.execute(self.root/'release','a'*64,11031,'T0','baseline')
            self.assertTrue((slot/'HOLD.json').exists());self.assertTrue((slot/'CLAIM.json').exists())
            with self.assertRaises(FileExistsError):c.execute(self.root/'release','a'*64,11031,'T0','baseline')
    def test_release_budget_initial_dev_and_risk_refs_cannot_be_omitted(self):
        with self.assertRaises((KeyError,PermissionError)):c._initial_dev({}, {})
    def test_wrong_risk_denominator_is_not_silently_accepted(self):
        original=PACKAGE.parents[1]/'artifacts/sr_satdose_20261112/intake/TRAIN_RISK_CLARIFICATION_20261114.json'
        value=t.read_json(original);path=self.root/'risk.json';t.write_once(path,value)
        c._risk(c._ref(path))
        value['per_window_conflict_response_count']=512;path.write_text(t.canonical(value).decode())
        with self.assertRaises(PermissionError):c._risk(c._ref(path))
    def test_mutated_pinned_baseline_rejected(self):
        p=self.root/'baseline.json';t.write_once(p,{});reference=c._ref(p);p.write_text('changed')
        with self.assertRaises(PermissionError):c._read(reference)


if __name__=='__main__':unittest.main()
