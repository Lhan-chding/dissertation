"""CPU fake sampler/teacher/builder + real496 Adam/COMMIT/state restore.

The process identity and GPU builder are explicitly mocked. These tests prove
control flow and bit-exact comparison, never actual process/GPU acceptance.
"""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import torch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import canary_replay as c
sys.path.pop(0)
spec=importlib.util.spec_from_file_location('_canary_step_fixture',Path(__file__).with_name('test_training_step.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
t=c.tx


@unittest.skipUnless((f.f.fixtures.NATIVE/'sr_f3_ce/training.py').exists(),'Pinned native CPU optimizer required')
class CanaryReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(f.TrainingStepTests.doClassCleanups);f.TrainingStepTests.setUpClass()

    def setUp(self):
        fixture=f.TrainingStepTests('test_real_transaction_adam_codec_and_idempotent_composition')
        fixture.setUp();self.addCleanup(fixture.doCleanups);self.fixture=fixture
        self.root=fixture.root;self.native=fixture.tiny.ops;self.handle=fixture.handle
        fixture.sources['canary_replay']=dict(path=str(Path(c.__file__).resolve()),sha256=t.file_sha(c.__file__))
        identity=copy.deepcopy(self.handle.path_identity);identity['execution_sources_sha256']=c.object_sha(fixture.sources)
        self.handle.path_identity=copy.deepcopy(identity);fixture.fixture.identity=identity
        genesis=fixture.fixture.genesis(identity)
        fixture.store=t.TransactionStore(self.root,fixture.fixture.freeze,fixture.fixture.freeze_sha,identity,
            native=self.native,genesis_reference=genesis)
        self.handle.genesis_reference=genesis;self.store=fixture.store
        value=t.read_json(t.bound(self.root,genesis))
        f.f.dump(self.handle.output/'FORK_LOAD.json',dict(identity=identity,fork_receipt=value['fork_receipt'],resume=False))
        self.first=fixture.run_one();self.second=fixture.run_one(2)
        self._teacher(self.handle)
        self.process=dict(host='CPU_FAKE_HOST',boot_id='CPU_FAKE_BOOT',pid=101,start_ticks=1001)
        patcher=patch.object(c,'_process_identity',side_effect=lambda:copy.deepcopy(self.process));patcher.start();self.addCleanup(patcher.stop)
        self.base=c._paths(self.root,self.store.start,self.store.arm);self.base.mkdir(parents=True)
        self.spec_path=self.base/'TECHNICAL_EXECUTION.json';self.sources=fixture.sources
        self.spec=dict(schema=c.SCHEMA,permission='READONLY_STEP2_CANARY_ONLY',identity=identity,
            design_freeze_sha256=self.store.freeze_sha,execution_sources=self.sources,
            baseline_audit=str(self.handle.output.relative_to(self.root)),restored_audit=str((self.base/'restored/audit').relative_to(self.root)),
            slots={phase:str((self.base/phase).relative_to(self.root)) for phase in ('baseline','restored')},
            commits={str(step):t.reference(self.root,self.store.path/f'step-{step:03d}/COMMIT.json') for step in (1,2)})
        f.f.dump(self.spec_path,self.spec);self.spec_sha=t.file_sha(self.spec_path)
        self.build_calls=[];self.teacher_calls=[];self.restored_handle=None;self.different_teacher=False;self.teacher_failure=False

    def _teacher(self,handle):
        r=handle.runtime;r._satdose_active=None;r._train_phase=None;r._satdose_output=handle.output
        if not hasattr(r,'_satdose_accounting'):r._satdose_accounting=[]
        def event(purpose):
            value=dict(kind='extra_forward_sequences',count=1,
                metadata=dict(execution_phase=r._train_phase,purpose=purpose))
            r._satdose_accounting.append(value)
            with (r._satdose_output/'accounting.jsonl').open('ab') as stream:stream.write(t.canonical(value)+b'\n')
        def reference(records,ref):
            event('training_reference')
            return [torch.tensor(records[0]['sampler_logprobs'],dtype=torch.float32)-.1]
        def policy(records,*,purpose,grad):
            self.assertFalse(grad);self.assertFalse(torch.is_grad_enabled())
            self.teacher_calls.append(records[0]['request_id']);event(purpose)
            if self.teacher_failure:raise RuntimeError('CPU injected teacher failure')
            value=next(r.model.parameters()).detach().reshape(1).repeat(len(records[0]['tokens']))
            if self.different_teacher:value=value+.001
            return [value]
        def unbind():r._satdose_active=None;r._train_phase=None
        r.batch_reference_forward=reference;r.batch_sequence_forward=policy;r.unbind_request=unbind

    def fake_builder(self,registry,lock,**kwargs):
        self.build_calls.append(copy.copy(kwargs))
        self.assertEqual(kwargs['resume_step'],2)
        self.assertEqual(kwargs['expected_commit_sha256'],self.second['commit_sha256'])
        self.assertIsInstance(kwargs['commit_store'],c._LockedReadOnlyStore)
        self.assertFalse(any(Path(kwargs['output_dir']).iterdir()))
        # Constructor is mocked; actual native state codec restoration is not.
        with patch.object(type(self.fixture.tiny.scheduler),'step',new=self.fixture.scheduler_class_step):
            r,opt,scheduler=self.fixture.tiny.make_runtime()
        r._vision_hook=SimpleNamespace(remove=Mock());r._satdose_accounting=[]
        fork=SimpleNamespace(reference=copy.deepcopy(self.fixture.tiny.reference),optimizer=opt,scheduler=scheduler)
        handle=c.runtime.TrainingHandle(r,fork,self.native,self.fixture.storage,registry,11031,'T0',
            kwargs['output_dir'],kwargs['expected_identity'],ready=False)
        genesis=t.read_json(t.bound(self.root,self.store.genesis_reference))
        f.f.dump(handle.output/'FORK_LOAD.json',dict(identity=handle.path_identity,fork_receipt=genesis['fork_receipt'],resume=True))
        handle.restore_committed(kwargs['commit_store'],2,kwargs['expected_commit_sha256'],kwargs['expected_identity'])
        self._teacher(handle);self.restored_handle=handle
        return handle

    def baseline(self,**changes):
        options=dict(spec_path=self.spec_path,expected_spec_sha256=self.spec_sha,execution_sources=self.sources)
        options.update(changes);return c.baseline(self.handle,self.store,**options)

    def restored(self):
        with patch.object(c.runtime,'build_training_runtime',side_effect=self.fake_builder):
            return c.restored(self.store,self.fixture.registry,self.root/'source_lock.json',
                spec_path=self.spec_path,expected_spec_sha256=self.spec_sha,execution_sources=self.sources)

    def test_fixed128_baseline_and_new_process_real_state_restore_bitexact(self):
        before=self.native.state_hash(self.native.capture_rng());self.teacher_calls=[]
        baseline=self.baseline()
        self.assertEqual(baseline['status'],c.BASELINE)
        self.assertEqual(self.teacher_calls,[row['request_id'] for row in self.store.plans[2]])
        self.assertEqual(self.native.state_hash(self.native.capture_rng()),before)
        original=t.rows(t.bound(self.root,baseline['raw']));original_bytes={r['native_capture_path']:Path(r['native_capture_path']).read_bytes() for r in original}
        self.process['pid']=102;self.process['start_ticks']=1002;self.teacher_calls=[]
        restored=self.restored()
        self.assertEqual(restored['status'],c.RESTORED);self.assertEqual(restored['permission'],'NO_STEP3_OR_AGGREGATE_ADMISSION')
        self.assertEqual(len(self.build_calls),1);self.assertTrue(self.restored_handle._closed)
        self.assertEqual(self.teacher_calls,[row['request_id'] for row in self.store.plans[2]])
        self.assertEqual(restored['measured_counts'],{'training_policy':128,'training_reference':128})
        self.assertEqual((len(self.fixture.generated),self.fixture.adam_calls,self.fixture.scheduler_calls),(256,2,2))
        self.assertEqual(self.native.state_hash(self.native.capture_rng()),before)
        tensors=torch.load(t.bound(self.root,restored['tensors']),weights_only=True)
        self.assertEqual(len(tensors),128);self.assertTrue(all(torch.is_tensor(r['teacher_logprobs']) for r in tensors))
        relocation=t.read_json(t.bound(self.root,restored['relocation']))['rows'];self.assertEqual(len(relocation),256)
        for item in relocation:
            self.assertEqual(item['source']['sha256'],item['destination']['sha256'])
            self.assertNotEqual(item['source']['path'],item['destination']['path'])
        for path,content in original_bytes.items():self.assertEqual(Path(path).read_bytes(),content)

    def test_same_process_or_repeated_baseline_cannot_launch_reload(self):
        self.baseline()
        with self.assertRaisesRegex(PermissionError,'different process'):self.restored()
        self.assertFalse(self.build_calls);self.assertFalse((self.base/'restored').exists())
        with self.assertRaisesRegex(PermissionError,'already occupied'):self.baseline()
        self.assertEqual(len(self.teacher_calls),128)

    def test_unknown_slot_and_bad_spec_block_before_teacher_or_builder(self):
        (self.base/'baseline').mkdir()
        with self.assertRaisesRegex(PermissionError,'UNKNOWN'):self.baseline()
        self.assertFalse(self.teacher_calls);self.assertFalse(self.build_calls)
        with self.assertRaisesRegex(PermissionError,'technical spec'):self.baseline(expected_spec_sha256='0'*64)

    def test_teacher_failure_leaves_hold_and_never_retries(self):
        self.teacher_failure=True
        with self.assertRaisesRegex(RuntimeError,'teacher failure'):self.baseline()
        self.assertTrue((self.base/'baseline/HOLD.json').exists());self.assertTrue(self.handle._closed)
        self.assertEqual(len(self.teacher_calls),1)
        with self.assertRaises(PermissionError):self.baseline()
        self.assertEqual(len(self.teacher_calls),1)
        self.assertEqual(self.fixture.adam_calls,2)

    def test_changed_teacher_vector_rejects_reload_and_discards(self):
        self.baseline();self.process['pid']=102;self.process['start_ticks']=1002;self.different_teacher=True
        with self.assertRaisesRegex(PermissionError,'not bit-exact'):self.restored()
        self.assertTrue((self.base/'restored/HOLD.json').exists());self.assertTrue(self.restored_handle._closed)
        with self.assertRaisesRegex(PermissionError,'UNKNOWN'):self.restored()
        self.assertEqual(len(self.build_calls),1)

    def test_private_lease_only_inside_real_lock_and_no_updates(self):
        with c._locked_chain(self.store) as (lease,chain):
            self.assertEqual(lease.inspect_committed(2,chain[1][0])['step'],2)
            marker=lease.inspect_committed(2,chain[1][0]);marker['step']=7
            self.assertEqual(lease.inspect_committed(2,chain[1][0])['step'],2)
            with self.assertRaisesRegex(PermissionError,'cannot update'):lease.commit_step(3)
            with self.assertRaisesRegex(PermissionError,'holds the lock'):self.store.inspect_committed(2,chain[1][0])
            self.process['start_ticks']+=1
            with self.assertRaisesRegex(PermissionError,'foreign'):lease.inspect_committed(2,chain[1][0])
            self.process['start_ticks']-=1
        with self.assertRaisesRegex(PermissionError,'Expired'):lease.inspect_committed(2,chain[1][0])

    def test_source_or_state_mutation_blocks_baseline(self):
        changed=copy.deepcopy(self.sources);changed['canary_replay']['sha256']='f'*64
        with self.assertRaisesRegex(PermissionError,'spec identity/sources'):self.baseline(execution_sources=changed)
        with torch.no_grad():next(self.handle.runtime.model.parameters()).add_(.01)
        with self.assertRaisesRegex(PermissionError,'Dirty/uncommitted'):self.baseline()
        self.assertTrue((self.base/'baseline/HOLD.json').exists());self.assertFalse(self.teacher_calls)

    def test_original_capture_changed_during_phase_prevents_complete(self):
        original=self.handle.canary_teacher;calls=[]
        def changed(row,*args,**kwargs):
            result=original(row,*args,**kwargs);calls.append(row['request_id'])
            if len(calls)==128:
                path=Path(row['native_capture_path']);path.write_bytes(path.read_bytes()+b'\n')
            return result
        with patch.object(self.handle,'canary_teacher',side_effect=changed):
            with self.assertRaisesRegex(PermissionError,'artifact SHA mismatch'):self.baseline()
        self.assertEqual(len(calls),128);self.assertTrue((self.base/'baseline/HOLD.json').exists())
        self.assertFalse((self.base/'baseline/COMPLETE.json').exists());self.assertTrue(self.handle._closed)

    def test_baseline_requires_actual_nonresume_fork_evidence(self):
        path=self.handle.output/'FORK_LOAD.json';value=t.read_json(path);value['resume']=True;f.f.dump(path,value)
        with self.assertRaisesRegex(PermissionError,'loader/fresh R24'):self.baseline()
        self.assertFalse(self.teacher_calls)


if __name__=='__main__':unittest.main()
