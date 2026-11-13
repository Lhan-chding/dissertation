"""Explicit CPU fake admission/sampler/forward; real native Adam, codec and txn.

No model forward, GPU gate, quota measurement or production authority is tested.
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
from execution import training_step as s
sys.path.pop(0)
spec=importlib.util.spec_from_file_location('_step_transaction_fixture',Path(__file__).with_name('test_training_transaction.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
t=s.tx


class FakeAdmissionForCPUOnly(s.AuthenticatedAdmission):
    """Test boundary, NOT a GPU/DRY verifier or deployable admission authority."""
    def __init__(self,refs):
        self.refs=refs;self.calls=0;self.foreign=False
        super().__init__(dict(path=str(Path(__file__).resolve()),sha256=t.file_sha(__file__)))

    def verify(self,*,root,identity,design_freeze_sha256,target_step,sampled_endpoint):
        self.calls+=1
        sampled_endpoint=copy.deepcopy(sampled_endpoint)
        if self.foreign:sampled_endpoint['arm']='T1'
        return dict(identity_sha256=s.object_sha(identity),design_freeze_sha256=design_freeze_sha256,
                    target_step=target_step,sampled_endpoint=copy.deepcopy(sampled_endpoint),**copy.deepcopy(self.refs))


@unittest.skipUnless((f.fixtures.NATIVE/'sr_f3_ce/training.py').exists(),'Pinned native CPU optimizer sources required')
class TrainingStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(f.TrainingTransactionTests.doClassCleanups)
        f.TrainingTransactionTests.setUpClass()

    def setUp(self):
        fixture=f.TrainingTransactionTests('test_real_adam_two_steps_resume_and_commit_idempotency')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.tiny=fixture.fixture;self.root=fixture.root
        self.authority=FakeAdmissionForCPUOnly({})
        self.sources={key:dict(path=str(Path(path).resolve()),sha256=t.file_sha(path)) for key,path in (
            ('training_step',s.__file__),('training_runtime',s.training_runtime.__file__),
            ('training_state',s.training_state.__file__),('training_transaction',t.__file__),
            ('production',s.production.__file__),('calibration',s.calibration.__file__),
            ('satdose_core',s.calibration.core.__file__),('scorer_bindings',s.calibration.PACKAGE/'config/source_bindings.json'))}
        self.sources['admission_verifier']=self.authority.verifier_reference.copy()
        fixture.identity['execution_sources_sha256']=s.object_sha(self.sources)
        frozen=t.read_json(fixture.freeze)
        runtime_dir=Path(s.training_runtime.frozen.__file__).parent
        frozen['runtime_sources']={p.name:t.file_sha(p) for p in runtime_dir.glob('*.py')}
        f.dump(fixture.freeze,frozen);fixture.freeze_sha=t.file_sha(fixture.freeze)
        fixture.genesis_reference=fixture.genesis(fixture.identity)
        self.store=t.TransactionStore(self.root,fixture.freeze,fixture.freeze_sha,fixture.identity,
            native=self.tiny.ops,genesis_reference=fixture.genesis_reference)
        self.output=self.root/'training_audit';self.output.mkdir()
        self.requests={row['request_id']:dict(row,scope='TRAIN',case_id='CPU_CASE',input_id='CPU_INPUT',
            prompt='public only',image_file='fake.png',image_sha256='e'*64,assistant_prefill='{')
            for row in t.rows(self.root/'train_manifest.jsonl')}
        self.registry=SimpleNamespace(root=self.root,freeze_path=fixture.freeze,freeze_sha256=fixture.freeze_sha,
            config=t.read_json(self.root/'config.json'),request=lambda rid:copy.deepcopy(self.requests[rid]))
        runtime=self.tiny.runtime;runtime._satdose_accounting=[];runtime._satdose_output=self.output
        runtime._vision_hook=SimpleNamespace(remove=Mock())
        fork=SimpleNamespace(reference=self.tiny.reference,optimizer=self.tiny.optimizer,scheduler=self.tiny.scheduler)
        self.storage=SimpleNamespace(validate_snapshot_stats=Mock())
        self.handle=s.training_runtime.TrainingHandle(runtime,fork,self.tiny.ops,self.storage,self.registry,
            11031,'T0',self.output,fixture.identity)
        self.handle.genesis_reference=fixture.genesis_reference
        common=dict(experiment_id=t.EXPERIMENT,config_sha256=frozen['config_sha256'])
        f.dump(self.root/'CPU_REGISTRATION.json',dict(common,schema='sr-satdose-registration-v1',mode='production',
            artifacts={'design_freeze':dict(path='DESIGN_FREEZE.json',sha256=fixture.freeze_sha)},CPU_FAKE=True))
        reg=t.reference(self.root,self.root/'CPU_REGISTRATION.json')
        f.dump(self.root/'CPU_GPU.json',dict(common,schema='sr-satdose-gpu-verification-v1',mode='production',
            registration_sha256=reg['sha256'],CPU_FAKE=True))
        gpu=t.reference(self.root,self.root/'CPU_GPU.json')
        f.dump(self.root/'CPU_ADMISSION.json',dict(common,schema='sr-satdose-admission-v1',
            status='ADMITTED_FIRST_TWO_STEPS_ONLY',permission='FOUR_PATHS_MAX_STEP_2',failures=[],
            registration_sha256=reg['sha256'],gpu_report_sha256=gpu['sha256'],CPU_FAKE=True))
        self.authority.refs=dict(admission=t.reference(self.root,self.root/'CPU_ADMISSION.json'),registration=reg,gpu_report=gpu)
        self.generated=[];self.native_bodies=[];self.captures=[];self.adam_calls=0;self.scheduler_calls=0
        self.fail_runner=False;self.fail_scheduler=False
        self.expected_runtime,self.expected_optimizer,self.expected_scheduler=self.tiny.make_runtime()
        self.scheduler_class_step=type(self.tiny.scheduler).step
        self.kernels=SimpleNamespace(check_numerics=Mock(),sources={'CPU_FAKE':'not_native_forward'})
        self.deps=dict(runner=self.fake_runner,scorer=self.fake_scorer,kernels=self.kernels,
            cases={'CPU_CASE':dict(task=dict(world={},query={}),audit={})},
            groups=t.rows(self.root/'training_groups.jsonl'))
        for patcher in (patch.object(s,'_dependencies',return_value=self.deps),
                        patch.object(s.production,'capture_update',side_effect=self.fake_capture),
                        patch.object(s.calibration.core,'geometry',return_value={'authorised':{'A'}})):
            patcher.start();self.addCleanup(patcher.stop)
        original_adam=self.tiny.optimizer.step;original_scheduler=self.tiny.scheduler.step
        def adam():
            self.adam_calls+=1
            self.assertTrue((self.store.path/f'.staging-step-{self.handle._step+1:03d}'/'UPDATE_INTENT.json').is_file())
            return original_adam()
        def scheduler(instance):
            self.scheduler_calls+=1
            if self.fail_scheduler:raise RuntimeError('CPU injected after Adam')
            return original_scheduler()
        for patcher in (patch.object(self.tiny.optimizer,'step',side_effect=adam),
                        patch.object(type(self.tiny.scheduler),'step',new=scheduler)):
            patcher.start();self.addCleanup(patcher.stop)

    def event(self,kind,count,**metadata):
        value=dict(kind=kind,count=count,metadata=dict(execution_phase='TRAIN',**metadata))
        self.handle.runtime._satdose_accounting.append(value)
        with (self.output/'accounting.jsonl').open('ab') as stream:stream.write(t.canonical(value)+b'\n')

    def fake_runner(self,request,root,capture):
        self.generated.append(request['request_id'])
        self.event('completion_attempts',1)
        if self.fail_runner:raise RuntimeError('CPU injected missing completion')
        self.event('generated_tokens',2)
        body=dict(seed=request['seed'],tokens=[3,4],raw_tokens=[3,4],completion_token_count=2,
            sampler_logprobs=[-.3,-.4],sampler_logprob_source='actual_generate_raw_logits_selected_token',
            assistant_prefill='{',full_text='{"CPU":"synthetic"}',generated_text='"CPU":"synthetic"}',
            generation_status='COMPLETE',technical_validation_errors=[],finish_reason='balanced',
            prompt=request['prompt'],model_input={k:request[k] for k in s.PUBLIC},prompt_tensor_hash='d'*64)
        capture(body);self.native_bodies.append(copy.deepcopy(body))

    def fake_scorer(self,*args,**kwargs):
        self.assertEqual(kwargs,dict(finished=True,finish_reason='balanced'))
        return dict(J=1,S=1,known_mask={'S':1},witness={'selected':['A']})

    def fake_capture(self,model,rows,forward,reference,*,kernels,config,identity,plan):
        self.captures.append(copy.deepcopy(dict(rows=rows,identity=identity,plan=plan)))
        self.assertEqual(len(rows),128);self.assertEqual(len(plan['groups']),16)
        self.assertEqual(identity,dict(self.handle.endpoint(),parent='b0'))
        self.assertEqual([r['request_id'] for r in rows],[r['request_id'] for r in self.store.plans[plan['step']]])
        for row in rows:
            self.assertEqual(row['scope'],'TRAIN');self.assertEqual(row['step'],identity['step']+1)
            self.assertEqual(row['J'],row['native_score']['J'])
            self.event('extra_forward_sequences',1,purpose='training_reference')
            self.event('extra_forward_sequences',1,purpose='training_policy')
            self.handle.storage_forwards.append(dict(row_id=row['request_id'],forward_returned=True,stats={'CPU_FAKE':True}))
        return dict(status='NUMERICAL_CAPTURE_NOT_UPDATE_AUTHORIZATION',sequence_denominator=128,
            row_forward_callback_calls=128,backward_calls=128,optimizer_created=False,optimizer_steps=0,scheduler_steps=0,
            parameter_updates=0,source_sha256=self.kernels.sources,policy_sha256=identity['policy_sha256'],
            reference_sha256=identity['reference_sha256'],
            numerics={'CPU_FAKE':True},postclip_gradients={n:torch.full_like(p,.01) for n,p in model.named_parameters()})

    def storage_ready(self,stage,step):
        path=self.output/'steps'/f'step-{step:03d}'/'CPU_QUOTA.json';f.dump(path,dict(CPU_FAKE=True))
        return dict(schema='sr-satdose-storage-readiness-v1',ready=True,quota_verified=True,writable_verified=True,
            required_bytes=100,effective_available_bytes=200,evidence=[t.reference(self.root,path)])

    def run_one(self,step=1,**kwargs):
        options=dict(step=step,admission=self.authority,storage_ready=self.storage_ready,execution_sources=self.sources)
        options.update(kwargs)
        return s.run_step(self.handle,self.store,**options)

    def test_real_transaction_adam_codec_and_idempotent_composition(self):
        # A second clip would change an already verified production gradient.
        with patch.object(torch.nn.utils,'clip_grad_norm_',side_effect=AssertionError('second clip forbidden')):
            result=self.run_one()
        self.assertEqual(result['status'],'COMMITTED');self.assertEqual(self.handle.endpoint()['step'],1)
        for parameter in self.expected_runtime.model.parameters():parameter.grad=torch.full_like(parameter,.01)
        self.expected_optimizer.step();self.scheduler_class_step(self.expected_scheduler)
        for actual,expected in zip(self.tiny.runtime.model.parameters(),self.expected_runtime.model.parameters(),strict=True):
            self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(self.tiny.ops.state_hash(self.tiny.optimizer.state_dict()),
                         self.tiny.ops.state_hash(self.expected_optimizer.state_dict()))
        self.assertEqual(self.tiny.ops.state_hash(self.tiny.scheduler.state_dict()),
                         self.tiny.ops.state_hash(self.expected_scheduler.state_dict()))
        self.assertEqual((len(self.generated),len(self.captures),self.adam_calls,self.scheduler_calls),(128,1,1,1))
        self.assertTrue(all(float(v['step'])==1 for v in self.tiny.optimizer.state.values()))
        self.assertTrue(all(p.grad is None for p in self.tiny.runtime.model.parameters()))
        before=(len(self.generated),len(self.captures),self.adam_calls,self.scheduler_calls,self.authority.calls)
        duplicate=self.run_one()
        self.assertEqual(duplicate['status'],'ALREADY_COMMITTED')
        self.assertEqual(before,(len(self.generated),len(self.captures),self.adam_calls,self.scheduler_calls,self.authority.calls))
        marker=t.read_json(Path(result['checkpoint'])/'COMMIT.json')
        self.assertEqual(marker['sampler_state']['generated_responses'],128)
        self.assertEqual(marker['native_update']['path'],'training_audit/steps/step-001/NATIVE_UPDATE.json')
        self.assertEqual(result['permission'],'NO_TRAINING_ADMISSION')

    def test_next_step_uses_advanced_endpoint_and_exact_next_frozen128(self):
        first=self.run_one();first_endpoint=self.handle.endpoint()
        second=self.run_one(2)
        self.assertEqual(second['status'],'COMMITTED');self.assertEqual(self.handle.endpoint()['step'],2)
        self.assertEqual((len(self.generated),len(self.captures),self.adam_calls,self.scheduler_calls),(256,2,2,2))
        self.assertEqual(self.captures[1]['identity'],dict(first_endpoint,parent='b0'))
        self.assertEqual([r['request_id'] for r in self.captures[1]['rows']],
                         [r['request_id'] for r in self.store.plans[2]])
        marker=t.read_json(Path(second['checkpoint'])/'COMMIT.json')
        self.assertEqual(marker['transaction']['previous_commit_sha256'],first['commit_sha256'])
        self.assertEqual(marker['sampler_state']['generated_responses'],256)
        self.assertTrue(all(float(v['step'])==2 for v in self.tiny.optimizer.state.values()))
        self.assertEqual(self.run_one(1)['status'],'ALREADY_COMMITTED')
        self.assertEqual(self.adam_calls,2)

    def test_raw_metadata_is_new_dict_preserves_all_native_complete_bytes(self):
        requests=copy.deepcopy(self.requests)
        result=self.run_one();self.assertEqual(self.requests,requests)
        raw=t.rows(self.output/'steps/step-001/RAW.jsonl')
        for row,body in zip(raw,self.native_bodies,strict=True):
            capture=Path(row['native_capture_path'])
            self.assertEqual(t.read_json(capture),body)
            complete=t.read_json(capture.with_name(capture.name.replace('.CAPTURE.json','.COMPLETE.json')))
            self.assertNotIn('native_score',complete);self.assertNotIn('scope',complete)
            self.assertTrue(all(row[k]==v for k,v in complete.items()))
            self.assertEqual(row['step'],1);self.assertEqual(row['arm'],'T0');self.assertEqual(row['exact_authorised'],1)
        self.store.inspect_committed(1,result['commit_sha256'])

    def test_unknown_after_adam_never_retries_or_advances(self):
        self.fail_scheduler=True
        with self.assertRaisesRegex(RuntimeError,'after Adam'):self.run_one()
        self.assertEqual((self.adam_calls,self.scheduler_calls),(1,1));self.assertEqual(self.handle._step,0)
        self.assertTrue(self.handle._closed)
        self.assertTrue((self.store.path/'.staging-step-001/UPDATE_INTENT.json').is_file())
        self.assertFalse((self.store.path/'step-001/COMMIT.json').exists())
        with self.assertRaises(PermissionError):self.run_one()
        self.assertEqual((len(self.generated),self.adam_calls),(128,1))
        with self.assertRaisesRegex(PermissionError,'occupied staging'):
            self.store.commit_step(1,collect_evidence=Mock(),prepare_update=Mock(),optimizer_step=Mock(),
                scheduler_step=Mock(),write_checkpoint=Mock(),storage_ready=Mock())

    def test_failed_native_generation_persists_attempt_and_discard(self):
        self.fail_runner=True
        with self.assertRaisesRegex(RuntimeError,'missing completion'):self.run_one()
        self.assertEqual((len(self.generated),self.adam_calls,self.scheduler_calls),(1,0,0))
        self.assertEqual(len(list((self.output/'steps/step-001').glob('*.BEGIN.json'))),1)
        self.assertTrue(self.handle._closed)
        with self.assertRaises(PermissionError):self.run_one()
        self.assertEqual(len(self.generated),1)

    def test_publish_then_advance_failure_preserves_verified_commit(self):
        with patch.object(self.handle,'advance_committed',side_effect=RuntimeError('CPU advance failure')):
            with self.assertRaisesRegex(RuntimeError,'advance failure'):self.run_one()
        directory=self.output/'steps/step-001'
        self.assertFalse((directory/'HOLD.json').exists())
        self.assertTrue((directory/'POST_PUBLICATION_FAILURE.json').is_file());self.assertTrue(self.handle._closed)
        self.assertEqual(t.read_json(directory/'POST_PUBLICATION_FAILURE.json')['status'],
                         'COMMIT_PRESENT_HANDLE_DISCARDED_REQUIRES_VALIDATION')
        commit=self.store.path/'step-001/COMMIT.json'
        self.store.inspect_committed(1,t.file_sha(commit))
        self.assertEqual(self.run_one()['status'],'ALREADY_COMMITTED')
        self.assertEqual((len(self.generated),self.adam_calls,self.scheduler_calls),(128,1,1))

    def test_plain_gates_unpinned_authority_and_future_step_are_inaccessible(self):
        for changes in ({'admission':{'status':'ADMITTED_FIRST_TWO_STEPS_ONLY'}},{'step':2},
                        {'execution_sources':dict(self.sources,admission_verifier={'path':'x','sha256':'f'*64})}):
            with self.subTest(changes=changes),self.assertRaises(PermissionError):self.run_one(**changes)
        self.assertFalse(self.generated);self.assertEqual(self.adam_calls,0)
        with self.assertRaises(TypeError):s.AuthenticatedAdmission(self.authority.verifier_reference)

    def test_cross_arm_authority_scope_and_dirty_policy_deny_before_generation(self):
        self.authority.foreign=True
        with self.assertRaisesRegex(PermissionError,'scope differs'):self.run_one()
        self.authority.foreign=False
        with torch.no_grad():next(self.tiny.runtime.model.parameters()).add_(.1)
        with self.assertRaisesRegex(PermissionError,'Dirty/uncommitted'):self.run_one()
        self.assertFalse(self.generated);self.assertEqual(self.adam_calls,0)

    def test_changed_audited_source_or_receipt_refuses_before_generation(self):
        changed=copy.deepcopy(self.sources);changed['production']['sha256']='0'*64
        with self.assertRaisesRegex(PermissionError,'implementation not bound'):self.run_one(execution_sources=changed)
        gpu=self.root/'CPU_GPU.json';gpu.write_bytes(gpu.read_bytes()+b'\n')
        with self.assertRaises(PermissionError):self.run_one()
        self.assertFalse(self.generated)

    def test_accounting_mismatch_and_live_quota_failure_prevent_adam(self):
        def no_space(stage,step):
            value=self.storage_ready(stage,step);value['effective_available_bytes']=0;return value
        with self.assertRaisesRegex(PermissionError,'actual quota'):self.run_one(storage_ready=no_space)
        self.assertEqual(self.adam_calls,0)
        self.assertFalse((self.store.path/'.staging-step-001/UPDATE_INTENT.json').exists())
        self.assertTrue(self.handle._closed)

    def assert_history_rejected(self,commit_sha):
        callbacks={key:Mock(side_effect=AssertionError('historical audit ran callback')) for key in
            ('collect_evidence','prepare_update','optimizer_step','scheduler_step','write_checkpoint','storage_ready')}
        for step in (1,2):
            with self.assertRaises((PermissionError,FileNotFoundError)):
                self.store.commit_step(step,**callbacks)
        with self.assertRaises((PermissionError,FileNotFoundError)):
            self.handle.advance_committed(self.store,1,commit_sha)
        for callback in callbacks.values():callback.assert_not_called()

    def test_committed_native_update_capture_begin_deleted_or_tampered_block_every_audit(self):
        result=self.run_one();directory=self.output/'steps/step-001'
        row=t.rows(directory/'RAW.jsonl')[0];capture=Path(row['native_capture_path'])
        paths=(directory/'NATIVE_UPDATE.json',capture,capture.with_name(capture.name.replace('.CAPTURE.json','.BEGIN.json')))
        for path in paths:
            original=path.read_bytes()
            for action in ('delete','tamper'):
                with self.subTest(path=path.name,action=action):
                    if action=='delete':path.unlink()
                    else:path.write_bytes(original+b'\n')
                    try:self.assert_history_rejected(result['commit_sha256'])
                    finally:path.write_bytes(original)
        self.assertEqual((len(self.generated),self.adam_calls,self.scheduler_calls),(128,1,1))
        self.assertEqual(self.run_one()['status'],'ALREADY_COMMITTED')
        (directory/'NATIVE_UPDATE.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_one()
        self.assertEqual((len(self.generated),self.adam_calls),(128,1))

    def test_native_update_missing_or_semantically_rebound_never_passes_history(self):
        result=self.run_one();commit_path=Path(result['checkpoint'])/'COMMIT.json'
        update_path=self.output/'steps/step-001/NATIVE_UPDATE.json'
        commit_bytes=commit_path.read_bytes();update_bytes=update_path.read_bytes()
        marker=t.read_json(commit_path);update=t.read_json(update_path)
        fields={'schema':'foreign','identity':dict(self.handle.path_identity,arm='T1'),'step':2,
            'policy_before_sha256':'0'*64,'policy_after_sha256':'0'*64,'reference_sha256':'0'*64,
            'optimizer_calls':True,'scheduler_calls':2,'parameter_update_norm':-1,
            'production_capture':t.reference(self.root,self.root/'CPU_GPU.json')}
        for key,value in fields.items():
            with self.subTest(field=key):
                f.dump(update_path,dict(update,**{key:value}))
                changed=copy.deepcopy(marker);changed['native_update']=t.reference(self.root,update_path)
                f.dump(commit_path,changed)
                try:self.assert_history_rejected(t.file_sha(commit_path))
                finally:commit_path.write_bytes(commit_bytes);update_path.write_bytes(update_bytes)
        changed=copy.deepcopy(marker);del changed['native_update'];f.dump(commit_path,changed)
        try:self.assert_history_rejected(t.file_sha(commit_path))
        finally:commit_path.write_bytes(commit_bytes)
        self.assertEqual((len(self.generated),self.adam_calls),(128,1))

    def test_override_source_check_cannot_authorize_foreign_verify_implementation(self):
        class BypassAttempt(FakeAdmissionForCPUOnly):
            def check_source(self):pass
        # Constructor always invokes the base source audit, even when overridden.
        with self.assertRaisesRegex(PermissionError,'Unauthenticated admission verifier source'):
            s.AuthenticatedAdmission.__init__(BypassAttempt.__new__(BypassAttempt),
                dict(path=str(Path(s.__file__).resolve()),sha256=t.file_sha(s.__file__)))
        authority=BypassAttempt(self.authority.refs)
        authority.verifier_reference=dict(path=str(Path(s.__file__).resolve()),sha256=t.file_sha(s.__file__))
        with self.assertRaisesRegex(PermissionError,'Unauthenticated admission verifier source'):
            self.run_one(admission=authority)
        with self.assertRaisesRegex(PermissionError,'Unauthenticated admission verifier source'):
            s._authorization(self.handle,authority,1)
        self.assertFalse(self.generated)

    def test_recorded_accounting_cannot_be_replaced_by_expected_count(self):
        self.event('extra_forward_sequences',128,purpose='training_policy')
        self.event('extra_forward_sequences',128,purpose='training_reference')
        self.assertEqual(s._accounting(self.handle,0,128)['measured_counts']['training_policy'],128)
        (self.output/'accounting.jsonl').write_bytes(b'')
        with self.assertRaisesRegex(PermissionError,'Persisted native accounting'):s._accounting(self.handle,0,128)


if __name__=='__main__':unittest.main()
