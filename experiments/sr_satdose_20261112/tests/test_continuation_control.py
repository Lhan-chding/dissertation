"""CPU controller tests with explicit fake GPU/admission/quota boundaries.

The integration subset uses actual native 496-tensor Adam, scheduler, safe state
codec and immutable transactions. Synthetic sampler/forward/risk observations do
not certify production admission, GPU correctness or independent process exit.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import continuation_control as c
sys.path.pop(0)
t=c.tx
spec=importlib.util.spec_from_file_location('_continuation_step_fixture',Path(__file__).with_name('test_training_step.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(t.canonical(value))


def decision(step=2,hold=False):
    pending=[f'11031/T0/{step}'] if step in (8,16) else []
    return dict(status=('GLOBAL_SCIENTIFIC_HOLD' if hold else 'PENDING_REQUIRED_DEV' if pending else
                        'NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED'),global_hold=hold,pending_required_dev=pending)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name).resolve()
        for p in (patch.object(c,'ROOT',self.root),patch.object(c.first_two,'ROOT',self.root),patch.object(c.storage,'ROOT',self.root)):
            p.start();self.addCleanup(p.stop)
        self.trusted={};self.endpoints={};self.identity=dict(CPU_FAKE=True)
        for key in c.PATHS:
            start,arm=key.split('/');refs=[]
            for step in (1,2):
                p=self.root/f'training_transactions/start-{start}/{arm}/step-{step:03d}/COMMIT.json'
                dump(p,dict(identity=self.identity,step=step));refs.append(c.ref(p))
            self.trusted[key]=refs
            for phase,status in (('baseline',c.first_two.BASELINE_STATUS),('reload',c.first_two.RELOAD_STATUS)):
                dump(self.root/f'first_two_control/start-{start}/{arm}/{phase}/RESULT.json',
                     dict(status=status,permission='NO_STEP3_OR_CANARY_ADMISSION',process={'CPU':phase}))
            directory=self.root/f'trained_dev/start-{start}/{arm}/step-002'
            self.add_dev(key+'/2',directory,dict(commit=refs[1]))
        for start in (11031,11032):
            self.add_dev(f'{start}/START/0',self.root/f'start_dev/start-{start}',None)
        dump(self.root/'first_two_launcher/RESULT.json',dict(status='FOUR_FIRST_TWO_PATHS_EXIT_OBSERVED_NOT_CANARY_ADMISSION'))
    def add_dev(self,key,directory,checkpoint):
        seed,arm,step=key.split('/')
        binding=dict(scope='DEV',endpoint=dict(start_seed=int(seed),arm=arm,step=int(step)))
        if checkpoint:binding['checkpoint']=checkpoint
        dump(directory/'ENDPOINT_BINDING.json',binding);dump(directory/'endpoint/RECEIPT.json',{'CPU_FAKE':True})
        refs=dict(receipt=c.ref(directory/'endpoint/RECEIPT.json'),binding=c.ref(directory/'ENDPOINT_BINDING.json'))
        self.endpoints[key]=refs
        dump(directory/'RESULT.json',dict(status=('START_DEV_ENDPOINT_COMPLETE' if arm=='START' else 'TRAINED_DEV_ENDPOINT_COMPLETE'),
            responses=108,receipt=refs['receipt'],endpoint_binding=refs['binding'],process_identity={'CPU':key}))
    def test_only_fixed_chunks_and_paths(self):
        for first,last in ((3,8),(9,16),(17,24)):
            self.assertEqual(c._slot(11031,'T0',first).name,f'chunk-{first:03d}-{last:03d}')
        for args in ((11031,'MAIN',3),(11031,'T2',3),(11033,'T0',3),(11031,'T0',2),(11031,'T0',4),(11031,'T0',25),(True,'T0',3)):
            with self.subTest(args=args),self.assertRaises(PermissionError):c._slot(*args)
    def test_global_worker_lease_excludes_and_detects_replacement(self):
        with c.Lease() as lease:
            with self.assertRaises(BlockingIOError):
                with c.Lease():pass
            (self.root/'DIAGNOSTIC_WORKER.lock').rename(self.root/'OLD.lock')
            (self.root/'DIAGNOSTIC_WORKER.lock').touch()
            with self.assertRaisesRegex(PermissionError,'replaced'):lease.active()
        with c.Lease():pass
    def test_complete_exact_four_path_snapshot_and_prior_dev_barrier(self):
        c._cursor(self.trusted);c._barrier(self.trusted,self.endpoints,11031,'T0',3)
        c._completed_attempts(self.trusted,self.endpoints,{})
    def test_any_missing_prior_dev_blocks_all_paths(self):
        for key in self.endpoints:
            changed=copy.deepcopy(self.endpoints);changed.pop(key)
            with self.subTest(key=key),self.assertRaisesRegex(PermissionError,'barriers'):
                c._barrier(self.trusted,changed,11031,'T0',3)
    def test_dev_cannot_bind_foreign_checkpoint(self):
        key='11032/T1/2';p=c.bound(self.endpoints[key]['binding']);value=t.read_json(p)
        value['checkpoint']['commit']=self.trusted['11031/T0'][0];dump(p,value);self.endpoints[key]['binding']=c.ref(p)
        with self.assertRaisesRegex(PermissionError,'checkpoint'):c._barrier(self.trusted,self.endpoints,11031,'T0',3)
    def test_unknown_transaction_and_changed_commit_ref_rejected(self):
        p=self.root/'training_transactions/start-11031/T0/.staging-step-003';p.mkdir()
        with self.assertRaisesRegex(PermissionError,'cursor'):c._cursor(self.trusted)
        p.rmdir();dump(c.bound(self.trusted['11031/T0'][1]),{'tampered':True})
        with self.assertRaises(PermissionError):c._cursor(self.trusted)
    def test_other_path_midchunk_or_advanced_past_current_chunk_rejected(self):
        for length in (3,7,9,16,24):
            refs=copy.deepcopy(self.trusted);refs['11032/T0']=[{}]*length
            with self.subTest(length=length),self.assertRaisesRegex(PermissionError,'boundary'):
                c._barrier(refs,self.endpoints,11031,'T0',3)
    def test_occupied_unknown_chunk_or_dev_cannot_be_retried(self):
        slot=c._slot(11031,'T0',3);slot.mkdir(parents=True)
        with self.assertRaisesRegex(PermissionError,'Occupied'):c._completed_attempts(self.trusted,self.endpoints,{})
        slot.rmdir();(self.root/'trained_dev/start-11032/T0/step-008').mkdir()
        with self.assertRaises(FileNotFoundError):c._completed_attempts(self.trusted,self.endpoints,{})
    def test_global_holds_and_unknown_staging_across_controllers(self):
        for family in ('first_two_launcher','training_control','continuation_control','trained_dev','training_transactions'):
            path=self.root/family/'HOLD.json';dump(path,{})
            with self.subTest(family=family),self.assertRaisesRegex(PermissionError,'failed'):c._hold_free()
            path.unlink()
        p=self.root/'training_transactions/start-11031/T0/.staging-step-003';p.mkdir()
        with self.assertRaisesRegex(PermissionError,'failed'):c._hold_free()
    def test_unresolved_begin_and_unknown_original_worker_slot_rejected(self):
        p=self.root/'first_two_control/start-11031/T0/baseline/CPU.BEGIN.json';dump(p,{'CPU':True})
        with self.assertRaisesRegex(PermissionError,'Unresolved'):c._hold_free()
        p.unlink();(p.parent.parent/'unknown-phase').mkdir()
        with self.assertRaisesRegex(PermissionError,'phase'):c._completed_attempts(self.trusted,self.endpoints,{})
    def test_readonly_canary_relocation_pair_is_deferred_to_full_source_bound_audit(self):
        directory=self.root/'canary_replay/start-11031/T0/restored/audit/original_tokens'
        dump(directory/'CPU.BEGIN.json',{'CPU_ORIGINAL':True});dump(directory/'CPU.CAPTURE.json',{'CPU_ORIGINAL':True})
        c._hold_free()  # Production preflight still must pass ContinuingAdmission.
        (directory/'CPU.CAPTURE.json').unlink()
        with self.assertRaisesRegex(PermissionError,'Unresolved'):c._hold_free()
    def test_symlink_artifact_rejected(self):
        p=self.root/'trained_dev/link';p.symlink_to(self.root/'first_two_launcher/RESULT.json')
        with self.assertRaisesRegex(PermissionError,'Noncanonical'):c._hold_free()
    def test_fresh_process_rejects_prior_worker_or_dev_identity(self):
        with self.assertRaisesRegex(PermissionError,'fresh'):c._fresh({'CPU':'baseline'})
        with self.assertRaisesRegex(PermissionError,'fresh'):c._fresh({'CPU':'11031/T0/2'})
        c._fresh({'CPU':'genuinely-new-fixture'})
    def test_preflight_rejects_unknown_before_admission_or_model(self):
        c._slot(11031,'T0',3).mkdir(parents=True)
        with patch.object(c,'verify_release') as release,patch.object(c.training_runtime,'build_training_runtime') as model:
            with self.assertRaisesRegex(PermissionError,'Occupied'):c.preflight(self.root/'R.json','0'*64,11031,'T0',3)
            release.assert_not_called();model.assert_not_called()
    def test_later_chunks_require_all_prior_dev_barriers(self):
        for first,last in ((9,16),(17,24)):
            with self.subTest(first=first):
                trusted=copy.deepcopy(self.trusted);endpoints=copy.deepcopy(self.endpoints)
                for key in c.PATHS:
                    start,arm=key.split('/')
                    for step in range(3,first):
                        p=self.root/f'training_transactions/start-{start}/{arm}/step-{step:03d}/COMMIT.json'
                        dump(p,dict(identity=self.identity,step=step));trusted[key].append(c.ref(p))
                    for step in (8,16):
                        if step>=first:continue
                        directory=self.root/f'trained_dev/start-{start}/{arm}/step-{step:03d}'
                        self.add_dev(f'{key}/{step}',directory,dict(commit=trusted[key][step-1]))
                        endpoints[f'{key}/{step}']=self.endpoints[f'{key}/{step}']
                c._barrier(trusted,endpoints,11031,'T0',first)
                endpoints.pop(f'11032/T1/{first-1}')
                with self.assertRaisesRegex(PermissionError,'barriers'):c._barrier(trusted,endpoints,11031,'T0',first)
    def test_source_pinned_release_rejects_changed_identity_and_scope(self):
        modules={'continuation_control':c,'continuation_verifier':c.continuation_authority,'risk_verifier':c.risk}
        sources={name:dict(path=str(Path(module.__file__).resolve()),sha256=t.file_sha(module.__file__))
                 for name,module in modules.items()}
        inventory={str(Path(v['path']).relative_to(c.PACKAGE)):v['sha256'] for v in sources.values()}
        original=dict(sources=inventory,execution_sources=sources,admission_spec={},admission={},risk_clarification={})
        path=self.root/'CPU_ORIGINAL.json';dump(path,original)
        release=dict(schema='sr-satdose-continuation-release-v1',status='AUDITED_FOR_ONE_FIXED_TRAIN_CHUNK',
            permission='ONE_FIXED_CHUNK_NO_MAIN',experiment_id=c.EXPERIMENT,evidence_root=str(self.root),release_root=str(c.PACKAGE),
            design_freeze_sha256=c.FREEZE_SHA,config_sha256=c.CONFIG_SHA,original_train_release=c.ref(path),**original)
        path=self.root/'CPU_RELEASE.json'
        with patch.object(c.first_two,'verify_release',return_value=original):
            dump(path,release);self.assertEqual(c.verify_release(path,c.sha(path))[0],release)
            for field,value in (('permission','MAIN'),('design_freeze_sha256','0'*64),('admission',{'changed':True})):
                modified=copy.deepcopy(release);modified[field]=value;dump(path,modified)
                with self.subTest(field=field),self.assertRaises(PermissionError):c.verify_release(path,c.sha(path))
            changed=copy.deepcopy(original);changed['execution_sources']['continuation_control']['sha256']='0'*64
            dump(path,release)
            with patch.object(c.first_two,'verify_release',return_value=changed),self.assertRaisesRegex(PermissionError,'source pin'):
                c.verify_release(path,c.sha(path))
            with self.assertRaisesRegex(PermissionError,'Untrusted'):c.verify_release(path,'0'*64)
    def test_full_preflight_calls_bound_continuing_authority_before_any_model(self):
        # Explicit mocked heavy authority/native codec; real controller snapshot, DEV,
        # canonical references, selection and source-binding checks remain exercised.
        sources={'risk_verifier':{'sha256':'a'*64},'admission_verifier':{'CPU_FAKE':True},
                 'continuation_verifier':{'CPU_FAKE':True}}
        original=dict(execution_sources=sources)
        for key,refs in self.trusted.items():
            start,arm=key.split('/')
            for step in (1,2):
                p=c.bound(refs[step-1]);dump(p,dict(step=step,identity=dict(train_seed=int(start),arm=arm,
                    execution_sources_sha256=c.object_sha(sources)),policy_sha256='p',reference_sha256='r',
                    transaction=dict(genesis_reference={'CPU_FAKE':True})))
                refs[step-1]=c.ref(p)
            p=c.bound(self.endpoints[key+'/2']['binding']);v=t.read_json(p)
            v['checkpoint']['commit']=refs[1];dump(p,v);self.endpoints[key+'/2']['binding']=c.ref(p)
            result_path=p.parent/'RESULT.json';result=t.read_json(result_path)
            result['endpoint_binding']=c.ref(p);dump(result_path,result)
        dump(self.root/'CPU_SPEC.json',dict(root=str(self.root),sources={'authority':sources['admission_verifier']},
                                           training_execution_sources_sha256=c.object_sha(sources)))
        dump(self.root/'CPU_LOCK.json',dict(native_src='CPU_FAKE',native_files={}))
        dump(self.root/'CPU_CANARY.json',{'CPU_FAKE':True});dump(self.root/'CPU_RELEASE.json',{'CPU_FAKE':True})
        dump(self.root/'DESIGN_FREEZE.json',dict(artifacts={'source_lock':c.ref(self.root/'CPU_LOCK.json'),'config':{'CPU_FAKE':True}}))
        release=dict(selection=dict(start_seed=11031,arm='T0',first_step=3,last_step=8),trusted_commits=self.trusted,
            dev_endpoints=self.endpoints,prior_chunks={},admission_spec=c.ref(self.root/'CPU_SPEC.json'),admission={},
            risk_clarification={},canary_spec=c.ref(self.root/'CPU_CANARY.json'),storage_budget={})
        authority=SimpleNamespace(risk_snapshot=lambda:decision(),verify=Mock())
        marker=c.read(self.trusted['11031/T0'][-1]);store=SimpleNamespace(inspect_committed=Mock(return_value=marker))
        with patch.object(c,'verify_release',return_value=(release,original)),patch.object(c.canary_replay,'_process_identity',return_value={'CPU':'fresh'}),\
             patch.object(c.first_two,'_registry',return_value='CPU_REGISTRY'),patch.object(c.first_two,'_risk'),\
             patch.object(c.admission_authority,'ContinuingAdmission',return_value=authority) as constructor,\
             patch.object(c.fork_loader,'load_native_ops',return_value='CPU_NATIVE'),patch.object(c.tx,'TransactionStore',return_value=store),\
             patch.object(c.storage,'_budget',return_value={'scope':'CONTINUATION_FIXED_CHUNK_ONLY'}),\
             patch.object(c.training_runtime,'build_training_runtime') as builder:
            result=c.preflight(self.root/'CPU_RELEASE.json','0'*64,11031,'T0',3)
            constructor.assert_called_once();self.assertEqual(authority.verify.call_args.kwargs['target_step'],3)
            self.assertEqual(result['initial'],self.trusted['11031/T0'][-1]);store.inspect_committed.assert_called_once()
            builder.assert_not_called();self.assertFalse(result['slot'].exists())
    def test_live_storage_budget_expiration_or_capacity_failure_precedes_model(self):
        budget=dict(scope='CONTINUATION_FIXED_CHUNK_ONLY',required_bytes=100,max_observation_age_seconds=10)
        with patch.object(c.storage,'_budget',return_value=budget),patch.object(c.storage,'_observe',return_value={'effective_available_bytes':99}),\
             patch.object(c.storage,'_probe') as probe:
            with self.assertRaisesRegex(PermissionError,'quota'):c._pre_model_storage(self.root,{})
            probe.assert_not_called()
        with patch.object(c.storage,'_budget',side_effect=[budget,PermissionError('expired')]),\
             patch.object(c.storage,'_observe',return_value={'effective_available_bytes':200}),patch.object(c.storage,'_probe',return_value={}):
            with self.assertRaisesRegex(PermissionError,'expired'):c._pre_model_storage(self.root,{})
    def test_cli_has_no_arbitrary_end_step_or_main_authority(self):
        with patch.object(c,'preflight',return_value={}) as preflight:
            self.assertEqual(c.main(['--release-receipt',str(self.root/'R.json'),'--release-sha256','0'*64,
                                     '--start-seed','11031','--arm','T0','--first-step','3']),0)
            preflight.assert_called_once()


class ExecuteCompositionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name).resolve()
        p=patch.object(c,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        self.slot=c._slot(11031,'T0',3)
        self.ctx=dict(slot=self.slot,start=11031,arm='T0',first=3,last=8,process={'CPU':1},
            release_ref={'path':'CPU_RELEASE.json','sha256':'0'*64},initial={'path':'step2','sha256':'2'*64},identity={'CPU':'original'},
            release={'selection':dict(start_seed=11031,arm='T0',first_step=3,last_step=8),'storage_budget':{'CPU':True}},
            sources={'CPU':'original'},registry=object(),source_lock=self.root/'SOURCE_LOCK.json',store=object())
        self.handle=SimpleNamespace(discard=Mock());self.result=dict(status=c.STATUS,permission=c.PERMISSION)
        self.builders=[]
        def builder(*args,**kwargs):self.builders.append(kwargs);return self.handle
        def preflight(*args):
            c.require(not self.slot.exists(),'Occupied attempt no retry');return self.ctx
        for p in (patch.object(c,'preflight',side_effect=preflight),patch.object(c.first_two,'_allocation',return_value={'CPU':True}),
                  patch.object(c,'_pre_model_storage',return_value={'CPU_FAKE':True}),patch.object(c,'_guard'),
                  patch.object(c.training_runtime,'build_training_runtime',side_effect=builder),
                  patch.object(c,'_run_chunk',return_value=self.result)):
            mock=p.start();self.addCleanup(p.stop)
    def execute(self):return c.execute(self.root/'R.json','0'*64,11031,'T0',3)
    def test_builder_restores_exact_existing_checkpoint_identity_and_state(self):
        result=self.execute();self.assertEqual(result,self.result)
        call=self.builders[0];self.assertEqual(call['resume_step'],2)
        self.assertIs(call['commit_store'],self.ctx['store']);self.assertEqual(call['expected_commit_sha256'],'2'*64)
        self.assertEqual(call['expected_identity'],self.ctx['identity']);self.assertEqual(call['execution_sources'],self.ctx['sources'])
        self.handle.discard.assert_called_once();self.assertTrue((self.slot/'RESULT.json').exists())
        with self.assertRaisesRegex(PermissionError,'Occupied'):self.execute()
        self.assertEqual(len(self.builders),1)
    def test_model_failure_keeps_unknown_claim_and_global_hold_without_retry(self):
        with patch.object(c.training_runtime,'build_training_runtime',side_effect=RuntimeError('CPU injected restore failure')):
            with self.assertRaisesRegex(RuntimeError,'restore'):self.execute()
        self.assertTrue((self.slot/'CLAIM.json').exists());self.assertTrue((self.slot/'HOLD.json').exists())
        self.assertTrue((self.root/'training_control/HOLD.json').exists());self.assertFalse((self.slot/'RESULT.json').exists())
        with self.assertRaisesRegex(PermissionError,'Occupied'):self.execute()
    def test_postcommit_risk_error_retains_hold_and_disposes_runtime(self):
        with patch.object(c,'_run_chunk',side_effect=RuntimeError('CPU injected failed risk refresh')):
            with self.assertRaisesRegex(RuntimeError,'refresh'):self.execute()
        self.handle.discard.assert_called_once();self.assertFalse((self.slot/'RESULT.json').exists())
        self.assertEqual(t.read_json(self.root/'training_control/HOLD.json')['error_type'],'RuntimeError')


@unittest.skipUnless((f.f.fixtures.NATIVE/'sr_f3_ce/training.py').exists(),'Pinned actual native CPU optimizer required')
class ActualNativeChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f.TrainingStepTests.setUpClass();cls.addClassCleanup(f.TrainingStepTests.doClassCleanups)
    def setUp(self):
        fixture=f.TrainingStepTests('test_real_transaction_adam_codec_and_idempotent_composition')
        fixture.setUp();self.addCleanup(fixture.doCleanups);self.f=fixture;self.root=fixture.root
        p=patch.object(c,'ROOT',self.root);p.start();self.addCleanup(p.stop)
        fixture.run_one(1);fixture.run_one(2)
        frozen=t.read_json(fixture.fixture.freeze)
        path=self.root/'CPU_SYNTHETIC_CANARY_NOT_PRODUCTION.json'
        dump(path,dict(status='CANARY_RESUME_PASS',experiment_id=t.EXPERIMENT,config_sha256=frozen['config_sha256'],
                       design_freeze_sha256=fixture.registry.freeze_sha256,CPU_FAKE=True))
        fixture.authority.refs['canary']=c.ref(path)
        self.refreshes=[];self.stop_at=None;self.fail_at=None;self.current=decision()
        def refresh(inputs):
            step=len(inputs['trusted_commits']['11031/T0']);self.refreshes.append(copy.deepcopy(inputs))
            if self.fail_at==step:raise RuntimeError('CPU injected risk refresh failure')
            self.current=decision(step,hold=self.stop_at==step);return copy.deepcopy(self.current)
        fixture.authority.refresh_risk=refresh;fixture.authority.risk_snapshot=lambda:copy.deepcopy(self.current)
        slot=c._slot(11031,'T0',3);slot.mkdir(parents=True)
        refs=[c.ref(fixture.store.path/f'step-{step:03d}/COMMIT.json') for step in (1,2)]
        self.ctx=dict(slot=slot,start=11031,arm='T0',first=3,last=8,store=fixture.store,authority=fixture.authority,
            identity=copy.deepcopy(fixture.handle.path_identity),endpoint=fixture.handle.endpoint(),sources=fixture.sources,
            initial=refs[-1],release_ref={'CPU_FAKE':True},process={'CPU_FAKE':True},
            release=dict(storage_budget={'CPU_FAKE':True},original_train_release={'CPU_FAKE':True}),
            risk_inputs=dict(trusted_commits={'11031/T0':refs,'11031/T1':[{'unchanged':1},{'unchanged':2}]},dev_endpoints={'CPU':True}))
        for p in (patch.object(c,'_guard'),patch.object(c.storage,'make_storage_ready',return_value=fixture.storage_ready)):
            p.start();self.addCleanup(p.stop)
    def test_six_real_native_updates_exact768_responses_order_and_risk_after_each_commit(self):
        result=c._run_chunk(self.f.handle,self.ctx)
        self.assertEqual((len(self.f.generated),self.f.adam_calls,self.f.scheduler_calls),(1024,8,8))
        self.assertEqual(result['train_responses'],768);self.assertEqual(result['formal_optimizer_steps'],6)
        self.assertEqual([len(r['trusted_commits']['11031/T0']) for r in self.refreshes],[3,4,5,6,7,8])
        self.assertTrue(all(r['trusted_commits']['11031/T1']==[{'unchanged':1},{'unchanged':2}] for r in self.refreshes))
        self.assertEqual(result['pending_required_dev'],['11031/T0/8']);self.assertFalse(result['main_authorized'])
        self.assertFalse(result['process_exit_observed']);self.assertFalse((self.f.store.path/'step-009').exists())
        for step in range(3,9):
            raw=t.rows(self.f.output/f'steps/step-{step:03d}/RAW.jsonl')
            self.assertEqual([r['request_id'] for r in raw],[r['request_id'] for r in self.f.store.plans[step]])
            self.assertTrue((self.ctx['slot']/f'RISK_AFTER_STEP_{step:03d}.json').exists())
        self.assertTrue(all(float(v['step'])==8 for v in self.f.tiny.optimizer.state.values()))
    def test_true_risk_hold_stops_all_after_committed_step_without_undo_or_retry(self):
        self.stop_at=4
        with self.assertRaisesRegex(PermissionError,'scientific'):c._run_chunk(self.f.handle,self.ctx)
        self.assertEqual((self.f.adam_calls,len(self.f.generated)),(4,512))
        self.assertTrue((self.f.store.path/'step-004/COMMIT.json').exists());self.assertFalse((self.f.store.path/'step-005').exists())
        self.assertTrue((self.root/'training_control/HOLD.json').exists())
        self.assertFalse((self.ctx['slot']/'RESULT.json').exists())
        with self.assertRaisesRegex(PermissionError,'Restored'):c._run_chunk(self.f.handle,self.ctx)
        self.assertEqual(self.f.adam_calls,4)
    def test_failed_refresh_cannot_use_previous_snapshot_or_advance(self):
        self.fail_at=3
        with self.assertRaisesRegex(RuntimeError,'refresh'):c._run_chunk(self.f.handle,self.ctx)
        self.assertEqual(self.f.adam_calls,3);self.assertTrue((self.f.store.path/'step-003/COMMIT.json').exists())
        self.assertFalse((self.f.store.path/'step-004').exists());self.assertEqual(len(self.refreshes),1)
    def test_incomplete_generation_produces_no_new_optimizer_step(self):
        self.f.fail_runner=True
        with self.assertRaisesRegex(RuntimeError,'missing completion'):c._run_chunk(self.f.handle,self.ctx)
        self.assertEqual(self.f.adam_calls,2);self.assertFalse((self.f.store.path/'step-003/COMMIT.json').exists())
        self.assertFalse(self.refreshes)


if __name__=='__main__':unittest.main()
