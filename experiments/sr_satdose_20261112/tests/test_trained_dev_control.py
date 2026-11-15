"""CPU controller tests: real durable bridge, synthetic native sampler only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import trained_dev_control as c
from analysis import evaluation


class FakeHandle:
    def __init__(self,fixture,authority,output):
        self.fx=fixture;self.authority=authority;self.output=output;self.attempted=0;self.closed=False
        self.ep=copy.deepcopy(authority.ctx['endpoint']);self.auth(None)
    def auth(self,rid):
        a=self.authority;x=a.ctx
        return a.verify(root=c.ROOT,selection=x['selection'],endpoint=self.ep,design_freeze_sha256=c.FREEZE_SHA,
            execution_sources_sha256=x['sources_sha256'],request_ids_sha256=c.object_sha([r['request_id'] for r in x['requests']]),
            request_id=rid)
    def endpoint(self):return copy.deepcopy(self.ep)
    def __call__(self,request,root,capture):
        self.auth(request['request_id']);self.attempted+=1;self.fx.attempts.append(request['request_id'])
        if self.fx.fail_at==self.attempted:raise RuntimeError('synthetic draw interrupted')
        if self.fx.hook:self.fx.hook(self)
        capture(dict(tokens=[12,13],raw_tokens=[12,13],completion_token_count=2,sampler_logprobs=[-.2,-.3],
            sampler_logprob_source='actual_generate_raw_logits_selected_token',seed=request['seed'],assistant_prefill='{',
            full_text='{"selected":["A"]}',generated_text='"selected":["A"]}',
            model_input={k:request[k] for k in ('prompt','image_file','image_sha256')},prompt_tensor_hash='a'*64,
            finish_reason='balanced',generation_status='COMPLETE',technical_validation_errors=[]))
        with (self.output/'accounting.jsonl').open('a') as f:
            for kind,count in [('completion_attempts',1),('generated_tokens',2)]:
                f.write(json.dumps(dict(kind=kind,count=count,metadata=dict(group_seed=request['seed'])))+'\n')
    def close(self):
        self.closed=True
        return dict(self.ep,attempted_requests=self.attempted,optimizer_steps=0,scheduler_steps=0,parameter_updates=0)


class TrainedDevControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.attempts=[];self.handles=[];self.fail_at=None;self.hook=None;self.global_hold=False
        self.process=dict(host='CPU_FIXTURE',boot_id='BOOT',pid=101,start_ticks=1)
        self.cfg=dict(experiment_id=c.EXPERIMENT,parent='b0',start_bindings={str(s):dict(policy_sha256=str(s)[-1]*64,
            reference_sha256='a'*64) for s in (11031,11032)})
        self.write('config.json',self.cfg);self.config_sha=c.sha(self.root/'config.json')
        self.write('source_lock.json',{'native_src':'/CPU_ONLY','native_files':{}})
        (self.root/'image.png').write_bytes(b'CPU image placeholder');self.manifest=[];self.requests={}
        for start in (11031,11032):
            for arm in ('T0','T1'):
                for step in (2,8,16):
                    for i in range(108):
                        rid=f'CPU/DEV/{start}/{arm}/{step}/{i}'
                        m=dict(request_id=rid,split='DEV',start_seed=start,arm=arm,step=step,seed=1000+i,panel='list8',
                            view='conflict',root_id=f'root-{i}',structure='TIES',draw=0,case_id='case',input_id='input')
                        self.manifest.append(m);self.requests[rid]=dict(m,scope='DEV',prompt='public unchanged',
                            image_file='image.png',image_sha256=c.sha(self.root/'image.png'),assistant_prefill='{')
        (self.root/'dev_manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in self.manifest))
        self.freeze=dict(config_sha256=self.config_sha,artifacts={k:self.ref(self.root/p) for k,p in
            [('config','config.json'),('source_lock','source_lock.json'),('dev_manifest','dev_manifest.jsonl')]})
        self.write('DESIGN_FREEZE.json',self.freeze);self.freeze_sha=c.sha(self.root/'DESIGN_FREEZE.json')
        for key,value in [('ROOT',self.root),('FREEZE_SHA',self.freeze_sha),('CONFIG_SHA',self.config_sha)]:
            self.patch(c,key,value)
        self.patch(c,'_process_identity',side_effect=lambda:copy.deepcopy(self.process))
        self.registry=SimpleNamespace(root=self.root,freeze_path=self.root/'DESIGN_FREEZE.json',freeze_sha256=self.freeze_sha,
            config=self.cfg,requests=self.requests,request=lambda rid:copy.deepcopy(self.requests[rid]))
        self.execution={'admission_verifier':self.source('execution/admission_authority.py'),
                        'risk_verifier':self.source('analysis/risk.py')}
        self.original={'execution_sources':self.execution}
        self.write('admission.json',{'CPU_ONLY':True});self.write('registration.json',{});self.write('gpu.json',{})
        self.write('canary.json',{});self.write('original.json',{});self.write('risk.json',{});self.write('canary_pass.json',{})
        self.spec=dict(sources={'authority':self.execution['admission_verifier']},
            training_execution_sources_sha256=c.object_sha(self.execution),documents={k:self.ref(self.root/p) for k,p in
                [('registration','registration.json'),('gpu_report','gpu.json')]})
        self.write('spec.json',self.spec);self.write('budget.json',dict(scope='TRAINED_DEV_ONLY',required_bytes=4096,max_observation_age_seconds=300))
        self.release=dict(schema='sr-satdose-trained-dev-release-v1',release_root=str(PACKAGE),design_freeze_sha256=self.freeze_sha,
            sources={f'execution/{name}.py':c.sha(PACKAGE/'execution'/(name+'.py')) for name in
            ('trained_dev_control','evaluation_runtime','evaluation_loader','training_state','training_transaction','admission_authority')},
            admission=self.ref(self.root/'admission.json'),admission_spec=self.ref(self.root/'spec.json'),
            original_train_release=self.ref(self.root/'original.json'),risk_clarification=self.ref(self.root/'risk.json'),
            canary_spec=self.ref(self.root/'canary.json'),storage_budget=self.ref(self.root/'budget.json'),trusted_commits={},continuation_results={},
            dev_endpoints={f'{s}/START/0':{'receipt':self.ref(self.root/'admission.json'),'binding':self.ref(self.root/'admission.json')}
                           for s in (11031,11032)})
        self.stores={};self.set_commits(2)
        self.release_path=self.root/'release.json';self.update_release()
        self.base=SimpleNamespace(_observed=SimpleNamespace(stable=Mock()))
        self.canary=SimpleNamespace(execution_sources=self.execution,stable=Mock(),receipt=self.ref(self.root/'canary_pass.json'),
            identities={},spec={'process_exit_proofs':{}})
        launcher=dict(status='FOUR_FIRST_TWO_PATHS_EXIT_OBSERVED_NOT_CANARY_ADMISSION',release=self.release['original_train_release'],paths={})
        for index,key in enumerate(c.PATHS):
            start,arm=key.split('/');identity=c.read(c.bound(self.release['trusted_commits'][key][-1]))['identity']
            self.canary.identities[key]=identity
            self.write(f'first_two_launcher/start-{start}/{arm}/EXIT_PROOF.json',{})
            proof=self.ref(self.root/f'first_two_launcher/start-{start}/{arm}/EXIT_PROOF.json')
            self.canary.spec['process_exit_proofs'][key]=proof
            launched=dict(status='FIRST_TWO_PATH_EXIT_OBSERVED_NOT_CANARY_ADMISSION',process_exit_proof=proof)
            for phase,status in [('baseline',c.first_two.BASELINE_STATUS),('reload',c.first_two.RELOAD_STATUS)]:
                process=dict(host='CPU_FIXTURE',boot_id='BOOT',pid=500+index*2+(phase=='reload'),start_ticks=1)
                name=f'first_two_control/start-{start}/{arm}/{phase}'
                self.write(name+'/RESULT.json',dict(status=status,release=self.release['original_train_release'],
                    start_seed=int(start),arm=arm,identity=identity,process=process))
                self.write(name+'/CLAIM.json',dict(schema='sr-satdose-first-two-claim-v1',status='UNKNOWN_UNTIL_COMPLETE_NO_RETRY',
                    release=self.release['original_train_release'],start_seed=int(start),arm=arm,phase=phase,process=process))
                launched[phase+'_result']=self.ref(self.root/name/'RESULT.json')
            launcher['paths'][key]=launched
        self.write('first_two_launcher/RESULT.json',launcher)
        self.verify=self.patch(c,'verify_release',side_effect=lambda *_:(self.release,self.original))
        self.patch(c,'_registry',return_value=self.registry)
        self.patch(c.first_two,'_allocation',return_value={})
        self.budget=self.patch(c.storage,'_budget',return_value=c.read(self.root/'budget.json'))
        self.storage=self.patch(c,'storage_check',return_value={'CPU_ONLY':True})
        self.base_factory=self.patch(c.admission_authority,'FirstTwoStepAdmission',return_value=self.base)
        self.canary_verify=self.patch(c.continuation_authority,'verify_canary',return_value=self.canary)
        self.risk=self.patch(c.risk,'assess',side_effect=self.assess)
        self.patch(c.fork_loader,'load_native_ops',return_value=object())
        self.patch(c.tx,'TransactionStore',side_effect=lambda root,freeze,sha,identity,**kw:self.stores[f"{identity['train_seed']}/{identity['arm']}"])
        self.factory_mock=self.patch(c.runtime,'build_evaluation_runtime',side_effect=self.factory)

    def patch(self,obj,key,*args,**kwargs):
        p=patch.object(obj,key,*args,**kwargs);result=p.start();self.addCleanup(p.stop);return result
    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    def ref(self,path):return dict(path=str(path.relative_to(self.root)),sha256=c.sha(path))
    def source(self,name):return dict(path=str(PACKAGE/name),sha256=c.sha(PACKAGE/name))
    def update_release(self):
        self.release_number=getattr(self,'release_number',0)+1
        name=f'release-{self.release_number:03d}.json';self.write(name,self.release)
        self.release_path=self.root/name;self.release_sha=c.sha(self.release_path)
    def set_commits(self,n):
        for key in c.PATHS:
            start,arm=key.split('/');path=self.root/'training_transactions'/f'start-{start}'/arm
            refs=[]
            for step in range(1,n+1):
                slot=path/f'step-{step:03d}';slot.mkdir(parents=True,exist_ok=True);(slot/'state.pt').write_bytes(b'CPU_ONLY')
                marker=dict(step=step,policy_sha256='b'*64,reference_sha256='a'*64,state_sha256=c.sha(slot/'state.pt'),
                    identity=dict(train_seed=int(start),arm=arm,execution_sources_sha256=c.object_sha(self.execution)),
                    transaction={'genesis_reference':self.ref(self.root/'admission.json')})
                (slot/'COMMIT.json').write_text(json.dumps(marker));refs.append(self.ref(slot/'COMMIT.json'))
            self.release['trusted_commits'][key]=refs
            def inspect(step,digest,path=path):
                p=path/f'step-{step:03d}/COMMIT.json';c.require(c.sha(p)==digest,'Commit SHA changed');return c.read(p)
            self.stores[key]=SimpleNamespace(path=path,inspect_committed=inspect)
    def assess(self,*args,**kw):
        n=len(next(iter(kw['trusted_commits'].values())))
        due={f'{key}/{step}' for key in c.PATHS for step in (2,8,16) if step<=n}
        pending=sorted(due-set(kw['dev_endpoints']))
        hold=self.global_hold and len(kw['dev_endpoints'])>2
        return dict(status='GLOBAL_SCIENTIFIC_HOLD' if hold else 'PENDING_REQUIRED_DEV' if pending else 'NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED',
            global_hold=hold,pending_required_dev=pending,hold_reasons=['HOLD_SEVERE_REGRESSION'] if hold else [],observed_files={})
    def factory(self,*args,**kw):
        self.assertEqual(kw['scope'],'DEV');self.assertIn(kw['arm'],('T0','T1'));self.assertIn(kw['step'],(2,8,16))
        self.assertIs(kw['commit_store'],self.stores[f"{kw['start_seed']}/{kw['arm']}"])
        self.assertEqual(kw['expected_identity'],c.read(c.bound(self.release['trusted_commits'][f"{kw['start_seed']}/{kw['arm']}"][-1]))['identity'])
        self.assertTrue(not any(Path(kw['output_dir']).iterdir()))
        h=FakeHandle(self,kw['authority'],Path(kw['output_dir']));self.handles.append(h);return h
    def execute(self,start=11031,arm='T0',step=2):return c.execute(self.release_path,self.release_sha,start,arm,step)

    def test_exact108_actual_bridge_raw_and_explicit_committed_checkpoint(self):
        result=self.execute();self.assertEqual(result['status'],'TRAINED_DEV_ENDPOINT_COMPLETE')
        receipt=c.read(c.bound(result['receipt']));binding=c.read(c.bound(result['endpoint_binding']))
        self.assertEqual(binding['checkpoint']['commit'],self.release['trusted_commits']['11031/T0'][1])
        self.assertEqual(receipt['responses'],108);self.assertTrue(self.handles[0].closed)
        selected=[r for r in self.manifest if (r['start_seed'],r['arm'],r['step'])==(11031,'T0',2)]
        raw=c.rows(c.bound(receipt['raw']));self.assertEqual([r['request_id'] for r in raw],[r['request_id'] for r in selected])
        for row,m,reference in zip(raw,selected,receipt['complete_files']):
            complete=c.read(c.bound(reference));self.assertTrue(all(row[k]==v for k,v in complete.items()))
            inp=dict(prompt='public unchanged',image_file='image.png',assistant_prefill='{')
            self.assertEqual(evaluation._capture(self.root,c.bound(result['receipt']).parent,reference,m,binding['endpoint'],
                self.freeze_sha,inp,dict(png_sha256=c.sha(self.root/'image.png'))),complete)
        self.assertEqual(self.risk.call_count,2);self.assertEqual(len(self.attempts),108)
        self.assertEqual(self.base._observed.stable.call_count,3)

    def test_all12_registered_dev_endpoints_preserve_checkpoint_and_request_population(self):
        for step in (2,8,16):
            if step>2:
                self.set_commits(step);first,last=((3,8) if step==8 else (9,16))
                for index,key in enumerate(c.PATHS):
                    start,arm=key.split('/');name=f'continuation_control/start-{start}/{arm}/chunk-{first:03d}-{last:03d}'
                    process=dict(host='CPU_FIXTURE',boot_id='BOOT',pid=9000+step+index,start_ticks=1)
                    self.write(name+'/CLAIM.json',dict(status='ATTEMPTED_UNKNOWN_NO_RETRY',process=process,
                        release=self.release['original_train_release']))
                    self.write(name+'/RESULT.json',dict(schema='sr-satdose-continuation-result-v1',status='CONTINUATION_CHUNK_COMPLETE',
                        permission='NO_AUTOMATIC_SUCCESSOR_OR_MAIN',start_seed=int(start),arm=arm,first_step=first,last_step=last,
                        identity=self.canary.identities[key],commits=self.release['trusted_commits'][key][first-1:last],
                        initial_commit=self.release['trusted_commits'][key][first-2],process=process,
                        release=self.release['original_train_release'],post_risk=self.ref(self.root/'risk.json')))
                    self.release['continuation_results'][f'{key}/{first}-{last}']=self.ref(self.root/name/'RESULT.json')
            for key in c.PATHS:
                start,arm=key.split('/');self.process['pid']+=1;self.update_release()
                result=self.execute(int(start),arm,step)
                self.assertEqual(result['commit'],self.release['trusted_commits'][key][step-1])
                self.release['dev_endpoints'][f'{key}/{step}']=dict(receipt=result['receipt'],binding=result['endpoint_binding'])
        self.assertEqual(len(self.attempts),1296);self.assertEqual(len(set(self.attempts)),1296)
        self.assertTrue(all(h.closed for h in self.handles));self.assertEqual(len(self.handles),12)

    def test_repeated_endpoint_never_retries(self):
        self.execute()
        with self.assertRaisesRegex(PermissionError,'never retry'):self.execute()
        self.assertEqual(len(self.attempts),108)

    def test_interrupted_draw_retains_unknown_and_blocks_other_paths(self):
        self.fail_at=5
        with self.assertRaises(RuntimeError):self.execute()
        slot=c._slot(11031,'T0',2);self.assertTrue((slot/'HOLD.json').exists())
        self.assertEqual(len(list((slot/'endpoint').glob('*.BEGIN.json'))),5)
        self.assertEqual(len(list((slot/'endpoint').glob('*.COMPLETE.json'))),4)
        with self.assertRaises(PermissionError):self.execute(11032,'T1')
        self.assertEqual(len(self.attempts),5)

    def test_canary_and_full_admission_required_before_claim_or_model(self):
        self.canary_verify.side_effect=PermissionError('actual canary failed')
        with self.assertRaisesRegex(PermissionError,'canary failed'):self.execute()
        self.assertFalse((self.root/'trained_dev').exists());self.factory_mock.assert_not_called()
        self.canary_verify.side_effect=None;self.base_factory.side_effect=PermissionError('actual admission failed')
        with self.assertRaisesRegex(PermissionError,'admission failed'):self.execute()
        self.factory_mock.assert_not_called()

    def test_existing_global_hold_blocks_before_model(self):
        self.write('training_control/HOLD.json',{})
        with self.assertRaisesRegex(PermissionError,'Global failed'):self.execute()
        self.factory_mock.assert_not_called()

    def test_launcher_and_continuation_holds_block_otherwise_valid_canary(self):
        for name in ('first_two_launcher/HOLD.json','continuation_control/ERROR.json'):
            self.write(name,{})
            with self.assertRaisesRegex(PermissionError,'Global failed'):self.execute()
            (self.root/name).unlink()
        self.factory_mock.assert_not_called()

    def test_unknown_first_two_reload_and_generation_block_dev(self):
        p=self.root/'first_two_control/start-11032/T1/reload/RESULT.json';saved=p.read_bytes();p.unlink()
        with self.assertRaises(FileNotFoundError):self.execute()
        p.write_bytes(saved)
        self.write('first_two_control/start-11032/T1/baseline/audit/extra.BEGIN.json',{})
        with self.assertRaisesRegex(PermissionError,'Unresolved first-two generation'):self.execute()
        self.factory_mock.assert_not_called()

    def test_scientific_hold_blocks_but_pending_dev_is_allowed(self):
        self.risk.side_effect=None;self.risk.return_value=dict(global_hold=True,status='GLOBAL_SCIENTIFIC_HOLD',pending_required_dev=['11031/T0/2'])
        with self.assertRaisesRegex(PermissionError,'safe scheduled'):self.execute()
        self.factory_mock.assert_not_called()

    def test_post_dev_scientific_stop_published_without_discarding_result(self):
        self.global_hold=True;result=self.execute()
        self.assertTrue(result['global_scientific_hold']);self.assertTrue((self.root/'training_control/HOLD.json').is_file())
        self.assertEqual(c.read(c.bound(result['receipt']))['responses'],108)
        with self.assertRaises(PermissionError):self.execute(11032,'T1')

    def test_all_four_paths_must_reach_same_barrier(self):
        self.release['trusted_commits']['11032/T1'].pop();self.update_release()
        with self.assertRaisesRegex(PermissionError,'same fixed DEV barrier'):self.execute()
        self.factory_mock.assert_not_called()

    def test_later_checkpoint_cannot_skip_earlier_dev_barrier(self):
        self.set_commits(8);self.update_release()
        with patch.object(c,'_continuation_complete',return_value=[]):
            with self.assertRaisesRegex(PermissionError,'Earlier scheduled DEV'):self.execute(step=8)
        self.factory_mock.assert_not_called()

    def test_unknown_continuation_attempt_blocks_even_with_valid_canary(self):
        self.write('continuation_control/start-11031/T0/chunk-003-008/CLAIM.json',{'status':'ATTEMPTED_UNKNOWN_NO_RETRY'})
        with self.assertRaisesRegex(PermissionError,'Unknown/incomplete continuation'):self.execute()
        self.factory_mock.assert_not_called()

    def test_step8_requires_four_successful_pinned_continuation_results(self):
        self.set_commits(8);self.update_release()
        with self.assertRaisesRegex(PermissionError,'completed continuation references'):self.execute(step=8)
        self.factory_mock.assert_not_called()

    def test_no_start_main_step24_or_unregistered_path(self):
        for args in [(11031,'START',0),(11031,'T0',24),(9,'T1',2),(11031,'T2',2)]:
            with self.assertRaisesRegex(PermissionError,'fixed trained'):self.execute(*args)
        self.factory_mock.assert_not_called()

    def test_foreign_canary_training_source_identity_rejected(self):
        self.canary.execution_sources={'foreign':True}
        with self.assertRaisesRegex(PermissionError,'another TRAIN'):self.execute()

    def test_original_train_identity_cannot_change(self):
        p=c.bound(self.release['trusted_commits']['11031/T0'][-1]);m=c.read(p);m['identity']['execution_sources_sha256']='c'*64;p.write_text(json.dumps(m))
        self.release['trusted_commits']['11031/T0'][-1]=self.ref(p);self.update_release()
        with self.assertRaisesRegex(PermissionError,'original TRAIN identity'):self.execute()

    def test_storage_failure_retains_claim_no_model(self):
        self.storage.side_effect=PermissionError('insufficient capacity')
        with self.assertRaisesRegex(PermissionError,'insufficient capacity'):self.execute()
        self.assertTrue((c._slot(11031,'T0',2)/'CLAIM.json').exists());self.factory_mock.assert_not_called()

    def test_changed_commit_during_draw_blocks_next_request(self):
        def hook(h):
            if h.attempted==1:c.bound(self.release['trusted_commits']['11032/T1'][0]).write_text('{}')
        self.hook=hook
        with self.assertRaises(PermissionError):self.execute()
        self.assertEqual(len(self.attempts),1)

    def test_training_advance_during_endpoint_is_rejected(self):
        def hook(h):
            if h.attempted==1:(self.stores['11032/T1'].path/'step-003').mkdir()
        self.hook=hook
        with self.assertRaisesRegex(PermissionError,'advanced during'):self.execute()

    def test_authority_rejects_main_and_out_of_order_request(self):
        def factory(*args,**kw):
            h=self.factory(*args,**kw);a=h.authority;x=a.ctx
            for changes in ({'scope':'MAIN'},{'step':8},{'arm':'START'}):
                with self.assertRaisesRegex(PermissionError,'scope differs'):
                    a.verify(root=c.ROOT,selection=dict(x['selection'],**changes),endpoint=x['endpoint'],
                        design_freeze_sha256=c.FREEZE_SHA,execution_sources_sha256=x['sources_sha256'],
                        request_ids_sha256=c.object_sha([r['request_id'] for r in x['requests']]),request_id=None)
            with self.assertRaisesRegex(PermissionError,'Repeated/out-of-order'):h.auth(x['requests'][1]['request_id'])
            return h
        self.factory_mock.side_effect=factory;self.execute()

    def test_prior_endpoint_must_be_bound_and_fresh_process(self):
        result=self.execute()
        with self.assertRaisesRegex(PermissionError,'incomplete or omitted'):self.execute(11031,'T1')
        self.release['dev_endpoints']['11031/T0/2']=dict(receipt=result['receipt'],binding=result['endpoint_binding']);self.update_release()
        with self.assertRaisesRegex(PermissionError,'fresh process'):self.execute(11031,'T1')
        self.process['pid']+=1;self.execute(11031,'T1');self.assertEqual(len(self.attempts),216)

    def test_global_lease_and_closed_authority(self):
        with c.Lease():
            with self.assertRaises(BlockingIOError):self.execute()
        self.execute()
        with self.assertRaisesRegex(PermissionError,'Inactive global'):self.handles[0].authority._fast()

    def test_missing_requested_pending_dev_is_rejected(self):
        self.risk.side_effect=None;self.risk.return_value=dict(global_hold=False,status='PENDING_REQUIRED_DEV',pending_required_dev=['11031/T1/2'])
        with self.assertRaisesRegex(PermissionError,'safe scheduled'):self.execute()

    def test_wrong_storage_scope_is_rejected_before_claim(self):
        self.budget.return_value={'scope':'FIRST_TWO_AND_CANARY_ONLY'}
        with self.assertRaisesRegex(PermissionError,'storage scope'):self.execute()
        self.assertFalse((self.root/'trained_dev').exists())

    def test_no_model_preflight_and_source_mutation_boundary(self):
        ctx=c.preflight(self.release_path,self.release_sha,11031,'T0',2)
        self.assertEqual(len(ctx['requests']),108);self.factory_mock.assert_not_called();self.assertFalse((self.root/'trained_dev').exists())
        def hook(h):
            if h.attempted==1:self.write('admission.json',{'changed':True})
        self.hook=hook
        with self.assertRaises(PermissionError):self.execute()
        self.assertEqual(len(self.attempts),1)

    def test_corrupt_checkpoint_at_completion_prevents_receipt(self):
        def hook(h):
            if h.attempted==108:
                h.authority.ctx['store'].inspect_committed=Mock(side_effect=PermissionError('changed checkpoint'))
        self.hook=hook
        with self.assertRaisesRegex(PermissionError,'changed checkpoint'):self.execute()
        self.assertFalse((c._slot(11031,'T0',2)/'endpoint/RECEIPT.json').exists())

    def test_extra_unreferenced_draw_cannot_be_hidden_by_complete_receipt(self):
        def hook(h):
            if h.attempted==108:(h.output/'extra.BEGIN.json').write_text('{}')
        self.hook=hook
        with self.assertRaisesRegex(PermissionError,'Extra or unresolved DEV'):self.execute()
        self.assertFalse((c._slot(11031,'T0',2)/'endpoint/RECEIPT.json').exists())

    def test_real_release_verifier_rejects_wrong_hash_before_imported_train_verifier(self):
        # Access the real implementation through the patch wrapper's target.
        from importlib import util
        spec=util.spec_from_file_location('execution._trained_dev_release_test',PACKAGE/'execution/trained_dev_control.py')
        module=util.module_from_spec(spec);sys.modules[spec.name]=module
        self.addCleanup(lambda:sys.modules.pop(spec.name,None));spec.loader.exec_module(module)
        with patch.object(module.first_two,'verify_release') as original:
            with self.assertRaisesRegex(PermissionError,'Untrusted trained DEV release'):
                module.verify_release(self.release_path,'0'*64)
            original.assert_not_called()

    def test_real_release_verifier_requires_preserved_original_sources_and_new_controller_pins(self):
        from importlib import util
        spec=util.spec_from_file_location('execution._trained_dev_release_pins_test',PACKAGE/'execution/trained_dev_control.py')
        module=util.module_from_spec(spec);sys.modules[spec.name]=module
        self.addCleanup(lambda:sys.modules.pop(spec.name,None));spec.loader.exec_module(module)
        module.ROOT=self.root;module.FREEZE_SHA=self.freeze_sha;module.CONFIG_SHA=self.config_sha
        original=copy.deepcopy(self.original)
        original.update(release_root=str(PACKAGE),sources=self.release['sources'],**{k:self.release[k] for k in
            ('admission_spec','admission','risk_clarification')})
        original['execution_sources'].update(continuation_verifier=self.source('execution/continuation_authority.py'),
            trained_dev_control=self.source('execution/trained_dev_control.py'))
        for name in ('analysis/risk.py','execution/continuation_authority.py'):
            self.release['sources'][name]=c.sha(PACKAGE/name)
        self.release.update(schema='sr-satdose-trained-dev-release-v1',status='AUDITED_FOR_TRAINED_DEV_ONLY',
            permission='TRAINED_DEV_ONLY_NO_TRAINING_OR_MAIN',experiment_id=c.EXPERIMENT,evidence_root=str(self.root),
            release_root=str(PACKAGE),design_freeze_sha256=self.freeze_sha,config_sha256=self.config_sha)
        self.update_release()
        with patch.object(module.first_two,'verify_release',return_value=original), \
             patch.object(module.first_two,'ROOT',self.root),patch.object(module.storage,'ROOT',self.root), \
             patch.object(module.storage,'FREEZE_SHA',self.freeze_sha):
            module.verify_release(self.release_path,self.release_sha)
            original['execution_sources']['trained_dev_control']['sha256']='a'*64
            with self.assertRaisesRegex(PermissionError,'dependency pin'):
                module.verify_release(self.release_path,self.release_sha)


if __name__=='__main__':unittest.main()
