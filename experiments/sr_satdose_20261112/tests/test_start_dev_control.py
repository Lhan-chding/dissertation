"""CPU controller/authority tests with synthetic sampler; no GPU/admission proof."""
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
from execution import start_dev_control as c
from analysis import evaluation


class FakeHandle:
    """Synthetic CPU sampler used only behind a patched model factory."""
    def __init__(self,fx,authority,output):
        self.fx=fx;self.authority=authority;self.output=output;self.attempted=0;self.closed=False
        self.ep=copy.deepcopy(authority.ctx['endpoint'])
        self.receipt={'CPU_FIXTURE_ONLY':True}
        authority.verify(root=c.ROOT,selection=authority.ctx['selection'],endpoint=self.ep,design_freeze_sha256=c.FREEZE_SHA,
            execution_sources_sha256=authority.ctx['sources_sha256'],request_ids_sha256=c.object_sha([r['request_id'] for r in authority.ctx['requests']]),request_id=None)
    def endpoint(self):return copy.deepcopy(self.ep)
    def __call__(self,request,root,capture):
        a=self.authority;ctx=a.ctx
        a.verify(root=root,selection=ctx['selection'],endpoint=self.ep,design_freeze_sha256=c.FREEZE_SHA,
            execution_sources_sha256=ctx['sources_sha256'],request_ids_sha256=c.object_sha([r['request_id'] for r in ctx['requests']]),request_id=request['request_id'])
        self.fx.attempts.append(request['request_id']);self.attempted+=1
        if self.fx.fail_at==self.attempted:raise RuntimeError('synthetic interrupted draw')
        if self.fx.hook:self.fx.hook(self)
        raw=dict(tokens=[12,13],raw_tokens=[12,13],completion_token_count=2,sampler_logprobs=[-.2,-.3],
            sampler_logprob_source='actual_generate_raw_logits_selected_token',seed=request['seed'],
            assistant_prefill='{',full_text='{"selected":["A"]}',generated_text='"selected":["A"]}',
            model_input={k:request[k] for k in ('prompt','image_file','image_sha256')},prompt_tensor_hash='a'*64,
            finish_reason='balanced',generation_status='COMPLETE',technical_validation_errors=[])
        capture(raw)
        with (self.output/'accounting.jsonl').open('a') as f:
            for kind,count in [('completion_attempts',1),('generated_tokens',2)]:
                f.write(json.dumps(dict(kind=kind,count=count,metadata=dict(group_seed=request['seed'])))+'\n')
    def close(self):
        self.closed=True
        return dict(self.ep,attempted_requests=self.attempted,optimizer_steps=0,scheduler_steps=0,parameter_updates=0)


class StartDevControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.attempts=[];self.fail_at=None;self.hook=None;self.handles=[]
        self.process=dict(host='CPU_FIXTURE',boot_id='CPU_FIXTURE_BOOT',pid=11031,start_ticks='1')
        proc=patch.object(c,'_process_identity',side_effect=lambda:copy.deepcopy(self.process));proc.start();self.addCleanup(proc.stop)
        self.cfg=dict(experiment_id=c.EXPERIMENT,parent='b0',start_bindings={str(s):dict(policy_sha256=str(s)[-1]*64,
            reference_sha256='a'*64) for s in (11031,11032)})
        self.write('config.json',self.cfg);self.config_sha=c.sha(self.root/'config.json')
        self.write('source_lock.json',{})
        (self.root/'image.png').write_bytes(b'CPU image placeholder')
        self.manifest=[];self.requests={}
        for start in (11031,11032):
            for i in range(108):
                rid=f'CPU/DEV/{start}/{i}'
                m=dict(request_id=rid,split='DEV',start_seed=start,arm='START',step=0,seed=1000+i,panel='list8',
                    view='conflict',root_id=f'root-{i}',structure='TIES',draw=0,case_id='case',input_id='input')
                self.manifest.append(m)
                self.requests[rid]=dict(m,scope='DEV',prompt='public unchanged',image_file='image.png',
                    image_sha256=c.sha(self.root/'image.png'),assistant_prefill='{')
        (self.root/'dev_manifest.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in self.manifest))
        self.freeze=dict(config_sha256=self.config_sha,artifacts={k:self.ref(self.root/p) for k,p in
            [('config','config.json'),('source_lock','source_lock.json'),('dev_manifest','dev_manifest.jsonl')]})
        self.write('DESIGN_FREEZE.json',self.freeze);self.freeze_sha=c.sha(self.root/'DESIGN_FREEZE.json')
        for key,value in [('ROOT',self.root),('FREEZE_SHA',self.freeze_sha),('CONFIG_SHA',self.config_sha)]:
            p=patch.object(c,key,value);p.start();self.addCleanup(p.stop)
        self.registry=SimpleNamespace(root=self.root,freeze_path=self.root/'DESIGN_FREEZE.json',freeze_sha256=self.freeze_sha,
            config=self.cfg,requests=self.requests,request=lambda rid:copy.deepcopy(self.requests[rid]))
        self.write('registration.json',{'CPU_FIXTURE':True});self.write('gpu.json',{'CPU_FIXTURE':True})
        self.candidate=dict(status='ADMITTED_FIRST_TWO_STEPS_ONLY',permission='FOUR_PATHS_MAX_STEP_2',CPU_FIXTURE=True)
        self.write('admission.json',self.candidate)
        self.write('budget.json',dict(scope='START_DEV_ONLY',required_bytes=4096,max_observation_age_seconds=300))
        self.spec=dict(root=str(self.root),documents={k:self.ref(self.root/p) for k,p in
            [('config','config.json'),('registration','registration.json'),('gpu_report','gpu.json'),('independent_audit','gpu.json'),
             ('binding','gpu.json'),('waiver','gpu.json')]},sources={'authority':dict(path=str(PACKAGE/'execution/admission_authority.py'),
                sha256=c.sha(PACKAGE/'execution/admission_authority.py'))},training_execution_sources_sha256='a'*64)
        self.write('spec.json',self.spec)
        self.release=dict(sources={f'execution/{name}.py':c.sha(PACKAGE/'execution'/(name+'.py')) for name in
            ('start_dev_control','evaluation_runtime','evaluation_loader','training_state','training_transaction','admission_authority')},
            admission_spec=self.ref(self.root/'spec.json'),admission=self.ref(self.root/'admission.json'),storage_budget=self.ref(self.root/'budget.json'))
        self.write('release.json',self.release);self.release_path=self.root/'release.json';self.release_sha=c.sha(self.release_path)
        self.env=patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_NTASKS':'1','SLURM_NNODES':'1','SLURM_RESTART_COUNT':'0'})
        self.env.start();self.addCleanup(self.env.stop)
        self.patches=[patch.object(c,'verify_release',return_value=self.release),patch.object(c,'_registry',return_value=self.registry),
            patch.object(c.storage,'_budget',return_value=c.read(self.root/'budget.json')),
            patch('execution.admission_authority.prepare',return_value=self.candidate),
            patch.object(c,'storage_check',return_value={'CPU_FIXTURE':True}),
            patch.object(c.runtime,'build_evaluation_runtime',side_effect=self.factory)]
        self.mocks=[p.start() for p in self.patches]
        for p in self.patches:self.addCleanup(p.stop)

    def write(self,name,value):(self.root/name).write_text(json.dumps(value))
    def ref(self,path):return dict(path=str(path.relative_to(self.root)),sha256=c.sha(path))
    def factory(self,*args,**kw):
        self.assertEqual((kw['scope'],kw['arm'],kw['step']),('DEV','START',0))
        self.assertTrue(not any(Path(kw['output_dir']).iterdir()))
        h=FakeHandle(self,kw['authority'],Path(kw['output_dir']));self.handles.append(h);return h
    def execute(self,start=11031):
        self.process['pid']=start
        return c.execute(self.release_path,self.release_sha,start)

    def test_both_starts_exact_216_order_raw_and_completion_binding(self):
        for start in (11031,11032):
            result=self.execute(start);self.assertEqual(result['status'],'START_DEV_ENDPOINT_COMPLETE')
            receipt=c.read(c.bound(result['receipt']));binding=c.read(c.bound(result['endpoint_binding']))
            self.assertEqual(receipt['responses'],108);self.assertIsNone(binding['checkpoint'])
            raw=c.rows(c.bound(receipt['raw']));selected=[r for r in self.manifest if r['start_seed']==start]
            self.assertEqual([r['request_id'] for r in raw],[r['request_id'] for r in selected])
            for row,m,reference in zip(raw,selected,receipt['complete_files']):
                complete=c.read(c.bound(reference));capture=c.read(Path(complete['native_capture_path']))
                self.assertTrue(all(row[k]==v for k,v in complete.items()))
                self.assertTrue(all(complete[k]==v for k,v in capture.items()))
                self.assertEqual(row['suffix_text'],row['generated_text'])
                self.assertEqual(row['scope'],'DEV');self.assertEqual(row['split'],'DEV')
                # Actual analysis capture validator checks unmodified BEGIN/CAPTURE/COMPLETE.
                inp=dict(prompt='public unchanged',image_file='image.png',assistant_prefill='{')
                self.assertEqual(evaluation._capture(self.root,c.bound(result['receipt']).parent,reference,m,
                    binding['endpoint'],self.freeze_sha,inp,dict(png_sha256=c.sha(self.root/'image.png'))),complete)
            self.assertTrue(self.handles[-1].closed)
        self.assertEqual(self.attempts,[r['request_id'] for r in self.manifest])
        self.assertEqual(self.mocks[3].call_count,2)  # Full prepare once per fresh START, never per draw.

    def test_second_start_before_first_is_refused(self):
        with self.assertRaisesRegex(PermissionError,'must complete first'):self.execute(11032)
        self.assertEqual(self.attempts,[]);self.assertEqual(self.handles,[])

    def test_unknown_claim_and_failed_draw_never_resume(self):
        self.fail_at=5
        with self.assertRaisesRegex(RuntimeError,'interrupted draw'):self.execute()
        directory=self.root/'start_dev/start-11031'
        self.assertTrue((directory/'HOLD.json').is_file());self.assertFalse((directory/'RESULT.json').exists())
        self.assertEqual(len(list((directory/'endpoint').glob('*.BEGIN.json'))),5)
        self.assertEqual(len(list((directory/'endpoint').glob('*.COMPLETE.json'))),4)
        with self.assertRaises(PermissionError):self.execute()
        with self.assertRaises(PermissionError):self.execute(11032)
        self.assertEqual(len(self.attempts),5)

    def test_completed_endpoint_is_not_replayed(self):
        self.execute()
        with self.assertRaisesRegex(PermissionError,'Existing START DEV'):self.execute()
        self.assertEqual(len(self.attempts),108)

    def test_global_lock_excludes_second_controller(self):
        with c.Lease():
            with self.assertRaises(BlockingIOError):self.execute()
        self.assertEqual(self.handles,[])
        with c.Lease() as lease:lease.active()

    def test_invalid_start_and_slurm_restart_rejected_before_model(self):
        with self.assertRaisesRegex(PermissionError,'two fixed'):self.execute(999)
        with patch.dict(os.environ,SLURM_RESTART_COUNT='1'):
            with self.assertRaisesRegex(PermissionError,'Fresh one-task'):self.execute()
        self.assertEqual(self.handles,[])

    def test_no_model_when_complete_admission_verification_fails(self):
        self.mocks[3].side_effect=PermissionError('full evidence failed')
        with self.assertRaisesRegex(PermissionError,'full evidence failed'):self.execute()
        self.assertEqual(self.handles,[]);self.assertFalse((self.root/'start_dev').exists())

    def test_published_admission_must_equal_full_candidate(self):
        self.mocks[3].return_value=dict(self.candidate,altered=True)
        with self.assertRaisesRegex(PermissionError,'Published full admission'):self.execute()
        self.assertEqual(self.handles,[])

    def test_storage_failure_claim_is_retained_before_model(self):
        self.mocks[4].side_effect=PermissionError('insufficient capacity')
        with self.assertRaisesRegex(PermissionError,'insufficient capacity'):self.execute()
        self.assertTrue((self.root/'start_dev/start-11031/CLAIM.json').exists())
        self.assertTrue((self.root/'start_dev/start-11031/HOLD.json').exists())
        self.assertEqual(self.handles,[])

    def test_constructor_rejects_main_trained_endpoint_and_other_scope(self):
        def factory(*args,**kw):
            a=kw['authority'];ctx=a.ctx
            for change in ({'scope':'MAIN'},{'arm':'T0','step':2},{'scope':'TRAIN'}):
                with self.assertRaisesRegex(PermissionError,'scope differs'):
                    a.verify(root=c.ROOT,selection=dict(ctx['selection'],**change),endpoint=ctx['endpoint'],
                        design_freeze_sha256=c.FREEZE_SHA,execution_sources_sha256=ctx['sources_sha256'],
                        request_ids_sha256=c.object_sha([r['request_id'] for r in ctx['requests']]),request_id=None)
            return self.factory(*args,**kw)
        self.mocks[5].side_effect=factory;self.execute()

    def test_source_or_admission_mutation_blocks_next_request(self):
        def hook(handle):
            if handle.attempted==1:self.write('admission.json',{'changed':True})
        self.hook=hook
        with self.assertRaises(PermissionError):self.execute()
        self.assertEqual(len(self.attempts),1)
        self.assertTrue((self.root/'start_dev/start-11031/HOLD.json').exists())

    def test_new_global_hold_during_run_blocks_next_request(self):
        def hook(handle):
            if handle.attempted==1:
                p=self.root/'training_transactions';p.mkdir();(p/'HOLD.json').write_text('{}')
        self.hook=hook
        with self.assertRaisesRegex(PermissionError,'Global failed/unknown'):self.execute()
        self.assertEqual(len(self.attempts),1)

    def test_prior_complete_tampering_blocks_second_start(self):
        r=self.execute();receipt=c.read(c.bound(r['receipt']))
        c.bound(receipt['complete_files'][0]).write_text('{}')
        with self.assertRaises(PermissionError):self.execute(11032)
        self.assertEqual(len(self.attempts),108)

    def test_prior_begin_tampering_blocks_second_start(self):
        self.execute()
        path=next((self.root/'start_dev/start-11031/endpoint').glob('*.BEGIN.json'))
        value=c.read(path);value['seed']+=1;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(PermissionError,'capture chain differs'):self.execute(11032)
        self.assertEqual(len(self.attempts),108)

    def test_controller_authority_refuses_out_of_order_request(self):
        def factory(*args,**kw):
            h=self.factory(*args,**kw);a=kw['authority'];ctx=a.ctx
            with self.assertRaisesRegex(PermissionError,'Repeated/out-of-order'):
                a.verify(root=c.ROOT,selection=ctx['selection'],endpoint=ctx['endpoint'],design_freeze_sha256=c.FREEZE_SHA,
                    execution_sources_sha256=ctx['sources_sha256'],request_ids_sha256=c.object_sha([r['request_id'] for r in ctx['requests']]),
                    request_id=ctx['requests'][1]['request_id'])
            return h
        self.mocks[5].side_effect=factory;self.execute()

    def test_closed_real_lease_revokes_authority(self):
        self.execute()
        with self.assertRaisesRegex(PermissionError,'Inactive global'):
            self.handles[0].authority._fast()

    def test_second_start_same_process_rejected(self):
        self.execute()
        with self.assertRaisesRegex(PermissionError,'fresh process'):
            c.execute(self.release_path,self.release_sha,11032)
        self.assertFalse((self.root/'start_dev/start-11032').exists())

    def test_release_hash_rejected_by_real_verifier(self):
        self.patches[0].stop()
        with self.assertRaisesRegex(PermissionError,'Untrusted START DEV release'):
            c.verify_release(self.release_path,'0'*64)

    def test_live_storage_observations_and_probe_are_required(self):
        self.patches[4].stop();directory=self.root/'probe';directory.mkdir()
        with patch.object(c.storage,'_observe',return_value={'effective_available_bytes':10000}) as observe, \
             patch.object(c.storage,'_probe',return_value={'file_fsync':True}) as probe:
            result=c.storage_check(directory,self.release['storage_budget'])
        self.assertEqual(observe.call_count,2);probe.assert_called_once_with(directory)
        self.assertEqual(result['budget'],self.release['storage_budget'])
        with patch.object(c.storage,'_observe',return_value={'effective_available_bytes':1}), \
             patch.object(c.storage,'_probe') as probe:
            with self.assertRaisesRegex(PermissionError,'Insufficient DEV storage'):c.storage_check(directory,self.release['storage_budget'])
            probe.assert_not_called()


if __name__=='__main__':unittest.main()
