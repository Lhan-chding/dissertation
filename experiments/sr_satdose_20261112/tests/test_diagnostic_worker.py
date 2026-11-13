"""CPU mocked control flow only. No native model, CUDA, generation or GPU PASS."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import diagnostic_worker as w


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.model = torch.nn.Linear(1,1,bias=False)
        self.original = self.model.weight.detach().clone()
        self.endpoint = dict(experiment_id=w.shadow.EXPERIMENT,start_seed=11031,arm='START',step=0,
                             policy_sha256='a'*64,reference_sha256='b'*64)
        self.b = dict(shadow_slots=w.shadow.expected_slots(),design_freeze=dict(sha256='f'*64),
                      backend_equivalence_slots={str(s):f'backend_equivalence/start-{s}' for s in (11031,11032)},
                      max_backend_equivalence_attempts=2,
                      source_lock=dict(path='lock.json',sha256=''))
        (self.root/'lock.json').write_text('{}')
        self.b['source_lock']['sha256'] = w.shadow.file_sha(self.root/'lock.json')
        state = self.root/'state.pt';state.write_bytes(b'original checkpoint fixture')
        self.source = dict(original_state_path=str(state),state_sha256=w.shadow.file_sha(state))
        self.rows=[]
        dry=self.root/'original_dry';dry.mkdir()
        for i in range(256):
            rid=str(i);name=hashlib.sha256(rid.encode()).hexdigest()+'.CAPTURE.json'
            row=dict(request_id=rid,group_id=str(i//8),tokens=[1,2],sampler_logprobs=[-.1,-.2],
                     prompt_tensor_hash='p',model_input={'prompt':'fixture'},seed=i)
            cap=dry/name;cap.write_text(json.dumps(row))
            cap.with_name(name.replace('.CAPTURE.json','.BEGIN.json')).write_text(json.dumps(dict(
                request_id=rid,endpoint=self.endpoint,design_freeze_sha256='f'*64)))
            row.update(native_capture_path=str(cap),native_capture_sha256=w.shadow.file_sha(cap))
            self.rows.append(row)
        self.before=copy.deepcopy(self.rows)
        prepared=dict(starts={'11031':dict(expected_identity=dict(self.endpoint,parent='b0'),
            probe=dict(rows=self.rows[:128],half_group_ids=[[str(i) for i in range(8)],[str(i) for i in range(8,16)]]),
            witness=dict(rows=self.rows[128:],half_group_ids=[[str(i) for i in range(16,24)],[str(i) for i in range(24,32)]]),
            production_plan={'scope':'DRY'})})
        self.ctx=dict(root=self.root,binding=self.b,config={'start_bindings':{'11031':self.source}},
            registry=object(),prepared=prepared,protected=[self.b['source_lock']],binding_sha='c'*64,
            lock={'native_src':'/not-loaded','legacy_root':'/not-loaded'})
        self.handles=[];self.shadow_calls=0
        self.process=dict(host='same-node',pid=1,start_ticks='100',boot_id='boot')
        self.patches=[patch.object(w.native_runtime,'build_runtime',side_effect=self.builder),
            patch.object(w.production,'load_native_kernels',return_value=object()),
            patch.object(w.diagnostics,'load_native_math',return_value=object()),
            patch.object(w.production,'capture_update',side_effect=self.capture),
            patch.object(w.diagnostics,'accumulate_component',side_effect=self.component),
            patch.object(w.shadow,'one_trial',side_effect=self.shadow)]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def builder(self, registry, source_lock, *, start_seed, output_dir):
        self.assertIs(registry,self.ctx['registry'])
        self.assertNotEqual(output_dir,self.root/'original_dry')
        self.assertEqual(list(output_dir.iterdir()),[])
        runtime=SimpleNamespace(model=self.model,_satdose_accounting=[])
        # Any accidental generation call fails the fixture instead of silently succeeding.
        def forbidden(*args,**kwargs):raise AssertionError('generation is forbidden')
        runtime.generate=runtime._generate_native_group=forbidden
        handle=SimpleNamespace(runtime=runtime,output=output_dir,storage_forwards=[],closed=False,
            fork=SimpleNamespace(optimizer=None,scheduler=None,reference={},receipt=dict(
                source_state_path=self.source['original_state_path'],source_state_sha256=self.source['state_sha256'],
                optimizer_created=False,model_calls=0,optimizer_steps=0)))
        handle.endpoint=lambda:copy.deepcopy(self.endpoint)
        def forward(row):
            self.assertTrue(Path(row['native_capture_path']).is_relative_to(output_dir))
            for purpose in ('training_reference','training_policy'):
                event=dict(kind='extra_forward_sequences',count=1,metadata=dict(purpose=purpose))
                runtime._satdose_accounting.append(event)
                with (output_dir/'accounting.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            handle.storage_forwards.append(dict(row_id=row['request_id'],stats={'synthetic_only':True}))
        handle.forward_original=forward
        handle.verify_storage_after_backward=lambda:None
        def close():
            handle.closed=True
            self.assertTrue(torch.equal(self.model.weight,self.original))
            return dict(closed=True,optimizer_steps=0,parameter_updates=0)
        handle.close=close
        self.handles.append(handle)
        return handle

    def capture(self, model, rows, forward, reference, **kwargs):
        for row in rows:forward(row)
        return dict(preclip_gradients={'weight':torch.ones(1)},postclip_gradients={'weight':torch.ones(1)})

    def component(self, model, rows, forward, native, component):
        for row in rows:forward(row)
        return dict(gradients={'weight':torch.zeros(1) if component=='KL' else torch.ones(1)},
                    component=component,synthetic_only=True)

    def shadow(self, policy, gradients, endpoint, *, start_seed, trial, **kwargs):
        self.shadow_calls+=1
        self.assertIsNot(policy['weight'],self.model.weight)
        self.assertNotEqual(policy['weight'].data_ptr(),self.model.weight.data_ptr())
        target=self.root/w.shadow.CLAIMS_NAME/f'start-{start_seed}'/f'trial-{trial}'
        target.mkdir(parents=True)
        w._save_tensor(target/'RESULT.pt',{'delta':{'weight':torch.ones(1)}})
        w.shadow.write_once(target/'COMPLETE.json',dict(result_sha256=w.shadow.file_sha(target/'RESULT.pt')))

    def run_trial(self,trial=0,process=None):
        return w._execute(self.ctx,11031,trial,process or self.process)

    def slot(self,trial=0):
        return self.root/'shadow_inputs/start-11031'/f'trial-{trial}'

    def test_complete_flow_no_generation_one_shadow_real_accounting(self):
        result=self.run_trial()
        self.assertEqual(result['status'],'DIAGNOSTIC_SLOT_COMPLETE_NOT_GPU_ADMISSION')
        self.assertEqual(self.shadow_calls,1)
        self.assertTrue(self.handles[0].closed)
        self.assertEqual(self.rows,self.before)
        accounting=w.shadow.read_json(self.slot()/'ACCOUNTING.json')
        self.assertEqual(accounting['production']['measured_forward_sequences'],
                         {'training_policy':128,'training_reference':128})
        self.assertEqual(len(accounting),13)
        self.assertEqual(len(w.shadow.read_json(self.slot()/'audit/RELOCATION.json')['files']),512)
        self.assertEqual(result['remaining_checks'],w.PENDING)

    def test_new_process_same_host_trial1_rebuilds_and_recaptures(self):
        self.run_trial()
        with self.assertRaisesRegex(PermissionError,'new Python process'):self.run_trial(1)
        self.assertFalse(self.slot(1).exists())
        result=self.run_trial(1,dict(self.process,pid=2,start_ticks='200'))
        self.assertEqual(len(self.handles),2)
        self.assertEqual(self.shadow_calls,2)
        self.assertTrue(all(h.closed for h in self.handles))
        self.assertEqual(len(w.shadow.read_json(self.slot(1)/'ACCOUNTING.json')),1)
        self.assertEqual(result['status'],'DIAGNOSTIC_SLOT_COMPLETE_NOT_GPU_ADMISSION')

    def test_pre_shadow_replay_hold_preserved_closes_and_blocks_dependencies(self):
        with patch.object(w.diagnostics,'compare_replay',return_value={'bit_exact':False}):
            with self.assertRaisesRegex(PermissionError,'not bit-exact'):self.run_trial()
        self.assertEqual(self.shadow_calls,0)
        self.assertTrue(self.handles[0].closed)
        self.assertTrue((self.slot()/'HOLD.json').exists())
        with self.assertRaises((FileNotFoundError,PermissionError)):self.run_trial(1,dict(self.process,pid=2))
        with self.assertRaisesRegex(PermissionError,'HOLD'):w._no_holds(self.root,self.b,binding_sha='c'*64)

    def test_post_shadow_direction_hold_blocks_retry(self):
        with patch.object(w.diagnostics,'direction_summary',return_value={'direction_checks_satisfied':False}):
            with self.assertRaisesRegex(PermissionError,'heldout'):self.run_trial()
        self.assertEqual(self.shadow_calls,1)
        self.assertTrue((self.slot()/'DIRECTION.json').exists())
        self.assertTrue((self.slot()/'HOLD.json').exists())
        self.assertFalse((self.slot()/'WORKER_COMPLETE.json').exists())
        self.assertTrue(self.handles[0].closed)
        with self.assertRaises(PermissionError):self.run_trial()
        self.assertEqual(self.shadow_calls,1)

    def test_accounting_mismatch_fails_before_shadow_and_finally_closes(self):
        original=self.capture
        def mismatch(*args,**kwargs):
            result=original(*args,**kwargs)
            self.handles[-1].runtime._satdose_accounting.pop()
            return result
        with patch.object(w.production,'capture_update',side_effect=mismatch):
            with self.assertRaisesRegex(PermissionError,'accounting differ'):self.run_trial()
        self.assertEqual(self.shadow_calls,0)
        self.assertTrue(self.handles[0].closed)

    def test_shadow_failure_no_second_call_and_close(self):
        def fail(*args,**kwargs):
            self.shadow_calls+=1
            raise RuntimeError('unknown optimizer outcome')
        with patch.object(w.shadow,'one_trial',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'unknown optimizer'):self.run_trial()
        self.assertEqual(self.shadow_calls,1)
        self.assertTrue(self.handles[0].closed)
        self.assertTrue((self.slot()/'HOLD.json').exists())

    def test_tampered_original_begin_rejected_without_shadow(self):
        first=Path(self.rows[0]['native_capture_path'])
        begin=first.with_name(first.name.replace('.CAPTURE.json','.BEGIN.json'))
        data=json.loads(begin.read_text());data['endpoint']['step']=1;begin.write_text(json.dumps(data))
        with self.assertRaisesRegex(PermissionError,'BEGIN identity'):self.run_trial()
        self.assertEqual(self.shadow_calls,0)
        self.assertTrue(self.handles[0].closed)

    def test_public_entry_cpu_denied_and_no_dependency_injection(self):
        with patch.object(w,'_USED',False),patch.object(w,'_preflight',return_value=self.ctx),\
             patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(PermissionError,'requires two CUDA'):
                w.run_worker(start_seed=11031,trial=0,expected_binding_sha256='c'*64)
        self.assertFalse(self.handles)
        with self.assertRaises(TypeError):
            w.run_worker(start_seed=11031,trial=0,expected_binding_sha256='c'*64,handle=object())

    def test_public_process_guard_and_preflight_fail_closed(self):
        with patch.object(w,'_USED',True),patch.object(w,'_preflight') as preflight:
            with self.assertRaisesRegex(PermissionError,'fresh Python process'):
                w.run_worker(start_seed=11031,trial=0,expected_binding_sha256='c'*64)
            preflight.assert_not_called()
        with patch.object(w,'_USED',False),patch.object(w,'_preflight',side_effect=PermissionError('incomplete DRY')):
            with self.assertRaisesRegex(PermissionError,'incomplete DRY'):
                w.run_worker(start_seed=11031,trial=0,expected_binding_sha256='c'*64)
        self.assertFalse(self.handles)

    def test_crash_left_empty_or_claimed_slot_blocks_other_start_before_build(self):
        for kind in ('empty_worker', 'worker_claim', 'empty_shadow', 'shadow_claim'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve()
                path=(root/'shadow_inputs/start-11031/trial-0' if 'worker' in kind else
                      root/'shadow_claims/start-11031/trial-0')
                path.mkdir(parents=True)
                if kind.endswith('claim'):
                    (path/('WORKER_CLAIM.json' if 'worker' in kind else 'CLAIM.json')).write_text('{}')
                ctx=dict(self.ctx,root=root)
                with self.assertRaisesRegex(PermissionError,'UNKNOWN'):
                    w._execute(ctx,11032,0,self.process)
                self.assertFalse(self.handles)

    def test_nonblocking_global_lock_prevents_cross_slot_race(self):
        with w._worker_lock(self.root):
            with self.assertRaisesRegex(PermissionError,'global lock'):
                self.run_trial()
        self.assertFalse(self.handles)
        self.assertFalse(self.slot().exists())

    def test_runtime_changed_backend_or_fork_loader_rejected_against_frozen_copy(self):
        for filename in ('backend.py','fork_loader.py'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as temp:
                runtime=Path(temp)
                for name in ('native_runtime.py','backend.py','fork_loader.py'):
                    (runtime/name).write_text('# original frozen bytes\n')
                sources={p.name:w.shadow.file_sha(p) for p in runtime.glob('*.py')}
                freeze=dict(runtime_sources=copy.deepcopy(sources),
                            artifacts={name:dict(sha256=sha) for name,sha in sources.items()})
                (runtime/filename).write_text('# modified actual runtime, frozen copy unchanged\n')
                with patch.object(w.native_runtime,'__file__',str(runtime/'native_runtime.py')):
                    with self.assertRaisesRegex(PermissionError,'runtime source inventory/SHA'):
                        w._verify_runtime_sources(freeze)
                self.assertEqual(freeze['runtime_sources'],sources)
                self.assertFalse(self.handles)

    def test_imported_runtime_module_cannot_be_shadowed(self):
        runtime=Path(w.native_runtime.__file__).resolve().parent
        sources={p.name:w.shadow.file_sha(p) for p in runtime.glob('*.py') if p.is_file()}
        freeze=dict(runtime_sources=sources,artifacts={name:dict(sha256=sha) for name,sha in sources.items()})
        fake=SimpleNamespace(__file__='/different/backend.py')
        with patch.dict(sys.modules,{'runtime.backend':fake}):
            with self.assertRaisesRegex(PermissionError,'Shadowed imported runtime'):
                w._verify_runtime_sources(freeze)


    def test_equivalence_failed_or_unknown_blocks_diagnostic_model_constructor(self):
        for status in ('empty','claim','hold','invalid_complete'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve()
                path=root/'backend_equivalence/start-11031';path.mkdir(parents=True)
                if status=='claim':(path/'CLAIM.json').write_text('{}')
                if status=='hold':(path/'HOLD.json').write_text('{}')
                if status=='invalid_complete':(path/'COMPLETE.json').write_text('{}')
                with self.assertRaises(PermissionError):
                    w._execute(dict(self.ctx,root=root),11032,0,self.process)
                self.assertFalse(self.handles)

    def test_missing_equivalence_budget_cannot_silently_bypass_gate(self):
        binding=copy.deepcopy(self.b);del binding['backend_equivalence_slots']
        with self.assertRaisesRegex(PermissionError,'fixed backend-equivalence'):
            w._execute(dict(self.ctx,binding=binding),11031,0,self.process)
        self.assertFalse(self.handles)


    def prefix_fixture(self, *, complete=True):
        self.b['config'] = dict(path='config.json',sha256='e'*64)
        control=self.root/'prefix_control';control.mkdir()
        protocol={'fixture':'独立前置澄清'}
        preflight=dict(schema='sr-satdose-prefix-preflight-v1',status='READY_FOR_PREFIX_CALIB_ONLY',
            permission='NO_TRAINING',model_calls=0,starts=[11031,11032],responses_per_start=48,
            design_freeze_sha256='f'*64,config_sha256='e'*64,protocol_clarification=protocol)
        (control/'PREFLIGHT.json').write_text(json.dumps(preflight))
        if not complete:return control,preflight,None
        analyses={}
        for start in (11031,11032):
            run=self.root/'runs'/f'PREFIX_CALIB_{start}';run.mkdir(parents=True)
            raw=run/'PREFIX_RAW.jsonl';raw.write_text('synthetic completed raw fixture\n')
            receipt=dict(status='CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION',scope='PREFIX_CALIB',
                start_seed=start,responses=48,optimizer_steps=0,parameter_updates=0,optimizer_created=False,
                scheduler_created=False,design_freeze_sha256='f'*64,source_lock_sha256=self.b['source_lock']['sha256'],
                raw=dict(path='PREFIX_RAW.jsonl',sha256=w.shadow.file_sha(raw)))
            receipt_path=run/'CALIBRATION_COMPLETE.json';receipt_path.write_text(json.dumps(receipt))
            checks=dict(all_views_known_at_least_14=True,stable_correct_at_least_12=True,
                        stable_minus_revtie_at_least_4=True)
            analysis=dict(schema='sr-satdose-prefix-calibration-analysis-v1',start_seed=start,
                status='PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED',permission='NO_TRAINING',
                candidate_gate_passed=True,candidate_decision_rule='strict_complete_gate_OR_sensitivity_complete_gate',
                design_freeze_sha256='f'*64,config_sha256='e'*64,protocol_clarification=protocol,
                readouts={name:dict(checks=checks,passed=True) for name in ('strict','sensitivity')},
                qualifying_readouts=['strict','sensitivity'],raw=w._ref(self.root,raw),
                completion_receipt=w._ref(self.root,receipt_path))
            (control/f'PREFIX_{start}_ANALYSIS.json').write_text(json.dumps(analysis))
            analyses[str(start)]=analysis
        result=dict(status='PREFIX_METRICS_COMPLETE_GPU_CHECK_STILL_REQUIRED',permission='NO_TRAINING',
            responses=96,preflight_sha256=w.diagnostic_inputs.core.object_sha(preflight),prefix_analyses=analyses)
        (control/'RESULT.json').write_text(json.dumps(result))
        return control,preflight,result

    def test_prefix_empty_or_claim_only_blocks_gpu_before_model(self):
        control=self.root/'prefix_control';control.mkdir()
        with self.assertRaisesRegex(PermissionError,'PREFIX UNKNOWN'):
            self.run_trial()
        self.assertFalse(self.handles)
        control.rmdir()
        self.prefix_fixture(complete=False)
        with self.assertRaisesRegex(PermissionError,'PREFIX UNKNOWN'):
            self.run_trial()
        self.assertFalse(self.handles)

    def test_prefix_current_exemption_is_internal_and_never_ignores_hold(self):
        control,_,_=self.prefix_fixture(complete=False)
        w._no_holds(self.root,self.b,binding_sha='c'*64,current_prefix=True)
        (control/'RESULT.json').write_text(json.dumps({'status':'HOLD_PREFIX_CONTROL'}))
        with self.assertRaisesRegex(PermissionError,'PREFIX HOLD'):
            w._no_holds(self.root,self.b,binding_sha='c'*64,current_prefix=True)
        with self.assertRaises(TypeError):
            w.run_worker(start_seed=11031,trial=0,expected_binding_sha256='c'*64,current_prefix=True)
        self.assertFalse(self.handles)

    def test_prefix_complete_chain_allows_gpu_and_unicode_preflight_hash(self):
        self.prefix_fixture()
        result=self.run_trial()
        self.assertEqual(result['status'],'DIAGNOSTIC_SLOT_COMPLETE_NOT_GPU_ADMISSION')
        self.assertEqual(len(self.handles),1)

    def test_prefix_raw_tamper_blocks_gpu_constructor(self):
        self.prefix_fixture()
        (self.root/'runs/PREFIX_CALIB_11032/PREFIX_RAW.jsonl').write_text('changed raw')
        with self.assertRaisesRegex(PermissionError,'evidence changed'):
            self.run_trial()
        self.assertFalse(self.handles)

    def test_prefix_preflight_or_analysis_chain_tamper_blocks_gpu(self):
        control,preflight,_=self.prefix_fixture()
        preflight['protocol_clarification']['fixture']='changed'
        (control/'PREFLIGHT.json').write_text(json.dumps(preflight))
        with self.assertRaisesRegex(PermissionError,'PREFIX HOLD/incomplete RESULT'):
            self.run_trial()
        self.assertFalse(self.handles)

    def test_prefix_one_start_failed_even_with_result_complete_blocks_gpu(self):
        control,_,result=self.prefix_fixture()
        analysis=result['prefix_analyses']['11032'];analysis['status']='HOLD_PREFIX_MEASUREMENT'
        (control/'PREFIX_11032_ANALYSIS.json').write_text(json.dumps(analysis))
        (control/'RESULT.json').write_text(json.dumps(result))
        with self.assertRaisesRegex(PermissionError,'Both complete PREFIX starts must PASS'):
            self.run_trial()
        self.assertFalse(self.handles)

    def test_prefix_incomplete_receipt_rejected_before_raw_bytes_are_opened(self):
        control,_,result=self.prefix_fixture()
        receipt_path=self.root/'runs/PREFIX_CALIB_11031/CALIBRATION_COMPLETE.json'
        receipt=json.loads(receipt_path.read_text());receipt['responses']=47
        receipt_path.write_text(json.dumps(receipt))
        analysis=result['prefix_analyses']['11031'];analysis['completion_receipt']=w._ref(self.root,receipt_path)
        (control/'PREFIX_11031_ANALYSIS.json').write_text(json.dumps(analysis))
        (control/'RESULT.json').write_text(json.dumps(result))
        original=w._bound
        def bound(root,ref):
            self.assertFalse(ref['path'].endswith('PREFIX_RAW.jsonl'),'partial raw must not be read')
            return original(root,ref)
        with patch.object(w,'_bound',side_effect=bound):
            with self.assertRaisesRegex(PermissionError,'completion is incomplete'):
                self.run_trial()
        self.assertFalse(self.handles)

    def test_allzero_unknown_or_hold_blocks_shadow_constructor(self):
        self.b.update(allzero_slots={str(s):f'allzero_check/start-{s}' for s in (11031,11032)},max_allzero_attempts=2)
        path=self.root/'allzero_check/start-11031';path.mkdir(parents=True)
        with self.assertRaisesRegex(PermissionError,'Allzero HOLD/UNKNOWN'):self.run_trial()
        (path/'HOLD.json').write_text('{"synthetic_cpu_only":true}')
        with self.assertRaisesRegex(PermissionError,'Allzero HOLD/UNKNOWN'):self.run_trial()
        self.assertFalse(self.handles);self.assertEqual(self.shadow_calls,0)

    def test_allzero_occupied_directory_without_binding_never_bypasses_gate(self):
        (self.root/'allzero_check/start-11031').mkdir(parents=True)
        with self.assertRaisesRegex(PermissionError,'without its fixed technical binding'):self.run_trial()
        self.assertFalse(self.handles);self.assertEqual(self.shadow_calls,0)

    def test_allzero_complete_control_fixture_requires_all_bound_evidence(self):
        # Control-schema fixture only: bytes are NOT real GPU tensor evidence.
        self.b.update(allzero_slots={str(s):f'allzero_check/start-{s}' for s in (11031,11032)},max_allzero_attempts=2)
        path=self.root/'allzero_check/start-11031';path.mkdir(parents=True)
        comparison=dict(pg_exact_zero=True,kl_retained=True,recomposition_relative_l2=0.)
        names=('ORIGINAL_INPUTS.json','PG_ZERO.pt','KL.pt','TOTAL.pt','COMPARISON.json','ACCOUNTING.json')
        for name in names:(path/name).write_text(json.dumps(comparison) if name=='COMPARISON.json' else 'synthetic CPU control fixture')
        complete=dict(schema='sr-satdose-allzero-check-v1',status='ALLZERO_CAPTURE_COMPLETE_NOT_GPU_ADMISSION',
            start_seed=11031,technical_binding_sha256='c'*64,permission='NO_TRAINING',close=dict(closed=True),
            optimizer_steps=0,scheduler_steps=0,new_generations=0,parameter_updates=0,explicit_diagnostic_advantage=0.,
            original_scores_unchanged=True,sequence_denominator=128,trainable_tensors=496,
            measured_forward_sequences=dict(training_policy=384,training_reference=384),comparison=comparison,
            evidence={name:w._ref(self.root,path/name) for name in names})
        (path/'COMPLETE.json').write_text(json.dumps(complete))
        w._no_holds(self.root,self.b,binding_sha='c'*64)
        with self.assertRaisesRegex(PermissionError,'identity/counts'):
            w._no_holds(self.root,self.b,binding_sha='d'*64)
        (path/'KL.pt').write_text('tampered')
        with self.assertRaisesRegex(PermissionError,'evidence changed'):self.run_trial()
        self.assertFalse(self.handles);self.assertEqual(self.shadow_calls,0)

    def test_unbound_evaluation_slot_blocks_shared_worker_before_build(self):
        path=self.root/'evaluation_loader_checks/start-11031';path.mkdir(parents=True)
        with self.assertRaisesRegex(PermissionError,'fixed technical binding'):self.run_trial()
        self.assertFalse(self.handles)

    def test_declared_evaluation_unknown_or_hold_blocks_shared_worker(self):
        sources={name:dict(path='/synthetic/'+name+'.py',sha256='0'*64)
                 for name in ('evaluation_loader','training_state','training_transaction')}
        self.b.update(evaluation_loader_slots={str(s):f'evaluation_loader_checks/start-{s}' for s in (11031,11032)},
            max_evaluation_loader_attempts=2,evaluation_execution_sources=sources,
            expected_evaluation_execution_sources_sha256=w._sha_object(sources),code=sources)
        path=self.root/'evaluation_loader_checks/start-11032';path.mkdir(parents=True)
        with self.assertRaisesRegex(PermissionError,'HOLD/UNKNOWN'):self.run_trial()
        (path/'HOLD.json').write_text('{}')
        with self.assertRaisesRegex(PermissionError,'HOLD/UNKNOWN'):self.run_trial()
        self.assertFalse(self.handles)



if __name__=='__main__':unittest.main()
