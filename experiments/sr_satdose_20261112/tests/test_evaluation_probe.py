"""CPU orchestration fixtures: GPU observations and loader construction are mocked.

These tests never load a real model, call CUDA, or establish GPU acceptance.
The actual loader's 496-tensor/policy-only boundaries have separate CPU tests.
"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from execution import evaluation_probe as p
import test_diagnostic_worker as worker_tests

w=p.worker
REAL_GPU_IDENTITY=p._gpu_identity


class EvaluationProbeTests(unittest.TestCase):
    def setUp(self):
        self.fixture=worker_tests.WorkerTests('test_complete_flow_no_generation_one_shadow_real_accounting')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.ctx=self.fixture.ctx;self.b=self.ctx['binding'];self.root=self.ctx['root']
        self.calls=0;self.handles=[];self.rng='1'*64
        for start in (11031,11032):
            source=dict(self.fixture.source,policy_sha256='a'*64,reference_sha256='b'*64)
            self.ctx['config']['start_bindings'][str(start)]=source
            self.ctx['prepared']['starts'][str(start)]=dict(expected_identity=dict(
                self.fixture.endpoint,start_seed=start,parent='b0'))
        parent=self.root/'parent';parent.mkdir();(parent/'config.json').write_text('{}')
        self.ctx['lock'].update(parent_path=str(parent),parent_files={'config.json':w.shadow.file_sha(parent/'config.json')},native_files={})
        (self.root/'lock.json').write_text(json.dumps(self.ctx['lock']))
        self.b['source_lock']=w._ref(self.root,self.root/'lock.json')
        self.ctx['protected']=[self.b['source_lock']]
        config=self.root/'config.json';config.write_text(json.dumps(self.ctx['config']))
        control,preflight,result=self.fixture.prefix_fixture()
        self.b['config']=w._ref(self.root,config)
        preflight['config_sha256']=self.b['config']['sha256']
        (control/'PREFLIGHT.json').write_text(json.dumps(preflight))
        for start,analysis in result['prefix_analyses'].items():
            analysis['config_sha256']=self.b['config']['sha256']
            (control/f'PREFIX_{start}_ANALYSIS.json').write_text(json.dumps(analysis))
        result['preflight_sha256']=w.diagnostic_inputs.core.object_sha(preflight)
        (control/'RESULT.json').write_text(json.dumps(result))
        modules={'evaluation_probe':p,'evaluation_loader':p.loader,
                 'training_state':p.loader.state,'training_transaction':p.loader.transactions}
        self.b['code']={name:dict(path=str(Path(mod.__file__).resolve()),sha256=w.shadow.file_sha(mod.__file__))
                        for name,mod in modules.items()}
        sources={name:ref for name,ref in self.b['code'].items() if name!='evaluation_probe'}
        self.b.update(evaluation_loader_slots=p.expected_slots(),max_evaluation_loader_attempts=2,
            evaluation_execution_sources=sources,expected_evaluation_execution_sources_sha256=p.loader.object_sha(sources))
        self.ops=SimpleNamespace(capture_rng=lambda:self.rng,state_hash=lambda value:value,
                                restore_rng=self.forbidden)
        self.gpu=dict(visible_gpu_count=2,arithmetic_device=0,synthetic_cpu_fixture=True,
                      devices=[{'index':i,'name':'SYNTHETIC CPU FIXTURE'} for i in (0,1)])
        for item in (patch.object(p.loader.fork_loader,'load_native_ops',return_value=self.ops),
                     patch.object(p.loader,'build_evaluation_loader',side_effect=self.builder),
                     patch.object(p,'_gpu_identity',return_value=self.gpu),
                     patch.object(p,'_inventory',side_effect=self.inventory),
                     patch.object(torch.optim,'Adam',side_effect=self.forbidden),
                     patch.object(torch.optim,'AdamW',side_effect=self.forbidden),
                     patch.object(torch.nn.Module,'__call__',side_effect=self.forbidden)):
            item.start();self.addCleanup(item.stop)

    @staticmethod
    def forbidden(*args,**kwargs):raise AssertionError('forward/generation/optimizer/RNG restore forbidden')

    def builder(self,registry,lock,*,start_seed,arm,step,output_dir,execution_sources,expected_execution_sources_sha256):
        self.calls+=1
        self.assertEqual((arm,step),('START',0));self.assertEqual(list(output_dir.iterdir()),[])
        self.assertEqual(execution_sources,self.b['evaluation_execution_sources'])
        self.assertEqual(expected_execution_sources_sha256,self.b['expected_evaluation_execution_sources_sha256'])
        expected=self.ctx['prepared']['starts'][str(start_seed)]['expected_identity']
        audit={k:v for k,v in expected.items() if k!='parent'}
        audit.update(model_calls=0,optimizer_constructed=False,optimizer_steps=0,training_rng_restored=False,
            permission='LOAD_AUDIT_ONLY_NO_GENERATION_NO_MAIN_RELEASE')
        source=self.ctx['config']['start_bindings'][str(start_seed)]
        files={source['original_state_path']:source['state_sha256']}
        files.update({str(Path(self.ctx['lock']['parent_path'])/name):sha for name,sha in self.ctx['lock']['parent_files'].items()})
        receipt=dict(schema='sr-satdose-evaluation-loader-v1',
            selection={k:expected[k] for k in ('experiment_id','start_seed','arm','step')},
            original_files=files,evaluation_sources_sha256=expected_execution_sources_sha256,
            design_freeze_sha256=self.b['design_freeze']['sha256'],source_lock_sha256=self.b['source_lock']['sha256'],
            caller_rng_sha256=self.rng,gpu_acceptance=False,synthetic_cpu_fixture=True)
        w.shadow.write_once(output_dir/'EVALUATION_LOAD.json',receipt)
        handle=SimpleNamespace(receipt=receipt,audit=lambda:copy.deepcopy(audit),closed=0,
                               generate=self.forbidden,forward=self.forbidden)
        def close():handle.closed+=1;return dict(audit,closed=True)
        handle.close=close;self.handles.append(handle)
        return handle

    def inventory(self,handle,ops,expected):
        return dict(trainable_tensors=496,reference_tensors=496,reference_dtype='torch.float32',policy_sha256=expected['policy_sha256'],reference_sha256=expected['reference_sha256'],
            dtype='torch.float32',device='cuda:0',optimizer_created=False,scheduler_created=False,
            measured_forward_sequences=0,measured_backward_calls=0,synthetic_cpu_fixture=True,
            tensors=[dict(name=f'lora_{i}',shape=[1],dtype='torch.float32',device='cuda:0') for i in range(496)])

    def run_probe(self,start=11031):return p._execute(self.ctx,start,self.fixture.process)
    def slot(self,start=11031):return self.root/p.expected_slots()[str(start)]

    def test_two_fixed_start_audits_no_work_and_no_trained_claim(self):
        for start in (11031,11032):
            result=self.run_probe(start)
            self.assertEqual(result['tested_scope'],'START_R24_ONLY')
            self.assertFalse(result['trained_checkpoint_gpu_verified'])
            self.assertEqual(result['permission'],'NO_TRAINING')
            self.assertEqual(result['model_constructions'],1)
            self.assertTrue(all(result[key]==0 for key in ('forward_sequences','new_generations','backward_calls','optimizer_steps','scheduler_steps','parameter_updates')))
            w._no_holds(self.root,self.b,binding_sha=self.ctx['binding_sha'])
        self.assertEqual(self.calls,2);self.assertEqual([h.closed for h in self.handles],[1,1])
        with self.assertRaises(FileExistsError):self.run_probe()
        self.assertEqual(self.calls,2)

    def test_constructor_unknown_persists_hold_and_blocks_other_gpu_workers(self):
        with patch.object(p.loader,'build_evaluation_loader',side_effect=RuntimeError('unknown construction')) as build:
            with self.assertRaisesRegex(RuntimeError,'unknown construction'):self.run_probe()
            self.assertEqual(build.call_count,1)
        self.assertTrue((self.slot()/'HOLD.json').is_file())
        with self.assertRaisesRegex(PermissionError,'HOLD/UNKNOWN'):self.run_probe(11032)
        with self.assertRaisesRegex(PermissionError,'HOLD/UNKNOWN'):self.fixture.run_trial()
        self.assertFalse(self.fixture.handles);self.assertEqual(self.calls,0)

    def test_failed_observation_closes_once_and_never_completes(self):
        with patch.object(p,'_inventory',side_effect=PermissionError('dirty gradients')):
            with self.assertRaisesRegex(PermissionError,'dirty gradients'):self.run_probe()
        self.assertEqual(self.handles[0].closed,1)
        self.assertTrue((self.slot()/'HOLD.json').exists());self.assertFalse((self.slot()/'COMPLETE.json').exists())

    def test_close_failure_is_not_retried(self):
        original=self.builder
        def builder(*args,**kwargs):
            handle=original(*args,**kwargs)
            def close():handle.closed+=1;raise RuntimeError('close failed')
            handle.close=close;return handle
        with patch.object(p.loader,'build_evaluation_loader',side_effect=builder):
            with self.assertRaisesRegex(RuntimeError,'close failed'):self.run_probe()
        self.assertEqual(self.handles[0].closed,1);self.assertTrue((self.slot()/'HOLD.json').exists())

    def test_caller_rng_mutation_is_hold_without_restoring_training_rng(self):
        def inventory(*args):self.rng='2'*64;return self.inventory(*args)
        with patch.object(p,'_inventory',side_effect=inventory):
            with self.assertRaisesRegex(PermissionError,'RNG'):self.run_probe()
        self.assertTrue((self.slot()/'HOLD.json').exists());self.assertEqual(self.handles[0].closed,1)

    def test_source_change_is_hold_and_preserved_for_audit(self):
        source=Path(self.fixture.source['original_state_path'])
        def inventory(*args):source.write_text('external mutation');return self.inventory(*args)
        with patch.object(p,'_inventory',side_effect=inventory):
            with self.assertRaisesRegex(PermissionError,'source|files changed'):self.run_probe()
        self.assertEqual(source.read_text(),'external mutation');self.assertTrue((self.slot()/'HOLD.json').exists())

    def test_unbound_code_and_incomplete_prefix_fail_before_construction(self):
        self.b['code']['evaluation_probe']['sha256']='0'*64
        with self.assertRaisesRegex(PermissionError,'Unbound'):self.run_probe()
        self.assertEqual(self.calls,0);self.assertFalse(self.slot().exists())
        self.b['code']['evaluation_probe']['sha256']=w.shadow.file_sha(p.__file__)
        (self.root/'prefix_control/RESULT.json').unlink()
        with self.assertRaisesRegex(PermissionError,'PREFIX UNKNOWN'):self.run_probe()
        self.assertEqual(self.calls,0)

    def test_complete_evidence_tamper_blocks_cross_worker(self):
        self.run_probe();(self.slot()/'SOURCES.json').write_text('{}')
        with self.assertRaisesRegex(PermissionError,'evidence changed'):self.fixture.run_trial()
        self.assertFalse(self.fixture.handles)

    def test_global_lock_and_unknown_slot_block_before_model(self):
        with w._worker_lock(self.root):
            with self.assertRaisesRegex(PermissionError,'global lock'):self.run_probe()
        self.slot().mkdir(parents=True)
        with self.assertRaisesRegex(PermissionError,'HOLD/UNKNOWN'):self.run_probe(11032)
        self.assertEqual(self.calls,0)

    def test_public_api_has_no_injection_and_requires_gpu(self):
        with self.assertRaises(TypeError):p.run_probe(start_seed=11031,expected_binding_sha256='x',handle=object())
        # Undo only the synthetic observation helper for the actual public CUDA guard.
        with patch.object(p,'_gpu_identity',side_effect=REAL_GPU_IDENTITY),patch.object(torch.cuda,'is_available',return_value=False),patch.object(p,'_USED',False),patch.object(p,'_preflight') as preflight:
            with self.assertRaisesRegex(PermissionError,'CUDA'):p.run_probe(start_seed=11031,expected_binding_sha256='x')
            preflight.assert_not_called()
        self.assertEqual(self.calls,0)

    def test_preflight_reuses_full_dry_gate_and_requires_complete_prefix(self):
        with patch.object(w,'_preflight',return_value=self.ctx) as preflight:
            self.assertIs(p._preflight(11031,'c'*64),self.ctx)
            preflight.assert_called_once_with('c'*64,11031,0)
        with patch.object(w,'_preflight',side_effect=PermissionError('both full DRY required')):
            with self.assertRaisesRegex(PermissionError,'full DRY'):p._preflight(11031,'c'*64)
        self.assertEqual(self.calls,0)


if __name__=='__main__':unittest.main()
