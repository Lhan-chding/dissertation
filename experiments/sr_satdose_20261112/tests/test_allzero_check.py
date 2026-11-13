"""Actual pinned native math on CPU; mocked device/storage telemetry is NOT GPU evidence."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import allzero_check as a
sys.path.pop(0)
spec=importlib.util.spec_from_file_location('_allzero_native_fixture',Path(__file__).with_name('test_production.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


@unittest.skipUnless((fixtures.NATIVE/'sr_f6/objective.py').exists() and
                    (fixtures.LEGACY/'adapter/training.py').exists(),'Pinned native sources required')
class AllzeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(fixtures.ProductionTests.doClassCleanups)
        fixtures.ProductionTests.setUpClass();cls.k=fixtures.ProductionTests.k

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.audit=self.root/'original_dry';self.audit.mkdir()
        factory=fixtures.ProductionTests('test_independent_production_equals_diagnostics_bit_exact')
        self.model,self.rows,self.native_forward,self.reference,self.identity,self.config,self.plan=factory.fixture()
        self.config['loss']=dict(kl_coefficient=.02)
        source=self.root/'original_R24.pt';source.write_bytes(b'SYNTHETIC_CPU_SOURCE_NOT_MODEL_WEIGHTS')
        self.config['start_bindings']['11031'].update(original_state_path=str(source),state_sha256=a.shadow.file_sha(source))
        self.endpoint={k:v for k,v in self.identity.items() if k!='parent'}
        self.runtime=SimpleNamespace(model=self.model,_satdose_accounting=[])
        def forbidden(*args,**kwargs):raise AssertionError('No generation/Adam allowed')
        self.runtime.generate=forbidden
        self.closed=False;self.hooks=[]
        self.handle=SimpleNamespace(runtime=self.runtime,output=self.audit,storage_forwards=[],
            fork=SimpleNamespace(reference=self.reference,optimizer=None,scheduler=None),
            endpoint=lambda:copy.deepcopy(self.endpoint),close=self.close,forward_original=self.forward,
            verify_storage_after_backward=self.verify_storage,
            storage=SimpleNamespace(gpu_peak_stats=lambda **kwargs:dict(synthetic_cpu_only=True)))
        for i,row in enumerate(self.rows):
            row.update(seed=i,prompt_tensor_hash='p'*64,model_input=dict(prompt='CPU fixture',image_file='image.png',image_sha256='i'*64))
            name=hashlib.sha256(row['request_id'].encode()).hexdigest()+'.CAPTURE.json'
            path=self.audit/name;path.write_text(json.dumps(row))
            path.with_name(name.replace('.CAPTURE.json','.BEGIN.json')).write_text(json.dumps(dict(
                request_id=row['request_id'],endpoint=self.endpoint,design_freeze_sha256='f'*64)))
            row.update(native_capture_path=str(path),native_capture_sha256=a.shadow.file_sha(path))
        self.before=copy.deepcopy(self.rows)

    def forward(self,row):
        # Real CPU autograd saved-tensor hooks; no CUDA/offload claim is made.
        for purpose in ('training_reference','training_policy'):
            event=dict(kind='extra_forward_sequences',count=1,metadata=dict(purpose=purpose))
            self.runtime._satdose_accounting.append(event)
            with (self.handle.output/'accounting.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        stats=dict(synthetic_cpu_only=True,packed=0,unpacked=0)
        def pack(tensor):stats['packed']+=1;return tensor.detach().clone()
        def unpack(tensor):stats['unpacked']+=1;return tensor
        entry=dict(row_id=row['request_id'],stats=stats,forward_returned=True)
        self.handle.storage_forwards.append(entry)
        with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
            return self.native_forward(row)

    def verify_storage(self):
        for entry in self.handle.storage_forwards:
            self.assertTrue(entry['stats']['synthetic_cpu_only'])
            self.assertGreater(entry['stats']['packed'],0);self.assertGreater(entry['stats']['unpacked'],0)

    def close(self):
        self.closed=True
        return dict(closed=True,optimizer_steps=0,parameter_updates=0)

    def capture(self,component,rows=None,kernels=None):
        return a._capture_component(self.handle,rows if rows is not None else self.rows,
                                    kernels or self.k,component,expected_tensors=1)

    def captures(self):return {name:self.capture(name) for name in a.COMPONENTS}

    def context(self):
        lock=self.root/'lock.json';lock.write_text('{}')
        prepared=self.root/'prepared.json';prepared.write_text('{"synthetic_cpu_only":true}')
        binding=dict(shadow_slots=a.shadow.expected_slots(),design_freeze=dict(sha256='f'*64),
            backend_equivalence_slots={str(s):f'backend_equivalence/start-{s}' for s in (11031,11032)},
            max_backend_equivalence_attempts=2,allzero_slots=a.expected_slots(),max_allzero_attempts=2,
            source_lock=a.worker._ref(self.root,lock),diagnostic_inputs=a.worker._ref(self.root,prepared))
        return dict(root=self.root,binding=binding,binding_sha='c'*64,registry=object(),config=self.config,
            protected=[binding['source_lock'],binding['diagnostic_inputs']],
            lock=dict(native_src=str(fixtures.NATIVE),legacy_root=str(fixtures.LEGACY)),
            prepared=dict(starts={'11031':dict(expected_identity=self.identity,production_plan=self.plan,
                                              probe=dict(rows=self.rows))}))

    def patches(self):
        capture=a._capture_component
        def build(*args,output_dir,**kwargs):self.handle.output=output_dir;return self.handle
        def cpu_capture(*args,**kwargs):return capture(*args,**kwargs,expected_tensors=1)
        return patch.object(a.native_runtime,'build_runtime',side_effect=build),patch.object(a,'_capture_component',side_effect=cpu_capture)

    def test_real_native_zero_override_keeps_original_rewards_and_kl(self):
        rng=self.k.state_hash(self.k.capture_rng());before=self.model.weight.detach().clone()
        with patch.object(torch.optim,'AdamW',side_effect=AssertionError('No Adam construction')):
            captures=self.captures();comparison=a.compare(captures)
        self.assertEqual(self.rows,self.before);self.assertTrue(torch.equal(before,self.model.weight))
        self.assertEqual(self.k.state_hash(self.k.capture_rng()),rng);self.assertIsNone(self.model.weight.grad)
        self.assertEqual({r['J'] for r in self.rows},{0,1})
        self.assertTrue(any(term['original_group_advantage']!=0 for term in captures['PG_ZERO']['native_terms']))
        self.assertTrue(comparison['pg_exact_zero']);self.assertTrue(comparison['kl_retained'])
        self.assertGreater(comparison['kl_gradient_l2'],0);self.assertGreater(comparison['weighted_kl_mean'],0)
        self.assertLessEqual(comparison['recomposition_relative_l2'],1e-5)
        for capture in captures.values():
            self.assertEqual(capture['forward_calls'],128);self.assertEqual(capture['backward_calls'],128)
            self.assertEqual(capture['sequence_denominator'],128);self.assertEqual(capture['optimizer_steps'],0)
        self.assertEqual(len(self.handle.storage_forwards),384)

    def test_mathematically_zero_kl_is_reported_without_fake_positive_requirement(self):
        self.reference['weight'].copy_(self.model.weight.detach())
        comparison=a.compare(self.captures())
        self.assertEqual(comparison['kl_gradient_l2'],0);self.assertEqual(comparison['weighted_kl_mean'],0)
        self.assertTrue(comparison['kl_retained']);self.assertEqual(comparison['recomposition_relative_l2'],0)

    def test_nonzero_pg_or_total_recomposition_failure_rejected(self):
        captures=self.captures()
        changed=copy.deepcopy(captures);changed['PG_ZERO']['gradients']['weight'].add_(.01)
        with self.assertRaisesRegex(PermissionError,'not exactly zero'):a.compare(changed)
        changed=copy.deepcopy(captures);changed['TOTAL']['gradients']['weight'].add_(.01)
        with self.assertRaisesRegex(PermissionError,'recomposition'):a.compare(changed)
        changed=copy.deepcopy(captures);changed['KL']['traces'][0]['reference_logprobs'][0]+=.1
        with self.assertRaisesRegex(PermissionError,'vectors'):a.compare(changed)

    def test_full128_original_sampler_and_group_order_are_enforced(self):
        with self.assertRaisesRegex(PermissionError,'complete fixed probe128'):self.capture('PG_ZERO',self.rows[:-1])
        changed=copy.deepcopy(self.rows);changed[0]['sampler_logprobs']=[-9.,-9.]
        original=self.handle.forward_original
        self.handle.forward_original=lambda row:original(self.rows[int(row['request_id'])])
        with self.assertRaisesRegex(ValueError,'Sampler differs'):self.capture('TOTAL',changed)
        self.handle.forward_original=original
        changed=copy.deepcopy(self.rows);changed[0]['group_id']='wrong'
        with self.assertRaisesRegex(ValueError,'G8 order'):self.capture('TOTAL',changed)

    def test_forward_failure_restores_rng_mode_and_clears_gradients(self):
        self.model.train();rng=self.k.state_hash(self.k.capture_rng())
        def fail(row):
            torch.rand(5);self.model.weight.grad=torch.ones_like(self.model.weight);raise RuntimeError('synthetic forward failure')
        self.handle.forward_original=fail
        with self.assertRaisesRegex(RuntimeError,'synthetic forward'):self.capture('TOTAL')
        self.assertTrue(self.model.training);self.assertIsNone(self.model.weight.grad)
        self.assertEqual(rng,self.k.state_hash(self.k.capture_rng()))

    def test_policy_or_original_score_mutation_is_rejected(self):
        original=self.handle.forward_original;before=self.model.weight.detach().clone()
        def mutate(row):
            with torch.no_grad():self.model.weight.add_(.01)
            return original(row)
        self.handle.forward_original=mutate
        with self.assertRaisesRegex(PermissionError,'Policy mutated'):self.capture('TOTAL')
        self.assertTrue(torch.equal(self.model.weight,before));self.assertIsNone(self.model.weight.grad)
        def mutate_score(row):
            row['native_score']['J']=1-row['native_score']['J'];return original(row)
        self.handle.forward_original=mutate_score
        with self.assertRaisesRegex(PermissionError,'rows/scores changed'):self.capture('TOTAL')

    def test_orchestration_records_real_cpu_accounting_and_no_retry(self):
        ctx=self.context();build,capture=self.patches()
        with build as builder,capture,patch.object(torch.optim,'AdamW',side_effect=AssertionError('No Adam')):
            result=a._execute(ctx,11031)
            self.assertEqual(result['status'],'ALLZERO_CAPTURE_COMPLETE_NOT_GPU_ADMISSION')
            self.assertEqual(result['measured_forward_sequences'],dict(training_policy=384,training_reference=384))
            self.assertEqual(result['trainable_tensors'],1)  # honest CPU toy, cannot satisfy production496 gate
            self.assertEqual(result['optimizer_steps'],0);self.assertEqual(result['new_generations'],0)
            self.assertEqual(self.rows,self.before);self.assertTrue(self.closed)
            with self.assertRaises(PermissionError):a._execute(ctx,11031)
            self.assertEqual(builder.call_count,1)

    def test_bad_accounting_preserves_hold_and_blocks_other_phases(self):
        ctx=self.context();build,capture=self.patches();original=a.worker._accounting
        def fail(*args,**kwargs):
            original(*args,**kwargs);raise PermissionError('synthetic accounting failure')
        with build,capture,patch.object(a.worker,'_accounting',side_effect=fail):
            with self.assertRaisesRegex(PermissionError,'accounting failure'):a._execute(ctx,11031)
        self.assertTrue(self.closed);self.assertTrue((self.root/'allzero_check/start-11031/HOLD.json').exists())
        with self.assertRaisesRegex(PermissionError,'Allzero HOLD/UNKNOWN'):
            a.worker._no_holds(self.root,ctx['binding'],binding_sha=ctx['binding_sha'])
        from execution import backend_equivalence
        with patch.object(a.native_runtime,'build_runtime') as build:
            with self.assertRaisesRegex(PermissionError,'Allzero HOLD/UNKNOWN'):
                backend_equivalence._execute(ctx,11031)
            build.assert_not_called()

    def test_occupied_unbound_or_unknown_slot_fails_before_constructor(self):
        ctx=self.context();path=self.root/'allzero_check/start-11031';path.mkdir(parents=True)
        (path/'CLAIM.json').write_text('{}')
        with patch.object(a.native_runtime,'build_runtime') as build:
            with self.assertRaisesRegex(PermissionError,'Allzero HOLD/UNKNOWN'):a._execute(ctx,11032)
            build.assert_not_called()
        del ctx['binding']['allzero_slots']
        with self.assertRaisesRegex(PermissionError,'without its fixed technical binding'):
            a.worker._no_holds(self.root,ctx['binding'],binding_sha=ctx['binding_sha'])

    def test_global_lock_and_cpu_public_entry_cannot_start_model(self):
        ctx=self.context()
        with a.worker._worker_lock(self.root),patch.object(a.native_runtime,'build_runtime') as build:
            with self.assertRaisesRegex(PermissionError,'global lock'):a._execute(ctx,11031)
            build.assert_not_called()
        with patch.object(torch.cuda,'is_available',return_value=False),patch.object(a,'_preflight') as preflight:
            with self.assertRaisesRegex(PermissionError,'two visible CUDA'):
                a.run_allzero(start_seed=11031,expected_binding_sha256='c'*64)
            preflight.assert_not_called()
        with self.assertRaises(TypeError):a.run_allzero(start_seed=11031,expected_binding_sha256='c'*64,handle=self.handle)

    def test_preflight_requires_source_sha_and_exact_two_slots(self):
        ctx=self.context()
        ctx['binding']['code']={'allzero_check':dict(path=str(Path(a.__file__).resolve()),sha256=a.shadow.file_sha(a.__file__))}
        with patch.object(a.worker,'_preflight',return_value=ctx):
            self.assertIs(a._preflight(11031,'c'*64),ctx)
            ctx['binding']['code']['allzero_check']['sha256']='0'*64
            with self.assertRaisesRegex(PermissionError,'Unbound allzero source'):a._preflight(11031,'c'*64)
            ctx['binding']['max_allzero_attempts']=3
            with self.assertRaisesRegex(PermissionError,'two-slot'):a._preflight(11031,'c'*64)


if __name__=='__main__':unittest.main()
