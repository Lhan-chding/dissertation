"""CPU native-math and mocked-storage tests; these are NOT GPU equivalence tests."""
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import backend_equivalence as e

NATIVE=Path(os.environ.get('SATDOSE_NATIVE_SRC',str(ROOT.parent/'dissertation for NTU/sr-f6/src')))
LEGACY=ROOT/'.worktrees/sr-ord-train-20261103/sr_ord_train'


@unittest.skipUnless((NATIVE/'sr_f6/objective.py').exists() and (LEGACY/'adapter/training.py').exists(), 'Pinned native sources required')
class EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespaces={p.relative_to(NATIVE).parts[0].removesuffix('.py') for p in NATIVE.rglob('*.py')}
        namespaces.add('runtime_contract')
        before={name:mod for name,mod in sys.modules.items() if name.split('.')[0] in namespaces}
        def restore():
            for name in list(sys.modules):
                if name.split('.')[0] in namespaces:del sys.modules[name]
            sys.modules.update(before)
        cls.addClassCleanup(restore)
        for name in before:del sys.modules[name]
        cls.k=e.production.load_native_kernels(NATIVE,LEGACY)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.audit=self.root/'audit';self.audit.mkdir()
        self.model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():self.model.weight.fill_(-.4)
        self.endpoint=dict(experiment_id=e.shadow.EXPERIMENT,start_seed=11031,arm='START',step=0,
                           policy_sha256='a'*64,reference_sha256='b'*64)
        self.rows=[];self.requests={}
        for i in range(128):
            count=1 if i in (2,5) else 3
            public=dict(prompt='fixture',image_file='image.png',image_sha256='c'*64)
            row=dict(request_id=str(i),group_id=str(i//8),J=int(i%8>=4),exact_authorised=int(i%3==0),
                     tokens=[4]*count,sampler_logprobs=torch.full((count,),-.4).tolist(),seed=i,
                     prompt_tensor_hash='d'*64,model_input=public,policy_sha256='a'*64,reference_sha256='b'*64)
            self.requests[str(i)]=dict(public,request_id=str(i),scope='DRY',assistant_prefill='{',seed=i)
            self.rows.append(row)
        self.hook_count=0;self.storage_validated=0;self.active=None;self.closed=False
        self.reference={'weight':self.model.weight.detach().clone()-.05}
        self.runtime=SimpleNamespace(model=self.model,_satdose_accounting=[])
        self.runtime.bind_request=self.bind;self.runtime.unbind_request=self.unbind
        self.runtime.batch_reference_forward=self.reference_forward
        self.runtime.batch_sequence_forward=self.policy_forward
        def forbidden(*args,**kwargs):raise AssertionError('generation forbidden')
        self.runtime.generate=forbidden
        self.registry=SimpleNamespace(root=self.root,freeze_sha256='f'*64,request=lambda rid:copy.deepcopy(self.requests[rid]))
        self.handle=SimpleNamespace(runtime=self.runtime,registry=self.registry,output=self.audit,
            endpoint=lambda:copy.deepcopy(self.endpoint),close=self.close,
            fork=SimpleNamespace(reference=self.reference,optimizer=None,scheduler=None),
            storage=SimpleNamespace(GPUSavedTensors=self.hook,gpu_peak_stats=self.peaks,
                                    validate_snapshot_stats=self.validate_storage))
        for row in self.rows:self.persist(row)

    def persist(self,row):
        name=hashlib.sha256(row['request_id'].encode()).hexdigest()+'.CAPTURE.json'
        path=self.audit/name;path.write_text(json.dumps(row))
        path.with_name(name.replace('.CAPTURE.json','.BEGIN.json')).write_text(json.dumps(dict(
            request_id=row['request_id'],endpoint=self.endpoint,design_freeze_sha256='f'*64)))
        row.update(native_capture_path=str(path),native_capture_sha256=e.shadow.file_sha(path))

    def bind(self,request):
        self.assertIsNone(self.active);self.active=request

    def unbind(self):
        self.assertIsNotNone(self.active);self.active=None

    def reserve(self,purpose):
        event=dict(kind='extra_forward_sequences',count=1,metadata={'purpose':purpose})
        self.runtime._satdose_accounting.append(event)
        with (self.handle.output/'accounting.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')

    def reference_forward(self,records,reference):
        self.reserve('training_reference')
        return [torch.tensor(records[0]['sampler_logprobs'])-.05]

    def policy_forward(self,records,*,purpose,grad):
        self.assertIsNotNone(self.active);self.assertTrue(grad)
        self.reserve(purpose)
        return [self.model(torch.ones(1)).reshape(1).repeat(len(records[0]['tokens']))]

    def hook(self,*args,**kwargs):
        self.hook_count+=1
        stats=dict(synthetic_cpu_only=True,copied_tensors=0,unpacked_copies=0)
        def pack(tensor):stats['copied_tensors']+=1;return tensor.detach().clone()
        def unpack(tensor):stats['unpacked_copies']+=1;return tensor
        @contextmanager
        def context():
            with torch.autograd.graph.saved_tensors_hooks(pack,unpack):yield
        return SimpleNamespace(stats=stats,context=context)

    def peaks(self,*,reset=False):
        # Explicit CPU fake telemetry. No CUDA API is called or certified.
        return {name:dict(allocated_bytes=0,peak_allocated_bytes=0) for name in ('cuda:0','cuda:1')}

    def validate_storage(self,stats):
        self.assertTrue(stats['synthetic_cpu_only'])
        self.assertGreater(stats['copied_tensors'],0);self.assertGreater(stats['unpacked_copies'],0)
        self.storage_validated+=1

    def close(self):self.closed=True;return dict(closed=True)

    def capture(self,row=None,*,two_device=False,kernels=None):
        return e._capture(self.handle,row or self.rows[2],-1.,kernels or self.k,
                          two_device=two_device,expected_tensors=1)

    def test_shortest_original_tie_uses_order_and_complete_g8_advantage(self):
        selected=e.select_shortest(self.rows,self.k)
        self.assertEqual(selected['row'],self.rows[2])
        self.assertEqual(selected['original_index'],2)
        self.assertEqual(selected['group_request_ids'],list(map(str,range(8))))
        self.assertEqual(selected['advantage'],self.k.advantages([0]*4+[1]*4)[2])
        changed=copy.deepcopy(self.rows)
        for row in changed:row['J']=1-row['J'];row['exact_authorised']=1-row['exact_authorised']
        self.assertEqual(e.select_shortest(changed,self.k)['row']['request_id'],'2')
        self.assertEqual(selected['sequence_denominator'],128)

    def test_actual_native_loss_denominator_gradient_and_two_paths_equal(self):
        before=self.model.weight.detach().clone();rng=self.k.state_hash(self.k.capture_rng())
        baseline=self.capture()
        self.assertEqual(self.hook_count,0)
        second=self.capture(two_device=True)
        self.assertEqual(self.hook_count,1);self.assertEqual(self.storage_validated,1)
        self.assertTrue(e.compare(baseline,second)['bit_exact'])
        current=self.model(torch.ones(1)).reshape(1)
        terms=self.k.sequence_loss(current,torch.tensor(self.rows[2]['sampler_logprobs']),
                                   torch.tensor(self.rows[2]['sampler_logprobs'])-.05,-1.,0.)
        (terms['loss']/128).backward()
        self.assertTrue(torch.equal(baseline['preclip_gradients']['weight'],self.model.weight.grad))
        self.assertTrue(torch.equal(baseline['scaled_loss'],terms['loss'].detach()/128))
        self.model.zero_grad(set_to_none=True)
        self.assertTrue(torch.equal(before,self.model.weight))
        self.assertEqual(rng,self.k.state_hash(self.k.capture_rng()))
        self.assertEqual(baseline['original_tokens'],self.rows[2]['tokens'])

    def test_original_capture_request_and_endpoint_tamper_fail_closed(self):
        for key,value in [('tokens',[999]),('policy_sha256','wrong'),('prepared',{}),('seed',999)]:
            with self.subTest(key=key):
                row=dict(self.rows[2],**{key:value})
                with self.assertRaises(PermissionError):self.capture(row)
                self.assertIsNone(self.active);self.assertIsNone(self.model.weight.grad)
        self.assertEqual(self.hook_count,0)

    def test_forward_exception_unbinds_clears_grad_and_restores_rng_mode(self):
        self.model.train();before=self.k.state_hash(self.k.capture_rng())
        def fail(*args,**kwargs):
            torch.rand(5);self.model.weight.grad=torch.ones_like(self.model.weight)
            raise RuntimeError('synthetic forward failure')
        self.runtime.batch_sequence_forward=fail
        with self.assertRaisesRegex(RuntimeError,'synthetic forward'):self.capture(two_device=True)
        self.assertIsNone(self.active);self.assertIsNone(self.model.weight.grad)
        self.assertTrue(self.model.training)
        self.assertEqual(before,self.k.state_hash(self.k.capture_rng()))

    def test_policy_mutation_restored_then_rejected(self):
        before=self.model.weight.detach().clone();original=self.policy_forward
        def mutate(*args,**kwargs):
            with torch.no_grad():self.model.weight.add_(.1)
            return original(*args,**kwargs)
        self.runtime.batch_sequence_forward=mutate
        with self.assertRaisesRegex(PermissionError,'Policy mutated'):self.capture()
        self.assertTrue(torch.equal(before,self.model.weight));self.assertIsNone(self.model.weight.grad)

    def test_baseline_gpu1_allocation_sentinel_is_fail_closed_in_mock(self):
        self.handle.storage.gpu_peak_stats=lambda **kw:{'cuda:1':{'allocated_bytes':4,'peak_allocated_bytes':4}}
        with self.assertRaisesRegex(PermissionError,'already has tensor allocations'):self.capture()
        self.assertFalse(self.runtime._satdose_accounting)

    def test_tensor_mismatch_is_not_equal(self):
        result=self.capture();other=copy.deepcopy(result);other['teacher'][0]+=.1
        self.assertEqual(e.compare(result,other)['differences'],['teacher'])

    def execution_context(self):
        lock=self.root/'lock.json';lock.write_text('{}')
        binding=dict(source_lock=e.worker._ref(self.root,lock),design_freeze={'sha256':'f'*64},
                     shadow_slots=e.shadow.expected_slots(),backend_equivalence_slots=e.expected_slots(),max_backend_equivalence_attempts=2)
        return dict(root=self.root,binding=binding,binding_sha='c'*64,registry=self.registry,
            lock=dict(native_src=str(NATIVE),legacy_root=str(LEGACY)),protected=[],prepared=dict(starts={'11031':dict(
                expected_identity=dict(self.endpoint,parent='b0'),probe={'rows':self.rows})}))

    def execution_patches(self):
        actual_capture=e._capture
        def build(*args,output_dir,**kwargs):self.handle.output=output_dir;return self.handle
        def capture(*args,**kwargs):return actual_capture(*args,**kwargs,expected_tensors=1)
        return patch.object(e.native_runtime,'build_runtime',side_effect=build),patch.object(e,'_capture',side_effect=capture)

    def test_writeonce_original_evidence_no_optimizer_or_retry(self):
        ctx=self.execution_context();p1,p2=self.execution_patches()
        with p1 as builder,p2:
            complete=e._execute(ctx,11031)
            self.assertEqual(complete['status'],'BACKEND_EQUIVALENCE_CAPTURED_NOT_GPU_ADMISSION')
            self.assertFalse(complete['one_visible_gpu_startup_verified'])
            self.assertEqual(complete['long_sequence_stress'],'PENDING')
            self.assertTrue(self.closed);self.assertEqual(self.hook_count,1)
            with self.assertRaises(FileExistsError):e._execute(ctx,11031)
            self.assertEqual(builder.call_count,1)

    def test_unknown_claim_blocks_other_slot_before_constructor(self):
        ctx=self.execution_context()
        path=self.root/e.expected_slots()['11031'];path.mkdir(parents=True)
        (path/'CLAIM.json').write_text('{}')
        with patch.object(e.native_runtime,'build_runtime') as builder:
            with self.assertRaisesRegex(PermissionError,'HOLD/UNKNOWN'):e._execute(ctx,11032)
            builder.assert_not_called()

    def test_execution_failure_preserves_hold_and_closes(self):
        ctx=self.execution_context();p1,p2=self.execution_patches()
        with p1,p2,patch.object(e,'compare',return_value={'bit_exact':False,'differences':['teacher']}):
            with self.assertRaisesRegex(PermissionError,'gradients differ'):e._execute(ctx,11031)
        path=self.root/e.expected_slots()['11031']
        self.assertTrue((path/'HOLD.json').is_file());self.assertTrue(self.closed)
        self.assertTrue((path/'BASELINE.pt').exists());self.assertTrue((path/'TWO_DEVICE.pt').exists())
        with self.assertRaisesRegex(PermissionError,'HOLD/UNKNOWN'):e._execute(ctx,11031)

    def test_public_cpu_gate_and_no_handle_output_override(self):
        with patch.object(e,'_USED',False),patch.object(e,'_preflight',return_value={}),\
             patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(PermissionError,'Two visible CUDA'):
                e.run_equivalence(start_seed=11031,expected_binding_sha256='c'*64)
        with self.assertRaises(TypeError):
            e.run_equivalence(start_seed=11031,expected_binding_sha256='c'*64,output=self.root)


if __name__=='__main__':unittest.main()
