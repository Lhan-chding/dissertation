"""CPU synthetic checks for independent production arithmetic; no GPU claims."""
import copy
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import production as p
from runtime import diagnostics as d

NATIVE = Path(os.environ.get('SATDOSE_NATIVE_SRC',str(ROOT.parent/'dissertation for NTU/sr-f6/src')))
LEGACY = Path(os.environ.get('SATDOSE_LEGACY_PACKAGE',str(ROOT/'.worktrees/sr-ord-train-20261103/sr_ord_train')))


@unittest.skipUnless((NATIVE/'sr_f6/objective.py').exists() and (LEGACY/'adapter/training.py').exists(), 'Pinned full native sources required')
class ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Earlier suites intentionally import immutable vendored scorer copies.
        # Isolate native namespaces for this CPU fixture; production shadowing
        # rejection is unchanged and is never disabled in load_native_kernels.
        namespaces={path.relative_to(NATIVE).parts[0].removesuffix('.py')
                    for path in NATIVE.rglob('*.py')}
        namespaces.add('runtime_contract')
        before={name:module for name,module in sys.modules.items()
                if name.split('.')[0] in namespaces}
        def restore():
            for name in list(sys.modules):
                if name.split('.')[0] in namespaces:
                    del sys.modules[name]
            sys.modules.update(before)
        cls.addClassCleanup(restore)
        for name in before:
            del sys.modules[name]
        cls.k = p.load_native_kernels(NATIVE,LEGACY)
        cls.d = d.load_native_math(NATIVE/'sr_f6/objective.py',LEGACY)

    def fixture(self, *, zeros=False, arm='START'):
        model = torch.nn.Linear(2,1,bias=False)
        with torch.no_grad(): model.weight.copy_(torch.tensor([[.2,-.1]]))
        reference = {'weight':model.weight.detach().clone()-.01}
        policy_sha = self.k.state_hash({'weight':model.weight.detach().clone()})
        reference_sha = self.k.state_hash(reference)
        config = dict(experiment_id=p.EXPERIMENT,start_bindings={'11031':dict(policy_sha256=policy_sha,reference_sha256=reference_sha)})
        identity = dict(experiment_id=p.EXPERIMENT,parent='b0',start_seed=11031,arm=arm,step=0,
                        policy_sha256=policy_sha,reference_sha256=reference_sha)
        scope, step = ('DRY',0) if arm=='START' else ('TRAIN',1)
        rows,groups=[],[]
        for i in range(128):
            x=[1.,float(i%8)];current=float(model(torch.tensor(x)).detach())
            j=0 if zeros else int(i%8>=4)
            rows.append(dict(request_id=str(i),group_id=str(i//8),root_id='root'+str(i//8),start_seed=11031,
                scope=scope,arm=arm,step=step,policy_sha256=policy_sha,reference_sha256=reference_sha,
                tokens=[1,2],sampler_logprobs=[current,current],J=j,native_score={'J':j},exact_authorised=int(i%8>=4),
                generation_status='COMPLETE',technical_validation_errors=[],finish_reason='balanced',assistant_prefill='{',
                sampler_logprob_source='actual_generate_raw_logits_selected_token',x=x))
        for i in range(16):
            groups.append(dict(group_id=str(i),root_id='root'+str(i),request_ids=[str(j) for j in range(i*8,i*8+8)]))
        plan=dict(scope=scope,step=step,groups=groups)
        def forward(row):
            x=torch.tensor(row['x'])
            current=model(x).reshape(1).repeat(2)
            with torch.no_grad(): ref=(reference['weight']@x).reshape(1).repeat(2)
            return current,torch.tensor(row['sampler_logprobs']),ref
        return model,rows,forward,reference,identity,config,plan

    def capture(self,f,forward=None):
        model,rows,fn,reference,identity,config,plan=f
        return p.capture_update(model,rows,forward or fn,reference,kernels=self.k,identity=identity,config=config,plan=plan,expected_tensors=1)

    def test_independent_production_equals_diagnostics_bit_exact(self):
        f=self.fixture();model,rows,forward,*_=f
        result=self.capture(f)
        diag=d.accumulate_component(model,rows,forward,self.d,'total',expected_tensors=1)
        self.assertTrue(torch.equal(result['preclip_gradients']['weight'],diag['gradients']['weight']))
        self.assertEqual(result['sequence_denominator'],128)
        self.assertEqual(result['backward_calls'],128)
        self.assertEqual(result['row_forward_callback_calls'],128)
        self.assertNotIn('forward_calls',result)
        self.assertLessEqual(result['numerics']['gradient_norm_after_clip'],1.00001)
        self.assertFalse(result['optimizer_created'])

    def test_start_policy_may_differ_from_original_reference(self):
        for arm in ('T0','T1'):
            with self.subTest(arm=arm):
                f=self.fixture(arm=arm)
                self.assertNotEqual(f[4]['policy_sha256'],f[4]['reference_sha256'])
                self.assertEqual(self.capture(f)['optimizer_steps'],0)

    def test_zero_advantage_keeps_kl_and_all128_denominator(self):
        f=self.fixture(zeros=True);model,rows,forward,*_=f
        result=self.capture(f)
        j=d.accumulate_component(model,rows,forward,self.d,'J',expected_tensors=1)
        total=d.accumulate_component(model,rows,forward,self.d,'total',expected_tensors=1)
        self.assertEqual(torch.count_nonzero(j['gradients']['weight']),0)
        self.assertGreater(torch.count_nonzero(result['preclip_gradients']['weight']),0)
        self.assertTrue(torch.equal(result['preclip_gradients']['weight'],total['gradients']['weight']))
        self.assertEqual(result['totals']['native'],0)
        self.assertGreater(result['totals']['kl'],0)
        self.assertEqual(len(result['traces']),128)

    def test_no_adam_constructor_or_body_called(self):
        with patch.object(torch.optim,'AdamW',side_effect=AssertionError('Adam forbidden')):
            result=self.capture(self.fixture())
        self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(result['scheduler_steps'],0)

    def test_failures_clear_grad_and_restore_full_rng_training(self):
        f=self.fixture();model,_,forward,*_=f
        model.train();rng=self.k.state_hash(self.k.capture_rng());before=model.weight.detach().clone()
        def fail(row):
            random.random();np.random.rand();torch.rand(1)
            if row['request_id']=='3': raise RuntimeError('injected before end')
            return forward(row)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            self.capture(f,fail)
        self.assertIsNone(model.weight.grad);self.assertTrue(model.training)
        self.assertEqual(self.k.state_hash(self.k.capture_rng()),rng)
        self.assertTrue(torch.equal(before,model.weight))

    def test_policy_mutation_rejected_and_restored(self):
        f=self.fixture();model,_,forward,*_=f;before=model.weight.detach().clone()
        def mutate(row):
            if row['request_id']=='0':
                with torch.no_grad(): model.weight.add_(.001)
            return forward(row)
        with self.assertRaisesRegex(PermissionError,'policy changed'):self.capture(f,mutate)
        self.assertTrue(torch.equal(before,model.weight));self.assertIsNone(model.weight.grad)

    def test_frozen_buffer_mutation_rejected(self):
        f=self.fixture();model,_,forward,*_=f;model.register_buffer('frozen',torch.zeros(1))
        def mutate(row):
            if row['request_id']=='0':
                model.frozen.add_(1);model.frozen.sub_(1)
            return forward(row)
        with self.assertRaisesRegex(PermissionError,'discard runtime'):self.capture(f,mutate)
        self.assertIsNone(model.weight.grad)

    def test_original_sampler_gate_and_tamper_fail_closed(self):
        f=self.fixture();forward=f[2]
        def wrong(row):
            a,b,c=forward(row);return a,b+.1,c
        with self.assertRaisesRegex(PermissionError,'sampler changed'):self.capture(f,wrong)
        f=self.fixture()
        for row in f[1]: row['sampler_logprobs']=(torch.tensor(row['sampler_logprobs'])+.02).tolist()
        with self.assertRaisesRegex(ValueError,'sampler/teacher'):self.capture(f)
        self.assertIsNone(f[0].weight.grad)

    def test_missing_or_reordered_rows_rejected_before_forward(self):
        for change in ('missing','order','technical'):
            f=self.fixture()
            if change=='missing':f[1].pop()
            if change=='order':f[1][0],f[1][1]=f[1][1],f[1][0]
            if change=='technical':f[1][0]['generation_status']='OOM'
            with self.assertRaises(PermissionError):
                self.capture(f,lambda row: (_ for _ in ()).throw(AssertionError('must not forward')))

    def test_existing_gradients_are_preserved_on_rejection(self):
        f=self.fixture();f[0].weight.grad=torch.ones_like(f[0].weight)
        with self.assertRaisesRegex(PermissionError,'Preexisting'):self.capture(f)
        self.assertTrue(torch.equal(f[0].weight.grad,torch.ones_like(f[0].weight)))

    def test_real_native_reference_swap_is_allowed(self):
        old=list(sys.path)
        try:
            sys.path.insert(0,str(NATIVE))
            from mm_dev.runtime import reference_parameters
        finally:sys.path[:]=old
        f=self.fixture();model,_,_,reference,*_=f;before=model.weight.detach().clone()
        def actual(row):
            x=torch.tensor(row['x'])
            with torch.no_grad(),reference_parameters(model,reference):ref=model(x).reshape(1).repeat(2)
            return model(x).reshape(1).repeat(2),torch.tensor(row['sampler_logprobs']),ref
        result=self.capture(f,actual)
        self.assertEqual(result['optimizer_steps'],0)
        self.assertTrue(torch.equal(before,model.weight));self.assertIsNone(model.weight.grad)


if __name__=='__main__':unittest.main()
