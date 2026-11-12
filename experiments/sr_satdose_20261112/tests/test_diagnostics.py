"""CPU fixtures exercising actual pinned native loss, never GPU evidence."""
import importlib.util
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import diagnostics as d

OBJECTIVE = Path(os.environ.get('SATDOSE_NATIVE_OBJECTIVE', str(ROOT.parent / 'dissertation for NTU/sr-f6/src/sr_f6/objective.py')))
LEGACY = Path(os.environ.get('SATDOSE_LEGACY_PACKAGE', str(ROOT / '.worktrees/sr-ord-train-20261103/sr_ord_train')))


@unittest.skipUnless(OBJECTIVE.exists() and (LEGACY/'adapter/training.py').exists(), 'Pinned legacy math files required')
class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = d.load_native_math(OBJECTIVE, LEGACY)

    def fixture(self, *, zeros=False, count=128, tokens=2):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.2, -0.1]]))
        rows=[]
        for i in range(count):
            x = [1.0, float(i%8)]
            c = int(i%8 >= 4)
            value = float(model(torch.tensor(x)).detach())
            rows.append(dict(request_id=str(i),group_id=str(i//8),J=0 if zeros else c,
                             exact_authorised=c,tokens=[1]*tokens,sampler_logprobs=[value]*tokens,x=x))
        def forward(row):
            current = model(torch.tensor(row['x'])).reshape(1).repeat(len(row['tokens']))
            sampler = torch.tensor(row['sampler_logprobs'], dtype=torch.float32)
            reference = sampler - 0.1
            return current,sampler,reference
        return model,rows,forward

    def capture(self, model, rows, forward, component):
        return d.accumulate_component(model,rows,forward,self.native,component,expected_tensors=1)['gradients']

    def test_native_replay_recomposition_and_unchanged(self):
        model,rows,forward=self.fixture();before=model.weight.detach().clone()
        random.seed(45);rng=random.getstate();trng=torch.get_rng_state().clone()
        total=self.capture(model,rows,forward,'total')
        replay=self.capture(model,rows,forward,'total')
        j=self.capture(model,rows,forward,'J');kl=self.capture(model,rows,forward,'KL')
        source=self.capture(model,rows,forward,'source')
        self.assertTrue(d.compare_replay(total,replay)['bit_exact'])
        self.assertLessEqual(d.recomposition_error(j,kl,total),1e-5)
        self.assertAlmostEqual(d.cosine(j,source),1)
        self.assertTrue(torch.equal(before,model.weight))
        self.assertIsNone(model.weight.grad)
        self.assertEqual(rng,random.getstate());self.assertTrue(torch.equal(trng,torch.get_rng_state()))

    def test_real_native_reference_swap_restores_policy_despite_version_change(self):
        native_root=OBJECTIVE.resolve().parents[1]
        if not (native_root/'mm_dev/runtime.py').is_file():
            self.skipTest('Complete native runtime source required')
        old_path=list(sys.path)
        try:
            sys.path.insert(0,str(native_root))
            from mm_dev import runtime
        finally:
            sys.path[:]=old_path
        self.assertEqual(Path(runtime.__file__).resolve(),native_root/'mm_dev/runtime.py')
        model,rows,_=self.fixture()
        before=model.weight.detach().clone();version=model.weight._version
        reference={'weight':before-0.01}
        reference_before=reference['weight'].clone()
        def native_forward(row):
            x=torch.tensor(row['x'])
            with torch.no_grad(),runtime.reference_parameters(model,reference):
                ref=model(x).reshape(1).repeat(len(row['tokens']))
            current=model(x).reshape(1).repeat(len(row['tokens']))
            return current,torch.tensor(row['sampler_logprobs']),ref
        result=self.capture(model,rows,native_forward,'total')
        self.assertGreater(d.dot(result,result),0)
        self.assertEqual(model.weight._version,version+2*len(rows))
        self.assertTrue(torch.equal(model.weight,before))
        self.assertTrue(torch.equal(reference['weight'],reference_before))
        self.assertIsNone(model.weight.grad)

    def test_frozen_mutation_is_still_rejected_even_when_value_restored(self):
        for kind in ('parameter','buffer'):
            with self.subTest(kind=kind):
                model,rows,forward=self.fixture()
                if kind=='parameter':
                    model.register_parameter('frozen',torch.nn.Parameter(torch.zeros(1),requires_grad=False))
                else:
                    model.register_buffer('frozen',torch.zeros(1))
                def mutate(row):
                    if row['request_id']=='0':
                        with torch.no_grad():
                            model.frozen.add_(1);model.frozen.sub_(1)
                    return forward(row)
                with self.assertRaisesRegex(ValueError,'frozen parameter/buffer'):
                    self.capture(model,rows,mutate,'total')

    def test_zero_j_keeps_kl_and_denominator(self):
        model,rows,forward=self.fixture(zeros=True)
        j=self.capture(model,rows,forward,'J');total=self.capture(model,rows,forward,'total')
        self.assertEqual(d.dot(j,j),0);self.assertGreater(d.dot(total,total),0)
        with self.assertRaisesRegex(ValueError,'zero gradient'):
            d.cosine(j,total)

    def test_witness_uses_sum_and_loo(self):
        model,rows,forward=self.fixture(tokens=2)
        h2=self.capture(model,rows,forward,'witness')
        for row in rows:
            row['tokens']*=2;row['sampler_logprobs']*=2
        h4=self.capture(model,rows,forward,'witness')
        self.assertTrue(torch.equal(h4['weight'],2*h2['weight']))
        self.assertAlmostEqual(d.witness_coefficients(rows,self.native)[0],-4/7)

    def test_gradient_cleanup_and_rng_on_failure(self):
        model,rows,forward=self.fixture();state=torch.get_rng_state().clone()
        def fail(row):
            torch.rand(2)
            if row['request_id']=='2':raise RuntimeError('intentional')
            return forward(row)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            self.capture(model,rows,fail,'total')
        self.assertIsNone(model.weight.grad);self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_reject_tampered_sampler_and_existing_grad(self):
        model,rows,forward=self.fixture()
        def wrong(row):
            a,b,c=forward(row);return a,b+1,c
        with self.assertRaisesRegex(ValueError,'Sampler differs'):
            self.capture(model,rows,wrong,'total')
        model.weight.grad=torch.zeros_like(model.weight)
        with self.assertRaisesRegex(ValueError,'Existing gradients'):
            self.capture(model,rows,forward,'total')

    def test_disjoint_frozen_partition(self):
        _,rows,_=self.fixture()
        halves=[[str(i) for i in range(8)],[str(i) for i in range(8,16)]]
        blocks=d.validate_partition(rows,halves)
        self.assertEqual([len(b) for b in blocks],[64,64])
        halves[1][0]='0'
        with self.assertRaises(ValueError):d.validate_partition(rows,halves)

    def test_shadow_copies_replay_and_once_per_start(self):
        model,rows,forward=self.fixture()
        total=self.capture(model,rows,forward,'total')
        policy={'weight':model.weight.detach().clone()};before=policy['weight'].clone()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            receipt,delta,clipped=d.shadow_pair(policy,total,root,11031,expected_tensors=1)
            self.assertEqual(receipt['shadow_optimizer_steps'],2)
            self.assertGreater(d.dot(delta,delta),0)
            self.assertTrue(torch.equal(before,policy['weight']))
            self.assertTrue(torch.equal(before,model.weight))
            with self.assertRaises(FileExistsError):d.shadow_pair(policy,total,root,11031,expected_tensors=1)
            d.shadow_pair(policy,total,root,11032,expected_tensors=1)
            with self.assertRaises(ValueError):d.shadow_pair(policy,total,root,11033,expected_tensors=1)

    def test_pins_and_missing_grad_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            objective=Path(tmp)/'objective.py';objective.write_text('raise RuntimeError("must not import")')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                d.load_native_math(objective,LEGACY)
        model,rows,forward=self.fixture()
        model.register_parameter('unused',torch.nn.Parameter(torch.zeros(1)))
        with self.assertRaisesRegex(ValueError,'Missing/nonfinite'):
            d.accumulate_component(model,rows,forward,self.native,'total',expected_tensors=2)

    def test_direction_floor_uses_repeat_not_between_halves(self):
        def capture(value):
            return {'gradients': {'x':torch.tensor([value],dtype=torch.float32)}}
        captures={name:capture(1) for name in ('J','source','total','J_half_1','J_half_2',
                  'source_half_1','source_half_2','witness_half_1','witness_half_1_repeat')}
        captures['KL']=capture(0)
        captures['witness_half_2']=capture(100)
        captures['witness_half_2_repeat']=capture(100)
        summary=d.direction_summary(captures,{'x':torch.tensor([1.])},{'x':torch.tensor([1.])})
        self.assertEqual(summary['effect_repeat_error_floor'],0)
        self.assertTrue(summary['direction_checks_satisfied'])
        captures['witness_half_1_repeat']=capture(3)
        summary=d.direction_summary(captures,{'x':torch.tensor([1.])},{'x':torch.tensor([1.])})
        self.assertEqual(summary['effect_repeat_error_floor'],2)
        self.assertFalse(summary['direction_checks_satisfied'])


if __name__=='__main__':unittest.main()
