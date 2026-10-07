from __future__ import annotations
import importlib.util, itertools, json, math, random, unittest
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from contracts import *
from statistics_reference import primary_effect, cross_macro
ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'manifests'

def read(name):return [json.loads(x) for x in (M/name).read_text().splitlines() if x.strip()]

class MathContractTests(unittest.TestCase):
    def setUp(self):
        self.t,self.a=make_task([1,2,3,4],[1,9,3,4],3,1,'test','DONOR_TRAIN',0)
    def test_unique_repair(self):self.assertEqual(solve_public(self.t),[[1,2,3,4]])
    def test_copy_not_accepted(self):self.assertFalse(verify(self.t,[1,9,3,4]))
    def test_wrong_value_not_accepted(self):self.assertFalse(verify(self.t,[1,7,3,4]))
    def test_bool_rejected(self):self.assertIsNone(parse('[true,2,3,4]'))
    def test_float_rejected(self):self.assertIsNone(parse('[1.0,2,3,4]'))
    def test_wrapped_json_rejected(self):self.assertIsNone(parse('```[1,2,3,4]```'))
    def test_multiple_json_rejected(self):self.assertIsNone(parse('[1,2,3,4] [1,2,3,4]'))
    def test_oob_parse_retained(self):self.assertEqual(parse('[1,-2,3,4]'),[1,-2,3,4]);self.assertFalse(valid([1,-2,3,4]))
    def test_edit_identity(self):
        for y in itertools.product([0,1,2,3,4,9],repeat=4):
            o=[1,9,3,4];x=[1,2,3,4];j=1
            m=sum(y[k]!=o[k] for k in range(4));b=sum(y[k]!=o[k] for k in range(4) if k!=j)
            l=int(y[j]!=o[j]);f=int(y[j]==x[j])
            self.assertEqual(m,b+l);self.assertLessEqual(f,l)
    def test_false_equivalence_counterexample(self):
        y=[1,7,3,4];self.assertEqual(sum(v!=w for v,w in zip(y,[1,9,3,4])),1);self.assertNotEqual(y[1],2)
    def test_projection_does_not_read_truth(self):self.assertEqual(deletion_projection(self.t,[87,2,-1,17]),[1,2,3,4])
    def test_projection_missing_value(self):self.assertIsNone(deletion_projection(self.t,[1,7,3,4]))
    def test_projection_equivalence(self):
        for y in itertools.product([-1,1,2,4,9],repeat=4):
            self.assertEqual(deletion_projection(self.t,list(y)) is not None,y[1]==2)
    def test_projection_unparsed(self):self.assertIsNone(deletion_projection(self.t,None))
    def test_main_sign(self):self.assertAlmostEqual(specificity(.4,.7,.8,.5),-.3)
    def test_interaction_not_absolute_harm(self):
        # Both B and C above A=.1, yet structural contrast can be negative.
        self.assertLess(specificity(.6,.8,.8,.6),0)
    def test_wilson_zero_not_zero_upper(self):self.assertGreater(wilson(0,8)[1],0.3)
    def test_wilson_all_not_one_lower(self):self.assertLess(wilson(8,8)[0],0.8)
    def test_conditional_empty_null(self):self.assertIsNone(safe_ratio(0,0))
    def test_passk_correct(self):self.assertAlmostEqual(pass_at_k(8,2,2),1-30/56)
    def test_passk_zero(self):self.assertEqual(pass_at_k(8,0,8),0)
    def test_passk_invalid(self):
        with self.assertRaises(ValueError):pass_at_k(8,2,9)
    def test_lr(self):self.assertEqual(lr_at(1),1.25e-6);self.assertEqual(lr_at(8),1e-5);self.assertEqual(lr_at(256),1e-5)
    def test_microbatch_partition(self):self.assertEqual(sum(grouped_microbatches(),[]),list(range(16)))
    def test_donor_isolated(self):self.assertIn([11],grouped_microbatches())
    def test_macro_not_count_weighted(self):
        c={(h,j):0 for h in range(4) for j in range(4)};c[2,2]=c[3,3]=1
        self.assertEqual(cross_macro(c),.125)
    def test_bootstrap_retains_pairing(self):
        p=np.full((2,3,3,128,2),.6)
        p[:,:,1,:,1]=.4;p[:,:,2,:,0]=.4
        out=primary_effect(p,bootstrap_replicates=100)
        self.assertAlmostEqual(out['Gamma'],-.2)
        self.assertTrue(np.allclose(out['ci95_root_conditional'],[-.2,-.2]))
    def test_bootstrap_seed_reproducible(self):
        p=np.random.default_rng(1).random((2,3,3,128,2))
        self.assertEqual(primary_effect(p,bootstrap_replicates=100),primary_effect(p,bootstrap_replicates=100))

class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=read('donors_public.jsonl');cls.a={r['task_id']:r for r in read('donors_audit.jsonl')}
        cls.t=read('donor_targets.jsonl');cls.p=json.loads((ROOT/'protocol.json').read_text())
    def test_build_counts(self):
        x=json.loads((M/'BUILD_REPORT.json').read_text())
        self.assertEqual((x['common_tasks'],x['replay_tasks'],x['donor_roots'],x['confirm_tasks']),(182,64,32,800))
    def test_donor_targets_equal(self):
        g=defaultdict(list)
        for x in self.t:g[x['root_id']].append(x['target'])
        self.assertEqual(len(g),32)
        for vals in g.values():self.assertEqual(len(vals),3);self.assertEqual(len(set(vals)),1)
    def test_donor_observations_equal(self):
        g=defaultdict(list)
        for x in self.d:g[x['root_id']].append(x)
        for ts in g.values():
            self.assertEqual(len({tuple(x['observed']) for x in ts}),1)
            self.assertEqual(len({tuple(map(tuple,x['H_original'])) for x in ts}),3)
            self.assertEqual(len({x['operation'] for x in ts}),1)
    def test_j2_only(self):self.assertEqual({x['corrupted_index'] for x in self.a.values()},{1})
    def test_donors_verify(self):
        for t in self.d:self.assertEqual(solve_public(t),[self.a[t['task_id']]['true_world']])
    def test_all_new_unique(self):
        for panel in ['E_DIAG','E_CONFIRM']:
            a={x['task_id']:x for x in read(panel+'/audit_only.jsonl')}
            for t in read(panel+'/tasks_public.jsonl'):self.assertEqual(solve_public(t),[a[t['task_id']]['true_world']])
    def test_new_roots_disjoint(self):
        g=defaultdict(set)
        for r in read('new_root_registry_AUDIT_ONLY.jsonl'):g[r['split']].add(tuple(sorted(r['true_world'])))
        for a,b in itertools.combinations(g,2):self.assertFalse(g[a]&g[b])
    def test_confirm_counts(self):
        rows=read('E_CONFIRM/audit_only.jsonl');c=Counter((r['center'],r['corrupted_index']) for r in rows if r['center'] is not None)
        self.assertEqual(sum(c.values()),704)
        self.assertEqual(c[2,2],128);self.assertEqual(c[3,3],128)
        self.assertEqual(set(v for k,v in c.items() if k not in [(2,2),(3,3)]),{32})
    def test_main_roots_paired(self):
        rows=read('E_CONFIRM/audit_only.jsonl')
        g3={r['root_id'] for r in rows if r['center']==2 and r['corrupted_index']==2}
        g4={r['root_id'] for r in rows if r['center']==3 and r['corrupted_index']==3}
        self.assertEqual(g3,g4);self.assertEqual(len(g3),128)
    def test_common_replay_validate(self):
        for f in ['common_targets.jsonl','replay_targets.jsonl']:
            for r in read(f):self.assertTrue(verify(r['task'],json.loads(r['target'])))
    def test_role_no_audit_in_prompt(self):
        for r in self.d:
            p=public_prompt(r);self.assertEqual(set(p),{'system','user'})
            self.assertNotIn(r['task_id'],p['user']);self.assertNotIn('corrupted_index',p['user'])
    def test_schedule_sizes_and_shared_slots(self):
        for seed in (108701,108702,108703):
            rows=read(f'schedule_{seed}.jsonl');self.assertEqual(len(rows),3*256*16)
            g=defaultdict(list)
            for r in rows:g[(r['update'],r['slot'])].append(r)
            for (s,slot),v in g.items():
                self.assertEqual(len(v),3)
                if slot==11:self.assertEqual(len({x['donor_root'] for x in v}),1)
                else:self.assertEqual(len({x['task_id'] for x in v}),1)
    def test_schedule_role_counts(self):
        rows=read('schedule_108701.jsonl')
        for a in ARMS:
            c=Counter(r['role'] for r in rows if r['arm']==a)
            self.assertEqual(c,dict(common=2816,donor=256,replay=1024))
    def test_exact_donor_exposure(self):
        for seed in (108701,108702,108703):
            rows=read(f'schedule_{seed}.jsonl')
            c=Counter((r['arm'],r['donor_root']) for r in rows if r['role']=='donor')
            self.assertEqual(set(c.values()),{8});self.assertEqual(len(c),96)
    def test_actual_different_schedules(self):
        r1=read('schedule_108701.jsonl');r2=read('schedule_108702.jsonl')
        self.assertNotEqual([r['task_id'] for r in r1],[r['task_id'] for r in r2])
    def test_job_matrix(self):
        jobs=read('training_jobs.jsonl');self.assertEqual(len(jobs),18)
        self.assertEqual(len({(j['parent'],j['block'],j['arm']) for j in jobs}),18)
    def test_workload(self):
        w=experiment_workload();self.assertEqual(w['formal_generation_total'],145280)
        self.assertEqual(w['formal_sft_updates'],4608);self.assertEqual(w['formal_sft_target_exposures'],73728)
        self.assertEqual(self.p['workload'],w)
    def test_no_extra_training_factors(self):
        self.assertFalse(self.p['scope']['new_rl']);self.assertFalse(self.p['scope']['new_discovery'])
        self.assertFalse(self.p['training']['dose_grid'])

class TinyLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        spec=importlib.util.spec_from_file_location('vdt_source_loss',ROOT/'sources/VDT_sft_loss.py')
        module=importlib.util.module_from_spec(spec)
        import sys;sys.modules[spec.name]=module;spec.loader.exec_module(module)
        cls.loss=module
    def test_shift_and_eos_mask(self):
        torch=self.torch
        labels=torch.tensor([[-100,-100,2,3,-100]])
        logits=torch.zeros((1,5,5),requires_grad=True)
        means,_,mask=self.loss.per_sequence_nll(logits,labels)
        self.assertEqual(mask.tolist(),[[False,True,True,False]])
        self.assertAlmostEqual(float(means.detach()[0]),math.log(5),places=6)
        means.sum().backward()
        self.assertTrue(torch.equal(logits.grad[0,0],torch.zeros(5)))
        self.assertTrue(torch.equal(logits.grad[0,4],torch.zeros(5)))
    def test_variable_target_sequence_mean(self):
        torch=self.torch
        labels=torch.tensor([[-100,1,2,3],[-100,-100,-100,2]])
        means,_,mask=self.loss.per_sequence_nll(torch.zeros((2,4,5)),labels)
        self.assertEqual(mask.sum(-1).tolist(),[3,1])
        self.assertTrue(torch.allclose(means,torch.full((2,),math.log(5))))
    def test_microbatch_accumulation_equal_weight(self):
        torch=self.torch
        x=torch.arange(64,dtype=torch.float64).reshape(16,4)/64
        y=torch.arange(16)%3
        base=torch.nn.Linear(4,3,bias=True,dtype=torch.float64)
        other=torch.nn.Linear(4,3,bias=True,dtype=torch.float64);other.load_state_dict(base.state_dict())
        torch.nn.functional.cross_entropy(base(x),y,reduction='mean').backward()
        for ids in grouped_microbatches():
            torch.nn.functional.cross_entropy(other(x[ids]),y[ids],reduction='sum').div(16).backward()
        for p,q in zip(base.parameters(),other.parameters()):self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-12,rtol=1e-12))
    def test_seed_only_not_training_replication(self):
        torch=self.torch
        def run(seed):
            torch.manual_seed(seed)
            w=torch.nn.Parameter(torch.tensor([1.]))
            opt=torch.optim.AdamW([w],lr=.01,weight_decay=0)
            for v in [1.,2.,3.]:
                opt.zero_grad();((w-v)**2).sum().backward();opt.step()
            return w.detach()
        self.assertTrue(torch.equal(run(1),run(999)))
    def test_zero_grad_and_none_differ(self):
        torch=self.torch
        def run(skip):
            w=torch.nn.Parameter(torch.tensor([1.]))
            opt=torch.optim.AdamW([w],lr=.01,weight_decay=0)
            w.grad=torch.tensor([1.]);opt.step()
            w.grad=None if skip else torch.tensor([0.]);opt.step()
            return w.detach()
        self.assertFalse(torch.equal(run(True),run(False)))

if __name__=='__main__':unittest.main(verbosity=2)
