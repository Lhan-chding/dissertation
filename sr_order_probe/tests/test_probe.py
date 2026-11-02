from __future__ import annotations
import copy, itertools, json, math, subprocess, sys, unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from order_probe.common import exact,finite,seed_for
from order_probe.data import build_worlds,validate_root,oracle_sort,oracle_enumerate
from order_probe.inputs import NATURAL,PREFIX,prompt,assistant_prefill,prefix_orders,validate_prefix_triplet,request_rows
from order_probe.measurement import extract,measure,first_object_end
from order_probe.statistics import _contrast,paired_bootstrap,holm,root_effects
CFG=json.loads((R/'config/plan.json').read_text());FMT=(R/'assets/format_contract_v3.txt').read_text()
ROOTS=build_worlds(CFG,'MAIN');SMOKE=build_worlds(CFG,'SMOKE');TASK=ROOTS[0]['views'][0]


def correct_response(t=TASK, condition='C0'):
    w=t['world'];gold=t['gold']
    evidence=[{'series':s,'category':c,'value':w['series'][s][i]} for s in ('Alpha','Beta') for i,c in enumerate(w['categories'])]
    return {'evidence':evidence,'selected':list(gold['selected']),
            'bindings':[{'category':c,'operand':v} for c,v in zip(gold['selected'],gold['operands'])],
            'aggregate':t['query']['aggregate'],'answer':gold['answer']}

class DataTests(unittest.TestCase):
    def test_root_count(self): self.assertEqual(len(ROOTS),48)
    def test_smoke_count(self): self.assertEqual(len(SMOKE),4)
    def test_structure_balance(self): self.assertEqual(set(Counter(r['structure'] for r in ROOTS).values()),{12})
    def test_factor_balance(self):
        for st in {r['structure'] for r in ROOTS}:
            c=Counter((r['direction'],r['aggregate']) for r in ROOTS if r['structure']==st)
            self.assertEqual(set(c.values()),{3})
    def test_deterministic(self): self.assertEqual(ROOTS,build_worlds(CFG,'MAIN'))
    def test_paired_views(self):
        for r in ROOTS: validate_root(r)
    def test_dual_oracles(self):
        for r in ROOTS:
            for t in r['views']: self.assertEqual(oracle_sort(t['world'],t['query']),oracle_enumerate(t['world'],t['query']))
    def test_gold_finite(self):
        self.assertTrue(all(finite(Fraction(t['gold']['answer'])) for r in ROOTS for t in r['views']))
    def test_no_shared_worlds(self):
        a={r['views'][0]['unlabeled_world_query_key'] for r in ROOTS};b={r['views'][0]['unlabeled_world_query_key'] for r in SMOKE}
        self.assertFalse(a&b);self.assertEqual(len(a),48)
    def test_names_balanced_not_world_replicates(self):
        for r in ROOTS:self.assertEqual({v['rename'] for v in r['views']},{'ALIGNED','CONFLICT'})
    def test_all_boundary_choices_answer_separate(self):
        for r in ROOTS:
            t=r['views'][0]; vals=t['world']['series']['Beta'];sets=list(itertools.combinations(t['boundary_indices'],t['required_from_boundary']))
            sums=[sum(vals[i] for i in s) for s in sets];self.assertEqual(len(sums),len(set(sums)))
    def test_placebo_same_size_unselected(self):
        for r in ROOTS:
            for t in r['views']:
                self.assertEqual(len(t['placebo_indices']),len(t['boundary_indices']))
                self.assertFalse(set(t['placebo_indices'])&set(t['gold']['indices']))
    def test_no_outcome_selection(self):self.assertFalse(CFG['data']['reject_on_model_outcome'])

class InputTests(unittest.TestCase):
    def test_budget(self):
        req,t=request_rows(ROOTS,CFG,FMT);self.assertEqual(len(req),4320);self.assertEqual(len(t),96)
        self.assertEqual(Counter(x['branch'] for x in req),{'NATURAL':3456,'PREFIX':864})
    def test_smoke_budget(self):self.assertEqual(len(request_rows(SMOKE,CFG,FMT)[0]),216)
    def test_enum_same_members(self):
        for r in ROOTS:
            t=r['views'][0];self.assertEqual(sorted(t['enum_permutation']),list(range(10)))
    def test_enum_boundary_conflict(self):
        for r in ROOTS:
            t=r['views'][0];b=t['boundary_indices'];k=t['required_from_boundary']
            self.assertNotEqual(set([i for i in t['enum_permutation'] if i in b][:k]),set(b[:k]))
    def test_c0_exact_historical_examples(self):
        examples=json.loads((R/'sources/historical_prompt_examples.json').read_text())
        # The two archived items are TOPK Alpha -> Beta. Reconstruct the public task
        # directly from their original prompt and checklist (not hidden outcomes).
        import re
        for e in examples:
            old=e['model_input']['prompt'];qmatch=re.search(r'Select the (\d+) categories with the (largest|smallest) Alpha values. Compute the (sum|mean) of Beta',old)
            if qmatch is None:continue
            names=json.loads(re.search(r'in this public list: (\[.*?\])\.',old).group(1))
            t={'world':{'categories':names},'query':{'family':'TOPK','selection_series':'Alpha','value':'Beta','k':int(qmatch[1]),'direction':qmatch[2],'aggregate':qmatch[3]}}
            self.assertEqual(prompt(t,'C0',FMT),old)
    def test_nolist_keeps_format(self):
        p=prompt(TASK,'C_NOLIST',FMT);self.assertIn(FMT,p)
        for name in TASK['world']['categories']:self.assertNotIn('"'+name+'"',p)
    def test_ref_not_change_others(self):
        p=prompt(TASK,'C_REF',FMT);self.assertIn('The order in which you write evidence does not change',p);self.assertIn(FMT,p)
    def test_order_instructions_only_append(self):
        for c in ('C_PRINT','C_ALT'):self.assertTrue(prompt(TASK,c,FMT).startswith(prompt(TASK,'C0',FMT)+'\n'))
    def test_prefix_same_base_prompt(self):
        for c in PREFIX:self.assertEqual(prompt(TASK,c,FMT),prompt(TASK,'C_NOLIST',FMT))
    def test_prefix_validation_all(self):
        for r in ROOTS:
            for t in r['views']:validate_prefix_triplet(t)
    def test_prefix_equal_perturbation_count(self):
        for r in ROOTS:
            o=prefix_orders(r['views'][0]);base=o['P_STABLE']
            self.assertEqual(sum(a!=b for a,b in zip(base,o['P_REVTIE'])),sum(a!=b for a,b in zip(base,o['P_PLACEBO'])))
    def test_prefix_no_gold_target_fields(self):
        for c in PREFIX:
            p=assistant_prefill(TASK,c)
            for field in ('"selected"','"answer"','"bindings"'):self.assertNotIn(field,p)
    def test_prefix_beta_not_spatial_list(self):
        p=assistant_prefill(TASK,'P_STABLE');facts=json.loads(p[:-1]+'}')['evidence'];names=[e['category'] for e in facts if e['series']=='Beta']
        self.assertEqual(names,sorted(names))
    def test_common_seeds_within_branch(self):
        rows,_=request_rows(SMOKE,CFG,FMT);rows=[r for r in rows if r['qid']==SMOKE[0]['views'][0]['qid'] and r['parent']=='b0']
        self.assertEqual(len({r['seed'] for r in rows if r['condition'] in NATURAL}),1)
    def test_sampler_no_private_world(self):
        rows,_=request_rows(SMOKE,CFG,FMT)
        for row in rows:
            for k in ('world','query','gold','rename','boundary_indices','gold_answer'):self.assertNotIn(k,row)
    def test_unknown_condition_raises(self):
        with self.assertRaises(ValueError):prompt(TASK,'NEW_ARM',FMT)

class ScoreTests(unittest.TestCase):
    def test_correct(self):
        r=measure(json.dumps(correct_response()),TASK,'C0','balanced')
        self.assertEqual((r['A'],r['S'],r['B'],r['V_probe']),(1,1,1,1))
    def test_answer_not_gated_by_new_field(self):
        o=correct_response();o['selected']=[{'category':x} for x in o['selected']]
        r=measure(json.dumps(o),TASK,'C0','balanced');self.assertEqual(r['A'],1);self.assertIsNone(r['S']);self.assertEqual(r['V_probe'],0)
    def test_duplicate_json_key(self):self.assertIsNone(extract('{"answer":1,"answer":2}')[0])
    def test_duplicate_selection_is_failure(self):
        o=correct_response();o['selected'].append(o['selected'][0]);r=measure(json.dumps(o),TASK,'C0','balanced')
        self.assertEqual(r['S'],0);self.assertEqual(r['S_known'],1)
    def test_incomplete_unknown(self):
        r=measure(json.dumps(correct_response()),TASK,'C0','length');self.assertIsNone(r['S']);self.assertEqual(r['A'],1)
    def test_no_work_answer_repair(self):
        o=correct_response();o['answer']=-999;o['work']='The correct answer is '+TASK['gold']['answer']
        self.assertEqual(measure(json.dumps(o),TASK,'C0','balanced')['A'],0)
    def test_no_second_json_repair(self):
        self.assertIsNone(extract('{bad}\n'+json.dumps(correct_response()))[0])
    def test_order_truth_separated(self):
        o=correct_response();o['evidence'][0]['value']=-123;r=measure(json.dumps(o),TASK,'C_PRINT','balanced')
        self.assertEqual(r['rank_order_print'],1);self.assertEqual(r['rank_values_true'],0);self.assertEqual(r['order_compliance'],1)
    def test_missing_rank_order_unknown(self):
        o=correct_response();o['evidence']=o['evidence'][1:];r=measure(json.dumps(o),TASK,'C0','balanced')
        self.assertEqual(r['rank_order_known'],0);self.assertEqual(r['S'],1)
    def test_stable_prefix_truthful(self):
        pre=assistant_prefill(TASK,'P_STABLE');o=correct_response();del o['evidence']
        full=pre+json.dumps(o)[1:];r=measure(full,TASK,'P_STABLE','balanced')
        self.assertEqual((r['A'],r['S'],r['rank_values_true']),(1,1,1));self.assertEqual(r['forced_evidence'],1)
    def test_scanner_prefix_suffix(self):
        pre=assistant_prefill(TASK,'P_STABLE');o=correct_response();del o['evidence'];full=pre+json.dumps(o)[1:]
        self.assertIsNone(first_object_end(pre));self.assertEqual(first_object_end(full),len(full))
    def test_scanner_escaped_quote_and_brace(self):
        text=json.dumps({'evidence':[],'work':'brace } and quote " and slash \\','answer':1});self.assertEqual(first_object_end(text),len(text))
    def test_nonfinite_rejected(self):self.assertIsNone(extract('{"answer":NaN}')[0])
    def test_exact_fraction(self):self.assertEqual(exact('17/3'),Fraction(17,3))
    def test_number_bounds(self):
        with self.assertRaises(ValueError):exact('71.66666666666667')

class StatisticsTests(unittest.TestCase):
    def test_zero_effect(self):
        r=paired_bootstrap(np.zeros(48),['a']*12+['b']*12+['c']*12+['d']*12,B=100)
        self.assertEqual(r['estimate'],0);self.assertEqual(r['p_approx_centered_bootstrap'],1)
    def test_constant_nonzero_flag(self):
        r=paired_bootstrap(np.ones(48),['a']*48,B=100);self.assertIsNone(r['p_approx_centered_bootstrap'])
    def test_holm(self):self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])
    def test_interaction_sign(self):
        v={(g,c):1.0 for g in ('ALIGNED','CONFLICT') for c in NATURAL+PREFIX};v['CONFLICT','C_ALT']=0
        self.assertEqual(_contrast(v,'I_ALT'),1)
    def test_specificity_contrast(self):
        v={(g,c):1.0 for g in ('ALIGNED','CONFLICT') for c in NATURAL+PREFIX}
        for g in ('ALIGNED','CONFLICT'):v[g,'P_REVTIE']=0
        self.assertEqual(_contrast(v,'D_SPEC'),1);self.assertEqual(_contrast(v,'D_PLACEBO'),0)
    def test_bootstrap_reproducible(self):
        a=paired_bootstrap(np.arange(48)/48,['a']*48,B=200,seed=4);b=paired_bootstrap(np.arange(48)/48,['a']*48,B=200,seed=4);self.assertEqual(a,b)
    def test_protocol_forbids_training(self):
        self.assertFalse(CFG['authorization']['train']);self.assertEqual(CFG['authorization']['optimizer_updates'],0)
    def test_not_run_preflight_blocked(self):
        p=subprocess.run([sys.executable,str(R/'scripts/check_ready.py'),'--receipt',str(R/'templates/PREFLIGHT_NOT_RUN.json')],capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertIn('BLOCKED',p.stdout)

if __name__=='__main__':unittest.main()
