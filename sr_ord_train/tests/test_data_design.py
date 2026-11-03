"""CPU scientific semantics tests; no model, torch runtime, network, image or GPU required."""
import copy
import itertools
import json
import os
from fractions import Fraction
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import data_design as d


def named_facts(t):
    w=t['world'];return {name:(w['series']['Alpha'][i],w['series']['Beta'][i]) for i,name in enumerate(w['categories'])}


def response(task):
    w=task['world'];g=task['gold'];facts=named_facts(task)
    ev=[{'series':'Alpha','category':name,'value':pair[0]} for name,pair in facts.items()]
    ev.extend({'series':'Beta','category':name,'value':facts[name][1]} for name in g['selected'])
    return json.dumps({'evidence':ev,'selected':g['selected'],'bindings':[{'category':name,'operand':facts[name][1]} for name in g['selected']],
                       'aggregate':task['query']['aggregate'],'answer':g['answer']})


class TestOrderData(unittest.TestCase):
    def test_dual_oracles_exact_and_balanced(self):
        seen=set()
        for n in (8,10):
            cells=set()
            for i in range(32):
                t=d.make_world('UNIT',i,n); cells.add((t['structure'],t['query']['direction'],t['query']['aggregate']))
                w,q=t['world'],t['query']; self.assertEqual(t,d.make_world('UNIT',i,n));self.assertEqual(t['gold'],d.oracle_enumerate(w,q))
                self.assertEqual(len(t['strictly_better_indices']),2); self.assertNotIn(t['unlabeled_world_query_key'],seen);seen.add(t['unlabeled_world_query_key'])
                vs=[Fraction(sum(w['series']['Beta'][j] for j in sub+tuple(t['strictly_better_indices'])),q['k'] if q['aggregate']=='mean' else 1)
                    for sub in itertools.combinations(t['boundary_indices'],t['r'])]
                self.assertEqual(len(vs),len(set(vs)));self.assertIn(Fraction(t['gold']['answer']),vs)
                self.assertEqual(t['matched_placebo_available'], n==10 or t['structure']!='4_choose_2')
            self.assertEqual(len(cells),16)
        t=d.make_world('A',1);u=d.make_world('B',1);self.assertNotEqual(t['world_query_key'],u['world_query_key']);self.assertNotEqual(t['root_id'],u['root_id'])
        self.assertNotEqual(t['root_id'],d.make_world('A',1,master_seed=8)['root_id'])
        self.assertNotEqual(t['root_id'],d.make_world('A',1,aggregate='mean')['root_id'])

    def test_native_scores_not_a_probe_substitute(self):
        d._v3();from sr_f6.measurement import score_response
        for i in range(16):
            t=d.make_world('GRADE',i);s=score_response(response(t),t['world'],t['query']);self.assertEqual((s['J'],s['V'],s['S'],s['B']),(1,1,1,1))
            bad=json.loads(response(t));bad['answer']=999;self.assertEqual(score_response(json.dumps(bad),t['world'],t['query'])['J'],0)
            c=d.build_context(t,'BASE','R',0);self.assertEqual(c['prompt'],d._v3().public_prompt(t['world']['categories'],t['query']))
        ten=d.make_world('GRADE',0,10);self.assertFalse(d.build_context(ten,'BASE','R',0)['audit']['native_J_applicable'])
        with self.assertRaises((ValueError,AssertionError)):score_response(response(ten),ten['world'],ten['query'])

    def test_prompt_matching_treatment_is_real_and_outcome_blind(self):
        t=d.make_world('MANIP',7);initial=copy.deepcopy(t)
        for ch in ('LIST','REPORT'):
            agreements=[]
            for seed in range(40):
                g=d.build_context(t,ch,'G',seed);target=d.build_context(t,ch,'D',seed)
                self.assertEqual(set(g['public']),{'prompt','assistant_prefill'});self.assertEqual(g['public']['prompt'],g['prompt'])
                self.assertTrue(target['audit']['list_conflict']);agreements.append(g['audit']['list_conflict'])
                mark=lambda z:z['prompt'].replace(json.dumps(z['audit']['list_order']),'<SINGLE LIST>')
                self.assertEqual(mark(g),mark(target));self.assertEqual(g['prompt'].count(d.RANK_PREFIX),1)
                self.assertEqual(target['task'],t);self.assertEqual(g['task'],t)
                self.assertEqual(target['audit']['expected_report_order'] is not None,ch=='REPORT')
                if g['audit']['list_conflict']:self.assertEqual(target['audit']['list_order'],g['audit']['list_order'])
            self.assertEqual(set(agreements),{False,True})
            blind=copy.deepcopy(t);del blind['gold'];del blind['boundary'];del blind['boundary_indices'];del blind['r']
            self.assertEqual(d._order(blind,ch,17,False),d._order(t,ch,17,False))
        self.assertEqual(t,initial)

    def test_2x2_prompt_factors_and_no_list(self):
        t=d.make_world('A',3,10)
        for i in range(8):
            a,b,c,e=[d.build_context(t,ch,'R',i) for ch in ('BASE','CLARIFY','ENUM','CLARIFY_CONFLICT')]
            self.assertEqual(a['prompt']+'\n'+d.CLEAR_SENTENCE,b['prompt']);self.assertEqual(c['prompt']+'\n'+d.CLEAR_SENTENCE,e['prompt']);self.assertEqual(c['audit']['list_order'],e['audit']['list_order']);self.assertTrue(e['audit']['list_conflict'])
        no=d.build_context(t,'NO_LIST','R',0);self.assertIn(d.NO_LIST_LINE,no['prompt']);self.assertNotIn(d.RANK_PREFIX,no['prompt']);self.assertIsNone(no['audit']['list_order'])

    def test_reverse_alphabetic_transfer_is_constructed_not_screened(self):
        for i in range(16):
            t=d.make_world('XFER',i,10);views=d.transfer_views(t);original=copy.deepcopy(t)
            for role,v in views.items():
                self.assertEqual(named_facts(v),named_facts(t));self.assertEqual(v['root_id'],t['root_id']);self.assertEqual(v['gold'],d.oracle_enumerate(v['world'],v['query']))
                x=d.build_context(t,'TRANSFER',role,777);self.assertEqual(x['task'],v);self.assertEqual(x['audit']['report_conflict'],role=='CONFLICT');self.assertEqual(x['audit']['expected_report_order'],sorted(v['world']['categories'],reverse=True))
                self.assertEqual(d.build_context(v,'TRANSFER',role,1)['task'],v);self.assertIsNone(x['audit']['list_order']);self.assertIn(d.NO_LIST_LINE,x['prompt']);self.assertNotIn(d.RANK_PREFIX,x['prompt'])
            self.assertEqual(t,original);self.assertNotEqual(views['ALIGNED']['gold']['answer'],views['CONFLICT']['gold']['answer'])

    def test_swap_factorial_graph_should_change_list_should_not(self):
        for i in range(16):
            t=d.make_world('SWAP',i);panel=d.swap_four(t,33);self.assertEqual(set(panel),{'G0_L0','G0_L1','G1_L0','G1_L1'})
            for gi,li in itertools.product(range(2),repeat=2):
                row=panel[f'G{gi}_L{li}'];other=panel[f'G{1-gi}_L{li}'];flip_list=panel[f'G{gi}_L{1-li}'];view=row['task']
                self.assertEqual(named_facts(view),named_facts(t));self.assertEqual(view['root_id'],t['root_id']);self.assertEqual(row['audit']['list_conflict'],gi!=li)
                self.assertEqual(row['audit']['list_order'],other['audit']['list_order']);self.assertEqual(row['prompt'],other['prompt'])
                self.assertEqual(view['world'],flip_list['task']['world']);self.assertEqual(view['gold'],flip_list['task']['gold']);self.assertNotEqual(row['prompt'],flip_list['prompt'])
                self.assertNotEqual(view['gold']['answer'],other['task']['gold']['answer']);self.assertEqual(len(set(view['gold']['selected'])^set(other['task']['gold']['selected'])),2)

if __name__=='__main__':unittest.main()
