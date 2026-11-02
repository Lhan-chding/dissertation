#!/usr/bin/env python3
"""TEST FIXTURE ONLY: deterministic invented responders, never model evidence.
The fixture always obeys task gold except when it follows a conflicting list/order.
It checks complete raw->score->root-effect plumbing, not the scientific hypothesis.
"""
import copy,json,sys
from pathlib import Path
from fractions import Fraction
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from order_probe.common import read_jsonl,write_json,enc
from order_probe.inputs import prefix_orders
from order_probe.measurement import measure
from order_probe.statistics import root_effects,PRIMARY

def main():
    tasks={t['qid']:t for t in read_jsonl(R/'data/MAIN_TASKS_PRIVATE.jsonl')}
    requests=read_jsonl(R/'data/MAIN_REQUESTS_PUBLIC.jsonl');rows=[]
    for req in requests:
        t=tasks[req['qid']];w=t['world'];c=req['condition'];names=w['categories']
        ids=t['gold']['indices'].copy()
        if c=='C_ENUM':
            tied=[i for i in t['enum_permutation'] if i in t['boundary_indices']][:t['required_from_boundary']]
            ids=sorted(t['strictly_better_indices']+tied)
        elif c=='C_ALT':ids=t['alphabetical_selected_indices']
        elif c.startswith('P_'):ids=sorted(prefix_orders(t)[c][:t['query']['k']])
        values=[w['series']['Beta'][i] for i in ids]
        answer=Fraction(sum(values))
        if t['query']['aggregate']=='mean':answer/=len(ids)
        obj={'selected':[names[i] for i in ids],
             'bindings':[{'category':names[i],'operand':w['series']['Beta'][i]} for i in ids],
             'aggregate':t['query']['aggregate'],'answer':enc(answer)}
        if c.startswith('P_'):
            text=req['assistant_prefill']+json.dumps(obj)[1:]
        else:
            order=sorted(range(10),key=lambda i:names[i]) if c=='C_ALT' else list(range(10))
            obj={'evidence':[{'series':s,'category':names[i],'value':w['series'][s][i]} for s in ('Alpha','Beta') for i in order],**obj}
            text=json.dumps(obj)
        rows.append({k:req[k] for k in ('request_id','qid','root_id','split','parent','condition','branch','draw')}|
                    {'rename':t['rename'],'structure':t['structure'],**measure(text,t,c,'balanced')})
    effects={}
    for metric in ('A','S_hit'):
        for contrast in PRIMARY:
            roots,strata,vals=root_effects(rows,metric,contrast,['b0','b1','b2'])
            assert len(roots)==48 and all(abs(v-1)<1e-12 for v in vals),(metric,contrast)
            effects[metric+'/'+contrast]=float(vals.mean())
    write_json(R/'qa/SYNTHETIC_PIPELINE_ONLY.json',{'status':'PASS_SYNTHETIC_FIXTURE_ONLY',
        'invented_responses_checked':len(rows),'real_model_calls':0,'optimizer_updates':0,
        'expected_contrasts':effects,'warning':'These numbers are deliberately constructed test values, not model results.'})
    print('4320 synthetic fixture responses scored; expected paired effects verified; real model calls 0')
if __name__=='__main__':main()
