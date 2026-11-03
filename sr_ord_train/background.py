"""Public CPU constructor. No model data or outcomes are used."""
import sys, random, json
from pathlib import Path
from fractions import Fraction
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor/native_src'))
from srf6_ref import core, design
from sr_f6.interface_v3_protocol import public_prompt
from srf5_ref.oracle import evaluate_a
FAMILIES=('BINDING','IMPACT','SIGNED','INTERVAL','K_DIRECTION')

def normalize(world, query):
    a=core.oracle_a(world,query); b=core.oracle_b(world,query); n=evaluate_a(world,query)
    if a!=b or set(a['selected'])!=set(n['selected']) or a['answer']!=n['answer']:raise ValueError('oracle disagreement')
    return {**a,'answer':core.enc(a['answer']),'operands':[core.enc(x) for x in a['operands']]}

def background(family,i):
    if family not in FAMILIES:raise ValueError(family)
    rid=f'OT1_TRAIN_BG_{family}_{i:03d}';q=design.base_query(family,i,8)
    for attempt in range(256):
        rng=random.Random(core.seed_for(2026110304,rid,attempt))
        w={'categories':rng.sample(list(core.LABELS),8),'domain':[10,90],'series':{s:rng.sample(design.GRID,8) for s in ('Alpha','Beta')}}
        if family=='INTERVAL':
            w['series']['Alpha'][3:5]=[24,72];w['series']['Beta'][3:5]=[60,30]
        if family=='IMPACT' and i%3==0:
            idx=core.select_a(w,q); key=q['selection_series'];ascending=q['direction']=='smallest'; ranked=sorted(range(8),key=lambda j:w['series'][key][j],reverse=not ascending)
            w['series']['Beta'][ranked[q['k']]]=w['series']['Beta'][ranked[q['k']-1]]
        a=normalize(w,q)
        if a['answer'] is None or not core.finite_decimal(Fraction(a['answer'])):continue
        return dict(root_id=rid,qid=rid,world=w,query=q,gold=a,structure=family,family=family,attempt=attempt)
    raise ValueError('background exhausted '+rid)

def retention():
    c=json.loads((BASE/'history/historical_sr_f6.json').read_text())
    for fam in core.FAMILIES:
        for i in range(4):
            ret=design.make_root('DEV_EVAL',fam,i,c['data']['seeds']['DEV_EVAL'])
            for row in ret['tasks']['P0']:
                w,q=row['world'],row['query'];gold=normalize(w,q)
                if gold['answer']!=row['gold_answer'] or gold['selected']!=row['gold_selected']: raise ValueError('historical P0 mismatch')
                yield {k:row[k] for k in ('root_id','qid','world','query','variant','family')}|{'structure':fam,'gold':gold}
