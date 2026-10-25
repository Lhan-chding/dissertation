"""Versioned five-arm credits. All eight observations remain in the denominator."""
from __future__ import annotations
import math,random
from collections import defaultdict
ARMS=('J','HARD','DENSE','LOCAL','SHUF')

def standardize_binary(js):
    if len(js)!=8 or any(type(x) not in (int,bool) or x not in (0,1) for x in js):raise ValueError('G8 binary rewards required')
    mu=sum(js)/8;sd=math.sqrt(sum((v-mu)**2 for v in js)/8)
    return [0.0]*8 if sd==0 else [(v-mu)/sd for v in js]

def credits(rows,arm,seed,lam=0.25):
    if arm not in ARMS or len(rows)!=8 or lam!=0.25:raise ValueError('unregistered arm/G/lambda')
    if len({r['qid'] for r in rows})!=1:raise ValueError('never center across related questions')
    for r in rows:
        if r['S'] not in ('PASS','FAIL','UNKNOWN') or r['P'] not in ('PASS','FAIL','UNKNOWN'):raise ValueError('label enum')
    js=[r['J'] for r in rows]
    eligible=[i for i,r in enumerate(rows) if r['V'] and r['finish'] and r['S']!='UNKNOWN' and r['mask_ok']]
    true=[0.0]*8;shuf=[0.0]*8;perm=list(range(8));strata=defaultdict(list)
    if len(eligible)>=2:
        mu=sum(rows[i]['S']=='PASS' for i in eligible)/len(eligible)
        for i in eligible:true[i]=lam*(int(rows[i]['S']=='PASS')-mu)
        for i in eligible:strata[(rows[i]['J'],rows[i]['P'])].append(i)
        rng=random.Random(seed)
        for key in sorted(strata):
            dest=strata[key];src=dest[:];rng.shuffle(src)
            for i,j in zip(dest,src):perm[i]=j
        for i in eligible:shuf[i]=lam*(int(rows[perm[i]]['S']=='PASS')-mu)
    native=standardize_binary([r['J']*int(r['S']=='PASS') for r in rows]) if arm=='HARD' else standardize_binary(js)
    aux=true if arm in ('DENSE','LOCAL') else shuf if arm=='SHUF' else [0.0]*8
    swappable=any(len({rows[i]['S'] for i in inds})>1 for inds in strata.values())
    return {'arm':arm,'native':native,'aux':aux,'eligible':eligible,'permutation':perm,
      'scope':'full' if arm=='DENSE' else 'selected' if arm in ('LOCAL','SHUF') else 'none',
      'counterfactual_true':true,'counterfactual_shuffle':shuf,
      'nonzero_group':any(abs(x)>1e-12 for x in true),'swappable_group':swappable,
      'changed_group':any(abs(a-b)>1e-12 for a,b in zip(true,shuf)),
      'all_J_zero':not any(js),'all_J_one':all(js),
      'native_zero':not any(standardize_binary(js)),
      'positive_aux_J0_answer0':sum(aux[i]>0 and rows[i]['J']==0 and rows[i]['value_correct']==0 for i in range(8)),
      'positive_aux_J0_answer1':sum(aux[i]>0 and rows[i]['J']==0 and rows[i]['value_correct']==1 for i in range(8)),
      'positive_aux_J1':sum(aux[i]>0 and rows[i]['J']==1 for i in range(8))}
