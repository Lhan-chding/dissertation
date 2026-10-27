"""Fixed-dose whole-sequence semantic credit. No online weight fitting.
Conditional centering is a specified surrogate, NOT an unbiased estimator of
J + lambda*S or a guarantee of no J interference in shared neural parameters.
"""
from __future__ import annotations
from collections import defaultdict
import math, random

COMPONENTS=('S','B')

def native_advantage(j:list[int])->list[float]:
    if not j or any(x not in (0,1) for x in j): raise ValueError('binary nonempty J required')
    m=sum(j)/len(j); sd=math.sqrt(sum((x-m)**2 for x in j)/len(j))
    return [0.0]*len(j) if sd==0 else [(x-m)/sd for x in j]

def credit(rows:list[dict],mode:str='COND',lam:float=.5,seed:int=0)->dict:
    if mode not in ('J','COND','GLOBAL','SHUF'): raise ValueError('credit mode')
    if not 0<=lam<=.5: raise ValueError('registered dose bound')
    n=len(rows); a=native_advantage([r['J'] for r in rows]); raw=[0.0]*n
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if not r.get('V') or not r.get('finished'): continue
        mask=tuple(r.get(k) is not None for k in COMPONENTS)
        for k in COMPONENTS:
            if r.get(k) is not None and r[k] not in (0,1): raise ValueError('invalid semantic label')
        if not any(mask): continue
        key=(None if mode=='GLOBAL' else r['J'],mask)
        groups[key].append(i)
    parts=[[0.0]*n for _ in COMPONENTS]
    for (_,mask),idx in groups.items():
        if len(idx)<2:continue
        for c,k in enumerate(COMPONENTS):
            if not mask[c]:continue
            mean=sum(rows[i][k] for i in idx)/len(idx)
            for i in idx: parts[c][i]=(rows[i][k]-mean)/len(COMPONENTS)
    for i in range(n):raw[i]=sum(p[i] for p in parts)
    conditional=[lam*x for x in raw]
    applied=conditional[:]; permutations=[]
    if mode=='SHUF':
        rng=random.Random(seed)
        # Permute the completed credit inside the SAME J/availability stratum.
        # Preserves per-stratum multiset, L1/L2, zero sum, and all UNKNOWN-zero positions.
        for key,idx in groups.items():
            src=idx[:];rng.shuffle(src)
            for dest,origin in zip(idx,src):applied[dest]=conditional[origin]
            permutations.append({'stratum':str(key),'dest':idx,'src':src})
    if mode=='J': applied=[0.0]*n
    return {'native':a,'candidate':conditional,'applied':applied,'total':[x+y for x,y in zip(a,applied)],
            'component_unscaled':parts, 'strata':[list(x) for x in groups.values()],
            'nonzero':any(abs(x)>1e-15 for x in applied),
            'permutation_changed':any(abs(x-y)>1e-15 for x,y in zip(applied,conditional)) if mode=='SHUF' else False,
            'permutations':permutations}
