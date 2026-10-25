"""Constructive development generator. Labels are not model inputs.
No model-score rejection, no sample selection based on generated responses.
Resampling only repairs predeclared mathematical invariants and is logged.
"""
from __future__ import annotations
import copy, hashlib, json, random
from fractions import Fraction
from .oracle import evaluate_a,evaluate_b,encode,heldout_tags,boundary_tie

FAMILIES=('BINDING','IMPACT','TIES','SIGNED','INTERVAL','K_DIRECTION')
TARGET={'BINDING':'H_BIND','IMPACT':'H_DIFF_BOTTOM','TIES':'H_TIE_BOTTOM',
        'SIGNED':'H_DIFF_K4','INTERVAL':'H_INTERVAL_FILTER_MEAN','K_DIRECTION':'H_K4_BOTTOM_MEAN'}
GRID=list(range(12,85,6))

def seed_for(master:int,*parts)->int:
    s=json.dumps([master,*parts],ensure_ascii=False,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8],'big') & ((1<<63)-1)

def tq(rank='Alpha',value='Beta',k=3,direction='largest',aggregate='sum'):
    return dict(family='TOPK',selection_series=rank,value=value,k=k,direction=direction,aggregate=aggregate)

def iq(n,positive=True,aggregate='sum',inc_start=True,inc_end=True):
    return dict(family='INTERVAL',start=1,end=n-2,include_start=inc_start,include_end=inc_end,
                positive_only=positive,value='diff',aggregate=aggregate)

def _base(rng,n):
    cats=list(('Mug','Lamp','Desk','Vase','Chair','Clock','Tray','Book')[:n]);rng.shuffle(cats)
    return dict(categories=cats,series={'Alpha':rng.sample(GRID,n),'Beta':rng.sample(GRID,n)},domain=[10,90])

def _permuted(w,p):
    return dict(categories=[w['categories'][i] for i in p],series={s:[v[i] for i in p] for s,v in w['series'].items()},domain=[10,90])

def _perturb_payload(w,q,idx,delta):
    out=copy.deepcopy(w)
    # For rank Alpha and diff=Alpha-Beta, changing Beta by -delta changes payload by +delta.
    col='Beta' if q.get('selection_series')=='Alpha' else 'Alpha'
    change=-delta if q['value']=='diff' and col=='Beta' else delta
    out['series'][col][idx]+=change
    if not 10<=out['series'][col][idx]<=90:raise ValueError('bounded perturbation')
    return out

def _four_with_payload(w,q):
    sel=evaluate_a(w,q)['indices']; non=[i for i in range(len(w['categories'])) if i not in sel]
    if not sel or not non:raise ValueError('needs selected and non-selected positions')
    ws=[copy.deepcopy(w),_perturb_payload(w,q,sel[0],6),_perturb_payload(w,q,non[0],6),_perturb_payload(w,q,sel[0],-6)]
    return [(x,copy.deepcopy(q)) for x in ws]

def construct(family,index,rng,ood=False):
    n=8 if family in ('INTERVAL','K_DIRECTION','SIGNED') or index%2 else 6
    w=_base(rng,n)
    if ood:
        if family=='BINDING':return _four_with_payload(w,tq('Beta','Alpha',3,'largest','mean'))
        if family=='IMPACT':return _four_with_payload(w,tq('Alpha','diff',3,'smallest','mean'))
        if family=='SIGNED':return _four_with_payload(w,tq('Alpha','diff',4,'largest','mean'))
        if family=='K_DIRECTION':return _four_with_payload(w,tq('Alpha','Beta',4,'smallest','mean'))
    if family=='BINDING':
        return [(copy.deepcopy(w),tq(r,v,3,'largest' if index%2==0 else 'smallest','sum')) for r,v in [('Alpha','Alpha'),('Alpha','Beta'),('Beta','Alpha'),('Beta','Beta')]]
    if family=='IMPACT':
        q=tq('Alpha','Beta',3,'largest' if index%2==0 else 'smallest','sum' if index%3==0 else 'mean')
        # Controlled pair collision in every third root; not a model-outcome filter.
        order=sorted(range(n),key=lambda i:((-1 if q['direction']=='largest' else 1)*w['series']['Alpha'][i],i))
        if index%3==0:w['series']['Beta'][order[2]]=w['series']['Beta'][order[3]]
        return _four_with_payload(w,q)
    if family=='TIES':
        direction='smallest' if ood else 'largest'
        # k=3 boundary passes through an exactly tied block; payloads remain diverse.
        levels=([12,30,30,30,60,72,78,84] if ood else [84,66,66,66,36,24,18,12])[:n]
        rng.shuffle(levels);w['series']['Alpha']=levels
        q=tq('Alpha','Beta',3,direction,'sum' if not ood else 'mean')
        tievalue=30 if ood else 66
        tiepos=[i for i,x in enumerate(levels) if x==tievalue]
        p=list(range(n));p[tiepos[0]],p[tiepos[-1]]=p[tiepos[-1]],p[tiepos[0]]
        # A true change of tie membership, not a semantics-preserving transformation.
        w2=_permuted(w,p)
        return [(copy.deepcopy(w),copy.deepcopy(q)),(w2,copy.deepcopy(q)),
                (_perturb_payload(w,q,evaluate_a(w,q)['indices'][0],6),copy.deepcopy(q)),
                (_perturb_payload(w2,q,evaluate_a(w2,q)['indices'][0],6),copy.deepcopy(q))]
    if family=='SIGNED':
        return [(copy.deepcopy(w),tq('Alpha','diff',k,'largest',op)) for k,op in [(2,'sum'),(2,'mean'),(3,'sum'),(3,'mean')]]
    if family=='INTERVAL':
        # Interior contains both signs and at least one positive value under every endpoint variant.
        mid=n//2;w['series']['Alpha'][mid]=72;w['series']['Beta'][mid]=30
        neg=mid-1;w['series']['Alpha'][neg]=24;w['series']['Beta'][neg]=60
        pos=True if ood else index%2==0
        op='mean' if ood or not pos else 'sum'
        return [(copy.deepcopy(w),iq(n,pos,op,a,b)) for a,b in [(True,True),(False,True),(True,False),(False,False)]]
    if family=='K_DIRECTION':
        return [(copy.deepcopy(w),tq('Alpha','Beta',k,d,'sum')) for k,d in [(2,'largest'),(2,'smallest'),(4,'largest'),(4,'smallest')]]
    raise ValueError(family)

def terminate_decimal(x):
    if x is None:return False
    d=x.denominator
    for p in (2,5):
        while d%p==0:d//=p
    return d==1

def make_root(split,family,index,seed,ood=False):
    if family not in FAMILIES:raise ValueError('unknown family')
    rid=f'{split}_{family}_{index:04d}'
    failures=[]
    for attempt in range(128):
        rng=random.Random(seed_for(seed,rid,attempt))
        try:
            pairs=construct(family,index,rng,ood)
            tasks=[]
            for v,(world,q) in enumerate(pairs):
                aa=evaluate_a(world,q);bb=evaluate_b(world,q)
                if aa!=bb:raise AssertionError('oracle disagreement')
                if not aa['indices']:raise ValueError('empty core query')
                if not terminate_decimal(aa['answer']):raise ValueError('non-terminating primary numeric answer')
                tags=heldout_tags(world,q)
                if ood and TARGET[family] not in tags:raise ValueError('missing reserved conjunction')
                if not ood and tags:raise ValueError('forbidden conjunction in training/support')
                task=dict(qid=f'{rid}_v{v}',root_id=rid,split=split,cell=family,variant=v,
                          world=world,query=q,gold_answer=encode(aa['answer']),gold_selected=aa['selected'],
                          heldout_tags=tags,construction_attempt=attempt,
                          render_style='r0',generator_version='srf5-cpu-v1')
                tasks.append(task)
            # Root relations are task truth, never inferred from a model's response.
            base=evaluate_a(tasks[0]['world'],tasks[0]['query'])
            for t in tasks:
                ev=evaluate_a(t['world'],t['query'])
                t['relation_to_v0']={'answer_delta':encode(ev['answer']-base['answer']),
                   'selected_same_ids':set(ev['selected'])==set(base['selected']),
                   'note':'oracle-verified relation; not an independent observation'}
                t['construction_rejections']=failures[:]
            return tasks
        except ValueError as e:failures.append(str(e))
    raise RuntimeError(f'construction exhausted for {rid}: {failures[-5:]}')

def make_bank(split,roots_per_family,seed,ood=False):
    out=[]
    for f in FAMILIES:
        for i in range(roots_per_family):out.extend(make_root(split,f,i,seed,ood))
    return out

def world_shape(world):
    # Ignore category spellings but preserve printed order. Also check unordered shape separately.
    return tuple(zip(world['series']['Alpha'],world['series']['Beta']))

def audit_splits(banks):
    by_shape={};by_multiset={};rootmap={};collision=[]
    for split,rows in banks.items():
        for t in rows:
            old=rootmap.setdefault(t['root_id'],split)
            if old!=split:raise AssertionError('root split leakage')
            shape=world_shape(t['world']);unordered=tuple(sorted(shape))
            for table,key,kind in [(by_shape,shape,'printed_world'),(by_multiset,unordered,'renamed_reordered_world')]:
                prev=table.setdefault(key,split)
                if prev!=split:collision.append(dict(kind=kind,from_split=prev,to_split=split,qid=t['qid']))
    if collision:raise AssertionError(f'world collisions across splits: {collision[:5]}')
    return {'roots':len(rootmap),'queries':sum(len(v) for v in banks.values()),'cross_split_world_collisions':0}

def make_interface_bank(roots_per_family,seed):
    """Four orthogonal stress views per root: baseline, render, interface, numeric.
    These are secondary endpoints and must be reported separately by variant.
    """
    rows=[]
    for family in FAMILIES:
        for i in range(roots_per_family):
            base=make_root('test_interface',family,i,seed)[0]
            for v in range(4):
                t=copy.deepcopy(base);t['variant']=v;t['qid']=f"{base['root_id']}_v{v}"
                t['protocol']='natural_work_v2' if v==2 else 'structured_v3'
                t['render_style']='r1' if v==1 else 'r0'
                t['stress_kind']=('reference','render_only','natural_work_interface','off_grid_numeric')[v]
                if v==3:
                    idx=evaluate_a(t['world'],t['query'])['indices'][0]
                    q=t['query'];col='Beta' if q['value'] in ('Beta','diff') else 'Alpha'
                    t['world']['series'][col][idx]+=1
                a=evaluate_a(t['world'],t['query']);b=evaluate_b(t['world'],t['query'])
                assert a==b
                t['gold_answer']=encode(a['answer']);t['gold_selected']=a['selected']
                t['relation_to_v0']={'answer_delta':encode(a['answer']-evaluate_a(base['world'],base['query'])['answer']),
                  'selected_same_ids':set(a['selected'])==set(base['gold_selected']),
                  'note':'one registered stress factor; do not pool with primary COMPOSITION'}
                rows.append(t)
    return rows
