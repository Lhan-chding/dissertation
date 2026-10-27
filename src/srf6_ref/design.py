"""Model-outcome-blind paired P0/P1 constructor and finite-class certificates.
P0 samples uniformly within prechosen edit-cost bins. P1 maximizes response-
signature coverage within exactly the SAME bins. P0 is never chosen to fail.
These certificates concern this finite executable candidate library, not a
claim to identify the model's hidden reasoning algorithm.
"""
from __future__ import annotations
import copy, itertools, json, random
from .core import *

GRID=list(range(12,85,6))
TARGET={'BINDING':'H_BIND','IMPACT':'H_DIFF_BOTTOM','TIES':'H_TIE_BOTTOM',
        'SIGNED':'H_DIFF_K4','INTERVAL':'H_INTERVAL_FILTER_MEAN','K_DIRECTION':'H_K4_BOTTOM_MEAN'}

def base_query(family:str,index:int,n:int,ood:bool=False)->dict:
    q={'family':'TOPK','selection_series':'Alpha','value':'Beta','k':3,'direction':'largest','aggregate':'sum'}
    if family=='BINDING':
        q.update(selection_series='Alpha' if index%2==0 else 'Beta',value='Beta' if index%2==0 else 'Alpha',k=2+index%2)
        if ood:q.update(selection_series='Beta',value='Alpha',aggregate='mean')
    elif family=='IMPACT':
        q.update(direction='largest' if index%2==0 else 'smallest',aggregate='sum' if index%3==0 else 'mean')
        if ood:q.update(value='diff',direction='smallest',aggregate='mean')
    elif family=='TIES':
        q.update(direction='smallest' if ood else 'largest',aggregate='sum')
    elif family=='SIGNED':
        q.update(value='diff',k=2+index%2,aggregate='sum' if index%3==0 else 'mean')
        if ood:q.update(k=4,aggregate='mean')
    elif family=='INTERVAL':
        q={'family':'INTERVAL','start':1,'end':n-2,'include_start':bool(index%3),
           'include_end':bool(index%4),'positive_only':bool(index%2==0),'value':'diff',
           'aggregate':'sum' if index%2==0 else 'mean'}
        if ood:q.update(positive_only=True,aggregate='mean',include_start=True,include_end=True)
    elif family=='K_DIRECTION':
        q.update(k=2 if index%2 else 4,direction='smallest' if index%3 else 'largest')
        if index%2:q['aggregate']='mean'
        if ood:q.update(k=4,direction='smallest',aggregate='mean')
    else:raise ValueError(family)
    return q

def wrong_rules(q:dict)->list[str]:
    out=['boundary_swap','wrong_value','wrong_aggregate']
    if q['family']=='TOPK':out+=['wrong_rank','wrong_direction','k_minus','k_plus','reverse_tie']
    else:out+=['toggle_start','toggle_end','toggle_filter']
    if q['value']=='diff':out+=['reverse_difference']
    return out

def wrong_value(w:dict,q:dict,rule:str):
    g=oracle_a(w,q); qq=copy.deepcopy(q)
    if rule=='boundary_swap':
        ids=g['indices'][:]
        outside=[i for i in range(len(w['categories'])) if i not in ids]
        if q['family']=='INTERVAL': outside.sort(key=lambda i:(q['start']<=i<=q['end'],i))
        if not ids or not outside:return None
        ids[-1]=outside[0]
        return reduce_values([value_at(w,q,i) for i in ids],q['aggregate'])
    if rule=='wrong_value':qq['value']='Alpha' if q['value']!='Alpha' else 'Beta'
    elif rule=='wrong_aggregate':qq['aggregate']='mean' if q['aggregate']=='sum' else 'sum'
    elif rule=='wrong_rank':qq['selection_series']='Beta' if q['selection_series']=='Alpha' else 'Alpha'
    elif rule=='wrong_direction':qq['direction']='smallest' if q['direction']=='largest' else 'largest'
    elif rule=='k_minus':qq['k']=max(1,q['k']-1)
    elif rule=='k_plus':qq['k']=min(len(w['categories'])-1,q['k']+1)
    elif rule=='toggle_start':qq['include_start']=not q['include_start']
    elif rule=='toggle_end':qq['include_end']=not q['include_end']
    elif rule=='toggle_filter':qq['positive_only']=not q['positive_only']
    elif rule=='reverse_difference':return reduce_values([-value_at(w,q,i) for i in g['indices']],q['aggregate'])
    elif rule=='reverse_tie':
        v=w['series'][q['selection_series']];sign=-1 if q['direction']=='largest' else 1
        ids=sorted(range(len(v)),key=lambda i:(sign*v[i],-i))[:q['k']]
        return reduce_values([value_at(w,q,i) for i in ids],q['aggregate'])
    elif rule!='boundary_swap':raise ValueError(rule)
    return oracle_a(w,qq)['answer']

def force_boundary_collision(w:dict,q:dict)->dict:
    out=copy.deepcopy(w);g=oracle_a(out,q);ids=g['indices'];outside=[i for i in range(len(out['categories'])) if i not in ids]
    if q['family']=='INTERVAL':outside.sort(key=lambda i:(q['start']<=i<=q['end'],i))
    if not ids or not outside:raise ValueError('collision needs inside/outside')
    a,b=ids[-1],outside[0]
    if q['value']=='Beta':out['series']['Beta'][b]=out['series']['Beta'][a]
    elif q['value']=='Alpha':out['series']['Alpha'][b]=out['series']['Alpha'][a]
    else:out['series']['Beta'][b]=out['series']['Alpha'][b]-(out['series']['Alpha'][a]-out['series']['Beta'][a])
    if any(v<12 or v>84 or v%6 for col in out['series'].values() for v in col):raise ValueError('collision bounds')
    gg=oracle_a(out,q)
    if gg['indices']!=ids or wrong_value(out,q,'boundary_swap')!=gg['answer']:raise ValueError('collision not preserved')
    return out

def valid_task(w:dict,q:dict,ood:bool=False,family:str|None=None)->bool:
    g=oracle_a(w,q)
    if not g['indices'] or len(g['indices'])==len(w['categories']) or g['answer'] is None or not finite_decimal(g['answer']):return False
    tags=heldout_tags(w,q)
    return (TARGET[family] in tags) if ood else not tags

def edit_pool(w:dict,q:dict,ood:bool=False,family:str|None=None,delta:int=6)->list[dict]:
    ga=oracle_a(w,q);rules=wrong_rules(q);base={h:wrong_value(w,q,h) for h in rules}
    pool=[]
    for col in ('Alpha','Beta'):
        for idx in range(len(w['categories'])):
            for d in (-delta,delta):
                ww=copy.deepcopy(w);ww['series'][col][idx]+=d
                if not 12<=ww['series'][col][idx]<=84:continue
                if not valid_task(ww,q,ood,family):continue
                ge=oracle_a(ww,q);dy=ge['answer']-ga['answer'];mask=0;absolute_mask=0
                for k,h in enumerate(rules):
                    v=wrong_value(ww,q,h)
                    if v is None or base[h] is None:continue
                    if v-ge['answer'] != base[h]-ga['answer']:mask|=1<<k
                    if v!=ge['answer']:absolute_mask|=1<<k
                # Matching bins use true task properties, never model scores or candidate-rule IDs.
                desc=(col,1 if d>0 else -1,abs(d),len(ge['indices'])!=len(ga['indices']),enc(dy))
                pool.append({'edit_id':f'{col}:{idx}:{d}','world':ww,'column':col,'index':idx,'delta':d,
                  'descriptor':desc,'response_mask':mask,'absolute_mask':absolute_mask,'answer_delta':enc(dy)})
    return pool

def paired_views(w:dict,q:dict,rng:random.Random,ood:bool=False,family:str|None=None)->dict:
    pool=edit_pool(w,q,ood,family);bins={}
    for p in pool:bins.setdefault(p['descriptor'],[]).append(p)
    if len(bins)<3:raise ValueError('fewer than three edit bins')
    keys=sorted(bins,key=str);rng.shuffle(keys);keys=keys[:3]
    choices=[]
    for key in keys:
        vals=bins[key][:];rng.shuffle(vals);choices.append(vals)
    p0=[rng.choice(vals) for vals in choices]
    # Exact maximization over three small bins. No rerolling of weak baselines.
    best=None;bestscore=-1
    for comb in itertools.product(*choices):
        mask=0
        for e in comb:mask|=e['response_mask']
        score=mask.bit_count()
        if score>bestscore:bestscore=score;best=comb
    p1=list(best);allmask=0
    for e in pool:allmask|=e['response_mask']
    m0=m1=0
    for e in p0:m0|=e['response_mask']
    for e in p1:m1|=e['response_mask']
    rules=wrong_rules(q);base=oracle_a(w,q)
    alias=[h for k,h in enumerate(rules) if (allmask>>k)&1 and wrong_value(w,q,h)==base['answer']]
    active=[h for k,h in enumerate(rules) if (allmask>>k)&1]
    denom=len(active)
    return {'P0':p0,'P1':p1,'rules':rules,'active_rules':active,'base_alias_rules':alias,
            'uncovered_by_pool':[h for h in rules if h not in active],
            'coverage_P0':m0.bit_count()/denom if denom else 0.,'coverage_P1':m1.bit_count()/denom if denom else 0.,
            'coverage_all_registered_P0':m0.bit_count()/len(rules),'coverage_all_registered_P1':m1.bit_count()/len(rules),
            'blind_P1':[h for k,h in enumerate(rules) if (allmask>>k)&1 and not (m1>>k)&1],
            'descriptors':keys,'pool_size':len(pool)}

def make_root(split:str,family:str,index:int,master:int,ood:bool=False,zero_alias:bool=False)->dict:
    rid=f'{split}_{family}_{index:04d}';errors=[]
    forced=(index%3==0 and not ood and split not in ('TEST_ID','DEV_EVAL'))
    for attempt in range(256):
        rng=random.Random(seed_for(master,rid,attempt));n=8 if index%2 or family in ('SIGNED','INTERVAL','K_DIRECTION') else 6
        cats=list(LABELS[:n]);rng.shuffle(cats)
        w={'categories':cats,'series':{s:rng.sample(GRID,n) for s in ('Alpha','Beta')},'domain':[10,90]}
        q=base_query(family,index,n,ood)
        if family=='TIES':
            v=([12,30,30,30,60,72,78,84] if ood else [84,66,66,66,36,24,18,12])[:n];rng.shuffle(v);w['series']['Alpha']=v
        if family=='INTERVAL':
            mid=n//2;w['series']['Alpha'][mid]=72;w['series']['Beta'][mid]=30
            w['series']['Alpha'][mid-1]=24;w['series']['Beta'][mid-1]=60
        try:
            if forced:w=force_boundary_collision(w,q)
            if not valid_task(w,q,ood,family):raise ValueError('base validity or heldout constraints')
            pair=paired_views(w,q,rng,ood,family)
            if not pair['active_rules']:raise ValueError('no active executable candidate')
            if forced and 'boundary_swap' not in pair['base_alias_rules']:raise ValueError('collision lacks distinguishing witness')
            # Zero-alias is only with respect to active members of this declared finite library.
            if zero_alias and pair['base_alias_rules']:raise ValueError('candidate alias at test anchor')
            outputs={}
            for factor in ('P0','P1'):
                views=[{'world':w,'edit_id':'anchor','descriptor':None,'answer_delta':'0'}]+pair[factor]
                tasks=[]
                for v,e in enumerate(views):
                    ww=e['world'];a=oracle_a(ww,q);b=oracle_b(ww,q)
                    if a!=b:raise AssertionError('dual oracle disagreement')
                    if zero_alias and any(wrong_value(ww,q,h)==a['answer'] for h in pair['active_rules']):raise ValueError('candidate alias at test view')
                    tasks.append({'qid':f'{rid}_{factor}_v{v}','root_id':rid,'split':split,'family':family,'factor':factor,'variant':v,
                      'world':ww,'query':q,'gold_answer':enc(a['answer']),'gold_selected':a['selected'],
                      'gold_bindings':dict(zip(a['selected'],map(enc,a['operands']))),'edit_id':e['edit_id'],
                      'edit_descriptor':e['descriptor'],'answer_delta':e['answer_delta'],'heldout_tags':heldout_tags(ww,q),
                      'engineered_collision_anchor':forced,'attempt':attempt})
                outputs[factor]=tasks
            cert={k:v for k,v in pair.items() if k not in ('P0','P1')}
            cert.update(root_id=rid,engineered_collision_anchor=forced,attempt=attempt,rejections=errors)
            return {'root_id':rid,'certificate':cert,'tasks':outputs}
        except ValueError as e:errors.append(str(e))
    raise RuntimeError(f'construction exhausted {rid}: {errors[-8:]}')

def build_bank(split:str,roots_per_family:int,seed:int,ood:bool=False,zero_alias:bool=False)->list[dict]:
    return [make_root(split,f,i,seed,ood,zero_alias) for f in FAMILIES for i in range(roots_per_family)]

def visible_prompt(t:dict)->str:
    w,q=t['world'],t['query'];value={'Alpha':'Alpha','Beta':'Beta','diff':'the signed Alpha minus Beta'}[q['value']]
    if q['family']=='TOPK':s=f"Select the {q['k']} categories with the {q['direction']} {q['selection_series']} values. "
    else:
        s=f"Use the interval from {w['categories'][q['start']]} to {w['categories'][q['end']]} in PRINTED LEFT-TO-RIGHT order. "
        s+=f"{'Include' if q['include_start'] else 'Exclude'} the start and {'include' if q['include_end'] else 'exclude'} the end. "
        if q['positive_only']:s+='Within that interval keep only categories with Alpha strictly greater than Beta. '
    s+=f"Compute the {q['aggregate']} of {value} for the final selected categories. "
    s+='All chart quantities are integer counts in [10,90]. Break ranking ties by printed left-to-right order. '
    s+='Return exactly one JSON object with evidence, selected, bindings, aggregate, answer and optional work. '
    s+='evidence lists enough true chart facts as {"series":...,"category":...,"value":...}. '
    s+='selected lists your FINAL adopted categories after all conditions. '
    s+='bindings lists each adopted category and the task-requested operand as {"category":...,"operand":...}; the order is irrelevant. '
    s+='aggregate is "sum" or "mean". answer is an exact integer, finite decimal, or quoted fraction. '
    s+='work may contain concise free-form working. Do not add other text. '
    return s


def make_classout_root(family:str,index:int,master:int)->dict:
    """New double-cell ±12 edits, sampled uniformly, never coverage-optimized.
    Test queries stay in the non-COMPOSITION grammar to separate edit transfer.
    """
    rid=f'TEST_CLASSOUT_{family}_{index:04d}'
    errors=[]
    for attempt in range(256):
        rng=random.Random(seed_for(master,rid,attempt));n=8 if index%2 or family in ('SIGNED','INTERVAL','K_DIRECTION') else 6
        cats=list(LABELS[:n]);rng.shuffle(cats)
        w={'categories':cats,'series':{c:rng.sample(GRID,n) for c in ('Alpha','Beta')},'domain':[10,90]}
        q=base_query(family,index,n,False)
        if family=='TIES':
            vals=[84,66,66,66,36,24,18,12][:n];rng.shuffle(vals);w['series']['Alpha']=vals
        if not valid_task(w,q):continue
        pool=[]
        cells=[(c,i) for c in ('Alpha','Beta') for i in range(n)]
        for a,b in itertools.combinations(cells,2):
            if a[1]==b[1]:continue
            for da,db in itertools.product((-12,12),repeat=2):
                ww=copy.deepcopy(w);ww['series'][a[0]][a[1]]+=da;ww['series'][b[0]][b[1]]+=db
                if any(v<12 or v>84 for col in ww['series'].values() for v in col):continue
                if not valid_task(ww,q):continue
                pool.append((ww,[[a[0],a[1],da],[b[0],b[1],db]]))
        if len(pool)<3:errors.append('fewer than three valid double edits');continue
        edits=rng.sample(pool,3);tasks=[];ga=oracle_a(w,q)
        for v,(ww,desc) in enumerate([(w,[])]+edits):
            g=oracle_a(ww,q)
            if g!=oracle_b(ww,q):raise AssertionError('CLASSOUT dual oracle disagreement')
            tasks.append({'qid':f'{rid}_P0_v{v}','root_id':rid,'split':'TEST_CLASSOUT','family':family,
               'factor':'P0','variant':v,'world':ww,'query':q,'gold_answer':enc(g['answer']),
               'gold_selected':g['selected'],'gold_bindings':dict(zip(g['selected'],map(enc,g['operands']))),
               'edit_id':'anchor' if v==0 else str(desc),'edit_descriptor':desc,'answer_delta':enc(g['answer']-ga['answer']),
               'heldout_tags':heldout_tags(ww,q),'engineered_collision_anchor':False,'attempt':attempt})
        return {'root_id':rid,'certificate':{'kind':'unoptimized_two_cell_12','attempt':attempt,
                'engineered_collision_anchor':False,'training_edit_catalog_used':False,'rejections':errors},'tasks':{'P0':tasks}}
    raise RuntimeError(f'CLASSOUT exhausted {rid}')


def compound_wrong_value(w:dict,q:dict,kind:int):
    """Held-out DOUBLE query mutations, never used to choose training edits."""
    qq=copy.deepcopy(q)
    if kind not in (0,1,2):raise ValueError('compound candidate kind')
    if q['family']=='TOPK':
        if kind==0:
            qq['selection_series']='Beta' if q['selection_series']=='Alpha' else 'Alpha'
            qq['direction']='smallest' if q['direction']=='largest' else 'largest'
        elif kind==1:
            qq['direction']='smallest' if q['direction']=='largest' else 'largest'
            qq['value']='Alpha' if q['value']!='Alpha' else 'Beta'
        else:
            qq['k']=min(len(w['categories'])-1,q['k']+1)
            qq['aggregate']='mean' if q['aggregate']=='sum' else 'sum'
    else:
        if kind==0:qq['include_start']=not q['include_start'];qq['positive_only']=not q['positive_only']
        elif kind==1:qq['include_end']=not q['include_end'];qq['aggregate']='mean' if q['aggregate']=='sum' else 'sum'
        else:qq['value']='Alpha' if q['value']!='Alpha' else 'Beta';qq['aggregate']='mean' if q['aggregate']=='sum' else 'sum'
    return oracle_b(w,qq)['answer']
