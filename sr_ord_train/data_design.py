"""CPU-only paired TOPK chart worlds, reference-channel interventions, and strict input separation.

Import from the package root. Never serialize the entire returned context as a model request:
`ctx['public']` is the public text projection; render `ctx['task']['world']` separately.
No model, network, performance-dependent filtering, filesystem output, or reward implementation.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

VERSION = 'order-train-data-v1'
MASTER_SEED = 2026110301
STRUCTURES = ((2, 1), (3, 1), (3, 2), (4, 2))
NAMES = ('Book','Bowl','Chair','Clock','Desk','Frame','Lamp','Mug','Plate','Shelf','Spoon','Stool','Table','Tray','Vase','Watch')
GRID = tuple(range(12, 85, 6))
TIE_SENTENCE = 'Break ranking ties by printed left-to-right order.'
CLEAR_SENTENCE = ('Break ranking ties by the category positions on the chart horizontal axis, from left to right. '
                  'The order of a category list and the order in which you write evidence do not change these chart positions.')
CHECKLIST = 'EVIDENCE CHECKLIST FOR THIS QUESTION:'
RANK_PREFIX = '- Ranking: report the original Alpha value for EVERY category in this public list: '
NO_LIST_LINE = '- Ranking: report the original Alpha value for EVERY category shown on the chart.'
OPERAND_LINE = '- Operand: report the original Beta value of each finally selected category in evidence.'
ONCE_LINE = '- A cell already reported for ranking or filtering is reported only once. Values written only in bindings or work do not count as evidence.'
REPORT_INSTRUCTION = "For the Alpha evidence entries, list categories in the exact order of the public list above. Keep the question's tie-breaking rule unchanged."
REVERSE_INSTRUCTION = "For the Alpha evidence entries, list categories in reverse alphabetical order by category label. Keep the question's tie-breaking rule unchanged."
ALT_INSTRUCTION = "For the Alpha evidence entries, list categories in alphabetical order by category label. Keep the question's tie-breaking rule unchanged."
PRINT_INSTRUCTION = "For the Alpha evidence entries, list categories in printed left-to-right order. Keep the question's tie-breaking rule unchanged."


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def _seed(*parts):
    return int(_hash([VERSION, *parts])[:16], 16)


def _structure(s, index):
    if s is None:
        return STRUCTURES[index % len(STRUCTURES)]
    if isinstance(s, str):
        m = re.fullmatch(r'([234])(?:_choose_|x|/)([12])', s)
        if not m:
            raise ValueError('structure must be (m,r) or m_choose_r')
        s = tuple(map(int, m.groups()))
    if not isinstance(s,(tuple,list)) or len(s)!=2 or any(type(z) is not int for z in s):
        raise ValueError('structure must be two integers')
    m,r=s
    if m not in (2,3,4) or r not in (1,2) or r>=m: raise ValueError('need 1 <= r < m, m in {2,3,4}, r in {1,2}')
    return m,r


def _gold(world, query, chosen):
    chosen=sorted(chosen); xs=[world['series']['Beta'][i] for i in chosen]
    answer=Fraction(sum(xs),len(xs) if query['aggregate']=='mean' else 1)
    return {'indices':chosen, 'selected':[world['categories'][i] for i in chosen], 'operands':[str(x) for x in xs], 'answer':str(answer)}


def oracle_sort(world, query):
    """Native task: Alpha TOPK, tie by chart position; sum/mean of selected Betas."""
    if (set(query)!= {'family','selection_series','value','k','direction','aggregate'} or query['family']!='TOPK'
            or query['selection_series']!='Alpha' or query['value']!='Beta' or query['aggregate'] not in ('sum','mean')
            or query['direction'] not in ('largest','smallest') or type(query['k']) is not int
            or not 0<query['k']<=len(world['categories'])):
        raise ValueError('unsupported query, not an extension of native J')
    a=world['series']['Alpha']; sign=-1 if query['direction']=='largest' else 1
    return _gold(world,query,sorted(range(len(a)),key=lambda i:(sign*a[i],i))[:query['k']])


def oracle_enumerate(world, query):
    """Independent selection and arithmetic: enumerate k-subsets, reject any outranked member."""
    a=world['series']['Alpha']; n=len(a); k=query['k']; winners=[]
    for ids in itertools.combinations(range(n),k):
        chosen=set(ids)
        if all((a[i]>a[j] if query['direction']=='largest' else a[i]<a[j]) or (a[i]==a[j] and i<j)
               for i in ids for j in range(n) if j not in chosen):
            winners.append(ids)
    if len(winners)!=1: raise ValueError('independent oracle not unique')
    ix=list(winners[0]); bs=[Fraction(world['series']['Beta'][i]) for i in ix]; y=sum(bs,Fraction())
    if query['aggregate']=='mean': y/=len(bs)
    return {'indices':ix,'selected':[world['categories'][i] for i in ix],'operands':[str(v) for v in bs],'answer':str(y)}


def _derive(task):
    w=task['world'];q=task['query'];cs=w['categories']; a=w['series']['Alpha'];b=w['series']['Beta']; n=len(cs)
    if n not in (8,10) or len(set(cs))!=n or not all(type(x) is str for x in cs): raise ValueError('invalid labels')
    if w['domain']!=[10,90] or len(a)!=n or len(b)!=n or any(type(v) is not int or not 10<=v<=90 for v in a+b): raise ValueError('invalid chart cells')
    g=oracle_sort(w,q);ind=oracle_enumerate(w,q)
    if g!=ind: raise AssertionError('dual oracle mismatch')
    kth=sorted(a,reverse=q['direction']=='largest')[q['k']-1]; ids=[i for i,v in enumerate(a) if v==kth]
    better=[i for i,v in enumerate(a) if (v>kth if q['direction']=='largest' else v<kth)]
    r=q['k']-len(better);m=len(ids);_structure((m,r),0)
    if len(better)!=2: raise ValueError('must have exactly two strictly better objects')
    sums=[sum(b[i] for i in comb) for comb in itertools.combinations(ids,r)]
    if len(sums)!=len(set(sums)): raise ValueError('numeric collision between rank-admissible selections')
    # Where support permits, pre-identify a wholly unselected equal-size tie for optional placebo.
    outside=[i for i in range(n) if i not in set(better+ids)]; places=next(([j for j in outside if a[j]==a[i]] for i in outside if sum(a[j]==a[i] for j in outside)>=m),[])
    task.update(gold=g,boundary_indices=ids,boundary=[cs[i] for i in ids],boundary_categories=[cs[i] for i in ids],
                r=r,required_from_boundary=r,strictly_better_indices=better,strictly_better=[cs[i] for i in better],
                structure=f'{m}_choose_{r}',placebo_indices=places[:m],matched_placebo_available=len(places)>=m,
                world_query_key=_hash({'world':w,'query':q}),
                unlabeled_world_query_key=_hash({'world':dict(w,categories=[f'position_{i}' for i in range(n)]),'query':q}))
    return task


def make_world(split, index, n=8, structure=None, *, master_seed=MASTER_SEED, direction=None, aggregate=None):
    """Generate one private, JSON-compatible base task. Stable split and index; 16-index balanced default cycle."""
    if not isinstance(split,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',split):raise ValueError('unsafe split name')
    if type(index) is not int or index<0 or type(n) is not int or n not in (8,10) or type(master_seed) is not int:raise ValueError('invalid index/n/master_seed')
    m,r=_structure(structure,index); direction=('largest','smallest')[(index//4)%2] if direction is None else direction
    aggregate=('sum','mean')[(index//8)%2] if aggregate is None else aggregate
    if direction not in ('largest','smallest') or aggregate not in ('sum','mean'):raise ValueError('invalid direction/aggregate')
    root=f'OT1_{split}_n{n}_{index:05d}_m{m}r{r}_{direction[0]}{aggregate[0]}_{master_seed}'; rng=random.Random(_seed(master_seed,root,direction,aggregate))
    tie=rng.choice((42,48));lo=[v for v in GRID if v<tie];hi=[v for v in GRID if v>tie]
    good,bad=(hi,lo) if direction=='largest' else (lo,hi); psize=min(m,n-2-m); pv=rng.choice(bad)
    a=rng.sample(good,2)+[tie]*m+[pv]*psize+rng.sample([v for v in bad if v!=pv],n-2-m-psize);rng.shuffle(a)
    bound=[i for i,v in enumerate(a) if v==tie];cs=rng.sample(NAMES,n)
    for attempt in range(4096):
        b=rng.sample(GRID,n); sums=[sum(b[i] for i in k) for k in itertools.combinations(bound,r)]
        if len(sums)==len(set(sums)):break
    else:raise RuntimeError('could not avoid structural answer collision; never filter on model outcomes')
    w={'categories':cs,'series':{'Alpha':a,'Beta':b},'domain':[10,90]}
    q={'family':'TOPK','selection_series':'Alpha','value':'Beta','k':2+r,'direction':direction,'aggregate':aggregate}
    return _derive({'schema':VERSION,'split':split,'root_id':root,'qid':root+'_base','view':'BASE','world':w,'query':q,
                    'structural_attempt':attempt,'master_seed':master_seed,'index':index,'n':n})


def _reorder(t,p,suffix):
    """p[new_position] is old_position. Move label and BOTH values together."""
    z=copy.deepcopy(t);w=t['world'];n=len(w['categories']);
    if sorted(p)!=list(range(n)):raise ValueError('invalid whole-row permutation')
    z['world']={'categories':[w['categories'][i] for i in p], 'series':{k:[vs[i] for i in p] for k,vs in w['series'].items()},'domain':w['domain'][:]}
    z['view']=suffix;z['qid']=t['root_id']+'_'+suffix.lower();return _derive(z)


def transfer_views(task):
    """Two deterministic reverse-alphabetic compatibility views; preserve every named cell, not only value marginals."""
    cs=task['world']['categories'];b=task['boundary_indices'];out={}
    for label,descending in [('ALIGNED',True),('CONFLICT',False)]:
        p=list(range(len(cs)));byname=sorted(b,key=cs.__getitem__,reverse=descending)
        for slot,old in zip(b,byname):p[slot]=old
        t=_reorder(task,p,'TRANSFER_'+label);t['rename']=label;t['report_relation']=label
        reverse=sorted(cs,reverse=True); disag=set(induced_choice(t,reverse))!=set(t['gold']['selected'])
        if disag!=(label=='CONFLICT'):raise AssertionError('reverse alphabetic view relation')
        out[label]=t
    return out


def induced_choice(task, order):
    """Selection that would follow chart Alpha values, but use the proposed name order for tied values."""
    cs=task['world']['categories']; a=task['world']['series']['Alpha']; q=task['query']
    if not isinstance(order,(list,tuple)) or len(order)!=len(cs) or set(order)!=set(cs):raise ValueError('order must list all names once')
    p={nm:j for j,nm in enumerate(order)};sg=-1 if q['direction']=='largest' else 1
    return [cs[i] for i in sorted(range(len(cs)),key=lambda i:(sg*a[i],p[cs[i]]))[:q['k']]]


def _order(task,channel,seed,targeted):
    names=task['world']['categories']; rng=random.Random(_seed(task['root_id'],channel,seed,'name-permutation'))
    for tries in range(1,4097):
        prop=rng.sample(names,len(names))
        # Short-circuit is intentional: the generic sampler never reads the boundary or gold to choose its order.
        if not targeted or set(induced_choice(task,prop))!=set(task['gold']['selected']):return prop,tries
    raise RuntimeError('conditional permutation sampler exhausted its structural cap')


def _v3():
    vendor=Path(__file__).resolve().parent/'vendor'/'native_src'
    if str(vendor) not in sys.path: sys.path.insert(0,str(vendor))
    from sr_f6 import interface_v3_protocol as mod
    return mod


def _base(task):
    w=task['world'];q=task['query'];cs=w['categories'];mod=_v3()
    if len(cs)==8:
        return mod.public_prompt(list(cs),dict(q))  # Real F6 native public template including checklist.
    if len(cs)!=10:raise ValueError('unsupported chart size')
    # Frozen 10-category probe text, same v3 field grammar but no F6 J at 10 categories.
    question=(f"Select the {q['k']} categories with the {q['direction']} Alpha values. Compute the {q['aggregate']} of Beta for the final selected categories. "
              'All chart quantities are integer counts in [10,90]. '+TIE_SENTENCE+' ')
    return question+'\n'+mod.format_contract()+'\n'+'\n'.join([CHECKLIST,RANK_PREFIX+json.dumps(cs,ensure_ascii=False)+'.',OPERAND_LINE,ONCE_LINE])


def _context(t, ch, policy, seed, names, instruction='', clarify=False, tries=0,omit_list=False):
    p=_base(t);cs=t['world']['categories'];old=RANK_PREFIX+json.dumps(cs,ensure_ascii=False)+'.'
    if p.count(old)!=1:raise ValueError('historical ranking checklist differs; review adapter rather than appending a second public list')
    p=p.replace(old,NO_LIST_LINE if omit_list else RANK_PREFIX+json.dumps(names,ensure_ascii=False)+'.',1)
    # Appending a narrow clarification leaves original wording/otherwise identical checklist intact.
    if clarify:p+='\n'+CLEAR_SENTENCE
    if instruction:p+='\n'+instruction
    target=None if omit_list else induced_choice(t,names); pub={'prompt':p,'assistant_prefill':'{'}
    target_report=(list(names) if ch=='REPORT' else sorted(cs,reverse=True) if ch=='TRANSFER' else sorted(cs) if ch=='ALT' else list(cs) if ch=='PRINT' else None)
    audit={'channel':ch,'policy':policy,'seed':seed,'list_order':None if omit_list else list(names),
           'expected_report_order':target_report,'order_attempts':tries,'list_induced_selected':target,
           'list_conflict':None if omit_list else set(target)!=set(t['gold']['selected']),
           'report_conflict':None if target_report is None else set(induced_choice(t,target_report))!=set(t['gold']['selected']),
           'native_J_applicable':len(cs)==8}
    return {'task':t,'prompt':p,'assistant_prefill':'{','public':pub,'audit':audit}


def build_context(task,channel,policy,seed):
    """One model-facing prompt (without gold/diagnostics) plus private material for rendering/scoring/audit.

    Main API: BASE; LIST or REPORT with G/D; ENUM, CLARIFY, CLARIFY_CONFLICT;
    NO_LIST; TRANSFER with CONFLICT/ALIGNED or default CONFLICT. ALT and PRINT are descriptive controls.
    Only TRANSFER can alter chart positions; always render the returned task.world, never the input rather than that return value.
    """
    if type(seed) is not int:raise ValueError('seed must be integer')
    ch=str(channel).upper();ch={'C0':'BASE','C_ENUM':'ENUM','C_REF':'CLARIFY','NOLIST':'NO_LIST','C_NOLIST':'NO_LIST','C_ALT':'ALT','C_PRINT':'PRINT'}.get(ch,ch)
    po=str(policy).upper();t=copy.deepcopy(task)
    if ch=='TRANSFER':
        rel=po if po in ('ALIGNED','CONFLICT') else t.get('report_relation','CONFLICT')
        if rel not in ('ALIGNED','CONFLICT'):raise ValueError('bad TRANSFER compatibility')
        if t.get('report_relation')!=rel:t=transfer_views(t)[rel]
        return _context(t,ch,po,seed,t['world']['categories'],REVERSE_INSTRUCTION,omit_list=True)
    cs=t['world']['categories']
    if ch in ('LIST','REPORT','ENUM','CLARIFY_CONFLICT'):
        targeted=ch in ('ENUM','CLARIFY_CONFLICT') or po in ('D','TARGET','TARGETED','DIAGNOSTIC')
        if ch in ('LIST','REPORT') and not targeted and po not in ('G','GENERIC','RANDOM'):raise ValueError('LIST/REPORT needs policy G or D')
        names,tries=_order(t,'REPORT' if ch=='REPORT' else 'LIST',seed,targeted)
        return _context(t,ch,po,seed,names,REPORT_INSTRUCTION if ch=='REPORT' else '',clarify=ch=='CLARIFY_CONFLICT',tries=tries)
    if ch not in ('BASE','CLARIFY','NO_LIST','ALT','PRINT'):raise ValueError('unknown channel '+ch)
    return _context(t,ch,po,seed,cs,{'ALT':ALT_INSTRUCTION,'PRINT':PRINT_INSTRUCTION}.get(ch,''),clarify=ch=='CLARIFY',omit_list=ch=='NO_LIST')


def swap_four(task,seed):
    """Return G0_L0/G0_L1/G1_L0/G1_L1: fixed-by-name list orders crossed with a named whole-row position swap."""
    if type(seed) is not int:raise ValueError('integer seed required')
    base=copy.deepcopy(task);cs=base['world']['categories'];b=base['boundary_indices'];r=base['r']
    rng=random.Random(_seed(base['root_id'],seed,'position-swap'));i,j=rng.choice(b[:r]),rng.choice(b[r:])
    p=list(range(len(cs)));p[i],p[j]=p[j],p[i]; gs=[_reorder(base,list(range(len(cs))),'SWAP_G0'),_reorder(base,p,'SWAP_G1')]
    lists=[g['world']['categories'][:] for g in gs];out={}
    if set(gs[0]['gold']['selected'])==set(gs[1]['gold']['selected']) or gs[0]['gold']['answer']==gs[1]['gold']['answer']:raise AssertionError('swap failed to change the gold')
    for gi,graph in enumerate(gs):
        for li,names in enumerate(lists):
            ctx=_context(copy.deepcopy(graph),'SWAP','FROZEN',seed,names);ctx.update(graph_swap=gi,list_swap=li)
            ctx['audit'].update(swap_names=[cs[i],cs[j]],swap_positions=[i,j],required_relation='ALIGNED' if gi==li else 'CONFLICT')
            if ctx['audit']['list_conflict']!=(gi!=li):raise AssertionError('swap grid mismatch')
            out[f'G{gi}_L{li}']=ctx
    return out
