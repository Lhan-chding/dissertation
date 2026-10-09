"""CPU contract for SR-F1: executable queries, exact evidence scoring, reward/advantage APIs.

This is NOT a Qwen/GPU training implementation. All answers are derived from typed
world/query records. Do not pass records containing gold or root IDs to the model.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import re
from decimal import Decimal
from fractions import Fraction
from typing import Any
import numpy as np

VERSION = 'SR-F1-20261009'
FAMILIES = ('CROSS', 'THRESHOLD', 'TOPK', 'INTERVAL')
ARMS = ('A', 'J', 'PART', 'DEC', 'GATE')
EPS = 1e-8


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big') % (2**63-1)


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def fraction_of(x: Any, *, text: bool = False) -> Fraction:
    if isinstance(x, bool) or x is None:
        raise ValueError('Not a numeric value')
    if isinstance(x, (int, Decimal, Fraction)):
        z = Fraction(x)
    elif isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError('Nonfinite')
        z = Fraction(Decimal(str(x)))
    elif text and isinstance(x, str) and re.fullmatch(r'-?\d+(?:/[1-9]\d*)?', x):
        z = Fraction(x)
    else:
        raise ValueError('Unsupported numeric representation')
    if abs(z.numerator) > 10**12 or z.denominator > 10**9:
        raise ValueError('Numeric magnitude exceeds protocol limits')
    return z


def encode_answer(x: Fraction | None) -> int | str | None:
    if x is None:
        return None
    return x.numerator if x.denominator == 1 else f'{x.numerator}/{x.denominator}'


def validate_world(w: dict) -> None:
    cats = w['categories']
    if len(cats) != 4 or len(set(cats)) != 4:
        raise ValueError('Four distinct named categories required')
    for s in ('Alpha', 'Beta'):
        if s not in w['series'] or len(w['series'][s]) != 4:
            raise ValueError('Required series missing')
    for vals in w['series'].values():
        if len(vals) != 4 or any(type(v) is not int or not 0 <= v <= 100 for v in vals):
            raise ValueError('World values must be integers in [0,100]')


def validate_query(q: dict) -> None:
    if q.get('family') not in FAMILIES:
        raise ValueError('Unknown family')
    f = q['family']
    permitted = {
        'CROSS': {'family','comparator','aggregate'},
        'THRESHOLD': {'family','comparator','threshold','aggregate'},
        'TOPK': {'family','direction','k','aggregate'},
        'INTERVAL': {'family','start','end','include_start','aggregate'},
    }
    if set(q) != permitted[f]:
        raise ValueError('Unexpected/missing AST keys')
    if q['aggregate'] not in ('sum','mean','count','range','max','min'):
        raise ValueError('Unsupported aggregation')
    if f in ('CROSS','THRESHOLD') and q['comparator'] not in ('gt','ge'):
        raise ValueError('Unsupported comparison')
    if f == 'THRESHOLD' and (type(q['threshold']) is not int or not 0 <= q['threshold'] <= 100):
        raise ValueError('Bad threshold')
    if f == 'TOPK' and (q['direction'] not in ('largest','smallest') or type(q['k']) is not int or q['k'] != 2):
        raise ValueError('Top/bottom exactly two')
    if f == 'INTERVAL' and (q['start'] != 1 or q['end'] != 3 or type(q['include_start']) is not bool):
        raise ValueError('Interval convention mismatch')


def base_query(family: str, threshold: int = 50) -> dict:
    if family == 'CROSS':
        return dict(family=family, comparator='gt', aggregate='sum')
    if family == 'THRESHOLD':
        return dict(family=family, comparator='ge', threshold=threshold, aggregate='count')
    if family == 'TOPK':
        return dict(family=family, direction='largest', k=2, aggregate='sum')
    if family == 'INTERVAL':
        return dict(family=family, start=1, end=3, include_start=True, aggregate='max')
    raise ValueError(family)


def selected_indices(w: dict, q: dict) -> list[int]:
    validate_world(w); validate_query(q)
    a, b = w['series']['Alpha'], w['series']['Beta']
    f = q['family']
    if f in ('CROSS','THRESHOLD'):
        rhs = b if f == 'CROSS' else [q['threshold']]*4
        return [i for i in range(4) if (a[i] > rhs[i] if q['comparator']=='gt' else a[i] >= rhs[i])]
    if f == 'TOPK':
        # Ties are resolved by category order, which is explicitly printed in prompt.
        key = (lambda i: (-a[i], i)) if q['direction']=='largest' else (lambda i: (a[i], i))
        return sorted(range(4), key=key)[:q['k']]
    return list(range(q['start'] + (not q['include_start']), q['end']+1))


def required_keys(w: dict, q: dict) -> list[tuple[str,str]]:
    validate_world(w); validate_query(q)
    # Evidence means all cells needed to VERIFY the query, not merely contributors to the answer.
    if q['family'] == 'THRESHOLD':
        return [('Alpha', c) for c in w['categories']]
    indices = list(range(4)) if q['family'] in ('CROSS','TOPK') else selected_indices(w,q)
    return [(s,w['categories'][i]) for i in indices for s in ('Alpha','Beta')]


def execute(w: dict, q: dict) -> Fraction | None:
    inds = selected_indices(w, q)
    a, b = w['series']['Alpha'], w['series']['Beta']
    f = q['family']
    vals = [Fraction(b[i] if f=='TOPK' else a[i]-b[i] if f=='INTERVAL' else a[i]) for i in inds]
    agg = q['aggregate']
    if agg == 'count':
        return Fraction(len(vals))
    if agg == 'sum':
        return sum(vals, Fraction(0))
    if not vals:
        return None
    if agg == 'mean': return sum(vals, Fraction(0))/len(vals)
    if agg == 'range': return max(vals)-min(vals)
    if agg == 'max': return max(vals)
    if agg == 'min': return min(vals)
    raise ValueError(agg)


def gold_evidence(w: dict, q: dict) -> list[dict]:
    return [{'series':s, 'category':c, 'value':w['series'][s][w['categories'].index(c)]} for s,c in required_keys(w,q)]


def gold_output(w: dict, q: dict) -> dict:
    return {'evidence': gold_evidence(w,q), 'answer': encode_answer(execute(w,q))}


def composition_query(q: dict) -> dict:
    out = copy.deepcopy(q)
    out['aggregate'] = {'CROSS':'range','THRESHOLD':'mean','TOPK':'mean','INTERVAL':'mean'}[q['family']]
    return out


def operator_variant(q: dict) -> dict:
    out = copy.deepcopy(q)
    out['aggregate'] = {'CROSS':'mean','THRESHOLD':'sum','TOPK':'range','INTERVAL':'min'}[q['family']]
    return out


def ast_skeleton(q: dict) -> str:
    z = dict(q)
    if 'threshold' in z: z['threshold'] = '<threshold>'
    return canonical(z)


def boundary_pair(w: dict, q: dict) -> tuple[dict,dict,dict]:
    w2, q0, q1 = copy.deepcopy(w), copy.deepcopy(q), copy.deepcopy(q)
    f = q['family']
    if f == 'CROSS':
        w2['series']['Alpha'][1] = w2['series']['Beta'][1]
        q0['comparator'], q1['comparator'] = 'gt','ge'
    elif f == 'THRESHOLD':
        w2['series']['Alpha'][1] = q['threshold']
        q0['comparator'], q1['comparator'] = 'gt','ge'
    elif f == 'TOPK':
        q0['direction'], q1['direction'] = 'largest','smallest'
    else:
        q0['include_start'], q1['include_start'] = True,False
    return w2,q0,q1


def relevant_variant(w: dict, q: dict, salt: str) -> dict:
    keys = required_keys(w,q)
    candidates = [(s, w['categories'].index(c),v) for s,c in keys for v in range(10,91)]
    candidates.sort(key=lambda z: stable_seed(salt,*z))
    before = execute(w,q)
    for s,i,v in candidates:
        if v == w['series'][s][i]: continue
        out = copy.deepcopy(w); out['series'][s][i] = v
        if execute(out,q) != before:
            return out
    raise ValueError('No single-cell answer-changing intervention found')


def variants(w: dict, q: dict, salt: str) -> list[tuple[str,dict,dict]]:
    w_extra = copy.deepcopy(w)
    w_extra['series']['Gamma'] = [15+(stable_seed(salt,'gamma',i)%71) for i in range(4)]
    wb,qb0,qb1 = boundary_pair(w,q)
    return [
        ('v0',copy.deepcopy(w),copy.deepcopy(q)),
        ('v1',copy.deepcopy(w),copy.deepcopy(q)),
        ('v2',w_extra,copy.deepcopy(q)),
        ('v3',relevant_variant(w,q,salt),copy.deepcopy(q)),
        ('v4',copy.deepcopy(w),operator_variant(q)),
        ('v5',copy.deepcopy(w),copy.deepcopy(q)),
        ('v6',wb,qb0), ('v7',copy.deepcopy(wb),qb1),
    ]


CONTRACT = '''Report the required chart cells, then the answer. Output one JSON object with exactly two top-level keys: "evidence" and "answer". Each evidence entry has exactly "series", "category", and "value". Use the chart's exact series/category labels. Evidence order does not matter. Do not add duplicates or unrelated cells. For comparisons between Alpha and Beta or ranking Alpha to retrieve Beta, report BOTH series for ALL four categories, including excluded categories. For a threshold applied only to Alpha, report ALL four Alpha cells. For an interval query, report BOTH series for every category inside the stated interval. Values are integer counts in [0,100]. A mean may be an exact integer or an exact fraction string such as "7/3"; do not round it. For an empty selected set, sum and count are 0; mean, range, max and min are null. For one selected value, range is 0. If rankings tie, break ties in the category order printed on the chart. Do not include prose, code fences, or a reasoning trace.'''
PLAIN_CONTRACT = '''Answer the question using the chart. Output only a JSON object with key "answer"; do not output evidence or a reasoning trace. Counts are integers. Express non-integer means as exact fraction strings such as "7/3", not rounded decimals. For an empty set: sum/count are 0; mean/range/max/min are null. A single-value range is 0. Resolve ranking ties by the category order printed on the chart.'''


def question_text(w: dict, q: dict, surface: str='train_a') -> str:
    f,agg = q['family'],q['aggregate']
    op = {'sum':'sum','mean':'arithmetic mean','count':'number','range':'range (maximum minus minimum)','max':'maximum','min':'minimum'}[agg]
    if f=='CROSS':
        comp = 'strictly greater than' if q['comparator']=='gt' else 'at least as large as'
        condition = f'the categories where Alpha is {comp} Beta'
        target = 'the Alpha values'
    elif f=='THRESHOLD':
        comp = 'strictly above' if q['comparator']=='gt' else 'at least'
        condition = f'the categories where Alpha is {comp} {q["threshold"]} counts'
        target = 'the Alpha values'
    elif f=='TOPK':
        condition = f'the two categories with the {q["direction"]} Alpha values'
        target = 'the Beta values at those categories'
    else:
        start,end = w['categories'][q['start']],w['categories'][q['end']]
        condition = f'the categories from {start} through {end}, including both endpoints' if q['include_start'] else f'the categories after {start} and through {end}, excluding {start} and including {end}'
        target = 'the per-category signed differences Alpha minus Beta'
    if agg=='count':
        if surface=='test_c': return f'Count how many categories qualify: consider only {condition}. Return that count.'
        if surface=='train_b': return f'How many categories belong to {condition}?'
        return f'What is the number of categories in {condition}?'
    if surface=='test_c': return f'First restrict attention to {condition}. Then evaluate the {op} of {target} over that restriction.'
    if surface=='train_b': return f'Considering only {condition}, calculate the {op} of {target}.'
    return f'For {condition}, what is the {op} of {target}?'


def model_text(w: dict,q: dict,surface: str='train_a',plain: bool=False) -> str:
    return question_text(w,q,surface)+'\n\n'+(PLAIN_CONTRACT if plain else CONTRACT)


def _no_duplicate_pairs(pairs):
    z={}
    for k,v in pairs:
        if k in z: raise ValueError('Duplicate JSON key')
        z[k]=v
    return z


def load_json_strict(text: str) -> Any:
    if not isinstance(text,str) or len(text.encode())>65536:
        raise ValueError('Output not a bounded string')
    return json.loads(text, parse_float=Decimal, parse_int=int,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')),
                      object_pairs_hook=_no_duplicate_pairs)


def score(text: str,w: dict,q: dict) -> dict:
    """Scoring uses gold on the verifier side only. No searching/repair/rounding to gold.
    Missing evidence does NOT erase an otherwise evaluable answer; malformed whole
    JSON, duplicate keys or unknown top-level keys invalidate the whole record.
    """
    out=dict(L_json=0,L_evidence=0,L_answer=0,E=0,p_read=0.0,P=0,A=0,J=0,
             C=None, extra_count=0,missing_count=len(required_keys(w,q)),duplicate_count=0,
             reason='unparsed')
    gold_ans=execute(w,q); keys=required_keys(w,q)
    truth={(e['series'],e['category']):Fraction(e['value']) for e in gold_evidence(w,q)}
    try:
        obj=load_json_strict(text)
        if not isinstance(obj,dict) or not set(obj).issubset({'evidence','answer'}):
            raise ValueError('Wrong object/root keys')
        out['L_json']=1
    except (ValueError,TypeError,OverflowError,RecursionError,json.JSONDecodeError):
        out['reason']='invalid_json'; return out
    if 'answer' in obj:
        try:
            ans=None if obj['answer'] is None else fraction_of(obj['answer'],text=True)
            out['L_answer']=1; out['A']=int(ans==gold_ans)
        except (ValueError,TypeError,ZeroDivisionError,OverflowError):
            ans='invalid'
    else: ans='invalid'
    evidence={}; seen=set()
    try:
        es=obj['evidence']
        if not isinstance(es,list) or len(es)>32: raise ValueError('Wrong evidence list')
        for item in es:
            if not isinstance(item,dict) or set(item)!= {'series','category','value'}:
                raise ValueError('Wrong evidence fields')
            s,c=item['series'],item['category']
            if not isinstance(s,str) or not isinstance(c,str): raise ValueError('Wrong entity keys')
            key=(s,c)
            if key in seen:
                out['duplicate_count']+=1; raise ValueError('Duplicate entity')
            seen.add(key)
            v=fraction_of(item['value'])
            if v.denominator!=1 or not 0<=v<=100: raise ValueError('Evidence out of domain')
            evidence[key]=v
        out['L_evidence']=1
        out['extra_count']=len(set(evidence)-set(truth))
        out['missing_count']=len(set(truth)-set(evidence))
        out['E']=int(set(evidence)==set(truth))
        out['p_read']=sum(evidence.get(key)==truth[key] for key in truth)/len(truth)
        out['P']=int(out['E'] and all(evidence[k]==truth[k] for k in truth))
        if out['E'] and out['L_answer']:
            predicted=copy.deepcopy(w)
            for (s,c),v in evidence.items():
                predicted['series'][s][w['categories'].index(c)]=int(v)
            out['C']=int(ans==execute(predicted,q))
    except (KeyError,ValueError,TypeError,ZeroDivisionError,OverflowError):
        out['L_evidence']=out['E']=out['P']=0;out['p_read']=0.0
    out['J']=int(out['P'] and out['A'])
    out['reason']='scored'
    return out


def zscore(r: np.ndarray) -> np.ndarray:
    r=np.asarray(r,dtype=np.float64)
    if r.ndim !=1 or not np.isfinite(r).all(): raise ValueError('Bad reward vector')
    if len(r)==0: raise ValueError('Empty group')
    if np.all(r==r[0]): return np.zeros_like(r)
    return (r-r.mean())/(r.std(ddof=0)+EPS)


def gate_one(j: np.ndarray,e: np.ndarray,p: np.ndarray,kappa: float=.25) -> tuple[np.ndarray,str]:
    j,e,p=map(lambda x:np.asarray(x,dtype=float),(j,e,p))
    if not 0<=kappa<1: raise ValueError('kappa outside [0,1)')
    if np.any(j) and not np.all(j):
        a=zscore(j); failed=(j==0); h=.5*e*(1+p)
        u=h[failed]-h[failed].mean(); scale=np.max(np.abs(u))
        if scale>0:
            m=np.min(-a[failed]); a[failed]+=kappa*m*u/scale
        return a,'target_J_refined'
    if np.all(j): return np.zeros_like(j),'all_joint_correct'
    if np.ptp(e)>0: return zscore(e),'evidence'
    if np.all(e==1) and np.ptp(p)>0: return zscore(p),'readings'
    return np.zeros_like(j),'no_semantic_contrast'


def reward_advantages(arm: str,scores: list[list[dict]], beta: float=.5,kappa: float=.25) -> tuple[np.ndarray,dict]:
    if arm not in ARMS: raise ValueError(arm)
    if not scores or len({len(g) for g in scores})!=1: raise ValueError('Ragged groups')
    a=np.asarray([[s['A'] for s in g] for g in scores],dtype=float)
    j=np.asarray([[s['J'] for s in g] for g in scores],dtype=float)
    e=np.asarray([[s['E'] for s in g] for g in scores],dtype=float)
    p=np.asarray([[s['p_read'] for s in g] for g in scores],dtype=float)
    if not (np.isin(a,[0,1]).all() and np.isin(j,[0,1]).all() and np.isin(e,[0,1]).all() and np.isfinite(p).all() and ((0<=p)&(p<=1)).all()):
        raise ValueError('Invalid component scores')
    if np.any(j>a) or np.any(j>e) or np.any(j>p): raise ValueError('Joint success contract violated')
    h=.5*e*(1+p)
    info={'arm':arm,'group_size':a.shape[1],'prompt_count':a.shape[0], 'branches':[], 'std_convention':'population'}
    if arm=='A': r=a;adv=np.stack([zscore(row) for row in r])
    elif arm=='J':r=j;adv=np.stack([zscore(row) for row in r])
    elif arm=='PART':
        if not 0<beta<1: raise ValueError('beta outside (0,1)')
        r=j+beta*(1-j)*h;adv=np.stack([zscore(row) for row in r])
    elif arm=='DEC':
        # GDPO-style: per-component, per-prompt normalisation, then full-batch normalisation.
        # Weights (1,.5,.5) match the scalar PART components before normalisation.
        # The final policy-update scaling is still the GDPO-style batch normalisation.
        components=np.stack([j,e,e*p],axis=-1)
        norm=np.empty_like(components)
        for b in range(a.shape[0]):
            for k in range(3): norm[b,:,k]=zscore(components[b,:,k])
        weights=np.array([1.0,.5,.5])
        pre=(norm*weights).sum(axis=-1)
        adv=zscore(pre.reshape(-1)).reshape(pre.shape)
        r=None
        info['component_weights']=weights.tolist()
        info['component_std']=components.std(axis=1).tolist()
        info['normalised_components']=norm.tolist()
        info['pre_batch_advantage']=pre.tolist()
        info['pre_batch_mean']=float(pre.mean());info['pre_batch_std']=float(pre.std())
    else:
        rows=[gate_one(j[b],e[b],p[b],kappa) for b in range(a.shape[0])]
        adv=np.stack([x[0] for x in rows]); info['branches']=[x[1] for x in rows];r=None
    info['scalar_reward']=None if r is None else r.tolist()
    info['final_advantage']=adv.tolist()
    # Never normalise or clip the returned advantage again inside the GPU trainer.
    return adv,info


def credit_audit(base: np.ndarray,new: np.ndarray,event: np.ndarray) -> dict:
    base,new,event=map(lambda x:np.asarray(x,dtype=float),(base,new,event))
    if base.shape != new.shape or base.shape!=event.shape: raise ValueError('Shape mismatch')
    within=0.0
    for value in np.unique(base):
        cells=new[base==value];within+=len(cells)*float(cells.var())/len(base)
    return {'within_coarse_ties':within,'W_plus':float((np.maximum(new,0)*event).mean()),
            'W_minus':float((np.maximum(-new,0)*event).mean())}
