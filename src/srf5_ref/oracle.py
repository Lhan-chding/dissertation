"""Two independent task evaluators; neither consumes a model's answer.
Oracle A sorts rows. Oracle B enumerates admissible subsets and verifies ranks.
Supported semantics intentionally stay within the previous TOPK/INTERVAL AST.
"""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations
from typing import Any


def validate(world: dict, query: dict) -> None:
    if set(world) != {'categories','series','domain'}:
        raise ValueError('unknown/missing world field')
    cats=world['categories']; n=len(cats)
    if n not in (6,8) or len(set(cats))!=n or not all(isinstance(c,str) and c for c in cats):
        raise ValueError('category schema')
    if world['domain'] != [10,90] or set(world['series']) != {'Alpha','Beta'}:
        raise ValueError('world domain/series')
    if any(len(v)!=n or any(type(x) is not int or not 10<=x<=90 for x in v) for v in world['series'].values()):
        raise ValueError('world values')
    base={'family','value','aggregate'}
    fields={'TOPK':base|{'selection_series','k','direction'},
            'INTERVAL':base|{'start','end','include_start','include_end','positive_only'}}
    if query.get('family') not in fields or set(query)!=fields[query['family']]:
        raise ValueError('unsupported query field; never silently ignore extensions')
    if query['value'] not in ('Alpha','Beta','diff') or query['aggregate'] not in ('sum','mean'):
        raise ValueError('operand/aggregate')
    if query['family']=='TOPK':
        if query['selection_series'] not in ('Alpha','Beta') or type(query['k']) is not int or not 1<=query['k']<=n:
            raise ValueError('rank column/k')
        if query['direction'] not in ('largest','smallest'):
            raise ValueError('direction')
    else:
        if any(type(query[x]) is not int for x in ('start','end')) or not 0<=query['start']<=query['end']<n:
            raise ValueError('printed positions')
        if any(type(query[x]) is not bool for x in ('include_start','include_end','positive_only')):
            raise ValueError('boolean endpoint/filter flags')


def evaluate_a(world:dict, query:dict) -> dict:
    validate(world,query)
    a,b=world['series']['Alpha'],world['series']['Beta']; n=len(a)
    if query['family']=='TOPK':
        vs=world['series'][query['selection_series']]
        sign=-1 if query['direction']=='largest' else 1
        inds=sorted(range(n),key=lambda i:(sign*vs[i],i))[:query['k']]
    else:
        lo=query['start']+int(not query['include_start']); hi=query['end']-int(not query['include_end'])
        inds=[i for i in range(lo,hi+1) if not query['positive_only'] or a[i]>b[i]]
    vals=[Fraction(a[i] if query['value']=='Alpha' else b[i] if query['value']=='Beta' else a[i]-b[i]) for i in inds]
    answer=sum(vals,Fraction(0)) if query['aggregate']=='sum' else (sum(vals,Fraction(0))/len(vals) if vals else None)
    return {'indices':sorted(inds),'selected':[world['categories'][i] for i in sorted(inds)],'answer':answer}


def evaluate_b(world:dict, query:dict) -> dict:
    # Separate selection implementation, no calls to A, its sorting, or its operand helper.
    validate(world,query)
    rows=[(i,world['categories'][i],world['series']['Alpha'][i],world['series']['Beta'][i]) for i in range(len(world['categories']))]
    if query['family']=='TOPK':
        col=2 if query['selection_series']=='Alpha' else 3
        def precedes(x,y):
            if x[col]==y[col]: return x[0]<y[0]
            return x[col]>y[col] if query['direction']=='largest' else x[col]<y[col]
        matches=[]
        for candidate in combinations(rows,query['k']):
            ids={r[0] for r in candidate}
            if all(precedes(x,y) for x in candidate for y in rows if y[0] not in ids):
                matches.append(candidate)
        if len(matches)!=1: raise AssertionError('rank definition not unique')
        chosen=list(matches[0])
    else:
        chosen=[]
        for r in rows:
            left=(r[0]>query['start'] or (r[0]==query['start'] and query['include_start']))
            right=(r[0]<query['end'] or (r[0]==query['end'] and query['include_end']))
            if left and right and (not query['positive_only'] or r[2]>r[3]): chosen.append(r)
    total=Fraction(0)
    for r in chosen:
        if query['value']=='Alpha': total+=Fraction(r[2])
        elif query['value']=='Beta': total+=Fraction(r[3])
        else: total+=Fraction(r[2])-Fraction(r[3])
    answer=total if query['aggregate']=='sum' else (total/Fraction(len(chosen)) if chosen else None)
    ids=sorted(r[0] for r in chosen)
    return {'indices':ids,'selected':[world['categories'][i] for i in ids],'answer':answer}


def encode(x: Fraction|None) -> int|str|None:
    if x is None:return None
    return x.numerator if x.denominator==1 else f'{x.numerator}/{x.denominator}'


def required_certificate_keys(world:dict, query:dict, facts:dict) -> set|None:
    """Sufficiency must use submitted facts to establish selection, not hidden gold sets.
    This is a sound, deliberately incomplete template. Historical scorer remains canonical
    in the server integration and must be cross-checked on these supported tasks.
    """
    n=len(world['categories'])
    if query['family']=='TOPK':
        col=query['selection_series']; ranking={(col,i) for i in range(n)}
        if not ranking<=facts.keys():return None
        sign=-1 if query['direction']=='largest' else 1
        chosen=sorted(range(n),key=lambda i:(sign*facts[(col,i)],i))[:query['k']]
        required=set(ranking)
    else:
        lo=query['start']+int(not query['include_start']); hi=query['end']-int(not query['include_end'])
        candidate=list(range(lo,hi+1)); required=set()
        if query['positive_only']:
            required={(c,i) for i in candidate for c in ('Alpha','Beta')}
            if not required<=facts.keys():return None
            chosen=[i for i in candidate if facts['Alpha',i]>facts['Beta',i]]
        else:chosen=candidate
    cols=('Alpha','Beta') if query['value']=='diff' else (query['value'],)
    required|={(c,i) for i in chosen for c in cols}
    return required


def canonical_evidence(world:dict,query:dict)->list:
    allfacts={(c,i):v for c,vs in world['series'].items() for i,v in enumerate(vs)}
    required=required_certificate_keys(world,query,allfacts)
    assert required is not None
    return [{'series':c,'category':world['categories'][i],'value':world['series'][c][i]} for c,i in sorted(required)]


def boundary_tie(world:dict, query:dict)->bool:
    if query['family']!='TOPK' or query['k']==len(world['categories']):return False
    vals=sorted(world['series'][query['selection_series']],reverse=query['direction']=='largest')
    return vals[query['k']-1]==vals[query['k']]


def heldout_tags(world:dict,q:dict)->list[str]:
    """Global exclusion predicates: scan ALL training families, not only named cells."""
    tags=[]
    if q['family']=='TOPK':
        if q['selection_series']=='Beta' and q['value']=='Alpha' and q['aggregate']=='mean':tags.append('H_BIND')
        if q['selection_series']=='Alpha' and q['value']=='diff' and q['direction']=='smallest' and q['aggregate']=='mean':tags.append('H_DIFF_BOTTOM')
        if boundary_tie(world,q) and q['direction']=='smallest':tags.append('H_TIE_BOTTOM')
        if q['value']=='diff' and q['k']==4 and q['aggregate']=='mean':tags.append('H_DIFF_K4')
        if q['k']==4 and q['direction']=='smallest' and q['aggregate']=='mean':tags.append('H_K4_BOTTOM_MEAN')
    elif q['positive_only'] and q['aggregate']=='mean':tags.append('H_INTERVAL_FILTER_MEAN')
    return tags
