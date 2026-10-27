"""Gold-blind extraction; task-scoped oracle S and B; execution is diagnostic.
Unknown is None, not a scored failure. Empty/duplicated/extra task claims that
can be unambiguously judged are FAIL. Binding truth for an arbitrary chosen set
is NOT the training B label: B must cover the query's required operands.
"""
from __future__ import annotations
import json
from decimal import Decimal
from .core import exact, enc, oracle_a, reduce_values, value_at

REQUIRED = {'evidence', 'selected', 'bindings', 'aggregate', 'answer'}

def _unique(pairs):
    d = {}
    for k, v in pairs:
        if k in d: raise ValueError('duplicate JSON key')
        d[k] = v
    return d

def parse(text: str) -> dict:
    try:
        if not isinstance(text, str) or len(text) > 100000: raise ValueError('text bound')
        o = json.loads(text, object_pairs_hook=_unique, parse_float=Decimal,
                       parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        if not isinstance(o, dict) or not REQUIRED <= o.keys() or set(o)-REQUIRED-{'work'}: raise ValueError('top-level schema')
        if not isinstance(o['selected'], list) or not all(type(x) is str for x in o['selected']): raise ValueError('selected shape')
        if not isinstance(o['bindings'], list): raise ValueError('binding shape')
        if not isinstance(o['evidence'], list): raise ValueError('evidence shape')
        if o['aggregate'] not in ('sum','mean'): raise ValueError('aggregate shape')
        if 'work' in o and not isinstance(o['work'],str): raise ValueError('work shape')
        for b in o['bindings']:
            if not isinstance(b, dict) or set(b) != {'category','operand'} or not isinstance(b['category'],str): raise ValueError('binding entry')
            exact(b['operand'])  # null/non-numeric is protocol-invalid in every arm; no opt-out of negative B credit
        for e in o['evidence']:
            if not isinstance(e,dict) or set(e) != {'series','category','value'}: raise ValueError('evidence entry')
            if not isinstance(e['series'],str) or not isinstance(e['category'],str): raise ValueError('evidence identifiers')
            exact(e['value'])
        if o['answer'] is not None: exact(o['answer'])
        return {'valid':True,'object':o,'reason':None}
    except (ValueError, TypeError, OverflowError, ArithmeticError, RecursionError) as e:
        return {'valid':False,'object':None,'reason':str(e)}

def answer_only(text: str, expected) -> int:
    """Primary answer score ignores missing/incorrect extra semantic fields.
    Still requires one unambiguous JSON answer; never regex-select a gold-like number.
    """
    try:
        o=json.loads(text,object_pairs_hook=_unique,parse_float=Decimal,
                     parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        return int(isinstance(o,dict) and o.get('answer') is not None and exact(o['answer'])==expected)
    except (ValueError,TypeError,ArithmeticError,RecursionError):return 0

def labels(text: str, w: dict, q: dict, finished: bool=True) -> dict:
    p = parse(text)
    r = {'V':int(p['valid']), 'finished':int(finished), 'S':None,'B':None,'E_raw':None,'E_anchored':None,
         'A':answer_only(text,oracle_a(w,q)['answer']),'P':None,'consistency':None,'reason':p['reason']}
    if not p['valid'] or not finished: return r
    o = p['object']; g = oracle_a(w,q)
    target = dict(zip(g['selected'],g['operands']))
    sel = o['selected']
    r['S'] = int(len(sel)==len(set(sel)) and set(sel)==set(g['selected']))
    if o['answer'] is not None: r['A'] = int(exact(o['answer']) == g['answer'])
    names = [b['category'] for b in o['bindings']]
    r['consistency'] = int(len(names)==len(set(names)) and len(sel)==len(set(sel)) and set(names)==set(sel))
    # A wrong scope is verifiable FAIL; uninterpretable missing operand is UNKNOWN.
    if len(names)!=len(set(names)) or set(names)!=set(target): r['B']=0
    elif any(b['operand'] is None for b in o['bindings']): r['B']=None
    else: r['B']=int(all(exact(b['operand'])==target[b['category']] for b in o['bindings']))
    if o['bindings'] and all(b['operand'] is not None for b in o['bindings']) and o['answer'] is not None:
        val = reduce_values([exact(b['operand']) for b in o['bindings']],o['aggregate'])
        r['E_raw'] = int(val == exact(o['answer']))
    elif not o['bindings']: r['E_raw']=0
    if r['B'] is not None:
        # This is B AND correct operator AND A; not a new independent reward.
        r['E_anchored'] = int(r['B']==1 and o['aggregate']==q['aggregate'] and r['A']==1)
    if o['evidence']:
        try:
            keys=[];truth=True
            for e in o['evidence']:
                key=(e['series'],e['category']); keys.append(key)
                if e['series'] not in w['series'] or e['category'] not in w['categories'] or e['value'] is None: raise ValueError('unknown fact')
                truth &= exact(e['value'])==exact(w['series'][e['series']][w['categories'].index(e['category'])])
            r['P']=int(truth and len(keys)==len(set(keys)))
        except (ValueError,ArithmeticError): r['P']=None
    return r

def full_truth_response(w:dict,q:dict)->str:
    g=oracle_a(w,q)
    return json.dumps({'evidence':[{'series':s,'category':c,'value':enc(w['series'][s][i])} for s in ('Alpha','Beta') for i,c in enumerate(w['categories'])],
        'selected':g['selected'], 'bindings':[{'category':c,'operand':enc(v)} for c,v in zip(g['selected'],g['operands'])],
        'aggregate':q['aggregate'], 'answer':enc(g['answer'])}, ensure_ascii=False)

def legacy_projection(text:str)->str:
    """Projection into F5 M3 schema; production adapter must verify parity."""
    p=parse(text)
    if not p['valid']: raise ValueError(p['reason'])
    o=p['object']; return json.dumps({k:o[k] for k in ('evidence','selected','answer','work') if k in o},default=str)


def natural_answer_score(text:str, expected)->int:
    """Frozen natural-interface panel: one final 'Answer: <exact number>' line.
    No gold-dependent searching; multiple Answer: declarations are rejected.
    """
    import re
    if not isinstance(text,str):return 0
    declarations=list(re.finditer(r'(?im)^\s*Answer\s*:',text))
    if len(declarations)!=1:return 0
    match=re.search(r'(?im)^\s*Answer\s*:\s*([^\n]+)\s*\Z',text)
    if not match:return 0
    try:return int(exact(match.group(1).strip())==expected)
    except (ValueError,TypeError,ArithmeticError):return 0
