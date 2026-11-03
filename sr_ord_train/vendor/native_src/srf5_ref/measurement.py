"""Gold-blind structured extraction followed by gold-aware verification.
No regex harvesting of category names from prose. UNKNOWN is not a semantic FAIL.
Server baseline J must use the preserved native scorer through a v3 wrapper;
this independent reference accepts a sound subset of the old certificate templates.
"""
from __future__ import annotations
import json, math, re
from decimal import Decimal
from fractions import Fraction
from .oracle import evaluate_a, required_certificate_keys


def rational(value):
    if isinstance(value,bool) or value is None:raise ValueError('not numeric')
    if isinstance(value,Decimal):
        if not value.is_finite() or abs(value.adjusted())>30:raise ValueError('decimal bounds')
        out=Fraction(value)
    elif isinstance(value,(int,Fraction)):
        out=Fraction(value)
    elif isinstance(value,float):
        if not math.isfinite(value):raise ValueError('not finite')
        out=Fraction(str(value))
    elif isinstance(value,str):
        if len(value)>64 or not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?|[+-]?\d+\s*/\s*[+-]?\d+',value.strip()):raise ValueError('invalid numeric grammar')
        out=Fraction(value.replace(' ',''))
    else:raise ValueError('invalid numeric type')
    if abs(out.numerator)>10**12 or out.denominator>10**9:raise ValueError('rational bounds')
    return out


def _unique(pairs):
    d={}
    for key,value in pairs:
        if key in d:raise ValueError('duplicate JSON key')
        d[key]=value
    return d


def parse_response(text:str)->dict:
    """Input signature intentionally has no world, query, or gold parameter."""
    result={'status':'UNKNOWN','reason':None,'object':None,'selected':None,'selected_span':None}
    try:
        if not isinstance(text,str) or len(text)>64000:raise ValueError('text bounds')
        decoder=json.JSONDecoder(object_pairs_hook=_unique,parse_float=Decimal,
             parse_constant=lambda _:(_ for _ in ()).throw(ValueError('non-finite JSON')))
        start=len(text)-len(text.lstrip());obj,end=decoder.raw_decode(text,start)
        if text[end:].strip() or not isinstance(obj,dict):raise ValueError('one JSON object required')
        if set(obj) not in ({'evidence','selected','answer'},{'evidence','work','selected','answer'}):raise ValueError('top-level keys')
        if not isinstance(obj['evidence'],list):raise ValueError('evidence type')
        if 'work' in obj and not isinstance(obj['work'],str):raise ValueError('work type')
        if not isinstance(obj['selected'],list) or not all(type(x) is str for x in obj['selected']):raise ValueError('selected type')
        if not (obj['answer'] is None or type(obj['answer']) in (str,int,Decimal)):raise ValueError('answer type')
        for entry in obj['evidence']:
            if not isinstance(entry,dict) or set(entry)!={'series','category','value'}:raise ValueError('evidence entry schema')
            if not isinstance(entry['series'],str) or not isinstance(entry['category'],str):raise ValueError('fact identifiers')
        # Scan top-level raw spans, skipping nested objects/strings via JSON decoder.
        pos=start+1;span=None
        while True:
            while text[pos].isspace():pos+=1
            if text[pos]=='}':break
            key,kend=decoder.raw_decode(text,pos);pos=kend
            while text[pos].isspace():pos+=1
            if text[pos]!=':':raise ValueError('missing colon')
            pos+=1
            while text[pos].isspace():pos+=1
            lo=pos;_,pos=decoder.raw_decode(text,pos)
            if key=='selected':span=[lo,pos]
            while text[pos].isspace():pos+=1
            if text[pos]==',':pos+=1;continue
            if text[pos]=='}':break
            raise ValueError('top-level separator')
        if span is None or text[span[0]:span[1]][0]!='[':raise ValueError('missing array span')
        result.update(status='KNOWN',reason='CANONICAL_SELECTED_FIELD',object=obj,selected=list(obj['selected']),selected_span=span)
    except (ValueError,TypeError,IndexError,ArithmeticError,RecursionError) as e:
        result['reason']=str(e)
    return result


def verify_selection(extraction:dict,expected:list[str])->str:
    if extraction['status']!='KNOWN':return 'UNKNOWN'
    got=extraction['selected']
    # Duplicate elements, extra labels, wrong cardinality and empty arrays are semantic failures,
    # not structural invalidity. A correct order permutation is accepted.
    normalized=[x.strip().casefold() for x in got]
    target={x.strip().casefold() for x in expected}
    return 'PASS' if len(normalized)==len(set(normalized)) and set(normalized)==target else 'FAIL'


def score_response(text,world,query,finished=True):
    e=parse_response(text);gold=evaluate_a(world,query)
    res=dict(V=int(e['status']=='KNOWN'),finish=int(finished),P='UNKNOWN',S=verify_selection(e,gold['selected']),
      certificate_truth=0,certificate_sufficient=0,value_correct=0,value_parseable=0,J=0,
      selected_span=e['selected_span'],reason=e['reason'])
    if e['status']!='KNOWN':return res
    obj=e['object']
    try:
        answer=None if obj['answer'] is None else rational(obj['answer'])
        res['value_parseable']=1;res['value_correct']=int(answer==gold['answer'])
    except (ValueError,TypeError,ZeroDivisionError,OverflowError):pass
    try:
        facts={};cats=world['categories']
        for ent in obj['evidence']:
            col=ent['series'];cat=ent['category']
            if col not in ('Alpha','Beta') or cat not in cats:raise ValueError('unknown fact key')
            key=(col,cats.index(cat))
            if key in facts:raise ValueError('duplicate fact key')
            facts[key]=rational(ent['value'])
        truth=all(x==world['series'][c][i] for (c,i),x in facts.items())
        res['P']='PASS' if facts and truth else 'FAIL' if facts else 'UNKNOWN'
        need=required_certificate_keys(world,query,facts)
        res['certificate_truth']=int(truth)
        res['certificate_sufficient']=int(need is not None and need<=facts.keys())
    except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError):res['P']='UNKNOWN'
    res['J']=int(res['V'] and res['finish'] and res['value_correct'] and res['certificate_truth'] and res['certificate_sufficient'])
    return res


def map_token_mask(token_ids,text,span,decode,prefill=''):
    """Decode original sampled IDs only; no re-tokenization of an edited response.
    decode must disable cleanup and use the same special-token convention as storage.
    Boundary-crossing tokens are excluded. Prefix instability causes abstention.
    """
    empty=[False]*len(token_ids)
    if span is None:return empty,'NO_SPAN'
    decoded=decode(token_ids)
    if prefill+decoded!=text:return empty,'FULL_DECODE_MISMATCH'
    lo,hi=span
    if not 0<=lo<hi<=len(text):return empty,'INVALID_SPAN'
    last='';mask=[]
    for i in range(len(token_ids)):
        now=decode(token_ids[:i+1])
        if not now.startswith(last) or not decoded.startswith(now):return empty,'UNSTABLE_PREFIX'
        a,b=len(prefill)+len(last),len(prefill)+len(now)
        mask.append(b>a and lo<=a and b<=hi);last=now
    return mask,'MAPPED' if any(mask) else 'NO_CONTAINED_TOKENS'
