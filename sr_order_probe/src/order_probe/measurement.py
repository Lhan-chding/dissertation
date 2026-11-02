"""Frozen field-local outcomes and order diagnostics; extraction never receives gold.
No answer recovery from work, last-good-JSON search, or gold-guided repair.
"""
from __future__ import annotations
import json
from decimal import Decimal
from .common import exact
from .inputs import prefix_orders


def _unique(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError('DUPLICATE_JSON_KEY')
        out[k]=v
    return out


def _nonfinite(_): raise ValueError('NONFINITE_JSON')


def extract(text: str) -> tuple[dict | None, str | None]:
    if not isinstance(text,str) or len(text)>100000: return None,'TEXT_BOUND'
    try:
        obj=json.loads(text,object_pairs_hook=_unique,parse_float=Decimal,parse_constant=_nonfinite)
        if not isinstance(obj,dict): raise ValueError('TOP_LEVEL_NOT_OBJECT')
        return obj,None
    except (ValueError,TypeError,ArithmeticError,RecursionError) as exc:
        return None,str(exc)


def first_object_end(text: str) -> int | None:
    """Character endpoint for stop-scanner tests. Must scan prefill + suffix."""
    if not isinstance(text,str): raise TypeError('text required')
    depth=0; quoted=False; escaped=False; begun=False
    for i,ch in enumerate(text):
        if not begun:
            if ch.isspace(): continue
            if ch!='{': raise ValueError('Assistant response must begin with object')
            begun=True; depth=1; continue
        if quoted:
            if escaped: escaped=False
            elif ch=='\\': escaped=True
            elif ch=='"': quoted=False
            continue
        if ch=='"': quoted=True
        elif ch in '{[': depth+=1
        elif ch in '}]':
            depth-=1
            if depth<0: raise ValueError('Unbalanced delimiters')
            if depth==0: return i+1
    return None


def _shape(obj: dict) -> bool:
    required={'evidence','selected','bindings','aggregate','answer'}
    try:
        if not obj.keys()>=required or set(obj)-required-{'work'}: return False
        if 'work' in obj and not isinstance(obj['work'],str): return False
        if not isinstance(obj['selected'],list) or not all(type(x)is str for x in obj['selected']): return False
        if not isinstance(obj['evidence'],list) or not isinstance(obj['bindings'],list): return False
        for x in obj['evidence']:
            if not isinstance(x,dict) or set(x)!={'series','category','value'}: return False
            if type(x['series'])is not str or type(x['category'])is not str: return False
            exact(x['value'])
        for x in obj['bindings']:
            if not isinstance(x,dict) or set(x)!={'category','operand'} or type(x['category'])is not str: return False
            exact(x['operand'])
        if obj['aggregate'] not in ('sum','mean'): return False
        if obj['answer'] is not None: exact(obj['answer'])
        return True
    except (ValueError,TypeError,ArithmeticError): return False


def measure(text: str, task: dict, condition: str, finish_reason: str) -> dict:
    obj,error=extract(text)
    result={'A':0,'A_parseable':0,'S':None,'S_hit':0,'S_known':0,'B':None,
            'V_probe':0,'json_valid':int(obj is not None),'complete':int(finish_reason in ('balanced','eos')),
            'parse_error':error,'S_reason':None,'B_reason':None,'selected':None,
            'rank_order_known':0,'rank_values_true':None,'order_reason':None,
            'rank_order_print':None,'rank_order_alphabetical':None,'rank_order_numeric_world':None,
            'boundary_order_print':None,'boundary_first_r_gold':None,
            'own_order_predicts_selected':None,'boundary_choice_admissible':None,
            'reversed_boundary_choice_hit':0,'enumeration_choice_hit':0,'alphabetical_choice_hit':0,
            'order_compliance':None,'forced_evidence':int(condition.startswith('P_'))}
    if obj is None:
        result['S_reason']=result['B_reason']=error
        return result
    result['V_probe']=int(_shape(obj))
    try:
        value=exact(obj['answer']); result['A_parseable']=1
        result['A']=int(value==exact(task['gold']['answer']))
    except (ValueError,TypeError,KeyError,ArithmeticError): pass
    if not result['complete']:
        result['S_reason']=result['B_reason']='INCOMPLETE_GENERATION'
        return result
    names=task['world']['categories']; gold=set(task['gold']['selected'])
    selected=obj.get('selected')
    if isinstance(selected,list) and all(type(v)is str for v in selected):
        result['selected']=selected
        unique=len(selected)==len(set(selected))
        result['S']=int(unique and set(selected)==gold); result['S_hit']=result['S']; result['S_known']=1
        admissible=(unique and len(selected)==task['query']['k'] and set(selected)<=set(names))
        ids={names.index(n) for n in selected if n in names}
        strict=set(task['strictly_better_indices']); boundary=set(task['boundary_indices'])
        result['boundary_choice_admissible']=int(admissible and strict<=ids and ids<=strict|boundary)
        def selection_from(order):
            tied=[i for i in order if i in boundary][:task['required_from_boundary']]
            return {names[i] for i in strict|set(tied)}
        candidates={
            'reversed_boundary_choice_hit':selection_from(prefix_orders(task)['P_REVTIE']),
            'enumeration_choice_hit':selection_from(task['enum_permutation']),
            'alphabetical_choice_hit':selection_from(sorted(range(len(names)),key=lambda i:names[i]))}
        for key,target in candidates.items(): result[key]=int(admissible and set(selected)==target)
    else: result['S_reason']='SELECTED_SHAPE_OR_MISSING'
    try:
        entries=obj['bindings']
        if not isinstance(entries,list): raise ValueError('BINDINGS_SHAPE')
        got=[]
        for e in entries:
            if not isinstance(e,dict) or set(e)!={'category','operand'} or type(e['category'])is not str:
                raise ValueError('BINDING_ENTRY')
            got.append((e['category'],exact(e['operand'])))
        target=dict(zip(task['gold']['selected'],map(exact,task['gold']['operands'])))
        result['B']=int(len(got)==len(dict(got)) and dict(got)==target)
    except (KeyError,ValueError,TypeError,ArithmeticError) as exc: result['B_reason']=str(exc)
    try:
        entries=obj['evidence']
        if not isinstance(entries,list): raise ValueError('EVIDENCE_SHAPE')
        rank=[]
        for e in entries:
            if not isinstance(e,dict): raise ValueError('EVIDENCE_ENTRY')
            if e.get('series')=='Alpha':
                if type(e.get('category'))is not str or e['category'] not in names:
                    raise ValueError('RANK_CATEGORY')
                rank.append((names.index(e['category']),e.get('value')))
        ids=[i for i,_ in rank]
        if len(ids)!=len(names) or len(set(ids))!=len(names): raise ValueError('RANK_NOT_FULL_UNIQUE')
        result['rank_order_known']=1
        try: result['rank_values_true']=int(all(exact(v)==task['world']['series']['Alpha'][i] for i,v in rank))
        except (ValueError,TypeError,ArithmeticError): result['rank_values_true']=None
        result['rank_order_print']=int(ids==list(range(len(names))))
        result['rank_order_alphabetical']=int(ids==sorted(range(len(names)),key=lambda i:names[i]))
        vals=[task['world']['series']['Alpha'][i] for i in ids]
        result['rank_order_numeric_world']=int(vals==sorted(vals,reverse=task['query']['direction']=='largest'))
        boundary=set(task['boundary_indices']); ordered=[i for i in ids if i in boundary]
        result['boundary_order_print']=int(ordered==sorted(boundary))
        r=task['required_from_boundary']
        chosen=set(task['strictly_better_indices'])|set(ordered[:r])
        result['boundary_first_r_gold']=int(chosen==set(task['gold']['indices']))
        if result['S_known']:
            result['own_order_predicts_selected']=int(len(selected)==len(set(selected)) and set(selected)=={names[i] for i in chosen})
        if condition=='C_PRINT': result['order_compliance']=result['rank_order_print']
        if condition=='C_ALT': result['order_compliance']=result['rank_order_alphabetical']
    except (ValueError,TypeError,KeyError) as exc: result['order_reason']=str(exc)
    return result
