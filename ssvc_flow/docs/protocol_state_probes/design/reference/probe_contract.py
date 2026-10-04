"""Deterministic reference contracts, NOT a GPU runner or a learned observer.

Units: canonical variable order is a,b,c,d; an output order maps emission
positions to canonical positions. Truth and DPE labels never enter render().
"""
from __future__ import annotations
import hashlib
import json
import math
from fractions import Fraction
from typing import Sequence
from ptlc_reference import run, one_row, transform_star, external

NAMES = 'abcd'
FWD = (0, 1, 2, 3)
REV = (3, 2, 1, 0)
PROTOCOLS = ('O0', 'A1', 'B0', 'B1', 'BR', 'L00', 'L01', 'L10', 'L11')
SYSTEM = ('You are repairing a four-integer chart record. Return only one JSON array\n'
          'containing four integers from 0 to 99. Do not include an explanation.')
OPERATIONS = {'sum4': 'a+b+c+d', 'difference_pairs': '(a+b)-(c+d)',
              'range4': 'max(a,b,c,d)-min(a,b,c,d)'}
PUBLIC_KEYS = {'observed', 'H', 'b', 'family', 'operation', 'original_system', 'original_user'}

def permutation(order):
    order = tuple(order)
    if len(order) != 4 or any(type(k) is not int for k in order) or sorted(order) != list(FWD):
        raise ValueError('Expected a permutation of 0,1,2,3')
    return order

def emit(canonical, order):
    order = permutation(order)
    if len(canonical) != 4: raise ValueError('Four coordinates required')
    return [canonical[k] for k in order]

def restore(emitted, order):
    order = permutation(order)
    if len(emitted) != 4: raise ValueError('Four coordinates required')
    y = [None] * 4
    for value, k in zip(emitted, order): y[k] = value
    return y

def parse_four_ints(raw):
    """No extraction, JSON repairs, rounding, booleans, or domain clipping."""
    if not isinstance(raw, str): return None
    try: y = json.loads(raw)
    except (ValueError, TypeError, RecursionError): return None
    if not isinstance(y, list) or len(y) != 4 or any(type(v) is not int for v in y): return None
    return y

def valid_domain(y):
    return y is not None and len(y) == 4 and all(type(v) is int and 0 <= v <= 99 for v in y)

def value(world, operation):
    a,b,c,d = world
    if operation == 'sum4': return a+b+c+d
    if operation == 'difference_pairs': return a+b-c-d
    if operation == 'range4': return max(world)-min(world)
    raise ValueError('Unknown operation')

def event_label(raw, order, truth, operation):
    emitted = parse_four_ints(raw)
    if emitted is None: return 'I', None
    y = restore(emitted, order)
    if not valid_domain(y): return 'I', y
    if list(y) == list(truth): return 'X', y
    return ('S' if value(y,operation)==value(truth,operation) else 'W'), y

def anchors(H, order):
    order = permutation(order)
    return [k for k in order if not one_row(H,k,order)]

def dpe(H, j, order):
    return one_row(H, j, permutation(order))

def relation_values(H,b,y):
    if y is None: return None
    return [sum(Fraction(a)*Fraction(v) for a,v in zip(row,y)) == Fraction(rhs)
            for row,rhs in zip(H,b)]

def matrix_from_scene(scene):
    cue=scene['cue']; fam=cue['family']
    if fam == 'trend': return [[1,-2,1,0],[0,1,-2,1]],[0,0]
    if fam == 'cross_series':
        H=[]; b=[]
        for i,j,s in sorted(cue['edges']):
            row=[0]*4;row[i]=row[j]=1;H.append(row);b.append(s)
        return H,b
    if fam == 'duplicate_encoding':
        row=[0]*4;row[cue['known_index']]=1
        return [row],[cue['known_value']]
    raise ValueError('Unsupported family')

def public_from_record(record):
    scene=record['scene']; H,b=matrix_from_scene(scene)
    return dict(observed=list(scene['observed_world']), H=H, b=b,
                family=record['family'], operation=record['operation'],
                original_system=record['prompt']['system'], original_user=record['prompt']['user'])

def strict_public(public):
    if set(public) != PUBLIC_KEYS:
        raise ValueError('Renderer accepts PUBLIC_KEYS only; truth/j/DPE/outputs forbidden')
    if len(public['observed']) != 4 or not valid_domain(public['observed']):
        raise ValueError('Observed world must be four integers in 0..99')
    if public['operation'] not in OPERATIONS: raise ValueError('Unknown operation')
    if len(public['H']) != len(public['b']) or any(len(row)!=4 for row in public['H']):
        raise ValueError('Bad equation shape')

def format_linear(row,rhs):
    parts=[]
    ordered=list(zip(NAMES,row))
    # Keep B1 explicitly x_leaf - x_first_leaf, not -x_first_leaf + x_leaf.
    ordered.sort(key=lambda pair: Fraction(pair[1]) < 0)
    for name,coef in ordered:
        coef=Fraction(coef)
        if not coef: continue
        mag=abs(coef)
        term=name if mag==1 else f'{mag}*{name}'
        if not parts: parts.append(('-' if coef<0 else '')+term)
        else: parts.append((' - ' if coef<0 else ' + ')+term)
    return ''.join(parts)+' = '+str(Fraction(rhs))

def _b0_order(H,b):
    _,_,meta=transform_star(H,b,FWD)
    c,f=meta['center'],meta['first_leaf']
    byleaf={next(k for k,a in enumerate(row) if a and k!=c):(row,rhs)
            for row,rhs in zip(H,b)}
    leaves=[f]+[k for k in FWD if k not in (c,f)]
    return [list(byleaf[k][0]) for k in leaves],[byleaf[k][1] for k in leaves]

def _replace_once(text,old,new):
    if text.count(old)!=1: raise ValueError('Template span missing or ambiguous: '+old)
    return text.replace(old,new,1)

def render(public, protocol):
    """Minimal edits of frozen original messages. Returns no truth/scene metadata."""
    strict_public(public)
    if protocol not in PROTOCOLS: raise ValueError('Unknown protocol')
    H=[list(row) for row in public['H']];b=list(public['b'])
    user=public['original_user'];sys=public['original_system']; order=FWD
    if protocol in ('A1','L01','L11'): order=REV
    if order==REV:
        user=_replace_once(user,'Return [a,b,c,d], not the downstream answer.',
                                'Return [d,c,b,a], not the downstream answer.')
    if protocol.startswith('L'):
        input_order=REV if protocol in ('L10','L11') else FWD
        old='The observed record is '+json.dumps(public['observed'],separators=(',',':'))+'.'
        new='The observed record is '+', '.join(f'{NAMES[k]}={public["observed"][k]}' for k in input_order)+'.'
        user=_replace_once(user,old,new)
    if protocol in ('B0','B1','BR'):
        if public['family']!='cross_series': raise ValueError('B0/B1 apply only to sum-star cross')
        if protocol=='B0':H,b=_b0_order(H,b)
        elif protocol=='BR':H,b=list(reversed(H)),list(reversed(b))
        else:H,b,_=transform_star(H,b,FWD)
        prefix='The following relationships are reliable:\n'; suffix='\nThe downstream calculation is:'
        if user.count(prefix)!=1 or user.count(suffix)!=1:raise ValueError('Equation span ambiguous')
        before,remaining=user.split(prefix);old,after=remaining.split(suffix)
        user=before+prefix+'\n'.join(format_linear(row,rhs) for row,rhs in zip(H,b))+suffix+after
    if sys!=SYSTEM:raise ValueError('Unknown parent system; review, do not silently rewrite')
    return dict(system=sys,user=user,output_order=list(order),H_display=H,b_display=b)

def predict(public,protocol):
    text=render(public,protocol)
    out={}
    for variant in ('V1','V2'):
        y,trace=run(text['H_display'],text['b_display'],public['observed'],text['output_order'],variant=variant)
        out[variant]={'canonical':external(y),'emitted':external(emit(y,text['output_order'])),
                      'integer_prediction':all(x.denominator==1 for x in y),
                      'in_domain':all(x.denominator==1 and 0<=x<=99 for x in y),'trace':trace}
    return out

def diagnostic_features(raw,public,truth,protocol):
    rendered=render(public,protocol);order=rendered['output_order']
    event,y=event_label(raw,order,truth,public['operation'])
    parsed=y is not None; valid=event!='I';pred=predict(public,protocol)
    obs=public['observed'];a=anchors(rendered['H_display'],order)
    matches={v:bool(parsed and pred[v]['integer_prediction'] and y==pred[v]['canonical']) for v in pred}
    copy=bool(parsed and y==obs)
    orig=relation_values(public['H'],public['b'],y)
    disp=relation_values(rendered['H_display'],rendered['b_display'],y)
    bitmask=int(copy)+2*int(matches['V1'])+4*int(matches['V2'])
    result={'event':event,'parse_four_ints':parsed,'domain_valid':valid,'canonical_world':y,
            'copy_parsed':copy,'match_V1':matches['V1'],'match_V2':matches['V2'],
            'match_mask':bitmask if parsed else None,
            'outside_copy_ptlc_union':bool(parsed and bitmask==0),
            'anchor_mask':a,'anchor_all_parsed':bool(parsed and all(y[k]==obs[k] for k in a)),
            'C_orig':sum(orig)/len(orig) if valid else 0.0,
            'C_display':sum(disp)/len(disp) if valid else 0.0,
            'C_alg_orig':sum(orig)/len(orig) if parsed else None,
            'C_alg_display':sum(disp)/len(disp) if parsed else None,
            'F':None,'B':None,'M':None}
    if valid:
        js=[k for k in FWD if truth[k]!=obs[k]]
        if len(js)!=1:raise ValueError('Repair audit requires exactly one corruption')
        j=js[0]
        result.update(F=int(y[j]==truth[j]),B=sum(y[k]!=truth[k] for k in FWD if k!=j),
                      M=sum(y[k]!=obs[k] for k in FWD))
    return result

def sample_seed(experiment_id, checkpoint_id, case_id, draw_index, role='main'):
    if type(draw_index) is not int or draw_index<0:raise ValueError('Bad draw index')
    s=json.dumps([experiment_id,checkpoint_id,case_id,role,draw_index],ensure_ascii=False,separators=(',',':'))
    return int(hashlib.sha256(s.encode()).hexdigest()[:16],16) % (2**63-1)

def contrast(counts,n,weights):
    """Fixed-panel point estimate; independent stream plug-in MC SE is descriptive."""
    if len(counts)!=len(n) or len(n)!=len(weights) or not counts:raise ValueError('shape')
    estimate=0.;var=0.;boundary=False
    for x,m,w in zip(counts,n,weights):
        if type(x)is not int or type(m)is not int or not 0<=x<=m or m<2:raise ValueError('counts')
        p=x/m;estimate+=w*p;var+=w*w*p*(1-p)/(m-1);boundary|=x in (0,m)
    return {'estimate':estimate,'plugin_mc_se':math.sqrt(var), 'has_boundary_cells':boundary,
            'plugin_se_is_not_a_coverage_guarantee':True}

def zero_success_upper(n,alpha=.05):
    if type(n)is not int or n<1 or not 0<alpha<1:raise ValueError('bounds')
    return -math.expm1(math.log(alpha)/n)
