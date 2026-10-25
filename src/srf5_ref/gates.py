"""Mechanical feasibility gates, not empirical efficacy or evidence authenticity checks."""
from __future__ import annotations
import math

def _numeric(v):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)

def _at_least(v,threshold):
    return _numeric(v) and v>=threshold

def check_calibration(report):
    errors=[]
    if not isinstance(report,dict):
        return {'status':'FAIL','errors':['report must be an object']}
    if report.get('actual_response_count')!=4608:errors.append('expected 4608 actual calibration responses')
    if not _at_least(report.get('unique_reviewed_responses'),360):errors.append('insufficient deduplicated extraction/span reviews')
    if report.get('unresolved_systematic_measurement_errors',1)!=0:errors.append('unresolved measurement error')
    blocks=report.get('blocks',{})
    if not isinstance(blocks,dict):blocks={}
    for b in ('b0','b1','b2'):
        r=blocks.get(b,{})
        if not isinstance(r,dict):r={}
        constraints=[('groups',192,'eq'),('known_fraction',.9,'ge'),('minimum_family_known_fraction',.8,'ge'),
            ('mask_success_given_known',.99,'ge'),('nonzero_groups',20,'ge'),('swappable_groups',10,'ge'),
            ('changed_groups',5,'ge'),('families_with_signal',4,'ge'),('roots_with_signal',12,'ge'),('truncation_fraction',.02,'le')]
        for key,threshold,op in constraints:
            value=r.get(key)
            passed=_numeric(value) and ((value==threshold) if op=='eq' else (value>=threshold) if op=='ge' else (value<=threshold))
            if not passed:errors.append(f'{b}: {key} must be {op} {threshold}')
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'scope':'mechanical G2 criteria; independently verify raw evidence'}

def check_pilot(report):
    errors=[]
    if not isinstance(report,dict):
        return {'status':'FAIL','errors':['report must be an object']}
    if report.get('actual_response_count')!=23040:errors.append('expected 23040 pilot responses')
    if report.get('numerical_and_runtime_checks')!='PASS':errors.append('numerical/runtime checks missing')
    blocks=report.get('blocks',{})
    if not isinstance(blocks,dict):blocks={}
    for b in ('b0','b1','b2'):
        r=blocks.get(b,{})
        if not isinstance(r,dict):r={}
        windows=r.get('windows',[])
        if not isinstance(windows,list):windows=[]
        if len(windows)!=3:errors.append(f'{b}: three four-step windows required')
        for k,w in enumerate(windows):
            if not isinstance(w,dict):w={}
            if not _at_least(w.get('LOCAL_nonzero_groups'),4):errors.append(f'{b}/{k}: insufficient LOCAL intervention')
            if not _at_least(w.get('SHUF_changed_groups'),2):errors.append(f'{b}/{k}: insufficient SHUF contrast')
        if not _at_least(r.get('signal_roots'),12) or not _at_least(r.get('signal_families'),4):errors.append(f'{b}: narrow signal support')
        if not _at_least(r.get('S_known_fraction'),.9):errors.append(f'{b}: observation collapse')
        if r.get('verified_intervention_resampling_intervention') is not True:errors.append(f'{b}: no verified on-policy feedback sequence')
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'scope':'G3 feasibility, not performance significance'}
