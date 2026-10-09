import copy
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
from semantic_contract import *

W={'categories':['January','February','March','April'],'series':{'Alpha':[30,42,26,17],'Beta':[20,50,14,21]},'unit':'count'}
Q=base_query('CROSS')

def out(w=W,q=Q):return gold_output(w,q)
def check(o,w=W,q=Q):return score(json.dumps(o),w,q)

def test_example_sum():assert execute(W,Q)==56

def test_example_mean():assert execute(W,operator_variant(Q))==28

def test_evidence_all_eight_not_contributors():assert len(required_keys(W,Q))==8

def test_numeric_change():
    w=copy.deepcopy(W);w['series']['Alpha'][1]=60
    assert execute(w,Q)==116

def test_boundary_pair():
    w,q0,q1=boundary_pair(W,Q)
    assert execute(w,q0)==56 and execute(w,q1)==106

def test_topk():assert execute(W,base_query('TOPK'))==70

def test_threshold_count():assert execute(W,base_query('THRESHOLD',30))==2

def test_interval_signed_max():assert execute(W,base_query('INTERVAL'))==12

def test_rank_tie_order():
    w=copy.deepcopy(W);w['series']['Alpha']=[90,90,90,20]
    assert selected_indices(w,base_query('TOPK'))==[0,1]

@pytest.mark.parametrize('agg,expected',[('sum',0),('count',0),('mean',None),('range',None),('max',None),('min',None)])
def test_empty_contract(agg,expected):
    q=base_query('THRESHOLD',99);q['aggregate']=agg
    assert execute(W,q)==expected

def test_single_range_zero():
    q=base_query('THRESHOLD',40);q['aggregate']='range'
    assert execute(W,q)==0

def test_mean_fraction():
    q=base_query('THRESHOLD',20);q['aggregate']='mean'
    assert execute(W,q)==Fraction(98,3)

def test_perfect():
    s=check(out());assert (s['E'],s['P'],s['A'],s['J'],s['C'])==(1,1,1,1,1)

def test_reorder_evidence():
    o=out();o['evidence'].reverse();assert check(o)['J']==1

def test_missing_evidence_answer_survives():
    s=check({'answer':56});assert s['A']==1 and s['J']==0

def test_missing_answer_evidence_survives():
    o=out();del o['answer'];s=check(o);assert s['P']==1 and s['A']==0

def test_contributor_only_is_not_full_evidence():
    o=out();o['evidence']=[v for v in o['evidence'] if v['series']=='Alpha' and v['category'] in ('January','March')]
    s=check(o);assert s['E']==0 and s['J']==0 and s['A']==1

def test_extra_entity():
    o=out();o['evidence'].append({'series':'Gamma','category':'January','value':50})
    s=check(o);assert s['E']==0 and s['extra_count']==1 and s['J']==0

def test_duplicate_entity():
    o=out();o['evidence'].append(o['evidence'][0]);assert check(o)['L_evidence']==0

def test_duplicate_root_key():assert score('{"answer":1,"answer":56}',W,Q)['L_json']==0

def test_duplicate_inner_key():assert score('{"evidence":[{"value":1,"value":2}],"answer":56}',W,Q)['L_json']==0

@pytest.mark.parametrize('bad',['NaN','Infinity','-Infinity'])
def test_nonfinite_json(bad):assert score('{"answer":'+bad+'}',W,Q)['L_json']==0

@pytest.mark.parametrize('bad',[True,False,'56.0','50+6','56/0','5.6e1',{},[]])
def test_bad_answer_type(bad):assert check({'answer':bad})['L_answer']==0

@pytest.mark.parametrize('val',[56,56.0,'56','56/1','112/2'])
def test_equivalent_answer_repr(val):assert check({'answer':val})['A']==1

@pytest.mark.parametrize('bad',[True,False,-1,101,20.5,'30',None])
def test_bad_evidence_value(bad):
    o=out();o['evidence'][0]['value']=bad;assert check(o)['L_evidence']==0

def test_finite_wrong_answer_evaluable():assert check({'answer':50})['L_answer']==1

def test_null_is_not_blanket_success():assert check({'answer':None})['A']==0

def test_empty_null_is_valid_success():
    q=base_query('THRESHOLD',99);q['aggregate']='mean'
    assert check(out(W,q),W,q)['J']==1

def test_no_code_fence_extraction():assert score('```json\n'+json.dumps(out())+'\n```',W,Q)['L_json']==0

def test_unknown_root_key_fatal():
    o=out();o['reasoning']='answer is 56';assert check(o)['L_json']==0

def test_unknown_item_key():
    o=out();o['evidence'][0]['certainty']=1;assert check(o)['E']==0

def test_domain_without_gold_rounding():
    o=out();o['evidence'][0]['value']=31;assert check(o)['P']==0

def test_correct_evidence_wrong_answer_C():
    o=out();o['answer']=57;s=check(o);assert s['P']==1 and s['A']==0 and s['C']==0

def test_threshold_only_A_required():assert len(required_keys(W,base_query('THRESHOLD')))==4

def test_interval_exclusive_evidence():
    q=base_query('INTERVAL');q['include_start']=False
    assert len(required_keys(W,q))==4

def test_wrong_semantic_condition_same_world():
    wb,q0,q1=boundary_pair(W,Q)
    assert score(json.dumps(out(wb,q0)),wb,q1)['J']==0

def test_relevant_intervention_changes_answer():assert execute(relevant_variant(W,Q,'unit'),Q)!=execute(W,Q)

def test_distractor_invariance():
    w=copy.deepcopy(W);w['series']['Gamma']=[5,6,7,8];assert execute(w,Q)==execute(W,Q)

def test_unit_contract_exact_no_tolerance():assert check({'answer':56.001})['A']==0


def ss(a=0,j=0,e=0,p=0):return {'A':a,'J':j,'E':e,'p_read':p}
G=[ss(1,1,1,1)]*2+[ss(0,0,1,.5)]*3+[ss(0,0,0,0)]*3

@pytest.mark.parametrize('arm',ARMS)
def test_reward_finite(arm):
    v,_=reward_advantages(arm,[G,G]);assert np.isfinite(v).all() and np.max(np.abs(v.mean(axis=1)))<1e-12

@pytest.mark.parametrize('arm',ARMS)
def test_all_correct_zero(arm):
    v,_=reward_advantages(arm,[[ss(1,1,1,1)]*8]);assert np.array_equal(v,np.zeros((1,8)))

@pytest.mark.parametrize('arm',ARMS)
def test_all_wrong_same_zero(arm):
    v,_=reward_advantages(arm,[[ss()]*8]);assert np.array_equal(v,np.zeros((1,8)))

def test_gate_mixed_properties():
    a,_=reward_advantages('J',[G]);b,info=reward_advantages('GATE',[G]);
    assert np.allclose(a[0,:2],b[0,:2]);assert (b[0,2:]<0).all();assert b[0,2]>b[0,5]
    assert info['branches']==['target_J_refined']

def test_gate_evidence():
    g=[ss(0,0,1,0)]*4+[ss()]*4
    a,info=reward_advantages('GATE',[g]);assert info['branches']==['evidence'] and a[0,0]>0>a[0,7]

def test_gate_readings():
    g=[ss(0,0,1,1)]*4+[ss(0,0,1,.5)]*4
    a,info=reward_advantages('GATE',[g]);assert info['branches']==['readings'] and a[0,0]>0

def test_D_bottleneck_zero():
    a,info=reward_advantages('GATE',[[ss(0,0,1,1)]*8]);assert not a.any() and info['branches']==['no_semantic_contrast']

def test_gate_no_evidence_condition_no_read_reward():
    g=[ss(0,0,0,.5)]*4+[ss(0,0,0,0)]*4
    a,info=reward_advantages('GATE',[g]);assert not a.any()

def test_gate_zero_kappa_pureJ():
    a,_=reward_advantages('GATE',[G],kappa=0);b,_=reward_advantages('J',[G]);assert np.allclose(a,b)

def test_part_success_ranks_first():
    _,info=reward_advantages('PART',[G]);r=np.array(info['scalar_reward'])[0];assert r[:2].min()==1 and r[2:].max()<1

def test_part_matches_dec_raw_components():
    g=G
    _,info=reward_advantages('PART',[g]);r=np.array(info['scalar_reward'])[0]
    j=np.array([s['J'] for s in g]);e=np.array([s['E'] for s in g]);p=np.array([s['p_read'] for s in g])
    assert np.allclose(r,(j+.5*e+.5*e*p)/2)

def test_dec_fullbatch_includes_zero_groups():
    g0=[ss()]*8
    a,info=reward_advantages('DEC',[G,g0]);assert np.isclose(a.std(),1,atol=1e-7)
    assert np.allclose(a[1],0,atol=1e-12)
    assert info['component_weights']==[1,.5,.5]

def test_new_information_without_new_contrast():
    a=np.array([1,1,1,1,1,1,0,0.]);j=np.array([1,1,1,1,0,0,0,0.])
    assert np.ptp(a)>0 and np.ptp(j)>0
    assert credit_audit(a,zscore(j),j)['within_coarse_ties']>0

def test_kappa_invalid():
    with pytest.raises(ValueError):reward_advantages('GATE',[G],kappa=1)

def test_reward_nan_rejected():
    with pytest.raises(ValueError):reward_advantages('PART',[[ss(p=float('nan'))]*8])

def test_joint_contract_rejected():
    with pytest.raises(ValueError):reward_advantages('J',[[ss(0,1,0,0)]*8])

def test_counterexample_not_guaranteed_improvement():
    scores=np.array([1.,-10.,28/3]);probs=np.array([.25,.375,.375])
    a,_=reward_advantages('J',[G]);b,_=reward_advantages('GATE',[G])
    da=(2*a[0,0]*scores[0]+3*a[0,2]*scores[1]+3*a[0,5]*scores[2])/8
    db=(2*b[0,0]*scores[0]+3*b[0,2]*scores[1]+3*b[0,5]*scores[2])/8
    assert da>0 and db<0 and abs(np.dot(scores,probs))<1e-12

def test_boundary_surface_words_not_equal():
    w,q0,q1=boundary_pair(W,Q)
    assert question_text(w,q0)!=question_text(w,q1)

def test_plain_contract_no_gold_values():
    t=model_text(W,Q,plain=True);assert '56' not in t and '50' not in t

def test_prompt_sidecar_not_dependency():
    w=copy.deepcopy(W);w['hidden_answer']='LEAK'
    assert model_text(w,Q)==model_text(W,Q)

def test_train_test_surface_different():assert question_text(W,Q,'train_a')!=question_text(W,Q,'test_c')

def test_composition_different_aggregation():
    assert composition_query(Q)['aggregate']=='range'
