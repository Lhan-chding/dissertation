import hashlib,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference'))
from semantic_contract import *
from validate_manifest import independent_answer
from paired_analysis import paired_root_ci
R=[json.loads(s) for s in (ROOT/'manifests/TASKS_GOLD_AUDIT_ONLY.jsonl').read_text().splitlines()]

def test_all_4800_dual_oracle_and_gold():
    assert len(R)==4800
    for r in R:
        assert encode_answer(independent_answer(r['world'],r['query']))==r['answer']
        assert score(canonical(gold_output(r['world'],r['query'])),r['world'],r['query'])['J']==1

def test_root_identity_496():assert len({r['root_id'] for r in R})==496

def test_unique_question_ids():assert len({r['qid'] for r in R})==4800

def test_training_pool_exact():assert len([r for r in R if r['pool']=='TRAIN'])==1536

def test_train_AST_not_composition():
    a={r['ast_skeleton'] for r in R if r['pool']=='TRAIN'}
    b={r['ast_skeleton'] for r in R if r['pool']=='TEST_COMPOSITION'}
    assert len(a)==12 and len(b)==4 and not a&b

def test_test_expression_not_training():assert all(r['surface']!='test_c' for r in R if r['pool']=='TRAIN')

def test_test_style_not_training():assert all(r['style']!='heldout' for r in R if r['pool']=='TRAIN')

def test_schedule_actual_runs():
    runs=json.loads((ROOT/'manifests/RUN_MATRIX.json').read_text());assert len(runs)==15
    assert Counter(r['arm'] for r in runs)==dict.fromkeys(ARMS,3)
    for p in (ROOT/'manifests/schedules').glob('*.jsonl'):
        slots=[json.loads(s) for s in p.read_text().splitlines()]
        assert len(slots)==1536 and len({s['qid'] for s in slots})==1536
        assert set(Counter(s['step'] for s in slots).values())=={16}

def test_diagnostics_not_train():
    d=[json.loads(s) for s in (ROOT/'manifests/DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl').read_text().splitlines()]
    assert len(d)==320 and all(not x['allowed_in_training'] for x in d)
    train={r['qid'] for r in R if r['pool']=='TRAIN'}
    assert not train & {x['qid'] for x in d}

def test_budget_no_time_cutoff():
    c=json.loads((ROOT/'config/SR_F1.json').read_text());assert c['resources']['cumulative_gpu_hours_limit'] is None
    assert not c['resources']['stop_on_elapsed_time']

def test_seed_specific_ci_zero():
    z=np.zeros((3,8));o=paired_root_ci(z,z,['a']*4+['b']*4,20)
    assert o['CI95_pp']==[0.,0.]

def test_ci_positive_shift():
    z=np.zeros((3,8));o=paired_root_ci(z+.1,z,['a']*4+['b']*4,20)
    assert np.allclose(o['CI95_pp'],[10,10])

def test_ci_missing_rejected():
    a=np.zeros((3,8));a[0,0]=np.nan
    with pytest.raises(ValueError):paired_root_ci(a,a,['a']*8)

def test_partial_matches_components_on_real_golds_and_corruptions():
    for r in R[::100]:
        gold=gold_output(r['world'],r['query']);wrong=dict(gold);wrong['answer']=10000
        s1=score(canonical(gold),r['world'],r['query']);s0=score(canonical(wrong),r['world'],r['query'])
        group=[s1]*4+[s0]*4
        _,info=reward_advantages('PART',[group]);raw=info['scalar_reward'][0]
        assert all(abs(raw[i]-(s['J']+.5*s['E']+.5*s['E']*s['p_read'])/2)<1e-12 for i,s in enumerate(group))
