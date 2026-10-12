"""Read-only registered diagnostics and fixed baseline credit after unified release.

Usage: PYTHONPATH=src python companion_analysis.py EVIDENCE_DIR OUTPUT_DIR
Requires independently rescored baseline and strictly parsed normalized answer metadata.
No generation, model loading, or training is performed.
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sr_f1.contract import reward_advantages
from sr_f12.protocol import ARMS, BASELINE, SEEDS, object_hash
from sr_f12.analysis import _score

METRICS = ('A', 'E', 'p_read', 'P', 'J')
VIEWS = ('keys_hint', 'gold_values', 'text_values', 'neutral_hint', 'blank_image')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def balanced(rows, metric):
    questions = defaultdict(list)
    for r in rows:
        questions[(r['family'], r['root_id'], r['qid'])].append(r['score'][metric])
    roots = defaultdict(list)
    for (f, root, _), values in questions.items():
        roots[(f, root)].append(float(np.mean(values)))
    families = defaultdict(list)
    for (f, _), values in roots.items():
        families[f].append(float(np.mean(values)))
    return float(np.mean([np.mean(v) for v in families.values()]))


def rate(n, d):
    return dict(numerator=int(n), denominator=int(d), probability=n/d if d else None,
                status='DEFINED' if d else 'UNDEFINED_NO_SUPPORT')


def panel_summary(rr):
    support = [r for r in rr if r['score']['P']==1]
    return dict(records=len(rr), questions=len({r['qid'] for r in rr}),
                roots=len({r['root_id'] for r in rr}),
                metrics={k:balanced(rr,k) for k in METRICS},
                A_failure_given_P=rate(sum(r['score']['A']==0 for r in support),len(support)),
                invalid_json=sum(r['score']['L_json']==0 for r in rr),
                invalid_answer=sum(r['score']['L_answer']==0 for r in rr))


def diagnostics(rows, baseline):
    panels=defaultdict(list)
    for r in baseline:
        if r['pool']=='MONITOR_DIAGNOSTIC_ONLY':panels[(BASELINE,r['view'])].append(r)
    for r in rows:
        if r['pool']=='MONITOR_DIAGNOSTIC_ONLY':
            assert r['model_id']!=BASELINE
            panels[(r['model_id'],r['view'])].append(r)
    models=[BASELINE]+[f'SRF1_2_{a}_s{s}' for a in ARMS for s in SEEDS]
    assert set(panels)=={(m,v) for m in models for v in VIEWS}
    absolute=[];changes=[]
    for m in models:
        for v in VIEWS:
            rr=panels[(m,v)];k=8 if m==BASELINE else 4
            assert len(rr)==64*k
            assert {r['qid'] for r in rr}=={r['qid'] for r in panels[(BASELINE,v)]}
            for fam in ['ALL']+sorted({r['family'] for r in rr}):
                sub=rr if fam=='ALL' else [r for r in rr if r['family']==fam]
                d=dict(model=m,view=v,family=fam,**panel_summary(sub));absolute.append(d)
                if m!=BASELINE:
                    br=panels[(BASELINE,v)]
                    if fam!='ALL':br=[r for r in br if r['family']==fam]
                    changes.append(dict(model=m,view=v,family=fam,
                        delta_from_baseline_pp={key:100*(d['metrics'][key]-balanced(br,key)) for key in METRICS}))
    means=[]
    for a in ARMS:
        for v in VIEWS:
            for fam in ['ALL','CROSS','INTERVAL','THRESHOLD','TOPK']:
                ds=[d for d in absolute if d['model'].startswith(f'SRF1_2_{a}_s') and d['view']==v and d['family']==fam]
                cs=[d for d in changes if d['model'].startswith(f'SRF1_2_{a}_s') and d['view']==v and d['family']==fam]
                assert len(ds)==len(cs)==3
                means.append(dict(arm=a,view=v,family=fam,
                    means={k:float(np.mean([d['metrics'][k] for d in ds])) for k in METRICS},
                    delta_from_baseline_pp={k:float(np.mean([d['delta_from_baseline_pp'][k] for d in cs])) for k in METRICS},
                    seed_range={k:[min(d['metrics'][k] for d in ds),max(d['metrics'][k] for d in ds)] for k in METRICS}))
    original=[r for r in baseline if r['pool']=='MONITOR' and r['variant']=='v0']
    assert len(original)==512
    within=[]
    for m in models:
        for fam in ['ALL','CROSS','INTERVAL','THRESHOLD','TOPK']:
            for left,right in [('gold_values','neutral_hint')]+([(v,'original') for v in VIEWS] if m==BASELINE else []):
                l=panels[(m,left)];r=original if right=='original' else panels[(m,right)]
                if fam!='ALL':l=[x for x in l if x['family']==fam];r=[x for x in r if x['family']==fam]
                assert {x['qid'] for x in l}=={x['qid'] for x in r}
                within.append(dict(model=m,family=fam,left=left,right=right,
                    delta_pp={k:100*(balanced(l,k)-balanced(r,k)) for k in METRICS}))
    return dict(status='RELEASED_DIAGNOSTICS_DESCRIPTIVE',baseline_records=2560,endpoint_records=15360,
        weighting='draws -> question -> root -> family; then three seeds equally',
        scope='Same question/view before-after, baseline K8 and endpoint K4; independent generations, no draw-wise coupling; descriptive, no confirmatory significance claim.',
        limitations=['Endpoint original MONITOR was deliberately omitted. No endpoint diagnostic-original delta is available; TEST_ID is not substituted.',
                    'Views grant different information and are not pure perception/computation causal identification.',
                    'Conditional A failure denominators are raw draws with P=1; zero support stays undefined.'],
        absolute=absolute,changes=changes,arm_means=means,within_view_contrasts=within)


def credit(baseline):
    rr=[r for r in baseline if r['pool']=='MONITOR'];assert len(rr)==4096
    qids=sorted({r['qid'] for r in rr});assert len(qids)==512
    lookup={(r['qid'],r['draw']):r for r in rr};assert len(lookup)==4096
    records=[]
    for b in range(32):
        qq=qids[b*16:(b+1)*16]
        scores=[[lookup[(q,d)]['score'] for d in range(8)] for q in qq]
        c=np.array([[[s['A'],s['J'],s['E'],s['p_read'],s['E']*s['p_read']] for s in g] for g in scores])
        rec=dict(block=b,qids=qq,component_order=['A','J','E','p_read','E*p_read'],
                 component_means=c.mean(axis=1).tolist(),component_population_std=c.std(axis=1).tolist(),
                 component_population_covariance=[np.cov(g,rowvar=False,ddof=0).tolist() for g in c],arms={})
        for a in ARMS:
            adv,info=reward_advantages(a,scores);adv=np.asarray(adv)
            assert adv.shape==(16,8)
            info['positive_negative_credit']={event:dict(W_plus=float((np.maximum(adv,0)*c[:,:,i]).sum()),
                 W_minus=float((np.maximum(-adv,0)*c[:,:,i]).sum()),event_count=int(c[:,:,i].sum())) for event,i in [('A',0),('J',1),('E',2)]}
            info['within_prompt_coarse_tie_variance']={event:[float(sum(np.sum(g==v)/8*av[g==v].var() for v in np.unique(g)))
                 for g,av in zip(c[:,:,i],adv)] for event,i in [('A',0),('J',1)]}
            info['nonconstant_advantage_groups']=int(sum(np.ptp(g)>0 for g in adv))
            if a=='DEC':
                std=np.array(info['component_std']);batch=info['pre_batch_std'];weights=np.array(info['component_weights'])
                info['effective_component_scale']=np.where(std>0,weights/((std+1e-8)*(batch+1e-8)),0).tolist()
                info['scale_definition']='weight/((per-prompt population std+EPS)*(full128 pre-advantage std+EPS)); 0 for constant component; EPS=1e-8'
            rec['arms'][a]=info
        records.append(rec)
    totals=[]
    for a in ARMS:
        totals.append(dict(arm=a,groups=512,nonconstant_groups=sum(r['arms'][a]['nonconstant_advantage_groups'] for r in records),
            positive_negative_credit={ev:{k:sum(r['arms'][a]['positive_negative_credit'][ev][k] for r in records) for k in ['W_plus','W_minus','event_count']} for ev in ['A','J','E']},
            mean_coarse_tie_variance={ev:float(np.mean([x for r in records for x in r['arms'][a]['within_prompt_coarse_tie_variance'][ev]])) for ev in ['A','J']},
            branches=dict(Counter(x for r in records for x in r['arms'][a]['branches']))))
    return dict(status='OFFLINE_COMMON_BASELINE_FOUR_ARM_CREDIT_AUDIT',records=4096,prompts=512,blocks=32,
        arms=list(ARMS),scope='All four registered reward rules applied offline to the SAME fixed 16x8 baseline MONITOR blocks. Not realised trained-model credit or a training intervention; PART is not computed.',
        blocks_definition='protocol evaluation_slots sorts MONITOR qids lexically; index//16. Population std/covariance over eight draws within prompt.',
        W_definition='sum(max(+/-advantage,0)*event) over all4096 trajectories; totals divided by4096 give per-trajectory credit. Event counts retained.',
        totals=totals,blocks_detail=records)


def structures(rows, metadata):
    selected=[r for r in rows if r['pool']=='TEST_ID']
    meta={r['slot_id']:r for r in metadata if r['stage']=='endpoint' and r['pool']=='TEST_ID'}
    assert len(meta)==len(selected)
    look={tuple(r[k] for k in ('model_id','protocol','root_id','chart','variant','draw')):r for r in selected}
    assert len(look)==len(selected)
    grouped=defaultdict(list)
    for key,a in look.items():
        m,pr,root,chart,v,draw=key
        partners=['v1','v2','v3','v4','v5'] if v=='v0' else ['v7'] if v=='v6' else []
        for other in partners:
            b=look[(m,pr,root,chart,other,draw)];ma=meta[a['slot_id']];mb=meta[b['slot_id']]
            assert ma['score_A']==a['score']['A'] and mb['score_A']==b['score']['A']
            na=ma['normalized_answer'];nb=mb['normalized_answer']
            # Valid null is a dictionary; missing/invalid parse is None.
            valid=na is not None and nb is not None and a['score']['L_answer']==1 and b['score']['L_answer']==1
            eq=(na==nb) if valid else None
            strata=['ALL']
            if other=='v4':strata+=['same_gold_answer' if ma['gold_normalized']==mb['gold_normalized'] else 'different_gold_answer']
            for st in strata:
                for metric in (['A'] if pr=='answer_only' else ['A','J']):
                    grouped[(m,pr,v+'_'+other,st,metric,a['family'],root)].append((a['score'][metric]*b['score'][metric],eq))
    roots=[]
    for key,vals in sorted(grouped.items()):
        m,pr,pair,st,metric,f,root=key;valid=[x[1] for x in vals if x[1] is not None]
        roots.append(dict(model=m,protocol=pr,pair=pair,stratum=st,metric=metric,family=f,root_id=root,
                     pairs=len(vals),both_correct_count=sum(x[0] for x in vals),both_correct_probability=float(np.mean([x[0] for x in vals])),
                     agreement=rate(sum(valid),len(valid)),invalid_answer_pairs=len(vals)-len(valid)))
    groups=defaultdict(list)
    for r in roots:groups[tuple(r[k] for k in ['model','protocol','pair','stratum','metric'])].append(r)
    totals=[]
    for key,rs in sorted(groups.items()):
        m,pr,pair,st,metric=key;fs=sorted({r['family'] for r in rs})
        supported=[r for r in rs if r['agreement']['denominator']]
        supportedfs=sorted({r['family'] for r in supported})
        totals.append(dict(model=m,protocol=pr,pair=pair,stratum=st,metric=metric,root_count=len(rs),families=fs,
            both_correct_probability=float(np.mean([np.mean([r['both_correct_probability'] for r in rs if r['family']==f]) for f in fs])),
            agreement_balanced_on_supported_roots=float(np.mean([np.mean([r['agreement']['probability'] for r in supported if r['family']==f]) for f in supportedfs])) if supported else None,
            agreement=rate(sum(r['agreement']['numerator'] for r in rs),sum(r['agreement']['denominator'] for r in rs)),
            total_pairs=sum(r['pairs'] for r in rs),invalid_answer_pairs=sum(r['invalid_answer_pairs'] for r in rs),
            unsupported_roots=len(rs)-len(supported)))
    return dict(status='REGISTERED_STRUCTURE_DESCRIPTIVE',source_TEST_ID_records=len(selected),metadata_TEST_ID_records=len(meta),
        weighting='paired independent draws within chart-question pair -> root -> family; all six registered pairings kept, v4 additionally stratified by exact gold equality',
        scope='both-correct includes every pair and failures remain zero. Agreement is conditional on both answers parsing validly (including valid null); denominator and failures retained. Agreement alone is not correctness or causal robustness.',
        roots=roots,model_totals=totals)


def main(root,out):
    complete=read(root/'endpoint_orchestration/COMPLETE.json');assert complete['endpoint_models']==13 and complete['endpoint_records']==227700
    state=read(root/'endpoint_orchestration/STATE.json');assert state['scores_sealed'] is False and state['endpoints_complete']==13
    rows=[];inputs=[]
    for item in complete['releases']:
        p=root/'released/evaluation'/f"{item['model_id']}.json";d=read(p)
        assert object_hash(d)==item['release_sha256'] and object_hash(d['records'])==d['records_sha256']
        rows+=d['records'];inputs.append(dict(path=str(p),sha256=sha(p)))
    bp=root.parent/'audit/audit_baseline_scored_rows.json';baseline=read(bp)
    if isinstance(baseline,dict):baseline=baseline['records']
    assert len(baseline)==6656
    for r in baseline:_score(r)
    inputs.append(dict(path=str(bp),sha256=sha(bp)))
    normp=root.parent/'audit/audit_normalized_answers.jsonl';meta=[json.loads(l) for l in normp.read_text().splitlines()]
    inputs.append(dict(path=str(normp),sha256=sha(normp)))
    for filename,d in [('DIAGNOSTIC_VIEWS.json',diagnostics(rows,baseline)),('REWARD_ADVANTAGE_AUDIT.json',credit(baseline)),('STRUCTURAL_RESULTS.json',structures(rows,meta))]:
        if filename=='STRUCTURAL_RESULTS.json':
            detail_path=root.parent/'audit/STRUCTURAL_ROOT_DETAILS.json'
            details=d.pop('roots');write(detail_path,details)
            d['root_details']=dict(path=str(detail_path),records=len(details),
                                   file_sha256=sha(detail_path),records_sha256=object_hash(details))
        d['inputs']=inputs;d['script_sha256']=sha(Path(__file__));write(out/filename,d)
    print('PASS diagnostic views, four-arm offline credit, six structural pairings')

if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
