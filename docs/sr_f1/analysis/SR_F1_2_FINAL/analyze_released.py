"""Reproduce fixed SR-F1.2 final statistics from authenticated released scores.

Usage: PYTHONPATH=src python analyze_released.py EVIDENCE_DIR OUTPUT_DIR
No generation, model loading, reward changes, or reads of sealed answers.
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sr_f12.analysis import primary_contrasts, evaluate_predictions, _score, FAMILIES
from sr_f12.protocol import BASELINE, ARMS, SEEDS, PRIMARY_CONTRASTS, object_hash


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def root_values(rows, metric):
    qs = defaultdict(list)
    fs = {}
    for r in rows:
        qs[(r['root_id'], r['qid'])].append(r['score'][metric])
        fs[r['root_id']] = r['family']
    roots = defaultdict(list)
    for (root, _), v in qs.items():
        roots[root].append(np.mean(v))
    return {k: float(np.mean(v)) for k, v in roots.items()}, fs


def balanced(rows, metric):
    values, fam = root_values(rows, metric)
    return float(np.mean([np.mean([v for k, v in values.items() if fam[k] == f])
                         for f in sorted(set(fam.values()))]))


def secondary_ci(panels, pool, protocol, metric, left, right):
    models = lambda a: [BASELINE] * 3 if a == 'BASELINE' else [f'SRF1_2_{a}_s{s}' for s in SEEDS]
    data = {}
    for m in set(models(left) + models(right)):
        data[m], fam = root_values(panels[(m, pool, protocol, None)], metric)
    roots = sorted(data[models(left)[0]])
    assert all(set(v) == set(roots) for v in data.values())
    delta = np.array([[data[l][r] - data[rr][r] for r in roots]
                      for l, rr in zip(models(left), models(right))])
    rng = np.random.default_rng(812901)
    draws = np.zeros(5000)
    seed_effects = np.zeros(3)
    for f in sorted(FAMILIES):
        ix = np.array([i for i, r in enumerate(roots) if fam[r] == f])
        assert len(ix)
        seed_effects += delta[:, ix].mean(axis=1) / 4
        draws += delta.mean(axis=0)[rng.choice(ix, (5000, len(ix)), replace=True)].mean(axis=1) / 4
    return dict(pool=pool, protocol=protocol, metric=metric, contrast=f'{left}-{right}',
                effect_pp=float(seed_effects.mean()*100), seed_effects_pp=(seed_effects*100).tolist(),
                CI95_pp=(np.quantile(draws, [.025,.975])*100).tolist(),
                scope='secondary, unadjusted paired family-stratified root CI; conditional on three realised seeds')


def chart_ci(rows_by_model, arm, split, metric):
    # Question-weighted accuracy, resampling image clusters with all their questions.
    models = [f'SRF1_2_{arm}_s{s}' for s in SEEDS]
    ds = {m: {r['qid']: r for r in rows_by_model[m] if split == 'all' or r['variant'] == split}
          for m in [BASELINE]+models}
    keys = sorted(ds[BASELINE]); assert all(set(v)==set(keys) for v in ds.values())
    clusters = defaultdict(list)
    for q in keys:
        assert all(ds[m][q]['root_id']==ds[BASELINE][q]['root_id'] for m in models)
        clusters[ds[BASELINE][q]['root_id']].append(q)
    clusters = list(clusters.values())
    counts = np.array([len(g) for g in clusters])
    sums = np.array([[sum(ds[m][q]['score'][metric]-ds[BASELINE][q]['score'][metric] for q in g)
                      for g in clusters] for m in models], float)
    rng=np.random.default_rng(812901); draws=[]
    for _ in range(50):
        ix=rng.integers(len(clusters),size=(100,len(clusters)))
        draws.extend((sums.mean(axis=0)[ix].sum(axis=1)/counts[ix].sum(axis=1)).tolist())
    return dict(arm=arm,split=split,metric=metric,n_questions=len(keys),n_images=len(clusters),
                effect_pp=float(sums.mean(axis=0).sum()/counts.sum()*100),
                seed_effects_pp=(sums.sum(axis=1)/counts.sum()*100).tolist(),
                CI95_pp=(np.quantile(draws,[.025,.975])*100).tolist(),
                scope='secondary unadjusted image-cluster CI; question-weighted; conditional on three realised seeds')


def main(root, out):
    out.mkdir(parents=True,exist_ok=True)
    complete=read(root/'endpoint_orchestration/COMPLETE.json')
    acceptance=read(root/'endpoint_orchestration/MAIN_ACCEPTANCE.json')
    state=read(root/'endpoint_orchestration/STATE.json')
    assert complete['status']=='EVALUATIONS_COMPLETE_REPORT_PENDING'
    assert complete['endpoint_models']==13 and complete['endpoint_records']==227700
    assert acceptance['status']=='ALL_TWELVE_TERMINAL_AND_STATE_VERIFIED'
    assert state['endpoints_complete']==13 and state['scores_sealed'] is False
    expected={BASELINE}|{f'SRF1_2_{a}_s{s}' for a in ARMS for s in SEEDS}
    assert {x['model_id'] for x in complete['releases']}==expected
    rows=[]; inputs=[]
    for item in complete['releases']:
        p=root/'released/evaluation'/f"{item['model_id']}.json";d=read(p)
        assert object_hash(d)==item['release_sha256']
        assert d['status']=='RELEASED_AFTER_ALL_MAIN_COMPLETE'
        assert d['authorization_sha256']==object_hash(acceptance['main_completion'])
        assert d['count']==len(d['records'])==item['records']
        assert d['records_sha256']==object_hash(d['records'])
        assert len({x['slot_id'] for x in d['records']})==len(d['records'])
        assert all(x['model_id']==item['model_id'] for x in d['records'])
        for r in d['records']:
            if r['protocol']=='evidence_answer':_score(r)
        rows.extend(d['records']);inputs.append(dict(file=str(p.relative_to(root)),sha256=sha(p),records=d['count']))
    assert len(rows)==227700
    panels=defaultdict(list)
    for r in rows:panels[(r['model_id'],r['pool'],r['protocol'],r['view'])].append(r)
    absolute=[];long=[]
    for (m,p,pr,v), rr in sorted(panels.items(),key=lambda x:str(x[0])):
        metrics=[k for k in rr[0]['score'] if all(isinstance(r['score'].get(k),(float,int)) for r in rr)]
        ischart=p=='CHARTQA'
        values={k:float(np.mean([r['score'][k] for r in rr])) if ischart else balanced(rr,k) for k in metrics}
        support=[r for r in rr if r['score'].get('P')==1]
        conditional=dict(P_support=len(support),A_fail_given_P_count=sum(r['score']['A']==0 for r in support),
                         A_fail_given_P_rate=sum(r['score']['A']==0 for r in support)/len(support) if support else None)
        absolute.append(dict(model=m,pool=p,protocol=pr,view=v,n=len(rr),roots=len({r['root_id'] for r in rr}),metrics=values,conditional=conditional))
        for family in sorted({r['family'] for r in rr}):
            for variant in ['ALL']+sorted({r['variant'] for r in rr}):
                sub=[r for r in rr if r['family']==family and (variant=='ALL' or r['variant']==variant)]
                if not sub:continue
                for metric in metrics:
                    value=float(np.mean([r['score'][metric] for r in sub])) if ischart else balanced(sub,metric)
                    long.append(dict(model=m,pool=p,protocol=pr,view=v or 'original',family=family,variant=variant,
                                     metric=metric,value=value,n=len(sub)))
    predictions=read(root/'SEMANTIC_PREDICTIONS.json')
    primary=primary_contrasts(rows,released=True)
    secondary=[]
    for pool,protocol,metrics in [('TEST_ID','answer_only',['A']),('TEST_ID','evidence_answer',['A','J']),
        ('TEST_COMPOSITION','evidence_answer',['A','J']),('TEST_LANGUAGE','evidence_answer',['A','J']),('TEST_RENDER','evidence_answer',['A','J'])]:
        for metric in metrics:
            for left,right in list(PRIMARY_CONTRASTS)+[(a,'BASELINE') for a in ARMS]:
                secondary.append(secondary_ci(panels,pool,protocol,metric,left,right))
    chart={m:panels[(m,'CHARTQA','plain_answer',None)] for m in sorted(expected)}
    chart_absolute=[dict(model=m,split=split,n=len(rr),metrics={k:float(np.mean([r['score'][k] for r in rr])) for k in ['relaxed_accuracy','exact_match']})
                    for m,rs in chart.items() for split in ['all','human','augmented']
                    for rr in [[r for r in rs if split=='all' or r['variant']==split]]]
    result=dict(status='AUTHENTICATED_RELEASE_ANALYZED',records=len(rows),inputs=inputs,
                weighting='draws -> question -> root -> family equal -> seed equal; ChartQA question mean with image-cluster CI',
                absolute=absolute,primary=primary,predictions=evaluate_predictions(rows,predictions,released=True),
                secondary=secondary,chartqa_absolute=chart_absolute,
                chartqa_changes=[chart_ci(chart,a,s,k) for a in ARMS for s in ['all','human','augmented'] for k in ['relaxed_accuracy','exact_match']],
                limitations=['Root CIs are conditional on three realised training seeds, not training-seed population CIs.',
                              'Secondary CIs are unadjusted; no equivalence or retention certification.',
                              'No endpoint MONITOR: original-versus-view endpoint comparison is unavailable.'])
    for name,value in [('ANALYSIS.json',result),('PRIMARY_CONTRASTS.json',primary),('TRANSFER_RESULTS.json',secondary),
                       ('SEMANTIC_PREDICTION_RESULTS.json',result['predictions']),('CHARTQA_RESULTS.json',dict(absolute=chart_absolute,changes=result['chartqa_changes']))]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (out/'RESULTS_LONG.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(long[0]),lineterminator='\n');w.writeheader();w.writerows(long)
    print(json.dumps(dict(primary=primary,predictions=result['predictions']),ensure_ascii=False,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
