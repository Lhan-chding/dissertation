"""Nested whole-lineage diagnostic on the completed first-four development matrix.

No GPU, no final-test access, and no global freeze. Outer fitting takes only
training labels; decisions are persisted before held outcomes are scored.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from src.prospective_selection.features import LEVELS, PreDecisionPacket
from src.prospective_selection.kernels import FeatureKernel
from src.prospective_selection.selectors import ACTIONS, ALPHA_GRID, FrozenSelector, best_static, choose, krr_fit


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def table(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def alpha_choice(scores):
    peak=max(scores.values())
    return max(a for a,s in scores.items() if peak-s < .001+1e-12)


def split_lineages(packets, held):
    return ([p for p in packets if p.lineage_id!=held],
            [p for p in packets if p.lineage_id==held])


def grid_predictions(fit, labels, held):
    if set(labels)!={p.origin_id for p in fit}:
        raise ValueError('Only exact fit-origin labels may enter fitting')
    if {p.lineage_id for p in fit}&{p.lineage_id for p in held}:
        raise ValueError('Whole-lineage holdout overlap')
    y=np.array([[labels[p.origin_id][a] for a in ACTIONS] for p in fit])
    default=best_static(fit,labels)
    kernel=FeatureKernel.fit(fit)
    gram=kernel.gram(fit); cross=kernel.gram(held,fit)
    result={}
    for alpha in ALPHA_GRID:
        mean,beta=krr_fit(gram,y-y[:,:1],alpha)
        preds=mean+cross@beta
        result[alpha]=[choose(dict(zip(ACTIONS, map(float,row))),default) for row in preds]
    return result,default,kernel,gram


def fit_outer(train, labels, held):
    if len({p.lineage_id for p in train})!=3 or len({p.lineage_id for p in held})!=1:
        raise ValueError('Pilot requires three training lineages and one holdout')
    if set(labels)!={p.origin_id for p in train}:
        raise ValueError('Outer held labels forbidden')
    inner=[]
    for val_lid in sorted({p.lineage_id for p in train}):
        fit,val=split_lineages(train,val_lid)
        fit_labels={p.origin_id:labels[p.origin_id] for p in fit}
        grid,default,_,_=grid_predictions(fit,fit_labels,val)
        for alpha,decisions in grid.items():
            inner.append(dict(validation_lineage=val_lid,fit_lineages=sorted({p.lineage_id for p in fit}),
                alpha=alpha,best_static=default,origins=[p.origin_id for p in val],decisions=decisions,
                selected_J=float(np.mean([labels[p.origin_id][d['recipe']] for p,d in zip(val,decisions)]))))
    scores={a:float(np.mean([x['selected_J'] for x in inner if x['alpha']==a])) for a in ALPHA_GRID}
    alpha=alpha_choice(scores)
    grid,default,kernel,gram=grid_predictions(train,labels,held)
    model=FrozenSelector.fit(train,labels,alpha=alpha,default_action=default,
        tuning_scores=[dict(alpha=a,selected_action_utility=s) for a,s in scores.items()])
    decisions=[model.choose_action(p) for p in held]
    for p,d,fast in zip(held,decisions,grid[alpha]):
        assert d['recipe']==fast['recipe']
        assert max(abs(d['predictions'][a]-fast['predictions'][a]) for a in ACTIONS)<1e-12
    return dict(alpha=alpha,best_static=default,inner=inner,decisions=decisions,
        alpha_grid=grid,model=model.to_dict(),kernel=kernel.to_dict(),gram=gram.tolist(),
        eigenvalues=np.linalg.eigvalsh(gram).tolist(),feature_sizes=[kernel.diagnostics(p) for p in held])


def run(source,out):
    if out.exists() and any(p.name!='PILOT_PLAN_zh.md' for p in out.iterdir()):
        raise ValueError('Preserve partial or complete diagnostic; use fresh output')
    d=json.loads(source.read_text());assert len(d['branches'])==88 and not d['missing']
    labels={};all_labels={}
    for b in d['branches']:
        p=b['task']['payload'];o=p['origin_id'];a=p['recipe_id'];j=b['evaluations']['32']['summary']['J']
        assert p['repeat']==1 and p['lineage_id'] in range(61001,61005)
        assert a not in all_labels.setdefault(o,{})
        all_labels[o][a]=j
        if a in ACTIONS:labels.setdefault(o,{})[a]=j
    expected_origins={f'{lid}_t{step}' for lid in range(61001,61005) for step in (32,96)}
    expected_recipes=set(ACTIONS)|{'GDPO_R4','SAW_R4','DIRECT_REPAIR_R4'}
    assert set(all_labels)==expected_origins and all(set(v)==expected_recipes for v in all_labels.values())
    assert len(d['prestates'])==8 and {p['origin'] for p in d['prestates']}==expected_origins
    for entry in d['prestates']:
        assert set(entry['packets'])==set(LEVELS)
        for level,packet in entry['packets'].items():
            assert packet['origin_id']==entry['origin'] and packet['feature_level']==level
            assert packet['origin_id']==f"{packet['lineage_id']}_t{packet['step']}"
    packets={level:[PreDecisionPacket.from_dict(p['packets'][level]) for p in sorted(d['prestates'],key=lambda x:x['origin'])] for level in LEVELS}
    assert len(labels)==8 and all(set(v)==set(ACTIONS) for v in labels.values())
    decisions=[];sensitivity=[];prediction_rows=[];folds=[]
    # This phase generates all held choices before any held score is joined below.
    for level in LEVELS:
        for lid in sorted({p.lineage_id for p in packets[level]}):
            train,held=split_lineages(packets[level],lid)
            fitlabels={p.origin_id:labels[p.origin_id] for p in train}
            r=fit_outer(train,fitlabels,held)
            stem=f'{lid}_{level}'
            save(out/'models'/f'{stem}.json',r.pop('model'))
            save(out/'folds'/f'{stem}.json',dict(held_lineage=lid,fit_lineages=sorted({p.lineage_id for p in train}),
                fit_origins=[p.origin_id for p in train],held_origins=[p.origin_id for p in held],**r))
            for p,choice in zip(held,r['decisions']):
                decisions.append(dict(level=level,lineage=lid,origin=p.origin_id,step=p.to_dict()['step'],
                    alpha=r['alpha'],best_static=r['best_static'],**choice))
            for a,choices in r['alpha_grid'].items():
                for p,c in zip(held,choices):sensitivity.append(dict(level=level,lineage=lid,origin=p.origin_id,alpha=a,recipe=c['recipe'],predicted_best=c['predicted_best'],margin=c['margin_to_default']))
            folds.append(dict(level=level,held_lineage=lid,alpha=r['alpha'],best_static=r['best_static'],
                inner_selected_J=float(np.mean([x['selected_J'] for x in r['inner'] if x['alpha']==r['alpha']])),eigenvalues=r['eigenvalues']))
            print('FOLD_COMPLETE',stem,flush=True)
    save(out/'decisions_before_scoring.json',decisions)
    decision_hash=hashlib.sha256((out/'decisions_before_scoring.json').read_bytes()).hexdigest()
    scored=[]
    for row in decisions:
        o=row['origin'];truth=labels[o];best=max(truth.values());default=row['best_static'];pred=row['predictions']
        for a in ACTIONS:
            prediction_rows.append(dict(level=row['level'],lineage=row['lineage'],origin=o,recipe=a,
                predicted_delta_R0=pred[a],actual_delta_R0=truth[a]-truth['R0'],actual_J=truth[a],chosen=a==row['recipe']))
        scored.append(dict(level=row['level'],lineage=row['lineage'],origin=o,step=row['step'],alpha=row['alpha'],
            recipe=row['recipe'],predicted_best=row['predicted_best'],best_static=default,
            margin=row['margin_to_default'],same_as_static=row['recipe']==default,
            tie_override=row['predicted_best']!=default and row['recipe']==default,
            J=truth[row['recipe']],best_static_J=truth[default],delta_static_pp=100*(truth[row['recipe']]-truth[default]),
            observed_oracle_J=best,observed_regret_pp=100*(best-truth[row['recipe']]),
            GDPO_J=all_labels[o]['GDPO_R4'],SAW_J=all_labels[o]['SAW_R4'],DIRECT_REPAIR_J=all_labels[o]['DIRECT_REPAIR_R4'],
            prediction_delta_MAE=float(np.mean([abs(pred[a]-(truth[a]-truth['R0'])) for a in ACTIONS]))))
    for s in sensitivity:s['J']=labels[s['origin']][s['recipe']]
    lineage=[]
    for level in LEVELS:
        for lid in sorted({r['lineage'] for r in scored}):
            rows=[r for r in scored if r['level']==level and r['lineage']==lid]
            lineage.append(dict(level=level,lineage=lid,**{k:float(np.mean([r[k] for r in rows])) for k in ['J','best_static_J','delta_static_pp','observed_regret_pp','GDPO_J','SAW_J','DIRECT_REPAIR_J','prediction_delta_MAE']}))
    summary=[]
    for level in LEVELS:
        rows=[r for r in lineage if r['level']==level];orig=[r for r in scored if r['level']==level]
        summary.append(dict(level=level,**{k:float(np.mean([r[k] for r in rows])) for k in ['J','best_static_J','delta_static_pp','observed_regret_pp','GDPO_J','SAW_J','DIRECT_REPAIR_J','prediction_delta_MAE']},
            static_choices=sum(r['same_as_static'] for r in orig),tie_overrides=sum(r['tie_override'] for r in orig),
            alpha_choices=[r['alpha'] for r in folds if r['level']==level]))
    table(out/'selected_outcomes.csv',scored);table(out/'all_candidate_predictions.csv',prediction_rows)
    table(out/'alpha_sensitivity_POSTHOC.csv',sensitivity);table(out/'lineage_scores.csv',lineage)
    table(out/'method_summary.csv',summary)
    paired=[]
    for lid in sorted({r['lineage'] for r in lineage}):
        z={r['level']:r for r in lineage if r['lineage']==lid}
        paired.append(dict(lineage=lid,delta_Z3_Z2_pp=100*(z[LEVELS[3]]['J']-z[LEVELS[2]]['J'])))
    save(out/'summary.json',dict(status='EXPLORATORY_FIRST_FOUR_NESTED_LOLO_COMPLETE',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),decisions_sha256=decision_hash,
        outer_folds=4,inner_folds_per_outer=3,independent_lineages=4,held_origins=8,methods=summary,paired_lineages=paired,
        mean_Z3_Z2_pp=float(np.mean([x['delta_Z3_Z2_pp'] for x in paired])),final_test_run=False,global_freeze_written=False,
        caveat='Previously inspected development labels; exploratory replay, not prospective confirmation.'))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.input,a.out)
