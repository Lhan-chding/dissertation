#!/usr/bin/env python3
"""Independent read-only historical replay audit. Writes only to --out.

Checks archive SHA256s, all native training selection witnesses, all evaluation
selection fields, reward arrays against historical path receipts, algebraic G8
advantage invariants, and the two starting-checkpoint hashes from old receipts. No torch, GPU, model calls, scheduler calls or network.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from satdose_core import (KEY, exact_hit, file_sha, geometry, group_metrics,
                          native_advantages, parse_selected, read_json, read_jsonl,
                          require, source_class, write_json)


def audit(bundle: Path) -> dict:
    bundle=bundle.resolve(strict=True)
    pins=read_json(bundle/'SHA256SUMS.json')
    for name, info in pins.items():
        p=(bundle/name).resolve(strict=True)
        require(p.is_relative_to(bundle), 'unsafe historical checksum path')
        require(p.stat().st_size==info['bytes'] and file_sha(p)==info['sha256'], 'historical SHA mismatch: '+name)
    cases={x['case_id']:x for x in read_jsonl(bundle/'frozen/private_cases.jsonl')}
    scores={tuple(x[k] for k in KEY):x for x in read_jsonl(bundle/'results/eval_scores.jsonl')}
    require(len(scores)==26624, 'unexpected evaluation key coverage')
    eval_n=0;eval_geometry=0;seen=set();authority=defaultdict(Counter)
    for raw in read_jsonl(bundle/'results/eval_raw.jsonl'):
        key=tuple(raw[k] for k in KEY)
        require(key in scores and key not in seen, 'missing or duplicate eval raw key')
        seen.add(key);sc=scores[key];eval_n+=1
        require(raw['case_id']==sc['case_id'] and raw['input_id']==sc['input_id'], 'eval identity mismatch')
        require(raw['full_text']==raw['assistant_prefill']+raw['suffix_text'], 'prefill concatenation mismatch')
        _,sel=parse_selected(raw['full_text'],raw['finish_reason'])
        task=cases[raw['case_id']]['task']
        require(int(sel is not None)==sc['S_known'], 'eval S_known disagreement')
        require(exact_hit(sel,task['gold']['selected'])==sc['S_hit'], 'eval S_hit disagreement')
        if 'boundary_indices' in task:
            geo=geometry(task,cases[raw['case_id']]['audit'],raw['panel']);eval_geometry+=1
            cls=source_class(sel,geo)
            if raw['panel']=='authority8' and raw['parent']=='b0' and raw['arm']=='R' and raw['step']==24:
                authority[f"{raw['train_seed']}:{raw['view']}"][cls]+=1
    require(seen==scores.keys(), 'evaluation raw/score key set mismatch')
    train_groups=defaultdict(list);train_n=0;selection_mismatches=[];nesting=[];jpositive=0
    for row in read_jsonl(bundle/'training/training_native_scores_projection.jsonl'):
        ns=row['native_score'];w=ns['witness'];train_n+=1
        known=bool(ns['known_mask']['S'])
        sel=w.get('selected') if known else None
        c=exact_hit(sel,w['gold_selected'])
        if c != (int(ns['S']) if known else 0):selection_mismatches.append(row['request_id'])
        jpositive+=ns['J']
        if ns['J']==1 and c==0:nesting.append(row['request_id'])
        train_groups[row['group_id']].append((row,c,int(known)))
    require(train_n==36864 and len(train_groups)==4608, 'training row/group coverage')
    require(not selection_mismatches, 'native S witness disagrees with exact set')
    dose=defaultdict(list); group_n=0;request_ids=set();receipt_groups={}
    for path in (bundle/'training/path_receipts').glob('*.json'):
        receipt=read_json(path)
        for update in receipt['updates']:
            for g in update['groups']:
                require(g['group_id'] not in receipt_groups, 'duplicate receipt group')
                receipt_groups[g['group_id']]=g
    require(len(receipt_groups)==4608, 'receipt group coverage')
    for gid,rows in train_groups.items():
        rows.sort(key=lambda x:x[0]['request_id'])
        require(len(rows)==8, 'incomplete G8')
        ids=[r['request_id'] for r,c,k in rows]
        require(len(set(ids))==8 and not request_ids.intersection(ids), 'duplicate training draw')
        request_ids.update(ids)
        js=[r['native_score']['J'] for r,c,k in rows]
        meta=rows[0][0];task=cases[meta['case_id']]['task'];audit0=cases[meta['case_id']]['audit']
        rec=receipt_groups[gid]
        require(rec['request_ids']==ids and rec['J']==js, 'projection/receipt J mismatch')
        a=native_advantages(js)
        require(abs(sum(a))<1e-10 and max(map(abs,a)) <= math.sqrt(7)+1e-10, 'advantage algebra failure')
        cvec=[c for r,c,k in rows];known=[k for r,c,k in rows]
        if 'boundary_indices' in task:
            geo=geometry(task,audit0)
            for r,c,k in rows:
                sel=r['native_score']['witness'].get('selected') if k else None
                require(c==exact_hit(sel,geo['authorised']), 'TIES oracle/selection mismatch')
            if audit0.get('list_conflict') is True:
                m=group_metrics(js,cvec,known)
                dose[(meta['parent'],meta['arm'],audit0['channel'])].append(m)
        group_n+=1
    summaries=[]
    for (parent,arm,channel),xs in sorted(dose.items()):
        summaries.append(dict(parent=parent,arm=arm,channel=channel,groups=len(xs),
                              signal_groups=sum(x['signal'] for x in xs),
                              J0_source_mixed=sum(x['all_J0'] and x['source_mixed'] for x in xs),
                              dose_sum=sum(x['dose'] for x in xs),dose_mean=sum(x['dose'] for x in xs)/len(xs)))
    starts={}
    for s in (11031,11032):
        r=read_json(bundle/f'training/path_receipts/b0-{s}-R.json')
        starts[str(s)]=dict(policy_sha256=r['evaluation_policy_sha256']['24'],
                            state_sha256=r['evidence']['checkpoint_24']['sha256'],
                            reference_sha256=r['reference_tensor_sha256'],
                            original_state_path=f"/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2/run/paths/b0-{s}-R/checkpoints/step-024/state.pt")
    # Execute the existing pinned native golden fixtures, including J=1/S=0.
    sys.path.insert(0,str(bundle))
    try:
        import runtime_contract
        fixtures=runtime_contract.native_fixtures(bundle/'vendor/native_src')
    finally:
        sys.path.remove(str(bundle))
    return dict(status='LOCAL_HISTORICAL_AUDIT_PASS_NOT_GPU_ADMISSION',
                archive_files_verified=len(pins),training_rows=train_n,training_groups=group_n,
                native_selection_witness_mismatches=0,evaluation_rows=eval_n,
                evaluation_selection_mismatches=0,evaluation_geometry_rows=eval_geometry,
                historical_J_positive=jpositive,J1_selection_not_correct=len(nesting),
                J1_selection_not_correct_request_ids=nesting,
                conflict_training_dose=summaries,authority8_R24={k:dict(v) for k,v in sorted(authority.items())},
                start_bindings=starts,native_fixtures=fixtures,
                gpu_model_calls=0,optimizer_updates=0,
                limitations=['No production adapter/objective file or model tensors are present in this bundle.',
                             'Historical scores are independently checked, not a prospective dry run.',
                             'A J-positive/wrong-selected response is allowed by the historical reward definition; never silently modify J.'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    require(not a.out.resolve().is_relative_to(a.bundle.resolve()), 'output must not alter source bundle')
    require(not a.out.exists(), 'refuse to overwrite audit evidence')
    r=audit(a.bundle);write_json(a.out,r)
    print(json.dumps({k:r[k] for k in ('status','archive_files_verified','training_rows','training_groups','evaluation_rows','J1_selection_not_correct')},indent=2))

if __name__=='__main__': main()
