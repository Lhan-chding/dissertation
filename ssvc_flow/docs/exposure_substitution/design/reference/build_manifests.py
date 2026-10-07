"""Build new CPU task/schedule manifests from an existing VDT results archive."""
from __future__ import annotations
import argparse, csv, io, json, random, zipfile
from collections import Counter
from pathlib import Path
from contracts import *


def load_lines(z, name):
    return [json.loads(s) for s in z.read(name).decode().splitlines() if s.strip()]


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-zip', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(args.results_zip)
    tp = {r['task_id']:r for r in load_lines(z, 'run_v2/cohort/T_train/tasks_public.jsonl')}
    ta = {r['task_id']:r for r in load_lines(z, 'run_v2/cohort/T_train/audit_only.jsonl')}
    exposure = list(csv.DictReader(io.StringIO(z.read('project_docs/final_analysis/discovery_fit_exposures.csv').decode())))
    groups = {}
    for r in exposure:
        if r['role']=='focus' and r['job'].endswith('.SELF_O0'):
            groups.setdefault(r['job'],set()).add(r['task_id'])
    if len(groups)!=4:
        raise ValueError('Expected the four historical SELF_O0 data views')
    common_ids = set.intersection(*groups.values())
    removed = {i for i in common_ids if ta[i]['center'] in (0,2,3) and ta[i]['corrupted_index']==1}
    common_ids -= removed
    common=[]
    for i in sorted(common_ids):
        t=dict(tp[i]);t['root_id']=t['base_instance_id'];t['split']='COMMON_TRAIN'
        common.append({'task':t,'target':canonical_target(ta[i]['true_world']),
                       'source':'intersection_of_four_historical_SELF_O0_focus_views',
                       'original_task_id':i})
        assert verify(t,ta[i]['true_world'])
    write_lines(root/'common_targets.jsonl',common)
    write_lines(root/'common_audit.jsonl',[dict(ta[i],split='COMMON_TRAIN') for i in sorted(common_ids)])
    rp=load_lines(z,'run_v2/cohort/R_replay/tasks_public.jsonl')
    rt=load_lines(z,'run_v2/cohort/R_replay/verified_targets.jsonl')
    # Original order is by task_id in both files; join by actual recorded ID rather than assume order.
    r_by_id={r['task']['task_id']:r for r in rt}
    replay=[]
    for t0 in rp:
        t=dict(t0);t['split']='REPLAY';t['root_id']=t['base_instance_id']
        r=r_by_id[t['task_id']]
        y=r['canonical_vector']
        assert verify(t,y)
        replay.append({'task':t,'target':canonical_target(y),'source':r['source']})
    write_lines(root/'replay_targets.jsonl',replay)
    hist=json.loads(z.read('project_docs/evidence/historical_inputs/historical_exclusion.json'))
    seen={tuple(a) for a in hist['truth_orbits']}
    for n in z.namelist():
        if n.startswith('run_v2/cohort/') and n.endswith('audit_only.jsonl'):
            seen.update(tuple(sorted(a['true_world'])) for a in load_lines(z,n))
    seen.update(tuple(sorted(r['canonical_vector'])) for r in rt)
    starting_exclusion_count=len(seen)
    roots=[]
    def get_root(rng, split, index, family):
        if family=='trend':
            # Enumerate an actually available finite set; no silent domain expansion.
            choices=[(a,a+d,a+2*d,a+3*d) for a in range(100) for d in range(1,34)
                     if a+3*d<=99 and (a,a+d,a+2*d,a+3*d) not in seen]
            if not choices:raise ValueError('No unused trend orbits remain')
            x=list(rng.choice(choices))
            if rng.randrange(2):x.reverse()
        else:
            for _ in range(100000):
                x=rng.sample(range(100),4)
                if tuple(sorted(x)) not in seen:break
            else:raise ValueError('Unable to sample new root')
        seen.add(tuple(sorted(x)))
        replacements=[]
        for v in x:
            u=rng.randrange(99); replacements.append(u+(u>=v))
        rid=stable_id('SER-J2-root',split,index,family,x,replacements)
        rr=dict(root_id=rid,split=split,family=family,true_world=x,replacements=replacements)
        roots.append(rr)
        return rr
    def from_root(rr,h,j,split,index):
        x=rr['true_world'];o=x[:];o[j]=rr['replacements'][j]
        return make_task(x,o,h,j,rr['root_id'],split,index,rr['family'])
    # Matched donors: same observed values, same target; only relation hub differs.
    donor_tasks=[];donor_aud=[];donor_targets=[];donor_ids={};donor_roots=[]
    rng=random.Random(107071)
    for i in range(32):
        rr=get_root(rng,'DONOR_TRAIN',i,'cross_series');donor_roots.append(rr['root_id'])
        for arm,h in ARMS.items():
            t,a=from_root(rr,h,1,'DONOR_TRAIN',i)
            donor_tasks.append(t);donor_aud.append(a)
            donor_targets.append(dict(arm=arm,task_id=t['task_id'],root_id=rr['root_id'],target=canonical_target(rr['true_world']),label_source='generated_gold_for_controlled_intervention'))
            donor_ids[(rr['root_id'],arm)]=t['task_id']
    write_lines(root/'donors_public.jsonl',donor_tasks)
    write_lines(root/'donors_audit.jsonl',donor_aud)
    write_lines(root/'donor_targets.jsonl',donor_targets)
    # E_diag: full 16 cross cells, two matched roots; eight trend and eight duplicate.
    tasks=[];aud=[];rng=random.Random(107072)
    for i in range(2):
        rr=get_root(rng,'E_DIAG',i,'cross_series')
        for h in range(4):
            for j in range(4):
                t,a=from_root(rr,h,j,'E_DIAG',i);tasks.append(t);aud.append(a)
    for family in ('trend','duplicate_encoding'):
        for i in range(8):
            rr=get_root(rng,'E_DIAG',i,family)
            t,a=from_root(rr,None,i%4,'E_DIAG',i);tasks.append(t);aud.append(a)
    write_lines(root/'E_DIAG/tasks_public.jsonl',tasks);write_lines(root/'E_DIAG/audit_only.jsonl',aud)
    # E_confirm: 128 matched cross roots. First 32 expand to all 16 cells;
    # remaining 96 expand only to the two primary protected hub cells.
    ct=[];ca=[];rng=random.Random(107073)
    for i in range(128):
        rr=get_root(rng,'E_CONFIRM',i,'cross_series')
        cells=[(h,j) for h in range(4) for j in range(4)] if i<32 else [(2,2),(3,3)]
        for h,j in cells:
            t,a=from_root(rr,h,j,'E_CONFIRM',i);t['root_cohort']='CORE32' if i<32 else 'EXTRA96'
            a['root_cohort']=t['root_cohort'];ct.append(t);ca.append(a)
    for family,n in [('trend',64),('duplicate_encoding',32)]:
        for i in range(n):
            rr=get_root(rng,'E_CONFIRM',i,family)
            t,a=from_root(rr,None,i%4,'E_CONFIRM',i);t['root_cohort']=family;a['root_cohort']=family
            ct.append(t);ca.append(a)
    write_lines(root/'E_CONFIRM/tasks_public.jsonl',ct);write_lines(root/'E_CONFIRM/audit_only.jsonl',ca)
    write_lines(root/'new_root_registry_AUDIT_ONLY.jsonl',roots)
    all_tasks=donor_tasks+tasks+ct
    audits={a['task_id']:a for a in donor_aud+aud+ca}
    for t in all_tasks:
        assert solve_public(t)==[audits[t['task_id']]['true_world']]
    prompt_records=[]
    for t in all_tasks:
        prompt_records.append({'task_id':t['task_id'],'split':t['split'],'prompt':public_prompt(t)})
    write_lines(root/'new_prompts_public.jsonl',prompt_records)
    for seed in (108701,108702,108703):
        rows=schedule(sorted(common_ids),[r['task']['task_id'] for r in replay],donor_roots,donor_ids,seed)
        write_lines(root/f'schedule_{seed}.jsonl',rows)
    training_jobs=[dict(parent=p,block=b,block_seed=s,arm=a,updates=256)
                   for p in ('S96','REP96') for b,s in enumerate((108701,108702,108703)) for a in ARMS]
    write_lines(root/'training_jobs.jsonl',training_jobs)
    # Train-only sentinels. Their output is diagnostic, never used as held-out performance.
    rng=random.Random(107074)
    sentinels=dict(common=rng.sample(sorted(common_ids),8),
                   replay=rng.sample([r['task']['task_id'] for r in replay],4),
                   donor_roots=rng.sample(donor_roots,4))
    (root/'fit_sentinels.json').write_text(json.dumps(sentinels,indent=2))
    summary=dict(source_archive=Path(args.results_zip).name,
                 historical_exclusion_orbits=starting_exclusion_count,
                 common_intersection_before_removal=len(common_ids)+len(removed),
                 common_removed_donor_cells=len(removed),common_tasks=len(common),
                 replay_tasks=len(replay),donor_roots=32,donor_variants=len(donor_tasks),
                 diagnostic_tasks=len(tasks),confirm_tasks=len(ct),new_roots=len(roots),
                 new_unique_repair_checks=len(all_tasks),
                 confirm_cross_cells=dict(Counter(f"c{a['center']+1}j{a['corrupted_index']+1}" for a in ca if a['center'] is not None)),
                 workload=experiment_workload(),
                 limitation='Exclusion covers the archived historical index and VDT cohort; Codex must add later local runs before first GPU use. No model result was queried.')
    (root/'BUILD_REPORT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
