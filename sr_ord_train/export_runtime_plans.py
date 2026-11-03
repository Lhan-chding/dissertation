#!/usr/bin/env python3
"""Export deterministic expected G8 indices solely from the PRE-OUTCOME frozen registration.
No model, scheduler, training/evaluation receipts, or generated response files are opened.
The INDEX pins per-file raw-byte hashes for runtime_contract.py --plan-sha256.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, tempfile
from collections import defaultdict
from pathlib import Path
from itertools import product

ROOT=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor'/'native_src'))
from srf6_ref.core import seed_for

FILES=('training_groups','smoke_groups','evaluation_manifest','clarification_manifest','private_cases','inputs','images')
PARENTS=('b0','b1');SEEDS=(11031,11032);ARMS=('R','G','D')

def need(ok,message):
    if not ok:raise ValueError(message)

def unique_json(pairs):
    d=dict(pairs);need(len(d)==len(pairs),'duplicate JSON keys');return d

def loads(value):
    def reject(x):raise ValueError('nonfinite JSON '+x)
    return json.loads(value,object_pairs_hook=unique_json,parse_constant=reject)

def file_hash(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')

def object_hash(x):return hashlib.sha256(canonical(x)).hexdigest()

def read_freeze(frozen):
    f=loads((frozen/'FREEZE.json').read_bytes());need(f['version']=='SR-ORD-TRAIN-1.0','unregistered freeze version')
    need(set(f['files'])=={n+'.jsonl' for n in FILES},'missing/unexpected frozen files')
    for name,expected in f['files'].items():need(file_hash(frozen/name)==expected,'frozen bytes mismatch '+name)
    return f

def rows(frozen,name):
    with (frozen/(name+'.jsonl')).open('rb') as fh:return [loads(line) for line in fh]

def indexed(rows,key):
    d={r[key]:r for r in rows};need(len(d)==len(rows),'duplicate '+key);return d

def export(frozen):
    frozen=Path(frozen).resolve(strict=True);registration=read_freeze(frozen)
    cs=indexed(rows(frozen,'private_cases'),'case_id'); public=indexed(rows(frozen,'inputs'),'input_id')
    train=rows(frozen,'training_groups');smoke=rows(frozen,'smoke_groups')
    need(len(train)==4608 and len(smoke)==24,'wrong registered group count')
    paths=defaultdict(lambda:defaultdict(dict));ss={};group_slots={};rid_seen=set();seen_seeds={}
    def group(row,stem,slot_key):
        p=row['parent']; t=cs[row['case_id']]; need(t['input_id']==row['input_id'] and row['input_id'] in public and t['task']['root_id']==row['root_id'],'binding mismatch')
        seed=row['group_seed'];need(type(seed) is int and p in PARENTS and row['draws']==8,'invalid frozen draw plan')
        need(group_slots.setdefault(seed,slot_key)==slot_key,'same group seed in unrelated slots')
        seeds=[seed_for(seed,'draw',i) for i in range(8)]
        for i,sd in enumerate(seeds):need(seen_seeds.setdefault(sd,(slot_key,i))==(slot_key,i),'derived seed reused outside paired slot')
        ids=[f'{stem}/d{i:02d}' for i in range(8)];need(not rid_seen.intersection(ids),'duplicate request ID');rid_seen.update(ids)
        return {'group_id':stem,'parent':p,'qid':t['task']['qid'],'root_id':row['root_id'],'input_id':row['input_id'],'case_id':row['case_id'],
                'group_seed':seed,'seeds':seeds,'request_ids':ids}
    for row in train:
        p,s,a,step,slot=(row[k] for k in ('parent','train_seed','arm','step','slot'))
        need(s in SEEDS and a in ARMS and step in range(1,25) and slot in range(16),'invalid path')
        key=(p,s,a); stepmap=paths[key][step];need(slot not in stepmap,'duplicate slot')
        stepmap[slot]=group(row,f'OT1/TRAIN/{p}/{a}/seed{s}/s{step:02d}/g{slot:02d}',('train',p,s,step,slot))
    need(len(paths)==12,'missing path');data={}
    for p,s,a in product(PARENTS,SEEDS,ARMS):
        dest=[]
        for n in range(1,25):
            g=paths[p,s,a][n];need(set(g)==set(range(16)),'incomplete update');dest.append({'step':n,'groups':[g[x] for x in range(16)]})
        data[f'paths/{p}-{s}-{a}.json']=canonical(dest)+b'\n'
    for row in smoke:
        p,i=row['parent'],row['group_index'];need(p in PARENTS and i in range(12) and (p,i) not in ss,'incorrect smoke indices')
        ss[p,i]=group(row,f'OT1/DEV_SMOKE/{p}/g{i:02d}',('smoke',p,i))
    need(len(ss)==24,'incomplete smoke');data['smoke_slots.json']=canonical([ss[k] for k in sorted(ss)])+b'\n'
    idx={'version':'order-runtime-plan-export-v1','scientific_freeze_sha256':file_hash(frozen/'FREEZE.json'),
         'source_files':registration['files'],'derivation':"seed_for(group_seed, 'draw', draw_index_0_to_7)",
         'provenance':'Derived solely from frozen pre-outcome registration; never infer or replace expected slots from receipts.',
         'paths':12,'train_groups':len(train),'smoke_groups':len(smoke),'distinct_request_ids':len(rid_seen),
         'files':{n:hashlib.sha256(content).hexdigest() for n,content in sorted(data.items())}}
    data['INDEX.json']=json.dumps(idx,sort_keys=True,indent=2,ensure_ascii=False).encode()+b'\n'
    dest=frozen/'receipt_plans'
    if dest.exists():
        need(set(str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file())==set(data),'existing export file set differs')
        need(all((dest/k).read_bytes()==v for k,v in data.items()),'divergent existing export: do not rewrite registration')
    else:
        staging=Path(tempfile.mkdtemp(prefix='.receipt_plans_',dir=frozen))
        try:
            for n,b in data.items():
                f=staging/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
            os.rename(staging,dest)
        finally:
            if staging.exists():shutil.rmtree(staging)
    return idx

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frozen',type=Path,default=ROOT/'frozen');a=p.parse_args()
    try:print(json.dumps(export(a.frozen),ensure_ascii=False,indent=2))
    except (OSError,ValueError,TypeError,KeyError) as e:p.exit(2,'FAIL_CLOSED: '+str(e)+'\n')
if __name__=='__main__':main()
