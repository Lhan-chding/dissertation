#!/usr/bin/env python3
"""Build and validate an outcome-blind slot schedule, not charts/model outputs.

Production chart generation/rendering must use the audited project generator.
This does not authorize training. Writing into an existing output file is refused.
"""
from __future__ import annotations
import argparse
import itertools
from collections import Counter, defaultdict
from pathlib import Path
import json
import random
from satdose_core import ARMS, STARTS, STRUCTURES, object_sha, read_jsonl, require, seed_for, write_json

MASTER = 2026111201
CELLS = tuple(itertools.product(('largest', 'smallest'), ('sum', 'mean')))


def build() -> list[dict]:
    rows = []
    for start in STARTS:
        for step in range(1, 25):
            cells = [(si, ci) for si in range(4) for ci in range(4)]
            random.Random(seed_for(MASTER, 'slot_order', start, step)).shuffle(cells)
            for slot, (si, ci) in enumerate(cells):
                direction, aggregate = CELLS[ci]
                root = f'SAT1_TRAIN_s{start}_u{step:02d}_g{slot:02d}'
                gseed = seed_for(MASTER, 'G8', start, step, slot)
                # One aligned cell per structure per update, rotated across all four cells.
                is_conflict = ci != ((step-1+si) % 4)
                for arm in ARMS:
                    rid = f'SAT1/TRAIN/{start}/{arm}/s{step:02d}/g{slot:02d}'
                    rows.append(dict(group_id=rid, start_seed=start, parent='b0', arm=arm,
                                     step=step, slot=slot, root_id=root, n=8, family='TIES',
                                     structure=STRUCTURES[si], direction=direction, aggregate=aggregate,
                                     channel='LIST', planned_conflict=(arm=='T1' and is_conflict),
                                     group_seed=gseed, world_seed=seed_for(MASTER, 'world', start, step, slot),
                                     permutation_seed=seed_for(MASTER, 'list', start, step, slot),
                                     draw_seeds=[seed_for(gseed, 'draw', i) for i in range(8)],
                                     request_ids=[rid+f'/d{i:02d}' for i in range(8)]))
    validate(rows)
    return rows


def validate(rows: list[dict]) -> dict:
    require(len(rows) == 1536, 'exactly 2 starts x 2 arms x 24 x 16 groups required')
    expected = {(s,a,t,g) for s in STARTS for a in ARMS for t in range(1,25) for g in range(16)}
    got = set(); paired = defaultdict(dict); bystep = defaultdict(list); bycell=Counter()
    reqids=set(); gidset=set(); rootowner={}; allseeds={}
    for row in rows:
        s,a,t,g = (row[k] for k in ('start_seed','arm','step','slot'))
        require(type(s) is int and type(t) is int and type(g) is int, 'noninteger schedule identity')
        k=(s,a,t,g)
        require(k in expected and k not in got, 'unexpected/duplicate schedule slot')
        got.add(k)
        require(row['family']=='TIES' and row['n']==8 and row['parent']=='b0' and row['channel']=='LIST',
                'unregistered task/channel')
        require(row['structure'] in STRUCTURES and (row['direction'],row['aggregate']) in CELLS,
                'unregistered stratum')
        require(type(row['planned_conflict']) is bool, 'conflict flag must be boolean')
        require(len(row['draw_seeds'])==8 and len(set(row['draw_seeds']))==8,
                'incomplete/duplicate G8 draw seeds')
        require(len(row['request_ids'])==8 and len(set(row['request_ids']))==8,
                'incomplete/duplicate request ids')
        require(not reqids.intersection(row['request_ids']), 'reused request id')
        reqids.update(row['request_ids'])
        require(row['group_id'] not in gidset, 'reused group_id');gidset.add(row['group_id'])
        owner=(s,t,g)
        require(rootowner.setdefault(row['root_id'],owner)==owner, 'root reused in another step/start')
        for ds in row['draw_seeds']:
            require(type(ds) is int and 0 <= ds < 2**63, 'invalid draw seed')
            require(allseeds.setdefault(ds,owner)==owner, 'unintended seed reuse')
        require(a!='T0' or not row['planned_conflict'], 'T0 must be aligned')
        paired[owner][a]=row
        bystep[s,a,t].append(row)
        bycell[s,a,row['structure'],row['direction'],row['aggregate'],row['planned_conflict']]+=1
    require(got==expected, 'missing planned slot')
    for pair in paired.values():
        x,y=pair['T0'],pair['T1']
        for key in ('root_id','structure','direction','aggregate','world_seed','group_seed','draw_seeds','permutation_seed'):
            require(x[key]==y[key], 'unpaired '+key)
    for (s,a,t), group in bystep.items():
        require(len(group)==16, 'wrong groups per update')
        require(sum(r['planned_conflict'] for r in group)==(12 if a=='T1' else 0), 'wrong per-update dose')
        require(len({(r['structure'],r['direction'],r['aggregate']) for r in group})==16,
                'per-update structure/direction/aggregate imbalance')
    for s in STARTS:
        for st in STRUCTURES:
            for d, ag in CELLS:
                require(bycell[s,'T1',st,d,ag,True]==18 and bycell[s,'T1',st,d,ag,False]==6,
                        'cumulative cell imbalance')
    return dict(status='CPU_SCHEDULE_VALIDATED_NOT_GPU_AUTHORIZATION', paths=4, updates=96,
                groups=1536, unique_train_roots=768, responses=12288,
                T1_conflict_groups_per_path=288, schedule_content_sha256=object_sha(rows))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out',type=Path);mode.add_argument('--check',type=Path)
    a=ap.parse_args()
    if a.check:
        print(json.dumps(validate(list(read_jsonl(a.check))),indent=2));return
    require(not a.out.exists(), 'refuse to overwrite existing schedule')
    rows=build();a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
    write_json(a.out.with_suffix('.summary.json'),validate(rows))
    print(a.out)

if __name__=='__main__':main()
