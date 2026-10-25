"""Six 12-update windows; each window covers 8 roots per family exactly."""
from __future__ import annotations
import random
from collections import defaultdict
from .generator import FAMILIES,seed_for

def build_schedule(tasks,master,block,repeat,expected_roots_per_family=48):
    roots=defaultdict(list)
    for t in tasks:roots[t['root_id']].append(t)
    families={f:[] for f in FAMILIES}
    for root,rows in roots.items():
        if len(rows)!=4 or {r['variant'] for r in rows}!={0,1,2,3}:raise ValueError('root must have four variants')
        if len({r['cell'] for r in rows})!=1:raise ValueError('family mismatch')
        families[rows[0]['cell']].append(root)
    if expected_roots_per_family not in (8,48) or any(len(v)!=expected_roots_per_family for v in families.values()):raise ValueError('registered 8-pilot/48-main roots per family required')
    rng=random.Random(seed_for(master,'root-order',block,repeat))
    for ids in families.values():ids.sort();rng.shuffle(ids)
    result=[]
    for window in range(expected_roots_per_family//8):
        current=[r for f in FAMILIES for r in families[f][8*window:8*(window+1)]]
        rng.shuffle(current)
        for k in range(12):
            step=12*window+k+1;chosen=current[4*k:4*(k+1)];groups=[]
            for root in chosen:
                for task in sorted(roots[root],key=lambda x:x['variant']):
                    gi=len(groups)
                    groups.append(dict(qid=task['qid'],root_id=root,cell=task['cell'],group_index=gi,
                         draw_seeds=[seed_for(master,'train',block,repeat,step,task['qid'],d) for d in range(8)],
                         shuffle_seed=seed_for(master,'conditional-shuffle',block,repeat,step,task['qid'])))
            result.append(dict(step=step,window=window,groups=groups))
    return result
