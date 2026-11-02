#!/usr/bin/env python3
"""Read only user-provided historical DEVELOPMENT task members, not sealed tests."""
import argparse,json,sys
from pathlib import Path
from zipfile import ZipFile
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from order_probe.common import read_jsonl,write_json,file_sha
from order_probe.data import world_key

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--archive-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sources=[('SR_F6_REVIEW_20261028.zip','DEV_CAL_TASKS_PRIVATE.jsonl'),
             ('SR_F6_PILOT_DEV_FOR_GPT_CLAUDE_20261102.zip','04_DESIGN/DEV_EVAL_TASKS_PRIVATE.jsonl'),
             ('SR_F6_PILOT_DEV_FOR_GPT_CLAUDE_20261102.zip','04_DESIGN/DEV_PILOT_TASKS_PRIVATE.jsonl')]
    old=set();unlabeled=set();details=[]
    for filename,suffix in sources:
        path=a.archive_dir/filename
        with ZipFile(path) as z:
            members=[m for m in z.namelist() if m.endswith(suffix)]
            if len(members)!=1:raise RuntimeError(f'Exact historical member ambiguous: {filename}:{suffix}')
            rows=[json.loads(s) for s in z.read(members[0]).decode().splitlines() if s]
        for t in rows:
            old.add(world_key(t['world'],t['query']));unlabeled.add(world_key(t['world'],t['query'],strip_names=True))
        details.append({'archive':filename,'archive_sha256':file_sha(path),'member':members[0],'rows':len(rows)})
    overlaps=[];n=0
    for split in ('SMOKE','MAIN'):
        for t in read_jsonl(a.inputs/f'{split}_TASKS_PRIVATE.jsonl'):
            n+=1
            if t['world_query_key'] in old or t['unlabeled_world_query_key'] in unlabeled:overlaps.append(t['qid'])
    write_json(a.out,{'status':'PASS_WITH_RECORDED_SCOPE' if not overlaps else 'FAIL','new_views':n,
        'historical_sources':details,'overlap_qids':overlaps,
        'scope':'Only named previously-used F6 development/census/pilot data. No claim about inaccessible historical or sealed tests.',
        'server_historical_used_data_index':'PENDING'})
    print(f'{n} new views; {len(overlaps)} historical matches in recorded source scope')
    if overlaps:raise SystemExit(2)
if __name__=='__main__':main()
