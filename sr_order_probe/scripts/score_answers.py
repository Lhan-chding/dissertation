#!/usr/bin/env python3
"""Score one completed split, preserving model failures; no best-of retries."""
import argparse, json, sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from order_probe.common import read_jsonl,write_jsonl
from order_probe.measurement import measure

def main():
    p=argparse.ArgumentParser(); p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--answers',type=Path,required=True);p.add_argument('--split',choices=['MAIN','SMOKE'],default='MAIN')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    requests=read_jsonl(a.inputs/f'{a.split}_REQUESTS_PUBLIC.jsonl')
    tasks={t['qid']:t for t in read_jsonl(a.inputs/f'{a.split}_TASKS_PRIVATE.jsonl')}
    raw=read_jsonl(a.answers);byid={r['request_id']:r for r in raw}
    if len(byid)!=len(raw): raise SystemExit('Duplicate saved requests: no silent last-row selection')
    if set(byid)!={r['request_id'] for r in requests}: raise SystemExit('Incomplete/extra split: report INCOMPLETE, do not impute system failures')
    out=[]
    for req in requests:
        row=byid[req['request_id']];task=tasks[req['qid']]
        if row['seed']!=req['seed']: raise ValueError('seed mismatch')
        if row['assistant_prefill']!=req['assistant_prefill']: raise ValueError('prefill changed')
        full=req['assistant_prefill']+row['suffix_text']
        if row.get('full_text',full)!=full: raise ValueError('full text is not literal prefill+suffix')
        out.append({k:req[k] for k in ('request_id','qid','root_id','split','parent','condition','branch','draw')}|
                   {k:task[k] for k in ('rename','structure','placebo_side')}|
                   {'finish_reason':row['finish_reason'],**measure(full,task,req['condition'],row['finish_reason'])})
    if a.out.exists(): raise SystemExit('Preserve existing scores; use a new explicit path')
    write_jsonl(a.out,out);print(f'Scored {len(out)} existing responses; generated 0; updates 0')
if __name__=='__main__':main()
