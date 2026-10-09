"""Streaming independent re-score of a runtime raw-output JSONL.
Required raw fields: qid, raw_text. All additional runtime identity fields are retained.
Examples: python reference/recompute_scores.py --raw RAW.jsonl --out SCORED.jsonl
"""
import argparse,json
from pathlib import Path
from semantic_contract import score,canonical

def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,default=Path('manifests/TASKS_GOLD_AUDIT_ONLY.jsonl'))
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.raw.resolve()==a.out.resolve():raise ValueError('Never overwrite raw evidence')
    tasks={t['qid']:t for t in map(json.loads,a.tasks.read_text().splitlines())}
    n=0
    with a.raw.open() as fi,a.out.open('x') as fo:
        for line in fi:
            rec=json.loads(line);t=tasks[rec['qid']]
            rec['independent_score']=score(rec['raw_text'],t['world'],t['query'])
            fo.write(canonical(rec)+'\n');n+=1
    print(json.dumps({'rescored_rows':n,'raw_preserved':True}))
if __name__=='__main__':main()
