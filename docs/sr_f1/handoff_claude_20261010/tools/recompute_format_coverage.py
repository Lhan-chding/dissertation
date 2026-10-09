#!/usr/bin/env python3
"""Offline original-text scoring only; never loads a model or runs an experiment."""
import collections
import json
from pathlib import Path
import sys

def compute(root):
    root=Path(root).resolve()
    sys.path.insert(0,str(root/'code/src'))
    from sr_f1.contract import digest,score
    tasks={r['qid']:r for r in map(json.loads,(root/'code/docs/sr_f1/package/manifests/TASKS_GOLD_AUDIT_ONLY.jsonl').read_text().splitlines())}
    summary={}; rows=[]
    for label in ['before','confirm','after']:
        counts=collections.Counter(); finished=collections.Counter(); covered=truncated=0
        files=sorted((root/'evidence/server/engineering/format'/label).glob('*-[0-7].json'))
        for f in files:
            r=json.loads(f.read_text()); assert r['record_hash']==digest({k:v for k,v in r.items() if k!='record_hash'}),str(f)
            t=tasks[r['qid']];s=score(r['raw_text'],t['world'],t['query'])
            ok=bool(s['L_json'] and s['L_answer'] and s['L_evidence'])
            kind='covered' if ok else 'whole_json_invalid' if not s['L_json'] else 'answer_not_scorable' if not s['L_answer'] else 'evidence_not_scorable'
            counts[kind]+=1;covered+=int(ok);truncated+=int(r['truncated']);finished[r['finish_reason']]+=1
            rows.append(dict(panel=label,file=f.relative_to(root).as_posix(),qid=r['qid'],draw=r['sample_index'],failure_type=kind,covered=ok,truncated=r['truncated'],raw_text=r['raw_text'],record_hash=r['record_hash']))
        summary[label]=dict(records=len(files),expected_records=256,complete=len(files)==256,covered=covered,coverage_of_included_only=covered/len(files) if files else None,truncated=truncated,finish_reasons=dict(finished),failure_types=dict(counts),all_raw_hashes_valid=True)
        receipt=root/'evidence/server/engineering/format'/label/'COVERAGE.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text());assert len(files)==saved['responses'] and covered==saved['covered'] and truncated==saved['truncated']
            summary[label]['complete_server_receipt_reproduced']=True
    return dict(model_calls=0,read_only=True,warning='after may be partial; before and confirm are different panels, not a paired learning contrast',panels=summary),rows

if __name__=='__main__':
    value,_=compute(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
    print(json.dumps(value,ensure_ascii=False,indent=2))
