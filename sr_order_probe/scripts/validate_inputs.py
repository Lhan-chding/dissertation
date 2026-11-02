#!/usr/bin/env python3
"""Validate candidate data/condition bytes. Never consumes model outcomes."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from order_probe.common import file_sha,read_jsonl,write_json,digest,text_sha
from order_probe.data import validate_root
from order_probe.inputs import prompt,assistant_prefill,validate_prefix_triplet,NATURAL,PREFIX

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((R/'config/plan.json').read_text());fmt=(R/'assets/format_contract_v3.txt').read_text();summary={}
    for split,n in [('MAIN',4320),('SMOKE',216)]:
        roots=json.loads((a.inputs/f'{split}_ROOTS_PRIVATE.json').read_text())
        tasks=read_jsonl(a.inputs/f'{split}_TASKS_PRIVATE.jsonl');byqid={t['qid']:t for t in tasks}
        reqs=read_jsonl(a.inputs/f'{split}_REQUESTS_PUBLIC.jsonl')
        assert len(reqs)==n and len({r['request_id'] for r in reqs})==n
        for root in roots:
            validate_root(root)
            for t in root['views']:validate_prefix_triplet(t)
        for req in reqs:
            t=byqid[req['qid']]
            assert req['prompt']==prompt(t,req['condition'],fmt)
            assert req['assistant_prefill']==assistant_prefill(t,req['condition'])
            assert req['prompt_sha256']==text_sha(req['prompt']) and req['prefill_sha256']==text_sha(req['assistant_prefill'])
            assert req['image_sha256']==file_sha(a.inputs/req['image_file'])
            assert not any(k in req for k in ['world','gold','rename','boundary_indices'])
            assert req['sampling']==cfg['generation']
        summary[split]={'roots':len(roots),'views':len(tasks),'requests':len(reqs),'parents':dict(Counter(r['parent'] for r in reqs)),
                        'conditions':dict(Counter(r['condition'] for r in reqs))}
    for item in json.loads((a.inputs/'MANIFEST.json').read_text()):
        assert file_sha(a.inputs/item['path'])==item['sha256'],item['path']
    write_json(a.out,{'status':'PASS_CPU_DATA_ONLY','summary':summary,'actual_processor':'NOT_RUN',
         'server_runtime':'NOT_RUN','new_model_inference':0,'optimizer_updates':0})
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
