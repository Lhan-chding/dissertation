#!/usr/bin/env python3
"""Build immutable candidate probe inputs; no model execution, no authorization receipt."""
import argparse, json, sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(R/'src'))
from order_probe.common import file_sha, write_json, write_jsonl
from order_probe.data import build_worlds, validate_root
from order_probe.inputs import request_rows, validate_prefix_triplet
from order_probe.render import render

def main():
    a=argparse.ArgumentParser(); a.add_argument('--out',type=Path,required=True)
    a.add_argument('--render',action='store_true')
    a.add_argument('--font',default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    args=a.parse_args()
    if args.out.exists() and any(args.out.iterdir()): raise SystemExit('Output exists: preserve it; do not overwrite frozen inputs')
    args.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((R/'config/plan.json').read_text()); fmt=(R/'assets/format_contract_v3.txt').read_text()
    totals={}; all_keys=[]
    for split in ('SMOKE','MAIN'):
        roots=build_worlds(cfg,split)
        for root in roots:
            validate_root(root)
            for task in root['views']: validate_prefix_triplet(task)
        public,private=request_rows(roots,cfg,fmt)
        if args.render:
            for task in private: render(task,args.out/'images'/(task['world_query_key']+'.png'),args.font)
        for req in public:
            p=args.out/req['image_file']
            req['image_sha256']=file_sha(p) if p.is_file() else None
        write_json(args.out/f'{split}_ROOTS_PRIVATE.json',roots)
        write_jsonl(args.out/f'{split}_TASKS_PRIVATE.jsonl',private)
        write_jsonl(args.out/f'{split}_REQUESTS_PUBLIC.jsonl',public)
        totals[split]={'roots':len(roots),'views':len(private),'requests':len(public)}
        all_keys += [v['unlabeled_world_query_key'] for v in private[::2]]
    if len(all_keys)!=len(set(all_keys)): raise AssertionError('Smoke and main numerical worlds overlap')
    write_json(args.out/'BUILD_RECEIPT.json',{'status':'CPU_CONSTRUCTION_ONLY','counts':totals,
        'images_rendered':args.render,'historical_exclusion':'SERVER_CHECK_REQUIRED',
        'processor_visual_check':'NOT_RUN','GPU_inference':'NOT_RUN','optimizer_updates':0,
        'config_sha256':file_sha(R/'config/plan.json'),
        'font_path_local':args.font if args.render else None,
        'font_sha256':file_sha(args.font) if args.render else None})
    manifest=[{'path':str(p.relative_to(args.out)),'bytes':p.stat().st_size,'sha256':file_sha(p)}
              for p in sorted(args.out.rglob('*')) if p.is_file()]
    write_json(args.out/'MANIFEST.json',manifest)
    print(json.dumps(totals,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
