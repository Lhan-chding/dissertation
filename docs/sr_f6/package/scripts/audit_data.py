"""Construct the registered reference banks and audit without any model queries.
Only DEV_CAL examples are written out. Formal test outcomes remain unused.
The reference CLASSOUT arithmetic is covered; real-processor readability is NOT covered.
"""
import json,sys,time
from pathlib import Path
from collections import defaultdict,Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from srf6_ref.design import build_bank,visible_prompt,make_classout_root
from srf6_ref.core import fingerprint,FAMILIES
root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/experiment.json').read_text());out={};seen={};fail=[];start=time.time()
for split,n in cfg['data']['roots_per_family'].items():
    if split=='TEST_CLASSOUT':
        bank=[make_classout_root(f,i,cfg['data']['seeds'][split]) for f in FAMILIES for i in range(n)]
        out[split]={'roots':len(bank),'reference_views_including_both_factors':len(bank)*4,
          'construction_attempt_max':max(x['certificate']['attempt'] for x in bank),'coverage':'not optimized; held-out double edits'}
        for r in bank:
            for task in r['tasks']['P0']:
                fp=fingerprint(task['world']);prev=seen.setdefault(fp,split)
                if prev!=split:fail.append({'from':prev,'to':split,'qid':task['qid']})
        print(split,len(bank),'CLASSOUT',flush=True)
        continue
    ood=split=='TEST_COMPOSITION';zero=ood
    bank=build_bank(split,n,cfg['data']['seeds'][split],ood,zero)
    stats={'roots':len(bank),'reference_views_including_both_factors':len(bank)*8,'construction_attempt_max':max(x['certificate']['attempt'] for x in bank),'by_family':{}}
    stats['fraction_roots_strict_coverage_gain']=sum(x['certificate']['coverage_P1']>x['certificate']['coverage_P0']+1e-12 for x in bank)/len(bank)
    for f in FAMILIES:
        sub=[x for x in bank if x['tasks']['P0'][0]['family']==f]
        c0=sum(x['certificate']['coverage_P0'] for x in sub)/len(sub);c1=sum(x['certificate']['coverage_P1'] for x in sub)/len(sub)
        stats['by_family'][f]={'coverage_P0':c0,'coverage_P1':c1,'gap':c1-c0,
            'engineered_anchors':sum(x['certificate']['engineered_collision_anchor'] for x in sub),
            'anchor_with_any_candidate_alias':sum(bool(x['certificate']['base_alias_rules']) for x in sub)}
    for r in bank:
        for p in ('P0','P1'):
            for task in r['tasks'][p]:
                fp=fingerprint(task['world']);prev=seen.setdefault(fp,split)
                if prev!=split:fail.append({'from':prev,'to':split,'qid':task['qid']})
    out[split]=stats
    print(split,stats['roots'],stats['construction_attempt_max'],flush=True)
    if split=='DEV_CAL':
        examples=[bank[next(i for i,r in enumerate(bank) if r['tasks']['P0'][0]['family']==f)] for f in FAMILIES]
        (root/'examples/DEV_ROOTS_PRIVATE.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2,default=str))
        visible=[]
        for r in examples:
            for p in ('P0','P1'):
                for t in r['tasks'][p]:visible.append({'qid':t['qid'],'prompt':visible_prompt(t),'image_artifact_needed':True})
        (root/'examples/MODEL_INPUT_TEXT_PREVIEW.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in visible)+'\n')
result={'status':'PASS' if not fail else 'FAIL','elapsed_seconds':time.time()-start,'banks':out,'cross_split_duplicates':fail,
 'limitations':['Only CPU task/reference arithmetic and finite candidate coverage; no model-outcome effect or actual chart-processor readability tested.',
 'Natural-interface, legacy and ChartQA panels require production manifests; all rendered inputs require real processor checks.',
 'P1 coverage is relative to active candidates in the stated edit catalog, not all possible model failures.']}
(root/'qa/full_reference_data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('status',result['status']);sys.exit(0 if not fail else 1)
