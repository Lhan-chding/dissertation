"""Deterministic, model-free generation of SR-F1 worlds, questions, gold and streams.
Run from package root: python reference/build_manifest.py --out manifests
Produces no model inference, GPU training or claims of image readability.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from semantic_contract import *

POOL_ROOTS = {'ENGINE':8,'FORMAT':16,'FORMAT_CONFIRM':16,'BRIDGE':8,
              'TRAIN':96,'MONITOR':32,'TEST_ID':128,
              'TEST_COMPOSITION':64,'TEST_LANGUAGE':64,'TEST_RENDER':64}
PAIRED_SEEDS = [71001,71002,71003]


def make_root(pool: str, i: int, used: set) -> tuple[str,dict,dict]:
    f=FAMILIES[i%4];root=f'{pool.lower()}-{i:04d}'
    rng=random.Random(stable_seed(VERSION,pool,i,'world'))
    for attempt in range(10000):
        a=rng.sample(range(10,91),4); b=rng.sample(range(10,91),4)
        w={'categories':['January','February','March','April'], 'series':{'Alpha':a,'Beta':b}, 'unit':'count'}
        q=base_query(f,50)
        if f=='INTERVAL' and not (a[1]-b[1]>max(a[2]-b[2],a[3]-b[3])):continue
        # Includes some genuinely empty masks; not selected using any model response.
        target_empty=(i//4)%8==0
        if f in ('CROSS','THRESHOLD'):
            if (len(selected_indices(w,q))==0) != target_empty:continue
        vv=variants(w,q,root)
        if execute(vv[6][1],vv[6][2])==execute(vv[7][1],vv[7][2]):continue
        sigs={sha_obj({'Alpha':v[1]['series']['Alpha'],'Beta':v[1]['series']['Beta']}) for v in vv}
        if sigs & used:continue
        used.update(sigs)
        return root,w,q
    raise RuntimeError(f'Cannot construct {root}; do not silently reduce a cell.')


def build(outdir: Path) -> dict:
    outdir.mkdir(parents=True,exist_ok=True)
    used=set(); records=[];streams=[];roots=[]
    for pool,nroots in POOL_ROOTS.items():
        for i in range(nroots):
            root,w,q=make_root(pool,i,used)
            roots.append({'root_id':root,'pool':pool,'world':w,'base_query':q,'family':q['family']})
            if pool=='TEST_COMPOSITION':
                vv=[('v0',w,composition_query(q))]
            elif pool in ('FORMAT','FORMAT_CONFIRM','TEST_LANGUAGE','TEST_RENDER'):
                vv=[('v0',w,q)]
            else: vv=variants(w,q,root)
            for chart in ('grouped_bar','line'):
                for v,ww,qq in vv:
                    surface='test_c' if pool=='TEST_LANGUAGE' else 'train_b' if v=='v5' else 'train_a'
                    style='heldout' if pool=='TEST_RENDER' else 'equivalent' if v=='v1' else 'standard'
                    qid=f'{root}-{chart}-{v}'
                    image_identity=sha_obj({'world':ww,'chart':chart,'style':style})
                    image_file=f'images/{pool}/{image_identity}.png'
                    text=model_text(ww,qq,surface)
                    plain=model_text(ww,qq,surface,True)
                    rec={'qid':qid,'pool':pool,'root_id':root,'family':q['family'],'chart':chart,
                         'variant':v,'surface':surface,'style':style,'world':ww,'query':qq,
                         'image_file':image_file,'source_image_key':image_identity,
                         'required_evidence':gold_evidence(ww,qq),'answer':encode_answer(execute(ww,qq)),
                         'selected_indices':selected_indices(ww,qq),'prompt_hash':sha_obj(text),
                         'plain_prompt_hash':sha_obj(plain),'ast_skeleton':ast_skeleton(qq)}
                    records.append(rec)
                    streams.append({'qid':qid,'image_file':image_file,'text':text,'plain_text':plain})
    def writejl(name,rows):
        p=outdir/name
        p.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
    writejl('ROOTS_AUDIT_ONLY.jsonl',roots)
    writejl('TASKS_GOLD_AUDIT_ONLY.jsonl',records)
    writejl('MODEL_INPUTS.jsonl',streams)
    train=[r for r in records if r['pool']=='TRAIN']
    strata=defaultdict(list)
    for r in train:strata[(r['family'],r['chart'])].append(r['qid'])
    assert len(train)==1536 and len(strata)==8 and all(len(v)==192 for v in strata.values())
    streamdir=outdir/'schedules';streamdir.mkdir(exist_ok=True)
    for seed in PAIRED_SEEDS:
        buckets={k:sorted(v,key=lambda qid:stable_seed(seed,'input',qid)) for k,v in strata.items()}
        slots=[]
        for step in range(96):
            batch=[(k,qid) for k,v in buckets.items() for qid in v[2*step:2*step+2]]
            batch.sort(key=lambda z:stable_seed(seed,step,z[1]))
            for slot,((family,chart),qid) in enumerate(batch):
                slots.append({'step':step+1,'slot':slot,'qid':qid,'family':family,'chart':chart,
                              'paired_seed':seed,'rollout_seeds':[stable_seed(seed,qid,j,'rollout') for j in range(8)]})
        assert len(slots)==1536 and len({r['qid'] for r in slots})==1536
        writejl(Path('schedules')/f'train_seed_{seed}.jsonl',slots)
    runs=[{'run_id':f'SRF1_{arm}_s{seed}','arm':arm,'paired_seed':seed,'common_start':'SRF1_COMMON_START',
           'schedule':f'manifests/schedules/train_seed_{seed}.jsonl','H':96,'B':16,'G':8,
           'checkpoint_steps':[0,32,64,96]} for seed in PAIRED_SEEDS for arm in ARMS]
    (outdir/'RUN_MATRIX.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
    # Train-fit sentinel is fixed BEFORE results and is sampled from training, never a test claim.
    sentinel=[]
    for key,vals in strata.items():
        sentinel.extend(sorted(vals,key=lambda qid:stable_seed('train_fit',qid))[:4])
    (outdir/'TRAIN_FIT_QIDS.json').write_text(json.dumps(sentinel,indent=2),encoding='utf-8')
    poolcounts=Counter(r['pool'] for r in records)
    skeletons={r['ast_skeleton'] for r in train}
    comp={r['ast_skeleton'] for r in records if r['pool']=='TEST_COMPOSITION'}
    assert not skeletons & comp
    assert len(records)==4800 and len(roots)==496
    summary={'version':VERSION,'root_count':len(roots),'task_count':len(records),
             'pool_roots':POOL_ROOTS,'pool_tasks':dict(poolcounts),'training_paths':len(runs),
             'training_steps_total':len(runs)*96,'training_prompt_exposures':len(runs)*96*16,
             'training_rollouts':len(runs)*96*16*8,
             'train_ast_count':len(skeletons),'composition_ast_count':len(comp),
             'ast_overlap':len(skeletons&comp),'images_rendered':False,
             'unique_images_to_render':len({r['image_file'] for r in records})}
    (outdir/'BUILD_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    hashes={str(p.relative_to(outdir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(outdir.rglob('*')) if p.is_file() and p.name!='MANIFEST_SHA256.json'}
    (outdir/'MANIFEST_SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('manifests'))
    print(json.dumps(build(p.parse_args().out),indent=2))
