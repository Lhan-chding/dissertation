"""Independent enumeration and invariants; no model inference."""
from __future__ import annotations
import json,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
from semantic_contract import *


def independent_answer(w,q):
    # Alternative evaluator using entity rows; deliberately not calling execute/selected_indices.
    cats=w['categories'];rs=[{'i':i,'A':Fraction(w['series']['Alpha'][i]),'B':Fraction(w['series']['Beta'][i])} for i in range(4)]
    if q['family']=='CROSS':
        rs=[r for r in rs if r['A']>r['B'] or (q['comparator']=='ge' and r['A']==r['B'])];vals=[r['A'] for r in rs]
    elif q['family']=='THRESHOLD':
        rs=[r for r in rs if r['A']>q['threshold'] or (q['comparator']=='ge' and r['A']==q['threshold'])];vals=[r['A'] for r in rs]
    elif q['family']=='TOPK':
        rs=sorted(rs,key=lambda r: ((-1 if q['direction']=='largest' else 1)*r['A'],r['i']))[:2];vals=[r['B'] for r in rs]
    else:
        rs=[r for r in rs if (r['i']>=q['start'] if q['include_start'] else r['i']>q['start']) and r['i']<=q['end']];vals=[r['A']-r['B'] for r in rs]
    a=q['aggregate']
    if a=='sum':return sum(vals,Fraction(0))
    if a=='count':return Fraction(len(vals))
    if not vals:return None
    funcs={'mean':lambda:sum(vals,Fraction(0))/len(vals),'range':lambda:sorted(vals)[-1]-sorted(vals)[0], 'min':lambda:sorted(vals)[0], 'max':lambda:sorted(vals)[-1]}
    return funcs[a]()


def main(root=Path('.')):
    ms=root/'manifests';rs=[json.loads(x) for x in (ms/'TASKS_GOLD_AUDIT_ONLY.jsonl').read_text().splitlines()]
    ins=[json.loads(x) for x in (ms/'MODEL_INPUTS.jsonl').read_text().splitlines()]
    roots=[json.loads(x) for x in (ms/'ROOTS_AUDIT_ONLY.jsonl').read_text().splitlines()]
    assert len({x['qid'] for x in rs})==len(rs)==4800
    assert len({x['root_id'] for x in roots})==len(roots)==496
    assert all(set(x)=={'qid','image_file','text','plain_text'} for x in ins)
    mismatch=[];families=defaultdict(dict)
    for r in rs:
        answer=independent_answer(r['world'],r['query'])
        if encode_answer(answer)!=r['answer']:mismatch.append(r['qid'])
        s=score(canonical(gold_output(r['world'],r['query'])),r['world'],r['query'])
        assert s['J']==1,r['qid']
        families[(r['root_id'],r['chart'])][r['variant']]=r
    assert not mismatch
    paired=0
    for key,vs in families.items():
        if len(vs)!=8:continue
        for v in ['v1','v2','v5']:assert vs[v]['answer']==vs['v0']['answer']
        assert vs['v0']['answer']!=vs['v3']['answer']
        assert vs['v6']['answer']!=vs['v7']['answer']
        assert vs['v0']['source_image_key']==vs['v5']['source_image_key']
        assert vs['v6']['source_image_key']==vs['v7']['source_image_key']
        paired+=1
    schedulechecks=0
    trainids={r['qid'] for r in rs if r['pool']=='TRAIN'}
    for p in sorted((ms/'schedules').glob('*.jsonl')):
        slots=[json.loads(x) for x in p.read_text().splitlines()]
        assert {x['qid'] for x in slots}==trainids and len(slots)==1536
        for st in range(1,97):
            batch=[x for x in slots if x['step']==st]
            assert len(batch)==16 and set(Counter((x['family'],x['chart']) for x in batch).values())=={2}
        schedulechecks+=1
    trainasts={r['ast_skeleton'] for r in rs if r['pool']=='TRAIN'}
    compasts={r['ast_skeleton'] for r in rs if r['pool']=='TEST_COMPOSITION'}
    assert not trainasts&compasts
    for r in rs:
        if r['pool']=='TEST_LANGUAGE':assert r['surface']=='test_c'
        if r['pool']=='TRAIN':assert r['surface']!='test_c' and r['style']!='heldout'
    declared=json.loads((ms/'MANIFEST_SHA256.json').read_text())
    for n,h in declared.items():assert hashlib.sha256((ms/n).read_bytes()).hexdigest()==h
    report={'root_count':len(roots),'questions':len(rs),'independent_gold_mismatches':mismatch,
            'gold_outputs_scored_J1':len(rs),'complete_chart_variant_families_checked':paired,
            'training_streams_checked':schedulechecks,'steps_per_stream':96,'prompts_per_step':16,
            'train_composition_AST_overlap':0,'manifest_hashes_checked':len(declared)}
    (root/'validation'/'MANIFEST_VALIDATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
