"""Bind already opened F6 DEV_EVAL v3 images. Reads explicit approved inputs only. No regeneration."""
import argparse,hashlib,json,shutil
from pathlib import Path
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):
    with p.open() as f:return [json.loads(l) for l in f if l.strip()]
def keyed(rows,key):
    m={r[key]:r for r in rows}
    if len(m)!=len(rows):raise ValueError('duplicate '+key)
    return m
def bind(frozen, private_path, v3_public, old_root, out):
    # The caller independently checks the source identities against the historical authorized phase manifest.
    cases=keyed(load(frozen/'private_cases.jsonl'),'case_id');inputs=keyed(load(frozen/'inputs.jsonl'),'input_id')
    ids={r['case_id'] for r in load(frozen/'evaluation_manifest.jsonl') if r['panel']=='retention8'}
    src=keyed(load(private_path),'qid'); pub=keyed(load(v3_public),'qid'); qids={cases[c]['task']['qid'] for c in ids}
    if len(qids)!=96 or set(src)!=qids or set(pub)!=qids:raise ValueError('requires exactly the 96 registered DEV_EVAL P0 rows')
    if out.exists():raise FileExistsError('new directory required')
    records=[]
    for c in sorted(ids):
        t=cases[c]['task'];q=t['qid'];p=pub[q];s=src[q];modelinp=p['model_input'];new=inputs[cases[c]['input_id']]
        if t['root_id']!=s['root_id'] or s['factor']!='P0' or s['split']!='DEV_EVAL' or t['world']!=s['world'] or t['query']!=s['query'] or t['gold']['answer']!=s['gold_answer']: raise ValueError('changed historical world/query/gold: '+q)
        if p['protocol']!='structured_v4' or modelinp['prompt']!=new['prompt'] or p.get('prompt_sha256')!=sha(new['prompt'].encode()): raise ValueError('historical v3 overlay prompt mismatch: '+q)
        img=(old_root/modelinp['image_file']).resolve();img.relative_to(old_root.resolve())
        if not img.is_file() or sha(img.read_bytes())!=modelinp['image_sha256']:raise ValueError('historical image mismatch: '+q)
        if new['image_file']!='legacy/'+q+'.png':raise ValueError('unexpected local target')
        records.append({'qid':q,'relative':new['image_file'],'source':str(img),'sha256':modelinp['image_sha256']})
    out.mkdir(parents=True);(out/'legacy').mkdir()
    for r in records:shutil.copyfile(r['source'],out/r['relative'])
    receipt={'status':'ORIGINAL_DEV_EVAL_P0_V3_BOUND','rows':len(records),'private_file_sha256':sha(private_path.read_bytes()),'public_file_sha256':sha(v3_public.read_bytes()),'files':records}
    (out/'LEGACY_BINDING.json').write_text(json.dumps(receipt,indent=2));print(receipt['status'],len(records))
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for k in ['frozen','private','public-v3','old-root','out']:a.add_argument('--'+k,type=Path,required=True)
    x=a.parse_args();bind(x.frozen,x.private,x.public_v3,x.old_root,x.out)
