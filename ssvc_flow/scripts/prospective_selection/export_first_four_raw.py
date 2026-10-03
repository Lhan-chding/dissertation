"""Server-side, read-only first-four analysis archive; no tensor payloads or T data.

Writes only a new archive/export directory under this experiment. Preserves every
JSON/JSONL artifact under selected completed task directories, including prior
attempt traces. Authoritative COMMIT receipts distinguish successful records.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

BASE=Path('/projects/varunssd/louis-ssvc/prospective_selection_v2_20260924').resolve()
ROOT=BASE/'campaign'
OUT=BASE/'pilot_four_20261004_raw_export_v2'
LIDS={str(x) for x in range(61001,61005)}
SMALL={'.json','.jsonl','.txt','.log','.py','.yaml','.yml','.toml','.md','.sh'}


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    files={};omitted=[]
    def add(p,name=None):
        p=p.resolve()
        assert p.is_file()
        name=name or str(p.relative_to(BASE))
        assert not name.startswith('/') and '..' not in Path(name).parts
        if name in files:assert files[name]==p
        files[name]=p
    def walk(folder):
        for p in sorted(folder.rglob('*')):
            if not p.is_file():continue
            if p.suffix in SMALL:add(p)
            else:omitted.append(dict(path=str(p),bytes=p.stat().st_size,reason='Tensor/binary/cache kept server-side; no rehash'))
    for stem in ['sources','prestate','branches','historical_E']:
        for folder in sorted((ROOT/stem).iterdir()):
            if stem=='historical_E' or folder.name.split('_')[0] in LIDS:walk(folder)
    tids=[]
    for p in (ROOT/'tasks').glob('*.json'):
        t=json.loads(p.read_text())
        if str(t['payload'].get('lineage_id')) in LIDS or t['kind']=='historical_e':tids.append(p.stem)
    assert len(tids)==116
    for tid in tids:
        for stem in ['tasks','completed','intents','submissions']:
            p=ROOT/stem/(tid+'.json')
            if p.exists():add(p)
        for stem in ['recoveries','cancelled_submissions','rejected_submissions','submission_errors','logs','startup']:
            for p in (ROOT/stem).glob(tid+'*'):
                if p.is_dir():walk(p)
                elif p.is_file() and p.suffix in SMALL:add(p)
    for name in ['protocol.json','FIRST_FOUR_DELIVERED.json']:add(ROOT/name)
    walk(ROOT/'schedules')
    old=BASE/'fixed_pipelines_20260930'
    for p in old.rglob('*'):
        if p.is_file() and p.suffix in SMALL:add(p)
    for p in BASE.glob('*.json'):
        if any(k in p.name for k in ['VERIFIED','LAUNCH','RECOVERY','CANCELLED','CODE_VERSION']):add(p)
    add(BASE/'runtime.json')
    for snap in ['code_bb23e9b','code_e59b9ee','code_414df78','code_afd3119']:
        walk(BASE/snap/'ssvc_flow/src')
    prepared=json.loads((BASE/'prepared_development.json').read_text())
    subset={k:prepared[k] for k in ['schema','source_prompts','continuation_prompts','source_schedule','branch_schedules','input_bindings','data_id']}
    subset['panels']={k:prepared['panels'][k] for k in ['P','D']}
    runtime=json.loads((BASE/'runtime.json').read_text());hist=json.loads(Path(runtime['historical_runtime']).read_text())
    epanel=json.loads(Path(hist['panels']['path']).read_text())['panels']['E']
    subset['panels']['E_old']=epanel
    subset['omitted']='T/T_H8 excluded. D_H8 is deterministic _diagnostic_subset(D); preserve original evaluator code.'
    prep=OUT/'prepared_first_four_ONLY.json';prep.write_text(json.dumps(subset,ensure_ascii=False)+'\n');add(prep,'inputs/prepared_first_four_ONLY.json')
    image_index={};missing_images=[]
    for prompt in subset['source_prompts']+subset['continuation_prompts']+sum(subset['panels'].values(),[]):
        scene=prompt['scene'];p=(Path(prompt.get('data_root') or hist['data_root'])/scene['image_path']).resolve()
        if p.is_file():
            name='inputs/images/'+scene['image_hash']+p.suffix
            add(p,name);image_index[str(p)]=name
        else:missing_images.append({'prompt_id':prompt['prompt_id'],'path':str(p),'interface':prompt['interface']})
    manifest=[]
    for i,(name,p) in enumerate(sorted(files.items())):
        manifest.append(dict(name=name,source=str(p),bytes=p.stat().st_size,sha256=digest(p)))
        if i%5000==0:print('HASHED',i,len(files),flush=True)
    meta={'status':'RAW_ANALYSIS_EXPORT','scope':'First four lineages + 16 historical E, source/prestate/branch raw outputs and metadata, inputs and code; no final test',
          'files':manifest,'omitted_files':omitted,'image_path_map':image_index,'missing_images':missing_images,
          'not_a_training_rerun_bundle':True,'notes':['Base weights and optimizer/adapter tensors remain on server.','All selected-directory traces included; authoritative COMMIT receipts identify accepted samples.','No final T/T_H8 prompts or outcomes.','Outer model/decisions are packaged separately by local pilot.']}
    metadata=json.dumps(meta,ensure_ascii=False,indent=2).encode();(OUT/'RAW_MANIFEST.json').write_bytes(metadata)
    archive=OUT/'FIRST_FOUR_RAW_ANALYSIS.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for row in manifest:tar.add(row['source'],arcname=row['name'],recursive=False)
        info=tarfile.TarInfo('RAW_MANIFEST.json');info.size=len(metadata);tar.addfile(info,io.BytesIO(metadata))
    receipt={'status':'ARCHIVE_CREATED','files':len(files),'raw_bytes':sum(x['bytes'] for x in manifest),'archive_bytes':archive.stat().st_size,'sha256':digest(archive),'manifest_sha256':hashlib.sha256(metadata).hexdigest(),'omitted_tensor_files':len(omitted)}
    (OUT/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
