"""Build portable core/full ZIPs and validate every member by read-to-EOF."""
import hashlib,json,platform,zipfile
from pathlib import Path
import numpy


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    root=Path.cwd();assert root.name=='ssvc_flow'
    out=root/'artifacts/pilot_four_20261004';pilot=root/'docs/prospective_selection/pilot_four_20261004'
    files={}
    for folder in [pilot,root/'docs/prospective_selection/first_four_20261004',root/'docs/prospective_selection/design']:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files['ssvc_flow/'+str(p.relative_to(root))]=p
    for p in (root/'src').rglob('*.py'):files['ssvc_flow/'+str(p.relative_to(root))]=p
    for p in (root/'scripts/prospective_selection').glob('*.py'):files['ssvc_flow/'+str(p.relative_to(root))]=p
    for name in ['test_pilot_four.py','test_first_four_archive.py','test_prospective_selectors.py','test_prospective_features.py']:
        p=root/'tests'/name;files['ssvc_flow/tests/'+name]=p
    for name in ['RAW_MANIFEST.json','EXPORT_RECEIPT.json']:
        files['raw/'+name]=out/name
    files['START_HERE_zh.md']=out/'START_HERE_zh.md'
    manifest={name:{'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in files.items()}
    metadata={'schema':'first-four-pilot-analysis-package-v1','python':platform.python_version(),'numpy':numpy.__version__,
        'scope':'Exploratory nested lineage replay and full first-four raw outputs; no model/Adam tensor payload or final T',
        'files':manifest}
    receipts=[]
    for full in [False,True]:
        name='SSVC_FOUR_LINEAGE_FULL_DATA_20261004.zip' if full else 'SSVC_FOUR_LINEAGE_ANALYSIS_CORE_20261004.zip'
        target=out/name;assert not target.exists()
        mf=dict(metadata);mf['files']=dict(manifest)
        if full:
            raw=out/'FIRST_FOUR_RAW_ANALYSIS.tar.gz';mf['files']['raw/'+raw.name]={'bytes':raw.stat().st_size,'sha256':sha(raw)}
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for key,p in files.items():z.write(p,key)
            if full:z.write(raw,'raw/'+raw.name,compress_type=zipfile.ZIP_STORED)
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(mf,indent=2))
        with zipfile.ZipFile(target) as z:
            names=z.namelist();assert len(names)==len(set(names))
            assert set(names)==set(mf['files'])|{'PACKAGE_MANIFEST.json'}
            for n in names:
                assert not n.startswith('/') and '..' not in Path(n).parts
                data=z.read(n) # CRC is checked while reading all bytes.
                if n in mf['files']:assert hashlib.sha256(data).hexdigest()==mf['files'][n]['sha256']
        receipts.append(dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),status='PASS',unique_safe_paths=True,all_members_crc_and_sha256_verified=True,includes_raw_outputs=full))
        print(json.dumps(receipts[-1]),flush=True)
    (pilot/'PACKAGE_VERIFICATION.json').write_text(json.dumps(receipts,indent=2)+'\n')
    (out/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+Path(r['path']).name+'\n' for r in receipts))

if __name__=='__main__':main()
