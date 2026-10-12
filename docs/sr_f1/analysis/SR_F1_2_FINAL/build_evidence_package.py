"""Build and verify a bounded final analysis package; no server mutation."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

repo=Path(sys.argv[1]).resolve()
dest=repo/'artifacts/sr_f12_final_20261012'
report=repo/'docs/sr_f1/analysis/SR_F1_2_FINAL'
archive=dest/'SR_F1_2_FINAL_EVIDENCE_20261012.zip'
def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

files=set()
for directory in [report,report.parent/'SR_F1_2_THREE_SEED_TRAIN_FIT',dest/'evidence',dest/'audit']:
    files.update(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink()
                 and '__pycache__' not in p.parts and p.name not in ['DELIVERY_RECEIPT.json'])
for name in ['docs/sr_f1/EXECUTION_STATUS_zh.md','docs/sr_f1/SR_F1_2_EARLY_SEALED_EVALUATION_zh.md',
             'docs/sr_f1/SR_F1_2_ROLLING_SCHEDULING_zh.md','docs/sr_f1/SR_F1_2_CHECKPOINT_STORAGE_zh.md']:
    files.add(repo/name)
files.update((repo/'docs/sr_f1/amendments/SR_F1_2_20261010').glob('*'))
files={p for p in files if p.is_file()}
manifest={str(p.relative_to(repo)):dict(bytes=p.stat().st_size,sha256=checksum(p)) for p in sorted(files)}
intro='''# SR-F1.2 最终报告与可验证证据

入口：[中文报告及复现说明](docs/sr_f1/analysis/SR_F1_2_FINAL/START_HERE_zh.md)。

先运行 `python verify_package.py` 校验各文件；再按入口命令复算。包内含已释放评分和审计元数据，没有模型权重、完整原始解释或大型checkpoint；服务器原始路径和hash保留，非自包含重训练包。
'''
verifier='''import hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'SHA256SUMS.json').read_text())
for relative,record in manifest.items():
    p=root/relative
    assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
    assert p.resolve().is_relative_to(root) and p.is_file()
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert p.stat().st_size==record['bytes'] and h.hexdigest()==record['sha256'],relative
print('PASS',len(manifest),'files')
'''
extras={'START_HERE_zh.md':intro.encode(),'verify_package.py':verifier.encode()}
for name,body in extras.items():manifest[name]=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):z.write(p,str(p.relative_to(repo)))
    for n,b in extras.items():z.writestr(n,b)
    z.writestr('SHA256SUMS.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(archive) as z:
    names=z.namelist()
    assert len(names)==len(set(names)) and set(names)==set(manifest)|{'SHA256SUMS.json'}
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    assert z.testzip() is None
    for name,record in manifest.items():
        h=hashlib.sha256();size=0
        with z.open(name) as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
        assert size==record['bytes'] and h.hexdigest()==record['sha256'],name
result=dict(status='VERIFIED_ZIP_CRC_AND_ALL_SHA256',zip=str(archive),bytes=archive.stat().st_size,
            sha256=checksum(archive),content_files=len(manifest),zip_entries=len(names),
            uncompressed_bytes=sum(v['bytes'] for v in manifest.values()),
            unique_safe_paths=True,all_entries_read_to_eof=True,
            scope='Offline released-score analysis and audit evidence; excludes weights/full checkpoints/full raw answers')
(dest/'PACKAGE_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
