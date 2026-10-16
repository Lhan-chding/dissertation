"""Read-only final ablation evidence capture; no model or scheduler mutation."""
import argparse, hashlib, io, json, os, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def encoded(obj):return (json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
a=p.parse_args();r=Path(a.root).resolve(strict=True);o=Path(a.output)
assert r==Path('/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation')
assert o.parent==r.parent/'archives' and o.name=='ABLATION_COMPLETE_20261016.tar.gz'
assert not o.exists() and not o.with_suffix(o.suffix+'.json').exists()
m=json.loads((r/'deployment/SOURCE_MANIFEST.json').read_text())
assert m['code_commit']=='652e52d530520a276b6420ff0f4ee479dcde6db1'
expected={x['path']:x for x in m['files']}
actual={x.relative_to(r/'code').as_posix():x for x in (r/'code').rglob('*') if x.is_file()}
assert set(expected)==set(actual)
for name,path in actual.items():
    assert not path.is_symlink() and path.suffix not in ('.pyc','.pyo')
    assert path.stat().st_size==expected[name]['bytes'] and sha(path)==expected[name]['sha256']
sys.path.insert(0,str(r/'code/src'))
from sr_f3_ce.ablation import verify_selection
from sr_f3_ce.orchestration import Slurm, TEACHER_QOS
v=verify_selection(r);assert v['outputs']==2304 and len(v['raw_evidence'])==9
slurm=Slurm();queue=slurm.queue('varun024');terminals={}
for item in v['raw_evidence']:
    saved=json.loads((r/item['terminal']['path']).read_text());job=saved['job_id']
    assert job not in queue
    current=slurm.terminal(job)
    for k in ('job_id','state','exit_code','user','account','qos','name'):assert saved[k]==current[k],(job,k)
    assert current['state']=='COMPLETED' and current['exit_code']=='0:0' and current['name'].endswith('-LH')
    terminals[job]=current
job='207281';assert job not in queue
current=slurm.terminal(job)
assert all(current.get(k)==x for k,x in dict(job_id=job,state='COMPLETED',exit_code='0:0',user='varun024',account='rose',qos=TEACHER_QOS,name='SRF3CE-ABLDRIVER-N0001-LH').items())
terminals[job]=current
end=json.loads((r/'operations/ablation_driver/drivers/driver-0001/END.json').read_text())
assert end['status']=='COMPLETE' and end['result']['selection_status']==v['status']
old=r.parent.parent/'sr_f3_ce_20261016'
assert sha(old/'STOP')=='69dec4010e4f81a9bb251b9fe4ceee2a98c5caea25a624d8904d59ef6d6e3b08'
paths=set()
for d in ('ablation','operations','authorization','deployment/native_cpu'):
    for f in (r/d).rglob('*'):
        assert not f.is_symlink(),str(f)
        if f.is_file():paths.add(f)
for name in ('RUN_IDENTITY.json','ABLATION_MANIFEST.json','PROTOCOL_SELECTION.json','NATIVE_AUDIT_SOURCE.json'):
    paths.add(r/name)
for f in (r/'deployment').iterdir():
    if f.is_file() and f.suffix in ('.json','.txt'):paths.add(f)
items=[]
for f in sorted(paths):
    assert f.is_file() and not f.is_symlink()
    items.append(dict(path=f.relative_to(r).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
status=dict(status='COMPLETE_FIXED_ABLATION_EVIDENCE_CAPTURE',captured_at=datetime.now(timezone.utc).isoformat(),root=str(r),outputs=2304,groups=576,raw_request_pending_originals=1152,selection_status=v['status'],selected_revision=v['selected_revision'],counts=v['counts'],decision_uses_only=v['decision_uses_only'],selection_sha256=sha(r/'PROTOCOL_SELECTION.json'),scientific_execution_authorized=False,science_paths_started=0,live_terminal_verification=terminals,unrelated_queue_preserved=queue,old_stop_preserved=True,capture_script_sha256=sha(Path(__file__)),omissions=['Base weights, parent adapters, copied data and candidate/code trees are not duplicated in this phase archive. Source, candidates, native evidence and old full history are separate members of the enclosing delivery. This is not a standalone retraining bundle.'])
status_bytes=encoded(status)
items.append(dict(path='CAPTURE_STATUS.json',bytes=len(status_bytes),sha256=hashlib.sha256(status_bytes).hexdigest()))
manifest_bytes=encoded(dict(schema_version=1,files=items,total_payload_bytes=sum(x['bytes'] for x in items)))
o.parent.mkdir(parents=True,exist_ok=True)
with o.open('xb') as out:
    with tarfile.open(fileobj=out,mode='w:gz',compresslevel=6) as tar:
        for item in items:
            if item['path']=='CAPTURE_STATUS.json':
                info=tarfile.TarInfo(item['path']);info.size=len(status_bytes);info.mode=0o644;tar.addfile(info,io.BytesIO(status_bytes))
            else:tar.add(r/item['path'],arcname=item['path'],recursive=False)
        info=tarfile.TarInfo('CAPTURE_MANIFEST.json');info.size=len(manifest_bytes);info.mode=0o644;tar.addfile(info,io.BytesIO(manifest_bytes))
for item in items:
    if item['path']!='CAPTURE_STATUS.json':
        f=r/item['path'];assert sha(f)==item['sha256'] and f.stat().st_size==item['bytes'],item['path']
receipt=dict(status='SERVER_CAPTURE_COMPLETE',archive=o.name,bytes=o.stat().st_size,sha256=sha(o),members=len(items)+1,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),capture_status=status)
with o.with_suffix(o.suffix+'.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(receipt,ensure_ascii=False))
