"""Read-only compact export of audited first-four metadata via SSH.

No model, checkpoint payload, raw text, training output, or final-test data is
read. Prior raw audits are reused; this export validates small receipt hashes.
"""
import argparse
import base64
import gzip
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import json,pathlib,hashlib,datetime
p=pathlib.Path('/projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/fixed_pipelines_20260930')
files={}
def read(path):
    path=path.resolve();data=path.read_bytes()
    files[str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    return json.loads(data)
c=read(p/'config.json');r=pathlib.Path(c['root']);reviewed=dict(c['baseline'])
for f in (p/'verified').glob('*.json'):
    a=read(f);reviewed[a['task_id']]=a
out={'schema':'first-four-progress-v1','extracted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'scope':'Audited development endpoints and historical E; no final tests or selector fitting',
     'verification':'Prior raw audits reused; receipt/task hashes rechecked; exported metadata hashed',
     'branches':[],'historical_E':[],'prestates':[],'missing':[],'status':read(p/'STATUS.json')}
for tid in c['targets']:
    task=read(r/'tasks'/(tid+'.json'))
    assert files[str((r/'tasks'/(tid+'.json')).resolve())]['sha256']==c['task_hashes'][tid]
    if tid not in reviewed:
        out['missing'].append(task);continue
    a=reviewed[tid];read(r/'completed'/(tid+'.json'))
    assert files[str((r/'completed'/(tid+'.json')).resolve())]['sha256']==a['completion_sha256']
    q=task['payload'];item={'task':task,'audit':a}
    if task['kind']=='branch':
        d=r/'branches'/q['origin_id']/q['recipe_id']/'repeat_1'
        item['training']=read(d/'COMPLETE.json');item['manifest']=read(d/'MANIFEST.json')
        item['evaluations']={str(h):read(d/'evaluations'/('H%02d'%h)/'COMPLETE.json') for h in (8,32)}
        # Individual optimizer metadata is small and contains no text or weights.
        item['updates']=[]
        for segment in item['training']['authoritative_segments']:
            seg=read(pathlib.Path(segment))
            for binding in seg['updates']:
                u=read(pathlib.Path(binding['path']))
                assert files[str(pathlib.Path(binding['path']).resolve())]['sha256']==binding['sha256']
                item['updates'].append({k:u[k] for k in ('step','loss','grad_norm_preclip','sampling_seconds','update_seconds')})
        out['branches'].append(item)
    else:
        assert task['kind']=='historical_e'
        item['evaluation']=read(r/'historical_E'/q['policy_id']/'COMPLETE.json')
        out['historical_E'].append(item)
for seed in range(61001,61005):
    for step in (32,96):
        origin=f'{seed}_t{step}';d=r/'prestate'/origin
        out['prestates'].append({'origin':origin,'complete':read(d/'COMPLETE.json'),
            'packets':{f.stem:read(f) for f in sorted(d.glob('Z*.json'))}})
out['source_files']=files
print('SSVC_EXPORT_JSON')
print(json.dumps(out,separators=(',',':'),allow_nan=False))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--metadata-only',action='store_true');p.add_argument('--compact',action='store_true');args=p.parse_args()
    if args.out.exists():raise ValueError('Keep earlier snapshots; output already exists')
    remote=REMOTE
    if args.metadata_only:
        start=remote.index('        for segment in item[')
        end=remote.index("        out['branches'].append(item)",start)
        remote=remote[:start]+remote[end:]
    if args.compact:
        remote=remote.replace("out['source_files']=files", """out['source_files']=files
out['parent_snapshot_sha256']=hashlib.sha256(json.dumps(out,separators=(',',':'),allow_nan=False).encode()).hexdigest()
for b in out['branches']:
    b.pop('training');b.pop('manifest')
    for ev in b['evaluations'].values():ev.pop('chunks',None)
for e in out['historical_E']:e['evaluation'].pop('chunks',None)
""")
    remote=remote.replace("print('SSVC_EXPORT_JSON')\nprint(json.dumps(out,separators=(',',':'),allow_nan=False))",
        "import gzip,base64\nprint('SSVC_GZIP_BASE64')\nprint(base64.b64encode(gzip.compress(json.dumps(out,separators=(',',':'),allow_nan=False).encode())).decode())")
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','ConnectionAttempts=1','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
        'eee-cluster',"python3 - <<'SSVC_EXPORT_SCRIPT'\n"+remote+"\nSSVC_EXPORT_SCRIPT"],
        stdin=subprocess.DEVNULL,capture_output=True,timeout=900)
    if result.returncode:
        raise RuntimeError('Read-only export failed: '+result.stderr[-3000:].decode(errors='replace'))
    data=json.loads(gzip.decompress(base64.b64decode(result.stdout.split(b'SSVC_GZIP_BASE64\n',1)[1].strip(),validate=True)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(data,f,separators=(',',':'),allow_nan=False)
    print(json.dumps({'branches':len(data['branches']),'historical_E':len(data['historical_E']),
        'prestates':len(data['prestates']),'missing':len(data['missing']),'out':str(args.out)}))

if __name__=='__main__':main()
