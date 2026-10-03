
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
target=p/'ANALYSIS_METADATA_20261003.json'
with target.open('x') as f: json.dump(out,f,separators=(',',':'),allow_nan=False)
print(str(target),len(out['branches']),target.stat().st_size)
