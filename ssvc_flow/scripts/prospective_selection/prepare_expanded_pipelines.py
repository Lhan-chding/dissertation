"""Register the fixed development/tuning remainder after first-four delivery.

Run once on the server, with the old entry stopped and no project GPU jobs.
This creates assignments only; it never submits jobs or changes crontab.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys


def assign(tasks, specs):
    """Whole lineages stay in one lane, so no cross-lane dependency wait."""
    lanes={str(i):[] for i in range(1,6)}
    coverage=[]
    for index,spec in enumerate(specs):
        lid=spec['seed']
        selected=[t for t in tasks.values() if t['payload'].get('lineage_id')==lid]
        counts=Counter(t['kind'] for t in selected)
        if counts!={'source':1,'prestate':2,'branch':22}:
            raise ValueError('Expected one source, two prestates, 22 branches per lineage')
        def order(t):
            p=t['payload']
            return ({'source':0,'prestate':1,'branch':2}[t['kind']],p.get('origin_step',0),p.get('recipe_id',''))
        ids=[t['task_id'] for t in sorted(selected,key=order)]
        lanes[str(index%5+1)].extend(ids);coverage.extend(ids)
    if len(coverage)!=len(set(coverage)) or set(coverage)!=set(tasks):
        raise ValueError('Fixed lists must cover all new tasks once')
    return lanes


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-config',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);args=p.parse_args()
    old=json.loads(args.old_config.read_text());control=args.control.resolve()
    spec=importlib.util.spec_from_file_location('expanded_controller',control/'server_controller.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    c.require(not (control/'config.json').exists(),'Config exists; inspect rather than repeat setup')
    c.require((Path(old['control'])/'STOP').exists(),'Old entry must be stopped before registration')
    c.require(not c.active_jobs(),'Existing project GPU jobs must finish before expansion')
    root=Path(old['root']);protocol=c.read(root/'protocol.json')
    sys.path.insert(0,old['project_root'])
    from src.prospective_selection.jobs import TaskRegistry
    from src.prospective_selection.orchestration import register_development
    receipt=root/'FIRST_FOUR_DELIVERED.json';delivered=c.read(receipt)
    c.require(delivered['status']=='FIRST_FOUR_DELIVERED' and delivered['verified_branches']==88
              and delivered['verified_historical_E']==16 and delivered['lineages']==[61001,61002,61003,61004],
              'First-four delivery receipt incomplete')
    for path,sha in delivered['report_files'].items():c.require(c.digest(path)==sha,'Delivered report changed')
    runtime=c.read(old['runtime']);prepared=runtime['prepared']
    c.require(c.digest(prepared['path'])==prepared['sha256'],'Prepared data changed')
    registry=TaskRegistry(root,protocol)
    before={t['task_id']:t for t in registry.tasks()}
    c.require(len(before)==117,'Unexpected old task scope; inspect prior setup')
    baseline={}
    for tid,t in before.items():
        f=root/'completed'/(tid+'.json');v=c.read(f)
        c.require(v['receipt']['status']=='COMPLETE','Old task incomplete')
        baseline[tid]={'kind':t['kind'],'completion_sha256':c.digest(f)}
    registered=register_development(protocol,root,c.read(prepared['path']),first_four=False)
    after={t['task_id']:t for t in registry.tasks()}
    new={tid:t for tid,t in after.items() if tid not in before}
    specs=protocol['origins']['development'][4:]+protocol['origins']['tuning']
    c.require(len(new)==300 and len(specs)==12,'Unexpected expansion size')
    for tid in new:
        c.require(not any((root/area/(tid+'.json')).exists() for area in ['intents','submissions','completed']),
                  'New task already attempted; inspect')
    lanes=assign(new,specs)
    cfg={**old,'phase':'development-tuning','control':str(control),
         'first_four_delivery':{'path':str(receipt),'sha256':c.digest(receipt)},
         'allowed_lineages':[s['seed'] for s in specs],
         'targets':sorted(new),'task_hashes':{tid:c.digest(root/'tasks'/(tid+'.json')) for tid in after},
         'baseline':baseline,'inflight_at_setup':[], 'lanes':lanes,
         'max_gpu_jobs':7,'available_gpus':5,'setup_at':c.now()}
    cfg['small_file_hashes']={path:c.digest(path) for path in old.get('small_file_hashes',{})}
    for path in [control/'server_controller.py',control/'launch_pipeline.sbatch',control/'prepare_expanded_pipelines.py',Path(prepared['path'])]:
        cfg['small_file_hashes'][str(path)]=c.digest(path)
    for area in ['verified','ticks','allocations']:(control/area).mkdir(exist_ok=True)
    c.inspect_scope(cfg,registry)
    for lane,ids in lanes.items():c.atomic(control/f'lane_{lane}_tasks.json',[after[tid] for tid in ids])
    c.atomic(control/'REGISTRATION.json',registered)
    c.atomic(control/'config.json',cfg)
    print(json.dumps({'status':'FIXED_LISTS_PREPARED_NOT_SUBMITTED','new_tasks':len(new),
                      'counts':dict(Counter(t['kind'] for t in new.values())),
                      'lane_counts':{k:len(v) for k,v in lanes.items()}}))


if __name__=='__main__':main()
