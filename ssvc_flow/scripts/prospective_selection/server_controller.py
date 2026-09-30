"""Five fixed sequential GPU lists, with CPU allocation handoff only.

The immutable worker snapshot is imported, never modified. New terminal outputs
are audited and descriptively compared before the next fixed experiment can execute in that same GPU job.
No failed/unknown submission is retried; no scientific delivery gate is written.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

QOS = "soujanya-poria-startfund-2026-03"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bound(binding, parent):
    p = Path(binding['path']).resolve()
    require(p.is_relative_to(parent.resolve()), 'raw binding outside task directory')
    data = p.read_bytes()
    require(hashlib.sha256(data).hexdigest() == binding['sha256'], 'raw binding hash mismatch')
    return data


def compare(a, b):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), 'summary fields differ')
        return max([compare(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), 'summary lengths differ')
        return max([compare(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, float):
        require(isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b)
                and abs(a-b) < 1e-12, 'summary numeric mismatch')
        return abs(a-b)
    require(a == b, 'summary value mismatch')
    return 0.0


def verify_evaluation(path, expected, prompts, draws, state_hash):
    from src.modeling_v3.io import canonical_hash
    from src.prospective_selection.evaluation import summarize_rows
    m, c = read(path/'MANIFEST.json'), read(path/'COMPLETE.json')
    require(c['status'] == 'EVALUATION_COMPLETE' and c['identity'] == m, 'evaluation incomplete')
    require(m['checkpoint_state_hash'] == state_hash and m['draws'] == draws, 'evaluation binding')
    for k, v in expected.items():
        require(m[k] == v, 'evaluation identity: ' + k)
    chunks = [read(f) for f in path.glob('prompts/*/COMMIT.json')]
    require(len(chunks) == len(c['chunks']) == prompts, 'evaluation prompt count')
    require({canonical_hash(x) for x in chunks} == {canonical_hash(x) for x in c['chunks']},
            'evaluation chunks differ')
    rows = []
    for chunk in chunks:
        require(chunk['identity_hash'] == canonical_hash(m) and chunk['count'] == draws,
                'chunk identity/count')
        rr = [json.loads(line) for line in bound(chunk['samples'], path).splitlines()]
        require(len(rr) == draws, 'draw count')
        for row in rr:
            for k in ('lineage_id','origin_id','policy_id','panel_id','horizon','repeat','role'):
                require(row[k] == m[k], 'row identity: ' + k)
        rows.extend(rr)
    require(len(rows) == prompts*draws and len({x['sample_id'] for x in rows}) == len(rows),
            'duplicate or missing evaluation samples')
    delta = compare(summarize_rows(rows), c['summary'])
    return {'outputs': len(rows), 'J': c['summary']['J'], 'max_summary_difference': delta}


def verify_task(root, task):
    from src.modeling_v3.io import canonical_hash
    p = task['payload']
    origin = p['origin_id']
    if task['kind'] == 'historical_e':
        result = verify_evaluation(root/'historical_E'/p['policy_id'], {
            'lineage_id':p['lineage_id'], 'origin_id':origin, 'policy_id':p['policy_id'],
            'panel_id':'E_old','role':'historical_E','horizon':32,'repeat':0},
            72, 32, p['checkpoint']['state_hash'])
        return {'kind':'historical_e', 'origin_id':origin, 'recipe':p['policy_id'], **result}
    require(task['kind'] == 'branch' and p['role'] == 'development'
            and int(p['lineage_id']) in (61001,61002,61003,61004) and p['repeat'] == 1,
            'task outside first-four branch scope')
    recipe = p['recipe_id']
    path = root/'branches'/origin/recipe/'repeat_1'
    m, c = read(path/'MANIFEST.json'), read(path/'COMPLETE.json')
    mh = canonical_hash(m)
    require(m['kind'] == 'PROSPECTIVE_TRAINING' and m['fixture'] is False
            and m['runtime_identity']['execution_kind'] == 'REAL_CUDA_MODEL', 'non-real training')
    require(c['status'] == 'TRAINING_COMPLETE' and c['identity'] == m['identity']
            and c['steps'] == 32 and c['training_outputs'] == 1024, 'incomplete training')
    for k in ('origin_id','lineage_id','recipe_id','repeat','schedule_id','role'):
        require(m['identity'][k] == p[k], 'training identity: ' + k)
    require(c['checkpoints']['0']['state_hash'] == m['initial_state_hash'], 'initial state binding')
    for f in path.parents[1].glob('*/repeat_1/MANIFEST.json'):
        require(read(f)['initial_state_hash'] == m['initial_state_hash'], 'different origin initial state')
    require(len(c['authoritative_segments']) == 4, 'segment count')
    rows, updates = [], []
    for i, f in enumerate(c['authoritative_segments']):
        require(Path(f).resolve().is_relative_to(path.resolve()), 'segment outside task')
        s = read(f)
        require(s['manifest_hash'] == mh and (s['start'],s['stop']) == (8*i+1,8*i+8), 'segment identity')
        cp = s['checkpoint']
        require(cp['identity'] == {'manifest_hash':mh,'step':s['stop']}
                and Path(cp['path']).resolve().is_relative_to(path.resolve())
                and Path(cp['path']).stat().st_size > 0, 'checkpoint metadata')
        if str(s['stop']) in c['checkpoints']:
            require(cp == c['checkpoints'][str(s['stop'])], 'checkpoint link mismatch')
        rr = [json.loads(line) for line in bound(s['samples'],path).splitlines()]
        require(len(rr) == 256, 'training sample count')
        for row in rr:
            for k,v in {'origin_id':origin,'recipe':recipe,'repeat':1,'role':'train',
                        'schedule_id':p['schedule_id']}.items():
                require(row[k] == v, 'training sample identity: ' + k)
        rows.extend(rr)
        for u in s['updates']:
            update = json.loads(bound(u,path))
            require(math.isfinite(update['loss']) and math.isfinite(update['grad_norm_preclip']),
                    'non-finite update')
            updates.append(update['step'])
    require(len(rows) == len({x['sample_key'] for x in rows}) == 1024
            and sorted(updates) == list(range(1,33)), 'training coverage')
    evals = {}
    for h,n,d,panel,role in [(8,24,8,'D_H8','diagnostic'),(32,144,16,'D','evaluation')]:
        evals[str(h)] = verify_evaluation(path/'evaluations'/f'H{h:02d}', {
            'origin_id':origin,'lineage_id':p['lineage_id'], 'policy_id':f'{origin}:{recipe}:1:H{h}',
            'horizon':h,'panel_id':panel,'role':role,'repeat':1},n,d,c['checkpoints'][str(h)]['state_hash'])
    return {'kind':'branch','origin_id':origin,'recipe':recipe,'J':evals['32']['J'],
            'training_outputs':1024,'finite_updates':32,'evaluations':evals}


def run(command):
    return subprocess.check_output(command, text=True, timeout=60).strip()


def active_jobs():
    rows = []
    for line in run(['squeue','--noheader','--me','--format=%i|%j|%b']).splitlines():
        job,name,tres = line.split('|')
        if name.startswith('ps2-'):
            require(tres.strip() in ('gres/gpu:pro6000:1','gpu:pro6000:1'), 'non-PRO6000 project job')
            rows.append({'job_id':job.strip(),'gpus':1})
    return rows


def accounting(ids):
    if not ids:
        return {}
    result = {}
    for line in run(['sacct','-X','-n','-j',','.join(sorted(ids)),
                     '--format=JobID,State,ExitCode,Elapsed,Start,End','-P']).splitlines():
        fields = line.split('|')
        if fields[0] in ids:
            require(fields[0] not in result, 'ambiguous accounting')
            result[fields[0]] = fields
    return result


def latest_submission(root, tid):
    recovery = root/'recoveries'/tid/'attempt_1'
    path = recovery/'submission.json' if (recovery/'intent.json').exists() else root/'submissions'/f'{tid}.json'
    require(path.exists(), 'unknown submission; do not resubmit: ' + tid)
    return str(read(path)['job_id'])


def inspect_scope(cfg, registry):
    root = registry.root
    for path,expected in cfg.get('small_file_hashes',{}).items():
        require(digest(path)==expected,'runtime/launcher version changed: '+path)
    require(digest(root/'protocol.json') == cfg['protocol_sha256'], 'protocol changed')
    assigned=[t for lane in cfg['lanes'].values() for t in lane]
    require(set(cfg['lanes'])=={'1','2','3','4','5'} and len(assigned)==len(set(assigned)), 'duplicate/missing lane assignment')
    require(set(assigned)==set(cfg['targets'])-set(cfg['baseline'])-set(cfg['inflight_at_setup']), 'fixed lane coverage mismatch')
    tasks = {t['task_id']:t for t in registry.tasks()}
    require(set(tasks) == set(cfg['task_hashes']), 'registered task scope changed')
    for tid in tasks:
        require(digest(root/'tasks'/f'{tid}.json') == cfg['task_hashes'][tid], 'registered task changed')
        registry.task(tid)
    for tid,b in cfg['baseline'].items():
        require(digest(root/'completed'/f'{tid}.json') == b['completion_sha256'], 'reviewed completion changed')
    return tasks


def reviewed_records(cfg, tasks):
    root, control = Path(cfg['root']), Path(cfg['control'])
    reviewed = dict(cfg['baseline'])
    for f in (control/'verified').glob('*.json'):
        a=read(f);tid=a['task_id']
        require(tid in tasks and a['task_sha256']==cfg['task_hashes'][tid], 'audit task mismatch')
        require(digest(root/'completed'/f'{tid}.json')==a['completion_sha256'], 'audited completion changed')
        reviewed[tid]=a
    return reviewed


def audit_completed(cfg, task, execution):
    root, control = Path(cfg['root']), Path(cfg['control'])
    tid=task['task_id'];completion=root/'completed'/f'{tid}.json'
    require(completion.exists() and read(completion)['receipt']['status']=='COMPLETE', 'missing completion')
    value=verify_task(root,task)
    reviewed=reviewed_records(cfg,{t:None for t in cfg['task_hashes']})
    scores={a['recipe']:a['J'] for a in reviewed.values()
            if a.get('kind')==value['kind'] and a.get('origin_id')==value['origin_id']}
    if scores:
        value.update(previous_best_J=max(scores.values()),delta_previous_best=value['J']-max(scores.values()))
    value.update(status='VERIFIED',task_id=tid,observed_at=now(),
                 task_sha256=cfg['task_hashes'][tid],completion_sha256=digest(completion),**execution,
                 scientific_interpretation='Descriptive development endpoint only; retain all arms and seeds.')
    atomic(control/'verified'/f'{tid}.json',value)
    return value


def publish_block(control, exc):
    value={'status':'BLOCKED','observed_at':now(),'job_id':os.environ.get('SLURM_JOB_ID'),
           'error_type':type(exc).__name__,'error':str(exc)}
    atomic(control/'BLOCKED.json',value)
    print(json.dumps(value),flush=True)


def lane_allocations(cfg):
    result=[]
    for p in sorted((Path(cfg['control'])/'allocations').glob('*.intent.json')):
        receipt=p.with_name(p.name.replace('.intent.json','.submitted.json'))
        require(receipt.exists(),'pipeline submission unknown; do not retry: '+p.name)
        result.append(read(receipt))
    return result


def claim_task(cfg, registry, job_id, lane, tid):
    """Execute the next fixed entry; no shared task selection or work stealing."""
    from src.prospective_selection.jobs import _publish
    root,control=registry.root,Path(cfg['control'])
    with registry._lock():
        if (control/'STOP').exists() or (control/'BLOCKED.json').exists():return None
        tasks=inspect_scope(cfg,registry);reviewed=reviewed_records(cfg,tasks)
        require(tid in cfg['lanes'][str(lane)], 'task not assigned to this fixed lane')
        if tid in reviewed:return 'DONE'
        require(tid in cfg['targets'], 'task outside first-four scope')
        require(any(x['job_id']==job_id and x['lane']==lane for x in lane_allocations(cfg)),
                'unregistered GPU lane')
        running_ids={x['job_id'] for x in active_jobs()}
        require(job_id in running_ids and len(running_ids)<=5,'GPU lane not live or capacity exceeded')
        require(not (root/'intents'/f'{tid}.json').exists(), 'task already attempted; no automatic retry')
        task=registry.assert_can_execute(tid)
        _publish(root/'intents'/f'{tid}.json',{'task_id':tid,'created_at':now(),
                 'status':'PIPELINE_FIXED_ASSIGNMENT','gpus':1,'pipeline_lane':lane,'job_id':job_id})
        _publish(root/'submissions'/f'{tid}.json',{'task_id':tid,'job_id':job_id,
                 'status':'PIPELINE_ASSIGNED','pipeline_lane':lane})
        return task


def pipeline(cfg, registry, lane):
    control=Path(cfg['control']);job=os.environ['SLURM_JOB_ID'];start=time.monotonic()
    # sbatch can start before its submitting process has saved the job-id receipt.
    for _ in range(60):
        receipts=[read(f) for f in (control/'allocations').glob('*.submitted.json')]
        if any(x['job_id']==job and x['lane']==lane for x in receipts):break
        time.sleep(1)
    else:raise ValueError('GPU allocation registration missing')
    for tid in cfg['lanes'][str(lane)]:
        # A fresh worker has the original 24 h cap; leave 2 h allocation margin.
        if time.monotonic()-start >= 46*3600:break
        task=claim_task(cfg,registry,job,lane,tid)
        if task=='DONE':continue
        if task is None:break
        tid=task['task_id']
        atomic(control/f'lane_{lane}.json',{'status':'EXECUTING','job_id':job,'task_id':tid,'at':now()})
        print(json.dumps({'event':'START_TASK','job_id':job,'task_id':tid,'key':task['key'],'at':now()}),flush=True)
        command=[cfg['python'],'-m','src.prospective_selection.cli','worker',
                 '--config',str(registry.root/'protocol.json'),'--root',str(registry.root),
                 '--task',str(registry.root/'tasks'/f'{tid}.json'),'--runtime',cfg['runtime'],'--allow-gpu']
        began=time.monotonic()
        completed=subprocess.run(command,cwd=cfg['project_root'],timeout=24*3600)
        require(completed.returncode==0,'worker failed; preserve attempt and stop: '+tid)
        value=audit_completed(cfg,task,{'job_id':job,'worker_exit_code':0,
                    'execution_evidence':'GPU subprocess returned 0; allocation may still be RUNNING',
                    'worker_elapsed_seconds':time.monotonic()-began,'pipeline_lane':lane})
        print(json.dumps({'event':'VERIFIED_TASK','task_id':tid,'J':value['J'],'at':now()}),flush=True)
    atomic(control/f'lane_{lane}.json',{'status':'DRAINED','job_id':job,'at':now()})


def launch_lane(cfg, config_path, lane):
    control=Path(cfg['control']);token=uuid.uuid4().hex
    intent=control/'allocations'/f'{token}.intent.json'
    atomic(intent,{'status':'SUBMISSION_INTENT_UNRESOLVED','lane':lane,'at':now()})
    script=Path(__file__).with_name('launch_pipeline.sbatch')
    job=run(['sbatch','--parsable','--qos='+QOS,'--no-requeue','--job-name=ps2-pipe-'+str(lane),
             '--output='+str(control/'%j.out'),'--error='+str(control/'%j.err'),str(script),
             cfg['python'],str(Path(__file__).resolve()),str(config_path),str(lane)]).split(';')[0]
    require(job.isdigit(),'pipeline submission unknown; no retry')
    result={'job_id':job,'lane':lane,'at':now(),'status':'SUBMITTED'}
    atomic(control/'allocations'/f'{token}.submitted.json',result)
    return result


def handoff_tick(cfg,registry,config_path,submit=True):
    """Only replaces finished allocations; tasks run back-to-back inside each lane."""
    root,control=registry.root,Path(cfg['control'])
    tasks=inspect_scope(cfg,registry);reviewed=reviewed_records(cfg,tasks)
    active=active_jobs();active_ids={x['job_id'] for x in active}
    allocations=lane_allocations(cfg);lane_ids={x['job_id'] for x in allocations}
    submissions={f.stem:latest_submission(root,f.stem) for f in (root/'intents').glob('*.json')}
    require(active_ids <= set(submissions.values()) | lane_ids,'unrecognized project GPU allocation')
    require(len(active_ids)<=5,'project GPU capacity exceeded')
    pending={tid:j for tid,j in submissions.items() if tid not in reviewed and j not in active_ids}
    states=accounting(set(pending.values()) | (lane_ids-active_ids))
    for tid,job in pending.items():
        state=states.get(job)
        require(state is not None and state[1:3]==['COMPLETED','0:0'],'failed/unknown task: '+job)
        require(job not in lane_ids,'GPU lane ended without task audit; explicit recovery required')
        audit_completed(cfg,tasks[tid],{'job_id':job,'sacct':state})
    for job in lane_ids-active_ids:
        require(states.get(job,[None,None,None])[1:3]==['COMPLETED','0:0'],'failed/unknown GPU lane: '+job)
    reviewed=reviewed_records(cfg,tasks)
    done=set(cfg['targets']) <= set(reviewed)
    result={'status':'FIRST_FOUR_COLLECTED_AWAITING_ANALYSIS' if done else 'RUNNING',
            'observed_at':now(),'controller_job_id':os.environ.get('SLURM_JOB_ID'),
            'verified_branches':sum(x.get('kind')=='branch' for x in reviewed.values()),
            'verified_historical_E':sum(x.get('kind')=='historical_e' for x in reviewed.values()),
            'active_gpu_jobs':active,'launched_lanes':[]}
    return result


def launch_only(cfg, registry, config_path):
    """Lightweight login-node submission only; raw audits run inside GPU jobs.

    Used just for migration and 72 h allocation rollover, never between the
    experiments in a lane. Pending and ambiguous allocations consume slots.
    """
    root,control=registry.root,Path(cfg['control'])
    tasks=inspect_scope(cfg,registry);reviewed=reviewed_records(cfg,tasks)
    active=active_jobs();active_ids={x['job_id'] for x in active}
    allocations=lane_allocations(cfg);lane_ids={x['job_id'] for x in allocations}
    submissions={f.stem:latest_submission(root,f.stem) for f in (root/'intents').glob('*.json')}
    require(active_ids <= set(submissions.values())|lane_ids,'unrecognized GPU allocation')
    require(len(active_ids)<=5,'GPU capacity exceeded')
    terminal_ids=(lane_ids|{j for t,j in submissions.items() if t not in reviewed})-active_ids
    states=accounting(terminal_ids)
    for job in terminal_ids:
        state=states.get(job)
        if state is None or state[1] in ('PENDING','RUNNING','COMPLETING','CONFIGURING'):
            return {'status':'WAIT_ACCOUNTING_NO_SUBMISSION','observed_at':now(),'job_id':job}
        require(state[1:3]==['COMPLETED','0:0'],'failed allocation; manual review: '+job)
        if job in lane_ids:
            require(all(t in reviewed for t,j in submissions.items() if j==job),
                    'lane ended with unaudited work; no retry')
    done=set(cfg['targets'])<=set(reviewed)
    result={'status':'FIRST_FOUR_COLLECTED_AWAITING_ANALYSIS' if done else 'RUNNING',
            'observed_at':now(),'verified_branches':sum(x.get('kind')=='branch' for x in reviewed.values()),
            'verified_historical_E':sum(x.get('kind')=='historical_e' for x in reviewed.values()),
            'active_gpu_jobs':active,'launched_lanes':[]}
    if done:return result
    busy={x['lane'] for x in allocations if x['job_id'] in active_ids}
    with registry._lock():
        for lane in range(1,6):
            if lane in busy or all(t in reviewed for t in cfg['lanes'][str(lane)]):continue
            live={x['job_id'] for x in active_jobs()}
            if len(live|{x['job_id'] for x in result['launched_lanes']})>=5:break
            result['launched_lanes'].append(launch_lane(cfg,config_path,lane))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--lane',type=int,choices=range(1,6))
    parser.add_argument('--launch-only',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args(argv)
    cfg=read(args.config);control=Path(cfg['control']);root=Path(cfg['root'])
    sys.path.insert(0,cfg['project_root'])
    from src.prospective_selection.jobs import TaskRegistry
    registry=TaskRegistry(root,read(root/'protocol.json'))
    for area in ('verified','ticks','allocations'):(control/area).mkdir(parents=True,exist_ok=True)
    require(args.lane or args.launch_only or args.audit_only, 'choose a lane, submission or audit mode')
    require(not args.lane or os.environ.get('SLURM_JOB_ID'),'GPU lane requires Slurm')
    lockname=f'lane_{args.lane}.lock' if args.lane else 'controller.lock'
    with (control/lockname).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('ALREADY_RUNNING');return
        try:
            if (control/'STOP').exists() or (control/'BLOCKED.json').exists():return
            if args.lane:
                # Initial handoff audits old jobs inside a Slurm allocation.
                with (control/'controller.lock').open('a') as audit_lock:
                    fcntl.flock(audit_lock,fcntl.LOCK_EX)
                    handoff_tick(cfg,registry,args.config.resolve(),submit=False)
                pipeline(cfg,registry,args.lane);return
            result=launch_only(cfg,registry,args.config.resolve()) if args.launch_only else handoff_tick(cfg,registry,args.config.resolve(),submit=False)
            atomic(control/'ticks'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex+'.json'),result)
            atomic(control/'STATUS.json',result);print(json.dumps(result),flush=True)
        except Exception as exc:
            publish_block(control,exc);raise


if __name__=='__main__':main()
