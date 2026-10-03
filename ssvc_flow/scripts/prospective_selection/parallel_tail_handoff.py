"""Explicit user-authorized R6 tail handoff; preserve the loaded fixed lists.

STOP prevents old processes from claiming their next entry. Their current
workers and audits finish normally. Only the reserved R6 may run separately.
Never remove STOP automatically or retry an unresolved submission.
"""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import time


def validate_reservation(c, cfg, registry):
    root, control = registry.root, Path(cfg['control'])
    c.require(not (control/'STOP').exists(), 'existing STOP requires review')
    c.require(not (control/'BLOCKED.json').exists(), 'existing BLOCKED requires review')
    tasks = c.inspect_scope(cfg, registry)
    reviewed = c.reviewed_records(cfg, tasks)
    remaining = set(cfg['targets']) - set(reviewed)
    keys = {tuple(tasks[t]['key'][:2]) for t in remaining}
    c.require(keys <= {('61004_t96', r) for r in ('R1','R6','GDPO_R4')},
              'scope differs from authorized final three tasks')
    candidates = [t for t in remaining if tasks[t]['key'][:2] == ['61004_t96','R6']]
    c.require(len(candidates) == 1, 'R6 absent or already verified')
    tid = candidates[0]
    c.require(tid in cfg['lanes']['2'], 'R6 original lane mismatch')
    c.require(all(t in reviewed for t in cfg['lanes']['3']), 'spare lane not complete')
    for area in ('intents','submissions','completed'):
        c.require(not (root/area/(tid+'.json')).exists(), 'R6 already attempted; do not duplicate')
    allocations = c.lane_allocations(cfg)
    live = {x['job_id'] for x in c.active_jobs()}
    c.require(len(live) < 5, 'no project capacity')
    c.require(not any(x['lane']==3 and x['job_id'] in live for x in allocations), 'lane 3 still live')
    known = {x['job_id'] for x in allocations}
    c.require(live <= known, 'unknown active allocation')
    states = c.accounting(known-live)
    c.require(all(states.get(j, [None,None,None])[1:3] == ['COMPLETED','0:0']
                  for j in known-live), 'unknown or failed old allocation')
    registry.assert_can_execute(tid)
    return tid


def submit(c, cfg, registry, config_path):
    from src.prospective_selection.jobs import _publish
    control = Path(cfg['control'])
    with (control/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return submit_locked(c, cfg, registry, config_path)


def submit_locked(c, cfg, registry, config_path):
    from src.prospective_selection.jobs import _publish
    control = Path(cfg['control'])
    with registry._lock():
        tid = validate_reservation(c,cfg,registry)
        reservation = {'status':'USER_AUTHORIZED_TAIL_HANDOFF','task_id':tid,
                       'at':c.now(),'source_lane':2,'execution_lane':3,
                       'script_sha256':c.digest(__file__),
                       'controller_sha256':c.digest(control/'server_controller.py')}
        _publish(control/'TAIL_HANDOFF.json',reservation)
        c.atomic(control/'STOP',reservation)
        _publish(registry.root/'intents'/(tid+'.json'),reservation)
        # launch_lane records the allocation intent before sbatch. Any unknown
        # outcome remains reserved and STOP remains in place for manual review.
        token = 'tail_r6_20261003'
        _publish(control/'allocations'/(token+'.intent.json'),reservation)
        job = c.run(['sbatch','--parsable','--qos='+c.QOS,'--no-requeue',
                     '--job-name=ps2-tail-r6','--output='+str(control/'%j.out'),
                     '--error='+str(control/'%j.err'),str(control/'launch_pipeline.sbatch'),
                     cfg['python'],str(Path(__file__).resolve()),str(config_path),'3']).split(';')[0]
        c.require(job.isdigit(),'unknown submission; do not retry')
        receipt = {'job_id':job,'lane':3,'task_id':tid,'at':c.now(),'status':'SUBMITTED'}
        _publish(control/'allocations'/(token+'.submitted.json'),receipt)
        _publish(registry.root/'submissions'/(tid+'.json'),receipt)
        c.atomic(control/'TAIL_HANDOFF_SUBMITTED.json',receipt)
        print(receipt,flush=True)


def execute(c, cfg, registry):
    control=Path(cfg['control']);job=os.environ['SLURM_JOB_ID']
    a=c.read(control/'TAIL_HANDOFF.json');tid=a['task_id']
    for _ in range(60):
        if (control/'TAIL_HANDOFF_SUBMITTED.json').exists():break
        time.sleep(1)
    with registry._lock():
        c.require(c.read(control/'STOP')==a,'handoff STOP changed')
        c.require(not (control/'BLOCKED.json').exists(),'blocked')
        c.require(c.digest(__file__)==a['script_sha256'],'handoff script changed')
        c.require(c.digest(control/'server_controller.py')==a['controller_sha256'],'controller changed')
        c.require(c.read(control/'TAIL_HANDOFF_SUBMITTED.json')['job_id']==job,'wrong allocation')
        c.require(c.read(registry.root/'submissions'/(tid+'.json'))['job_id']==job,'wrong task submission')
        c.inspect_scope(cfg,registry)
        task=registry.assert_can_execute(tid)
        c.require(task['key'][:2]==['61004_t96','R6'],'wrong task')
        live={x['job_id'] for x in c.active_jobs()}
        c.require(job in live and len(live)<=5,'capacity or job mismatch')
    c.atomic(control/'TAIL_HANDOFF_RUNNING.json',{'job_id':job,'task_id':tid,'at':c.now()})
    began=time.monotonic()
    result=c.subprocess.run([cfg['python'],'-m','src.prospective_selection.cli','worker',
        '--config',str(registry.root/'protocol.json'),'--root',str(registry.root),
        '--task',str(registry.root/'tasks'/(tid+'.json')),'--runtime',cfg['runtime'],'--allow-gpu'],
        cwd=cfg['project_root'],timeout=24*3600)
    c.require(result.returncode==0,'R6 worker failed; preserve attempt')
    value=c.audit_completed(cfg,task,{'job_id':job,'worker_exit_code':0,
        'worker_elapsed_seconds':time.monotonic()-began,'execution_evidence':'Explicit tail handoff subprocess exited 0'})
    c.atomic(control/'TAIL_HANDOFF_COMPLETE.json',value)
    tasks=c.inspect_scope(cfg,registry);reviewed=c.reviewed_records(cfg,tasks)
    c.atomic(control/'STATUS.json',{'status':'FIRST_FOUR_COLLECTED_AWAITING_ANALYSIS'
        if set(cfg['targets'])<=set(reviewed) else 'TAIL_HANDOFF_AWAITING_OTHER_ACTIVE_TASKS',
        'observed_at':c.now(),'verified_branches':sum(a.get('kind')=='branch' for a in reviewed.values()),
        'verified_historical_E':sum(a.get('kind')=='historical_e' for a in reviewed.values()),
        'active_gpu_jobs':c.active_jobs(),'launched_lanes':[]})


def report_failure(c, control, exc, submitting):
    # A pre-reservation refusal (e.g. the old lane just started R6) has
    # changed no scheduling state and must not stop that healthy lane.
    if submitting and not (control/'TAIL_HANDOFF.json').exists():
        print({'status':'HANDOFF_NOT_PERFORMED','reason':str(exc)},flush=True)
        return
    c.publish_block(control,exc)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--submit',action='store_true');p.add_argument('--lane',type=int,choices=[3])
    args=p.parse_args()
    import json
    cfg=json.loads(args.config.read_text());control=Path(cfg['control'])
    spec=importlib.util.spec_from_file_location('fixed_controller',control/'server_controller.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    sys.path.insert(0,cfg['project_root'])
    from src.prospective_selection.jobs import TaskRegistry
    registry=TaskRegistry(Path(cfg['root']),c.read(Path(cfg['root'])/'protocol.json'))
    try:
        if args.submit:submit(c,cfg,registry,args.config.resolve())
        else:
            c.require(args.lane==3 and os.environ.get('SLURM_JOB_ID'),'GPU allocation required')
            execute(c,cfg,registry)
    except Exception as exc:
        report_failure(c,control,exc,args.submit)
        raise


if __name__=='__main__':main()
