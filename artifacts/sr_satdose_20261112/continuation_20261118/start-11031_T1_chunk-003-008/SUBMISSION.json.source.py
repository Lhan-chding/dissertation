#!/usr/bin/env python3
"""Submit exactly one authenticated fixed TRAIN continuation chunk. Default is local-only plan.

--execute performs one held sbatch, persists its accepted identity, releases the
existing worker lease, then releases ONLY that newly returned verified job once.
Any occupied attempt or uncertain network/scheduler response forbids resubmission.
No cancellation, cleanup, other-job mutation, implicit successor or MAIN is implemented.
The 24-hour walltime is a scheduler ceiling, not a scientific stopping rule.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

WORKSPACE=Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ROOT='/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1'
CODE=ROOT+'/train_code_71018e5'
PYTHON='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
PREPARER_SHA='737954eb256b1c29f5a381e864d3a1bc74a5fa1a8e95ab4a3373c2e66ad151c2'
PINS={
 'FIRST_TWO_CODE_DEPLOYMENT_71018e5_20261116.json':'de42793b8d839e7d72d8eb399d37c0fb9e842d02f838709cde5b1a0e3ebbbeb8',
 'ORIGINAL_TRAIN_EXECUTION_SOURCES_71018e5.json':'5370b089381609c2e8d5d5ec7555c3dd441babaed242bd66b7cbfad02f679dd7',
}
ORIGINAL=dict(path='execution_releases/FIRST_TWO_RELEASE_71018e5.json',
 sha256='22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f')

REMOTE=r'''
import sys,os,json,hashlib,fcntl,subprocess,re,shlex
from pathlib import Path
from datetime import datetime,timezone
p=json.load(sys.stdin)
ROOT=Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
CODE=ROOT/'train_code_71018e5';QOS='soujanya-poria-startfund-2026-03'
locks=[];directory=None;observations=[];job=None;accepted=None

def require(ok,message):
    if not ok:raise RuntimeError(message)
def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Noncanonical file '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def ref(path):return dict(path=str(path),sha256=sha(path))
def inventory():
    require(CODE.resolve(strict=True)==CODE and CODE.is_dir(),'Original code root changed')
    values={}
    for path in CODE.rglob('*'):
        require(path.resolve(strict=True)==path and not path.is_symlink(),'Noncanonical code entry')
        if path.is_file():values[str(path.relative_to(CODE))]=sha(path)
        else:require(path.is_dir(),'Nonordinary code entry')
    require(values==p['sources'] and len(values)==84,'Original source84 changed')
def syncdir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def write(name,value):
    data=json.dumps(value,sort_keys=True,allow_nan=False).encode()+b'\n'
    with (directory/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    syncdir(directory)
    return ref(directory/name)
def acquire(name,mode):
    path=ROOT/name;sha(path)
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:fcntl.flock(fd,mode|fcntl.LOCK_NB)
    except BaseException:os.close(fd);raise
    locks.append((fd,path))
def stable_locks():
    for fd,path in locks:
        actual=path.stat();opened=os.fstat(fd)
        require(not path.is_symlink() and (actual.st_dev,actual.st_ino)==(opened.st_dev,opened.st_ino),'Lease replaced')
def unlock():
    while locks:
        fd,path=locks.pop();os.close(fd)
def run(argv):
    environment={k:v for k,v in os.environ.items() if k!='SLURM_CLUSTERS'}
    if argv[0]=='sbatch':
        environment={k:v for k,v in environment.items() if not k.startswith('SBATCH_') and k!='CUDA_VISIBLE_DEVICES'}
    completed=subprocess.run(argv,capture_output=True,text=True,check=False,env=environment)
    value=dict(argv=argv,returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr,observed_at_utc=utc())
    observations.append(value)
    return value

def scheduler_before(name):
    prior=run(['sacct','-j','1740834','--noheader','--parsable2',
      '--format=JobIDRaw,State%40,ExitCode,JobName%120,Account%40,QOS%80,End'])
    require(prior['returncode']==0,'Cannot observe first-two scheduler')
    records=[line.strip().split('|') for line in prior['stdout'].splitlines() if line.strip()]
    require(all(len(row)==7 for row in records),'Malformed prerequisite scheduler record')
    parents=[row for row in records if row[0]=='1740834']
    require(len(parents)==1 and parents[0][1:3]==['COMPLETED','0:0'] and
      parents[0][3:6]==['LH-SATDOSE-FIRST-TWO','rose',QOS] and parents[0][6] not in ('','Unknown','None'),
      'First-two 1740834 not confirmed COMPLETED 0:0 with original identity')
    require(all(row[0]=='1740834' or row[0].startswith('1740834.') for row in records) and
      all(row[1:3]==['COMPLETED','0:0'] for row in records),'Failed/unknown prerequisite child')
    queue=run(['squeue','--user=varun024','--noheader','--format=%i|%j|%T'])
    history=run(['sacct','--user=varun024','--starttime=2026-11-12T00:00:00','--name='+name,
      '--noheader','--parsable2','--format=JobIDRaw,JobName%120,State%40'])
    for value in (queue,history):
        require(value['returncode']==0,'Cannot exclude duplicate scheduler job')
        rows=[line.strip().split('|') for line in value['stdout'].splitlines() if line.strip()]
        require(all(len(row)==3 for row in rows),'Malformed duplicate-job query')
        require(all(row[1].strip()!=name for row in rows),'Existing same-name job: never resubmit')
    return dict(prerequisite=prior,queue=queue,history=history)

def verify_own_job(job,name,held):
    observation=run(['scontrol','show','job','--oneliner',str(job)])
    require(observation['returncode']==0,'Cannot verify newly submitted job identity')
    fields={}
    for token in shlex.split(observation['stdout']):
        if '=' in token:
            key,value=token.split('=',1)
            require(key not in fields,'Duplicate scontrol identity field')
            fields[key]=value
    expected=dict(JobId=str(job),JobName=name,Account='rose',QOS=QOS,Partition='cluster02',
       NumCPUs='8',NumTasks='1',Requeue='0',Command=str(CODE/'scheduling/run_continuation.sh'),
       WorkDir=str(CODE),StdOut=str(directory/f'slurm-{job}.out'),StdErr=str(directory/f'slurm-{job}.err'))
    require(all(fields.get(k)==v for k,v in expected.items()) and fields.get('NumNodes') in ('1','1-1') and
       fields.get('TimeLimit') in ('1-00:00:00','24:00:00') and
       fields.get('UserId','').startswith('varun024('),'New job identity/resources differ; do not release')
    tres=dict(item.split('=',1) for item in fields.get('ReqTRES','').split(',') if '=' in item)
    require(all(tres.get(k)==v for k,v in {'cpu':'8','mem':'66G','node':'1','gres/gpu':'2','gres/gpu:pro6000':'2'}.items()),
       'New job requested resources differ')
    if held:
        require(fields.get('JobState')=='PENDING' and fields.get('Reason')=='JobHeldUser' and fields.get('Priority')=='0',
                'New job is not confirmed user-held')
    else:
        require(fields.get('JobState') in ('PENDING','RUNNING','COMPLETING','COMPLETED') and
           fields.get('Reason') not in ('JobHeldUser','JobHeldAdmin') and
           (fields.get('JobState')!='PENDING' or int(fields.get('Priority','0'))>0),
           'Job release not confirmed or job failed; never resubmit')
        require(fields.get('JobState')!='COMPLETED' or fields.get('ExitCode')=='0:0',
                'Completed accepted job exit code failed/unknown')
    return dict(observation=observation,fields=fields)


try:
    require(ROOT.resolve(strict=True)==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Canonical CPU-only bootstrap required')
    inventory()
    acquire('FIRST_TWO_LAUNCHER.lock',fcntl.LOCK_SH)
    acquire('DIAGNOSTIC_WORKER.lock',fcntl.LOCK_EX)
    sys.path.insert(0,str(CODE))
    from execution import continuation_control as c
    require(c.PACKAGE==CODE and c.ROOT==ROOT,'Shadowed frozen original namespace')
    preparation=c.read(p['preparation_ref'])
    require(preparation==p['preparation'],'Remote PREFLIGHT differs from local successful report')
    start,arm,first,last=p['start_seed'],p['arm'],p['first_step'],p['last_step']
    selection=dict(start_seed=start,arm=arm,first_step=first,last_step=last)
    require(c.CHUNKS.get(first)==last and preparation['selection']==selection,'Fixed chunk selection differs')
    target=c._slot(start,arm,first)
    release_ref=preparation['release'];release,original=c.verify_release(c.bound(release_ref),release_ref['sha256'])
    require(release['original_train_release']==p['original'] and release['sources']==p['sources'] and
      original['execution_sources']==p['execution_sources'] and len(original['execution_sources'])==19 and
      release['selection']==selection and release['canary_spec']==p['canary_spec'],
      'Original TRAIN source84/19-key identity, canary or chunk differs')
    require(release['storage_budget']==preparation['storage_budget'] and
      preparation['original_train_release']==p['original'] and
      preparation['preparation_script_sha256']==p['preparer_sha'],'Preparation lineage/budget/source differs')
    prep_prefix=f'CONTINUATION_71018e5_s{start}_{arm}_{first:03d}_{last:03d}'
    expected_prep=ROOT/'execution_releases'/(prep_prefix+'_PREFLIGHT.json')
    require(c.bound(p['preparation_ref'])==expected_prep and
      c.bound(release_ref)==expected_prep.with_name(prep_prefix+'_RELEASE.json') and
      c.bound(preparation['intent'])==expected_prep.with_name(prep_prefix+'_INTENT.json') and
      c.bound(preparation['risk'])==expected_prep.with_name(prep_prefix+'_RISK.json') and
      c.bound(preparation['storage_budget'])==expected_prep.with_name(prep_prefix+'_BUDGET.json'),
      'Preparation artifact relocated or mismatched')
    require(not os.path.lexists(expected_prep.with_name(prep_prefix+'_ERROR.json')),'Preparation has retained ERROR')
    intent_proof=c.read(preparation['intent'])
    require(intent_proof.get('status')=='CPU_PREPARATION_INTENT_UNKNOWN_UNTIL_PREFLIGHT' and
      intent_proof['preparation_script_sha256']==p['preparer_sha'] and intent_proof['selection']==selection and
      intent_proof['original_train_release']==p['original'] and intent_proof['canary_spec']==release['canary_spec'] and
      intent_proof['trusted_commits']==release['trusted_commits'] and
      intent_proof['dev_endpoints']==release['dev_endpoints'] and intent_proof['prior_chunks']==release['prior_chunks'],
      'Preparation intent/canary/checkpoint/DEV lineage differs')
    # Actual original CPU admission: all four previous DEV barriers, original
    # checkpoint states, per-path cursors, scientific risk and published canary.
    ctx=c.preflight(c.bound(release_ref),release_ref['sha256'],start,arm,first)
    require(ctx['risk']==c.read(preparation['risk']) and ctx['risk']['global_hold'] is False,
            'Actual risk differs from independently published preparation')
    require(preparation['cursor']=={key:len(refs) for key,refs in release['trusted_commits'].items()} and
      preparation['completed_dev_endpoints']==len(release['dev_endpoints']) and
      preparation['completed_chunks']==len(release['prior_chunks']),'Preparation population/cursor changed')
    require(preparation['required_bytes']==c.storage._budget(release['storage_budget'],c.storage._now())['required_bytes'],
            'Budget byte count differs from preparation')
    c._hold_free();require(not os.path.lexists(target),'Existing continuation attempt: never retry')
    schedule_root=ROOT.parent/'scheduling'
    require(schedule_root.resolve(strict=True)==schedule_root and schedule_root.is_dir(),'Canonical scheduling root missing')
    # Pending/unknown operators are not made safe by an absent GPU attempt slot.
    # Check both DEV and TRAIN submissions, preserving the one-worker contract.
    for family in ('trained_dev_71018e5','continuation_71018e5'):
        parent=schedule_root/family
        if not os.path.lexists(parent):continue
        require(parent.resolve(strict=True)==parent and parent.is_dir(),'Noncanonical submission root')
        for item in parent.rglob('*'):
            require(not item.is_symlink() and 'ERROR' not in item.name and 'HOLD' not in item.name,
                    'Symlink or failed submission evidence')
        for sd in parent.iterdir():
            require(sd.name in ('start-11031','start-11032') and sd.is_dir(),'Unknown submission start')
            for ad in sd.iterdir():
                require(ad.name in ('T0','T1') and ad.is_dir(),'Unknown submission arm')
                for slot in ad.iterdir():
                    require(slot.is_dir(),'Nonordinary submission slot')
                    if family=='trained_dev_71018e5':
                        require(slot.name in ('step-002','step-008','step-016'),'Unknown submitted DEV')
                        completed=ROOT/'trained_dev'/sd.name/ad.name/slot.name/'RESULT.json'
                        status='TRAINED_DEV_ENDPOINT_COMPLETE'
                    else:
                        require(slot.name in ('chunk-003-008','chunk-009-016','chunk-017-024'),'Unknown submitted chunk')
                        completed=ROOT/'continuation_control'/sd.name/ad.name/slot.name/'RESULT.json'
                        status='CONTINUATION_CHUNK_COMPLETE'
                    require((slot/'SUBMISSION.json').is_file() and (slot/'RELEASE_RESULT.json').is_file() and completed.is_file(),
                            'Prior submitted DEV/TRAIN active or unknown; never submit a competing worker')
                    require(c.tx.read_json(completed).get('status')==status,'Prior submitted controller not complete')
    base=schedule_root/'continuation_71018e5'
    planned_directory=base/f'start-{start}'/arm/f'chunk-{first:03d}-{last:03d}'
    require(not os.path.lexists(planned_directory),'Submission slot occupied: no retry')
    name=f'LH-SATDOSE-TRAIN-{start}-{arm}-{first:03d}-{last:03d}'
    scheduler=scheduler_before(name)
    ctx['authority'].verify(root=ROOT,identity=ctx['identity'],design_freeze_sha256=c.FREEZE_SHA,
      target_step=first,sampled_endpoint=ctx['endpoint'])
    c._hold_free();stable_locks();inventory()
    storage=c._pre_model_storage(ROOT/'execution_releases',release['storage_budget'])
    scheduler=scheduler_before(name);c._hold_free();stable_locks()
    for child in (base,base/f'start-{start}',base/f'start-{start}'/arm):
        if not child.exists():child.mkdir();syncdir(child.parent)
        require(child.resolve(strict=True)==child and child.is_dir(),'Noncanonical scheduling descendant')
    planned_directory.mkdir();syncdir(planned_directory.parent);directory=planned_directory
    args=['sbatch','--parsable','--hold','--job-name='+name,'--account=rose','--qos='+QOS,
      '--partition=cluster02','--nodes=1','--ntasks=1','--cpus-per-task=8','--gpus=pro6000:2',
      '--mem=66G','--time=24:00:00','--no-requeue','--export=ALL','--chdir='+str(CODE),
      '--output='+str(directory/'slurm-%j.out'),'--error='+str(directory/'slurm-%j.err'),
      str(CODE/'scheduling/run_continuation.sh'),str(c.bound(release_ref)),release_ref['sha256'],str(start),arm,str(first)]
    intent=dict(status='SUBMISSION_INTENT_UNKNOWN_UNTIL_RESPONSE_NO_RETRY',created_at_utc=utc(),
      selection=selection,planned_train_responses=128*(last-first+1),planned_formal_optimizer_steps=last-first+1,
      args=args,preparation=p['preparation_ref'],release=release_ref,canary_spec=release['canary_spec'],
      source_sha256=p['source_sha256'],scheduler=scheduler,storage=storage,
      walltime_rationale='24h scheduler ceiling, same as completed first-two job; not a scientific stop threshold or completion promise.',
      permission='ONE_FIXED_TRAIN_CHUNK_NO_IMPLICIT_SUCCESSOR_OR_MAIN')
    write('INTENT.json',intent)
    response=run(args) # The ONLY sbatch call. A rejected or ambiguous response is never retried.
    write('RESPONSE.json',response)
    require(response['returncode']==0 and re.fullmatch(r'[1-9][0-9]*',response['stdout'].strip()),
            'Submission rejected/unknown; preserve attempt, never resubmit')
    job=int(response['stdout'].strip())
    held=verify_own_job(job,name,True)
    accepted=dict(status='CONTINUATION_SUBMITTED_HELD_IDENTITY_VERIFIED',job_id=job,job_name=name,
      selection=selection,release=release_ref,preparation=p['preparation_ref'],held_observation=held,
      intent=ref(directory/'INTENT.json'),response=ref(directory/'RESPONSE.json'),accepted_at_utc=utc())
    write('SUBMISSION.json',accepted)
    stable_locks();unlock()
    held_again=verify_own_job(job,name,True)
    release_args=['scontrol','release',str(job)]
    write('RELEASE_INTENT.json',dict(status='RELEASE_ONCE_UNKNOWN_UNTIL_RESPONSE',job_id=job,args=release_args,
      submission=ref(directory/'SUBMISSION.json'),identity=held_again,created_at_utc=utc()))
    release_response=run(release_args)
    write('RELEASE_RESPONSE.json',release_response)
    require(release_response['returncode']==0,'Release failed/unknown: preserve known job, never resubmit')
    after=verify_own_job(job,name,False)
    accounting=run(['sacct','-j',str(job),'--noheader','--parsable2',
      '--format=JobIDRaw,State%40,ExitCode,JobName%120,Account%40,QOS%80'])
    require(accounting['returncode']==0,'Cannot query accepted job accounting; preserve known job ID')
    account_rows=[row.strip().split('|') for row in accounting['stdout'].splitlines() if row.strip()]
    require(all(len(row)==6 for row in account_rows),'Malformed accepted job accounting')
    parents=[row for row in account_rows if row[0]==str(job)]
    require(len(parents)<=1 and (not parents or parents[0][3:]==[name,'rose',QOS]),'Accepted accounting identity differs')
    require(not parents or parents[0][1] in ('PENDING','RUNNING','COMPLETING','COMPLETED'),
            'Accepted job failed/unknown; preserve no retry')
    require(not parents or parents[0][1]!='COMPLETED' or parents[0][2]=='0:0','Completed job has nonzero exit')
    result=dict(status='CONTINUATION_SUBMITTED_RELEASE_CONFIRMED',job_id=job,job_name=name,selection=selection,
      release=release_ref,submission=ref(directory/'SUBMISSION.json'),release_response=ref(directory/'RELEASE_RESPONSE.json'),
      scheduler=after,accounting=accounting,accounting_parent_visible=bool(parents),completed_at_utc=utc(),
      permission='ONE_FIXED_TRAIN_CHUNK_NO_IMPLICIT_SUCCESSOR_OR_MAIN',other_jobs_modified=False)
    write('RELEASE_RESULT.json',result)
    output=dict(result=result,receipt=ref(directory/'RELEASE_RESULT.json'))
except BaseException as exc:
    output=dict(status='SUBMISSION_OR_RELEASE_FAILED_OR_UNKNOWN_NO_RETRY',error_type=type(exc).__name__,error=str(exc),
      job_id=job,accepted=accepted,submission_directory=str(directory) if directory else None,
      observations=observations,recovery='Inspect exact job/attempt. Never resubmit, cancel, or release another job.')
    if directory is not None:
        try:write('ERROR.json',output)
        except BaseException:pass
finally:unlock()
print('SATDOSE_SUBMIT_JSON='+json.dumps(output,sort_keys=True,allow_nan=False),flush=True)
sys.exit(0 if 'result' in output else 2)
'''


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    def pairs(items):
        value=dict(items);require(len(value)==len(items),'Duplicate JSON key');return value
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
      parse_constant=lambda value:require(False,'Nonfinite JSON'))
def rootref(value):
    require(set(value)=={'path','sha256'} and isinstance(value['path'],str),'Malformed root reference')
    path=Path(value['path'])
    require(not path.is_absolute() and '..' not in path.parts and str(path)==value['path'] and
      re.fullmatch('[0-9a-f]{64}',value['sha256'] or ''),'Invalid root reference')
    return value
def local_ref(value):
    path=Path(value['path'])
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink() and
      sha(path)==value['sha256'],'Local provenance file missing/noncanonical/changed')
    return path
def write_once(path,value):
    require(path.parent.resolve(strict=True)==path.parent,'Canonical output parent required')
    data=value if isinstance(value,bytes) else json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def payload_from_receipt(path,expected):
    require(re.fullmatch('[0-9a-f]{64}',expected or ''),'Explicit preparation receipt SHA required')
    local_ref(dict(path=str(path),sha256=expected));receipt=read(path)
    require(receipt.get('status')=='CPU_PREPARATION_PUBLICATION_RECEIPT_CAPTURED' and receipt.get('returncode')==0,
            'Continuation preparation did not return successfully')
    source=receipt['local_operation_source']
    require(source['sha256']==PREPARER_SHA,'Unexpected continuation preparer source')
    source_path=local_ref(source)
    require(source.get('bytes')==source_path.stat().st_size,'Preparation source byte count differs')
    for value in receipt['raw_logs'].values():local_ref(value)
    require(set(receipt['raw_logs'])=={'stdout','stderr'},'Incomplete preparation raw logs')
    stdout=Path(receipt['raw_logs']['stdout']['path']).read_text(errors='replace')
    stderr=Path(receipt['raw_logs']['stderr']['path']).read_text(errors='replace')
    publication=receipt['publication'];report=publication['report']
    require(stdout==receipt['stdout'] and stderr==receipt['stderr'] and json.loads(stdout)==publication,
            'Raw preparation stdout/stderr differs from recorded publication')
    rootref(publication['preflight'])
    for key in ('intent','release','storage_budget','risk','original_train_release'):rootref(report[key])
    require(report.get('status')=='CONTINUATION_CPU_PREFLIGHT_COMPLETE_NO_MODEL_CALLS' and
      report.get('permission')=='NO_GPU_SUBMISSION' and report.get('preparation_script_sha256')==PREPARER_SHA and
      all(type(report.get(k)) is int and report[k]==0 for k in ('model_calls','optimizer_calls','gpu_jobs_submitted')),
      'Continuation preparation completion fields differ')
    selected=report['selection']
    require(set(selected)=={'start_seed','arm','first_step','last_step'} and
      type(selected['start_seed']) is int and selected['start_seed'] in (11031,11032) and selected['arm'] in ('T0','T1') and
      type(selected['first_step']) is int and type(selected['last_step']) is int and
      {3:8,9:16,17:24}.get(selected['first_step'])==selected['last_step'],'Unknown fixed chunk')
    require(publication['release_path']==str(Path(ROOT)/report['release']['path']) and
      publication['release_sha256']==report['release']['sha256'] and report['original_train_release']==ORIGINAL,
      'Preparation release/original reference differs')
    summary=receipt['summary']
    require(summary.get('mode')=='WRITE_ONCE_CPU_PREPARATION' and summary.get('selection')==selected and
      summary.get('original_train_release')==ORIGINAL and summary.get('gpu_submission') is False,
      'Local preparation summary differs')
    rootref(summary['canary_spec'])
    required_devs=2+4*sum(step<=selected['first_step']-1 for step in (2,8,16))
    require(report.get('completed_dev_endpoints')==required_devs and
      type(report.get('required_bytes')) is int and report['required_bytes']>0,'Missing prior four-path DEV barriers/budget')
    values=[]
    for name,expected_sha in PINS.items():
        pin=WORKSPACE/'artifacts/sr_satdose_20261112/intake'/name
        local_ref(dict(path=str(pin),sha256=expected_sha));values.append(read(pin))
    deployment,execution=values
    require(deployment['release_root']==CODE and len(deployment['sources'])==84 and len(execution)==19,'Original release identity differs')
    for value in execution.values():
        relative=str(Path(value['path']).relative_to(CODE))
        require(deployment['sources'][relative]==value['sha256'],'Original execution source pin differs')
    return dict(preparation=report,preparation_ref=publication['preflight'],preparer_sha=PREPARER_SHA,
      preparation_local=dict(path=str(path),sha256=expected),sources=deployment['sources'],execution_sources=execution,
      original=ORIGINAL,canary_spec=summary['canary_spec'],**selected)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-receipt',type=Path,required=True)
    parser.add_argument('--preparation-sha256',required=True)
    parser.add_argument('--receipt',type=Path,required=True,help='New local submission receipt; parent must exist')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    path=args.preparation_receipt.resolve(strict=True)
    payload=payload_from_receipt(path,args.preparation_sha256)
    source_path=Path(__file__).resolve();source=source_path.read_bytes();payload['source_sha256']=hashlib.sha256(source).hexdigest()
    compile(REMOTE,'<remote-continuation-submit>','exec')
    if not args.execute:
        print(json.dumps(dict(status='LOCAL_PLAN_ONLY_NO_SSH_OR_SUBMISSION',start_seed=payload['start_seed'],
          arm=payload['arm'],first_step=payload['first_step'],last_step=payload['last_step'],preparation=payload['preparation_ref'],
          protocol='One held sbatch; verify/persist exact new job; unlock worker; release only that job once.',
          resources=dict(account='rose',qos='soujanya-poria-startfund-2026-03',partition='cluster02',gpus='pro6000:2',
            cpus=8,memory='66G',time='24:00:00',requeue=False),source_sha256=payload['source_sha256'])))
        return 0
    require(args.receipt.is_absolute(),'Absolute local output path required')
    output=args.receipt.parent.resolve(strict=True)/args.receipt.name
    files={name:output.with_name(output.name+suffix) for name,suffix in
      [('intent','.intent.json'),('source','.source.py'),('stdout','.stdout.log'),('stderr','.stderr.log')]}
    require(not any(os.path.lexists(p) for p in [output,*files.values()]),'Local attempt occupied: never retry')
    write_once(files['source'],source)
    source_ref=dict(path=str(files['source']),sha256=sha(files['source']))
    write_once(files['intent'],dict(status='SSH_SUBMISSION_ATTEMPT_UNKNOWN_NO_RETRY',source=source_ref,payload=payload,
      stdout_path=str(files['stdout']),stderr_path=str(files['stderr'])))
    command=' '.join(shlex.quote(part) for part in ['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
      'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',PYTHON,'-I','-B','-c',REMOTE])
    try:
        outfd=os.open(files['stdout'],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(outfd,'wb') as out:
            errfd=os.open(files['stderr'],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(errfd,'wb') as err:
                try:
                    process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',
                      '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','eee-cluster',command],
                      input=json.dumps(payload),text=True,stdout=out,stderr=err,check=False)
                finally:out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
        stdout=files['stdout'].read_text(errors='replace');stderr=files['stderr'].read_text(errors='replace')
        lines=[line for line in stdout.splitlines() if line.startswith('SATDOSE_SUBMIT_JSON=')]
        response=json.loads(lines[0].split('=',1)[1]) if len(lines)==1 else None
        value=dict(status='SUBMISSION_RESPONSE_RETURNED' if response is not None else 'SUBMISSION_UNKNOWN_NO_RETRY',
          returncode=process.returncode,response=response,source=source_ref,stdout=stdout,stderr=stderr,
          raw_logs={k:dict(path=str(files[k]),sha256=sha(files[k])) for k in ('stdout','stderr')})
        write_once(output,value);print(json.dumps(dict(receipt=str(output),**value),sort_keys=True))
        return 0 if process.returncode==0 and response and 'result' in response else 2
    except BaseException as exc:
        if not os.path.lexists(output):
            write_once(output,dict(status='SUBMISSION_UNKNOWN_NO_RETRY',error_type=type(exc).__name__,error=str(exc),
              source=source_ref,intent=str(files['intent']),raw_logs={k:dict(path=str(files[k]),sha256=sha(files[k]))
                for k in ('stdout','stderr') if files[k].is_file()}))
        raise


if __name__=='__main__':raise SystemExit(main())
