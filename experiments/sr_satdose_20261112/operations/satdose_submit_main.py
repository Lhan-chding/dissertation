#!/usr/bin/env python3
"""Submit one fixed MAIN endpoint only after actual preparation and full terminal publication.

Default plan is local only. --execute performs exactly one held submit, verifies
and persists its exact identity, unlocks, then releases only that job once.
No retries, cancellation, other-job mutation, training or implicit next endpoint.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

PREPARER_SHA='6e40ae7735fc5d6e2386cf0e996a4db4a3fa90d6eea79d13a8d14d03325302b5'


def require(ok,message):
    if not ok:raise PermissionError('HOLD_MAIN_SUBMISSION: '+message)
def load_prepare(path):
    path=Path(path)
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Noncanonical prepare dependency')
    source=path.read_bytes();require(hashlib.sha256(source).hexdigest()==PREPARER_SHA,'Prepare dependency source changed')
    name='_satdose_main_prepare_local_dependency';spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);exec(compile(source,str(path),'exec'),module.__dict__)
    return module

REMOTE=r'''
import sys,os,json,hashlib,fcntl,subprocess,re,shlex,types
from pathlib import Path
from datetime import datetime,timezone
p=_PAYLOAD
ROOT=Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
CODE=ROOT/'external_main_code_4190bc3';QOS='soujanya-poria-startfund-2026-03'
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
       NumCPUs='8',NumTasks='1',Requeue='0',Command=str(CODE/'scheduling/run_main.sh'),
       WorkDir=str(CODE),StdOut=str(directory/f'slurm-{job}.out'),StdErr=str(directory/f'slurm-{job}.err'))
    require(all(fields.get(k)==v for k,v in expected.items()) and fields.get('NumNodes') in ('1','1-1') and
       fields.get('TimeLimit') in (('1-00:00:00','24:00:00') if p['responses']==1952 else ('06:00:00',)) and
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
    require(ROOT.resolve(strict=True)==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only fixed root required')
    endpoint=p['endpoint'];start,arm,step=endpoint['start_seed'],endpoint['arm'],endpoint['step']
    prepare_dir=ROOT/'execution_releases/main_preparations_4190bc3'/f'start-{start}'/arm/f'step-{step:03d}'
    source_path=prepare_dir/'PREPARER_SOURCE.py'
    require(sha(source_path)==PREPARER_SHA,'Wrong preparation verifier source')
    source=source_path.read_bytes();require(hashlib.sha256(source).hexdigest()==PREPARER_SHA,'Verifier changed during read')
    namespace={'__name__':'_satdose_main_prepare_submission_verifier','__file__':str(source_path)}
    exec(compile(source,str(source_path),'exec'),namespace);P=types.SimpleNamespace(**namespace)
    require(P.selection(start,arm,step)==(endpoint,p['responses']),'Unregistered MAIN endpoint/count')
    acquire('FIRST_TWO_LAUNCHER.lock',fcntl.LOCK_SH)
    acquire('DIAGNOSTIC_WORKER.lock',fcntl.LOCK_EX)
    c,ctx,prepared_inputs,report,actual_dir=P.revalidate_prepared(p['manifest'],PREPARER_SHA,p['pins'])
    require(actual_dir==prepare_dir and report==p['preparation_report'],'Remote preparation differs from successful local receipt')
    release_ref=report['release'];release=ctx['release'];target=c._slot(start,arm,step)
    require(not os.path.lexists(target),'MAIN target already attempted; never retry')
    schedule_root=ROOT.parent/'scheduling'
    require(schedule_root.resolve(strict=True)==schedule_root and schedule_root.is_dir(),'Canonical scheduling root missing')
    for family,control,status in (('trained_dev_71018e5','trained_dev','TRAINED_DEV_ENDPOINT_COMPLETE'),
      ('continuation_71018e5','continuation_control','CONTINUATION_CHUNK_COMPLETE'),('main_4190bc3','main_control','MAIN_ENDPOINT_COMPLETE')):
        parent=schedule_root/family
        if not os.path.lexists(parent):continue
        require(parent.resolve(strict=True)==parent and parent.is_dir(),'Noncanonical submission root')
        for item in parent.rglob('*'):
            require(not item.is_symlink() and 'ERROR' not in item.name and 'HOLD' not in item.name,'Failed/unknown submission evidence')
        for sd in parent.iterdir():
            require(sd.name in ('start-11031','start-11032') and sd.is_dir(),'Unknown submitted start')
            for ad in sd.iterdir():
                require(ad.name in (('START','T0','T1') if control=='main_control' else ('T0','T1')) and ad.is_dir(),'Unknown submitted arm')
                for slot in ad.iterdir():
                    require(slot.is_dir(),'Nonordinary submission slot')
                    if control=='continuation_control':require(slot.name in ('chunk-003-008','chunk-009-016','chunk-017-024'),'Unknown submitted chunk')
                    elif control=='trained_dev':require(slot.name in ('step-002','step-008','step-016'),'Unknown submitted DEV')
                    else:
                        require(re.fullmatch('step-[0-9]{3}',slot.name),'Malformed submitted MAIN')
                        P.selection(int(sd.name.removeprefix('start-')),ad.name,int(slot.name.removeprefix('step-')))
                    completed=ROOT/control/sd.name/ad.name/slot.name/'RESULT.json'
                    require((slot/'SUBMISSION.json').is_file() and (slot/'RELEASE_RESULT.json').is_file() and completed.is_file(),
                      'Prior submitted worker active/unknown; do not submit another MAIN')
                    require(P.read(completed).get('status')==status,'Prior submitted controller incomplete')
    base=schedule_root/'main_4190bc3';planned_directory=base/f'start-{start}'/arm/f'step-{step:03d}'
    require(not os.path.lexists(planned_directory),'Submission slot occupied; never retry')
    name=f'LH-SATDOSE-MAIN-{start}-{arm}-s{step:02d}'
    scheduler=scheduler_before(name)
    ctx['terminal'].stable();ctx['prior'].stable();c._hold_free();stable_locks()
    P.validate_terminal(prepared_inputs['terminal_manifest'],P.TERMINAL,P.TERMINAL/'PUBLISHER_SOURCE.py',remote=True)
    storage=c.storage_check(prepare_dir,release['storage_budget'])
    scheduler=scheduler_before(name);c._hold_free();stable_locks()
    for child in (base,base/f'start-{start}',base/f'start-{start}'/arm):
        if not child.exists():child.mkdir();syncdir(child.parent)
        require(child.resolve(strict=True)==child and child.is_dir(),'Noncanonical submission descendant')
    planned_directory.mkdir();syncdir(planned_directory.parent);directory=planned_directory
    walltime='24:00:00' if p['responses']==1952 else '06:00:00'
    args=['sbatch','--parsable','--hold','--job-name='+name,'--account=rose','--qos='+QOS,
      '--partition=cluster02','--nodes=1','--ntasks=1','--cpus-per-task=8','--gpus=pro6000:2',
      '--mem=66G','--time='+walltime,'--no-requeue','--export=ALL','--chdir='+str(CODE),
      '--output='+str(directory/'slurm-%j.out'),'--error='+str(directory/'slurm-%j.err'),
      str(CODE/'scheduling/run_main.sh'),str(c.bound(release_ref)),release_ref['sha256'],str(start),arm,str(step)]
    intent=dict(status='SUBMISSION_INTENT_UNKNOWN_UNTIL_RESPONSE_NO_RETRY',created_at_utc=utc(),endpoint=endpoint,
      planned_responses=p['responses'],args=args,preparation_manifest=p['manifest'],release=release_ref,
      terminal_publication=report['terminal_publication'],source_sha256=p['source_sha256'],scheduler=scheduler,storage=storage,
      permission='ONE_FIXED_MAIN_ENDPOINT_NO_TRAINING_OR_IMPLICIT_SUCCESSOR',walltime_note='Scheduler ceiling, not scientific stop or completion guarantee.')
    write('INTENT.json',intent)
    response=run(args) # Exactly one held sbatch. Never retry an uncertain response.
    write('RESPONSE.json',response)
    require(response['returncode']==0 and re.fullmatch(r'[1-9][0-9]*',response['stdout'].strip()),'Submission rejected/unknown; never resubmit')
    job=int(response['stdout'].strip());held=verify_own_job(job,name,True)
    accepted=dict(status='MAIN_SUBMITTED_HELD_IDENTITY_VERIFIED',job_id=job,job_name=name,endpoint=endpoint,
      planned_responses=p['responses'],release=release_ref,held_observation=held,intent=ref(directory/'INTENT.json'),
      response=ref(directory/'RESPONSE.json'),accepted_at_utc=utc())
    write('SUBMISSION.json',accepted)
    stable_locks();unlock()
    held_again=verify_own_job(job,name,True)
    c._hold_free()
    P.validate_terminal(prepared_inputs['terminal_manifest'],P.TERMINAL,P.TERMINAL/'PUBLISHER_SOURCE.py',remote=True)
    release_args=['scontrol','release',str(job)]
    write('RELEASE_INTENT.json',dict(status='RELEASE_ONCE_UNKNOWN_UNTIL_RESPONSE',job_id=job,args=release_args,
      submission=ref(directory/'SUBMISSION.json'),identity=held_again,created_at_utc=utc()))
    release_response=run(release_args)
    write('RELEASE_RESPONSE.json',release_response)
    require(release_response['returncode']==0,'Release failed/unknown; preserve exact job ID, never resubmit')
    after=verify_own_job(job,name,False)
    accounting=run(['sacct','-j',str(job),'--noheader','--parsable2',
      '--format=JobIDRaw,State%40,ExitCode,JobName%120,Account%40,QOS%80'])
    require(accounting['returncode']==0,'Cannot query accepted job accounting; preserve exact job ID')
    rows=[row.strip().split('|') for row in accounting['stdout'].splitlines() if row.strip()]
    require(all(len(row)==6 for row in rows),'Malformed accepted job accounting')
    parents=[row for row in rows if row[0]==str(job)]
    require(len(parents)<=1 and (not parents or parents[0][3:]==[name,'rose',QOS]),'Accounting identity differs')
    require(not parents or parents[0][1] in ('PENDING','RUNNING','COMPLETING','COMPLETED'),'Job failed/unknown after release')
    require(not parents or parents[0][1]!='COMPLETED' or parents[0][2]=='0:0','Completed accepted job exit failed')
    result=dict(status='MAIN_SUBMITTED_RELEASE_CONFIRMED',job_id=job,job_name=name,endpoint=endpoint,planned_responses=p['responses'],
      release=release_ref,submission=ref(directory/'SUBMISSION.json'),release_response=ref(directory/'RELEASE_RESPONSE.json'),
      scheduler=after,accounting=accounting,accounting_parent_visible=bool(parents),completed_at_utc=utc(),
      permission='ONE_FIXED_MAIN_ENDPOINT_NO_TRAINING_OR_IMPLICIT_SUCCESSOR',other_jobs_modified=False)
    write('RELEASE_RESULT.json',result);output=dict(result=result,receipt=ref(directory/'RELEASE_RESULT.json'))
except BaseException as exc:
    output=dict(status='MAIN_SUBMISSION_OR_RELEASE_FAILED_OR_UNKNOWN_NO_RETRY',error_type=type(exc).__name__,error=str(exc),
      job_id=job,accepted=accepted,submission_directory=str(directory) if directory else None,observations=observations,
      recovery='Inspect only this exact known job/attempt; never resubmit, cancel, or release another job.')
    if directory is not None:
        try:write('ERROR.json',output)
        except BaseException:pass
finally:unlock()
'''


def remote_submit(payload,source):
    require(hashlib.sha256(source).hexdigest()==payload['source_sha256'],'Submit source bytes differ')
    namespace=dict(_PAYLOAD=payload,PREPARER_SHA=PREPARER_SHA)
    exec(compile(REMOTE,'<main-submission-remote>','exec'),namespace)
    return namespace['output']

def local_preparation(P,path,expected):
    require(re.fullmatch('[0-9a-f]{64}',expected or '') and P.sha(path)==expected,'Explicit local preparation receipt SHA differs')
    receipt=P.read(path)
    require(receipt.get('status')=='MAIN_OPERATOR_RESPONSE_CAPTURED' and type(receipt.get('returncode')) is int and receipt['returncode']==0 and
      receipt['source']['sha256']==PREPARER_SHA,'No successful pinned MAIN preparation')
    for value in [receipt['source'],*receipt['raw_logs'].values()]:
        require(value==P.localref(Path(value['path'])),'Preparation raw source/log bytes differ')
    require(set(receipt['raw_logs'])=={'stdout','stderr'},'Missing preparation raw logs')
    stdout=Path(receipt['raw_logs']['stdout']['path']).read_bytes()
    marked=[line for line in stdout.splitlines() if line.startswith(b'SATDOSE_MAIN_OPERATOR=')]
    response=receipt['response'];require(len(marked)==1 and json.loads(marked[0].split(b'=',1)[1])==response,'Original preparation response differs')
    manifest=response['manifest'];report=response['result'];endpoint=manifest['endpoint']
    selected,count=P.selection(endpoint['start_seed'],endpoint['arm'],endpoint['step'])
    require(endpoint==selected and manifest.get('status')=='MAIN_PREPARATION_COMPLETE' and set(manifest['files'])==P.FILES and
      report.get('status')=='MAIN_CPU_PREFLIGHT_AND_STORAGE_COMPLETE' and report.get('permission')=='NO_MODEL_CALLS_OR_GPU_SUBMISSION' and
      report['endpoint']==endpoint and report['responses']==count and
      all(type(report.get(k)) is int and report[k]==0 for k in ('model_calls','optimizer_calls','gpu_submissions')),
      'Incomplete MAIN preparation result')
    directory=P.preparation_dir(endpoint)
    for name,value in manifest['files'].items():
        require(set(value)=={'path','sha256','bytes'} and value['path']==str(directory/name) and
          re.fullmatch('[0-9a-f]{64}',value['sha256'] or '') and type(value['bytes']) is int and value['bytes']>=0,
          'Preparation manifest relocated or malformed')
    require(manifest['files']['PREPARER_SOURCE.py']['sha256']==PREPARER_SHA and
      report['release']==dict(path=str((directory/'RELEASE.json').relative_to(P.ROOT)),sha256=manifest['files']['RELEASE.json']['sha256']),
      'Preparation source or release reference differs')
    require(report['prior_endpoints']==[entry[:3] for entry in P.ORDER].index((endpoint['start_seed'],endpoint['arm'],endpoint['step'])) and
      all(type(report.get(k)) is int and report[k]==v for k,v in
        [('formal_optimizer_steps',96),('train_responses',12288),('dev_endpoints',14),('dev_responses',1512)]),
      'Missing full terminal population or exact MAIN prefix')
    return dict(endpoint=endpoint,responses=count,manifest=manifest,preparation_report=report,pins=P.local_pins(),
      preparation_local=dict(path=str(path),sha256=expected))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-receipt',type=Path,required=True);parser.add_argument('--preparation-sha256',required=True)
    parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    P=load_prepare(Path(__file__).resolve().with_name('satdose_prepare_main.py'))
    payload=local_preparation(P,args.preparation_receipt.resolve(strict=True),args.preparation_sha256)
    source=Path(__file__).resolve().read_bytes();payload['source_sha256']=hashlib.sha256(source).hexdigest()
    compile(REMOTE,'<main-submission-remote>','exec')
    if not args.execute:
        print(json.dumps(dict(status='LOCAL_MAIN_SUBMISSION_PLAN_ONLY_NO_SSH',endpoint=payload['endpoint'],responses=payload['responses'],
          walltime='24:00:00' if payload['responses']==1952 else '06:00:00',account='rose',qos='soujanya-poria-startfund-2026-03',
          partition='cluster02',gpus='pro6000:2',cpus=8,memory='66G',requeue=False,
          protocol='One held sbatch, verified durable identity, unlock, release only that exact job once.')));return 0
    return P.invoke_remote(payload,source,args.receipt,'remote_submit',source_filename=Path(__file__).resolve())


if __name__=='__main__':raise SystemExit(main())
