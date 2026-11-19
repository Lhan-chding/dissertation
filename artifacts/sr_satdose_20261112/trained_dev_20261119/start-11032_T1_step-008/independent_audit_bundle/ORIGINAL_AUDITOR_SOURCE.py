#!/usr/bin/env python3
"""Independently score one actual completed trained DEV; default is local plan.

--execute never generates, trains, submits, retries or grants a successor. It
audits the explicitly bound own job, preserves negative/scientific-HOLD results,
publishes analysis-only evidence once, and downloads exact bytes. An occupied
publication directory is UNKNOWN/retained evidence, never a retry invitation.
"""
import argparse
import base64
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import traceback

WORKSPACE=Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ROOT=Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
CODE=ROOT/'train_code_71018e5'
PYTHON='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
QOS='soujanya-poria-startfund-2026-03'
SUBMITTER_SHA='40c465fea1719dc13d31b390c7583180b00d5cd9d05e90addee7f04bcc307bf5'
CANARY_SHA='44a8db68c655bea987629ea210da1a32e268806f6e0c7b1506d6d77828d4b260'
ORIGINAL=dict(path='execution_releases/FIRST_TWO_RELEASE_71018e5.json',
              sha256='22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f')
PINS={
 'FIRST_TWO_CODE_DEPLOYMENT_71018e5_20261116.json':'de42793b8d839e7d72d8eb399d37c0fb9e842d02f838709cde5b1a0e3ebbbeb8',
 'ORIGINAL_TRAIN_EXECUTION_SOURCES_71018e5.json':'5370b089381609c2e8d5d5ec7555c3dd441babaed242bd66b7cbfad02f679dd7'}
FILES={'CLAIM.json','SCHEDULER.json','OWN_JOB_STDOUT.log','OWN_JOB_STDERR.log','ORIGINAL_RESULT.json',
       'ORIGINAL_POST_RISK.json','ANALYSIS.json','RISK_REPLAY.json','AUDIT_REPORT.json','PUBLICATION.json','ERROR.json'}
PERMISSION='ANALYSIS_ONLY_NO_TRAINING_MAIN_OR_SUCCESSOR_AUTHORITY'


def require(ok,message):
    if not ok:raise PermissionError('HOLD_DEV_AUDIT: '+message)
def utc():return datetime.now(timezone.utc).isoformat()
def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def loads(data):
    def pairs(items):
        value=dict(items);require(len(value)==len(items),'Duplicate JSON key');return value
    return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda x:require(False,'Nonfinite JSON'))
def read(path):return loads(Path(path).read_bytes())
def ordinary(path):
    path=Path(path);require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and
                          not path.is_symlink(),'Noncanonical regular file: '+str(path));return path
def sha(path):
    path=ordinary(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def absolute_ref(path):return dict(path=str(ordinary(path)),sha256=sha(path))
def root_ref(path):return dict(path=str(ordinary(path).relative_to(ROOT)),sha256=sha(path))
def check_ref(reference,*,absolute=False):
    require(isinstance(reference,dict) and set(reference)=={'path','sha256'} and isinstance(reference['path'],str) and
            re.fullmatch('[0-9a-f]{64}',reference['sha256'] or ''),'Malformed trusted reference')
    path=Path(reference['path'])
    require(path.is_absolute()==absolute and '..' not in path.parts and str(path)==reference['path'],'Malformed reference path')
    return path
def bound(reference):
    path=ROOT/check_ref(reference);require(sha(path)==reference['sha256'],'Root artifact SHA differs');return path
def pinned_absolute(reference):
    path=check_ref(reference,absolute=True);require(sha(path)==reference['sha256'],'Absolute artifact SHA differs');return path
def write_once(path,value):
    path=Path(path);require(path.parent.resolve(strict=True)==path.parent,'Noncanonical output parent')
    data=value if isinstance(value,bytes) else canonical(value)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def endpoint(value):
    require(set(value)=={'start_seed','arm','step','responses'} and type(value['start_seed']) is int and
            value['start_seed'] in (11031,11032) and value['arm'] in ('T0','T1') and
            type(value['step']) is int and value['step'] in (2,8,16) and type(value['responses']) is int and
            value['responses']==108,'Only a fixed trained DEV108 endpoint is accepted')
    return value
def scheduling_path(ep):
    return ROOT.parent/'scheduling/trained_dev_71018e5'/f"start-{ep['start_seed']}"/ep['arm']/f"step-{ep['step']:03d}"
def audit_directory(job,ep):
    return ROOT/'execution_releases'/f"trained_dev_audit_job_{job}_s{ep['start_seed']}_{ep['arm']}_step{ep['step']:03d}"


def one_json_marker(data,prefix):
    lines=[line for line in data.splitlines() if line.startswith(prefix)]
    require(len(lines)==1,'Missing/ambiguous raw operation response')
    return loads(lines[0][len(prefix):])
def worker_stdout(data,result):
    values=[]
    for line in data.splitlines():
        if not line.startswith(b'{'):continue
        try:value=loads(line)
        except (ValueError,PermissionError):continue
        if isinstance(value,dict) and value.get('status') in ('TRAINED_DEV_ENDPOINT_COMPLETE','HOLD_TRAINED_DEV_CONTROL'):
            values.append(value)
    require(len(values)==1 and values[0]==result,'Own job stdout does not exactly bind completed endpoint result')


def scheduler(job):
    args=['sacct','-j',str(job),'--noheader','--parsable2',
          '--format=JobIDRaw,State%40,ExitCode,JobName%120,Account%40,QOS%80,Partition,Start,End,NodeList']
    environment={key:value for key,value in os.environ.items() if key!='SLURM_CLUSTERS'}
    response=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,env=environment)
    require(response.returncode==0,'Cannot read explicit own job accounting')
    fields=('job_id','state','exit_code','job_name','account','qos','partition','start','end','nodes');records=[]
    for line in response.stdout.decode().splitlines():
        if not line.strip():continue
        items=[x.strip() for x in line.split('|')];require(len(items)==len(fields),'Malformed scheduler record')
        row=dict(zip(fields,items));require(row['job_id']==str(job) or row['job_id'].startswith(str(job)+'.'),
                                           'Scheduler returned a foreign job')
        records.append(row)
    require(len({r['job_id'] for r in records})==len(records),'Duplicate own-job scheduler records')
    return dict(args=args,returncode=response.returncode,stdout=response.stdout.decode(),stderr=response.stderr.decode(),records=records)
def completed_scheduler(observation,job,ep,scientific_hold):
    rows=observation['records'];parents=[r for r in rows if r['job_id']==str(job)]
    require(len(parents)==1,'Missing/ambiguous own job parent')
    parent=parents[0];name=f"LH-SATDOSE-DEV-{ep['start_seed']}-{ep['arm']}-s{ep['step']:02d}"
    require(parent['job_name']==name and parent['account']=='rose' and parent['qos']==QOS and
            parent['partition']=='cluster02' and parent['end'] not in ('','Unknown','None'), 'Own job identity/completion differs')
    expected=('FAILED','2:0') if scientific_hold else ('COMPLETED','0:0')
    require((parent['state'],parent['exit_code'])==expected,'Job terminal status differs from actual scientific-HOLD result')
    allowed_ids={str(job),str(job)+'.batch',str(job)+'.extern',str(job)+'.0'}
    require(len({r['job_id'] for r in rows})==len(rows) and
            all(r['job_id'] in allowed_ids for r in rows) and
            sum(r['job_id']==str(job)+'.0' for r in rows)==1,'Unexpected/missing actual single-task model step')
    for row in rows:
        allowed={('COMPLETED','0:0')}|({('FAILED','2:0')} if scientific_hold else set())
        require((row['state'],row['exit_code']) in allowed,'Technical failure/unknown is not planned scientific exit2')


@contextmanager
def idle_lock(name):
    path=ordinary(ROOT/name);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
        stat=os.fstat(fd);require((stat.st_dev,stat.st_ino)==(path.stat().st_dev,path.stat().st_ino),'Worker lease replaced')
        yield
        require(not path.is_symlink() and (stat.st_dev,stat.st_ino)==(path.stat().st_dev,path.stat().st_ino),'Worker lease changed')
    finally:os.close(fd)


def verify_closed(result,binding,claim,close,runtime_receipt,ep,release_ref,original):
    require(result.get('schema')=='sr-satdose-trained-dev-result-v1' and result.get('status')=='TRAINED_DEV_ENDPOINT_COMPLETE' and
            result.get('permission')=='NO_TRAINING_OR_MAIN' and all(result.get(k)==ep[k] for k in ep) and
            result.get('release')==release_ref and type(result.get('global_scientific_hold')) is bool,'Incomplete/foreign DEV result')
    identity=binding['endpoint'];process=result['process_identity']
    require(binding.get('schema')=='sr-satdose-evaluation-endpoint-binding-v1' and binding['scope']=='DEV' and
            all(identity[k]==ep[k] for k in ('start_seed','arm','step')) and binding['requests']==108 and
            claim.get('schema')=='sr-satdose-trained-dev-claim-v1' and claim.get('status')=='ATTEMPTED_UNKNOWN_NO_RETRY' and
            claim.get('process_identity')==process and claim.get('release')==release_ref and
            all(claim.get(k)==ep[k] for k in ep),'Original claim/binding/process identity differs')
    require(close.get('closed') is True and close.get('scope')=='DEV' and close.get('attempted_requests')==108 and
            all(close.get(k)==v for k,v in identity.items()) and close.get('request_ids_sha256')==binding['request_ids_sha256'] and
            all(type(close.get(k)) is int and close[k]==0 for k in ('optimizer_steps','scheduler_steps','parameter_updates')),
            'DEV close is incomplete or changed state')
    require(runtime_receipt.get('schema')=='sr-satdose-evaluation-runtime-v1' and runtime_receipt.get('endpoint')==identity and
            runtime_receipt.get('requests')==108 and runtime_receipt.get('request_ids_sha256')==binding['request_ids_sha256'] and
            runtime_receipt.get('execution_sources_sha256')==binding['execution_sources_sha256'] and
            runtime_receipt.get('authority_source')==original['execution_sources']['trained_dev_control'] and
            runtime_receipt.get('optimizer_constructed') is False and runtime_receipt.get('scheduler_constructed') is False and
            runtime_receipt.get('training_rng_restored') is False,'GPU evaluation runtime construction differs')


def remote_worker(payload):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Independent scoring requires CPU-only environment')
    ep=endpoint(payload['endpoint']);job=payload['job_id'];require(type(job) is int and job>0,'Invalid explicit own job')
    directory=audit_directory(job,ep);require(not os.path.lexists(directory),'Existing audit attempt: preserve, never retry')
    require(directory.parent.resolve(strict=True)==directory.parent and directory.parent.is_dir(),
            'Noncanonical audit publication parent')
    require(ROOT.resolve(strict=True)==ROOT and CODE.resolve(strict=True)==CODE,'Wrong original canonical roots')
    actual={}
    for path in CODE.rglob('*'):
        require(path.resolve(strict=True)==path and not path.is_symlink(),'Noncanonical original code entry')
        if path.is_file():actual[str(path.relative_to(CODE))]=sha(path)
        else:require(path.is_dir(),'Special file in original code inventory')
    require(len(actual)==84 and actual==payload['sources'] and len(payload['execution_sources'])==19,'Original84/19 source inventory differs')
    for reference in payload['execution_sources'].values():
        path=pinned_absolute(reference);require(path.is_relative_to(CODE) and actual[str(path.relative_to(CODE))]==reference['sha256'],
                                               'Original execution source differs')
    expected_submission=scheduling_path(ep)/'RELEASE_RESULT.json'
    require(pinned_absolute(payload['submission_ref'])==expected_submission and read(expected_submission)==payload['submission'],
            'Local submission object does not reproduce actual own server receipt')
    submitted=payload['submission'];require(submitted['job_id']==job and submitted['endpoint']==ep,'Explicit job/endpoint mismatch')
    observation=scheduler(job);parents=[r for r in observation['records'] if r['job_id']==str(job)]
    require(len(parents)==1 and parents[0]['state'] not in ('PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED') and
            parents[0]['end'] not in ('','Unknown','None'),'Own job is not terminal; no audit attempt was created')
    sys.path.insert(0,str(CODE));sys.dont_write_bytecode=True
    from execution import trained_dev_control as control
    from analysis import evaluation,risk
    from runtime.backend import object_sha
    import torch
    torch.set_num_threads(1)
    require(control.PACKAGE==CODE and control.ROOT==ROOT,'Shadowed frozen original modules')
    with ExitStack() as stack:
        for name in ('FIRST_TWO_LAUNCHER.lock','DIAGNOSTIC_WORKER.lock'):stack.enter_context(idle_lock(name))
        directory.mkdir();write_once(directory/'CLAIM.json',dict(status='ANALYSIS_ATTEMPT_UNKNOWN_UNTIL_PUBLICATION',job_id=job,
            endpoint=ep,submission=payload['submission_ref'],operation_source_sha256=payload['operation_source_sha256'],
            permission=PERMISSION,started_at_utc=utc()))
        try:
            write_once(directory/'SCHEDULER.json',observation)
            schedule=scheduling_path(ep);observed={str(expected_submission):sha(expected_submission)}
            def tracked(path):
                path=ordinary(path);value=sha(path);require(str(path) not in observed or observed[str(path)]==value,'Audit input mutated')
                observed[str(path)]=value;return path
            for label in ('STDOUT','STDERR'):
                path=tracked(schedule/f"slurm-{job}.{'out' if label=='STDOUT' else 'err'}")
                write_once(directory/f'OWN_JOB_{label}.log',path.read_bytes())
            acceptance_path=pinned_absolute(submitted['submission']);acceptance=read(tracked(acceptance_path))
            intent_path=pinned_absolute(acceptance['intent']);intent=read(tracked(intent_path))
            require(acceptance_path==schedule/'SUBMISSION.json' and intent_path==schedule/'INTENT.json' and
                    acceptance['job_id']==job and acceptance['endpoint']==ep and acceptance['release']==submitted['release'] and
                    intent['endpoint']==ep and intent['release']==submitted['release'] and intent['source_sha256']==SUBMITTER_SHA,
                    'Own scheduler submission lineage differs')
            release_ref=submitted['release'];release,original=control.verify_release(bound(release_ref),release_ref['sha256'])
            require(release['original_train_release']==ORIGINAL and original['sources']==actual and
                    original['execution_sources']==payload['execution_sources'] and release['canary_spec']['sha256']==CANARY_SHA,
                    'Original TRAIN/canary identity differs')
            expected_command=[str(CODE/'scheduling/run_trained_dev.sh'),str(bound(release_ref)),release_ref['sha256'],
                              str(ep['start_seed']),ep['arm'],str(ep['step'])]
            require(intent['args'][-6:]==expected_command and '--output='+str(schedule/'slurm-%j.out') in intent['args'] and
                    '--error='+str(schedule/'slurm-%j.err') in intent['args'],'Job command/log location does not bind exact endpoint')
            held=acceptance['held_observation']['fields']
            require(held['JobId']==str(job) and held['Command']==expected_command[0] and held['StdOut']==str(schedule/f'slurm-{job}.out') and
                    held['StdErr']==str(schedule/f'slurm-{job}.err'),'Actual accepted scheduler command/log identity differs')
            for reference in (release_ref,release['admission'],release['admission_spec'],release['canary_spec'],ORIGINAL):tracked(bound(reference))
            slot=control._slot(ep['start_seed'],ep['arm'],ep['step']);output=slot/'endpoint'
            result_path=tracked(slot/'RESULT.json');result=read(result_path)
            write_once(directory/'ORIGINAL_RESULT.json',result_path.read_bytes())
            worker_stdout((directory/'OWN_JOB_STDOUT.log').read_bytes(),result)
            binding=read(tracked(bound(result['endpoint_binding'])));claim=read(tracked(slot/'CLAIM.json'))
            close=read(tracked(bound(result['close'])));runtime_receipt=read(tracked(output/'EVALUATION_RUNTIME.json'))
            verify_closed(result,binding,claim,close,runtime_receipt,ep,release_ref,original)
            require(bound(result['endpoint_binding'])==slot/'ENDPOINT_BINDING.json' and bound(result['close'])==slot/'CLOSE.json' and
                    bound(result['receipt'])==output/'RECEIPT.json' and bound(result['post_risk'])==slot/'POST_RISK.json',
                    'Completed DEV artifacts were relocated')
            require(result['admission']==release['admission'] and result['commit']==release['trusted_commits'][f"{ep['start_seed']}/{ep['arm']}"][-1] and
                    result['canary']==read(bound(release['canary_spec']))['published'],'DEV admission/canary/checkpoint differs')
            receipt=read(tracked(bound(result['receipt'])))
            require(close['accounting']==control.rows(bound(receipt['accounting'])),'CLOSE/native accounting differs')
            freeze_path=tracked(ROOT/'DESIGN_FREEZE.json');freeze=read(freeze_path)
            analysis=evaluation.analyze_completed_endpoint(bound(freeze['artifacts']['config']),freeze_path,ROOT,bound(result['receipt']),
                expected_receipt_sha256=result['receipt']['sha256'],endpoint_binding_path=bound(result['endpoint_binding']),
                expected_endpoint_binding_sha256=result['endpoint_binding']['sha256'])
            require(len(analysis['rows'])==108 and analysis['summary']['metrics']['assigned']==108,'Independent score population differs')
            write_once(directory/'ANALYSIS.json',analysis)
            post_path=tracked(bound(result['post_risk']));post=read(post_path)
            write_once(directory/'ORIGINAL_POST_RISK.json',post_path.read_bytes())
            endpoints=loads(canonical(release['dev_endpoints']))
            key=f"{ep['start_seed']}/{ep['arm']}/{ep['step']}";require(key not in endpoints,'Current endpoint already existed before this invocation')
            endpoints[key]=dict(receipt=result['receipt'],binding=result['endpoint_binding'])
            replay=risk.assess(ROOT,config_ref=freeze['artifacts']['config'],
                freeze_ref=dict(path='DESIGN_FREEZE.json',sha256=original['design_freeze_sha256']),
                clarification_ref=release['risk_clarification'],trusted_commits=release['trusted_commits'],dev_endpoints=endpoints,
                expected_source_sha256=original['execution_sources']['risk_verifier']['sha256'])
            write_once(directory/'RISK_REPLAY.json',replay)
            require(replay==post and result['global_scientific_hold'] is post['global_hold'],'Independent POST_RISK object differs')
            hold=post['global_hold'];completed_scheduler(observation,job,ep,hold)
            if hold:
                global_path=tracked(ROOT/'training_control/HOLD.json');global_hold=read(global_path)
                require(global_hold.get('status')=='GLOBAL_SCIENTIFIC_HOLD' and global_hold.get('hold_reasons')==post['hold_reasons'] and
                        global_hold.get('evidence')==result['post_risk'],'Scientific-HOLD global record differs')
            existing_hold=None
            global_path=ROOT/'training_control/HOLD.json'
            if global_path.exists():
                tracked(global_path);existing_hold=dict(reference=root_ref(global_path),value=read(global_path))
            for path,value in replay['observed_files'].items():
                require(sha(path)==value,'Replayed risk input mutated');observed[str(path)]=value
            for path,value in observed.items():require(sha(path)==value,'Observed source/input changed during audit')
            control.verify_release(bound(release_ref),release_ref['sha256'])
            report=dict(schema='sr-satdose-independent-trained-dev-audit-v1',
                status='TRAINED_DEV_INDEPENDENT_AUDIT_COMPLETE_SCIENTIFIC_HOLD' if hold else 'TRAINED_DEV_INDEPENDENT_AUDIT_COMPLETE',
                permission=PERMISSION,job_id=job,endpoint=ep,release=release_ref,submission=payload['submission_ref'],
                result=root_ref(result_path),analysis=root_ref(directory/'ANALYSIS.json'),original_post_risk=root_ref(directory/'ORIGINAL_POST_RISK.json'),
                risk_replay=root_ref(directory/'RISK_REPLAY.json'),exact_post_risk_object_reproduced=True,global_scientific_hold=hold,
                hold_reasons=post['hold_reasons'],pending_required_dev=post['pending_required_dev'],responses=108,
                optimizer_steps=0,scheduler_steps=0,parameter_updates=0,audit_model_calls=0,audit_optimizer_calls=0,
                original_train_release=ORIGINAL,source_files=84,execution_source_keys=19,observed_files=observed,
                operation_source_sha256=payload['operation_source_sha256'],completed_at_utc=utc(),
                scientific_outcome_preserved=True,next_stage_authorized=False,existing_global_hold=existing_hold,
                limitation='Analysis-only; pending DEV/global HOLD and every subsequent authority remain separate.')
            write_once(directory/'AUDIT_REPORT.json',report)
            publication=dict(status='ANALYSIS_ONLY_ARTIFACTS_PUBLISHED',permission=PERMISSION,job_id=job,endpoint=ep,
                global_scientific_hold=hold,report=root_ref(directory/'AUDIT_REPORT.json'),
                files={p.name:root_ref(p) for p in sorted(directory.iterdir()) if p.is_file()},completed_at_utc=utc())
            write_once(directory/'PUBLICATION.json',publication)
        except BaseException as exc:
            write_once(directory/'ERROR.json',dict(status='INDEPENDENT_DEV_AUDIT_FAILED_OR_UNKNOWN_NO_RETRY',permission=PERMISSION,
                error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),job_id=job,endpoint=ep,
                completed_at_utc=utc(),recovery='Preserve all original and audit artifacts. Do not resample or overwrite.'))
            return manifest(job,ep)
    return manifest(job,ep)


def manifest(job,ep):
    directory=audit_directory(job,ep);require(directory.resolve(strict=True)==directory,'Noncanonical audit directory')
    files={}
    for path in directory.iterdir():
        require(path.name in FILES,'Unexpected audit publication file');ordinary(path)
        files[path.name]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    return dict(status='ANALYSIS_PUBLICATION_COMPLETE' if 'PUBLICATION.json' in files and 'ERROR.json' not in files else
        'AUDIT_FAILED_OR_UNKNOWN_ARTIFACTS_RETAINED',success='PUBLICATION.json' in files and 'ERROR.json' not in files,
        job_id=job,endpoint=ep,files=files,permission=PERMISSION)


def local_payload(args):
    path=pinned_absolute(dict(path=str(args.submission_receipt),sha256=args.submission_sha256));receipt=read(path)
    require(receipt.get('status')=='SUBMISSION_RESPONSE_RETURNED' and receipt.get('returncode')==0,'No successful actual submission receipt')
    require(receipt['source']['sha256']==SUBMITTER_SHA,'Unexpected submitter implementation');pinned_absolute(receipt['source'])
    for reference in receipt['raw_logs'].values():pinned_absolute(reference)
    response=receipt['response'];submitted=response['result']
    raw=Path(receipt['raw_logs']['stdout']['path']).read_bytes()
    require(one_json_marker(raw,b'SATDOSE_SUBMIT_JSON=')==response,'Actual submit stdout and receipt disagree')
    ep=endpoint(dict(start_seed=args.start_seed,arm=args.arm,step=args.step,responses=108))
    require(submitted.get('status')=='TRAINED_DEV_SUBMITTED_RELEASE_CONFIRMED' and type(submitted.get('job_id')) is int and
            submitted['job_id']==args.job_id and submitted['endpoint']==ep and submitted.get('other_jobs_modified') is False,
            'Explicit endpoint/job does not match own successful submit receipt')
    require(check_ref(response['receipt'],absolute=True)==scheduling_path(ep)/'RELEASE_RESULT.json','Foreign scheduling receipt path')
    deployment,execution=[read(pinned_absolute(dict(path=str(WORKSPACE/'artifacts/sr_satdose_20261112/intake'/name),sha256=value)))
                          for name,value in PINS.items()]
    require(deployment['release_root']==str(CODE) and len(deployment['sources'])==84 and len(execution)==19,'Frozen local84/19 identity differs')
    return dict(job_id=args.job_id,endpoint=ep,submission=submitted,submission_ref=response['receipt'],
        sources=deployment['sources'],execution_sources=execution,
        local_submission_ref=dict(path=str(path),sha256=args.submission_sha256))


def local(args):
    payload=local_payload(args);source=Path(__file__).read_bytes();payload['operation_source_sha256']=digest(source)
    if not args.execute:
        print(json.dumps(dict(status='LOCAL_PLAN_ONLY_NO_SSH',job_id=args.job_id,endpoint=payload['endpoint'],
            permission=PERMISSION,remote_output=str(audit_directory(args.job_id,payload['endpoint'])),source_sha256=digest(source)),indent=2));return 0
    require(args.local_dir is not None and args.local_dir.is_absolute(),'Explicit absolute new local output directory required')
    destination=args.local_dir.parent.resolve(strict=True)/args.local_dir.name
    require(not os.path.lexists(destination),'Existing local audit attempt: never overwrite/retry');destination.mkdir()
    source_path=destination/'OPERATION_SOURCE.py';write_once(source_path,source)
    write_once(destination/'LOCAL_INTENT.json',dict(status='SSH_AUDIT_ATTEMPT_UNKNOWN_UNTIL_RESPONSE',payload=payload,
        operation_source=absolute_ref(source_path),created_at_utc=utc(),permission=PERMISSION))
    encoded=base64.b64encode(canonical(payload)).decode()
    bootstrap=('import hashlib,sys; s=sys.stdin.buffer.read(); '
               f'assert hashlib.sha256(s).hexdigest()=={digest(source)!r}; '
               "exec(compile(s,'<pinned-independent-dev-auditor>','exec'))")
    command=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1',
        'TRANSFORMERS_OFFLINE=1','OMP_NUM_THREADS=1',PYTHON,'-I','-B','-c',bootstrap,'--remote-worker',encoded])
    stdout=destination/'REMOTE_STDOUT.log';stderr=destination/'REMOTE_STDERR.log'
    try:
        with stdout.open('xb') as out,stderr.open('xb') as err:
            try:response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',
                '-o','ServerAliveCountMax=4','eee-cluster',command],input=source,stdout=out,stderr=err,check=False)
            finally:out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
        value=one_json_marker(stdout.read_bytes(),b'SATDOSE_DEV_AUDIT_JSON=')
        require(value['job_id']==args.job_id and value['endpoint']==payload['endpoint'],'Remote audit response substituted endpoint/job')
        write_once(destination/'REMOTE_MANIFEST.json',value);downloaded={}
        for name,reference in sorted(value['files'].items()):
            require(name in FILES and Path(reference['path'])==audit_directory(args.job_id,payload['endpoint'])/name,'Download escapes own audit')
            cmd=shlex.join(['cat','--',reference['path']])
            got=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','eee-cluster',cmd],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
            require(len(got.stdout)==reference['bytes'] and digest(got.stdout)==reference['sha256'],'Downloaded audit bytes differ')
            write_once(destination/name,got.stdout);downloaded[name]=dict(reference,local_path=str(destination/name))
        write_once(destination/'LOCAL_RECEIPT.json',dict(status='EXACT_ANALYSIS_ARTIFACTS_DOWNLOADED',permission=PERMISSION,
            job_id=args.job_id,endpoint=payload['endpoint'],audit_success=value['success'],remote_returncode=response.returncode,
            operation_source=absolute_ref(source_path),files=downloaded,raw_logs=dict(stdout=absolute_ref(stdout),stderr=absolute_ref(stderr)),
            completed_at_utc=utc(),next_stage_authorized=False))
        print(json.dumps(dict(status=value['status'],local_dir=str(destination),files=len(downloaded),permission=PERMISSION),indent=2))
        return 0 if response.returncode==0 and value['success'] else 2
    except BaseException as exc:
        if not (destination/'LOCAL_ERROR.json').exists():
            write_once(destination/'LOCAL_ERROR.json',dict(status='AUDIT_OR_DOWNLOAD_FAILED_UNKNOWN_NO_RETRY',permission=PERMISSION,
                error_type=type(exc).__name__,error=str(exc),operation_source=absolute_ref(source_path),
                raw_logs={k:absolute_ref(p) for k,p in [('stdout',stdout),('stderr',stderr)] if p.is_file()},
                recovery='Read retained local/remote artifacts; never rerun an occupied publication or resample.'))
        raise


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if len(argv)==2 and argv[0]=='--remote-worker':
        payload=loads(base64.b64decode(argv[1],validate=True))
        try:value=remote_worker(payload)
        except BaseException as exc:
            traceback.print_exc();value=dict(status='AUDIT_NOT_STARTED_OR_UNKNOWN',success=False,job_id=payload['job_id'],
                endpoint=payload['endpoint'],files={},error=str(exc),permission=PERMISSION)
        print('SATDOSE_DEV_AUDIT_JSON='+json.dumps(value,sort_keys=True));return 0 if value['success'] else 2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission-receipt',type=Path,required=True);parser.add_argument('--submission-sha256',required=True)
    parser.add_argument('--job-id',type=int,required=True);parser.add_argument('--start-seed',type=int,choices=(11031,11032),required=True)
    parser.add_argument('--arm',choices=('T0','T1'),required=True);parser.add_argument('--step',type=int,choices=(2,8,16),required=True)
    parser.add_argument('--local-dir',type=Path);parser.add_argument('--execute',action='store_true')
    return local(parser.parse_args(argv))


if __name__=='__main__':raise SystemExit(main())
