#!/usr/bin/env python3
"""Prepare/publish actual SATDOSE canary evidence after a completed first-two job.

Default --mode plan is LOCAL ONLY. --mode publish explicitly invokes SSH and
writes only a fresh job-specific publication directory; --mode download is
read-only remote recovery. This script never submits jobs or calls a model.
No overwrite/retry of an existing publication attempt is allowed.
"""
import argparse
import base64
from contextlib import ExitStack, contextmanager
import datetime
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

ROOT=Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
PYTHON='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
INTAKE=Path('/Users/louis/Documents/ChatGPT/dissertation-ntu/artifacts/sr_satdose_20261112/intake/FIRST_TWO_CODE_DEPLOYMENT_71018e5_20261116.json')
KEYS={'first_two_control','training_runtime','training_step','training_state','training_transaction','production',
      'calibration','satdose_core','scorer_bindings','admission_verifier','canary_replay','frozen_fingerprint',
      'storage_ready','evaluation_analysis','canary_controller','continuation_verifier','risk_verifier',
      'trained_dev_control','continuation_control'}
FILES={'CLAIM.json','SCHEDULER_SACCT.txt','PRECHECK.json','INDEPENDENT_CANARY_AUDIT.json',
       'CANARY_INPUT_SPEC.json','CANARY_RESUME_PASS.json','CANARY_SPEC.json','PUBLICATION_RECEIPT.json','ERROR.json'}


def require(ok,message):
    if not ok:raise PermissionError('HOLD_CANARY_PUBLICATION: '+message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):result.update(data)
    return result.hexdigest()
def read(path):
    def pairs(values):
        result=dict(values);require(len(result)==len(values),'duplicate JSON key');return result
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
                      parse_constant=lambda value:require(False,'nonfinite JSON'))
def ordinary(path):
    path=Path(path);require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and
                          not path.is_symlink(),'noncanonical ordinary file: '+str(path));return path
def ref(path):return dict(path=str(ordinary(path).relative_to(ROOT)),sha256=sha(path))
def write_once(path,value):
    data=value if isinstance(value,bytes) else canonical(value)
    with Path(path).open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextmanager
def existing_lock(path):
    ordinary(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require((os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(path.stat().st_dev,path.stat().st_ino),'lock replaced')
        yield
    finally:os.close(fd)


def scheduler(job):
    command=['sacct','-j',job,'--noheader','--parsable2',
             '--format=JobIDRaw,State%40,ExitCode,JobName%120,Account%40,QOS%80,Start,End,NodeList']
    response=subprocess.run(command,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    names=['job_id','state','exit_code','job_name','account','qos','start','end','nodes']
    records=[dict(zip(names,(item.strip() for item in line.split('|')))) for line in response.stdout.decode().splitlines() if line.strip()]
    actual=[row for row in records if row['job_id']==job]
    require(len(actual)==1,'exact scheduler parent job missing/ambiguous')
    selected=actual[0]
    require(selected['state']=='COMPLETED' and selected['exit_code']=='0:0' and selected['account']=='rose' and
            selected['qos']=='soujanya-poria-startfund-2026-03' and selected['end'] not in ('Unknown','None',''),
            'actual first-two job is not successful completed teacher-QoS job')
    require(all(row['state']=='COMPLETED' and row['exit_code']=='0:0' for row in records),
            'retained scheduler child record failed/unknown')
    return response.stdout,dict(command=command,records=records,selected=selected)


def publication_directory(job):return ROOT/'execution_releases'/('canary_job_'+job)


def remote_worker(payload):
    """Executed only by explicit --mode publish, in a fresh CPU-only interpreter."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only audit environment required')
    require(ROOT.resolve(strict=True)==ROOT,'fixed evidence root changed')
    job=payload['job_id'];release_path=ordinary(payload['release_path'])
    require(release_path.is_relative_to(ROOT/'execution_releases') and sha(release_path)==payload['release_sha256'],
            'explicit FIRST_TWO_RELEASE path/SHA differs')
    release=read(release_path);deployment=payload['deployment'];package=Path(release['release_root'])
    require(package==Path(deployment['release_root']) and package.resolve(strict=True)==package and
            package.is_relative_to(ROOT) and release['sources']==deployment['sources'] and len(release['sources'])==84,
            'immutable 84-source release differs from independent deployment intake')
    actual={}
    for path in package.rglob('*'):
        require(path.resolve(strict=True)==path and not path.is_symlink(),'noncanonical release entry')
        if path.is_file():actual[str(path.relative_to(package))]=sha(path)
        else:require(path.is_dir(),'nonordinary release entry')
    require(actual==release['sources'],'full actual release inventory/hash differs')
    require(set(release['execution_sources'])==KEYS,'original execution identity is not exact 19-key inventory')
    for name,reference in release['execution_sources'].items():
        path=ordinary(reference['path'])
        require(path.is_relative_to(package) and sha(path)==reference['sha256']==actual[str(path.relative_to(package))],
                'execution source pin differs: '+name)
    raw_scheduler,job_observation=scheduler(job)
    # Import only after independently checking every source against local intake.
    os.chdir(package);sys.path.insert(0,str(package));sys.dont_write_bytecode=True
    from execution import first_two_control as worker, canary_audit, continuation_authority, admission_authority
    from execution import training_transaction as tx
    from scheduling import first_two_launcher as launcher
    from runtime.backend import object_sha
    import torch
    torch.set_num_threads(1)
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve(),'wrong production CPU interpreter')
    require(worker.verify_release(release_path,payload['release_sha256'])==release,'native release verification differs')
    require(release['execution_sources']['canary_controller']==dict(path=str(Path(launcher.__file__).resolve()),
                                                                  sha256=sha(launcher.__file__)),'actual launcher source differs')
    directory=publication_directory(job);require(not os.path.lexists(directory),'existing publication attempt; audit retained files, never retry/overwrite')
    with ExitStack() as stack:
        for name in ('FIRST_TWO_LAUNCHER.lock','DIAGNOSTIC_WORKER.lock'):stack.enter_context(existing_lock(ROOT/name))
        directory.mkdir();write_once(directory/'CLAIM.json',dict(status='ATTEMPTED_UNKNOWN_UNTIL_PUBLICATION_COMPLETE',
            job_id=job,release=ref(release_path),publisher_sha256=payload['publisher_sha256'],
            deployment_intake_sha256=payload['deployment_intake_sha256'],started_at_utc=utc(),model_calls=0))
        try:
            write_once(directory/'SCHEDULER_SACCT.txt',raw_scheduler)
            worker._no_unknown()
            launcher_path=ROOT/'first_two_launcher/RESULT.json';result=read(ordinary(launcher_path))
            require(result.get('schema')=='sr-satdose-first-two-launcher-result-v1' and
                    result.get('status')=='FOUR_FIRST_TWO_PATHS_EXIT_OBSERVED_NOT_CANARY_ADMISSION' and
                    result.get('permission')=='NO_STEP3_OR_CANARY_ADMISSION' and result.get('release')==ref(release_path) and
                    (result.get('train_responses'),result.get('formal_optimizer_steps'),result.get('model_worker_processes'))==(1024,8,8) and
                    set(result['paths'])==set(canary_audit.PATHS),'incomplete/wrong actual launcher result')
            claim=read(ROOT/'first_two_launcher/CLAIM.json');controller=claim['controller_process']
            source=release['execution_sources']['canary_controller'];canary_audit._process(controller)
            require(claim['release']==ref(release_path) and claim['controller_source']==source,'launcher claim belongs to another release')
            families=dict(trusted_specs={},frozen_witnesses={},process_exit_proofs={});processes=[];workers={}
            for start,arm in launcher.PATHS:
                key=f'{start}/{arm}';slot=ROOT/f'first_two_launcher/start-{start}/{arm}';path_result=read(slot/'RESULT.json')
                require(path_result==result['paths'][key] and path_result['status']=='FIRST_TWO_PATH_EXIT_OBSERVED_NOT_CANARY_ADMISSION',
                        'launcher per-path result differs')
                contexts={}
                for phase in ('baseline','reload'):
                    prefix=phase.upper();launch_path=slot/(prefix+'_LAUNCH_INTENT.json')
                    intent=read(launch_path);spawn=read(slot/(prefix+'_SPAWNED.json'));wait=read(slot/(prefix+'_WAIT_OBSERVATION.json'))
                    baseline_sha=None if phase=='baseline' else path_result['baseline_result']['sha256']
                    require(intent['command']==launcher._command(release_path,payload['release_sha256'],start,arm,phase,baseline_sha) and
                            intent['phase']==phase and intent['controller_process']==controller and intent['controller_source']==source,
                            'actual launch command/source differs')
                    process=wait['worker_process'];canary_audit._process(process)
                    require(spawn['launch']==wait['launch']==ref(launch_path) and spawn['worker_process']==process and
                            spawn['controller_process']==wait['controller_process']==controller and
                            spawn['controller_source']==wait['controller_source']==source and
                            spawn['direct_child_pid']==wait['waited_pid']==process['pid'] and wait['kind']=='DIRECT_CHILD_WAIT' and
                            type(wait['returncode']) is int and wait['returncode']==0 and
                            intent['monotonic_ns']<spawn['monotonic_ns']<wait['monotonic_ns'],'actual direct-child wait chain differs')
                    ctx=launcher._result(start,arm,phase,{'process':process},ref(release_path),release)
                    require(ctx['reference']==path_result[phase+'_result'],'worker RESULT not bound by launcher')
                    worker_claim=read(ROOT/f'first_two_control/start-{start}/{arm}/{phase}/CLAIM.json')
                    require(worker_claim['process']==process and worker_claim['release']==ref(release_path) and
                            worker_claim['start_seed']==start and worker_claim['arm']==arm and worker_claim['phase']==phase and
                            worker_claim['allocation']['job_id']==job and worker_claim['allocation']['qos']==job_observation['selected']['qos'],
                            'explicit scheduler job did not run this worker')
                    processes.append(process);contexts[phase]=ctx
                require(contexts['baseline']['result']['identity']==contexts['reload']['result']['identity'] and
                        contexts['baseline']['result']['identity']['execution_sources_sha256']==object_sha(release['execution_sources']),
                        'baseline/reload original scientific identity differs')
                families['trusted_specs'][key]=path_result['canary_spec']
                families['frozen_witnesses'][key]=path_result['frozen_witnesses']
                families['process_exit_proofs'][key]=path_result['process_exit_proof']
                workers[key]={phase:ctx['reference'] for phase,ctx in contexts.items()}
            require(len({json.dumps(p,sort_keys=True) for p in processes})==8,'worker process incarnation reused')
            original_spec=read(tx.bound(ROOT,release['admission_spec']))
            original=admission_authority.FirstTwoStepAdmission(original_spec,admission=release['admission'])
            original._observed.stable()
            original_admission=read(tx.bound(ROOT,release['admission']))
            require(original_admission['prefix_measurement_passed'] is False and
                    original_admission['prefix_gate_decision']=='PASS_USER_OVERRIDE','original PREFIX failure/override was rewritten')
            precheck=dict(status='ACTUAL_FOUR_PATHS_READY_FOR_INDEPENDENT_CPU_AUDIT_NOT_PASS',job=job_observation,
                release=ref(release_path),launcher=ref(launcher_path),workers=workers,source_files=84,execution_source_keys=sorted(KEYS),
                execution_sources_sha256=object_sha(release['execution_sources']),prefix_measurement_passed=False,
                prefix_gate_decision='PASS_USER_OVERRIDE',model_calls=0)
            write_once(directory/'PRECHECK.json',precheck)
            audit=canary_audit.audit_four_paths(ROOT,**families,execution_sources=release['execution_sources'],
                expected_auditor_sha256=release['execution_sources']['frozen_fingerprint']['sha256'],
                expected_freeze_sha256=release['design_freeze_sha256'])
            # Preserve exact raw report even if it reports PENDING or a negative result.
            write_once(directory/'INDEPENDENT_CANARY_AUDIT.json',audit)
            spec=dict(root=str(ROOT),helper=release['execution_sources']['continuation_verifier'],
                auditor=release['execution_sources']['frozen_fingerprint'],report=ref(directory/'INDEPENDENT_CANARY_AUDIT.json'),
                design_freeze=ref(ROOT/'DESIGN_FREEZE.json'),config=original_spec['documents']['config'],
                execution_sources=release['execution_sources'],**families)
            write_once(directory/'CANARY_INPUT_SPEC.json',spec)
            candidate=continuation_authority.prepare_canary(spec)  # Real second independent CPU audit; never a PASS factory.
            original._observed.stable();worker.verify_release(release_path,payload['release_sha256']);scheduler(job)
            write_once(directory/'CANARY_RESUME_PASS.json',candidate)
            published_spec=dict(spec,published=ref(directory/'CANARY_RESUME_PASS.json'))
            write_once(directory/'CANARY_SPEC.json',published_spec)
            continuation_authority.verify_canary(published_spec).stable()
            receipt=dict(status='ACTUAL_CANARY_PUBLISHED_AND_INDEPENDENTLY_REVERIFIED',job_id=job,
                release=ref(release_path),source_files=84,execution_source_keys=sorted(KEYS),
                publisher_sha256=payload['publisher_sha256'],deployment_intake_sha256=payload['deployment_intake_sha256'],
                files={p.name:ref(p) for p in sorted(directory.iterdir()) if p.is_file()},completed_at_utc=utc(),
                model_calls=0,optimizer_calls=0,main_authorized=False,
                next_action='Separate trained DEV2 and risk-gated continuation controls remain required',
                preservation='All original evidence/negative outcomes retained; no source, tensor, checkpoint or raw deletion')
            write_once(directory/'PUBLICATION_RECEIPT.json',receipt)
        except BaseException as exc:
            write_once(directory/'ERROR.json',dict(status='HOLD_CANARY_PUBLICATION_NO_RETRY',job_id=job,
                error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),model_calls=0,
                recovery='Retain all files; do not overwrite or rerun this publication directory'))
            return remote_manifest(job,success=False)
    return remote_manifest(job,success=True)


def remote_manifest(job,success=None):
    directory=publication_directory(job);require(directory.resolve(strict=True)==directory,'noncanonical publication directory')
    files={}
    for path in directory.iterdir():
        require(path.name in FILES,'unknown publication artifact');ordinary(path)
        files[path.name]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    completed='PUBLICATION_RECEIPT.json' in files and 'ERROR.json' not in files
    return dict(status='CANARY_PUBLICATION_COMPLETE' if completed else 'HOLD_OR_INCOMPLETE_PUBLICATION_RETAINED',
                job_id=job,files=files,success=completed and success is not False)


def local(args):
    require(re.fullmatch(r'[1-9][0-9]*',args.job_id),'explicit numeric first-two job ID required')
    require(re.fullmatch(r'[0-9a-f]{64}',args.release_sha256),'explicit release SHA256 required')
    release_path=Path(args.release_path)
    require(release_path.is_absolute() and '..' not in release_path.parts and release_path.is_relative_to(ROOT/'execution_releases'),
            'release outside fixed execution_releases root')
    deployment=read(args.deployment_intake)
    require(len(deployment['sources'])==84 and deployment['status']=='IMMUTABLE_ORIGINAL_TRAIN_CODE_DEPLOYED_NO_MODEL_CALLS',
            'wrong independent deployment intake')
    code=Path(__file__).read_bytes();publisher_sha=sha_bytes(code)
    payload=dict(job_id=args.job_id,release_path=str(release_path),release_sha256=args.release_sha256,
        deployment=deployment,deployment_intake_sha256=sha(args.deployment_intake),publisher_sha256=publisher_sha)
    if args.mode=='plan':
        print(json.dumps(dict(status='LOCAL_PREPARATION_ONLY_NO_SSH_NO_PUBLICATION',job_id=args.job_id,
            release_path=str(release_path),release_sha256=args.release_sha256,publisher_sha256=publisher_sha,
            source_files=84,execution_source_keys=sorted(KEYS),remote_output=str(publication_directory(args.job_id))),indent=2));return 0
    require(args.ssh_host and re.fullmatch(r'[A-Za-z0-9_.@:-]+',args.ssh_host) and not args.ssh_host.startswith('-'),'explicit SSH host required')
    require(args.local_dir is not None,'explicit local download directory required')
    destination=args.local_dir.resolve();require(not destination.exists(),'local receipt directory already exists; choose a new recovery directory')
    destination.mkdir(parents=True)
    # Preserve exact executed stdin bytes even if SSH fails before any manifest.
    source_copy=destination/'PUBLISHER_SOURCE.py'
    write_once(source_copy,code)
    require(sha(source_copy)==publisher_sha,'local publisher source copy differs')
    encoded=base64.b64encode(canonical(payload)).decode()
    bootstrap=('import hashlib,sys; s=sys.stdin.buffer.read(); '
               f'assert hashlib.sha256(s).hexdigest()=={publisher_sha!r}; '
               "exec(compile(s,'<source-pinned-canary-publisher>','exec'))")
    remote_command=shlex.join(['env','-u','PYTHONPATH','-u','PYTHONHOME','-u','PYTORCH_CUDA_ALLOC_CONF',
        'CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONNOUSERSITE=1',
        'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','OMP_NUM_THREADS=1',PYTHON,'-B','-c',bootstrap,
        '--remote-worker' if args.mode=='publish' else '--remote-manifest',encoded])
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',args.ssh_host,remote_command]
    with (destination/'REMOTE_STDOUT.log').open('xb') as out,(destination/'REMOTE_STDERR.log').open('xb') as err:
        response=subprocess.run(command,input=code,stdout=out,stderr=err)
    output=(destination/'REMOTE_STDOUT.log').read_bytes()
    lines=[line for line in output.splitlines() if line.startswith(b'SATDOSE_PUBLISH_MANIFEST=')]
    require(len(lines)==1,'no unique publication manifest; retained local stdout/stderr, do not retry publication')
    manifest=json.loads(lines[0].split(b'=',1)[1]);require(manifest['job_id']==args.job_id,'remote manifest job differs')
    write_once(destination/'REMOTE_MANIFEST.json',manifest)
    downloaded={}
    for name,reference in sorted(manifest['files'].items()):
        require(name in FILES and Path(reference['path'])==publication_directory(args.job_id)/name,'download manifest escapes designated artifacts')
        cat=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',args.ssh_host,shlex.join(['cat','--',reference['path']])]
        result=subprocess.run(cat,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        require(len(result.stdout)==reference['bytes'] and sha_bytes(result.stdout)==reference['sha256'],'download exact-byte SHA differs: '+name)
        write_once(destination/name,result.stdout);downloaded[name]=dict(reference,local_path=str(destination/name))
    write_once(destination/'LOCAL_DOWNLOAD_RECEIPT.json',dict(status='EXACT_REMOTE_BYTES_DOWNLOADED',
        publication_success=manifest['success'],remote_returncode=response.returncode,job_id=args.job_id,
        publisher_sha256=publisher_sha,operation_mode=args.mode,
        local_operation_source=dict(path=str(source_copy),sha256=sha(source_copy),bytes=len(code)),
        files=downloaded,completed_at_utc=utc(),
        raw_training_data='Retained in original server paths; this receipt package downloads audit/spec/publication records only'))
    print(json.dumps(dict(status=manifest['status'],local_dir=str(destination),files=len(downloaded)),indent=2))
    return 0 if response.returncode==0 and manifest['success'] else 2


def main():
    if len(sys.argv)==3 and sys.argv[1] in ('--remote-worker','--remote-manifest'):
        payload=json.loads(base64.b64decode(sys.argv[2],validate=True))
        try:result=remote_worker(payload) if sys.argv[1]=='--remote-worker' else remote_manifest(payload['job_id'])
        except BaseException as exc:
            traceback.print_exc()
            result=dict(status='HOLD_BEFORE_PUBLICATION',job_id=payload['job_id'],files={},success=False,error=str(exc))
        print('SATDOSE_PUBLISH_MANIFEST='+json.dumps(result,sort_keys=True));return 0 if result['success'] else 2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('plan','publish','download'),default='plan')
    parser.add_argument('--job-id',required=True);parser.add_argument('--release-path',required=True)
    parser.add_argument('--release-sha256',required=True);parser.add_argument('--deployment-intake',type=Path,default=INTAKE)
    parser.add_argument('--ssh-host');parser.add_argument('--local-dir',type=Path)
    return local(parser.parse_args())


if __name__=='__main__':raise SystemExit(main())
