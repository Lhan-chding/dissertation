#!/usr/bin/env python3
"""Publish actual terminal evidence only after every frozen TRAIN/DEV path closes.

Default: read fixed local intake pins and print a plan, with no network or writes.
--execute: one source-pinned SSH CPU audit/publication, then exact-byte download.
--download: read-only recovery download; never repeat an uncertain publication.
No GPU, scheduler submission, cleanup, altered scientific gate or automatic MAIN
launch exists here. Original TRAIN84/19-key and external4190bc3 remain immutable.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack, contextmanager
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import traceback

PROJECT=Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ROOT=Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
TRAIN=ROOT/'train_code_71018e5';EXTERNAL=ROOT/'external_main_code_4190bc3'
PYTHON='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
TRAIN_COMMIT='71018e5720b593b4a37624a8f6b6b3a618aa3f03'
EXTERNAL_COMMIT='4190bc30da4550f213279bf16372e41b9af03f16'
CANARY=dict(path='execution_releases/canary_job_1740834/CANARY_SPEC.json',
            sha256='44a8db68c655bea987629ea210da1a32e268806f6e0c7b1506d6d77828d4b260')
PUBLICATION=ROOT/'execution_releases/terminal_4190bc3'
ARTIFACTS=PROJECT/'artifacts/sr_satdose_20261112'
LOCAL_INPUTS={
 'train_intake':(ARTIFACTS/'intake/FIRST_TWO_CODE_DEPLOYMENT_71018e5_20261116.json','de42793b8d839e7d72d8eb399d37c0fb9e842d02f838709cde5b1a0e3ebbbeb8'),
 'external_intake':(ARTIFACTS/'intake/EXTERNAL_CODES_DEPLOYMENT_4190bc3_20261116.json','b585eaf9f3d8243c956ff79b152f5afbcb47aabb5ca72d42c663d11f25ce379b'),
 'original_release_record':(ARTIFACTS/'intake/FIRST_TWO_RELEASE_PUBLICATION_71018e5_20261116.json','8f28b06a1015cddceba31a5cf76545e4f1d4762d74686c39fc1dcb3d8382f697'),
 'canary_copy':(ARTIFACTS/'canary_1740834_20261116/CANARY_SPEC.json',CANARY['sha256'])}
KEYS={'first_two_control','training_runtime','training_step','training_state','training_transaction','production',
      'calibration','satdose_core','scorer_bindings','admission_verifier','canary_replay','frozen_fingerprint',
      'storage_ready','evaluation_analysis','canary_controller','continuation_verifier','risk_verifier',
      'trained_dev_control','continuation_control'}
PATHS=tuple(f'{s}/{a}' for s in (11031,11032) for a in ('T0','T1'))
FILES={'CLAIM.json','PUBLISHER_SOURCE.py','PAYLOAD.json','PRECHECK.json','FINAL_RISK.json','AUDIT_SPEC.json',
       'INDEPENDENT_TERMINAL_AUDIT.json','AUTHORITY_INPUT_SPEC.json','TERMINAL_ADMISSION.json',
       'TERMINAL_SPEC.json','PUBLICATION_RECEIPT.json','ERROR.json'}
PREFIX=b'SATDOSE_TERMINAL_MANIFEST='


def require(ok,message):
    if not ok:raise PermissionError('HOLD_TERMINAL_PUBLICATION: '+message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def ordinary(path):
    path=Path(path);require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and
                          not path.is_symlink(),'Noncanonical ordinary file: '+str(path));return path
def sha(path):
    value=hashlib.sha256()
    with ordinary(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):value.update(data)
    return value.hexdigest()
def read(path):
    def pairs(items):
        result=dict(items);require(len(result)==len(items),'Duplicate JSON key');return result
    return json.loads(ordinary(path).read_bytes(),object_pairs_hook=pairs,
        parse_constant=lambda value:require(False,'Nonfinite JSON'))
def reference(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value['path'],str) and
            isinstance(value['sha256'],str) and re.fullmatch('[0-9a-f]{64}',value['sha256']),'Invalid trusted reference')
def bound(value):
    reference(value);relative=Path(value['path'])
    require(not relative.is_absolute() and '..' not in relative.parts,'Reference escapes evidence root')
    path=ROOT/relative;require(sha(path)==value['sha256'],'Bound evidence changed');return path
def ref(path):return dict(path=str(ordinary(path).relative_to(ROOT)),sha256=sha(path))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_once(path,value):
    data=value if isinstance(value,bytes) else canonical(value)
    with Path(path).open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    fsync(Path(path).parent)
def fsync(directory):
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def inventory(directory):
    require(directory.resolve(strict=True)==directory and directory.is_dir(),'Noncanonical source directory')
    out={}
    for path in directory.rglob('*'):
        require(path.resolve(strict=True)==path and not path.is_symlink(),'Source symlink/noncanonical entry')
        if path.is_file():out[str(path.relative_to(directory))]=sha(path)
        else:require(path.is_dir(),'Nonordinary source entry')
    return out


def _pin_metadata(payload):
    train=payload['train_intake'];external=payload['external_intake'];record=payload['original_release_record']
    original=record['release'];canary=payload['canary_copy']
    require(train.get('status')=='IMMUTABLE_ORIGINAL_TRAIN_CODE_DEPLOYED_NO_MODEL_CALLS' and
            train.get('git_commit')==TRAIN_COMMIT and train.get('release_root')==str(TRAIN) and len(train['sources'])==84,
            'Original TRAIN deployment intake differs')
    require(external.get('status')=='EXTERNAL_CODE_DEPLOYED_NO_TERMINAL_OR_MAIN_ADMISSION' and
            external.get('git_commit')==EXTERNAL_COMMIT and external.get('original_train_root')==str(TRAIN) and
            external.get('original_train_inventory_unchanged') is True and
            external['releases']['main']['root']==str(EXTERNAL) and set(external['releases']['main']['sources'])==
            {'execution/main_control.py','execution/terminal_authority.py','execution/terminal_audit.py','scheduling/run_main.sh'},
            'External4190bc3 MAIN deployment intake differs')
    require(record['path']==str(ROOT/'execution_releases/FIRST_TWO_RELEASE_71018e5.json') and
            re.fullmatch('[0-9a-f]{64}',record['sha256']) and original.get('schema')=='sr-satdose-first-two-release-v1' and
            original.get('status')=='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY' and
            original.get('permission')=='FOUR_PATHS_MAX_STEP_2_NO_STEP3' and original.get('git_commit')==TRAIN_COMMIT and
            original.get('evidence_root')==str(ROOT) and original.get('release_root')==str(TRAIN) and
            original['sources']==train['sources'] and set(original['execution_sources'])==KEYS,
            'Original published TRAIN source/permission identity differs')
    require(payload['canary_reference']==CANARY and canary['root']==str(ROOT) and
            canary['execution_sources']==original['execution_sources'],'Published actual canary lineage differs')
    for name,reference_value in original['execution_sources'].items():
        reference(reference_value);path=Path(reference_value['path'])
        require(path.is_relative_to(TRAIN) and '..' not in path.parts and
                original['sources'].get(str(path.relative_to(TRAIN)))==reference_value['sha256'],
                'Original execution source pin differs: '+name)
    return original


def local_payload(args):
    payload={};pins={}
    for name in LOCAL_INPUTS:
        path=ordinary(getattr(args,name));digest=getattr(args,name+'_sha256')
        require(re.fullmatch('[0-9a-f]{64}',digest or '') and sha(path)==digest,'Pinned local input changed: '+name)
        payload[name]=read(path);pins[name]=dict(path=str(path),sha256=digest)
    require(pins['canary_copy']['sha256']==CANARY['sha256'],'Actual canary SHA differs from observed publication')
    source=ordinary(Path(__file__).resolve()).read_bytes()
    payload.update(schema='sr-satdose-terminal-operator-payload-v1',local_input_pins=pins,
                   canary_reference=dict(CANARY),publisher_sha256=sha_bytes(source))
    _pin_metadata(payload);return payload,source


@contextmanager
def idle_lock(name):
    path=ordinary(ROOT/name);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
        before=os.fstat(fd);current=path.stat()
        require((before.st_dev,before.st_ino)==(current.st_dev,current.st_ino),'Existing worker lease replaced')
        yield
        current=path.stat()
        require(not path.is_symlink() and (before.st_dev,before.st_ino)==(current.st_dev,current.st_ino),'Worker lease changed')
    finally:os.close(fd)


def load_source(path,digest,name):
    require(sha(path)==digest and name not in sys.modules,'External source changed/name occupied')
    data=path.read_bytes();require(sha_bytes(data)==digest,'External source changed during read')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:exec(compile(data,str(path),'exec'),module.__dict__)
    except BaseException:sys.modules.pop(name,None);raise
    return module


def bootstrap(payload):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1',
            'CPU-only bytecode-free environment required')
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve(),'Wrong production CPU interpreter')
    require(ROOT.resolve(strict=True)==ROOT and ROOT.is_dir(),'Missing fixed evidence root')
    original=_pin_metadata(payload);record=payload['original_release_record']
    original_ref=dict(path=str(Path(record['path']).relative_to(ROOT)),sha256=record['sha256'])
    require(read(bound(original_ref))==original and inventory(TRAIN)==original['sources'],
            'Actual immutable84 source/release differs from independent local intake')
    external=payload['external_intake']['releases']['main']
    require(inventory(EXTERNAL)==external['sources'],'Actual immutable external4190bc3 inventory differs')
    require(read(bound(CANARY))==payload['canary_copy'],'Actual canary bytes/object differs from published copy')
    admission=read(bound(original['admission']))
    require(admission.get('prefix_measurement_passed') is False and admission.get('prefix_gate_decision')=='PASS_USER_OVERRIDE',
            'Original PREFIX measurement failure/authorized exception was rewritten')
    require(not os.path.lexists(ROOT/'main_control'),'MAIN attempt exists before first terminal publication')
    for value in original['execution_sources'].values():require(sha(Path(value['path']))==value['sha256'],'Original execution source changed')
    authority_ref=dict(path=str(EXTERNAL/'execution/terminal_authority.py'),sha256=external['sources']['execution/terminal_authority.py'])
    auditor_ref=dict(path=str(EXTERNAL/'execution/terminal_audit.py'),sha256=external['sources']['execution/terminal_audit.py'])
    authority=load_source(Path(authority_ref['path']),authority_ref['sha256'],'_satdose_actual_terminal_authority')
    modules=authority.import_original(TRAIN)
    auditor=load_source(Path(auditor_ref['path']),auditor_ref['sha256'],'_satdose_actual_terminal_auditor')
    import torch
    torch.set_num_threads(1)
    return dict(original=original,original_ref=original_ref,authority=authority,auditor=auditor,
                authority_ref=authority_ref,auditor_ref=auditor_ref,modules=modules,external=external)


def terminal_population(ctx):
    modules=ctx['modules'];control=modules['execution.continuation_control'] if 'execution.continuation_control' in modules else sys.modules['execution.continuation_control']
    control._hold_free();trusted={};endpoints={};chunks={}
    for key in PATHS:
        start,arm=key.split('/');directory=ROOT/f'training_transactions/start-{start}/{arm}'
        require(directory.resolve(strict=True)==directory and directory.is_dir() and
                {p.name for p in directory.iterdir()}=={f'step-{n:03d}' for n in range(1,25)},
                'Exactly24 actual committed steps per path required')
        trusted[key]=[ref(directory/f'step-{n:03d}/COMMIT.json') for n in range(1,25)]
        for step in (2,8,16):
            result=read(ROOT/f'trained_dev/start-{start}/{arm}/step-{step:03d}/RESULT.json')
            require(result.get('status')=='TRAINED_DEV_ENDPOINT_COMPLETE' and result.get('responses')==108 and
                    result.get('global_scientific_hold') is False,'Actual trained DEV is incomplete/HOLD')
            endpoints[f'{key}/{step}']=dict(receipt=result['receipt'],binding=result['endpoint_binding'])
        for a,b in ((3,8),(9,16),(17,24)):
            chunk=f'chunk-{a:03d}-{b:03d}';path=ROOT/f'continuation_control/start-{start}/{arm}/{chunk}/RESULT.json'
            result=read(path)
            require(result.get('status')=='CONTINUATION_CHUNK_COMPLETE' and result.get('main_authorized') is False,
                    'Actual continuation chunk incomplete/foreign')
            chunks[f'{key}/{chunk}']=ref(path)
    for start in (11031,11032):
        result=read(ROOT/f'start_dev/start-{start}/RESULT.json')
        require(result.get('status')=='START_DEV_ENDPOINT_COMPLETE' and result.get('responses')==108,'Actual START DEV incomplete')
        endpoints[f'{start}/START/0']=dict(receipt=result['receipt'],binding=result['endpoint_binding'])
    control._cursor(trusted);control._completed_attempts(trusted,endpoints,chunks)
    original=ctx['original'];canary=read(bound(CANARY));freeze=read(ROOT/'DESIGN_FREEZE.json')
    require(sha(ROOT/'DESIGN_FREEZE.json')==original['design_freeze_sha256'],'Actual design freeze changed')
    inputs=dict(config_ref=freeze['artifacts']['config'],freeze_ref=canary['design_freeze'],
        clarification_ref=original['risk_clarification'],trusted_commits=trusted,dev_endpoints=endpoints,
        expected_source_sha256=original['execution_sources']['risk_verifier']['sha256'])
    return control,inputs,chunks


def validate_publication(directory,files):
    """Completeness is parsed and hash-bound, never inferred from file existence.

    Also used after download, with the same remote references but local bytes.
    It validates the publication envelope, not a substitute third tensor audit.
    """
    require(set(files)==FILES-{'ERROR.json'},'Missing/extra successful publication artifacts')
    for name,value in files.items():
        path=directory/name
        require(Path(value['path'])==PUBLICATION/name and sha(path)==value['sha256'] and path.stat().st_size==value['bytes'],
                'Publication artifact changed or is incomplete: '+name)
    def expected(name):return dict(path=str((PUBLICATION/name).relative_to(ROOT)),sha256=files[name]['sha256'])
    receipt=read(directory/'PUBLICATION_RECEIPT.json')
    require(receipt.get('status')=='ACTUAL_TERMINAL_PUBLISHED_AND_INDEPENDENTLY_REVERIFIED' and
            receipt.get('permission')=='PUBLISHED_TERMINAL_AUTHORITY_ONLY_NO_JOB_SUBMISSION' and
            type(receipt.get('independent_terminal_audit_runs')) is int and receipt['independent_terminal_audit_runs']==2 and
            all(type(receipt.get(k)) is int and receipt[k]==0 for k in ('model_calls','optimizer_calls')),
            'Incomplete/foreign publication receipt')
    require(receipt.get('files')=={name:expected(name) for name in files if name!='PUBLICATION_RECEIPT.json'} and
            receipt.get('terminal_spec')==expected('TERMINAL_SPEC.json'),'Receipt file references differ from actual publication')
    payload=read(directory/'PAYLOAD.json');claim=read(directory/'CLAIM.json')
    require(receipt.get('publisher_sha256')==payload.get('publisher_sha256')==claim.get('publisher_sha256')==
            files['PUBLISHER_SOURCE.py']['sha256'],'Publication source/payload/claim identity differs')
    spec=read(directory/'TERMINAL_SPEC.json');candidate=read(directory/'TERMINAL_ADMISSION.json')
    require(spec.get('published')==expected('TERMINAL_ADMISSION.json') and
            spec.get('report')==expected('INDEPENDENT_TERMINAL_AUDIT.json') and spec.get('audit_spec')==expected('AUDIT_SPEC.json') and
            candidate.get('status')=='TRAINING_TERMINAL_VERIFIED_MAIN_ALLOWED' and
            candidate.get('permission')=='FIXED_MAIN_MANIFEST_ONLY_NO_TRAINING',
            'Published authority envelope incomplete or foreign')
    return receipt


def manifest(success=None):
    require(PUBLICATION.resolve(strict=True)==PUBLICATION and PUBLICATION.is_dir(),'Noncanonical terminal publication')
    files={}
    for path in PUBLICATION.iterdir():
        require(path.name in FILES,'Unknown terminal publication artifact');ordinary(path)
        files[path.name]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    complete=False;validation_error=None
    try:validate_publication(PUBLICATION,files);complete=True
    except Exception as exc:validation_error=dict(error_type=type(exc).__name__,error=str(exc))
    complete=complete and success is not False
    return dict(schema='sr-satdose-terminal-operator-manifest-v1',status=('ACTUAL_TERMINAL_PUBLICATION_COMPLETE' if complete else
        'HOLD_OR_UNKNOWN_TERMINAL_PUBLICATION_RETAINED'),publication=str(PUBLICATION),files=files,success=complete,
        completion_validation_error=validation_error)


def _remote_worker_once(payload,source,attempt):
    require(sha_bytes(source)==payload['publisher_sha256'],'Publisher executed bytes differ')
    require(not os.path.lexists(PUBLICATION),'Existing terminal publication attempt: no automatic retry or overwrite')
    ctx=bootstrap(payload)
    with ExitStack() as stack:
        # Hold shared leases across both auditors and publication. Their nested
        # shared leases are compatible; every model worker requires exclusive.
        for name in ('DIAGNOSTIC_WORKER.lock','FIRST_TWO_LAUNCHER.lock'):stack.enter_context(idle_lock(name))
        control,inputs,chunks=terminal_population(ctx)
        require(not os.path.lexists(PUBLICATION),'Concurrent terminal publication attempt')
        require(PUBLICATION.parent.resolve(strict=True)==PUBLICATION.parent,'Missing canonical releases directory')
        PUBLICATION.mkdir();attempt['created']=True;fsync(PUBLICATION.parent)
        try:
            write_once(PUBLICATION/'PUBLISHER_SOURCE.py',source);write_once(PUBLICATION/'PAYLOAD.json',payload)
            write_once(PUBLICATION/'CLAIM.json',dict(status='ATTEMPTED_UNKNOWN_UNTIL_PUBLICATION_COMPLETE',
                started_at_utc=utc(),publisher_sha256=payload['publisher_sha256'],original_train_release=ctx['original_ref'],
                canary=CANARY,model_calls=0,optimizer_calls=0))
            write_once(PUBLICATION/'PRECHECK.json',dict(status='ACTUAL_POPULATIONS_READY_FOR_CPU_AUDIT_NOT_PASS',
                original_train_release=ctx['original_ref'],canary=CANARY,commits=sum(map(len,inputs['trusted_commits'].values())),
                dev_endpoints=len(inputs['dev_endpoints']),continuation_chunks=len(chunks),source_files=84,execution_source_keys=sorted(KEYS)))
            decision=ctx['modules']['analysis.risk'].assess(ROOT,**inputs)
            write_once(PUBLICATION/'FINAL_RISK.json',decision)
            require(decision.get('status')=='NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED' and decision.get('global_hold') is False and
                    decision.get('hold_reasons')==[] and decision.get('pending_required_dev')==[],
                    'Actual final risk is HOLD or incomplete')
            audit_spec=dict(schema='sr-satdose-terminal-audit-spec-v1',root=str(ROOT),auditor=ctx['auditor_ref'],
                original_train_release=ctx['original_ref'],admission_spec=ctx['original']['admission_spec'],admission=ctx['original']['admission'],
                canary_spec=CANARY,trusted_commits=inputs['trusted_commits'],dev_endpoints=inputs['dev_endpoints'],
                continuation_results=chunks,final_risk=ref(PUBLICATION/'FINAL_RISK.json'))
            write_once(PUBLICATION/'AUDIT_SPEC.json',audit_spec)
            report=ctx['auditor'].audit(audit_spec)
            write_once(PUBLICATION/'INDEPENDENT_TERMINAL_AUDIT.json',report)  # Preserve any negative result unchanged.
            authority_spec=dict(schema='sr-satdose-terminal-authority-spec-v1',root=str(ROOT),authority=ctx['authority_ref'],
                auditor=ctx['auditor_ref'],external_root=str(EXTERNAL),external_sources=ctx['external']['sources'],
                original_train_release=ctx['original_ref'],audit_spec=ref(PUBLICATION/'AUDIT_SPEC.json'),
                report=ref(PUBLICATION/'INDEPENDENT_TERMINAL_AUDIT.json'))
            write_once(PUBLICATION/'AUTHORITY_INPUT_SPEC.json',authority_spec)
            candidate=ctx['authority'].prepare(authority_spec)  # Actual second independent audit, not a manufactured PASS.
            require(candidate.get('status')=='TRAINING_TERMINAL_VERIFIED_MAIN_ALLOWED' and
                    candidate.get('permission')=='FIXED_MAIN_MANIFEST_ONLY_NO_TRAINING','Terminal authority did not admit fixed MAIN')
            control._hold_free();control._cursor(inputs['trusted_commits'])
            require(inventory(TRAIN)==ctx['original']['sources'] and inventory(EXTERNAL)==ctx['external']['sources'],
                    'Immutable source inventory changed before publication')
            write_once(PUBLICATION/'TERMINAL_ADMISSION.json',candidate)
            published=dict(authority_spec,published=ref(PUBLICATION/'TERMINAL_ADMISSION.json'))
            write_once(PUBLICATION/'TERMINAL_SPEC.json',published)
            ctx['authority'].verify(published).stable()
            write_once(PUBLICATION/'PUBLICATION_RECEIPT.json',dict(status='ACTUAL_TERMINAL_PUBLISHED_AND_INDEPENDENTLY_REVERIFIED',
                permission='PUBLISHED_TERMINAL_AUTHORITY_ONLY_NO_JOB_SUBMISSION',publisher_sha256=payload['publisher_sha256'],
                original_train_release=ctx['original_ref'],canary=CANARY,terminal_spec=ref(PUBLICATION/'TERMINAL_SPEC.json'),
                files={p.name:ref(p) for p in sorted(PUBLICATION.iterdir())},completed_at_utc=utc(),
                independent_terminal_audit_runs=2,model_calls=0,optimizer_calls=0,
                next_action='A separate source-bound fixed MAIN release and scheduler action remain necessary',
                consumption_rule='Only accept this complete receipt with no ERROR; intermediate admission/spec files alone are not a successful operator publication',
                preservation='No original source, state, raw output or negative evidence modified or deleted'))
        except BaseException as exc:
            write_once(PUBLICATION/'ERROR.json',dict(status='HOLD_TERMINAL_PUBLICATION_NO_RETRY',error_type=type(exc).__name__,
                error=str(exc),traceback=traceback.format_exc(),model_calls=0,
                recovery='Retain all artifacts and use download-only recovery; no automatic publication retry'))
            return manifest(success=False)
    return manifest(success=True)


def remote_worker(payload,source):
    attempt={}
    try:return _remote_worker_once(payload,source,attempt)
    except BaseException as exc:
        # Includes shared-lease exit failures after the last authority check.
        # Never mark or touch a pre-existing/concurrent publisher's directory.
        if attempt.get('created'):
            if not os.path.lexists(PUBLICATION/'ERROR.json'):
                write_once(PUBLICATION/'ERROR.json',dict(status='HOLD_TERMINAL_PUBLICATION_NO_RETRY',
                    error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),model_calls=0,
                    recovery='Publication is invalid/UNKNOWN even if candidate artifacts exist; download only'))
            return manifest(success=False)
        raise


def _command(host,source_sha,payload,download=False):
    require(isinstance(host,str) and re.fullmatch('[A-Za-z0-9_.@:-]+',host) and not host.startswith('-'),'Explicit safe SSH host required')
    encoded=base64.b64encode(canonical(payload)).decode()
    bootstrap_code=("import hashlib,sys; s=sys.stdin.buffer.read(); "
        f"assert hashlib.sha256(s).hexdigest()=={source_sha!r}; "
        "g={'__name__':'__main__','__file__':'<pinned-terminal-publisher>','_PINNED_SOURCE_BYTES':s}; "
        "exec(compile(s,'<pinned-terminal-publisher>','exec'),g)")
    remote=shlex.join(['env','-u','PYTHONPATH','-u','PYTHONHOME','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
        'PYTHONNOUSERSITE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','OMP_NUM_THREADS=1',PYTHON,'-I','-B','-c',bootstrap_code,
        '--remote-download' if download else '--remote-worker',encoded])
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',host,remote]


def _download(destination,host,remote):
    require(remote.get('schema')=='sr-satdose-terminal-operator-manifest-v1' and remote.get('publication')==str(PUBLICATION) and
            isinstance(remote.get('files'),dict) and type(remote.get('success')) is bool,'Foreign/malformed remote manifest')
    downloaded={}
    for name,value in sorted(remote['files'].items()):
        require(name in FILES and set(value)=={'path','sha256','bytes'} and Path(value['path'])==PUBLICATION/name and
                type(value['bytes']) is int and value['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',value['sha256']),
                'Download escapes fixed publication artifact')
        command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',host,shlex.join(['cat','--',value['path']])]
        with (destination/(name+'.DOWNLOAD_STDOUT')).open('xb') as out,(destination/(name+'.DOWNLOAD_STDERR')).open('xb') as err:
            response=subprocess.run(command,stdout=out,stderr=err)
        data=(destination/(name+'.DOWNLOAD_STDOUT')).read_bytes()
        require(response.returncode==0 and len(data)==value['bytes'] and sha_bytes(data)==value['sha256'],
                'Failed or nonidentical artifact download: '+name)
        write_once(destination/name,data);downloaded[name]=dict(value,local_path=str(destination/name))
    return downloaded


def local(args):
    payload,source=local_payload(args);mode='publish' if args.execute else 'download' if args.download else 'plan'
    if mode=='plan':
        result=dict(status='LOCAL_TERMINAL_PLAN_ONLY_NO_NETWORK_NO_PUBLICATION',publisher_sha256=payload['publisher_sha256'],
            inputs=payload['local_input_pins'],original_train_release_sha256=payload['original_release_record']['sha256'],
            canary=CANARY,publication=str(PUBLICATION),required_actual_counts=dict(COMMIT=96,TRAIN=12288,DEV_endpoints=14,DEV=1512,chunks=12),
            actual_terminal_state='NOT_OBSERVED_BY_LOCAL_PLAN',model_calls=0)
        print(json.dumps(result,indent=2));return 0
    require(args.local_dir is not None,'Explicit fresh local-dir required for retained source/logs')
    command=_command(args.ssh_host,payload['publisher_sha256'],payload,download=mode=='download')
    destination=args.local_dir.absolute();require(not os.path.lexists(destination),'Existing local operator attempt; never retry or overwrite')
    destination.mkdir(parents=True);require(destination.resolve(strict=True)==destination,'Noncanonical local receipt directory')
    fsync(destination.parent)
    write_once(destination/'OPERATOR_SOURCE.py',source);write_once(destination/'LOCAL_PAYLOAD.json',payload)
    write_once(destination/'LOCAL_INTENT.json',dict(status='UNKNOWN_UNTIL_EXACT_DOWNLOAD',mode=mode,command=command,
        publisher_sha256=payload['publisher_sha256'],created_at_utc=utc(),publication=str(PUBLICATION)))
    try:
        with (destination/'REMOTE_STDOUT.log').open('xb') as out,(destination/'REMOTE_STDERR.log').open('xb') as err:
            response=subprocess.run(command,input=source,stdout=out,stderr=err)
        output=(destination/'REMOTE_STDOUT.log').read_bytes();marked=[line for line in output.splitlines() if line.startswith(PREFIX)]
        require(len(marked)==1,'No unique remote manifest; retain UNKNOWN and use download-only recovery')
        remote=json.loads(marked[0][len(PREFIX):]);write_once(destination/'REMOTE_MANIFEST.json',remote)
        downloaded=_download(destination,args.ssh_host,remote)
        if remote['success']:validate_publication(destination,remote['files'])
        write_once(destination/'LOCAL_DOWNLOAD_RECEIPT.json',dict(status='EXACT_TERMINAL_ARTIFACT_BYTES_DOWNLOADED',mode=mode,
            publication_success=remote['success'],remote_returncode=response.returncode,files=downloaded,
            publisher_sha256=payload['publisher_sha256'],completed_at_utc=utc(),
            evidence_boundary='Only authentic downloaded terminal artifacts; no GPU, scheduler or MAIN generation performed'))
        print(json.dumps(dict(status=remote['status'],publication_success=remote['success'],local_dir=str(destination)),indent=2))
        return 0 if response.returncode==0 and remote['success'] else 2
    except BaseException as exc:
        write_once(destination/'LOCAL_ERROR.json',dict(status='HOLD_OR_UNKNOWN_OPERATOR_RETAINED_NO_RETRY',mode=mode,
            error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),publication=str(PUBLICATION)))
        raise


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if len(argv)==2 and argv[0] in ('--remote-worker','--remote-download'):
        payload=json.loads(base64.b64decode(argv[1],validate=True));source=globals().get('_PINNED_SOURCE_BYTES')
        try:
            require(isinstance(source,bytes) and sha_bytes(source)==payload['publisher_sha256'],'Remote publisher source pin missing')
            result=remote_worker(payload,source) if argv[0]=='--remote-worker' else manifest()
        except BaseException as exc:
            traceback.print_exc();result=dict(schema='sr-satdose-terminal-operator-manifest-v1',status='HOLD_BEFORE_TERMINAL_PUBLICATION',
                publication=str(PUBLICATION),files={},success=False,error_type=type(exc).__name__,error=str(exc))
        print(PREFIX.decode()+json.dumps(result,sort_keys=True));return 0 if result['success'] else 2
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true');mode.add_argument('--download',action='store_true')
    parser.add_argument('--ssh-host');parser.add_argument('--local-dir',type=Path)
    for name,(path,digest) in LOCAL_INPUTS.items():
        parser.add_argument('--'+name.replace('_','-'),type=Path,default=path)
        parser.add_argument('--'+name.replace('_','-')+'-sha256',default=digest)
    return local(parser.parse_args(argv))


if __name__=='__main__':raise SystemExit(main())
