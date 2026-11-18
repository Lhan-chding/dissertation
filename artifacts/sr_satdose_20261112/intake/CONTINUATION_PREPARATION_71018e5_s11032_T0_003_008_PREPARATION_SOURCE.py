#!/usr/bin/env python3
"""Prepare one fixed SATDOSE continuation release; never submit a GPU job.

Without --execute this checks local published pins and prints the intended
scope. --execute performs the explicitly selected, write-once server CPU
preparation. An existing preparation intent, release, budget or actual worker
attempt is not retried. Inspect its existing evidence after an interruption.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

PROJECT=Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ROOT='/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1'
CODE=ROOT+'/train_code_71018e5'
PYTHON='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
TAG='71018e5'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def load(path):
    def pairs(items):
        result=dict(items);require(len(result)==len(items),'Duplicate JSON key');return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda value:require(False,'Nonfinite JSON '+value))


def ref(value):
    require(isinstance(value,dict) and isinstance(value.get('path'),str) and
            re.fullmatch('[0-9a-f]{64}',value.get('sha256','')),'Malformed local published reference')
    p=Path(value['path'])
    if p.is_absolute():p=p.relative_to(ROOT)
    require('..' not in p.parts,'Reference escapes evidence root')
    return dict(path=str(p),sha256=value['sha256'])


def write_once(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())


REMOTE=r'''
import sys,json,pathlib,hashlib,os,fcntl,copy,datetime,contextlib
d=json.load(sys.stdin);root=pathlib.Path(d['root']);code=pathlib.Path(d['code'])
def need(ok,message):
    if not ok:raise PermissionError('HOLD_PREPARATION: '+message)
def sha(path):
    need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Noncanonical file')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def bound(reference):
    p=pathlib.Path(reference['path']);need(not p.is_absolute() and '..' not in p.parts,'Reference escapes root')
    p=root/p;need(sha(p)==reference['sha256'],'Externally pinned reference changed: '+str(p));return p
def reference(path):return dict(path=str(path.relative_to(root)),sha256=sha(path))
def read(reference):return json.loads(bound(reference).read_text())
def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def publish(path,value):
    b=canonical(value)
    with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return reference(path)
@contextlib.contextmanager
def idle(name):
    path=root/name;need(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Missing existing worker lease')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
        stat=os.fstat(fd);need((stat.st_dev,stat.st_ino)==(path.stat().st_dev,path.stat().st_ino),'Replaced worker lease')
        yield
        need(not path.is_symlink() and (stat.st_dev,stat.st_ino)==(path.stat().st_dev,path.stat().st_ino),'Worker lease changed')
    finally:os.close(fd)
need(root.resolve(strict=True)==root and code.resolve(strict=True)==code and code.parent==root,'Wrong canonical roots')
actual={}
for p in code.rglob('*'):
    need(not p.is_symlink() and p.resolve(strict=True)==p,'Noncanonical source entry')
    if p.is_file():actual[str(p.relative_to(code))]=sha(p)
need(len(actual)==84 and actual==d['sources'] and len(d['execution_sources'])==19,'Frozen 84-file/19-key identity changed')
original=read(d['original_train_release'])
need(original['release_root']==str(code) and original['evidence_root']==str(root) and original['sources']==actual and
     original['execution_sources']==d['execution_sources'] and original['admission_spec']==d['admission_spec'] and
     original['admission']==d['admission'],'Original TRAIN release/local pins differ')
need(original['git_commit']==d['git_commit'],'Original frozen commit differs')
sys.path.insert(0,str(code))
from execution import continuation_control as c, first_two_control as first_two, admission_authority, storage_ready as storage
from runtime.backend import object_sha
need(c.PACKAGE==code and c.ROOT==root and c.FREEZE_SHA==original['design_freeze_sha256'],'Shadowed original control plane')
for name,reference_value in original['execution_sources'].items():
    p=pathlib.Path(reference_value['path']);need(p.is_relative_to(code) and sha(p)==reference_value['sha256'],'Execution source changed')
start,arm,first=d['start'],d['arm'],d['first'];last=c.CHUNKS[first]
need(d['last']==last,'Fixed chunk differs');slot=c._slot(start,arm,first)
prefix=f'CONTINUATION_{d["tag"]}_s{start}_{arm}_{first:03d}_{last:03d}'
directory=root/'execution_releases';need(directory.resolve(strict=True)==directory and directory.is_dir(),'Missing release directory')
paths={key:directory/(prefix+'_'+key+'.json') for key in ('INTENT','BUDGET','RELEASE','RISK','PREFLIGHT','ERROR')}
result={};intent=None
with contextlib.ExitStack() as stack:
    for name in ('DIAGNOSTIC_WORKER.lock','FIRST_TWO_LAUNCHER.lock'):stack.enter_context(idle(name))
    c._hold_free();need(not os.path.lexists(slot),'Existing actual chunk attempt: never retry')
    need(not any(os.path.lexists(p) for p in paths.values()),'Existing preparation evidence: inspect it, never retry automatically')
    original=first_two.verify_release(bound(d['original_train_release']),d['original_train_release']['sha256'])
    canary_spec=read(d['canary_spec'])
    need(canary_spec['root']==str(root) and canary_spec['execution_sources']==original['execution_sources'],
         'Canary binds another original TRAIN identity')
    trusted={};endpoints={};prior={}
    for key in c.PATHS:
        seed,which=key.split('/');parent=root/f'training_transactions/start-{seed}/{which}'
        need(parent.resolve(strict=True)==parent and parent.is_dir(),'Missing actual committed path')
        entries=sorted(parent.iterdir());need(entries and all(p.is_dir() and not p.is_symlink() for p in entries),'Unknown transaction entry')
        trusted[key]=[reference(p/'COMMIT.json') for p in entries]
    for seed in (11031,11032):
        value=json.loads((root/f'start_dev/start-{seed}/RESULT.json').read_text())
        endpoints[f'{seed}/START/0']=dict(receipt=value['receipt'],binding=value['endpoint_binding'])
    for family in ('trained_dev','continuation_control'):
        parent=root/family
        if not parent.exists():continue
        need(parent.resolve(strict=True)==parent and parent.is_dir(),'Noncanonical controller family')
        for sd in parent.iterdir():
            need(sd.name in ('start-11031','start-11032') and sd.is_dir() and not sd.is_symlink(),'Unknown controller start')
            for ad in sd.iterdir():
                need(ad.name in ('T0','T1') and ad.is_dir() and not ad.is_symlink(),'Unknown controller arm')
                for leaf in ad.iterdir():
                    need(leaf.is_dir() and not leaf.is_symlink(),'Unknown controller attempt')
                    result_path=leaf/'RESULT.json';result_ref=reference(result_path);value=read(result_ref)
                    key=f'{sd.name[6:]}/{ad.name}'
                    if family=='trained_dev':
                        need(leaf.name in ('step-002','step-008','step-016'),'Unknown trained DEV slot')
                        endpoints[f'{key}/{int(leaf.name[5:])}']=dict(receipt=value['receipt'],binding=value['endpoint_binding'])
                    else:
                        need(leaf.name in ('chunk-003-008','chunk-009-016','chunk-017-024'),'Unknown chunk slot')
                        prior[f'{key}/{leaf.name}']=result_ref
    c._cursor(trusted);c._barrier(trusted,endpoints,start,arm,first);c._completed_attempts(trusted,endpoints,prior)
    intent=publish(paths['INTENT'],dict(status='CPU_PREPARATION_INTENT_UNKNOWN_UNTIL_PREFLIGHT',created_at_utc=storage._now().isoformat(),
        selection=dict(start_seed=start,arm=arm,first_step=first,last_step=last),original_train_release=d['original_train_release'],
        canary_spec=d['canary_spec'],trusted_commits=trusted,dev_endpoints=endpoints,prior_chunks=prior,
        preparation_script_sha256=d['script_sha256'],model_calls=0,gpu_submission_permission=False))
    try:
        freeze=json.loads((root/'DESIGN_FREEZE.json').read_text());admission_spec=read(original['admission_spec'])
        risk_inputs=dict(config_ref=freeze['artifacts']['config'],freeze_ref=dict(path='DESIGN_FREEZE.json',sha256=c.FREEZE_SHA),
            clarification_ref=original['risk_clarification'],trusted_commits=trusted,dev_endpoints=endpoints,
            expected_source_sha256=original['execution_sources']['risk_verifier']['sha256'])
        authority=admission_authority.ContinuingAdmission(admission_spec,admission=original['admission'],continuation_spec=dict(
            root=str(root),helper=original['execution_sources']['continuation_verifier'],execution_sources=original['execution_sources'],
            canary_spec=canary_spec,risk_inputs=risk_inputs))
        decision=authority.risk_snapshot();need(decision['global_hold'] is False,'Actual scientific HOLD')
        current=read(trusted[f'{start}/{arm}'][-1]);identity=current['identity']
        endpoint=dict(experiment_id=c.EXPERIMENT,start_seed=start,arm=arm,step=first-1,
            policy_sha256=current['policy_sha256'],reference_sha256=current['reference_sha256'])
        authority.verify(root=root,identity=identity,design_freeze_sha256=c.FREEZE_SHA,target_step=first,sampled_endpoint=endpoint)
        # Observed complete checkpoint/audit sizes plus conservative long-response allowance.
        checkpoint_sizes=[];capture_sizes=[];triplet_sizes=[]
        for refs in trusted.values():
            for commit_ref in refs:
                commit_path=bound(commit_ref);marker=read(commit_ref)
                files=list(commit_path.parent.iterdir());need(all(p.is_file() and not p.is_symlink() for p in files),'Nonordinary checkpoint')
                checkpoint_sizes.append(sum(p.stat().st_size for p in files))
                local=marker['transaction']['evidence'];ep=commit_path.parent/local['path']
                need(sha(ep)==local['sha256'],'Committed evidence changed');evidence=json.loads(ep.read_text())
                precommit=read(evidence['precommit_reference']);capture=bound(precommit['production_capture'])
                capture_sizes.append(capture.stat().st_size)
                rollout=read(evidence['rollout_receipt'])
                for complete_ref in rollout['complete_files']:
                    complete=bound(complete_ref);stem=complete.name.removesuffix('.COMPLETE.json')
                    files=[complete.parent/(stem+suffix) for suffix in ('.BEGIN.json','.CAPTURE.json','.COMPLETE.json')]
                    need(all(p.resolve(strict=True)==p and p.is_file() and not p.is_symlink() for p in files),'Incomplete original response triplet')
                    triplet_sizes.append(sum(p.stat().st_size for p in files))
        need(checkpoint_sizes and capture_sizes and triplet_sizes,'No genuine observed storage samples')
        per_response=max(4*max(triplet_sizes),1<<20);count=last-first+1
        calculated=2*(count*(max(checkpoint_sizes)+max(capture_sizes)+128*per_response)+max(checkpoint_sizes))+(1<<30)
        required=max(16<<30,calculated);now=storage._now()
        budget=dict(schema='sr-satdose-storage-budget-v1',scope='CONTINUATION_FIXED_CHUNK_ONLY',experiment_id=c.EXPERIMENT,
            evidence_root=str(root),design_freeze_sha256=c.FREEZE_SHA,required_bytes=required,created_at_utc=now.isoformat(),
            expires_at_utc=(now+datetime.timedelta(days=7)).isoformat(),max_observation_age_seconds=300,
            rationale='One fixed chunk only. Twice observed max checkpoint+production capture and128 response allowances per new step, plus one checkpoint and1GiB metadata; minimum16GiB. Per response=max(4x largest original triplet,1MiB). Engineering allowance, not an upper bound or quota reservation. Live quota/statvfs/fsync remains mandatory before model and every Adam.',
            sizing=dict(observed_checkpoints=len(checkpoint_sizes),observed_responses=len(triplet_sizes),max_checkpoint_bytes=max(checkpoint_sizes),
                max_capture_bytes=max(capture_sizes),max_triplet_bytes=max(triplet_sizes),per_response_allowance_bytes=per_response,
                new_steps=count,new_responses=count*128,calculated_bytes=calculated,minimum_bytes=16<<30))
        before=storage._observe(directory);need(before['effective_available_bytes']>=required,'Insufficient observed shared quota')
        probe=storage._probe(directory);after=storage._observe(directory)
        need(after['effective_available_bytes']>=required,'Insufficient post-fsync shared quota')
        budget_ref=publish(paths['BUDGET'],budget)
        release=dict(schema='sr-satdose-continuation-release-v1',status='AUDITED_FOR_ONE_FIXED_TRAIN_CHUNK',
            permission='ONE_FIXED_CHUNK_NO_MAIN',experiment_id=c.EXPERIMENT,evidence_root=str(root),release_root=str(code),
            design_freeze_sha256=c.FREEZE_SHA,config_sha256=c.CONFIG_SHA,sources=original['sources'],
            original_train_release=d['original_train_release'],admission_spec=original['admission_spec'],admission=original['admission'],
            risk_clarification=original['risk_clarification'],canary_spec=d['canary_spec'],storage_budget=budget_ref,
            trusted_commits=trusted,dev_endpoints=endpoints,prior_chunks=prior,
            selection=dict(start_seed=start,arm=arm,first_step=first,last_step=last))
        release_ref=publish(paths['RELEASE'],release)
        # This is the frozen controller's complete CPU-only entrypoint, no --execute.
        ctx=c.preflight(paths['RELEASE'],release_ref['sha256'],start,arm,first)
        need(ctx['risk']==decision,'Evidence/risk changed between independent preparation and controller preflight')
        final_before=storage._observe(directory);need(final_before['effective_available_bytes']>=required,'Post-preflight quota insufficient')
        final_probe=storage._probe(directory);final_after=storage._observe(directory)
        need(final_after['effective_available_bytes']>=required,'Post-preflight fsync quota insufficient')
        storage._budget(budget_ref,storage._now())
        risk_ref=publish(paths['RISK'],ctx['risk'])
        report=dict(status='CONTINUATION_CPU_PREFLIGHT_COMPLETE_NO_MODEL_CALLS',permission='NO_GPU_SUBMISSION',
            intent=intent,release=release_ref,storage_budget=budget_ref,risk=risk_ref,selection=release['selection'],
            checked_at_utc=storage._now().isoformat(),original_train_release=d['original_train_release'],
            cursor={key:len(refs) for key,refs in trusted.items()},completed_dev_endpoints=len(endpoints),completed_chunks=len(prior),
            observed_quota_before=before,observed_quota_after=after,writability_probe=probe,
            final_quota_before=final_before,final_quota_after=final_after,final_writability_probe=final_probe,
            required_bytes=required,model_calls=0,optimizer_calls=0,gpu_jobs_submitted=0,
            preparation_script_sha256=d['script_sha256'])
        report_ref=publish(paths['PREFLIGHT'],report)
        result=dict(report=report,preflight=report_ref,release_path=str(paths['RELEASE']),release_sha256=release_ref['sha256'])
    except BaseException as exc:
        error=dict(status='CPU_PREPARATION_FAILED_NO_AUTOMATIC_RETRY',intent=intent,error_type=type(exc).__name__,error=str(exc),
            finished_at_utc=storage._now().isoformat(),model_calls=0,optimizer_calls=0,gpu_jobs_submitted=0)
        if not paths['ERROR'].exists():publish(paths['ERROR'],error)
        raise
print(json.dumps(result,sort_keys=True,allow_nan=False))
'''


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed',type=int,choices=(11031,11032),required=True)
    parser.add_argument('--arm',choices=('T0','T1'),required=True)
    parser.add_argument('--first-step',type=int,choices=(3,9,17),required=True)
    parser.add_argument('--canary-spec-path',required=True)
    parser.add_argument('--canary-spec-sha256',required=True)
    parser.add_argument('--first-two-publication',type=Path,
        default=PROJECT/'artifacts/sr_satdose_20261112/intake/FIRST_TWO_RELEASE_PUBLICATION_71018e5_20261116.json')
    parser.add_argument('--execute',action='store_true',help='Run write-once SSH CPU preparation; never submit GPU')
    args=parser.parse_args(argv);art=PROJECT/'artifacts/sr_satdose_20261112/intake'
    publication=load(args.first_two_publication);original_ref=ref(publication)
    dep=load(art/'FIRST_TWO_CODE_DEPLOYMENT_71018e5_20261116.json')
    execution=load(art/'ORIGINAL_TRAIN_EXECUTION_SOURCES_71018e5.json')
    admission=load(art/'FIRST_TWO_ADMISSION_PUBLICATION_71018e5_20261116.json')
    require(dep['release_root']==CODE and dep['git_commit']=='71018e5720b593b4a37624a8f6b6b3a618aa3f03' and
            len(dep['sources'])==84 and len(execution)==19,'Original local immutable inventory differs')
    if 'release' in publication:
        require(publication['release']['sources']==dep['sources'] and publication['release']['execution_sources']==execution,
                'Local original TRAIN release differs from deployed pins')
    last={3:8,9:16,17:24}[args.first_step]
    source_bytes=Path(__file__).read_bytes()
    payload=dict(root=ROOT,code=CODE,tag=TAG,start=args.start_seed,arm=args.arm,first=args.first_step,last=last,
        original_train_release=original_ref,canary_spec=ref(dict(path=args.canary_spec_path,sha256=args.canary_spec_sha256)),
        sources=dep['sources'],execution_sources=execution,admission_spec=admission['spec'],admission=admission['admission'],
        git_commit=dep['git_commit'],script_sha256=hashlib.sha256(source_bytes).hexdigest())
    summary=dict(mode='WRITE_ONCE_CPU_PREPARATION' if args.execute else 'LOCAL_PLAN_ONLY_NO_SSH',
        selection=dict(start_seed=args.start_seed,arm=args.arm,first_step=args.first_step,last_step=last),
        original_train_release=original_ref,canary_spec=payload['canary_spec'],gpu_submission=False)
    if not args.execute:print(json.dumps(summary,indent=2));return 0
    prefix=f'CONTINUATION_PREPARATION_{TAG}_s{args.start_seed}_{args.arm}_{args.first_step:03d}_{last:03d}'
    intent=art/(prefix+'_LOCAL_INTENT.json');receipt=art/(prefix+'_LOCAL_RECEIPT.json')
    source_copy=art/(prefix+'_PREPARATION_SOURCE.py')
    stdout_path=art/(prefix+'_REMOTE_STDOUT.log');stderr_path=art/(prefix+'_REMOTE_STDERR.log')
    require(not any(os.path.lexists(p) for p in (intent,receipt,source_copy,stdout_path,stderr_path)),
            'Existing local preparation attempt; inspect it rather than retry')
    # Preserve the exact operation bytes before any SSH, even on connection failure.
    with source_copy.open('xb') as stream:
        stream.write(source_bytes);stream.flush();os.fsync(stream.fileno())
    require(hashlib.sha256(source_copy.read_bytes()).hexdigest()==payload['script_sha256'],
            'Raw operation source copy differs')
    source_ref=dict(path=str(source_copy),sha256=payload['script_sha256'],bytes=len(source_bytes))
    started=datetime.now(timezone.utc).isoformat()
    write_once(intent,dict(summary,started_at_utc=started,script_sha256=payload['script_sha256'],
        local_operation_source=source_ref,status='LOCAL_INTENT_UNKNOWN_UNTIL_SSH_RECEIPT'))
    command='env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 '+shlex.quote(PYTHON)+' -I -B -c '+shlex.quote(REMOTE)
    try:
        with stdout_path.open('xb') as out,stderr_path.open('xb') as err:
            process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','eee-cluster',command],
                input=json.dumps(payload).encode(),stdout=out,stderr=err)
            out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
    except BaseException as exc:
        write_once(receipt,dict(status='LOCAL_OR_SSH_INVOCATION_FAILED_REMOTE_OUTCOME_UNKNOWN',
            started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),returncode=None,
            stdout=stdout_path.read_bytes().decode(errors='replace') if stdout_path.exists() else '',
            stderr=stderr_path.read_bytes().decode(errors='replace') if stderr_path.exists() else '',
            error=str(exc),error_type=type(exc).__name__,summary=summary,
            stdout_path=str(stdout_path),stderr_path=str(stderr_path),local_operation_source=source_ref,recovery='Inspect existing local/remote records; no automatic retry'))
        raise
    value=dict(status='SSH_CPU_PREPARATION_FAILED' if process.returncode else 'SSH_CPU_PREPARATION_RETURNED_ZERO',
        started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),returncode=process.returncode,
        stdout=stdout_path.read_bytes().decode(errors='replace'),stderr=stderr_path.read_bytes().decode(errors='replace'),summary=summary,
        raw_logs={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                  for k,p in [('stdout',stdout_path),('stderr',stderr_path)]},local_operation_source=source_ref)
    exit_code=process.returncode
    if process.returncode==0:
        try:
            value['publication']=json.loads(value['stdout'])
            value['status']='CPU_PREPARATION_PUBLICATION_RECEIPT_CAPTURED'
        except (ValueError,TypeError) as exc:
            value.update(status='SSH_ZERO_BUT_PUBLICATION_RECEIPT_PARSE_FAILED',parse_error=str(exc),
                recovery='Retain exact stdout/stderr and inspect remote publication; no automatic retry')
            exit_code=2
    write_once(receipt,value)
    print(value['stdout']);print(value['stderr'],file=sys.stderr)
    print('Local receipt:',receipt)
    return exit_code


if __name__=='__main__':raise SystemExit(main())
