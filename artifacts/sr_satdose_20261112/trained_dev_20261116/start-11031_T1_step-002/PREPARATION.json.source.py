#!/usr/bin/env python3
"""Prepare ONE frozen TRAINED_DEV release and run CPU preflight; never submit/execute GPU.

Default: local trust/static checks only. --execute-preparation opts into one SSH
invocation. Existing local intent or any remote publication name blocks retries.
Remote uncertainty requires inspection; use no new tag to bypass an unknown run.
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

WORKSPACE = Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ROOT = '/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1'
CODE = ROOT + '/train_code_71018e5'
PYTHON = '/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
PINS = {
    'FIRST_TWO_CODE_DEPLOYMENT_71018e5_20261116.json': 'de42793b8d839e7d72d8eb399d37c0fb9e842d02f838709cde5b1a0e3ebbbeb8',
    'ORIGINAL_TRAIN_EXECUTION_SOURCES_71018e5.json': '5370b089381609c2e8d5d5ec7555c3dd441babaed242bd66b7cbfad02f679dd7',
    'FIRST_TWO_ADMISSION_PUBLICATION_71018e5_20261116.json': '58610a047a8f26c98879c74302a70c9c44ba72215e77874795c9ba3467378101',
    'TRAIN_RISK_CLARIFICATION_PUBLICATION_20261116.json': '07aa41a7d089ff320e043fd9d00c62f4910398e54874702c243fd7fdd65e5f5b',
}
EXECUTION_KEYS = set('first_two_control training_runtime training_step training_state training_transaction production calibration satdose_core scorer_bindings admission_verifier canary_replay frozen_fingerprint storage_ready evaluation_analysis canary_controller continuation_verifier risk_verifier trained_dev_control continuation_control'.split())

REMOTE = r'''
import sys, os, json, hashlib, re, fcntl
from pathlib import Path
from datetime import datetime, timezone, timedelta
p=json.load(sys.stdin)
ROOT=Path(p['root']); CODE=Path(p['code']); published={}; locks=[]
def require(ok,message):
    if not ok: raise RuntimeError(message)
def digest(path):
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Noncanonical file '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def inventory():
    require(CODE.resolve(strict=True)==CODE and CODE.is_dir(),'Noncanonical original code root')
    out={}
    for f in CODE.rglob('*'):
        require(f.resolve(strict=True)==f and not f.is_symlink(),'Code symlink')
        if f.is_file(): out[str(f.relative_to(CODE))]=digest(f)
        else: require(f.is_dir(),'Nonordinary source entry')
    require(out==p['sources'] and len(out)==84,'Full original source84 mismatch')
def canonical(v): return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def acquire(name,mode):
    path=ROOT/name
    require(path.resolve(strict=True)==path and path.is_file(),'Existing canonical lock required')
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try: fcntl.flock(fd,mode|fcntl.LOCK_NB)
    except BaseException: os.close(fd);raise
    locks.append((fd,path))
def stable_locks():
    for fd,path in locks:
        a=os.fstat(fd);b=path.stat()
        require(not path.is_symlink() and (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'Lease replaced')
def publish(name,value):
    path=pubdir/(prefix+'_'+name+'.json')
    c.tx.write_once(path,value)
    reference=c.ref(path);published[name]=reference
    return reference
try:
    require(ROOT.resolve(strict=True)==ROOT and ROOT.is_dir(),'Noncanonical evidence root')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only CUDA mask missing')
    inventory()                         # Authenticate before importing project code.
    acquire('FIRST_TWO_LAUNCHER.lock',fcntl.LOCK_SH)
    acquire('DIAGNOSTIC_WORKER.lock',fcntl.LOCK_EX)
    sys.path.insert(0,str(CODE))
    from execution import trained_dev_control as c
    from execution import continuation_control
    require(c.PACKAGE==CODE and c.ROOT==ROOT,'Shadowed original namespace')
    original=c.first_two.verify_release(c.bound(p['original_train_release']),p['original_train_release']['sha256'])
    require(original['sources']==p['sources'] and original['execution_sources']==p['execution_sources'],
            'Original TRAIN source84/original19 identity differs')
    require(original['admission_spec']==p['admission_spec'] and original['admission']==p['admission'] and
            original['risk_clarification']==p['risk_clarification'],'Original admission/risk refs differ')
    c._hold_free()
    target=c._slot(p['start_seed'],p['arm'],p['step'])
    require(not os.path.lexists(target),'Target DEV attempted already: no retry')
    pubdir=ROOT/'execution_releases'
    require(pubdir.resolve(strict=True)==pubdir and pubdir.is_dir(),'Publication directory missing/noncanonical')
    prefix='TRAINED_DEV_'+p['tag']+'_start-'+str(p['start_seed'])+'_'+p['arm']+'_step-'+str(p['step']).zfill(3)
    require(not any(pubdir.glob(prefix+'_*')),'Publication tag occupied; inspect existing evidence, no retry')
    endpoint_suffix='_start-'+str(p['start_seed'])+'_'+p['arm']+'_step-'+str(p['step']).zfill(3)
    for old_claim in pubdir.glob('TRAINED_DEV_*'+endpoint_suffix+'_PREPARATION_CLAIM.json'):
        c.sha(old_claim)
        old_result=old_claim.with_name(old_claim.name.removesuffix('_PREPARATION_CLAIM.json')+'_PREPARATION_RESULT.json')
        require(old_result.is_file(),'Prior endpoint preparation remains UNKNOWN; do not bypass with another tag')
        c.sha(old_result)
        require(c.read(old_result).get('status')=='TRAINED_DEV_RELEASE_CPU_PREFLIGHT_AND_STORAGE_OBSERVED',
                'Prior endpoint preparation did not complete')
    freeze=c.read(ROOT/'DESIGN_FREEZE.json')
    canary=c.continuation_authority.verify_canary(c.read(c.bound(p['canary_spec'])))
    require(canary.execution_sources==p['execution_sources'],'Foreign canary identity')
    c.first_two._initial_dev(original,freeze)
    endpoints={}
    for start in (11031,11032):
        old=c.read(c.bound(original['start_dev_results'][str(start)]))
        endpoints[f'{start}/START/0']=dict(receipt=old['receipt'],binding=old['endpoint_binding'])
    # An exact all-four-path barrier: no later, missing, staging or unknown slot.
    transactions=ROOT/'training_transactions'
    require({x.name for x in transactions.iterdir()}=={'TRANSACTION.lock','start-11031','start-11032'},'Unknown transaction start/lock')
    transaction_lock=transactions/'TRANSACTION.lock'
    require(transaction_lock.resolve(strict=True)==transaction_lock and transaction_lock.is_file() and
            not transaction_lock.is_symlink(),'Noncanonical transaction lock')
    commits={}
    for start in (11031,11032):
        sd=transactions/f'start-{start}'
        require({x.name for x in sd.iterdir()}=={'T0','T1'},'Unknown transaction arm')
        for arm in ('T0','T1'):
            key=f'{start}/{arm}';ad=sd/arm
            require({x.name for x in ad.iterdir()}=={f'step-{i:03d}' for i in range(1,p['step']+1)},'Not exact four-path barrier')
            commits[key]=[]
            for i in range(1,p['step']+1):
                reference=c.ref(ad/f'step-{i:03d}'/'COMMIT.json');marker=c.read(c.bound(reference))
                require(marker['step']==i and marker['identity']==canary.identities[key] and
                        marker['identity']['execution_sources_sha256']==c.object_sha(p['execution_sources']),
                        'Commit step/original identity mismatch')
                commits[key].append(reference)
    # Derive only complete existing endpoint references; _attempts audits extra slots.
    devroot=ROOT/'trained_dev'
    if devroot.exists():
        for result_path in devroot.glob('start-*/*/step-*/RESULT.json'):
            result=c.read(result_path)
            start=result.get('start_seed');arm=result.get('arm');step=result.get('step')
            require(type(start) is int and type(step) is int and step<=p['step'] and
                    result_path==c._slot(start,arm,step)/'RESULT.json','Relocated/future DEV result')
            require(result.get('global_scientific_hold') is False,'Prior DEV global scientific HOLD')
            prior=c.verify_release(c.bound(result['release']),result['release']['sha256'])[0]
            require(prior['original_train_release']==p['original_train_release'] and
                    prior['canary_spec']==p['canary_spec'],'Prior DEV lineage changed')
            endpoints[f'{start}/{arm}/{step}']=dict(receipt=result['receipt'],binding=result['endpoint_binding'])
    c._attempts(endpoints)
    chunks={}
    for key in c.PATHS:
        start,arm=key.split('/')
        for first,last in ((3,8),(9,16)):
            if last>p['step']: continue
            path=ROOT/'continuation_control'/f'start-{start}'/arm/f'chunk-{first:03d}-{last:03d}'/'RESULT.json'
            result=c.read(path)
            continuation_control.verify_release(c.bound(result['release']),result['release']['sha256'])
            chunks[f'{key}/{first}-{last}']=c.ref(path)
    release=dict(schema='sr-satdose-trained-dev-release-v1',status='AUDITED_FOR_TRAINED_DEV_ONLY',
        permission='TRAINED_DEV_ONLY_NO_TRAINING_OR_MAIN',experiment_id=c.EXPERIMENT,evidence_root=str(ROOT),
        release_root=str(CODE),design_freeze_sha256=c.FREEZE_SHA,config_sha256=c.CONFIG_SHA,
        sources=p['sources'],original_train_release=p['original_train_release'],admission_spec=p['admission_spec'],
        admission=p['admission'],risk_clarification=p['risk_clarification'],canary_spec=p['canary_spec'],
        trusted_commits=commits,dev_endpoints=endpoints,continuation_results=chunks)
    c._first_two_complete(release,canary);c._continuation_complete(release,canary,p['step'])
    spec=c.read(c.bound(release['admission_spec']))
    require(spec['training_execution_sources_sha256']==c.object_sha(p['execution_sources']) and
            spec['sources']['authority']==p['execution_sources']['admission_verifier'],'Admission source identity differs')
    base=c.admission_authority.FirstTwoStepAdmission(spec,admission=release['admission'])
    decision=c._assess(release,freeze,original,endpoints)
    require(decision['global_hold'] is False and decision['status']=='PENDING_REQUIRED_DEV' and
            f"{p['start_seed']}/{p['arm']}/{p['step']}" in decision['pending_required_dev'] and
            all(k.rsplit('/',1)[-1]==str(p['step']) for k in decision['pending_required_dev']),
            'Risk HOLD, earlier incomplete barrier, or endpoint not due')
    # Measure actual completed DEV outputs. The allowance is engineering headroom,
    # not a claim that future response sizes are bounded or quota is reserved.
    sizes=[];sizing=[];auxiliary=[]
    for key,refs in sorted(endpoints.items()):
        receipt=c.read(c.bound(refs['receipt']));files=receipt['complete_files']
        require(len(files)==108,'Unexpected completed DEV sizing population')
        total=0
        for reference in files:
            complete=c.bound(reference);stem=complete.name.removesuffix('.COMPLETE.json')
            triple=[complete.parent/(stem+suffix) for suffix in ('.BEGIN.json','.CAPTURE.json','.COMPLETE.json')]
            for path in triple: c.sha(path)
            n=sum(path.stat().st_size for path in triple);sizes.append(n);total+=n
        extra=sum(c.bound(receipt[k]).stat().st_size for k in ('raw','accounting'))
        auxiliary.append(extra)
        sizing.append(dict(key=key,receipt=refs['receipt'],responses=108,triplet_bytes=total,raw_accounting_bytes=extra))
    require(len(sizes)>=216,'Missing actual START DEV sizing evidence')
    per_response=max(1<<20,4*max(sizes))
    auxiliary_allowance=4*max(auxiliary)
    metadata_allowance=(64<<20)+4*len(canonical(decision))
    required=108*per_response+auxiliary_allowance+metadata_allowance
    c._hold_free();base._observed.stable();canary.stable();stable_locks();inventory()
    now=datetime.now(timezone.utc)
    budget=dict(schema='sr-satdose-storage-budget-v1',scope='TRAINED_DEV_ONLY',experiment_id=c.EXPERIMENT,
        evidence_root=str(ROOT),design_freeze_sha256=c.FREEZE_SHA,required_bytes=required,
        max_observation_age_seconds=300,created_at_utc=now.isoformat(),
        expires_at_utc=(now+timedelta(hours=p['budget_hours'])).isoformat(),
        planned_responses=108,observed_completed_response_count=len(sizes),observed_max_triplet_bytes=max(sizes),
        per_response_allowance_bytes=per_response,raw_accounting_allowance_bytes=auxiliary_allowance,
        metadata_allowance_bytes=metadata_allowance,observed_endpoints=sizing,
        rationale='4x largest observed DEV triplet, minimum 1 MiB per response; 4x largest observed raw/accounting; 64 MiB plus 4x actual risk JSON. Engineering allowance, not a proven upper bound or quota reservation.')
    # Publication begins only after real predecessor/risk/source checks above.
    publish('PREPARATION_CLAIM',dict(status='PREPARATION_IN_PROGRESS_NO_AUTOMATIC_RETRY',
        permission='NO_MODEL_CALLS_OR_GPU_SUBMISSION',process=c._process_identity(),request=p,
        started_at_utc=now.isoformat()))
    release['storage_budget']=publish('STORAGE_BUDGET',budget)
    risk_ref=publish('OBSERVED_RISK',decision)
    release_ref=publish('RELEASE',release)
    ctx=c.preflight(c.bound(release_ref),release_ref['sha256'],p['start_seed'],p['arm'],p['step'])
    # Actual quota/statvfs and fsynced write probe, after the long CPU preflight.
    # This is a preparation-directory snapshot; controller must check actual slot later.
    c._hold_free();ctx['base']._observed.stable();ctx['canary'].stable();stable_locks();inventory()
    require(not os.path.lexists(target),'Target attempted during preparation')
    storage=c.storage_check(pubdir,release['storage_budget'])
    c._hold_free();stable_locks();inventory()
    result=dict(status='TRAINED_DEV_RELEASE_CPU_PREFLIGHT_AND_STORAGE_OBSERVED',
        permission='NO_MODEL_CALLS_OR_GPU_SUBMISSION',start_seed=p['start_seed'],arm=p['arm'],step=p['step'],
        responses=108,release=release_ref,storage_budget=release['storage_budget'],observed_risk=risk_ref,
        storage=storage,storage_observation_directory=str(pubdir),
        storage_limit='Snapshot, not reservation. Future controller must rerun on actual endpoint directory.',
        original_source_count=84,original_execution_source_count=19,
        committed_updates=sum(map(len,commits.values())),completed_dev_endpoints=len(endpoints),
        pending_required_dev=ctx['risk']['pending_required_dev'],
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    receipt=publish('PREPARATION_RESULT',result)
    response=dict(result=result,receipt=receipt,published=published)
except BaseException as exc:
    response=dict(status='PREPARATION_FAILED_OR_INCOMPLETE_DO_NOT_EXECUTE',error_type=type(exc).__name__,
        error=str(exc),published=published,permission='NO_MODEL_CALLS_OR_GPU_SUBMISSION',
        recovery='Inspect existing publication and process; never automatically retry or bypass an unknown attempt.')
finally:
    for fd,path in reversed(locks): os.close(fd)
print('SATDOSE_PREPARE_JSON='+json.dumps(response,sort_keys=True,allow_nan=False),flush=True)
sys.exit(0 if 'result' in response else 2)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path, digest):
    require(re.fullmatch('[0-9a-f]{64}', digest or ''), 'Explicit reference SHA256 required')
    relative = Path(path)
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == path,
            'Reference must be a normalized path relative to fixed evidence ROOT')
    return dict(path=path, sha256=digest)


def write_bytes_once(path, data):
    require(path.is_absolute() and path.parent.resolve(strict=True) == path.parent, 'Canonical local receipt parent required')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write_once(path, value):
    write_bytes_once(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed', type=int, choices=(11031, 11032), required=True)
    parser.add_argument('--arm', choices=('T0', 'T1'), required=True)
    parser.add_argument('--step', type=int, choices=(2, 8, 16), required=True)
    parser.add_argument('--original-train-release', required=True)
    parser.add_argument('--original-train-sha256', required=True)
    parser.add_argument('--canary-spec', required=True)
    parser.add_argument('--canary-spec-sha256', required=True)
    parser.add_argument('--publication-tag', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--budget-hours', type=int, default=168)
    parser.add_argument('--execute-preparation', action='store_true')
    args = parser.parse_args()
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.publication_tag), 'Unsafe publication tag')
    require(1 <= args.budget_hours <= 168, 'Budget lifetime must be 1..168 hours')
    trust = WORKSPACE/'artifacts/sr_satdose_20261112/intake'
    values = []
    for name, expected in PINS.items():
        path = trust/name
        require(path.resolve(strict=True) == path and sha(path) == expected, 'Local trust receipt changed: '+name)
        values.append(json.loads(path.read_text()))
    deployment, execution, admission, risk = values
    require(deployment['release_root'] == CODE and len(deployment['sources']) == 84, 'Unexpected original release')
    require(set(execution) == EXECUTION_KEYS and len(execution) == 19, 'Unexpected execution keys')
    for value in execution.values():
        relative = str(Path(value['path']).relative_to(CODE))
        require(deployment['sources'][relative] == value['sha256'], 'Execution pin not in original source84')
        require(sha(WORKSPACE/'experiments/sr_satdose_20261112'/relative) == value['sha256'], 'Local execution source changed')
    payload = dict(root=ROOT, code=CODE, sources=deployment['sources'], execution_sources=execution,
        admission_spec=admission['spec'], admission=admission['admission'],
        risk_clarification={k:risk[k] for k in ('path','sha256')},
        original_train_release=reference(args.original_train_release,args.original_train_sha256),
        canary_spec=reference(args.canary_spec,args.canary_spec_sha256),
        start_seed=args.start_seed, arm=args.arm, step=args.step, tag=args.publication_tag,
        budget_hours=args.budget_hours, preparation_script_sha256=sha(Path(__file__).resolve()))
    compile(REMOTE, '<remote-cpu-preparation>', 'exec')
    if not args.execute_preparation:
        print(json.dumps(dict(status='LOCAL_PINS_AND_REMOTE_SYNTAX_VERIFIED_ONLY',
            original_source_count=84, original_execution_source_count=19,
            remote_code_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),
            warning='No SSH, server observation, publication, or actual preflight occurred. Future refs have not been authenticated.')))
        return 0
    require(args.receipt.is_absolute(), 'Absolute local receipt path required')
    args.receipt = args.receipt.parent.resolve(strict=True)/args.receipt.name
    intent = args.receipt.with_name(args.receipt.name+'.intent.json')
    source_copy = args.receipt.with_name(args.receipt.name+'.source.py')
    stdout_path = args.receipt.with_name(args.receipt.name+'.stdout.log')
    stderr_path = args.receipt.with_name(args.receipt.name+'.stderr.log')
    require(not any(os.path.lexists(path) for path in (args.receipt,intent,source_copy,stdout_path,stderr_path)),
            'Existing local receipt/intent/source/logs: no retry')
    source_bytes = Path(__file__).resolve().read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest()==payload['preparation_script_sha256'], 'Script changed during preparation')
    write_bytes_once(source_copy,source_bytes)
    source_reference=dict(path=str(source_copy),sha256=sha(source_copy))
    write_once(intent, dict(status='REMOTE_PREPARATION_ATTEMPTED_UNKNOWN_UNTIL_RECEIPT',request=payload,
        source=source_reference,stdout_path=str(stdout_path),stderr_path=str(stderr_path)))
    command = ' '.join(shlex.quote(x) for x in [
        'env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
        PYTHON,'-I','-B','-c',REMOTE])
    # One invocation; intentionally no application timeout/retry. Interrupted SSH
    # can leave published artifacts, and the intent permanently preserves UNKNOWN.
    try:
        outfd=os.open(stdout_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(outfd,'wb') as out:
            errfd=os.open(stderr_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(errfd,'wb') as err:
                try:
                    process = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',
                        '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','eee-cluster',command],
                        input=json.dumps(payload), text=True, stdout=out, stderr=err, check=False)
                finally:
                    out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
        stdout=stdout_path.read_text(errors='replace');stderr=stderr_path.read_text(errors='replace')
        lines = [line for line in stdout.splitlines() if line.startswith('SATDOSE_PREPARE_JSON=')]
        value = json.loads(lines[0].split('=',1)[1]) if len(lines) == 1 else None
        receipt = dict(status='REMOTE_PREPARATION_RETURNED' if value is not None else 'REMOTE_PREPARATION_UNKNOWN_NO_RETRY',
            returncode=process.returncode, response=value, stdout=stdout, stderr=stderr,
            source=source_reference,stdout_file=dict(path=str(stdout_path),sha256=sha(stdout_path)),
            stderr_file=dict(path=str(stderr_path),sha256=sha(stderr_path)),
            script_sha256=payload['preparation_script_sha256'])
        write_once(args.receipt,receipt)
        print(json.dumps(dict(local_receipt=str(args.receipt),status=receipt['status'],response=value),sort_keys=True))
        return 0 if process.returncode == 0 and value and 'result' in value else 2
    except BaseException as exc:
        if not os.path.lexists(args.receipt):
            write_once(args.receipt,dict(status='REMOTE_PREPARATION_UNKNOWN_NO_RETRY',
                error_type=type(exc).__name__,error=str(exc),intent=str(intent),source=source_reference,
                raw_logs=[dict(path=str(path),sha256=sha(path)) for path in (stdout_path,stderr_path) if path.is_file()]))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
