#!/usr/bin/env python3
"""Prepare one fixed MAIN endpoint after complete actual terminal publication.

Default local plan never contacts a server. --execute performs one CPU-only
preparation, preserving exact source bytes and logs. No scheduler/model call,
retry, overwrite, extra endpoint, original TRAIN or external4190 edit exists.
This module also exposes read-only verification for its separately pinned submitter.
"""
import argparse
import base64
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

PROJECT=Path('/Users/louis/Documents/ChatGPT/dissertation-ntu')
ROOT=Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
TRAIN=ROOT/'train_code_71018e5';EXTERNAL=ROOT/'external_main_code_4190bc3'
TERMINAL=ROOT/'execution_releases/terminal_4190bc3'
PREPARATIONS=ROOT/'execution_releases/main_preparations_4190bc3'
PYTHON='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
PUBLISHER_SHA='49e1188ea76d5945ee21faad394b0ea06064bd18f1f8fe0b836e43940c783511'
ORIGINAL=dict(path='execution_releases/FIRST_TWO_RELEASE_71018e5.json',sha256='22f3b7b260670628398154652455520cd25169234f66a827d3d117def924438f')
CANARY=dict(path='execution_releases/canary_job_1740834/CANARY_SPEC.json',sha256='44a8db68c655bea987629ea210da1a32e268806f6e0c7b1506d6d77828d4b260')
PINS={
 'FIRST_TWO_CODE_DEPLOYMENT_71018e5_20261116.json':'de42793b8d839e7d72d8eb399d37c0fb9e842d02f838709cde5b1a0e3ebbbeb8',
 'ORIGINAL_TRAIN_EXECUTION_SOURCES_71018e5.json':'5370b089381609c2e8d5d5ec7555c3dd441babaed242bd66b7cbfad02f679dd7',
 'EXTERNAL_CODES_DEPLOYMENT_4190bc3_20261116.json':'b585eaf9f3d8243c956ff79b152f5afbcb47aabb5ca72d42c663d11f25ce379b'}
ORDER=tuple([(s,'START',0,1952) for s in (11031,11032)]+
 [(s,a,t,1952 if t==24 else 256) for s in (11031,11032) for a in ('T0','T1') for t in (8,16,24)])
FILES={'PREPARER_SOURCE.py','INPUT.json','INTENT.json','TERMINAL_MANIFEST.json','STORAGE_BUDGET.json','RELEASE.json','PREFLIGHT.json','PREPARATION_RECEIPT.json'}


def require(ok,message):
    if not ok:raise PermissionError('HOLD_MAIN_PREPARATION: '+message)
def utc():return datetime.now(timezone.utc).isoformat()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def read(path):
    def pairs(items):
        value=dict(items);require(len(value)==len(items),'Duplicate JSON key');return value
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
      parse_constant=lambda value:require(False,'Nonfinite JSON'))
def ordinary(path):
    path=Path(path);require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),
      'Noncanonical ordinary file: '+str(path));return path
def sha(path):
    h=hashlib.sha256()
    with ordinary(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def reference(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value['path'],str) and
      re.fullmatch('[0-9a-f]{64}',value['sha256'] or ''),'Invalid reference')
    path=Path(value['path']);require(not path.is_absolute() and '..' not in path.parts and str(path)==value['path'],'Reference escapes root')
    return value
def bound(value):
    reference(value);path=ROOT/value['path'];require(sha(path)==value['sha256'],'Root artifact SHA differs');return path
def ref(path):return dict(path=str(ordinary(path).relative_to(ROOT)),sha256=sha(path))
def localref(path):return dict(path=str(ordinary(path)),sha256=sha(path),bytes=path.stat().st_size)
def fsync(directory):
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def write_once(path,value):
    require(path.parent.resolve(strict=True)==path.parent,'Noncanonical output parent')
    data=value if isinstance(value,bytes) else canonical(value)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fsync(path.parent);return path

def selection(start,arm,step):
    require(type(start) is int and type(step) is int,'Integer endpoint required')
    match=[entry for entry in ORDER if entry[:3]==(start,arm,step)]
    require(len(match)==1,'Only the fourteen frozen MAIN endpoints')
    return dict(start_seed=start,arm=arm,step=step),match[0][3]
def key(endpoint):return f"{endpoint['start_seed']}/{endpoint['arm']}/{endpoint['step']}"
def preparation_dir(endpoint):return PREPARATIONS/f"start-{endpoint['start_seed']}"/endpoint['arm']/f"step-{endpoint['step']:03d}"
def load_source(path,expected,name):
    require(sha(path)==expected and name not in sys.modules,'Source changed or fresh namespace unavailable')
    data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==expected,'Source changed during read')
    spec=importlib.util.spec_from_file_location(name,path);require(spec and spec.loader,'Cannot load authenticated module')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:exec(compile(data,str(path),'exec'),module.__dict__)
    except BaseException:sys.modules.pop(name,None);raise
    return module

def validate_terminal(manifest,directory,publisher_path,*,remote=False):
    require(manifest.get('schema')=='sr-satdose-terminal-operator-manifest-v1' and
      manifest.get('status')=='ACTUAL_TERMINAL_PUBLICATION_COMPLETE' and manifest.get('success') is True and
      manifest.get('publication')==str(TERMINAL) and manifest.get('completion_validation_error') is None,
      'Terminal manifest is incomplete, negative, or UNKNOWN')
    require(directory.resolve(strict=True)==directory and directory.is_dir(),'Noncanonical terminal evidence directory')
    require(not os.path.lexists(directory/'ERROR.json') and not os.path.lexists(directory/'LOCAL_ERROR.json'),
      'Terminal publication retained ERROR')
    # Authenticated, pure envelope verifier. Never call publisher bootstrap/publish here.
    name='_satdose_main_terminal_envelope'
    pub=load_source(publisher_path,PUBLISHER_SHA,name)
    try:
        require(set(manifest['files'])==pub.FILES-{'ERROR.json'},'Incomplete eleven-artifact terminal manifest')
        if remote:require({p.name for p in directory.iterdir()}==set(manifest['files']),'Extra/missing remote publication artifact')
        receipt=pub.validate_publication(directory,manifest['files'])
    finally:sys.modules.pop(name,None)
    require(receipt['publisher_sha256']==PUBLISHER_SHA and receipt['original_train_release']==ORIGINAL and
      receipt['canary']==CANARY,'Terminal publication source/original/canary differs')
    return receipt

def local_terminal(directory,manifest_sha,download_sha):
    directory=directory.resolve(strict=True)
    require(re.fullmatch('[0-9a-f]{64}',manifest_sha or '') and re.fullmatch('[0-9a-f]{64}',download_sha or ''),'Explicit terminal evidence hashes required')
    require(sha(directory/'REMOTE_MANIFEST.json')==manifest_sha and sha(directory/'LOCAL_DOWNLOAD_RECEIPT.json')==download_sha,
      'Local terminal manifest/download receipt changed')
    manifest=read(directory/'REMOTE_MANIFEST.json');download=read(directory/'LOCAL_DOWNLOAD_RECEIPT.json')
    require(download.get('status')=='EXACT_TERMINAL_ARTIFACT_BYTES_DOWNLOADED' and download.get('publication_success') is True and
      type(download.get('remote_returncode')) is int and download['remote_returncode']==0 and download.get('publisher_sha256')==PUBLISHER_SHA,
      'No completed successful exact terminal download')
    require(set(download['files'])==set(manifest['files']),'Download manifest population differs')
    for name,value in download['files'].items():
        require(value==dict(manifest['files'][name],local_path=str(directory/name)),'Download file binding differs')
    publisher=Path(__file__).resolve().with_name('satdose_publish_actual_terminal.py')
    validate_terminal(manifest,directory,publisher)
    return manifest

def local_pins():
    values=[]
    for name,digest in PINS.items():
        path=PROJECT/'artifacts/sr_satdose_20261112/intake'/name
        require(sha(path)==digest,'Local immutable intake changed');values.append(read(path))
    train,execution,external=values
    require(train['release_root']==str(TRAIN) and len(train['sources'])==84 and len(execution)==19 and
      external['original_train_inventory_unchanged'] is True and external['releases']['main']['root']==str(EXTERNAL),
      'Original/external inventory metadata differs')
    for value in execution.values():require(train['sources'][str(Path(value['path']).relative_to(TRAIN))]==value['sha256'],'Original source19 pin differs')
    return dict(sources=train['sources'],execution_sources=execution,external_sources=external['releases']['main']['sources'])

def inventory(directory):
    require(directory.resolve(strict=True)==directory and directory.is_dir(),'Noncanonical source directory')
    value={}
    for path in directory.rglob('*'):
        require(path.resolve(strict=True)==path and not path.is_symlink(),'Noncanonical source entry')
        if path.is_file():value[str(path.relative_to(directory))]=sha(path)
        else:require(path.is_dir(),'Nonordinary source entry')
    return value

@contextmanager
def existing_lease(name,exclusive):
    path=ordinary(ROOT/name);fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,(fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|fcntl.LOCK_NB)
        def stable():
            a=os.fstat(fd);b=path.stat()
            require(not path.is_symlink() and (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'Existing lease replaced')
        stable();yield stable;stable()
    finally:os.close(fd)

def authenticate_remote(payload):
    require(ROOT.resolve(strict=True)==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Fixed CPU-only root/environment required')
    require(inventory(TRAIN)==payload['sources'] and len(payload['sources'])==84 and
      inventory(EXTERNAL)==payload['external_sources'],'Original/external source inventories differ')
    original=read(bound(ORIGINAL))
    require(original['sources']==payload['sources'] and original['execution_sources']==payload['execution_sources'] and
      len(original['execution_sources'])==19 and original['release_root']==str(TRAIN),'Original source84/19 identity differs')
    receipt=validate_terminal(payload['terminal_manifest'],TERMINAL,TERMINAL/'PUBLISHER_SOURCE.py',remote=True)
    spec=read(bound(receipt['terminal_spec']))
    require(spec['external_root']==str(EXTERNAL) and spec['external_sources']==payload['external_sources'] and
      spec['original_train_release']==ORIGINAL,'Terminal spec changes frozen code identity')
    c=load_source(EXTERNAL/'execution/main_control.py',payload['external_sources']['execution/main_control.py'],'_satdose_main_operator_controller')
    require(list(c.allocations().items())==[(tuple(item[:3]),item[3]) for item in ORDER],'Frozen MAIN order/budgets differ')
    authority=load_source(EXTERNAL/'execution/terminal_authority.py',payload['external_sources']['execution/terminal_authority.py'],
      '_satdose_main_operator_terminal_preverify')
    terminal=authority.verify(spec) # All 96 COMMIT/12288 TRAIN/14 DEV1512 and observed bytes, no new draws.
    require(terminal.original==original and terminal.package==TRAIN and terminal.report['global_hold'] is False and
      terminal.report['pending_required_dev']==[],'Actual terminal state differs')
    torch=sys.modules.get('torch')
    if torch is not None:torch.set_num_threads(1)
    c._hold_free()
    return c,terminal,receipt

def verify_prefix(c,endpoint):
    order=[f'{s}/{a}/{t}' for s,a,t,n in ORDER];index=order.index(key(endpoint));existing=c._existing_slots()
    require(set(existing)==set(order[:index]),'MAIN requires exact completed prefix; occupied/extra/missing attempt')
    result={}
    for prior in order[:index]:
        path=existing[prior]/'RESULT.json';value=read(path)
        require(value.get('status')=='MAIN_ENDPOINT_COMPLETE' and f"{value['start_seed']}/{value['arm']}/{value['step']}"==prior,
          'Prior MAIN endpoint incomplete')
        result[prior]=ref(path)
    return result

def validate_preparation_bundle(manifest,expected_source):
    endpoint=manifest['endpoint'];selection(**dict(start=endpoint['start_seed'],arm=endpoint['arm'],step=endpoint['step']))
    directory=preparation_dir(endpoint)
    require(manifest.get('status')=='MAIN_PREPARATION_COMPLETE' and set(manifest['files'])==FILES and
      directory.resolve(strict=True)==directory and {p.name for p in directory.iterdir()}==FILES,
      'Incomplete/unknown MAIN preparation bundle')
    for name,value in manifest['files'].items():
        require(value==localref(directory/name),'MAIN preparation bytes/paths differ')
    receipt=read(directory/'PREPARATION_RECEIPT.json');report=read(directory/'PREFLIGHT.json')
    require(receipt.get('status')=='MAIN_PREPARATION_COMPLETE' and receipt['preparer_sha256']==expected_source==sha(directory/'PREPARER_SOURCE.py') and
      receipt['endpoint']==endpoint and receipt['files']=={name:ref(directory/name) for name in FILES-{'PREPARATION_RECEIPT.json'}} and
      receipt['preflight']==ref(directory/'PREFLIGHT.json'),'Preparation receipt/source lineage differs')
    require(report.get('status')=='MAIN_CPU_PREFLIGHT_AND_STORAGE_COMPLETE' and report['endpoint']==endpoint and
      report['release']==ref(directory/'RELEASE.json') and report['storage_budget']==ref(directory/'STORAGE_BUDGET.json') and
      report['terminal_publication']==ref(TERMINAL/'PUBLICATION_RECEIPT.json'),'Preparation report references differ')
    payload=read(directory/'INPUT.json')
    require(payload['source_sha256']==expected_source and payload['endpoint']==endpoint and
      read(directory/'TERMINAL_MANIFEST.json')==payload['terminal_manifest'],'Preparation request differs')
    return directory,payload,report

def revalidate_prepared(manifest,expected_source,expected_pins):
    directory,payload,report=validate_preparation_bundle(manifest,expected_source)
    require(all(payload[name]==expected_pins[name] for name in ('sources','execution_sources','external_sources')),
      'Prepared inputs differ from independently pinned original/external inventories')
    c,terminal,publication=authenticate_remote(payload)
    endpoint,count=selection(report['endpoint']['start_seed'],report['endpoint']['arm'],report['endpoint']['step'])
    release=read(directory/'RELEASE.json')
    require(release['terminal_manifest']==ref(directory/'TERMINAL_MANIFEST.json') and
      release['terminal_publication']==ref(TERMINAL/'PUBLICATION_RECEIPT.json') and
      release['terminal_spec']==publication['terminal_spec'] and release['original_train_release']==ORIGINAL and
      release['external_sources']==payload['external_sources'] and release['external_root']==str(EXTERNAL) and
      report['responses']==count,'Prepared MAIN source/terminal/population differs')
    ctx=c.preflight(directory/'RELEASE.json',sha(directory/'RELEASE.json'),endpoint['start_seed'],endpoint['arm'],endpoint['step'])
    require(ctx['count']==count and ctx['release']['completed_endpoints']==verify_prefix(c,endpoint),'Current MAIN prefix changed')
    return c,ctx,payload,report,directory

def remote_prepare(payload,source):
    endpoint,count=selection(payload['endpoint']['start_seed'],payload['endpoint']['arm'],payload['endpoint']['step'])
    require(hashlib.sha256(source).hexdigest()==payload['source_sha256'],'Preparation source bytes differ')
    directory=preparation_dir(endpoint);created=False
    try:
        require(not os.path.lexists(directory),'MAIN preparation slot occupied; no retry')
        with ExitStack() as stack:
            stable_launcher=stack.enter_context(existing_lease('FIRST_TWO_LAUNCHER.lock',False))
            stable_worker=stack.enter_context(existing_lease('DIAGNOSTIC_WORKER.lock',True))
            c,terminal,publication=authenticate_remote(payload)
            previous=verify_prefix(c,endpoint);require(not os.path.lexists(c._slot(endpoint['start_seed'],endpoint['arm'],endpoint['step'])),'MAIN target already attempted')
            # Observed DEV + preceding MAIN sizes; no invented quota or success observation.
            receipts=[refs['receipt'] for refs in terminal.audit_spec['dev_endpoints'].values()]
            receipts += [read(bound(value))['receipt'] for value in previous.values()]
            triplets=[];aux=[];observed=[]
            for value in receipts:
                receipt=read(bound(value));n=receipt['responses'];require(type(n) is int and n>0 and len(receipt['complete_files'])==n,'Incomplete sizing endpoint')
                total=0
                for complete_ref in receipt['complete_files']:
                    complete=bound(complete_ref);stem=complete.name.removesuffix('.COMPLETE.json')
                    paths=[complete.parent/(stem+suffix) for suffix in ('.BEGIN.json','.CAPTURE.json','.COMPLETE.json')]
                    sizes=[ordinary(path).stat().st_size for path in paths];triplets.append(sum(sizes));total+=sum(sizes)
                extra=sum(bound(receipt[k]).stat().st_size for k in ('raw','accounting'));aux.append(math.ceil(extra/n))
                observed.append(dict(receipt=value,responses=n,triplet_bytes=total,raw_accounting_bytes=extra))
            require(len(triplets)>=1512,'All14 DEV sizing evidence missing')
            per_response=max(1<<20,4*max(triplets));metadata=128<<20
            required=count*(per_response+4*max(aux))+metadata
            now=datetime.now(timezone.utc)
            budget=dict(schema='sr-satdose-storage-budget-v1',scope='MAIN_ONLY',experiment_id=c.EXPERIMENT,evidence_root=str(ROOT),
              design_freeze_sha256=c.FREEZE_SHA,required_bytes=required,max_observation_age_seconds=300,
              created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(days=7)).isoformat(),planned_responses=count,
              per_response_allowance_bytes=per_response,raw_accounting_per_response_allowance_bytes=4*max(aux),metadata_allowance_bytes=metadata,
              observed_endpoints=observed,rationale='4x largest observed original triplet, minimum1MiB per fixed response; 4x largest raw/accounting per-response mean;128MiB metadata. Engineering allowance, not upper bound or reservation.')
            terminal.stable();c._hold_free();stable_launcher();stable_worker()
            require(not os.path.lexists(directory),'Concurrent MAIN preparation attempt')
            require(PREPARATIONS.parent.resolve(strict=True)==PREPARATIONS.parent,'Noncanonical publication parent')
            for parent in (PREPARATIONS,PREPARATIONS/f"start-{endpoint['start_seed']}",directory.parent):
                if not parent.exists():parent.mkdir();fsync(parent.parent)
                require(parent.resolve(strict=True)==parent and parent.is_dir(),'Noncanonical preparation parent')
            directory.mkdir();fsync(directory.parent);created=True
            write_once(directory/'PREPARER_SOURCE.py',source);write_once(directory/'INPUT.json',payload)
            write_once(directory/'INTENT.json',dict(status='MAIN_PREPARATION_UNKNOWN_UNTIL_RECEIPT',endpoint=endpoint,
              preparer_sha256=payload['source_sha256'],terminal_publication=ref(TERMINAL/'PUBLICATION_RECEIPT.json'),started_at_utc=utc()))
            write_once(directory/'TERMINAL_MANIFEST.json',payload['terminal_manifest']);write_once(directory/'STORAGE_BUDGET.json',budget)
            release=dict(schema='sr-satdose-main-release-v1',status='AUDITED_FOR_FIXED_MAIN_ONLY',permission='FIXED_MAIN_ONLY_NO_TRAINING',
              experiment_id=c.EXPERIMENT,evidence_root=str(ROOT),design_freeze_sha256=c.FREEZE_SHA,config_sha256=c.CONFIG_SHA,
              external_root=str(EXTERNAL),external_sources=payload['external_sources'],original_train_release=ORIGINAL,
              terminal_authority=dict(path=str(EXTERNAL/'execution/terminal_authority.py'),sha256=payload['external_sources']['execution/terminal_authority.py']),
              terminal_spec=publication['terminal_spec'],storage_budget=ref(directory/'STORAGE_BUDGET.json'),endpoint=endpoint,
              completed_endpoints=previous,terminal_publication=ref(TERMINAL/'PUBLICATION_RECEIPT.json'),terminal_manifest=ref(directory/'TERMINAL_MANIFEST.json'))
            write_once(directory/'RELEASE.json',release)
            ctx=c.preflight(directory/'RELEASE.json',sha(directory/'RELEASE.json'),endpoint['start_seed'],endpoint['arm'],endpoint['step'])
            require(ctx['count']==count,'Actual MAIN budget differs')
            ctx['terminal'].stable();ctx['prior'].stable();c._hold_free();stable_worker();stable_launcher()
            validate_terminal(payload['terminal_manifest'],TERMINAL,TERMINAL/'PUBLISHER_SOURCE.py',remote=True)
            storage=c.storage_check(directory,release['storage_budget'])
            report=dict(status='MAIN_CPU_PREFLIGHT_AND_STORAGE_COMPLETE',permission='NO_MODEL_CALLS_OR_GPU_SUBMISSION',
              endpoint=endpoint,responses=count,release=ref(directory/'RELEASE.json'),storage_budget=release['storage_budget'],
              terminal_publication=release['terminal_publication'],terminal_spec=publication['terminal_spec'],
              prior_endpoints=len(previous),formal_optimizer_steps=96,train_responses=12288,dev_endpoints=14,dev_responses=1512,
              storage=storage,model_calls=0,optimizer_calls=0,gpu_submissions=0,checked_at_utc=utc())
            write_once(directory/'PREFLIGHT.json',report)
            write_once(directory/'PREPARATION_RECEIPT.json',dict(status='MAIN_PREPARATION_COMPLETE',endpoint=endpoint,
              preparer_sha256=payload['source_sha256'],preflight=ref(directory/'PREFLIGHT.json'),
              files={name:ref(directory/name) for name in FILES-{'PREPARATION_RECEIPT.json'}},completed_at_utc=utc()))
            manifest=dict(status='MAIN_PREPARATION_COMPLETE',endpoint=endpoint,files={name:localref(directory/name) for name in sorted(FILES)})
            validate_preparation_bundle(manifest,payload['source_sha256'])
            return dict(result=report,manifest=manifest)
    except BaseException as exc:
        value=dict(status='MAIN_PREPARATION_FAILED_OR_UNKNOWN_NO_RETRY',error_type=type(exc).__name__,error=str(exc),
          endpoint=endpoint,preparation_directory=str(directory),model_calls=0,gpu_submissions=0)
        if created:
            try:write_once(directory/'ERROR.json',value)
            except BaseException:pass
        return value

def invoke_remote(payload,source,output,worker,source_filename=None):
    require(output.is_absolute(),'Absolute local receipt required');output=output.parent.resolve(strict=True)/output.name
    paths={name:output.with_name(output.name+suffix) for name,suffix in
      [('source','.source.py'),('intent','.intent.json'),('stdout','.stdout.log'),('stderr','.stderr.log')]}
    require(not any(os.path.lexists(path) for path in [output,*paths.values()]),'Existing local attempt; never retry')
    write_once(paths['source'],source);write_once(paths['intent'],dict(status='REMOTE_ATTEMPT_UNKNOWN_UNTIL_RECEIPT',source=localref(paths['source']),payload=payload))
    bootstrap="import sys,hashlib,json,base64; s=sys.stdin.buffer.read(); p=json.loads(base64.b64decode(sys.argv[1])); assert hashlib.sha256(s).hexdigest()==p['source_sha256']; n={'__name__':'_satdose_pinned_operator','__file__':sys.argv[2]}; exec(compile(s,sys.argv[2],'exec'),n); r=n[sys.argv[3]](p,s); print('SATDOSE_MAIN_OPERATOR='+json.dumps(r,sort_keys=True)); sys.exit(0 if 'result' in r else 2)"
    command=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
      PYTHON,'-I','-B','-c',bootstrap,base64.b64encode(canonical(payload)).decode(),str(source_filename or Path(__file__).resolve()),worker])
    try:
        with paths['stdout'].open('xb') as out,paths['stderr'].open('xb') as err:
            try:response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',
              '-o','ServerAliveCountMax=4','eee-cluster',command],input=source,stdout=out,stderr=err,check=False)
            finally:out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
        stdout=paths['stdout'].read_bytes();marked=[line for line in stdout.splitlines() if line.startswith(b'SATDOSE_MAIN_OPERATOR=')]
        value=json.loads(marked[0].split(b'=',1)[1]) if len(marked)==1 else None
        receipt=dict(status='MAIN_OPERATOR_RESPONSE_CAPTURED' if value is not None else 'MAIN_OPERATOR_UNKNOWN_NO_RETRY',
          returncode=response.returncode,response=value,source=localref(paths['source']),raw_logs={name:localref(paths[name]) for name in ('stdout','stderr')})
        write_once(output,receipt);print(json.dumps(dict(local_receipt=str(output),**receipt),sort_keys=True))
        return 0 if response.returncode==0 and value and 'result' in value else 2
    except BaseException as exc:
        if not os.path.lexists(output):write_once(output,dict(status='MAIN_OPERATOR_UNKNOWN_NO_RETRY',error_type=type(exc).__name__,error=str(exc),
          source=localref(paths['source']),raw_logs={name:localref(paths[name]) for name in ('stdout','stderr') if paths[name].is_file()}))
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-dir',type=Path,required=True)
    parser.add_argument('--terminal-manifest-sha256',required=True);parser.add_argument('--terminal-download-receipt-sha256',required=True)
    parser.add_argument('--start-seed',type=int,choices=(11031,11032),required=True);parser.add_argument('--arm',choices=('START','T0','T1'),required=True)
    parser.add_argument('--step',type=int,choices=(0,8,16,24),required=True);parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    endpoint,count=selection(args.start_seed,args.arm,args.step)
    manifest=local_terminal(args.terminal_dir,args.terminal_manifest_sha256,args.terminal_download_receipt_sha256)
    source=Path(__file__).resolve().read_bytes()
    payload=dict(**local_pins(),endpoint=endpoint,terminal_manifest=manifest,source_sha256=hashlib.sha256(source).hexdigest())
    if not args.execute:
        print(json.dumps(dict(status='LOCAL_MAIN_PREPARATION_PLAN_ONLY_NO_SSH',endpoint=endpoint,responses=count,
          ordinal=[entry[:3] for entry in ORDER].index((args.start_seed,args.arm,args.step))+1,total_endpoints=14,total_responses=13760,
          actual_server_state='NOT_OBSERVED_BY_LOCAL_PLAN')));return 0
    return invoke_remote(payload,source,args.receipt,'remote_prepare')


if __name__=='__main__':raise SystemExit(main())
