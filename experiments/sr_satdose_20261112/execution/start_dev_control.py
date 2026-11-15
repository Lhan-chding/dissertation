"""Fixed START DEV controller: 108 original requests per start; no TRAIN/MAIN.

The externally pinned release authenticates a full admission verifier and every
source before model construction. Each fresh process owns the existing global
worker lock and one write-once endpoint claim. Unknown attempts are never resumed.
This module grants no training permission and never constructs an optimizer.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import socket
import time

from . import evaluation_runtime as runtime, storage_ready as storage
from runtime.backend import FrozenRegistry, GenerationBridge, object_sha

ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
PACKAGE = Path(__file__).resolve().parents[1]
EXPERIMENT = 'SR-SATDOSE-1-20261112'
FREEZE_SHA = '6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
CONFIG_SHA = '58841639b0f36a69ac77824bc8c683ef7fbeb5c50a41e9975b89b54880f4ecfe'
tx = runtime.tx
require = tx.require


def sha(path):
    path=Path(path)
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Canonical regular file required')
    return tx.file_sha(path)


def ref(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))
def bound(reference):return tx.bound(ROOT,reference)
def read(path):return tx.read_json(path)
def rows(path):return [tx.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _process_identity():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid=os.getpid(),start_ticks=fields[19])


def inventory(directory):
    result={}
    for parent,dirs,files in os.walk(directory,followlinks=False):
        for name in dirs:
            p=Path(parent)/name;require(p.resolve(strict=True)==p and not p.is_symlink(),'Release directory symlink')
        for name in files:
            p=Path(parent)/name;result[str(p.relative_to(directory))]=sha(p)
    return result


def verify_release(path,expected):
    path=Path(path)
    require(re.fullmatch('[a-f0-9]{64}',expected or '') is not None and sha(path)==expected,'Untrusted START DEV release')
    value=read(path)
    fields=dict(schema='sr-satdose-start-dev-release-v1',status='AUDITED_FOR_START_DEV_ONLY',
        permission='START_DEV_ONLY_NO_TRAINING',experiment_id=EXPERIMENT,evidence_root=str(ROOT),
        release_root=str(PACKAGE),design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA)
    require(all(value.get(k)==v for k,v in fields.items()) and path.is_relative_to(ROOT) and
            not path.is_relative_to(PACKAGE),'START DEV release identity differs')
    actual=inventory(PACKAGE)
    required={'execution/start_dev_control.py','execution/evaluation_runtime.py','execution/evaluation_loader.py',
        'execution/training_state.py','execution/training_transaction.py','execution/storage_ready.py',
        'execution/admission_authority.py','runtime/backend.py','runtime/native_runtime.py','runtime/fork_loader.py'}
    require(required<=set(actual) and actual==value['sources'],'Complete START DEV release inventory differs')
    freeze=read(ROOT/'DESIGN_FREEZE.json')
    require(sha(ROOT/'DESIGN_FREEZE.json')==FREEZE_SHA and
            {k.removeprefix('runtime/'):v for k,v in actual.items() if k.startswith('runtime/') and k.endswith('.py')}==
            freeze['runtime_sources'],'Frozen runtime changed')
    for name,module in [('evaluation_runtime',runtime),('evaluation_loader',runtime.loader),
                        ('training_state',runtime.state),('training_transaction',tx),('storage_ready',storage)]:
        require(Path(module.__file__).resolve()==PACKAGE/'execution'/(name+'.py'),'Shadowed controller dependency')
    require(storage.ROOT==ROOT and storage.FREEZE_SHA==FREEZE_SHA,'Storage implementation identity differs')
    return value


class Lease:
    """Real nonblocking global lock, released on every exception."""
    def __init__(self):self.fd=None
    def __enter__(self):
        require(ROOT.resolve(strict=True)==ROOT,'Noncanonical evidence root')
        self.fd=os.open(ROOT/'DIAGNOSTIC_WORKER.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(self.fd);self.fd=None;raise
        return self
    def active(self):
        require(self.fd is not None,'Inactive global evaluation lease')
        path=ROOT/'DIAGNOSTIC_WORKER.lock';a=os.fstat(self.fd);b=path.stat()
        require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and not path.is_symlink(),'Replaced global lock')
    def __exit__(self,*args):
        if self.fd is not None:os.close(self.fd);self.fd=None


def _prior_start(start,release_reference,admission_reference):
    parent=ROOT/'start_dev'
    if not parent.exists():require(start==11031,'11031 START DEV must complete first');return
    require(parent.resolve(strict=True)==parent and parent.is_dir(),'Noncanonical START DEV root')
    require({p.name for p in parent.iterdir()}<={'start-11031','start-11032'},'Unexpected START DEV attempt')
    require(not os.path.lexists(parent/f'start-{start}'),'Existing START DEV attempt: never retry')
    if start==11031:
        require(not any(parent.iterdir()),'Unknown/out-of-order START DEV attempt')
    else:
        directory=parent/'start-11031'
        require(directory.is_dir() and not any('HOLD' in p.name or 'ERROR' in p.name for p in directory.rglob('*')),
                'Previous START DEV failed/unknown')
        done=read(directory/'RESULT.json')
        require(done.get('status')=='START_DEV_ENDPOINT_COMPLETE' and done.get('start_seed')==11031 and
                done.get('responses')==108 and done.get('design_freeze_sha256')==FREEZE_SHA and
                done.get('release')==release_reference and done.get('admission')==admission_reference,
                'Previous START DEV incomplete/foreign release')
        endpoint_dir=directory/'endpoint'
        require(bound(done['receipt'])==endpoint_dir/'RECEIPT.json' and
                bound(done['endpoint_binding'])==directory/'ENDPOINT_BINDING.json' and
                bound(done['close'])==directory/'CLOSE.json','Previous completion was relocated')
        receipt=read(bound(done['receipt']));binding=read(bound(done['endpoint_binding']))
        require(receipt.get('responses')==108 and receipt.get('status')==
                'EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION','Previous completion receipt invalid')
        require(receipt['endpoint_binding_sha256']==done['endpoint_binding']['sha256'] and
                receipt['endpoint_before']==receipt['endpoint_after']==binding['endpoint'] and
                binding['endpoint']['start_seed']==11031 and binding['endpoint']['arm']=='START' and
                binding['endpoint']['step']==0 and binding['scope']=='DEV','Previous endpoint binding differs')
        for key in ('raw','accounting'):bound(receipt[key])
        require(len(receipt['complete_files'])==108,'Previous completion population differs')
        frozen=read(ROOT/'DESIGN_FREEZE.json')
        expected=[row for row in rows(bound(frozen['artifacts']['dev_manifest']))
                  if (row['start_seed'],row['arm'],row['step'])==(11031,'START',0)]
        require(len(expected)==108,'Previous frozen endpoint population differs')
        for manifest,reference in zip(expected,receipt['complete_files'],strict=True):
            stem=hashlib.sha256(manifest['request_id'].encode()).hexdigest()
            path=bound(reference);require(path==endpoint_dir/(stem+'.COMPLETE.json'),'Previous request order/location differs')
            completed=read(path);captured=endpoint_dir/(stem+'.CAPTURE.json');begin=endpoint_dir/(stem+'.BEGIN.json')
            require(completed['request_id']==manifest['request_id'] and completed['seed']==manifest['seed'] and
                    completed['native_capture_path']==str(captured) and sha(captured)==completed['native_capture_sha256'] and
                    read(begin)==dict(request_id=manifest['request_id'],seed=manifest['seed'],endpoint=binding['endpoint'],
                        design_freeze_sha256=FREEZE_SHA,status='ATTEMPTED_UNKNOWN'),'Previous original capture chain differs')


def _hold_free():
    for name in ('training_transactions','start_dev'):
        directory=ROOT/name
        if directory.exists():
            require(not directory.is_symlink(),'Symlink control directory')
            for p in directory.rglob('*'):
                require(not p.is_symlink(),'Symlink attempt evidence')
                require('HOLD' not in p.name and 'ERROR' not in p.name and not p.name.startswith('.staging'),
                        'Global failed/unknown attempt')


def _registry(freeze):
    path=lambda key:bound(freeze['artifacts'][key])
    return FrozenRegistry(path('config'),ROOT,ROOT/'DESIGN_FREEZE.json',
        [path(k) for k in ('train_manifest','dry_manifest','dev_manifest','prefix_calib_manifest','main_manifest')],
        path('inputs'),path('images'))


def preflight(release_path,release_sha,start):
    require(type(start) is int and start in (11031,11032),'Only the two fixed START DEV endpoints')
    release=verify_release(release_path,release_sha);_hold_free()
    _prior_start(start,ref(Path(release_path)),release['admission'])
    spec=read(bound(release['admission_spec']))
    authority=importlib.import_module('execution.admission_authority')
    require(Path(authority.__file__).resolve()==PACKAGE/'execution/admission_authority.py' and
            spec['sources']['authority']==dict(path=str(PACKAGE/'execution/admission_authority.py'),
                sha256=release['sources']['execution/admission_authority.py']),'Foreign admission verifier')
    candidate=authority.prepare(spec)
    require(read(bound(release['admission']))==candidate and candidate.get('status')=='ADMITTED_FIRST_TWO_STEPS_ONLY' and
            candidate.get('permission')=='FOUR_PATHS_MAX_STEP_2','Published full admission differs')
    freeze=read(ROOT/'DESIGN_FREEZE.json');registry=_registry(freeze)
    require(sha(bound(freeze['artifacts']['config']))==CONFIG_SHA,'Config identity differs')
    selection,requests=runtime._selection(registry,start,'START',0,'DEV')
    manifest=[r for r in rows(bound(freeze['artifacts']['dev_manifest']))
              if (r['start_seed'],r['arm'],r['step'])==(start,'START',0)]
    require([r['request_id'] for r in requests]==[r['request_id'] for r in manifest] and len(manifest)==108,
            'START DEV fixed manifest order differs')
    source=registry.config['start_bindings'][str(start)]
    endpoint={k:v for k,v in selection.items() if k!='scope'}|dict(policy_sha256=source['policy_sha256'],
                                                               reference_sha256=source['reference_sha256'])
    budget=storage._budget(release['storage_budget'],storage._now())
    require(budget.get('scope')=='START_DEV_ONLY','Storage budget scope differs')
    sources={name:dict(path=str(PACKAGE/'execution'/(name+'.py')),sha256=release['sources']['execution/'+name+'.py'])
             for name in ('evaluation_runtime','evaluation_loader','training_state','training_transaction')}
    sources['authority']=dict(path=str(Path(__file__).resolve()),sha256=release['sources']['execution/start_dev_control.py'])
    return dict(release=release,release_ref=ref(Path(release_path)),spec=spec,candidate=candidate,freeze=freeze,
        registry=registry,selection=selection,requests=requests,manifest=manifest,endpoint=endpoint,
        sources=sources,sources_sha256=object_sha(sources),start=start,budget=budget)


class StartDevAuthority(runtime.AuthenticatedEvaluationAuthority):
    """Only START DEV while the real global lease and bound claim remain live."""
    def __init__(self,release_path,release_sha,start,lease):
        require(type(lease) is Lease,'Actual controller lease required');lease.active()
        self.lease=lease;self.ctx=preflight(release_path,release_sha,start);self.index=0;self.claim=None
        self.process=_process_identity()
        if start==11032:
            previous=read(ROOT/'start_dev/start-11031/RESULT.json')
            require(previous['process_identity']!=self.process,'Each START DEV endpoint requires a fresh process')
        self.output=ROOT/'start_dev'/f'start-{start}'/'endpoint'
        super().__init__(self.ctx['sources']['authority'])

    def activate(self,claim):
        require(self.claim is None,'Controller claim already activated')
        value=read(bound(claim));c=self.ctx
        require(value==dict(schema='sr-satdose-start-dev-claim-v1',status='ATTEMPTED_UNKNOWN_NO_RETRY',
            start_seed=c['start'],responses=108,release=c['release_ref'],admission=c['release']['admission'],
            design_freeze_sha256=FREEZE_SHA,request_ids_sha256=object_sha([r['request_id'] for r in c['requests']]),
            process_identity=self.process,output=str(self.output)),'Controller claim identity differs')
        self.claim=copy.deepcopy(claim)

    def _fast(self):
        self.lease.active();require(self.claim is not None,'Missing live controller claim');bound(self.claim);_hold_free()
        require(_process_identity()==self.process,'Controller process identity changed')
        c=self.ctx
        require(sha(ROOT/'DESIGN_FREEZE.json')==FREEZE_SHA,'Freeze changed')
        bound(c['release_ref'])
        for key in ('admission_spec','admission','storage_budget'):bound(c['release'][key])
        for reference in c['spec']['documents'].values():bound(reference)
        # Small immutable source/receipt hashes only; no repeated tensor audit or DRY rescoring.
        for path,digest in c['release']['sources'].items():require(sha(PACKAGE/path)==digest,'Release source changed')
        for reference in c['spec']['sources'].values():require(sha(Path(reference['path']))==reference['sha256'],'Admission source changed')

    def verify(self,*,root,selection,endpoint,design_freeze_sha256,execution_sources_sha256,request_ids_sha256,request_id):
        self._fast();c=self.ctx
        require(root==ROOT and selection==c['selection'] and selection['scope']=='DEV' and
                selection['arm']=='START' and selection['step']==0 and endpoint==c['endpoint'] and
                design_freeze_sha256==FREEZE_SHA and execution_sources_sha256==c['sources_sha256'] and
                request_ids_sha256==object_sha([r['request_id'] for r in c['requests']]),'START-only DEV authority scope differs')
        out=dict(schema='sr-satdose-authenticated-evaluation-authority-v1',selection=selection,endpoint=endpoint,
            design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA,execution_sources_sha256=execution_sources_sha256,
            request_ids_sha256=request_ids_sha256,request_id=request_id,admission=c['release']['admission'],
            registration=c['spec']['documents']['registration'],gpu_report=c['spec']['documents']['gpu_report'])
        if request_id is not None:
            require(self.index<108 and c['requests'][self.index]['request_id']==request_id,'Repeated/out-of-order START DEV request')
            if self.index:
                previous=self.output/(hashlib.sha256(c['requests'][self.index-1]['request_id'].encode()).hexdigest()+'.COMPLETE.json')
                require(previous.is_file(),'Previous draw remains UNKNOWN')
            begin=self.output/(hashlib.sha256(request_id.encode()).hexdigest()+'.BEGIN.json')
            out['begin']=ref(begin)
            require(read(begin)==dict(request_id=request_id,seed=c['requests'][self.index]['seed'],endpoint=endpoint,
                design_freeze_sha256=FREEZE_SHA,status='ATTEMPTED_UNKNOWN'),'Original START DEV BEGIN differs')
            self.index+=1
        return out


def storage_check(directory,budget_ref):
    require(directory.resolve(strict=True)==directory and directory.is_dir() and directory.is_relative_to(ROOT),
            'Noncanonical START DEV storage directory')
    started=storage._now();tick=time.monotonic();budget=storage._budget(budget_ref,started)
    require(budget.get('scope')=='START_DEV_ONLY','Wrong DEV storage budget scope')
    before=storage._observe(directory);require(before['effective_available_bytes']>=budget['required_bytes'],'Insufficient DEV storage')
    probe=storage._probe(directory);after=storage._observe(directory);finished=storage._now()
    storage._budget(budget_ref,finished)
    require(min(before['effective_available_bytes'],after['effective_available_bytes'])>=budget['required_bytes'] and
            0<=time.monotonic()-tick<=budget['max_observation_age_seconds'] and
            (finished-started).total_seconds()<=budget['max_observation_age_seconds'],'Stale/insufficient DEV storage')
    return dict(schema='sr-satdose-start-dev-storage-v1',budget=budget_ref,observations=[before,after],probe=probe,
                checked_at_utc=finished.isoformat(),permission='START_DEV_ONLY_NO_TRAINING')


def _allocation():
    require(re.fullmatch('[0-9]+',os.environ.get('SLURM_JOB_ID','')) is not None and
            os.environ.get('SLURM_NTASKS')=='1' and os.environ.get('SLURM_NNODES')=='1' and
            os.environ.get('SLURM_RESTART_COUNT','0')=='0','Fresh one-task one-node Slurm allocation required')


def execute(release_path,release_sha,start):
    _allocation()
    with Lease() as lease:
        authority=StartDevAuthority(release_path,release_sha,start,lease);c=authority.ctx
        parent=ROOT/'start_dev';parent.mkdir(exist_ok=True)
        directory=parent/f'start-{start}';directory.mkdir(exist_ok=False);tx.fsync_directory(parent)
        output=directory/'endpoint';output.mkdir();tx.fsync_directory(directory)
        claim=dict(schema='sr-satdose-start-dev-claim-v1',status='ATTEMPTED_UNKNOWN_NO_RETRY',start_seed=start,responses=108,
            release=c['release_ref'],admission=c['release']['admission'],design_freeze_sha256=FREEZE_SHA,
            request_ids_sha256=object_sha([r['request_id'] for r in c['requests']]),
            process_identity=authority.process,output=str(output))
        tx.write_once(directory/'CLAIM.json',claim);authority.activate(ref(directory/'CLAIM.json'))
        handle=None
        try:
            tx.write_once(directory/'STORAGE.json',storage_check(output,c['release']['storage_budget']))
            binding=dict(schema='sr-satdose-evaluation-endpoint-binding-v1',scope='DEV',endpoint=c['endpoint'],requests=108,
                request_ids_sha256=claim['request_ids_sha256'],design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA,
                source_lock_sha256=c['freeze']['artifacts']['source_lock']['sha256'],
                manifest_sha256=c['freeze']['artifacts']['dev_manifest']['sha256'],
                execution_sources_sha256=c['sources_sha256'],checkpoint=None)
            tx.write_once(directory/'ENDPOINT_BINDING.json',binding)
            handle=runtime.build_evaluation_runtime(c['registry'],bound(c['freeze']['artifacts']['source_lock']),
                start_seed=start,arm='START',step=0,scope='DEV',output_dir=output,authority=authority,
                authority_source_reference=c['sources']['authority'],execution_sources=c['sources'],
                expected_execution_sources_sha256=c['sources_sha256'])
            require(handle.endpoint()==c['endpoint'],'Constructed START DEV identity differs')
            bridge=GenerationBridge(c['registry'],output,handle,handle.endpoint);completes=[]
            raw_path=output/'RAW.jsonl'
            fd=os.open(raw_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as raw:
                for manifest in c['manifest']:
                    value=bridge.generate_one(manifest['request_id'])
                    row=dict(value,**{k:v for k,v in manifest.items() if k!='request_id'},scope='DEV',suffix_text=value['generated_text'])
                    raw.write(tx.canonical(row)+b'\n');raw.flush();os.fsync(raw.fileno())
                    completes.append(ref(output/(hashlib.sha256(manifest['request_id'].encode()).hexdigest()+'.COMPLETE.json')))
            before=handle.endpoint();closed=handle.close();handle=None
            require(before==c['endpoint'] and closed['attempted_requests']==108 and
                    all(closed[k]==0 for k in ('optimizer_steps','scheduler_steps','parameter_updates')),'START DEV changed policy or count')
            authority._fast();verify_release(release_path,release_sha)
            events=rows(output/'accounting.jsonl');require(len(events)==216,'START DEV accounting population differs')
            for index,manifest in enumerate(c['manifest']):
                generated=read(bound(completes[index]))
                for event,kind,count in zip(events[2*index:2*index+2],('completion_attempts','generated_tokens'),(1,len(generated['tokens']))):
                    require(event['kind']==kind and event['count']==count and event['metadata']['group_seed']==manifest['seed'],
                            'START DEV native accounting differs')
            tx.write_once(directory/'CLOSE.json',closed)
            receipt=dict(schema='sr-satdose-evaluation-completion-v1',status='EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION',
                scope='DEV',endpoint_binding_sha256=sha(directory/'ENDPOINT_BINDING.json'),endpoint_before=c['endpoint'],endpoint_after=before,
                responses=108,optimizer_created=False,scheduler_created=False,optimizer_steps=0,scheduler_steps=0,parameter_updates=0,
                extra_forward_sequences=0,raw=ref(raw_path),accounting=ref(output/'accounting.jsonl'),complete_files=completes)
            tx.write_once(output/'RECEIPT.json',receipt)
            result=dict(schema='sr-satdose-start-dev-result-v1',status='START_DEV_ENDPOINT_COMPLETE',permission='NO_TRAINING',
                start_seed=start,responses=108,design_freeze_sha256=FREEZE_SHA,release=c['release_ref'],
                process_identity=authority.process,
                admission=c['release']['admission'],receipt=ref(output/'RECEIPT.json'),
                endpoint_binding=ref(directory/'ENDPOINT_BINDING.json'),close=ref(directory/'CLOSE.json'))
            tx.write_once(directory/'RESULT.json',result);return result
        except BaseException as exc:
            if handle is not None:
                try:handle.close()
                except BaseException:pass
            hold=dict(status='HOLD_START_DEV_TECHNICAL',permission='NO_TRAINING',start_seed=start,
                error_type=type(exc).__name__,error=str(exc),recovery='AUDIT_EXISTING_ATTEMPT_NO_RESAMPLING')
            tx.write_once(directory/'HOLD.json',hold)
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-receipt',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    p.add_argument('--start-seed',type=int,choices=(11031,11032),required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    try:
        value=execute(a.release_receipt,a.release_sha256,a.start_seed) if a.execute else preflight(a.release_receipt,a.release_sha256,a.start_seed)
        if not a.execute:value=dict(status='START_DEV_PREFLIGHT_COMPLETE',start_seed=a.start_seed,responses=108,permission='NO_MODEL_CALLS')
    except Exception as exc:
        value=dict(status='HOLD_START_DEV_CONTROL',permission='NO_TRAINING',error_type=type(exc).__name__,error=str(exc))
    print(json.dumps(value,sort_keys=True,allow_nan=False));return 2 if value['status'].startswith('HOLD') else 0


if __name__=='__main__':raise SystemExit(main())
