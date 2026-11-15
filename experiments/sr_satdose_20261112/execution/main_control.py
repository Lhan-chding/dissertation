"""Standalone fixed MAIN controller, loaded only after full terminal admission.

The standard-library bootstrap authenticates this external release and the
entire original TRAIN package before importing project modules. Each fresh
process evaluates one prescribed endpoint; it cannot train, choose checkpoints,
resume an occupied attempt, add draws, or authorize a successor experiment.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

ROOT=Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
EXPERIMENT='SR-SATDOSE-1-20261112'
FREEZE_SHA='6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
CONFIG_SHA='58841639b0f36a69ac77824bc8c683ef7fbeb5c50a41e9975b89b54880f4ecfe'


def require(value,message):
    if not value:raise PermissionError('HOLD: '+message)


def read(path):
    def pairs(values):
        out=dict(values);require(len(out)==len(values),'Duplicate JSON key');return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda value:require(False,'Nonfinite JSON'))


def sha(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),
            'Noncanonical source/evidence')
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def ref(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))


def bound(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value['path'],str) and
            re.fullmatch('[0-9a-f]{64}',value['sha256'] or ''),'Invalid trusted reference')
    p=Path(value['path']);require(not p.is_absolute() and '..' not in p.parts,'Artifact escapes root')
    path=ROOT/p;require(sha(path)==value['sha256'],'Artifact SHA mismatch');return path


def inventory(root):
    root=Path(root);require(root.resolve(strict=True)==root and root.is_dir(),'Noncanonical code root')
    result={}
    for parent,dirs,files in os.walk(root,followlinks=False):
        for name in dirs:
            p=Path(parent)/name;require(p.resolve(strict=True)==p and not p.is_symlink(),'Code directory symlink')
        for name in files:
            p=Path(parent)/name;result[str(p.relative_to(root))]=sha(p)
    return result


def allocations():
    values={(start,'START',0):1952 for start in (11031,11032)}
    values.update({(start,arm,step):(1952 if step==24 else 256)
                   for start in (11031,11032) for arm in ('T0','T1') for step in (8,16,24)})
    return values


def endpoint_key(start,arm,step):return f'{start}/{arm}/{step}'


def _slot(start,arm,step):
    require(type(start) is int and type(step) is int and (start,arm,step) in allocations(),'Only the fixed fourteen MAIN endpoints')
    return ROOT/'main_control'/f'start-{start}'/arm/f'step-{step:03d}'


def _configure(terminal_module,terminal):
    """Only called after bootstrap authenticates the original namespace files."""
    global TA,runtime,tx,backend,storage,first_two,evaluation,fork_loader
    TA=terminal_module;modules=terminal.modules
    runtime=modules['execution.evaluation_runtime'];tx=modules['execution.training_transaction']
    backend=modules['runtime.backend'];storage=modules['execution.storage_ready']
    first_two=modules['execution.first_two_control'];evaluation=modules['analysis.evaluation'];fork_loader=modules['runtime.fork_loader']


def bootstrap(release_path,release_sha):
    """No nonstandard/project import occurs before both inventories pass."""
    sys.dont_write_bytecode=True
    path=Path(release_path)
    require(re.fullmatch('[0-9a-f]{64}',release_sha or '') and sha(path)==release_sha and
            path.is_relative_to(ROOT),'Untrusted MAIN release receipt')
    release=read(path);external=Path(release['external_root'])
    fields=dict(schema='sr-satdose-main-release-v1',status='AUDITED_FOR_FIXED_MAIN_ONLY',
        permission='FIXED_MAIN_ONLY_NO_TRAINING',experiment_id=EXPERIMENT,evidence_root=str(ROOT),
        design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA)
    require(all(release.get(k)==v for k,v in fields.items()) and external.is_relative_to(ROOT) and
            not path.is_relative_to(external),'MAIN release identity differs')
    actual=inventory(external);require(actual==release['external_sources'],'External MAIN inventory differs')
    here=Path(__file__).resolve();require(here.is_relative_to(external) and
        actual.get(str(here.relative_to(external)))==sha(here),'Running MAIN controller is not in authenticated release')
    original=read(bound(release['original_train_release']));package=Path(original['release_root'])
    require(package.is_relative_to(ROOT) and not package.is_relative_to(external) and not external.is_relative_to(package) and
            original.get('schema')=='sr-satdose-first-two-release-v1' and
            original.get('status')=='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY' and
            original.get('experiment_id')==EXPERIMENT and original.get('evidence_root')==str(ROOT) and
            inventory(package)==original['sources'],'Original immutable TRAIN inventory differs')
    authority=release['terminal_authority'];source=Path(authority['path'])
    require(source.is_relative_to(external) and sha(source)==authority['sha256']==actual.get(str(source.relative_to(external))),
            'Untrusted external terminal authority')
    terminal_spec=read(bound(release['terminal_spec']))
    require(terminal_spec['authority']==authority and terminal_spec['original_train_release']==release['original_train_release'] and
            terminal_spec['external_root']==str(external) and terminal_spec['external_sources']==actual,
            'Terminal and MAIN releases bind different sources')
    name='satdose_terminal_authority_'+authority['sha256'][:16]
    require(name not in sys.modules,'External terminal authority already loaded; fresh MAIN process required')
    data=source.read_bytes();require(hashlib.sha256(data).hexdigest()==authority['sha256'] and sha(source)==authority['sha256'],
                                   'Terminal source changed during read')
    spec=importlib.util.spec_from_file_location(name,source);require(spec and spec.loader,'Cannot load terminal authority')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:exec(compile(data,str(source),'exec'),module.__dict__)
    except BaseException:sys.modules.pop(name,None);raise
    terminal=module.verify(terminal_spec);_configure(module,terminal)
    require(terminal.root==ROOT and terminal.original==original and terminal.package==package,'Terminal namespace differs')
    require(storage.ROOT==first_two.ROOT==ROOT and storage.FREEZE_SHA==FREEZE_SHA,'Foreign original controller environment')
    return release,terminal


def _process():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid=os.getpid(),start_ticks=int(fields[19]))


class Lease:
    def __init__(self):self.fd=None
    def __enter__(self):
        require(ROOT.resolve(strict=True)==ROOT,'Noncanonical MAIN root')
        self.fd=os.open(ROOT/'DIAGNOSTIC_WORKER.lock',os.O_RDWR|os.O_NOFOLLOW)
        try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(self.fd);self.fd=None;raise
        return self
    def active(self):
        require(self.fd is not None,'Inactive MAIN lease')
        p=ROOT/'DIAGNOSTIC_WORKER.lock';a=os.fstat(self.fd);b=p.stat()
        require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and not p.is_symlink(),'Replaced global worker lease')
    def __exit__(self,*args):
        if self.fd is not None:os.close(self.fd);self.fd=None


def _hold_free():
    for name in ('HOLD.json','ERROR.json','training_control/HOLD.json','training_control/ERROR.json',
                 'first_two_launcher/HOLD.json','first_two_launcher/ERROR.json','main_control/HOLD.json','main_control/ERROR.json'):
        require(not os.path.lexists(ROOT/name),'Global HOLD/error forbids MAIN')
    # Check live control flags before every draw. Scandir uses directory-entry
    # metadata: this does not rehash model weights or replay full terminal data.
    for name in ('training_transactions','training_control','first_two_control','first_two_launcher',
                 'continuation_control','canary_replay','start_dev','trained_dev','main_control'):
        parent=ROOT/name
        if not os.path.lexists(parent):continue
        require(parent.resolve(strict=True)==parent and parent.is_dir() and not parent.is_symlink(),
                'Noncanonical live control root')
        pending=[parent]
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    require(not entry.is_symlink(),'Symlink in live control evidence')
                    require('HOLD' not in entry.name and 'ERROR' not in entry.name and not entry.name.startswith('.staging'),
                            'Global failed/unknown control attempt forbids MAIN')
                    if entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))


def _existing_slots():
    parent=ROOT/'main_control';found={}
    if not parent.exists():return found
    require(parent.resolve(strict=True)==parent and parent.is_dir(),'Noncanonical MAIN root')
    for sd in parent.iterdir():
        require(sd.name in ('start-11031','start-11032') and sd.is_dir() and not sd.is_symlink(),'Unknown MAIN start slot')
        start=int(sd.name.removeprefix('start-'))
        for ad in sd.iterdir():
            require(ad.name in ('START','T0','T1') and ad.is_dir() and not ad.is_symlink(),'Unknown MAIN arm slot')
            for slot in ad.iterdir():
                require(re.fullmatch('step-[0-9]{3}',slot.name) and slot.is_dir() and not slot.is_symlink(),'Unknown MAIN step slot')
                step=int(slot.name.removeprefix('step-'));require(slot==_slot(start,ad.name,step),'Unregistered MAIN slot')
                require(not any((slot/name).exists() for name in ('HOLD.json','ERROR.json')),'Prior MAIN failed/unknown')
                found[endpoint_key(start,ad.name,step)]=slot
    return found


def _capture_inventory(directory,request_ids):
    expected={hashlib.sha256(rid.encode()).hexdigest() for rid in request_ids}
    for suffix in ('.BEGIN.json','.CAPTURE.json','.COMPLETE.json'):
        actual={str(p.relative_to(directory)).removesuffix(suffix) for p in directory.rglob('*'+suffix)}
        require(actual==expected,'Extra/missing/unresolved MAIN generation')


def _prior(release,terminal,start,arm,step,registry):
    order=[endpoint_key(*key) for key in allocations()];key=endpoint_key(start,arm,step);index=order.index(key)
    expected=set(order[:index]);require(set(release['completed_endpoints'])==expected,'MAIN requires the exact preceding endpoint prefix')
    existing=_existing_slots();require(set(existing)==expected,'Occupied incomplete/extra MAIN attempt; never retry')
    observed=TA.Observed(ROOT);processes=[]
    for prior in order[:index]:
        value=observed.json(release['completed_endpoints'][prior]);s,a,t=prior.split('/');s=int(s);t=int(t);slot=existing[prior]
        require(observed.ref(release['completed_endpoints'][prior])==slot/'RESULT.json' and
                value.get('status')=='MAIN_ENDPOINT_COMPLETE' and value.get('start_seed')==s and value.get('arm')==a and
                value.get('step')==t and value.get('responses')==allocations()[s,a,t] and value.get('terminal')==terminal.receipt,
                'Prior MAIN endpoint incomplete or foreign terminal')
        old_release=observed.json(value['release'])
        require(old_release.get('schema')=='sr-satdose-main-release-v1' and old_release.get('external_sources')==release['external_sources'] and
                old_release.get('external_root')==release['external_root'] and
                old_release.get('original_train_release')==release['original_train_release'] and
                old_release.get('terminal_spec')==release['terminal_spec'],'Prior MAIN source/terminal release differs')
        require(observed.ref(value['receipt'])==slot/'endpoint/RECEIPT.json' and
                observed.ref(value['endpoint_binding'])==slot/'ENDPOINT_BINDING.json' and
                observed.ref(value['close'])==slot/'CLOSE.json','Prior MAIN artifacts relocated')
        config=bound(read(registry.freeze_path)['artifacts']['config'])
        evaluated=evaluation.analyze_completed_endpoint(config,registry.freeze_path,ROOT,observed.ref(value['receipt']),
            expected_receipt_sha256=value['receipt']['sha256'],endpoint_binding_path=observed.ref(value['endpoint_binding']),
            expected_endpoint_binding_sha256=value['endpoint_binding']['sha256'])
        require(evaluated['summary']['scope']=='MAIN' and evaluated['summary']['metrics']['assigned']==allocations()[s,a,t],
                'Prior MAIN independent population validation failed')
        receipt=observed.json(value['receipt']);request_ids=[]
        for reference in receipt['complete_files']:
            p=observed.ref(reference);complete=read(p);request_ids.append(complete['request_id'])
            observed.path(Path(complete['native_capture_path']),complete['native_capture_sha256'])
            observed.path(p.with_name(p.name.replace('.COMPLETE.json','.BEGIN.json')))
        observed.ref(receipt['raw']);observed.ref(receipt['accounting']);observed.tree(slot)
        _capture_inventory(slot/'endpoint',request_ids);processes.append(value['process_identity'])
    observed.stable();return observed,processes


def preflight(release_path,release_sha,start,arm,step):
    directory=_slot(start,arm,step);require(not os.path.lexists(directory),'Existing MAIN attempt; never retry')
    release,terminal=bootstrap(release_path,release_sha);_hold_free()
    require(release.get('endpoint')==dict(start_seed=start,arm=arm,step=step),'MAIN release names a different endpoint')
    freeze=read(ROOT/'DESIGN_FREEZE.json');path=lambda key:bound(freeze['artifacts'][key])
    registry=backend.FrozenRegistry(path('config'),ROOT,ROOT/'DESIGN_FREEZE.json',
        [path(k) for k in ('train_manifest','dry_manifest','dev_manifest','prefix_calib_manifest','main_manifest')],path('inputs'),path('images'))
    selection,requests=runtime._selection(registry,start,arm,step,'MAIN');count=allocations()[start,arm,step]
    manifest=[row for row in tx.rows(path('main_manifest')) if (row['start_seed'],row['arm'],row['step'])==(start,arm,step)]
    require(len(manifest)==count and [r['request_id'] for r in manifest]==[r['request_id'] for r in requests],
            'Original MAIN population/order differs')
    require(evaluation.endpoint_allocations('MAIN')==allocations(),'Frozen MAIN allocation contract differs')
    evaluation.validate_manifest(tx.rows(path('main_manifest')),'MAIN')
    prior,processes=_prior(release,terminal,start,arm,step,registry);process=_process()
    require(process not in processes,'Every MAIN endpoint requires a fresh process')
    store=identity=commit=None;source=registry.config['start_bindings'][str(start)]
    policy=source['policy_sha256'];checkpoint=None
    if arm!='START':
        commit=terminal.audit_spec['trusted_commits'][f'{start}/{arm}'][step-1];marker=read(bound(commit));identity=marker['identity']
        require(identity['train_seed']==start and identity['arm']==arm and
                identity['execution_sources_sha256']==backend.object_sha(terminal.original['execution_sources']),
                'MAIN cannot substitute original TRAIN identity')
        lock=read(path('source_lock'));native=fork_loader.load_native_ops(lock['native_src'],lock['native_files'])
        store=tx.TransactionStore(ROOT,ROOT/'DESIGN_FREEZE.json',FREEZE_SHA,identity,native=native,
            genesis_reference=marker['transaction']['genesis_reference'])
        require(store.inspect_committed(step,commit['sha256'])==marker,'Selected MAIN checkpoint changed')
        policy=marker['policy_sha256'];checkpoint=dict(commit=commit,state=ref(store.path/f'step-{step:03d}/state.pt'))
    endpoint={k:v for k,v in selection.items() if k!='scope'}|dict(policy_sha256=policy,reference_sha256=source['reference_sha256'])
    modules={'evaluation_runtime':runtime,'evaluation_loader':runtime.loader,'training_state':runtime.state,'training_transaction':tx}
    sources={name:dict(path=str(Path(module.__file__).resolve()),sha256=sha(Path(module.__file__).resolve())) for name,module in modules.items()}
    for value in sources.values():
        p=Path(value['path']);require(p.is_relative_to(terminal.package) and
            terminal.original['sources'][str(p.relative_to(terminal.package))]==value['sha256'],'Evaluation changed original library source')
    sources['authority']=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).resolve()))
    budget=storage._budget(release['storage_budget'],storage._now());require(budget.get('scope')=='MAIN_ONLY','Wrong MAIN storage scope')
    terminal.stable();prior.stable()
    return dict(release=release,release_ref=ref(Path(release_path)),terminal=terminal,freeze=freeze,registry=registry,
        selection=selection,requests=requests,manifest=manifest,count=count,endpoint=endpoint,sources=sources,
        sources_sha256=backend.object_sha(sources),start=start,arm=arm,step=step,directory=directory,process=process,
        prior=prior,store=store,identity=identity,commit=commit,checkpoint=checkpoint,budget=budget)


def _authority_type():
    class MainAuthority(runtime.AuthenticatedEvaluationAuthority):
        def __init__(self,ctx,lease):
            require(type(lease) is Lease,'Actual MAIN lease required');lease.active()
            self.ctx=ctx;self.lease=lease;self.index=0;self.claim=None;self.process=ctx['process'];self.output=ctx['directory']/'endpoint'
            super().__init__(ctx['sources']['authority'])
        def claim_value(self):
            c=self.ctx
            return dict(schema='sr-satdose-main-claim-v1',status='ATTEMPTED_UNKNOWN_NO_RETRY',start_seed=c['start'],arm=c['arm'],
                step=c['step'],responses=c['count'],release=c['release_ref'],terminal=c['terminal'].receipt,
                design_freeze_sha256=FREEZE_SHA,request_ids_sha256=backend.object_sha([r['request_id'] for r in c['requests']]),
                process_identity=self.process,output=str(self.output))
        def activate(self,reference):
            require(self.claim is None and read(bound(reference))==self.claim_value(),'MAIN claim differs/already active')
            self.claim=copy.deepcopy(reference)
        def fast(self):
            self.lease.active();require(self.claim is not None,'Missing MAIN claim');bound(self.claim);_hold_free()
            require(_process()==self.process,'MAIN process identity changed');c=self.ctx;bound(c['release_ref'])
            require(sha(ROOT/'DESIGN_FREEZE.json')==FREEZE_SHA,'Frozen MAIN design changed')
            for name in ('original_train_release','terminal_spec','storage_budget'):bound(c['release'][name])
            for base,files in ((Path(c['release']['external_root']),c['release']['external_sources']),
                               (c['terminal'].package,c['terminal'].original['sources'])):
                for name,digest in files.items():require(sha(base/name)==digest,'MAIN source changed')
            bound(c['terminal'].receipt)
            for value in c['release']['completed_endpoints'].values():bound(value)
            expected=set(c['release']['completed_endpoints'])|{endpoint_key(c['start'],c['arm'],c['step'])}
            require(set(_existing_slots())==expected,'Unexpected concurrent MAIN attempt')
        def boundary(self):
            self.fast();c=self.ctx;require(self.index in (0,c['count']),'Incomplete MAIN endpoint boundary')
            c['terminal'].stable();c['prior'].stable()
            _capture_inventory(self.output,[r['request_id'] for r in c['requests'][:self.index]])
            if c['store'] is not None:c['store'].inspect_committed(c['step'],c['commit']['sha256'])
        def verify(self,*,root,selection,endpoint,design_freeze_sha256,execution_sources_sha256,request_ids_sha256,request_id):
            self.fast();c=self.ctx
            require(root==ROOT and selection==c['selection'] and selection['scope']=='MAIN' and endpoint==c['endpoint'] and
                    design_freeze_sha256==FREEZE_SHA and execution_sources_sha256==c['sources_sha256'] and
                    request_ids_sha256==backend.object_sha([r['request_id'] for r in c['requests']]),'MAIN authority scope differs')
            admission_spec=read(bound(c['terminal'].audit_spec['admission_spec']))
            out=dict(schema='sr-satdose-authenticated-evaluation-authority-v1',selection=selection,endpoint=endpoint,
                design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA,execution_sources_sha256=execution_sources_sha256,
                request_ids_sha256=request_ids_sha256,request_id=request_id,admission=c['terminal'].audit_spec['admission'],
                registration=admission_spec['documents']['registration'],gpu_report=admission_spec['documents']['gpu_report'],
                terminal=c['terminal'].receipt)
            if request_id is not None:
                require(self.index<c['count'] and c['requests'][self.index]['request_id']==request_id,'Repeated/out-of-order MAIN request')
                if self.index:
                    previous=self.output/(hashlib.sha256(c['requests'][self.index-1]['request_id'].encode()).hexdigest()+'.COMPLETE.json')
                    require(previous.is_file(),'Prior MAIN draw remains UNKNOWN')
                begin=self.output/(hashlib.sha256(request_id.encode()).hexdigest()+'.BEGIN.json');out['begin']=ref(begin)
                require(read(begin)==dict(request_id=request_id,seed=c['requests'][self.index]['seed'],endpoint=endpoint,
                    design_freeze_sha256=FREEZE_SHA,status='ATTEMPTED_UNKNOWN'),'Original MAIN BEGIN differs')
                self.index+=1
            return out
    return MainAuthority


def storage_check(directory,budget_ref):
    require(directory.resolve(strict=True)==directory and directory.is_dir() and directory.is_relative_to(ROOT),'Noncanonical MAIN storage')
    started=storage._now();tick=time.monotonic();budget=storage._budget(budget_ref,started)
    require(budget.get('scope')=='MAIN_ONLY','Wrong MAIN storage scope')
    before=storage._observe(directory);require(before['effective_available_bytes']>=budget['required_bytes'],'Insufficient MAIN storage')
    probe=storage._probe(directory);after=storage._observe(directory);finished=storage._now();storage._budget(budget_ref,finished)
    require(min(before['effective_available_bytes'],after['effective_available_bytes'])>=budget['required_bytes'] and
            0<=time.monotonic()-tick<=budget['max_observation_age_seconds'] and
            (finished-started).total_seconds()<=budget['max_observation_age_seconds'],'Stale/insufficient MAIN storage')
    return dict(schema='sr-satdose-main-storage-v1',budget=budget_ref,observations=[before,after],probe=probe,
                checked_at_utc=finished.isoformat(),permission='MAIN_ONLY_NO_TRAINING')


def execute(release_path,release_sha,start,arm,step):
    # Terminal independent replay is already published; verification can occur
    # under this exclusive lease without requesting the auditor's shared lease.
    with Lease() as lease:
        ctx=preflight(release_path,release_sha,start,arm,step);first_two._allocation()
        authority=_authority_type()(ctx,lease);c=ctx;directory=c['directory']
        directory.mkdir(parents=True,exist_ok=False);tx.fsync_directory(directory.parent)
        output=directory/'endpoint';output.mkdir();tx.fsync_directory(directory)
        tx.write_once(directory/'CLAIM.json',authority.claim_value());authority.activate(ref(directory/'CLAIM.json'))
        handle=None
        try:
            tx.write_once(directory/'STORAGE.json',storage_check(output,c['release']['storage_budget']));authority.boundary()
            binding=dict(schema='sr-satdose-evaluation-endpoint-binding-v1',scope='MAIN',endpoint=c['endpoint'],requests=c['count'],
                request_ids_sha256=backend.object_sha([r['request_id'] for r in c['requests']]),design_freeze_sha256=FREEZE_SHA,
                config_sha256=CONFIG_SHA,source_lock_sha256=c['freeze']['artifacts']['source_lock']['sha256'],
                manifest_sha256=c['freeze']['artifacts']['main_manifest']['sha256'],execution_sources_sha256=c['sources_sha256'],
                checkpoint=c['checkpoint'])
            tx.write_once(directory/'ENDPOINT_BINDING.json',binding)
            handle=runtime.build_evaluation_runtime(c['registry'],bound(c['freeze']['artifacts']['source_lock']),start_seed=start,arm=arm,
                step=step,scope='MAIN',output_dir=output,authority=authority,authority_source_reference=c['sources']['authority'],
                execution_sources=c['sources'],expected_execution_sources_sha256=c['sources_sha256'],commit_store=c['store'],
                expected_identity=c['identity'],expected_commit_sha256=None if c['commit'] is None else c['commit']['sha256'])
            require(handle.endpoint()==c['endpoint'],'Actual MAIN model endpoint differs')
            bridge=backend.GenerationBridge(c['registry'],output,handle,handle.endpoint);complete=[];raw_path=output/'RAW.jsonl'
            descriptor=os.open(raw_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(descriptor,'wb') as raw:
                for manifest in c['manifest']:
                    # The frozen bridge independently retains its MAIN seal.
                    # This is the exact published, independently verified
                    # terminal document, not a controller-created boolean.
                    value=bridge.generate_one(manifest['request_id'],terminal=read(bound(c['terminal'].receipt)))
                    row=dict(value,**{k:v for k,v in manifest.items() if k!='request_id'},scope='MAIN',suffix_text=value['generated_text'])
                    raw.write(tx.canonical(row)+b'\n');raw.flush();os.fsync(raw.fileno())
                    complete.append(ref(output/(hashlib.sha256(manifest['request_id'].encode()).hexdigest()+'.COMPLETE.json')))
            after=handle.endpoint();closed=handle.close();handle=None
            require(after==c['endpoint'] and closed['attempted_requests']==c['count'] and
                    all(closed[k]==0 for k in ('optimizer_steps','scheduler_steps','parameter_updates')),'MAIN changed policy/count')
            authority.boundary();events=tx.rows(output/'accounting.jsonl');require(len(events)==2*c['count'],'MAIN accounting count differs')
            for i,manifest in enumerate(c['manifest']):
                generated=read(bound(complete[i]))
                for event,kind,count in zip(events[2*i:2*i+2],('completion_attempts','generated_tokens'),(1,len(generated['tokens']))):
                    require(event['kind']==kind and event['count']==count and event['metadata']['group_seed']==manifest['seed'],
                            'Actual MAIN accounting differs')
            tx.write_once(directory/'CLOSE.json',closed)
            receipt=dict(schema='sr-satdose-evaluation-completion-v1',status='EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION',
                scope='MAIN',endpoint_binding_sha256=sha(directory/'ENDPOINT_BINDING.json'),endpoint_before=c['endpoint'],endpoint_after=after,
                responses=c['count'],optimizer_created=False,scheduler_created=False,optimizer_steps=0,scheduler_steps=0,parameter_updates=0,
                extra_forward_sequences=0,raw=ref(raw_path),accounting=ref(output/'accounting.jsonl'),complete_files=complete)
            tx.write_once(output/'RECEIPT.json',receipt)
            result=dict(schema='sr-satdose-main-result-v1',status='MAIN_ENDPOINT_COMPLETE',permission='NO_TRAINING_OR_SUCCESSOR_EXPERIMENT',
                start_seed=start,arm=arm,step=step,responses=c['count'],design_freeze_sha256=FREEZE_SHA,release=c['release_ref'],
                terminal=c['terminal'].receipt,process_identity=c['process'],checkpoint=c['checkpoint'],receipt=ref(output/'RECEIPT.json'),
                endpoint_binding=ref(directory/'ENDPOINT_BINDING.json'),close=ref(directory/'CLOSE.json'))
            tx.write_once(directory/'RESULT.json',result);return result
        except BaseException as exc:
            if handle is not None:
                try:handle.close()
                except BaseException:pass
            tx.write_once(directory/'HOLD.json',dict(status='HOLD_MAIN_TECHNICAL',permission='NO_GENERATION_OR_TRAINING',
                start_seed=start,arm=arm,step=step,error_type=type(exc).__name__,error=str(exc),
                recovery='AUDIT_EXISTING_ATTEMPT_NO_RESAMPLING'))
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release-receipt',type=Path,required=True)
    p.add_argument('--release-sha256',required=True);p.add_argument('--start-seed',type=int,choices=(11031,11032),required=True)
    p.add_argument('--arm',choices=('START','T0','T1'),required=True);p.add_argument('--step',type=int,choices=(0,8,16,24),required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    try:
        value=execute(a.release_receipt,a.release_sha256,a.start_seed,a.arm,a.step) if a.execute else \
            preflight(a.release_receipt,a.release_sha256,a.start_seed,a.arm,a.step)
        if not a.execute:value=dict(status='MAIN_PREFLIGHT_COMPLETE',start_seed=a.start_seed,arm=a.arm,step=a.step,
                                    responses=value['count'],permission='NO_MODEL_CALLS')
    except Exception as exc:value=dict(status='HOLD_MAIN_CONTROL',permission='NO_GENERATION_OR_TRAINING',
                                      error_type=type(exc).__name__,error=str(exc))
    print(json.dumps(value,sort_keys=True,allow_nan=False));return 2 if value['status'].startswith('HOLD') else 0


if __name__=='__main__':raise SystemExit(main())
