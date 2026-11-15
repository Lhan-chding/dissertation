"""Fixed DEV2/8/16 after authenticated four-path canary; never TRAIN or MAIN.

This controller is part of the immutable original TRAIN package. Evaluation has
its own source identity while checkpoints retain the original TRAIN identity.
Every endpoint requires a fresh process, the global worker lease, actual current
COMMITs and a still-missing scheduled DEV endpoint. No occupied slot is retried.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

from . import evaluation_runtime as runtime, storage_ready as storage, first_two_control as first_two
from . import admission_authority, continuation_authority
from analysis import risk
from runtime import fork_loader
from runtime.backend import FrozenRegistry, GenerationBridge, object_sha

ROOT=Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
PACKAGE=Path(__file__).resolve().parents[1]
EXPERIMENT='SR-SATDOSE-1-20261112'
FREEZE_SHA='6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
CONFIG_SHA='58841639b0f36a69ac77824bc8c683ef7fbeb5c50a41e9975b89b54880f4ecfe'
PATHS=tuple(f'{s}/{a}' for s in (11031,11032) for a in ('T0','T1'))
tx=runtime.tx
require=tx.require


def sha(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),
            'Canonical regular file required')
    return tx.file_sha(path)


def ref(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))
def bound(reference):return tx.bound(ROOT,reference)
def read(path):return tx.read_json(path)
def rows(path):return tx.rows(path)


def _process_identity():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid=os.getpid(),start_ticks=int(fields[19]))


class Lease:
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


def _hold_free():
    for name in ('HOLD.json','ERROR.json'):
        require(not os.path.lexists(ROOT/name),'Global HOLD/error')
    for name in ('training_transactions','training_control','first_two_control','first_two_launcher',
                 'continuation_control','canary_replay','start_dev','trained_dev'):
        directory=ROOT/name
        if not directory.exists():continue
        require(directory.resolve(strict=True)==directory and directory.is_dir(),'Noncanonical control directory')
        for p in directory.rglob('*'):
            require(not p.is_symlink(),'Symlink attempt evidence')
            require('HOLD' not in p.name and 'ERROR' not in p.name and not p.name.startswith('.staging'),
                    'Global failed/unknown attempt')


def _first_two_complete(release,canary):
    """Keep successful canary evidence separate from live controller completion."""
    parent=ROOT/'first_two_control'
    require(parent.is_dir() and {p.name for p in parent.iterdir()}=={'start-11031','start-11032'},
            'Missing/extra first-two controller start')
    launcher=read(ROOT/'first_two_launcher/RESULT.json')
    require(launcher.get('status')=='FOUR_FIRST_TWO_PATHS_EXIT_OBSERVED_NOT_CANARY_ADMISSION' and
            launcher.get('release')==release['original_train_release'] and set(launcher.get('paths',{}))==set(PATHS),
            'First-two launcher remains incomplete/foreign')
    processes=[]
    for start_dir in parent.iterdir():
        require({p.name for p in start_dir.iterdir()}=={'T0','T1'},'Missing/extra first-two arm')
        start=int(start_dir.name.removeprefix('start-'))
        for arm_dir in start_dir.iterdir():
            require({p.name for p in arm_dir.iterdir()}=={'baseline','reload'},'Missing/extra first-two worker phase')
            key=f'{start}/{arm_dir.name}';launched=launcher['paths'][key]
            require(launched.get('status')=='FIRST_TWO_PATH_EXIT_OBSERVED_NOT_CANARY_ADMISSION' and
                    launched.get('process_exit_proof')==canary.spec['process_exit_proofs'][key],
                    'Launcher path differs from independently audited exit proof')
            for phase,status in (('baseline',first_two.BASELINE_STATUS),('reload',first_two.RELOAD_STATUS)):
                slot=arm_dir/phase;result=read(slot/'RESULT.json');claim=read(slot/'CLAIM.json')
                require(result.get('status')==status and result.get('release')==release['original_train_release'] and
                        result.get('start_seed')==start and result.get('arm')==arm_dir.name and
                        result.get('identity')==canary.identities[key] and
                        bound(launched[phase+'_result'])==slot/'RESULT.json',
                        'First-two worker incomplete/foreign')
                require(claim.get('schema')=='sr-satdose-first-two-claim-v1' and
                        claim.get('status')=='UNKNOWN_UNTIL_COMPLETE_NO_RETRY' and
                        claim.get('release')==release['original_train_release'] and claim.get('start_seed')==start and
                        claim.get('arm')==arm_dir.name and claim.get('phase')==phase and
                        claim.get('process')==result.get('process'),'First-two worker claim differs')
                processes.append(result['process'])
                for begin in slot.rglob('*.BEGIN.json'):
                    stem=begin.name.removesuffix('.BEGIN.json')
                    require((begin.parent/(stem+'.CAPTURE.json')).is_file() and
                            (begin.parent/(stem+'.COMPLETE.json')).is_file(),'Unresolved first-two generation attempt')
    return processes


def _continuation_complete(release,canary,step):
    expected={f'{key}/{first}-{last}' for key in PATHS for first,last in ((3,8),(9,16)) if last<=step}
    require(set(release['continuation_results'])==expected,'Missing/extra completed continuation references')
    parent=ROOT/'continuation_control';found=set();processes=[]
    if parent.exists():
        for sd in parent.iterdir():
            require(sd.name in ('start-11031','start-11032') and sd.is_dir(),'Unknown continuation start slot')
            start=int(sd.name.removeprefix('start-'))
            for ad in sd.iterdir():
                require(ad.name in ('T0','T1') and ad.is_dir(),'Unknown continuation arm slot')
                for slot in ad.iterdir():
                    require(slot.name in ('chunk-003-008','chunk-009-016') and slot.is_dir(),
                            'Unknown/future continuation chunk during DEV')
                    first,last=(int(x) for x in slot.name.removeprefix('chunk-').split('-'))
                    key=f'{start}/{ad.name}';name=f'{key}/{first}-{last}'
                    require(name in expected and bound(release['continuation_results'][name])==slot/'RESULT.json',
                            'Unknown/incomplete continuation chunk')
                    result=read(slot/'RESULT.json');claim=read(slot/'CLAIM.json')
                    require(result.get('schema')=='sr-satdose-continuation-result-v1' and
                            result.get('status')=='CONTINUATION_CHUNK_COMPLETE' and
                            result.get('permission')=='NO_AUTOMATIC_SUCCESSOR_OR_MAIN' and
                            result.get('start_seed')==start and result.get('arm')==ad.name and
                            result.get('first_step')==first and result.get('last_step')==last and
                            result.get('identity')==canary.identities[key] and
                            result.get('commits')==release['trusted_commits'][key][first-1:last] and
                            result.get('initial_commit')==release['trusted_commits'][key][first-2],
                            'Continuation result does not close original committed chunk')
                    require(claim.get('status')=='ATTEMPTED_UNKNOWN_NO_RETRY' and
                            claim.get('process')==result.get('process') and claim.get('release')==result.get('release'),
                            'Continuation claim/result process differs')
                    bound(result['release']);bound(result['post_risk']);processes.append(result['process']);found.add(name)
    require(found==expected,'Missing completed continuation controller chunk')
    return processes


def _slot(start,arm,step):
    require(type(start) is int and start in (11031,11032) and arm in ('T0','T1') and
            type(step) is int and step in (2,8,16),'Only fixed trained DEV2/8/16 endpoints')
    return ROOT/'trained_dev'/f'start-{start}'/arm/f'step-{step:03d}'


def _capture_inventory(directory,stems):
    for suffix in ('.BEGIN.json','.CAPTURE.json','.COMPLETE.json'):
        actual={str(p.relative_to(directory)).removesuffix(suffix) for p in directory.rglob('*'+suffix)}
        require(actual==set(stems),'Extra or unresolved DEV generation attempt')


def _attempts(endpoints,*,current=None):
    """Every existing endpoint must be supplied and complete, except live self."""
    parent=ROOT/'trained_dev';found=set()
    if not parent.exists():
        require(not any('/START/' not in key for key in endpoints),'Missing completed DEV directories')
        return
    for start_dir in parent.iterdir():
        require(start_dir.name in ('start-11031','start-11032') and start_dir.is_dir(),'Unknown DEV start slot')
        start=int(start_dir.name.removeprefix('start-'))
        for arm_dir in start_dir.iterdir():
            require(arm_dir.name in ('T0','T1') and arm_dir.is_dir(),'Unknown DEV arm slot')
            for slot in arm_dir.iterdir():
                require(slot.name in ('step-002','step-008','step-016') and slot.is_dir(),'Unknown DEV checkpoint slot')
                key=f'{start}/{arm_dir.name}/{int(slot.name.removeprefix("step-"))}'
                if current==slot:continue
                result=read(slot/'RESULT.json')
                require(result.get('status')=='TRAINED_DEV_ENDPOINT_COMPLETE' and key in endpoints and
                        result.get('start_seed')==start and result.get('arm')==arm_dir.name and
                        result.get('step')==int(slot.name.removeprefix('step-')) and result.get('responses')==108 and
                        result.get('design_freeze_sha256')==FREEZE_SHA,'Existing DEV incomplete or omitted')
                require(endpoints[key]==dict(receipt=result['receipt'],binding=result['endpoint_binding']),
                        'Prior DEV release references differ')
                require(bound(result['receipt'])==slot/'endpoint/RECEIPT.json' and
                        bound(result['endpoint_binding'])==slot/'ENDPOINT_BINDING.json' and
                        bound(result['close'])==slot/'CLOSE.json','Relocated prior DEV evidence')
                receipt=read(bound(result['receipt']))
                stems=[Path(r['path']).name.removesuffix('.COMPLETE.json') for r in receipt['complete_files']]
                require(len(stems)==len(set(stems))==108,'Prior DEV capture population differs')
                _capture_inventory(slot/'endpoint',stems)
                prior_release=read(bound(result['release']))
                require(prior_release.get('schema')=='sr-satdose-trained-dev-release-v1' and
                        prior_release.get('release_root')==str(PACKAGE) and
                        prior_release.get('design_freeze_sha256')==FREEZE_SHA and
                        prior_release.get('admission')==result['admission'],
                        'Prior endpoint release provenance differs')
                found.add(key)
    require(found=={key for key in endpoints if '/START/' not in key},'Invented or omitted prior DEV endpoint')


def verify_release(path,expected):
    path=Path(path)
    require(re.fullmatch('[0-9a-f]{64}',expected or '') and sha(path)==expected,'Untrusted trained DEV release')
    value=read(path)
    fields=dict(schema='sr-satdose-trained-dev-release-v1',status='AUDITED_FOR_TRAINED_DEV_ONLY',
        permission='TRAINED_DEV_ONLY_NO_TRAINING_OR_MAIN',experiment_id=EXPERIMENT,evidence_root=str(ROOT),
        release_root=str(PACKAGE),design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA)
    require(all(value.get(k)==v for k,v in fields.items()) and path.is_relative_to(ROOT) and
            not path.is_relative_to(PACKAGE),'Trained DEV release identity differs')
    original=first_two.verify_release(bound(value['original_train_release']),value['original_train_release']['sha256'])
    require(original['sources']==value['sources'] and original['release_root']==str(PACKAGE) and
            all(value[k]==original[k] for k in ('admission_spec','admission','risk_clarification')),
            'Original TRAIN source/admission identity changed')
    sources=original['execution_sources']
    expected_modules={'continuation_verifier':continuation_authority,'risk_verifier':risk,
                      'trained_dev_control':__import__(__name__,fromlist=['x'])}
    for name,module in expected_modules.items():
        p=Path(module.__file__).resolve()
        require(p.is_relative_to(PACKAGE) and sources.get(name)==dict(path=str(p),sha256=sha(p)) and
                value['sources'].get(str(p.relative_to(PACKAGE)))==sha(p),'Missing original TRAIN dependency pin: '+name)
    require(first_two.ROOT==storage.ROOT==ROOT and storage.FREEZE_SHA==FREEZE_SHA,'Foreign controller dependency root')
    return value,original


def _registry(freeze):
    path=lambda key:bound(freeze['artifacts'][key])
    return FrozenRegistry(path('config'),ROOT,ROOT/'DESIGN_FREEZE.json',
        [path(k) for k in ('train_manifest','dry_manifest','dev_manifest','prefix_calib_manifest','main_manifest')],
        path('inputs'),path('images'))


def _assess(release,freeze,original,endpoints):
    return risk.assess(ROOT,config_ref=freeze['artifacts']['config'],freeze_ref=dict(path='DESIGN_FREEZE.json',sha256=FREEZE_SHA),
        clarification_ref=release['risk_clarification'],trusted_commits=release['trusted_commits'],dev_endpoints=endpoints,
        expected_source_sha256=original['execution_sources']['risk_verifier']['sha256'])


def preflight(release_path,release_sha,start,arm,step):
    directory=_slot(start,arm,step);_hold_free()
    require(not os.path.lexists(directory),'Existing trained DEV attempt; never retry')
    release,original=verify_release(release_path,release_sha)
    _attempts(release['dev_endpoints'])
    current_process=_process_identity()
    for prior in (ROOT/'trained_dev').glob('start-*/*/step-*/RESULT.json'):
        require(read(prior).get('process_identity')!=current_process,'Each trained DEV requires a fresh process')
    spec=read(bound(release['admission_spec']))
    require(spec['sources']['authority']==original['execution_sources']['admission_verifier'] and
            spec['training_execution_sources_sha256']==object_sha(original['execution_sources']),
            'Original admission/TRAIN execution identity differs')
    base=admission_authority.FirstTwoStepAdmission(spec,admission=release['admission'])
    canary=continuation_authority.verify_canary(read(bound(release['canary_spec'])))
    require(canary.execution_sources==original['execution_sources'],'Canary belongs to another TRAIN release')
    require(current_process not in _first_two_complete(release,canary),'DEV must use a fresh post-canary process')
    freeze=read(ROOT/'DESIGN_FREEZE.json');registry=_registry(freeze)
    trusted=release['trusted_commits']
    require(set(trusted)==set(PATHS) and all(isinstance(v,list) and len(v)==step for v in trusted.values()),
            'All four paths must reach the same fixed DEV barrier')
    require(current_process not in _continuation_complete(release,canary,step),
            'DEV must use a fresh post-training process')
    decision=_assess(release,freeze,original,release['dev_endpoints'])
    key=f'{start}/{arm}/{step}'
    require(decision['global_hold'] is False and decision['status']=='PENDING_REQUIRED_DEV' and
            key in decision['pending_required_dev'],'Requested DEV is not an unresolved safe scheduled endpoint')
    require(all(name.rsplit('/',1)[-1]==str(step) for name in decision['pending_required_dev']),
            'Earlier scheduled DEV barrier remains incomplete')
    selected_ref=trusted[f'{start}/{arm}'][-1];marker=read(bound(selected_ref));identity=marker['identity']
    require(marker['step']==step and identity['train_seed']==start and identity['arm']==arm and
            identity['execution_sources_sha256']==object_sha(original['execution_sources']),
            'Selected checkpoint differs from original TRAIN identity')
    lock=read(bound(freeze['artifacts']['source_lock']))
    native=fork_loader.load_native_ops(lock['native_src'],lock['native_files'])
    store=tx.TransactionStore(ROOT,ROOT/'DESIGN_FREEZE.json',FREEZE_SHA,identity,native=native,
        genesis_reference=marker['transaction']['genesis_reference'])
    require(store.inspect_committed(step,selected_ref['sha256'])==marker,'Actual checkpoint changed')
    selection,requests=runtime._selection(registry,start,arm,step,'DEV')
    manifest=[r for r in rows(bound(freeze['artifacts']['dev_manifest']))
              if (r['start_seed'],r['arm'],r['step'])==(start,arm,step)]
    require(len(manifest)==108 and [r['request_id'] for r in manifest]==[r['request_id'] for r in requests],
            'Original DEV population/order differs')
    endpoint={k:v for k,v in selection.items() if k!='scope'}|dict(policy_sha256=marker['policy_sha256'],
        reference_sha256=marker['reference_sha256'])
    sources={name:dict(path=str(PACKAGE/'execution'/(name+'.py')),sha256=release['sources']['execution/'+name+'.py'])
             for name in ('evaluation_runtime','evaluation_loader','training_state','training_transaction')}
    sources['authority']=dict(path=str(Path(__file__).resolve()),sha256=release['sources']['execution/trained_dev_control.py'])
    budget=storage._budget(release['storage_budget'],storage._now())
    require(budget.get('scope')=='TRAINED_DEV_ONLY','Wrong trained DEV storage scope')
    base._observed.stable();canary.stable()
    return dict(release=release,original=original,release_ref=ref(Path(release_path)),spec=spec,base=base,canary=canary,
        freeze=freeze,registry=registry,selection=selection,requests=requests,manifest=manifest,endpoint=endpoint,
        sources=sources,sources_sha256=object_sha(sources),start=start,arm=arm,step=step,directory=directory,
        process=current_process,store=store,identity=identity,commit=selected_ref,marker=marker,risk=decision,budget=budget)


class TrainedDevAuthority(runtime.AuthenticatedEvaluationAuthority):
    def __init__(self,release_path,release_sha,start,arm,step,lease):
        require(type(lease) is Lease,'Actual global controller lease required');lease.active()
        self.lease=lease;self.ctx=preflight(release_path,release_sha,start,arm,step);self.index=0;self.claim=None
        self.process=self.ctx['process'];self.output=self.ctx['directory']/'endpoint'
        super().__init__(self.ctx['sources']['authority'])
    def _claim(self):
        c=self.ctx
        return dict(schema='sr-satdose-trained-dev-claim-v1',status='ATTEMPTED_UNKNOWN_NO_RETRY',
            start_seed=c['start'],arm=c['arm'],step=c['step'],responses=108,release=c['release_ref'],
            admission=c['release']['admission'],commit=c['commit'],design_freeze_sha256=FREEZE_SHA,
            request_ids_sha256=object_sha([r['request_id'] for r in c['requests']]),process_identity=self.process,output=str(self.output))
    def activate(self,reference):
        require(self.claim is None and read(bound(reference))==self._claim(),'Controller claim differs/already active')
        self.claim=copy.deepcopy(reference)
    def _fast(self):
        self.lease.active();require(self.claim is not None,'Missing live endpoint claim');bound(self.claim);_hold_free()
        require(_process_identity()==self.process,'Controller process identity changed')
        c=self.ctx;require(sha(ROOT/'DESIGN_FREEZE.json')==FREEZE_SHA,'Freeze changed');bound(c['release_ref'])
        for key in ('admission_spec','admission','storage_budget','original_train_release','canary_spec','risk_clarification'):
            bound(c['release'][key])
        for reference in c['spec']['documents'].values():bound(reference)
        for path,digest in c['release']['sources'].items():require(sha(PACKAGE/path)==digest,'Release source changed')
        for reference in c['spec']['sources'].values():require(sha(Path(reference['path']))==reference['sha256'],'Admission source changed')
        _first_two_complete(c['release'],c['canary'])
        _continuation_complete(c['release'],c['canary'],c['step'])
        for refs in c['release']['trusted_commits'].values():
            for reference in refs:bound(reference)
        for key,refs in c['release']['trusted_commits'].items():
            start,arm=key.split('/');directory=ROOT/'training_transactions'/f'start-{start}'/arm
            require({p.name for p in directory.iterdir()}=={f'step-{i:03d}' for i in range(1,len(refs)+1)},
                    'Training advanced during fixed DEV barrier')
        _attempts(c['release']['dev_endpoints'],current=c['directory'])
    def boundary(self):
        self._fast();c=self.ctx;c['base']._observed.stable();c['canary'].stable()
        require(self.index in (0,108),'Incomplete DEV endpoint boundary')
        _capture_inventory(self.output,[hashlib.sha256(r['request_id'].encode()).hexdigest() for r in c['requests'][:self.index]])
        require(c['store'].inspect_committed(c['step'],c['commit']['sha256'])==c['marker'],'Selected checkpoint changed')
        for path,digest in c['risk']['observed_files'].items():require(sha(Path(path))==digest,'Risk evidence changed')
    def verify(self,*,root,selection,endpoint,design_freeze_sha256,execution_sources_sha256,request_ids_sha256,request_id):
        self._fast();c=self.ctx
        require(root==ROOT and selection==c['selection'] and selection['scope']=='DEV' and
                selection['arm'] in ('T0','T1') and selection['step'] in (2,8,16) and endpoint==c['endpoint'] and
                design_freeze_sha256==FREEZE_SHA and execution_sources_sha256==c['sources_sha256'] and
                request_ids_sha256==object_sha([r['request_id'] for r in c['requests']]),'Trained DEV scope differs')
        out=dict(schema='sr-satdose-authenticated-evaluation-authority-v1',selection=selection,endpoint=endpoint,
            design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA,execution_sources_sha256=execution_sources_sha256,
            request_ids_sha256=request_ids_sha256,request_id=request_id,admission=c['release']['admission'],
            registration=c['spec']['documents']['registration'],gpu_report=c['spec']['documents']['gpu_report'])
        if request_id is not None:
            require(self.index<108 and c['requests'][self.index]['request_id']==request_id,'Repeated/out-of-order trained DEV request')
            if self.index:
                previous=self.output/(hashlib.sha256(c['requests'][self.index-1]['request_id'].encode()).hexdigest()+'.COMPLETE.json')
                require(previous.is_file(),'Previous draw remains UNKNOWN')
            begin=self.output/(hashlib.sha256(request_id.encode()).hexdigest()+'.BEGIN.json');out['begin']=ref(begin)
            require(read(begin)==dict(request_id=request_id,seed=c['requests'][self.index]['seed'],endpoint=endpoint,
                design_freeze_sha256=FREEZE_SHA,status='ATTEMPTED_UNKNOWN'),'Original trained DEV BEGIN differs')
            self.index+=1
        return out


def storage_check(directory,budget_ref):
    require(directory.resolve(strict=True)==directory and directory.is_dir() and directory.is_relative_to(ROOT),
            'Noncanonical DEV storage directory')
    started=storage._now();tick=time.monotonic();budget=storage._budget(budget_ref,started)
    require(budget.get('scope')=='TRAINED_DEV_ONLY','Wrong DEV storage scope')
    before=storage._observe(directory);require(before['effective_available_bytes']>=budget['required_bytes'],'Insufficient DEV storage')
    probe=storage._probe(directory);after=storage._observe(directory);finished=storage._now();storage._budget(budget_ref,finished)
    require(min(before['effective_available_bytes'],after['effective_available_bytes'])>=budget['required_bytes'] and
            0<=time.monotonic()-tick<=budget['max_observation_age_seconds'] and
            (finished-started).total_seconds()<=budget['max_observation_age_seconds'],'Stale/insufficient DEV storage')
    return dict(schema='sr-satdose-trained-dev-storage-v1',budget=budget_ref,observations=[before,after],probe=probe,
        checked_at_utc=finished.isoformat(),permission='TRAINED_DEV_ONLY_NO_TRAINING_OR_MAIN')


def execute(release_path,release_sha,start,arm,step):
    first_two._allocation()
    with Lease() as lease:
        authority=TrainedDevAuthority(release_path,release_sha,start,arm,step,lease);c=authority.ctx
        directory=c['directory'];directory.mkdir(parents=True,exist_ok=False);tx.fsync_directory(directory.parent)
        output=directory/'endpoint';output.mkdir();tx.fsync_directory(directory)
        tx.write_once(directory/'CLAIM.json',authority._claim());authority.activate(ref(directory/'CLAIM.json'))
        handle=None
        try:
            tx.write_once(directory/'STORAGE.json',storage_check(output,c['release']['storage_budget']))
            tx.write_once(directory/'PRE_RISK.json',c['risk']);authority.boundary()
            binding=dict(schema='sr-satdose-evaluation-endpoint-binding-v1',scope='DEV',endpoint=c['endpoint'],requests=108,
                request_ids_sha256=object_sha([r['request_id'] for r in c['requests']]),design_freeze_sha256=FREEZE_SHA,
                config_sha256=CONFIG_SHA,source_lock_sha256=c['freeze']['artifacts']['source_lock']['sha256'],
                manifest_sha256=c['freeze']['artifacts']['dev_manifest']['sha256'],execution_sources_sha256=c['sources_sha256'],
                checkpoint=dict(commit=c['commit'],state=ref(c['store'].path/f'step-{step:03d}/state.pt')))
            tx.write_once(directory/'ENDPOINT_BINDING.json',binding)
            handle=runtime.build_evaluation_runtime(c['registry'],bound(c['freeze']['artifacts']['source_lock']),
                start_seed=start,arm=arm,step=step,scope='DEV',output_dir=output,authority=authority,
                authority_source_reference=c['sources']['authority'],execution_sources=c['sources'],
                expected_execution_sources_sha256=c['sources_sha256'],commit_store=c['store'],expected_identity=c['identity'],
                expected_commit_sha256=c['commit']['sha256'])
            require(handle.endpoint()==c['endpoint'],'Constructed trained DEV identity differs')
            bridge=GenerationBridge(c['registry'],output,handle,handle.endpoint);completes=[];raw_path=output/'RAW.jsonl'
            fd=os.open(raw_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as raw:
                for manifest in c['manifest']:
                    value=bridge.generate_one(manifest['request_id'])
                    row=dict(value,**{k:v for k,v in manifest.items() if k!='request_id'},scope='DEV',suffix_text=value['generated_text'])
                    raw.write(tx.canonical(row)+b'\n');raw.flush();os.fsync(raw.fileno())
                    completes.append(ref(output/(hashlib.sha256(manifest['request_id'].encode()).hexdigest()+'.COMPLETE.json')))
            before=handle.endpoint();closed=handle.close();handle=None
            require(before==c['endpoint'] and closed['attempted_requests']==108 and
                    all(closed[k]==0 for k in ('optimizer_steps','scheduler_steps','parameter_updates')),'DEV changed policy/count')
            authority.boundary();verify_release(release_path,release_sha)
            events=rows(output/'accounting.jsonl');require(len(events)==216,'DEV accounting population differs')
            for index,manifest in enumerate(c['manifest']):
                generated=read(bound(completes[index]))
                for event,kind,count in zip(events[2*index:2*index+2],('completion_attempts','generated_tokens'),(1,len(generated['tokens']))):
                    require(event['kind']==kind and event['count']==count and event['metadata']['group_seed']==manifest['seed'],
                            'DEV native accounting differs')
            tx.write_once(directory/'CLOSE.json',closed)
            receipt=dict(schema='sr-satdose-evaluation-completion-v1',status='EVALUATION_ENDPOINT_CAPTURE_COMPLETE_NO_TRAINING_PERMISSION',
                scope='DEV',endpoint_binding_sha256=sha(directory/'ENDPOINT_BINDING.json'),endpoint_before=c['endpoint'],endpoint_after=before,
                responses=108,optimizer_created=False,scheduler_created=False,optimizer_steps=0,scheduler_steps=0,parameter_updates=0,
                extra_forward_sequences=0,raw=ref(raw_path),accounting=ref(output/'accounting.jsonl'),complete_files=completes)
            tx.write_once(output/'RECEIPT.json',receipt)
            endpoints=copy.deepcopy(c['release']['dev_endpoints'])
            endpoints[f'{start}/{arm}/{step}']=dict(receipt=ref(output/'RECEIPT.json'),binding=ref(directory/'ENDPOINT_BINDING.json'))
            post=_assess(c['release'],c['freeze'],c['original'],endpoints);tx.write_once(directory/'POST_RISK.json',post)
            result=dict(schema='sr-satdose-trained-dev-result-v1',status='TRAINED_DEV_ENDPOINT_COMPLETE',permission='NO_TRAINING_OR_MAIN',
                start_seed=start,arm=arm,step=step,responses=108,design_freeze_sha256=FREEZE_SHA,release=c['release_ref'],
                process_identity=authority.process,admission=c['release']['admission'],canary=c['canary'].receipt,commit=c['commit'],
                receipt=ref(output/'RECEIPT.json'),endpoint_binding=ref(directory/'ENDPOINT_BINDING.json'),close=ref(directory/'CLOSE.json'),
                post_risk=ref(directory/'POST_RISK.json'),global_scientific_hold=post['global_hold'])
            tx.write_once(directory/'RESULT.json',result)
            if post['global_hold']:
                control=ROOT/'training_control';control.mkdir(exist_ok=True)
                require(control.resolve(strict=True)==control,'Noncanonical global HOLD directory')
                tx.write_once(control/'HOLD.json',dict(status='GLOBAL_SCIENTIFIC_HOLD',hold_reasons=post['hold_reasons'],
                    evidence=ref(directory/'POST_RISK.json'),permission='NO_TRAINING_OR_MAIN'))
            return result
        except BaseException as exc:
            if handle is not None:
                try:handle.close()
                except BaseException:pass
            tx.write_once(directory/'HOLD.json',dict(status='HOLD_TRAINED_DEV_TECHNICAL',permission='NO_TRAINING_OR_MAIN',
                start_seed=start,arm=arm,step=step,error_type=type(exc).__name__,error=str(exc),
                recovery='AUDIT_EXISTING_ATTEMPT_NO_RESAMPLING'))
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-receipt',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    p.add_argument('--start-seed',type=int,choices=(11031,11032),required=True)
    p.add_argument('--arm',choices=('T0','T1'),required=True);p.add_argument('--step',type=int,choices=(2,8,16),required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    try:
        value=execute(a.release_receipt,a.release_sha256,a.start_seed,a.arm,a.step) if a.execute else \
            preflight(a.release_receipt,a.release_sha256,a.start_seed,a.arm,a.step)
        if not a.execute:value=dict(status='TRAINED_DEV_PREFLIGHT_COMPLETE',start_seed=a.start_seed,arm=a.arm,step=a.step,
                                    responses=108,permission='NO_MODEL_CALLS')
    except Exception as exc:value=dict(status='HOLD_TRAINED_DEV_CONTROL',permission='NO_TRAINING_OR_MAIN',
                                      error_type=type(exc).__name__,error=str(exc))
    print(json.dumps(value,sort_keys=True,allow_nan=False))
    return 2 if value['status'].startswith('HOLD') or value.get('global_scientific_hold') else 0


if __name__=='__main__':raise SystemExit(main())
