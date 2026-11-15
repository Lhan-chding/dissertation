"""One fresh process for one fixed TRAIN chunk; no implicit successor or MAIN.

The original first-two execution identity is retained. A separately SHA-pinned
invocation supplies exact existing COMMITs, completed DEV barriers and original
canary proof. Only this invocation's actual new COMMITs may extend that snapshot.
Risk assessment happens outside transaction locks after every committed update.
Every occupied incomplete attempt is UNKNOWN and cannot be automatically retried.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
from pathlib import Path
import re
import time

from . import (first_two_control as first_two, admission_authority, continuation_authority,
               canary_replay, training_runtime, training_step, training_transaction as tx,
               storage_ready as storage)
from analysis import risk
from runtime import fork_loader
from runtime.backend import object_sha

ROOT=first_two.ROOT
PACKAGE=Path(__file__).resolve().parents[1]
FREEZE_SHA=first_two.FREEZE_SHA
CONFIG_SHA=first_two.CONFIG_SHA
EXPERIMENT=tx.EXPERIMENT
PATHS=tuple(f'{s}/{a}' for s in (11031,11032) for a in ('T0','T1'))
CHUNKS={3:8,9:16,17:24}
STATUS='CONTINUATION_CHUNK_COMPLETE'
PERMISSION='NO_AUTOMATIC_SUCCESSOR_OR_MAIN'
require=tx.require


def sha(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),
            'Canonical ordinary file required')
    return tx.file_sha(path)


def ref(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha(path))
def bound(reference):return tx.bound(ROOT,reference)
def read(reference):return tx.read_json(bound(reference))


def _slot(start,arm,first):
    require(type(start) is int and start in (11031,11032) and arm in ('T0','T1') and
            type(first) is int and first in CHUNKS,'Only one registered path and fixed chunk3–8/9–16/17–24')
    return ROOT/'continuation_control'/f'start-{start}'/arm/f'chunk-{first:03d}-{CHUNKS[first]:03d}'


class Lease:
    def __init__(self):self.fd=None
    def __enter__(self):
        require(ROOT.resolve(strict=True)==ROOT,'Noncanonical evidence root')
        self.fd=os.open(ROOT/'DIAGNOSTIC_WORKER.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB);self.active()
        except BaseException:os.close(self.fd);self.fd=None;raise
        return self
    def active(self):
        require(self.fd is not None,'Missing sole-worker lease')
        path=ROOT/'DIAGNOSTIC_WORKER.lock';actual=path.stat();opened=os.fstat(self.fd)
        require(not path.is_symlink() and (actual.st_dev,actual.st_ino)==(opened.st_dev,opened.st_ino),
                'Sole-worker lease was replaced')
    def __exit__(self,*args):
        if self.fd is not None:os.close(self.fd);self.fd=None


def _hold_free():
    for name in ('HOLD.json','ERROR.json'):
        require(not os.path.lexists(ROOT/name),'Global technical HOLD/error')
    for name in ('training_transactions','training_control','first_two_control','first_two_launcher',
                 'canary_replay','start_dev','trained_dev','continuation_control'):
        directory=ROOT/name
        if not os.path.lexists(directory):continue
        require(directory.resolve(strict=True)==directory and directory.is_dir(),'Noncanonical control directory')
        for path in directory.rglob('*'):
            require(not path.is_symlink() and path.resolve(strict=True)==path,'Noncanonical control evidence')
            require('HOLD' not in path.name and 'ERROR' not in path.name and not path.name.startswith('.staging'),
                    'Global failed/unknown attempt')
        relocated={ROOT/f'canary_replay/start-{seed}/{arm}/restored/audit/original_tokens'
                   for seed in (11031,11032) for arm in ('T0','T1')}
        for begin in directory.rglob('*.BEGIN.json'):
            stem=begin.name.removesuffix('.BEGIN.json')
            # Readonly canary relocation copies original BEGIN+CAPTURE only.
            # ContinuingAdmission verifies these exact bytes against the original
            # independent audit on construction and every authority check.
            require((begin.parent/(stem+'.CAPTURE.json')).is_file() and
                    (begin.parent in relocated or (begin.parent/(stem+'.COMPLETE.json')).is_file()),
                    'Unresolved generation attempt')


def _cursor(trusted):
    require(isinstance(trusted,dict) and set(trusted)==set(PATHS),'Exact four committed paths required')
    for key,refs in trusted.items():
        require(isinstance(refs,list) and 2<=len(refs)<=24,'Invalid committed cursor')
        start,arm=key.split('/');directory=ROOT/f'training_transactions/start-{start}/{arm}'
        require(directory.resolve(strict=True)==directory and directory.is_dir(),'Missing canonical committed path')
        require({p.name for p in directory.iterdir()}=={f'step-{step:03d}' for step in range(1,len(refs)+1)},
                'Unknown transaction or published cursor changed')
        for step,reference in enumerate(refs,1):
            require(bound(reference)==directory/f'step-{step:03d}/COMMIT.json','Committed snapshot relocated or changed')


def _barrier(trusted,endpoints,start,arm,first):
    """All four prior DEV endpoints are required even while paths stagger."""
    boundary=first-1;last=CHUNKS[first];key=f'{start}/{arm}'
    require(set(trusted)==set(PATHS) and len(trusted[key])==boundary and
            all(isinstance(v,list) and len(v) in (boundary,last) for v in trusted.values()),
            'Chunk must start at exact boundary; other paths must be at boundary or completed chunk')
    required={f'{s}/START/0' for s in (11031,11032)}
    required|={f'{key}/{step}' for key in PATHS for step in (2,8,16) if step<=boundary}
    require(isinstance(endpoints,dict) and set(endpoints)==required,'All four prior DEV barriers must complete, without future endpoints')
    for key,refs in endpoints.items():
        require(set(refs)=={'receipt','binding'},'Malformed DEV evidence references')
        binding=read(refs['binding']);ep=binding['endpoint']
        require(binding.get('scope')=='DEV' and key==f"{ep['start_seed']}/{ep['arm']}/{ep['step']}",'DEV endpoint identity differs')
        bound(refs['receipt'])
        if ep['arm']!='START':
            require(binding['checkpoint']['commit']==trusted[f"{ep['start_seed']}/{ep['arm']}"][ep['step']-1],
                    'DEV checkpoint does not match actual committed prefix')


def _completed_attempts(trusted,endpoints,prior_chunks,*,current=None):
    """Require published controller completion in addition to transaction markers."""
    for family in ('start_dev','first_two_control'):
        parent=ROOT/family
        require(parent.is_dir() and {p.name for p in parent.iterdir()}=={'start-11031','start-11032'},
                'Missing/extra original controller start')
    for seed in (11031,11032):
        parent=ROOT/f'first_two_control/start-{seed}'
        require({p.name for p in parent.iterdir()}=={'T0','T1'},'Missing/extra first-two controller arm')
        for arm in ('T0','T1'):
            require({p.name for p in (parent/arm).iterdir()}=={'baseline','reload'},'Missing/extra first-two phase')
        result=tx.read_json(ROOT/f'start_dev/start-{seed}/RESULT.json')
        require(result.get('status')=='START_DEV_ENDPOINT_COMPLETE' and result.get('responses')==108 and
                endpoints[f'{seed}/START/0']==dict(receipt=result['receipt'],binding=result['endpoint_binding']),
                'Initial DEV controller incomplete')
    for key in PATHS:
        seed,arm=key.split('/')
        for phase,status in (('baseline',first_two.BASELINE_STATUS),('reload',first_two.RELOAD_STATUS)):
            result=tx.read_json(ROOT/f'first_two_control/start-{seed}/{arm}/{phase}/RESULT.json')
            require(result.get('status')==status and result.get('permission')=='NO_STEP3_OR_CANARY_ADMISSION',
                    'First-two worker incomplete')
    launcher=tx.read_json(ROOT/'first_two_launcher/RESULT.json')
    require(launcher.get('status')=='FOUR_FIRST_TWO_PATHS_EXIT_OBSERVED_NOT_CANARY_ADMISSION',
            'First-two external process observation incomplete')
    # All existing trained DEV attempts must be the externally supplied, complete population.
    found=set();parent=ROOT/'trained_dev'
    if parent.exists():
        for sd in parent.iterdir():
            require(sd.name in ('start-11031','start-11032') and sd.is_dir(),'Unknown trained DEV start')
            for ad in sd.iterdir():
                require(ad.name in ('T0','T1') and ad.is_dir(),'Unknown trained DEV arm')
                for directory in ad.iterdir():
                    require(directory.name in ('step-002','step-008','step-016') and directory.is_dir(),'Unknown trained DEV slot')
                    key=f'{sd.name[6:]}/{ad.name}/{int(directory.name[5:])}'
                    result=tx.read_json(directory/'RESULT.json')
                    require(result.get('status')=='TRAINED_DEV_ENDPOINT_COMPLETE' and result.get('responses')==108 and
                            key in endpoints and endpoints[key]==dict(receipt=result['receipt'],binding=result['endpoint_binding']),
                            'Unknown, omitted or incomplete trained DEV')
                    found.add(key)
    require(found=={k for k in endpoints if '/START/' not in k},'Missing trained DEV controller completion')
    require(isinstance(prior_chunks,dict),'Missing prior chunk references')
    found=set();parent=ROOT/'continuation_control'
    if parent.exists():
        for sd in parent.iterdir():
            require(sd.name in ('start-11031','start-11032') and sd.is_dir(),'Unknown continuation start')
            for ad in sd.iterdir():
                require(ad.name in ('T0','T1') and ad.is_dir(),'Unknown continuation arm')
                for directory in ad.iterdir():
                    require(directory.name in {f'chunk-{a:03d}-{b:03d}' for a,b in CHUNKS.items()} and directory.is_dir(),
                            'Unknown continuation chunk')
                    if directory==current:continue
                    name=f'{sd.name[6:]}/{ad.name}/{directory.name}'
                    require(name in prior_chunks,'Occupied untrusted continuation attempt; no retry')
                    require(bound(prior_chunks[name])==directory/'RESULT.json','Prior chunk receipt relocated')
                    result=read(prior_chunks[name]);a,b=map(int,directory.name.split('-')[1:])
                    key=f'{sd.name[6:]}/{ad.name}'
                    require(result.get('schema')=='sr-satdose-continuation-result-v1' and result.get('status')==STATUS and
                            result.get('permission')==PERMISSION and result.get('start_seed')==int(sd.name[6:]) and
                            result.get('arm')==ad.name and result.get('first_step')==a and result.get('last_step')==b and
                            result.get('commits')==trusted[key][a-1:b] and len(trusted[key])>=b,
                            'Prior chunk completion differs from committed prefix')
                    claim=tx.read_json(directory/'CLAIM.json')
                    require(claim.get('status')=='ATTEMPTED_UNKNOWN_NO_RETRY' and claim.get('process')==result.get('process') and
                            claim.get('release')==result.get('release') and result.get('initial_commit')==trusted[key][a-2] and
                            result.get('identity')==read(trusted[key][a-2])['identity'],
                            'Prior chunk claim/original identity differs')
                    bound(result['release']);bound(result['post_risk'])
                    found.add(name)
    require(found==set(prior_chunks),'Invented prior chunk completion')
    # Any published steps beyond two must belong to a complete earlier invocation,
    # except the exact current invocation while it is committing its own steps.
    for key,refs in trusted.items():
        for a,b in CHUNKS.items():
            if len(refs)<a:continue
            seed,arm=key.split('/');directory=_slot(int(seed),arm,a)
            if directory==current:continue
            require(len(refs)>=b and f'{key}/{directory.name}' in found,'Published steps lack complete chunk receipt')


def verify_release(path,expected):
    path=Path(path)
    require(re.fullmatch('[0-9a-f]{64}',expected or '') and sha(path)==expected and
            path.is_relative_to(ROOT) and not path.is_relative_to(PACKAGE),'Untrusted continuation release')
    release=tx.read_json(path)
    fields=dict(schema='sr-satdose-continuation-release-v1',status='AUDITED_FOR_ONE_FIXED_TRAIN_CHUNK',
        permission='ONE_FIXED_CHUNK_NO_MAIN',experiment_id=EXPERIMENT,evidence_root=str(ROOT),release_root=str(PACKAGE),
        design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA)
    require(all(release.get(k)==v for k,v in fields.items()),'Continuation release scope/identity differs')
    original=first_two.verify_release(bound(release['original_train_release']),release['original_train_release']['sha256'])
    require(release['sources']==original['sources'] and all(release[k]==original[k] for k in
            ('admission_spec','admission','risk_clarification')),'Original TRAIN source/admission identity changed')
    sources=original['execution_sources']
    modules={'continuation_control':__import__(__name__,fromlist=['x']),
             'continuation_verifier':continuation_authority,'risk_verifier':risk}
    for name,module in modules.items():
        p=Path(module.__file__).resolve()
        require(p.is_relative_to(PACKAGE) and sources.get(name)==dict(path=str(p),sha256=sha(p)) and
                release['sources'].get(str(p.relative_to(PACKAGE)))==sha(p),'Missing original TRAIN source pin: '+name)
    require(ROOT==first_two.ROOT==storage.ROOT and FREEZE_SHA==storage.FREEZE_SHA,'Foreign dependency root')
    return release,original


def _risk_inputs(release,original,freeze):
    return dict(config_ref=freeze['artifacts']['config'],freeze_ref=dict(path='DESIGN_FREEZE.json',sha256=FREEZE_SHA),
        clarification_ref=release['risk_clarification'],trusted_commits=copy.deepcopy(release['trusted_commits']),
        dev_endpoints=copy.deepcopy(release['dev_endpoints']),expected_source_sha256=original['execution_sources']['risk_verifier']['sha256'])


def _fresh(process):
    for family,pattern,field in (('first_two_control','start-*/*/*/RESULT.json','process'),
                                ('continuation_control','start-*/*/*/RESULT.json','process'),
                                ('trained_dev','start-*/*/*/RESULT.json','process_identity'),
                                ('start_dev','start-*/RESULT.json','process_identity')):
        for path in (ROOT/family).glob(pattern):
            require(tx.read_json(path).get(field)!=process,'Every chunk requires a fresh process incarnation')


def preflight(release_path,release_sha,start,arm,first):
    slot=_slot(start,arm,first);_hold_free()
    require(not os.path.lexists(slot),'Occupied continuation attempt; no automatic retry')
    release,original=verify_release(release_path,release_sha)
    require(release.get('selection')==dict(start_seed=start,arm=arm,first_step=first,last_step=CHUNKS[first]),
            'Invocation differs from externally pinned chunk')
    trusted=release['trusted_commits'];endpoints=release['dev_endpoints']
    _cursor(trusted);_barrier(trusted,endpoints,start,arm,first)
    _completed_attempts(trusted,endpoints,release['prior_chunks'])
    process=canary_replay._process_identity();_fresh(process)
    sources=original['execution_sources'];spec=read(release['admission_spec'])
    require(spec['root']==str(ROOT) and spec['sources']['authority']==sources['admission_verifier'] and
            spec['training_execution_sources_sha256']==object_sha(sources),'Original admission source identity differs')
    freeze=tx.read_json(ROOT/'DESIGN_FREEZE.json');registry=first_two._registry(freeze)
    first_two._risk(release['risk_clarification'])
    risk_inputs=_risk_inputs(release,original,freeze)
    authority=admission_authority.ContinuingAdmission(spec,admission=release['admission'],continuation_spec=dict(
        root=str(ROOT),helper=sources['continuation_verifier'],execution_sources=sources,
        canary_spec=read(release['canary_spec']),risk_inputs=risk_inputs))
    decision=authority.risk_snapshot();require(decision['global_hold'] is False,'Global scientific HOLD')
    initial=trusted[f'{start}/{arm}'][-1];marker=read(initial);identity=marker['identity']
    require(marker['step']==first-1 and identity['train_seed']==start and identity['arm']==arm and
            identity['execution_sources_sha256']==object_sha(sources),'Selected checkpoint identity differs')
    endpoint=dict(experiment_id=EXPERIMENT,start_seed=start,arm=arm,step=first-1,
                  policy_sha256=marker['policy_sha256'],reference_sha256=marker['reference_sha256'])
    authority.verify(root=ROOT,identity=identity,design_freeze_sha256=FREEZE_SHA,target_step=first,sampled_endpoint=endpoint)
    source_lock=bound(freeze['artifacts']['source_lock']);lock=tx.read_json(source_lock)
    native=fork_loader.load_native_ops(lock['native_src'],lock['native_files'])
    store=tx.TransactionStore(ROOT,ROOT/'DESIGN_FREEZE.json',FREEZE_SHA,identity,native=native,
                              genesis_reference=marker['transaction']['genesis_reference'])
    require(store.inspect_committed(first-1,initial['sha256'])==marker,'Actual resume checkpoint changed')
    budget=storage._budget(release['storage_budget'],storage._now())
    require(budget.get('scope')=='CONTINUATION_FIXED_CHUNK_ONLY','Wrong continuation storage scope')
    return dict(slot=slot,start=start,arm=arm,first=first,last=CHUNKS[first],release=release,original=original,
        release_ref=ref(release_path),sources=sources,identity=identity,initial=initial,endpoint=endpoint,
        registry=registry,source_lock=source_lock,store=store,authority=authority,risk_inputs=risk_inputs,
        risk=decision,process=process)


def _guard(ctx):
    ctx['lease'].active();bound(ctx['claim']);
    _hold_free();require(canary_replay._process_identity()==ctx['process'],'Controller process changed')
    verify_release(bound(ctx['release_ref']),ctx['release_ref']['sha256'])
    for key in ('admission_spec','admission','risk_clarification','canary_spec','storage_budget'):
        bound(ctx['release'][key])
    _cursor(ctx['risk_inputs']['trusted_commits'])
    _completed_attempts(ctx['risk_inputs']['trusted_commits'],ctx['risk_inputs']['dev_endpoints'],
                        ctx['release']['prior_chunks'],current=ctx['slot'])


def _pre_model_storage(directory,budget_ref):
    began=time.monotonic();budget=storage._budget(budget_ref,storage._now())
    require(budget.get('scope')=='CONTINUATION_FIXED_CHUNK_ONLY','Wrong continuation storage scope')
    before=storage._observe(directory)
    require(before['effective_available_bytes']>=budget['required_bytes'],'Insufficient pre-model quota')
    probe=storage._probe(directory);after=storage._observe(directory)
    storage._budget(budget_ref,storage._now())
    require(after['effective_available_bytes']>=budget['required_bytes'] and
            time.monotonic()-began<=budget['max_observation_age_seconds'],'Insufficient or stale quota after fsync')
    return dict(budget=budget_ref,observations=[before,after],probe=probe,permission='NO_TRAINING_ADMISSION')


def _hold(ctx,*,error=None,risk_ref=None):
    result=dict(status='HOLD_CONTINUATION_NO_RETRY',permission='NO_TRAINING_OR_MAIN',release=ctx['release_ref'],
        start_seed=ctx['start'],arm=ctx['arm'],first_step=ctx['first'],last_step=ctx['last'],
        risk=risk_ref,error_type=(type(error).__name__ if error is not None else None),
        error=(str(error) if error is not None else None))
    for path in (ctx['slot']/'HOLD.json',ROOT/'training_control/HOLD.json'):
        if not os.path.lexists(path):
            path.parent.mkdir(parents=True,exist_ok=True);tx.write_once(path,result)


def _run_chunk(handle,ctx):
    commits=[];risk_refs=[];store=ctx['store']
    require(handle.path_identity==ctx['identity'] and handle.endpoint()==ctx['endpoint'],
            'Restored live endpoint differs from original committed checkpoint')
    for step in range(ctx['first'],ctx['last']+1):
        _guard(ctx)
        require(handle.endpoint()['step']==step-1 and not os.path.lexists(store.path/f'step-{step:03d}'),
                'Only exact unattempted next step may generate')
        result=training_step.run_step(handle,store,step=step,admission=ctx['authority'],
            storage_ready=storage.make_storage_ready(ctx['release']['storage_budget']),execution_sources=ctx['sources'])
        require(result.get('status')=='COMMITTED' and handle.endpoint()['step']==step,'Step did not freshly commit')
        commit=ref(store.path/f'step-{step:03d}/COMMIT.json');marker=read(commit)
        require(result.get('commit_sha256')==commit['sha256'] and marker['identity']==ctx['identity'] and
                marker['step']==step and marker['sampler_state']['generated_responses']==128*step,
                'Returned COMMIT does not match actual fixed-budget update')
        commits.append(commit)
        ctx['risk_inputs']['trusted_commits'][f"{ctx['start']}/{ctx['arm']}"].append(commit)
        # No TransactionStore lock is held here. Never calculate risk inside its callbacks.
        decision=ctx['authority'].refresh_risk(copy.deepcopy(ctx['risk_inputs']))
        path=ctx['slot']/f'RISK_AFTER_STEP_{step:03d}.json';tx.write_once(path,decision);risk_refs.append(ref(path))
        if decision['global_hold']:
            _hold(ctx,risk_ref=risk_refs[-1]);raise PermissionError('Actual scientific risk triggered global HOLD')
        _guard(ctx)
    decision=ctx['authority'].risk_snapshot()
    pending=decision['pending_required_dev']
    if ctx['last'] in (8,16):
        require(f"{ctx['start']}/{ctx['arm']}/{ctx['last']}" in pending and
                all(int(key.rsplit('/',1)[1])==ctx['last'] for key in pending),
                'Completed chunk must stop at its actual required DEV barrier')
    else:require(not pending,'Unresolved DEV barrier at final TRAIN checkpoint')
    return dict(schema='sr-satdose-continuation-result-v1',status=STATUS,permission=PERMISSION,
        start_seed=ctx['start'],arm=ctx['arm'],first_step=ctx['first'],last_step=ctx['last'],identity=ctx['identity'],
        process=ctx['process'],release=ctx['release_ref'],initial_commit=ctx['initial'],commits=commits,
        train_responses=128*len(commits),formal_optimizer_steps=len(commits),scheduler_steps=len(commits),
        post_risk=risk_refs[-1],risk_assessments=risk_refs,pending_required_dev=pending,
        original_train_release=ctx['release']['original_train_release'],process_exit_observed=False,main_authorized=False)


def execute(release_path,release_sha,start,arm,first):
    first_two._allocation()
    with Lease() as lease:
        ctx=preflight(release_path,release_sha,start,arm,first);slot=ctx['slot']
        slot.mkdir(parents=True,exist_ok=False);tx.fsync_directory(slot.parent)
        tx.write_once(slot/'CLAIM.json',dict(schema='sr-satdose-continuation-claim-v1',status='ATTEMPTED_UNKNOWN_NO_RETRY',
            selection=ctx['release']['selection'],process=ctx['process'],release=ctx['release_ref'],initial_commit=ctx['initial'],
            allocation=first_two._allocation()))
        ctx['claim']=ref(slot/'CLAIM.json');ctx['lease']=lease
        handle=None
        try:
            tx.write_once(slot/'STORAGE.json',_pre_model_storage(slot,ctx['release']['storage_budget']))
            _guard(ctx)
            output=slot/'audit';output.mkdir();tx.fsync_directory(slot)
            handle=training_runtime.build_training_runtime(ctx['registry'],ctx['source_lock'],start_seed=start,arm=arm,
                output_dir=output,execution_sources=ctx['sources'],commit_store=ctx['store'],resume_step=first-1,
                expected_commit_sha256=ctx['initial']['sha256'],expected_identity=ctx['identity'])
            result=_run_chunk(handle,ctx);handle.discard();handle=None
            _guard(ctx);tx.write_once(slot/'RESULT.json',result);return result
        except BaseException as exc:
            try:_hold(ctx,error=exc)
            finally:
                if handle is not None:handle.discard()
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-receipt',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    parser.add_argument('--start-seed',type=int,choices=(11031,11032),required=True)
    parser.add_argument('--arm',choices=('T0','T1'),required=True)
    parser.add_argument('--first-step',type=int,choices=tuple(CHUNKS),required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    try:
        value=(execute if args.execute else preflight)(args.release_receipt,args.release_sha256,args.start_seed,args.arm,args.first_step)
        if not args.execute:value=dict(status='CONTINUATION_PREFLIGHT_COMPLETE_NO_MODEL_CALLS',permission='NO_TRAINING_OR_MAIN')
    except Exception as exc:
        value=dict(status='HOLD_CONTINUATION_CONTROL',permission='NO_TRAINING_OR_MAIN',error_type=type(exc).__name__,error=str(exc))
    print(tx.canonical(value).decode());return 2 if value['status'].startswith('HOLD') else 0


if __name__=='__main__':raise SystemExit(main())
