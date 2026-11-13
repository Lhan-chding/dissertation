"""Fixed step2 before/after process-reload teacher evidence, never admission.

No CLI, authority, launcher, generation or optimizer capability is provided.
The caller must independently authenticate the fixed per-path technical spec
and its SHA before calling either phase. Restored execution constructs its own
fresh runtime using the pinned builder and trusted committed state. Four-path
aggregation and step3 authorization remain external and unimplemented here.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import socket

from . import training_runtime as runtime, training_state as state, training_transaction as tx, training_step
from runtime.backend import object_sha

require=state.require
SCHEMA='sr-satdose-canary-replay-spec-v1'
BASELINE='STEP2_BASELINE_COMPLETE_NOT_RESUME_ADMISSION'
RESTORED='STEP2_RELOAD_BIT_EXACT_NOT_AGGREGATE_ADMISSION'


def _process_identity():
    """Linux process incarnation; no weak PID-only or platform fallback."""
    stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    ticks=int(stat[19]);require(boot and ticks>0,'Missing Linux process incarnation')
    return dict(host=socket.gethostname(),boot_id=boot,pid=os.getpid(),start_ticks=ticks)


class _LockedReadOnlyStore(tx.TransactionStore):
    """Private active-lease view: avoids reentrant flock inside canary_teacher.

    Parent holds the real transaction lock and verifies the full chain at both
    boundaries. Cached markers are copies, only usable within that process and
    active lease. No update/restore implementation or mutable callback is exposed.
    """
    def __init__(self,original,chain):
        self.__dict__.update(original.__dict__)
        self._canary_chain=copy.deepcopy(chain);self._canary_process=_process_identity();self._canary_active=True

    def inspect_committed(self,step,expected_commit_sha256):
        require(self._canary_active and _process_identity()==self._canary_process,'Expired or foreign canary lock lease')
        require(type(step) is int and 1<=step<=len(self._canary_chain) and
                self._canary_chain[step-1][0]==expected_commit_sha256,'Canary lease COMMIT differs')
        return copy.deepcopy(self._canary_chain[step-1][1])

    def commit_step(self,*args,**kwargs):
        raise PermissionError('Read-only canary lease cannot update')


@contextmanager
def _locked_chain(store):
    with tx.exclusive_lock(store.store):
        store._design();chain=store._chain()
        require(len(chain)==2,'Canary requires exactly two committed steps')
        lease=_LockedReadOnlyStore(store,chain)
        try:
            yield lease,copy.deepcopy(chain)
            store._design()
            require(store._chain()==chain,'Committed chain changed during locked canary')
        finally:lease._canary_active=False


def _paths(root,start,arm):
    require(start in (11031,11032) and arm in ('T0','T1'),'Foreign canary path')
    return root/'canary_replay'/f'start-{start}'/arm


def _sources(identity,sources):
    require(identity.get('execution_sources_sha256')==object_sha(sources),'Unbound original execution inventory')
    expected={'canary_replay':__file__,'training_runtime':runtime.__file__,'training_state':state.__file__,
              'training_transaction':tx.__file__,'training_step':training_step.__file__}
    require(set(expected)<=set(sources),'Incomplete canary execution source inventory')
    for key,ref in sources.items():
        path=Path(ref['path'])
        require(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink() and
                tx.file_sha(path)==ref['sha256'],'Changed canary source: '+key)
        if key in expected:require(path==Path(expected[key]).resolve(),'Shadowed canary source: '+key)


def _spec(store,path,expected_sha,sources):
    path=Path(path)
    base=_paths(store.root,store.start,store.arm)
    require(path==base/'TECHNICAL_EXECUTION.json' and path.resolve(strict=True)==path and
            not path.is_symlink() and tx.file_sha(path)==expected_sha,'Untrusted/nonfixed canary technical spec')
    spec=tx.read_json(path)
    require(spec.get('schema')==SCHEMA and spec.get('permission')=='READONLY_STEP2_CANARY_ONLY' and
            spec.get('identity')==store.identity and spec.get('design_freeze_sha256')==store.freeze_sha and
            spec.get('execution_sources')==sources,'Foreign canary spec identity/sources')
    _sources(store.identity,sources)
    for phase in ('baseline','restored'):
        require(spec['slots'][phase]==str((base/phase).relative_to(store.root)),'Canary slot may not be relocated')
    require(spec['restored_audit']==str((base/'restored/audit').relative_to(store.root)),'Restored audit must be fixed/fresh')
    audit=store.root/spec['baseline_audit']
    require(audit.resolve(strict=True)==audit and audit.is_relative_to(store.root) and
            not audit.is_relative_to(store.store) and not audit.is_relative_to(base),'Noncanonical original training audit')
    for step in (1,2):
        ref=spec['commits'][str(step)]
        require(tx.bound(store.root,ref)==store.path/f'step-{step:03d}/COMMIT.json','Canary trusted COMMIT path differs')
    return spec,base,tx.reference(store.root,path)


def _complete(root,slot,phase):
    require(not (slot/'HOLD.json').exists() and (slot/'COMPLETE.json').is_file(),
            'Occupied canary slot is HOLD/UNKNOWN; no automatic retry')
    result=tx.read_json(slot/'COMPLETE.json')
    require(result.get('schema')=='sr-satdose-canary-replay-result-v1' and result.get('phase')==phase and
            result.get('status')==(BASELINE if phase=='baseline' else RESTORED) and
            result.get('permission')=='NO_STEP3_OR_AGGREGATE_ADMISSION','Invalid canary COMPLETE')
    require(result.get('request_count')==128 and result.get('measured_counts')==
            {'training_policy':128,'training_reference':128} and
            result.get('identity',{}).get('arm')==slot.parent.name and
            'start-'+str(result.get('identity',{}).get('train_seed'))==slot.parent.parent.name,
            'Foreign/incomplete fixed-slot canary result')
    require(tx.bound(root,result['claim'])==slot/'CLAIM.json' and
            tx.bound(root,result['spec'])==slot.parent/'TECHNICAL_EXECUTION.json','Relocated canary claim/spec')
    claim=tx.read_json(tx.bound(root,result['claim']))
    require(claim.get('phase')==phase and claim.get('identity')==result.get('identity') and
            claim.get('process')==result.get('process') and claim.get('spec')==result.get('spec'),
            'Canary claim/result chain differs')
    tx.bound(root,result['spec'])
    for key,name in (('tensors','TEACHER.pt'),('state_identity','STATE_IDENTITY.json'),
                     ('accounting','ACCOUNTING.json'),('relocation','RELOCATION.json')):
        require(tx.bound(root,result[key])==slot/name,'Relocated canary phase evidence')
    for key in ('fork_load','raw'):tx.bound(root,result[key])
    if phase=='restored':
        require(tx.bound(root,result['baseline'])==slot.parent/'baseline/COMPLETE.json','Restored baseline reference differs')
    return result


def _no_unknown(root):
    for start in (11031,11032):
        for arm in ('T0','T1'):
            base=_paths(root,start,arm)
            for phase in ('baseline','restored'):
                slot=base/phase
                if slot.exists():
                    require(slot.resolve(strict=True)==slot and slot.is_dir(),'Noncanonical canary slot')
                    _complete(root,slot,phase)


def _state_identity(handle,marker):
    handle._live_commit(marker)
    model=handle.runtime.model;native=handle.native_ops
    require(all(p.grad is None for p in model.parameters()),'Canary cannot consume existing gradients')
    policy=native.trainable_state(model)
    require(len(policy)==496 and all(str(v.dtype)=='torch.float32' for v in policy.values()),'Complete FP32 LoRA identity required')
    order=state.optimizer_parameter_order(model,handle.fork.optimizer)
    require(order==handle.path_identity['optimizer_parameter_order'],'Actual optimizer order differs')
    values=dict(policy=policy,reference=handle.fork.reference,optimizer=handle.fork.optimizer.state_dict(),
                scheduler=handle.fork.scheduler.state_dict(),rng=native.capture_rng())
    hashes={key+'_sha256':native.state_hash(value) for key,value in values.items()}
    require(all(hashes[key]==marker[key] for key in hashes),'Live state does not match trusted step2')
    return dict(identity=copy.deepcopy(handle.path_identity),trainable_tensors=496,optimizer_parameter_order=order,
                model_training=model.training,**hashes)


def _originals(store,marker):
    directory=store.path/'step-002'
    evidence=tx.read_json(tx.bound(directory,marker['transaction']['evidence']))
    receipt=tx.read_json(tx.bound(store.root,evidence['rollout_receipt']))
    rows=tx.rows(tx.bound(store.root,receipt['raw']))
    require(len(rows)==128 and [r['request_id'] for r in rows]==[r['request_id'] for r in store.plans[2]],
            'Canary must preserve every original step2 row in frozen order')
    return rows,copy.deepcopy(receipt['raw'])


def _copy_once(source,destination,expected_sha):
    require(source.resolve(strict=True)==source and tx.file_sha(source)==expected_sha,'Relocation source changed')
    descriptor=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with source.open('rb') as src,os.fdopen(descriptor,'wb') as dst:
        while data:=src.read(1<<20):dst.write(data)
        dst.flush();os.fsync(dst.fileno())
    require(tx.file_sha(destination)==expected_sha,'Relocation changed original bytes')
    tx.fsync_directory(destination.parent)


def _relocate(handle,rows,phase):
    moved=[];receipt=[]
    if phase=='restored':
        destination=handle.output/'original_tokens';destination.mkdir();tx.fsync_directory(handle.output)
    for original in rows:
        row=copy.deepcopy(original);source=Path(row['native_capture_path'])
        begin=source.with_name(source.name.replace('.CAPTURE.json','.BEGIN.json'))
        for src,sha in ((source,row['native_capture_sha256']),(begin,row['native_begin_sha256'])):
            require(src.is_relative_to(handle.registry.root),'Original capture outside experiment')
            dst=destination/src.name if phase=='restored' else src
            if phase=='restored':_copy_once(src,dst,sha)
            else:require(tx.file_sha(src)==sha,'Original baseline capture changed')
            receipt.append(dict(source=tx.reference(handle.registry.root,src),destination=tx.reference(handle.registry.root,dst)))
        if phase=='restored':row['native_capture_path']=str(destination/source.name)
        require({k:v for k,v in row.items() if k!='native_capture_path'}==
                {k:v for k,v in original.items() if k!='native_capture_path'},'Relocation changed original row')
        moved.append(row)
    return moved,receipt


def _accounting(handle,begin):
    require(tx.rows(handle.runtime._satdose_output/'accounting.jsonl')==handle.runtime._satdose_accounting,
            'Persisted canary accounting differs')
    events=copy.deepcopy(handle.runtime._satdose_accounting[begin:]);counts={}
    for event in events:
        require(event['kind']=='extra_forward_sequences' and type(event['count']) is int and event['count']==1 and
                event['metadata']['execution_phase']=='CANARY_RESTORE_TEACHER' and
                event['metadata']['purpose'] in ('training_policy','training_reference'),'Canary generated or ran an unexpected kernel')
        purpose=event['metadata']['purpose'];counts[purpose]=counts.get(purpose,0)+event['count']
    require(counts=={'training_policy':128,'training_reference':128},'Incomplete measured fixed128 canary accounting')
    return dict(events=events,measured_counts=counts)


def _save_tensor(path,value):
    import torch
    with path.open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    tx.fsync_directory(path.parent)


def _compare(base,current):
    import torch
    require(len(base)==len(current)==128,'Canary tensor population differs')
    for left,right in zip(base,current,strict=True):
        require(set(left)==set(right),'Canary tensor schema differs')
        for key in left:
            if key in ('teacher_logprobs','reference_logprobs'):
                require(torch.is_tensor(left[key]) and torch.is_tensor(right[key]) and
                        left[key].dtype==right[key].dtype==torch.float32 and
                        torch.equal(left[key],right[key]),'Restored teacher/reference is not bit-exact')
            else:require(left[key]==right[key],'Canary original tokens/identity/accounting differ: '+key)


def _fork(handle,store,resume):
    path=handle.output/'FORK_LOAD.json';value=tx.read_json(path)
    genesis=tx.read_json(tx.bound(store.root,store.genesis_reference));source=store.identity['source_binding']
    expected=dict(start_seed=store.start,arm=store.arm,source_state_path=source['original_state_path'],
        source_state_sha256=source['state_sha256'],policy_sha256=source['policy_sha256'],
        reference_sha256=source['reference_sha256'],optimizer_created=True,source_optimizer_restored=False,trainable_tensors=496)
    require(value.get('identity')==store.identity and value.get('resume') is resume and
            all(value.get('fork_receipt',{}).get(k)==v for k,v in expected.items()),'Wrong actual loader/fresh R24 receipt')
    if not resume:require(value['fork_receipt']==genesis['fork_receipt'],'Baseline was not the original uninterrupted fork')
    return tx.reference(store.root,path)


def _phase(store,*,phase,spec_path,expected_spec_sha256,execution_sources,handle=None,registry=None,source_lock_path=None):
    require(type(store) is tx.TransactionStore,'Actual transaction owner required')
    require(phase in ('baseline','restored'),'Invalid canary phase')
    spec,base,spec_ref=_spec(store,spec_path,expected_spec_sha256,execution_sources)
    process=_process_identity();slot=base/phase;claimed=False;created=False
    try:
        with _locked_chain(store) as (lease,chain):
            _no_unknown(store.root)
            require(not slot.exists(),'Canary phase already occupied; no automatic retry')
            require([spec['commits'][str(n)]['sha256'] for n in (1,2)]==[x[0] for x in chain],
                    'Technical spec differs from trusted full COMMIT chain')
            baseline=None
            if phase=='baseline':
                require(isinstance(handle,runtime.TrainingHandle) and handle.output==store.root/spec['baseline_audit'],
                        'Baseline must use original uninterrupted step2 handle')
            else:
                baseline=_complete(store.root,base/'baseline','baseline')
                require(baseline['spec']==spec_ref and baseline['identity']==store.identity and baseline['process']!=process,
                        'Restored phase requires same trusted baseline and a different process incarnation')
                require(handle is None and registry is not None and source_lock_path is not None,'Reload cannot accept a reused handle')
                require(registry.root==store.root and registry.freeze_sha256==store.freeze_sha and
                        registry.freeze_path==store.freeze_path,'Restored registry differs before model construction')
                frozen=tx.read_json(store.freeze_path)
                require(Path(source_lock_path).resolve(strict=True)==tx.bound(store.root,frozen['artifacts']['source_lock']),
                        'Restored source lock differs before model construction')
            slot.mkdir();tx.fsync_directory(base);claimed=True
            claim=dict(schema='sr-satdose-canary-replay-claim-v1',phase=phase,identity=store.identity,
                spec=spec_ref,process=process,status='UNKNOWN_UNTIL_COMPLETE_NO_RETRY')
            tx.write_once(slot/'CLAIM.json',claim)
            if phase=='restored':
                output=store.root/spec['restored_audit'];output.mkdir();tx.fsync_directory(slot)
                handle=runtime.build_training_runtime(registry,source_lock_path,start_seed=store.start,arm=store.arm,
                    output_dir=output,execution_sources=execution_sources,commit_store=lease,resume_step=2,
                    expected_commit_sha256=chain[1][0],expected_identity=copy.deepcopy(store.identity))
                created=True
            handle._store(lease);training_step._verify_execution(handle,execution_sources)
            require(handle.endpoint()['step']==2 and handle._commit_sha==chain[1][0] and
                    handle._commit_dir==store.path/'step-002','Canary must start at exact live committed step2')
            fork=_fork(handle,store,phase=='restored');before=_state_identity(handle,chain[1][1])
            originals,raw_ref=_originals(store,chain[1][1]);rows,relocation=_relocate(handle,originals,phase)
            tx.write_once(slot/'RELOCATION.json',dict(rows=relocation,raw=raw_ref))
            index=len(handle.runtime._satdose_accounting);captured=[]
            for row in rows:
                captured.append(handle.canary_teacher(row,lease,sampled_commit_sha256=chain[0][0],teacher_commit_sha256=chain[1][0]))
            require(_state_identity(handle,chain[1][1])==before and _process_identity()==process,
                    'Canary changed live state or process')
            accounting=_accounting(handle,index)
            require(tx.reference(store.root,tx.bound(store.root,raw_ref))==raw_ref,'Original raw changed during canary')
            if baseline is not None:
                import torch
                previous=torch.load(tx.bound(store.root,baseline['tensors']),map_location='cpu',weights_only=True)
                _compare(previous,captured)
                require(tx.read_json(tx.bound(store.root,baseline['state_identity']))==before,
                        'Full496 policy/Adam/scheduler/RNG/reference identity differs after reload')
            _sources(store.identity,execution_sources)
            require(tx.file_sha(spec_path)==expected_spec_sha256,'Technical execution spec changed')
            # Recheck full immutable history before publishing COMPLETE, while locked.
            store._design();require(store._chain()==chain,'Original committed bytes changed during canary')
            _save_tensor(slot/'TEACHER.pt',captured)
            tx.write_once(slot/'STATE_IDENTITY.json',before);tx.write_once(slot/'ACCOUNTING.json',accounting)
            result=dict(schema='sr-satdose-canary-replay-result-v1',phase=phase,status=BASELINE if phase=='baseline' else RESTORED,
                permission='NO_STEP3_OR_AGGREGATE_ADMISSION',identity=store.identity,spec=spec_ref,process=process,
                request_count=len(captured),sampled_commit_sha256=chain[0][0],teacher_commit_sha256=chain[1][0],
                claim=tx.reference(store.root,slot/'CLAIM.json'),tensors=tx.reference(store.root,slot/'TEACHER.pt'),
                state_identity=tx.reference(store.root,slot/'STATE_IDENTITY.json'),accounting=tx.reference(store.root,slot/'ACCOUNTING.json'),
                relocation=tx.reference(store.root,slot/'RELOCATION.json'),fork_load=fork,raw=raw_ref,
                measured_counts=accounting['measured_counts'])
            if baseline is not None:result['baseline']=tx.reference(store.root,base/'baseline/COMPLETE.json')
            tx.write_once(slot/'COMPLETE.json',result)
        return result
    except BaseException as exc:
        try:
            if claimed and not (slot/'HOLD.json').exists():
                tx.write_once(slot/'HOLD.json',dict(status='HOLD_CANARY_NO_RETRY',error_type=type(exc).__name__,error=str(exc)))
        finally:
            if handle is not None:handle.discard()
        raise
    finally:
        if created and handle is not None:handle.discard()


def baseline(handle,store,*,spec_path,expected_spec_sha256,execution_sources):
    """Capture all128 on the uninterrupted committed step2 handle; no admission."""
    return _phase(store,phase='baseline',handle=handle,spec_path=spec_path,
                  expected_spec_sha256=expected_spec_sha256,execution_sources=execution_sources)


def restored(store,registry,source_lock_path,*,spec_path,expected_spec_sha256,execution_sources):
    """New process only; calls actual pinned builder with explicit trusted step2 restore."""
    return _phase(store,phase='restored',registry=registry,source_lock_path=source_lock_path,spec_path=spec_path,
                  expected_spec_sha256=expected_spec_sha256,execution_sources=execution_sources)
