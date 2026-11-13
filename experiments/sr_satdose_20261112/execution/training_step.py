"""Non-CLI single-step composition; no admission implementation or auto retries.

A production AuthenticatedAdmission implementation is deliberately NOT supplied.
Its independently audited source digest must be supplied by the trusted control
plane. It must verify the complete DRY/GPU/registration chain and, above step2,
all required canary evidence, not simply echo a PASS string. This adapter checks
its identity/receipt bindings again but cannot manufacture that external trust.
CPU fakes exercise composition only and are not production authorities.

TransactionStore alone invokes the one Adam and one scheduler callbacks. Original
128 response bytes and scores are immutable; production.capture_update executes
once, and its verified postclip gradients are copied without a second clip.
No model is constructed here. The caller owns verified runtime construction,
step2 restart/canary, actual quota checks, scheduling and final GPU admission.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import importlib
import inspect
import math
import os
import sys
from pathlib import Path

from . import production, training_runtime, training_state, training_transaction as tx
from runtime.backend import GenerationBridge, NativeKernelRunner, PUBLIC, object_sha
from analysis import calibration

require=training_state.require


class AuthenticatedAdmission(ABC):
    """Trust interface only. No production verifier exists in this module.

    verifier_reference is an independently trusted absolute source path/SHA,
    not a receipt produced by this adapter. The concrete verify method must be
    defined in that source file; inherited/decorated foreign implementations are
    rejected. An external audited control plane must pin this reference.
    """
    def __init__(self,verifier_reference):
        self.verifier_reference=copy.deepcopy(verifier_reference)
        AuthenticatedAdmission.check_source(self)

    def check_source(self):
        reference=self.verifier_reference
        require(getattr(self.verify,'__self__',None) is self and
                getattr(self.verify,'__func__',None) is type(self).verify,
                'Admission verifier method replaced after source audit')
        actual=Path(inspect.getsourcefile(type(self).verify) or '').resolve(strict=True)
        require(set(reference)=={'path','sha256'} and Path(reference['path']).is_absolute() and
                actual==Path(reference['path']) and not actual.is_symlink() and
                tx.file_sha(actual)==reference['sha256'],'Unauthenticated admission verifier source')

    @abstractmethod
    def verify(self,*,root,identity,design_freeze_sha256,target_step,sampled_endpoint):
        """Return authenticated scoped evidence refs after independent verification.

        Required keys: identity_sha256, design_freeze_sha256, target_step,
        sampled_endpoint, admission, registration, gpu_report; canary above step2.
        Each evidence reference is root-relative {path,sha256}. This is a security
        boundary, not permission to return self-authored status dictionaries.
        """
        raise PermissionError('No authenticated production admission verifier installed')


def _authorization(handle,authority,step,*,existing=None):
    require(isinstance(authority,AuthenticatedAdmission),'AuthenticatedAdmission required; plain gates are forbidden')
    AuthenticatedAdmission.check_source(authority)
    endpoint=handle.endpoint()
    require(type(step) is int and step==endpoint['step']+1 and 1<=step<=24,'Only exact next committed step may generate')
    evidence=(authority.verify(root=handle.registry.root,identity=copy.deepcopy(handle.path_identity),
        design_freeze_sha256=handle.registry.freeze_sha256,target_step=step,sampled_endpoint=copy.deepcopy(endpoint))
        if existing is None else copy.deepcopy(existing))
    require(isinstance(evidence,dict) and evidence.get('identity_sha256')==object_sha(handle.path_identity) and
            evidence.get('design_freeze_sha256')==handle.registry.freeze_sha256 and
            type(evidence.get('target_step')) is int and evidence['target_step']==step and
            evidence.get('sampled_endpoint')==endpoint,'Admission verification scope differs')
    root=handle.registry.root
    documents={name:tx.read_json(tx.bound(root,evidence[name])) for name in ('admission','registration','gpu_report')}
    frozen=tx.read_json(handle.registry.freeze_path)
    admission,registration,gpu=(documents[name] for name in ('admission','registration','gpu_report'))
    config_sha=frozen['config_sha256']
    require(admission.get('schema')=='sr-satdose-admission-v1' and
            admission.get('experiment_id')==training_state.EXPERIMENT and
            admission.get('status')=='ADMITTED_FIRST_TWO_STEPS_ONLY' and
            admission.get('permission')=='FOUR_PATHS_MAX_STEP_2' and admission.get('failures')==[] and
            admission.get('config_sha256')==config_sha and
            admission.get('registration_sha256')==evidence['registration']['sha256'] and
            admission.get('gpu_report_sha256')==evidence['gpu_report']['sha256'],'Admission receipt identity/bindings differ')
    require(registration.get('schema')=='sr-satdose-registration-v1' and registration.get('mode')=='production' and
            registration.get('experiment_id')==training_state.EXPERIMENT and registration.get('config_sha256')==config_sha and
            registration.get('artifacts',{}).get('design_freeze')==dict(path='DESIGN_FREEZE.json',sha256=handle.registry.freeze_sha256),
            'Registration is not bound to this freeze')
    require(gpu.get('schema')=='sr-satdose-gpu-verification-v1' and gpu.get('mode')=='production' and
            gpu.get('experiment_id')==training_state.EXPERIMENT and gpu.get('config_sha256')==config_sha and
            gpu.get('registration_sha256')==evidence['registration']['sha256'],'GPU report binding differs')
    gates={'admission':admission}
    if step>2:
        canary=tx.read_json(tx.bound(root,evidence['canary']))
        require(canary.get('status')=='CANARY_RESUME_PASS' and canary.get('experiment_id')==training_state.EXPERIMENT and
                canary.get('config_sha256')==config_sha and canary.get('design_freeze_sha256')==handle.registry.freeze_sha256,
                'Authenticated canary required above step2')
        gates['canary']=canary
    return copy.deepcopy(evidence),gates


def _verify_execution(handle,execution_sources):
    require(handle.path_identity.get('execution_sources_sha256')==object_sha(execution_sources),
            'Step implementation not bound to runtime identity')
    freeze=tx.read_json(handle.registry.freeze_path)
    directory=Path(training_runtime.frozen.__file__).resolve().parent
    require({p.name:tx.file_sha(p) for p in directory.glob('*.py') if p.is_file()}==freeze['runtime_sources'],
            'Actual frozen runtime source inventory changed')
    for name,module in tuple(sys.modules.items()):
        if name.startswith('runtime.'):
            expected=directory/(name.removeprefix('runtime.').replace('.','/')+'.py')
            require(getattr(module,'__file__',None) and Path(module.__file__).resolve()==expected,
                    'Shadowed actual runtime dependency: '+name)
    for key,module in [('training_step',__file__),('training_runtime',training_runtime.__file__),
                       ('training_state',training_state.__file__),('training_transaction',tx.__file__),
                       ('production',production.__file__),('calibration',calibration.__file__),
                       ('satdose_core',calibration.core.__file__)]:
        actual=Path(module).resolve();reference=execution_sources[key]
        require(Path(reference['path']).resolve()==actual and tx.file_sha(actual)==reference['sha256'],
                'Unbound current step dependency: '+key)
    path=calibration.PACKAGE/'config/source_bindings.json'
    reference=execution_sources['scorer_bindings']
    require(Path(reference['path']).resolve()==path.resolve() and tx.file_sha(path)==reference['sha256'],
            'Unbound native scorer pin inventory')


def _dependencies(handle,store):
    root=store.root;freeze=tx.read_json(store.freeze_path)
    lock=tx.read_json(tx.bound(root,freeze['artifacts']['source_lock']))
    native_root=Path(lock['native_src']).resolve();legacy_root=Path(lock['legacy_root']).resolve()
    for directory,key in ((native_root,'native_files'),(legacy_root,'legacy_files')):
        require({str(p.relative_to(directory)) for p in directory.rglob('*.py') if p.is_file()}==set(lock[key]),
                'Actual native/legacy inventory differs')
    sources={str((native_root/rel).resolve()):sha for rel,sha in lock['native_files'].items()}
    sources.update({str((legacy_root/rel).resolve()):sha for rel,sha in lock['legacy_files'].items()})
    for path,sha in sources.items():require(tx.file_sha(path)==sha,'Pinned native/legacy dependency changed')
    native=importlib.import_module('sr_f6.runtime');legacy=importlib.import_module('gpu_production_v2.backend')
    protocol=importlib.import_module('sr_f1.json_protocol')
    require(sources.get(str(Path(protocol.__file__).resolve()))==tx.file_sha(protocol.__file__),'Unbound JSON stopping helper')
    runner=NativeKernelRunner(handle.runtime,native,legacy,protocol.balanced_cut,source_hashes=sources)
    scorer,_=calibration.load_scorer(native_root)
    kernels=production.load_native_kernels(native_root,legacy_root)
    case_rows=tx.rows(tx.bound(root,freeze['artifacts']['private_cases']))
    cases={row['case_id']:row for row in case_rows}
    require(len(cases)==len(case_rows),'Duplicate private cases')
    groups=tx.rows(tx.bound(root,freeze['artifacts']['training_groups']))
    return dict(runner=runner,scorer=scorer,kernels=kernels,cases=cases,groups=groups)


def _join_score(original,request,case,scorer):
    """Never mutate CAPTURE/COMPLETE, and never send private truth to generation."""
    require(original['seed']==request['seed'] and original['model_input']=={k:request[k] for k in PUBLIC},
            'Original native seed/public input differs')
    row=copy.deepcopy(original)
    for key,value in request.items():
        if key in row:require(row[key]==value,'Frozen metadata would overwrite original COMPLETE: '+key)
        row[key]=copy.deepcopy(value)
    require(row['scope']=='TRAIN' and row['assistant_prefill']=='{','Training must use original unforced prefix')
    task=case['task']
    score=scorer(row['full_text'],task['world'],task['query'],finished=row['finish_reason']=='balanced',
                 finish_reason=row['finish_reason'])
    known=int(score['known_mask']['S'])
    geometry=calibration.core.geometry(task,case['audit'])
    selected=score['witness'].get('selected') if known else None
    authorised=calibration.core.exact_hit(selected,geometry['authorised']) if known else 0
    require(type(score['J']) is int and score['J'] in (0,1) and type(authorised) is int and authorised in (0,1) and
            (score['S'] if known else 0)==authorised,'Native score differs from independent authorised-set oracle')
    capture=Path(row['native_capture_path'])
    begin=capture.with_name(capture.name.replace('.CAPTURE.json','.BEGIN.json'))
    row.update(native_begin_sha256=tx.file_sha(begin),
               native_score=copy.deepcopy(score),J=score['J'],exact_authorised=authorised,
               suffix_text=row['generated_text'],native_finished_semantics='balanced_only')
    return row


def _accounting(handle,begin,expected_rows,*,generation=False):
    require(tx.rows(handle.runtime._satdose_output/'accounting.jsonl')==handle.runtime._satdose_accounting,
            'Persisted native accounting differs')
    events=copy.deepcopy(handle.runtime._satdose_accounting[begin:])
    totals={}
    for event in events:
        require(type(event['count']) is int and event['count']>=0,'Invalid native accounting count')
        key=(event['kind'] if generation else event['metadata'].get('purpose'))
        if generation:require(event['kind'] in ('completion_attempts','generated_tokens'),'Unexpected forward during rollout')
        else:require(event['kind']=='extra_forward_sequences' and key in ('training_policy','training_reference'),
                     'Unexpected accounting during production backward')
        totals[key]=totals.get(key,0)+event['count']
    if generation:require(totals.get('completion_attempts')==expected_rows,'Measured native rollout attempts differ')
    else:require(totals=={'training_policy':expected_rows,'training_reference':expected_rows},'Measured native forwards differ')
    return dict(events=events,measured_counts=totals)


def _tensor_file(path,value):
    import torch
    with path.open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    tx.fsync_directory(path.parent)


def run_step(handle,store,*,step,admission,storage_ready,execution_sources):
    """One next step, or verified idempotent existing COMMIT; no model construction.

    storage_ready(stage,step) MUST perform current quota/writable checks and
    return the existing TransactionStore storage-readiness schema with evidence.
    No default storage or admission provider is available. On failure, runtime
    is discarded; a durable transaction UNKNOWN is never automatically retried.
    """
    import torch
    require(isinstance(handle,training_runtime.TrainingHandle) and isinstance(store,tx.TransactionStore),
            'Actual TRAIN handle and transaction store required')
    require(isinstance(admission,AuthenticatedAdmission) and callable(storage_ready),
            'Authenticated admission and live storage callback required')
    require(type(step) is int and 1<=step<=24,'Step outside frozen training budget')
    AuthenticatedAdmission.check_source(admission)
    require(execution_sources.get('admission_verifier')==admission.verifier_reference,
            'Admission verifier must be pinned in original runtime execution identity')
    _verify_execution(handle,execution_sources);handle._store(store)
    directory=handle.output/'steps'/f'step-{step:03d}'
    require(handle.output.resolve()==handle.output and handle.output.is_relative_to(store.root) and
            not handle.output.is_relative_to(store.store),'Training capture audit must be outside published transaction tree')
    # Idempotency is owned by the store. Do not require an old endpoint or invoke
    # authority/generation/optimizer callbacks when a COMMIT already exists.
    authorization=None
    if not (store.path/f'step-{step:03d}'/'COMMIT.json').is_file():
        authorization,_=_authorization(handle,admission,step)
    context={};calls=dict(capture=0,optimizer=0,scheduler=0)

    def collect(stage,original_requests):
        require(len(original_requests)==128 and len({r['request_id'] for r in original_requests})==128,
                'Exactly 128 frozen requests required')
        evidence,gates=_authorization(handle,admission,step,existing=authorization)
        context['endpoint']=handle.endpoint();context['authorization']=evidence
        deps=_dependencies(handle,store);context['deps']=deps
        groups=[g for g in deps['groups'] if g['start_seed']==handle.start_seed and g['arm']==handle.arm and g['step']==step]
        require(len(groups)==16 and [rid for group in groups for rid in group['request_ids']]==
                [r['request_id'] for r in original_requests] and all(len(g['request_ids'])==8 for g in groups),
                'Frozen original 16G8 order differs')
        context['plan']=dict(scope='TRAIN',step=step,groups=copy.deepcopy(groups))
        directory.parent.mkdir(exist_ok=True)
        require(directory.parent.resolve()==directory.parent,'Noncanonical per-step audit path')
        directory.mkdir(exist_ok=False);tx.fsync_directory(directory.parent)
        tx.write_once(directory/'AUTHORIZATION.json',dict(evidence=evidence,verifier=admission.verifier_reference))
        bridge=GenerationBridge(handle.registry,directory,deps['runner'],handle.endpoint)
        raw=[];complete=[];begin=len(handle.runtime._satdose_accounting)
        with (directory/'RAW.jsonl').open('xb') as stream:
            for original_request in original_requests:
                request=handle.registry.request(original_request['request_id'])
                require(all(request.get(k)==v for k,v in original_request.items()),'Registry substituted frozen request')
                _authorization(handle,admission,step,existing=evidence)
                generated=bridge.generate_one(request['request_id'],**gates)
                row=_join_score(generated,request,deps['cases'][request['case_id']],deps['scorer'])
                stream.write(tx.canonical(row)+b'\n');stream.flush();os.fsync(stream.fileno())
                raw.append(row)
                complete.append(tx.reference(store.root,Path(row['native_capture_path']).with_name(
                    Path(row['native_capture_path']).name.replace('.CAPTURE.json','.COMPLETE.json'))))
        context['rows']=raw
        accounting=_accounting(handle,begin,128,generation=True)
        require(accounting['measured_counts'].get('generated_tokens')==sum(len(r['tokens']) for r in raw),
                'Measured original token count differs')
        tx.write_once(directory/'GENERATION_ACCOUNTING.json',accounting)
        raw_ref=tx.reference(store.root,directory/'RAW.jsonl')
        scores=dict(schema='sr-satdose-training-scores-v1',identity=handle.path_identity,step=step,raw=raw_ref,
            native_rescoring_complete=True,rows=[{k:r[k] for k in ('request_id','native_score','J','exact_authorised')} for r in raw])
        tx.write_once(directory/'SCORES.json',scores)
        receipt=dict(schema='sr-satdose-training-rollout-v1',status='COMPLETE_NOT_TRAINING_ADMISSION',
            identity=handle.path_identity,step=step,responses=128,design_freeze_sha256=store.freeze_sha,
            source_lock_sha256=store.source_lock['sha256'],raw=raw_ref,complete_files=complete,
            scores=tx.reference(store.root,directory/'SCORES.json'),authorization=tx.reference(store.root,directory/'AUTHORIZATION.json'))
        tx.write_once(directory/'ROLLOUT_COMPLETE.json',receipt)
        return tx.reference(store.root,directory/'ROLLOUT_COMPLETE.json')

    def prepare(stage,rollout_ref):
        calls['capture']+=1;require(calls['capture']==1,'Production capture may execute only once')
        require(handle.endpoint()==context['endpoint'],'Policy moved between rollout and production capture')
        begin=len(handle.runtime._satdose_accounting);storage_begin=len(handle.storage_forwards)
        kernels=context['deps']['kernels']
        produced=production.capture_update(handle.runtime.model,context['rows'],handle.forward_original,handle.fork.reference,
            kernels=kernels,config=handle.registry.config,identity=dict(handle.endpoint(),parent='b0'),plan=context['plan'])
        require(produced['status']=='NUMERICAL_CAPTURE_NOT_UPDATE_AUTHORIZATION' and produced['sequence_denominator']==128 and
                produced['row_forward_callback_calls']==produced['backward_calls']==128 and
                produced['optimizer_created'] is False and
                produced['optimizer_steps']==produced['scheduler_steps']==produced['parameter_updates']==0 and
                produced['policy_sha256']==context['endpoint']['policy_sha256'] and
                produced['reference_sha256']==context['endpoint']['reference_sha256'] and
                produced['source_sha256']==kernels.sources,
                'Incomplete independent production capture')
        accounting=_accounting(handle,begin,128)
        require(len(handle.storage_forwards)-storage_begin==128,'Missing saved-tensor row evidence')
        handle.verify_storage_after_backward()
        accounting['storage']=copy.deepcopy(handle.storage_forwards[storage_begin:])
        _tensor_file(directory/'PRODUCTION_CAPTURE.pt',produced)
        tx.write_once(directory/'PRODUCTION_ACCOUNTING.json',accounting)
        context['capture_ref']=tx.reference(store.root,directory/'PRODUCTION_CAPTURE.pt')
        context['rng']=handle.native_ops.capture_rng()
        precommit=dict(schema='sr-satdose-training-precommit-v1',identity=handle.path_identity,step=step,
            rollout_receipt=rollout_ref,gradient_stage='postclip',numerical_checks_passed=True,
            production_capture=context['capture_ref'],
            artifacts=[context['capture_ref'],tx.reference(store.root,directory/'PRODUCTION_ACCOUNTING.json'),
                       tx.reference(store.root,directory/'AUTHORIZATION.json')],rng_sha256=handle.native_ops.state_hash(context['rng']))
        tx.write_once(directory/'PRECOMMIT.json',precommit)
        return tx.reference(store.root,directory/'PRECOMMIT.json')

    def optimizer_step():
        calls['optimizer']+=1;require(calls['optimizer']==1 and calls['capture']==1,'Duplicate/unprepared native Adam')
        _authorization(handle,admission,step)  # External authority re-verifies before the sole Adam call.
        _verify_execution(handle,execution_sources)
        capture=torch.load(tx.bound(store.root,context['capture_ref']),map_location='cpu',weights_only=True)
        kernels=context['deps']['kernels'];kernels.check_numerics(capture['numerics'])
        require(handle.endpoint()==context['endpoint'],'Pre-Adam endpoint changed')
        native=handle.native_ops
        require(native.state_hash(native.capture_rng())==native.state_hash(context['rng']),'Pre-Adam RNG changed')
        native.validate_optimizer(handle.fork.optimizer,handle.fork.scheduler,model=handle.runtime.model)
        require(training_state.optimizer_parameter_order(handle.runtime.model,handle.fork.optimizer)==
                handle.path_identity['optimizer_parameter_order'],'Actual Adam parameter mapping differs')
        parameters={n:p for n,p in handle.runtime.model.named_parameters() if p.requires_grad}
        gradients=capture['postclip_gradients']
        require(list(parameters)==list(gradients) and len(parameters)==496,'Postclip parameter order/inventory differs')
        require(all(p.grad is None for p in handle.runtime.model.parameters()),'Preexisting gradient before Adam')
        context['policy_before']=native.trainable_state(handle.runtime.model)
        for name,param in parameters.items():
            value=gradients[name]
            require(value.dtype==torch.float32 and value.shape==param.shape and not value.requires_grad and
                    bool(torch.isfinite(value).all()),'Invalid verified postclip gradient')
            param.grad=value.to(param.device).clone()
            require(torch.equal(param.grad.detach().cpu(),value),'Postclip transfer changed gradient')
        try:handle.fork.optimizer.step()  # Already clipped by production.capture_update; never clip again.
        finally:handle.fork.optimizer.zero_grad(set_to_none=True)
        require(len(handle.fork.optimizer.state)==496 and
                all(float(v['step'])==step for v in handle.fork.optimizer.state.values()),'Actual Adam cursor differs')

    def scheduler_step():
        calls['scheduler']+=1;require(calls['scheduler']==1 and calls['optimizer']==1,'Duplicate/out-of-order native scheduler')
        handle.fork.scheduler.step()
        native=handle.native_ops
        require(handle.fork.scheduler.last_epoch==step,'Actual scheduler cursor differs')
        require(native.finite_tree(handle.fork.optimizer.state_dict()) and native.finite_tree(handle.fork.scheduler.state_dict()) and
                native.finite_tree(native.trainable_state(handle.runtime.model)),'Nonfinite post-Adam state')
        require(native.state_hash(handle.fork.reference)==context['endpoint']['reference_sha256'],'Original reference changed')
        require(native.state_hash(native.capture_rng())==native.state_hash(context['rng']),'Native update consumed RNG')
        after=native.trainable_state(handle.runtime.model)
        norm=math.sqrt(sum(float((after[n].double()-v.double()).square().sum()) for n,v in context['policy_before'].items()))
        tx.write_once(directory/'NATIVE_UPDATE.json',dict(schema='sr-satdose-training-native-update-v1',
            status='UPDATED_UNCOMMITTED_NOT_ADMISSION',
            identity=handle.path_identity,step=step,policy_before_sha256=context['endpoint']['policy_sha256'],
            policy_after_sha256=native.state_hash(after),reference_sha256=native.state_hash(handle.fork.reference),
            parameter_update_norm=norm,optimizer_calls=calls['optimizer'],scheduler_calls=calls['scheduler'],
            production_capture=context['capture_ref']))

    def checkpoint(stage,fields):
        require(calls==dict(capture=1,optimizer=1,scheduler=1),'Step did not execute exactly one native update')
        return training_state.write_checkpoint_files(stage,handle.runtime,handle.fork.optimizer,handle.fork.scheduler,
            handle.fork.reference,handle.native_ops,identity=handle.path_identity,step=step,
            sampler_state=dict(committed_step=step,next_step=step+1,generated_responses=step*128),
            commit_fields=dict(fields,native_update=tx.reference(store.root,directory/'NATIVE_UPDATE.json')))

    try:
        result=store.commit_step(step,collect_evidence=collect,prepare_update=prepare,optimizer_step=optimizer_step,
                                scheduler_step=scheduler_step,write_checkpoint=checkpoint,storage_ready=storage_ready)
        if result['status']=='COMMITTED':handle.advance_committed(store,step,result['commit_sha256'])
        return result
    except BaseException as exc:
        # Publication may have completed before an exception (including fsync or
        # cursor validation). Never poison immutable evidence of a valid COMMIT.
        # Existing COMMIT is not trusted here; the next read must fully inspect it.
        published=(store.path/f'step-{step:03d}'/'COMMIT.json').exists()
        failure_path=directory/('POST_PUBLICATION_FAILURE.json' if published else 'HOLD.json')
        try:
            if directory.exists() and not failure_path.exists():
                tx.write_once(failure_path,dict(status=('COMMIT_PRESENT_HANDLE_DISCARDED_REQUIRES_VALIDATION'
                    if published else 'HOLD_TRAINING_STEP_NO_RETRY'),
                    published_commit_requires_independent_validation=published,error_type=type(exc).__name__,
                    error=str(exc),calls=calls))
        finally:handle.discard()
        raise
