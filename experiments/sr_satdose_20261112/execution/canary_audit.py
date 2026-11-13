"""Independent read-only four-path canary audit; never publish admission.

Trusted spec/source digests are input, not discovered as 'latest'. CPU-loaded
state and teacher tensors are revalidated. GPU frozen-parameter and process-exit
proofs are source-bound producer observations, not second offline observations
of GPU weights or operating-system process death. Missing either proof family
returns PENDING, never the positive four-path evidence status.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import sys

from . import canary_replay as replay, training_state as state, training_transaction as tx, training_runtime as runtime
from runtime.backend import object_sha
from runtime import fork_loader

PATHS=tuple(f'{start}/{arm}' for start in (11031,11032) for arm in ('T0','T1'))
WITNESS_SCHEMA='sr-satdose-first2-frozen-parameters-v1'
EXIT_SCHEMA='sr-satdose-canary-baseline-exit-v1'
require=state.require


class _Evidence:
    def __init__(self,root):self.root=root;self.observed={}
    def path(self,path,expected=None):
        path=Path(path)
        require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Noncanonical canary evidence')
        actual=tx.file_sha(path)
        require(expected is None or actual==expected,'Canary evidence SHA differs')
        require(path not in self.observed or self.observed[path]==actual,'Canary evidence changed during audit')
        self.observed[path]=actual;return path
    def ref(self,ref):return self.path(tx.bound(self.root,ref),ref['sha256'])
    def json(self,ref):return tx.read_json(self.ref(ref))
    def tensor(self,path,expected=None):
        import torch
        return torch.load(self.path(path,expected),map_location='cpu',weights_only=True)
    def stable(self):
        require(all(tx.file_sha(path)==sha for path,sha in self.observed.items()),'Evidence changed during canary audit')


def _self(expected):
    path=Path(__file__).resolve()
    require(re.fullmatch('[0-9a-f]{64}',expected or '') and tx.file_sha(path)==expected,'Independent canary auditor source SHA differs')
    return dict(path=str(path),sha256=expected)


def _sources(sources,freeze,evidence):
    required={'training_runtime':runtime.__file__,'training_state':state.__file__,
              'training_transaction':tx.__file__,'canary_replay':replay.__file__}
    require(set(required)<=set(sources),'Missing actual canary producer dependencies')
    for name,ref in sources.items():
        path=evidence.path(Path(ref['path']),ref['sha256'])
        if name in required:require(path==Path(required[name]).resolve(),'Shadowed canary producer dependency')
    directory=Path(fork_loader.__file__).resolve().parent
    actual={p.name:tx.file_sha(p) for p in directory.glob('*.py') if p.is_file()}
    require(actual==freeze['runtime_sources'],'Actual frozen runtime inventory differs')
    for path in directory.glob('*.py'):evidence.path(path,actual[path.name])
    for name,module in tuple(sys.modules.items()):
        if name.startswith('runtime.'):
            require(getattr(module,'__file__',None) and Path(module.__file__).resolve()==
                    directory/(name.removeprefix('runtime.').replace('.','/')+'.py'),'Shadowed frozen runtime import')


def _native(lock,evidence):
    root=Path(lock['native_src'])
    require(root.resolve(strict=True)==root,'Noncanonical native source root')
    require({str(p.relative_to(root)) for p in root.rglob('*.py') if p.is_file()}==set(lock['native_files']),
            'Incomplete native source inventory')
    for rel,sha in lock['native_files'].items():evidence.path(root/rel,sha)
    ops=fork_loader.load_native_ops(root,lock['native_files'])
    return ops


@contextmanager
def _existing_lock(root):
    path=root/'training_transactions/TRANSACTION.lock'
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Existing transaction lock required')
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(descriptor,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise PermissionError('Active training transaction; no canary audit') from exc
        yield
    finally:os.close(descriptor)


def _process(value):
    require(isinstance(value,dict) and set(value)=={'host','boot_id','pid','start_ticks'} and
            all(isinstance(value[k],str) and value[k] for k in ('host','boot_id')) and
            all(type(value[k]) is int and value[k]>0 for k in ('pid','start_ticks')),'Invalid process incarnation')
    return value


def _same(left,right):
    import torch
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and set(left)==set(right) and all(_same(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left) is type(right) and len(left)==len(right) and all(_same(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left==right


def _delta(before,after,order):
    import torch
    names=[p['name'] for group in order for p in group]
    require(len(names)==496 and set(before)==set(after)==set(names),'Full496 policy delta inventory differs')
    terms=[];changed=0
    for name in names:
        a,b=before[name],after[name]
        require(torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype==torch.float32 and
                a.shape==b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'Invalid actual policy delta tensor')
        terms.append(float((b.double()-a.double()).square().sum()));changed+=int(not torch.equal(a,b))
    norm=math.sqrt(sum(terms))
    require(changed>0 and math.isfinite(norm) and norm>0,'Actual committed policy update is zero/nonfinite')
    return dict(parameter_update_norm=norm,changed_tensors=changed,compared_tensors=496)


def _teacher(values,rows,prior,current,commits):
    import torch
    require(isinstance(values,list) and len(values)==len(rows)==128,'Canary must contain original full128 tensors')
    expected_events=[]
    sampled=dict(experiment_id=state.EXPERIMENT,start_seed=current['identity']['train_seed'],arm=current['identity']['arm'],
        step=1,policy_sha256=prior['policy_sha256'],reference_sha256=prior['reference_sha256'])
    teacher=dict(sampled,step=2,policy_sha256=current['policy_sha256'])
    for value,row in zip(values,rows,strict=True):
        require(value.get('status')=='CANARY_TEACHER_CAPTURE_NOT_RESUME_ADMISSION' and value.get('request_id')==row['request_id'] and
                value.get('tokens')==row['tokens'] and value.get('sampler_logprobs')==row['sampler_logprobs'] and
                value.get('sampled_endpoint')==sampled and value.get('teacher_endpoint')==teacher and
                value.get('sampled_commit_sha256')==commits[0] and value.get('teacher_commit_sha256')==commits[1] and
                all(type(value.get(k)) is int and value[k]==0 for k in ('backward_calls','optimizer_steps','new_generations')),
                'Canary substituted original token/order/endpoint/counters')
        for key in ('teacher_logprobs','reference_logprobs'):
            tensor=value[key]
            require(torch.is_tensor(tensor) and tensor.device.type=='cpu' and tensor.dtype==torch.float32 and
                    tensor.shape==(len(row['tokens']),) and bool(torch.isfinite(tensor).all()),'Invalid actual canary teacher tensor')
        events=value['accounting']
        require(isinstance(events,list) and len(events)==2 and
                {e['metadata'].get('purpose') for e in events}=={'training_policy','training_reference'} and
                all(e.get('kind')=='extra_forward_sequences' and type(e.get('count')) is int and e['count']==1 and
                e['metadata'].get('execution_phase')=='CANARY_RESTORE_TEACHER' for e in events),'Canary accounting permits unexpected model work')
        expected_events.extend(events)
    return expected_events


def _phase(e,store,spec,spec_ref,phase,rows,raw_ref,chain):
    slot=e.root/spec['slots'][phase];result=replay._complete(e.root,slot,phase)
    result_ref=tx.reference(e.root,slot/'COMPLETE.json');e.ref(result_ref)
    require(result['identity']==store.identity and result['spec']==spec_ref and result['raw']==raw_ref and
            result['sampled_commit_sha256']==chain[0][0] and result['teacher_commit_sha256']==chain[1][0],
            'Canary result differs from trusted spec/commits/raw')
    _process(result['process']);claim=e.json(result['claim'])
    require(claim['schema']=='sr-satdose-canary-replay-claim-v1' and claim['status']=='UNKNOWN_UNTIL_COMPLETE_NO_RETRY',
            'Canary claim schema differs')
    for key in ('tensors','state_identity','accounting','relocation','fork_load','raw'):e.ref(result[key])
    snapshot=e.json(result['state_identity']);marker=chain[1][1]
    require(snapshot['identity']==store.identity and snapshot['trainable_tensors']==496 and
            snapshot['optimizer_parameter_order']==store.identity['optimizer_parameter_order'] and
            type(snapshot['model_training']) is bool and all(snapshot[key+'_sha256']==marker[key+'_sha256']
                for key in ('policy','optimizer','scheduler','rng','reference')),'Canary observed state identity differs from actual committed state')
    expected_audit=e.root/spec['baseline_audit'] if phase=='baseline' else e.root/spec['restored_audit']
    require(e.ref(result['fork_load'])==expected_audit/'FORK_LOAD.json','Canary loader receipt relocated')
    fork=e.json(result['fork_load']);genesis=e.json(store.genesis_reference);source=store.identity['source_binding']
    require(fork['identity']==store.identity and fork['resume'] is (phase=='restored'),'Wrong fresh/resumed constructor identity')
    expected=dict(schema='sr-satdose-fork-v1',experiment_id=state.EXPERIMENT,start_seed=store.start,arm=store.arm,
        source_state_path=source['original_state_path'],source_state_sha256=source['state_sha256'],
        policy_sha256=source['policy_sha256'],reference_sha256=source['reference_sha256'],optimizer_created=True,
        source_optimizer_restored=False,trainable_tensors=496,model_calls=0,optimizer_steps=0)
    require(all(fork['fork_receipt'].get(k)==v for k,v in expected.items()),'Canary fresh R24/original reference fork differs')
    if phase=='baseline':require(fork['fork_receipt']==genesis['fork_receipt'],'Baseline not original fresh fork')
    moved=e.json(result['relocation'])
    require(moved['raw']==raw_ref and len(moved['rows'])==256,'Canary relocation coverage differs')
    expected_moves=[]
    for row in rows:
        capture=Path(row['native_capture_path']);begin=capture.with_name(capture.name.replace('.CAPTURE.json','.BEGIN.json'))
        for original,sha in ((capture,row['native_capture_sha256']),(begin,row['native_begin_sha256'])):
            e.path(original,sha)
            destination=original if phase=='baseline' else expected_audit/'original_tokens'/original.name
            e.path(destination,sha)
            expected_moves.append(dict(source=tx.reference(e.root,original),destination=tx.reference(e.root,destination)))
    require(moved['rows']==expected_moves,'Canary relocated/reordered/substituted original bytes')
    vectors=e.tensor(e.ref(result['tensors']))
    events=_teacher(vectors,rows,chain[0][1],marker,[x[0] for x in chain])
    account=e.json(result['accounting'])
    require(account['events']==events and account['measured_counts']=={'training_policy':128,'training_reference':128} and
            result['measured_counts']==account['measured_counts'],'Actual per-row/accounting evidence differs')
    ledger=tx.rows(e.path(expected_audit/'accounting.jsonl'))
    require([event for event in ledger if event.get('metadata',{}).get('execution_phase')=='CANARY_RESTORE_TEACHER']==events,
            'Original runtime canary accounting log differs')
    return dict(result=result,reference=result_ref,snapshot=snapshot,vectors=vectors)


def _one_path(e,spec_ref,sources,native,freeze_path,freeze_sha):
    spec=e.json(spec_ref);identity=spec['identity'];state._validate_identity(identity)
    start,arm=identity['train_seed'],identity['arm']
    require(spec.get('schema')==replay.SCHEMA and spec.get('permission')=='READONLY_STEP2_CANARY_ONLY' and
            spec.get('design_freeze_sha256')==freeze_sha and spec.get('execution_sources')==sources and
            identity['execution_sources_sha256']==object_sha(sources),'Untrusted canary technical spec/source identity')
    base=e.root/'canary_replay'/f'start-{start}'/arm
    require(e.ref(spec_ref)==base/'TECHNICAL_EXECUTION.json' and spec['slots']==
            {phase:str((base/phase).relative_to(e.root)) for phase in ('baseline','restored')} and
            spec['restored_audit']==str((base/'restored/audit').relative_to(e.root)),'Canary spec fixed paths differ')
    marker=e.json(spec['commits']['2']);genesis=marker['transaction']['genesis_reference']
    store=tx.TransactionStore(e.root,freeze_path,freeze_sha,identity,native=native,genesis_reference=genesis)
    chain=store._chain()
    require(len(chain)==2 and all(spec['commits'][str(step)]==tx.reference(e.root,store.path/f'step-{step:03d}/COMMIT.json')
            and spec['commits'][str(step)]['sha256']==chain[step-1][0] for step in (1,2)),
            'Canary trusted chain must contain exactly original first2 commits')
    payloads=[]
    for step,(commit_sha,commit) in enumerate(chain,1):
        state.validate_checkpoint(store.path/f'step-{step:03d}',commit_sha,identity,native)
        payloads.append(e.tensor(store.path/f'step-{step:03d}/state.pt',commit['state_sha256']))
    source=identity['source_binding'];r24=e.tensor(Path(source['original_state_path']),source['state_sha256'])
    require(r24['schema']==fork_loader.LEGACY_SCHEMA and r24['step']==24 and
            all(r24['identity'].get(k)==v for k,v in dict(parent='b0',train_seed=start,arm='R').items()) and
            r24['reference_sha256']==source['reference_sha256'] and native.state_hash(r24['policy'])==source['policy_sha256'],
            'Wrong actual R24 source policy/reference identity')
    deltas=[_delta(before,after['policy'],identity['optimizer_parameter_order']) for before,after in
            ((r24['policy'],payloads[0]),(payloads[0]['policy'],payloads[1]))]
    rows,raw_ref=replay._originals(store,chain[1][1]);e.ref(raw_ref)
    phases={phase:_phase(e,store,spec,spec_ref,phase,rows,raw_ref,chain) for phase in ('baseline','restored')}
    require(phases['baseline']['result']['process']!=phases['restored']['result']['process'],
            'Reload did not use a different process incarnation')
    require(phases['restored']['result']['baseline']==phases['baseline']['reference'],'Restored phase binds another baseline')
    require(_same(phases['baseline']['vectors'],phases['restored']['vectors']), 'Actual restored teacher/reference arrays are not bit-exact')
    require(phases['baseline']['snapshot']==phases['restored']['snapshot'],'Canary observed state identity before/after differs')
    return dict(store=store,chain=chain,spec=spec,spec_ref=spec_ref,phases=phases,deltas=deltas)


def _fingerprint_values(model):
    """Hash every named non-LoRA parameter (aliases included), never model.forward."""
    import torch
    parameters=list(model.named_parameters(remove_duplicate=False))
    frozen=[(name,value) for name,value in parameters if 'lora_' not in name]
    require(frozen and all(not value.requires_grad for _,value in frozen),'Non-LoRA parameters must all be frozen')
    versions=[(value,value._version) for _,value in parameters];mode=model.training;rows=[]
    with torch.no_grad():
        for name,value in frozen:
            require(not value.is_floating_point() or value.dtype==torch.float32,'FP32 frozen floating parameters required')
            metadata=dict(name=name,shape=list(value.shape),dtype=str(value.dtype),device=str(value.device),requires_grad=False)
            cpu=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest=hashlib.sha256(tx.canonical(dict(shape=list(value.shape),dtype=str(value.dtype))))
            digest.update(memoryview(cpu.numpy()))
            rows.append(dict(metadata,tensor_sha256=digest.hexdigest()))
    require(model.training==mode and all(value._version==version for value,version in versions),'Fingerprint mutated model')
    inventory=[{k:v for k,v in row.items() if k!='tensor_sha256'} for row in rows]
    return dict(scope='ALL_NAMED_NON_LORA_PARAMETERS_ALIASES_INCLUDED',parameters=rows,
                inventory_sha256=object_sha(inventory),values_sha256=object_sha(rows),buffers_included=False)


def fingerprint_cpu_for_tests(model):
    """Pure CPU synthetic helper. Its mode is rejected by production proof audit."""
    require(all(p.device.type=='cpu' for p in model.parameters()),'CPU synthetic helper only')
    return dict(schema=WITNESS_SCHEMA,mode='SYNTHETIC_CPU_TEST',permission='NO_TRAINING',**_fingerprint_values(model))


def fingerprint_first_two(handle,*,phase,execution_sources,expected_auditor_sha256):
    """Future actual TRAIN worker observation; returns data, writes no artifact.

    Caller records BEFORE_FIRST_STEP and AFTER_STEP2 under its existing sole
    worker/transaction discipline. Hashing may transfer one parameter to CPU;
    no forward/backward/Adam, RNG consumption, or GPU model construction occurs.
    Buffers are explicitly outside this all-non-LoRA-PARAMETER witness contract.
    """
    import torch
    producer=_self(expected_auditor_sha256)
    require(isinstance(handle,runtime.TrainingHandle) and phase in ('BEFORE_FIRST_STEP','AFTER_STEP2') and
            execution_sources.get('frozen_fingerprint')==producer and
            handle.path_identity['execution_sources_sha256']==object_sha(execution_sources),'Unbound fingerprint producer identity')
    endpoint=handle.endpoint();require(endpoint['step']==(0 if phase=='BEFORE_FIRST_STEP' else 2),'Wrong frozen witness phase/cursor')
    model=handle.runtime.model
    require(all(p.device==torch.device('cuda:0') for p in model.parameters()) and
            len([p for p in model.parameters() if p.requires_grad])==496,'Actual CUDA0 model with complete496 LoRA required')
    require(torch.cuda.is_available() and torch.cuda.device_count()==2 and torch.cuda.current_device()==0,'Actual two-GPU inventory required')
    freeze=tx.read_json(handle.registry.freeze_path)
    runtime.verify_sources(handle.registry,tx.bound(handle.registry.root,freeze['artifacts']['source_lock']),execution_sources)
    gpu=[]
    for index in (0,1):
        value=torch.cuda.get_device_properties(index)
        gpu.append(dict(index=index,name=value.name,total_memory=value.total_memory,major=value.major,minor=value.minor,
                        uuid=str(getattr(value,'uuid','UNAVAILABLE'))))
    require(gpu[0]['name']==gpu[1]['name'] and 'PRO6000' in re.sub('[^A-Z0-9]','',gpu[0]['name'].upper()) and
            all(gpu[0][k]==gpu[1][k] for k in ('total_memory','major','minor')),'Actual GPU pair differs from original backend')
    native=handle.native_ops
    def hashes():
        return {key+'_sha256':native.state_hash(value) for key,value in dict(policy=native.trainable_state(model),
            reference=handle.fork.reference,optimizer=handle.fork.optimizer.state_dict(),scheduler=handle.fork.scheduler.state_dict(),
            rng=native.capture_rng()).items()}
    before=hashes();require(handle.runtime._satdose_active is None and all(p.grad is None for p in model.parameters()),'Active/dirty fingerprint input')
    result=_fingerprint_values(model)
    require(before==hashes() and endpoint==handle.endpoint(),'Fingerprint changed policy/reference/Adam/scheduler/RNG')
    return dict(schema=WITNESS_SCHEMA,mode='CUDA_OBSERVATION',permission='NO_TRAINING',phase=phase,
        producer=producer,identity=copy.deepcopy(handle.path_identity),process=replay._process_identity(),endpoint=endpoint,
        gpu_inventory=gpu,protected_state=before,model_calls=0,optimizer_steps=0,backward_calls=0,rng_unchanged=True,**result)


def _frozen_pair(e,pair,ctx,sources):
    require(set(pair)=={'before','after'},'Missing frozen parameter before/after witness')
    values=[e.json(pair[key]) for key in ('before','after')];identity=ctx['store'].identity
    producer=sources['frozen_fingerprint'];e.path(Path(producer['path']),producer['sha256'])
    for witness,phase,step in zip(values,('BEFORE_FIRST_STEP','AFTER_STEP2'),(0,2),strict=True):
        require(witness.get('schema')==WITNESS_SCHEMA and witness.get('mode')=='CUDA_OBSERVATION' and
                witness.get('permission')=='NO_TRAINING' and witness.get('phase')==phase and witness.get('identity')==identity and
                witness.get('producer')==producer and witness.get('process')==ctx['phases']['baseline']['result']['process'] and
                witness.get('scope')=='ALL_NAMED_NON_LORA_PARAMETERS_ALIASES_INCLUDED' and witness.get('buffers_included') is False and
                witness.get('rng_unchanged') is True and all(type(witness.get(k)) is int and witness[k]==0
                    for k in ('model_calls','optimizer_steps','backward_calls')),'Invalid actual frozen-parameter witness')
        endpoint=witness['endpoint'];expected=ctx['chain'][1][1] if step else e.json(ctx['store'].genesis_reference)
        policy=identity['source_binding']['policy_sha256'] if step==0 else expected['policy_sha256']
        require(endpoint==dict(experiment_id=state.EXPERIMENT,start_seed=identity['train_seed'],arm=identity['arm'],step=step,
                policy_sha256=policy,reference_sha256=identity['source_binding']['reference_sha256']), 'Frozen witness endpoint differs')
        hashes=witness['protected_state']
        require(hashes['policy_sha256']==policy and hashes['reference_sha256']==identity['source_binding']['reference_sha256'] and
                all(hashes[k+'_sha256']==expected[k+'_sha256'] for k in ('optimizer','scheduler','rng')),'Frozen witness state/RNG lineage differs')
        rows=witness['parameters'];gpu=witness['gpu_inventory']
        require(isinstance(rows,list) and rows and len({r['name'] for r in rows})==len(rows) and
                all('lora_' not in r['name'] and r['device']=='cuda:0' and r['requires_grad'] is False and
                re.fullmatch('[0-9a-f]{64}',r['tensor_sha256']) for r in rows),'Incomplete/invalid frozen parameter witness inventory')
        require(witness['inventory_sha256']==object_sha([{k:v for k,v in r.items() if k!='tensor_sha256'} for r in rows]) and
                witness['values_sha256']==object_sha(rows),'Frozen parameter witness digest differs')
        require(len(gpu)==2 and [r['index'] for r in gpu]==[0,1] and gpu[0]['name']==gpu[1]['name'] and
                'PRO6000' in re.sub('[^A-Z0-9]','',gpu[0]['name'].upper()),'Missing actual GPU inventory in witness')
    require(values[0]['parameters']==values[1]['parameters'] and values[0]['gpu_inventory']==values[1]['gpu_inventory'],
            'Frozen/base/vision parameter bytes changed across first2')
    return dict(status='SOURCE_BOUND_GPU_OBSERVATIONS_IDENTICAL',before=pair['before'],after=pair['after'],
                parameters=len(values[0]['parameters']),second_offline_gpu_weight_observation=False,buffers_verified=False)


def _exit_proof(e,ref,ctx,sources):
    proof=e.json(ref);source=sources['canary_controller'];e.path(Path(source['path']),source['sha256'])
    baseline=ctx['phases']['baseline'];restored=ctx['phases']['restored']
    require(proof.get('schema')==EXIT_SCHEMA and proof.get('identity')==ctx['store'].identity and
            proof.get('spec')==ctx['spec_ref'] and proof.get('controller_source')==source and
            proof.get('baseline_complete')==baseline['reference'] and proof.get('restored_claim')==restored['result']['claim'] and
            proof.get('baseline_process')==baseline['result']['process'] and
            proof.get('restored_process')==restored['result']['process'],'Unbound controller exit proof')
    controller=_process(proof['controller_process']);wait=e.json(proof['wait_event']);launch=e.json(proof['restored_launch_event'])
    for event,kind in ((wait,'BASELINE_WAIT_RETURNED'),(launch,'RESTORED_LAUNCH_REQUESTED')):
        require(event.get('schema')=='sr-satdose-controller-process-event-v1' and event.get('event')==kind and
                event.get('controller_source')==source and event.get('controller_process')==controller and
                event.get('identity')==ctx['store'].identity and event.get('spec')==ctx['spec_ref'] and
                type(event.get('monotonic_ns')) is int and event['monotonic_ns']>0,'Invalid source-bound controller event')
    require(wait['monotonic_ns']<launch['monotonic_ns'] and type(wait.get('returncode')) is int and wait['returncode']==0 and
            wait.get('worker_process')==baseline['result']['process'] and launch.get('worker_process')==restored['result']['process'] and
            wait.get('worker_claim')==baseline['result']['claim'] and launch.get('worker_claim')==restored['result']['claim'],
            'Baseline exit not observed before restored launch')
    if wait.get('kind')=='DIRECT_CHILD_WAIT':
        require(wait.get('waited_pid')==baseline['result']['process']['pid'] and
                all(controller[k]==baseline['result']['process'][k] for k in ('host','boot_id')),'Waited on wrapper/foreign PID')
    else:
        require(wait.get('kind')=='SINGLE_TASK_SRUN_WAIT' and type(wait.get('wrapper_pid')) is int and wait['wrapper_pid']>0 and
                wait.get('slurm_state')=='COMPLETED' and wait.get('slurm_exit_code')=='0:0' and
                type(wait.get('task_count')) is int and wait['task_count']==1 and
                all(isinstance(wait.get(k),str) and wait[k] for k in ('slurm_job_id','slurm_step_id')),
                'Incomplete actual single-task Slurm completion')
        observation=e.json(wait['worker_slurm_observation'])
        require(observation.get('schema')=='sr-satdose-controller-worker-slurm-v1' and
                observation.get('identity')==ctx['store'].identity and observation.get('spec')==ctx['spec_ref'] and
                observation.get('worker_process')==baseline['result']['process'] and observation.get('worker_claim')==baseline['result']['claim'] and
                observation.get('controller_source')==source and all(observation.get(k)==wait[k] for k in
                    ('slurm_job_id','slurm_step_id','task_count')),'Slurm wrapper was not bound to actual model worker')
    return dict(status='SOURCE_BOUND_CONTROLLER_EXIT_OBSERVATION_VERIFIED',evidence=ref,
                offline_process_death_reobserved=False)


def audit_four_paths(root,*,trusted_specs,execution_sources,expected_auditor_sha256,expected_freeze_sha256,
                     frozen_witnesses=None,process_exit_proofs=None):
    """No writes, model, GPU calls, optimizer or admission publication.

    trusted_specs is exactly {11031/T0,11031/T1,11032/T0,11032/T1: {path,sha256}}.
    Optional proof families are future-worker/controller artifacts: omission is
    PENDING, never the positive complete-audit status. Supplied families must be
    exact4, source-bound and valid; malformed evidence raises a technical HOLD.
    """
    auditor=_self(expected_auditor_sha256);root=Path(root)
    require(root.resolve(strict=True)==root and set(trusted_specs)==set(PATHS),'Exact four trusted paths required')
    for family in (frozen_witnesses,process_exit_proofs):
        require(family is None or set(family)==set(PATHS),'Proof family missing/extra path')
    e=_Evidence(root);freeze_path=e.path(root/'DESIGN_FREEZE.json',expected_freeze_sha256);freeze=tx.read_json(freeze_path)
    require(freeze['schema']=='sr-satdose-design-freeze-v1' and freeze['experiment_id']==state.EXPERIMENT,'Foreign original freeze')
    _sources(execution_sources,freeze,e);lock=e.json(freeze['artifacts']['source_lock'])
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:
        native=_native(lock,e);reports={};contexts={}
        with _existing_lock(root):
            replay._no_unknown(root)
            for key in PATHS:
                ctx=_one_path(e,trusted_specs[key],execution_sources,native,freeze_path,expected_freeze_sha256)
                require(key==f"{ctx['store'].start}/{ctx['store'].arm}",'Trusted spec mapped to another path')
                contexts[key]=ctx
                reports[key]=dict(status='CORE_CANARY_TENSORS_AND_STATE_VERIFIED',request_count=128,
                    baseline=ctx['phases']['baseline']['reference'],restored=ctx['phases']['restored']['reference'],
                    actual_nonzero_policy_deltas=ctx['deltas'],policy_adam_scheduler_rng='VERIFIED_FROM_ACTUAL_SAFE_CPU_STATE',
                    reference='ORIGINAL_B0_HASH_AND_OBSERVED_REFERENCE_LOGPROBS_BOUND_NO_NEW_GPU_OBSERVATION')
                if frozen_witnesses is not None:reports[key]['frozen_parameters']=_frozen_pair(e,frozen_witnesses[key],ctx,execution_sources)
                if process_exit_proofs is not None:reports[key]['baseline_exit']=_exit_proof(e,process_exit_proofs[key],ctx,execution_sources)
            for ctx in contexts.values():
                ctx['store']._design();require(ctx['store']._chain()==ctx['chain'],'Original chain changed during aggregate audit')
            _sources(execution_sources,freeze,e);e.stable()
        pending=[]
        if frozen_witnesses is None:pending.append('PENDING_FIRST2_FROZEN_BASE_VISION_PARAMETER_WITNESSES')
        if process_exit_proofs is None:pending.append('PENDING_CONTROLLER_BASELINE_EXIT_PROOFS')
        return dict(schema='sr-satdose-four-path-canary-audit-v1',status='PENDING_CANARY_EVIDENCE' if pending else
            'FOUR_PATH_CANARY_EVIDENCE_AUDITED_NOT_ADMISSION',permission='NO_TRAINING',auditor=auditor,
            design_freeze_sha256=expected_freeze_sha256,paths=reports,pending=pending,
            limitations=['No final CANARY_RESUME_PASS publication or step3 authorization.',
                'GPU parameter fingerprints and process exits are authenticated producer observations, not re-observed offline.',
                'Frozen witness contract covers parameters, not mutable model buffers.'],
            observed_files={str(path):sha for path,sha in e.observed.items()})
    finally:sys.dont_write_bytecode=previous
