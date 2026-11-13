"""Two fixed START/R24 GPU loader audit slots; no inference or training.

Actual evaluation_loader construction is used once per fresh process. The
shared diagnostic lock/gates and complete DRY/PREFIX evidence precede loading.
Trained checkpoint loading is NOT exercised or certified here. These captures
are evidence for independent review, never whole-GPU acceptance or admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import diagnostic_worker as worker, evaluation_loader as loader, shadow_replay as shadow

require = shadow.require
_USED = False
EVIDENCE_NAMES = ('CLAIM.json','GPU_IDENTITY.json','TENSOR_INVENTORY.json','SOURCES.json',
                  'CALLER_STATE.json','LOAD_AUDIT.json','CLOSE.json','audit/EVALUATION_LOAD.json')


def expected_slots():
    return {str(start):f'evaluation_loader_checks/start-{start}' for start in (11031,11032)}


def _binding_fields(binding):
    require(binding.get('evaluation_loader_slots') == expected_slots() and
            binding.get('max_evaluation_loader_attempts') == 2, 'Fixed two-slot evaluation-loader binding required')
    sources = binding.get('evaluation_execution_sources', {})
    require(set(sources) == {'evaluation_loader','training_state','training_transaction'} and
            loader.object_sha(sources) == binding.get('expected_evaluation_execution_sources_sha256'),
            'Evaluation execution source inventory/SHA differs')
    modules = {'evaluation_probe':Path(__file__).resolve(), 'evaluation_loader':Path(loader.__file__).resolve(),
               'training_state':Path(loader.state.__file__).resolve(),
               'training_transaction':Path(loader.transactions.__file__).resolve()}
    for name,path in modules.items():
        ref = binding['code'][name]
        require(Path(ref['path']).resolve() == path and shadow.file_sha(path) == ref['sha256'],
                'Unbound evaluation-probe dependency: '+name)
        if name != 'evaluation_probe':
            require(sources[name] == ref, 'Evaluator source binding differs from technical code pin')


def _prefix_required(ctx):
    require((ctx['root']/'prefix_control/RESULT.json').is_file(),
            'Both complete passing PREFIX starts required before evaluation loader check')
    worker._prefix_gate(ctx['root'],ctx['binding'])


def _preflight(start,binding_sha):
    # Revalidates both complete DRY populations/native scores, runtime inventory,
    # source lock, and global HOLD/UNKNOWN gates. No model construction occurs.
    ctx = worker._preflight(binding_sha,start,0)
    _binding_fields(ctx['binding'])
    _prefix_required(ctx)
    return ctx


def _gpu_identity():
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2 and torch.cuda.current_device() == 0,
            'Evaluation probe requires the production two visible CUDA GPUs')
    devices=[]
    for index in (0,1):
        prop=torch.cuda.get_device_properties(index)
        devices.append(dict(index=index,name=torch.cuda.get_device_name(index),
            total_memory=prop.total_memory,major=prop.major,minor=prop.minor,uuid=str(getattr(prop,'uuid','unavailable'))))
    return dict(visible_gpu_count=2,arithmetic_device=0,devices=devices,
        peer_0_to_1=bool(torch.cuda.can_device_access_peer(0,1)),
        peer_1_to_0=bool(torch.cuda.can_device_access_peer(1,0)),
        torch_version=torch.__version__,cuda_version=torch.version.cuda)


def _inventory(handle,ops,expected):
    """Observe actual tensors without a forward; handle has no generation API."""
    import torch
    inner=handle._underlying
    require(inner.fork.optimizer is None and inner.fork.scheduler is None and
            not inner.runtime._satdose_accounting and not inner.storage_forwards,
            'Evaluation construction performed model/optimizer work')
    model=inner.runtime.model
    trainable={name:param for name,param in model.named_parameters() if param.requires_grad}
    require(len(trainable)==496 and set(trainable)==set(inner.fork.reference) and
            all('lora_' in name for name in trainable),'Exact 496 original LoRA tensors required')
    require(all(value.dtype==torch.float32 and value.grad is None for value in inner.fork.reference.values()),
            'Original reference must remain 496 FP32 tensors without gradients')
    require(all(p.grad is None for p in model.parameters()),'Evaluation has dirty gradients')
    require(all(str(value.device)=='cuda:0' for value in (*model.parameters(),*model.buffers())) and
            all(not value.is_floating_point() or value.dtype==torch.float32
                for value in (*model.parameters(),*model.buffers())), 'Actual FP32 model must remain on cuda:0')
    policy=ops.state_hash(ops.trainable_state(model));reference=ops.state_hash(inner.fork.reference)
    require(policy==expected['policy_sha256'] and reference==expected['reference_sha256'],
            'Actual evaluation tensor hashes differ from R24/b0')
    return dict(trainable_tensors=496,reference_tensors=496,reference_dtype='torch.float32',
        policy_sha256=policy,reference_sha256=reference,
        dtype='torch.float32',device='cuda:0',optimizer_created=False,scheduler_created=False,
        measured_forward_sequences=0,measured_backward_calls=0,
        tensors=[dict(name=name,shape=list(param.shape),dtype=str(param.dtype),device=str(param.device))
                 for name,param in trainable.items()])


def _execute(ctx,start,process):
    with worker._worker_lock(ctx['root']):
        return _execute_locked(ctx,start,process)


def _execute_locked(ctx,start,process):
    root,binding=ctx['root'],ctx['binding']
    require(type(start) is int and start in (11031,11032),'Only fixed START loader slots exist')
    worker._no_holds(root,binding,binding_sha=ctx['binding_sha'])
    _binding_fields(binding);_prefix_required(ctx)
    destination=root/expected_slots()[str(start)]
    destination.parent.mkdir(exist_ok=True)
    require(destination.parent.resolve()==destination.parent,'Noncanonical evaluation-loader root')
    destination.mkdir(exist_ok=False);shadow._sync_dir(destination.parent)
    shadow.write_once(destination/'CLAIM.json',dict(status='RESERVED_UNKNOWN_NO_RETRY',start_seed=start,
        technical_binding_sha256=ctx['binding_sha'],process_identity=process,
        endpoint_arm='START',endpoint_step=0,max_model_constructions=1,max_new_generations=0,
        max_forward_sequences=0,max_backward_calls=0,max_optimizer_steps=0,max_scheduler_steps=0))
    audit=destination/'audit';audit.mkdir()
    handle=None;close=None;close_attempted=False;built=0;before_sources={};rng_before=None;ops=None
    source=ctx['config']['start_bindings'][str(start)]
    try:
        expected=ctx['prepared']['starts'][str(start)]['expected_identity']
        require(expected['arm']=='START' and expected['step']==0 and expected['start_seed']==start,
                'Evaluation-loader probe cannot select trained endpoint')
        before_sources={source['original_state_path']:source['state_sha256']}
        before_sources.update({str(Path(ctx['lock']['parent_path'])/name):sha
                               for name,sha in ctx['lock']['parent_files'].items()})
        for path,sha in before_sources.items():require(shadow.file_sha(path)==sha,'Original loader source changed')
        ops=loader.fork_loader.load_native_ops(ctx['lock']['native_src'],ctx['lock']['native_files'])
        rng_before=ops.state_hash(ops.capture_rng())
        gpu=_gpu_identity()
        shadow.write_once(destination/'GPU_IDENTITY.json',gpu)
        # One call site and no retries. A failed constructor leaves CLAIM/HOLD.
        handle=loader.build_evaluation_loader(ctx['registry'],worker._bound(root,binding['source_lock']),
            start_seed=start,arm='START',step=0,output_dir=audit,
            execution_sources=binding['evaluation_execution_sources'],
            expected_execution_sources_sha256=binding['expected_evaluation_execution_sources_sha256'])
        built=1
        observed=handle.audit()
        require(all(observed.get(k)==v for k,v in expected.items() if k!='parent') and
                observed.get('permission')=='LOAD_AUDIT_ONLY_NO_GENERATION_NO_MAIN_RELEASE' and
                observed.get('model_calls')==0 and observed.get('optimizer_constructed') is False and
                observed.get('training_rng_restored') is False,'Evaluation audit endpoint/side effects differ')
        receipt=shadow.read_json(audit/'EVALUATION_LOAD.json')
        require(receipt==handle.receipt and receipt['selection']=={k:expected[k] for k in
                ('experiment_id','start_seed','arm','step')} and receipt['original_files']==before_sources and
                receipt['evaluation_sources_sha256']==binding['expected_evaluation_execution_sources_sha256'] and
                receipt['design_freeze_sha256']==binding['design_freeze']['sha256'] and
                receipt['source_lock_sha256']==binding['source_lock']['sha256'] and
                receipt['caller_rng_sha256']==rng_before,'Actual loader receipt chain differs')
        inventory=_inventory(handle,ops,expected)
        shadow.write_once(destination/'TENSOR_INVENTORY.json',inventory)
        shadow.write_once(destination/'LOAD_AUDIT.json',observed)
        require(ops.state_hash(ops.capture_rng())==rng_before,'Evaluation loader changed caller RNG')
        close_attempted=True
        close=handle.close()
        require(close==dict(observed,closed=True),'Evaluation close changed endpoint or side effects')
        shadow.write_once(destination/'CLOSE.json',close)
        after_sources={path:shadow.file_sha(path) for path in before_sources}
        require(after_sources==before_sources,'Original R24/parent files changed during loader probe')
        _binding_fields(binding)
        execution_after={name:dict(path=ref['path'],sha256=shadow.file_sha(ref['path']))
                         for name,ref in binding['code'].items()
                         if name in ('evaluation_probe','evaluation_loader','training_state','training_transaction')}
        require(execution_after=={name:binding['code'][name] for name in execution_after},
                'Evaluation execution source changed during load')
        shadow.write_once(destination/'SOURCES.json',dict(before=before_sources,after=after_sources,unchanged=True,
            execution_before={name:binding['code'][name] for name in execution_after},execution_after=execution_after))
        rng_after=ops.state_hash(ops.capture_rng())
        require(rng_after==rng_before,'Evaluation close changed caller RNG')
        shadow.write_once(destination/'CALLER_STATE.json',dict(rng_before_sha256=rng_before,
            rng_after_sha256=rng_after,caller_rng_preserved=True,training_rng_restored=False,
            optimizer_created=False,scheduler_created=False))
        for ref in ctx['protected']:worker._bound(root,ref)
        _prefix_required(ctx)
    except BaseException as exc:
        shadow.write_once(destination/'HOLD.json',dict(status='HOLD_EVALUATION_LOADER_NO_RETRY',
            technical_binding_sha256=ctx['binding_sha'],start_seed=start,model_constructions_completed=built,
            model_construction_outcome='COMPLETE' if built else 'UNKNOWN_OR_NOT_REACHED',
            error_type=type(exc).__name__,error=str(exc),permission='NO_TRAINING'))
        raise
    finally:
        try:
            if handle is not None and not close_attempted:
                close_attempted=True
                close=handle.close()
            for path,sha in before_sources.items():require(shadow.file_sha(path)==sha,'Original loader source changed at cleanup')
            if ops is not None and rng_before is not None:
                require(ops.state_hash(ops.capture_rng())==rng_before,'Caller RNG differs at cleanup')
        except BaseException as exc:
            if not (destination/'HOLD.json').exists():
                shadow.write_once(destination/'HOLD.json',dict(status='HOLD_EVALUATION_LOADER_CLEANUP',error=str(exc)))
            raise
    complete=dict(schema='sr-satdose-evaluation-loader-check-v1',
        status='EVALUATION_START_LOAD_CAPTURE_COMPLETE_NOT_GPU_ADMISSION',start_seed=start,
        technical_binding_sha256=ctx['binding_sha'],evaluation_sources_sha256=binding['expected_evaluation_execution_sources_sha256'],
        tested_scope='START_R24_ONLY',trained_checkpoint_gpu_verified=False,model_constructions=1,
        trainable_tensors=496,policy_sha256=inventory['policy_sha256'],reference_sha256=inventory['reference_sha256'],
        forward_sequences=0,backward_calls=0,new_generations=0,optimizer_steps=0,scheduler_steps=0,
        parameter_updates=0,caller_rng_preserved=True,training_rng_restored=False,
        actual_gpu_identity=gpu,close=close,permission='NO_TRAINING',
        evidence={name:worker._ref(root,destination/name) for name in EVIDENCE_NAMES})
    shadow.write_once(destination/'COMPLETE.json',complete)
    return complete


def run_probe(*,start_seed,expected_binding_sha256):
    global _USED
    require(not _USED and type(start_seed) is int and start_seed in (11031,11032),
            'One fixed START evaluation-loader check per fresh process')
    _gpu_identity()
    with worker._worker_lock(shadow.PRODUCTION_ROOT):
        ctx=_preflight(start_seed,expected_binding_sha256)
        _USED=True
        return _execute_locked(ctx,start_seed,worker._process_identity())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed',type=int,choices=(11031,11032),required=True)
    parser.add_argument('--binding-sha256',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run_probe(start_seed=args.start_seed,expected_binding_sha256=args.binding_sha256),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
