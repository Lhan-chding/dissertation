"""Explicit zero-advantage native-objective technical check, not zero-J data.

Three fresh complete-128 original-token replays capture PG_zero, unweighted KL,
and total preclip gradients. Only the diagnostic objective's advantage argument
is overridden to zero. Actual J/native_score, requests, tokens, sampler and
reference are never rewritten. No generation, clipping, Adam, scheduler, formal
update or global GPU admission occurs. Two write-once slots are source-bound by
the future technical binding and share the global diagnostic process lock.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from . import diagnostic_worker as worker, production, shadow_replay as shadow
from runtime import diagnostics, native_runtime

require = shadow.require
_USED = False
COMPONENTS = {'PG_ZERO':'J', 'KL':'KL', 'TOTAL':'total'}


def expected_slots():
    return {str(start):f'allzero_check/start-{start}' for start in (11031,11032)}


def _preflight(start, binding_sha):
    ctx = worker._preflight(binding_sha,start,0)
    binding = ctx['binding']
    require(binding.get('allzero_slots') == expected_slots() and binding.get('max_allzero_attempts') == 2,
            'Fixed two-slot allzero technical binding required')
    actual = Path(__file__).resolve()
    require(Path(binding['code']['allzero_check']['path']).resolve() == actual and
            shadow.file_sha(actual) == binding['code']['allzero_check']['sha256'], 'Unbound allzero source')
    return ctx


def _capture_component(handle, rows, kernels, component, *, expected_tensors=496):
    """Pinned native arithmetic on original vectors; CPU toy seam stays private."""
    import torch
    require(component in COMPONENTS and len(rows) == 128, 'One complete fixed probe128 required')
    model = handle.runtime.model
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    require(len(parameters) == expected_tensors and set(parameters) == set(handle.fork.reference),
            'Policy/reference trainable inventory differs')
    if expected_tensors == 496:
        require(all('lora_' in n for n in parameters) and
                all(str(v.device) == 'cuda:0' for v in (*model.parameters(),*model.buffers())),
                'Expected 496 original LoRA tensors on cuda:0')
    require(all(p.grad is None for p in model.parameters()), 'Preexisting gradients forbidden')
    original = copy.deepcopy(rows)
    before = {n:p.detach().cpu().clone() for n,p in parameters.items()}
    reference_sha = kernels.state_hash(handle.fork.reference)
    rng = kernels.capture_rng(); mode = model.training
    frozen = [(v,v._version) for v in (*[p for p in model.parameters() if not p.requires_grad],*model.buffers())]
    terms_trace = []
    def explicit_zero_loss(current,sampler,reference,original_advantage,aux):
        require(aux == 0., 'Auxiliary objective is forbidden')
        # Reuse the original pinned native function. Do not overwrite row J,
        # mutate its native score, substitute sampler/old, or implement GRPO.
        terms = kernels.sequence_loss(current,sampler,reference,0.0,0.0)
        require(all(bool(torch.isfinite(torch.as_tensor(v)).all()) for v in terms.values()),
                'Nonfinite native zero-advantage objective')
        terms_trace.append(dict(original_group_advantage=float(original_advantage),
            diagnostic_advantage=0.0, **{k:float(terms[k].detach()) for k in ('native','kl','kl_unweighted','loss')}))
        return terms
    native = diagnostics.NativeMath(explicit_zero_loss,kernels.advantages,
        dict(native=copy.deepcopy(kernels.sources),diagnostic_override='advantage=0.0; actual J/native_score unchanged',
             override_source_sha256=shadow.file_sha(__file__)))
    failure = None; mutation = []; result = None
    try:
        peak_before = handle.storage.gpu_peak_stats(reset=True)
        result = diagnostics.accumulate_component(model,rows,handle.forward_original,native,
                                                  COMPONENTS[component],expected_tensors=expected_tensors)
        require(len(terms_trace) == result['forward_calls'] == result['backward_calls'] == 128,
                'Full native forward/backward inventory differs')
        handle.verify_storage_after_backward()
        result.update(component=component,explicit_diagnostic_advantage=0.0,original_scores_unchanged=rows==original,
            original_rows_sha256=worker._sha_object(original),native_terms=terms_trace,
            original_reference_sha256=reference_sha,policy_before_sha256=kernels.state_hash(before),
            gpu_memory_before=peak_before,gpu_memory_after=handle.storage.gpu_peak_stats(reset=False),
            trainable_tensors=len(parameters),optimizer_steps=0,scheduler_steps=0,new_generations=0,parameter_updates=0,
            status='ALLZERO_COMPONENT_CAPTURE_NOT_GPU_ADMISSION')
    except BaseException as exc:
        failure = exc
    finally:
        try:
            if any(v._version != version for v,version in frozen): mutation.append('Frozen tensor changed; discard runtime')
            if kernels.state_hash(handle.fork.reference) != reference_sha: mutation.append('Reference changed; discard runtime')
            if rows != original: mutation.append('Original rows/scores changed')
            with torch.no_grad():
                for name,param in parameters.items():
                    if not torch.equal(param.detach().cpu(),before[name]):
                        mutation.append('Policy mutated: '+name);param.copy_(before[name])
        finally:
            for param in model.parameters():param.grad=None
            try:kernels.restore_rng(rng)
            finally:model.train(mode)
    if mutation:raise PermissionError('; '.join(mutation)) from failure
    if failure is not None:raise failure
    require(kernels.state_hash(kernels.capture_rng()) == kernels.state_hash(rng), 'RNG restoration differs')
    result.update(policy_after_sha256=kernels.state_hash({n:p.detach().cpu() for n,p in parameters.items()}),
                  reference_after_sha256=kernels.state_hash(handle.fork.reference),rng_unchanged=True)
    return result


def compare(captures):
    import torch
    require(set(captures) == set(COMPONENTS), 'Missing explicit-zero components')
    pg,kl,total = (captures[name] for name in COMPONENTS)
    for capture in (pg,kl,total):
        require(capture['sequence_denominator'] == 128 and capture['original_scores_unchanged'] is True and
                capture['explicit_diagnostic_advantage'] == 0.0 and len(capture['native_terms']) == 128 and
                all(row['native'] == 0. and row['loss'] == row['kl'] for row in capture['native_terms']),
                'Zero-advantage branch lost native KL or has nonzero PG loss')
        require(capture['policy_before_sha256'] == capture['policy_after_sha256'] and
                capture['original_reference_sha256'] == capture['reference_after_sha256'] and capture['rng_unchanged'] is True,
                'Zero-advantage capture changed policy/reference/RNG')
    require(pg['traces'] == kl['traces'] == total['traces'] and
            pg['native_terms'] == kl['native_terms'] == total['native_terms'] and
            pg['original_rows_sha256'] == kl['original_rows_sha256'] == total['original_rows_sha256'] and
            pg['policy_before_sha256'] == kl['policy_before_sha256'] == total['policy_before_sha256'] and
            pg['original_reference_sha256'] == kl['original_reference_sha256'] == total['original_reference_sha256'],
            'Replayed original vectors, scores or native terms changed across components')
    grads = [capture['gradients'] for capture in (pg,kl,total)]
    require(list(grads[0]) == list(grads[1]) == list(grads[2]), 'Component tensor order/inventory differs')
    require(all(torch.count_nonzero(value).item() == 0 for value in grads[0].values()), 'PG_zero gradient is not exactly zero')
    recomposed = {name:grads[0][name]+.02*grads[1][name] for name in grads[0]}
    exact = all(torch.equal(recomposed[name],grads[2][name]) for name in recomposed)
    error = diagnostics.recomposition_error(*grads)
    require(math.isfinite(error) and error <= 1e-5, 'Zero-advantage recomposition exceeds existing 1e-5 limit')
    return dict(pg_exact_zero=True,kl_retained=True,recomposition_bit_exact=exact,recomposition_relative_l2=error,
        kl_gradient_l2=math.sqrt(diagnostics.dot(grads[1],grads[1])),total_gradient_l2=math.sqrt(diagnostics.dot(grads[2],grads[2])),
        kl_unweighted_mean=sum(row['kl_unweighted'] for row in kl['native_terms'])/128,
        weighted_kl_mean=sum(row['kl'] for row in total['native_terms'])/128,
        explicit_diagnostic_advantage=0.0,kl_coefficient=.02,sequence_denominator=128,
        policy_sha256=total['policy_before_sha256'],original_reference_sha256=total['original_reference_sha256'],
        original_scores_unchanged=True,interpretation='explicit_zero_advantage_technical_branch_not_actual_all_zero_rewards',
        zero_kl_is_allowed_when_native_math_is_zero=True)


def _execute(ctx,start):
    """Internal control seam for CPU tests; production preflight holds this lock."""
    with worker._worker_lock(ctx['root']):
        return _execute_locked(ctx,start)


def _execute_locked(ctx,start):
    root,binding = ctx['root'],ctx['binding']
    worker._no_holds(root,binding,binding_sha=ctx['binding_sha'])
    require(binding.get('allzero_slots') == expected_slots() and binding.get('max_allzero_attempts') == 2,
            'Fixed allzero slots required')
    destination = root/expected_slots()[str(start)]
    destination.parent.mkdir(exist_ok=True)
    require(destination.parent.resolve() == destination.parent, 'Noncanonical allzero root')
    destination.mkdir(exist_ok=False);shadow._sync_dir(destination.parent)
    shadow.write_once(destination/'CLAIM.json',dict(status='RESERVED_UNKNOWN_NO_RETRY',start_seed=start,
        technical_binding_sha256=ctx['binding_sha'],explicit_diagnostic_advantage=0.0,
        max_optimizer_steps=0,max_new_generations=0,max_policy_forward_sequences=384,max_reference_forward_sequences=384))
    audit=destination/'audit';audit.mkdir()
    handle=None;relocation=[];accounts={};captures={};close=None;comparison=None
    source=ctx['config']['start_bindings'][str(start)]
    source_path=Path(source['original_state_path'])
    try:
        require(shadow.file_sha(source_path)==source['state_sha256'],'Original R24 state changed')
        selected=ctx['prepared']['starts'][str(start)]
        original=copy.deepcopy(selected['probe']['rows'])
        shadow.write_once(destination/'ORIGINAL_INPUTS.json',dict(
            prepared_diagnostic_inputs=binding['diagnostic_inputs'],rows=original,
            actual_rewards_preserved=True,explicit_diagnostic_advantage=0.0))
        handle=native_runtime.build_runtime(ctx['registry'],worker._bound(root,binding['source_lock']),
                                          start_seed=start,output_dir=audit)
        endpoint=handle.endpoint()
        require(dict(endpoint,parent='b0') == selected['expected_identity'], 'Loaded R24/reference differs')
        require(handle.fork.optimizer is None and handle.fork.scheduler is None and not handle.runtime._satdose_accounting,
                'Read-only constructor created optimizer or performed model work')
        rows,relocation=worker._relocate(root,audit,original,endpoint,binding['design_freeze']['sha256'])
        kernels=production.load_native_kernels(ctx['lock']['native_src'],ctx['lock']['legacy_root'])
        require(ctx['config']['loss']['kl_coefficient'] == .02, 'Frozen KL coefficient differs')
        production.validate_batch(rows,selected['production_plan'],selected['expected_identity'],ctx['config'],
                                  endpoint['policy_sha256'],endpoint['reference_sha256'],kernels)
        for component in COMPONENTS:
            begin,storage_begin=len(handle.runtime._satdose_accounting),len(handle.storage_forwards)
            captures[component]=_capture_component(handle,rows,kernels,component)
            accounts[component]=worker._accounting(handle,begin,storage_begin,128)
            worker._save_tensor(destination/(component+'.pt'),captures[component])
        comparison=compare(captures)
        shadow.write_once(destination/'COMPARISON.json',comparison)
        shadow.write_once(destination/'ACCOUNTING.json',accounts)
        require(selected['probe']['rows'] == original and handle.endpoint() == endpoint,
                'Original input/score or runtime endpoint changed')
    except BaseException as exc:
        shadow.write_once(destination/'HOLD.json',dict(status='HOLD_ALLZERO_NO_RETRY',error_type=type(exc).__name__,error=str(exc),
            technical_binding_sha256=ctx['binding_sha'],observed_accounting=copy.deepcopy(handle.runtime._satdose_accounting) if handle else [],
            observed_storage=copy.deepcopy(handle.storage_forwards) if handle else []))
        raise
    finally:
        try:
            if handle is not None:
                close=handle.close()
                require(close.get('closed') is True and close.get('optimizer_steps') == close.get('parameter_updates') == 0,
                        'Allzero runtime did not close unchanged with zero updates')
            require(shadow.file_sha(source_path)==source['state_sha256'],'Original R24 state changed during allzero check')
            for ref in ctx['protected']:worker._bound(root,ref)
            for entry in relocation:
                require(shadow.file_sha(entry['source']) == shadow.file_sha(entry['destination']) == entry['sha256'],
                        'Original/relocated capture changed')
        except BaseException as exc:
            if not (destination/'HOLD.json').exists():
                shadow.write_once(destination/'HOLD.json',dict(status='HOLD_ALLZERO_CLEANUP',error=str(exc)))
            raise
    complete=dict(schema='sr-satdose-allzero-check-v1',status='ALLZERO_CAPTURE_COMPLETE_NOT_GPU_ADMISSION',
        start_seed=start,technical_binding_sha256=ctx['binding_sha'],permission='NO_TRAINING',close=close,
        comparison=comparison,explicit_diagnostic_advantage=0.0,original_scores_unchanged=True,sequence_denominator=128,
        trainable_tensors=captures['TOTAL']['trainable_tensors'],optimizer_steps=0,scheduler_steps=0,new_generations=0,parameter_updates=0,
        measured_forward_sequences={purpose:sum(item['measured_forward_sequences'][purpose] for item in accounts.values())
            for purpose in ('training_policy','training_reference')},
        evidence={name:worker._ref(root,destination/name) for name in
            ('ORIGINAL_INPUTS.json','PG_ZERO.pt','KL.pt','TOTAL.pt','COMPARISON.json','ACCOUNTING.json')})
    shadow.write_once(destination/'COMPLETE.json',complete)
    return complete


def run_allzero(*,start_seed,expected_binding_sha256):
    global _USED
    require(not _USED and type(start_seed) is int and start_seed in (11031,11032), 'One registered allzero attempt per fresh process')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2 and torch.cuda.current_device() == 0,
            'Allzero technical check requires the production two visible CUDA GPUs')
    with worker._worker_lock(shadow.PRODUCTION_ROOT):
        ctx=_preflight(start_seed,expected_binding_sha256)
        _USED=True
        return _execute_locked(ctx,start_seed)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed',type=int,choices=(11031,11032),required=True)
    parser.add_argument('--binding-sha256',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run_allzero(start_seed=args.start_seed,expected_binding_sha256=args.binding_sha256),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
