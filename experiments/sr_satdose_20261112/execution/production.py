"""Independent SATDOSE production numerical core, with no optimizer capability.

The future transaction driver can use the returned clipped gradients only after
its separate admission/commit checks. This module never samples, constructs or
calls Adam, steps a scheduler, or publishes a checkpoint. It independently
executes the pinned legacy per-row numerical sequence, rather than calling the
component diagnostic accumulator. Every capture is evidence, not GPU admission.

Caller must authenticate the complete finite plan, native rescoring, original
captures/tokens and current committed endpoint. ``forward(row)`` must be the
registered native policy/reference bridge with production saved-tensor storage.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path
import sys

EXPERIMENT = 'SR-SATDOSE-1-20261112'
PINS = {
    'sr_f6/objective.py': '7be5b90a6d91261aa9f23803f217de8c42c30de6050157c23ccb7d3b2c880cea',
    'sr_f5/training.py': '58013b4dec4902b3e1261e565c6de491c693d5e7b0ab4cb3b903913f47dc3cd0',
    'mm_core/training.py': '43b027fb35710411d28a0309bd926c2b23172de4ae5fe7022dc2c66fb8377af2',
}
LEGACY_PINS = {
    'adapter/training.py': 'aac309a9cb1f3eee8e6371e4b588cc54154f7d1088282b8861f48dcf01fde3c7',
    'runtime_contract.py': '84b8ea2d1d70dc03de50ca7f17d5dff9b5e4be4227061efae0a23d661e514dd6',
}


def require(condition, message):
    if not condition:
        raise PermissionError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class NativeKernels:
    sequence_loss: object
    advantages: object
    numerical_audit: object
    check_numerics: object
    state_hash: object
    capture_rng: object
    restore_rng: object
    sources: dict


def load_native_kernels(native_src, legacy_package):
    """Import pinned native arithmetic/audit; caller binds entire dependency tree."""
    native_src, legacy_package = Path(native_src).resolve(strict=True), Path(legacy_package).resolve(strict=True)
    for root, pins in ((native_src, PINS), (legacy_package, LEGACY_PINS)):
        for name, sha in pins.items():
            require(file_sha(root / name) == sha, 'Pinned production source changed: ' + name)
    existing = sys.modules.get('runtime_contract')
    require(existing is None or Path(existing.__file__).resolve() == legacy_package / 'runtime_contract.py',
            'Shadowed legacy runtime contract')
    old_path = list(sys.path)
    try:
        sys.path[:0] = [str(native_src), str(legacy_package)]
        modules = {}
        for rel in PINS:
            name = rel[:-3].replace('/', '.')
            mod = importlib.import_module(name)
            require(Path(mod.__file__).resolve() == native_src / rel, 'Shadowed native dependency: ' + name)
            modules[name] = mod
        contract = importlib.import_module('runtime_contract')
        spec = importlib.util.spec_from_file_location('_satdose_production_pinned_training', legacy_package/'adapter/training.py')
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
    finally:
        sys.path[:] = old_path
    core = modules['mm_core.training']
    return NativeKernels(modules['sr_f6.objective'].sequence_loss, training.native_j_advantages,
        modules['sr_f5.training'].numerical_audit, contract.check_numerics,
        core.state_hash, core.capture_rng, core.restore_rng,
        {'native': dict(PINS), 'legacy': dict(LEGACY_PINS)})


def validate_batch(rows, plan, identity, config, policy_hash, reference_hash, kernels):
    """New scientific identity, exact finite G8 order and original endpoint checks."""
    require(config['experiment_id'] == identity.get('experiment_id') == EXPERIMENT, 'Wrong experiment')
    start = identity.get('start_seed')
    require(type(start) is int and start in (11031, 11032), 'Wrong start')
    require(identity.get('parent') == 'b0', 'Wrong parent')
    step = identity.get('step'); arm = identity.get('arm'); scope = plan.get('scope')
    require(type(step) is int and 0 <= step < 24, 'Invalid committed step')
    require((scope == 'DRY' and arm == 'START' and step == plan.get('step') == 0)
            or (scope == 'TRAIN' and arm in ('T0', 'T1') and plan.get('step') == step+1),
            'Invalid scope/path/cursor')
    binding = config['start_bindings'][str(start)]
    require(identity.get('policy_sha256') == policy_hash and
            identity.get('reference_sha256') == reference_hash == binding['reference_sha256'],
            'Policy/reference tensor identity differs')
    if step == 0:
        require(policy_hash == binding['policy_sha256'], 'Step0 must be the R24 policy')
    require(len(rows) == 128 and len(plan['groups']) == 16, 'Exactly sixteen complete G8 groups required')
    require(len({r['request_id'] for r in rows}) == 128, 'Duplicate original request')
    require(len({g['group_id'] for g in plan['groups']}) == len({g['root_id'] for g in plan['groups']}) == 16,
            'Duplicate group/root')
    coefficients, groups = [], []
    for offset, group in zip(range(0,128,8), plan['groups'], strict=True):
        block = rows[offset:offset+8]
        require([r['request_id'] for r in block] == group['request_ids'] and len(set(group['request_ids'])) == 8,
                'Original within-G8 order differs')
        js = []
        for row in block:
            require(row['group_id'] == group['group_id'] and row['root_id'] == group['root_id']
                    and row['start_seed'] == start and row['arm'] == arm
                    and row['scope'] == scope and row['step'] == plan['step'], 'Original row identity differs')
            require(row['policy_sha256'] == policy_hash and row['reference_sha256'] == reference_hash,
                    'Stale/foreign original trajectory')
            require(row['generation_status'] == 'COMPLETE' and row['technical_validation_errors'] == [],
                    'Technical failure is not a zero reward')
            require(row['finish_reason'] in ('balanced','eos','length') and row['assistant_prefill'] == '{',
                    'Invalid original termination/prefix')
            require(type(row['J']) is int and row['J'] in (0,1) and row['native_score']['J'] == row['J'],
                    'Invalid/unbound native J')
            tokens, sampler = row['tokens'], row['sampler_logprobs']
            require(type(tokens) is list and 1 <= len(tokens) <= 1536 and all(type(t) is int and t >= 0 for t in tokens),
                    'Invalid original token sequence')
            require(type(sampler) is list and len(sampler) == len(tokens) and
                    all(type(v) in (int,float) and math.isfinite(v) for v in sampler), 'Invalid original sampler')
            require(row['sampler_logprob_source'] == 'actual_generate_raw_logits_selected_token', 'Wrong sampler provenance')
            js.append(row['J'])
        advantage = kernels.advantages(js)
        coefficients.extend(advantage)
        groups.append(dict(group_id=group['group_id'], root_id=group['root_id'], J=js, advantages=advantage))
    return coefficients, groups


def capture_update(model, rows, forward, reference, *, kernels, config, identity, plan, expected_tensors=496):
    """Independent native loss/128 backward, then explicit pre/post-clip capture.

    Returns detached CPU gradient evidence and the exact native numerical audit.
    For calibration, construct new dictionaries, never mutate original evidence:
    ``identity = dict(handle.endpoint(), parent='b0')`` and
    ``rows = [dict(row, J=row['native_score']['J']) for row in completed_raw]``.
    The plan must come from the SHA-verified frozen probe mapping.
    No optimizer object is accepted. Existing gradients are rejected. Finally
    clears newly accumulated gradients and restores full RNG/training mode.
    LoRA mutation is restored then rejected; frozen mutation invalidates the
    runtime and requires discarding/reloading it, since no full base copy exists.
    """
    import torch
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    require(len(parameters) == expected_tensors and parameters, 'Trainable inventory differs')
    if expected_tensors == 496:
        require(all('lora_' in n for n in parameters), 'Non-LoRA trainable tensor')
    require(set(parameters) == set(reference), 'Reference inventory differs')
    require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in (*model.parameters(),*model.buffers())),
            'FP32 model/buffers required')
    require(all(not isinstance(m,torch.nn.Dropout) or m.p == 0 for m in model.modules()), 'Dropout forbidden')
    require(all(p.grad is None for p in model.parameters()), 'Preexisting gradients would be lost')
    require(all(torch.is_tensor(v) and not v.requires_grad and v.dtype == torch.float32 and
                v.shape == parameters[n].shape and bool(torch.isfinite(v).all()) for n,v in reference.items()),
            'Reference tensor contract differs')
    before = {n:p.detach().cpu().clone() for n,p in parameters.items()}
    require(all(bool(torch.isfinite(v).all()) for v in before.values()), 'Nonfinite policy')
    policy_hash, reference_hash = kernels.state_hash(before), kernels.state_hash(reference)
    coefficients, groups = validate_batch(rows,plan,identity,config,policy_hash,reference_hash,kernels)
    rows_before, identity_before, plan_before = copy.deepcopy(rows), copy.deepcopy(identity), copy.deepcopy(plan)
    versions = [(v,v._version) for v in (*[p for p in model.parameters() if not p.requires_grad],*model.buffers())]
    rng, training = kernels.capture_rng(), model.training
    traces, totals = [], {k:0. for k in ('loss','native','aux','kl','kl_unweighted')}
    result, failure, mutation = None, None, []
    row_forward_callback_calls = backward_calls = 0
    try:
        model.eval()
        # These operations and their order mirror the pinned adapter update:
        # reference/policy vectors -> native objective -> loss/128 backward ->
        # per-sequence scalar totals -> native numerical audit -> global clip.
        for row, advantage in zip(rows,coefficients,strict=True):
            row_forward_callback_calls += 1
            current,sampler,ref = forward(row)
            require(current.ndim == 1 and current.numel() == len(row['tokens']) and
                    current.shape == sampler.shape == ref.shape, 'Original token alignment differs')
            require(sampler.detach().cpu().tolist() == row['sampler_logprobs'], 'Original sampler changed')
            traces.append(dict(row_id=row['request_id'],qid=row.get('qid',row['root_id']),tokens=list(row['tokens']),
                sampler_logprobs=sampler.detach().cpu().tolist(),train_logprobs=current.detach().cpu().tolist(),
                reference_logprobs=ref.detach().cpu().tolist()))
            terms = kernels.sequence_loss(current,sampler,ref,advantage,0.)
            (terms['loss']/128).backward()
            backward_calls += 1
            for name in totals:
                totals[name] += float(terms[name].detach())/128
        numerics = kernels.numerical_audit(traces)
        require(numerics['status'] == 'FINITE' and all(math.isfinite(v) for v in totals.values()), 'Nonfinite numerical audit')
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
                'Missing/nonfinite gradient')
        preclip = {n:p.grad.detach().cpu().clone() for n,p in parameters.items()}
        norm_before = float(torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,error_if_nonfinite=True))
        norm_after = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters.values()))
        postclip = {n:p.grad.detach().cpu().clone() for n,p in parameters.items()}
        numerics.update(kl_unweighted=totals['kl_unweighted'],gradient_norm_before_clip=norm_before,
                        gradient_norm_after_clip=norm_after)
        kernels.check_numerics(numerics)
        result = dict(status='NUMERICAL_CAPTURE_NOT_UPDATE_AUTHORIZATION',preclip_gradients=preclip,
            postclip_gradients=postclip,numerics=numerics,totals=totals,traces=traces,groups=groups,
            policy_sha256=policy_hash,reference_sha256=reference_hash,source_sha256=copy.deepcopy(kernels.sources),
            row_forward_callback_calls=row_forward_callback_calls,backward_calls=backward_calls,
            sequence_denominator=128,optimizer_created=False,
            optimizer_steps=0,scheduler_steps=0,parameter_updates=0)
    except BaseException as exc:
        failure = exc
    finally:
        try:
            if any(v._version != version for v,version in versions): mutation.append('frozen parameter/buffer changed; discard runtime')
            if rows != rows_before or identity != identity_before or plan != plan_before: mutation.append('caller-owned registration/rows changed')
            if kernels.state_hash(reference) != reference_hash: mutation.append('reference changed; discard runtime')
            with torch.no_grad():
                for n,p in parameters.items():
                    if not torch.equal(before[n],p.detach().cpu()):
                        mutation.append('policy changed: '+n)
                        p.copy_(before[n])
        except BaseException as cleanup_error:
            mutation.append('state cleanup failed; discard runtime: '+repr(cleanup_error))
        finally:
            for p in model.parameters(): p.grad = None
            try:
                kernels.restore_rng(rng)
            finally:
                model.train(training)
    if mutation:
        raise PermissionError('; '.join(mutation)) from failure
    if failure is not None:
        raise failure
    require(kernels.state_hash(kernels.capture_rng()) == kernels.state_hash(rng), 'RNG restoration differs')
    return result
