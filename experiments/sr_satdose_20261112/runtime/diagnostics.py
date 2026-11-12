"""Read-only SATDOSE gradient primitives; never a GPU admission certificate.

The caller must validate DRY identities/tokens against DESIGN_FREEZE before use.
`forward(row)` must call the registered production backend with original tokens,
returning (current, original_sampler, frozen_reference) FP32 vectors. Its saved-
tensor context must match production. This module never generates or calls the
legacy update/optimizer. Tensor evidence alone is not production replay evidence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

OBJECTIVE_SHA = '7be5b90a6d91261aa9f23803f217de8c42c30de6050157c23ccb7d3b2c880cea'
TRAINING_SHA = 'aac309a9cb1f3eee8e6371e4b588cc54154f7d1088282b8861f48dcf01fde3c7'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class NativeMath:
    sequence_loss: object
    advantages: object
    sources: dict


def load_native_math(objective_file, legacy_package):
    """Import actual pinned functions; do not reimplement the native objective."""
    objective_file = Path(objective_file).resolve(strict=True)
    package = Path(legacy_package).resolve(strict=True)
    training_file = package / 'adapter/training.py'
    require(file_sha(objective_file) == OBJECTIVE_SHA, 'Native objective SHA mismatch')
    require(file_sha(training_file) == TRAINING_SHA, 'Native training SHA mismatch')
    existing = sys.modules.get('runtime_contract')
    require(existing is None or Path(existing.__file__).resolve() == package / 'runtime_contract.py',
            'Shadowed legacy runtime_contract')
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(package))
        objective = load('_satdose_pinned_objective', objective_file)
        training = load('_satdose_pinned_training', training_file)
    finally:
        sys.path[:] = original_path
    return NativeMath(objective.sequence_loss, training.native_j_advantages,
                      {'objective_sha256': OBJECTIVE_SHA, 'training_sha256': TRAINING_SHA})


def coefficients(rows, native, field):
    require(len(rows) > 0 and len(rows) % 8 == 0, 'Complete G8 groups required')
    ids = [r['request_id'] for r in rows]
    require(len(ids) == len(set(ids)), 'Duplicate request')
    result, groups = [], set()
    for start in range(0, len(rows), 8):
        block = rows[start:start + 8]
        gid = block[0]['group_id']
        require(gid not in groups and all(r['group_id'] == gid for r in block), 'G8 order/membership differs')
        groups.add(gid)
        values = [r[field] for r in block]
        # Native implementation checks exact binary labels and population std.
        result.extend(native.advantages(values))
    return result


def witness_coefficients(rows, native):
    coefficients(rows, native, 'exact_authorised')  # Validate binary complete groups.
    result = []
    for start in range(0, len(rows), 8):
        values = [r['exact_authorised'] for r in rows[start:start + 8]]
        result.extend(float(c - (sum(values) - c) / 7) for c in values)
    return result


def validate_partition(rows, ordered_group_halves):
    """Require a pre-frozen, disjoint partition, without choosing from scores."""
    group_ids = [r['group_id'] for r in rows[::8]]
    require(len(ordered_group_halves) == 2, 'Two fixed halves required')
    flat = [g for half in ordered_group_halves for g in half]
    require(len(flat) == len(set(flat)) == len(group_ids) and set(flat) == set(group_ids),
            'Frozen halves must partition every complete group')
    require(len(ordered_group_halves[0]) == len(ordered_group_halves[1]), 'Unequal halves')
    lookup = {g: rows[8*i:8*i+8] for i, g in enumerate(group_ids)}
    return [[row for g in half for row in lookup[g]] for half in ordered_group_halves]


def _snapshot_rng(torch):
    import numpy as np
    return (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(torch, state):
    import numpy as np
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def accumulate_component(model, rows, forward, native, component, *, expected_tensors=496):
    """Fresh forwards in fixed row order; return unclipped CPU gradient tensors.

    `component` is total/J/source/KL/witness. Whole batch must have 128 rows;
    a frozen split half has 64. All rows remain in the sequence denominator.
    Witness probability derivatives use completion logprob SUM, never mean.
    CPU toy use is permitted for tests but does not emit a production PASS.
    """
    import torch
    require(component in ('total', 'J', 'source', 'KL', 'witness'), 'Unknown component')
    require(len(rows) in (64, 128), 'Full batch or fixed half required')
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(len(parameters) == expected_tensors and parameters, 'Trainable inventory differs')
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'FP32 parameters required')
    require(all(not b.is_floating_point() or b.dtype == torch.float32 for b in model.buffers()), 'FP32 buffers required')
    require(not any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()), 'Dropout forbidden')
    require(all(p.grad is None for p in model.parameters()), 'Existing gradients would be overwritten')
    values = (witness_coefficients(rows, native) if component == 'witness' else
              coefficients(rows, native, 'exact_authorised' if component == 'source' else 'J'))
    before = {n: p.detach().cpu().clone() for n, p in parameters.items()}
    # Native reference_parameters temporarily copies reference LoRA values into
    # the trainable tensors and restores policy values before policy forward.
    # Those legitimate copies advance LoRA versions twice per reference call;
    # compare their final values below, not their version counters. Frozen base
    # parameters and buffers must never participate in the reference swap.
    versions = [(v, v._version) for v in
                (*[p for p in model.parameters() if not p.requires_grad], *model.buffers())]
    rng, training = _snapshot_rng(torch), model.training
    traces = []
    try:
        model.eval()
        for row, advantage in zip(rows, values, strict=True):
            current, sampler, reference = forward(row)
            require(current.ndim == 1 and current.numel() == len(row['tokens']), 'Original token alignment differs')
            require(1 <= len(row['tokens']) <= 1536 and all(type(t) is int and t >= 0 for t in row['tokens']), 'Invalid original tokens')
            require(current.dtype == sampler.dtype == reference.dtype == torch.float32, 'FP32 vectors required')
            require(current.shape == sampler.shape == reference.shape, 'Vector alignment differs')
            require(current.requires_grad and not sampler.requires_grad and not reference.requires_grad, 'Detached sampler/reference required')
            require(all(bool(torch.isfinite(v).all()) for v in (current, sampler, reference)), 'Nonfinite logprob')
            require(sampler.detach().cpu().tolist() == row['sampler_logprobs'], 'Sampler differs from original trajectory')
            if component == 'witness':
                loss = advantage * current.sum()
            else:
                result = native.sequence_loss(current, sampler, reference, advantage, 0.0)
                loss = result[{'total':'loss', 'J':'native', 'source':'native', 'KL':'kl_unweighted'}[component]]
            (loss / len(rows)).backward()
            traces.append({'request_id': row['request_id'], 'token_count': current.numel(),
                           'train_logprobs': current.detach().cpu().tolist(),
                           'reference_logprobs': reference.detach().cpu().tolist(),
                           'sampler_logprobs': sampler.detach().cpu().tolist()})
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
                'Missing/nonfinite gradient tensor')
        gradients = {n: p.grad.detach().cpu().clone() for n, p in parameters.items()}
    finally:
        for p in model.parameters():
            p.grad = None
        _restore_rng(torch, rng)
        model.train(training)
    require(all(v._version == version for v, version in versions), 'Forward mutated a frozen parameter/buffer')
    require(all(torch.equal(before[n], p.detach().cpu()) for n, p in parameters.items()), 'Policy changed')
    return {'gradients': gradients, 'traces': traces, 'component': component,
            'forward_calls': len(rows), 'backward_calls': len(rows), 'optimizer_steps': 0,
            'sequence_denominator': len(rows), 'source_sha256': dict(native.sources),
            'status': 'COMPONENT_CAPTURED_NOT_GPU_ADMISSION'}


def dot(left, right):
    require(set(left) == set(right) and left, 'Gradient tensor inventory differs')
    result = 0.0
    for name in left:
        require(left[name].shape == right[name].shape, 'Gradient shape differs')
        result += float((left[name].double() * right[name].double()).sum())
    require(math.isfinite(result), 'Nonfinite gradient inner product')
    return result


def cosine(left, right):
    norm = math.sqrt(dot(left, left) * dot(right, right))
    require(norm > 0, 'HOLD_DIRECTION_UNCERTAIN: zero gradient cosine undefined')
    return max(-1.0, min(1.0, dot(left, right) / norm))


def compare_replay(production_gradients, diagnostic_gradients):
    import torch
    require(set(production_gradients) == set(diagnostic_gradients), 'Replay inventory differs')
    differences = [n for n in production_gradients if not torch.equal(production_gradients[n], diagnostic_gradients[n])]
    return {'bit_exact': not differences, 'different_tensor_names': differences}


def recomposition_error(j, kl, total):
    require(set(j) == set(kl) == set(total), 'Component inventory differs')
    # Reconstruct in FP32, matching the defined gradient objects.
    delta = {n: j[n] + 0.02 * kl[n] - total[n] for n in total}
    norm = math.sqrt(dot(total, total))
    error = math.sqrt(dot(delta, delta))
    return error / norm if norm else (0.0 if error == 0 else math.inf)


def shadow_pair(policy, gradients, output_root, start_seed, *, expected_tensors=496):
    """Two fresh Adam steps on isolated copies; durable reservation prevents retry.

    Exactly one write-once directory per registered start allows at most FOUR
    shadow optimizer calls per output_root. Caller binds this unique output_root
    into DESIGN_FREEZE; creating another root is not an authorized replay.
    An interrupted reservation must be audited, not rerun. No production tensor
    or optimizer is passed into Adam. Successful return includes actual deltas.
    """
    import torch
    require(type(start_seed) is int and start_seed in (11031, 11032), 'Unregistered start')
    require(set(policy) == set(gradients) and len(policy) == expected_tensors, 'Shadow tensor inventory differs')
    require(all(v.dtype == torch.float32 and v.shape == gradients[n].shape and
                bool(torch.isfinite(v).all()) and bool(torch.isfinite(gradients[n]).all()) for n,v in policy.items()),
            'Shadow dtype/shape/finite contract differs')
    root = Path(output_root)
    require(root.is_dir() and root.resolve() == root.absolute(), 'Canonical shadow evidence root required')
    destination = root / str(start_seed)
    destination.mkdir(exist_ok=False)
    (destination / 'ATTEMPT.json').write_text(json.dumps({'status':'RESERVED_UNKNOWN', 'max_optimizer_steps':2,
                                                       'start_seed':start_seed})+'\n')
    deltas, clips, states = [], [], []
    for trial in range(2):
        # Preserve original parameter order; Adam foreach=False is fixed.
        params = {n: torch.nn.Parameter(v.detach().clone()) for n, v in policy.items()}
        opt = torch.optim.AdamW(list(params.values()), lr=1e-5, betas=(0.9,0.999), eps=1e-8,
                               weight_decay=0.0, foreach=False)
        for n,p in params.items():
            p.grad = gradients[n].detach().to(p.device).clone()
        norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0, error_if_nonfinite=True))
        require(math.isfinite(norm) and norm <= 1000, 'Preclip numerical sentinel failed')
        clipped = {n:p.grad.detach().cpu().clone() for n,p in params.items()}
        require(math.sqrt(dot(clipped, clipped)) <= 1.00001, 'Postclip numerical sentinel failed')
        opt.step()
        delta = {n:(p.detach()-policy[n]).cpu().clone() for n,p in params.items()}
        require(all(bool(torch.isfinite(v).all()) for v in delta.values()), 'Nonfinite shadow update')
        with (destination / f'trial_{trial}.pt').open('xb') as stream:
            torch.save({'delta':delta,'clipped':clipped,'optimizer':opt.state_dict()},stream)
        deltas.append(delta); clips.append(clipped); states.append(opt.state_dict())
        del opt, params
    require(compare_replay(deltas[0], deltas[1])['bit_exact'], 'Shadow direct/replay differs')
    receipt = {'status':'SHADOW_PAIR_COMPLETE_NOT_ADMISSION','start_seed':start_seed,'shadow_optimizer_steps':2,
               'formal_optimizer_steps':0,'direct_replay_delta_bit_exact':True,
               'evidence':{f'trial_{i}.pt':file_sha(destination/f'trial_{i}.pt') for i in range(2)}}
    with (destination/'COMPLETE.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    return receipt, deltas[0], clips[0]


def save_components(destination, captures):
    """Persist full gradients and traces; never declare GPU or direction PASS."""
    import torch
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    files = {}
    for name, capture in captures.items():
        require(name.replace('_','').isalnum(), 'Unsafe component name')
        path = destination / (name + '.pt')
        with path.open('xb') as stream:
            torch.save(capture, stream)
        files[path.name] = file_sha(path)
    with (destination/'INDEX.json').open('x') as stream:
        json.dump({'status':'EVIDENCE_ONLY_NOT_ADMISSION','files':files},stream,indent=2)
    return files


def direction_summary(captures, delta, production_gradients):
    """Compute registered metrics from separately persisted full tensor captures.

    Requires repeated witness calculation on the SAME halves; the repeat error
    floor is not the between-half difference. Returns HOLD when a direction is
    undefined/nonpositive. Even all checks true does not grant GPU admission.
    """
    keys = ('J','source','KL','total','J_half_1','J_half_2','source_half_1','source_half_2',
            'witness_half_1','witness_half_2','witness_half_1_repeat','witness_half_2_repeat')
    require(all(k in captures for k in keys), 'Missing component/half/repeat evidence')
    gradients = {k:captures[k]['gradients'] for k in keys}
    effects = [dot(gradients[f'witness_half_{i}'],delta) for i in (1,2)]
    repeats = [dot(gradients[f'witness_half_{i}_repeat'],delta) for i in (1,2)]
    floor = max(abs(a-b) for a,b in zip(effects,repeats,strict=True))
    cosines, reasons = {}, []
    for name,left,right in (('J_vs_source_pg_cosine','J','source'),
                            ('J_split_half_cosine','J_half_1','J_half_2'),
                            ('source_split_half_cosine','source_half_1','source_half_2')):
        try:
            cosines[name]=cosine(gradients[left],gradients[right])
        except ValueError:
            cosines[name]=None
            reasons.append(name+': undefined')
        if cosines[name] is not None and cosines[name] <= 0:
            reasons.append(name+': nonpositive')
    for i,effect in enumerate(effects,1):
        if effect <= floor:
            reasons.append(f'heldout half {i}: does not exceed repeat error floor')
    replay = compare_replay(production_gradients,gradients['total'])
    if not replay['bit_exact']:reasons.append('production replay not bit-exact')
    error = recomposition_error(gradients['J'],gradients['KL'],gradients['total'])
    if not math.isfinite(error) or error > 1e-5:reasons.append('component recomposition exceeds 1e-5')
    return {**cosines,'heldout_effect_half_1':effects[0],'heldout_effect_half_2':effects[1],
            'effect_repeat_error_floor':floor,'recomposition_relative_l2':error if math.isfinite(error) else None,
            'production_replay':replay,'direction_checks_satisfied':not reasons,'hold_reasons':reasons,
            'status':'HOLD_DIRECTION_UNCERTAIN' if reasons else 'DIRECTION_CHECKS_ONLY_NOT_GPU_ADMISSION'}
