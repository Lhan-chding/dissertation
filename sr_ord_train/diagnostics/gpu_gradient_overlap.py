"""Two original-row CUDA snapshot probes against immutable job-264303 evidence.

Short rows compare to the old successful baseline; long rows compare ONLY to
the already-saved CPU-snapshot result. The original long baseline remains OOM.
No CPU-snapshot execution, sampling, optimizer/scheduler step or COMMIT occurs.
CPU tensors here are serialized gradient evidence, not execution offload.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import time
import traceback

from adapter.transactions import file_hash, read_json, require, write_once
from diagnostics.gpu_saved_tensors import GPUSavedTensors, OFFLOAD_MODE, gpu_peak_stats, validate_gpu_pair
from diagnostics.gpu_recovery_state import original_rng_projection
from diagnostics.memory_equivalence import select_probes

PRIOR_JOB = '264303'
PRIOR_PATHS = frozenset(('b0-11031-R', 'b1-11031-R', 'b1-11031-G'))
UNAVAILABLE_PATH = 'b0-11031-G'
EXPECTED_GRADIENTS = 496
MANIFEST_FIELDS = ('path', 'checkpoint_step', 'next_step', 'binding_sha256', 'identity',
                   'commit_sha256', 'state_sha256', 'policy_sha256', 'reference_sha256',
                   'rng_sha256', 'plan_step', 'probes')
EXACT_FIELDS = ('request_id', 'token_path_sha256', 'prompt_tensor_hash', 'advantage',
                'train_logprobs', 'reference_logprobs', 'scalar_terms', 'tensor_terms',
                'scaled_loss', 'immutable_state_hashes', 'restored_rng_sha256',
                'gradient_names', 'gradient_none_names')


def _file_ref(path, root):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.resolve(strict=True) == path and path.is_relative_to(root),
            'Historical evidence is missing/noncanonical/outside its registered directory')
    return {'path': str(path), 'sha256': file_hash(path)}


def _recheck_refs(refs):
    for ref in refs:
        require(file_hash(ref['path']) == ref['sha256'], 'Historical evidence changed: ' + ref['path'])


def load_prior_evidence(package, manifest):
    """Pin old bytes before any new forward; never infer old success from filenames."""
    package = Path(package).resolve(strict=True)
    name = manifest['path']
    require(name in PRIOR_PATHS | {UNAVAILABLE_PATH}, 'Unregistered overlap path')
    root = package / 'run' / f'memory_diag_{PRIOR_JOB}_{name}'
    if name == UNAVAILABLE_PATH:
        require(not root.exists(), 'Unexpected b0G historical evidence requires explicit review')
        return {'status': 'NOT_AVAILABLE', 'path': name, 'prior_job': PRIOR_JOB,
                'reason': 'Job 264303 did not include b0-11031-G', 'refs': [], 'probes': []}
    manifest_ref = _file_ref(root / 'MANIFEST.json', root)
    previous = read_json(manifest_ref['path'])
    require(all(previous.get(key) == manifest.get(key) for key in MANIFEST_FIELDS),
            'Historical checkpoint/policy/reference/RNG/plan/probes differ')
    require(len(manifest['probes']) == 2, 'Fixed first and longest probes must be distinct')
    refs, probes = [manifest_ref], []
    for index, mode in ((0, 'baseline'), (1, 'bounded_cpu_saved_tensors')):
        folder = root / f'probe-{index}-{mode}'
        result_ref = _file_ref(folder / 'RESULT.json', root)
        old = read_json(result_ref['path'])
        require(old.get('status') == 'COMPLETE' and old.get('mode') == mode
                and old.get('probe') == manifest['probes'][index]
                and old.get('request_id') == manifest['probes'][index]['request_id'],
                'Historical comparison result is incomplete or from another probe')
        require(old.get('state_unchanged_after_attempt') is True
                and old.get('rng_restored_after_attempt') is True
                and old.get('restored_rng_sha256') == manifest['rng_sha256'],
                'Historical state/RNG audit is not verified')
        immutable = old['immutable_state_hashes']
        require(immutable['policy'] == manifest['policy_sha256']
                and immutable['reference'] == manifest['reference_sha256'],
                'Historical model identity differs')
        require(old.get('new_generations') == old.get('optimizer_steps') == old.get('scheduler_steps') == 0,
                'Historical diagnostic had unregistered model actions')
        gradient_ref = _file_ref(old['gradient_file'], folder)
        require(gradient_ref['sha256'] == old['gradient_file_sha256'], 'Historical gradient bytes differ')
        require(len(old['gradient_names']) == len(set(old['gradient_names'])) == EXPECTED_GRADIENTS,
                'Historical 496-gradient inventory differs')
        refs.extend((result_ref, gradient_ref))
        probes.append({'mode': mode, 'result': old, 'result_ref': result_ref, 'gradient_ref': gradient_ref})
    failed_ref = _file_ref(root / 'probe-1-baseline/RESULT.json', root)
    failed = read_json(failed_ref['path'])
    require(failed.get('mode') == 'baseline' and failed.get('status') == 'OOM'
            and failed.get('probe') == manifest['probes'][1], 'Original long baseline OOM evidence differs')
    refs.append(failed_ref)
    return {'status': 'HISTORICAL_EVIDENCE_PINNED', 'prior_job': PRIOR_JOB, 'path': name,
            'refs': refs, 'probes': probes, 'original_long_baseline_status': 'OOM',
            'original_long_baseline_ref': failed_ref,
            'long_comparator': 'Already-saved CPU-snapshot evidence only; no CPU-snapshot rerun'}


def compare_fields(actual, previous):
    """Missing scalar, vector, identity or gradient inventory fields cannot pass."""
    require(all(key in actual and key in previous for key in EXACT_FIELDS), 'Missing exact comparison field')
    return {key: actual[key] == previous[key] for key in EXACT_FIELDS}


def compare_gradients(actual, previous, *, torch):
    """Compare every saved gradient on CUDA 0, preserving dtype/shape/None status."""
    require(isinstance(actual, dict) and isinstance(previous, dict)
            and actual.keys() == previous.keys() and len(actual) == EXPECTED_GRADIENTS,
            'Exactly matching 496-gradient dictionaries required')
    comparisons = {}
    for name in actual:
        left, right = actual[name], previous[name]
        exact, maximum = left is None and right is None, None
        if left is not None and right is not None:
            require(torch.is_tensor(left) and torch.is_tensor(right)
                    and left.dtype == right.dtype and left.shape == right.shape,
                    'Historical gradient shape/dtype differs: ' + name)
            a = left.to(device='cuda:0', non_blocking=False)
            b = right.to(device='cuda:0', non_blocking=False)
            require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
                    'Nonfinite gradient evidence: ' + name)
            exact = bool(torch.equal(a, b))
            maximum = float((a.double() - b.double()).abs().max()) if a.numel() else 0.
            del a, b
        comparisons[name] = {'exact_equal': exact, 'max_absolute_difference': maximum}
    return comparisons


def _state(runtime, optimizer, scheduler, reference):
    from mm_core.training import state_hash, trainable_state
    return {'policy': state_hash(trainable_state(runtime.model)), 'reference': state_hash(reference),
            'optimizer': state_hash(optimizer.state_dict()), 'scheduler': state_hash(scheduler.state_dict()),
            'runtime_identity': copy.deepcopy(runtime.identity),
            'model_modes': {name: module.training for name, module in runtime.model.named_modules()}}


def _objective_evidence(terms, torch):
    """Check tensors on CUDA; CPU copies are serialized evidence only."""
    import math
    scalars, tensors = {}, {}
    for name, value in terms.items():
        if type(value) in (int, float, bool):
            require(math.isfinite(value), 'Nonfinite objective metadata: ' + name)
            scalars[name] = value
            continue
        require(torch.is_tensor(value) and str(value.device) == 'cuda:0'
                and bool(torch.isfinite(value).all()), 'Invalid CUDA objective: ' + name)
        detached = value.detach()
        if detached.ndim == 0:
            scalars[name] = float(detached)
        else:
            tensors[name] = {'shape': list(detached.shape), 'dtype': str(detached.dtype),
                             'values': detached.cpu().tolist()}
    require('loss' in scalars, 'Scalar native loss required')
    return scalars, tensors


def _cleanup_attempt(actions):
    """Attempt every cleanup, especially RNG restore, even after an earlier failure."""
    errors = []
    for name, action in actions:
        try:
            action()
        except BaseException as exc:
            errors.append((name, exc))
    return errors


def _probe(runtime, optimizer, scheduler, records, reference, manifest, output, index,
           coefficients, prior):
    import torch
    from adapter.training import _verify_prepared
    from mm_core.training import capture_rng, restore_rng, state_hash
    from sr_f6.objective import sequence_loss

    started = time.monotonic()
    probe = manifest['probes'][index]
    position = next(i for i, row in enumerate(records) if row['request_id'] == probe['request_id'])
    row, advantage = records[position], coefficients[position]
    folder = output / f'probe-{index}-{OFFLOAD_MODE}'
    folder.mkdir(exist_ok=False)
    before = _state(runtime, optimizer, scheduler, reference)
    rng = capture_rng()
    require(len(rng['cuda']) == 2 and state_hash(original_rng_projection(rng)) == manifest['rng_sha256'],
            'Two-GPU state does not project to original checkpoint RNG')
    rng_sha, gpu1_rng_sha = state_hash(rng), state_hash(rng['cuda'][1])
    result = {'status': 'UNFINISHED', 'mode': OFFLOAD_MODE, 'probe': probe,
              'new_generations': 0, 'optimizer_steps': 0, 'scheduler_steps': 0,
              'original_long_baseline_status': prior.get('original_long_baseline_status', 'NOT_AVAILABLE'),
              'original_long_gradient_equivalence': 'UNVERIFIED',
              'new_cpu_snapshot_execution': False, 'cpu_activation_offload_bytes': 0}
    error, hook = None, None
    current = ref = loss = scaled = sampler = gradients = None
    currents = refs = prepared = item = previous_gradients = None
    try:
        require(all(p.grad is None for p in runtime.model.parameters()), 'Probe must begin without gradients')
        runtime.model.eval()
        optimizer.zero_grad(set_to_none=True)
        prepared = runtime.prepare_input(row['model_input'], Path(runtime.identity['run_root']), protocol='structured_v4')
        _verify_prepared(row, prepared, torch)
        item = dict(row, prepared=prepared)
        with torch.no_grad():
            refs = runtime.batch_reference_forward([item], reference)
        require(len(refs) == 1, 'One original reference sequence required')
        ref = refs[0]
        hook = GPUSavedTensors((*runtime.model.parameters(), *runtime.model.buffers()), row_id=row['row_id'])
        with hook.context():
            currents = runtime.batch_sequence_forward([item], purpose='training_policy', grad=True)
            require(len(currents) == 1, 'One original policy sequence required')
            current = currents[0]
            require(str(current.device) == str(ref.device) == 'cuda:0'
                    and current.ndim == ref.ndim == 1 and current.shape == ref.shape
                    and len(current) == len(row['tokens']), 'Original CUDA token probability paths differ')
            sampler = torch.tensor(row['sampler_logprobs'], device='cuda:0', dtype=torch.float32)
            loss = sequence_loss(current, sampler, ref, advantage, 0.)
            scaled = loss['loss'] / 128
        gpu1_before_backward = state_hash(capture_rng()['cuda'][1])
        require(gpu1_before_backward == gpu1_rng_sha, 'Snapshot GPU consumed RNG during forward')
        runtime.reserve('diagnostic_gradient_calls', 1, method='gpu_saved_tensor_historical_overlap',
                        request_id=row['request_id'], optimizer_steps=0)
        scaled.backward()
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        require(hook.stats['copied_bytes'] > 0 and hook.stats['unpacked_copies'] > 0,
                'GPU saved-tensor copy/unpack mechanism was not exercised')
        result['gpu1_rng_unchanged_forward_backward'] = state_hash(capture_rng()['cuda'][1]) == gpu1_rng_sha
        require(result['gpu1_rng_unchanged_forward_backward'], 'Snapshot GPU consumed RNG during backward')
        gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                     for name, p in runtime.model.named_parameters() if p.requires_grad}
        require(len(gradients) == EXPECTED_GRADIENTS and any(v is not None for v in gradients.values()),
                'Actual 496-gradient inventory differs')
        gradient_path = folder / 'adapter_gradients.pt'
        with gradient_path.open('xb') as stream:
            torch.save(gradients, stream)
            stream.flush()
            os.fsync(stream.fileno())
        scalar_terms, tensor_terms = _objective_evidence(loss, torch)
        result.update(status='COMPLETE', request_id=row['request_id'], tokens=list(row['tokens']),
                      token_path_sha256=row['token_path_sha256'], prompt_tensor_hash=row['prompt_tensor_hash'],
                      advantage=advantage, train_logprobs=current.detach().cpu().tolist(),
                      reference_logprobs=ref.detach().cpu().tolist(), sampler_logprobs=list(row['sampler_logprobs']),
                      scalar_terms=scalar_terms, tensor_terms=tensor_terms, scaled_loss=float(scaled.detach()),
                      gradient_file=str(gradient_path), gradient_file_sha256=file_hash(gradient_path),
                      gradient_names=list(gradients), gradient_none_names=[name for name, value in gradients.items() if value is None],
                      immutable_state_hashes={key: before[key] for key in ('policy','reference','optimizer','scheduler')},
                      restored_rng_sha256=manifest['rng_sha256'])
        # Scalars above are gathered from the actual loss; gradients are checked
        # on CUDA 0 below. The original gradient file is read only as evidence.
        if prior['status'] == 'NOT_AVAILABLE':
            result['historical_comparison'] = {'status': 'NOT_AVAILABLE', 'reason': prior['reason']}
            for value in gradients.values():
                if value is not None:
                    require(bool(torch.isfinite(value.to(device='cuda:0')).all()), 'Nonfinite new gradient')
        else:
            old = prior['probes'][index]
            _recheck_refs(prior['refs'])
            fields = compare_fields(result, old['result'])
            previous_gradients = torch.load(old['gradient_ref']['path'], map_location='cpu', weights_only=True)
            compared = compare_gradients(gradients, previous_gradients, torch=torch)
            exact = all(fields.values()) and all(value['exact_equal'] for value in compared.values())
            result['historical_comparison'] = {
                'status': 'EXACT_SAVED_EVIDENCE_OVERLAP' if exact else 'SAVED_EVIDENCE_DIFFERENCE',
                'previous_mode': old['mode'], 'previous_result': old['result_ref'],
                'previous_gradient': old['gradient_ref'], 'exact_fields': fields,
                'gradient_comparison': compared, 'compared_gradient_tensors': len(compared),
                'original_long_gradient_equivalence': 'UNVERIFIED'}
            require(exact, 'New CUDA probe differs from fixed historical evidence')
            del previous_gradients
        _recheck_refs(prior['refs'])
    except BaseException as exc:
        error = exc
        result.update(status='OOM' if type(exc).__name__ == 'OutOfMemoryError' else 'FAILED',
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        # Release all graph/snapshot references before checking the next row.
        current = ref = loss = scaled = sampler = gradients = None
        currents = refs = prepared = item = previous_gradients = None
        def restore_modes():
            for name, module in runtime.model.named_modules():
                module.train(before['model_modes'][name])
        def check_state():
            result['state_unchanged_after_attempt'] = _state(runtime, optimizer, scheduler, reference) == before
            require(result['state_unchanged_after_attempt'], 'Probe changed model/reference/optimizer/scheduler/mode')
        def check_gpu1():
            result['gpu1_rng_unchanged_after_attempt'] = state_hash(capture_rng()['cuda'][1]) == gpu1_rng_sha
            require(result['gpu1_rng_unchanged_after_attempt'], 'Snapshot GPU consumed RNG')
        def restore_and_check_rng():
            restore_rng(rng)
            result['restored_two_gpu_rng_sha256'] = state_hash(capture_rng())
            result['rng_restored_after_attempt'] = result['restored_two_gpu_rng_sha256'] == rng_sha
            result['restored_rng_sha256'] = state_hash(original_rng_projection(capture_rng()))
            require(result['rng_restored_after_attempt'] and result['restored_rng_sha256'] == manifest['rng_sha256'],
                    'Full two-GPU RNG restoration differs')
        audit_errors = _cleanup_attempt((
            ('zero_grad', lambda: optimizer.zero_grad(set_to_none=True)),
            ('restore_model_modes', restore_modes), ('immutable_state', check_state),
            ('gpu1_rng', check_gpu1), ('restore_full_rng', restore_and_check_rng)))
        if audit_errors:
            result['final_audit_errors'] = [{'stage': stage, 'type': type(exc).__name__, 'error': str(exc)}
                                          for stage, exc in audit_errors]
            if error is None:
                error = audit_errors[0][1]
                result['status'] = 'FAILED_FINAL_AUDIT'
        if hook is not None:
            result['saved_tensor_offload'] = dict(hook.stats)
        try:
            result['gpu_memory'] = gpu_peak_stats()
        except Exception as memory_error:
            result['gpu_memory_error'] = str(memory_error)
        result['elapsed_seconds_including_gradient_and_comparison_evidence'] = time.monotonic() - started
        write_once(folder / 'RESULT.json', result)
    if error is not None:
        raise RuntimeError('GPU historical overlap failed; original evidence retained at ' + str(folder)) from error
    return result


def verify_overlap(runtime, optimizer, scheduler, records, reference, manifest, output):
    """Run exactly the fixed first/longest original probes, or fail before full batch."""
    from adapter.training import validate_records
    from mm_core.training import state_hash, trainable_state
    started = time.monotonic()
    package = Path(runtime.identity['run_root']).resolve(strict=True)
    destination = Path(output)
    require(destination.is_absolute() and destination.parent.resolve(strict=True) == destination.parent
            and destination.is_relative_to(package / 'run')
            and not destination.is_relative_to(package / 'run/paths') and not destination.exists(),
            'Overlap evidence requires a new directory outside production paths')
    require(runtime.identity['step'] == manifest['checkpoint_step']
            and state_hash(trainable_state(runtime.model)) == manifest['policy_sha256']
            and state_hash(reference) == manifest['reference_sha256'], 'Restored original checkpoint differs')
    expected_ids = [rid for group in manifest['plan_step']['groups'] for rid in group['request_ids']]
    require(manifest['probes'] == select_probes(records, expected_ids) and len(manifest['probes']) == 2,
            'Only fixed first and longest original probes are allowed')
    coefficients, _ = validate_records(runtime.identity, records, manifest['plan_step'],
                                        manifest['policy_sha256'], manifest['reference_sha256'])
    pair = validate_gpu_pair()
    prior = load_prior_evidence(package, manifest)
    destination.mkdir()
    write_once(destination / 'PRIOR_EVIDENCE.json', {k:v for k,v in prior.items() if k != 'probes'})
    summary = {'status': 'UNFINISHED', 'path': manifest['path'], 'gpu_pair': pair,
               'new_generations': 0, 'optimizer_steps': 0, 'scheduler_steps': 0,
               'production_recovery_authorized': False, 'prior_job': PRIOR_JOB,
               'original_long_baseline_status': prior.get('original_long_baseline_status', 'NOT_AVAILABLE'),
               'original_long_gradient_equivalence': 'UNVERIFIED',
               'prior_cpu_result_reused_as_evidence_only': prior['status'] != 'NOT_AVAILABLE',
               'new_cpu_snapshot_execution': False, 'probes': []}
    try:
        for index in (0, 1):
            result = _probe(runtime, optimizer, scheduler, records, reference, manifest,
                            destination, index, coefficients, prior)
            result_path = destination / f'probe-{index}-{OFFLOAD_MODE}' / 'RESULT.json'
            summary['probes'].append({'probe': result['probe'], 'status': result['status'],
                                       'comparison': result['historical_comparison']['status'],
                                       'result': {'path': str(result_path), 'sha256': file_hash(result_path)}})
        _recheck_refs(prior['refs'])
        summary['status'] = ('NOT_AVAILABLE' if prior['status'] == 'NOT_AVAILABLE'
                             else 'EXACT_HISTORICAL_OVERLAP')
    except BaseException as exc:
        summary.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        summary['elapsed_seconds_including_probe_evidence'] = time.monotonic() - started
        write_once(destination / 'SUMMARY.json', summary)
    return summary
