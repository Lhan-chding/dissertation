"""Independent GPU-only full-original-batch memory/gate probe; intentionally stops before Adam.

The production update function remains unchanged. A verified optimizer pre-hook
captures its already-clipped gradients and raises a dedicated BaseException,
allowing the original finally block to clear gradients and restore RNG. No new
sampling, optimizer body, scheduler step, production write, or COMMIT is allowed.
Passing is engineering evidence only: long-row baseline gradient equivalence
remains unverified where the original gradient execution ran out of memory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import sys
import traceback
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from adapter.transactions import file_hash, read_json, require, write_once
from diagnostics.gpu_capacity_v2 import GPUSavedTensors, CAP_BYTES, validate_snapshot_stats
from diagnostics.gpu_capacity_overlap_v2 import verify_overlap
from diagnostics.gpu_recovery_state import make_diagnostic_binding, restore_latest_gpu, actual_cuda_identity
from diagnostics.memory_equivalence import (host_preflight,
    MINIMUM_HOST_MIB, inspect_source, readonly_path, production_snapshot, record_final_audit)


class StopBeforeAdam(BaseException):
    """Intentional pre-body stop, deliberately outside the native Exception handler."""


def step_memory_budget(host_budget, worker_mib):
    require(type(worker_mib) is int and worker_mib > 0, 'Positive explicit worker memory grant required')
    result = dict(host_budget, explicit_step_grant_mib=worker_mib)
    result['effective_mib'] = min(host_budget['effective_mib'], worker_mib)
    result['status'] = 'PASS' if result['effective_mib'] >= MINIMUM_HOST_MIB else 'BLOCKED_HOST_MEMORY'
    return result


def optimizer_hook_evidence(optimizer):
    import importlib
    import torch
    module = importlib.import_module('torch.optim.optimizer')
    register = inspect.getsource(torch.optim.Optimizer.register_step_pre_hook)
    wrapper = inspect.getsource(torch.optim.Optimizer.profile_hook_step)
    require(wrapper.index('_optimizer_step_pre_hooks') < wrapper.index('out = func(*args, **kwargs)'),
            'Installed optimizer pre-hook ordering is not verified')
    require(not module._global_optimizer_pre_hooks and not module._global_optimizer_post_hooks and
            not optimizer._optimizer_step_pre_hooks and not optimizer._optimizer_step_post_hooks,
            'Unexpected optimizer hooks may mutate the isolated state')
    return {'torch_version': torch.__version__, 'source_file': str(Path(inspect.getfile(torch.optim.Optimizer))),
            'source_file_sha256': file_hash(inspect.getfile(torch.optim.Optimizer)),
            'register_step_pre_hook_source': register, 'profile_hook_step_source': wrapper,
            'profile_hook_step_sha256': hashlib.sha256(wrapper.encode()).hexdigest(),
            'pre_hooks_before_adam_body': True}


def immutable_state(runtime, optimizer, scheduler, reference):
    from mm_core.training import state_hash, trainable_state
    return {'policy': state_hash(trainable_state(runtime.model)), 'reference': state_hash(reference),
            'optimizer': state_hash(optimizer.state_dict()), 'scheduler': state_hash(scheduler.state_dict()),
            'runtime_identity': copy.deepcopy(runtime.identity), 'model_training': runtime.model.training}


def collect_baseline(runtime, records, reference, destination):
    """Original full-token policy/reference forward, no gradient and no offload."""
    import torch
    from adapter.training import _verify_prepared
    from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
    before = state_hash(trainable_state(runtime.model)), state_hash(reference)
    rng = capture_rng()
    old_training = runtime.model.training
    result = []
    try:
        runtime.model.eval()
        with Path(destination).open('x') as stream:
            for row in records:
                prepared = runtime.prepare_input(row['model_input'], Path(runtime.identity['run_root']),
                                                 protocol='structured_v4')
                _verify_prepared(row, prepared, torch)
                item = dict(row, prepared=prepared)
                with torch.no_grad():
                    refs = runtime.batch_reference_forward([item], reference)
                    currents = runtime.batch_sequence_forward([item], purpose='training_policy', grad=False)
                require(len(refs) == len(currents) == 1 and currents[0].shape == refs[0].shape and
                        currents[0].ndim == 1 and len(currents[0]) == len(row['tokens']),
                        'Original baseline trajectory length differs')
                value = {'row_id': row['row_id'], 'token_path_sha256': row['token_path_sha256'],
                         'prompt_tensor_hash': row['prompt_tensor_hash'], 'tokens': row['tokens'],
                         'train_logprobs': currents[0].detach().cpu().tolist(),
                         'reference_logprobs': refs[0].detach().cpu().tolist()}
                require(all(math.isfinite(v) for v in value['train_logprobs']+value['reference_logprobs']),
                        'Nonfinite original no-grad baseline')
                result.append(value)
                stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        restore_rng(rng)
        runtime.model.train(old_training)
    require((state_hash(trainable_state(runtime.model)), state_hash(reference)) == before,
            'Baseline changed policy/reference')
    return result


def compare_probability_vectors(baseline, audit):
    require(len(baseline) == len(audit) == 128, 'Full 128 probability vectors required')
    comparisons = []
    for expected, actual in zip(baseline, audit, strict=True):
        require(expected['row_id'] == actual['row_id'] and expected['tokens'] == actual['tokens'] and
                expected['token_path_sha256'] == actual['token_path_sha256'], 'Baseline/audit trajectory differs')
        item = {'row_id': expected['row_id']}
        for key in ('train_logprobs', 'reference_logprobs'):
            left, right = expected[key], actual[key]
            require(len(left) == len(right) == len(expected['tokens']), 'Probability vector length differs')
            item[key] = {'exact_equal': left == right,
                         'max_absolute_difference': max(abs(a-b) for a, b in zip(left, right, strict=True))}
        comparisons.append(item)
    return {'status': 'EXACT_ALL_128' if all(v[k]['exact_equal'] for v in comparisons
                  for k in ('train_logprobs','reference_logprobs')) else 'PROBABILITY_DIFFERENCE',
            'rows': comparisons, 'gradient_equivalence': 'UNVERIFIED_FOR_ORIGINAL_LONG_BASELINE_OOM'}


def probe_update(runtime, optimizer, scheduler, records, reference, plan, output):
    """Call unchanged update, intercept only grad forwards and the pre-Adam hook."""
    import torch
    from adapter.training import update
    from mm_core.training import capture_rng, state_hash
    output = Path(output)
    require(output.is_dir() and len(records) == 128, 'Existing isolated output and full batch required')
    evidence = optimizer_hook_evidence(optimizer)
    before = immutable_state(runtime, optimizer, scheduler, reference)
    rng_before = state_hash(capture_rng())
    audit_file = Path(runtime.identity['audit_root']) / 'NUMERICAL_TOKEN_AUDIT.jsonl'
    require(not audit_file.exists(), 'Isolated numerical audit must be new')
    write_once(output / 'OPTIMIZER_PREHOOK_SOURCE.json', evidence)
    forwards, capture = [], {'pre_hook_calls': 0}
    original = runtime.batch_sequence_forward
    had_instance_method = 'batch_sequence_forward' in vars(runtime)
    instance_method = vars(runtime).get('batch_sequence_forward')

    def bounded_forward(items, *, purpose, grad):
        if not grad:
            return original(items, purpose=purpose, grad=grad)
        require(purpose == 'training_policy' and len(items) == 1, 'Unexpected gradient forward')
        hook = GPUSavedTensors((*runtime.model.parameters(), *runtime.model.buffers()), row_id=items[0]['row_id'])
        entry = {'row_id': items[0]['row_id'], 'stats': hook.stats}
        forwards.append(entry)
        try:
            with hook.context():
                value = original(items, purpose=purpose, grad=grad)
            entry['forward_returned'] = True
            return value
        except Exception as error:
            entry.update(error_type=type(error).__name__, error=str(error))
            raise

    def before_adam(opt, args, kwargs):
        capture['pre_hook_calls'] += 1
        try:
            require(opt is optimizer and capture['pre_hook_calls'] == 1, 'Unexpected optimizer hook invocation')
            events = [json.loads(line) for line in audit_file.read_text().splitlines()]
            require(len(events) == 1 and events[0]['status'] == 'PREUPDATE_PASS' and
                    events[0]['precommit_gate'] == 'PASS_BEFORE_ADAM' and
                    len(events[0]['probability_vectors']) == 128, 'Native pre-Adam audit not complete')
            if hasattr(runtime, 'reserve'):
                runtime.reserve('diagnostic_gradient_calls', 128, method='full_batch_pre_adam_probe',
                                completed=True, optimizer_body_calls=0)
            gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                         for name, parameter in runtime.model.named_parameters() if parameter.requires_grad}
            require(gradients and any(v is not None for v in gradients.values()) and
                    all(v is None or bool(torch.isfinite(v).all()) for v in gradients.values()),
                    'Missing/nonfinite clipped adapter gradients')
            norm = math.sqrt(sum(float(v.double().square().sum()) for v in gradients.values() if v is not None))
            require(norm <= 1+1e-5 and math.isclose(norm, events[0]['numerics']['gradient_norm_after_clip'],
                    rel_tol=1e-6, abs_tol=1e-7), 'Captured clipped gradient norm differs')
            gradient_file = output / 'CLIPPED_ADAPTER_GRADIENTS.pt'
            with gradient_file.open('xb') as stream:
                torch.save(gradients, stream)
            capture.update(status='PRE_ADAM_CAPTURED', gradient_file=str(gradient_file),
                gradient_file_sha256=file_hash(gradient_file), gradient_names=list(gradients),
                gradient_none_names=[k for k,v in gradients.items() if v is None],
                gradient_norm_after_clip=norm, native_audit_sha256=file_hash(audit_file))
        except Exception as error:
            capture.update(status='PRE_ADAM_CAPTURE_FAILED', error_type=type(error).__name__,
                           error=str(error), traceback=traceback.format_exc())
        raise StopBeforeAdam()

    runtime.batch_sequence_forward = bounded_forward
    handle = optimizer.register_step_pre_hook(before_adam)
    stopped = False
    try:
        update(runtime, optimizer, scheduler, records, reference, plan)
        raise RuntimeError('Optimizer pre-hook did not stop update before Adam')
    except StopBeforeAdam:
        stopped = True
    finally:
        handle.remove()
        if had_instance_method:
            runtime.batch_sequence_forward = instance_method
        else:
            del runtime.batch_sequence_forward
        write_once(output / 'BOUNDED_FORWARD_RECORDS.json', {'forwards': forwards,
                   'cap_reset': 'each training_policy gradient forward', 'production_recovery_authorized': False})
        write_once(output / 'PRE_ADAM_CAPTURE.json', capture)
        write_once(output / 'DIAGNOSTIC_BACKWARD_ACCOUNTING.json', {
            'gradient_forwards_attempted': len(forwards),
            'gradient_forwards_returned': sum(v.get('forward_returned') is True for v in forwards),
            'backward_calls_completed': 128 if capture['pre_hook_calls'] == 1 else None,
            'backward_completion_status': 'ALL_128_BEFORE_PRE_ADAM_HOOK' if capture['pre_hook_calls'] == 1
                                          else 'UNKNOWN_PARTIAL_NO_COMPLETION_INFERENCE_FROM_FORWARD_COUNT',
            'optimizer_body_calls': 0 if stopped else None, 'new_generations': 0})
    require(stopped and capture.get('status') == 'PRE_ADAM_CAPTURED' and capture['pre_hook_calls'] == 1,
            'Full batch did not reach a verified pre-Adam stop')
    require(len(forwards) == 128 and [v['row_id'] for v in forwards] == [r['row_id'] for r in records],
            'Full batch gradient forward order/count differs')
    for entry in forwards:
        validate_snapshot_stats(entry['stats'])
    require(immutable_state(runtime, optimizer, scheduler, reference) == before and
            state_hash(capture_rng()) == rng_before, 'Native finally did not preserve state/RNG')
    require(all(p.grad is None for p in runtime.model.parameters()), 'Native finally did not clear gradients')
    event = read_json_lines(audit_file)[0]
    return {'status': 'FULL_BATCH_PRE_ADAM_PASS_NO_UPDATE', 'capture': capture,
            'native_audit': event, 'gradient_forwards': 128, 'optimizer_body_calls': 0,
            'scheduler_steps': 0, 'new_generations': 0, 'state_unchanged': True, 'rng_restored': True,
            'saved_tensor_copied_bytes_total': sum(v['stats']['copied_bytes'] for v in forwards),
            'saved_tensor_max_bytes_per_forward': max(v['stats']['copied_bytes'] for v in forwards),
            'original_long_gradient_equivalence': 'UNVERIFIED', 'production_recovery_authorized': False}


def read_json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def require_output(output):
    output = Path(os.path.abspath(output))
    require(output.parent == PACKAGE/'run' and output.name.startswith('gpu_capacity_v2_diag_') and
            output.parent.resolve(strict=True) == output.parent and not output.exists() and not output.is_symlink(),
            'Output must be a new separate package/run/gpu_capacity_v2_diag_* directory')
    output.mkdir()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True, type=Path)
    parser.add_argument('--path', required=True, choices=('b0-11031-G',))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--worker-memory-mib', required=True, type=int)
    args = parser.parse_args()
    output = require_output(args.output)
    started = time.monotonic()
    result = {'status': 'UNFINISHED', 'path': args.path, 'new_generations': 0, 'optimizer_steps': 0,
              'scheduler_steps': 0, 'production_recovery_authorized': False,
              'original_long_gradient_equivalence': 'UNVERIFIED',
              'diagnostic_source_sha256': file_hash(__file__),
              'gpu_storage_source_sha256': file_hash(PACKAGE/'diagnostics/gpu_capacity_v2.py'),
              'gpu_storage_capacity_bytes': CAP_BYTES}
    backend, manifest = None, None
    rng_hash = None
    original_path = PACKAGE/'run/paths'/args.path
    before = None
    try:
        budget = step_memory_budget(host_preflight(), args.worker_memory_mib)
        budget.pop('offload_cap_mib', None)
        budget['interpretation'] = 'Conservative base-model loading host floor; no activation CPU offload'
        budget['cpu_activation_offload_bytes'] = 0
        write_once(output/'HOST_PREFLIGHT.json', budget)
        require(budget['status'] == 'PASS', 'BLOCKED_HOST_MEMORY')
        write_once(output/'GPU_PAIR_PREFLIGHT.json', actual_cuda_identity())
        manifest, records = inspect_source(args.binding.resolve(strict=True), args.path)
        write_once(output/'MANIFEST.json', manifest)
        binding = read_json(args.binding.resolve(strict=True))
        identity = manifest['identity']
        binding['path_identity'] = {k: identity[k] for k in ('arm','train_seed','run_id')}
        execution = output/'worker'
        execution.mkdir()
        with readonly_path(original_path) as tx:
            from diagnostics.gpu_backend import NativeBackend
            from adapter.training import validate_records, canonical_hash
            from adapter.seal_smoke import independent_scores
            binding = make_diagnostic_binding(args.binding.resolve(strict=True), output,
                [Path(__file__), PACKAGE/'diagnostics/gpu_saved_tensors.py',
                 PACKAGE/'diagnostics/memory_equivalence.py',
                 PACKAGE/'diagnostics/gpu_gradient_overlap.py',
                 PACKAGE/'diagnostics/gpu_capacity_v2.py',
                 PACKAGE/'diagnostics/gpu_capacity_overlap_v2.py',
                 PACKAGE/'diagnostics/run_gpu_capacity_v2.sh',
                 PACKAGE/'diagnostics/GPU_CAPACITY_V2.md'], path_identity=binding['path_identity'])
            write_once(output/'GPU_DIAGNOSTIC_BINDING.json', binding)
            backend = NativeBackend(binding, identity['parent'], execution, 'TRAINING')
            import torch
            from mm_core.training import state_hash, capture_rng, restore_rng
            runtime = backend.runtime
            require(all(str(p.device) == 'cuda:0' for p in (*runtime.model.parameters(), *runtime.model.buffers())),
                    'GPU oracle model and buffers must remain on cuda:0')
            marker = restore_latest_gpu(tx, runtime, backend.optimizer, backend.scheduler, backend.reference, binding, execution)
            require(marker['step'] == manifest['checkpoint_step'], 'Original checkpoint cursor differs')
            require(all(v.device.type == 'cpu' and v.numel() == 1 or str(v.device) == 'cuda:0'
                        for state in backend.optimizer.state.values() for v in state.values() if torch.is_tensor(v)),
                    'GPU oracle optimizer tensor moved outside cuda:0')
            runtime.identity['plan_step_sha256'] = canonical_hash(manifest['plan_step'])
            cases = {r['case_id']:r for r in read_json_lines(PACKAGE/'frozen/private_cases.jsonl')}
            scorer = independent_scores(PACKAGE, records, cases)
            write_once(output/'INDEPENDENT_NATIVE_SCORES.json', scorer)
            scores = {s['request_id']:s['score'] for s in scorer['scores']}
            records = [dict(row, score=scores[row['request_id']]) for row in records]
            validate_records(runtime.identity, records, manifest['plan_step'], backend.policy_hash(),
                             state_hash(backend.reference))
            before = immutable_state(runtime, backend.optimizer, backend.scheduler, backend.reference)
            rng = capture_rng()
            rng_hash = state_hash(rng)  # Explicit two-GPU lineage is separately verified by restore_latest_gpu.
            [torch.cuda.reset_peak_memory_stats(device) for device in range(2)]
            result['capacity_gradient_overlap'] = verify_overlap(runtime, backend.optimizer, backend.scheduler,
                records, backend.reference, manifest, output/'overlap')
            baseline = collect_baseline(runtime, records, backend.reference, output/'BASELINE_PROBABILITIES.jsonl')
            require(len(baseline) == 128, 'Original no-grad baseline incomplete')
            restore_rng(rng)
            probe_started = time.monotonic()
            probe = probe_update(runtime, backend.optimizer, backend.scheduler, records, backend.reference,
                                 manifest['plan_step'], output)
            result['full_batch_end_to_end_seconds_including_evidence'] = time.monotonic() - probe_started
            comparison = compare_probability_vectors(baseline, probe['native_audit']['probability_vectors'])
            write_once(output/'PROBABILITY_COMPARISON.json', comparison)
            result.update(probe, probability_comparison=comparison['status'],
                          baseline_responses=128, baseline_sha256=file_hash(output/'BASELINE_PROBABILITIES.jsonl'))
            require(comparison['status'] == 'EXACT_ALL_128', 'No-grad baseline and offload probabilities differ')
            require(probe['saved_tensor_copied_bytes_total'] > 0, 'Offload path was not exercised')
    except Exception as error:
        result.update(status='OOM' if type(error).__name__ == 'OutOfMemoryError' else 'FAILED',
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        def audit():
            if backend is not None:
                result['gpu_peak_allocated_bytes'] = [backend.torch.cuda.max_memory_allocated(d) for d in range(2)]
                result['gpu_peak_reserved_bytes'] = [backend.torch.cuda.max_memory_reserved(d) for d in range(2)]
                if before is not None:
                    result['state_unchanged_after_attempt'] = immutable_state(
                        backend.runtime, backend.optimizer, backend.scheduler, backend.reference) == before
                    require(result['state_unchanged_after_attempt'], 'Diagnostic changed isolated state')
                    result['rng_restored_after_attempt'] = rng_hash is not None and state_hash(capture_rng()) == rng_hash
                    require(result['rng_restored_after_attempt'], 'Diagnostic RNG changed')
            if manifest is not None:
                with readonly_path(original_path):
                    result['production_files_unchanged'] = production_snapshot(original_path) == manifest['production_files']
                require(result['production_files_unchanged'], 'Production evidence changed')
        record_final_audit(result, audit)
        accounting = output/'DIAGNOSTIC_BACKWARD_ACCOUNTING.json'
        if accounting.exists():
            result['diagnostic_backward_accounting'] = read_json(accounting)
        if result.get('final_audit_error') and result['status'] == 'FULL_BATCH_PRE_ADAM_PASS_NO_UPDATE':
            result['status'] = 'FAILED_FINAL_AUDIT'
        result['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != 'darwin' else 1)
        result['allocator_environment'] = {k:os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')}
        result['elapsed_seconds'] = time.monotonic() - started
        result['activation_storage'] = 'cuda:1; arithmetic cuda:0; CPU fallback forbidden'
        write_once(output/'SUMMARY.json', result)
    print(json.dumps({k:result[k] for k in ('path','status','optimizer_steps','production_recovery_authorized')}))
    raise SystemExit(0 if result['status'] == 'FULL_BATCH_PRE_ADAM_PASS_NO_UPDATE' else 1)


if __name__ == '__main__':
    main()
