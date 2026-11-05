"""GPU-only saved-tensor execution around the unchanged registered update.

The first recovered b0-G step9 update must match the retained diagnostic gradients
before Adam is permitted. All other arithmetic, sequence order and gates are
the original adapter.training.update. No new sampling occurs in this module.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import time

from adapter.transactions import file_hash, read_json, require, write_once
from diagnostics.gpu_capacity_v2 import (GPUSavedTensors, CAP_BYTES, validate_snapshot_stats,
    gpu_peak_stats as gpu_memory_snapshot)


def check_optimizer_hooks(optimizer):
    import torch
    from torch.optim import optimizer as module
    source = inspect.getsource(torch.optim.Optimizer.profile_hook_step)
    require(source.index('_optimizer_step_pre_hooks') < source.index('out = func(*args, **kwargs)'),
            'Pre-Adam hook order not verified')
    require(not module._global_optimizer_pre_hooks and not module._global_optimizer_post_hooks
            and not optimizer._optimizer_step_pre_hooks and not optimizer._optimizer_step_post_hooks,
            'Unexpected optimizer hooks')
    return hashlib.sha256(source.encode()).hexdigest()


def compare_audit(actual, expected):
    # GPU1 has independently recorded RNG state. All science/numerics must agree.
    omit = {'rng_before_sha256'}
    require({k: v for k, v in actual.items() if k not in omit} ==
            {k: v for k, v in expected.items() if k not in omit},
            'Production pre-Adam objective/probability/numerics differs from diagnostic')


def compare_gradients(model, path, expected_sha):
    import torch
    require(file_hash(path) == expected_sha, 'Diagnostic gradient bytes changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    actual = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    require(set(saved) == set(actual) and bool(saved), 'Gradient inventory differs')
    for name, value in saved.items():
        current = actual[name]
        if value is None:
            require(current is None, 'Gradient presence differs: ' + name)
        else:
            require(current is not None and current.dtype == value.dtype and current.shape == value.shape
                    and str(current.device) == 'cuda:0', 'Gradient dtype/shape/device differs: ' + name)
            require(torch.equal(current, value.to(device=current.device)), 'Gradient differs: ' + name)
    return len(saved)


def first_recovery_evidence(runtime, step, binding, actual_audit):
    """Require exact accepted b0-G step9 evidence at the actual Adam boundary."""
    if runtime.identity['run_id'] != 'SR-ORD-TRAIN-1/b0-11031-G' or step != 9:
        return {'first_recovery_comparison': 'NOT_THIS_REGISTERED_B0G_STEP9'}
    acceptance_ref = binding['gpu_production']['acceptance']
    require(file_hash(acceptance_ref['path']) == acceptance_ref['sha256'], 'Production acceptance changed')
    acceptance = read_json(acceptance_ref['path'])
    summary_ref = acceptance['diagnostic_summary']
    require(file_hash(summary_ref['path']) == summary_ref['sha256'], 'Diagnostic summary changed')
    summary = read_json(summary_ref['path'])
    require(summary['path'] == 'b0-11031-G' and summary['status'] == 'FULL_BATCH_PRE_ADAM_PASS_NO_UPDATE'
            and summary['native_audit']['run_id'] == runtime.identity['run_id']
            and summary['native_audit']['step'] == step,
            'Accepted diagnostic path/step differs from first recovery')
    compare_audit(actual_audit, summary['native_audit'])
    capture = summary['capture']
    gradient_path = Path(capture['gradient_file'])
    require(gradient_path.parent == Path(summary_ref['path']).parent,
            'Gradient evidence must belong to accepted diagnostic')
    count = compare_gradients(runtime.model, gradient_path, capture['gradient_file_sha256'])
    require(count == 496, 'Exactly496 registered adapter gradients required')
    return dict(first_recovery_comparison='EXACT_DIAGNOSTIC_128_AND_CLIPPED_GRADIENTS',
                compared_gradients=count, diagnostic_summary=copy.deepcopy(summary_ref),
                gradient_sha256=capture['gradient_file_sha256'])


def verify_storage(forwards, records):
    require(len(forwards) == len(records) == 128 and
            [v['row_id'] for v in forwards] == [v['row_id'] for v in records], 'Gradient sequence order differs')
    for row in forwards:
        require(row.get('forward_returned') is True, 'GPU saved-tensor forward incomplete')
        validate_snapshot_stats(row['stats'])


def validate_compute_placement(runtime, optimizer):
    import torch
    require(all(str(p.device) == 'cuda:0' for p in (*runtime.model.parameters(), *runtime.model.buffers())),
            'Production model/buffers moved off cuda:0')
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                require(str(value.device) == 'cuda:0' or (key == 'step' and value.numel() == 1),
                        'Adam state moved off cuda:0')


def production_update(runtime, optimizer, scheduler, records, reference, plan, execution, binding):
    import torch
    from adapter.training import update
    from mm_core.training import state_hash
    execution = Path(execution)
    step = plan['step']
    check_path = execution / f'GPU_PRE_ADAM_CHECK_step-{step:03d}.json'
    accounting_path = execution / f'GPU_UPDATE_ACCOUNTING_step-{step:03d}.json'
    require(not check_path.exists() and not accounting_path.exists(), 'Step execution is not reusable')
    source_sha = check_optimizer_hooks(optimizer)
    validate_compute_placement(runtime, optimizer)
    gpu_memory_snapshot(reset=True)
    gpu1 = torch.cuda.get_rng_state(1).clone()
    original = runtime.batch_sequence_forward
    had_forward = 'batch_sequence_forward' in vars(runtime)
    old_forward = vars(runtime).get('batch_sequence_forward')
    old_scheduler = scheduler.step
    had_scheduler = 'step' in vars(scheduler)
    scheduler_override = vars(scheduler).get('step')
    forwards = []
    counts = dict(pre_adam_calls=0, adam_permitted=False, optimizer_returned_calls=0,
                  scheduler_entered_calls=0, scheduler_returned_calls=0)
    result = None
    succeeded = False
    started = time.monotonic()

    def bounded_forward(items, *, purpose, grad):
        if not grad:
            return original(items, purpose=purpose, grad=grad)
        require(purpose == 'training_policy' and len(items) == 1, 'Unexpected gradient forward')
        hook = GPUSavedTensors((*runtime.model.parameters(), *runtime.model.buffers()), row_id=items[0]['row_id'])
        entry = dict(row_id=items[0]['row_id'], stats=hook.stats)
        forwards.append(entry)
        with hook.context():
            value = original(items, purpose=purpose, grad=grad)
        entry['forward_returned'] = True
        return value

    def before_adam(opt, args, kwargs):
        counts['pre_adam_calls'] += 1
        check = dict(status='DENIED_BEFORE_ADAM', step=step, optimizer_hook_source_sha256=source_sha,
                     cpu_activation_offload_bytes=0, original_long_gradient_equivalence='UNVERIFIED_ORIGINAL_BASELINE_OOM')
        try:
            require(opt is optimizer and counts['pre_adam_calls'] == 1, 'Duplicate optimizer call')
            verify_storage(forwards, records)
            require(torch.equal(gpu1, torch.cuda.get_rng_state(1)), 'GPU1 RNG changed during backward')
            events = [json.loads(line) for line in (execution/'NUMERICAL_TOKEN_AUDIT.jsonl').read_text().splitlines()]
            matches = [e for e in events if e['step'] == step and e['status'] == 'PREUPDATE_PASS']
            require(len(matches) == 1 and matches[0]['precommit_gate'] == 'PASS_BEFORE_ADAM',
                    'Missing/ambiguous native pre-Adam evidence')
            check['gpu1_rng_sha256'] = state_hash(gpu1)
            check.update(first_recovery_evidence(runtime, step, binding, matches[0]))
            check.update(status='GPU_PRE_ADAM_PASS', gradient_forwards=128,
                         copied_bytes=sum(v['stats']['copied_bytes'] for v in forwards), gpu1_rng_unchanged=True)
        except BaseException as error:
            check.update(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            write_once(check_path, check)
        counts['adam_permitted'] = True

    def after_adam(opt, args, kwargs):
        counts['optimizer_returned_calls'] += 1

    def counted_scheduler(*args, **kwargs):
        counts['scheduler_entered_calls'] += 1
        value = old_scheduler(*args, **kwargs)
        counts['scheduler_returned_calls'] += 1
        return value

    runtime.batch_sequence_forward = bounded_forward
    scheduler.step = counted_scheduler
    pre = optimizer.register_step_pre_hook(before_adam)
    post = optimizer.register_step_post_hook(after_adam)
    try:
        result = update(runtime, optimizer, scheduler, records, reference, plan)
        require(counts == dict(pre_adam_calls=1, adam_permitted=True, optimizer_returned_calls=1,
                               scheduler_entered_calls=1, scheduler_returned_calls=1), 'Update call counts differ')
        result['gpu_pre_adam_check'] = dict(path=str(check_path), sha256=file_hash(check_path))
        succeeded = True
        return result
    finally:
        pre.remove(); post.remove()
        if had_forward:
            runtime.batch_sequence_forward = old_forward
        else:
            del runtime.batch_sequence_forward
        if had_scheduler:
            scheduler.step = scheduler_override
        else:
            del scheduler.step
        memory_error = None
        try:
            memory = gpu_memory_snapshot()
        except Exception as error:
            memory = None
            memory_error = dict(error_type=type(error).__name__, error=str(error))
        write_once(accounting_path, dict(counts, step=step, status='UPDATED_UNCOMMITTED' if succeeded and memory_error is None else 'FAILED',
            elapsed_seconds=time.monotonic()-started, forwards=forwards, new_generations=0,
            gpu_memory=memory, gpu_memory_error=memory_error,
            note='adam_permitted means body may have started; optimizer_returned_calls counts successful returns only'))
        if succeeded:
            require(memory_error is None, 'GPU telemetry failed after update; accounting retained; no COMMIT')
            result['gpu_update_accounting'] = dict(path=str(accounting_path), sha256=file_hash(accounting_path))
