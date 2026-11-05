"""Fixed b0G first-row 16/48 GiB exact overlap and known failed-row 48 GiB probe.

All arithmetic is the original CUDA 0 native objective. CPU tensors are saved
comparison evidence only. Original failed long-row gradients remain unavailable.
"""
from __future__ import annotations
import copy
import os
from pathlib import Path
import time
import traceback
from adapter.transactions import file_hash, read_json, require, write_once
from diagnostics.gpu_saved_tensors import GPUSavedTensors as Original16
from diagnostics.gpu_capacity_v2 import GPUSavedTensors as Capacity48, gpu_peak_stats, validate_gpu_pair, validate_snapshot_stats
from diagnostics.gpu_recovery_state import original_rng_projection
from diagnostics.gpu_gradient_overlap import (EXPECTED_GRADIENTS, _state, _objective_evidence,
    _cleanup_attempt, _recheck_refs, compare_fields, compare_gradients)

FAILURE_JOB = '274335'
FAILURE_ROW = 'OT1/TRAIN/b0/G/seed11031/s09/g07/d02'


def failure_evidence(package, records):
    """Pin the exact retained failure; never select rows using model outcomes."""
    root = Path(package)/'run/paths/b0-11031-G/executions'
    files = list(root.glob(FAILURE_JOB+'-*/GPU_UPDATE_ACCOUNTING_step-009.json'))
    require(len(files) == 1, 'Exactly one retained b0G GPU failure receipt required')
    f = files[0]
    require(f.resolve(strict=True) == f and not f.is_symlink(), 'Canonical failure receipt required')
    value = read_json(f)
    require(value['status'] == 'FAILED' and value['step'] == 9 and
            value['pre_adam_calls'] == value['optimizer_returned_calls'] ==
            value['scheduler_entered_calls'] == value['scheduler_returned_calls'] == 0 and
            value['new_generations'] == 0 and value['adam_permitted'] is False,
            'Failure must precede every Adam/scheduler body and generation')
    forwards = value['forwards']
    require(len(forwards) == 59 and all(v.get('forward_returned') is True for v in forwards[:-1]) and
            forwards[-1].get('forward_returned') is not True and forwards[-1]['row_id'] == FAILURE_ROW and
            [v['row_id'] for v in forwards] == [v['row_id'] for v in records[:59]],
            'Fixed first/failed original-row ordering differs')
    require(forwards[-1]['stats']['copied_bytes'] == 16*1024**3 and
            forwards[-1]['stats']['cap_skips'] > 0, 'Failure did not exhaust old GPU snapshot capacity')
    return {'path':str(f), 'sha256':file_hash(f)}


def _probe(runtime, optimizer, scheduler, records, reference, manifest, output, index,
           coefficients, prior, *, hook_class, mode):
    import torch
    from adapter.training import _verify_prepared
    from mm_core.training import capture_rng, restore_rng, state_hash
    from sr_f6.objective import sequence_loss

    started = time.monotonic()
    probe = manifest['probes'][index]
    position = next(i for i, row in enumerate(records) if row['request_id'] == probe['request_id'])
    row, advantage = records[position], coefficients[position]
    folder = output / f'probe-{index}-{mode}'
    folder.mkdir(exist_ok=False)
    before = _state(runtime, optimizer, scheduler, reference)
    rng = capture_rng()
    require(len(rng['cuda']) == 2 and state_hash(original_rng_projection(rng)) == manifest['rng_sha256'],
            'Two-GPU state does not project to original checkpoint RNG')
    rng_sha, gpu1_rng_sha = state_hash(rng), state_hash(rng['cuda'][1])
    result = {'status': 'UNFINISHED', 'mode': mode, 'probe': probe,
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
        hook = hook_class((*runtime.model.parameters(), *runtime.model.buffers()), row_id=row['row_id'])
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
        runtime.reserve('diagnostic_gradient_calls', 1, method='gpu_capacity_v2_overlap',
                        request_id=row['request_id'], optimizer_steps=0)
        scaled.backward()
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        require(hook.stats['copied_bytes'] > 0 and hook.stats['unpacked_copies'] > 0,
                'GPU saved-tensor copy/unpack mechanism was not exercised')
        validate_snapshot_stats(hook.stats, expected_cap=(16 if hook_class is Original16 else 48)*1024**3)
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
    from adapter.training import validate_records
    from mm_core.training import state_hash, trainable_state
    require(manifest['path'] == 'b0-11031-G' and manifest['checkpoint_step'] == 8 and
            manifest['next_step'] == 9 and len(records) == 128, 'Only frozen b0G step9 is eligible')
    package = Path(runtime.identity['run_root']).resolve(strict=True)
    output = Path(output)
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent and
            output.is_relative_to(package/'run') and not output.is_relative_to(package/'run/paths') and
            not output.exists(), 'New isolated comparison output required')
    require(runtime.identity['step'] == 8 and state_hash(trainable_state(runtime.model)) == manifest['policy_sha256']
            and state_hash(reference) == manifest['reference_sha256'], 'Checkpoint identity differs')
    coefficients, _ = validate_records(runtime.identity, records, manifest['plan_step'],
                                       manifest['policy_sha256'], manifest['reference_sha256'])
    failed = failure_evidence(package, records)
    selected = copy.deepcopy(manifest)
    selected['probes'] = [dict(request_id=records[i]['request_id'], completion_tokens=len(records[i]['tokens']),
                              roles=[role]) for i,role in ((0,'fixed_first_original'),(58,'retained_failure_274335'))]
    output.mkdir()
    write_once(output/'FIXED_PROBES.json', {'probes':selected['probes'], 'failure':failed,
                                         'old_capacity_bytes':16*1024**3,'new_capacity_bytes':48*1024**3})
    summary = {'status':'UNFINISHED', 'path':manifest['path'], 'gpu_pair':validate_gpu_pair(),
               'new_generations':0,'optimizer_steps':0,'scheduler_steps':0,
               'production_recovery_authorized':False,'original_long_gradient_equivalence':'UNVERIFIED',
               'failure_evidence':failed,'probes':[]}
    empty_prior = {'status':'NOT_AVAILABLE','reason':'Original failing long-row gradient unavailable', 'refs':[]}
    try:
        old = _probe(runtime, optimizer, scheduler, records, reference, selected, output, 0,
                     coefficients, empty_prior, hook_class=Original16, mode='original_gpu16')
        old_result = output/'probe-0-original_gpu16/RESULT.json'
        old_ref = {'path':str(old_result),'sha256':file_hash(old_result)}
        gradient_ref = {'path':old['gradient_file'],'sha256':old['gradient_file_sha256']}
        prior = {'status':'PINNED_GPU16','refs':[old_ref,gradient_ref],
                 'probes':[{'mode':'original_gpu16','result':old,'result_ref':old_ref,'gradient_ref':gradient_ref}]}
        new = _probe(runtime, optimizer, scheduler, records, reference, selected, output, 0,
                     coefficients, prior, hook_class=Capacity48, mode='capacity48_v2')
        require(new['historical_comparison']['status'] == 'EXACT_SAVED_EVIDENCE_OVERLAP',
                'Changed capacity differs on fixed original first row')
        long = _probe(runtime, optimizer, scheduler, records, reference, selected, output, 1,
                      coefficients, empty_prior, hook_class=Capacity48, mode='capacity48_v2')
        require(long['saved_tensor_offload']['copied_bytes'] > 16*1024**3,
                'Failed-row probe did not exercise increased GPU capacity')
        _recheck_refs([failed,old_ref,gradient_ref])
        summary.update(status='GPU16_GPU48_FIRST_ROW_EXACT_FAILED_ROW48_PASS',
            original_first_gradient_comparison=new['historical_comparison'],
            failed_row_status=long['status'],
            probes=[{'mode':r['mode'],'probe':r['probe'],'status':r['status']} for r in (old,new,long)])
    except BaseException as exc:
        summary.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_once(output/'SUMMARY.json',summary)
    return summary
