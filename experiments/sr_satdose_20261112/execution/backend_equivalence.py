"""Original shortest-probe replay with single-/two-device activation storage.

Both GPUs remain visible in BOTH paths. Baseline uses nullcontext and records
zero PyTorch tensor-allocation peak on cuda:1; the alternative uses the pinned
GPUSavedTensors cuda:0 -> cuda:1 snapshot hook. This is NOT a one-visible-GPU
startup test, full-batch certification, long-sequence stress or GPU admission.
No tokens are generated, truncated or synthesized, and no optimizer is created.

The sole selected row is the shortest original fixed probe response (stable
original-order tie break), regardless of J/c. Its advantage is computed from
its complete original G8 with the pinned native function, and loss retains /128.
Each start has one fixed write-once attempt under the technical binding. CPU
fixtures exercise orchestration/arithmetic only and cannot establish GPU facts.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from . import diagnostic_worker as worker, production, shadow_replay as shadow
from runtime import native_runtime
from runtime.backend import PUBLIC

require = shadow.require
_USED = False


def expected_slots():
    return {str(start): f'backend_equivalence/start-{start}' for start in (11031,11032)}


def select_shortest(probe_rows, kernels):
    """Selection uses lengths/order only; scoring is consulted only AFTER selection."""
    require(len(probe_rows) == 128 and len({r['request_id'] for r in probe_rows}) == 128,
            'Complete fixed probe128 required')
    require(all(isinstance(r['tokens'],list) and 1 <= len(r['tokens']) <= 1536 and
                all(type(t) is int and t >= 0 for t in r['tokens']) for r in probe_rows),
            'Original complete token vectors required')
    index = min(range(len(probe_rows)), key=lambda i: len(probe_rows[i]['tokens']))
    group_start = index // 8 * 8
    group = probe_rows[group_start:group_start+8]
    require(len({r['group_id'] for r in group}) == 1 and
            sum(r['group_id'] == group[0]['group_id'] for r in probe_rows) == 8,
            'Selected row lacks original contiguous complete G8')
    advantages = kernels.advantages([r['J'] for r in group])
    require(len(advantages) == 8, 'Native G8 advantage inventory differs')
    return dict(row=copy.deepcopy(probe_rows[index]), original_index=index,
                group_request_ids=[r['request_id'] for r in group],
                group_J=[r['J'] for r in group], advantage=float(advantages[index-group_start]),
                sequence_denominator=128, selection='minimum_original_token_length_stable_original_order')


def _no_unknown(root, binding):
    for start, rel in expected_slots().items():
        path = root / rel
        if path.exists():
            require(path.resolve() == path and not (path / 'HOLD.json').exists() and
                    (path / 'COMPLETE.json').is_file(), 'Backend-equivalence slot is HOLD/UNKNOWN; no retry')
            complete = shadow.read_json(path / 'COMPLETE.json')
            require(complete['status'] == 'BACKEND_EQUIVALENCE_CAPTURED_NOT_GPU_ADMISSION' and
                    complete['bit_exact'] is True and complete['start_seed'] == int(start),
                    'Backend-equivalence completion differs')
            for ref in complete['evidence'].values(): worker._bound(root, ref)


def _preflight(start, binding_sha):
    # Includes both complete DRY gates, full native scorer revalidation, complete
    # actual runtime source inventory and all imported runtime module paths.
    ctx = worker._preflight(binding_sha, start, 0)
    b = ctx['binding']
    require(b['backend_equivalence_slots'] == expected_slots() and
            b['max_backend_equivalence_attempts'] == 2, 'Fixed backend-equivalence slots/budget differ')
    actual = Path(__file__).resolve()
    require(Path(b['code']['backend_equivalence']['path']).resolve() == actual and
            shadow.file_sha(actual) == b['code']['backend_equivalence']['sha256'],
            'Backend-equivalence source is not technically bound')
    _no_unknown(ctx['root'], b)
    return ctx


def _forward_original(handle, row, *, two_device):
    """Frozen forward_original validation/math, with an explicit storage context.

    Baseline never constructs a saved-tensor hook. Neither path monkeypatches
    the runtime, changes teacher/reference math, or invokes generation.
    """
    runtime = handle.runtime
    request = handle.registry.request(row['request_id'])
    require(request['scope'] == 'DRY' and request['assistant_prefill'] == '{', 'Only original DRY replay')
    endpoint = handle.endpoint()
    require(row['policy_sha256'] == endpoint['policy_sha256'] and
            row['reference_sha256'] == endpoint['reference_sha256'], 'Stale replay policy/reference')
    capture = Path(row['native_capture_path']).resolve(strict=True)
    require(capture.is_relative_to(handle.output) and shadow.file_sha(capture) == row['native_capture_sha256'],
            'Capture changed/outside new audit')
    name = hashlib.sha256(row['request_id'].encode()).hexdigest() + '.CAPTURE.json'
    require(capture.name == name, 'Capture request name differs')
    begin = shadow.read_json(capture.with_name(name.replace('.CAPTURE.json','.BEGIN.json')))
    require(begin['request_id'] == row['request_id'] and begin['endpoint'] == endpoint and
            begin['design_freeze_sha256'] == handle.registry.freeze_sha256, 'Original BEGIN identity differs')
    saved = shadow.read_json(capture)
    for key in ('tokens','sampler_logprobs','prompt_tensor_hash','model_input','seed'):
        require(row[key] == saved[key], 'Original trajectory differs: '+key)
    require(row['seed'] == request['seed'] and row['model_input'] == {k:request[k] for k in PUBLIC} and
            'prepared' not in row, 'Frozen request/prepared boundary differs')
    record = dict(row, run_root=str(handle.registry.root), root=str(handle.registry.root),
                  protocol='structured_v4', protocol_revision='v2', assistant_prefill='{')
    runtime.bind_request(request)
    try:
        reference = runtime.batch_reference_forward([record], handle.fork.reference)[0].detach()
        hook = (handle.storage.GPUSavedTensors((*runtime.model.parameters(),*runtime.model.buffers()),
                                             row_id=row['request_id']) if two_device else None)
        context = hook.context() if hook is not None else nullcontext()
        with context:
            current = runtime.batch_sequence_forward([record], purpose='training_policy', grad=True)[0]
        sampler = current.new_tensor(row['sampler_logprobs'])
        require(current.shape == sampler.shape == reference.shape, 'Original vectors differ in shape')
        return current, sampler, reference, hook
    finally:
        runtime.unbind_request()


def _accounting(handle, index):
    events = copy.deepcopy(handle.runtime._satdose_accounting[index:])
    disk = [json.loads(line) for line in (handle.output/'accounting.jsonl').read_text().splitlines() if line.strip()]
    require(disk == handle.runtime._satdose_accounting, 'Disk/runtime forward accounting differs')
    totals = dict(training_policy=0,training_reference=0)
    for event in events:
        require(event['kind'] == 'extra_forward_sequences' and type(event['count']) is int and
                event['count'] > 0 and event['metadata']['purpose'] in totals,
                'Unexpected accounting/generation in backend comparison')
        totals[event['metadata']['purpose']] += event['count']
    require(totals == dict(training_policy=1,training_reference=1), 'Actual single-row native forwards differ')
    return dict(events=events, measured_forward_sequences=totals)


def _capture(handle, row, advantage, kernels, *, two_device, expected_tensors=496):
    import torch
    model = handle.runtime.model
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    require(len(parameters) == expected_tensors and parameters and
            (expected_tensors != 496 or all('lora_' in n for n in parameters)), 'Trainable inventory differs')
    require(all(p.grad is None for p in model.parameters()), 'Preexisting gradients forbidden')
    if expected_tensors == 496:
        require(all(str(v.device) == 'cuda:0' for v in (*model.parameters(),*model.buffers())),
                'All production model tensors must remain on cuda:0')
    require(all(not v.is_floating_point() or v.dtype == torch.float32
                for v in (*model.parameters(),*model.buffers())), 'FP32 model required')
    require(not any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules()), 'Dropout forbidden')
    before = {n:p.detach().cpu().clone() for n,p in parameters.items()}
    frozen = [(v,v._version) for v in (*[p for p in model.parameters() if not p.requires_grad],*model.buffers())]
    reference_sha = kernels.state_hash(handle.fork.reference)
    rng, training, row_before = kernels.capture_rng(), model.training, copy.deepcopy(row)
    accounting_start = len(handle.runtime._satdose_accounting)
    failure, mutation, result = None, [], None
    try:
        model.eval()
        peak_before = handle.storage.gpu_peak_stats(reset=True)
        if not two_device:
            require(peak_before['cuda:1']['allocated_bytes'] == 0,
                    'Baseline cuda:1 already has tensor allocations')
        current,sampler,reference,hook = _forward_original(handle,row,two_device=two_device)
        require(current.ndim == 1 and current.numel() == len(row['tokens']) and
                current.dtype == sampler.dtype == reference.dtype == torch.float32 and
                current.requires_grad and not sampler.requires_grad and not reference.requires_grad and
                all(bool(torch.isfinite(x).all()) for x in (current,sampler,reference)), 'FP32 native vectors required')
        require(sampler.detach().cpu().tolist() == row['sampler_logprobs'], 'Original sampler changed')
        terms = kernels.sequence_loss(current,sampler,reference,advantage,0.)
        loss = terms['loss'] / 128
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
                'Missing/nonfinite preclip gradients')
        require(all(bool(torch.isfinite(torch.as_tensor(value)).all()) for value in terms.values()), 'Nonfinite native objective')
        if hook is not None: handle.storage.validate_snapshot_stats(hook.stats)
        peak_after = handle.storage.gpu_peak_stats(reset=False)
        if not two_device:
            require(peak_after['cuda:1']['peak_allocated_bytes'] == 0 and
                    peak_after['cuda:1']['allocated_bytes'] == 0, 'Baseline allocated tensors on cuda:1')
        result = dict(teacher=current.detach().cpu().clone(),reference=reference.cpu().clone(),
            sampler=sampler.cpu().clone(),loss=terms['loss'].detach().cpu().clone(),
            scaled_loss=loss.detach().cpu().clone(),terms={k:torch.as_tensor(v).detach().cpu().clone() for k,v in terms.items()},
            preclip_gradients={n:p.grad.detach().cpu().clone() for n,p in parameters.items()},
            storage=copy.deepcopy(hook.stats) if hook is not None else None,
            gpu_memory_before=peak_before,gpu_memory_after=peak_after,accounting=_accounting(handle,accounting_start),
            original_tokens=list(row['tokens']),original_sampler=list(row['sampler_logprobs']),
            advantage=advantage,sequence_denominator=128,backward_calls=1,optimizer_steps=0,
            visible_gpu_count=2,storage_path='two_device_saved_tensors' if two_device else 'single_device_activation_retention',
            status='BACKEND_PATH_CAPTURE_NOT_GPU_ADMISSION',sources=copy.deepcopy(kernels.sources))
    except BaseException as exc:
        failure = exc
    finally:
        try:
            if any(v._version != version for v,version in frozen): mutation.append('Frozen tensor changed; discard runtime')
            if kernels.state_hash(handle.fork.reference) != reference_sha: mutation.append('Reference changed; discard runtime')
            if row != row_before: mutation.append('Caller row changed')
            with torch.no_grad():
                for name,param in parameters.items():
                    if not torch.equal(param.detach().cpu(),before[name]):
                        mutation.append('Policy mutated: '+name)
                        param.copy_(before[name])
        finally:
            for param in model.parameters(): param.grad = None
            try: kernels.restore_rng(rng)
            finally: model.train(training)
    if mutation: raise PermissionError('; '.join(mutation)) from failure
    if failure is not None: raise failure
    require(kernels.state_hash(kernels.capture_rng()) == kernels.state_hash(rng), 'RNG restoration differs')
    return result


def compare(baseline, two_device):
    import torch
    differences=[]
    for field in ('original_tokens','original_sampler','advantage','sequence_denominator'):
        require(baseline[field] == two_device[field], 'Comparison changed original trajectory/objective: '+field)
    for field in ('teacher','reference','sampler','loss','scaled_loss'):
        if not torch.equal(baseline[field],two_device[field]): differences.append(field)
    for field in ('terms','preclip_gradients'):
        require(list(baseline[field]) == list(two_device[field]), 'Comparison tensor inventory differs')
        differences.extend(field+'.'+name for name in baseline[field]
                           if not torch.equal(baseline[field][name],two_device[field][name]))
    return dict(bit_exact=not differences,differences=differences)


def _execute(ctx, start):
    root,b = ctx['root'],ctx['binding']
    with worker._worker_lock(root):
        worker._no_holds(root,b,binding_sha=ctx['binding_sha'])
        _no_unknown(root,b)
        selected=ctx['prepared']['starts'][str(start)]
        kernels=production.load_native_kernels(ctx['lock']['native_src'],ctx['lock']['legacy_root'])
        selection=select_shortest(selected['probe']['rows'],kernels)
        destination = root / expected_slots()[str(start)]
        destination.parent.mkdir(exist_ok=True)
        require(destination.parent.resolve() == destination.parent, 'Noncanonical backend-equivalence root')
        destination.mkdir(exist_ok=False)
        shadow._sync_dir(destination.parent)
        shadow.write_once(destination/'CLAIM.json',dict(status='RESERVED_UNKNOWN_NO_RETRY',start_seed=start,
            technical_binding_sha256=ctx['binding_sha'],max_original_row_replays=2,max_optimizer_steps=0,
            request_id=selection['row']['request_id'],original_index=selection['original_index'],
            original_g8_request_ids=selection['group_request_ids'],advantage=selection['advantage'],
            sequence_denominator=128,selected_tokens_sha256=worker._sha_object(selection['row']['tokens'])))
        audit=destination/'audit';audit.mkdir()
        handle=None;relocation=[];close=None
        try:
            shadow.write_once(destination/'SELECTION.json',selection)
            handle=native_runtime.build_runtime(ctx['registry'],worker._bound(root,b['source_lock']),
                                               start_seed=start,output_dir=audit)
            endpoint=handle.endpoint()
            require(dict(endpoint,parent='b0') == selected['expected_identity'], 'Loaded R24 identity differs')
            require(handle.fork.optimizer is None and handle.fork.scheduler is None and
                    not handle.runtime._satdose_accounting, 'Constructor must be read-only without model forwards')
            rows,relocation=worker._relocate(root,audit,[selection['row']],endpoint,b['design_freeze']['sha256'])
            baseline=_capture(handle,rows[0],selection['advantage'],kernels,two_device=False)
            worker._save_tensor(destination/'BASELINE.pt',baseline)
            second=_capture(handle,rows[0],selection['advantage'],kernels,two_device=True)
            worker._save_tensor(destination/'TWO_DEVICE.pt',second)
            comparison=compare(baseline,second)
            shadow.write_once(destination/'COMPARISON.json',comparison)
            require(comparison['bit_exact'], 'Backend teacher/reference/loss/preclip gradients differ')
            require(handle.endpoint() == endpoint, 'Endpoint changed during backend comparison')
        except BaseException as exc:
            shadow.write_once(destination/'HOLD.json',dict(status='HOLD_BACKEND_EQUIVALENCE_NO_RETRY',
                error_type=type(exc).__name__,error=str(exc),
                observed_accounting=copy.deepcopy(handle.runtime._satdose_accounting) if handle else []))
            raise
        finally:
            try:
                if handle is not None: close=handle.close()
                for ref in ctx['protected']: worker._bound(root,ref)
                for entry in relocation:
                    require(shadow.file_sha(entry['source']) == shadow.file_sha(entry['destination']) == entry['sha256'],
                            'Original/relocated evidence changed')
            except BaseException as exc:
                if not (destination/'HOLD.json').exists():
                    shadow.write_once(destination/'HOLD.json',dict(status='HOLD_BACKEND_CLEANUP',error=str(exc)))
                raise
        complete=dict(schema='sr-satdose-backend-equivalence-v1',start_seed=start,
            technical_binding_sha256=ctx['binding_sha'],status='BACKEND_EQUIVALENCE_CAPTURED_NOT_GPU_ADMISSION',
            **comparison,close=close,permission='NO_TRAINING',optimizer_steps=0,new_generations=0,
            tested='single_device_activation_retention_vs_two_device_saved_tensors_with_two_gpus_visible',
            one_visible_gpu_startup_verified=False,long_sequence_stress='PENDING',
            evidence={name:worker._ref(root,destination/name) for name in
                      ('SELECTION.json','BASELINE.pt','TWO_DEVICE.pt','COMPARISON.json')})
        shadow.write_once(destination/'COMPLETE.json',complete)
        return complete


def run_equivalence(*, start_seed, expected_binding_sha256):
    global _USED
    require(not _USED, 'One backend-equivalence worker per fresh process')
    ctx=_preflight(start_seed,expected_binding_sha256)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2 and torch.cuda.current_device() == 0,
            'Two visible CUDA GPUs required for this technical comparison')
    _USED=True
    return _execute(ctx,start_seed)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed',type=int,choices=(11031,11032),required=True)
    parser.add_argument('--binding-sha256',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run_equivalence(start_seed=args.start_seed,expected_binding_sha256=args.binding_sha256),sort_keys=True))
    return 0


if __name__ == '__main__': raise SystemExit(main())
