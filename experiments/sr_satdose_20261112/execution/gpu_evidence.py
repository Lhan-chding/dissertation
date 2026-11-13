"""Read-only CPU audit of completed finite GPU evidence, never admission.

Run this independently SHA-pinned file OUTSIDE the immutable GPU release:
  python -B /audit/gpu_evidence.py --binding-sha256 SHA --auditor-sha256 SHA
The binding authenticates the full deployed release before any release import.
No model, optimizer, generation, CUDA operation, output file or lock file is
created. JSON is returned/printed; the caller owns any subsequent publication.
Evaluation GPU tensors were not persisted: their metadata/source receipts are
audited, not represented as a second offline observation of those GPU tensors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
STARTS = (11031, 11032)
LONG_ROW = 'SAT1/DRY/11032/g004/d07'


def require(ok, message):
    if not ok:
        raise ValueError('HOLD_GPU_EVIDENCE: ' + message)


def _sha(path):
    path = Path(path)
    require(path.is_file() and path.resolve(strict=True) == path and not path.is_symlink(),
            'Noncanonical ordinary evidence: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _read(path):
    def pairs(items):
        require(len(dict(items)) == len(items), 'Duplicate JSON key')
        return dict(items)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'Nonfinite JSON'))


class Evidence:
    """Hash actual bytes before reading; recheck all observed bytes at the end."""
    def __init__(self, root):
        self.root = Path(root)
        self.observed = {}

    def path(self, relative, expected=None):
        relative = Path(relative)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Escaped evidence path')
        path = self.root / relative
        digest = _sha(path)
        require(expected is None or digest == expected, 'Artifact SHA differs: ' + str(relative))
        require(path not in self.observed or self.observed[path] == digest, 'Evidence changed during audit')
        self.observed[path] = digest
        return path

    def ref(self, ref):
        require(set(ref) == {'path', 'sha256'}, 'Malformed artifact reference')
        return self.path(ref['path'], ref['sha256'])

    def json(self, relative, expected=None):
        return _read(self.path(relative, expected))

    def tensor(self, relative, expected=None):
        import torch
        return torch.load(self.path(relative, expected), map_location='cpu', weights_only=True)

    def stable(self):
        require(all(_sha(p) == digest for p, digest in self.observed.items()), 'Evidence changed during audit')


@contextmanager
def _existing_shared_lock(root):
    """Never create a lock or join an active producer's exclusive section."""
    path = root / 'DIAGNOSTIC_WORKER.lock'
    require(path.is_file() and path.resolve(strict=True) == path, 'Missing/noncanonical existing worker lock')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('HOLD_GPU_EVIDENCE: producer still holds lock') from exc
        yield
    finally:
        os.close(fd)


def _release(evidence, binding, auditor_sha256):
    require(re.fullmatch('[a-f0-9]{64}', auditor_sha256 or '') and
            _sha(Path(__file__).resolve()) == auditor_sha256, 'Auditor source SHA differs')
    receipt_path = evidence.ref(binding['gpu_release'])
    receipt = _read(receipt_path)
    release = Path(receipt['release_root'])
    require(release.is_absolute() and release.resolve(strict=True) == release and
            not receipt_path.is_relative_to(release), 'Invalid independent release receipt')
    require(receipt['schema'] == 'sr-satdose-gpu-technical-release-v1' and
            receipt['status'] == 'AUDITED_FOR_GPU_CHECKS_ONLY' and receipt['permission'] == 'NO_TRAINING' and
            receipt['experiment_id'] == binding['experiment_id'] and receipt['evidence_root'] == str(evidence.root) and
            receipt['design_freeze_sha256'] == binding['design_freeze']['sha256'] and
            receipt['config_sha256'] == binding['config']['sha256'], 'Release receipt identity differs')
    actual = {}
    for directory, dirs, files in os.walk(release, followlinks=False):
        for name in dirs:
            p = Path(directory) / name
            require(p.resolve(strict=True) == p and not p.is_symlink(), 'Symlink in release')
        for name in files:
            p = Path(directory) / name
            actual[str(p.relative_to(release))] = _sha(p)
    require(actual == receipt['sources'], 'Full release inventory differs')
    for ref in binding['code'].values():
        path = Path(ref['path'])
        require(path.is_relative_to(release) and actual.get(str(path.relative_to(release))) == ref['sha256'],
                'Code pin outside authenticated release')
    return release, actual


def _gradients(values, policy):
    import torch
    require(len(policy) == 496 and list(values) == list(policy), 'Incomplete/reordered 496 tensor inventory')
    require(all('lora_' in n and torch.is_tensor(v) and v.device.type == 'cpu' and
                v.dtype == torch.float32 and v.shape == policy[n].shape and bool(torch.isfinite(v).all())
                for n, v in values.items()), 'Invalid actual FP32 tensor/shape/value')


def _same(left, right):
    import torch
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and
                left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and list(left) == list(right) and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _traces(traces, rows, *, production=False):
    require(len(traces) == len(rows) and len({r['request_id'] for r in rows}) == len(rows), 'Trace coverage differs')
    for trace, row in zip(traces, rows, strict=True):
        require(trace['row_id' if production else 'request_id'] == row['request_id'], 'Original trace order differs')
        n = len(row['tokens'])
        require(1 <= n <= 1536, 'Original token length invalid')
        require(trace['tokens'] == row['tokens'] if production else trace['token_count'] == n,
                'Original trace tokens/count differ')
        for field in ('sampler_logprobs', 'train_logprobs', 'reference_logprobs'):
            vector = trace[field]
            require(isinstance(vector, list) and len(vector) == n and
                    all(type(v) in (int, float) and math.isfinite(v) for v in vector), 'Nonfinite/misaligned vector')
        require(trace['sampler_logprobs'] == row['sampler_logprobs'], 'Original sampler vector differs')


def _accounting(account, rows, validate_stats):
    _events(account, len(rows))
    entries = account['storage']
    require(len(entries) == len(rows), 'Saved-tensor coverage differs')
    for entry, row in zip(entries, rows, strict=True):
        require(entry['row_id'] == entry['stats']['row_id'] == row['request_id'] and
                entry['forward_returned'] is True, 'Saved-tensor original row mismatch/incomplete')
        validate_stats(entry['stats'])


def _events(account, count):
    totals = dict(training_policy=0, training_reference=0)
    for event in account['events']:
        require(event['kind'] == 'extra_forward_sequences' and type(event['count']) is int and event['count'] > 0,
                'Unexpected actual accounting event')
        purpose = event['metadata']['purpose']
        require(purpose in totals, 'Generation/unregistered forward in accounting')
        totals[purpose] += event['count']
    require(totals == account['measured_forward_sequences'] ==
            dict(training_policy=count, training_reference=count), 'Actual forward counters differ')


def _production(capture, rows, policy, kernels):
    require(len(rows) == 128, 'Production full128 required')
    for key in ('row_forward_callback_calls', 'backward_calls', 'sequence_denominator'):
        require(type(capture[key]) is int and capture[key] == 128, 'Production counter differs: ' + key)
    require(capture['optimizer_created'] is False and all(type(capture[k]) is int and capture[k] == 0
            for k in ('optimizer_steps', 'scheduler_steps', 'parameter_updates')), 'Production updates forbidden')
    _traces(capture['traces'], rows, production=True)
    for key in ('preclip_gradients', 'postclip_gradients'):
        _gradients(capture[key], policy)
    recomputed = kernels.numerical_audit(capture['traces'])
    require(all(capture['numerics'].get(k) == v for k, v in recomputed.items()), 'Actual trace numerical audit differs')
    kernels.check_numerics(capture['numerics'])
    pre_norm = math.sqrt(sum(float(v.double().square().sum()) for v in capture['preclip_gradients'].values()))
    # Apply the original bound to the independent actual tensor norm as well.
    # If GPU FP32 and CPU FP64 reductions straddle 1000, HOLD for review; do
    # not silently widen the bound or require their scalar values to be equal.
    require(pre_norm <= 1000., 'Actual preclip norm exceeds original 1000 sentinel (CPU FP64 reduction)')
    post_norm = math.sqrt(sum(float(v.double().square().sum()) for v in capture['postclip_gradients'].values()))
    # GPU and CPU FP64 reductions need not be bit-identical. Both obey the
    # original sentinel; actual saved clip/replay tensors are compared exactly.
    require(post_norm <= 1.00001 and 0 <= capture['numerics']['gradient_norm_after_clip'] <= 1.00001,
            'Actual postclip norm exceeds original sentinel')
    require(set(capture['totals']) == {'loss','native','aux','kl','kl_unweighted'} and
            all(type(v) in (int,float) and math.isfinite(v) for v in capture['totals'].values()) and
            capture['totals']['aux'] == 0 and
            capture['totals']['kl_unweighted'] == capture['numerics']['kl_unweighted'], 'Invalid production scalar totals')


def _shadow_result(result, policy, produced, ops):
    import torch
    for key in ('delta', 'clipped'):
        _gradients(result[key], policy)
    require(_same(result['clipped'], produced['postclip_gradients']), 'Shadow/production actual clipping differs')
    optimizer = result['optimizer']
    ops.validate_optimizer_state(optimizer, result['scheduler'])
    require(len(optimizer['param_groups']) == 1 and
            all(group.get('foreach') is False and group.get('capturable') is False and group.get('fused') is None and
                all(group.get(key) is False for key in ('amsgrad','maximize','differentiable'))
                for group in optimizer['param_groups']), 'Shadow differs from native fresh AdamW execution modes')
    ids = [p for group in optimizer['param_groups'] for p in group['params']]
    require(len(ids) == len(set(ids)) == 496 and set(ids) == set(optimizer['state']), 'Adam actual parameter inventory differs')
    for ident, name in zip(ids, policy, strict=True):
        state = optimizer['state'][ident]
        require(set(state) == {'step','exp_avg','exp_avg_sq'} and torch.is_tensor(state['step']) and
                state['step'].numel() == 1 and float(state['step']) == 1, 'Adam actual step/state differs')
        for key in ('exp_avg','exp_avg_sq'):
            value = state[key]
            require(value.shape == policy[name].shape and value.dtype == torch.float32 and
                    bool(torch.isfinite(value).all()), 'Adam moment tensor differs')
        require(bool((state['exp_avg_sq'] >= 0).all()), 'Negative Adam second moment')
    require(result['scheduler']['last_epoch'] == 0, 'Shadow scheduler advanced')


def _relocation(e, audit, rows):
    """Authenticate every original CAPTURE/BEGIN plus its byte-identical copy."""
    manifest = e.json(audit / 'RELOCATION.json')['files']
    expected = {}
    relocated = []
    for row in rows:
        source = Path(row['native_capture_path'])
        require(source.is_relative_to(e.root), 'Original capture outside root')
        require(source.name == hashlib.sha256(row['request_id'].encode()).hexdigest() + '.CAPTURE.json',
                'Original capture filename differs')
        saved = e.json(source.relative_to(e.root), row['native_capture_sha256'])
        for key in ('tokens','sampler_logprobs','prompt_tensor_hash','model_input','seed'):
            require(saved[key] == row[key], 'Original CAPTURE trajectory differs')
        for p in (source, source.with_name(source.name.replace('.CAPTURE.json','.BEGIN.json'))):
            digest = _sha(e.path(p.relative_to(e.root)))
            dest = e.path(audit / p.name, digest)
            expected[str(p)] = dict(source=str(p), destination=str(dest), sha256=digest)
        begin = e.json(source.with_name(source.name.replace('.CAPTURE.json','.BEGIN.json')).relative_to(e.root))
        require(begin['request_id'] == row['request_id'] and begin['seed'] == row['seed'] and
                begin['design_freeze_sha256'] == row['design_freeze_sha256'] and
                begin['endpoint']['start_seed'] == row['start_seed'] and
                begin['endpoint']['policy_sha256'] == row['policy_sha256'] and
                begin['endpoint']['reference_sha256'] == row['reference_sha256'] and
                begin['endpoint']['arm'] == 'START' and begin['endpoint']['step'] == 0,
                'Original BEGIN endpoint differs')
        relocated.append(dict(row, native_capture_path=str(e.root / audit / source.name)))
    require(len(manifest) == len(expected) and {v['source']:v for v in manifest} == expected,
            'Original relocation inventory differs')
    return relocated


def _complete_chain(e, binding, binding_sha):
    """Require ALL ten completions; absent unattempted slots are not success."""
    for prefix in ('evaluation_loader_checks','backend_equivalence','allzero_check','shadow_inputs','shadow_claims'):
        parent = e.root/prefix
        require(parent.is_dir() and parent.resolve(strict=True) == parent and
                {p.name for p in parent.iterdir()} == {'start-11031','start-11032'},
                'Missing/unknown finite stage directory: ' + prefix)
        if prefix in ('shadow_inputs','shadow_claims'):
            for start in STARTS:
                directory = parent/f'start-{start}'
                require(directory.resolve(strict=True) == directory and
                        {p.name for p in directory.iterdir()} == {'trial-0','trial-1'}, 'Unknown shadow slot/budget')
    completions = {}
    for start in STARTS:
        for field in ('evaluation_loader_slots','backend_equivalence_slots','allzero_slots'):
            directory = Path(binding[field][str(start)])
            require(not (e.root / directory / 'HOLD.json').exists(), 'Existing phase HOLD')
            marker = e.json(directory / 'COMPLETE.json')
            require(marker['technical_binding_sha256'] == binding_sha and marker['start_seed'] == start,
                    'Foreign phase completion')
            for ref in marker['evidence'].values():
                e.ref(ref)
            completions[str(directory)] = marker
        for trial in (0, 1):
            slot = binding['shadow_slots'][f'{start}/{trial}']
            directory = Path(slot['input_receipt']).parent
            claim = Path(f'shadow_claims/start-{start}/trial-{trial}')
            require(not any((e.root / d / 'HOLD.json').exists() for d in (directory, claim)), 'Existing worker/shadow HOLD')
            w = e.json(directory / 'WORKER_COMPLETE.json')
            s = e.json(claim / 'COMPLETE.json', w['shadow_complete_sha256'])
            require(w['technical_binding_sha256'] == s['technical_binding_sha256'] == binding_sha and
                    w['start_seed'] == s['start_seed'] == start and w['trial'] == s['trial'] == trial and
                    s['status'] == 'SHADOW_TRIAL_COMPLETE_NOT_GPU_ADMISSION' and s['mode'] == 'production' and
                    all(type(s[k]) is int and s[k] == 1 for k in ('optimizer_body_calls','optimizer_steps')) and
                    all(type(s[k]) is int and s[k] == 0 for k in ('formal_optimizer_steps','scheduler_steps')),
                    'Shadow completion identity/counters differ')
            e.path(slot['restore'], w['restore_sha256'])
            e.path(slot['input_receipt'], s['input_receipt_sha256'])
            e.path(claim / 'RESULT.pt', s['result_sha256'])
            e.path(directory / 'ACCOUNTING.json', w['actual_accounting_sha256'])
            for ref in w['component_files'].values():
                e.ref(ref)
            completions[str(directory)] = dict(worker=w, shadow=s)
    return completions


def audit_evidence(expected_binding_sha256, *, auditor_sha256):
    """Production fixed-root audit; any missing/failing evidence raises HOLD."""
    e = Evidence(ROOT)
    binding = e.json('TECHNICAL_EXECUTION_BINDING.json', expected_binding_sha256)
    release, inventory = _release(e, binding, auditor_sha256)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(release))
    import torch
    previous_threads = torch.get_num_threads()
    require(torch.get_default_dtype() == torch.float32, 'Native numerical audit requires default FP32')
    torch.set_num_threads(1)  # Match the pinned producer OMP_NUM_THREADS=1 CPU numerical audit.
    try:
        modules = {name:importlib.import_module('execution.' + name) for name in
                   ('diagnostic_worker','production','shadow_replay','allzero_check','backend_equivalence','evaluation_probe')}
        for name, module in modules.items():
            require(Path(module.__file__).resolve() == Path(binding['code'][name]['path']), 'Shadowed audited import: ' + name)
        worker, production, shadow = (modules[n] for n in ('diagnostic_worker','production','shadow_replay'))
        with _existing_shared_lock(ROOT):
            ctx = worker._preflight(expected_binding_sha256, 11031, 0)  # CPU only: full DRY rescoring/source checks.
            modules['evaluation_probe']._binding_fields(binding)
            modules['evaluation_probe']._prefix_required(ctx)
            completions = _complete_chain(e, binding, expected_binding_sha256)
            worker.native_runtime.verify_source_lock(ctx['registry'], e.ref(binding['source_lock']))
            kernels = production.load_native_kernels(ctx['lock']['native_src'], ctx['lock']['legacy_root'])
            _, _, ops = shadow._binding(ROOT, expected_binding_sha256, 'production')
            storage_path = Path(ctx['lock']['legacy_root'])/'diagnostics/gpu_capacity_v2.py'
            spec = importlib.util.spec_from_file_location('_satdose_independent_storage_audit', storage_path)
            storage = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = storage
            spec.loader.exec_module(storage)
            # Native/source files are verified by frozen preflight/loaders, not an unbound callback.
            reports = {}
            for start in STARTS:
                reports[str(start)] = _audit_start(e, ctx, start, completions, modules, kernels, ops, storage)
            worker._no_holds(ROOT, binding, binding_sha=expected_binding_sha256)
            worker.native_runtime.verify_source_lock(ctx['registry'], e.ref(binding['source_lock']))
            _release(e, binding, auditor_sha256)
            e.stable()
            return dict(schema='sr-satdose-independent-gpu-evidence-audit-v1',
                status='FINITE_GPU_EVIDENCE_AUDITED_NOT_ADMISSION', permission='NO_TRAINING',
                technical_binding_sha256=expected_binding_sha256, auditor_sha256=auditor_sha256,
                completed_fixed_stages=10, shadow_optimizer_steps=4, formal_optimizer_steps=0,
                audit_model_calls=0, audit_optimizer_calls=0, starts=reports,
                evidence={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha
                          for p,sha in e.observed.items()},
                limitations=['Not the final fifteen-check GPU verification or training admission.',
                    'Evaluation START GPU tensors were not saved: receipt metadata is cross-checked against R24 CPU tensors.',
                    'No trained-checkpoint GPU evaluation loading is certified.',
                    'No production GPU memory peak was recorded; none is inferred.',
                    'Shadow Adam tensors and replay are audited; this auditor never executes Adam.'])
    finally:
        torch.set_num_threads(previous_threads)
        sys.path.remove(str(release))


def _audit_start(e, ctx, start, completions, modules, kernels, ops, storage):
    worker, shadow = modules['diagnostic_worker'], modules['shadow_replay']
    binding = ctx['binding']; selected = ctx['prepared']['starts'][str(start)]
    rows = selected['probe']['rows']; previous = None; direction = None
    source = ctx['config']['start_bindings'][str(start)]
    source_path = Path(source['original_state_path'])
    require(_sha(source_path) == source['state_sha256'], 'Original R24 source changed')
    e.observed[source_path] = source['state_sha256']
    for field in ('backend_equivalence_slots','allzero_slots'):
        closed = completions[binding[field][str(start)]]['close']
        require(closed['closed'] is True and closed['start_seed'] == start and closed['arm'] == 'START' and closed['step'] == 0 and
                closed['policy_sha256'] == source['policy_sha256'] and closed['reference_sha256'] == source['reference_sha256'] and
                closed['optimizer_steps'] == closed['parameter_updates'] == 0, 'Read-only phase close differs from R24')
    for trial in (0, 1):
        slot = binding['shadow_slots'][f'{start}/{trial}']; directory = Path(slot['input_receipt']).parent
        close = completions[str(directory)]['worker']; sc = completions[str(directory)]['shadow']
        endpoint = {k:v for k,v in selected['expected_identity'].items() if k != 'parent'}
        receipt, restored, capture, policy = shadow._load_inputs(e.root, binding, ctx['binding_sha'], start, trial,
            sc['input_receipt_sha256'], endpoint, ops, ctx['config'])
        e.ref(receipt['capture']); e.ref(receipt['restore'])
        produced = capture['production']
        _production(produced, rows, policy, kernels)
        require(capture['request_ids_sha256'] == _object_sha([r['request_id'] for r in rows]) and
                capture['token_manifest_sha256'] == _object_sha([{k:r[k] for k in
                    ('request_id','tokens','sampler_logprobs')} for r in rows]), 'Production original token manifest differs')
        _, groups = modules['production'].validate_batch(rows, selected['production_plan'], selected['expected_identity'],
            ctx['config'], produced['policy_sha256'], produced['reference_sha256'], kernels)
        require(produced['groups'] == groups, 'Actual native G8 advantages differ')
        claim = Path(f'shadow_claims/start-{start}/trial-{trial}')
        result = e.tensor(claim/'RESULT.pt', sc['result_sha256'])
        _shadow_result(result, policy, produced, ops)
        require(sc['loader_instance_id'] == restored['loader_instance_id'] and
                close['process_identity'] == restored['process_identity'] and close['close']['closed'] is True and
                close['close']['policy_sha256'] == produced['policy_sha256'] and
                close['close']['reference_sha256'] == produced['reference_sha256'] and
                close['close']['optimizer_steps'] == close['close']['parameter_updates'] == 0,
                'Worker restore/close identity differs')
        process = restored['process_identity']
        require(set(process) == {'host','pid','start_ticks','boot_id'} and type(process['pid']) is int and
                process['pid'] > 0 and all(isinstance(process[k], str) and process[k] for k in ('host','start_ticks','boot_id')),
                'Missing actual process identity')
        wc = e.json(directory/'WORKER_CLAIM.json')
        require(wc['start_seed'] == start and wc['trial'] == trial and wc['process_identity'] == process and
                wc['technical_binding_sha256'] == ctx['binding_sha'], 'Worker claim chain differs')
        shadow_claim = e.json(claim/'CLAIM.json')
        require(shadow_claim['schema'] == 'sr-satdose-shadow-claim-v1' and shadow_claim['mode'] == 'production' and
                shadow_claim['start_seed'] == start and shadow_claim['trial'] == trial and
                shadow_claim['technical_binding_sha256'] == ctx['binding_sha'] and
                shadow_claim['input_receipt_sha256'] == sc['input_receipt_sha256'] and
                shadow_claim['design_freeze_sha256'] == binding['design_freeze']['sha256'] and
                type(shadow_claim['max_optimizer_body_calls']) is int and shadow_claim['max_optimizer_body_calls'] == 1 and
                sc['caller_policy_unchanged'] is True and sc['production_clip_bit_exact'] is True and
                sc['restore_replay_compared'] is (trial == 1), 'Shadow exact-once claim/restore chain differs')
        accounts = e.json(directory/'ACCOUNTING.json', close['actual_accounting_sha256'])
        tasks = {'production':rows}
        captures = {}
        source_rows = rows + (selected['witness']['rows'] if trial == 0 else [])
        _relocation(e, directory/'audit', source_rows)
        if trial == 0:
            halves = worker.diagnostics.validate_partition(rows, selected['probe']['half_group_ids'])
            whalves = worker.diagnostics.validate_partition(selected['witness']['rows'], selected['witness']['half_group_ids'])
            tasks.update({x:rows for x in ('total','J','source','KL')})
            tasks.update({f'{x}_half_{i}':halves[i-1] for x in ('J','source') for i in (1,2)})
            tasks.update({f'witness_half_{i}{repeat}':whalves[i-1] for repeat in ('','_repeat') for i in (1,2)})
            require(set(close['component_files']) == set(tasks)-{'production'}, 'Incomplete component file inventory')
            for name in set(tasks)-{'production'}:
                ref = close['component_files'][name]
                require(ref['path'] == str(directory/(name+'.pt')), 'Relocated component file')
                component = captures[name] = e.tensor(ref['path'], ref['sha256'])
                _gradients(component['gradients'], policy); _traces(component['traces'], tasks[name])
                require(component['forward_calls'] == component['backward_calls'] == component['sequence_denominator'] == len(tasks[name])
                        and component['optimizer_steps'] == 0, 'Component actual counters differ')
            direction = worker.diagnostics.direction_summary(captures, result['delta'], produced['preclip_gradients'])
            require(direction['direction_checks_satisfied'] is True and not direction['hold_reasons'] and
                    direction == e.json(directory/'DIRECTION.json') == close['direction'], 'Actual direction/repeat-floor gate differs')
        else:
            require(not close['component_files'] and restored['loader_instance_id'] != previous['restore']['loader_instance_id'] and
                    restored['process_identity'] != previous['restore']['process_identity'], 'Replay did not use fresh process/instance')
            require(_same(result, previous['result']) and _same(produced, previous['production']), 'Actual shadow/production replay differs')
        require(set(accounts) == set(tasks), 'Unexpected/missing accounting stage')
        all_events = []
        for name, batch in tasks.items():
            require(e.json(directory/(name+'_ACCOUNTING.json')) == accounts[name], 'Duplicated stage accounting differs')
            _accounting(accounts[name], batch, storage.validate_snapshot_stats)
        # Native on-disk append order follows the fixed producer task order.
        for name in tasks:
            all_events.extend(accounts[name]['events'])
        disk_events = [_read_line(line) for line in e.path(directory/'audit/accounting.jsonl').read_text().splitlines() if line.strip()]
        require(disk_events == all_events, 'Actual native accounting log differs')
        previous = dict(result=result, production=produced, restore=restored)
    # Shared gate already verified all evaluation receipts and original file hashes.
    evaluation_dir = Path(binding['evaluation_loader_slots'][str(start)])
    tensor_inventory = e.json(evaluation_dir/'TENSOR_INVENTORY.json')['tensors']
    require([(r['name'],r['shape']) for r in tensor_inventory] == [(n,list(v.shape)) for n,v in policy.items()],
            'Evaluation metadata differs from actual original R24 tensor names/shapes')
    _audit_backend(e, ctx, start, rows, policy, modules, kernels, storage)
    _audit_allzero(e, ctx, start, rows, policy, modules, storage)
    lengths = {r['request_id']:len(r['tokens']) for r in rows}
    require(max(lengths.values()) == (1536 if start == 11032 else 709), 'Fixed observed probe length scope differs')
    if start == 11032:
        require([rid for rid,n in lengths.items() if n == 1536] == [LONG_ROW], 'Original long-row scope differs')
    return dict(production_rows=128, max_original_tokens=max(lengths.values()),
                long_original_row=LONG_ROW if start == 11032 else None, shadow_optimizer_steps=2,
                direction=direction, evaluation_scope='START_R24_RECEIPTS_AND_ORIGINAL_CPU_TENSOR_INVENTORY',
                memory_peak='NOT_RECORDED_FOR_PRODUCTION', permission='NO_TRAINING')


def _read_line(line):
    return json.loads(line, parse_constant=lambda _: require(False, 'Nonfinite accounting'))


def _audit_backend(e, ctx, start, rows, policy, modules, kernels, storage):
    import torch
    module = modules['backend_equivalence']; directory = Path(ctx['binding']['backend_equivalence_slots'][str(start)])
    selection = e.json(directory/'SELECTION.json')
    require(selection == module.select_shortest(rows, kernels), 'Backend selection differs from fixed shortest original row')
    row = selection['row']; _relocation(e, directory/'audit', [row])
    claim = e.json(directory/'CLAIM.json')
    require(claim['start_seed'] == start and claim['technical_binding_sha256'] == ctx['binding_sha'] and
            claim['max_original_row_replays'] == 2 and claim['max_optimizer_steps'] == 0 and
            claim['request_id'] == row['request_id'] and claim['original_index'] == selection['original_index'] and
            claim['original_g8_request_ids'] == selection['group_request_ids'] and claim['advantage'] == selection['advantage'] and
            claim['sequence_denominator'] == 128 and claim['selected_tokens_sha256'] == _object_sha(row['tokens']),
            'Backend claim/selection differs')
    captures = [e.tensor(directory/name) for name in ('BASELINE.pt','TWO_DEVICE.pt')]
    for index,c in enumerate(captures):
        _backend_identity(c, selection, kernels.sources, two_device=index == 1)
        _gradients(c['preclip_gradients'], policy)
        require(c['original_tokens'] == row['tokens'] and c['original_sampler'] == row['sampler_logprobs'] and
                c['sequence_denominator'] == 128 and c['backward_calls'] == 1 and c['optimizer_steps'] == 0,
                'Backend actual trajectory/counters differ')
        for name in ('teacher','reference','sampler'):
            value = c[name]
            require(value.dtype == torch.float32 and value.shape == (len(row['tokens']),) and
                    bool(torch.isfinite(value).all()), 'Backend invalid original logprob vector')
        require(c['sampler'].tolist() == row['sampler_logprobs'], 'Backend sampler tensor differs')
        require(all(bool(torch.isfinite(v).all()) for v in c['terms'].values()), 'Backend nonfinite native terms')
        _events(c['accounting'], 1)
    require(captures[0]['gpu_memory_after']['cuda:1']['peak_allocated_bytes'] == 0, 'Baseline used activation GPU')
    storage.validate_snapshot_stats(captures[1]['storage'])
    comparison = module.compare(*captures)
    require(comparison['bit_exact'] is True and comparison == e.json(directory/'COMPARISON.json'), 'Actual backend tensor comparison differs')
    disk = [_read_line(line) for line in e.path(directory/'audit/accounting.jsonl').read_text().splitlines() if line.strip()]
    require(disk == captures[0]['accounting']['events'] + captures[1]['accounting']['events'], 'Backend native accounting log differs')


def _backend_identity(capture, selection, sources, *, two_device):
    """Bind each real capture to selection, not merely to the other capture."""
    expected_path = 'two_device_saved_tensors' if two_device else 'single_device_activation_retention'
    require(capture['advantage'] == selection['advantage'] and
            capture['status'] == 'BACKEND_PATH_CAPTURE_NOT_GPU_ADMISSION' and
            capture['sources'] == sources and capture['visible_gpu_count'] == 2 and
            capture['storage_path'] == expected_path and
            (isinstance(capture['storage'],dict) if two_device else capture['storage'] is None),
            'Backend capture advantage/source/storage identity differs')


def _audit_allzero(e, ctx, start, rows, policy, modules, storage):
    module = modules['allzero_check']; directory = Path(ctx['binding']['allzero_slots'][str(start)])
    original = e.json(directory/'ORIGINAL_INPUTS.json')
    require(original['rows'] == rows and original['prepared_diagnostic_inputs'] == ctx['binding']['diagnostic_inputs'] and
            original['actual_rewards_preserved'] is True and original['explicit_diagnostic_advantage'] == 0.,
            'Allzero rewrote original rewards/inputs')
    relocated = _relocation(e, directory/'audit', rows)
    claim = e.json(directory/'CLAIM.json')
    require(claim['start_seed'] == start and claim['technical_binding_sha256'] == ctx['binding_sha'] and
            claim['explicit_diagnostic_advantage'] == 0. and claim['max_optimizer_steps'] == claim['max_new_generations'] == 0 and
            claim['max_policy_forward_sequences'] == claim['max_reference_forward_sequences'] == 384,
            'Allzero allocation claim differs')
    accounts = e.json(directory/'ACCOUNTING.json'); captures = {}
    require(set(accounts) == set(module.COMPONENTS), 'Allzero accounting component coverage differs')
    for name in module.COMPONENTS:
        c = captures[name] = e.tensor(directory/(name+'.pt'))
        _gradients(c['gradients'], policy); _traces(c['traces'], rows)
        require(c['original_rows_sha256'] == _object_sha(relocated) and
                c['forward_calls'] == c['backward_calls'] == c['sequence_denominator'] == 128 and
                all(type(c[k]) is int and c[k] == 0 for k in ('optimizer_steps','scheduler_steps','new_generations','parameter_updates')),
                'Allzero actual original input/counters differ')
        source = ctx['config']['start_bindings'][str(start)]
        require(c['policy_before_sha256'] == source['policy_sha256'] and
                c['original_reference_sha256'] == source['reference_sha256'] and
                all(type(v) in (int,float) and math.isfinite(v) for term in c['native_terms'] for v in term.values()),
                'Allzero actual baseline/native scalar terms differ')
        _accounting(accounts[name], rows, storage.validate_snapshot_stats)
    comparison = module.compare(captures)
    require(comparison == e.json(directory/'COMPARISON.json'), 'Actual allzero tensor/term comparison differs')
    disk = [_read_line(line) for line in e.path(directory/'audit/accounting.jsonl').read_text().splitlines() if line.strip()]
    require(disk == [event for name in module.COMPONENTS for event in accounts[name]['events']],
            'Allzero native accounting log differs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--auditor-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_evidence(args.binding_sha256, auditor_sha256=args.auditor_sha256)
    except Exception as exc:
        print(json.dumps(dict(status='HOLD_GPU_EVIDENCE_AUDIT', permission='NO_TRAINING',
                              error_type=type(exc).__name__, error=str(exc)), sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
