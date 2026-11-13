"""Single-process original-token diagnostic worker; never GPU admission.

Run each start/trial in a NEW Python process. Technical binding precedes GPU
work and binds this module, the prepared diagnostic inputs and both complete
DRY receipts. Only this worker constructs its handle; no caller handle, model,
forward, generation, optimizer or output-directory argument is accepted.

The frozen runtime constructor resolves an existing output_root, loads b0 and
R24 and installs accounting. Its native output_root can also own activation
artifacts. Therefore a new fixed slot/audit is mandatory; original DRY output
is never writable here. Original CAPTURE/BEGIN bytes are relocated into audit
with source/destination SHA receipts, without changing any original row/file.

Remaining admission checks (short-sequence 1/2 GPU equivalence, 1536-token
stress and combined independent review) are explicitly PENDING. Synthetic CPU
tests validate orchestration only; they cannot establish any GPU property.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import uuid

from . import diagnostic_inputs, production, shadow_replay as shadow
from runtime import diagnostics, native_runtime
from runtime.backend import FrozenRegistry

require = shadow.require
_USED = False
PENDING = ['short_sequence_one_vs_two_gpu_equivalence', '1536_token_stress',
           'combined_independent_gpu_admission_review']


def _sha_object(value):
    return hashlib.sha256(shadow.canonical(value)).hexdigest()


def _process_identity():
    # Linux kernel process start ticks + boot ID avoid PID reuse ambiguity.
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(host=socket.gethostname(), pid=os.getpid(), start_ticks=stat[19],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def _ref(root, path):
    return dict(path=str(path.relative_to(root)), sha256=shadow.file_sha(path))


def _bound(root, ref):
    path = shadow._local_file(root, ref['path'])
    require(shadow.file_sha(path) == ref['sha256'], 'Bound diagnostic evidence changed')
    return path


def _no_holds(root, binding, *, current=None):
    """Reject HOLD and crash-left UNKNOWN slots; only the active internal slot is exempt."""
    for start in (11031, 11032):
        for trial in (0, 1):
            slot = binding['shadow_slots'][f'{start}/{trial}']
            worker = (root / slot['input_receipt']).parent
            claim = root / shadow.CLAIMS_NAME / f'start-{start}' / f'trial-{trial}'
            require(not (worker / 'HOLD.json').exists() and not (claim / 'HOLD.json').exists(),
                    'Existing HOLD blocks dependent diagnostics')
            if (start, trial) == current:
                continue
            if worker.exists() or claim.exists():
                require((worker / 'WORKER_COMPLETE.json').is_file() and (claim / 'COMPLETE.json').is_file(),
                        'Incomplete occupied slot is UNKNOWN; read-only audit required')
                complete = shadow.read_json(worker / 'WORKER_COMPLETE.json')
                require(complete['status'] == 'DIAGNOSTIC_SLOT_COMPLETE_NOT_GPU_ADMISSION' and
                        complete['shadow_complete_sha256'] == shadow.file_sha(claim / 'COMPLETE.json'),
                        'Occupied slot completion chain differs')


@contextmanager
def _worker_lock(root):
    """One fixed nonblocking process lock across all four slots; never an admission receipt."""
    path = root / 'DIAGNOSTIC_WORKER.lock'
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PermissionError('Another diagnostic worker holds the global lock') from exc
        yield
    finally:
        os.close(descriptor)


def _verify_runtime_sources(freeze):
    runtime = Path(native_runtime.__file__).resolve().parent
    actual = {p.name: shadow.file_sha(p) for p in runtime.glob('*.py') if p.is_file()}
    require(actual == freeze.get('runtime_sources'), 'Actual runtime source inventory/SHA differs')
    require(set(actual.values()) <= {ref['sha256'] for ref in freeze['artifacts'].values()},
            'Actual runtime sources absent from original freeze artifacts')
    for name, module in tuple(sys.modules.items()):
        if name.startswith('runtime.'):
            path = getattr(module, '__file__', None)
            expected = runtime / (name.removeprefix('runtime.').replace('.', '/') + '.py')
            require(path is not None and Path(path).resolve() == expected and expected.name in actual,
                    'Shadowed imported runtime module: ' + name)
    return actual


def _preflight(binding_sha, start, trial):
    root = shadow.PRODUCTION_ROOT
    require(type(start) is int and start in (11031, 11032) and type(trial) is int and trial in (0, 1),
            'Only four registered worker slots exist')
    b, config, _ = shadow._binding(root, binding_sha, 'production')
    for name, module in [('diagnostic_worker', __file__), ('diagnostic_inputs', diagnostic_inputs.__file__),
                         ('diagnostics', diagnostics.__file__), ('native_runtime', native_runtime.__file__)]:
        actual = Path(module).resolve()
        require(Path(b['code'][name]['path']).resolve() == actual and
                shadow.file_sha(actual) == b['code'][name]['sha256'], 'Unbound worker dependency: ' + name)
    _no_holds(root, b)
    freeze = shadow.read_json(_bound(root, b['design_freeze']))
    _verify_runtime_sources(freeze)
    lock = shadow.read_json(_bound(root, b['source_lock']))
    prepared_path = _bound(root, b['diagnostic_inputs'])
    prepared = shadow.read_json(prepared_path)
    require(set(b['dry_receipts']) == {'11031', '11032'}, 'Both complete DRY receipts required')
    receipts = {int(s): _bound(root, ref) for s, ref in b['dry_receipts'].items()}
    # Reverify all 1,280 complete rows, their native scores/gates and fixed splits.
    checked = diagnostic_inputs.build_diagnostic_inputs(_bound(root, b['config']),
        _bound(root, b['design_freeze']), root, receipts, lock['native_src'])
    require(checked == prepared, 'Prepared diagnostic inputs differ from full DRY revalidation')
    artifacts = freeze['artifacts']
    registry = FrozenRegistry(_bound(root, b['config']), root, _bound(root, b['design_freeze']),
        [_bound(root, artifacts[k]) for k in ('train_manifest', 'dry_manifest', 'dev_manifest',
                                            'prefix_calib_manifest', 'main_manifest')],
        _bound(root, artifacts['inputs']), _bound(root, artifacts['images']))
    protected = [b['design_freeze'], b['config'], b['source_lock'], b['diagnostic_inputs'],
                 *b['dry_receipts'].values(), *[x['raw'] for x in prepared['starts'].values()]]
    return dict(root=root, binding=b, config=config, registry=registry, lock=lock,
                prepared=prepared, protected=protected, binding_sha=binding_sha)


def _save_tensor(path, data):
    import torch
    with path.open('xb') as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    shadow._sync_dir(path.parent)


def _relocate(root, audit, rows, endpoint, freeze_sha):
    relocated, manifest = [], []
    for original in rows:
        row = copy.deepcopy(original)
        source = Path(row['native_capture_path'])
        require(source.resolve(strict=True) == source and source.is_relative_to(root) and
                not source.is_relative_to(root / 'shadow_inputs'), 'Capture must be original DRY evidence')
        name = hashlib.sha256(row['request_id'].encode()).hexdigest() + '.CAPTURE.json'
        require(source.name == name and shadow.file_sha(source) == row['native_capture_sha256'],
                'Original capture SHA/name differs')
        begin = source.with_name(name.replace('.CAPTURE.json', '.BEGIN.json'))
        attempt = shadow.read_json(begin)
        require(attempt['request_id'] == row['request_id'] and attempt['endpoint'] == endpoint and
                attempt['design_freeze_sha256'] == freeze_sha, 'Original BEGIN identity differs')
        saved = shadow.read_json(source)
        for key in ('tokens', 'sampler_logprobs', 'prompt_tensor_hash', 'model_input', 'seed'):
            require(saved[key] == row[key], 'Original CAPTURE trajectory differs: ' + key)
        for src in (source, begin):
            sha = shadow.file_sha(src)
            data = src.read_bytes()
            require(hashlib.sha256(data).hexdigest() == sha, 'Original capture changed during copy')
            dest = audit / src.name
            with dest.open('xb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            require(shadow.file_sha(dest) == sha and shadow.file_sha(src) == sha, 'Relocation bytes differ')
            manifest.append(dict(source=str(src), destination=str(dest), sha256=sha))
        row['native_capture_path'] = str(audit / name)
        relocated.append(row)
    shadow.write_once(audit / 'RELOCATION.json', dict(schema='sr-satdose-capture-relocation-v1', files=manifest))
    return relocated, manifest


def _accounting(handle, begin, storage_begin, expected_rows):
    events = copy.deepcopy(handle.runtime._satdose_accounting[begin:])
    disk = []
    accounting_path = handle.output / 'accounting.jsonl'
    if accounting_path.exists():
        disk = [json.loads(line) for line in accounting_path.read_text().splitlines() if line.strip()]
    require(disk == handle.runtime._satdose_accounting, 'Disk/runtime accounting differ')
    totals = {'training_policy': 0, 'training_reference': 0}
    for event in events:
        require(event['kind'] == 'extra_forward_sequences' and type(event['count']) is int and event['count'] > 0,
                'Generation or unexpected accounting event in diagnostic worker')
        purpose = event['metadata']['purpose']
        require(purpose in totals, 'Unexpected forward purpose')
        totals[purpose] += event['count']
    # Expected counts are compared against measured native events, never emitted as measurements.
    require(totals == dict(training_policy=expected_rows, training_reference=expected_rows),
            'Actual native policy/reference accounting differs from replay rows')
    entries = copy.deepcopy(handle.storage_forwards[storage_begin:])
    require(len(entries) == expected_rows, 'Actual saved-tensor row count differs')
    handle.verify_storage_after_backward()
    return dict(events=events, measured_forward_sequences=totals, storage=entries)


def _before_shadow(captures, preclip):
    require(diagnostics.compare_replay(preclip, captures['total']['gradients'])['bit_exact'],
            'HOLD: production replay is not bit-exact')
    error = diagnostics.recomposition_error(captures['J']['gradients'], captures['KL']['gradients'],
                                            captures['total']['gradients'])
    require(math.isfinite(error) and error <= 1e-5, 'HOLD: component recomposition differs')
    pairs = [('J', 'source'), ('J_half_1', 'J_half_2'), ('source_half_1', 'source_half_2')]
    for left, right in pairs:
        require(diagnostics.cosine(captures[left]['gradients'], captures[right]['gradients']) > 0,
                'HOLD: nonpositive component direction ' + left + '/' + right)


def _execute(ctx, start, trial, process):
    with _worker_lock(ctx['root']):
        _no_holds(ctx['root'], ctx['binding'])
        return _execute_locked(ctx, start, trial, process)


def _execute_locked(ctx, start, trial, process):
    """Internal orchestration; CPU tests mock boundaries, never expose a production bypass."""
    import torch
    root, b, config = ctx['root'], ctx['binding'], ctx['config']
    slot = b['shadow_slots'][f'{start}/{trial}']
    destination = (root / slot['input_receipt']).parent
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    require(parent.resolve() == parent, 'Slot parent cannot be symlinked')
    if trial == 1:
        old = parent / 'trial-0'
        complete = shadow.read_json(old / 'WORKER_COMPLETE.json')
        restore = shadow.read_json(old / 'RESTORE.json')
        require(complete['status'] == 'DIAGNOSTIC_SLOT_COMPLETE_NOT_GPU_ADMISSION' and
                complete['technical_binding_sha256'] == ctx['binding_sha'] and
                complete['direction']['direction_checks_satisfied'] is True and
                complete['close']['closed'] is True, 'Trial0 worker incomplete or HOLD')
        require(restore['process_identity'] != process, 'Trial1 requires a new Python process')
        require(complete['restore_sha256'] == shadow.file_sha(old / 'RESTORE.json'), 'Trial0 restore changed')
        previous_shadow = root / shadow.CLAIMS_NAME / f'start-{start}' / 'trial-0' / 'COMPLETE.json'
        require(shadow.file_sha(previous_shadow) == complete['shadow_complete_sha256'], 'Trial0 shadow incomplete')
    _no_holds(root, b)
    destination.mkdir(exist_ok=False)
    shadow._sync_dir(parent)
    shadow.write_once(destination / 'WORKER_CLAIM.json', dict(status='RESERVED_NO_AUTOMATIC_RETRY',
        start_seed=start, trial=trial, technical_binding_sha256=ctx['binding_sha'], process_identity=process))
    audit = destination / 'audit'
    audit.mkdir()
    handle = None
    relocation, captures, stage_accounts, component_files = [], {}, {}, {}
    close = None
    try:
        source = config['start_bindings'][str(start)]
        source_path = Path(source['original_state_path'])
        require(shadow.file_sha(source_path) == source['state_sha256'], 'Original R24 changed')
        # Always constructs from original b0 + SHA-verified R24; never accepts a reused handle.
        handle = native_runtime.build_runtime(ctx['registry'], _bound(root, b['source_lock']),
                                             start_seed=start, output_dir=audit)
        endpoint = handle.endpoint()
        identity = dict(endpoint, parent='b0')
        selected = ctx['prepared']['starts'][str(start)]
        require(identity == selected['expected_identity'], 'Loaded endpoint differs from verified input')
        require(handle.fork.optimizer is None and handle.fork.scheduler is None,
                'Read-only runtime unexpectedly created an optimizer')
        require(not handle.runtime._satdose_accounting, 'Runtime construction performed unregistered model work')
        receipt = handle.fork.receipt
        require(receipt['source_state_path'] == str(source_path) and
                receipt['source_state_sha256'] == source['state_sha256'] and
                receipt['optimizer_created'] is False and receipt['model_calls'] == receipt['optimizer_steps'] == 0,
                'Actual fork receipt differs')
        instance = uuid.uuid4().hex
        common = dict(mode='production', technical_binding_sha256=ctx['binding_sha'],
                      design_freeze_sha256=b['design_freeze']['sha256'], endpoint=endpoint,
                      loader_instance_id=instance)
        restore = dict(common, schema='sr-satdose-restored-r24-v1', process_identity=process,
            source_state_path=str(source_path), source_state_sha256=shadow.file_sha(source_path),
            restored_from_original_state=True, actual_fork_receipt=copy.deepcopy(receipt))
        shadow.write_once(root / slot['restore'], restore)
        source_rows = selected['probe']['rows'] + (selected['witness']['rows'] if trial == 0 else [])
        rows, relocation = _relocate(root, audit, source_rows, endpoint, b['design_freeze']['sha256'])
        probe, witness = rows[:128], rows[128:]
        lock = ctx['lock']
        kernels = production.load_native_kernels(lock['native_src'], lock['legacy_root'])
        native = diagnostics.load_native_math(Path(lock['native_src']) / 'sr_f6/objective.py', lock['legacy_root'])
        def stage(name, batch, callback):
            before, sb = len(handle.runtime._satdose_accounting), len(handle.storage_forwards)
            result = callback()
            stage_accounts[name] = _accounting(handle, before, sb, len(batch))
            shadow.write_once(destination / (name + '_ACCOUNTING.json'), stage_accounts[name])
            return result
        produced = stage('production', probe, lambda: production.capture_update(handle.runtime.model, probe,
            handle.forward_original, handle.fork.reference, kernels=kernels, config=config,
            identity=identity, plan=selected['production_plan']))
        capture = dict(common, schema='sr-satdose-production-capture-v1', production=produced,
            request_ids_sha256=_sha_object([r['request_id'] for r in probe]),
            token_manifest_sha256=_sha_object([{k:r[k] for k in ('request_id','tokens','sampler_logprobs')} for r in probe]))
        _save_tensor(root / slot['capture'], capture)
        if trial == 0:
            halves = diagnostics.validate_partition(probe, selected['probe']['half_group_ids'])
            whalves = diagnostics.validate_partition(witness, selected['witness']['half_group_ids'])
            tasks = [(x, probe, x) for x in ('total','J','source','KL')]
            tasks += [(f'{x}_half_{i}', halves[i-1], x) for x in ('J','source') for i in (1,2)]
            tasks += [(f'witness_half_{i}{repeat}', whalves[i-1], 'witness')
                      for repeat in ('','_repeat') for i in (1,2)]
            for name, batch, component in tasks:
                captures[name] = stage(name, batch, lambda batch=batch,component=component:
                    diagnostics.accumulate_component(handle.runtime.model, batch, handle.forward_original, native, component))
                _save_tensor(destination / (name + '.pt'), captures[name])
                component_files[name] = _ref(root, destination / (name + '.pt'))
            shadow.write_once(destination / 'COMPONENTS.json', component_files)
            shadow.write_once(destination / 'ACCOUNTING.json', stage_accounts)
            _before_shadow(captures, produced['preclip_gradients'])
        _no_holds(root, b, current=(start, trial))
        if trial == 1:
            shadow.write_once(destination / 'ACCOUNTING.json', stage_accounts)
        input_receipt = dict(common, schema='sr-satdose-shadow-input-v1', start_seed=start, trial=trial,
                             restore=_ref(root, root / slot['restore']), capture=_ref(root, root / slot['capture']))
        shadow.write_once(root / slot['input_receipt'], input_receipt)
        policy = {n:p.detach().clone() for n,p in handle.runtime.model.named_parameters() if p.requires_grad}
        # This is the only shadow call site; one_trial reserves the fixed global slot before Adam.
        shadow.one_trial(policy, produced['preclip_gradients'], endpoint, start_seed=start, trial=trial,
            expected_binding_sha256=ctx['binding_sha'],
            expected_input_receipt_sha256=shadow.file_sha(root / slot['input_receipt']))
        claimdir = root / shadow.CLAIMS_NAME / f'start-{start}' / f'trial-{trial}'
        shadow_complete = shadow.read_json(claimdir / 'COMPLETE.json')
        require(shadow.file_sha(claimdir / 'RESULT.pt') == shadow_complete['result_sha256'], 'Shadow tensor evidence changed')
        if trial == 0:
            delta = torch.load(claimdir / 'RESULT.pt', map_location='cpu', weights_only=True)['delta']
            direction = diagnostics.direction_summary(captures, delta, produced['preclip_gradients'])
            shadow.write_once(destination / 'DIRECTION.json', direction)
            require(direction['direction_checks_satisfied'], 'HOLD: heldout direction uncertain')
        else:
            direction = dict(status='TRIAL0_DIRECTION_PRESERVED_RESTORE_REPLAY_ONLY',
                             direction_checks_satisfied=True)
        require(handle.endpoint() == endpoint, 'Read-only endpoint changed')
    except BaseException as exc:
        shadow.write_once(destination / 'HOLD.json', dict(status='HOLD_DIAGNOSTIC_NO_RETRY',
            technical_binding_sha256=ctx['binding_sha'], error_type=type(exc).__name__, error=str(exc),
            observed_accounting=copy.deepcopy(handle.runtime._satdose_accounting) if handle else [],
            observed_storage=copy.deepcopy(handle.storage_forwards) if handle else []))
        raise
    finally:
        try:
            if handle is not None:
                close = handle.close()
            for ref in ctx['protected']:
                _bound(root, ref)
            require(shadow.file_sha(source_path) == source['state_sha256'], 'Original R24 changed during worker')
            for entry in relocation:
                require(shadow.file_sha(entry['source']) == shadow.file_sha(entry['destination']) == entry['sha256'],
                        'Original/relocated capture changed')
        except BaseException as exc:
            if not (destination / 'HOLD.json').exists():
                shadow.write_once(destination / 'HOLD.json', dict(status='HOLD_DIAGNOSTIC_CLEANUP', error=str(exc)))
            raise
    complete = dict(schema='sr-satdose-diagnostic-worker-v1',
        status='DIAGNOSTIC_SLOT_COMPLETE_NOT_GPU_ADMISSION', start_seed=start, trial=trial,
        technical_binding_sha256=ctx['binding_sha'], process_identity=process, close=close,
        restore_sha256=shadow.file_sha(root / slot['restore']),
        shadow_complete_sha256=shadow.file_sha(claimdir / 'COMPLETE.json'), direction=direction,
        actual_accounting_sha256=shadow.file_sha(destination / 'ACCOUNTING.json'),
        component_files=component_files,
        remaining_checks=PENDING, permission='NO_TRAINING')
    shadow.write_once(destination / 'WORKER_COMPLETE.json', complete)
    return complete


def run_worker(*, start_seed, trial, expected_binding_sha256):
    global _USED
    require(not _USED, 'One diagnostic worker per fresh Python process')
    ctx = _preflight(expected_binding_sha256, start_seed, trial)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2 and torch.cuda.current_device() == 0,
            'Diagnostic worker requires two CUDA GPUs and arithmetic on cuda:0')
    _USED = True
    return _execute(ctx, start_seed, trial, _process_identity())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed', type=int, choices=(11031,11032), required=True)
    parser.add_argument('--trial', type=int, choices=(0,1), required=True)
    parser.add_argument('--binding-sha256', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_worker(start_seed=args.start_seed, trial=args.trial,
                               expected_binding_sha256=args.binding_sha256), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
