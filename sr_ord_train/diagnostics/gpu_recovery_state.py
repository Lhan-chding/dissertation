"""Isolated two-GPU diagnostic binding and one-to-two GPU RNG lineage.

No production binding/checkpoint is modified. ``make_diagnostic_binding`` returns
a new in-memory binding: the original implementation inventory remains an exact
prefix, and the diagnostic files are appended. Callers persist that binding in
their new diagnostic directory before loading ``gpu_backend.NativeBackend``.

``restore_latest_gpu`` mirrors PathTransactions.restore_latest, including every
metadata/content/runtime check. The ONLY state-semantic difference is explicit
RNG expansion: original Python/NumPy/CPU/GPU0 state is restored exactly; GPU1's
pre-existing state is retained and recorded separately. No random draw, forward,
optimizer step, transaction commit, or production write is performed here.
The caller must hold an original-path read lock for the entire operation.
"""
from __future__ import annotations

import copy
import csv
import os
from pathlib import Path
import socket
import subprocess

from adapter.backend import file_sha, object_sha, verify_file
from adapter.transactions import (SCHEMA, _finite, _plain, _rng_unpack, file_hash,
                                  read_json, require, write_once)

DIAGNOSTIC_SCHEMA = 'sr-ord-train-two-gpu-diagnostic-v1'
DIAGNOSTIC_SCOPE = 'ISOLATED_GPU_DIAGNOSTIC_NO_PRODUCTION_COMMIT'
_REQUIRED_FILES = ('diagnostics/gpu_backend.py', 'diagnostics/gpu_recovery_state.py')


def _execution_directory(package, path):
    path = _plain(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
            'Existing canonical diagnostic directory required')
    root = package / 'run'
    require(path != root and path.is_relative_to(root) and
            not path.is_relative_to(root / 'paths'), 'Diagnostic directory must be outside production paths')
    return path


def make_diagnostic_binding(source_binding_path, execution_root, additional_files=(), *, path_identity):
    """Return a derived binding; never rewrite RUN_BINDING or caller dictionaries."""
    source_path = _plain(source_binding_path)
    require(source_path.is_absolute(), 'Absolute original RUN_BINDING path required')
    source = read_json(source_path)
    require('gpu_diagnostic' not in source, 'Original binding, not a derived binding, required')
    package = Path(source['package_root'])
    root = _execution_directory(package, execution_root)
    require(not source_path.is_relative_to(root), 'Original binding cannot be diagnostic output')
    original = source['implementation_files']
    required = [package / p for p in _REQUIRED_FILES]
    extras = list(dict.fromkeys([*required, *(Path(p) for p in additional_files)]))
    seen = {r['path'] for r in original}
    rows = []
    for p in extras:
        p = _plain(p)
        require(p.is_absolute() and p.resolve(strict=True) == p and p.is_relative_to(package),
                'Canonical package diagnostic implementation required')
        require(str(p) not in seen, 'Diagnostic implementation overlaps original inventory')
        rows.append(dict(path=str(p), sha256=file_sha(p)))
        seen.add(str(p))
    result = copy.deepcopy(source)
    result['implementation_files'] = copy.deepcopy(original) + rows
    result['path_identity'] = copy.deepcopy(path_identity)
    result['gpu_diagnostic'] = dict(schema=DIAGNOSTIC_SCHEMA, scope=DIAGNOSTIC_SCOPE,
        source_run_binding=dict(path=str(source_path), sha256=file_hash(source_path)),
        execution_root=str(root), visible_gpu_count=2,
        original_implementation_sha256=object_sha(original))
    verify_diagnostic_binding(result, root)
    return result


def verify_diagnostic_binding(binding, run_dir):
    """Verify unchanged original fields and both original/new code before loading."""
    d = binding.get('gpu_diagnostic', {})
    require(d.get('schema') == DIAGNOSTIC_SCHEMA and d.get('scope') == DIAGNOSTIC_SCOPE
            and d.get('visible_gpu_count') == 2, 'Explicit isolated two-GPU binding required')
    require(set(d) == {'schema', 'scope', 'source_run_binding', 'execution_root',
                      'visible_gpu_count', 'original_implementation_sha256'}, 'Unknown diagnostic binding fields')
    source_path = verify_file(d['source_run_binding'])
    original = read_json(source_path)
    require('gpu_diagnostic' not in original, 'Diagnostic source must be original RUN_BINDING')
    package = Path(original['package_root'])
    require(package.is_absolute() and package.resolve(strict=True) == package, 'Canonical package required')
    root = _execution_directory(package, d['execution_root'])
    run = _execution_directory(package, run_dir)
    require(run.is_relative_to(root) and not source_path.is_relative_to(root), 'Diagnostic output root differs')
    ignored = {'path_identity', 'implementation_files', 'gpu_diagnostic'}
    require({k: v for k, v in binding.items() if k not in ignored} ==
            {k: v for k, v in original.items() if k not in ignored}, 'Original RUN_BINDING fields changed')
    source_rows = original['implementation_files']
    rows = binding['implementation_files']
    require(rows[:len(source_rows)] == source_rows and len(rows) > len(source_rows),
            'Original implementation inventory must remain an exact prefix')
    require(d['original_implementation_sha256'] == object_sha(source_rows), 'Original implementation digest differs')
    paths = [verify_file(r) for r in rows]
    require(len(set(paths)) == len(paths), 'Duplicate implementation binding')
    added = set(paths[len(source_rows):])
    require({package / p for p in _REQUIRED_FILES} <= added, 'Diagnostic backend/state code not bound')
    require(all(p.resolve(strict=True) == p and p.is_relative_to(package) for p in added),
            'Diagnostic code outside canonical package')
    require(Path(__file__).resolve() in added, 'Actual recovery helper is not bound')
    identity = binding.get('path_identity', {})
    require(set(identity) == {'arm', 'train_seed', 'run_id'} and identity['arm'] in ('R', 'G', 'D')
            and type(identity['train_seed']) is int and identity['train_seed'] in (11031, 11032)
            and isinstance(identity['run_id'], str) and identity['run_id'].startswith('SR-ORD-TRAIN-1/'),
            'Original training path identity required')
    return original


def actual_cuda_identity():
    """Record the two allocated PRO6000 devices; never disguise visible count."""
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Exactly two visible CUDA GPUs required')
    query = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = []
    for row in csv.reader(query.stdout.splitlines()):
        if row:
            require(len(row) == 5, 'Unexpected NVIDIA driver identity columns')
            rows.append(dict(zip(('name', 'uuid', 'pci_bus_id', 'driver_version', 'memory_mib'),
                                 (v.strip() for v in row))))
    normalize = lambda value: value.lower().removeprefix('gpu-').replace('-', '')
    devices = []
    for index in range(2):
        prop = torch.cuda.get_device_properties(index)
        uuid = str(getattr(prop, 'uuid', ''))
        require(uuid and 'RTX PRO 6000' in prop.name.upper(), 'Allocated GPU must be identifiable PRO6000')
        matched = [r for r in rows if normalize(r['uuid']) == normalize(uuid)]
        require(len(matched) == 1 and matched[0]['name'] == prop.name, 'CUDA/driver GPU identity mismatch')
        devices.append(dict(matched[0], cuda_index=index, cuda_uuid=uuid,
                            cuda_total_memory_bytes=prop.total_memory, compute_capability=[prop.major, prop.minor]))
    require(devices[0]['cuda_uuid'] != devices[1]['cuda_uuid'], 'Duplicate allocated GPU')
    require(devices[0]['name'] == devices[1]['name'] and
            devices[0]['compute_capability'] == devices[1]['compute_capability'], 'Matching GPU models required')
    peer = {'0_to_1': bool(torch.cuda.can_device_access_peer(0, 1)),
            '1_to_0': bool(torch.cuda.can_device_access_peer(1, 0))}
    require(all(peer.values()), 'GPU-only diagnostic requires bidirectional peer access')
    return dict(cuda_visible_device_count=2, devices=devices, peer_access=peer,
                cuda_runtime=torch.version.cuda, hostname=socket.gethostname(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), scope=DIAGNOSTIC_SCOPE)


def original_rng_projection(rng):
    require(set(rng) == {'python', 'numpy', 'cpu', 'cuda'} and len(rng['cuda']) == 2,
            'Two-GPU RNG snapshot required')
    result = copy.deepcopy(rng)
    result['cuda'] = result['cuda'][:1]
    return result


def restore_latest_gpu(tx, runtime, optimizer, scheduler, reference, diagnostic_binding, lineage_dir):
    """Original restore checks plus explicit GPU1 lineage, with no source writes.

    Return the unchanged original marker. Runtime step/policy cursor advances only
    after all checks and the new write-once lineage receipt succeed. Source tx
    identity and original checkpoint metadata remain unchanged.
    """
    tx._locked()
    import torch
    from mm_core.training import restore_rng, capture_rng, state_hash, trainable_state
    from sr_f3_ce.training import validate_optimizer
    original = verify_diagnostic_binding(diagnostic_binding, lineage_dir)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Two visible GPUs required for lineage')
    tx._check_runtime(runtime)
    require(runtime.identity.get('execution_mode') == 'TRAINING', 'Diagnostic training runtime required')
    require(runtime.identity.get('manifest_sha256') == original['manifest_sha256'], 'Runtime manifest differs')
    require(tx.identity.get('manifest_sha256') == original['manifest_sha256'] and
            tx.identity.get('source_sha256') == original['source_sha256'] and
            tx.identity.get('code_commit') == original['code_commit'] and
            tx.identity.get('implementation_sha256') == object_sha(original['implementation_files']),
            'Original transaction/binding identity differs')
    require(diagnostic_binding['path_identity'] == {k: tx.identity[k] for k in ('arm', 'train_seed', 'run_id')},
            'Diagnostic transaction path differs')
    package = Path(original['package_root'])
    name = f"{tx.identity['parent']}-{tx.identity['train_seed']}-{tx.identity['arm']}"
    require(tx.directory == package / 'run/paths' / name and read_json(tx.directory / 'IDENTITY.json') == tx.identity,
            'Original on-disk path identity differs')
    plan = package / 'frozen/receipt_plans/paths' / (name + '.json')
    index = read_json(package / 'frozen/receipt_plans/INDEX.json')
    require(file_hash(plan) == tx.identity['plan_sha256'] == index['files']['paths/' + name + '.json'],
            'Original frozen path plan differs')
    destination = _plain(Path(lineage_dir) / 'GPU_RECOVERY_LINEAGE.json')
    require(not destination.exists(), 'Diagnostic lineage already exists')
    # Uncached original validation prevents a previously cached marker from
    # hiding changed bytes. This changes only the diagnostic reader's cache.
    tx._markers = None
    markers = tx.published()
    require(markers, 'Committed original checkpoint required')
    require(not any('complete' not in a for a in tx._attempts.values()),
            'UNKNOWN attempt requires audit before restore/resampling')
    marker = markers[-1]
    require(state_hash(reference) == marker['reference_sha256'], 'Restore parent reference differs')
    path = tx.directory / 'checkpoints' / f"step-{marker['step']:03d}" / 'state.pt'
    require(file_hash(path) == marker['state_sha256'], 'State changed before restore')
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(state['schema'] == SCHEMA and state['identity'] == tx.identity
            and state['step'] == marker['step'] and state['sampler_state'] == marker['sampler_state'],
            'Serialized checkpoint metadata differs')
    require(state['reference_sha256'] == marker['reference_sha256']
            and state_hash(state['policy']) == marker['policy_sha256']
            and state_hash(state['optimizer']) == marker['optimizer_sha256']
            and state_hash(state['scheduler']) == marker['scheduler_sha256'], 'Serialized content differs')
    rng = _rng_unpack(state['rng'])
    require(state_hash(rng) == marker['rng_sha256'], 'Serialized RNG differs')
    require(set(rng) == {'python', 'numpy', 'cpu', 'cuda'} and len(rng['cuda']) == 1,
            'Original checkpoint must contain exactly one CUDA RNG state')
    parameters = {n: p for n, p in runtime.model.named_parameters() if p.requires_grad}
    require(set(parameters) == set(state['policy']) and _finite(state['policy'])
            and _finite(state['optimizer']), 'Restore parameter inventory/nonfinite state')
    before_rng = capture_rng()
    require(len(before_rng['cuda']) == 2, 'Independent GPU1 RNG missing')
    gpu1 = before_rng['cuda'][1].clone()
    with torch.no_grad():
        for name, value in state['policy'].items():
            require(value.dtype == torch.float32 and parameters[name].dtype == torch.float32
                    and value.shape == parameters[name].shape, 'Restore parameter dtype/shape differs')
            parameters[name].copy_(value)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    validate_optimizer(optimizer, scheduler, model=runtime.model)
    require(state_hash(optimizer.state_dict()) == marker['optimizer_sha256']
            and state_hash(scheduler.state_dict()) == marker['scheduler_sha256'], 'Loaded optimizer/scheduler differs')
    require(state_hash(trainable_state(runtime.model)) == marker['policy_sha256'], 'Loaded policy differs')
    expanded = copy.deepcopy(rng)
    expanded['cuda'].append(gpu1)
    restore_rng(expanded)
    restored = capture_rng()
    projected = original_rng_projection(restored)
    require(state_hash(projected) == marker['rng_sha256'], 'Original CPU/GPU0 RNG projection differs')
    require(state_hash(restored['cuda'][1]) == state_hash(gpu1), 'Independent GPU1 RNG changed during restore')
    components = {k: dict(source_sha256=state_hash(rng[k]), restored_sha256=state_hash(projected[k]))
                  for k in ('python', 'numpy', 'cpu', 'cuda')}
    require(all(v['source_sha256'] == v['restored_sha256'] for v in components.values()), 'RNG component differs')
    require(file_hash(path) == marker['state_sha256'], 'State changed during restore')
    receipt = dict(schema=DIAGNOSTIC_SCHEMA, scope=DIAGNOSTIC_SCOPE, status='EXACT_ORIGINAL_STATE_RESTORED',
        source_identity=copy.deepcopy(tx.identity), source_checkpoint=dict(path=str(path), sha256=marker['state_sha256']),
        source_commit_sha256=marker['commit_sha256'], source_step=marker['step'],
        source_run_binding=copy.deepcopy(diagnostic_binding['gpu_diagnostic']['source_run_binding']),
        diagnostic_binding_sha256=object_sha(diagnostic_binding), diagnostic_implementation_sha256=object_sha(diagnostic_binding['implementation_files']),
        policy_sha256=marker['policy_sha256'], reference_sha256=marker['reference_sha256'],
        optimizer_sha256=marker['optimizer_sha256'], scheduler_sha256=marker['scheduler_sha256'],
        source_rng_sha256=marker['rng_sha256'], restored_original_rng_projection_sha256=state_hash(projected),
        restored_full_two_gpu_rng_sha256=state_hash(restored), rng_components=components,
        gpu1_rng=dict(origin='preexisting_diagnostic_gpu1_state_retained_without_draw',
                      before_sha256=state_hash(gpu1), after_sha256=state_hash(restored['cuda'][1]),
                      state_bytes=gpu1.cpu().tolist(), dtype=str(gpu1.dtype)),
        production_commit_authorized=False, model_calls=0, optimizer_steps=0, scheduler_steps=0)
    write_once(destination, receipt)
    runtime.identity.update(step=marker['step'], policy_hash=marker['policy_sha256'])
    return copy.deepcopy(marker)
