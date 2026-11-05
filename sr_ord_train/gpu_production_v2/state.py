"""Exact scientific checkpoint restore with explicit GPU execution lineage.

New checkpoints keep the frozen scientific identity and carry an additional
immutable execution-version reference in sampler state (and UPDATE after zero).
"""
from __future__ import annotations

import copy
from pathlib import Path

from adapter.backend import object_sha
from adapter.transactions import SCHEMA, _finite, _plain, _rng_unpack, file_hash, read_json, require, write_once
from gpu_production_v2.binding import MODE, verify_production_binding, _evidence
from diagnostics.gpu_recovery_state import original_rng_projection

LINEAGE_SCHEMA = 'sr-ord-train-gpu-production-lineage-v2'
PROVENANCE_SCHEMA = 'sr-ord-train-gpu-step-provenance-v2'


def actual_cuda_identity():
    from diagnostics.gpu_recovery_state import actual_cuda_identity as diagnostic_identity
    value = diagnostic_identity()
    value['scope'] = 'REGISTERED_GPU_PRODUCTION_EXECUTION'
    return value


def _binding_file(binding, lineage_dir):
    directory = Path(lineage_dir)
    verify_production_binding(binding, directory)
    require(Path(__file__).resolve() == Path(binding['package_root']) / 'gpu_production_v2/state.py',
            'Actual production restore helper is not bound')
    path = directory / 'GPU_EXECUTION_BINDING.json'
    require(read_json(path) == binding, 'Persisted production execution binding differs')
    return path


def initialize_lineage(tx, runtime, optimizer, scheduler, reference, binding, lineage_dir):
    """Record a fresh parent's two-GPU RNG before step zero, without a draw."""
    tx._locked()
    import torch
    from mm_core.training import capture_rng, state_hash, trainable_state
    from sr_f3_ce.training import validate_optimizer
    _binding_file(binding, lineage_dir)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Two visible GPUs required')
    tx._check_runtime(runtime)
    require(not tx.published() and not tx._attempts, 'Fresh parent cannot have checkpoints or attempts')
    require(runtime.identity['step'] == 0 and state_hash(trainable_state(runtime.model)) == state_hash(reference),
            'Fresh parent policy differs from reference')
    validate_optimizer(optimizer, scheduler, model=runtime.model)
    require(not optimizer.state_dict()['state'] and scheduler.state_dict()['last_epoch'] == 0,
            'Fresh parent optimizer/scheduler already advanced')
    rng = capture_rng()
    require(set(rng) == {'python', 'numpy', 'cpu', 'cuda'} and len(rng['cuda']) == 2,
            'Fresh parent two-GPU RNG required')
    receipt = dict(schema=LINEAGE_SCHEMA, mode=MODE, status='INITIAL_PARENT_TWO_GPU', source_cuda_count=0,
        source_identity=copy.deepcopy(tx.identity), source_checkpoint=None, source_commit_sha256=None,
        source_step=None, source_run_binding=copy.deepcopy(binding['gpu_production']['source_run_binding']),
        execution_binding_sha256=object_sha(binding),
        execution_implementation_sha256=binding['gpu_production']['execution_implementation_sha256'],
        policy_sha256=state_hash(reference), reference_sha256=state_hash(reference),
        optimizer_sha256=state_hash(optimizer.state_dict()), scheduler_sha256=state_hash(scheduler.state_dict()),
        source_rng_sha256=None, restored_original_rng_projection_sha256=None,
        restored_full_two_gpu_rng_sha256=state_hash(rng), model_calls=0, optimizer_steps=0, scheduler_steps=0)
    write_once(Path(lineage_dir) / 'GPU_RECOVERY_LINEAGE.json', receipt)
    return None


def execution_provenance(binding, lineage_dir, step):
    """Reference immutable version/restore evidence in each atomic checkpoint."""
    require(type(step) is int and 0 <= step <= 24, 'Invalid GPU execution step')
    binding_file = _binding_file(binding, lineage_dir)
    path = Path(lineage_dir) / 'GPU_RECOVERY_LINEAGE.json'
    lineage = read_json(path)
    require(lineage['schema'] == LINEAGE_SCHEMA and lineage['mode'] == MODE
            and lineage['execution_binding_sha256'] == object_sha(binding), 'GPU lineage binding differs')
    require(all(lineage.get(k) == 0 for k in ('model_calls', 'optimizer_steps', 'scheduler_steps')),
            'Restore lineage must precede model calls/updates')
    source_step = lineage['source_step']
    require((source_step is None and lineage['status'] == 'INITIAL_PARENT_TWO_GPU')
            or (type(source_step) is int and source_step < step and lineage['status'] in
                ('EXACT_ORIGINAL_STATE_RESTORED', 'EXACT_TWO_GPU_STATE_RESTORED')),
            'New step does not follow restored source')
    expected_cuda = {'INITIAL_PARENT_TWO_GPU': 0, 'EXACT_ORIGINAL_STATE_RESTORED': 1,
                     'EXACT_TWO_GPU_STATE_RESTORED': 2}[lineage['status']]
    require(lineage['source_cuda_count'] == expected_cuda, 'Restore lineage CUDA count differs')
    if source_step is not None:
        require(lineage['source_rng_sha256'] == lineage['restored_original_rng_projection_sha256']
                and set(lineage['rng_components']) == {'python', 'numpy', 'cpu', 'cuda'}
                and all(v['source_sha256'] == v['restored_sha256'] for v in lineage['rng_components'].values())
                and lineage['gpu1_rng']['before_sha256'] == lineage['gpu1_rng']['after_sha256'],
                'Restore lineage RNG preservation failed')
        if expected_cuda == 2:
            require(lineage['source_rng_sha256'] == lineage['restored_full_two_gpu_rng_sha256'],
                    'Restored full two-GPU RNG differs')
    return dict(schema=PROVENANCE_SCHEMA, mode=MODE, step=step,
        execution_binding=dict(path=str(binding_file), sha256=file_hash(binding_file)),
        recovery_lineage=dict(path=str(path), sha256=file_hash(path)),
        source_commit_sha256=lineage['source_commit_sha256'], source_step=source_step)


def verify_checkpoint_provenance(package, marker):
    """Validate new checkpoint bridge; source identity remains the old lineage.

    Returns (derived_binding, lineage). The original transaction caller already
    validates COMMIT/state/UPDATE hashes. This additionally binds actual version.
    """
    package = _plain(package)
    prov = marker['sampler_state'].get('gpu_execution_provenance')
    if isinstance(prov, dict) and prov.get('schema') == 'sr-ord-train-gpu-step-provenance-v1':
        require(prov.get('mode') == 'cuda0_compute_cuda1_saved_tensors_16GiB',
                'Recognized v1 checkpoint mode required for explicit version transition')
        from gpu_production.state import verify_checkpoint_provenance as verify_v1_checkpoint
        return verify_v1_checkpoint(package, marker)
    require(isinstance(prov, dict) and set(prov) == {'schema', 'mode', 'step', 'execution_binding',
            'recovery_lineage', 'source_commit_sha256', 'source_step'}, 'Missing exact GPU checkpoint provenance')
    require(prov['schema'] == PROVENANCE_SCHEMA and prov['mode'] == MODE and prov['step'] == marker['step'],
            'GPU checkpoint provenance cursor differs')
    binding_file = _evidence(prov['execution_binding'], package)
    lineage_file = _evidence(prov['recovery_lineage'], package)
    require(binding_file.name == 'GPU_EXECUTION_BINDING.json' and lineage_file.name == 'GPU_RECOVERY_LINEAGE.json'
            and binding_file.parent == lineage_file.parent, 'GPU checkpoint evidence directory differs')
    binding = read_json(binding_file); original = verify_production_binding(binding, binding_file.parent)
    lineage = read_json(lineage_file)
    require(execution_provenance(binding, binding_file.parent, marker['step']) == prov,
            'GPU checkpoint provenance differs from immutable lineage')
    identity = marker['identity']
    require(lineage['source_identity'] == identity
            and identity['manifest_sha256'] == original['manifest_sha256']
            and identity['source_sha256'] == original['source_sha256']
            and identity['code_commit'] == original['code_commit']
            and identity['implementation_sha256'] == object_sha(original['implementation_files'])
            and binding['path_identity'] == {k: identity[k] for k in ('arm', 'train_seed', 'run_id')},
            'GPU checkpoint original scientific identity differs')
    require(lineage['source_run_binding'] == binding['gpu_production']['source_run_binding']
            and lineage['execution_implementation_sha256'] == binding['gpu_production']['execution_implementation_sha256'],
            'GPU lineage source/version differs')
    stage = package / 'run/paths' / identity['run_id'].split('/')[-1]
    require(binding_file.parent.parent == stage / 'executions', 'GPU checkpoint evidence path differs')
    if prov['source_step'] is None:
        require(lineage['source_checkpoint'] is None and prov['source_commit_sha256'] is None,
                'Fresh GPU lineage has old checkpoint')
        zero = marker if marker['step'] == 0 else read_json(stage / 'checkpoints/step-000/COMMIT.json')
        require(zero['identity'] == identity and zero['step'] == 0
                and zero['policy_sha256'] == zero['reference_sha256'] == lineage['policy_sha256'] == lineage['reference_sha256']
                and zero['optimizer_sha256'] == lineage['optimizer_sha256']
                and zero['scheduler_sha256'] == lineage['scheduler_sha256']
                and zero['rng_sha256'] == lineage['restored_full_two_gpu_rng_sha256']
                and lineage['source_rng_sha256'] is None
                and lineage['restored_original_rng_projection_sha256'] is None,
                'Fresh GPU origin differs from committed parent state')
    else:
        source_dir = stage / 'checkpoints' / f"step-{prov['source_step']:03d}"
        require(_evidence(lineage['source_checkpoint'], package) == source_dir / 'state.pt'
                and file_hash(source_dir / 'COMMIT.json') == prov['source_commit_sha256'],
                'Restored GPU source checkpoint changed')
        source = read_json(source_dir / 'COMMIT.json')
        require(source['identity'] == identity and source['step'] == prov['source_step']
                and all(source[k] == lineage[k] for k in ('policy_sha256', 'reference_sha256',
                                                        'optimizer_sha256', 'scheduler_sha256'))
                and source['rng_sha256'] == lineage['source_rng_sha256'], 'GPU restored source state differs')
    if marker['step']:
        update_file = stage / 'checkpoints' / f"step-{marker['step']:03d}" / 'UPDATE.json'
        require(file_hash(update_file) == marker['update_sha256']
                and read_json(update_file).get('gpu_execution_provenance') == prov,
                'UPDATE and checkpoint GPU provenance differ')
    return binding, lineage


def restore_latest_gpu(tx, runtime, optimizer, scheduler, reference, production_binding, lineage_dir):
    """Original restore checks plus explicit GPU1 lineage, with no source writes.

    Return the unchanged original marker. Runtime step/policy cursor advances only
    after all checks and the new write-once lineage receipt succeed. Source tx
    identity and original checkpoint metadata remain unchanged.
    """
    tx._locked()
    import torch
    from mm_core.training import restore_rng, capture_rng, state_hash, trainable_state
    from sr_f3_ce.training import validate_optimizer
    original = verify_production_binding(production_binding, lineage_dir)
    _binding_file(production_binding, lineage_dir)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Two visible GPUs required for lineage')
    tx._check_runtime(runtime)
    require(runtime.identity.get('execution_mode') == 'TRAINING', 'Production training runtime required')
    require(runtime.identity.get('manifest_sha256') == original['manifest_sha256'], 'Runtime manifest differs')
    require(tx.identity.get('manifest_sha256') == original['manifest_sha256'] and
            tx.identity.get('source_sha256') == original['source_sha256'] and
            tx.identity.get('code_commit') == original['code_commit'] and
            tx.identity.get('implementation_sha256') == object_sha(original['implementation_files']),
            'Original transaction/binding identity differs')
    require(production_binding['path_identity'] == {k: tx.identity[k] for k in ('arm', 'train_seed', 'run_id')},
            'Production transaction path differs')
    package = Path(original['package_root'])
    name = f"{tx.identity['parent']}-{tx.identity['train_seed']}-{tx.identity['arm']}"
    require(tx.directory == package / 'run/paths' / name and read_json(tx.directory / 'IDENTITY.json') == tx.identity,
            'Original on-disk path identity differs')
    plan = package / 'frozen/receipt_plans/paths' / (name + '.json')
    index = read_json(package / 'frozen/receipt_plans/INDEX.json')
    require(file_hash(plan) == tx.identity['plan_sha256'] == index['files']['paths/' + name + '.json'],
            'Original frozen path plan differs')
    destination = _plain(Path(lineage_dir) / 'GPU_RECOVERY_LINEAGE.json')
    require(not destination.exists(), 'Production lineage already exists')
    # Uncached original validation prevents a previously cached marker from
    # hiding changed bytes. This changes only this locked transaction's cache.
    tx._markers = None
    markers = tx.published()
    require(not any('complete' not in a for a in tx._attempts.values()),
            'UNKNOWN attempt requires audit before restore/resampling')
    if not markers:
        return initialize_lineage(tx, runtime, optimizer, scheduler, reference, production_binding, lineage_dir)
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
    require(set(rng) == {'python', 'numpy', 'cpu', 'cuda'} and len(rng['cuda']) in (1, 2),
            'Checkpoint must contain one original or two bound CUDA RNG states')
    source_cuda_count = len(rng['cuda'])
    if source_cuda_count == 2:
        verify_checkpoint_provenance(package, marker)
    else:
        require('gpu_execution_provenance' not in marker['sampler_state'], 'Original single-GPU checkpoint has GPU bridge metadata')
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
    if source_cuda_count == 1:
        expanded['cuda'].append(gpu1)
    else:
        gpu1 = rng['cuda'][1].clone()
    restore_rng(expanded)
    restored = capture_rng()
    projected = original_rng_projection(restored) if source_cuda_count == 1 else restored
    require(state_hash(projected) == marker['rng_sha256'], 'Original CPU/GPU0 RNG projection differs')
    require(state_hash(restored['cuda'][1]) == state_hash(gpu1), 'Independent GPU1 RNG changed during restore')
    components = {k: dict(source_sha256=state_hash(rng[k]), restored_sha256=state_hash(projected[k]))
                  for k in ('python', 'numpy', 'cpu', 'cuda')}
    require(all(v['source_sha256'] == v['restored_sha256'] for v in components.values()), 'RNG component differs')
    require(file_hash(path) == marker['state_sha256'], 'State changed during restore')
    receipt = dict(schema=LINEAGE_SCHEMA, mode=MODE, status='EXACT_ORIGINAL_STATE_RESTORED' if source_cuda_count == 1 else 'EXACT_TWO_GPU_STATE_RESTORED',
        source_cuda_count=source_cuda_count,
        source_identity=copy.deepcopy(tx.identity), source_checkpoint=dict(path=str(path), sha256=marker['state_sha256']),
        source_commit_sha256=marker['commit_sha256'], source_step=marker['step'],
        source_run_binding=copy.deepcopy(production_binding['gpu_production']['source_run_binding']),
        execution_binding_sha256=object_sha(production_binding), execution_implementation_sha256=production_binding['gpu_production']['execution_implementation_sha256'],
        policy_sha256=marker['policy_sha256'], reference_sha256=marker['reference_sha256'],
        optimizer_sha256=marker['optimizer_sha256'], scheduler_sha256=marker['scheduler_sha256'],
        source_rng_sha256=marker['rng_sha256'], restored_original_rng_projection_sha256=state_hash(projected),
        restored_full_two_gpu_rng_sha256=state_hash(restored), rng_components=components,
        gpu1_rng=dict(origin='preexisting_production_gpu1_state_retained_without_draw' if source_cuda_count == 1 else 'restored_committed_gpu1_state',
                      before_sha256=state_hash(gpu1), after_sha256=state_hash(restored['cuda'][1]),
                      state_bytes=gpu1.cpu().tolist(), dtype=str(gpu1.dtype)),
        model_calls=0, optimizer_steps=0, scheduler_steps=0)
    write_once(destination, receipt)
    runtime.identity.update(step=marker['step'], policy_hash=marker['policy_sha256'])
    return copy.deepcopy(marker)
