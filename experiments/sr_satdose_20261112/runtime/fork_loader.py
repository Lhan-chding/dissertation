"""SATDOSE R24 fork and explicit committed-state restore; no model construction.

Call ``load_native_ops`` only after auditing the complete native source binding.
``fork_runtime`` accepts a runtime already loaded from the original b0 adapter.
It freezes that adapter BEFORE copying the R24 policy and creates native fresh
AdamW. This module neither samples nor updates nor publishes checkpoints.
Resume requires a separately trusted COMMIT SHA and exact new path identity;
the driver must also validate the transaction's raw/UPDATE/COMMIT chain.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

SCHEMA = 'sr-satdose-state-v1'
LEGACY_SCHEMA = 'sr-ord-train-transaction-v1'
EXPERIMENT = 'SR-SATDOSE-1-20261112'


def require(condition, message):
    if not condition:
        raise PermissionError(message)


def file_sha(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Checkpoint must be a regular nonsymlink file')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_native_ops(native_src, source_pins):
    """Import real native hash/optimizer/RNG functions, with exact file pins."""
    root = Path(native_src).resolve(strict=True)
    names = ('mm_core.training', 'sr_f3_ce.training')
    for name in names:
        rel = name.replace('.', '/') + '.py'
        require(rel in source_pins and file_sha(root / rel) == source_pins[rel],
                'Native hash/optimizer source not pinned: ' + rel)
    sys.path.insert(0, str(root))
    try:
        mods = [importlib.import_module(name) for name in names]
    finally:
        sys.path.pop(0)
    for name, mod in zip(names, mods):
        require(Path(mod.__file__).resolve() == root / (name.replace('.', '/') + '.py'),
                'Shadowed native module: ' + name)
    core, training = mods
    return SimpleNamespace(state_hash=core.state_hash, trainable_state=core.trainable_state,
        capture_rng=core.capture_rng, restore_rng=core.restore_rng,
        fresh_optimizer=training.fresh_optimizer, validate_optimizer=training.validate_optimizer,
        validate_optimizer_state=training.validate_optimizer_state, finite_tree=training.finite_tree)


def _path(config, start_seed, arm):
    require(config['experiment_id'] == EXPERIMENT and config['parent'] == 'b0', 'Wrong experiment/parent')
    require(type(start_seed) is int and start_seed in (11031, 11032) and arm in ('T0', 'T1'), 'Wrong fork path')
    require(config['precision']['expected_trainable_tensors'] == 496, 'Expected all 496 LoRA tensors')
    return config['start_bindings'][str(start_seed)]


def _parameters(model, policy, native):
    import torch
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(len(parameters) == 496 and set(parameters) == set(policy), 'Exact 496-parameter inventory required')
    require(all('lora_' in n for n in parameters), 'Only LoRA parameters may be trainable')
    require(all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point())
            and all(b.dtype == torch.float32 for b in model.buffers() if b.is_floating_point()),
            'Model parameters/buffers must remain FP32')
    require(native.finite_tree(policy), 'Nonfinite policy')
    for name, value in policy.items():
        require(torch.is_tensor(value) and value.dtype == torch.float32
                and value.shape == parameters[name].shape, 'Policy tensor dtype/shape mismatch: ' + name)
    return parameters


def _copy(parameters, policy):
    import torch
    with torch.no_grad():
        for name, value in policy.items():
            parameters[name].copy_(value)


def _optimizer_check(native, model, optimizer, scheduler, step):
    native.validate_optimizer(optimizer, scheduler, model=model)
    state = optimizer.state_dict()
    require(all(g.get('foreach') is False for g in state['param_groups']), 'foreach must be false')
    require(scheduler.state_dict()['last_epoch'] == step, 'Scheduler cursor mismatch')
    require(native.finite_tree(state), 'Nonfinite Adam state')
    if step == 0:
        require(not state['state'], 'Fresh fork must not inherit Adam history')
    else:
        require(len(state['state']) == 496 and all(float(s['step']) == step for s in state['state'].values()),
                'Adam cursor/inventory mismatch')


def _fork_runtime(runtime, config, start_seed, arm, native, *, checkpoint_path=None, create_optimizer):
    """Fork an already-loaded original parent, returning frozen ref and fresh Adam.

    ``checkpoint_path`` supports a relocated byte-identical source only. All
    checks precede copying; failures restore the incoming policy and RNG.
    Callers must persist the returned receipt in their new immutable lineage.
    """
    import torch
    binding = _path(config, start_seed, arm)
    source = Path(checkpoint_path or binding['original_state_path'])
    require(runtime.identity.get('step', 0) == 0, 'Fork requires a newly loaded parent runtime')
    require(file_sha(source) == binding['state_sha256'], 'R24 checkpoint bytes differ')
    # Capture the original parent before reading or copying the R24 tensor tree.
    reference = native.trainable_state(runtime.model)
    parameters = _parameters(runtime.model, reference, native)
    require(native.state_hash(reference) == binding['reference_sha256'], 'Original b0 reference not loaded first')
    require(all(not t.requires_grad for t in reference.values()), 'Reference must be detached')
    state = torch.load(source, map_location='cpu', weights_only=True)
    require(state['schema'] == LEGACY_SCHEMA and state['step'] == 24, 'Expected legacy R24')
    old = state['identity']
    require(old.get('parent') == 'b0' and old.get('train_seed') == start_seed and old.get('arm') == 'R',
            'Legacy checkpoint path identity mismatch')
    require(state['reference_sha256'] == binding['reference_sha256'], 'Legacy reference differs')
    _parameters(runtime.model, state['policy'], native)
    require(native.state_hash(state['policy']) == binding['policy_sha256'], 'R24 policy tensor SHA differs')
    identity_before = copy.deepcopy(runtime.identity)
    rng_before = native.capture_rng()
    try:
        _copy(parameters, state['policy'])
        optimizer, scheduler = None, None
        if create_optimizer:
            optimizer, scheduler = native.fresh_optimizer(runtime.model)
            _optimizer_check(native, runtime.model, optimizer, scheduler, 0)
        require(native.state_hash(native.trainable_state(runtime.model)) == binding['policy_sha256'], 'Loaded R24 differs')
        require(native.state_hash(reference) == binding['reference_sha256'], 'Original reference changed')
        require(native.state_hash(native.capture_rng()) == native.state_hash(rng_before), 'Fork unexpectedly consumed RNG')
        require(file_sha(source) == binding['state_sha256'], 'R24 source changed while loading')
        runtime.identity.update(experiment_id=EXPERIMENT, parent_id='b0', reference_parent='b0',
            train_seed=start_seed, arm=arm, run_id=f'{EXPERIMENT}/b0-{start_seed}-{arm}',
            step=0, policy_hash=binding['policy_sha256'], reference_hash=binding['reference_sha256'])
    except BaseException:
        _copy(parameters, reference)
        native.restore_rng(rng_before)
        runtime.identity.clear(); runtime.identity.update(identity_before)
        raise
    receipt = dict(schema='sr-satdose-fork-v1', experiment_id=EXPERIMENT, start_seed=start_seed, arm=arm,
        source_state_path=str(source), source_state_sha256=binding['state_sha256'],
        policy_sha256=binding['policy_sha256'], reference_sha256=binding['reference_sha256'],
        optimizer_sha256=native.state_hash(optimizer.state_dict()) if optimizer is not None else None,
        scheduler_sha256=native.state_hash(scheduler.state_dict()) if scheduler is not None else None,
        optimizer_created=create_optimizer, rng_sha256=native.state_hash(rng_before),
        trainable_tensors=496, source_optimizer_restored=False, model_calls=0, optimizer_steps=0)
    return SimpleNamespace(reference=reference, optimizer=optimizer, scheduler=scheduler, receipt=receipt)


def fork_runtime(runtime, config, start_seed, arm, native, *, checkpoint_path=None):
    """Create a new training fork with native fresh AdamW, without any update."""
    return _fork_runtime(runtime, config, start_seed, arm, native,
                         checkpoint_path=checkpoint_path, create_optimizer=True)


def fork_readonly(runtime, config, start_seed, native, *, checkpoint_path=None):
    """Load the R24 START endpoint without constructing an optimizer/scheduler.

    The same source-byte, original-reference, tensor, RNG and no-side-effect
    checks as a training fork apply. No optimizer factory is invoked. This is
    suitable for DRY and endpoint inference, not permission to train.
    """
    result = _fork_runtime(runtime, config, start_seed, 'T0', native,
                           checkpoint_path=checkpoint_path, create_optimizer=False)
    runtime.identity.update(arm='START', run_id=f'{EXPERIMENT}/b0-{start_seed}-START')
    result.receipt.update(arm='START', mode='READ_ONLY_START', optimizer_created=False)
    return result


def _json(path):
    def pairs(items):
        require(len(dict(items)) == len(items), 'Duplicate JSON key')
        return dict(items)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def restore_committed(runtime, optimizer, scheduler, reference, native, *, checkpoint_dir,
                      expected_commit_sha256, expected_identity):
    """Restore an explicit new checkpoint; never selects latest or an old R24.

    Driver must hold the exclusive path lock and verify the full transaction
    chain before calling. This loader authenticates the supplied COMMIT bytes,
    serialized state, tensor inventory, Adam/scheduler, full RNG and sampler.
    """
    import numpy as np
    import torch
    identity = expected_identity
    require(identity.get('experiment_id') == EXPERIMENT and identity.get('parent') == 'b0'
            and identity.get('arm') in ('T0', 'T1') and identity.get('train_seed') in (11031, 11032),
            'Restore requires a new SATDOSE path identity')
    require(runtime.identity.get('experiment_id') == EXPERIMENT
            and runtime.identity.get('arm') == identity['arm']
            and runtime.identity.get('train_seed') == identity['train_seed']
            and runtime.identity.get('reference_parent') == 'b0', 'Runtime path identity mismatch')
    directory = Path(checkpoint_dir)
    marker_path, state_path = directory / 'COMMIT.json', directory / 'state.pt'
    require(file_sha(marker_path) == expected_commit_sha256, 'COMMIT bytes differ')
    marker = _json(marker_path)
    step = marker['step']
    require(type(step) is int and 0 <= step <= 24 and directory.name == f'step-{step:03d}', 'Explicit step mismatch')
    require(marker['schema'] == SCHEMA and marker['identity'] == identity, 'Checkpoint identity mismatch')
    require(file_sha(state_path) == marker['state_sha256'], 'State bytes differ')
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    require(state['schema'] == SCHEMA and state['identity'] == identity and state['step'] == step,
            'Serialized identity mismatch')
    require(state['sampler_state'] == marker['sampler_state'], 'Sampler state mismatch')
    sampler = state['sampler_state']
    require(sampler.get('committed_step') == step and sampler.get('next_step') == step + 1
            and sampler.get('generated_responses') == step * 128, 'Sampler cursor mismatch')
    require(native.state_hash(reference) == state['reference_sha256'] == marker['reference_sha256'],
            'Original reference mismatch')
    parameters = _parameters(runtime.model, state['policy'], native)
    for key in ('policy', 'optimizer', 'scheduler'):
        require(native.state_hash(state[key]) == marker[key + '_sha256'] and native.finite_tree(state[key]),
                'Serialized ' + key + ' differs')
    native.validate_optimizer_state(state['optimizer'], state['scheduler'])
    require(all(g.get('foreach') is False for g in state['optimizer']['param_groups']), 'Saved foreach differs')
    rng = copy.deepcopy(state['rng'])
    n = rng['numpy']; rng['numpy'] = (n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4])
    require(native.state_hash(rng) == marker['rng_sha256'], 'Serialized RNG differs')
    before = dict(policy=native.trainable_state(runtime.model), optimizer=copy.deepcopy(optimizer.state_dict()),
                  scheduler=copy.deepcopy(scheduler.state_dict()), rng=native.capture_rng(), identity=copy.deepcopy(runtime.identity))
    require(set(rng) == set(before['rng']) and len(rng['cuda']) == len(before['rng']['cuda']), 'RNG device inventory differs')
    try:
        _copy(parameters, state['policy'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        _optimizer_check(native, runtime.model, optimizer, scheduler, step)
        require(native.state_hash(optimizer.state_dict()) == marker['optimizer_sha256']
                and native.state_hash(scheduler.state_dict()) == marker['scheduler_sha256'], 'Restored Adam/scheduler differs')
        native.restore_rng(rng)
        require(native.state_hash(native.capture_rng()) == marker['rng_sha256'], 'Restored RNG differs')
        require(native.state_hash(native.trainable_state(runtime.model)) == marker['policy_sha256'], 'Restored policy differs')
        require(file_sha(state_path) == marker['state_sha256'] and file_sha(marker_path) == expected_commit_sha256,
                'Checkpoint changed during restore')
        runtime.identity.update(step=step, policy_hash=marker['policy_sha256'], reference_hash=marker['reference_sha256'])
    except BaseException:
        _copy(parameters, before['policy'])
        optimizer.load_state_dict(before['optimizer']); scheduler.load_state_dict(before['scheduler'])
        native.restore_rng(before['rng'])
        runtime.identity.clear(); runtime.identity.update(before['identity'])
        raise
    return copy.deepcopy(marker)


def audit_sources(config, source_bindings, native, *, parent_file_pins):
    """Read CPU tensors only; never instantiate Qwen or alter source files.

    PEFT serialized adapter names omit the '.default' adapter segment. The
    key projection below must be a bijection onto the actual R24 policy keys;
    the resulting exact native tensor digest must equal the frozen b0 digest.
    This proves source tensors, not correctness of a future model loader.
    """
    import torch
    from safetensors.torch import load_file
    rows = []
    inventory = None
    for seed in config['starts']:
        b = _path(config, seed, 'T0')
        path = Path(b['original_state_path'])
        before = file_sha(path)
        require(before == b['state_sha256'], 'Original R24 file differs')
        state = torch.load(path, map_location='cpu', weights_only=True)
        require(state['schema'] == LEGACY_SCHEMA and state['step'] == 24
                and state['identity'].get('parent') == 'b0' and state['identity'].get('arm') == 'R'
                and state['identity'].get('train_seed') == seed, 'Original R24 identity differs')
        policy = state['policy']
        require(len(policy) == 496 and all('lora_' in n and torch.is_tensor(t)
                and t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for n,t in policy.items()),
                'Original R24 tensor inventory/dtype/nonfinite mismatch')
        current = {n: list(t.shape) for n,t in policy.items()}
        require(inventory is None or inventory == current, 'R24 start tensor inventories differ')
        inventory = current
        require(native.state_hash(policy) == b['policy_sha256'], 'Original R24 tensor SHA mismatch')
        require(state['reference_sha256'] == b['reference_sha256'], 'Original R24 reference SHA mismatch')
        require(file_sha(path) == before, 'Original R24 changed during read')
        rows.append(dict(start_seed=seed, source_state_path=str(path), state_sha256=before,
            policy_sha256=native.state_hash(policy), reference_sha256=state['reference_sha256'],
            trainable_tensors=len(policy), tensor_shapes=current))
        del state, policy
    adapter = Path(source_bindings['model']['parents']['b0'])
    require(set(parent_file_pins) == {'adapter_config.json','adapter_model.safetensors'}, 'Exact parent file pins required')
    for name, pin in parent_file_pins.items():
        require(file_sha(adapter/name) == pin, 'Original b0 file differs: ' + name)
    serialized = load_file(str(adapter/'adapter_model.safetensors'), device='cpu')
    parent = {}
    for name, shape in inventory.items():
        serialized_name = name.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.')
        require(serialized_name in serialized, 'Missing original parent tensor: ' + serialized_name)
        tensor = serialized[serialized_name]
        require(list(tensor.shape) == shape and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
                'Original parent tensor shape/dtype/nonfinite mismatch')
        parent[name] = tensor
    require(len(parent) == len(serialized) == 496, 'Parent mapping is not bijective')
    parent_sha = native.state_hash(parent)
    require(parent_sha == source_bindings['parent_tensor_sha256']
            and all(r['reference_sha256'] == parent_sha for r in rows), 'Original b0 tensor SHA mismatch')
    for name,pin in parent_file_pins.items():
        require(file_sha(adapter/name) == pin, 'Original b0 changed during read')
    return dict(schema='sr-satdose-source-tensor-audit-v1', status='CPU_SOURCE_TENSORS_VERIFIED_NOT_GPU_ADMISSION',
        starts=rows, parent_path=str(adapter), parent_file_sha256=parent_file_pins,
        parent_tensor_sha256=parent_sha, model_calls=0, optimizer_steps=0,
        limitations=['No Qwen runtime instantiated; model loading order remains a separate GPU check.'])


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Read-only CPU R24 and original b0 tensor audit')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-bindings', type=Path, required=True)
    parser.add_argument('--native-src', type=Path, required=True)
    parser.add_argument('--native-pins', type=Path, required=True)
    parser.add_argument('--parent-file-pins', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refuse to overwrite audit evidence')
    native = load_native_ops(args.native_src, _json(args.native_pins))
    result = audit_sources(_json(args.config), _json(args.source_bindings), native,
                           parent_file_pins=_json(args.parent_file_pins))
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(result['status'])


if __name__ == '__main__':
    main()
