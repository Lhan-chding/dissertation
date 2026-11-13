"""SHA-bound endpoint load/audit only; no generation, optimizer or MAIN release.

START uses the actual frozen original-b0 -> R24 read-only loader. Trained
endpoints additionally validate the trusted new COMMIT/full transaction chain
and safely decode state, then copy only its 496 FP32 policy tensors. Training
Adam/scheduler/RNG state is validated as evidence but is never restored.
GPU construction/equivalence is not established by the CPU fixture tests.
"""
from __future__ import annotations

import copy
from pathlib import Path
import re
import sys

from runtime import native_runtime as frozen, fork_loader
from runtime.backend import FrozenRegistry, file_sha, object_sha, read_json, write_once
from . import training_state as state, training_transaction as transactions

require = state.require
CHECKPOINT_STEPS = (2, 8, 16, 24)


def verify_sources(registry, source_lock_path, execution_sources, expected_execution_sources_sha256):
    """Authenticate this separate loader inventory; never edit frozen runtime."""
    require(object_sha(execution_sources) == expected_execution_sources_sha256,
            'Untrusted evaluation source inventory SHA')
    expected = {'evaluation_loader': Path(__file__).resolve(),
                'training_state': Path(state.__file__).resolve(),
                'training_transaction': Path(transactions.__file__).resolve()}
    require(set(execution_sources) == set(expected), 'Evaluation source inventory differs')
    for name, path in expected.items():
        ref = execution_sources[name]
        require(set(ref) == {'path', 'sha256'} and Path(ref['path']).resolve(strict=True) == path and
                not Path(ref['path']).is_symlink() and file_sha(path) == ref['sha256'],
                'Unbound actual evaluation source: ' + name)
    directory = Path(frozen.__file__).resolve().parent
    freeze = read_json(registry.freeze_path)
    actual = {p.name:file_sha(p) for p in directory.glob('*.py') if p.is_file()}
    require(actual == freeze['runtime_sources'] and
            set(actual.values()) <= {x['sha256'] for x in freeze['artifacts'].values()},
            'Frozen runtime source inventory differs')
    for name, module in tuple(sys.modules.items()):
        if name.startswith('runtime.'):
            expected_path = directory / (name.removeprefix('runtime.').replace('.', '/') + '.py')
            require(getattr(module, '__file__', None) and Path(module.__file__).resolve() == expected_path,
                    'Shadowed frozen runtime import: ' + name)
    return frozen.verify_source_lock(registry, source_lock_path)


def registered_endpoint(registry, start_seed, arm, step):
    require(type(start_seed) is int and start_seed in (11031,11032) and
            type(step) is int and ((arm == 'START' and step == 0) or
            (arm in ('T0','T1') and step in CHECKPOINT_STEPS)), 'Unregistered endpoint selection')
    require(registry.config['experiment_id'] == state.EXPERIMENT and registry.config['parent'] == 'b0',
            'Wrong experiment/base parent')
    matches = [row for row in registry.requests.values()
               if row.get('start_seed') == start_seed and row.get('arm') == arm and row.get('step') == step
               and row.get('scope') in ('DEV','MAIN','DRY','PREFIX_CALIB')]
    require(matches, 'Endpoint absent from frozen request registration')
    # Recheck authoritative registry/freeze/image rather than trusting a mutable row.
    for row in matches:
        require(registry.request(row['request_id']) == row, 'Registered endpoint request changed')
    return dict(experiment_id=state.EXPERIMENT, start_seed=start_seed, arm=arm, step=step)


def _clean_runtime(handle, expected_policy):
    runtime, native = handle.runtime, handle.native_ops
    require(handle.fork.optimizer is None and handle.fork.scheduler is None,
            'Evaluation must not construct/restore optimizer or scheduler')
    require(getattr(runtime, '_satdose_active', None) is None and
            all(p.grad is None for p in runtime.model.parameters()), 'Dirty/active evaluation runtime')
    require(native.state_hash(native.trainable_state(runtime.model)) == expected_policy,
            'Dirty evaluation policy')
    source = handle.registry.config['start_bindings'][str(handle.start_seed)]
    require(native.state_hash(handle.fork.reference) == source['reference_sha256'],
            'Original b0 reference changed')
    fork_loader._parameters(runtime.model, native.trainable_state(runtime.model), native)


def _store_identity(handle, store, expected_identity):
    require(type(store) is transactions.TransactionStore and store.root == handle.registry.root and
            store.freeze_sha == handle.registry.freeze_sha256 and store.identity == expected_identity,
            'Foreign transaction store/identity/freeze')
    require(expected_identity['train_seed'] == handle.start_seed and expected_identity['parent'] == 'b0' and
            expected_identity['source_binding'] == handle.registry.config['start_bindings'][str(handle.start_seed)],
            'Checkpoint starting policy/reference source differs')
    state._validate_identity(expected_identity)


def load_committed_policy(handle, store, *, step, expected_commit_sha256, expected_identity):
    """Policy-only load into a newly audited R24 handle; no GPU claim/MAIN permission.

    The caller provides a trusted COMMIT digest and expected complete identity.
    This lower-level API still requires a real TransactionStore and runs its
    full chain/UNKNOWN checks, then the real weights_only state codec twice.
    """
    import torch
    require(type(step) is int and step in CHECKPOINT_STEPS, 'Explicit registered checkpoint step required')
    require(re.fullmatch('[0-9a-f]{64}', expected_commit_sha256 or '') is not None, 'Trusted COMMIT SHA required')
    _store_identity(handle, store, expected_identity)
    registered_endpoint(handle.registry, handle.start_seed, expected_identity['arm'], step)
    source = handle.registry.config['start_bindings'][str(handle.start_seed)]
    _clean_runtime(handle, source['policy_sha256'])
    require(handle.runtime.identity.get('step') == 0 and handle.runtime.identity.get('arm') == 'START',
            'Policy-only restore requires fresh R24 START runtime')
    native = handle.native_ops
    before_rng = native.capture_rng()
    before_rng_sha = native.state_hash(before_rng)
    before_policy = native.trainable_state(handle.runtime.model)
    before_identity = copy.deepcopy(handle.runtime.identity)
    parameters = fork_loader._parameters(handle.runtime.model, before_policy, native)
    copied = False
    try:
        marker = store.inspect_committed(step, expected_commit_sha256)
        require(marker['step'] == step and marker['identity'] == expected_identity, 'Inspected endpoint differs')
        directory = store.path/f'step-{step:03d}'
        require(state.validate_checkpoint(directory, expected_commit_sha256, expected_identity, native) == marker,
                'COMMIT/state codec disagreement')
        payload = torch.load(directory/'state.pt', map_location='cpu', weights_only=True)
        state._validate_state(payload, marker, expected_identity, native)
        policy = payload['policy']
        fork_loader._parameters(handle.runtime.model, policy, native)
        require(file_sha(directory/'state.pt') == marker['state_sha256'] and
                file_sha(directory/'COMMIT.json') == expected_commit_sha256, 'Checkpoint changed during decode')
        require(native.state_hash(native.capture_rng()) == before_rng_sha, 'Policy decoding consumed caller RNG')
        copied = True
        fork_loader._copy(parameters, policy)
        require(native.state_hash(native.trainable_state(handle.runtime.model)) == marker['policy_sha256'],
                'Loaded evaluation policy SHA differs')
        require(native.state_hash(handle.fork.reference) == source['reference_sha256'], 'Original reference changed')
        require(store.inspect_committed(step, expected_commit_sha256) == marker,
                'Transaction chain changed during load')
        require(file_sha(directory/'state.pt') == marker['state_sha256'], 'Original state file changed')
        require(native.state_hash(native.capture_rng()) == before_rng_sha, 'Policy-only load consumed caller RNG')
        handle.runtime.identity.update(arm=expected_identity['arm'], step=step,
            run_id=expected_identity['run_id'], policy_hash=marker['policy_sha256'],
            reference_hash=marker['reference_sha256'], execution_mode='EVALUATION_LOAD_ONLY')
        return dict(status='POLICY_ONLY_LOADED_NOT_EVALUATION_ADMISSION', step=step,
            policy_sha256=marker['policy_sha256'], reference_sha256=marker['reference_sha256'],
            commit_sha256=expected_commit_sha256, state_sha256=marker['state_sha256'],
            checkpoint_directory=str(directory), caller_rng_sha256=before_rng_sha,
            training_rng_restored=False, optimizer_constructed=False, optimizer_restored=False,
            scheduler_constructed=False, scheduler_restored=False, model_calls=0)
    except BaseException:
        if copied:
            fork_loader._copy(parameters, before_policy)
            handle.runtime.identity.clear(); handle.runtime.identity.update(before_identity)
        raise
    finally:
        # Preserve caller state, NEVER checkpoint['rng']; normally no setter call is needed.
        if native.state_hash(native.capture_rng()) != before_rng_sha:
            native.restore_rng(before_rng)
        require(native.state_hash(native.capture_rng()) == before_rng_sha, 'Caller RNG preservation failed')


class EvaluationLoadHandle:
    """Load-only audit surface, intentionally no generation/teacher/MAIN methods."""
    def __init__(self, underlying, selection, policy_sha256, source_files, receipt):
        self._underlying = underlying
        self._selection = copy.deepcopy(selection)
        self._policy_sha256 = policy_sha256
        self._source_files = copy.deepcopy(source_files)
        self.receipt = copy.deepcopy(receipt)
        self._closed = False
        # Existing frozen DRY bind_request rejects closed runtime. Do not expose
        # this object through a GenerationBridge or call frozen handle.close for trained policy.
        underlying.runtime.identity['execution_mode'] = 'EVALUATION_LOAD_ONLY'
        underlying.runtime._satdose_closed = True

    def audit(self):
        require(not self._closed, 'Evaluation loader already closed')
        handle = self._underlying
        require(handle.runtime._satdose_closed is True and
                handle.runtime.identity.get('execution_mode') == 'EVALUATION_LOAD_ONLY' and
                not getattr(handle.runtime, '_satdose_accounting', []),
                'Load-only runtime was reopened or performed model work')
        _clean_runtime(handle, self._policy_sha256)
        require(handle.runtime.identity['step'] == self._selection['step'] and
                handle.runtime.identity['arm'] == self._selection['arm'], 'Loaded endpoint cursor changed')
        for path, sha in self._source_files.items():
            require(file_sha(path) == sha, 'Original checkpoint/parent source changed')
        return dict(self._selection, policy_sha256=self._policy_sha256,
            reference_sha256=handle.native_ops.state_hash(handle.fork.reference),
            model_calls=0, optimizer_constructed=False, optimizer_steps=0, training_rng_restored=False,
            permission='LOAD_AUDIT_ONLY_NO_GENERATION_NO_MAIN_RELEASE')

    def close(self):
        require(not self._closed, 'Evaluation loader already closed')
        runtime = self._underlying.runtime
        try:
            result = self.audit()
            return dict(result, closed=True)
        finally:
            runtime.model.zero_grad(set_to_none=True)
            runtime._vision_hook.remove()
            self._closed = True


def build_evaluation_loader(registry, source_lock_path, *, start_seed, arm, step, output_dir,
                            execution_sources, expected_execution_sources_sha256,
                            commit_store=None, expected_identity=None, expected_commit_sha256=None):
    """Actual GPU construction via frozen loader, then optional policy-only load.

    No optimizer factory/restore, no training RNG restore, no sampling, no MAIN
    authorization. Constructor captures/restores caller RNG around the original
    model initialization; policy-only restore itself does not use training RNG.
    """
    require(isinstance(registry, FrozenRegistry), 'Real frozen request registry required')
    selection = registered_endpoint(registry, start_seed, arm, step)
    output = Path(output_dir).resolve(strict=True)
    require(output.is_dir() and output.is_relative_to(registry.root) and not any(output.iterdir()),
            'New empty isolated evaluation load audit required')
    if arm == 'START':
        require(all(x is None for x in (commit_store, expected_identity, expected_commit_sha256)),
                'START may not substitute a trained checkpoint')
    else:
        require(all(x is not None for x in (commit_store, expected_identity, expected_commit_sha256)) and
                expected_identity['arm'] == arm and expected_identity['train_seed'] == start_seed,
                'Explicit trained endpoint identity/COMMIT required')
        require(expected_identity.get('design_freeze_sha256') == registry.freeze_sha256 and
                expected_identity.get('config_sha256') == read_json(registry.freeze_path)['config_sha256'] and
                re.fullmatch('[0-9a-f]{64}', expected_identity.get('execution_sources_sha256', '')) is not None,
                'Trained endpoint lacks complete scientific/execution identity')
    lock, native_root, _, _, parent = verify_sources(registry, source_lock_path, execution_sources,
                                                     expected_execution_sources_sha256)
    native = fork_loader.load_native_ops(native_root, lock['native_files'])
    source = registry.config['start_bindings'][str(start_seed)]
    if arm != 'START':
        freeze = read_json(registry.freeze_path)
        require(expected_identity.get('source_sha256') == object_sha(lock['native_files']) and
                expected_identity.get('manifest_sha256') == freeze['artifacts']['train_manifest']['sha256'] and
                expected_identity.get('plan_sha256') == freeze['artifacts']['training_groups']['sha256'] and
                expected_identity.get('run_id') == f'{state.EXPERIMENT}/b0-{start_seed}-{arm}',
                'Trained endpoint source/plan/run identity differs')
        # Reject a missing/UNKNOWN/foreign chain before expensive GPU model construction.
        require(type(commit_store) is transactions.TransactionStore and commit_store.root == registry.root and
                commit_store.freeze_sha == registry.freeze_sha256 and commit_store.identity == expected_identity and
                expected_identity['source_binding'] == source, 'Foreign trained transaction identity')
        preflight = commit_store.inspect_committed(step, expected_commit_sha256)
        require(preflight['step'] == step and preflight['identity'] == expected_identity,
                'Trained checkpoint preflight differs')
        state.validate_checkpoint(commit_store.path/f'step-{step:03d}', expected_commit_sha256,
                                  expected_identity, native)
    original_files = {source['original_state_path']:source['state_sha256']}
    original_files.update({str(parent/name):sha for name,sha in lock['parent_files'].items()})
    for path, sha in original_files.items():
        require(file_sha(path) == sha, 'Original endpoint source bytes differ')
    before_rng = native.capture_rng(); before_rng_sha = native.state_hash(before_rng)
    underlying = result = None
    try:
        underlying = frozen.build_runtime(registry, source_lock_path, start_seed=start_seed, output_dir=output)
        _clean_runtime(underlying, source['policy_sha256'])
        if arm == 'START':
            loaded = dict(status='R24_START_LOADED_NOT_EVALUATION_ADMISSION',
                policy_sha256=source['policy_sha256'], reference_sha256=source['reference_sha256'],
                original_state_path=source['original_state_path'], original_state_sha256=source['state_sha256'],
                training_rng_restored=False, optimizer_constructed=False, model_calls=0)
        else:
            loaded = load_committed_policy(underlying, commit_store, step=step,
                expected_commit_sha256=expected_commit_sha256, expected_identity=expected_identity)
        receipt = dict(schema='sr-satdose-evaluation-loader-v1', selection=selection, loaded=loaded,
            design_freeze_sha256=registry.freeze_sha256, source_lock_sha256=file_sha(source_lock_path),
            evaluation_sources_sha256=expected_execution_sources_sha256,
            original_files=original_files, caller_rng_sha256=before_rng_sha,
            permission='LOAD_AUDIT_ONLY_NO_GENERATION_NO_MAIN_RELEASE', gpu_acceptance=False)
        result = EvaluationLoadHandle(underlying, selection, loaded['policy_sha256'], original_files, receipt)
        result.audit()
        # Source and caller state are checked before writing a successful receipt.
        if native.state_hash(native.capture_rng()) != before_rng_sha:
            native.restore_rng(before_rng)
        require(native.state_hash(native.capture_rng()) == before_rng_sha, 'Caller RNG preservation failed')
        write_once(output/'EVALUATION_LOAD.json', receipt)
        return result
    except BaseException:
        if underlying is not None:
            underlying.runtime.model.zero_grad(set_to_none=True)
            underlying.runtime._vision_hook.remove()
            underlying.runtime.identity['execution_mode'] = 'EVALUATION_LOAD_ONLY'
            underlying.runtime._satdose_closed = True
        raise
    finally:
        if native.state_hash(native.capture_rng()) != before_rng_sha:
            native.restore_rng(before_rng)
        require(native.state_hash(native.capture_rng()) == before_rng_sha, 'Caller RNG preservation failed')
