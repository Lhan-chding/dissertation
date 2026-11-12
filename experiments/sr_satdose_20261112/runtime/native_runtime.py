"""Construct the audited two-GPU SATDOSE runtime after a real design freeze.

No CLI or job submission is supplied. ``build_runtime`` loads the original b0,
then the SHA-verified R24 fork. It does not sample or update. The caller must use
GenerationBridge's durable attempt ledger. Forward replay only accepts original
persisted captures; the native capacity48 hook is used for each gradient row.
CPU tests do not certify GPU equivalence, full-batch capacity, or admission.
"""
from __future__ import annotations

import copy
import importlib
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from .backend import FrozenRegistry, PUBLIC, file_sha, object_sha, read_json, require
from .fork_loader import load_native_ops, fork_readonly


def verify_source_lock(registry, source_lock_path):
    """Bind full native inventory, old bridge/storage code and model bytes."""
    path = Path(source_lock_path).resolve(strict=True)
    freeze = read_json(registry.freeze_path)
    require(any((registry.root / x['path']).resolve() == path and x['sha256'] == file_sha(path)
                for x in freeze['artifacts'].values()), 'Source lock absent from design freeze')
    lock = read_json(path)
    native = Path(lock['native_src']).resolve(strict=True)
    legacy = Path(lock['legacy_root']).resolve(strict=True)
    actual = {str(p.relative_to(native)) for p in native.rglob('*.py') if p.is_file()}
    require(actual == set(lock['native_files']), 'Native source inventory incomplete')
    required = {'gpu_production_v2/backend.py', 'diagnostics/gpu_capacity_v2.py',
                'runtime_contract.py'}
    require(required <= set(lock['legacy_files']), 'Legacy bridge/storage sources missing')
    for root, key in ((native, 'native_files'), (legacy, 'legacy_files')):
        for rel, sha in lock[key].items():
            p = Path(rel)
            require(not p.is_absolute() and '..' not in p.parts, 'Source escapes root')
            f = root / p
            require(not f.is_symlink() and f.resolve().is_relative_to(root)
                    and file_sha(f) == sha, 'Bound source changed: ' + rel)
    require(lock['snapshot'] == registry.config['model_snapshot'], 'Wrong model snapshot')
    model = Path(lock['model_path']).resolve(strict=True)
    require(model.name == lock['snapshot'], 'Wrong model directory')
    files = {str(p.relative_to(model)) for p in model.rglob('*') if p.is_file()}
    require(files == set(lock['model_files']) and files, 'Model inventory incomplete')
    for rel, sha in lock['model_files'].items():
        require(file_sha(model / rel) == sha, 'Model bytes changed')
    parent = Path(lock['parent_path']).resolve(strict=True)
    require(set(lock['parent_files']) == {'adapter_config.json', 'adapter_model.safetensors'},
            'Original b0 adapter inventory required')
    for rel, sha in lock['parent_files'].items():
        require(file_sha(parent / rel) == sha, 'Original parent changed')
    return lock, native, legacy, model, parent


def _import(name, root):
    mod = importlib.import_module(name)
    require(Path(mod.__file__).resolve() == root / (name.replace('.', '/') + '.py'),
            'Shadowed module: ' + name)
    return mod


class RegisteredRuntimeMixin:
    """Control-plane checks around native preparation and unchanged forward math."""
    def bind_request(self, request):
        require(not getattr(self, '_satdose_closed', False), 'Runtime already closed')
        require(self._satdose_active is None, 'Nested runtime request')
        require(self._satdose_registry.request(request['request_id']) == request, 'Unregistered request')
        require(request.get('scope') in ('DRY', 'PREFIX_CALIB') and
                request.get('arm') == 'START' and request.get('step') == 0, 'Runtime only permits R24 calibration')
        require(request.get('start_seed', request.get('train_seed')) == self._satdose_start,
                'Request belongs to another starting model')
        self._satdose_active = copy.deepcopy(request)

    def unbind_request(self):
        require(self._satdose_active is not None, 'No active runtime request')
        self._satdose_active = None

    def _set_recipe(self):
        super()._set_recipe()
        for key, value in self._satdose_native.generation_recipe().items():
            setattr(self.generation_config, key, value)
        self.identity.update(runtime_recipe='SR-SATDOSE-native-full-sequence-1536',
                             generation_config_expanded=self.generation_config.to_dict())

    def prepare_input(self, model_input, run_root, *, protocol='structured_v4'):
        request = self._satdose_active
        require(request is not None, 'Unbound runtime request')
        require(protocol in ('structured', 'structured_v4') and
                Path(run_root).resolve() == self._satdose_registry.root, 'Unregistered input root/protocol')
        require(model_input == {k: request[k] for k in PUBLIC}, 'Public input differs from frozen request')
        image = self._satdose_registry.root / request['image_file']
        require(file_sha(image) == request['image_sha256'], 'Frozen image changed')
        # Include prefix: different prefix views can have identical image/prompt.
        key = object_sha([model_input, request['assistant_prefill']])
        if getattr(self, '_satdose_prepared_key', None) != key:
            self._srf5_prepared = self._satdose_legacy.prepare_public_input(
                self, request, self._satdose_registry.root)
            self._satdose_prepared_key = key
        return self._srf5_prepared

    def reserve(self, kind, count, **metadata):
        require(kind in ('completion_attempts', 'generated_tokens', 'extra_forward_sequences',
                         'diagnostic_gradient_calls'), 'Unknown native accounting kind')
        require(type(count) is int and count >= 0, 'Invalid accounting count')
        event = dict(kind=kind, count=count, metadata=metadata)
        self._satdose_accounting.append(event)
        with (self._satdose_output / 'accounting.jsonl').open('a') as stream:
            import os
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())

    def _validate_record_revision(self, record):
        require(record.get('protocol') in ('structured', 'structured_v4') and
                record.get('protocol_revision') == 'v2' and
                record.get('assistant_prefill') == self._satdose_active['assistant_prefill'],
                'Original record protocol/prefix differs')

    def _validate_prepared_revision(self, prepared):
        route = prepared.get('routing', {})
        require(route.get('protocol') in ('structured', 'structured_v4') and
                route.get('protocol_revision') == 'v2' and
                route.get('assistant_prefill') == self._satdose_active['assistant_prefill'],
                'Prepared protocol/prefix differs')

    def _batch_forward(self, prepared, token_rows, *, purpose, grad):
        # Exactly the old production full-sequence concatenation and mRoPE path.
        self._validate_prepared_revision(prepared)
        require(self._satdose_active.get('scope') == 'DRY' and
                purpose in ('training_policy', 'training_reference'), 'Forward replay outside DRY')
        require(len(token_rows) == 1 and 1 <= len(token_rows[0]) <= 1536, 'One original sequence required')
        torch = self.torch
        inputs = self._satdose_repeat_prompt(prepared['inputs'], len(token_rows))
        prompt_len, longest = inputs['input_ids'].shape[-1], max(map(len, token_rows))
        pad = self.processor.tokenizer.pad_token_id
        if pad is None:
            pad = self.eos_ids[0]
        completion = inputs['input_ids'].new_full((len(token_rows), longest), pad)
        mask = torch.zeros_like(completion)
        for index, tokens in enumerate(token_rows):
            completion[index, :len(tokens)] = completion.new_tensor(tokens)
            mask[index, :len(tokens)] = 1
        inputs['input_ids'] = torch.cat([inputs['input_ids'], completion], -1)
        inputs['attention_mask'] = torch.cat([inputs.get('attention_mask', mask.new_ones((len(token_rows), prompt_len))), mask], -1)
        inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(completion)], -1)
        return self._native_forward(inputs, completion_start=prompt_len,
                                    lengths=list(map(len, token_rows)), purpose=purpose, grad=grad)


class RuntimeHandle:
    def __init__(self, runtime, fork, native_ops, storage, registry, start_seed, output):
        self.runtime, self.fork, self.native_ops, self.storage = runtime, fork, native_ops, storage
        self.registry, self.start_seed, self.output = registry, start_seed, output
        self.storage_forwards = []

    def endpoint(self):
        expected = self.registry.config['start_bindings'][str(self.start_seed)]
        policy = self.native_ops.state_hash(self.native_ops.trainable_state(self.runtime.model))
        reference = self.native_ops.state_hash(self.fork.reference)
        require(policy == expected['policy_sha256'] and reference == expected['reference_sha256'],
                'Read-only R24/reference identity changed')
        return dict(experiment_id=self.registry.config['experiment_id'], start_seed=self.start_seed,
                    arm='START', step=0,
                    policy_sha256=policy, reference_sha256=reference)

    def forward_original(self, row):
        """Return current/sampler/reference vectors, with native capacity48 hooks.

        Caller runs backward before the next row. This method never steps Adam.
        Only original DRY captures from this R24 are admissible.
        """
        runtime = self.runtime
        request = self.registry.request(row['request_id'])
        require(request['scope'] == 'DRY' and request['assistant_prefill'] == '{', 'Only DRY gradient replay')
        endpoint = self.endpoint()
        require(row['policy_sha256'] == endpoint['policy_sha256'] and
                row['reference_sha256'] == endpoint['reference_sha256'], 'Stale replay policy/reference')
        capture_path = Path(row['native_capture_path']).resolve(strict=True)
        require(capture_path.is_relative_to(self.output) and file_sha(capture_path) == row['native_capture_sha256'],
                'Original capture changed or outside output')
        expected_name = hashlib.sha256(row['request_id'].encode()).hexdigest() + '.CAPTURE.json'
        require(capture_path.name == expected_name, 'Capture belongs to another request')
        begin = read_json(capture_path.with_name(expected_name.replace('.CAPTURE.json', '.BEGIN.json')))
        require(begin['request_id'] == row['request_id'] and begin['endpoint'] == endpoint
                and begin['design_freeze_sha256'] == self.registry.freeze_sha256, 'Original attempt identity differs')
        saved = read_json(capture_path)
        for key in ('tokens', 'sampler_logprobs', 'prompt_tensor_hash', 'model_input', 'seed'):
            require(row[key] == saved[key], 'Changed original trajectory: ' + key)
        require(row['seed'] == request['seed'] and row['model_input'] == {k: request[k] for k in PUBLIC},
                'Replay differs from frozen request')
        require('prepared' not in row, 'Untrusted prepared tensor injection')
        record = dict(row, run_root=str(self.registry.root), root=str(self.registry.root),
                      protocol='structured_v4', protocol_revision='v2', assistant_prefill='{')
        runtime.bind_request(request)
        try:
            reference = runtime.batch_reference_forward([record], self.fork.reference)[0].detach()
            hook = self.storage.GPUSavedTensors((*runtime.model.parameters(), *runtime.model.buffers()),
                                               row_id=row['request_id'])
            entry = dict(row_id=row['request_id'], stats=hook.stats)
            self.storage_forwards.append(entry)
            with hook.context():
                current = runtime.batch_sequence_forward([record], purpose='training_policy', grad=True)[0]
            entry['forward_returned'] = True
            sampler = current.new_tensor(row['sampler_logprobs'])
            require(current.shape == sampler.shape == reference.shape, 'Original token-vector alignment differs')
            return current, sampler, reference
        finally:
            runtime.unbind_request()

    def close(self):
        expected = self.registry.config['start_bindings'][str(self.start_seed)]
        endpoint = self.endpoint()
        require(endpoint['policy_sha256'] == expected['policy_sha256'] and
                endpoint['reference_sha256'] == expected['reference_sha256'], 'Read-only runtime changed')
        self.runtime.model.zero_grad(set_to_none=True)
        require(not getattr(self.runtime, '_satdose_closed', False), 'Runtime already closed')
        self.runtime._vision_hook.remove()
        self.runtime._satdose_closed = True
        return dict(endpoint, optimizer_steps=0, parameter_updates=0, closed=True)

    def verify_storage_after_backward(self):
        for entry in self.storage_forwards:
            require(entry.get('forward_returned') is True, 'Incomplete gradient forward')
            self.storage.validate_snapshot_stats(entry['stats'])


def build_runtime(registry, source_lock_path, *, start_seed, output_dir, checkpoint_path=None):
    """Construct original b0 then fork R24; requires exactly two audited GPUs."""
    require(isinstance(registry, FrozenRegistry), 'Real verified FrozenRegistry required')
    require(start_seed in (11031, 11032), 'Unknown start')
    output = Path(output_dir).resolve(strict=True)
    require(output.is_dir() and output.is_relative_to(registry.root), 'Output outside new experiment')
    lock, native_root, legacy_root, model, parent = verify_source_lock(registry, source_lock_path)
    sys.path[:0] = [str(native_root), str(legacy_root)]
    try:
        native = _import('sr_f6.runtime', native_root)
        base = _import('sr_f5.runtime', native_root).Runtime
        configure = _import('mm_dev.runtime', native_root).configure_audited_backend
        repeat = _import('sr_f12.runtime', native_root)._repeat_prompt
        legacy = _import('gpu_production_v2.backend', legacy_root)
        storage = _import('diagnostics.gpu_capacity_v2', legacy_root)
        native_ops = load_native_ops(native_root, lock['native_files'])
    except BaseException:
        del sys.path[:2]
        raise
    # Native methods use lazy absolute imports during model construction and replay.
    # Keep only these audited roots available for the lifetime of the runtime.
    bound_modules = {str((native_root / rel).resolve()) for rel in lock['native_files']}
    namespaces = {Path(rel).parts[0].removesuffix('.py') for rel in lock['native_files']}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, '__file__', None)
        if source and name.split('.')[0] in namespaces:
            require(str(Path(source).resolve()) in bound_modules, 'Shadowed native dependency: ' + name)
    import torch
    configure()
    storage.validate_gpu_pair()
    require(storage.CAP_BYTES == 48 * 1024**3, 'Unregistered saved-tensor capacity')
    class SatdoseRuntime(RegisteredRuntimeMixin, base):
        _satdose_native = native
        _satdose_legacy = legacy
        _satdose_repeat_prompt = staticmethod(repeat)
        _satdose_registry = registry
        _satdose_start = start_seed
        _satdose_output = output
        _satdose_active = None
    SatdoseRuntime._satdose_accounting = []
    native.seed_all(native.seed_for(2026110301, 'load', 'b0'))
    runtime = SatdoseRuntime(str(model), adapter_path=str(parent), output_root=output,
                             protocol_revision='v2', account=None)
    native.configure_parent_training(runtime, parent_adapter_config=read_json(parent / 'adapter_config.json'))
    original_dtype = runtime.identity['dtype']
    runtime.model = runtime.model.float()
    runtime.model.eval()
    runtime._srf5_numerics = dict(native.NUMERICS_V2)
    runtime.identity.update(dtype='float32', input_float_dtype='float32', probability_dtype='float32',
        model_cast_from_dtype=original_dtype, forward_parameter_dtypes=['float32'],
        forward_float_buffer_dtypes=['float32'], numerics=dict(native.NUMERICS_V2), step=0,
        manifest_sha256=registry.freeze_sha256, source_sha256=object_sha(lock['native_files']))
    fork = fork_readonly(runtime, registry.config, start_seed, native_ops, checkpoint_path=checkpoint_path)
    runtime.identity.update(arm='START', run_id=f"{registry.config['experiment_id']}/start-{start_seed}",
        execution_mode='DRY', run_root=str(registry.root), audit_root=str(output))
    return RuntimeHandle(runtime, fork, native_ops, storage, registry, start_seed, output)
