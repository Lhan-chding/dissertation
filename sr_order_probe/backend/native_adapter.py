"""Read-only SR-ORDER adapter over the unchanged F6 native sampling function.

The native function's code object is reused verbatim. A private globals mapping
binds four *serialization* helpers to the request's forced prefix; it does not
modify historical modules, logits, multinomial sampling, or model generation.
GPU validation is required separately. Importing this module never loads a model.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
import sys
import types
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE / 'src') not in sys.path:
    sys.path.insert(0, str(PACKAGE / 'src'))
from order_probe.common import file_sha, text_sha
from order_probe.measurement import first_object_end


def validate_request(request, config):
    """Reject private sidecars before projecting the three native model fields."""
    if any(k in request for k in ('world', 'gold', 'rename', 'boundary_indices', 'query')):
        raise PermissionError('Private task information at model boundary')
    if request.get('sampling') != config['generation']:
        raise PermissionError('Frozen generation configuration changed')
    for field, hash_field in [('prompt', 'prompt_sha256'), ('assistant_prefill', 'prefill_sha256')]:
        if not isinstance(request.get(field), str) or text_sha(request[field]) != request.get(hash_field):
            raise PermissionError('Input text digest mismatch: ' + field)
    prefix = request['assistant_prefill']
    if not prefix.startswith('{') or first_object_end(prefix) is not None:
        raise PermissionError('Prefix must be an unfinished top-level JSON object')
    if type(request.get('seed')) is not int or not 0 <= request['seed'] < 2**63:
        raise PermissionError('Invalid registered request seed')
    return {k: request[k] for k in ('prompt', 'image_file', 'image_sha256')}


def prefix_helpers(prefill, decode, native_stop, scan=first_object_end):
    """Bind serialization without patching native global state."""
    def get_prefill(protocol, revision):
        if protocol != 'structured' or revision != 'v2':
            raise PermissionError('Only the bound structured/v2 route is allowed')
        return prefill

    def boundary(tokenizer, tokens, protocol, *, revision='v2'):
        get_prefill(protocol, revision)
        for n in range(1, len(tokens) + 1):
            cut = scan(prefill + decode(tokenizer, tokens[:n]))
            if cut is not None:
                return n, cut
        return None, None

    def decoded(text, protocol, *, revision='v2'):
        get_prefill(protocol, revision)
        full = prefill + text
        cut = scan(full)
        error = 'unbalanced' if cut is None else ('trailing_after_close' if full[cut:].strip() else None)
        return dict(generated_text=text, assistant_prefill=prefill, prefill=prefill,
                    full_decoded_text=full, full_text=full, raw_text=full,
                    boundary_char=cut, balanced_cut_char=cut, final_tag_cut_char=None,
                    format_protocol_error=error)

    class PrefixStop(native_stop):
        def __init__(self, tokenizer, prompt_len, rows, protocol, *, revision='v2'):
            if rows != 1:
                raise PermissionError('Probe requires single-request batch')
            super().__init__(tokenizer, prompt_len, rows, protocol, revision=revision)
            # The inherited scanner concatenates this full prefix before decoding.
            self.prefill = get_prefill(protocol, revision)

    return dict(_prefill=get_prefill, first_boundary_token=boundary,
                decoded_protocol=decoded, BatchedProtocolStop=PrefixStop)


def bind_native_kernel(native_function, helpers):
    if set(helpers) != {'_prefill', 'first_boundary_token', 'decoded_protocol', 'BatchedProtocolStop'}:
        raise PermissionError('Only four serialization helpers may be rebound')
    namespace = dict(native_function.__globals__)
    namespace.update(helpers)
    bound = types.FunctionType(native_function.__code__, namespace,
                               native_function.__name__, native_function.__defaults__,
                               native_function.__closure__)
    bound.__kwdefaults__ = copy.deepcopy(native_function.__kwdefaults__)
    if bound.__code__ is not native_function.__code__:
        raise RuntimeError('Native kernel code was replaced')
    return bound


def import_native(binding):
    root = Path(binding['native_code_root']).resolve(strict=True)
    source = root / 'src'
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    native = importlib.import_module('sr_f6.runtime')
    if Path(native.__file__).resolve() != source / 'sr_f6/runtime.py':
        raise PermissionError('A different native runtime is already imported')
    expected = binding.get('native_runtime_sha256')
    if expected is not None and file_sha(native.__file__) != expected:
        raise PermissionError('Native runtime source hash changed')
    return native


def prepare_public_input(runtime, request, inputs_dir):
    """Full native processor call, same open assistant message, actual offsets.

    Available for processor-only visual checks; no model forward is performed.
    ``runtime`` must be a native SRF3CERuntime/FP32Runtime processor instance.
    """
    from sr_f1.runtime import SRRuntime
    safe = {k: request[k] for k in ('prompt', 'image_file', 'image_sha256')}
    prefix = request['assistant_prefill']
    prepared = SRRuntime.prepare_text(runtime, safe['prompt'], safe['image_file'], inputs_dir,
                                     expected_hash=safe['image_sha256'], protocol='plain_answer')
    prepared = runtime._append_native_prefill(prepared, safe, inputs_dir, prefix)
    prepared['inputs'] = {k: v.float() if runtime.torch.is_tensor(v) and v.is_floating_point() else v
                          for k, v in prepared['inputs'].items()}
    prepared['routing'].update(
        protocol='structured', protocol_revision='v2', assistant_prefill=prefix,
        prefill_is_context_not_loss_target=True, same_assistant_message=True,
        stop_rule='first_balanced_json_including_full_prefill',
        input_float_dtype='float32', probability_dtype='float32',
        forward_input_dtypes={k: str(v.dtype).removeprefix('torch.')
                              for k, v in prepared['inputs'].items() if runtime.torch.is_tensor(v)})
    prepared['model_input'] = safe
    return prepared


def _adapter_hashes(path):
    return {n: file_sha(Path(path) / n) for n in ('adapter_config.json', 'adapter_model.safetensors')}


def _versions(model):
    # Tensor versions include all persistent buffers as well as parameters.
    return {kind + ':' + n: (id(v), v._version)
            for kind, seq in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]
            for n, v in seq}


def _lora_tensors(model):
    return {n: v.detach() for n, v in model.named_parameters()
            if '.lora_A.' in n or '.lora_B.' in n}


class NativeBackend:
    def __init__(self, config, parent, run_dir):
        self.config, self.parent = copy.deepcopy(config), parent
        self.run_dir = Path(run_dir).resolve(strict=True)
        self.binding = copy.deepcopy(config['runtime_binding'])
        self.inputs_dir = Path(self.binding.get('inputs_dir', self.run_dir / 'INPUTS')).resolve(strict=True)
        if parent not in ('b0', 'b1', 'b2'):
            raise PermissionError('Unknown frozen parent')
        auth = config['authorization']
        if auth['train'] or auth['optimizer_updates'] != 0 or not auth['new_inference_only']:
            raise PermissionError('Only zero-update inference is supported')
        self.native = import_native(self.binding)
        import torch
        from mm_dev.runtime import configure_audited_backend, actual_cuda_identity
        from sr_f5.runtime import Runtime as FP32Runtime
        from sr_f6.source_binding import verify_bound_files, content_inventory, object_sha256
        self.torch = torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible CUDA GPU is required')
        configure_audited_backend()
        source_path = Path(self.binding['historical_source_binding']).resolve(strict=True)
        history = json.loads(source_path.read_text())
        adapter = Path(config['model']['parents'][parent]).resolve(strict=True)
        base = Path(config['model']['model_path']).resolve(strict=True)
        if base.name != config['model']['snapshot']:
            raise PermissionError('Snapshot directory differs from frozen identity')
        parent_record = history['parents'][parent]
        if Path(parent_record['path']).resolve(strict=True) != adapter:
            raise PermissionError('Historical parent binding does not match plan')
        if object_sha256(content_inventory(parent_record['files'])) != parent_record['content_sha256']:
            raise PermissionError('Historical parent content digest is inconsistent')
        expected_files = {str(adapter / n) for n in ('adapter_config.json', 'adapter_model.safetensors')}
        if {str(Path(r['path']).resolve()) for r in parent_record['files']} != expected_files:
            raise PermissionError('Historical adapter inventory is incomplete')
        if {str(p) for p in base.rglob('*') if p.is_file()} != {r['path'] for r in history['model_files']}:
            raise PermissionError('Historical snapshot inventory differs')
        self._history, self._history_path = history, source_path
        model_check = verify_bound_files(history['model_files'])
        parent_check = verify_bound_files(parent_record['files'])
        self._before = _adapter_hashes(adapter)
        expected_hashes = {Path(r['path']).name: r['sha256'] for r in parent_record['files']}
        if self._before != expected_hashes:
            raise PermissionError('Frozen parent bytes differ')
        self._accounting = []
        native, backend = self.native, self

        class ProbeRuntime(FP32Runtime):
            def _set_recipe(self):
                super()._set_recipe()
                for key, value in native.generation_recipe().items():
                    setattr(self.generation_config, key, value)
                self.identity.update(
                    generation_config_expanded=self.generation_config.to_dict(),
                    runtime_recipe='SR-ORDER-PROBE-native-1536-readonly',
                    experiment_id=config['experiment_id'])
            def prepare_input(self, model_input, run_root, *, protocol='structured'):
                if protocol != 'structured' or model_input != backend._active_safe:
                    raise PermissionError('Native input differs from active public request')
                if Path(run_root).resolve() != backend.inputs_dir:
                    raise PermissionError('Native input root changed')
                prepared = prepare_public_input(self, backend._active_request, backend.inputs_dir)
                self._srf5_prepared = prepared
                return prepared
            def reserve(self, kind, count, **metadata):
                if kind not in ('completion_attempts', 'generated_tokens'):
                    raise PermissionError('Training and extra forward accounting prohibited')
                backend._accounting.append(dict(kind=kind, count=count, metadata=metadata))

        native.seed_all(native.seed_for(config['data']['master_seed'], 'load', parent))
        runtime = ProbeRuntime(str(base), adapter_path=str(adapter), output_root=self.run_dir,
                               protocol_revision='v2', account=None)
        native.configure_parent_training(runtime, parent_adapter_config=json.loads((adapter / 'adapter_config.json').read_text()))
        self._source_dtype = runtime.identity['dtype']
        runtime.model = runtime.model.float()
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        runtime.model.eval()
        self._policy_hash = native.state_hash(_lora_tensors(runtime.model))
        self._parameter_dtypes = sorted({str(v.dtype).removeprefix('torch.') for v in runtime.model.parameters() if v.is_floating_point()})
        self._buffer_dtypes = sorted({str(v.dtype).removeprefix('torch.') for v in runtime.model.buffers() if v.is_floating_point()})
        if self._parameter_dtypes != ['float32'] or any(d != 'float32' for d in self._buffer_dtypes):
            raise RuntimeError('Full FP32 conversion did not take effect')
        self._after = _adapter_hashes(adapter)
        if self._after != self._before:
            raise PermissionError('Parent bytes changed during loading')
        runtime.identity.update(dtype='float32', parent_id=parent, run_id=config['experiment_id'], step=0,
                                input_float_dtype='float32', probability_dtype='float32',
                                numerics=dict(native.NUMERICS_V2), model_cast_from_dtype=self._source_dtype,
                                forward_parameter_dtypes=self._parameter_dtypes, forward_float_buffer_dtypes=self._buffer_dtypes)
        self.runtime, self._closed = runtime, False
        self._bound_versions = _versions(runtime.model)
        self._extra_identity = dict(parent=parent, snapshot=config['model']['snapshot'], model_path=str(base),
            adapter_path=str(adapter), parameter_dtypes=self._parameter_dtypes, buffer_dtypes=self._buffer_dtypes,
            trainable_parameters=[], optimizer=None, scheduler=None, optimizer_updates=0,
            parent_hashes_before=self._before, parent_hashes_after=self._after, policy_tensor_hash=self._policy_hash,
            hardware=actual_cuda_identity(), historical_source_binding_sha256=file_sha(source_path),
            bound_model_check=model_check, bound_parent_check=parent_check,
            native_kernel_file=str(Path(native.__file__).resolve()), native_kernel_file_sha256=file_sha(native.__file__),
            native_kernel_function_sha256=text_sha(inspect.getsource(native.Runtime._generate_native_group)),
            adapter_source_sha256=file_sha(__file__), execution_mode='READONLY_ORDER_PROBE',
            source_helpers={name: text_sha(inspect.getsource(obj)) for name, obj in {
                'prepare_text': importlib.import_module('sr_f1.runtime').SRRuntime.prepare_text,
                'append_native_prefill': runtime._append_native_prefill,
                'independent_draw_rng': native.independent_draw_rng,
                'BatchedProtocolStop': native.BatchedProtocolStop}.items()})
        self._active_request, self._active_safe = None, None
        self._readonly()

    def _readonly(self):
        if self._closed:
            return
        model = self.runtime.model
        if model.training or _versions(model) != self._bound_versions:
            raise PermissionError('Model mode or parameter/buffer versions changed')
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise PermissionError('Read-only model acquired gradients')

    def identity(self):
        self._readonly()
        return copy.deepcopy({**self.runtime.identity, **self._extra_identity})

    def generate_one(self, request):
        if self._closed or self._active_request is not None:
            raise PermissionError('Closed or nested generation')
        if request.get('parent') != self.parent:
            raise PermissionError('Request belongs to a different parent')
        safe = validate_request(request, self.config)
        self._readonly()
        self._active_request, self._active_safe = copy.deepcopy(request), safe
        native = self.native
        from sr_f1.json_protocol import balanced_cut
        helpers = prefix_helpers(request['assistant_prefill'], native.decode_generated, native.BatchedProtocolStop, scan=balanced_cut)
        generate = bind_native_kernel(native.Runtime._generate_native_group, helpers)
        captured = []
        self._accounting = []
        error = None
        try:
            with native.independent_draw_rng(self.torch, [request['seed']]):
                generate(self.runtime, safe, self.inputs_dir, [request['seed']], protocol='structured', on_completion=captured.append)
        except Exception as exc:
            if not captured:
                raise
            error = repr(exc)
        finally:
            self._active_request, self._active_safe = None, None
        return self._preserve_captured(request, captured, error)

    def _preserve_captured(self, request, captured, error):
        """Never turn post-sampling packaging failures into resampling permission.

        The caller writes the returned record before checking technical validity.
        If normalization fails, retain every unmodified native capture and mark
        the record invalid. No tensor access or probability calculation occurs
        in this fallback. An empty capture still indicates infrastructure failure.
        """
        if not captured:
            raise RuntimeError('Native generation returned no captured response')
        try:
            if len(captured) != 1:
                raise RuntimeError('Native generation returned multiple responses')
            return self._normalize_captured(request, copy.deepcopy(captured), error)
        except Exception as exc:
            original = captured[0]
            result = dict(original)
            result.update({k: request[k] for k in
                           ('request_id', 'qid', 'root_id', 'split', 'parent', 'condition', 'branch', 'draw')})
            result.update(seed=request['seed'],
                          generation_status='TECHNICAL_INVALID',
                          technical_validation_errors=[*original.get('technical_validation_errors', []),
                              'POST_CAPTURE_NORMALIZATION_FAILED:' + repr(exc)],
                          native_exception=error,
                          raw_native_captures=captured,
                          captured_response_count=len(captured),
                          assistant_prefill=request['assistant_prefill'],
                          suffix_text=original.get('generated_text'),
                          suffix_token_ids=original.get('tokens'),
                          generated_tokens=len(original.get('tokens', [])),
                          optimizer_updates=0,
                          recovery='PRESERVE_CAPTURED_OUTPUT_REPAIR_PACKAGING_ONLY_NO_RESAMPLING')
            return result

    def _normalize_captured(self, request, captured, error):
        row = captured[0]
        try:
            self._readonly()
        except Exception as exc:
            row['technical_validation_errors'].append('READONLY_CHECK_FAILED:' + repr(exc))
            row['generation_status'] = 'TECHNICAL_INVALID'
        prepared = self.runtime._srf5_prepared
        routing = prepared['routing']
        ids = prepared['inputs']['input_ids'][0].detach().cpu().tolist()
        boundary = dict(base_prompt_token_count=routing['base_prompt_token_count'],
                        prefill_token_ids=routing['assistant_prefill_token_ids'],
                        native_char_span=routing['assistant_prefill_native_char_span'],
                        native_offsets=routing['assistant_prefill_native_offsets'],
                        same_assistant_message=True)
        if ids[boundary['base_prompt_token_count']:] != boundary['prefill_token_ids']:
            raise RuntimeError('Stored input/prefill boundary mismatch')
        result = dict(row, **{k: request[k] for k in ('request_id', 'qid', 'root_id', 'split', 'parent', 'condition', 'branch', 'draw')})
        result.update(seed=request['seed'], suffix_text=row['generated_text'],
                      suffix_token_ids=row['tokens'], prompt_input_ids=ids,
                      prefill_boundary=boundary, generated_tokens=len(row['tokens']),
                      sampling=copy.deepcopy(request['sampling']),
                      image_sha256=request['image_sha256'], prompt_sha256=request['prompt_sha256'],
                      prefill_sha256=request['prefill_sha256'],
                      model_identity=copy.deepcopy({**self.runtime.identity, **self._extra_identity}),
                      optimizer_updates=0, seed_semantics='independent_per_draw_torch_generator',
                      sampled_token_probabilities=[math.exp(x) for x in row['sampler_logprobs']],
                      accounting=copy.deepcopy(self._accounting), native_exception=error,
                      stop_metadata=dict(eos_token_ids=list(self.runtime.eos_ids),
                                         observed_eos_token=next((x for x in row['tokens'] if x in self.runtime.eos_ids), None),
                                         first_balanced_token_count=row['balanced_token_count'],
                                         first_balanced_char=row['balanced_cut_char'],
                                         suffix_limit=1536),
                      native_prompt_inputs={k: v.detach().cpu().tolist() for k, v in prepared['inputs'].items()
                                            if k != 'pixel_values' and hasattr(v, 'detach')})
        if error and not result['technical_validation_errors']:
            result['technical_validation_errors'] = ['NATIVE_EXCEPTION_AFTER_CAPTURE:' + error]
            result['generation_status'] = 'TECHNICAL_INVALID'
        return result

    def close(self):
        if self._closed:
            return self.identity()
        self._readonly()
        after = _adapter_hashes(self._extra_identity['adapter_path'])
        policy = self.native.state_hash(_lora_tensors(self.runtime.model))
        if after != self._before or policy != self._policy_hash:
            raise PermissionError('Model/adapter changed during read-only run')
        self._extra_identity.update(parent_hashes_after=after, policy_tensor_hash_after=policy,
                                    end_model_unchanged=True)
        result = self.identity()
        self.runtime._vision_hook.remove()
        self._closed = True
        return result


def make_backend(config, parent, run_dir):
    return NativeBackend(config, parent, run_dir)
