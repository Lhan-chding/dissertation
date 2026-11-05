"""SR-ORD-TRAIN-1 registration around unchanged native sampling/forward math.

RUN_BINDING schema (all paths absolute except public image_file):
 experiment_id, package_root, inputs_root, manifest_sha256 (frozen/FREEZE.json),
 native_code_root, historical_source_binding, historical_source_binding_sha256,
 source_files=[{path,sha256}] (every native src/*.py),
 implementation_files=[{path,sha256}] (including this module),
 model={snapshot,model_path,parents:{b0:path,b1:path}},
 requests_manifest={path,sha256}, source_sha256 (canonical source_files digest).
 TRAINING additionally requires path_identity={arm,train_seed,run_id}.
 Each request has request_id, scope=DEV_SMOKE|CLARIFICATION|TRAIN, parent,
 prompt/prompt_sha256, assistant_prefill/prefill_sha256, image_file/image_sha256,
 seed and its frozen control-plane identifiers. Only public prompt, image and
 assistant prefill reach the processor. Endpoints require a separate later adapter.

No historical admission registry is changed or invoked. Import is CPU-only.
The driver owns checkpoint transactions; this backend records each sampling
attempt exclusively before calling the model, and never retries an uncertain call.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import types

EXPERIMENT = 'SR-ORD-TRAIN-1'
SNAPSHOT = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
PUBLIC = ('prompt', 'image_file', 'image_sha256')
CONTROL = {'request_id','row_id','scope','parent','group_id','qid','root_id','case_id',
           'input_id','draw','seed','arm','train_seed','step','structure','view',
           'split','panel','phase','slot','U_applicable','group_index','group_seed'}
REQUEST_FIELDS = CONTROL | set(PUBLIC) | {'prompt_sha256','assistant_prefill','prefill_sha256'}
MODES = {'TRAINING': 'TRAIN', 'DEV_SMOKE': 'DEV_SMOKE', 'CLARIFICATION': 'CLARIFICATION'}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def text_sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def object_sha(value):
    return text_sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))


def read_json(path):
    def pairs(values):
        result = dict(values)
        if len(result) != len(values):
            raise ValueError('Duplicate JSON keys')
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def canonical_directory(path):
    p = Path(path)
    if not p.is_absolute() or p.resolve(strict=True) != p or not p.is_dir():
        raise PermissionError('Expected existing canonical directory: ' + str(p))
    return p


def verify_file(ref):
    p = Path(ref['path'])
    if not p.is_absolute() or p.is_symlink() or not p.is_file() or file_sha(p) != ref['sha256']:
        raise PermissionError('Missing or changed bound file: ' + str(p))
    return p


def public_request(request, inputs_root):
    if not isinstance(request, dict) or set(request) - REQUEST_FIELDS:
        raise PermissionError('Unregistered/private fields at model boundary')
    for field, digest in [('prompt','prompt_sha256'),('assistant_prefill','prefill_sha256')]:
        if not isinstance(request.get(field), str) or text_sha(request[field]) != request.get(digest):
            raise PermissionError('Frozen text digest differs: ' + field)
    if not request['prompt'] or not request['assistant_prefill'].startswith('{'):
        raise PermissionError('Missing prompt or open JSON prefill')
    if request.get('scope') not in MODES.values() or request.get('parent') not in ('b0','b1'):
        raise PermissionError('Unregistered request scope/parent')
    if request['assistant_prefill'] != '{':
        raise PermissionError('This pre-evaluation registration permits only natural prefix')
    if type(request.get('seed')) is not int or not 0 <= request['seed'] < 2**63:
        raise PermissionError('Invalid draw seed')
    if not isinstance(request.get('request_id'), str) or not request['request_id']:
        raise PermissionError('Missing request identity')
    rel = PurePosixPath(request['image_file'])
    if rel.is_absolute() or '..' in rel.parts:
        raise PermissionError('Image escapes input root')
    image = Path(inputs_root) / rel
    if image.is_symlink() or not image.resolve(strict=True).is_relative_to(inputs_root):
        raise PermissionError('Image resolves outside input root')
    if file_sha(image) != request['image_sha256']:
        raise PermissionError('Frozen image bytes changed')
    return {k: request[k] for k in PUBLIC}


def verify_binding(binding, parent, run_dir, mode):
    """Fail before importing native model code or reading model tensors."""
    from gpu_production_v2.binding import verify_production_binding
    verify_production_binding(binding, run_dir)
    if mode != "TRAINING":
        raise PermissionError("GPU production bridge only permits registered training")
    if binding.get('experiment_id') != EXPERIMENT or parent not in ('b0','b1') or mode not in MODES:
        raise PermissionError('New experiment identity/mode required')
    package = canonical_directory(binding['package_root'])
    inputs = canonical_directory(binding['inputs_root'])
    run = canonical_directory(run_dir)
    native = canonical_directory(binding['native_code_root'])
    if run == native or run.is_relative_to(native):
        raise PermissionError('Cannot write historical source tree')
    if file_sha(package / 'frozen/FREEZE.json') != binding['manifest_sha256']:
        raise PermissionError('Frozen scientific manifest changed')
    frozen = read_json(package / 'frozen/FREEZE.json')
    for relative, expected_hash in frozen['files'].items():
        if file_sha(package / 'frozen' / relative) != expected_hash:
            raise PermissionError('Frozen input bytes changed: ' + relative)
    inventory = binding['source_files']
    paths = [verify_file(row) for row in inventory]
    expected = {p for p in (native / 'src').rglob('*.py') if p.is_file()}
    if len(paths) != len(set(paths)) or set(paths) != expected:
        raise PermissionError('Native Python dependency inventory incomplete')
    if object_sha(inventory) != binding['source_sha256']:
        raise PermissionError('Native source inventory digest changed')
    implementation = [verify_file(row) for row in binding['implementation_files']]
    if Path(__file__).resolve() not in implementation:
        raise PermissionError('Actual new wrapper is not bound')
    history_path = verify_file({'path':binding['historical_source_binding'],
                               'sha256':binding['historical_source_binding_sha256']})
    history = read_json(history_path)
    model = binding['model']
    base = canonical_directory(model['model_path'])
    adapter = canonical_directory(model['parents'][parent])
    if model['snapshot'] != SNAPSHOT or base.name != SNAPSHOT:
        raise PermissionError('Frozen model snapshot differs')
    record = history['parents'][parent]
    if record['path'] != str(adapter):
        raise PermissionError('Parent adapter differs from historical binding')
    from runtime_contract import PARENT_FILES, PINS
    for name, expected_hash in zip(('adapter_config.json','adapter_model.safetensors'), PARENT_FILES[parent]):
        if file_sha(adapter / name) != expected_hash:
            raise PermissionError('Parent file differs: ' + name)
    for relative, expected_hash in PINS.items():
        if file_sha(native / 'src' / relative) != expected_hash:
            raise PermissionError('Native frozen kernel differs: ' + relative)
    model_paths = {str(p) for p in base.rglob('*') if p.is_file()}
    if model_paths != {row['path'] for row in history['model_files']}:
        raise PermissionError('Snapshot file inventory differs')
    # Snapshot blobs may be symlinks in the verified model cache; hash their targets.
    for row in history['model_files']:
        if file_sha(row['path']) != row['sha256']:
            raise PermissionError('Model/tokenizer bytes differ: ' + row['path'])
    requests_path = verify_file(binding['requests_manifest'])
    requests = {}
    for line in requests_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        public_request(row, inputs)
        key = row['request_id']
        if key in requests:
            raise PermissionError('Duplicate frozen request identity')
        requests[key] = row
    if not requests:
        raise PermissionError('Empty registered request manifest')
    if mode == 'TRAINING':
        identity = binding['path_identity']
        if identity['arm'] not in ('R','G','D') or identity['train_seed'] not in (11031,11032):
            raise PermissionError('Unregistered training path')
        if not identity['run_id'].startswith(EXPERIMENT + '/'):
            raise PermissionError('Training path needs a distinct new run id')
    return dict(package=package, inputs=inputs, run=run, native=native,
                base=base, adapter=adapter, history=history, requests=requests)


def prefix_helpers(prefill, decode, native_stop, scan):
    """Preserved same-assistant serialization from the verified ORDER probe."""
    def get_prefill(protocol, revision):
        if protocol != 'structured' or revision != 'v2':
            raise PermissionError('Only structured/v2 is registered')
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
                raise PermissionError('Each draw uses an independent one-sequence call')
            super().__init__(tokenizer,prompt_len,rows,protocol,revision=revision)
            self.prefill = get_prefill(protocol,revision)
    return dict(_prefill=get_prefill,first_boundary_token=boundary,
                decoded_protocol=decoded,BatchedProtocolStop=PrefixStop)


def bind_native_kernel(native_function, helpers):
    if set(helpers) != {'_prefill','first_boundary_token','decoded_protocol','BatchedProtocolStop'}:
        raise PermissionError('Only prefix serialization helpers can be rebound')
    namespace = dict(native_function.__globals__)
    namespace.update(helpers)
    bound = types.FunctionType(native_function.__code__, namespace, native_function.__name__,
                               native_function.__defaults__,native_function.__closure__)
    bound.__kwdefaults__ = copy.deepcopy(native_function.__kwdefaults__)
    return bound


def prepare_public_input(runtime, request, inputs_dir):
    from sr_f1.runtime import SRRuntime
    safe = {k:request[k] for k in PUBLIC}
    prefix = request['assistant_prefill']
    prepared = SRRuntime.prepare_text(runtime,safe['prompt'],safe['image_file'],inputs_dir,
                                     expected_hash=safe['image_sha256'],protocol='plain_answer')
    prepared = runtime._append_native_prefill(prepared,safe,inputs_dir,prefix)
    prepared['inputs'] = {k:v.float() if runtime.torch.is_tensor(v) and v.is_floating_point() else v
                          for k,v in prepared['inputs'].items()}
    prepared['routing'].update(protocol='structured',protocol_revision='v2',assistant_prefill=prefix,
        prefill_is_context_not_loss_target=True,same_assistant_message=True,
        stop_rule='first_balanced_json_including_full_prefill',input_float_dtype='float32',
        probability_dtype='float32',forward_input_dtypes={k:str(v.dtype).removeprefix('torch.')
        for k,v in prepared['inputs'].items() if runtime.torch.is_tensor(v)})
    prepared['model_input'] = safe
    return prepared


def failure_safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return {'nonfinite_float':repr(value)}
    if isinstance(value,dict):
        return {k:failure_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [failure_safe(v) for v in value]
    return value


def write_once(path, value):
    """A persisted attempt survives crashes; an existing attempt is never retried."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


class NativeBackend:
    def __init__(self,binding,parent,run_dir,mode='DEV_SMOKE'):
        self.binding,self.parent,self.mode = copy.deepcopy(binding),parent,mode
        checked = verify_binding(binding,parent,run_dir,mode)
        self.checked = checked
        self.inputs_dir,self.run_dir = checked['inputs'],checked['run']
        self.requests = checked['requests']
        self._active_request = None
        self._accounting = []
        self._closed = False
        self._public = {}
        for request in self.requests.values():
            if self._matches_path(request):
                safe = {k:request[k] for k in PUBLIC}
                key = object_sha(safe)
                if key in self._public and self._public[key]['assistant_prefill'] != request['assistant_prefill']:
                    raise PermissionError('Ambiguous public input prefix')
                self._public[key] = request
        if not self._public:
            raise PermissionError('No registered requests for this path')
        sys.path.insert(0,str(checked['native'] / 'src'))
        native = importlib.import_module('sr_f6.runtime')
        if Path(native.__file__).resolve() != checked['native'] / 'src/sr_f6/runtime.py':
            raise PermissionError('Native runtime import is shadowed')
        self.native = native
        bound_sources = {Path(r['path']) for r in binding['source_files']}
        namespaces = {p.relative_to(checked['native']/'src').parts[0].removesuffix('.py') for p in bound_sources}
        for name,module in tuple(sys.modules.items()):
            source = getattr(module,'__file__',None)
            if name.split('.')[0] in namespaces and source and Path(source).resolve() not in bound_sources:
                raise PermissionError('Shadowed native dependency: '+name)
        import torch
        from mm_dev.runtime import configure_audited_backend
        from gpu_production_v2.state import actual_cuda_identity
        from sr_f5.runtime import Runtime as FP32Runtime
        from sr_f3_ce.training import fresh_optimizer
        from sr_f12.runtime import _repeat_prompt
        configure_audited_backend()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError('Exactly two visible CUDA GPUs required')
        self.torch = torch
        backend = self
        class OrderRuntime(FP32Runtime):
            def _set_recipe(self):
                super()._set_recipe()
                for key,value in native.generation_recipe().items():
                    setattr(self.generation_config,key,value)
                self.identity.update(
                    runtime_recipe='SR-ORD-TRAIN-1-native-full-sequence-1536',
                    generation_config_expanded=self.generation_config.to_dict())
            def prepare_input(self,model_input,run_root,*,protocol='structured_v4'):
                if protocol not in ('structured_v4','structured') or Path(run_root).resolve() != backend.inputs_dir:
                    raise PermissionError('Unregistered preparation root/protocol')
                key = object_sha(model_input)
                request = backend._public.get(key)
                if request is None or set(model_input) != set(PUBLIC):
                    raise PermissionError('Unregistered public input')
                public_request(request,backend.inputs_dir)
                if getattr(self,'_order_prepared_key',None) != key:
                    self._srf5_prepared = prepare_public_input(self,request,backend.inputs_dir)
                    self._order_prepared_key = key
                return self._srf5_prepared
            def reserve(self,kind,count,**metadata):
                if kind not in ('completion_attempts','generated_tokens','extra_forward_sequences','diagnostic_gradient_calls'):
                    raise PermissionError('Unregistered native accounting category')
                backend._accounting.append(dict(kind=kind,count=count,metadata=metadata))
                log = backend.run_dir / 'accounting.jsonl'
                with log.open('a') as f:
                    f.write(json.dumps(dict(kind=kind,count=count,metadata=metadata),sort_keys=True)+'\n')
                    f.flush(); os.fsync(f.fileno())
            def _prepared_records(self,records):
                for record in records:
                    request = backend.requests.get(record.get('request_id'))
                    if request is None or not backend._matches_path(request):
                        raise PermissionError('Forward request is outside registered path')
                    if record.get('model_input') != {k:request[k] for k in PUBLIC} or record.get('seed') != request['seed']:
                        raise PermissionError('Forward input differs from registered request')
                    capture = Path(record['native_capture_path'])
                    if not capture.is_absolute() or not capture.resolve(strict=True).is_relative_to(backend.run_dir.parent.parent):
                        raise PermissionError('Captured trajectory is outside this path')
                    if file_sha(capture) != record['native_capture_sha256']:
                        raise PermissionError('Native capture bytes changed')
                    saved = read_json(capture)
                    if saved['adapter_identity']['run_id'] != self.identity['run_id'] or saved['adapter_identity']['step'] != self.identity['step']:
                        raise PermissionError('Native capture is from another policy path/step')
                    if any(record.get(k) != saved.get(k) for k in ('tokens','sampler_logprobs','prompt_tensor_hash','model_input','seed')):
                        raise PermissionError('Forward differs from persisted original trajectory')
                return super()._prepared_records(records)
            def _validate_record_revision(self,record):
                if record.get('protocol') not in ('structured','structured_v4') or record.get('protocol_revision') != 'v2':
                    raise PermissionError('Unregistered record protocol')
                if record.get('assistant_prefill') != '{':
                    raise PermissionError('Training/replay prefill differs')
            def _validate_prepared_revision(self,prepared):
                r = prepared.get('routing',{})
                if r.get('protocol') not in ('structured','structured_v4') or r.get('protocol_revision') != 'v2' or r.get('assistant_prefill') != '{':
                    raise PermissionError('Prepared native protocol differs')
            def _batch_forward(self,prepared,token_rows,*,purpose,grad):
                # Same F6 concatenation/mRoPE/native forward, new admission identity.
                self._validate_prepared_revision(prepared)
                if backend.mode == 'CLARIFICATION' or (grad and backend.mode != 'TRAINING'):
                    raise PermissionError('Gradient/replay outside registered mode')
                if len(token_rows) != 1 or not token_rows[0] or len(token_rows[0]) > 1536:
                    raise ValueError('One full original 1..1536-token sequence required')
                inputs = _repeat_prompt(prepared['inputs'],len(token_rows))
                prompt_len,longest = inputs['input_ids'].shape[-1],max(map(len,token_rows))
                pad = self.processor.tokenizer.pad_token_id
                if pad is None: pad = self.eos_ids[0]
                completion = inputs['input_ids'].new_full((len(token_rows),longest),pad)
                mask = torch.zeros_like(completion)
                for index,tokens in enumerate(token_rows):
                    completion[index,:len(tokens)] = completion.new_tensor(tokens)
                    mask[index,:len(tokens)] = 1
                inputs['input_ids'] = torch.cat([inputs['input_ids'],completion],-1)
                inputs['attention_mask'] = torch.cat([inputs.get('attention_mask',mask.new_ones((len(token_rows),prompt_len))),mask],-1)
                inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(completion)],-1)
                return self._native_forward(inputs,completion_start=prompt_len,lengths=list(map(len,token_rows)),purpose=purpose,grad=grad)
        native.seed_all(native.seed_for(2026110301,'load',parent))
        runtime = OrderRuntime(str(checked['base']),adapter_path=str(checked['adapter']),
                               output_root=self.run_dir,protocol_revision='v2',account=None)
        native.configure_parent_training(runtime,parent_adapter_config=read_json(checked['adapter']/'adapter_config.json'))
        source_dtype = runtime.identity['dtype']
        runtime.model = runtime.model.float()
        self.adapter_names = tuple(n for n,p in runtime.model.named_parameters() if p.requires_grad)
        self.reference = {n:p.detach().cpu().clone() for n,p in runtime.model.named_parameters() if p.requires_grad}
        parent_hash = native.state_hash(self.reference)
        if parent_hash != binding['parent_tensor_sha256'][parent]:
            raise PermissionError('Loaded parent tensors differ from verified historical snapshot')
        if any(v.is_floating_point() and v.dtype != torch.float32 for v in (*runtime.model.parameters(),*runtime.model.buffers())):
            raise PermissionError('FP32 cast failed')
        if mode == 'CLARIFICATION':
            for p in runtime.model.parameters(): p.requires_grad_(False)
        runtime.model.eval()
        runtime._srf5_numerics = dict(native.NUMERICS_V2)
        path = binding.get('path_identity',{})
        run_id = path.get('run_id',EXPERIMENT+'/'+mode+'/'+parent)
        runtime.identity.update(experiment_id=EXPERIMENT,run_root=str(self.inputs_dir),audit_root=str(self.run_dir),
            execution_mode=mode,run_id=run_id,parent_id=parent,reference_parent=parent,step=0,
            arm=path.get('arm'),train_seed=path.get('train_seed'),reference_hash=parent_hash,
            source_sha256=binding['source_sha256'],dtype='float32',input_float_dtype='float32',
            probability_dtype='float32',model_cast_from_dtype=source_dtype,numerics=dict(native.NUMERICS_V2),
            manifest_sha256=binding['manifest_sha256'],
            forward_parameter_dtypes=['float32'],forward_float_buffer_dtypes=['float32'],snapshot=SNAPSHOT)
        self.runtime = runtime
        from runtime_contract import PARENT_FILES
        for name,expected_hash in zip(('adapter_config.json','adapter_model.safetensors'),PARENT_FILES[parent]):
            if file_sha(checked['adapter']/name) != expected_hash:
                raise PermissionError('Parent bytes changed during model load')
        self.optimizer,self.scheduler = fresh_optimizer(runtime.model) if mode == 'TRAINING' else (None,None)
        self.loaded_identity = dict(runtime.identity,hardware=actual_cuda_identity(),parent_tensor_sha256=parent_hash,
            native_runtime_sha256=file_sha(native.__file__),backend_sha256=file_sha(__file__),
            manifest_sha256=binding['manifest_sha256'],binding_sha256=object_sha(binding),
            parameter_names=list(self.adapter_names),model_calls=0,optimizer_created=self.optimizer is not None)
        write_once(self.run_dir/'LOADED_IDENTITY.json',self.loaded_identity)

    def _matches_path(self,request):
        if request['parent'] != self.parent or request['scope'] != MODES[self.mode]:
            return False
        if self.mode == 'TRAINING':
            path = self.binding['path_identity']
            return all(request.get(k) == path[k] for k in ('arm','train_seed'))
        return True

    def policy_hash(self):
        names = set(self.adapter_names)
        return self.native.state_hash({n:p.detach() for n,p in self.runtime.model.named_parameters() if n in names})

    def generate_one(self,request):
        if self._closed or self._active_request is not None:
            raise PermissionError('Closed or nested generation')
        expected = self.requests.get(request.get('request_id'))
        if expected != request or not self._matches_path(request):
            raise PermissionError('Request differs from new frozen registration')
        if self.mode == 'TRAINING' and request.get('step') != self.runtime.identity['step'] + 1:
            raise PermissionError('Stale or future policy request')
        safe = public_request(request,self.inputs_dir)
        policy = self.policy_hash()
        key = text_sha(request['request_id'])
        attempt = self.run_dir / 'generation_attempts' / key
        write_once(attempt.with_suffix('.BEGIN.json'),dict(request_id=request['request_id'],seed=request['seed'],
            policy_sha256=policy,reference_sha256=self.runtime.identity['reference_hash'],status='ATTEMPTED_UNKNOWN'))
        self._active_request = copy.deepcopy(request)
        self._accounting = []
        from sr_f1.json_protocol import balanced_cut
        helpers = prefix_helpers(request['assistant_prefill'],self.native.decode_generated,self.native.BatchedProtocolStop,balanced_cut)
        kernel = bind_native_kernel(self.native.Runtime._generate_native_group,helpers)
        captured = []
        def capture(row):
            captured.append(row)
            # Nonfinite native failures remain explicit evidence, never dropped.
            write_once(attempt.with_suffix('.CAPTURE.json'),failure_safe(row))
        error = None
        try:
            with self.native.independent_draw_rng(self.torch,[request['seed']]):
                kernel(self.runtime,safe,self.inputs_dir,[request['seed']],protocol='structured',on_completion=capture)
        except Exception as exc:
            write_once(attempt.with_suffix('.ERROR.json'),failure_safe(dict(
                request_id=request['request_id'],error_type=type(exc).__name__,error=repr(exc),
                captured_responses=len(captured),accounting=copy.deepcopy(self._accounting),
                policy_sha256=policy,recovery='AUDIT_EXISTING_ATTEMPT_NO_AUTOMATIC_RESAMPLING')))
            if not captured:
                raise
            error = repr(exc)
        finally:
            self._active_request = None
        if len(captured) != 1:
            raise RuntimeError('Native capture count differs; preserve attempt and audit')
        row = dict(captured[0])
        prepared = self.runtime._srf5_prepared
        row.update({k:v for k,v in request.items() if k in CONTROL})
        if row['finish_reason'] not in ('balanced','eos','length'):
            raise ValueError('Unregistered native finish reason')
        row.update(row_id=request['request_id'],protocol='structured_v4',backend_protocol='structured',
            finished=row['finish_reason']=='balanced',
            native_capture_path=str(attempt.with_suffix('.CAPTURE.json')),
            native_capture_sha256=file_sha(attempt.with_suffix('.CAPTURE.json')),
            protocol_revision='v2',run_id=self.runtime.identity['run_id'],policy_hash=policy,
            reference_hash=self.runtime.identity['reference_hash'],source_sha256=self.binding['source_sha256'],
            prompt_sha256=request['prompt_sha256'],image_sha256=request['image_sha256'],
            prefill_sha256=request['prefill_sha256'],suffix_text=row['generated_text'],
            seed_semantics='independent_per_draw_torch_generator',
            native_prompt_inputs={k:v.detach().cpu().tolist() for k,v in prepared['inputs'].items() if k!='pixel_values' and hasattr(v,'detach')},
            native_exception=error,accounting=copy.deepcopy(self._accounting))
        from adapter.training import trajectory_hash
        try:
            row['token_path_sha256'] = trajectory_hash(row)
        except (ValueError, TypeError) as exc:
            row['token_path_sha256'] = None
            row['technical_validation_errors'].append('INVALID_TRAJECTORY_DIGEST:'+repr(exc))
        if self.policy_hash() != policy or self.native.state_hash(self.reference) != self.runtime.identity['reference_hash']:
            row['technical_validation_errors'].append('POLICY_OR_REFERENCE_CHANGED_DURING_GENERATION')
        if error:
            row['technical_validation_errors'].append('NATIVE_EXCEPTION_AFTER_CAPTURE:'+error)
        if row['technical_validation_errors']:
            row['generation_status']='TECHNICAL_INVALID'
        write_once(attempt.with_suffix('.COMPLETE.json'),failure_safe(row))
        return row

    def processor_evidence(self, request_id, out):
        """Save real processor reconstruction without an additional model call."""
        if self._closed:
            raise PermissionError('Backend closed')
        request = self.requests.get(request_id)
        if request is None or not self._matches_path(request):
            raise PermissionError('Processor review request is not registered here')
        destination = Path(out)
        if not destination.is_absolute() or not destination.parent.resolve().is_relative_to(self.run_dir):
            raise PermissionError('Processor evidence must stay in worker output')
        destination.mkdir(parents=True,exist_ok=False)
        safe = public_request(request,self.inputs_dir)
        prepared = self.runtime.prepare_input(safe,self.inputs_dir,protocol='structured_v4')
        from mm_dev.data import reconstruct_rgb
        rgb = reconstruct_rgb(prepared['inputs']['pixel_values'],prepared['routing']['image_grid_thw'],
                              self.runtime.processor.image_processor)
        output = destination/'processed.png'
        rgb.save(output)
        receipt = dict(request_id=request_id,source_file=request['image_file'],
            source_sha256=request['image_sha256'],processed_file=str(output),
            processed_file_sha256=file_sha(output),processed_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            routing=copy.deepcopy(prepared['routing']),
            prompt_input_ids=prepared['inputs']['input_ids'][0].detach().cpu().tolist(),
            visual_review='PENDING',model_calls=0,optimizer_updates=0)
        write_once(destination/'PROCESSOR.json',receipt)
        return receipt

    def close(self):
        if self._closed:
            raise PermissionError('Backend already closed')
        if self.mode != 'TRAINING' and self.policy_hash() != self.loaded_identity['parent_tensor_sha256']:
            raise PermissionError('Zero-update policy changed')
        if self.native.state_hash(self.reference) != self.runtime.identity['reference_hash']:
            raise PermissionError('Frozen reference changed')
        self.runtime._vision_hook.remove()
        self._closed=True
        return dict(self.runtime.identity,policy_sha256=self.policy_hash(),closed=True)
