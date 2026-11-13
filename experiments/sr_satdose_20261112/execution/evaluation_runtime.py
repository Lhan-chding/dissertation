"""Independent DEV/MAIN native sampler, with no production authority or CLI.

DEVELOPMENT SNAPSHOT: behavior/integration tests are not yet implemented.
Not deployed or accepted for model execution. Syntax checks alone are not
evaluation admission; do not add this module to a production release yet.

The caller independently authenticates the execution inventory and verifier
source, owns the global control lock and GenerationBridge's durable no-retry
ledger, and supplies committed endpoint identities. This module does not reopen
EvaluationLoadHandle or change any frozen runtime class. MAIN authority must
verify four genuinely completed step24 chains and reject all HOLD/UNKNOWN state;
there is deliberately no default verifier or terminal-status shortcut here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import hashlib
import inspect
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from . import evaluation_loader as loader
from runtime import native_runtime as frozen, fork_loader
from runtime.backend import (FrozenRegistry, NativeKernelRunner, PUBLIC, file_sha,
                             object_sha, read_json, write_once)

state, tx = loader.state, loader.transactions
require = state.require


class AuthenticatedEvaluationAuthority(ABC):
    """Independently trusted source implementation; plain PASS dictionaries fail.

    verify must independently validate admission/registration/GPU evidence and
    exact endpoint registration. MAIN additionally requires full source-bound
    terminal semantics, four complete step24 chains, and no global HOLD/UNKNOWN.
    Its source reference is also supplied independently to the builder; an
    object's self-declared source is not sufficient authorization.
    """
    def __init__(self, verifier_reference):
        self.verifier_reference = copy.deepcopy(verifier_reference)
        AuthenticatedEvaluationAuthority.check_source(self, verifier_reference)

    def check_source(self, trusted_reference):
        require(self.verifier_reference == trusted_reference and
                getattr(self.verify, '__self__', None) is self and
                getattr(self.verify, '__func__', None) is type(self).verify,
                'Evaluation verifier source/method replaced')
        actual = Path(inspect.getsourcefile(type(self).verify) or '').resolve(strict=True)
        require(set(trusted_reference) == {'path','sha256'} and
                actual == Path(trusted_reference['path']) and actual.is_absolute() and not actual.is_symlink() and
                file_sha(actual) == trusted_reference['sha256'], 'Unauthenticated evaluation authority source')

    @abstractmethod
    def verify(self, *, root, selection, endpoint, design_freeze_sha256,
               execution_sources_sha256, request_ids_sha256, request_id):
        """Return exact bound scope plus authenticated root-relative references.

        schema=sr-satdose-authenticated-evaluation-authority-v1; selection,
        endpoint, design_freeze_sha256, config_sha256, execution_sources_sha256,
        request_ids_sha256, request_id; admission/registration/gpu_report refs.
        MAIN also terminal ref. A per-request call also requires begin ref to
        output_dir/sha256(request_id).BEGIN.json; construction request_id=None
        must not carry a BEGIN. All refs are {path,sha256}. Terminal document
        semantics belong to the separately audited verifier, not a new schema
        or a self-authored PASS introduced by this runtime.
        """
        raise PermissionError('No authenticated production evaluation authority installed')


def _execution_sources(authority, trusted_authority, sources, expected_sha):
    require(isinstance(authority, AuthenticatedEvaluationAuthority), 'AuthenticatedEvaluationAuthority required')
    AuthenticatedEvaluationAuthority.check_source(authority, trusted_authority)
    expected = {'evaluation_runtime':Path(__file__).resolve(),
                'evaluation_loader':Path(loader.__file__).resolve(),
                'training_state':Path(state.__file__).resolve(),
                'training_transaction':Path(tx.__file__).resolve(),
                'authority':Path(trusted_authority['path'])}
    require(set(sources) == set(expected) and object_sha(sources) == expected_sha and
            sources['authority'] == trusted_authority, 'Untrusted evaluation execution inventory')
    for name,path in expected.items():
        ref = sources[name]
        require(set(ref) == {'path','sha256'} and Path(ref['path']) == path and
                path.resolve(strict=True) == path and not path.is_symlink() and file_sha(path) == ref['sha256'],
                'Unbound evaluation source: ' + name)


def _registry(registry):
    """Rebuild from actual frozen artifacts before any expensive construction."""
    require(type(registry) is FrozenRegistry, 'Actual FrozenRegistry required')
    freeze = read_json(registry.freeze_path)
    require(file_sha(registry.freeze_path) == registry.freeze_sha256, 'Changed evaluation freeze')
    artifacts = freeze['artifacts']
    bound = lambda key: tx.bound(registry.root, artifacts[key])
    verified = FrozenRegistry(bound('config'), registry.root, registry.freeze_path,
        [bound(k) for k in ('train_manifest','dry_manifest','dev_manifest','prefix_calib_manifest','main_manifest')],
        bound('inputs'), bound('images'))
    require(verified.config == registry.config and verified.requests == registry.requests,
            'Caller registry differs from actual frozen manifests')
    return verified


def _selection(registry, start, arm, step, scope):
    require(scope in ('DEV','MAIN'), 'Only independently authorized DEV/MAIN')
    selection = dict(loader.registered_endpoint(registry, start, arm, step), scope=scope)
    require(step in ((0,2,8,16) if scope == 'DEV' else (0,8,16,24)), 'Endpoint outside prescribed evaluation scope')
    requests = [copy.deepcopy(row) for row in registry.requests.values()
                if row['scope'] == scope and row.get('start_seed') == start and row['arm'] == arm and row['step'] == step]
    expected_count = 108 if scope == 'DEV' else (1952 if step in (0,24) else 256)
    require(len(requests) == expected_count and len({r['request_id'] for r in requests}) == expected_count,
            'Incomplete/expanded frozen evaluation endpoint')
    for request in requests:
        require(registry.request(request['request_id']) == request, 'Changed evaluation request')
    return selection, requests


def _checkpoint_preflight(registry, selection, lock, native, store, identity, commit_sha):
    """Retain every high-level loader identity gate; never hold the store lock."""
    start,arm,step = (selection[k] for k in ('start_seed','arm','step'))
    source = registry.config['start_bindings'][str(start)]
    if arm == 'START':
        require(all(v is None for v in (store,identity,commit_sha)), 'START cannot substitute a checkpoint')
        policy_sha = source['policy_sha256']
    else:
        require(all(v is not None for v in (store,identity,commit_sha)) and
                identity['arm'] == arm and identity['train_seed'] == start, 'Explicit trained identity/COMMIT required')
        freeze = read_json(registry.freeze_path)
        require(identity.get('design_freeze_sha256') == registry.freeze_sha256 and
                identity.get('config_sha256') == freeze['config_sha256'] and
                re.fullmatch('[0-9a-f]{64}', identity.get('execution_sources_sha256','')) is not None and
                identity.get('source_sha256') == object_sha(lock['native_files']) and
                identity.get('manifest_sha256') == freeze['artifacts']['train_manifest']['sha256'] and
                identity.get('plan_sha256') == freeze['artifacts']['training_groups']['sha256'] and
                identity.get('run_id') == f'{state.EXPERIMENT}/b0-{start}-{arm}',
                'Trained endpoint scientific/source/plan/run identity differs')
        require(type(store) is tx.TransactionStore and store.root == registry.root and
                store.freeze_sha == registry.freeze_sha256 and store.identity == identity and
                identity['source_binding'] == source, 'Foreign trained transaction identity')
        marker = store.inspect_committed(step,commit_sha)
        require(marker['step'] == step and marker['identity'] == identity, 'Trained COMMIT selection differs')
        require(state.validate_checkpoint(store.path/f'step-{step:03d}',commit_sha,identity,native) == marker,
                'Trained COMMIT/state validation differs')
        policy_sha = marker['policy_sha256']
    return {k:v for k,v in selection.items() if k != 'scope'} | dict(
        policy_sha256=policy_sha, reference_sha256=source['reference_sha256'])


def _actual_tensors(underlying):
    import torch
    model = underlying.runtime.model
    trainable = {name:p for name,p in model.named_parameters() if p.requires_grad}
    reference = underlying.fork.reference
    require(len(trainable) == len(reference) == 496 and set(trainable) == set(reference) and
            all('lora_' in name and value.dtype == torch.float32 and not value.requires_grad and
                value.shape == trainable[name].shape and bool(torch.isfinite(value).all()) for name,value in reference.items()),
            'Evaluation requires actual 496 FP32 LoRA/original reference tensors')
    require(all(str(v.device) == 'cuda:0' and (not v.is_floating_point() or v.dtype == torch.float32)
                for v in (*model.parameters(),*model.buffers())) and
            all(p.grad is None for p in model.parameters()), 'Actual evaluation model is not clean FP32 cuda:0')
    return dict(trainable_tensors=496,reference_tensors=496,dtype='torch.float32',device='cuda:0')


def _authorization(registry, selection, endpoint, request_ids_sha, authority, trusted_authority,
                   execution_sha, output, request=None):
    require(isinstance(authority,AuthenticatedEvaluationAuthority), 'Plain evaluation permission is forbidden')
    AuthenticatedEvaluationAuthority.check_source(authority,trusted_authority)
    rid = None if request is None else request['request_id']
    value = authority.verify(root=registry.root,selection=copy.deepcopy(selection),endpoint=copy.deepcopy(endpoint),
        design_freeze_sha256=registry.freeze_sha256,execution_sources_sha256=execution_sha,
        request_ids_sha256=request_ids_sha,request_id=rid)
    AuthenticatedEvaluationAuthority.check_source(authority,trusted_authority)
    freeze = read_json(registry.freeze_path)
    require(isinstance(value,dict) and value.get('schema') == 'sr-satdose-authenticated-evaluation-authority-v1' and
            value.get('selection') == selection and value.get('endpoint') == endpoint and
            value.get('design_freeze_sha256') == registry.freeze_sha256 and value.get('config_sha256') == freeze['config_sha256'] and
            value.get('execution_sources_sha256') == execution_sha and value.get('request_ids_sha256') == request_ids_sha and
            value.get('request_id') == rid, 'Authenticated evaluation scope differs')
    docs = {key:read_json(tx.bound(registry.root,value[key])) for key in ('admission','registration','gpu_report')}
    admission,registration,gpu = (docs[k] for k in ('admission','registration','gpu_report'))
    require(admission.get('schema') == 'sr-satdose-admission-v1' and admission.get('experiment_id') == state.EXPERIMENT and
            admission.get('status') == 'ADMITTED_FIRST_TWO_STEPS_ONLY' and admission.get('permission') == 'FOUR_PATHS_MAX_STEP_2' and
            admission.get('failures') == [] and admission.get('config_sha256') == freeze['config_sha256'] and
            admission.get('registration_sha256') == value['registration']['sha256'] and
            admission.get('gpu_report_sha256') == value['gpu_report']['sha256'], 'Actual admission references differ')
    require(registration.get('schema') == 'sr-satdose-registration-v1' and registration.get('mode') == 'production' and
            registration.get('experiment_id') == state.EXPERIMENT and registration.get('config_sha256') == freeze['config_sha256'] and
            registration.get('artifacts',{}).get('design_freeze') == dict(path='DESIGN_FREEZE.json',sha256=registry.freeze_sha256),
            'Actual registration freeze differs')
    require(gpu.get('schema') == 'sr-satdose-gpu-verification-v1' and gpu.get('mode') == 'production' and
            gpu.get('experiment_id') == state.EXPERIMENT and gpu.get('config_sha256') == freeze['config_sha256'] and
            gpu.get('registration_sha256') == value['registration']['sha256'], 'Actual GPU report references differ')
    if selection['scope'] == 'MAIN':
        tx.bound(registry.root,value['terminal'])  # Semantic verification is the independently pinned authority's obligation.
    if request is None:
        require('begin' not in value, 'Constructor cannot claim an attempted request')
    else:
        expected = output/(hashlib.sha256(rid.encode()).hexdigest()+'.BEGIN.json')
        begin_path = tx.bound(registry.root,value['begin'])
        require(begin_path == expected and not any(expected.with_name(expected.name.replace('.BEGIN.json',suffix)).exists()
                for suffix in ('.CAPTURE.json','.COMPLETE.json','.ERROR.json')), 'Existing/foreign evaluation attempt: no retry')
        begin = read_json(begin_path)
        require(begin.get('request_id') == rid and begin.get('seed') == request['seed'] and begin.get('endpoint') == endpoint and
                begin.get('design_freeze_sha256') == registry.freeze_sha256 and begin.get('status') == 'ATTEMPTED_UNKNOWN',
                'Actual BEGIN identity differs')
    return copy.deepcopy(value)


class EvaluationRuntimeMixin:
    """New native adapter: original preparation/recipe; generation accounting only."""
    def _set_recipe(self):
        super()._set_recipe()
        for key,value in self._satdose_native.generation_recipe().items():
            setattr(self.generation_config,key,value)
        self.identity.update(runtime_recipe='SR-SATDOSE-native-full-sequence-1536',
                             generation_config_expanded=self.generation_config.to_dict())

    def bind_request(self,request):
        require(not self._satdose_closed and self._satdose_active is None and self._evaluation_handle is not None,
                'Closed/unready/nested evaluation request')
        self._evaluation_handle._authorize(request)
        self._satdose_active = copy.deepcopy(request)

    def unbind_request(self):
        require(self._satdose_active is not None,'No active evaluation request')
        self._satdose_active = None

    def prepare_input(self,*args,**kwargs):
        return frozen.RegisteredRuntimeMixin.prepare_input(self,*args,**kwargs)

    def _validate_record_revision(self,record):
        return frozen.RegisteredRuntimeMixin._validate_record_revision(self,record)

    def _validate_prepared_revision(self,prepared):
        return frozen.RegisteredRuntimeMixin._validate_prepared_revision(self,prepared)

    def reserve(self,kind,count,**metadata):
        require(self._satdose_active is not None and kind in ('completion_attempts','generated_tokens'),
                'Evaluation forbids teacher/gradient/unbound accounting')
        return frozen.RegisteredRuntimeMixin.reserve(self,kind,count,**metadata)

    def _batch_forward(self,*args,**kwargs):
        raise PermissionError('Evaluation runtime has no teacher/backward capability')

    def _native_forward(self,*args,**kwargs):
        raise PermissionError('Evaluation runtime has no teacher/backward capability')

    def batch_sequence_forward(self,*args,**kwargs):
        raise PermissionError('Evaluation runtime has no teacher/backward capability')

    def batch_reference_forward(self,*args,**kwargs):
        raise PermissionError('Evaluation runtime has no teacher/backward capability')


class EvaluationHandle:
    """Callable GenerationBridge runner; no optimizer, policy mutator or teacher API."""
    def __init__(self,underlying,selection,endpoint,requests,authority,trusted_authority,sources,source_sha,
                 original_files,source_lock_path,runner):
        self._underlying = underlying; self.registry = underlying.registry
        self.output = underlying.output; self._selection = copy.deepcopy(selection); self._endpoint = copy.deepcopy(endpoint)
        self._requests = {r['request_id']:copy.deepcopy(r) for r in requests}
        self._request_ids_sha = object_sha(list(self._requests))
        self._authority = authority; self._trusted_authority = copy.deepcopy(trusted_authority)
        self._sources = copy.deepcopy(sources); self._source_sha = source_sha; self._files = copy.deepcopy(original_files)
        self._source_lock_path = source_lock_path; self._runner = runner
        self._closed = False; self._poisoned = False; self._attempted = set(); self._generating = False
        self.receipt = None

    def endpoint(self):
        require(not self._closed and not self._poisoned,'Closed/failed evaluation runtime')
        inner = self._underlying; runtime = inner.runtime; native = inner.native_ops
        require(runtime._satdose_closed is False and inner.fork.optimizer is None and inner.fork.scheduler is None and
                runtime.identity['arm'] == self._endpoint['arm'] and runtime.identity['step'] == self._endpoint['step'] and
                runtime.identity['execution_mode'] == 'AUTHENTICATED_EVALUATION' and
                all(p.grad is None for p in runtime.model.parameters()), 'Dirty evaluation runtime')
        require(native.state_hash(native.trainable_state(runtime.model)) == self._endpoint['policy_sha256'] and
                native.state_hash(inner.fork.reference) == self._endpoint['reference_sha256'], 'Evaluation policy/reference changed')
        return copy.deepcopy(self._endpoint)

    def _authorize(self,request):
        require(self._generating and request['request_id'] not in self._attempted and
                self._requests.get(request['request_id']) == request == self.registry.request(request['request_id']),
                'Unregistered/repeated evaluation request')
        _execution_sources(self._authority,self._trusted_authority,self._sources,self._source_sha)
        endpoint = self.endpoint()
        require(request['scope'] == self._selection['scope'] and all(request[k] == endpoint[k]
                for k in ('start_seed','arm','step')), 'Request differs from authorized endpoint')
        evidence = _authorization(self.registry,self._selection,endpoint,self._request_ids_sha,self._authority,
            self._trusted_authority,self._source_sha,self.output,request)
        self._attempted.add(request['request_id'])
        return evidence

    def __call__(self,request,inputs_root,capture):
        require(Path(inputs_root).resolve(strict=True) == self.registry.root and
                self.registry.request(request['request_id']) == request,'GenerationBridge request/root differs')
        return self.generate(request['request_id'],capture)

    def generate(self,request_id,capture):
        require(callable(capture) and not self._generating and request_id in self._requests,'Invalid evaluation generation call')
        self.endpoint(); inner = self._underlying; native = inner.native_ops
        rng = native.capture_rng(); rng_sha = native.state_hash(rng)
        self._generating = True
        try:
            with inner.runtime.torch.no_grad():
                result = self._runner(copy.deepcopy(self._requests[request_id]),self.registry.root,capture)
            require(inner.runtime._satdose_active is None,'Native runner failed to unbind')
            self.endpoint()
            return result
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._generating = False
            if native.state_hash(native.capture_rng()) != rng_sha:native.restore_rng(rng)
            require(native.state_hash(native.capture_rng()) == rng_sha,'Caller evaluation draw RNG changed')

    def audit(self):
        endpoint = self.endpoint()
        for path,sha in self._files.items():require(file_sha(path) == sha,'Original evaluation source changed')
        _execution_sources(self._authority,self._trusted_authority,self._sources,self._source_sha)
        events = copy.deepcopy(self._underlying.runtime._satdose_accounting)
        require(all(event['kind'] in ('completion_attempts','generated_tokens') for event in events),'Teacher/update evaluation work detected')
        return dict(endpoint,scope=self._selection['scope'],request_ids_sha256=self._request_ids_sha,
            attempted_requests=len(self._attempted),accounting=events,optimizer_steps=0,scheduler_steps=0,
            parameter_updates=0,training_rng_restored=False,permission='EVALUATION_ONLY_NO_TRAINING_PERMISSION')

    def close(self):
        require(not self._closed,'Evaluation runtime already closed')
        try:
            return dict(self.audit(),closed=True)
        finally:
            self._underlying.runtime._satdose_closed = True
            self._underlying.runtime._vision_hook.remove()
            self._closed = True


def build_evaluation_runtime(registry,source_lock_path,*,start_seed,arm,step,scope,output_dir,
        authority,authority_source_reference,execution_sources,expected_execution_sources_sha256,
        commit_store=None,expected_identity=None,expected_commit_sha256=None):
    """Authorized real native construction. No default authority/CLI/BEGIN/retry.

    output_dir is also GenerationBridge's run_dir; existing generated attempts
    are not resumed. Never externally hold TransactionStore's lock here: its
    inspect calls acquire their own lock. Only the future driver's distinct
    global control lock may span this operation.
    """
    _execution_sources(authority,authority_source_reference,execution_sources,expected_execution_sources_sha256)
    registry = _registry(registry)
    selection,requests = _selection(registry,start_seed,arm,step,scope)
    output = Path(output_dir)
    require(output.is_absolute() and output.resolve(strict=True) == output and output.is_dir() and
            output.is_relative_to(registry.root) and not any(output.iterdir()), 'New empty canonical evaluation/ledger directory required')
    loader_sources = {key:execution_sources[key] for key in ('evaluation_loader','training_state','training_transaction')}
    lock,native_root,legacy_root,model,parent = loader.verify_sources(registry,source_lock_path,loader_sources,object_sha(loader_sources))
    ops = fork_loader.load_native_ops(native_root,lock['native_files'])
    rng = ops.capture_rng(); rng_sha = ops.state_hash(rng)
    runtime = result = None
    try:
        endpoint = _checkpoint_preflight(registry,selection,lock,ops,commit_store,expected_identity,expected_commit_sha256)
        authorization = _authorization(registry,selection,endpoint,object_sha([r['request_id'] for r in requests]),
            authority,authority_source_reference,expected_execution_sources_sha256,output)
        source = registry.config['start_bindings'][str(start_seed)]
        originals = {source['original_state_path']:source['state_sha256']}
        originals.update({str(parent/name):sha for name,sha in lock['parent_files'].items()})
        for path,sha in originals.items():require(file_sha(path) == sha,'Original checkpoint/parent source differs')
        sys.path[:0] = [str(native_root),str(legacy_root)]  # Native lazy imports need these SHA-verified roots.
        native = frozen._import('sr_f6.runtime',native_root)
        base = frozen._import('sr_f5.runtime',native_root).Runtime
        legacy = frozen._import('gpu_production_v2.backend',legacy_root)
        storage = frozen._import('diagnostics.gpu_capacity_v2',legacy_root)
        protocol = frozen._import('sr_f1.json_protocol',native_root)
        configure = frozen._import('mm_dev.runtime',native_root).configure_audited_backend
        allowed = {str(native_root/rel) for rel in lock['native_files']}
        namespaces = {Path(rel).parts[0].removesuffix('.py') for rel in lock['native_files']}
        for name,module in tuple(sys.modules.items()):
            if getattr(module,'__file__',None) and name.split('.')[0] in namespaces:
                require(str(Path(module.__file__).resolve()) in allowed,'Shadowed actual native module: '+name)
        configure(); storage.validate_gpu_pair()
        class Runtime(EvaluationRuntimeMixin,base):
            _satdose_native=native; _satdose_legacy=legacy; _satdose_registry=registry
            _satdose_start=start_seed; _satdose_output=output
            _satdose_active=None; _satdose_closed=False; _evaluation_handle=None
        Runtime._satdose_accounting = []
        native.seed_all(native.seed_for(2026110301,'load','b0'))
        runtime = Runtime(str(model),adapter_path=str(parent),output_root=output,protocol_revision='v2',account=None)
        native.configure_parent_training(runtime,parent_adapter_config=read_json(parent/'adapter_config.json'))
        original_dtype = runtime.identity['dtype']; runtime.model = runtime.model.float(); runtime.model.eval()
        runtime._srf5_numerics = dict(native.NUMERICS_V2)
        runtime.identity.update(dtype='float32',input_float_dtype='float32',probability_dtype='float32',
            model_cast_from_dtype=original_dtype,forward_parameter_dtypes=['float32'],forward_float_buffer_dtypes=['float32'],
            numerics=dict(native.NUMERICS_V2),arm='START',step=0,manifest_sha256=registry.freeze_sha256,
            source_sha256=object_sha(lock['native_files']))
        fork = fork_loader.fork_readonly(runtime,registry.config,start_seed,ops)
        underlying = SimpleNamespace(runtime=runtime,fork=fork,native_ops=ops,registry=registry,start_seed=start_seed,output=output)
        loader._clean_runtime(underlying,source['policy_sha256'])
        loaded = (dict(policy_sha256=source['policy_sha256'],reference_sha256=source['reference_sha256'],
                       optimizer_constructed=False,training_rng_restored=False,model_calls=0)
                  if arm == 'START' else loader.load_committed_policy(underlying,commit_store,step=step,
                      expected_commit_sha256=expected_commit_sha256,expected_identity=expected_identity))
        require(loaded['policy_sha256'] == endpoint['policy_sha256'] and not runtime._satdose_accounting,
                'Loaded evaluation endpoint differs or construction performed model work')
        actual_tensors = _actual_tensors(underlying)
        if arm != 'START':
            originals[str(commit_store.path/f'step-{step:03d}'/'state.pt')] = loaded['state_sha256']
            originals[str(commit_store.path/f'step-{step:03d}'/'COMMIT.json')] = expected_commit_sha256
        # A NEW runtime, never an EvaluationLoadHandle. Its execution identity is
        # installed only after authenticated policy-only loading succeeds.
        runtime.identity.update(arm=arm,step=step,execution_mode='AUTHENTICATED_EVALUATION',
            run_root=str(registry.root),audit_root=str(output),
            run_id=f'{state.EXPERIMENT}/b0-{start_seed}-{arm}')
        sources = {str(native_root/rel):sha for rel,sha in lock['native_files'].items()}
        sources.update({str(legacy_root/rel):sha for rel,sha in lock['legacy_files'].items()})
        runner = NativeKernelRunner(runtime,native,legacy,protocol.balanced_cut,source_hashes=sources)
        result = EvaluationHandle(underlying,selection,endpoint,requests,authority,authority_source_reference,
            execution_sources,expected_execution_sources_sha256,originals,source_lock_path,runner)
        runtime._evaluation_handle = result
        result.audit()
        _authorization(registry,selection,result.endpoint(),result._request_ids_sha,authority,
            authority_source_reference,expected_execution_sources_sha256,output)
        result.receipt = dict(schema='sr-satdose-evaluation-runtime-v1',selection=selection,endpoint=endpoint,
            design_freeze_sha256=registry.freeze_sha256,config_sha256=read_json(registry.freeze_path)['config_sha256'],
            source_lock_sha256=file_sha(source_lock_path),execution_sources_sha256=expected_execution_sources_sha256,
            authority_source=copy.deepcopy(authority_source_reference),authorization=authorization,
            request_ids_sha256=result._request_ids_sha,requests=len(requests),loaded=loaded,
            actual_tensors=actual_tensors,
            caller_rng_sha256=rng_sha,optimizer_constructed=False,scheduler_constructed=False,
            training_rng_restored=False,permission='EVALUATION_RUNTIME_ONLY_NO_TRAINING_ADMISSION')
        if ops.state_hash(ops.capture_rng()) != rng_sha:ops.restore_rng(rng)
        require(ops.state_hash(ops.capture_rng()) == rng_sha,'Caller load RNG preservation failed')
        write_once(output/'EVALUATION_RUNTIME.json',result.receipt)
        return result
    except BaseException:
        if runtime is not None:
            runtime._satdose_closed = True
            runtime._vision_hook.remove()
        raise
    finally:
        if ops.state_hash(ops.capture_rng()) != rng_sha:ops.restore_rng(rng)
        require(ops.state_hash(ops.capture_rng()) == rng_sha,'Caller load RNG preservation failed')
