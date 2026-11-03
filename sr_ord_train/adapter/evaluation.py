"""Post-training endpoint registration and read-only native evaluation.

register_evaluation(package, out, code_commit, worker_shards=1) acquires all 12
completed path locks and verifies all published checkpoints before opening the
formal evaluation manifest. It writes ENDPOINT_LOCK.json first, then registers
26 endpoints / 26,624 immutable requests and a separate EVAL_BINDING.json. The
original RUN_BINDING and training implementation remain unchanged.

run_endpoint(binding_path, endpoint_id, shard_index=0) loads only the registered
parent or step-12/24 adapter, creates no optimizer, and emits one immutable raw
file per registered request. Unknown attempts block automatic retries. Neither
importing this module nor running its CPU unit tests opens the real eval manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import copy
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

from adapter.backend import (SNAPSHOT, bind_native_kernel, canonical_directory,
    failure_safe, file_sha, object_sha, prefix_helpers, prepare_public_input,
    read_json, text_sha, verify_file, write_once)
from adapter.transactions import PathTransactions

EXPERIMENT = 'SR-ORD-TRAIN-1'
KEY = ('panel','root_id','structure','parent','train_seed','arm','step','view','draw')
PARENTS = ('b0','b1')
SEEDS = (11031,11032)
ARMS = ('R','G','D')
PUBLIC = ('prompt','image_file','image_sha256')
PUBLIC_WITH_PREFIX = (*PUBLIC,'assistant_prefill')


def require(ok,reason):
    if not ok:
        raise PermissionError(reason)


def load_rows(path):
    def pairs(values):
        require(len(dict(values)) == len(values),'Duplicate JSON key')
        return dict(values)
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    with Path(path).open() as f:
        return [json.loads(line,object_pairs_hook=pairs,parse_constant=invalid)
                for line in f if line.strip()]


def indexed(rows,key):
    result={r[key]:r for r in rows}
    require(len(result)==len(rows),'Duplicate '+key)
    return result


def endpoint_id(parent,arm,train_seed,step):
    require(parent in PARENTS,'Unknown parent')
    if arm == 'PARENT':
        require(train_seed is None and step == 0,'Parent must be evaluated once with null training seed')
        return parent+'-PARENT'
    require(arm in ARMS and type(train_seed) is int and train_seed in SEEDS
            and type(step) is int and step in (12,24),'Unknown trained endpoint')
    return f'{parent}-{train_seed}-{arm}-step-{step:03d}'


def expected_paths():
    return [(p,s,a,f'{p}-{s}-{a}') for p in PARENTS for s in SEEDS for a in ARMS]


def training_identity(run_binding,parent,seed,arm,plan_sha):
    """Exact identity written by driver.train_path, including implementation."""
    name=f'{parent}-{seed}-{arm}'
    return dict(experiment_id=EXPERIMENT,parent=parent,arm=arm,train_seed=seed,
        run_id=EXPERIMENT+'/'+name,manifest_sha256=run_binding['manifest_sha256'],
        source_sha256=run_binding['source_sha256'],plan_sha256=plan_sha,
        implementation_sha256=object_sha(run_binding['implementation_files']),
        code_commit=run_binding['code_commit'])


def validate_path_receipt(directory,identity,markers,run_binding_path,expected_plan_sha):
    """Accept only the independently replayed seal, with externally pinned plan."""
    receipt_path=directory/'PATH_RECEIPT.json'
    require(receipt_path.is_file() and not receipt_path.is_symlink(),
            'Missing atomic accepted PATH_RECEIPT: '+str(directory))
    receipt=read_json(receipt_path)
    require(receipt.get('seal_status')=='PATH_NATIVE_REPLAY_AND_RECEIPT_VERIFIED',
            'Path has not passed independent native replay and receipt verification')
    require(receipt.get('identity')==identity,'Sealed path identity differs')
    require(receipt['evidence']['native_binding']['sha256']==file_sha(run_binding_path),
            'Path seal uses a different original RUN_BINDING')
    require(receipt['parent_tensor_sha256']==markers[0]['policy_sha256'] and
            receipt['reference_tensor_sha256']==markers[0]['reference_sha256'],
            'Path seal parent/reference differs from actual commit')
    for step,key in ((0,'zero_checkpoint'),(12,'checkpoint_12'),(24,'checkpoint_24')):
        require(receipt['evidence'][key]['sha256']==markers[step]['state_sha256'],
                'Path seal checkpoint bytes differ from actual commit')
        if step:
            require(receipt['evaluation_policy_sha256'][str(step)]==markers[step]['policy_sha256'],
                    'Path seal endpoint tensor differs from actual commit')
    from runtime_contract import check_path,checked_refs
    checked_refs(directory,{'native_replay':receipt['native_replay']},('native_replay',))
    replay=read_json(directory/receipt['native_replay']['path'])
    require(replay['status']=='INDEPENDENT_NATIVE_REPLAY_PASS' and replay['responses']==3072,
            'Independent replay denominator/status differs')
    result=check_path(receipt,identity['manifest_sha256'],directory,expected_plan_sha)
    require(result['status']=='PATH_RECEIPT_STRUCTURE_AND_ARTIFACT_DIGESTS_VERIFIED' and
            result['steps']==24 and result['responses']==3072,
            'Standard path receipt check did not pass')
    return dict(path=str(receipt_path),sha256=file_sha(receipt_path))


@contextmanager
def completed_paths(package,run_binding):
    """All locks remain held through formal-manifest registration."""
    package=canonical_directory(package)
    index=read_json(package/'frozen/receipt_plans/INDEX.json')
    with ExitStack() as stack:
        result=[]
        for parent,seed,arm,name in expected_paths():
            directory=package/'run/paths'/name
            # Do not let PathTransactions manufacture an absent path or lock.
            for relative in ('IDENTITY.json','LOCK','ATTEMPTS.jsonl','checkpoints/step-024/COMMIT.json'):
                require((directory/relative).is_file() and not (directory/relative).is_symlink(),
                        'Training incomplete: '+name+'/'+relative)
            identity=read_json(directory/'IDENTITY.json')
            expected=training_identity(run_binding,parent,seed,arm,index['files']['paths/'+name+'.json'])
            require(identity==expected,'Training path identity differs: '+name)
            tx=stack.enter_context(PathTransactions(directory,identity))
            markers=tx.published()
            require([m['step'] for m in markers]==list(range(25)),
                    'Each path must have exactly 0..24 published commits: '+name)
            require(all('complete' in a for a in tx._attempts.values()),
                    'UNKNOWN training attempt blocks evaluation: '+name)
            require(len(tx._attempts)==3072,'Training response denominator differs: '+name)
            from runtime_contract import check_numerics
            for marker in markers[1:]:
                update=read_json(directory/'checkpoints'/f"step-{marker['step']:03d}"/'UPDATE.json')
                require(update.get('status')=='UPDATED_UNCOMMITTED' and
                    update.get('precommit_gate')=='PASS_BEFORE_ADAM' and
                    update.get('optimizer_updates')==1 and update.get('sequence_count')==128 and
                    update.get('step')==marker['step'],'Missing actual gated update: '+name)
                check_numerics(update['numerics'])
            require(markers[-1]['sampler_state']['generated_responses']==3072,
                    'Final sampler cursor differs: '+name)
            validate_path_receipt(directory,identity,markers,package/'execution/RUN_BINDING.json',
                                  index['files']['paths/'+name+'.json'])
            result.append((directory,identity,markers))
        yield result


def endpoint_descriptors(paths):
    """Build the immutable endpoint identities from fully verified commit chains."""
    endpoints={}
    for directory,identity,markers in paths:
        parent=identity['parent']
        first=markers[0]
        parent_id=endpoint_id(parent,'PARENT',None,0)
        descriptor=dict(endpoint_id=parent_id,parent=parent,arm='PARENT',train_seed=None,step=0,
            policy_sha256=first['policy_sha256'],reference_sha256=first['reference_sha256'])
        require(descriptor['policy_sha256']==descriptor['reference_sha256'],'Step zero not corresponding parent')
        require(parent_id not in endpoints or endpoints[parent_id]==descriptor,
                'Parent tensor identity differs across paired training paths')
        endpoints[parent_id]=descriptor
        for step in (12,24):
            marker=markers[step]
            eid=endpoint_id(parent,identity['arm'],identity['train_seed'],step)
            folder=directory/'checkpoints'/f'step-{step:03d}'
            endpoints[eid]=dict(endpoint_id=eid,parent=parent,arm=identity['arm'],train_seed=identity['train_seed'],
                step=step,identity=identity,policy_sha256=marker['policy_sha256'],
                reference_sha256=marker['reference_sha256'],
                checkpoint=dict(path=str(folder/'state.pt'),sha256=marker['state_sha256']),
                commit=dict(path=str(folder/'COMMIT.json'),sha256=marker['commit_sha256']),
                update=dict(path=str(folder/'UPDATE.json'),sha256=marker['update_sha256']))
    require(len(endpoints)==26,'All 26 endpoints required')
    return endpoints


def resolve_public(package,public,images,legacy):
    """Map original frozen image references without changing prompt or prefill."""
    original=public['image_file']
    rel=PurePosixPath(original)
    require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe frozen image path')
    if original.startswith('legacy/'):
        require(original in legacy,'Missing original retention image binding')
        relative='run/bound_assets/'+original
        expected=legacy[original]['sha256']
    else:
        require(original.startswith('images/') and original in images,'Unknown new frozen image')
        require(images[original]['legacy_original_required'] is False,'Legacy image substituted')
        relative='frozen/'+original
        expected=images[original]['png_sha256']
    path=package/relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(package),
            'Image escapes experiment root')
    require(file_sha(path)==expected,'Original image bytes differ')
    return dict(prompt=public['prompt'],assistant_prefill=public['assistant_prefill'],
        prompt_sha256=text_sha(public['prompt']),prefill_sha256=text_sha(public['assistant_prefill']),
        image_file=relative,image_sha256=expected)


def request_row(row,public,endpoint,qid):
    require(all(k in row for k in (*KEY,'case_id','input_id','seed','U_applicable')),
            'Incomplete frozen endpoint request')
    require(type(row['seed']) is int and 0<=row['seed']<2**63,'Invalid evaluation draw seed')
    eid=endpoint_id(row['parent'],row['arm'],row['train_seed'],row['step'])
    require(eid==endpoint['endpoint_id'],'Endpoint identity differs')
    rid='eval-'+object_sha({k:row[k] for k in KEY})
    return {**{k:row[k] for k in (*KEY,'case_id','input_id','seed','U_applicable')},
            'request_id':rid,'qid':qid,'scope':'EVALUATION','endpoint_id':eid,
            'policy_sha256':endpoint['policy_sha256'],**public}


def register_evaluation(package,out,code_commit,worker_shards=1):
    require(re.fullmatch('[a-f0-9]{40}',code_commit) is not None,'Actual new evaluation code commit required')
    require(type(worker_shards) is int and 1<=worker_shards<=8,'Invalid fixed worker partition')
    package=canonical_directory(package)
    out=Path(out)
    require(out.is_absolute() and out.parent.resolve().is_relative_to(package),
            'Evaluation registration must stay in experiment root')
    require(not out.exists(),'Evaluation registration is immutable')
    run_path=package/'execution/RUN_BINDING.json'
    run_binding=read_json(run_path)
    require(run_binding['experiment_id']==EXPERIMENT and run_binding['package_root']==str(package),
            'Wrong original run binding')
    require(file_sha(package/'frozen/FREEZE.json')==run_binding['manifest_sha256'],'Frozen manifest changed')
    with completed_paths(package,run_binding) as paths:
        endpoints=endpoint_descriptors(paths)
        out.mkdir(parents=True)
        lock=dict(schema='sr-ord-endpoint-lock-v1',experiment_id=EXPERIMENT,
            manifest_sha256=run_binding['manifest_sha256'],endpoints=endpoints,
            verified_paths=12,optimizer_updates=288,training_responses=36864,
            path_receipts=[dict(path=str(directory/'PATH_RECEIPT.json'),
                               sha256=file_sha(directory/'PATH_RECEIPT.json')) for directory,_,_ in paths],
            original_run_binding=dict(path=str(run_path),sha256=file_sha(run_path)),
            formal_manifest_opened_when_lock_written=False)
        write_once(out/'ENDPOINT_LOCK.json',lock)
        # No formal evaluation manifest reads occur above this line.
        freeze=read_json(package/'frozen/FREEZE.json')
        manifest=package/'frozen/evaluation_manifest.jsonl'
        require(file_sha(manifest)==freeze['files']['evaluation_manifest.jsonl'],'Formal manifest bytes differ')
        rows=load_rows(manifest)
        from analyze import index_rows,verify_design
        index_rows(rows);verify_design(rows,'retention8')
        require(len(rows)==26624,'Formal response budget differs')
        for name in ('inputs','images','private_cases'):
            require(file_sha(package/'frozen'/f'{name}.jsonl')==freeze['files'][name+'.jsonl'],
                    'Frozen support bytes differ: '+name)
        inputs=indexed(load_rows(package/'frozen/inputs.jsonl'),'input_id')
        images=indexed(load_rows(package/'frozen/images.jsonl'),'image_file')
        cases=indexed(load_rows(package/'frozen/private_cases.jsonl'),'case_id')
        legacy_path=package/'run/bound_assets/LEGACY_BINDING.json'
        require(file_sha(legacy_path)==run_binding['legacy_binding']['sha256'],'Historical retention binding changed')
        bound=read_json(legacy_path)
        require(bound['status']=='ORIGINAL_DEV_EVAL_P0_V3_BOUND' and bound['rows']==96,'Retention binding incomplete')
        legacy=indexed(bound['files'],'relative')
        requests=[];cached={}
        for row in rows:
            inp=inputs[row['input_id']]
            case=cases[row['case_id']]
            require(case['input_id']==row['input_id'] and case['task']['root_id']==row['root_id'],
                    'Frozen case/input/root mismatch')
            if row['input_id'] not in cached:
                cached[row['input_id']]=resolve_public(package,inp,images,legacy)
            ep=endpoints[endpoint_id(row['parent'],row['arm'],row['train_seed'],row['step'])]
            requests.append(request_row(row,cached[row['input_id']],ep,case['task']['qid']))
        require(len({r['request_id'] for r in requests})==26624,'Duplicate evaluation request')
        require(Counter(r['endpoint_id'] for r in requests)==Counter({k:1024 for k in endpoints}),
                'Endpoint denominator differs')
        request_path=out/'REQUESTS.jsonl'
        with request_path.open('x') as f:
            for r in requests:
                f.write(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        binding=dict(schema='sr-ord-eval-binding-v1',experiment_id=EXPERIMENT,code_commit=code_commit,
            package_root=str(package),original_run_binding=lock['original_run_binding'],
            endpoint_lock=dict(path=str(out/'ENDPOINT_LOCK.json'),sha256=file_sha(out/'ENDPOINT_LOCK.json')),
            requests_manifest=dict(path=str(request_path),sha256=file_sha(request_path)),
            evaluation_manifest=dict(path=str(manifest),sha256=file_sha(manifest)),
            implementation_files=[dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__))],
            worker_shards=worker_shards,endpoint_count=26,registered_responses=26624,
            optimizer_created=False,parameter_updates=0,status='REGISTERED_NOT_GENERATED')
        write_once(out/'EVAL_BINDING.json',binding)
        return binding


def verify_eval_binding(path):
    binding=read_json(path)
    require(binding['schema']=='sr-ord-eval-binding-v1' and binding['experiment_id']==EXPERIMENT,
            'Wrong evaluation registration')
    package=canonical_directory(binding['package_root'])
    original=read_json(verify_file(binding['original_run_binding']))
    require(original['package_root']==str(package),'Original package root differs')
    for ref in [*original['implementation_files'],*binding['implementation_files']]: verify_file(ref)
    require(str(Path(__file__).resolve()) in {r['path'] for r in binding['implementation_files']},
            'Actual evaluation implementation is not bound')
    lock=read_json(verify_file(binding['endpoint_lock']))
    require(lock['verified_paths']==12 and lock['optimizer_updates']==288 and lock['training_responses']==36864
            and len(lock['endpoints'])==26,'Incomplete endpoint lock')
    require(lock['original_run_binding']==binding['original_run_binding'],'Endpoint lock ancestry differs')
    require(len(lock['path_receipts'])==12,'Endpoint lock lacks all accepted path receipts')
    for ref in lock['path_receipts']:verify_file(ref)
    verify_file(binding['evaluation_manifest'])
    requests=load_rows(verify_file(binding['requests_manifest']))
    require(len(requests)==26624 and len({r['request_id'] for r in requests})==26624,'Registered request coverage differs')
    require(Counter(r['endpoint_id'] for r in requests)==Counter({k:1024 for k in lock['endpoints']}),
            'Registered endpoint coverage differs')
    return binding,original,lock,requests


class EvaluationBackend:
    """New read-only evaluation identity; no old stage loader or optimizer."""
    def __init__(self,binding_path,endpoint,output):
        self.binding,self.original,self.lock,requests=verify_eval_binding(binding_path)
        require(endpoint in self.lock['endpoints'],'Unknown endpoint')
        self.endpoint=self.lock['endpoints'][endpoint]
        self.requests={r['request_id']:r for r in requests if r['endpoint_id']==endpoint}
        self.package=canonical_directory(self.binding['package_root'])
        self.output=canonical_directory(output)
        require(self.output.is_relative_to(self.package/'run'),'Output outside independent run')
        native_root=canonical_directory(self.original['native_code_root'])
        sources=self.original['source_files']
        require(object_sha(sources)==self.original['source_sha256'],'Native inventory digest differs')
        actual={p for p in (native_root/'src').rglob('*.py') if p.is_file()}
        require({verify_file(r) for r in sources}==actual,'Native source membership changed')
        history=read_json(verify_file(dict(path=self.original['historical_source_binding'],
                                         sha256=self.original['historical_source_binding_sha256'])))
        model=self.original['model'];parent=self.endpoint['parent']
        base=canonical_directory(model['model_path']);adapter=canonical_directory(model['parents'][parent])
        require(model['snapshot']==SNAPSHOT and base.name==SNAPSHOT,'Snapshot differs')
        require(history['parents'][parent]['path']==str(adapter),'Parent source differs')
        from runtime_contract import PARENT_FILES,PINS
        for name,hx in zip(('adapter_config.json','adapter_model.safetensors'),PARENT_FILES[parent]):
            require(file_sha(adapter/name)==hx,'Parent bytes differ')
        for relative,hx in PINS.items():require(file_sha(native_root/'src'/relative)==hx,'Native kernel differs')
        require({str(p) for p in base.rglob('*') if p.is_file()}=={r['path'] for r in history['model_files']},
                'Snapshot inventory differs')
        for ref in history['model_files']:require(file_sha(ref['path'])==ref['sha256'],'Model/tokenizer bytes differ')
        sys.path.insert(0,str(native_root/'src'))
        native=importlib.import_module('sr_f6.runtime')
        require(Path(native.__file__).resolve()==native_root/'src/sr_f6/runtime.py','Shadowed native runtime')
        self.native=native
        import torch
        from mm_dev.runtime import configure_audited_backend,actual_cuda_identity
        from sr_f5.runtime import Runtime as FP32Runtime
        configure_audited_backend()
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One visible CUDA GPU required')
        self.torch=torch;self._active=None;self._accounting=[];self._closed=False
        backend=self
        class EndpointRuntime(FP32Runtime):
            def _set_recipe(self):
                super()._set_recipe()
                for key,value in native.generation_recipe().items():setattr(self.generation_config,key,value)
            def prepare_input(self,model_input,run_root,*,protocol='structured'):
                require(backend._active is not None and protocol=='structured' and Path(run_root).resolve()==backend.package,
                        'No registered active endpoint request')
                require(model_input=={k:backend._active[k] for k in PUBLIC},'Public endpoint payload differs')
                prepared=prepare_public_input(self,backend._active,backend.package)
                self._srf5_prepared=prepared
                return prepared
            def reserve(self,kind,count,**metadata):
                require(kind in ('completion_attempts','generated_tokens'),'Extra model forwards prohibited in final evaluation')
                event=dict(kind=kind,count=count,metadata=metadata)
                backend._accounting.append(event)
                with (backend.output/'accounting.jsonl').open('a') as f:
                    f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        native.seed_all(native.seed_for(2026110301,'load',parent))
        runtime=EndpointRuntime(str(base),adapter_path=str(adapter),output_root=self.output,protocol_revision='v2',account=None)
        native.configure_parent_training(runtime,parent_adapter_config=read_json(adapter/'adapter_config.json'))
        original_dtype=runtime.identity['dtype'];runtime.model=runtime.model.float()
        self.names=tuple(n for n,p in runtime.model.named_parameters() if p.requires_grad)
        require(self.names,'No inherited LoRA tensors')
        parent_hash=self.policy_hash(runtime)
        require(parent_hash==self.endpoint['reference_sha256'],'Loaded parent tensor identity differs')
        self._load_endpoint(runtime)
        for p in runtime.model.parameters():p.requires_grad_(False);p.grad=None
        runtime.model.eval()
        require(all(not v.is_floating_point() or v.dtype==torch.float32 for v in (*runtime.model.parameters(),*runtime.model.buffers())),
                'Endpoint is not wholly FP32')
        runtime._srf5_numerics=dict(native.NUMERICS_V2)
        runtime.identity.update(experiment_id=EXPERIMENT,execution_mode='EVALUATION',run_id=EXPERIMENT+'/EVAL/'+endpoint,
            parent_id=parent,step=self.endpoint['step'],dtype='float32',input_float_dtype='float32',
            probability_dtype='float32',model_cast_from_dtype=original_dtype,numerics=dict(native.NUMERICS_V2),
            forward_parameter_dtypes=['float32'],forward_float_buffer_dtypes=['float32'],snapshot=SNAPSHOT)
        self.runtime=runtime
        self.versions=self._versions()
        self.loaded=dict(runtime.identity,endpoint=copy.deepcopy(self.endpoint),hardware=actual_cuda_identity(),
            evaluation_binding_sha256=file_sha(binding_path),policy_sha256=self.policy_hash(runtime),
            parent_tensor_sha256=parent_hash,trainable_parameters=[],optimizer_created=False,parameter_updates=0)
        write_once(self.output/'LOADED_IDENTITY.json',self.loaded)

    def policy_hash(self,runtime=None):
        runtime=runtime or self.runtime
        return self.native.state_hash({n:p.detach() for n,p in runtime.model.named_parameters() if n in set(self.names)})

    def _load_endpoint(self,runtime):
        ep=self.endpoint
        if ep['arm']=='PARENT':
            require(self.policy_hash(runtime)==ep['policy_sha256'],'Parent endpoint tensor differs')
            return
        path=verify_file(ep['checkpoint']);commit=read_json(verify_file(ep['commit']));verify_file(ep['update'])
        require(commit['identity']==ep['identity'] and commit['step']==ep['step'] and
                commit['policy_sha256']==ep['policy_sha256'] and commit['state_sha256']==ep['checkpoint']['sha256'],
                'Committed endpoint binding differs')
        state=self.torch.load(path,map_location='cpu',weights_only=True)
        require(file_sha(path)==ep['checkpoint']['sha256'],'Checkpoint changed while loading')
        require(state['identity']==ep['identity'] and state['step']==ep['step'] and
                state['reference_sha256']==ep['reference_sha256'] and
                self.native.state_hash(state['policy'])==ep['policy_sha256'],'Serialized endpoint differs')
        parameters=dict(runtime.model.named_parameters())
        require(set(state['policy'])==set(self.names),'Endpoint LoRA inventory differs')
        with self.torch.no_grad():
            for name,tensor in state['policy'].items():
                require(tensor.dtype==self.torch.float32 and tensor.shape==parameters[name].shape and
                        bool(self.torch.isfinite(tensor).all()),'Invalid endpoint LoRA tensor')
                parameters[name].copy_(tensor)
        require(self.policy_hash(runtime)==ep['policy_sha256'],'Actually loaded endpoint tensor differs')
        # Optimizer/RNG payloads are never restored in evaluation.
        del state

    def _versions(self):
        return {kind+':'+n:(id(v),v._version) for kind,seq in
                [('p',self.runtime.model.named_parameters()),('b',self.runtime.model.named_buffers())] for n,v in seq}

    def _readonly(self):
        require(not self._closed and not self.runtime.model.training,'Endpoint closed or in training mode')
        require(self._versions()==self.versions,'Endpoint tensor versions changed')
        require(all(not p.requires_grad and p.grad is None for p in self.runtime.model.parameters()),'Endpoint acquired gradients')

    def generate_one(self,request):
        self._readonly()
        require(self._active is None and self.requests.get(request.get('request_id'))==request,'Unregistered endpoint request')
        require(request['policy_sha256']==self.endpoint['policy_sha256'],'Request endpoint policy changed')
        image=self.package/request['image_file']
        require(not image.is_symlink() and image.resolve(strict=True).is_relative_to(self.package) and
                file_sha(image)==request['image_sha256'],'Endpoint image bytes changed')
        require(text_sha(request['prompt'])==request['prompt_sha256'] and
                text_sha(request['assistant_prefill'])==request['prefill_sha256'],'Endpoint public text changed')
        key=text_sha(request['request_id']);attempt=self.output/'attempts'/key
        write_once(attempt.with_suffix('.BEGIN.json'),dict(request=request,status='ATTEMPTED_UNKNOWN'))
        self._active=copy.deepcopy(request);self._accounting=[];captured=[];error=None
        from sr_f1.json_protocol import balanced_cut
        helpers=prefix_helpers(request['assistant_prefill'],self.native.decode_generated,self.native.BatchedProtocolStop,balanced_cut)
        kernel=bind_native_kernel(self.native.Runtime._generate_native_group,helpers)
        def capture(row):
            captured.append(row);write_once(attempt.with_suffix('.CAPTURE.json'),failure_safe(row))
        try:
            with self.native.independent_draw_rng(self.torch,[request['seed']]):
                kernel(self.runtime,{k:request[k] for k in PUBLIC},self.package,[request['seed']],
                       protocol='structured',on_completion=capture)
        except Exception as exc:
            error=repr(exc)
            write_once(attempt.with_suffix('.ERROR.json'),dict(error=error,captured=len(captured),accounting=self._accounting))
            if not captured:raise
        finally:self._active=None
        require(len(captured)==1,'Native response count differs')
        row=dict(captured[0]);prepared=self.runtime._srf5_prepared;route=prepared['routing']
        row.update({k:request[k] for k in (*KEY,'case_id','input_id','qid','seed','U_applicable','request_id','endpoint_id')})
        row.update(row_id=request['request_id'],protocol='structured_v4',backend_protocol='structured',
            native_capture_path=str(attempt.with_suffix('.CAPTURE.json')),
            native_capture_sha256=file_sha(attempt.with_suffix('.CAPTURE.json')),
            protocol_revision='v2',policy_hash=self.endpoint['policy_sha256'],
            source_sha256=self.original['source_sha256'],reference_hash=self.endpoint['reference_sha256'],
            finished=row['finish_reason']=='balanced',
            prompt_sha256=request['prompt_sha256'],prefill_sha256=request['prefill_sha256'],
            image_sha256=request['image_sha256'],assistant_prefill=request['assistant_prefill'],
            suffix_text=row['generated_text'],suffix_token_ids=row['tokens'],
            prefill_token_count=len(route['assistant_prefill_token_ids']),generated_tokens=len(row['tokens']),
            seed_semantics='independent_per_draw_torch_generator',
            native_prompt_inputs={k:v.detach().cpu().tolist() for k,v in prepared['inputs'].items() if k!='pixel_values' and hasattr(v,'detach')},
            prefill_boundary=dict(base_prompt_token_count=route['base_prompt_token_count'],
                prefill_token_ids=route['assistant_prefill_token_ids'],native_char_span=route['assistant_prefill_native_char_span'],
                native_offsets=route['assistant_prefill_native_offsets'],same_assistant_message=True),
            model_identity=self.loaded,accounting=copy.deepcopy(self._accounting),native_exception=error,
            optimizer_updates=0,parameter_updates=0)
        require(row['full_text']==request['assistant_prefill']+row['suffix_text'],'Full prefix/suffix reconstruction differs')
        from adapter.training import trajectory_hash
        try:row['token_path_sha256']=trajectory_hash(row)
        except (ValueError,TypeError) as exc:
            row['token_path_sha256']=None
            row['technical_validation_errors'].append('INVALID_TRAJECTORY_DIGEST:'+repr(exc))
        try:self._readonly()
        except Exception as exc:
            row['technical_validation_errors'].append('ENDPOINT_CHANGED:'+repr(exc))
        if error:row['technical_validation_errors'].append('NATIVE_EXCEPTION_AFTER_CAPTURE:'+error)
        if row['technical_validation_errors']:row['generation_status']='TECHNICAL_INVALID'
        write_once(attempt.with_suffix('.COMPLETE.json'),failure_safe(row))
        return row

    def close(self):
        self._readonly()
        require(self.policy_hash()==self.endpoint['policy_sha256'],'Final endpoint tensor changed')
        self.runtime._vision_hook.remove();self._closed=True
        return dict(endpoint_id=self.endpoint['endpoint_id'],policy_sha256=self.endpoint['policy_sha256'],
                    optimizer_created=False,parameter_updates=0)


def run_endpoint(binding_path,endpoint,shard_index=0):
    binding,original,lock,all_requests=verify_eval_binding(binding_path)
    shards=binding['worker_shards']
    require(type(shard_index) is int and 0<=shard_index<shards,'Unregistered worker partition')
    package=canonical_directory(binding['package_root'])
    requests=[r for r in all_requests if r['endpoint_id']==endpoint]
    require(len(requests)==1024,'Endpoint request budget differs')
    requests=requests[shard_index::shards]
    stage=package/'run/evaluation'/endpoint/f'shard-{shard_index:02d}-of-{shards:02d}'
    stage.mkdir(parents=True,exist_ok=True)
    output=stage/'executions'/str(time.time_ns());output.mkdir(parents=True)
    backend=EvaluationBackend(binding_path,endpoint,output)
    completed=stage/'raw';completed.mkdir(exist_ok=True)
    for request in requests:
        key=text_sha(request['request_id']);rawpath=completed/(key+'.json');begin=stage/(key+'.BEGIN.json')
        markerpath=stage/(key+'.COMPLETE.json')
        if rawpath.exists():
            require(markerpath.is_file(),'Raw without completion journal is UNKNOWN')
            marker=read_json(markerpath)
            require(marker['raw_sha256']==file_sha(rawpath),'Existing raw bytes changed')
            row=read_json(rawpath)
            require(all(row[k]==request[k] for k in (*KEY,'case_id','input_id','seed','request_id')) and
                    row['policy_hash']==request['policy_sha256'],'Existing evaluation raw belongs to another request/endpoint')
            require(row['generation_status']=='COMPLETE' and not row['technical_validation_errors'],
                    'Existing technical failure requires audit')
            continue
        write_once(begin,dict(request=request,status='ATTEMPTED_UNKNOWN'))
        row=backend.generate_one(request)
        write_once(rawpath,failure_safe(row))
        write_once(markerpath,dict(request_id=request['request_id'],raw_sha256=file_sha(rawpath)))
        require(row['generation_status']=='COMPLETE' and not row['technical_validation_errors'],
                'Technical failure retained; audit before continuation')
    closed=backend.close()
    write_once(output/'WORKER_COMPLETE.json',dict(closed,status='ENDPOINT_SHARD_COMPLETE_REVIEW_PENDING',
        shard_index=shard_index,shards=shards,responses=len(requests)))


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    r=s.add_parser('register');r.add_argument('--package',type=Path,required=True);r.add_argument('--out',type=Path,required=True)
    r.add_argument('--code-commit',required=True);r.add_argument('--worker-shards',type=int,default=1)
    w=s.add_parser('worker');w.add_argument('--binding',type=Path,required=True);w.add_argument('--endpoint',required=True)
    w.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args()
    if a.command=='register':print(json.dumps(register_evaluation(a.package,a.out,a.code_commit,a.worker_shards),sort_keys=True))
    else:run_endpoint(a.binding,a.endpoint,a.shard_index)

if __name__=='__main__':main()
