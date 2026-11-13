"""Independent TRAIN adapter over pinned native math; no admission or update CLI.

No frozen module is modified or monkeypatched. Generation must use the frozen
GenerationBridge and a transaction owner's authenticated admission/canary gate.
The adapter additionally rejects stale requests and any uncommitted live policy.
Fresh step0 is the SHA-verified R24 genesis; published steps use TransactionStore
inspection plus the safe state codec. A dirty runtime is discarded, never silently
restored or described as unchanged. CPU boundary tests are not GPU acceptance.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

from . import training_state as state, training_transaction as transactions
from runtime import native_runtime as frozen, fork_loader
from runtime.backend import FrozenRegistry, PUBLIC, file_sha, object_sha, read_json

require = state.require


def verify_sources(registry, source_lock_path, execution_sources):
    """Verify actual frozen inventory and imported paths, then full native/model lock."""
    freeze=read_json(registry.freeze_path)
    directory=Path(frozen.__file__).resolve().parent
    actual={p.name:file_sha(p) for p in directory.glob('*.py') if p.is_file()}
    require(actual==freeze['runtime_sources'] and
            set(actual.values()) <= {r['sha256'] for r in freeze['artifacts'].values()},'Frozen runtime inventory differs')
    for name,module in tuple(sys.modules.items()):
        if name.startswith('runtime.'):
            expected=directory/(name.removeprefix('runtime.').replace('.','/')+'.py')
            require(getattr(module,'__file__',None) and Path(module.__file__).resolve()==expected,
                    'Shadowed frozen runtime module: '+name)
    for name,path in (('training_runtime',Path(__file__).resolve()),
                      ('training_state',Path(state.__file__).resolve()),
                      ('training_transaction',Path(transactions.__file__).resolve())):
        ref=execution_sources[name]
        require(Path(ref['path']).resolve()==path and file_sha(path)==ref['sha256'],'Unbound training source: '+name)
    return frozen.verify_source_lock(registry,source_lock_path)


class TrainingRuntimeMixin:
    def _set_recipe(self):
        super()._set_recipe()
        for key,value in self._satdose_native.generation_recipe().items():setattr(self.generation_config,key,value)
        self.identity.update(runtime_recipe='SR-SATDOSE-native-full-sequence-1536',
                             generation_config_expanded=self.generation_config.to_dict())

    def bind_request(self,request):
        require(not self._satdose_closed and self._satdose_active is None,'Closed/nested TRAIN request')
        require(self._satdose_registry.request(request['request_id'])==request,'Unregistered TRAIN request')
        endpoint=self._train_handle.endpoint()
        require(request['scope']=='TRAIN' and request['assistant_prefill']=='{' and
                request['start_seed']==endpoint['start_seed'] and request['arm']==endpoint['arm'] and
                request['step']==endpoint['step']+1,'TRAIN request differs from committed path/cursor')
        self._satdose_active=copy.deepcopy(request)
        self._train_phase='TRAIN'

    def unbind_request(self):
        require(self._satdose_active is not None,'No active TRAIN request')
        self._satdose_active=None
        self._train_phase=None

    def prepare_input(self,*args,**kwargs):
        return frozen.RegisteredRuntimeMixin.prepare_input(self,*args,**kwargs)

    def _validate_record_revision(self,record):
        return frozen.RegisteredRuntimeMixin._validate_record_revision(self,record)

    def _validate_prepared_revision(self,prepared):
        return frozen.RegisteredRuntimeMixin._validate_prepared_revision(self,prepared)

    def reserve(self,kind,count,**metadata):
        require(self._train_phase in ('TRAIN','CANARY_RESTORE_TEACHER'),'Unbound runtime accounting')
        if self._train_phase=='CANARY_RESTORE_TEACHER':
            require(kind=='extra_forward_sequences','Canary cannot generate tokens')
        frozen.RegisteredRuntimeMixin.reserve(self,kind,count,execution_phase=self._train_phase,**metadata)

    def _batch_forward(self,prepared,token_rows,*,purpose,grad):
        # Same frozen concatenation, attention mask and original native mRoPE implementation.
        self._validate_prepared_revision(prepared)
        require(self._satdose_active['scope']=='TRAIN' and purpose in ('training_policy','training_reference'),
                'Unregistered TRAIN teacher purpose')
        require(self._train_phase!='CANARY_RESTORE_TEACHER' or grad is False,'Canary backward is forbidden')
        require(len(token_rows)==1 and 1<=len(token_rows[0])<=1536,'One original full sequence required')
        torch=self.torch
        inputs=self._satdose_repeat_prompt(prepared['inputs'],1)
        prompt_len=inputs['input_ids'].shape[-1];longest=len(token_rows[0])
        pad=self.processor.tokenizer.pad_token_id
        if pad is None:pad=self.eos_ids[0]
        completion=inputs['input_ids'].new_full((1,longest),pad)
        mask=torch.zeros_like(completion)
        completion[0,:longest]=completion.new_tensor(token_rows[0]);mask[0,:longest]=1
        inputs['input_ids']=torch.cat([inputs['input_ids'],completion],-1)
        inputs['attention_mask']=torch.cat([inputs.get('attention_mask',mask.new_ones((1,prompt_len))),mask],-1)
        inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(completion)],-1)
        return self._native_forward(inputs,completion_start=prompt_len,lengths=[longest],purpose=purpose,grad=grad)


class TrainingHandle:
    def __init__(self,runtime,fork,native,storage,registry,start_seed,arm,output,path_identity,*,ready=True):
        self.runtime,self.fork,self.native_ops,self.storage=runtime,fork,native,storage
        self.registry,self.start_seed,self.arm,self.output=registry,start_seed,arm,Path(output)
        self.path_identity=copy.deepcopy(path_identity)
        self._step=0;self._commit_sha=None;self._commit_dir=None;self._ready=ready;self._closed=False
        self._policy=registry.config['start_bindings'][str(start_seed)]['policy_sha256']
        self._optimizer_hash=native.state_hash(fork.optimizer.state_dict())
        self._scheduler_hash=native.state_hash(fork.scheduler.state_dict())
        self.storage_forwards=[];self.genesis_reference=None
        runtime._train_handle=self

    def endpoint(self):
        require(not self._closed and self._ready,'Closed or unrestored training handle')
        source=self.registry.config['start_bindings'][str(self.start_seed)]
        policy=self.native_ops.state_hash(self.native_ops.trainable_state(self.runtime.model))
        reference=self.native_ops.state_hash(self.fork.reference)
        require(policy==self._policy and reference==source['reference_sha256'],
                'Dirty/uncommitted policy or changed original reference; generation forbidden')
        require(self.native_ops.state_hash(self.fork.optimizer.state_dict())==self._optimizer_hash and
                self.native_ops.state_hash(self.fork.scheduler.state_dict())==self._scheduler_hash,
                'Uncommitted optimizer/scheduler state; generation forbidden')
        require(self.runtime.identity['step']==self._step,'Runtime cursor changed before verified COMMIT')
        return dict(experiment_id=state.EXPERIMENT,start_seed=self.start_seed,arm=self.arm,step=self._step,
                    policy_sha256=policy,reference_sha256=reference)

    def _store(self,store):
        require(isinstance(store,transactions.TransactionStore) and store.root==self.registry.root and
                store.identity==self.path_identity and store.freeze_sha==self.registry.freeze_sha256,
                'Foreign/unverified transaction store')
        if self.genesis_reference is not None:
            require(store.genesis_reference==self.genesis_reference,'Transaction GENESIS differs')

    def _live_commit(self,marker):
        native=self.native_ops
        require(state.optimizer_parameter_order(self.runtime.model,self.fork.optimizer)==
                self.path_identity['optimizer_parameter_order'],'Optimizer parameter order changed')
        native.validate_optimizer(self.fork.optimizer,self.fork.scheduler,model=self.runtime.model)
        for key,value in [('policy',native.trainable_state(self.runtime.model)),
                          ('optimizer',self.fork.optimizer.state_dict()),('scheduler',self.fork.scheduler.state_dict()),
                          ('rng',native.capture_rng()),('reference',self.fork.reference)]:
            require(native.state_hash(value)==marker[key+'_sha256'],'Live state differs from published COMMIT: '+key)

    def advance_committed(self,store,step,expected_commit_sha256):
        """Call AFTER atomic publication and after commit_step releases its lock."""
        require(not self._closed and self._ready,'Closed/unrestored handle')
        self._store(store)
        marker=store.inspect_committed(step,expected_commit_sha256)
        require(marker['step']==step and marker['identity']==self.path_identity,'Published checkpoint identity differs')
        directory=store.path/f'step-{step:03d}'
        state.validate_checkpoint(directory,expected_commit_sha256,self.path_identity,self.native_ops)
        if step==self._step:
            require(expected_commit_sha256==self._commit_sha,'Different COMMIT at current cursor')
            self._live_commit(marker)
            return copy.deepcopy(marker)
        require(step==self._step+1 and marker['transaction']['previous_commit_sha256']==self._commit_sha,
                'Cannot skip or advance before the verified prior COMMIT')
        require(self.runtime.identity['step']==self._step,'Runtime cursor was advanced before publication')
        self._live_commit(marker)
        self._step=step;self._policy=marker['policy_sha256'];self._commit_sha=expected_commit_sha256;self._commit_dir=directory
        self._optimizer_hash=marker['optimizer_sha256'];self._scheduler_hash=marker['scheduler_sha256']
        self.runtime.identity.update(step=step,policy_hash=self._policy,reference_hash=marker['reference_sha256'])
        return copy.deepcopy(marker)

    def restore_committed(self,store,step,expected_commit_sha256,expected_identity):
        require(not self._closed and not self._ready and self._commit_sha is None,'Restore requires fresh unready loader')
        require(expected_identity==self.path_identity,'Resume expected identity differs')
        self._store(store)
        marker=store.inspect_committed(step,expected_commit_sha256)
        directory=store.path/f'step-{step:03d}'
        state.restore_checkpoint(self.runtime,self.fork.optimizer,self.fork.scheduler,self.fork.reference,self.native_ops,
            checkpoint_dir=directory,expected_commit_sha256=expected_commit_sha256,expected_identity=expected_identity)
        self._live_commit(marker)
        self._step=step;self._policy=marker['policy_sha256'];self._commit_sha=expected_commit_sha256;self._commit_dir=directory
        self._optimizer_hash=marker['optimizer_sha256'];self._scheduler_hash=marker['scheduler_sha256']
        self.genesis_reference=copy.deepcopy(store.genesis_reference);self._ready=True
        return self.endpoint()

    def _validate_original(self,row,sampled_endpoint):
        request=self.registry.request(row['request_id'])
        require(request['scope']=='TRAIN' and request['assistant_prefill']=='{' and request['arm']==self.arm and
                request['start_seed']==self.start_seed and request['step']==sampled_endpoint['step']+1,
                'Original trajectory path/cursor differs')
        require(row['policy_sha256']==sampled_endpoint['policy_sha256'] and
                row['reference_sha256']==sampled_endpoint['reference_sha256'],'Stale original sampled policy/reference')
        for key in ('start_seed','arm','step','scope'):
            require(row[key]==request[key],'Original request metadata differs: '+key)
        require(row.get('generation_status')=='COMPLETE' and row.get('technical_validation_errors')==[] and
                row.get('finish_reason') in ('balanced','eos','length') and row.get('assistant_prefill')=='{' and
                row.get('raw_tokens')==row['tokens'] and row.get('completion_token_count')==len(row['tokens']) and
                1<=len(row['tokens'])<=1536 and all(type(t) is int and t>=0 for t in row['tokens']) and
                len(row['sampler_logprobs'])==len(row['tokens']) and
                all(type(v) in (int,float) and math.isfinite(v) for v in row['sampler_logprobs']) and
                row.get('sampler_logprob_source')=='actual_generate_raw_logits_selected_token',
                'Incomplete/nonoriginal TRAIN capture')
        capture=Path(row['native_capture_path'])
        require(capture.resolve(strict=True)==capture and capture.is_relative_to(self.output) and
                file_sha(capture)==row['native_capture_sha256'],'Original capture changed/outside runtime audit')
        name=hashlib.sha256(row['request_id'].encode()).hexdigest()+'.CAPTURE.json'
        require(capture.name==name,'Capture request name differs')
        begin=read_json(capture.with_name(name.replace('.CAPTURE.json','.BEGIN.json')))
        require(begin['request_id']==row['request_id'] and begin['endpoint']==sampled_endpoint and
                begin['design_freeze_sha256']==self.registry.freeze_sha256,'Original BEGIN endpoint differs')
        saved=read_json(capture)
        for key in ('tokens','sampler_logprobs','prompt_tensor_hash','model_input','seed'):
            require(saved[key]==row[key],'Changed original trajectory: '+key)
        require('prepared' not in row and row['seed']==request['seed'] and
                row['model_input']=={k:request[k] for k in PUBLIC},'Unregistered original input')
        return request,dict(row,run_root=str(self.registry.root),root=str(self.registry.root),
                            protocol='structured_v4',protocol_revision='v2',assistant_prefill='{')

    def forward_original(self,row):
        endpoint=self.endpoint();request,record=self._validate_original(row,endpoint)
        runtime=self.runtime;runtime.bind_request(request)
        try:
            reference=runtime.batch_reference_forward([record],self.fork.reference)[0].detach()
            hook=self.storage.GPUSavedTensors((*runtime.model.parameters(),*runtime.model.buffers()),row_id=row['request_id'])
            entry=dict(row_id=row['request_id'],stats=hook.stats);self.storage_forwards.append(entry)
            with hook.context():current=runtime.batch_sequence_forward([record],purpose='training_policy',grad=True)[0]
            entry['forward_returned']=True
            sampler=current.new_tensor(row['sampler_logprobs'])
            require(current.shape==sampler.shape==reference.shape,'Original token vector alignment differs')
            return current,sampler,reference
        finally:runtime.unbind_request()

    def verify_storage_after_backward(self):
        for entry in self.storage_forwards:
            require(entry.get('forward_returned') is True,'Incomplete TRAIN forward')
            self.storage.validate_snapshot_stats(entry['stats'])

    def canary_teacher(self,row,store,*,sampled_commit_sha256,teacher_commit_sha256):
        """Read-only step1-sampled tokens under COMMITTED step2, before/after restart.

        This is not production training-policy replay: no backward, no optimizer,
        no generation, no changed original row policy fields. Full step2 rollout
        membership is checked through the actual published transaction evidence.
        """
        import torch
        teacher=self.endpoint();self._store(store)
        require(teacher['step']==2 and self._commit_sha==teacher_commit_sha256,'Canary requires committed step2')
        prior=store.inspect_committed(1,sampled_commit_sha256)
        current=store.inspect_committed(2,teacher_commit_sha256)
        require(current['transaction']['previous_commit_sha256']==sampled_commit_sha256,'Canary commit chain differs')
        sampled=dict(teacher,step=1,policy_sha256=prior['policy_sha256'],reference_sha256=prior['reference_sha256'])
        request,record=self._validate_original(row,sampled)
        directory=store.path/'step-002'
        evidence=transactions.read_json(transactions.bound(directory,current['transaction']['evidence']))
        receipt=transactions.read_json(transactions.bound(self.registry.root,evidence['rollout_receipt']))
        originals=transactions.rows(transactions.bound(self.registry.root,receipt['raw']))
        matches=[item for item in originals if item['request_id']==row['request_id']]
        require(len(matches)==1,'Canary original request absent from committed step2 rollout')
        original=matches[0]
        # Relocated capture path alone may differ; all original scientific fields remain identical.
        require({k:v for k,v in row.items() if k!='native_capture_path'}==
                {k:v for k,v in original.items() if k!='native_capture_path'},'Canary row substituted committed original')
        runtime=self.runtime
        require(runtime._satdose_active is None,'Nested canary request')
        before_rng=self.native_ops.capture_rng();before_mode=runtime.model.training
        before_policy=self.native_ops.trainable_state(runtime.model)
        before_optimizer=self.native_ops.state_hash(self.fork.optimizer.state_dict())
        before_scheduler=self.native_ops.state_hash(self.fork.scheduler.state_dict())
        versions=[(v,v._version) for v in (*[p for p in runtime.model.parameters() if not p.requires_grad],*runtime.model.buffers())]
        require(all(p.grad is None for p in runtime.model.parameters()),'Canary cannot overwrite existing gradients')
        index=len(runtime._satdose_accounting)
        runtime._satdose_active=copy.deepcopy(request);runtime._train_phase='CANARY_RESTORE_TEACHER'
        try:
            runtime.model.eval()
            with torch.no_grad():
                reference=runtime.batch_reference_forward([record],self.fork.reference)[0].detach()
                current_vector=runtime.batch_sequence_forward([record],purpose='training_policy',grad=False)[0].detach()
            require(current_vector.shape==reference.shape==(len(row['tokens']),) and
                    current_vector.dtype==reference.dtype==torch.float32 and
                    bool(torch.isfinite(current_vector).all()) and bool(torch.isfinite(reference).all()),'Invalid canary vectors')
            events=copy.deepcopy(runtime._satdose_accounting[index:])
            require(len(events)==2 and all(e['kind']=='extra_forward_sequences' and e['count']==1 and
                    e['metadata']['execution_phase']=='CANARY_RESTORE_TEACHER' for e in events) and
                    {e['metadata']['purpose'] for e in events}=={'training_policy','training_reference'},
                    'Canary accounting differs or generated tokens')
            result=dict(status='CANARY_TEACHER_CAPTURE_NOT_RESUME_ADMISSION',sampled_endpoint=sampled,teacher_endpoint=teacher,
                sampled_commit_sha256=sampled_commit_sha256,teacher_commit_sha256=teacher_commit_sha256,
                request_id=row['request_id'],tokens=list(row['tokens']),sampler_logprobs=list(row['sampler_logprobs']),
                teacher_logprobs=current_vector.cpu().clone(),reference_logprobs=reference.cpu().clone(),
                accounting=events,backward_calls=0,optimizer_steps=0,new_generations=0)
        finally:
            runtime.unbind_request()
            for p in runtime.model.parameters():p.grad=None
            try:self.native_ops.restore_rng(before_rng)
            finally:runtime.model.train(before_mode)
        require(all(v._version==version for v,version in versions),'Canary mutated frozen parameter/buffer')
        require(self.native_ops.state_hash(before_policy)==self.native_ops.state_hash(self.native_ops.trainable_state(runtime.model)) and
                self.native_ops.state_hash(self.fork.optimizer.state_dict())==before_optimizer and
                self.native_ops.state_hash(self.fork.scheduler.state_dict())==before_scheduler,'Canary mutated training state; discard')
        require(self.endpoint()==teacher and self.native_ops.state_hash(self.native_ops.capture_rng())==
                self.native_ops.state_hash(before_rng),'Canary changed endpoint/RNG')
        return result

    def discard(self):
        """Release hooks/gradients even for dirty parameters; do not claim unchanged."""
        if not self._closed:
            try:self.runtime.model.zero_grad(set_to_none=True)
            finally:
                try:self.runtime._vision_hook.remove()
                finally:self._closed=True;self._ready=False;self.runtime._satdose_closed=True
        return dict(discarded=True,state_unchanged_asserted=False)


def build_training_runtime(registry,source_lock_path,*,start_seed,arm,output_dir,execution_sources,
                           commit_store=None,resume_step=None,expected_commit_sha256=None,expected_identity=None):
    """Actually construct original b0, then R24/fresh native Adam; optional explicit restore.

    output_dir must exist and be empty under the frozen experiment root. The
    caller must authenticate execution_sources and admission separately. No old
    training CLI/update is called. No generated/raw bytes are read by construction.
    """
    require(isinstance(registry,FrozenRegistry) and start_seed in (11031,11032) and arm in ('T0','T1'),'Invalid TRAIN construction')
    output=Path(output_dir).resolve(strict=True)
    require(output.is_dir() and output.is_relative_to(registry.root) and not any(output.iterdir()),'New empty isolated TRAIN audit required')
    resume=commit_store is not None
    require((resume and all(v is not None for v in (resume_step,expected_commit_sha256,expected_identity))) or
            (not resume and all(v is None for v in (resume_step,expected_commit_sha256,expected_identity))), 'Incomplete explicit resume identity')
    lock,native_root,legacy_root,model,parent=verify_sources(registry,source_lock_path,execution_sources)
    sys.path[:0]=[str(native_root),str(legacy_root)]
    native=frozen._import('sr_f6.runtime',native_root);base=frozen._import('sr_f5.runtime',native_root).Runtime
    configure=frozen._import('mm_dev.runtime',native_root).configure_audited_backend
    repeat=frozen._import('sr_f12.runtime',native_root)._repeat_prompt
    legacy=frozen._import('gpu_production_v2.backend',legacy_root)
    storage=frozen._import('diagnostics.gpu_capacity_v2',legacy_root)
    ops=fork_loader.load_native_ops(native_root,lock['native_files'])
    namespaces={Path(rel).parts[0].removesuffix('.py') for rel in lock['native_files']}
    allowed={str((native_root/rel).resolve()) for rel in lock['native_files']}
    for name,module in tuple(sys.modules.items()):
        source=getattr(module,'__file__',None)
        if source and name.split('.')[0] in namespaces:
            require(str(Path(source).resolve()) in allowed,'Shadowed native dependency: '+name)
    configure();storage.validate_gpu_pair()
    require(storage.CAP_BYTES==48*1024**3,'Wrong saved tensor capacity')
    class Runtime(TrainingRuntimeMixin,base):
        _satdose_native=native;_satdose_legacy=legacy;_satdose_repeat_prompt=staticmethod(repeat)
        _satdose_registry=registry;_satdose_start=start_seed;_satdose_output=output
        _satdose_active=None;_satdose_closed=False;_train_phase=None
    Runtime._satdose_accounting=[]
    runtime=handle=None
    try:
        native.seed_all(native.seed_for(2026110301,'load','b0'))
        runtime=Runtime(str(model),adapter_path=str(parent),output_root=output,protocol_revision='v2',account=None)
        native.configure_parent_training(runtime,parent_adapter_config=read_json(parent/'adapter_config.json'))
        original_dtype=runtime.identity['dtype'];runtime.model=runtime.model.float();runtime.model.eval()
        runtime._srf5_numerics=dict(native.NUMERICS_V2)
        runtime.identity.update(dtype='float32',input_float_dtype='float32',probability_dtype='float32',
            model_cast_from_dtype=original_dtype,forward_parameter_dtypes=['float32'],forward_float_buffer_dtypes=['float32'],
            numerics=dict(native.NUMERICS_V2),step=0,manifest_sha256=registry.freeze_sha256,
            source_sha256=object_sha(lock['native_files']))
        fork=fork_loader.fork_runtime(runtime,registry.config,start_seed,arm,ops)
        runtime.identity.update(execution_mode='TRAIN',run_root=str(registry.root),audit_root=str(output))
        freeze=read_json(registry.freeze_path)
        base_identity=dict(experiment_id=state.EXPERIMENT,parent='b0',train_seed=start_seed,arm=arm,run_id=runtime.identity['run_id'],
            config_sha256=freeze['config_sha256'],design_freeze_sha256=registry.freeze_sha256,
            manifest_sha256=freeze['artifacts']['train_manifest']['sha256'],plan_sha256=freeze['artifacts']['training_groups']['sha256'],
            source_sha256=object_sha(lock['native_files']),execution_sources_sha256=object_sha(execution_sources))
        identity=state.bind_identity(base_identity,runtime.model,fork.optimizer,source_binding=registry.config['start_bindings'][str(start_seed)])
        handle=TrainingHandle(runtime,fork,ops,storage,registry,start_seed,arm,output,identity,ready=not resume)
        from runtime.backend import write_once
        write_once(output/'FORK_LOAD.json',dict(identity=identity,fork_receipt=fork.receipt,resume=resume))
        if resume:
            require(expected_identity==identity,'Resume source/order/implementation identity differs')
            handle.restore_committed(commit_store,resume_step,expected_commit_sha256,expected_identity)
        else:
            genesis=dict(schema='sr-satdose-training-genesis-v1',status='FRESH_R24_FORK_NOT_TRAINING_ADMISSION',
                permission='NO_TRAINING_ADMISSION',identity=identity,endpoint=handle.endpoint(),
                design_freeze_sha256=registry.freeze_sha256,source_binding=copy.deepcopy(identity['source_binding']),
                fork_receipt=copy.deepcopy(fork.receipt),optimizer_sha256=ops.state_hash(fork.optimizer.state_dict()),
                scheduler_sha256=ops.state_hash(fork.scheduler.state_dict()),rng_sha256=ops.state_hash(ops.capture_rng()),
                optimizer_steps=0,scheduler_steps=0,parameter_updates=0)
            write_once(output/'GENESIS.json',genesis)
            handle.genesis_reference=dict(path=str((output/'GENESIS.json').relative_to(registry.root)),sha256=file_sha(output/'GENESIS.json'))
        return handle
    except BaseException:
        if handle is not None:handle.discard()
        elif runtime is not None:
            try:runtime.model.zero_grad(set_to_none=True)
            finally:runtime._vision_hook.remove()
        raise
