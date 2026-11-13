"""Safe SATDOSE checkpoint codec over frozen native operations; no optimizer step.

The caller authenticates nativeOps with fork_loader.load_native_ops and the
complete source lock, and derives source_binding from the frozen start config.
This module writes only state.pt/COMMIT.json into an already exclusively owned
staging directory. It does NOT publish/rename a transaction, validate raw/UPDATE
chains, choose a checkpoint, generate tokens or authorize training. Restore
requires a caller-trusted COMMIT SHA and exact expected identity; the caller
must first validate the entire transaction chain and reject UNKNOWN intents.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

from runtime import fork_loader

SCHEMA = fork_loader.SCHEMA
EXPERIMENT = fork_loader.EXPERIMENT
SOURCE_KEYS = {'original_state_path','state_sha256','policy_sha256','reference_sha256'}
CORE_FIELDS = {'schema','identity','step','policy_sha256','reference_sha256','optimizer_sha256',
               'scheduler_sha256','rng_sha256','sampler_state','state_sha256','commit_sha256',
               'policy','optimizer','scheduler','rng','source_binding','optimizer_parameter_order',
               'optimizer_parameter_order_sha256'}


def require(condition, message):
    if not condition: raise PermissionError(message)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value):
    return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None


def _directory(path):
    path=Path(path).absolute()
    require(path.is_dir() and path.resolve(strict=True)==path,'Canonical existing checkpoint directory required')
    return path


def _file_sha(path):
    return fork_loader.file_sha(path)


def _sync(directory):
    fd=os.open(directory,os.O_RDONLY | os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def _write_json_once(path,value):
    fd=os.open(path,os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(_canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    _sync(path.parent)


def _cpu_tree(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_cpu_tree(v) for v in value)
    if isinstance(value,list):return [_cpu_tree(v) for v in value]
    return copy.deepcopy(value)


def _validate_rng(rng):
    import numpy as np
    import torch
    require(isinstance(rng,dict) and set(rng)=={'python','numpy','cpu','cuda'},'Native RNG inventory differs')
    p,n=rng['python'],rng['numpy']
    require(isinstance(p,tuple) and len(p)==3 and type(p[0]) is int and isinstance(p[1],tuple) and
            (p[2] is None or (type(p[2]) is float and math.isfinite(p[2]))),
            'Python RNG tuple required')
    require(isinstance(n,tuple) and len(n)==5 and isinstance(n[0],str) and
            isinstance(n[1],np.ndarray) and n[1].dtype==np.uint32 and n[1].ndim==1 and n[1].size>0 and
            type(n[2]) is int and type(n[3]) is int and n[3] in (0,1) and
            type(n[4]) is float and math.isfinite(n[4]),'Native NumPy uint32 RNG tuple required')
    require(isinstance(rng['cuda'],list),'CUDA RNG list required')
    for value in [rng['cpu'],*rng['cuda']]:
        require(torch.is_tensor(value) and value.device.type=='cpu' and value.dtype==torch.uint8 and
                value.ndim==1 and value.numel()>0 and not value.requires_grad,'CPU ByteTensor RNG state required')
    # Validate native structures without mutating any global RNG or touching CUDA.
    random.Random(0).setstate(p)
    np.random.RandomState(0).set_state(n)
    torch.Generator(device='cpu').set_state(rng['cpu'])


def pack_rng(rng):
    """Only NumPy ndarray becomes list; preserve all tuple/list types for hashing."""
    _validate_rng(rng)
    packed=_cpu_tree(rng)
    n=rng['numpy'];packed['numpy']=(n[0],n[1].tolist(),n[2],n[3],n[4])
    return packed


def unpack_rng(packed):
    """Restore weights_only-safe packed NumPy words without silent integer wrapping."""
    import numpy as np
    require(isinstance(packed,dict) and set(packed)=={'python','numpy','cpu','cuda'},'Packed RNG inventory differs')
    n=packed['numpy']
    require(isinstance(n,tuple) and len(n)==5 and isinstance(n[1],list) and n[1] and
            all(type(v) is int and 0<=v<2**32 for v in n[1]),'Packed NumPy uint32 words invalid')
    rng=copy.deepcopy(packed)
    rng['numpy']=(n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4])
    _validate_rng(rng)
    return rng


def optimizer_parameter_order(model,optimizer):
    """Bind serialized integer parameter IDs to actual ordered trainable names."""
    import torch
    require(isinstance(optimizer,torch.optim.AdamW),'Native AdamW required')
    named={name:p for name,p in model.named_parameters() if p.requires_grad}
    require(len(named)==496 and all('lora_' in name for name in named),'Exact 496 LoRA inventory required')
    by_id={id(p):name for name,p in named.items()}
    require(len(by_id)==496,'Aliased trainable parameter inventory')
    saved=optimizer.state_dict()['param_groups']
    require(len(saved)==len(optimizer.param_groups),'Optimizer group inventory differs')
    result=[];seen_objects=set();seen_ids=set()
    for actual,serialized in zip(optimizer.param_groups,saved,strict=True):
        require(len(actual['params'])==len(serialized['params']),'Optimizer group length differs')
        group=[]
        for parameter,pid in zip(actual['params'],serialized['params'],strict=True):
            require(id(parameter) in by_id and id(parameter) not in seen_objects and
                    type(pid) is int and pid not in seen_ids,'Duplicate/foreign optimizer parameter ID')
            require(parameter.dtype==torch.float32,'FP32 trainable parameter required')
            seen_objects.add(id(parameter));seen_ids.add(pid)
            group.append(dict(parameter_id=pid,name=by_id[id(parameter)],shape=list(parameter.shape),dtype=str(parameter.dtype)))
        result.append(group)
    require(seen_objects==set(by_id),'Optimizer must own every trainable parameter exactly once')
    return result


def _validate_identity(identity):
    require(isinstance(identity,dict) and identity.get('experiment_id')==EXPERIMENT and
            identity.get('parent')=='b0' and identity.get('arm') in ('T0','T1') and
            type(identity.get('train_seed')) is int and identity['train_seed'] in (11031,11032) and
            isinstance(identity.get('run_id'),str) and identity['run_id'],'Wrong SATDOSE path identity')
    source=identity.get('source_binding',{})
    require(set(source)==SOURCE_KEYS and all(_sha(source[key]) for key in SOURCE_KEYS-{'original_state_path'}) and
            isinstance(source['original_state_path'],str) and Path(source['original_state_path']).is_absolute() and
            '..' not in Path(source['original_state_path']).parts,'Frozen R24 source binding required')
    order=identity.get('optimizer_parameter_order')
    require(isinstance(order,list) and order and all(isinstance(g,list) and g for g in order),
            'Ordered optimizer identity required')
    flat=[v for group in order for v in group]
    require(len(flat)==496 and all(isinstance(v,dict) and set(v)=={'parameter_id','name','shape','dtype'} for v in flat),
            'Ordered optimizer identity inventory differs')
    require(all(type(v['parameter_id']) is int and isinstance(v['name'],str) and 'lora_' in v['name'] and
                v['dtype']=='torch.float32' and isinstance(v['shape'],list) and
                all(type(d) is int and d>=0 for d in v['shape']) for v in flat) and
            len({v['parameter_id'] for v in flat})==len({v['name'] for v in flat})==496,
            'Invalid ordered optimizer parameter binding')
    require(identity.get('optimizer_parameter_order_sha256')==_digest(order),'Optimizer order SHA differs')
    _canonical(identity)


def bind_identity(base_identity,model,optimizer,*,source_binding):
    """Return a new engineering identity; caller supplies the frozen start binding."""
    identity=copy.deepcopy(base_identity)
    require(not {'source_binding','optimizer_parameter_order','optimizer_parameter_order_sha256'} & set(identity),
            'Do not overwrite an existing engineering identity')
    identity['source_binding']=copy.deepcopy(source_binding)
    identity['optimizer_parameter_order']=optimizer_parameter_order(model,optimizer)
    identity['optimizer_parameter_order_sha256']=_digest(identity['optimizer_parameter_order'])
    _validate_identity(identity)
    return identity


def _runtime_identity(runtime,identity):
    actual=runtime.identity
    require(actual.get('experiment_id')==identity['experiment_id'] and actual.get('train_seed')==identity['train_seed'] and
            actual.get('arm')==identity['arm'] and actual.get('run_id')==identity['run_id'] and
            actual.get('reference_parent')=='b0','Runtime differs from expected path identity')


def _validate_state(state,marker,identity,native):
    import torch
    _validate_identity(identity)
    require(state['schema']==marker['schema']==SCHEMA and state['identity']==marker['identity']==identity,
            'Serialized path/source identity differs')
    step=state['step']
    require(type(step) is int and 0<=step<=24 and marker['step']==step,'Serialized step differs')
    sampler=state['sampler_state']
    require(isinstance(sampler,dict) and sampler==marker['sampler_state'] and
            type(sampler.get('committed_step')) is int and sampler['committed_step']==step and
            type(sampler.get('next_step')) is int and sampler['next_step']==step+1 and
            type(sampler.get('generated_responses')) is int and sampler['generated_responses']==128*step,
            'Sampler cursor differs')
    _canonical(sampler)
    order=identity['optimizer_parameter_order'];flat=[p for group in order for p in group]
    policy=state['policy']
    require(isinstance(policy,dict) and set(policy)=={p['name'] for p in flat},'Policy inventory differs')
    for entry in flat:
        value=policy[entry['name']]
        require(torch.is_tensor(value) and value.dtype==torch.float32 and not value.requires_grad and
                list(value.shape)==entry['shape'],'Policy dtype/shape differs')
    require(state['reference_sha256']==marker['reference_sha256']==identity['source_binding']['reference_sha256'],
            'Original b0 reference identity differs')
    if step==0:require(marker['policy_sha256']==identity['source_binding']['policy_sha256'],'Step0 must be original R24')
    for key in ('policy','optimizer','scheduler'):
        require(native.state_hash(state[key])==marker[key+'_sha256'] and native.finite_tree(state[key]),
                'Serialized '+key+' hash/nonfinite mismatch')
    native.validate_optimizer_state(state['optimizer'],state['scheduler'])
    groups=state['optimizer']['param_groups'];states=state['optimizer']['state']
    require(len(groups)==len(order) and all(g.get('foreach') is False and
            g['params']==[p['parameter_id'] for p in bound] for g,bound in zip(groups,order,strict=True)),
            'Serialized optimizer parameter order differs')
    require(state['scheduler'].get('last_epoch')==step,'Scheduler cursor differs')
    if step==0:require(not states,'Step0 Adam history must be empty')
    else:
        require(set(states)=={p['parameter_id'] for p in flat},'Adam state inventory differs')
        for entry in flat:
            item=states[entry['parameter_id']]
            require(set(item)=={'step','exp_avg','exp_avg_sq'} and torch.is_tensor(item['step']) and
                    item['step'].numel()==1 and float(item['step'])==step,'Adam state/counter differs')
            for key in ('exp_avg','exp_avg_sq'):
                require(torch.is_tensor(item[key]) and item[key].dtype==torch.float32 and
                        list(item[key].shape)==entry['shape'],'Adam moment shape/dtype differs')
    rng=unpack_rng(state['rng'])
    require(native.state_hash(rng)==marker['rng_sha256'],'Native unpacked RNG SHA differs')
    return rng


def write_checkpoint_files(directory,runtime,optimizer,scheduler,reference,native,*,identity,step,sampler_state,
                           commit_fields=None):
    """Write-once staging files, NOT an atomic transaction publication or admission.

    Additional transaction fields may bind previous COMMIT, original raw and
    UPDATE. They cannot overwrite any core state/identity/hash field. The
    transaction owner must validate those extra evidence chains independently.
    """
    import torch
    directory=_directory(directory)
    require(not (directory/'state.pt').exists() and not (directory/'COMMIT.json').exists(),
            'Checkpoint files already exist; never overwrite/retry')
    _validate_identity(identity);_runtime_identity(runtime,identity)
    require(optimizer_parameter_order(runtime.model,optimizer)==identity['optimizer_parameter_order'],
            'Actual optimizer parameter order differs')
    native.validate_optimizer(optimizer,scheduler,model=runtime.model)
    require(all(not v.is_floating_point() or v.dtype==torch.float32
                for v in (*runtime.model.parameters(),*runtime.model.buffers())),'FP32 model/buffers required')
    require(all(p.grad is None for p in runtime.model.parameters()),'Checkpoint requires cleared gradients')
    rng=native.capture_rng()
    state=dict(schema=SCHEMA,identity=copy.deepcopy(identity),step=step,policy=native.trainable_state(runtime.model),
        reference_sha256=native.state_hash(reference),optimizer=_cpu_tree(optimizer.state_dict()),
        scheduler=_cpu_tree(scheduler.state_dict()),rng=pack_rng(rng),sampler_state=copy.deepcopy(sampler_state))
    marker=dict(schema=SCHEMA,identity=copy.deepcopy(identity),step=step,reference_sha256=state['reference_sha256'],
                sampler_state=copy.deepcopy(sampler_state),rng_sha256=native.state_hash(rng))
    marker.update({key+'_sha256':native.state_hash(state[key]) for key in ('policy','optimizer','scheduler')})
    require(commit_fields is None or isinstance(commit_fields,dict),'Transaction fields must be a dictionary')
    extra=copy.deepcopy(commit_fields if commit_fields is not None else {})
    require(isinstance(extra,dict) and not set(extra)&CORE_FIELDS,'Transaction fields cannot overwrite core checkpoint fields')
    _canonical(extra);marker.update(extra)
    _validate_state(state,marker,identity,native)
    fd=os.open(directory/'state.pt',os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    _sync(directory)
    marker['state_sha256']=_file_sha(directory/'state.pt')
    # Ensure the exact saved payload is readable by the frozen safe loader before COMMIT bytes exist.
    loaded=torch.load(directory/'state.pt',map_location='cpu',weights_only=True)
    _validate_state(loaded,marker,identity,native)
    require(native.state_hash(native.capture_rng())==marker['rng_sha256'],'Checkpoint encoding consumed global RNG')
    _write_json_once(directory/'COMMIT.json',marker)
    return dict(commit=copy.deepcopy(marker),commit_sha256=_file_sha(directory/'COMMIT.json'),
                state_sha256=marker['state_sha256'])


def validate_checkpoint(checkpoint_dir,expected_commit_sha256,expected_identity,native):
    """Read-only schema/hash validation; staging names allowed, no model mutation."""
    import torch
    directory=_directory(checkpoint_dir)
    require(_sha(expected_commit_sha256) and _file_sha(directory/'COMMIT.json')==expected_commit_sha256,
            'Untrusted/different COMMIT SHA')
    _validate_identity(expected_identity)
    marker=fork_loader._json(directory/'COMMIT.json')
    require(marker.get('schema')==SCHEMA and marker.get('identity')==expected_identity,'Checkpoint path/source identity differs')
    require(_file_sha(directory/'state.pt')==marker['state_sha256'],'Checkpoint file bytes differ')
    state=torch.load(directory/'state.pt',map_location='cpu',weights_only=True)
    _validate_state(state,marker,expected_identity,native)
    require(_file_sha(directory/'COMMIT.json')==expected_commit_sha256 and
            _file_sha(directory/'state.pt')==marker['state_sha256'],'Checkpoint changed while validating')
    return copy.deepcopy(marker)


def restore_checkpoint(runtime,optimizer,scheduler,reference,native,*,checkpoint_dir,
                       expected_commit_sha256,expected_identity):
    """Prevalidate ordered parameter ownership, then use frozen restore_committed.

    Does not choose latest, retry UNKNOWN updates or validate transaction raw.
    On validation failure no model, optimizer, scheduler or RNG is modified.
    """
    _validate_identity(expected_identity);_runtime_identity(runtime,expected_identity)
    require(optimizer_parameter_order(runtime.model,optimizer)==expected_identity['optimizer_parameter_order'],
            'Actual optimizer parameter order differs')
    native.validate_optimizer(optimizer,scheduler,model=runtime.model)
    validate_checkpoint(checkpoint_dir,expected_commit_sha256,expected_identity,native)
    return fork_loader.restore_committed(runtime,optimizer,scheduler,reference,native,
        checkpoint_dir=checkpoint_dir,expected_commit_sha256=expected_commit_sha256,expected_identity=expected_identity)
