"""Four-slot, write-once isolated shadow Adam executor; never a training driver.

Production authority is the ONE TECHNICAL_EXECUTION_BINDING.json under the fixed
frozen experiment root. It must be written before diagnostic GPU work, bind the
original freeze/config/source lock and this executor/production source bytes,
and predeclare all four input slots. Actual restore/capture hashes are supplied
later by each slot's write-once INPUT_RECEIPT.json, before its durable CLAIM.
An existing claim is UNKNOWN or consumed and is NEVER retried automatically.

The caller must actually reload R24 in a new loader instance and recapture the
original fixed tokens for trial 1. This executor verifies those independent file
identities plus source checkpoint bytes, policy, gradients and replay equality;
it cannot prove a model reload from a self-authored JSON assertion alone. Only
the separately audited driver/runtime logs can establish that execution fact.
No model object, sampler, optimizer, output-directory override or checkpoint
publication API is accepted. GPU admission remains a separate decision.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

from . import production
from runtime.fork_loader import load_native_ops

EXPERIMENT = 'SR-SATDOSE-1-20261112'
PRODUCTION_ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
BINDING_NAME = 'TECHNICAL_EXECUTION_BINDING.json'
CLAIMS_NAME = 'shadow_claims'
MODES = {'production', 'SYNTHETIC_CPU_TEST_ONLY'}


def require(condition, message):
    if not condition:
        raise PermissionError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def file_sha(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), 'Regular evidence file required')
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        require(len(dict(items))==len(items),'Duplicate evidence key')
        return dict(items)
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_once(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    _sync_dir(path.parent)


def expected_slots():
    return {f'{start}/{trial}': {
        'input_receipt':f'shadow_inputs/start-{start}/trial-{trial}/INPUT_RECEIPT.json',
        'restore':f'shadow_inputs/start-{start}/trial-{trial}/RESTORE.json',
        'capture':f'shadow_inputs/start-{start}/trial-{trial}/PRODUCTION_CAPTURE.pt'}
        for start in (11031,11032) for trial in (0,1)}


def _local_file(root,relative):
    path=Path(relative)
    require(not path.is_absolute() and '..' not in path.parts,'Evidence escapes fixed root')
    full=root/path
    require(full.resolve(strict=True)==full and full.is_file() and not full.is_symlink(),
            'Canonical regular evidence required')
    return full


def _binding(root,expected_sha,mode):
    require(mode in MODES and root.resolve(strict=True)==root,'Canonical fixed root required')
    if mode=='production':require(root==PRODUCTION_ROOT,'Production claims root cannot be relocated')
    else:require(not root.is_relative_to(PRODUCTION_ROOT),'Synthetic tests cannot touch production claims')
    path=root/BINDING_NAME
    require(file_sha(path)==expected_sha,'Technical binding SHA differs')
    b=read_json(path)
    require(b['schema']=='sr-satdose-technical-execution-v1' and b['mode']==mode
            and b['experiment_id']==EXPERIMENT and b['experiment_root']==str(root), 'Wrong technical execution identity')
    require(b['shadow_claims_root']==CLAIMS_NAME and b['shadow_slots']==expected_slots()
            and b['max_shadow_optimizer_steps']==4 and b['steps_per_slot']==1,
            'Fixed global shadow slots/budget differ')
    freeze_path=_local_file(root,b['design_freeze']['path'])
    require(freeze_path==root/'DESIGN_FREEZE.json' and file_sha(freeze_path)==b['design_freeze']['sha256'],
            'Design freeze changed/relocated')
    freeze=read_json(freeze_path)
    require(freeze['experiment_id']==EXPERIMENT and freeze['schema']=='sr-satdose-design-freeze-v1', 'Wrong original freeze')
    for key in ('config','source_lock'):
        require(b[key]==freeze['artifacts'][key],'Technical binding substituted frozen '+key)
        require(file_sha(_local_file(root,b[key]['path']))==b[key]['sha256'],'Frozen '+key+' changed')
    for key,actual in (('shadow_replay',Path(__file__).resolve()),('production',Path(production.__file__).resolve())):
        require(Path(b['code'][key]['path']).resolve()==actual and file_sha(actual)==b['code'][key]['sha256'],
                'Unbound execution code: '+key)
    config=read_json(_local_file(root,b['config']['path']))
    lock=read_json(_local_file(root,b['source_lock']['path']))
    ops=load_native_ops(lock['native_src'],lock['native_files'])
    return b,config,ops


def _valid_endpoint(endpoint,start,config):
    expected=config['start_bindings'][str(start)]
    require(endpoint['experiment_id']==EXPERIMENT and endpoint['start_seed']==start
            and endpoint['arm']=='START' and endpoint['step']==0
            and endpoint.get('parent','b0')=='b0','Shadow requires original R24 START endpoint')
    require(endpoint['policy_sha256']==expected['policy_sha256'] and
            endpoint['reference_sha256']==expected['reference_sha256'],'Shadow endpoint hashes differ')
    return expected


def _load_inputs(root,b,binding_sha,start,trial,input_sha,endpoint,ops,config):
    import torch
    slot=b['shadow_slots'][f'{start}/{trial}']
    path=_local_file(root,slot['input_receipt'])
    require(file_sha(path)==input_sha,'Input receipt SHA differs')
    receipt=read_json(path)
    require(receipt['schema']=='sr-satdose-shadow-input-v1' and receipt['mode']==b['mode']
            and receipt['technical_binding_sha256']==binding_sha
            and receipt['design_freeze_sha256']==b['design_freeze']['sha256']
            and receipt['start_seed']==start and receipt['trial']==trial and receipt['endpoint']==endpoint,
            'Shadow input identity differs')
    evidence={}
    for key in ('restore','capture'):
        ref=receipt[key]
        require(ref['path']==slot[key],'Evidence moved outside fixed slot')
        evidence[key]=_local_file(root,ref['path'])
        require(file_sha(evidence[key])==ref['sha256'],'Changed '+key+' evidence')
    restored=read_json(evidence['restore'])
    capture=torch.load(evidence['capture'],map_location='cpu',weights_only=True)
    require(restored['schema']=='sr-satdose-restored-r24-v1' and capture['schema']=='sr-satdose-production-capture-v1',
            'Wrong restore/capture schema')
    for item in (restored,capture):
        require(item['mode']==b['mode'] and item['technical_binding_sha256']==binding_sha
                and item['design_freeze_sha256']==b['design_freeze']['sha256'] and item['endpoint']==endpoint,
                'Foreign restore/capture execution')
    instance=restored['loader_instance_id']
    require(type(instance) is str and re.fullmatch('[a-zA-Z0-9_-]{16,128}',instance)
            and capture['loader_instance_id']==instance,'Missing/mismatched independent loader instance')
    expected=_valid_endpoint(endpoint,start,config)
    require(restored['source_state_path']==expected['original_state_path'] and
            restored['source_state_sha256']==expected['state_sha256'] and restored['restored_from_original_state'] is True,
            'Restore did not bind original R24 checkpoint')
    source=Path(expected['original_state_path'])
    require(file_sha(source)==expected['state_sha256'],'Original R24 checkpoint bytes changed')
    state=torch.load(source,map_location='cpu',weights_only=True)
    require(ops.state_hash(state['policy'])==expected['policy_sha256'] and
            state['reference_sha256']==expected['reference_sha256'],'Original R24 tensor identity differs')
    prod=capture['production']
    require(prod['source_sha256']=={'native':production.PINS,'legacy':production.LEGACY_PINS},
            'Capture did not bind pinned native numerical kernels')
    require(prod['status']=='NUMERICAL_CAPTURE_NOT_UPDATE_AUTHORIZATION' and
            prod['policy_sha256']==expected['policy_sha256'] and prod['reference_sha256']==expected['reference_sha256']
            and prod['sequence_denominator']==128 and prod['row_forward_callback_calls']==128
            and prod['backward_calls']==128 and prod['optimizer_created'] is False
            and prod['optimizer_steps']==prod['scheduler_steps']==prod['parameter_updates']==0,
            'Production preclip capture incomplete or mutated')
    for key in ('request_ids_sha256','token_manifest_sha256'):
        require(type(capture[key]) is str and re.fullmatch('[a-f0-9]{64}',capture[key]),'Missing fixed original '+key)
    require(file_sha(source)==expected['state_sha256'],'Original R24 changed during input verification')
    return receipt,restored,capture,state['policy']


def _trial(root,policy,gradients,endpoint,*,start_seed,trial,expected_binding_sha256,
           expected_input_receipt_sha256,mode,expected_tensors,failpoint=None):
    import torch
    require(type(start_seed) is int and start_seed in (11031,11032) and type(trial) is int and trial in (0,1),
            'Only four registered shadow slots exist')
    b,config,ops=_binding(root,expected_binding_sha256,mode)
    if mode=='production':
        require(torch.cuda.is_available() and torch.cuda.current_device()==0,'Production shadow needs CUDA 0')
        require(expected_tensors==496,'Production requires all 496 LoRA tensors')
    receipt,restored,capture,source_policy=_load_inputs(root,b,expected_binding_sha256,start_seed,trial,
        expected_input_receipt_sha256,endpoint,ops,config)
    require(list(policy)==list(gradients)==list(source_policy)==list(capture['production']['preclip_gradients'])
            and len(policy)==expected_tensors,'Exact original parameter inventory/order required')
    require(all('lora_' in name for name in policy),'Only LoRA shadow tensors are permitted')
    for name,value in policy.items():
        require(torch.is_tensor(value) and not isinstance(value,torch.nn.Parameter) and not value.requires_grad
                and value.dtype==torch.float32 and value.shape==gradients[name].shape
                and bool(torch.isfinite(value).all()) and gradients[name].dtype==torch.float32
                and not gradients[name].requires_grad and bool(torch.isfinite(gradients[name]).all()),
                'Detached FP32 policy/gradient copies required')
        require(str(value.device)==('cuda:0' if mode=='production' else 'cpu'),'Wrong shadow arithmetic device')
        require(torch.equal(gradients[name].detach().cpu(),capture['production']['preclip_gradients'][name]),
                'Supplied preclip gradient differs from actual capture')
    require(ops.state_hash(policy)==endpoint['policy_sha256'],'Supplied policy is not original R24')
    policy_before=ops.state_hash(policy);gradient_before=ops.state_hash(gradients)
    identity_before=copy.deepcopy(endpoint)
    claims=root/CLAIMS_NAME
    claims.mkdir(exist_ok=True)
    require(claims.resolve()==claims,'Claims directory cannot be symlinked')
    previous=None
    if trial==1:
        olddir=claims/f'start-{start_seed}'/'trial-0'
        previous=read_json(olddir/'COMPLETE.json')
        oldclaim=read_json(olddir/'CLAIM.json')
        require(oldclaim['start_seed']==start_seed and oldclaim['trial']==0
                and oldclaim['technical_binding_sha256']==expected_binding_sha256
                and oldclaim['design_freeze_sha256']==b['design_freeze']['sha256']
                and oldclaim['input_receipt_sha256']==previous['input_receipt_sha256'],
                'Trial0 claim/complete chain differs')
        require(previous['schema']=='sr-satdose-shadow-trial-v1' and previous['start_seed']==start_seed
                and previous['trial']==0 and previous['status']=='SHADOW_TRIAL_COMPLETE_NOT_GPU_ADMISSION' and previous['mode']==mode
                and previous['technical_binding_sha256']==expected_binding_sha256
                and previous['optimizer_body_calls']==previous['optimizer_steps']==1,
                'Trial0 is incomplete/UNKNOWN')
        oldinput,oldrestore,oldcapture,_=_load_inputs(root,b,expected_binding_sha256,start_seed,0,
            previous['input_receipt_sha256'],endpoint,ops,config)
        require(restored['loader_instance_id']!=oldrestore['loader_instance_id'], 'Trial1 must use an independent R24 reload')
        require(receipt['capture']['path']!=oldinput['capture']['path'] and
                receipt['restore']['path']!=oldinput['restore']['path'],'Trial1 must recapture/reload independently')
        require(capture['request_ids_sha256']==oldcapture['request_ids_sha256'] and
                capture['token_manifest_sha256']==oldcapture['token_manifest_sha256'],'Replay changed original requests/tokens')
        require(all(torch.equal(gradients[n].cpu(),oldcapture['production']['preclip_gradients'][n]) for n in gradients),
                'Restored production gradient replay not bit-exact')
        require(file_sha(olddir/'RESULT.pt')==previous['result_sha256'],'Trial0 evidence changed')
    parent=claims/f'start-{start_seed}';parent.mkdir(exist_ok=True)
    require(parent.resolve()==parent,'Claims parent cannot be symlinked')
    destination=parent/f'trial-{trial}'
    # Atomic exclusive directory creation is itself a durable reservation. Any
    # pre-existing directory, including one with no completed CLAIM, is UNKNOWN.
    destination.mkdir(exist_ok=False);_sync_dir(parent)
    write_once(destination/'CLAIM.json',dict(schema='sr-satdose-shadow-claim-v1',mode=mode,
        status='RESERVED_UNKNOWN_NO_RETRY',start_seed=start_seed,trial=trial,
        technical_binding_sha256=expected_binding_sha256,input_receipt_sha256=expected_input_receipt_sha256,
        design_freeze_sha256=b['design_freeze']['sha256'],max_optimizer_body_calls=1))
    calls=returned=0
    try:
        if failpoint:failpoint('after_claim')
        params={n:torch.nn.Parameter(v.detach().clone()) for n,v in policy.items()}
        require(all(params[n].data_ptr()!=policy[n].data_ptr() for n in params),'Shadow aliases caller policy')
        isolated=SimpleNamespace(parameters=lambda:iter(params.values()))
        optimizer,scheduler=ops.fresh_optimizer(isolated)
        require(not optimizer.state and scheduler.state_dict()['last_epoch']==0,'Shadow Adam must be fresh')
        import importlib
        hooks=importlib.import_module('torch.optim.optimizer')
        require(not hooks._global_optimizer_pre_hooks and not hooks._global_optimizer_post_hooks
                and not optimizer._optimizer_step_pre_hooks and not optimizer._optimizer_step_post_hooks,
                'Unexpected optimizer hooks')
        for n,param in params.items():param.grad=gradients[n].detach().to(param.device).clone()
        norm_before=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True))
        norm_after=math.sqrt(sum(float(v.grad.detach().double().square().sum()) for v in params.values()))
        require(math.isfinite(norm_before) and norm_before<=1000 and norm_after<=1.00001,'Shadow gradient sentinel failed')
        clipped={n:p.grad.detach().cpu().clone() for n,p in params.items()}
        require(all(torch.equal(clipped[n],capture['production']['postclip_gradients'][n]) for n in clipped),
                'GPU shadow clip differs from actual production capture')
        def before_step(opt,args,kwargs):
            nonlocal calls
            calls+=1;require(calls==1 and opt is optimizer,'Duplicate shadow Adam body')
        pre=optimizer.register_step_pre_hook(before_step)
        try:
            if failpoint:failpoint('before_adam')
            optimizer.step();returned+=1
            if failpoint:failpoint('after_adam')
        finally:pre.remove()
        require(calls==returned==1 and len(optimizer.state)==expected_tensors and
                all(float(s['step'])==1 for s in optimizer.state.values()),'Fresh Adam counter/inventory differs')
        require(scheduler.state_dict()['last_epoch']==0,'Shadow scheduler must not step')
        delta={n:(p.detach()-policy[n]).cpu().clone() for n,p in params.items()}
        require(all(bool(torch.isfinite(t).all()) for t in delta.values()),'Nonfinite shadow update')
        result=dict(schema='sr-satdose-shadow-tensors-v1',mode=mode,delta=delta,clipped=clipped,
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())
        if previous:
            old=torch.load(claims/f'start-{start_seed}'/'trial-0'/'RESULT.pt',map_location='cpu',weights_only=True)
            require(all(torch.equal(delta[n],old['delta'][n]) for n in delta) and
                    ops.state_hash(result['optimizer'])==ops.state_hash(old['optimizer']) and
                    ops.state_hash(result['scheduler'])==ops.state_hash(old['scheduler']), 'Restored shadow replay differs')
        require(ops.state_hash(policy)==policy_before and ops.state_hash(gradients)==gradient_before
                and endpoint==identity_before,'Caller policy/gradient/endpoint changed')
        with (destination/'RESULT.pt').open('xb') as stream:
            torch.save(result,stream);stream.flush();os.fsync(stream.fileno())
        complete=dict(schema='sr-satdose-shadow-trial-v1',status='SHADOW_TRIAL_COMPLETE_NOT_GPU_ADMISSION',mode=mode,
            start_seed=start_seed,trial=trial,technical_binding_sha256=expected_binding_sha256,
            input_receipt_sha256=expected_input_receipt_sha256,loader_instance_id=restored['loader_instance_id'],
            result_sha256=file_sha(destination/'RESULT.pt'),optimizer_body_calls=calls,optimizer_steps=returned,
            formal_optimizer_steps=0,scheduler_steps=0,caller_policy_unchanged=True,
            restore_replay_compared=trial==1,production_clip_bit_exact=True,
            norm_before=norm_before,norm_after=norm_after)
        write_once(destination/'COMPLETE.json',complete)
        return complete
    except BaseException as exc:
        write_once(destination/'HOLD.json',dict(status='HOLD_SHADOW_UNKNOWN_NO_RETRY',mode=mode,
            error_type=type(exc).__name__,error=str(exc),optimizer_pre_hook_calls=calls,
            optimizer_returned_calls=returned,recovery='READ_ONLY_AUDIT_NO_AUTOMATIC_RETRY'))
        raise


def one_trial(policy,gradients,endpoint,*,start_seed,trial,expected_binding_sha256,expected_input_receipt_sha256):
    """Production-only CUDA/496 path. No caller-controlled output root exists."""
    return _trial(PRODUCTION_ROOT,policy,gradients,endpoint,start_seed=start_seed,trial=trial,
        expected_binding_sha256=expected_binding_sha256,expected_input_receipt_sha256=expected_input_receipt_sha256,
        mode='production',expected_tensors=496)


def one_synthetic_trial_for_test(root,policy,gradients,endpoint,*,start_seed,trial,
        expected_binding_sha256,expected_input_receipt_sha256,failpoint=None):
    """Explicit CPU fixture path; cannot create production receipts or claims."""
    return _trial(Path(root).resolve(strict=True),policy,gradients,endpoint,start_seed=start_seed,trial=trial,
        expected_binding_sha256=expected_binding_sha256,expected_input_receipt_sha256=expected_input_receipt_sha256,
        mode='SYNTHETIC_CPU_TEST_ONLY',expected_tensors=len(policy),failpoint=failpoint)
