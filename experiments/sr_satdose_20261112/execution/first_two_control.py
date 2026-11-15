"""Fixed per-path first-two-step worker and separate read-only reload worker.

An external controller pins this complete release and launches distinct processes.
This module never fabricates process-exit proof, canary PASS or step3 authority.
Baseline creates exactly 256 TRAIN responses and two native updates, then original
step2 teacher evidence. Reload uses the same two committed states and no sampler.
Every occupied incomplete slot is HOLD/UNKNOWN; no automatic retry/resampling.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import os
from pathlib import Path
import re
import subprocess
import time

from . import (admission_authority, canary_audit, canary_replay, storage_ready,
               training_runtime, training_step, training_state, training_transaction as tx)
from analysis import evaluation, calibration
from runtime import fork_loader
from runtime.backend import FrozenRegistry, object_sha

ROOT=Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
PACKAGE=Path(__file__).resolve().parents[1]
FREEZE_SHA='6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
CONFIG_SHA='58841639b0f36a69ac77824bc8c683ef7fbeb5c50a41e9975b89b54880f4ecfe'
EXPERIMENT=tx.EXPERIMENT
require=tx.require
STARTS=(11031,11032);ARMS=('T0','T1')
BASELINE_STATUS='FIRST_TWO_BASELINE_COMPLETE_NOT_CANARY_ADMISSION'
RELOAD_STATUS='FIRST_TWO_RELOAD_COMPLETE_NOT_CANARY_ADMISSION'


def _sha(path):
    path=Path(path)
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Noncanonical ordinary file')
    return tx.file_sha(path)


def _ref(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=_sha(path))
def _bound(reference):return tx.bound(ROOT,reference)
def _read(reference):return tx.read_json(_bound(reference))


def _inventory(directory):
    result={}
    for path in directory.rglob('*'):
        require(path.resolve(strict=True)==path and not path.is_symlink(),'Release symlink/noncanonical entry')
        if path.is_file():result[str(path.relative_to(directory))]=_sha(path)
        else:require(path.is_dir(),'Nonordinary release entry')
    return result


def _execution_sources(release):
    sources=release['execution_sources']
    modules={'first_two_control':__file__,'training_runtime':training_runtime.__file__,
        'training_step':training_step.__file__,'training_state':training_state.__file__,
        'training_transaction':tx.__file__,'production':training_step.production.__file__,
        'calibration':calibration.__file__,'satdose_core':calibration.core.__file__,
        'scorer_bindings':calibration.PACKAGE/'config/source_bindings.json',
        'admission_verifier':admission_authority.__file__,'canary_replay':canary_replay.__file__,
        'frozen_fingerprint':canary_audit.__file__,'storage_ready':storage_ready.__file__,
        'evaluation_analysis':evaluation.__file__}
    require(set(modules)|{'canary_controller'}<=set(sources),'Incomplete first-two source inventory')
    for name,reference in sources.items():
        path=Path(reference['path'])
        require(path.is_absolute() and path.is_relative_to(PACKAGE) and _sha(path)==reference['sha256'] and
                release['sources'].get(str(path.relative_to(PACKAGE)))==reference['sha256'],
                'Execution source outside authenticated release: '+name)
        if name in modules:require(path==Path(modules[name]).resolve(),'Shadowed first-two dependency: '+name)
    return copy.deepcopy(sources)


def verify_release(path,expected_sha):
    path=Path(path)
    require(re.fullmatch('[0-9a-f]{64}',expected_sha or '') and _sha(path)==expected_sha and
            path.is_relative_to(ROOT) and not path.is_relative_to(PACKAGE),'Untrusted first-two release receipt')
    release=tx.read_json(path)
    expected=dict(schema='sr-satdose-first-two-release-v1',status='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY',
        permission='FOUR_PATHS_MAX_STEP_2_NO_STEP3',experiment_id=EXPERIMENT,evidence_root=str(ROOT),
        release_root=str(PACKAGE),design_freeze_sha256=FREEZE_SHA,config_sha256=CONFIG_SHA)
    require(all(release.get(k)==v for k,v in expected.items()),'First-two release scope/identity differs')
    require(_inventory(PACKAGE)==release['sources'],'Full first-two release inventory differs')
    require(_sha(ROOT/'DESIGN_FREEZE.json')==FREEZE_SHA,'Frozen design changed')
    freeze=tx.read_json(ROOT/'DESIGN_FREEZE.json')
    require({p.removeprefix('runtime/'):digest for p,digest in release['sources'].items()
             if p.startswith('runtime/') and p.endswith('.py')}==freeze['runtime_sources'],
            'Original runtime inventory changed')
    require(storage_ready.ROOT==ROOT and storage_ready.FREEZE_SHA==FREEZE_SHA,'Foreign storage implementation')
    _execution_sources(release)
    return release


@contextmanager
def _lease():
    require(ROOT.resolve(strict=True)==ROOT,'Noncanonical fixed root')
    path=ROOT/'DIAGNOSTIC_WORKER.lock'
    descriptor=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=os.fstat(descriptor);actual=path.stat()
        require((before.st_dev,before.st_ino)==(actual.st_dev,actual.st_ino),'Global worker lock replaced')
        yield
    finally:os.close(descriptor)


def _paths(start,arm):
    require(type(start) is int and start in STARTS and arm in ARMS,'Unknown first-two path')
    return ROOT/'first_two_control'/f'start-{start}'/arm


def _no_unknown(current=None):
    for name in ('HOLD.json','ERROR.json','training_control/HOLD.json','training_control/ERROR.json',
                 'first_two_launcher/HOLD.json','first_two_launcher/ERROR.json'):
        require(not (ROOT/name).exists(),'Global training HOLD/error')
    base=ROOT/'first_two_control'
    if base.exists():
        require(base.resolve(strict=True)==base and not base.is_symlink(),'Noncanonical controller root')
        for sd in base.iterdir():
            require(sd.name in ('start-11031','start-11032') and sd.is_dir() and not sd.is_symlink(),'Unknown first-two start')
            for ad in sd.iterdir():
                require(ad.name in ARMS and ad.is_dir() and not ad.is_symlink(),'Unknown first-two arm')
                for slot in ad.iterdir():
                    require(slot.name in ('baseline','reload') and slot.is_dir() and not slot.is_symlink(),
                            'Unknown first-two phase')
                    require(not (slot/'HOLD.json').exists(),'First-two technical HOLD')
                    if current is not None and slot==current:continue
                    require((slot/'RESULT.json').is_file(),'Occupied incomplete first-two attempt; no retry')
                    result=tx.read_json(slot/'RESULT.json')
                    require(result.get('status')==(BASELINE_STATUS if slot.name=='baseline' else RELOAD_STATUS) and
                            result.get('permission')=='NO_STEP3_OR_CANARY_ADMISSION','Invalid completed first-two phase')
    transaction=ROOT/'training_transactions'
    if transaction.exists():
        require(transaction.resolve(strict=True)==transaction and not transaction.is_symlink(),'Noncanonical transaction root')
        for sd in transaction.iterdir():
            if sd.name=='TRANSACTION.lock':
                require(sd.is_file() and not sd.is_symlink(),'Invalid transaction lock');continue
            require(sd.name in ('start-11031','start-11032') and sd.is_dir() and not sd.is_symlink(),
                    'Unknown transaction start')
            for ad in sd.iterdir():
                require(ad.name in ARMS and ad.is_dir() and not ad.is_symlink(),'Unknown transaction arm')
                for step in ad.iterdir():
                    require(step.name in ('step-001','step-002') and step.is_dir() and not step.is_symlink() and
                            (step/'COMMIT.json').is_file(),'Unknown/uncommitted transaction or step beyond first two')
        for p in transaction.rglob('*'):
            require(not p.is_symlink(),'Symlink in transaction state')
            require(not p.name.startswith('.staging-') and p.name not in ('HOLD.json','ERROR.json'),
                    'Unknown/uncommitted transaction; no automatic retry')
    canary=ROOT/'canary_replay'
    if canary.exists():
        require(canary.resolve(strict=True)==canary and not canary.is_symlink(),'Noncanonical canary root')
        for sd in canary.iterdir():
            require(sd.name in ('start-11031','start-11032') and sd.is_dir() and not sd.is_symlink(),'Unknown canary start')
            for ad in sd.iterdir():
                require(ad.name in ARMS and ad.is_dir() and not ad.is_symlink(),'Unknown canary arm')
                for entry in ad.iterdir():
                    require(entry.name in ('TECHNICAL_EXECUTION.json','baseline','restored') and not entry.is_symlink() and
                            (entry.is_file() if entry.name=='TECHNICAL_EXECUTION.json' else entry.is_dir()),
                            'Unknown canary slot')
    canary_replay._no_unknown(ROOT)


def _registry(freeze):
    bound=lambda key:_bound(freeze['artifacts'][key])
    return FrozenRegistry(bound('config'),ROOT,ROOT/'DESIGN_FREEZE.json',
        [bound(key) for key in ('train_manifest','dry_manifest','dev_manifest','prefix_calib_manifest','main_manifest')],
        bound('inputs'),bound('images'))


def _initial_dev(release,freeze):
    require(set(release['start_dev_results'])==set(release['start_dev_analyses'])=={'11031','11032'},
            'Both complete START DEV endpoints and independent analyses required')
    processes=[];results={}
    for start in STARTS:
        key=str(start);ref=release['start_dev_results'][key]
        require(_bound(ref)==ROOT/f'start_dev/start-{start}/RESULT.json','START DEV result relocated')
        result=_read(ref)
        require(result.get('schema')=='sr-satdose-start-dev-result-v1' and result.get('status')=='START_DEV_ENDPOINT_COMPLETE' and
                result.get('permission')=='NO_TRAINING' and result.get('start_seed')==start and
                result.get('responses')==108 and result.get('design_freeze_sha256')==FREEZE_SHA,
                'Missing/wrong START DEV completion')
        directory=ROOT/f'start_dev/start-{start}'
        require(not any(p.name in ('HOLD.json','ERROR.json') or p.is_symlink() for p in directory.rglob('*')),
                'START DEV technical HOLD/symlink')
        old_release=_read(result['release']);old_admission=_read(result['admission'])
        require(old_release.get('schema')=='sr-satdose-start-dev-release-v1' and
                old_release.get('admission')==result['admission'] and
                old_release.get('design_freeze_sha256')==FREEZE_SHA and old_release.get('config_sha256')==CONFIG_SHA and
                old_admission.get('gpu_report_sha256')==_read(release['admission'])['gpu_report_sha256'] and
                old_admission.get('prefix_gate_decision')=='PASS_USER_OVERRIDE',
                'START DEV predecessor source/admission lineage differs')
        original_release=Path(old_release['release_root'])
        require(original_release.is_relative_to(ROOT) and _inventory(original_release)==old_release['sources'],
                'Original START DEV source release changed')
        process=result['process_identity']
        require(set(process)=={'host','pid','start_ticks','boot_id'} and type(process['pid']) is int and process['pid']>0 and
                all(process[k] for k in ('host','start_ticks','boot_id')),'Missing actual START DEV process incarnation')
        processes.append(process)
        require(_bound(result['receipt'])==directory/'endpoint/RECEIPT.json' and
                _bound(result['endpoint_binding'])==directory/'ENDPOINT_BINDING.json' and
                _bound(result['close'])==directory/'CLOSE.json','START DEV endpoint evidence relocated')
        close=_read(result['close'])
        require(close.get('attempted_requests')==108 and all(close.get(k)==0 for k in
                ('optimizer_steps','scheduler_steps','parameter_updates')),'START DEV side effects/count differ')
        actual=evaluation.analyze_completed_endpoint(_bound(freeze['artifacts']['config']),ROOT/'DESIGN_FREEZE.json',ROOT,
            _bound(result['receipt']),expected_receipt_sha256=result['receipt']['sha256'],
            endpoint_binding_path=_bound(result['endpoint_binding']),
            expected_endpoint_binding_sha256=result['endpoint_binding']['sha256'])
        require(actual==_read(release['start_dev_analyses'][key]),'Initial DEV independent CPU analysis differs')
        require(actual['summary']['scope']=='DEV' and actual['summary']['endpoint']['start_seed']==start and
                actual['summary']['endpoint']['arm']=='START' and actual['summary']['endpoint']['step']==0 and
                actual['summary']['metrics']['assigned']==108,'Wrong initial DEV analysis scope')
        results[key]=dict(result=ref,analysis=release['start_dev_analyses'][key])
    require(processes[0]!=processes[1],'START DEV endpoints reused one process')
    return results


def _risk(reference):
    value=_read(reference)
    require(value.get('schema')=='sr-satdose-training-risk-clarification-v1' and value.get('experiment_id')==EXPERIMENT and
            value.get('authorization')=='explicit_user_confirmation_before_any_TRAIN_generation' and
            value.get('user_answer')=='按完整建议冻结：训练率只统计每窗 384 条冲突回答' and
            value.get('start_seeds')==[11031,11032] and value.get('arm')=='T1' and
            value.get('windows')=='nonoverlapping steps 1-4, 5-8, 9-12, 13-16, 17-20, 21-24' and
            value.get('evaluate_each_path_independently') is True and value.get('per_window_conflict_g8_count')==48 and
            value.get('per_window_conflict_response_count')==384 and value.get('consecutive_qualifying_windows')==2 and
            value.get('scientific_thresholds_changed') is False and
            value.get('window_conditions_all_required')==dict(mean_d_less_than=.75,exact_authorised_rate_less_than=.8,
                exact_authorised_denominator='all 384 conflict responses only') and
            value.get('additional_condition')==dict(latest_same_path_DEV_conflict_response_count=32,S_less_than=.8,
                at_steps_8_and_16='complete current DEV before deciding') and
            value.get('trigger')=='any one T1 path satisfying all conditions causes global HOLD for all paths',
            'Frozen confirmed training risk interpretation differs')
    return value


def preflight(release_path,release_sha,start,arm,phase,*,baseline_result_sha256=None):
    require(phase in ('baseline','reload'),'Only baseline or read-only reload')
    release=verify_release(release_path,release_sha);_no_unknown()
    current_process=canary_replay._process_identity()
    for prior in (ROOT/'first_two_control').glob('start-*/*/*/RESULT.json'):
        require(tx.read_json(prior).get('process')!=current_process,
                'Each first-two/reload worker must use a fresh process incarnation')
    directory=_paths(start,arm);slot=directory/phase
    require(not slot.exists() and not slot.is_symlink(),'Occupied first-two slot; no automatic retry')
    sources=_execution_sources(release)
    spec=_read(release['admission_spec'])
    require(spec['root']==str(ROOT) and spec['sources']['authority']==sources['admission_verifier'] and
            spec['training_execution_sources_sha256']==object_sha(sources),'Admission/execution source identity differs')
    authority=admission_authority.FirstTwoStepAdmission(spec,admission=release['admission'])
    freeze=tx.read_json(ROOT/'DESIGN_FREEZE.json');registry=_registry(freeze)
    dev=_initial_dev(release,freeze);_risk(release['risk_clarification'])
    budget=storage_ready._budget(release['storage_budget'],storage_ready._now())
    require(budget.get('scope')=='FIRST_TWO_AND_CANARY_ONLY','Wrong first-two storage scope')
    if phase=='baseline':
        require(baseline_result_sha256 is None,'Baseline cannot supply an old-result authority')
        require(not directory.exists() and not (ROOT/f'training_transactions/start-{start}/{arm}').exists() and
                not canary_replay._paths(ROOT,start,arm).exists(),'Path already allocated; no fresh duplicate fork')
    else:
        prior=directory/'baseline/RESULT.json'
        require(re.fullmatch('[0-9a-f]{64}',baseline_result_sha256 or '') and
                _sha(prior)==baseline_result_sha256,'Reload requires externally pinned completed baseline')
    return dict(release=release,release_ref=_ref(Path(release_path)),sources=sources,authority=authority,
        freeze=freeze,registry=registry,source_lock=_bound(freeze['artifacts']['source_lock']),
        start=start,arm=arm,phase=phase,slot=slot,dev=dev,
        baseline_ref=(_ref(directory/'baseline/RESULT.json') if phase=='reload' else None))


def _pre_model_storage(directory,budget_ref):
    began=time.monotonic()
    budget=storage_ready._budget(budget_ref,storage_ready._now())
    require(budget.get('scope')=='FIRST_TWO_AND_CANARY_ONLY','Wrong storage scope')
    before=storage_ready._observe(directory)
    require(before['effective_available_bytes']>=budget['required_bytes'],'Insufficient pre-model quota')
    probe=storage_ready._probe(directory);after=storage_ready._observe(directory)
    storage_ready._budget(budget_ref,storage_ready._now())
    require(after['effective_available_bytes']>=budget['required_bytes'] and
            time.monotonic()-began<=budget['max_observation_age_seconds'],'Insufficient/stale quota after fsync')
    return dict(budget=budget_ref,observations=[before,after],probe=probe,permission='NO_TRAINING_ADMISSION')


def _allocation():
    require(os.environ.get('SLURM_JOB_ID','').isdigit() and os.environ.get('SLURM_NTASKS')=='1' and
            os.environ.get('SLURM_NNODES')=='1' and os.environ.get('SLURM_RESTART_COUNT','0')=='0',
            'One fresh non-requeued Slurm worker required')
    job=os.environ['SLURM_JOB_ID']
    result=subprocess.run(['scontrol','show','job',job,'-o'],text=True,capture_output=True)
    require(result.returncode==0,'Cannot independently verify Slurm allocation')
    fields=dict(word.split('=',1) for word in result.stdout.split() if '=' in word)
    require(fields.get('JobId')==job and fields.get('JobState')=='RUNNING' and
            fields.get('QOS')=='soujanya-poria-startfund-2026-03' and
            fields.get('Account')=='rose' and fields.get('NumNodes')=='1' and
            'gres/gpu:pro6000=2' in fields.get('AllocTRES','').split(','),
            'Actual teacher-QoS two-PRO6000 allocation required')
    return dict(job_id=os.environ['SLURM_JOB_ID'],step_id=os.environ.get('SLURM_STEP_ID',''),task_count=1,
                qos=fields['QOS'],observation=fields)


def _baseline(handle,store,ctx):
    """Compose real step adapter; tests may replace GPU kernels, never native Adam."""
    directory=ctx['slot'];sources=ctx['sources']
    before=canary_audit.fingerprint_first_two(handle,phase='BEFORE_FIRST_STEP',execution_sources=sources,
        expected_auditor_sha256=sources['frozen_fingerprint']['sha256'])
    tx.write_once(directory/'FROZEN_BEFORE.json',before)
    commits={}
    for step in (1,2):
        result=training_step.run_step(handle,store,step=step,admission=ctx['authority'],
            storage_ready=storage_ready.make_storage_ready(ctx['release']['storage_budget']),execution_sources=sources)
        require(result.get('status')=='COMMITTED' and handle.endpoint()['step']==step,'First-two step did not commit')
        commits[str(step)]=_ref(store.path/f'step-{step:03d}/COMMIT.json')
    after=canary_audit.fingerprint_first_two(handle,phase='AFTER_STEP2',execution_sources=sources,
        expected_auditor_sha256=sources['frozen_fingerprint']['sha256'])
    tx.write_once(directory/'FROZEN_AFTER.json',after)
    require(before['parameters']==after['parameters'] and before['gpu_inventory']==after['gpu_inventory'],
            'Actual base/vision/non-LoRA parameters changed')
    base=canary_replay._paths(ROOT,store.start,store.arm)
    base.mkdir(parents=True,exist_ok=False);tx.fsync_directory(base.parent)
    spec=dict(schema=canary_replay.SCHEMA,permission='READONLY_STEP2_CANARY_ONLY',identity=store.identity,
        design_freeze_sha256=store.freeze_sha,execution_sources=sources,
        baseline_audit=str(handle.output.relative_to(ROOT)),restored_audit=str((base/'restored/audit').relative_to(ROOT)),
        slots={phase:str((base/phase).relative_to(ROOT)) for phase in ('baseline','restored')},commits=commits)
    tx.write_once(base/'TECHNICAL_EXECUTION.json',spec)
    canary_ref=_ref(base/'TECHNICAL_EXECUTION.json')
    baseline=canary_replay.baseline(handle,store,spec_path=base/'TECHNICAL_EXECUTION.json',
        expected_spec_sha256=canary_ref['sha256'],execution_sources=sources)
    require(baseline['status']==canary_replay.BASELINE and baseline['request_count']==128,'Incomplete step2 baseline')
    return dict(schema='sr-satdose-first-two-result-v1',status=BASELINE_STATUS,permission='NO_STEP3_OR_CANARY_ADMISSION',
        start_seed=store.start,arm=store.arm,identity=store.identity,release=ctx['release_ref'],
        admission=ctx['release']['admission'],initial_dev=ctx['dev'],risk_clarification=ctx['release']['risk_clarification'],
        process=canary_replay._process_identity(),genesis=store.genesis_reference,commits=commits,
        train_responses=256,formal_optimizer_steps=2,scheduler_steps=2,canary_spec=canary_ref,
        baseline=_ref(base/'baseline/COMPLETE.json'),frozen_witnesses=dict(before=_ref(directory/'FROZEN_BEFORE.json'),
        after=_ref(directory/'FROZEN_AFTER.json')),process_exit_observed=False)


def _reload(ctx):
    baseline=_read(ctx['baseline_ref'])
    require(baseline.get('status')==BASELINE_STATUS and baseline.get('release')==ctx['release_ref'] and
            baseline.get('process')!=canary_replay._process_identity(),'Reload requires prior baseline in another process')
    identity=baseline['identity'];source=tx.read_json(ctx['source_lock'])
    native=fork_loader.load_native_ops(source['native_src'],source['native_files'])
    store=tx.TransactionStore(ROOT,ROOT/'DESIGN_FREEZE.json',FREEZE_SHA,identity,native=native,
                              genesis_reference=baseline['genesis'])
    canary=baseline['canary_spec'];path=_bound(canary)
    result=canary_replay.restored(store,ctx['registry'],ctx['source_lock'],spec_path=path,
        expected_spec_sha256=canary['sha256'],execution_sources=ctx['sources'])
    require(result['status']==canary_replay.RESTORED,'Reload actual tensor equality failed')
    return dict(schema='sr-satdose-first-two-result-v1',status=RELOAD_STATUS,permission='NO_STEP3_OR_CANARY_ADMISSION',
        start_seed=ctx['start'],arm=ctx['arm'],release=ctx['release_ref'],identity=identity,
        process=canary_replay._process_identity(),canary_spec=canary,restored=_ref(path.parent/'restored/COMPLETE.json'),
        train_responses=0,formal_optimizer_steps=0,scheduler_steps=0,process_exit_observed=False)


def execute(release_path,release_sha,start,arm,phase,*,baseline_result_sha256=None):
    _allocation()
    with _lease():
        ctx=preflight(release_path,release_sha,start,arm,phase,baseline_result_sha256=baseline_result_sha256);slot=ctx['slot']
        slot.mkdir(parents=True,exist_ok=False);tx.fsync_directory(slot.parent)
        tx.write_once(slot/'CLAIM.json',dict(schema='sr-satdose-first-two-claim-v1',status='UNKNOWN_UNTIL_COMPLETE_NO_RETRY',
            start_seed=start,arm=arm,phase=phase,process=canary_replay._process_identity(),release=ctx['release_ref'],allocation=_allocation()))
        handle=None
        try:
            tx.write_once(slot/'STORAGE.json',_pre_model_storage(slot,ctx['release']['storage_budget']))
            if phase=='baseline':
                output=slot/'audit';output.mkdir();tx.fsync_directory(slot)
                handle=training_runtime.build_training_runtime(ctx['registry'],ctx['source_lock'],start_seed=start,arm=arm,
                    output_dir=output,execution_sources=ctx['sources'])
                store=tx.TransactionStore(ROOT,ROOT/'DESIGN_FREEZE.json',FREEZE_SHA,handle.path_identity,
                    native=handle.native_ops,genesis_reference=handle.genesis_reference)
                result=_baseline(handle,store,ctx)
            else:result=_reload(ctx)
            verify_release(release_path,release_sha)
            if handle is not None:handle.discard();handle=None
            tx.write_once(slot/'RESULT.json',result);return result
        except BaseException as exc:
            if handle is not None:handle.discard()
            if not (slot/'HOLD.json').exists():
                tx.write_once(slot/'HOLD.json',dict(status='HOLD_FIRST_TWO_NO_RETRY',permission='NO_TRAINING',
                    error_type=type(exc).__name__,error=str(exc)))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-receipt',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    parser.add_argument('--start-seed',type=int,choices=STARTS,required=True);parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--phase',choices=('baseline','reload'),required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--baseline-result-sha256',help='Externally recorded baseline RESULT SHA; required for reload')
    args=parser.parse_args(argv)
    try:
        fn=execute if args.execute else preflight
        value=fn(args.release_receipt,args.release_sha256,args.start_seed,args.arm,args.phase,
                 baseline_result_sha256=args.baseline_result_sha256)
        if not args.execute:value=dict(status='FIRST_TWO_PREFLIGHT_COMPLETE_NO_MODEL_CALLS',start_seed=args.start_seed,arm=args.arm,phase=args.phase)
    except Exception as exc:
        value=dict(status='HOLD_FIRST_TWO_CONTROL',permission='NO_TRAINING',error_type=type(exc).__name__,error=str(exc))
    print(tx.canonical(value).decode());return 2 if value['status'].startswith('HOLD') else 0


if __name__=='__main__':raise SystemExit(main())
