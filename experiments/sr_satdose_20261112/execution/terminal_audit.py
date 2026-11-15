"""Independent terminal CPU audit, loaded outside the frozen TRAIN namespace.

Load this file under a unique module name after a stdlib bootstrap has pinned
the external MAIN code and inserted the ORIGINAL immutable TRAIN code root.
Never copy/rewrite original execution/analysis/runtime modules into MAIN code.
All model, optimizer, generation and publication permissions remain absent.
The audit runs before a MAIN worker takes the exclusive worker lease.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import copy
import fcntl
import os
from pathlib import Path
import sys

from execution import (admission_authority as base, continuation_authority,
                       continuation_control as continuation, first_two_control as first_two,
                       trained_dev_control as trained_dev, training_transaction as tx)
from analysis import risk
from runtime.backend import object_sha

require=tx.require
PATHS=tuple(continuation.PATHS)
CHUNKS={3:8,9:16,17:24}
SPEC_KEYS={'schema','root','auditor','original_train_release','admission_spec','admission',
           'canary_spec','trusted_commits','dev_endpoints','continuation_results','final_risk'}
STATUS='TERMINAL_TRAINING_EVIDENCE_AUDITED_NOT_MAIN_ADMISSION'


def _populations(spec):
    require(isinstance(spec,dict) and set(spec)==SPEC_KEYS and
            spec['schema']=='sr-satdose-terminal-audit-spec-v1','Incomplete/extra terminal audit spec')
    commits=spec['trusted_commits'];dev=spec['dev_endpoints'];chunks=spec['continuation_results']
    require(isinstance(commits,dict) and set(commits)==set(PATHS) and
            all(isinstance(refs,list) and len(refs)==24 for refs in commits.values()),'Exactly four genuine step24 prefixes required')
    expected_dev={f'{s}/START/0' for s in (11031,11032)}|{f'{key}/{n}' for key in PATHS for n in (2,8,16)}
    expected_chunks={f'{key}/chunk-{a:03d}-{b:03d}' for key in PATHS for a,b in CHUNKS.items()}
    require(isinstance(dev,dict) and set(dev)==expected_dev,'Exactly fourteen original DEV endpoints required')
    require(all(isinstance(item,dict) and set(item)=={'receipt','binding'} for item in dev.values()),'Malformed DEV references')
    require(isinstance(chunks,dict) and set(chunks)==expected_chunks,'Exactly twelve completed fixed TRAIN chunks required')


@contextmanager
def _idle_lock(root,name):
    """Existing cooperative lease only; no lock file creation or process claims."""
    path=root/name
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Missing/noncanonical existing worker lease')
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        try:fcntl.flock(descriptor,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise PermissionError('HOLD: active worker/launcher; terminal audit forbidden') from exc
        opened=os.fstat(descriptor);before=path.stat()
        require((opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino),'Replaced worker lease')
        yield
        after=path.stat()
        require(not path.is_symlink() and (after.st_dev,after.st_ino)==(opened.st_dev,opened.st_ino),'Worker lease replaced during terminal audit')
    finally:os.close(descriptor)


def _merge(observed,other):
    for path,digest in other.files.items():observed.absolute(path,digest)
    for path,entries in other.directories.items():
        require(path not in observed.directories or observed.directories[path]==entries,'Observed directory inventory differs')
        observed.directories[path]=copy.deepcopy(entries)
    for field in ('model_links','model_directories'):
        target=getattr(observed,field)
        for key,value in getattr(other,field).items():
            require(key not in target or target[key]==value,'Observed model link/inventory differs')
            target[key]=copy.deepcopy(value)


def _modules(release,observed):
    package=Path(release['release_root'])
    require(package==first_two.PACKAGE==continuation.PACKAGE==trained_dev.PACKAGE,'Copied or shadowed frozen TRAIN package')
    for name,module in tuple(sys.modules.items()):
        if name.startswith(('execution.','analysis.','runtime.')):
            path=Path(getattr(module,'__file__',''))
            require(path.is_absolute() and path.is_relative_to(package),'Foreign original TRAIN namespace module: '+name)
            relative=str(path.relative_to(package))
            require(release['sources'].get(relative)==tx.file_sha(path),'Unpinned imported TRAIN module: '+name)
            observed.absolute(path,release['sources'][relative])
    observed.tree(package)
    require({str(path.relative_to(package)):digest for path,digest in observed.files.items()
             if path.is_relative_to(package)}==release['sources'],'Original TRAIN source inventory differs')


def _launcher(observed,spec,original,canary):
    root=observed.root;value=tx.read_json(observed.absolute(root/'first_two_launcher/RESULT.json'))
    require(value.get('schema')=='sr-satdose-first-two-launcher-result-v1' and
            value.get('status')=='FOUR_FIRST_TWO_PATHS_EXIT_OBSERVED_NOT_CANARY_ADMISSION' and
            value.get('permission')=='NO_STEP3_OR_CANARY_ADMISSION' and value.get('release')==spec['original_train_release'] and
            set(value.get('paths',{}))==set(PATHS),'Original first-two launcher incomplete/foreign')
    for field,count in [('train_responses',1024),('formal_optimizer_steps',8),('model_worker_processes',8)]:
        require(type(value.get(field)) is int and value[field]==count,'First-two launcher budget differs')
    for key,item in value['paths'].items():
        start,arm=key.split('/');path=root/f'first_two_launcher/start-{start}/{arm}/RESULT.json'
        require(tx.read_json(observed.absolute(path))==item and item.get('status')=='FIRST_TWO_PATH_EXIT_OBSERVED_NOT_CANARY_ADMISSION' and
                item.get('canary_spec')==canary.spec['trusted_specs'][key] and
                item.get('process_exit_proof')==canary.spec['process_exit_proofs'][key] and
                item.get('frozen_witnesses')==canary.spec['frozen_witnesses'][key], 'First-two launcher path provenance differs')
        for phase,label in [('baseline','baseline_result'),('reload','reload_result')]:
            reference=item[label];worker=observed.json(reference)
            require(observed.ref(reference)==root/f'first_two_control/start-{start}/{arm}/{phase}/RESULT.json' and
                    worker['release']==spec['original_train_release'] and worker['identity']==canary.identities[key],
                    'First-two worker result source/identity differs')
    return value


def _chunks(observed,spec,original,canary):
    total=0
    for name,reference in spec['continuation_results'].items():
        key,chunk=name.rsplit('/',1);start,arm=key.split('/');a,b=map(int,chunk.split('-')[1:])
        value=observed.json(reference)
        require(observed.ref(reference)==observed.root/f'continuation_control/start-{start}/{arm}/{chunk}/RESULT.json',
                'Continuation terminal result relocated')
        require(value['identity']==canary.identities[key] and value['original_train_release']==spec['original_train_release'] and
                value.get('process_exit_observed') is False and value.get('main_authorized') is False,
                'Chunk changed original identity or manufactured process/MAIN authority')
        for field,count in [('train_responses',128*(b-a+1)),('formal_optimizer_steps',b-a+1),('scheduler_steps',b-a+1)]:
            require(type(value.get(field)) is int and value[field]==count,'Continuation chunk budget differs')
        release,_=continuation.verify_release(observed.ref(value['release']),value['release']['sha256'])
        require(release['original_train_release']==spec['original_train_release'] and release['canary_spec']==spec['canary_spec'] and
                release['selection']==dict(start_seed=int(start),arm=arm,first_step=a,last_step=b),
                'Continuation release binds another invocation')
        require(len(value['risk_assessments'])==b-a+1 and value['post_risk']==value['risk_assessments'][-1],
                'Missing committed-step risk assessments')
        for step,ref in zip(range(a,b+1),value['risk_assessments'],strict=True):
            report=observed.json(ref)
            require(observed.ref(ref)==observed.ref(reference).parent/f'RISK_AFTER_STEP_{step:03d}.json' and
                    report.get('schema')=='sr-satdose-scientific-risk-assessment-v1' and
                    report.get('global_hold') is False and report.get('hold_reasons')==[] and
                    report.get('trusted_commits',{}).get(key)==spec['trusted_commits'][key][:step],
                    'Historical chunk risk is incomplete, foreign or HOLD')
        total+=value['formal_optimizer_steps']
    require(total==88,'Continuation must add exactly eighty-eight formal updates')


def _dev(observed,spec,original,canary):
    for key,refs in spec['dev_endpoints'].items():
        start,arm,n=key.split('/');step=int(n)
        directory=(observed.root/f'start_dev/start-{start}' if arm=='START' else
                   observed.root/f'trained_dev/start-{start}/{arm}/step-{step:03d}')
        result=tx.read_json(observed.absolute(directory/'RESULT.json'))
        require(result['receipt']==refs['receipt'] and result['endpoint_binding']==refs['binding'] and result['responses']==108,
                'DEV controller does not close trusted endpoint')
        observed.ref(result['close'])
        if arm=='START':
            require(observed.ref(original['start_dev_results'][start])==directory/'RESULT.json',
                    'Original TRAIN release binds another START DEV')
            release=observed.json(result['release']);code=Path(release['release_root'])
            require(release.get('schema')=='sr-satdose-start-dev-release-v1' and
                    release.get('status')=='AUDITED_FOR_START_DEV_ONLY' and
                    release.get('permission')=='START_DEV_ONLY_NO_TRAINING' and
                    release.get('evidence_root')==str(observed.root) and code.is_relative_to(observed.root) and
                    code!=observed.root and release.get('design_freeze_sha256')==original['design_freeze_sha256'] and
                    release.get('config_sha256')==original['config_sha256'],'Foreign original START DEV release')
            observed.tree(code)
            require({str(p.relative_to(code)):digest for p,digest in observed.files.items() if p.is_relative_to(code)}==
                    release['sources'],'Original START DEV code changed')
        else:
            release,_=trained_dev.verify_release(observed.ref(result['release']),result['release']['sha256'])
            require(release['original_train_release']==spec['original_train_release'] and release['canary_spec']==spec['canary_spec'] and
                    result['canary']==canary.receipt and result['admission']==spec['admission'] and
                    result['commit']==spec['trusted_commits'][f'{start}/{arm}'][step-1] and
                    result.get('global_scientific_hold') is False,'Trained DEV source/canary/commit/HOLD differs')
            post=observed.json(result['post_risk'])
            require(post.get('global_hold') is False and post.get('hold_reasons')==[],'Historical DEV scientific HOLD')


def audit(spec):
    """Return independently checked terminal evidence; no artifact is written."""
    spec=copy.deepcopy(spec);_populations(spec);root=Path(spec['root'])
    require(root==first_two.ROOT==continuation.ROOT==trained_dev.ROOT and root.resolve(strict=True)==root,
            'Terminal audit must use original evidence root')
    observed=base._Observed(root)
    require(base._absolute(spec['auditor'])==Path(__file__).resolve(),'Untrusted independent terminal auditor')
    observed.absolute(Path(__file__).resolve(),spec['auditor']['sha256'])
    with ExitStack() as stack:
        for lock in ('DIAGNOSTIC_WORKER.lock','FIRST_TWO_LAUNCHER.lock'):
            stack.enter_context(_idle_lock(root,lock))
        continuation._hold_free()
        original=first_two.verify_release(observed.ref(spec['original_train_release']),spec['original_train_release']['sha256'])
        _modules(original,observed)
        require(original['admission_spec']==spec['admission_spec'] and original['admission']==spec['admission'],
                'Terminal spec substituted original admission')
        admission_spec=observed.json(spec['admission_spec'])
        require(admission_spec['root']==str(root) and admission_spec['sources']['authority']==original['execution_sources']['admission_verifier'] and
                admission_spec['training_execution_sources_sha256']==object_sha(original['execution_sources']),
                'Original full admission source identity differs')
        candidate,admission_observed,_,_,freeze=base._validate(admission_spec)
        require(observed.json(spec['admission'])==candidate,'Original admitted evidence no longer reproduces')
        _merge(observed,admission_observed)
        canary_spec=observed.json(spec['canary_spec']);canary=continuation_authority.verify_canary(canary_spec)
        require(canary.execution_sources==original['execution_sources'],'Canary has another original execution identity')
        _merge(observed,canary._observed)
        continuation._cursor(spec['trusted_commits'])
        continuation._completed_attempts(spec['trusted_commits'],spec['dev_endpoints'],spec['continuation_results'])
        _launcher(observed,spec,original,canary);_chunks(observed,spec,original,canary);_dev(observed,spec,original,canary)
        risk_ref=original['execution_sources']['risk_verifier']
        require(base._absolute(risk_ref)==Path(risk.__file__).resolve(),'Shadowed original risk verifier')
        inputs=dict(config_ref=freeze['artifacts']['config'],freeze_ref=canary_spec['design_freeze'],
            clarification_ref=original['risk_clarification'],trusted_commits=spec['trusted_commits'],dev_endpoints=spec['dev_endpoints'],
            expected_source_sha256=risk_ref['sha256'])
        actual=risk.assess(root,**inputs)
        require(actual==observed.json(spec['final_risk']) and actual['status']=='NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED' and
                actual['global_hold'] is False and actual['hold_reasons']==[] and actual['pending_required_dev']==[],
                'Final risk is stale, incomplete, HOLD or not independently reproduced')
        require(all(actual['paths'][key]['completed_steps']==24 for key in PATHS),'Nonterminal actual risk cursor')
        for path,digest in actual['observed_files'].items():observed.absolute(Path(path),digest)
        for family in ('training_transactions','first_two_control','first_two_launcher','canary_replay',
                       'start_dev','trained_dev','continuation_control'):
            observed.tree(root/family)
        continuation._hold_free();continuation._cursor(spec['trusted_commits']);canary.stable();admission_observed.stable()
        _modules(original,observed);observed.stable()
        result=dict(schema='sr-satdose-terminal-independent-audit-v1',status=STATUS,permission='NO_TRAINING_OR_MAIN',
            spec_sha256=object_sha(spec),auditor=copy.deepcopy(spec['auditor']),original_train_release=copy.deepcopy(spec['original_train_release']),
            formal_optimizer_steps=96,train_responses=12288,dev_endpoints=14,dev_responses=1512,
            terminal_steps={key:24 for key in PATHS},global_hold=False,pending_required_dev=[],
            original_admission=copy.deepcopy(spec['admission']),canary=copy.deepcopy(canary.receipt),
            final_risk=copy.deepcopy(spec['final_risk']),source_and_input_bytes_reverified=True,
            audit_model_calls=0,audit_optimizer_calls=0,
            observed_files={str(path):digest for path,digest in observed.files.items()},
            observed_directories={str(path):sorted(entries) for path,entries in observed.directories.items()},
            observed_model_links={str(path):dict(link=str(link),target=str(target)) for path,(link,target) in observed.model_links.items()},
            observed_model_directories={str(path):sorted(entries) for path,entries in observed.model_directories.items()},
            limitations=['No MAIN generation or further TRAIN authority is granted.',
                'Worker inactivity is verified through existing cooperative leases, not a new operating-system process-death observation.',
                'Original scientific PREFIX exception remains preserved in original admission.'])
    return result
