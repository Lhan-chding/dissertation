"""Authenticate the original four-path canary without extending its audit scope.

The independent tensor audit runs only when every chain has exactly two steps.
After continuation its trusted immutable report, all observed bytes and source
pins remain verifiable; this module never truncates a later transaction chain.
Canary evidence alone is not a TRAIN/DEV/MAIN generation authorization.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

from . import admission_authority as base, canary_audit, training_transaction as tx
from runtime.backend import object_sha

require=tx.require
PATHS=set(canary_audit.PATHS)
SPEC_KEYS={'root','helper','auditor','report','trusted_specs','frozen_witnesses',
           'process_exit_proofs','design_freeze','config','execution_sources'}


def _positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0


def _control_state(root,audited_files):
    """Reject new technical incidents/attempts before accepting any cache."""
    directory=root/'canary_replay'
    require(directory.resolve(strict=True)==directory and directory.is_dir(),'Missing original canary directory')
    require({p.name for p in directory.iterdir()}=={'start-11031','start-11032'},'Unknown/missing canary start')
    for start in directory.iterdir():
        require(start.is_dir() and not start.is_symlink() and {p.name for p in start.iterdir()}=={'T0','T1'},
                'Unknown/missing canary arm')
        for arm in start.iterdir():
            require(arm.is_dir() and not arm.is_symlink(),'Noncanonical canary arm')
            require({p.name for p in arm.iterdir() if p.is_dir()}=={'baseline','restored'},'Unknown/missing canary phase')
    for protected in (directory,root/'first_two_control'):
        if protected.exists():
            require(protected.resolve(strict=True)==protected and protected.is_dir(),'Noncanonical original control directory')
            require(not any(p.is_symlink() or p.name in ('HOLD.json','ERROR.json') or p.name.endswith('.ERROR.json')
                            for p in protected.rglob('*')),'Original canary/first-two technical HOLD/error')
    for begin in directory.rglob('*.BEGIN.json'):
        capture=begin.with_name(begin.name.replace('.BEGIN.json','.CAPTURE.json'))
        require(audited_files.get(str(begin))==tx.file_sha(begin) and capture.is_file() and
                audited_files.get(str(capture))==tx.file_sha(capture),'Unknown canary BEGIN outside original token relocation')
    first=root/'first_two_control'
    if first.exists():
        require({p.name for p in first.iterdir()}=={'start-11031','start-11032'},'Unknown/incomplete first-two starts')
        for start in first.iterdir():
            require(start.is_dir() and {p.name for p in start.iterdir()}=={'T0','T1'},'Unknown/incomplete first-two arms')
            for arm in start.iterdir():
                require(arm.is_dir() and {p.name for p in arm.iterdir()}=={'baseline','reload'},'Unknown/incomplete first-two phase')
                for phase in arm.iterdir():
                    require(phase.is_dir() and (phase/'RESULT.json').is_file(),'Incomplete first-two worker slot')
                    value=tx.read_json(phase/'RESULT.json')
                    expected='FIRST_TWO_'+('BASELINE' if phase.name=='baseline' else 'RELOAD')+'_COMPLETE_NOT_CANARY_ADMISSION'
                    require(value.get('status')==expected and value.get('permission')=='NO_STEP3_OR_CANARY_ADMISSION',
                            'Uncompleted original first-two worker')
                    begins=list(phase.rglob('*.BEGIN.json'))
                    require(len(begins)==(256 if phase.name=='baseline' else 0),'Unknown/missing first-two generation attempts')
                    for begin in begins:
                        require(all(begin.with_name(begin.name.replace('.BEGIN.json',suffix)).is_file()
                                    for suffix in ('.CAPTURE.json','.COMPLETE.json')),'Unclosed first-two BEGIN')
    for name in ('start_dev','trained_dev','trained_dev_control','continuation_control','training_control','first_two_launcher'):
        mutable=root/name
        if mutable.exists():
            require(mutable.resolve(strict=True)==mutable and not mutable.is_symlink(),'Noncanonical active control directory')
            require(not any(p.is_symlink() or p.name in ('HOLD.json','ERROR.json') or p.name.endswith('.ERROR.json')
                            for p in mutable.rglob('*')),'Global control-plane technical HOLD/error')
    canary_audit.replay._no_unknown(root)


def _inputs(spec,*,published):
    require(set(spec)==SPEC_KEYS|({'published'} if published else set()),'Incomplete/extra canary authority input')
    root=Path(spec['root']);require(root.is_absolute() and root.resolve(strict=True)==root,'Noncanonical canary root')
    observed=base._Observed(root)
    report=observed.json(spec['report'])
    _control_state(root,report.get('observed_files',{}))
    observed.tree(root/'canary_replay')
    if (root/'first_two_control').exists():observed.tree(root/'first_two_control')
    require(base._absolute(spec['helper'])==Path(__file__).resolve() and
            spec['execution_sources'].get('continuation_verifier')==spec['helper'],
            'Continuation helper absent from original execution identity')
    require(base._absolute(spec['auditor'])==Path(canary_audit.__file__).resolve() and
            spec['execution_sources'].get('frozen_fingerprint')==spec['auditor'],'Unpinned original canary auditor')
    for reference in spec['execution_sources'].values():
        observed.absolute(base._absolute(reference),reference['sha256'])
    for family in ('trusted_specs','frozen_witnesses','process_exit_proofs'):
        require(set(spec[family])==PATHS,'Canary proof family must cover exact four paths')
    freeze=observed.json(spec['design_freeze']);config=observed.json(spec['config'])
    require(spec['design_freeze']['path']=='DESIGN_FREEZE.json' and
            freeze.get('schema')=='sr-satdose-design-freeze-v1' and
            freeze.get('experiment_id')==config.get('experiment_id')==base.EXPERIMENT and
            freeze.get('config_sha256')==spec['config']['sha256'] and
            freeze['artifacts']['config']==spec['config'],'Canary design/config identity differs')
    return root,observed,freeze,config


def _candidate(spec,observed,report):
    """Validate externally pinned original audit, never accept a PASS string."""
    require(report.get('schema')=='sr-satdose-four-path-canary-audit-v1' and
            report.get('status')=='FOUR_PATH_CANARY_EVIDENCE_AUDITED_NOT_ADMISSION' and
            report.get('permission')=='NO_TRAINING' and report.get('pending')==[] and
            report.get('auditor')==spec['auditor'] and
            report.get('design_freeze_sha256')==spec['design_freeze']['sha256'] and
            set(report.get('paths',{}))==PATHS,'Incomplete original four-path canary audit')
    evidence=report.get('observed_files')
    require(isinstance(evidence,dict) and evidence,'Original canary audit missing observed hashes')
    for name,digest in evidence.items():observed.absolute(Path(name),digest)
    def audited(reference):
        path=observed.ref(reference)
        require(evidence.get(str(path))==reference['sha256'],'Canary proof not covered by independent audit')
        return tx.read_json(path)
    audited(spec['design_freeze'])
    for reference in spec['execution_sources'].values():
        require(evidence.get(reference['path'])==reference['sha256'],'Canary source absent from independent audit')
    identities={};commits={}
    for key in sorted(PATHS):
        start,arm=key.split('/');start=int(start)
        technical=audited(spec['trusted_specs'][key]);identity=technical['identity']
        require(technical.get('schema')==canary_audit.replay.SCHEMA and
                technical.get('permission')=='READONLY_STEP2_CANARY_ONLY' and
                technical.get('execution_sources')==spec['execution_sources'] and
                technical.get('design_freeze_sha256')==spec['design_freeze']['sha256'] and
                identity.get('execution_sources_sha256')==object_sha(spec['execution_sources']) and
                identity.get('train_seed')==start and identity.get('arm')==arm and
                identity.get('experiment_id')==base.EXPERIMENT,'Canary technical spec identity differs')
        prefix=f'canary_replay/start-{start}/{arm}'
        require(spec['trusted_specs'][key]['path']==prefix+'/TECHNICAL_EXECUTION.json' and
                technical['slots']=={phase:prefix+'/'+phase for phase in ('baseline','restored')} and
                set(technical['commits'])=={'1','2'},'Canary fixed paths/first-two commits differ')
        for step in (1,2):
            # The independently audited technical spec itself SHA-binds both
            # markers. The original auditor's observed map directly lists only
            # step2 COMMIT; TransactionStore verified step1 transitively.
            reference=technical['commits'][str(step)];marker=observed.json(reference)
            require(reference['path']==f'training_transactions/start-{start}/{arm}/step-{step:03d}/COMMIT.json' and
                    marker.get('identity')==identity and marker.get('step')==step,'Canary committed identity differs')
        row=report['paths'][key]
        require(row.get('status')=='CORE_CANARY_TENSORS_AND_STATE_VERIFIED' and row.get('request_count')==128 and
                row.get('policy_adam_scheduler_rng')=='VERIFIED_FROM_ACTUAL_SAFE_CPU_STATE' and
                row.get('reference')=='ORIGINAL_B0_HASH_AND_OBSERVED_REFERENCE_LOGPROBS_BOUND_NO_NEW_GPU_OBSERVATION',
                'Canary actual state/teacher evidence incomplete')
        deltas=row.get('actual_nonzero_policy_deltas')
        require(isinstance(deltas,list) and len(deltas)==2 and all(
                value.get('compared_tensors')==496 and type(value.get('changed_tensors')) is int and
                1<=value['changed_tensors']<=496 and _positive(value.get('parameter_update_norm')) for value in deltas),
                'Canary actual policy delta zero/nonfinite/incomplete')
        completions={}
        for phase in ('baseline','restored'):
            reference=row[phase];complete=audited(reference)
            require(reference['path']==prefix+'/'+phase+'/COMPLETE.json' and complete.get('identity')==identity and
                    complete.get('spec')==spec['trusted_specs'][key],'Canary phase references differ')
            completions[phase]=complete
        require(completions['baseline']['process']!=completions['restored']['process'],
                'Canary baseline/reload used same process')
        pair=spec['frozen_witnesses'][key];require(set(pair)=={'before','after'},'Missing actual frozen witnesses')
        frozen=row.get('frozen_parameters',{})
        require(frozen.get('status')=='SOURCE_BOUND_GPU_OBSERVATIONS_IDENTICAL' and
                frozen.get('before')==pair['before'] and frozen.get('after')==pair['after'] and
                type(frozen.get('parameters')) is int and frozen['parameters']>0,'Missing actual frozen-parameter audit')
        for reference in pair.values():audited(reference)
        exit_ref=spec['process_exit_proofs'][key];audited(exit_ref)
        require(row.get('baseline_exit',{}).get('status')=='SOURCE_BOUND_CONTROLLER_EXIT_OBSERVATION_VERIFIED' and
                row['baseline_exit'].get('evidence')==exit_ref,'Missing actual controller exit audit')
        identities[key]=copy.deepcopy(identity);commits[key]=copy.deepcopy(technical['commits'])
    return dict(schema='sr-satdose-canary-resume-admission-v1',status='CANARY_RESUME_PASS',
        permission='REMAINING_REGISTERED_TRAINING_STEPS_3_TO_24_SUBJECT_TO_RISK_AND_DEV_GATES',
        experiment_id=base.EXPERIMENT,config_sha256=spec['config']['sha256'],
        design_freeze_sha256=spec['design_freeze']['sha256'],independent_audit=copy.deepcopy(spec['report']),
        auditor=copy.deepcopy(spec['auditor']),authority_source=copy.deepcopy(spec['helper']),
        execution_sources_sha256=object_sha(spec['execution_sources']),identities=identities,commits=commits,
        trusted_specs=copy.deepcopy(spec['trusted_specs']),frozen_witnesses=copy.deepcopy(spec['frozen_witnesses']),
        process_exit_proofs=copy.deepcopy(spec['process_exit_proofs']),
        new_generation_calls=0,new_optimizer_calls=0,main_authorized=False)


def prepare_canary(spec):
    """No writes: re-run original exact-step2 audit and return publication data."""
    root,observed,_,_=_inputs(spec,published=False)
    report=observed.json(spec['report'])
    actual=canary_audit.audit_four_paths(root,trusted_specs=spec['trusted_specs'],
        execution_sources=spec['execution_sources'],expected_auditor_sha256=spec['auditor']['sha256'],
        expected_freeze_sha256=spec['design_freeze']['sha256'],frozen_witnesses=spec['frozen_witnesses'],
        process_exit_proofs=spec['process_exit_proofs'])
    require(actual==report,'Independent fresh canary replay differs from trusted original report')
    candidate=_candidate(spec,observed,report);observed.stable();return candidate


class VerifiedCanary:
    def __init__(self,spec):
        self.spec=copy.deepcopy(spec)
        self.root,self._observed,_,_=_inputs(self.spec,published=True)
        self.report=self._observed.json(self.spec['report'])
        candidate=_candidate(self.spec,self._observed,self.report)
        require(self._observed.json(self.spec['published'])==candidate,'Published canary differs from original audited evidence')
        self.receipt=copy.deepcopy(self.spec['published'])
        self.execution_sources=copy.deepcopy(self.spec['execution_sources'])
        self.identities=copy.deepcopy(candidate['identities']);self.stable()
    def stable(self):
        require(not any((self.root/name).exists() for name in
                ('HOLD.json','ERROR.json','training_control/HOLD.json','training_control/ERROR.json',
                 'first_two_launcher/HOLD.json','first_two_launcher/ERROR.json')),'Global training HOLD/error')
        _control_state(self.root,self.report['observed_files'])
        self._observed.stable()


def verify_canary(spec):
    """Read-only verification; never runs a model or grants endpoint permission."""
    return VerifiedCanary(spec)


class ContinuationVerifier:
    """One original identity, refreshed lock-external risk snapshot per step."""
    def __init__(self,original,spec):
        require(set(spec)=={'root','helper','execution_sources','canary_spec','risk_inputs'},
                'Incomplete continuation authority specification')
        self.spec=copy.deepcopy(spec);self.original=original
        self.root=Path(spec['root']);self.sources=copy.deepcopy(spec['execution_sources'])
        require(self.root==original._observed.root and spec['helper']==self.sources.get('continuation_verifier') and
                base._absolute(spec['helper'])==Path(__file__).resolve() and
                object_sha(self.sources)==original._spec['training_execution_sources_sha256'] and
                self.sources.get('admission_verifier')==original.verifier_reference,'Continuation changes original identity/sources')
        self.canary=verify_canary(spec['canary_spec'])
        require(self.canary.root==self.root and self.canary.execution_sources==self.sources and
                spec['canary_spec']['design_freeze']['sha256']==original._binding['design_freeze']['sha256'] and
                spec['canary_spec']['config']==original._spec['documents']['config'],'Canary differs from original admission')
        self._risk_fixed=None;self.refresh_risk(spec['risk_inputs'])

    def refresh_risk(self,inputs):
        """Must run outside TransactionStore's exclusive lock. No model calls."""
        self._risk_ready=False
        from analysis import risk
        require(set(inputs)=={'config_ref','freeze_ref','clarification_ref','trusted_commits','dev_endpoints',
                             'expected_source_sha256'},'Incomplete risk inputs')
        reference=self.sources.get('risk_verifier')
        require(reference is not None and base._absolute(reference)==Path(risk.__file__).resolve() and
                inputs['expected_source_sha256']==reference['sha256'] and
                inputs['config_ref']==self.original._spec['documents']['config'] and
                inputs['freeze_ref']==self.spec['canary_spec']['design_freeze'],'Risk source/config/freeze differs')
        fixed={k:copy.deepcopy(v) for k,v in inputs.items() if k not in ('trusted_commits','dev_endpoints')}
        require(self._risk_fixed is None or fixed==self._risk_fixed,'Risk authorization/threshold inputs changed')
        self.original._observed.stable();self.canary.stable()
        result=risk.assess(self.root,**copy.deepcopy(inputs))
        require(result.get('permission')=='RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY' and
                result.get('status') in ('NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED','PENDING_REQUIRED_DEV','GLOBAL_SCIENTIFIC_HOLD'),
                'Unauthenticated risk result')
        observed=base._Observed(self.root)
        require(result.get('observed_files'),'Risk assessment missing observed files')
        for path,digest in result['observed_files'].items():observed.absolute(Path(path),digest)
        observed.absolute(Path(reference['path']),reference['sha256'])
        self._risk_fixed=fixed;self._risk_inputs=copy.deepcopy(inputs);self._risk_result=copy.deepcopy(result)
        self._risk_observed=observed;self._cursor_stable();observed.stable();self._risk_ready=True
        return copy.deepcopy(result)

    def _cursor_stable(self):
        """No lock acquisition: caller run_step may already own exclusive lock."""
        expected=self._risk_inputs['trusted_commits']
        require(set(expected)==PATHS,'Risk snapshot omits a registered path')
        for key in sorted(PATHS):
            start,arm=key.split('/');directory=self.root/f'training_transactions/start-{start}/{arm}'
            require(directory.resolve(strict=True)==directory and not directory.is_symlink(),'Noncanonical committed path')
            references=expected[key]
            require(isinstance(references,list) and 2<=len(references)<=24,'Risk cursor outside original committed budget')
            actual=sorted(p for p in directory.iterdir() if p.name.startswith('step-'))
            require([p.name for p in actual]==[f'step-{step:03d}' for step in range(1,len(references)+1)],
                    'Published cursor changed since independent risk assessment')
            for step,reference in enumerate(references,1):
                require(tx.bound(self.root,reference)==directory/f'step-{step:03d}/COMMIT.json',
                        'Committed marker changed since independent risk assessment')

    def verify(self,*,root,identity,design_freeze_sha256,target_step,sampled_endpoint):
        require(self._risk_ready,'Missing/failed independent risk assessment; no cached fallback')
        require(type(target_step) is int and 3<=target_step<=24,'Continuation permits only exact registered steps3–24')
        require(Path(root)==self.root and design_freeze_sha256==self.original._binding['design_freeze']['sha256'],
                'Foreign continuation root/design')
        self.original._observed.stable();self.canary.stable();self._risk_observed.stable();self._cursor_stable()
        start=identity.get('train_seed');arm=identity.get('arm');key=f'{start}/{arm}'
        require(key in PATHS and type(start) is int and identity==self.canary.identities[key],
                'Continuation changes original first-two scientific identity')
        references=self._risk_inputs['trusted_commits'][key]
        require(len(references)==target_step-1,'Continuation target is not exact next committed step')
        marker=tx.read_json(tx.bound(self.root,references[-1]))
        endpoint=dict(experiment_id=base.EXPERIMENT,start_seed=start,arm=arm,step=target_step-1,
                      policy_sha256=marker['policy_sha256'],reference_sha256=marker['reference_sha256'])
        require(marker.get('identity')==identity and sampled_endpoint==endpoint,'Sampler does not use actual current committed policy')
        decision=self._risk_result
        require(decision.get('global_hold') is False and decision['status']!='GLOBAL_SCIENTIFIC_HOLD',
                'Global scientific HOLD forbids all continuation')
        pending=decision.get('pending_required_dev')
        require(isinstance(pending,list),'Missing current DEV barrier decision')
        if pending:
            require(decision['status']=='PENDING_REQUIRED_DEV','Inconsistent risk pending decision')
            barriers=[]
            for value in pending:
                parts=value.split('/')
                require(len(parts)==3 and '/'.join(parts[:2]) in PATHS and parts[2] in ('2','8','16'),
                        'Unregistered DEV barrier')
                barriers.append(int(parts[2]))
            require(target_step<=min(barriers),'Required DEV barrier must complete before advancing beyond it')
        else:require(decision['status']=='NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED','Risk assessment did not clear fixed checks')
        refs=self.original._spec['documents']
        return dict(identity_sha256=object_sha(identity),design_freeze_sha256=design_freeze_sha256,
            target_step=target_step,sampled_endpoint=copy.deepcopy(sampled_endpoint),
            admission=copy.deepcopy(self.original._admission),registration=copy.deepcopy(refs['registration']),
            gpu_report=copy.deepcopy(refs['gpu_report']),canary=copy.deepcopy(self.canary.receipt))
