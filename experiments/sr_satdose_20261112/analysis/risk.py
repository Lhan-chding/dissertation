"""Read-only, fixed-denominator scientific risk assessment; never admission.

``assess`` takes externally trusted COMMIT and DEV receipt/binding references,
validates complete actual transaction chains, independently re-scores TRAIN and
DEV, and returns decisions only. The caller must pin the complete deployed
source inventory as well as this module. No model, optimizer, generation,
checkpoint selection, retry, artifact publication or MAIN permission exists.
Unknown native/field-local selections retain their predefined zero-hit and
known-mask semantics; missing or technically failed answers are rejected.
"""
from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
import re

from . import calibration, evaluation
from execution import canary_audit, training_transaction as tx
from runtime import fork_loader

core=calibration.core
require=tx.require
PATHS=tuple(f'{s}/{a}' for s in (11031,11032) for a in ('T0','T1'))
DEV_STEPS=(2,8,16)
PERMISSION='RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY'


def _integer(value,low,high):
    require(type(value) is int and low<=value<=high,'Invalid integer count/cursor')
    return value


def _dev_counts(value):
    require(set(value)=={'conflict','aligned_no_list'},'Unexpected DEV risk metrics')
    result={}
    for key,n in [('conflict',32),('aligned_no_list',64)]:
        metric=value[key]
        require(metric['assigned']==n and type(metric['assigned']) is int,'DEV fixed denominator differs')
        hits=_integer(metric['counts']['S_hit'],0,n)
        known=_integer(metric['counts']['S_known'],0,n)
        require(hits<=known,'DEV UNKNOWN counted as a hit')
        result[key]=dict(assigned=n,hits=hits,known=known,unknown=n-known)
    return result


def _decision(progress,steps,dev):
    """Pure arithmetic on already authenticated per-step/group observations."""
    require(set(progress)==set(steps)==set(PATHS),'Exact four training paths required')
    expected_dev={f'{s}/START/0' for s in (11031,11032)}
    allowed=set(expected_dev)
    for key in PATHS:
        n=_integer(progress[key],0,24)
        require(len(steps[key])==n and [x['step'] for x in steps[key]]==list(range(1,n+1)),
                'Missing/duplicate/noncontiguous training metrics')
        expected=12 if key.endswith('/T1') else 0
        for item in steps[key]:
            require(type(item['conflict_g8_count']) is int and item['conflict_g8_count']==expected and
                    type(item['conflict_response_count']) is int and item['conflict_response_count']==8*expected and
                    len(item['conflict_group_doses'])==expected,'Wrong arm conflict denominator')
            require(all(type(d) in (int,float) and abs(d)<float('inf') for d in item['conflict_group_doses']),
                    'Nonfinite conflict dose')
            require(_integer(item['exact_authorised'],0,8*expected)<=_integer(item['known'],0,8*expected),
                    'Unknown TRAIN answer counted as authorised')
        allowed.update(f'{key}/{step}' for step in DEV_STEPS if step<=n)
    require(expected_dev<=set(dev)<=allowed,'Missing START, future or extra DEV endpoint')
    counts={key:_dev_counts(value) for key,value in dev.items()}
    pending=sorted(allowed-set(dev));signals=[];regressions=[];paths={}
    for key in PATHS:
        start,arm=key.split('/');n=progress[key];windows=[]
        for end in (range(4,n+1,4) if arm=='T1' else ()):
            block=steps[key][end-4:end]
            require(all(x['conflict_g8_count']==12 and x['conflict_response_count']==96 for x in block),
                    'Each step must contain original twelve conflict G8')
            hits=sum(_integer(x['exact_authorised'],0,96) for x in block)
            known=sum(_integer(x['known'],0,96) for x in block)
            require(hits<=known and all(x['exact_authorised']<=x['known'] for x in block),
                    'Unknown TRAIN answer counted as authorised')
            doses=[d for x in block for d in x['conflict_group_doses']]
            require(len(doses)==48 and all(type(d) in (int,float) and abs(d)<float('inf') for d in doses),
                    'Missing/nonfinite conflict group dose')
            # Rational comparison avoids changing strict/inclusive boundaries
            # through division rounding. Inputs are actual frozen core values.
            mean=sum(Fraction(d) for d in doses)/48
            qualifies=mean<Fraction(3,4) and Fraction(hits,384)<Fraction(4,5)
            windows.append(dict(first_step=end-3,last_step=end,conflict_g8_count=48,
                conflict_response_count=384,mean_d=float(mean),exact_authorised=hits,
                exact_authorised_rate=hits/384,known=known,unknown=384-known,
                low_d_and_low_authorised=qualifies))
        if arm=='T1':
            for previous,current in zip(windows,windows[1:]):
                end=current['last_step'];latest=max(step for step in DEV_STEPS if step<=end)
                name=f'{key}/{latest}'
                if name in counts and previous['low_d_and_low_authorised'] and current['low_d_and_low_authorised'] and \
                        Fraction(counts[name]['conflict']['hits'],32)<Fraction(4,5):
                    signals.append(dict(path=key,window_ends=[previous['last_step'],end],
                        dev_endpoint=name,dev_conflict=counts[name]['conflict']))
        baseline=counts[f'{start}/START/0']['aligned_no_list'];checks=[]
        for step in DEV_STEPS:
            name=f'{key}/{step}'
            if name not in counts:continue
            value=counts[name]['aligned_no_list']
            drop=Fraction(baseline['hits']-value['hits'],64)
            checks.append(dict(step=step,dev_endpoint=name,drop=float(drop),
                severe=drop>=Fraction(1,5),counts=value))
        for first,second in zip(checks,checks[1:]):
            if (first['step'],second['step']) in ((2,8),(8,16)) and first['severe'] and second['severe']:
                regressions.append(dict(path=key,dev_endpoints=[first['dev_endpoint'],second['dev_endpoint']],
                    baseline_endpoint=f'{start}/START/0',baseline=baseline))
        paths[key]=dict(completed_steps=n,windows=windows,scheduled_dev_regression_checks=checks)
    reasons=[]
    if signals:reasons.append('HOLD_SIGNAL_NOT_DELIVERED')
    if regressions:reasons.append('HOLD_SEVERE_REGRESSION')
    status='GLOBAL_SCIENTIFIC_HOLD' if reasons else ('PENDING_REQUIRED_DEV' if pending else 'NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED')
    return dict(schema='sr-satdose-scientific-risk-assessment-v1',status=status,permission=PERMISSION,
        global_hold=bool(reasons),hold_reasons=reasons,pending_required_dev=pending,
        next_training_step_requires_complete_dev=bool(pending),signals=signals,regressions=regressions,paths=paths,
        training_permission=False,main_permission=False,
        limitations=['No technical, canary, storage or training admission is conferred.',
                    'DEV is a budget safety check, not formal noninferiority or checkpoint selection.'])


class _Evidence:
    def __init__(self,root):self.root=root;self.files={}
    def path(self,path,expected=None):
        path=Path(path)
        require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),
                'Noncanonical risk evidence')
        digest=tx.file_sha(path)
        require(expected is None or digest==expected,'Risk evidence hash differs')
        require(path not in self.files or self.files[path]==digest,'Risk evidence changed during assessment')
        self.files[path]=digest;return path
    def ref(self,ref):
        require(isinstance(ref,dict) and set(ref)=={'path','sha256'},'Malformed trusted risk reference')
        return self.path(tx.bound(self.root,ref),ref['sha256'])
    def json(self,ref):return tx.read_json(self.ref(ref))
    def stable(self):
        for path,digest in list(self.files.items()):self.path(path,digest)

    def image(self,original):
        public=original['model_input'];relative=Path(public['image_file'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Image escapes frozen root')
        self.path(self.root/relative,public['image_sha256'])


def _clarification(value):
    required=dict(schema='sr-satdose-training-risk-clarification-v1',experiment_id=tx.EXPERIMENT,
        authorization='explicit_user_confirmation_before_any_TRAIN_generation',start_seeds=[11031,11032],arm='T1',
        evaluate_each_path_independently=True,windows='nonoverlapping steps 1-4, 5-8, 9-12, 13-16, 17-20, 21-24',
        per_window_conflict_g8_count=48,per_window_conflict_response_count=384,
        window_conditions_all_required=dict(mean_d_less_than=.75,exact_authorised_rate_less_than=.8,
            exact_authorised_denominator='all 384 conflict responses only'),consecutive_qualifying_windows=2,
        additional_condition=dict(latest_same_path_DEV_conflict_response_count=32,S_less_than=.8,
            at_steps_8_and_16='complete current DEV before deciding'),
        trigger='any one T1 path satisfying all conditions causes global HOLD for all paths',scientific_thresholds_changed=False)
    require(all(value.get(k)==v for k,v in required.items()),'Frozen risk clarification differs')


def _score_step(raw,groups,cases,scorer,start,arm,step):
    require(len(raw)==128 and len(groups)==16 and len({r['request_id'] for r in raw})==128,
            'Incomplete original training step')
    require([r['request_id'] for r in raw]==[rid for g in groups for rid in g['request_ids']],
            'Training metrics differ from original frozen G8 order')
    metrics=[]
    for offset,group in zip(range(0,128,8),groups,strict=True):
        require(len(group['request_ids'])==8 and group['start_seed']==start and group['arm']==arm and group['step']==step,
                'Foreign frozen training group')
        case=cases[group['case_id']];task=case['task'];geometry=core.geometry(task,case['audit'])
        require(type(group['planned_conflict']) is bool and geometry['alt'] is not None and
                (geometry['chart']!=geometry['alt'])==group['planned_conflict'], 'False conflict assignment')
        js=[];hits=[];known=[]
        for row in raw[offset:offset+8]:
            require(row['case_id']==group['case_id'] and row['group_id']==group['group_id'] and
                    row['start_seed']==start and row['arm']==arm and row['step']==step and row['scope']=='TRAIN' and
                    row['generation_status']=='COMPLETE' and row['technical_validation_errors']==[] and
                    row['finish_reason'] in ('balanced','eos','length'),'Incomplete/foreign raw training answer')
            score=scorer(row['full_text'],task['world'],task['query'],finished=row['finish_reason']=='balanced',
                         finish_reason=row['finish_reason'])
            k=int(score['known_mask']['S']);selected=score['witness'].get('selected') if k else None
            hit=core.exact_hit(selected,geometry['authorised']) if k else 0
            require(score==row['native_score'] and score['J']==row['J'] and hit==row['exact_authorised'] and
                    hit==(score['S'] if k else 0),'Independent native/geometry score differs')
            js.append(score['J']);hits.append(hit);known.append(k)
        metrics.append(dict(kind='CONFLICT' if group['planned_conflict'] else 'ALIGNED',**core.group_metrics(js,hits,known)))
    conflict=[x for x in metrics if x['kind']=='CONFLICT']
    expected=12 if arm=='T1' else 0
    require(len(conflict)==expected and len(metrics)-len(conflict)==16-expected,'Frozen conflict/aligned G8 allocation differs')
    return dict(step=step,conflict_g8_count=expected,conflict_response_count=8*expected,
        exact_authorised=sum(x['authorised'] for x in conflict),known=sum(x['known'] for x in conflict),
        conflict_group_doses=[x['dose'] for x in conflict])


def assess(root,*,config_ref,freeze_ref,clarification_ref,trusted_commits,dev_endpoints,expected_source_sha256):
    """Authenticate fixed actual evidence, then return a risk decision only.

    trusted_commits is exact four ``start/arm -> [COMMIT refs]`` contiguous
    current prefixes (empty before training). dev_endpoints maps ``start/arm/step``
    to exactly ``{receipt: ref, binding: ref}``; both START endpoints are required.
    Missing scheduled DEV yields PENDING_REQUIRED_DEV, never permission to skip.
    Technical failures raise PermissionError. Existing scientific HOLD triggers
    are retained when assessing later evidence; no later metric erases them.
    """
    root=Path(root);require(root.resolve(strict=True)==root and root.is_dir(),'Noncanonical risk root')
    e=_Evidence(root);source=e.path(Path(__file__).resolve(),expected_source_sha256)
    require(re.fullmatch('[0-9a-f]{64}',expected_source_sha256 or ''),'Missing independent verifier pin')
    cfg=e.json(config_ref);freeze=e.json(freeze_ref);_clarification(e.json(clarification_ref))
    require(e.ref(freeze_ref)==root/'DESIGN_FREEZE.json' and freeze['experiment_id']==cfg['experiment_id']==tx.EXPERIMENT and
            freeze['config_sha256']==config_ref['sha256'] and freeze['artifacts']['config']==config_ref,'Wrong risk design/config')
    require(set(trusted_commits)==set(PATHS) and isinstance(dev_endpoints,dict),'Exact four trusted paths required')
    lock=e.json(freeze['artifacts']['source_lock']);native_root=Path(lock['native_src'])
    for reference in freeze['artifacts'].values():e.ref(reference)
    for name,digest in lock['native_files'].items():e.path(native_root/name,digest)
    runtime_root=Path(fork_loader.__file__).resolve().parent
    require({p.name:tx.file_sha(p) for p in runtime_root.glob('*.py')}==freeze['runtime_sources'],'Frozen runtime differs')
    for p in runtime_root.glob('*.py'):e.path(p,freeze['runtime_sources'][p.name])
    for module in (calibration,evaluation,core,tx,canary_audit,canary_audit.state):e.path(Path(module.__file__).resolve())
    e.path(calibration.PACKAGE/'config/source_bindings.json')
    native=fork_loader.load_native_ops(native_root,lock['native_files']);scorer,_=calibration.load_scorer(native_root)
    groups=tx.rows(e.ref(freeze['artifacts']['training_groups']))
    cases=calibration._index(tx.rows(e.ref(freeze['artifacts']['private_cases'])),'case_id')
    tx_root=root/'training_transactions';require(tx_root.is_dir() and tx_root.resolve(strict=True)==tx_root,
        'Existing transaction directory required; risk analysis never creates it')
    steps={};progress={};chain_snapshots={};stores={};analyses={}
    with canary_audit._existing_lock(root):
        canary_audit.replay._no_unknown(root)
        for key in PATHS:
            start_text,arm=key.split('/');start=int(start_text);refs=trusted_commits[key]
            require(isinstance(refs,list) and len(refs)<=24,'Invalid committed prefix')
            directory=tx_root/f'start-{start}'/arm;steps[key]=[];progress[key]=len(refs)
            if not refs:
                require(not directory.exists() or not list(directory.iterdir()),'Omitted existing committed path')
                continue
            first=e.json(refs[0]);identity=first['identity']
            require(identity['train_seed']==start and identity['arm']==arm,'Trusted path mapped to foreign identity')
            require(re.fullmatch('[0-9a-f]{64}',identity.get('execution_sources_sha256','')),
                    'Nonproduction training identity has no authenticated source inventory')
            e.ref(first['transaction']['genesis_reference'])
            e.path(Path(identity['source_binding']['original_state_path']),identity['source_binding']['state_sha256'])
            store=tx.TransactionStore(root,e.ref(freeze_ref),freeze_ref['sha256'],identity,native=native,
                genesis_reference=first['transaction']['genesis_reference'])
            chain=store._chain();stores[key]=store;chain_snapshots[key]=chain
            require(len(chain)==len(refs),'Trusted prefix omits or invents actual commits')
            for step,((digest,commit),ref) in enumerate(zip(chain,refs,strict=True),1):
                parent=directory/f'step-{step:03d}'
                require(e.ref(ref)==parent/'COMMIT.json' and ref['sha256']==digest,'Trusted COMMIT path/hash differs')
                for p in parent.iterdir():e.path(p)
                evidence=tx.read_json(tx.bound(parent,commit['transaction']['evidence']))
                rollout=e.json(evidence['rollout_receipt']);raw=tx.rows(e.ref(rollout['raw']))
                e.ref(rollout['scores']);e.ref(rollout['authorization'])
                for capture_ref in rollout['complete_files']:
                    p=e.ref(capture_ref);original=tx.read_json(p)
                    e.path(Path(original['native_capture_path']),original['native_capture_sha256'])
                    e.path(p.with_name(p.name.replace('.COMPLETE.json','.BEGIN.json')))
                    e.image(original)
                precommit=e.json(evidence['precommit_reference'])
                for external in precommit['artifacts']:e.ref(external)
                selected=[g for g in groups if g['start_seed']==start and g['arm']==arm and g['step']==step]
                steps[key].append(_score_step(raw,selected,cases,scorer,start,arm,step))
        dev={}
        for key,refs in dev_endpoints.items():
            require(isinstance(refs,dict) and set(refs)=={'receipt','binding'},'Malformed DEV endpoint references')
            binding=e.json(refs['binding']);ep=binding['endpoint']
            require(key==f"{ep['start_seed']}/{ep['arm']}/{ep['step']}" and binding['scope']=='DEV','Foreign/non-DEV endpoint')
            result=evaluation.analyze_completed_endpoint(e.ref(config_ref),e.ref(freeze_ref),root,e.ref(refs['receipt']),
                expected_receipt_sha256=refs['receipt']['sha256'],endpoint_binding_path=e.ref(refs['binding']),
                expected_endpoint_binding_sha256=refs['binding']['sha256'])
            if ep['arm']!='START':
                path=f"{ep['start_seed']}/{ep['arm']}";step=ep['step']
                require(path in trusted_commits and type(step) is int and 1<=step<=len(trusted_commits[path]) and
                        binding['checkpoint']['commit']==trusted_commits[path][step-1],'DEV not bound to trusted actual checkpoint')
            risk=result['summary']['dev_risk_inputs'];dev[key]={k:risk[k] for k in ('conflict','aligned_no_list')}
            analyses[key]=result['summary']['evidence']
            for field in ('scorer_sources','analysis_sources'):
                for path,digest in result['summary'][field].items():e.path(Path(path),digest)
            receipt=e.json(refs['receipt']);e.ref(receipt['raw']);e.ref(receipt['accounting'])
            for capture in receipt['complete_files']:
                p=e.ref(capture);original=tx.read_json(p)
                e.path(Path(original['native_capture_path']),original['native_capture_sha256'])
                e.path(p.with_name(p.name.replace('.COMPLETE.json','.BEGIN.json')))
                e.image(original)
        decision=_decision(progress,steps,dev)
        for key,store in stores.items():require(store._chain()==chain_snapshots[key],'Committed chain changed during risk analysis')
        e.stable()
    decision.update(verifier=dict(path=str(source),sha256=expected_source_sha256),
        design_freeze=copy.deepcopy(freeze_ref),config=copy.deepcopy(config_ref),clarification=copy.deepcopy(clarification_ref),
        trusted_commits=copy.deepcopy(trusted_commits),dev_endpoints=copy.deepcopy(dev_endpoints),dev_analysis_evidence=analyses,
        observed_files={str(path):digest for path,digest in e.files.items()},
        source_inventory_must_be_authenticated_by_caller=True)
    return decision
