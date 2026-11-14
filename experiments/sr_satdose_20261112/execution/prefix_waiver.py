"""CPU-only verification of the explicit 2026-11-14 PREFIX measurement waiver.

No writes, CLI, model imports, generation, retries or training authority. The
caller authenticates the containing technical binding SHA and this module's
source before import. A waiver is not a measurement PASS: the failed original
11031 report is preserved byte-for-byte. Both original 48-response populations
must be complete before any raw scoring in verify_completed. Other GPU/training
checks remain the caller's responsibility, without any exemption here.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import math
from pathlib import Path
import re

from analysis import calibration

core=calibration.core
EXPERIMENT='SR-SATDOSE-1-20261112'
WAIVER_PATH='protocol_clarifications/PREFIX_GATE_WAIVER_20261114.json'
RESULT_PATH='prefix_waiver_control/RESULT.json'
ORIGINAL_RESULT='prefix_control/RESULT.json'
ORIGINAL_RELEASE='execution_releases/PREFIX_RELEASE_734e921.json'
ANSWER='豁免本次 PREFIX 门禁，保留未达标结果，其余 GPU／训练门禁照常执行'
QUESTION='你希望按哪种方式继续？当前 PREFIX 实测仍为 13/16，原门槛为 14/16。'
DISCLOSURE='must disclose post-outcome PREFIX calibration waiver; no claim of original preregistered calibration pass'
REMAINING=dict(start_seed=11032,responses=48,source='original frozen PREFIX_CALIB manifest',
               repeat_11031=False,resampling=False,new_samples=0)
RUNTIME={'backend.py','data_adapter.py','diagnostics.py','dry_driver.py','fork_loader.py','native_runtime.py','registration.py'}


def require(ok,message):
    if not ok:raise ValueError('PREFIX_WAIVER_REFUSED: '+message)


def _same(a,b):return core.object_sha(a)==core.object_sha(b)


def _bound(root,ref,expected_path=None):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'} and isinstance(ref['path'],str) and
            isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']), 'Invalid artifact reference')
    relative=Path(ref['path'])
    require(not relative.is_absolute() and '..' not in relative.parts and
            (expected_path is None or ref['path']==expected_path),'Relocated/escaping artifact')
    path=root/relative
    require(path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'Noncanonical artifact')
    require(core.file_sha(path)==ref['sha256'],'Artifact SHA differs: '+ref['path'])
    return path


def _root(value):
    root=Path(value)
    require(root.is_absolute() and root.resolve(strict=True)==root and root.is_dir(),'Canonical evidence root required')
    return root


def _no_errors(directory):
    require(directory.is_dir() and not directory.is_symlink() and directory.resolve()==directory,'Invalid evidence directory')
    for path in directory.rglob('*'):
        require(not path.is_symlink(),'Symlink in evidence directory')
        require(not(path.name in ('HOLD.json','ERROR.json') or path.name.endswith(('.HOLD.json','.ERROR.json'))),
                'Technical HOLD/ERROR is never waived')


def _inventory(directory,*,old=False):
    require(directory.is_absolute() and directory.is_dir() and directory.resolve()==directory,'Invalid release root')
    result={}
    for path in directory.rglob('*'):
        require(not path.is_symlink(),'Release symlink forbidden')
        if old and '__pycache__' in path.relative_to(directory).parts:continue
        if path.is_file():result[str(path.relative_to(directory))]=core.file_sha(path)
    return result


def _release(root,ref,freeze_sha,config_sha,*,waiver_ref=None,original_ref=None):
    path=_bound(root,ref);record=core.read_json(path);old=waiver_ref is None
    require(record.get('schema')==('sr-satdose-prefix-technical-release-v1' if old else 'sr-satdose-prefix-continuation-release-v1') and
            record.get('status')==('AUDITED_FOR_PREFIX_CALIB_ONLY' if old else 'AUDITED_FOR_PREFIX_11032_CONTINUATION_ONLY') and
            record.get('permission')=='NO_TRAINING','Wrong audited release scope')
    expected=dict(experiment_id=EXPERIMENT,evidence_root=str(root),design_freeze_sha256=freeze_sha,
                  config_sha256=config_sha,frozen_code_root=str(root/'code'))
    require(all(_same(record.get(k),v) for k,v in expected.items()),'Release scientific identity differs')
    directory=Path(record['release_root'])
    require(directory.is_relative_to(root) and directory!=root/'code' and not path.is_relative_to(directory),
            'Release must be separate from frozen code and its receipt')
    inventory=_inventory(directory,old=old)
    require(inventory==record.get('sources'),'Full release source inventory differs')
    required={'analysis/calibration.py','supplied_package/tools/satdose_core.py','supplied_package/config/experiment.json'}
    required.add('execution/prefix_control.py' if old else 'execution/prefix_continuation.py')
    if not old:required|={'execution/prefix_waiver.py','scheduling/run_prefix_continuation.sh'}
    require(required<=set(inventory) and inventory['supplied_package/config/experiment.json']==config_sha,
            'Missing/substituted release dependencies')
    require(inventory['analysis/calibration.py']==core.file_sha(calibration.__file__) and
            inventory['supplied_package/tools/satdose_core.py']==core.file_sha(core.__file__),
            'Actual original analysis/helper source differs')
    if not old:
        require(_same(record.get('waiver'),waiver_ref) and _same(record.get('original_release'),original_ref) and
                inventory['execution/prefix_waiver.py']==core.file_sha(__file__), 'Unbound actual waiver verifier/release lineage')
    return record


def verify_waiver(root,ref,*,expected_freeze_sha256,expected_config_sha256):
    """Verify explicit authorization and preserved original failure; never open raw."""
    root=_root(root);path=_bound(root,ref,WAIVER_PATH);document=core.read_json(path)
    fixed=dict(schema='sr-satdose-prefix-waiver-v1',status='USER_AUTHORIZED_MEASUREMENT_GATE_WAIVER',
        experiment_id=EXPERIMENT,evidence_root=str(root),scope='PREFIX_CALIB_MEASUREMENT_GATE_ONLY',
        design_freeze_sha256=expected_freeze_sha256,config_sha256=expected_config_sha256,
        original_job_id=1737594,preserve_original_results=True,rewrite_measurement_pass=False,
        remaining_calibration=REMAINING,require_both_complete_48_before_gpu=True,other_gates_unchanged=True,
        technical_failure_or_unknown_waived=False,main_confirmatory_claims=DISCLOSURE,permission='NO_TRAINING',
        authorization=dict(origin='explicit_user_reply',question=QUESTION,answer=ANSWER))
    require(set(document)==set(fixed)|{'created_at_utc','original_result','original_release'},'Waiver schema fields differ')
    require(all(_same(document.get(k),v) for k,v in fixed.items()),'Waiver scope, flags or explicit user authorization differs')
    timestamp=datetime.fromisoformat(document['created_at_utc'].replace('Z','+00:00'))
    require(timestamp.tzinfo is not None and timestamp.utcoffset()==timedelta(0),'Waiver timestamp must be UTC')
    freeze_path=_bound(root,dict(path='DESIGN_FREEZE.json',sha256=expected_freeze_sha256))
    frozen=core.read_json(freeze_path)
    require(frozen.get('schema')=='sr-satdose-design-freeze-v1' and frozen.get('experiment_id')==EXPERIMENT and
            frozen.get('config_sha256')==expected_config_sha256,'Original design differs')
    config_path=_bound(root,frozen['artifacts']['config'])
    require(core.file_sha(config_path)==expected_config_sha256,'Original config differs')
    _bound(root,document['original_release'],ORIGINAL_RELEASE)
    release=_release(root,document['original_release'],expected_freeze_sha256,expected_config_sha256)
    require(set(frozen['runtime_sources'])==RUNTIME and
            all(release['sources'].get('runtime/'+name)==sha for name,sha in frozen['runtime_sources'].items()),
            'Original release frozen runtime inventory differs')
    original_path=_bound(root,document['original_result'],ORIGINAL_RESULT)
    _no_errors(original_path.parent);original=core.read_json(original_path)
    require(original.get('status')=='HOLD_PREFIX_MEASUREMENT' and original.get('permission')=='NO_TRAINING' and
            type(original.get('stopped_start')) is int and original['stopped_start']==11031 and
            set(original.get('prefix_analyses',{}))=={'11031'},'Original outcome is not the authorized 11031 measurement HOLD')
    report=original['prefix_analyses']['11031']
    require(report.get('schema')=='sr-satdose-prefix-calibration-analysis-v1' and report.get('start_seed')==11031 and
            report.get('status')=='HOLD_PREFIX_MEASUREMENT' and report.get('candidate_gate_passed') is False and
            report.get('permission')=='NO_TRAINING' and report.get('qualifying_readouts')==[] and
            report.get('design_freeze_sha256')==expected_freeze_sha256 and report.get('config_sha256')==expected_config_sha256,
            'Original failed measurement was rewritten')
    stored=core.read_json(root/'prefix_control/PREFIX_11031_ANALYSIS.json')
    require(_same(stored,report),'Original 11031 analysis differs from preserved RESULT')
    preflight=core.read_json(root/'prefix_control/PREFLIGHT.json')
    require(preflight.get('schema')=='sr-satdose-prefix-preflight-v1' and preflight.get('status')=='READY_FOR_PREFIX_CALIB_ONLY' and
            preflight.get('permission')=='NO_TRAINING' and preflight.get('model_calls')==0 and
            preflight.get('starts')==[11031,11032] and preflight.get('responses_per_start')==48 and
            preflight.get('design_freeze_sha256')==expected_freeze_sha256 and preflight.get('config_sha256')==expected_config_sha256,
            'Original PREFLIGHT identity differs')
    released=preflight.get('release',{})
    require(released.get('path')==str(root/document['original_release']['path']) and
            released.get('sha256')==document['original_release']['sha256'] and
            released.get('release_root')==release['release_root'] and released.get('sources')==release['sources'],
            'Original PREFLIGHT audited release differs')
    require(_same(preflight.get('protocol_clarification'),report.get('protocol_clarification')) and
            report.get('protocol_clarification') is not None,'Original protocol identity absent/changed')
    return copy.deepcopy(document)


def _completion_headers(root,reports,freeze_sha,source_lock_sha):
    headers={}
    for start in (11031,11032):
        ref=reports[str(start)]['completion_receipt']
        path=_bound(root,ref,f'runs/PREFIX_CALIB_{start}/CALIBRATION_COMPLETE.json');_no_errors(path.parent)
        header=core.read_json(path)
        require(header.get('status')=='CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION' and
                header.get('scope')=='PREFIX_CALIB' and type(header.get('start_seed')) is int and header['start_seed']==start and
                type(header.get('responses')) is int and header['responses']==48 and
                header.get('design_freeze_sha256')==freeze_sha and header.get('source_lock_sha256')==source_lock_sha,
                'Both original 48-response populations must be complete before raw access')
        require(header.get('optimizer_created') is False and header.get('scheduler_created') is False and
                all(type(header.get(k)) is int and header[k]==0 for k in ('optimizer_steps','parameter_updates')),
                'PREFIX capture mutated parameters')
        headers[start]=(path,header)
    return headers


def _allocation_inventory(root,freeze_sha):
    """Original driver allocation/intent namespace cannot hide extra PREFIX runs."""
    runs=root/'runs';allocations=root/'calibration_allocations'
    for directory in (runs,allocations):
        require(directory.is_dir() and directory.resolve()==directory and not directory.is_symlink(),
                'Missing/noncanonical original allocation namespace')
    seen=set()
    for path in runs.iterdir():
        require(not path.is_symlink(),'Symlinked run directory')
        prefix=path.name.startswith('PREFIX')
        if path.is_dir() and (path/'INTENT.json').is_file():
            prefix=prefix or core.read_json(path/'INTENT.json').get('scope')=='PREFIX_CALIB'
        if prefix:seen.add(path.name)
    require(seen=={'PREFIX_CALIB_11031','PREFIX_CALIB_11032'},'Extra/missing original PREFIX run allocation')
    seen=set()
    for path in allocations.iterdir():
        require(path.is_file() and not path.is_symlink(),'Noncanonical calibration allocation')
        value=core.read_json(path)
        if path.name.startswith('PREFIX') or value.get('scope')=='PREFIX_CALIB':
            start=value.get('start_seed')
            require(type(start) is int and start in (11031,11032) and path.name==f'PREFIX_CALIB-{start}.json' and
                    start not in seen and value.get('scope')=='PREFIX_CALIB' and
                    value.get('status')=='CLAIMED_NO_AUTOMATIC_RETRY' and
                    value.get('output')==str(root/f'runs/PREFIX_CALIB_{start}') and
                    value.get('design_freeze_sha256')==freeze_sha,'Extra/substituted PREFIX allocation')
            seen.add(start)
    require(seen=={11031,11032},'Missing original PREFIX allocation')


def _capture_inventory(root,headers,manifest,config,freeze_sha):
    """Verify all 96 original attempts; do not open either raw JSONL here."""
    require(len(manifest)==96 and len({r['request_id'] for r in manifest})==96,'Frozen PREFIX allocation differs')
    for start,(path,header) in headers.items():
        selected={r['request_id']:r for r in manifest if r['start_seed']==start}
        require(len(selected)==48 and all(r['split']=='PREFIX_CALIB' and r['arm']=='START' and r['step']==0 for r in selected.values()),
                'Frozen START PREFIX allocation differs')
        refs=header['complete_files'];require(type(refs) is list and len(refs)==48,'Incomplete original COMPLETE inventory')
        observed=set();names=set()
        expected=dict(experiment_id=EXPERIMENT,start_seed=start,arm='START',step=0,
            **{k:config['start_bindings'][str(start)][k] for k in ('policy_sha256','reference_sha256')})
        require(all(header.get(k,{}).get('policy_sha256')==expected['policy_sha256'] and
                    header.get(k,{}).get('reference_sha256')==expected['reference_sha256'] for k in ('policy_before','policy_after')),
                'Original PREFIX source policy/reference differs')
        for ref in refs:
            require(Path(ref['path']).name==ref['path'],'Original COMPLETE must be directly under run')
            completed=_bound(path.parent,ref);row=core.read_json(completed);rid=row['request_id']
            require(rid in selected and rid not in observed,'Duplicate/extra original PREFIX completion');observed.add(rid)
            stem=hashlib.sha256(rid.encode()).hexdigest()
            require(completed.name==stem+'.COMPLETE.json','Original COMPLETE filename differs')
            capture=path.parent/(stem+'.CAPTURE.json');begin=path.parent/(stem+'.BEGIN.json')
            require(row['native_capture_path']==str(capture),'Original capture relocated')
            native=core.read_json(_bound(path.parent,dict(path=capture.name,sha256=row['native_capture_sha256'])))
            require(begin.resolve(strict=True)==begin and begin.is_file(),'Invalid BEGIN')
            started=core.read_json(begin)
            require(_same(started,dict(request_id=rid,seed=selected[rid]['seed'],endpoint=expected,
                design_freeze_sha256=freeze_sha,status='ATTEMPTED_UNKNOWN')),'Original BEGIN identity differs')
            require(all(k in row and _same(row[k],v) for k,v in native.items()),'Original CAPTURE changed in COMPLETE')
            require(all(k in native for k in ('tokens','raw_tokens','sampler_logprobs','prompt_tensor_hash','model_input','seed',
                'assistant_prefill','full_text','generated_text','finish_reason','generation_status','technical_validation_errors')),
                'Incomplete native capture proof')
            require(row.get('generation_status')=='COMPLETE' and row.get('technical_validation_errors')==[] and
                    row.get('finish_reason') in ('balanced','eos','length') and row.get('seed')==selected[rid]['seed'] and
                    row.get('design_freeze_sha256')==freeze_sha and row.get('policy_sha256')==expected['policy_sha256'] and
                    row.get('reference_sha256')==expected['reference_sha256'],'Technical/native endpoint failure is never waived')
            tokens=row['tokens'];probs=row['sampler_logprobs']
            require(type(tokens) is list and 1<=len(tokens)<=1536 and all(type(t) is int and t>=0 for t in tokens) and
                    _same(tokens,row['raw_tokens']) and type(row.get('completion_token_count')) is int and
                    row['completion_token_count']==len(tokens) and type(probs) is list and len(probs)==len(tokens) and
                    all(type(v) in (int,float) and math.isfinite(v) for v in probs) and
                    row.get('sampler_logprob_source')=='actual_generate_raw_logits_selected_token' and
                    row['full_text']==row['assistant_prefill']+row['generated_text'], 'Invalid original token/prefix capture')
            names.update(stem+suffix for suffix in ('.BEGIN.json','.CAPTURE.json','.COMPLETE.json'))
        require(observed==set(selected),'Missing original PREFIX requests')
        actual={p.name for p in path.parent.iterdir() if p.name.endswith(('.BEGIN.json','.CAPTURE.json','.COMPLETE.json'))}
        require(actual==names,'Unknown or extra PREFIX attempt is never waived')


def verify_completed(root,binding):
    """Independently replay all 96, preserving failure; never authorize training."""
    root=_root(root)
    freeze_ref=binding['design_freeze'];freeze=_bound(root,freeze_ref,'DESIGN_FREEZE.json');frozen=core.read_json(freeze)
    require(_same(binding['config'],frozen['artifacts']['config']) and
            _same(binding['source_lock'],frozen['artifacts']['source_lock']),'Substituted original config/source lock')
    config=_bound(root,binding['config']);_bound(root,binding['source_lock'])
    waiver=verify_waiver(root,binding['prefix_waiver'],expected_freeze_sha256=freeze_ref['sha256'],
                         expected_config_sha256=binding['config']['sha256'])
    require(_same(binding.get('prefix_result'),waiver['original_result']),'Original PREFIX result binding was substituted')
    path=_bound(root,binding['prefix_continuation'],RESULT_PATH);_no_errors(path.parent);result=core.read_json(path)
    fixed=dict(schema='sr-satdose-prefix-waiver-continuation-v1',status='PREFIX_CAPTURE_COMPLETE_WITH_AUTHORIZED_MEASUREMENT_WAIVER',
        gate_decision='PASS_USER_OVERRIDE',permission='NO_TRAINING',waiver=binding['prefix_waiver'],
        original_result=waiver['original_result'],original_release=waiver['original_release'],
        design_freeze_sha256=freeze_ref['sha256'],config_sha256=binding['config']['sha256'],
        responses=96,new_responses=48,resampled_responses=0)
    require(all(_same(result.get(k),v) for k,v in fixed.items()),'Incomplete/expanded waiver continuation')
    release=_release(root,result['release'],freeze_ref['sha256'],binding['config']['sha256'],
                     waiver_ref=binding['prefix_waiver'],original_ref=waiver['original_release'])
    require(result.get('source_inventory_sha256')==core.object_sha(release['sources']),'Continuation source digest differs')
    preflight=core.read_json(_bound(root,result['preflight'],'prefix_waiver_control/PREFLIGHT.json'))
    expected=dict(schema='sr-satdose-prefix-waiver-preflight-v1',status='READY_FOR_PREFIX_11032_ONLY',permission='NO_TRAINING',
        waiver=binding['prefix_waiver'],release=result['release'],original_result=waiver['original_result'],
        original_release=waiver['original_release'],design_freeze_sha256=freeze_ref['sha256'],
        config_sha256=binding['config']['sha256'],start_seed=11032,responses=48)
    require(all(_same(preflight.get(k),v) for k,v in expected.items()),'Continuation PREFLIGHT differs')
    claim=core.read_json(_bound(root,result['claim'],'prefix_waiver_control/CLAIM.json'))
    expected=dict(schema='sr-satdose-prefix-waiver-claim-v1',status='RESERVED_UNKNOWN_NO_RETRY',start_seed=11032,responses=48,
        waiver=binding['prefix_waiver'],release=result['release'],preflight=result['preflight'])
    require(all(_same(claim.get(k),v) for k,v in expected.items()),'Continuation CLAIM differs')
    reports=result['prefix_analyses'];require(set(reports)=={'11031','11032'},'Both PREFIX analyses required')
    original=core.read_json(_bound(root,waiver['original_result'],ORIGINAL_RESULT))['prefix_analyses']['11031']
    require(_same(reports['11031'],original),'Original 11031 failed result must be preserved exactly')
    require(_same(core.read_json(path.parent/'PREFIX_11032_ANALYSIS.json'),reports['11032']),'Stored continuation analysis differs')
    headers=_completion_headers(root,reports,freeze_ref['sha256'],binding['source_lock']['sha256'])
    _allocation_inventory(root,freeze_ref['sha256'])
    manifest=list(core.read_jsonl(_bound(root,frozen['artifacts']['prefix_calib_manifest'])))
    _capture_inventory(root,headers,manifest,core.read_json(config),freeze_ref['sha256'])
    for start in (11031,11032):
        report=reports[str(start)]
        require(_same(report.get('protocol_clarification'),original['protocol_clarification']),'Original PREFIX protocol changed')
        clarification=_bound(root,original['protocol_clarification']['clarification'])
        actual=calibration.analyze_prefix(config,freeze,root,headers[start][0],start,clarification=clarification)
        require(_same(actual,report),'Full original PREFIX analysis differs on replay: '+str(start))
    # Recheck immutable evidence and sources after CPU replay, not just headers.
    _allocation_inventory(root,freeze_ref['sha256'])
    _capture_inventory(root,headers,manifest,core.read_json(config),freeze_ref['sha256'])
    verify_waiver(root,binding['prefix_waiver'],expected_freeze_sha256=freeze_ref['sha256'],expected_config_sha256=binding['config']['sha256'])
    _bound(root,binding['prefix_continuation'],RESULT_PATH)
    _release(root,result['release'],freeze_ref['sha256'],binding['config']['sha256'],
             waiver_ref=binding['prefix_waiver'],original_ref=waiver['original_release'])
    for report in reports.values():
        _bound(root,report['raw']);_bound(root,report['completion_receipt'])
    return dict(status='PREFIX_WAIVER_CONTINUATION_AUDITED_NO_TRAINING',gate_decision='PASS_USER_OVERRIDE',permission='NO_TRAINING',
        waiver=copy.deepcopy(binding['prefix_waiver']),continuation=copy.deepcopy(binding['prefix_continuation']),
        responses=96,new_responses=48,resampled_responses=0,original_measurements_preserved=True,
        candidate_gate_passed={start:report['candidate_gate_passed'] for start,report in reports.items()},
        prefix_analyses=copy.deepcopy(reports),other_gates_unchanged=True,main_confirmatory_claims=DISCLOSURE)
