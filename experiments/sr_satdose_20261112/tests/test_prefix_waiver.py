"""Pure CPU synthetic waiver evidence; real analyze_prefix, no model/native GPU."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import prefix_waiver as w


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def lines(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
def ref(root,path):return dict(path=str(path.relative_to(root)),sha256=w.core.file_sha(path))


def make_fixture(root,*,second_passes=True):
    """Reusable complete96 synthetic fixture, NOT production user/GPU evidence."""
    root=Path(root).resolve(strict=True);cfg=dict(experiment_id=w.EXPERIMENT,parent='b0',start_bindings={
        str(s):dict(policy_sha256=str(i+1)*64,reference_sha256='b'*64) for i,s in enumerate((11031,11032))})
    config=root/'code/supplied_package/config/experiment.json';dump(config,cfg)
    dump(root/'runtime_source_lock.json',{})
    (root/'image.png').write_bytes(b'SYNTHETIC CPU IMAGE ONLY')
    cases=[];inputs=[];requests=[];raws={11031:[],11032:[]}
    for i in range(16):
        rt=f'SYNTHETIC_PREFIX_ROOT_{i}';structure=w.core.STRUCTURES[i//4]
        task=dict(root_id=rt,structure=structure,world=dict(categories=list('ABCDEFGHIJ')),
                  gold=dict(selected=['A','B','C']))
        for view in w.calibration.VIEWS:
            cid=f'{rt}/{view}/case';iid=f'{rt}/{view}/input';pref='{"evidence":[],'
            inp=dict(input_id=iid,image_file='image.png',prompt=f'SYNTHETIC {rt} {view}',assistant_prefill=pref)
            inputs.append(inp);cases.append(dict(case_id=cid,input_id=iid,task=task,audit=dict(native_J_applicable=False)))
            for start in (11031,11032):
                rid=f'SAT1/PREFIX_CALIB/{start}/START/s00/prefix10/{rt}/{view}/d00'
                man=dict(request_id=rid,split='PREFIX_CALIB',start_seed=start,arm='START',step=0,panel='prefix10',
                         view=view,root_id=rt,case_id=cid,input_id=iid,structure=structure,seed=i,draw=0)
                requests.append(man)
                selected=['A','B','D'] if view=='p_revtie' and i>=8 else ['A','B','C']
                text=pref+'"selected":'+json.dumps(selected)+'}';finish='balanced'
                if view=='p_stable' and i>=13 and (start==11031 or not second_passes):
                    text=pref+'"selected":["A"';finish='eos'
                raw=dict(man,scope='PREFIX_CALIB',full_text=text,generated_text=text[len(pref):],suffix_text=text[len(pref):],
                    assistant_prefill=pref,tokens=[11,12],raw_tokens=[11,12],completion_token_count=2,
                    sampler_logprobs=[-.2,-.3],sampler_logprob_source='actual_generate_raw_logits_selected_token',
                    model_input=dict(prompt=inp['prompt'],image_file='image.png',image_sha256=w.core.file_sha(root/'image.png')),
                    prompt_tensor_hash='c'*64,finish_reason=finish,generation_status='COMPLETE',technical_validation_errors=[],
                    native_score=None,native_J=None,native_J_applicable=False,dry_optimizer_steps=0,dry_parameter_updates=0,
                    **cfg['start_bindings'][str(start)])
                raws[start].append(raw)
    for name,rows in (('private_cases',cases),('inputs',inputs),('prefix_calib_manifest',requests),
                      ('images',[dict(image_file='image.png',png_sha256=w.core.file_sha(root/'image.png'))])):
        lines(root/(name+'.jsonl'),rows)
    runtime_sources={}
    for name in w.RUNTIME:
        path=root/'code/runtime'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# SYNTHETIC FROZEN CPU FIXTURE '+name+'\n');runtime_sources[name]=w.core.file_sha(path)
    frozen=dict(schema='sr-satdose-design-freeze-v1',experiment_id=w.EXPERIMENT,config_sha256=w.core.file_sha(config),
        runtime_sources=runtime_sources,artifacts=dict(config=ref(root,config),source_lock=ref(root,root/'runtime_source_lock.json')))
    for name in ('private_cases','inputs','prefix_calib_manifest','images'):frozen['artifacts'][name]=ref(root,root/(name+'.jsonl'))
    freeze=root/'DESIGN_FREEZE.json';dump(freeze,frozen);freeze_sha=w.core.file_sha(freeze);config_sha=w.core.file_sha(config)
    protocol=root/'protocol_clarifications/PREFIX_PROTOCOL_CLARIFICATION_20261113.json';proof=root/'protocol_clarifications/PRE_PREFIX_CONFIRMATION.json'
    identity=dict(experiment_id=w.EXPERIMENT,config_sha256=config_sha,design_freeze_sha256=freeze_sha)
    dump(proof,dict(schema='sr-satdose-pre-prefix-confirmation-v1',status='NO_PREFIX_ATTEMPTS_OBSERVED',
        **identity,prefix_attempts=0,prefix_allocations=[],checked_at='2026-11-13T00:00:00+00:00'))
    dump(protocol,dict(schema='sr-satdose-prefix-protocol-clarification-v1',**identity,decision_rule=w.calibration.PREFIX_RULE,
        decision_origin='agent_interpretation_after_user_requested_judgment',user_explicit_option_approval=False,
        pre_prefix_confirmation=ref(root,proof)))
    reports={}
    for start,rows in raws.items():
        directory=root/f'runs/PREFIX_CALIB_{start}';directory.mkdir(parents=True)
        dump(root/f'calibration_allocations/PREFIX_CALIB-{start}.json',dict(output=str(directory),
            design_freeze_sha256=freeze_sha,scope='PREFIX_CALIB',start_seed=start,status='CLAIMED_NO_AUTOMATIC_RETRY'))
        endpoint=dict(experiment_id=w.EXPERIMENT,start_seed=start,arm='START',step=0,**cfg['start_bindings'][str(start)])
        refs=[]
        for row in rows:
            row['design_freeze_sha256']=freeze_sha
            stem=hashlib.sha256(row['request_id'].encode()).hexdigest()
            keys=('seed','full_text','generated_text','assistant_prefill','tokens','raw_tokens','completion_token_count',
                  'sampler_logprobs','sampler_logprob_source','prompt_tensor_hash','model_input','finish_reason',
                  'generation_status','technical_validation_errors')
            capture=directory/(stem+'.CAPTURE.json');dump(capture,{k:row[k] for k in keys})
            dump(directory/(stem+'.BEGIN.json'),dict(request_id=row['request_id'],seed=row['seed'],endpoint=endpoint,
                 design_freeze_sha256=freeze_sha,status='ATTEMPTED_UNKNOWN'))
            row.update(native_capture_path=str(capture),native_capture_sha256=w.core.file_sha(capture))
            complete={k:row[k] for k in (*keys,'request_id','native_capture_path','native_capture_sha256',
                       'policy_sha256','reference_sha256','design_freeze_sha256')}
            path=directory/(stem+'.COMPLETE.json');dump(path,complete);refs.append(ref(directory,path))
        lines(directory/'PREFIX_RAW.jsonl',rows)
        receipt=dict(status='CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION',scope='PREFIX_CALIB',start_seed=start,responses=48,
            design_freeze_sha256=freeze_sha,source_lock_sha256=frozen['artifacts']['source_lock']['sha256'],
            optimizer_created=False,scheduler_created=False,optimizer_steps=0,parameter_updates=0,
            policy_before=endpoint,policy_after=endpoint,raw=ref(directory,directory/'PREFIX_RAW.jsonl'),complete_files=refs)
        path=directory/'CALIBRATION_COMPLETE.json';dump(path,receipt)
        reports[str(start)]=w.calibration.analyze_prefix(config,freeze,root,path,start,clarification=protocol)
    original=dict(status='HOLD_PREFIX_MEASUREMENT',permission='NO_TRAINING',stopped_start=11031,prefix_analyses={'11031':reports['11031']})
    original_path=root/w.ORIGINAL_RESULT;dump(original_path,original)
    dump(root/'prefix_control/PREFIX_11031_ANALYSIS.json',reports['11031'])
    def release_sources(directory,old):
        directory.mkdir()
        for target,source in (('analysis/calibration.py',Path(w.calibration.__file__)),
                             ('supplied_package/tools/satdose_core.py',Path(w.core.__file__)),
                             ('supplied_package/config/experiment.json',config)):
            dest=directory/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        if old:
            for name in w.RUNTIME:
                dest=directory/'runtime'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(root/'code/runtime'/name,dest)
            dest=directory/'execution/prefix_control.py';dest.parent.mkdir();dest.write_text('# SYNTHETIC ORIGINAL CONTROL')
        else:
            dest=directory/'execution/prefix_waiver.py';dest.parent.mkdir();shutil.copyfile(w.__file__,dest)
            (directory/'execution/prefix_continuation.py').write_text('# SYNTHETIC CONTROLLER, NO GPU')
            dest=directory/'scheduling/run_prefix_continuation.sh';dest.parent.mkdir();dest.write_text('# SYNTHETIC LAUNCHER, NO GPU')
        return w._inventory(directory,old=old)
    release_root=root/'original_release';sources=release_sources(release_root,True)
    common=dict(experiment_id=w.EXPERIMENT,evidence_root=str(root),design_freeze_sha256=freeze_sha,
                config_sha256=config_sha,frozen_code_root=str(root/'code'),permission='NO_TRAINING')
    old_release=dict(common,schema='sr-satdose-prefix-technical-release-v1',status='AUDITED_FOR_PREFIX_CALIB_ONLY',
                     release_root=str(release_root),sources=sources)
    old_release_path=root/w.ORIGINAL_RELEASE;dump(old_release_path,old_release);old_ref=ref(root,old_release_path)
    dump(root/'prefix_control/PREFLIGHT.json',dict(schema='sr-satdose-prefix-preflight-v1',status='READY_FOR_PREFIX_CALIB_ONLY',
        permission='NO_TRAINING',model_calls=0,starts=[11031,11032],responses_per_start=48,design_freeze_sha256=freeze_sha,
        config_sha256=config_sha,protocol_clarification=reports['11031']['protocol_clarification'],
        release=dict(path=str(old_release_path),sha256=old_ref['sha256'],release_root=str(release_root),sources=sources)))
    waiver=dict(schema='sr-satdose-prefix-waiver-v1',status='USER_AUTHORIZED_MEASUREMENT_GATE_WAIVER',**identity,
        evidence_root=str(root),created_at_utc='2026-11-14T03:45:37.617963+00:00',authorization=dict(origin='explicit_user_reply',question=w.QUESTION,answer=w.ANSWER),
        scope='PREFIX_CALIB_MEASUREMENT_GATE_ONLY',original_result=ref(root,original_path),original_job_id=1737594,
        original_release=old_ref,preserve_original_results=True,rewrite_measurement_pass=False,remaining_calibration=w.REMAINING,
        require_both_complete_48_before_gpu=True,other_gates_unchanged=True,technical_failure_or_unknown_waived=False,
        main_confirmatory_claims=w.DISCLOSURE,permission='NO_TRAINING')
    waiver_path=root/w.WAIVER_PATH;dump(waiver_path,waiver);waiver_ref=ref(root,waiver_path)
    new_root=root/'new_release';new_sources=release_sources(new_root,False)
    new_release=dict(common,schema='sr-satdose-prefix-continuation-release-v1',status='AUDITED_FOR_PREFIX_11032_CONTINUATION_ONLY',
        release_root=str(new_root),sources=new_sources,waiver=waiver_ref,original_release=old_ref)
    new_release_path=root/'execution_releases/PREFIX_CONTINUATION_RELEASE.json';dump(new_release_path,new_release);new_ref=ref(root,new_release_path)
    preflight=dict(schema='sr-satdose-prefix-waiver-preflight-v1',status='READY_FOR_PREFIX_11032_ONLY',permission='NO_TRAINING',
        waiver=waiver_ref,release=new_ref,original_result=waiver['original_result'],original_release=old_ref,
        design_freeze_sha256=freeze_sha,config_sha256=config_sha,start_seed=11032,responses=48)
    control=root/'prefix_waiver_control';dump(control/'PREFLIGHT.json',preflight)
    preflight_ref=ref(root,control/'PREFLIGHT.json')
    dump(control/'CLAIM.json',dict(schema='sr-satdose-prefix-waiver-claim-v1',status='RESERVED_UNKNOWN_NO_RETRY',
        start_seed=11032,responses=48,waiver=waiver_ref,release=new_ref,preflight=preflight_ref))
    dump(control/'PREFIX_11032_ANALYSIS.json',reports['11032'])
    result=dict(schema='sr-satdose-prefix-waiver-continuation-v1',status='PREFIX_CAPTURE_COMPLETE_WITH_AUTHORIZED_MEASUREMENT_WAIVER',
        gate_decision='PASS_USER_OVERRIDE',permission='NO_TRAINING',waiver=waiver_ref,original_result=waiver['original_result'],original_release=old_ref,
        design_freeze_sha256=freeze_sha,config_sha256=config_sha,responses=96,new_responses=48,resampled_responses=0,
        prefix_analyses=reports,release=new_ref,source_inventory_sha256=w.core.object_sha(new_sources),
        preflight=preflight_ref,claim=ref(root,control/'CLAIM.json'))
    dump(control/'RESULT.json',result)
    return dict(design_freeze=ref(root,freeze),config=frozen['artifacts']['config'],source_lock=frozen['artifacts']['source_lock'],
        prefix_waiver=waiver_ref,prefix_result=waiver['original_result'],prefix_continuation=ref(root,control/'RESULT.json'))


class PrefixWaiverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.binding=make_fixture(self.root)

    def verify(self):return w.verify_completed(self.root,self.binding)
    def waiver(self):
        return w.verify_waiver(self.root,self.binding['prefix_waiver'],expected_freeze_sha256=self.binding['design_freeze']['sha256'],
                               expected_config_sha256=self.binding['config']['sha256'])
    def change_result(self,fn):
        path=self.root/w.RESULT_PATH;value=w.core.read_json(path);fn(value);dump(path,value);self.binding['prefix_continuation']=ref(self.root,path)

    def test_actual96_replay_keeps13_of16_failure_and_performs_no_writes(self):
        before={str(p.relative_to(self.root)):w.core.file_sha(p) for p in self.root.rglob('*') if p.is_file()}
        with patch.object(w.calibration,'analyze_prefix',wraps=w.calibration.analyze_prefix) as replay:
            report=self.verify();self.assertEqual(replay.call_count,2)
        self.assertEqual(report['gate_decision'],'PASS_USER_OVERRIDE');self.assertEqual(report['permission'],'NO_TRAINING')
        self.assertEqual(report['candidate_gate_passed'],{'11031':False,'11032':True})
        original=report['prefix_analyses']['11031'];self.assertEqual(original['status'],'HOLD_PREFIX_MEASUREMENT')
        self.assertEqual(original['readouts']['strict']['views']['p_stable']['known'],13)
        self.assertEqual(before,{str(p.relative_to(self.root)):w.core.file_sha(p) for p in self.root.rglob('*') if p.is_file()})

    def test_both_failed_complete_measurements_remain_failed_under_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();binding=make_fixture(root,second_passes=False)
            self.assertEqual(w.verify_completed(root,binding)['candidate_gate_passed'],{'11031':False,'11032':False})

    def test_verify_authorization_does_not_read_raw(self):
        for path in self.root.glob('runs/*/PREFIX_RAW.jsonl'):path.unlink()
        self.assertEqual(self.waiver()['scope'],'PREFIX_CALIB_MEASUREMENT_GATE_ONLY')

    def test_flags_or_authorization_text_cannot_be_broadened_even_with_new_ref(self):
        path=self.root/w.WAIVER_PATH;original=w.core.read_json(path)
        for key,value in (('technical_failure_or_unknown_waived',True),('rewrite_measurement_pass',True),
                          ('other_gates_unchanged',False),('permission','TRAINING_ALLOWED'),('responses',96)):
            modified=copy.deepcopy(original)
            if key=='responses':modified['remaining_calibration'][key]=value
            else:modified[key]=value
            dump(path,modified);self.binding['prefix_waiver']=ref(self.root,path)
            with self.subTest(key=key),self.assertRaises(ValueError):self.waiver()

    def test_old_original_result_and_old_release_hashes_are_not_replaceable(self):
        (self.root/w.ORIGINAL_RESULT).write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA differs'):self.waiver()

    def test_both_headers_required_before_any_analysis_or_raw_access(self):
        path=self.root/'runs/PREFIX_CALIB_11032/CALIBRATION_COMPLETE.json';header=w.core.read_json(path);header['responses']=47;dump(path,header)
        def update(result):result['prefix_analyses']['11032']['completion_receipt']=ref(self.root,path)
        self.change_result(update)
        report=w.core.read_json(self.root/w.RESULT_PATH)['prefix_analyses']['11032']
        dump(self.root/'prefix_waiver_control/PREFIX_11032_ANALYSIS.json',report)
        (self.root/'runs/PREFIX_CALIB_11031/PREFIX_RAW.jsonl').unlink()
        with patch.object(w.calibration,'analyze_prefix',side_effect=AssertionError('partial raw scoring')):
            with self.assertRaisesRegex(ValueError,'Both original 48'):self.verify()

    def test_unknown_extra_attempt_never_waived(self):
        dump(self.root/'runs/PREFIX_CALIB_11032'/('0'*64+'.BEGIN.json'),{})
        with self.assertRaisesRegex(ValueError,'Unknown or extra'):self.verify()

    def test_extra_resampling_directory_and_allocation_cannot_hide_outside_fixed_runs(self):
        directory=self.root/'runs/PREFIX_CALIB_11031_retry';directory.mkdir()
        with self.assertRaisesRegex(ValueError,'run allocation'):self.verify()
        directory.rmdir()
        path=self.root/'calibration_allocations/PREFIX_CALIB-11031-retry.json'
        dump(path,dict(scope='PREFIX_CALIB',start_seed=11031))
        with self.assertRaisesRegex(ValueError,'PREFIX allocation'):self.verify()

    def test_technical_hold_in_either_old_or_new_control_never_waived(self):
        for directory in ('prefix_control','prefix_waiver_control','runs/PREFIX_CALIB_11032'):
            path=self.root/directory/'HOLD.json';dump(path,{})
            with self.subTest(directory=directory),self.assertRaisesRegex(ValueError,'Technical HOLD/ERROR'):self.verify()
            path.unlink()

    def test_native_capture_hash_and_begin_endpoint_tampering_rejected(self):
        path=next((self.root/'runs/PREFIX_CALIB_11032').glob('*.CAPTURE.json'));path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA differs'):self.verify()

    def test_same_gate_decision_but_changed_analysis_row_rejected_by_full_replay(self):
        def update(result):result['prefix_analyses']['11032']['rows'][0]['strict']['selected']=['WRONG']
        self.change_result(update)
        dump(self.root/'prefix_waiver_control/PREFIX_11032_ANALYSIS.json',w.core.read_json(self.root/w.RESULT_PATH)['prefix_analyses']['11032'])
        with self.assertRaisesRegex(ValueError,'Full original PREFIX analysis differs'):self.verify()

    def test_original_failed_analysis_cannot_be_relabelled_pass(self):
        self.change_result(lambda r:r['prefix_analyses']['11031'].update(status='PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED',candidate_gate_passed=True))
        with self.assertRaisesRegex(ValueError,'preserved exactly'):self.verify()

    def test_source_inventory_and_actual_verifier_bytes_are_bound(self):
        (self.root/'new_release/execution/prefix_waiver.py').write_text('# modified verifier')
        with self.assertRaisesRegex(ValueError,'source inventory'):self.verify()

    def test_missing_claim_no_override_label_and_resampling_rejected(self):
        original=w.core.read_json(self.root/w.RESULT_PATH)
        for key,value in (('gate_decision','PASS'),('new_responses',96),('resampled_responses',1)):
            dump(self.root/w.RESULT_PATH,original);self.change_result(lambda r:r.update({key:value}))
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Incomplete/expanded'):self.verify()
        dump(self.root/w.RESULT_PATH,original);self.binding['prefix_continuation']=ref(self.root,self.root/w.RESULT_PATH)
        (self.root/'prefix_waiver_control/CLAIM.json').unlink()
        with self.assertRaises(FileNotFoundError):self.verify()

    def test_binding_original_prefix_result_and_fixed_paths_required(self):
        self.binding['prefix_result']=self.binding['prefix_continuation']
        with self.assertRaisesRegex(ValueError,'result binding'):self.verify()
        self.binding['prefix_result']=w.core.read_json(self.root/w.WAIVER_PATH)['original_result']
        self.binding['prefix_continuation']['path']='../RESULT.json'
        with self.assertRaisesRegex(ValueError,'Relocated/escaping'):self.verify()


if __name__=='__main__':unittest.main()
