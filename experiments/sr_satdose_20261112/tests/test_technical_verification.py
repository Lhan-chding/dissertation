"""Synthetic CPU report-boundary tests; fixtures never constitute GPU evidence."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import technical_verification as t
sys.path.pop(0)


class TechnicalReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.audit_evidence={}
        self.verifier=self.root/'supplementary_verifier.py';self.verifier.write_text('# synthetic CPU proof producer\n')
        self.verifier_ref=dict(path=str(self.verifier),sha256=t.file_sha(self.verifier))
        self.auditor=self.root/'gpu_evidence.py'
        shutil.copyfile(PACKAGE/'execution/gpu_evidence.py',self.auditor)
        self.cfg=dict(experiment_id=t.EXPERIMENT,start_bindings={})
        for start in t.STARTS:
            p=self.root/f'original-{start}.pt';p.write_bytes(b'synthetic cpu state '+start.encode())
            self.cfg['start_bindings'][start]=dict(original_state_path=str(p),state_sha256=t.file_sha(p),
                policy_sha256='a'*64,reference_sha256='b'*64)
        self.config_ref=self.write('config.json',self.cfg)
        self.freeze_ref=self.write('DESIGN_FREEZE.json',dict(experiment_id=t.EXPERIMENT,config_sha256=self.config_ref['sha256']))
        self.registration_ref=self.write('REGISTRATION.json',dict(schema='sr-satdose-registration-v1',mode='production',
            experiment_id=t.EXPERIMENT,config_sha256=self.config_ref['sha256'],artifacts={'design_freeze':self.freeze_ref}))
        self.original_ref=self.write('prefix_control/RESULT.json',dict(status='HOLD_PREFIX_MEASUREMENT', stopped_start=11031, prefix_analyses={'11031':dict(candidate_gate_passed=False)}))
        self.waiver=dict(schema='sr-satdose-prefix-waiver-v1',status='USER_AUTHORIZED_MEASUREMENT_GATE_WAIVER',
            experiment_id=t.EXPERIMENT,scope='PREFIX_CALIB_MEASUREMENT_GATE_ONLY',evidence_root=str(self.root),
            design_freeze_sha256=self.freeze_ref['sha256'],config_sha256=self.config_ref['sha256'],original_result=self.original_ref,
            preserve_original_results=True,rewrite_measurement_pass=False,other_gates_unchanged=True,technical_failure_or_unknown_waived=False)
        self.waiver_ref=self.write('waiver.json',self.waiver)
        self.prefix_ref=self.write('prefix_complete.json',dict(status='PREFIX_CAPTURE_COMPLETE_WITH_AUTHORIZED_MEASUREMENT_WAIVER',
            gate_decision='PASS_USER_OVERRIDE',responses=96,resampled_responses=0,waiver=self.waiver_ref,original_result=self.original_ref,
            prefix_analyses={'11031':dict(candidate_gate_passed=False),'11032':dict(candidate_gate_passed=True)}))
        lock=self.write('source_lock.json',dict(synthetic=True))
        self.binding=dict(schema='sr-satdose-technical-execution-v1',mode='production',experiment_id=t.EXPERIMENT,
            experiment_root=str(self.root),config=self.config_ref,design_freeze=self.freeze_ref,source_lock=lock,
            prefix_result=self.original_ref,prefix_waiver=self.waiver_ref,prefix_continuation=self.prefix_ref)
        for field,prefix in [('evaluation_loader_slots','evaluation_loader_checks'),('backend_equivalence_slots','backend_equivalence'),('allzero_slots','allzero_check')]:
            self.binding[field]={s:f'{prefix}/start-{s}' for s in t.STARTS}
        self.binding_ref=self.write('TECHNICAL_EXECUTION_BINDING.json',self.binding,audited=True)
        self.audit=dict(schema='sr-satdose-independent-gpu-evidence-audit-v1',status='FINITE_GPU_EVIDENCE_AUDITED_NOT_ADMISSION',
            permission='NO_TRAINING',technical_binding_sha256=self.binding_ref['sha256'],auditor_sha256=t.AUDITOR_SHA,
            completed_fixed_stages=10,shadow_optimizer_steps=4,formal_optimizer_steps=0,audit_model_calls=0,audit_optimizer_calls=0,
            starts={},evidence=self.audit_evidence,limitations=['Synthetic fixtures: not actual GPU validation'])
        for start in t.STARTS:
            common=dict(technical_binding_sha256=self.binding_ref['sha256'],start_seed=int(start),permission='NO_TRAINING',optimizer_steps=0)
            self.write(f'evaluation_loader_checks/start-{start}/COMPLETE.json',dict(common,
                status='EVALUATION_START_LOAD_CAPTURE_COMPLETE_NOT_GPU_ADMISSION',tested_scope='START_R24_ONLY',
                trained_checkpoint_gpu_verified=False,trainable_tensors=496,caller_rng_preserved=True,
                policy_sha256='a'*64,reference_sha256='b'*64),audited=True)
            self.write(f'backend_equivalence/start-{start}/COMPLETE.json',dict(common,
                status='BACKEND_EQUIVALENCE_CAPTURED_NOT_GPU_ADMISSION',bit_exact=True,differences=[]),audited=True)
            self.write(f'allzero_check/start-{start}/COMPLETE.json',dict(common,
                status='ALLZERO_CAPTURE_COMPLETE_NOT_GPU_ADMISSION',comparison=dict(pg_exact_zero=True,kl_retained=True,
                original_scores_unchanged=True,explicit_diagnostic_advantage=0,kl_coefficient=.02,recomposition_relative_l2=1e-6)),audited=True)
            direction=dict(direction_checks_satisfied=True,hold_reasons=[],production_replay=dict(bit_exact=True,different_tensor_names=[]),
                recomposition_relative_l2=1e-6,**{k:.2 for k in t.COSINES},heldout_effect_half_1=.01,heldout_effect_half_2=.02,effect_repeat_error_floor=0.)
            self.audit['starts'][start]=dict(production_rows=128,max_original_tokens=709 if start=='11031' else 1536,
                long_original_row=None if start=='11031' else 'SAT1/DRY/11032/g004/d07',shadow_optimizer_steps=2,
                evaluation_scope='START_R24_RECEIPTS_AND_ORIGINAL_CPU_TENSOR_INVENTORY',memory_peak='NOT_RECORDED_FOR_PRODUCTION',permission='NO_TRAINING',direction=direction)
            for trial in (0,1):
                folder=f'shadow_inputs/start-{start}/trial-{trial}'
                for name in ['PRODUCTION_CAPTURE.pt','RESTORE.json','ACCOUNTING.json','WORKER_COMPLETE.json']:
                    self.write(folder+'/'+name,{'synthetic':True},audited=True)
                claim=f'shadow_claims/start-{start}/trial-{trial}'
                self.write(claim+'/RESULT.pt',{'synthetic':True},audited=True)
                self.write(claim+'/COMPLETE.json',dict(schema='synthetic',status='SHADOW_TRIAL_COMPLETE_NOT_GPU_ADMISSION',mode='production',
                    technical_binding_sha256=self.binding_ref['sha256'],start_seed=int(start),trial=trial,
                    optimizer_body_calls=1,optimizer_steps=1,formal_optimizer_steps=0,scheduler_steps=0),audited=True)
            self.write(f'shadow_inputs/start-{start}/trial-0/DIRECTION.json',direction,audited=True)
        starts={s:dict(policy_before_sha256='a'*64,policy_after_sha256='a'*64,reference_sha256='b'*64,
                      state_sha256=self.cfg['start_bindings'][s]['state_sha256'],responses=640) for s in t.STARTS}
        details={
          'start_and_reference_hash_verified':dict(starts=starts),
          'native_scorer_golden_pass':dict(fixture_count=9,failures=0,native_source_lock_sha256=lock['sha256']),
          'disjoint_splits_checked':dict(request_counts=t.COUNTS,new_splits_disjoint=True,historical_exclusion_verified=True,retention_exception_only=True),
          'gold_never_in_model_inputs':dict(total_requests=28936,model_input_fields=['image_file','image_sha256','prompt'],prefix_prefill_is_registered=True,private_metadata_absent=True),
          'dry_unchanged_policy':dict(total_responses=1280,optimizer_steps=0,parameter_updates=0,starts=starts)}
        self.proofs={}
        for name,d in details.items():
            self.proofs[name]=self.write('proofs/'+name+'.json',dict(schema='sr-satdose-supplementary-technical-proof-v1',
                check=name,passed=True,experiment_id=t.EXPERIMENT,config_sha256=self.config_ref['sha256'],design_freeze_sha256=self.freeze_ref['sha256'],
                verifier=self.verifier_ref,evidence={**{v['original_state_path']:v['state_sha256'] for v in self.cfg['start_bindings'].values()},
                    **{r['path']:r['sha256'] for r in (self.config_ref,self.freeze_ref,lock)}},details=d))

    def write(self,name,value,*,audited=False):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,allow_nan=False))
        ref=dict(path=name,sha256=t.file_sha(p))
        if audited:self.audit_evidence[name]=ref['sha256']
        return ref

    def build(self):
        ref=self.write('independent_audit.json',self.audit)
        return t.build_report(self.root,config_ref=self.config_ref,registration_ref=self.registration_ref,
            binding_ref=self.binding_ref,audit_ref=ref,auditor_ref=dict(path=str(self.auditor),sha256=t.AUDITOR_SHA),
            extra_proofs=self.proofs,source_sha256=t.file_sha(PACKAGE/'execution/technical_verification.py'))

    def mutate_marker(self,path,mutate):
        value=json.loads((self.root/path).read_text());mutate(value);self.write(path,value,audited=True)

    def test_report_has_fourteen_passes_and_original_prefix_false(self):
        report=self.build()
        self.assertEqual(len(report['checks']),15)
        self.assertEqual(sum(v['passed'] for v in report['checks'].values()),14)
        self.assertIs(report['checks']['prefix_calibration_pass']['passed'],False)
        self.assertEqual(report['checks']['prefix_calibration_pass']['evidence'],self.original_ref)
        self.assertEqual(report['prefix_gate_decision'],'PASS_USER_OVERRIDE')
        self.assertEqual(report['permission'],'NO_TRAINING')
        self.assertEqual(report['gradient']['11031']['J_split_half_cosine'],.2)

    def test_fake_audit_pass_without_raw_coverage_rejected(self):
        del self.audit_evidence['shadow_inputs/start-11032/trial-1/PRODUCTION_CAPTURE.pt']
        with self.assertRaisesRegex(ValueError,'Missing actual shadow'):self.build()

    def test_actual_evidence_hash_tamper_rejected(self):
        (self.root/'shadow_claims/start-11031/trial-0/RESULT.pt').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'SHA changed'):self.build()

    def test_unknown_fifth_shadow_attempt_rejected(self):
        (self.root/'shadow_claims/start-11031/trial-2').mkdir()
        with self.assertRaisesRegex(ValueError,'Unexpected shadow'):self.build()

    def test_shadow_counter_and_bool_counter_rejected(self):
        for count in [5,True]:
            with self.subTest(count=count):
                self.audit['shadow_optimizer_steps']=count
                with self.assertRaisesRegex(ValueError,'identity/counters'):self.build()

    def test_wrong_bound_audit_rejected(self):
        self.audit['technical_binding_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'identity/counters'):self.build()

    def test_long_row_is_exact_original_not_short_replacement(self):
        self.audit['starts']['11032']['max_original_tokens']=1535
        with self.assertRaisesRegex(ValueError,'fullbatch'):self.build()
        self.audit['starts']['11032']['max_original_tokens']=1536
        self.audit['starts']['11032']['long_original_row']='replacement'
        with self.assertRaisesRegex(ValueError,'fullbatch'):self.build()

    def test_full_batch_cannot_use_partial_rows(self):
        self.audit['starts']['11031']['production_rows']=127
        with self.assertRaisesRegex(ValueError,'fullbatch'):self.build()

    def test_nonpositive_and_nonfinite_directions_rejected(self):
        self.audit['starts']['11031']['direction']['source_split_half_cosine']=0
        with self.assertRaisesRegex(ValueError,'cosine'):self.build()
        self.audit['starts']['11031']['direction']['source_split_half_cosine']=.2
        self.audit['starts']['11031']['direction']['heldout_effect_half_1']=0
        with self.assertRaisesRegex(ValueError,'Heldout'):self.build()

    def test_backend_different_actual_marker_rejected(self):
        self.mutate_marker('backend_equivalence/start-11031/COMPLETE.json',lambda d:d.update(bit_exact=False))
        with self.assertRaisesRegex(ValueError,'Backend comparison'):self.build()

    def test_allzero_kl_missing_rejected(self):
        self.mutate_marker('allzero_check/start-11032/COMPLETE.json',lambda d:d['comparison'].update(kl_retained=False))
        with self.assertRaisesRegex(ValueError,'Allzero'):self.build()

    def test_fake_trained_loader_claim_rejected(self):
        self.mutate_marker('evaluation_loader_checks/start-11031/COMPLETE.json',lambda d:d.update(trained_checkpoint_gpu_verified=True))
        with self.assertRaisesRegex(ValueError,'START evaluation'):self.build()

    def test_scope_extended_waiver_rejected_even_with_updated_hash(self):
        self.waiver['technical_failure_or_unknown_waived']=True
        self.waiver_ref=self.write('waiver.json',self.waiver)
        self.binding['prefix_waiver']=self.waiver_ref
        self.binding_ref=self.write('TECHNICAL_EXECUTION_BINDING.json',self.binding,audited=True)
        # Direct proof-scope validator is reached only after identity checks; changed
        # binding cannot be smuggled into a previously recorded independent audit.
        with self.assertRaisesRegex(ValueError,'identity/counters'):self.build()

    def test_stale_supplementary_evidence_rejected(self):
        ref=self.proofs['native_scorer_golden_pass'];p=self.root/ref['path']
        d=json.loads(p.read_text());d['details']['failures']=1
        self.proofs['native_scorer_golden_pass']=self.write(ref['path'],d)
        with self.assertRaisesRegex(ValueError,'Golden'):self.build()

    def test_public_gold_field_rejected(self):
        ref=self.proofs['gold_never_in_model_inputs'];d=json.loads((self.root/ref['path']).read_text())
        d['details']['model_input_fields'].append('gold')
        self.proofs['gold_never_in_model_inputs']=self.write(ref['path'],d)
        with self.assertRaisesRegex(ValueError,'Public input'):self.build()

    def test_missing_or_failing_supplementary_check_rejected(self):
        del self.proofs['disjoint_splits_checked']
        with self.assertRaisesRegex(ValueError,'coverage'):self.build()

    def test_existing_hold_prevents_report(self):
        (self.root/'shadow_inputs/start-11031/trial-0/HOLD.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'HOLD'):self.build()

    def test_wrong_source_pin_rejected_before_any_audit(self):
        with self.assertRaisesRegex(ValueError,'constructor source'):
            t.build_report(self.root,config_ref={},registration_ref={},binding_ref={},audit_ref={},auditor_ref={},extra_proofs={},source_sha256='a'*64)

    def test_symlink_evidence_rejected(self):
        p=self.root/'shadow_claims/start-11031/trial-0/RESULT.pt';p.unlink();p.symlink_to(self.root/'config.json')
        with self.assertRaisesRegex(ValueError,'Noncanonical'):self.build()

    def test_duplicate_json_and_nonfinite_json_rejected(self):
        for raw in ['{"a":1,"a":2}', '{"x":NaN}']:
            p=self.root/'untrusted.json';p.write_text(raw)
            with self.assertRaises(ValueError):t.read_json(p)

if __name__=='__main__':unittest.main()
