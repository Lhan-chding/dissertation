"""CPU boundary tests, synthetic evidence; no assertion of production admission.

Full DRY/GPU/PREFIX semantics have their own real verifier suites. Here explicit
mocks isolate amended routing and source/receipt changes, while every fourteen-
gate/gradient/artifact check runs the actual new validator.
"""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import admission_authority as a
sys.path.pop(0)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


def ref(root,path):return dict(path=str(path.relative_to(root)),sha256=a.tx.file_sha(path))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();p=self.root/'proof.json';dump(p,{'CPU_SYNTHETIC':True})
        self.proof=ref(self.root,p);self.observed=a._Observed(self.root)
        self.cfg={'experiment_id':a.EXPERIMENT,'start_bindings':{str(s):{'policy_sha256':str(i+1)*64}
            for i,s in enumerate((11031,11032))}}
        self.direction={k:.1 for k in a.GRADIENT_FIELDS};self.direction['effect_repeat_error_floor']=.01
        self.audit={'shadow_optimizer_steps':4,'starts':{str(s):{'direction':copy.deepcopy(self.direction)} for s in (11031,11032)}}
        self.gpu=dict(schema='sr-satdose-gpu-verification-v1',mode='production',experiment_id=a.EXPERIMENT,
            config_sha256='c'*64,registration_sha256='d'*64,dry_optimizer_steps=0,dry_parameter_updates=0,
            shadow_optimizer_steps=4,prefix_gate_decision='PASS_USER_OVERRIDE',prefix_waiver=self.proof,
            observed_evidence={str(p):self.proof['sha256']},
            independent_audit=self.proof,checks={k:dict(passed=True,evidence=self.proof) for k in a.TECHNICAL},
            gradient={str(s):dict(self.direction,policy_before_sha256=self.cfg['start_bindings'][str(s)]['policy_sha256'],
                policy_after_sha256=self.cfg['start_bindings'][str(s)]['policy_sha256'],evidence=self.proof) for s in (11031,11032)})
        self.gpu['checks'][a.PREFIX]=dict(passed=False,evidence=self.proof)
    def verify(self):
        return a.verify_gpu_report(self.gpu,self.cfg,dict(path='config',sha256='c'*64),
            dict(path='registration',sha256='d'*64),self.observed,self.audit,self.proof,self.proof)
    def test_correct_false_measurement_and_explicit_override_are_preserved(self):
        before=copy.deepcopy(self.gpu);self.verify();self.assertEqual(before,self.gpu)
        self.assertIs(self.gpu['checks'][a.PREFIX]['passed'],False)
    def test_every_technical_gate_cannot_be_waived(self):
        for name in a.TECHNICAL:
            with self.subTest(name=name):
                self.gpu['checks'][name]['passed']=False
                with self.assertRaises(PermissionError):self.verify()
                self.gpu['checks'][name]['passed']=True
    def test_truthy_integer_gate_not_boolean(self):
        self.gpu['checks'][a.TECHNICAL[0]]['passed']=1
        with self.assertRaises(PermissionError):self.verify()
    def test_rewriting_prefix_measurement_true_rejected(self):
        self.gpu['checks'][a.PREFIX]['passed']=True
        with self.assertRaises(PermissionError):self.verify()
    def test_missing_or_extra_gate_rejected(self):
        del self.gpu['checks'][a.TECHNICAL[0]]
        with self.assertRaises(PermissionError):self.verify()
        self.gpu['checks']['invented']=dict(passed=True,evidence=self.proof)
        with self.assertRaises(PermissionError):self.verify()
    def test_evidence_tamper_rejected(self):
        (self.root/'proof.json').write_text('changed')
        with self.assertRaises(PermissionError):self.verify()
    def test_foreign_waiver_or_audit_rejected(self):
        for key in ('prefix_waiver','independent_audit'):
            old=self.gpu[key];self.gpu[key]={'path':'other','sha256':'f'*64}
            with self.subTest(key=key),self.assertRaises(PermissionError):self.verify()
            self.gpu[key]=old
    def test_no_unlabelled_override(self):
        self.gpu['prefix_gate_decision']='PASS'
        with self.assertRaises(PermissionError):self.verify()
    def test_zero_negative_nonfinite_and_out_of_range_cosines_rejected(self):
        for key in a.GRADIENT_FIELDS[:3]:
            for value in (0.,-.01,1.01,float('nan'),float('inf'),True):
                self.gpu['gradient']['11031'][key]=value
                with self.subTest(key=key,value=value),self.assertRaises(PermissionError):self.verify()
            self.gpu['gradient']['11031'][key]=self.direction[key]
    def test_effect_must_exceed_original_floor(self):
        self.gpu['gradient']['11032']['heldout_effect_half_1']=.01
        with self.assertRaises(PermissionError):self.verify()
    def test_plausible_positive_but_substituted_gradient_rejected(self):
        self.gpu['gradient']['11031']['J_vs_source_pg_cosine']=.2
        with self.assertRaises(PermissionError):self.verify()
    def test_probe_policy_update_rejected(self):
        self.gpu['gradient']['11031']['policy_after_sha256']='e'*64
        with self.assertRaises(PermissionError):self.verify()
    def test_budget_and_dry_updates_rejected(self):
        for key,value in [('dry_optimizer_steps',1),('dry_parameter_updates',True),('shadow_optimizer_steps',5),('shadow_optimizer_steps',3)]:
            old=self.gpu[key];self.gpu[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(PermissionError):self.verify()
            self.gpu[key]=old
    def test_evidence_symlink_and_escape_rejected(self):
        link=self.root/'link';link.symlink_to(self.root/'proof.json')
        for path in ('link','../proof.json','/tmp/proof.json'):
            self.gpu['checks'][a.TECHNICAL[0]]['evidence']=dict(path=path,sha256=self.proof['sha256'])
            with self.subTest(path=path),self.assertRaises((PermissionError,FileNotFoundError)):self.verify()
    def test_observed_changed_content_or_new_file_fails_without_mtime_cache(self):
        self.observed.tree(self.root);self.observed.stable()
        dump(self.root/'new.json',{})
        with self.assertRaises(PermissionError):self.observed.stable()
        (self.root/'new.json').unlink();(self.root/'proof.json').write_text('{"CPU_SYNTHETIC":false}\n')
        with self.assertRaises(PermissionError):self.observed.stable()
    def test_external_supplementary_input_and_verifier_are_in_cached_hash_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            external=Path(tmp).resolve()
            for name in ('historical_private_cases.jsonl','supplementary_auditor.py'):
                path=external/name;path.write_text('ORIGINAL CPU FIXTURE')
                self.gpu['observed_evidence'][str(path)]=a.tx.file_sha(path)
            self.verify();self.observed.stable()
            for path in external.iterdir():
                old=path.read_bytes();path.write_bytes(b'changed')
                with self.subTest(path=path),self.assertRaises(PermissionError):self.observed.stable()
                path.write_bytes(old)
    def test_missing_supplementary_inventory_or_unlisted_proof_fails(self):
        self.gpu['observed_evidence']={}
        with self.assertRaises(PermissionError):self.verify()
        self.gpu['observed_evidence']={str(self.root/'proof.json'):self.proof['sha256']}
        second=self.root/'other.json';dump(second,{})
        self.gpu['checks'][a.TECHNICAL[0]]['evidence']=ref(self.root,second)
        with self.assertRaises(PermissionError):self.verify()
    def test_original_frozen_gate_set_has_not_drifted(self):
        module=a._load(dict(path=str(PACKAGE/'supplied_package/tools/check_admission.py'),
            sha256=a.tx.file_sha(PACKAGE/'supplied_package/tools/check_admission.py')),'_authority_test_original')
        self.assertEqual(module.REQUIRED_TECH,a.TECHNICAL+(a.PREFIX,))


class AuthorityBoundaryTests(unittest.TestCase):
    """Explicit mocked full validation; tests scope/cache/bindings only."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.source=dict(path=str(Path(a.__file__).resolve()),sha256=a.tx.file_sha(a.__file__))
        self.candidate={'schema':'CPU_SYNTHETIC_NOT_PRODUCTION','permission':'FOUR_PATHS_MAX_STEP_2'}
        dump(self.root/'admission.json',self.candidate);self.admission=ref(self.root,self.root/'admission.json')
        dump(self.root/'proof.json',{'cpu':True})
        self.observed=a._Observed(self.root);self.observed.absolute(self.root/'proof.json')
        self.binding={'design_freeze':dict(path='DESIGN_FREEZE.json',sha256='f'*64)}
        self.freeze={'artifacts':{'train_manifest':dict(path='train',sha256='a'*64),'training_groups':dict(path='groups',sha256='b'*64)}}
        self.cfg={'start_bindings':{'11031':dict(policy_sha256='1'*64,reference_sha256='2'*64)}}
        self.spec={'root':str(self.root),'sources':{'authority':self.source},'documents':{
            'config':dict(path='config',sha256='c'*64),'registration':dict(path='reg',sha256='d'*64),
            'gpu_report':dict(path='gpu',sha256='e'*64)},'training_execution_sources_sha256':'9'*64}
        p=patch.object(a,'_validate',return_value=(copy.deepcopy(self.candidate),self.observed,self.cfg,self.binding,self.freeze))
        p.start();self.addCleanup(p.stop)
        self.authority=a.FirstTwoStepAdmission(self.spec,admission=self.admission)
        self.identity=dict(experiment_id=a.EXPERIMENT,parent='b0',train_seed=11031,arm='T0',run_id=f'{a.EXPERIMENT}/b0-11031-T0',
            config_sha256='c'*64,design_freeze_sha256='f'*64,execution_sources_sha256='9'*64,
            source_binding=self.cfg['start_bindings']['11031'],manifest_sha256='a'*64,plan_sha256='b'*64)
        self.endpoint=dict(experiment_id=a.EXPERIMENT,start_seed=11031,arm='T0',step=0,policy_sha256='1'*64,reference_sha256='2'*64)
    def verify(self,step=1):
        return self.authority.verify(root=self.root,identity=self.identity,design_freeze_sha256='f'*64,target_step=step,sampled_endpoint=self.endpoint)
    def test_step1_returns_bound_receipts_without_mutating_identity(self):
        before=copy.deepcopy(self.identity);out=self.verify()
        self.assertEqual(out['admission'],self.admission);self.assertEqual(before,self.identity)
    def test_published_receipt_must_equal_full_recomputed_candidate(self):
        dump(self.root/'admission.json',{'status':'PASS'})
        with self.assertRaises(PermissionError):a.FirstTwoStepAdmission(self.spec,admission=ref(self.root,self.root/'admission.json'))
    def test_step3_main_and_bool_step_rejected(self):
        for step in (3,24,'MAIN',True):
            with self.subTest(step=step),self.assertRaises(PermissionError):self.verify(step)
    def test_main_or_dev_path_rejected(self):
        for arm in ('MAIN','DEV','START'):
            self.identity['arm']=arm
            with self.subTest(arm=arm),self.assertRaises(PermissionError):self.verify()
    def test_changed_receipt_and_observed_bytes_block(self):
        for path in (self.root/'admission.json',self.root/'proof.json'):
            old=path.read_bytes();path.write_bytes(b'changed')
            with self.subTest(path=path),self.assertRaises(PermissionError):self.verify()
            path.write_bytes(old)
    def test_changed_verifier_source_pin_blocks(self):
        self.authority.verifier_reference['sha256']='0'*64
        with self.assertRaises(PermissionError):self.verify()
    def test_foreign_execution_source_and_reference_block(self):
        for key in ('execution_sources_sha256','config_sha256','plan_sha256'):
            old=self.identity[key];self.identity[key]='0'*64
            with self.subTest(key=key),self.assertRaises(PermissionError):self.verify()
            self.identity[key]=old
        self.endpoint['reference_sha256']='0'*64
        with self.assertRaises(PermissionError):self.verify()
    def test_global_hold_blocks(self):
        dump(self.root/'HOLD.json',{})
        with self.assertRaises(PermissionError):self.verify()
    def test_step2_requires_committed_prior_state(self):
        self.endpoint.update(step=1,policy_sha256='3'*64)
        with self.assertRaises((PermissionError,FileNotFoundError)):self.verify(2)
        d=self.root/'training_transactions/start-11031/T0/step-001';d.mkdir(parents=True)
        (d/'state.pt').write_bytes(b'CPU SYNTHETIC STATE NOT A REAL CHECKPOINT')
        dump(d/'COMMIT.json',dict(identity=self.identity,step=1,policy_sha256='3'*64,reference_sha256='2'*64,state_sha256=a.tx.file_sha(d/'state.pt')))
        self.assertEqual(self.verify(2)['target_step'],2)
        (d/'state.pt').write_bytes(b'changed')
        with self.assertRaises(PermissionError):self.verify(2)



class LiveGPUOccupancyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        for name in ('evaluation_loader_checks','backend_equivalence','allzero_check','shadow_inputs','shadow_claims'):
            for start in (11031,11032):
                path=self.root/name/f'start-{start}';path.mkdir(parents=True)
                if name.startswith('shadow_'):
                    for trial in (0,1):(path/f'trial-{trial}').mkdir()
    def test_fixed_completed_slot_inventory_has_no_extra_allocation(self):
        a._gpu_slots(self.root)
    def test_hold_already_present_before_cache_is_rejected(self):
        for relative in ('HOLD.json','shadow_inputs/start-11031/trial-0/HOLD.json',
                         'allzero_check/start-11032/ERROR.json'):
            path=self.root/relative;dump(path,{})
            with self.subTest(relative=relative),self.assertRaises(PermissionError):a._gpu_slots(self.root)
            path.unlink()
    def test_extra_unknown_slot_before_cache_is_rejected(self):
        for relative in ('shadow_inputs/start-11031/trial-2','backend_equivalence/start-11033'):
            path=self.root/relative;path.mkdir()
            with self.subTest(relative=relative),self.assertRaises(PermissionError):a._gpu_slots(self.root)
            path.rmdir()


class ActualPrefixIntegrationTests(unittest.TestCase):
    """Real full96 synthetic PREFIX verifier; no native GPU admission claim."""
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('_admission_prefix_fixture',Path(__file__).with_name('test_prefix_waiver.py'))
        cls.fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.fixture)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.binding=self.fixture.make_fixture(self.root)
    def verify(self):
        return a._verified_prefix(self.root,self.binding,self.binding['prefix_waiver'],self.fixture.w)
    def test_real96_verifier_keeps_original_failed_measurement(self):
        result=self.verify();self.assertEqual(result['responses'],96)
        self.assertEqual(result['candidate_gate_passed'],{'11031':False,'11032':True})
    def test_fake_user_authorization_and_broadened_scope_fail(self):
        path=self.root/self.fixture.w.WAIVER_PATH;original=a.tx.read_json(path)
        for key,value in [('scope','ALL_GATES'),('technical_failure_or_unknown_waived',True),
                          ('rewrite_measurement_pass',True),('other_gates_unchanged',False),
                          ('authorization',dict(origin='agent',answer='PASS'))]:
            changed=copy.deepcopy(original);changed[key]=value;dump(path,changed)
            self.binding['prefix_waiver']=ref(self.root,path)
            with self.subTest(key=key),self.assertRaises(ValueError):self.verify()
    def test_true_measurement_relabel_rejected_even_with_new_result_sha(self):
        path=self.root/self.fixture.w.RESULT_PATH;value=a.tx.read_json(path)
        value['prefix_analyses']['11031']['candidate_gate_passed']=True;dump(path,value)
        self.binding['prefix_continuation']=ref(self.root,path)
        with self.assertRaises(ValueError):self.verify()
    def test_extra_unknown_attempt_or_technical_hold_cannot_be_waived(self):
        path=self.root/'runs/PREFIX_CALIB_11032/HOLD.json';dump(path,{})
        with self.assertRaises(ValueError):self.verify()
        path.unlink();dump(path.parent/('0'*64+'.BEGIN.json'),{})
        with self.assertRaises(ValueError):self.verify()

    def test_fresh_process_new_release_analysis_with_pinned_old_prefix_verifier(self):
        release=self.root/'future_execution_release';release.mkdir()
        for name in ('execution','runtime','analysis'):
            shutil.copytree(PACKAGE/name,release/name,ignore=shutil.ignore_patterns('__pycache__'))
        for relative in ('supplied_package/tools/satdose_core.py','supplied_package/tools/check_admission.py',
                         'supplied_package/config/source_bindings.json'):
            p=release/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PACKAGE/relative,p)
        paths=dict(authority=release/'execution/admission_authority.py',
            check_admission=release/'supplied_package/tools/check_admission.py',
            satdose_core=release/'supplied_package/tools/satdose_core.py',
            scorer_bindings=release/'supplied_package/config/source_bindings.json',
            calibration=release/'analysis/calibration.py',
            prefix_waiver=PACKAGE/'execution/prefix_waiver.py',
            gpu_auditor=PACKAGE/'execution/gpu_evidence.py')
        sources={key:dict(path=str(path),sha256=a.tx.file_sha(path)) for key,path in paths.items()}
        payload=self.root/'namespace_fixture.json';dump(payload,dict(root=str(self.root),binding=self.binding,sources=sources))
        script='''
import json,sys,pathlib
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
from execution import admission_authority as a
d=json.loads(pathlib.Path(sys.argv[2]).read_text());root=pathlib.Path(d['root'])
original,waiver=a._sources({'sources':d['sources']},a._Observed(root))
assert pathlib.Path(waiver.__file__)==pathlib.Path(d['sources']['prefix_waiver']['path'])
assert pathlib.Path(waiver.calibration.__file__)==pathlib.Path(d['sources']['calibration']['path'])
result=a._verified_prefix(root,d['binding'],d['binding']['prefix_waiver'],waiver)
assert result['responses']==96 and result['candidate_gate_passed']['11031'] is False
print('CROSS_RELEASE_REAL_PREFIX_VERIFIER_PASS')
'''
        result=subprocess.run([sys.executable,'-B','-c',script,str(release),str(payload)],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('CROSS_RELEASE_REAL_PREFIX_VERIFIER_PASS',result.stdout)

if __name__=='__main__':unittest.main()
