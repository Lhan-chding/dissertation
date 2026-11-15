"""Real CPU canary replay + bounded continuation authority regression tests.

GPU fingerprints/controller records come from explicitly synthetic CPU fixtures;
no production GPU acceptance, process exit or permission is asserted by tests.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import continuation_authority as c, admission_authority as a
sys.path.pop(0)
spec=importlib.util.spec_from_file_location('_continuation_canary_fixture',Path(__file__).with_name('test_canary_audit.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
t=c.tx


@unittest.skipUnless((f.f.fixtures.NATIVE/'sr_f3_ce/training.py').exists(),'Pinned actual CPU native optimizer required')
class ActualCanaryAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f.CanaryAuditTests.setUpClass();cls.addClassCleanup(f.CanaryAuditTests.doClassCleanups)
        fixture=f.CanaryAuditTests('test_complete_synthetic_contract_actual_cpu_states_and_no_side_effects')
        cls.fixture=fixture;cls.addClassCleanup(fixture.doCleanups)
        original=fixture.make_path
        def make_path(*args):
            fixture.sources['continuation_verifier']=dict(path=str(Path(c.__file__).resolve()),sha256=t.file_sha(c.__file__))
            return original(*args)
        fixture.make_path=make_path;fixture.setUp()
        root=fixture.root;report=fixture.audit();f.dump(root/'INDEPENDENT_CANARY.json',report)
        cls.spec=dict(root=str(root),helper=fixture.sources['continuation_verifier'],
            auditor=dict(path=str(Path(c.canary_audit.__file__).resolve()),sha256=fixture.auditor_sha),
            report=fixture.ref(root/'INDEPENDENT_CANARY.json'),trusted_specs=fixture.trusted,
            frozen_witnesses=fixture.frozen,process_exit_proofs=fixture.exits,
            design_freeze=fixture.ref(fixture.case.freeze),config=fixture.ref(root/'config.json'),execution_sources=fixture.sources)
        candidate=c.prepare_canary(cls.spec);f.dump(root/'CANARY_RESUME_PASS.json',candidate)
        cls.spec=dict(cls.spec,published=fixture.ref(root/'CANARY_RESUME_PASS.json'))
    def test_actual_four_path_cpu_audit_and_published_candidate_match(self):
        verified=c.verify_canary(self.spec);verified.stable()
        self.assertEqual(set(verified.identities),c.PATHS)
        self.assertEqual(verified.receipt,self.spec['published'])
        self.assertFalse(t.read_json(t.bound(self.fixture.root,verified.receipt))['main_authorized'])
    def test_mutated_observed_teacher_bytes_rejected(self):
        verified=c.verify_canary(self.spec)
        name=next(p for p in verified.report['observed_files'] if p.endswith('TENSORS.pt')) if any(
            p.endswith('TENSORS.pt') for p in verified.report['observed_files']) else next(
            p for p in verified.report['observed_files'] if p.endswith('.pt') and str(self.fixture.root) in p)
        path=Path(name);old=path.read_bytes();self.addCleanup(path.write_bytes,old);path.write_bytes(b'changed')
        with self.assertRaises(PermissionError):verified.stable()
    def test_missing_proof_family_extra_path_and_source_change_rejected(self):
        for change in ('missing','extra','source'):
            spec=copy.deepcopy(self.spec)
            if change=='missing':spec['process_exit_proofs'].pop('11031/T0')
            elif change=='extra':spec['trusted_specs']['11033/T0']=spec['trusted_specs']['11031/T0']
            else:spec['helper']['sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(PermissionError):c.verify_canary(spec)
    def test_true_audit_cannot_be_replaced_by_pass_label(self):
        root=self.fixture.root;path=root/'FAKE_AUDIT.json';f.dump(path,{'status':'PASS'})
        spec=copy.deepcopy(self.spec);spec['report']=self.fixture.ref(path)
        with self.assertRaises(PermissionError):c.verify_canary(spec)
    def test_published_canary_must_equal_exact_candidate(self):
        root=self.fixture.root;path=root/'FAKE_CANARY.json'
        value=t.read_json(t.bound(root,self.spec['published']));value['permission']='MAIN_AND_UNLIMITED_TRAIN'
        f.dump(path,value);spec=copy.deepcopy(self.spec);spec['published']=self.fixture.ref(path)
        with self.assertRaises(PermissionError):c.verify_canary(spec)
    def test_known_negative_policy_delta_fails_even_if_report_says_verified(self):
        spec=copy.deepcopy(self.spec);report=copy.deepcopy(c.verify_canary(spec).report)
        report['paths']['11031/T0']['actual_nonzero_policy_deltas'][0]['parameter_update_norm']=0
        with self.assertRaises(PermissionError):c._candidate(spec,a._Observed(self.fixture.root),report)
    def test_new_canary_hold_or_unknown_phase_rejected_before_and_after_cache(self):
        verified=c.verify_canary(self.spec);root=self.fixture.root
        path=root/'canary_replay/start-11031/T0/baseline/HOLD.json';f.dump(path,{})
        try:
            with self.assertRaises(PermissionError):verified.stable()
            with self.assertRaises(PermissionError):c.verify_canary(self.spec)
        finally:path.unlink()
        path=root/'canary_replay/start-11032/T1/retry';path.mkdir()
        try:
            with self.assertRaises(PermissionError):verified.stable()
            with self.assertRaises(PermissionError):c.verify_canary(self.spec)
        finally:path.rmdir()
    def test_preexisting_unknown_begin_first_two_slot_and_other_control_hold_rejected(self):
        root=self.fixture.root
        begin=root/'canary_replay/start-11031/T0/baseline/new.BEGIN.json';f.dump(begin,{})
        try:
            with self.assertRaises(PermissionError):c.verify_canary(self.spec)
        finally:begin.unlink()
        slot=root/'first_two_control/start-11031/T0/baseline';slot.mkdir(parents=True);f.dump(slot/'CLAIM.json',{})
        try:
            with self.assertRaises(PermissionError):c.verify_canary(self.spec)
        finally:
            import shutil
            shutil.rmtree(root/'first_two_control')
        verified=c.verify_canary(self.spec);path=root/'continuation_control/start-11031/T1/HOLD.json';f.dump(path,{})
        try:
            with self.assertRaises(PermissionError):verified.stable()
            with self.assertRaises(PermissionError):c.verify_canary(self.spec)
        finally:
            import shutil
            shutil.rmtree(root/'continuation_control')


class ContinuationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.identity=dict(train_seed=11031,arm='T0',experiment_id=a.EXPERIMENT)
        self.observed=a._Observed(self.root);self.refs={};self.identities={}
        for key in c.PATHS:
            start,arm=key.split('/');identity=dict(train_seed=int(start),arm=arm,experiment_id=a.EXPERIMENT)
            self.identities[key]=identity;self.refs[key]=[]
            for step in (1,2):
                path=self.root/f'training_transactions/start-{start}/{arm}/step-{step:03d}/COMMIT.json'
                f.dump(path,dict(identity=identity,step=step,policy_sha256=str(step)*64,reference_sha256='r'*64))
                self.refs[key].append(t.reference(self.root,path));self.observed.absolute(path)
        v=c.ContinuationVerifier.__new__(c.ContinuationVerifier);self.v=v
        v.root=self.root;v._risk_ready=True
        v.original=SimpleNamespace(_observed=self.observed,_binding={'design_freeze':{'sha256':'f'*64}},
            _spec={'documents':{'registration':{},'gpu_report':{}}},_admission={})
        v.canary=SimpleNamespace(stable=self.observed.stable,identities=self.identities,receipt={'CPU':'canary'})
        v._risk_observed=self.observed;v._risk_inputs={'trusted_commits':self.refs}
        v._risk_result=dict(status='NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED',global_hold=False,pending_required_dev=[])
        self.arguments=dict(root=self.root,identity=self.identity,design_freeze_sha256='f'*64,target_step=3,
            sampled_endpoint=dict(experiment_id=a.EXPERIMENT,start_seed=11031,arm='T0',step=2,
                                  policy_sha256='2'*64,reference_sha256='r'*64))
    def test_step3_uses_actual_current_marker_without_lock_or_risk_call(self):
        from analysis import risk
        with t.exclusive_lock(self.root/'training_transactions'),patch.object(risk,'assess',side_effect=AssertionError('nested risk call')):
            result=self.v.verify(**self.arguments)
        self.assertEqual(result['target_step'],3)
        self.assertEqual(result['canary'],{'CPU':'canary'})
    def test_first_two_step25_bool_and_foreign_endpoint_rejected(self):
        for step in (1,2,25,True):
            with self.subTest(step=step),self.assertRaises(PermissionError):self.v.verify(**dict(self.arguments,target_step=step))
        with self.assertRaises(PermissionError):self.v.verify(**dict(self.arguments,identity=dict(self.identity,arm='MAIN')))
    def test_changed_cursor_or_committed_policy_rejected(self):
        path=self.root/'training_transactions/start-11032/T1/step-003/COMMIT.json';f.dump(path,{})
        with self.assertRaisesRegex(PermissionError,'cursor'):self.v.verify(**self.arguments)
    def test_actual_endpoint_must_equal_current_policy(self):
        endpoint=dict(self.arguments['sampled_endpoint'],policy_sha256='x'*64)
        with self.assertRaises(PermissionError):self.v.verify(**dict(self.arguments,sampled_endpoint=endpoint))
    def test_global_scientific_hold_and_failed_refresh_are_fail_closed(self):
        self.v._risk_result.update(status='GLOBAL_SCIENTIFIC_HOLD',global_hold=True)
        with self.assertRaises(PermissionError):self.v.verify(**self.arguments)
        self.v._risk_ready=False
        with self.assertRaisesRegex(PermissionError,'cached fallback'):self.v.verify(**self.arguments)
    def test_pending_dev2_blocks3_pending_dev8_allows_remaining_current_chunk(self):
        self.v._risk_result.update(status='PENDING_REQUIRED_DEV',pending_required_dev=['11032/T1/2'])
        with self.assertRaisesRegex(PermissionError,'DEV barrier'):self.v.verify(**self.arguments)
        self.v._risk_result['pending_required_dev']=['11032/T1/8']
        self.assertEqual(self.v.verify(**self.arguments)['target_step'],3)
    def test_same_file_facade_keeps_first_two_verifier_scope_separate(self):
        import inspect
        self.assertEqual(Path(inspect.getsourcefile(a.ContinuingAdmission.verify)).resolve(),Path(a.__file__).resolve())
        self.assertNotEqual(a.ContinuingAdmission.verify,a.FirstTwoStepAdmission.verify)
    def test_facade_rejects_runtime_replacement_of_pinned_helper_method(self):
        authority=a.ContinuingAdmission.__new__(a.ContinuingAdmission)
        authority.verifier_reference=dict(path=str(Path(a.__file__).resolve()),sha256=t.file_sha(a.__file__))
        authority._continuation=self.v
        self.v.spec={'helper':dict(path=str(Path(c.__file__).resolve()),sha256=t.file_sha(c.__file__))}
        self.assertEqual(authority.verify(**self.arguments)['target_step'],3)
        self.v.verify=lambda **kwargs:dict(status='PASS')
        with self.assertRaisesRegex(PermissionError,'method replaced'):authority.verify(**self.arguments)
    def risk_inputs(self):
        from analysis import risk
        source=dict(path=str(Path(risk.__file__).resolve()),sha256=t.file_sha(risk.__file__))
        config={'path':'config.json','sha256':'c'*64};freeze={'path':'DESIGN_FREEZE.json','sha256':'f'*64}
        self.v.sources={'risk_verifier':source};self.v.spec={'canary_spec':{'design_freeze':freeze}}
        self.v.original._spec['documents']['config']=config;self.v._risk_fixed=None
        return dict(config_ref=config,freeze_ref=freeze,clarification_ref={'path':'clarification.json','sha256':'d'*64},
            trusted_commits=self.refs,dev_endpoints={},expected_source_sha256=source['sha256'])
    def test_refresh_returns_real_report_copy_and_failure_poisons_old_cache(self):
        from analysis import risk
        inputs=self.risk_inputs();report=dict(self.v._risk_result,permission='RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY',
            observed_files={str(path):sha for path,sha in self.observed.files.items()})
        with patch.object(risk,'assess',return_value=report) as assess:
            returned=self.v.refresh_risk(inputs);self.assertEqual(assess.call_count,1)
            returned['global_hold']=True;self.assertFalse(self.v._risk_result['global_hold'])
        with patch.object(risk,'assess',side_effect=PermissionError('CPU synthetic missing evidence')):
            with self.assertRaises(PermissionError):self.v.refresh_risk(inputs)
        with self.assertRaisesRegex(PermissionError,'cached fallback'):self.v.verify(**self.arguments)
    def test_risk_clarification_cannot_change_between_steps(self):
        from analysis import risk
        inputs=self.risk_inputs();report=dict(self.v._risk_result,permission='RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY',
            observed_files={str(path):sha for path,sha in self.observed.files.items()})
        with patch.object(risk,'assess',return_value=report):self.v.refresh_risk(inputs)
        changed=copy.deepcopy(inputs);changed['clarification_ref']['sha256']='e'*64
        with patch.object(risk,'assess',side_effect=AssertionError('changed clarification reached assessor')):
            with self.assertRaisesRegex(PermissionError,'threshold inputs changed'):self.v.refresh_risk(changed)


if __name__=='__main__':unittest.main()
