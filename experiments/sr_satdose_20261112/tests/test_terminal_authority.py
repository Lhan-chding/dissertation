"""Standalone authority boundary fixtures, never actual training certification."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]


class TerminalAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.external=self.root/'external';self.external.mkdir();self.original=self.root/'original';self.original.mkdir()
        source=self.external/'terminal_authority.py';source.write_bytes((PACKAGE/'execution/terminal_authority.py').read_bytes())
        name='satdose_terminal_authority_fixture_'+str(id(self));spec=importlib.util.spec_from_file_location(name,source)
        self.a=importlib.util.module_from_spec(spec);sys.modules[name]=self.a;spec.loader.exec_module(self.a)
        self.addCleanup(lambda:sys.modules.pop(name,None));a=self.a
        (self.original/'fixture.py').write_text('# Synthetic original inventory, no model execution\n')
        self.write('DESIGN_FREEZE.json',{'fixture':True});self.write('evidence/admission.json',{'fixture':True})
        self.write('evidence/admission_spec.json',{'fixture':True});self.write('evidence/canary_spec.json',{'fixture':True})
        self.write('evidence/final_risk.json',{'fixture':True})
        original=dict(schema='sr-satdose-first-two-release-v1',status='AUDITED_FIRST_TWO_AND_READONLY_RELOAD_ONLY',
            experiment_id=a.EXPERIMENT,evidence_root=str(self.root),release_root=str(self.original),sources=a.inventory(self.original),
            design_freeze_sha256=a.sha(self.root/'DESIGN_FREEZE.json'),config_sha256='c'*64)
        self.write('original_release.json',original)
        auditor=self.external/'terminal_audit.py'
        auditor.write_text('import json\nfrom pathlib import Path\ndef audit(spec):\n    return json.loads(Path(spec["fixture_report"]).read_text())\n')
        self.audit_spec=dict(root=str(self.root),auditor=self.absref(auditor),original_train_release=self.ref('original_release.json'),
            fixture_report=str(self.root/'report.json'),trusted_commits={})
        for name in ('admission_spec','admission','canary_spec','final_risk'):self.audit_spec[name]=self.ref('evidence/'+name+'.json')
        for key in a.PATHS:
            refs=[]
            for step in range(1,25):
                name=f'evidence/{key}/{step}.json';self.write(name,{'fixture':True,'step':step});refs.append(self.ref(name))
            self.audit_spec['trusted_commits'][key]=refs
        self.write('audit_spec.json',self.audit_spec)
        self.model=self.root/'models/snapshots/frozen';self.model.mkdir(parents=True)
        target=self.root/'models/blobs'/('a'*64);target.parent.mkdir();target.write_bytes(b'fixed fixture model bytes')
        (self.model/'weights').symlink_to('../../blobs/'+target.name)
        files={str(p):a.sha(p) for p in (self.root/'evidence').rglob('*') if p.is_file()}
        files[str(target)]=a.sha(target)
        self.report=dict(schema='sr-satdose-terminal-independent-audit-v1',status='TERMINAL_TRAINING_EVIDENCE_AUDITED_NOT_MAIN_ADMISSION',
            permission='NO_TRAINING_OR_MAIN',auditor=self.audit_spec['auditor'],original_train_release=self.ref('original_release.json'),
            spec_sha256=a.object_sha(self.audit_spec),formal_optimizer_steps=96,train_responses=12288,dev_endpoints=14,dev_responses=1512,
            terminal_steps={key:24 for key in a.PATHS},global_hold=False,pending_required_dev=[],observed_files=files,
            source_and_input_bytes_reverified=True,audit_model_calls=0,audit_optimizer_calls=0,
            observed_directories={str(self.root/'evidence'):sorted(str(p.relative_to(self.root/'evidence')) for p in (self.root/'evidence').rglob('*'))},
            observed_model_links={str(self.model/'weights'):dict(link='../../blobs/'+target.name,target=str(target))},
            observed_model_directories={str(self.model):['weights']})
        self.write('report.json',self.report)
        self.spec=dict(schema='sr-satdose-terminal-authority-spec-v1',root=str(self.root),authority=self.absref(source),
            auditor=self.absref(auditor),external_root=str(self.external),external_sources=a.inventory(self.external),
            original_train_release=self.ref('original_release.json'),audit_spec=self.ref('audit_spec.json'),report=self.ref('report.json'))
        p=patch.object(a,'import_original',return_value={});self.import_mock=p.start();self.addCleanup(p.stop)
    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    def ref(self,name):return dict(path=name,sha256=self.a.sha(self.root/name))
    def absref(self,path):return dict(path=str(path),sha256=self.a.sha(path))
    def publish(self):
        candidate=self.a.prepare(self.spec);self.write('terminal.json',candidate)
        return dict(self.spec,published=self.ref('terminal.json'))
    def update_report(self):self.write('report.json',self.report);self.spec['report']=self.ref('report.json')

    def test_actual_authenticated_external_bytes_replay_then_exact_candidate_verification(self):
        spec=self.publish();verified=self.a.verify(spec)
        self.assertEqual(verified.receipt,spec['published']);self.assertEqual(verified.package,self.original)
        self.assertEqual(self.a.read(self.root/'terminal.json')['main_response_count'],13760)
        verified.stable()

    def test_partial_terminal_and_pending_or_hold_never_admit(self):
        for change in ({'formal_optimizer_steps':95},{'train_responses':12287},{'dev_endpoints':13},
                       {'dev_responses':1511},{'global_hold':True},{'pending_required_dev':['11031/T0/16']},
                       {'terminal_steps':{key:23 for key in self.a.PATHS}}):
            with self.subTest(change=change):
                prior=copy.deepcopy(self.report);self.report.update(change);self.update_report()
                with self.assertRaisesRegex(PermissionError,'budgets or risk'):self.a.prepare(self.spec)
                self.report=prior;self.update_report()

    def test_forged_published_status_is_rejected(self):
        spec=self.publish();value=self.a.read(self.root/'terminal.json');value['status']='PASS';self.write('terminal.json',value)
        spec['published']=self.ref('terminal.json')
        with self.assertRaisesRegex(PermissionError,'Published terminal admission'):self.a.verify(spec)

    def test_original_library_changed_or_added_is_rejected_before_import(self):
        (self.original/'fixture.py').write_text('# changed')
        with self.assertRaisesRegex(PermissionError,'inventory'):self.a.prepare(self.spec)
        self.import_mock.assert_not_called()

    def test_external_source_changed_is_rejected_before_original_import(self):
        (self.external/'extra.py').write_text('# added')
        with self.assertRaisesRegex(PermissionError,'External source inventory'):self.a.prepare(self.spec)
        self.import_mock.assert_not_called()

    def test_missing_independent_commit_observation_rejected(self):
        reference=self.audit_spec['trusted_commits']['11032/T1'][-1]
        self.report['observed_files'].pop(str(self.root/reference['path']));self.update_report()
        with self.assertRaisesRegex(PermissionError,'COMMIT not independently audited'):self.a.prepare(self.spec)

    def test_observed_bytes_and_added_unknown_file_invalidate_cached_terminal(self):
        verified=self.a.verify(self.publish())
        self.write('evidence/new_unknown.json',{})
        with self.assertRaisesRegex(PermissionError,'inventory differs'):verified.stable()
        (self.root/'evidence/new_unknown.json').unlink();self.write('evidence/admission.json',{'changed':True})
        with self.assertRaisesRegex(PermissionError,'Terminal evidence changed'):verified.stable()

    def test_cached_link_repoint_even_same_bytes_is_rejected(self):
        verified=self.a.verify(self.publish());link=self.model/'weights';old=link.resolve()
        other=old.parent/('b'*64);other.write_bytes(old.read_bytes());link.unlink();link.symlink_to('../../blobs/'+other.name)
        with self.assertRaisesRegex(PermissionError,'model link changed'):verified.stable()

    def test_model_inventory_addition_rejected(self):
        verified=self.a.verify(self.publish());(self.model/'extra').write_bytes(b'x')
        with self.assertRaisesRegex(PermissionError,'model inventory changed'):verified.stable()

    def test_new_global_hold_after_publication_blocks_initial_cached_verification(self):
        spec=self.publish();self.write('training_control/HOLD.json',{})
        with self.assertRaisesRegex(PermissionError,'Global terminal HOLD'):self.a.verify(spec)

    def test_audit_report_spec_and_source_identity_are_bound(self):
        self.report['spec_sha256']='a'*64;self.update_report()
        with self.assertRaisesRegex(PermissionError,'independent terminal audit'):self.a.prepare(self.spec)

    def test_actual_auditor_reexecution_must_match_report(self):
        with patch.object(self.a,'load_source',return_value=SimpleNamespace(audit=lambda spec:{'changed':True})):
            with self.assertRaisesRegex(PermissionError,'Fresh independent terminal audit'):self.a.prepare(self.spec)

    def test_import_original_rejects_copied_namespace_before_importing_any_module(self):
        other=self.root/'copied';other.mkdir();fake=SimpleNamespace(__file__=str(other/'evaluation_runtime.py'))
        # Bypass the fixture-only import mock to exercise the authenticated loader.
        loaded=importlib.util.spec_from_file_location('terminal_loader_boundary',self.external/'terminal_authority.py')
        module=importlib.util.module_from_spec(loaded);loaded.loader.exec_module(module)
        with patch.dict(sys.modules,{'execution.evaluation_runtime':fake}),patch.object(module.importlib,'import_module') as importer:
            with self.assertRaisesRegex(PermissionError,'Shadowed original'):module.import_original(self.original)
            importer.assert_not_called()


if __name__=='__main__':unittest.main()
