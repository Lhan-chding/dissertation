"""Offline normal-entrypoint guards. Synthetic fixtures are NOT actual admission.

No SSH/native risk/model execution. The only actual job fixture (1742958) is
used to prove exclusion from the normal path. All positive submits are synthetic.
"""
import base64
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

OPERATIONS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPERATIONS))
import satdose_audit_continuation_v2 as v


class Dependencies(unittest.TestCase):
    def setUp(self): self.dependencies = v.local_dependencies()

    def test_sources_exact_pins_and_no_remote_source_file_needed(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('No dependency filesystem reads')):
            a, helper, raw = v.load_modules(self.dependencies)
        self.assertEqual(a.SUBMITTER_SHA, 'eceec5aab4de56b53d8e1483d2e49093e28f4048b2ef818302bb3db499fb0f94')
        self.assertEqual(helper.ORIGINAL_SHA, v.ORIGINAL_SHA)
        self.assertEqual(a.digest(raw['namespace_verifier']), v.HELPER_SHA)

    def test_missing_extra_mutated_or_foreign_dependency_rejected(self):
        for mutation in ('missing', 'extra', 'source', 'sha', 'bytes'):
            value = copy.deepcopy(self.dependencies)
            if mutation == 'missing': value.pop('namespace_verifier')
            if mutation == 'extra': value['extra'] = value['original_auditor']
            if mutation == 'source': value['original_auditor']['source_b64'] = base64.b64encode(b'forged').decode()
            if mutation == 'sha': value['namespace_verifier']['sha256'] = 'f'*64
            if mutation == 'bytes': value['namespace_verifier']['bytes'] = True
            with self.assertRaises(ValueError): v.load_modules(value)

    def test_original_full_audit_and_main_functions_unchanged(self):
        a, helper, raw = v.load_modules(self.dependencies)
        audit = a.audit_remote; main = a.main
        v.configure(a, helper, raw)
        self.assertIs(a.audit_remote, audit); self.assertIs(a.main, main)
        self.assertTrue({'ORIGINAL_AUDITOR_SOURCE.py','NAMESPACE_VERIFIER_SOURCE.py'} <= a.SUCCESS_FILES)


class AttemptGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='satdose-normal-v2-OFFLINE-'); self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name).resolve(); self.root = self.project/'prepared'; self.root.mkdir()
        self.dependencies = v.local_dependencies(); self.a, helper, raw = v.load_modules(self.dependencies)
        v.configure(self.a,helper,raw);self.a.ROOT=self.root
        self.s = self.a.selection(11032,'T0',9)
        self.payload=dict(job_id=9999999,selection=self.s,operator_dependencies=self.dependencies,
                          operation_source_sha256=self.a.digest(b'OFFLINE_SOURCE'))

    def make(self, family, selection=None):
        s = selection or self.s
        path=self.a.ROOT.parent/family/f"start-{s['start_seed']}"/s['arm']/f"chunk-{s['first_step']:03d}-{s['last_step']:03d}"
        path.mkdir(parents=True);return path

    def test_new_normal_attempt_permitted_as_plan_guard(self): v.no_previous_attempt(self.a,self.payload)

    def test_known_recovery_job_hard_excluded(self):
        payload=dict(self.payload,job_id=1742958)
        with self.assertRaisesRegex(ValueError,'dedicated'):v.no_previous_attempt(self.a,payload)

    def test_recovery_payload_is_not_normal_admission(self):
        with self.assertRaises(ValueError):v.no_previous_attempt(self.a,dict(self.payload,known_failure={}))

    def test_any_previous_selected_slot_blocks_regardless_of_result(self):
        for family in ('continuation_audits_71018e5','continuation_audits_namespace_recovery_v1_1742958',
                       'continuation_audits_namespace_v2','continuation_audits_unknown_future'):
            with self.subTest(family=family):
                case=self.project/('case-'+family);case.mkdir();self.a.ROOT=case/'prepared';self.a.ROOT.mkdir()
                path=self.make(family)
                with self.assertRaisesRegex(ValueError,'Existing selected-chunk'):v.no_previous_attempt(self.a,self.payload)
                (path/'ERROR.json').write_bytes(b'{}')
                with self.assertRaises(ValueError):v.no_previous_attempt(self.a,self.payload)
                (path/'PUBLICATION.json').write_bytes(b'{}')
                with self.assertRaises(ValueError):v.no_previous_attempt(self.a,self.payload)
                # Each family is tested in isolation; keep all fixture evidence.

    def test_other_selected_chunks_do_not_block(self):
        self.make('continuation_audits_71018e5',self.a.selection(11031,'T0',3))
        self.make('continuation_audits_namespace_v2',self.a.selection(11032,'T1',9))
        v.no_previous_attempt(self.a,self.payload)

    def test_symlink_family_and_selected_ancestor_rejected(self):
        target=self.project/'unrelated';target.mkdir()
        link=self.project/'continuation_audits_symlink';link.symlink_to(target,target_is_directory=True)
        with self.assertRaises(ValueError):v.no_previous_attempt(self.a,self.payload)
        link.unlink();family=self.project/'continuation_audits_71018e5';family.mkdir()
        (family/'start-11032').symlink_to(target,target_is_directory=True)
        with self.assertRaises(ValueError):v.no_previous_attempt(self.a,self.payload)

    def test_only_current_created_slot_with_exact_source_intent_allowed_inside_audit(self):
        path=self.make(v.REMOTE_FAMILY)
        self.a.write_once(path/'OPERATION_SOURCE.py',b'OFFLINE_SOURCE')
        self.a.write_once(path/'INTENT.json',dict(payload=self.payload))
        with self.assertRaises(ValueError):v.no_previous_attempt(self.a,self.payload)
        v.no_previous_attempt(self.a,self.payload,current_created=True)
        (path/'OPERATION_SOURCE.py').write_bytes(b'changed')
        with self.assertRaises(ValueError):v.no_previous_attempt(self.a,self.payload,current_created=True)

    def test_old_slot_never_allowed_by_current_created_flag(self):
        self.make('continuation_audits_71018e5')
        with self.assertRaises(ValueError):v.no_previous_attempt(self.a,self.payload,current_created=True)


class LocalDrivers(unittest.TestCase):
    def setUp(self):
        fixtures=runpy.run_path(str(OPERATIONS/'tests/test_audit_continuation.py'))
        self.fixture=fixtures['OriginalReceiptChain']();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.directory=self.fixture.directory
        dependencies=v.local_dependencies();self.a,self.helper,self.raw=v.load_modules(dependencies)
        v.configure(self.a,self.helper,self.raw,local=True)
        self.payload=self.a.local_payload(self.args())
        self.payload['operation_source_sha256']=self.a.sha(Path(v.__file__))

    def args(self):return self.fixture.args()

    def argv(self):
        args=self.args()
        return ['--submission-receipt',str(args.submission_receipt),'--submission-sha256',args.submission_sha256,
            '--job-id',str(args.job_id),'--start-seed',str(args.start_seed),'--arm',args.arm,'--first-step',str(args.first_step)]

    def test_synthetic_local_plan_zero_subprocess_and_no_artifacts(self):
        before=set(self.directory.iterdir())
        with patch.object(self.a.subprocess,'run',side_effect=AssertionError('SSH forbidden')),redirect_stdout(io.StringIO()) as output:
            self.assertEqual(v.main(self.argv()),0)
        value=self.a.loads(output.getvalue());self.assertIn(v.REMOTE_FAMILY,value['remote_directory'])
        self.assertEqual(set(self.directory.iterdir()),before)

    def test_actual_1742958_submit_rejected_locally_without_ssh(self):
        base=OPERATIONS.parents[2]/'artifacts/sr_satdose_20261112/continuation_20261117/start-11031_T0_chunk-003-008'
        argv=['--submission-receipt',str(base/'SUBMISSION.json'),
            '--submission-sha256','ef21d0d78b17338a93c7abf6c262e7835390d18437e811626ad4ac8c766388a9',
            '--job-id','1742958','--start-seed','11031','--arm','T0','--first-step','3']
        with patch.object(self.a.subprocess,'run',side_effect=AssertionError('SSH forbidden')):
            with self.assertRaisesRegex(ValueError,'dedicated'):v.main(argv)

    def manifest(self):
        directory=self.a.audit_for(self.payload['selection'])
        files={name:dict(path=str(directory/name),sha256='a'*64,bytes=1) for name in self.a.SUCCESS_FILES}
        for name,raw in [('OPERATION_SOURCE.py',Path(v.__file__).read_bytes()),
                         ('ORIGINAL_AUDITOR_SOURCE.py',self.raw['original_auditor']),
                         ('NAMESPACE_VERIFIER_SOURCE.py',self.raw['namespace_verifier'])]:
            files[name].update(sha256=self.a.digest(raw),bytes=len(raw))
        return dict(schema='sr-satdose-continuation-audit-manifest-v1',success=True,
            status='CONTINUATION_AUDIT_PUBLICATION_COMPLETE',permission=self.a.PERMISSION,
            job_id=self.payload['job_id'],selection=self.payload['selection'],
            payload_sha256=self.a.digest(self.a.canonical(self.payload)),
            operation_source_sha256=self.payload['operation_source_sha256'],next_stage_authorized=False,files=files,
            bundle=dict(path=str(directory/'BUNDLE.tar.gz'),sha256='a'*64,bytes=1))

    def test_success_manifest_requires_both_exact_archived_dependency_sources(self):
        value=self.manifest();self.a.validate_manifest(value,self.payload)
        for name in ('ORIGINAL_AUDITOR_SOURCE.py','NAMESPACE_VERIFIER_SOURCE.py'):
            bad=copy.deepcopy(value);bad['files'].pop(name)
            with self.assertRaises(ValueError):self.a.validate_manifest(bad,self.payload)
            bad=copy.deepcopy(value);bad['files'][name]['sha256']='f'*64
            with self.assertRaises(ValueError):self.a.validate_manifest(bad,self.payload)

    def test_download_only_never_audits_and_transports_both_sources_on_stdin(self):
        manifest=self.manifest();saved=self.directory/'V2_MANIFEST.json';saved.write_bytes(self.a.canonical(manifest))
        output=self.directory/'download';calls=[]
        def ssh(argv,**kwargs):
            calls.append(argv);self.assertIn('--remote-download',argv[-1]);self.assertNotIn('--remote-worker',argv[-1])
            self.assertLess(len(argv[-1]),1500)
            envelope=self.a.loads(kwargs['input']);payload=envelope['data']['payload']
            self.assertEqual(v.dependency_bytes(payload['operator_dependencies']),self.raw)
            self.assertEqual(base64.b64decode(envelope['source_b64']),Path(v.__file__).read_bytes())
            kwargs['stdout'].write(b'PARTIAL OFFLINE BUNDLE');kwargs['stderr'].write(b'OFFLINE FAILURE')
            return type('Response',(),{'returncode':255})()
        with patch.object(self.a.subprocess,'run',side_effect=ssh),redirect_stdout(io.StringIO()):
            self.assertEqual(v.main(self.argv()+['--recover-manifest',str(saved),'--recover-sha256',self.a.sha(saved),
                '--local-dir',str(output),'--execute']),2)
        self.assertEqual(len(calls),1)
        self.assertEqual((output/'BUNDLE.tar.gz').read_bytes(),b'PARTIAL OFFLINE BUNDLE')
        self.assertTrue(self.a.loads((output/'LOCAL_RECEIPT.json').read_bytes())['remote_audit_success'])

    def test_changed_actual_job_or_selection_rejected_by_original_validator(self):
        argv=self.argv();argv[argv.index('--job-id')+1]='9999998'
        with self.assertRaises(ValueError):v.main(argv)


if __name__=='__main__':unittest.main()
