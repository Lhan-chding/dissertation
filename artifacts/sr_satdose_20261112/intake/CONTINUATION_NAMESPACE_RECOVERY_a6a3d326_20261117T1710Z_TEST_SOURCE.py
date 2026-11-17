"""Offline regression and actual downloaded-failure intake checks; no SSH/model.

The actual package tests only validate local already-downloaded bytes. Original
native full risk replay remains future server work and is not simulated as PASS.
"""
from contextlib import redirect_stdout
import copy
import importlib.machinery
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

OPERATIONS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPERATIONS))
import satdose_recover_continuation_namespace as r

FAILED_DIR = OPERATIONS.parents[2] / 'artifacts/sr_satdose_20261112/continuation_20261117/start-11031_T0_chunk-003-008/independent_audit'
SUBMIT = FAILED_DIR.parent / 'SUBMISSION.json'
SUBMIT_SHA = 'ef21d0d78b17338a93c7abf6c262e7835390d18437e811626ad4ac8c766388a9'


class NamespaceRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='satdose-namespace-OFFLINE-'); self.addCleanup(self.temp.cleanup)
        self.code = Path(self.temp.name).resolve(); self.a, self.original_source = r.load_original()
        self.modules = {}; self.sources = {}
        for name in ('execution', 'analysis', 'runtime'):
            directory = self.code / name; directory.mkdir()
            path = directory / 'example.py'; path.write_bytes(b'# OFFLINE inert source\n')
            self.sources[name + '/example.py'] = self.a.sha(path)
            spec = importlib.machinery.PathFinder.find_spec(name, [str(self.code)])
            module = importlib.util.module_from_spec(spec)
            self.modules[name] = module
        ordinary = ModuleType('execution.example'); ordinary.__file__ = str(self.code / 'execution/example.py')
        self.modules['execution.example'] = ordinary

    def verify(self):
        r.verify_modules_pinned(self.sources, code=self.code, modules=self.modules, ordinary=self.a.ordinary, sha=self.a.sha)

    def test_reproduces_original_path_none_bug_and_recovers_real_namespace(self):
        self.assertIsNone(self.modules['execution'].__file__)
        with patch.object(self.a, 'CODE', self.code), patch.object(self.a, 'sys', SimpleNamespace(modules=self.modules)):
            with self.assertRaises(TypeError): self.a.modules_pinned(self.sources)
        self.verify()

    def test_none_submodule_never_exempt(self):
        self.modules['execution.example'].__file__ = None
        with self.assertRaisesRegex(ValueError, 'Missing ordinary module'): self.verify()

    def test_missing_regular_module_rejected(self):
        self.modules['execution.example'].__file__ = str(self.code / 'execution/missing.py')
        with self.assertRaises((ValueError, FileNotFoundError)): self.verify()

    def test_foreign_regular_module_rejected(self):
        path = self.code / 'foreign.py'; path.write_bytes(b'# other\n')
        self.modules['execution.example'].__file__ = str(path)
        with self.assertRaises(ValueError): self.verify()

    def test_changed_regular_source_rejected(self):
        (self.code / 'execution/example.py').write_bytes(b'# changed\n')
        with self.assertRaises(ValueError): self.verify()

    def test_namespace_duplicate_paths_rejected(self):
        module = self.modules['execution']; module.__path__ = [str(self.code/'execution')]*2
        module.__spec__.submodule_search_locations = module.__path__
        with self.assertRaisesRegex(ValueError, 'Duplicate/foreign'): self.verify()

    def test_namespace_foreign_paths_rejected(self):
        module = self.modules['execution']; module.__path__ = [str(self.code/'analysis')]
        module.__spec__.submodule_search_locations = module.__path__
        with self.assertRaisesRegex(ValueError, 'Duplicate/foreign'): self.verify()

    def test_namespace_spec_path_disagreement_rejected(self):
        self.modules['execution'].__path__ = [str(self.code/'execution')]
        with self.assertRaisesRegex(ValueError, 'loader/spec'): self.verify()

    def test_fake_loader_or_spec_rejected(self):
        self.modules['execution'].__loader__ = object()
        with self.assertRaises(ValueError): self.verify()

    def test_namespace_init_file_or_inventory_disagrees(self):
        self.sources['execution/__init__.py'] = 'a'*64
        with self.assertRaises(ValueError): self.verify()
        self.sources.pop('execution/__init__.py'); (self.code/'execution/__init__.py').write_bytes(b'')
        with self.assertRaises(ValueError): self.verify()

    def test_no_blanket_ignore_none_modules(self):
        self.modules['runtime.injected'] = ModuleType('runtime.injected')
        with self.assertRaises(ValueError): self.verify()

    def test_original_full_audit_function_is_not_rewritten(self):
        original = self.a.audit_remote; original_main = self.a.main; original_sha = self.a.sha(OPERATIONS/'satdose_audit_continuation.py')
        r.configure(self.a, self.original_source)
        self.assertIs(self.a.audit_remote, original); self.assertIs(self.a.main, original_main)
        self.assertEqual(original_sha, r.ORIGINAL_SHA)
        self.assertEqual(self.a.audit_for(r.KNOWN_SELECTION), r.NEW_DIRECTORY)
        self.assertTrue(r.RECOVERY_FILES <= self.a.FILES)
        self.assertTrue(r.RECOVERY_FILES <= self.a.SUCCESS_FILES)
        with self.assertRaises(ValueError): self.a.audit_for(dict(r.KNOWN_SELECTION, arm='T1'))


class ActualDownloadedFailure(unittest.TestCase):
    def setUp(self):
        self.a, self.source = r.load_original()
        self.args = SimpleNamespace(submission_receipt=SUBMIT, submission_sha256=SUBMIT_SHA,
            job_id=r.KNOWN_JOB, start_seed=11031, arm='T0', first_step=3)
        self.payload = self.a.local_payload(self.args)
        self.manifest = self.a.loads((FAILED_DIR/'REMOTE_MANIFEST.json').read_bytes())
        self.contents = {name:(FAILED_DIR/'artifacts'/name).read_bytes() for name in r.FAILED_FILES}
        self.original_payload = self.a.loads(self.contents['INTENT.json'])['payload']

    def test_actual_complete_known_failure_tar_source_risk_and_submit_match(self):
        value = r.verify_local_failure(self.a, FAILED_DIR, r.FAILED_MANIFEST_SHA, self.payload)
        self.assertEqual(value['manifest'], self.manifest)
        self.assertTrue(value['binding']['original_failure_preserved'])
        self.assertFalse(value['binding']['original_scientific_hold'])
        self.assertTrue(value['binding']['complete_new_cpu_audit_required'])

    def test_wrong_manifest_pin_and_foreign_job_rejected(self):
        with self.assertRaises(ValueError): r.verify_local_failure(self.a, FAILED_DIR, 'a'*64, self.payload)
        value = dict(self.payload, job_id=1742959)
        with self.assertRaises(ValueError): r.verify_local_failure(self.a, FAILED_DIR, r.FAILED_MANIFEST_SHA, value)

    def test_unknown_partial_or_success_manifest_ineligible(self):
        cases = []
        value = copy.deepcopy(self.manifest); value['files'].pop('RISK_REPLAY.json'); cases.append(value)
        value = copy.deepcopy(self.manifest); value['success'] = True; cases.append(value)
        value = copy.deepcopy(self.manifest); value['operation_source_sha256'] = 'a'*64; cases.append(value)
        value = copy.deepcopy(self.manifest); value['payload_sha256'] = 'b'*64; cases.append(value)
        for value in cases:
            with self.assertRaises(ValueError): r.verify_failure_manifest(self.a, value, self.original_payload)

    def test_changed_error_risk_or_result_ineligible(self):
        for name in ('ERROR.json','RISK_REPLAY.json','ORIGINAL_RESULT.json','INCIDENTS.json'):
            contents = dict(self.contents); contents[name] += b' '
            with self.assertRaises(ValueError): r.validate_failed_members(self.a,self.manifest,self.original_payload,contents)

    def argv(self):
        return ['--submission-receipt',str(SUBMIT),'--submission-sha256',SUBMIT_SHA,'--job-id',str(r.KNOWN_JOB),
            '--start-seed','11031','--arm','T0','--first-step','3','--failed-audit-dir',str(FAILED_DIR),
            '--failed-manifest-sha256',r.FAILED_MANIFEST_SHA]

    def test_actual_local_plan_no_subprocess_or_write(self):
        with patch.object(self.a.subprocess,'run',side_effect=AssertionError('SSH forbidden')), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(r.main(self.argv()),0)
        value=self.a.loads(output.getvalue())
        self.assertEqual(value['remote_directory'],str(r.NEW_DIRECTORY));self.assertEqual(value['planned_audit_invocations'],1)
        self.assertFalse(value['next_stage_authorized'])

    def test_configured_payload_binds_exact_failed_manifest(self):
        r.configure(self.a,self.source,local=True)
        args=self.a.parser().parse_args(self.argv());payload=self.a.local_payload(args)
        self.assertEqual(payload['known_failure']['manifest'],self.manifest)
        self.assertEqual(payload['known_failure']['binding']['original_error_sha256'],r.FAILED_ERROR_SHA)
        self.assertEqual(payload['submitted']['release'],self.payload['submitted']['release'])

    def test_recovery_unknown_attempt_never_rerun(self):
        with tempfile.TemporaryDirectory(prefix='satdose-known-recovery-OFFLINE-') as temp:
            directory=Path(temp).resolve()/'occupied';directory.mkdir()
            with patch.object(self.a.subprocess,'run',side_effect=AssertionError('SSH forbidden')), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'Occupied'):
                    r.main(self.argv()+['--local-dir',str(directory),'--execute'])

    def recovery_manifest(self):
        r.configure(self.a,self.source,local=True)
        args=self.a.parser().parse_args(self.argv());payload=self.a.local_payload(args)
        payload['operation_source_sha256']=self.a.sha(Path(r.__file__))
        files={name:dict(path=str(r.NEW_DIRECTORY/name),sha256='a'*64,bytes=1) for name in self.a.SUCCESS_FILES}
        for name,raw in [('OPERATION_SOURCE.py',Path(r.__file__).read_bytes()),
                         ('ORIGINAL_AUDITOR_SOURCE.py',self.source),
                         ('KNOWN_FAILURE_BINDING.json',self.a.canonical(payload['known_failure']['binding']))]:
            files[name].update(sha256=self.a.digest(raw),bytes=len(raw))
        manifest=dict(schema='sr-satdose-continuation-audit-manifest-v1',success=True,
            status='CONTINUATION_AUDIT_PUBLICATION_COMPLETE',permission=self.a.PERMISSION,
            job_id=r.KNOWN_JOB,selection=r.KNOWN_SELECTION,payload_sha256=self.a.digest(self.a.canonical(payload)),
            operation_source_sha256=payload['operation_source_sha256'],next_stage_authorized=False,files=files,
            bundle=dict(path=str(r.NEW_DIRECTORY/'BUNDLE.tar.gz'),sha256='a'*64,bytes=1))
        return payload,manifest

    def test_complete_publication_requires_exact_original_source_and_failure_binding(self):
        payload,manifest=self.recovery_manifest()
        self.a.validate_manifest(manifest,payload)
        for name in r.RECOVERY_FILES:
            bad=copy.deepcopy(manifest);bad['files'][name]['sha256']='f'*64
            with self.assertRaises(ValueError):self.a.validate_manifest(bad,payload)
            bad=copy.deepcopy(manifest);bad['files'].pop(name)
            with self.assertRaises(ValueError):self.a.validate_manifest(bad,payload)

    def test_download_only_recovery_never_invokes_audit(self):
        payload,manifest=self.recovery_manifest();calls=[]
        with tempfile.TemporaryDirectory(prefix='satdose-download-recovery-OFFLINE-') as temp:
            directory=Path(temp).resolve();saved=directory/'MANIFEST.json';saved.write_bytes(self.a.canonical(manifest))
            output=directory/'new_download'
            def ssh(argv,**kwargs):
                calls.append(argv)
                self.assertIn('--remote-download',argv[-1]);self.assertNotIn('--remote-worker',argv[-1])
                kwargs['stdout'].write(b'PARTIAL OFFLINE DOWNLOAD');kwargs['stderr'].write(b'OFFLINE NETWORK FAILURE')
                return SimpleNamespace(returncode=255)
            with patch.object(self.a.subprocess,'run',side_effect=ssh),redirect_stdout(io.StringIO()):
                self.assertEqual(r.main(self.argv()+['--recover-manifest',str(saved),'--recover-sha256',self.a.sha(saved),
                    '--local-dir',str(output),'--execute']),2)
            self.assertEqual(len(calls),1)
            receipt=self.a.loads((output/'LOCAL_RECEIPT.json').read_bytes())
            self.assertTrue(receipt['remote_audit_success'])
            self.assertEqual((output/'BUNDLE.tar.gz').read_bytes(),b'PARTIAL OFFLINE DOWNLOAD')
            self.assertEqual((output/'OPERATION_SOURCE.py').read_bytes(),Path(r.__file__).read_bytes())


if __name__ == '__main__': unittest.main()
