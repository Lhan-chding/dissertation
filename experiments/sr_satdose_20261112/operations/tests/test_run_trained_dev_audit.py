"""Offline synthetic driver tests; no real SSH, audit, or model invocation."""
import base64
from contextlib import redirect_stdout
import copy
import importlib.util
import io
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

OPERATOR = Path(__file__).resolve().parents[1] / 'satdose_run_trained_dev_audit.py'


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


M=module('dev_driver_offline',OPERATOR)
F=module('dev_driver_synthetic_fixture',Path(__file__).with_name('test_recover_dev_audit_download.py'))


class Driver(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='satdose-dev-driver-OFFLINE-')
        self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name).resolve()
        self.args=SimpleNamespace(submission_receipt=self.directory/'FAKE_SUBMIT.json',submission_sha256='0'*64,
            job_id=F.JOB,start_seed=11031,arm='T0',step=2,local_dir=self.directory/'attempt',execute=True)
        self.auditor,self.auditor_source=M.load_pinned(M.AUDITOR_PATH,M.AUDITOR_SHA,'offline_original_auditor')
        self.recovery,self.recovery_source=M.load_pinned(M.RECOVERY_PATH,M.RECOVERY_SHA,'offline_original_recovery')
        self.payload=dict(job_id=F.JOB,endpoint=copy.deepcopy(F.EP),operation_source_sha256=M.AUDITOR_SHA,
            local_submission_ref=dict(path=str(self.args.submission_receipt),sha256='0'*64),
            submission={'OFFLINE_SYNTHETIC':True},submission_ref={'OFFLINE_SYNTHETIC':True},sources={},execution_sources={})
        self.blobs,self.manifest=F.synthetic();self.manifest_raw=M.canonical(self.manifest)
        self.prepared=(self.auditor,self.recovery,self.auditor_source,self.recovery_source,self.payload)
        self.calls=[]
    def run_driver(self,side_effect=None):
        with patch.object(M,'prepare',return_value=self.prepared),patch.object(M.subprocess,'run',side_effect=side_effect or self.fake_ssh),redirect_stdout(io.StringIO()):
            return M.local(self.args)
    def fake_ssh(self,argv,**kwargs):
        self.calls.append((argv,kwargs['input']))
        self.assertEqual(argv[0],'ssh')
        remote=shlex.split(argv[-1])
        if '--remote-worker' in remote:
            self.assertEqual(len(self.calls),1)
            self.assertEqual(kwargs['input'],self.auditor_source)
            payload=M.loads(base64.b64decode(remote[-1]))
            self.assertEqual(payload['operation_source_sha256'],M.AUDITOR_SHA)
            self.assertTrue((self.args.local_dir/'DRIVER_INTENT.json').is_file())
            self.assertEqual((self.args.local_dir/'ORIGINAL_AUDITOR_SOURCE.py').read_bytes(),self.auditor_source)
            kwargs['stdout'].write(b'OFFLINE native diagnostic\nSATDOSE_DEV_AUDIT_JSON='+self.manifest_raw)
            kwargs['stderr'].write(b'OFFLINE original audit stderr\n')
        elif '--remote-read-only' in remote:
            self.assertEqual(len(self.calls),2)
            self.assertEqual(kwargs['input'],self.recovery_source)
            manifest_path=self.args.local_dir/'audit/REMOTE_MANIFEST.json'
            self.assertEqual(manifest_path.read_bytes(),self.manifest_raw)
            kwargs['stdout'].write(F.bundle(self.blobs,self.manifest,self.manifest_raw))
            kwargs['stderr'].write(b'OFFLINE bundle stderr\n')
        else:self.fail('Unexpected command; old per-file cat or another operation is forbidden')
        return SimpleNamespace(returncode=0)
    def test_compile_and_both_original_pins_unchanged(self):
        compile(OPERATOR.read_bytes(),str(OPERATOR),'exec')
        self.assertEqual(M.digest(self.auditor_source),M.AUDITOR_SHA)
        self.assertEqual(M.digest(self.recovery_source),M.RECOVERY_SHA)
    def test_default_plan_has_no_process_or_output_directory(self):
        self.args.execute=False
        with patch.object(M,'prepare',return_value=self.prepared),patch.object(M.subprocess,'run',side_effect=AssertionError('network in plan')),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(M.local(self.args),0)
        self.assertEqual(M.loads(out.getvalue())['status'],'LOCAL_PLAN_ONLY_NO_SSH')
        self.assertFalse(self.args.local_dir.exists())
    def test_wrong_pin_cannot_execute_source(self):
        source=self.directory/'wrong.py';source.write_bytes(b'raise AssertionError("untrusted source executed")\n')
        with self.assertRaisesRegex(ValueError,'Pinned operation source SHA differs'):
            M.load_pinned(source,M.AUDITOR_SHA,'untrusted')
    def test_each_original_source_pin_checked_before_local_provenance(self):
        for constant in ('AUDITOR_SHA','RECOVERY_SHA'):
            with patch.object(M,constant,'0'*64),patch.object(M.subprocess,'run',side_effect=AssertionError('network')):
                with self.assertRaisesRegex(ValueError,'Pinned operation source SHA differs'):M.prepare(self.args)
    def test_source_symlink_rejected(self):
        alias=self.directory/'alias.py';alias.symlink_to(M.AUDITOR_PATH)
        with self.assertRaisesRegex(ValueError,'Noncanonical'):M.load_pinned(alias,M.AUDITOR_SHA,'alias')
    def test_original_local_payload_rejects_fake_submission(self):
        self.args.submission_receipt.write_bytes(M.canonical({'status':'NOT_SUBMITTED'}))
        self.args.submission_sha256=M.digest(self.args.submission_receipt.read_bytes())
        with patch.object(M.subprocess,'run',side_effect=AssertionError('network')):
            with self.assertRaisesRegex(PermissionError,'No successful actual submission receipt'):M.prepare(self.args)
    def test_success_original_source_once_then_one_bundle_download(self):
        self.assertEqual(self.run_driver(),0);self.assertEqual(len(self.calls),2)
        receipt=M.loads((self.args.local_dir/'DRIVER_RECEIPT.json').read_bytes())
        self.assertFalse(receipt['next_stage_authorized']);self.assertFalse(receipt['risk_or_scoring_modified'])
        self.assertEqual(receipt['original_audit_invocations'],1)
        self.assertEqual(receipt['bundle_download_invocations'],1)
        self.assertEqual(receipt['sources']['original_auditor']['sha256'],M.AUDITOR_SHA)
        for name,data in self.blobs.items():self.assertEqual((self.args.local_dir/'download'/name).read_bytes(),data)
        self.assertEqual((self.args.local_dir/'audit/REMOTE_STDERR.raw').read_bytes(),b'OFFLINE original audit stderr\n')
        self.assertNotEqual((self.args.local_dir/'download').parent,self.args.local_dir/'audit')
    def test_scientific_hold_preserved_and_no_successor_authority(self):
        self.blobs,self.manifest=F.synthetic(True);self.manifest_raw=M.canonical(self.manifest)
        self.assertEqual(self.run_driver(),0)
        receipt=M.loads((self.args.local_dir/'DRIVER_RECEIPT.json').read_bytes())
        self.assertTrue(receipt['validation']['global_scientific_hold']);self.assertFalse(receipt['next_stage_authorized'])
        self.assertEqual(receipt['validation']['hold_reasons'],['OFFLINE_FAKE_HOLD'])
    def test_original_negative_manifest_retained_without_download(self):
        negative=dict(status='AUDIT_FAILED_OR_UNKNOWN_ARTIFACTS_RETAINED',success=False,job_id=F.JOB,
            endpoint=F.EP,files={'ERROR.json':{'retained':'OFFLINE'}},permission=self.auditor.PERMISSION)
        raw=M.canonical(negative)
        def fail(argv,**kwargs):
            self.calls.append(argv);kwargs['stdout'].write(b'SATDOSE_DEV_AUDIT_JSON='+raw);kwargs['stderr'].write(b'negative details')
            return SimpleNamespace(returncode=2)
        with self.assertRaises(ValueError):self.run_driver(fail)
        self.assertEqual(len(self.calls),1);self.assertEqual((self.args.local_dir/'audit/REMOTE_MANIFEST.json').read_bytes(),raw)
        self.assertFalse((self.args.local_dir/'download').exists())
        error=M.loads((self.args.local_dir/'DRIVER_ERROR.json').read_bytes());self.assertFalse(error['original_audit_complete'])
    def test_false_manifest_even_with_zero_exit_never_downloaded(self):
        self.manifest.update(success=False,status='AUDIT_NOT_STARTED_OR_UNKNOWN');self.manifest_raw=M.canonical(self.manifest)
        with self.assertRaises(ValueError):self.run_driver()
        self.assertEqual(len(self.calls),1);self.assertFalse((self.args.local_dir/'download').exists())
    def test_download_failure_keeps_complete_audit_and_cannot_reaudit(self):
        def fail(argv,**kwargs):
            if not self.calls:return self.fake_ssh(argv,**kwargs)
            self.calls.append(argv);kwargs['stdout'].write(b'BROKEN DOWNLOAD'+b'x'*200000);kwargs['stderr'].write(b'full download stderr')
            return SimpleNamespace(returncode=0)
        with self.assertRaises(ValueError):self.run_driver(fail)
        self.assertEqual(len(self.calls),2)
        error=M.loads((self.args.local_dir/'DRIVER_ERROR.json').read_bytes())
        self.assertTrue(error['original_audit_complete']);self.assertEqual(error['phase'],'EXACT_BUNDLE_DOWNLOAD')
        self.assertEqual((self.args.local_dir/'download/SSH_STDOUT.raw').read_bytes(),b'BROKEN DOWNLOAD'+b'x'*200000)
        self.assertTrue((self.args.local_dir/'download/RECOVERY_ERROR.json').is_file())
        self.assertEqual((self.args.local_dir/'audit/REMOTE_MANIFEST.json').read_bytes(),self.manifest_raw)
        with self.assertRaisesRegex(ValueError,'never overwrite or retry'):self.run_driver(lambda *a,**k:self.fail('re-audit called'))
        self.assertEqual(len(self.calls),2)
    def test_audit_network_uncertainty_keeps_every_raw_byte_no_download(self):
        raw=b'partial original audit response'+b'z'*100000
        def fail(argv,**kwargs):
            self.calls.append(argv);kwargs['stdout'].write(raw);kwargs['stderr'].write(b'network unreachable detail')
            return SimpleNamespace(returncode=255)
        with self.assertRaises(ValueError):self.run_driver(fail)
        self.assertEqual(len(self.calls),1);self.assertEqual((self.args.local_dir/'audit/REMOTE_STDOUT.raw').read_bytes(),raw)
        self.assertFalse((self.args.local_dir/'download').exists())
    def test_manifest_success_with_uncertain_ssh_exit_is_not_download_authority(self):
        def fail(argv,**kwargs):self.fake_ssh(argv,**kwargs);return SimpleNamespace(returncode=255)
        with self.assertRaises(ValueError):self.run_driver(fail)
        self.assertEqual(len(self.calls),1);self.assertTrue((self.args.local_dir/'audit/REMOTE_MANIFEST.json').exists())
        self.assertFalse((self.args.local_dir/'download').exists())
    def test_ambiguous_duplicate_or_nonfinite_json_rejected(self):
        for raw in (b'SATDOSE_DEV_AUDIT_JSON={}\n'*2,
                    b'SATDOSE_DEV_AUDIT_JSON={"success":true,"success":false}\n',
                    b'SATDOSE_DEV_AUDIT_JSON={"value":NaN}\n'):
            with self.assertRaises(ValueError):M.manifest_from_stdout(raw)
    def test_foreign_endpoint_manifest_retained_but_never_downloaded(self):
        self.manifest['endpoint']=dict(F.EP,arm='T1');self.manifest_raw=M.canonical(self.manifest)
        with self.assertRaises(ValueError):self.run_driver()
        self.assertEqual(len(self.calls),1);self.assertTrue((self.args.local_dir/'audit/REMOTE_MANIFEST.json').exists())
        self.assertFalse((self.args.local_dir/'download').exists())
    def test_interruption_keeps_partial_raw_and_occupied_attempt(self):
        def fail(argv,**kwargs):
            self.calls.append(argv);kwargs['stdout'].write(b'partial');kwargs['stderr'].write(b'interrupted')
            raise KeyboardInterrupt('OFFLINE')
        with self.assertRaises(KeyboardInterrupt):self.run_driver(fail)
        self.assertEqual((self.args.local_dir/'audit/REMOTE_STDOUT.raw').read_bytes(),b'partial')
        self.assertTrue((self.args.local_dir/'DRIVER_ERROR.json').is_file())
        with self.assertRaisesRegex(ValueError,'never overwrite or retry'):self.run_driver()
        self.assertEqual(len(self.calls),1)
    def test_parent_symlink_or_existing_directory_rejected_before_ssh(self):
        alias=self.directory/'alias';alias.symlink_to(self.directory,target_is_directory=True)
        for destination in (alias/'attempt',self.directory):
            self.args.local_dir=destination
            with self.assertRaises(ValueError):self.run_driver(lambda *a,**k:self.fail('SSH called'))
        self.assertEqual(len(self.calls),0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
