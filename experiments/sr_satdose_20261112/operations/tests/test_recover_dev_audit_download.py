"""Synthetic offline recovery tests only; no SSH or production certification."""
import base64
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

OPERATOR = Path(__file__).resolve().parents[1] / 'satdose_recover_dev_audit_download.py'
SPEC = importlib.util.spec_from_file_location('dev_audit_recovery_offline', OPERATOR)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
EP = dict(start_seed=11031, arm='T0', step=2, responses=108)
JOB = 1741834


def synthetic(hold=False):
    """Invented local bytes exercise shape/consistency; not real run evidence."""
    def external(path):
        return dict(path=path, sha256=M.digest(path.encode()))
    slot = 'trained_dev/start-11031/T0/step-002'
    receipt = external(slot + '/endpoint/RECEIPT.json')
    binding = external(slot + '/ENDPOINT_BINDING.json')
    release = external('execution_releases/OFFLINE_FAKE_RELEASE.json')
    submit = external(str(M.ROOT.parent / 'scheduling/OFFLINE_FAKE_SUBMISSION.json'))
    evidence = dict(receipt_sha256=receipt['sha256'], endpoint_binding_sha256=binding['sha256'])
    summary = dict(schema='sr-satdose-evaluation-analysis-v1', scope='DEV',
        permission='ANALYSIS_ONLY_NO_TRAINING_OR_MAIN_RELEASE', endpoint={k: EP[k] for k in ('start_seed','arm','step')},
        metrics=dict(assigned=108), evidence=evidence)
    analysis = dict(rows=[dict(request_id=f'FAKE_{i}', start_seed=11031, arm='T0', step=2, S=None) for i in range(108)], summary=summary)
    observed = {M.reference_path(r): r['sha256'] for r in (receipt,binding,release,submit,M.ORIGINAL)}
    risk = dict(schema='sr-satdose-scientific-risk-assessment-v1', global_hold=hold,
        permission='RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY', training_permission=False, main_permission=False,
        hold_reasons=['OFFLINE_FAKE_HOLD'] if hold else [], pending_required_dev=['11031/T1/2'],
        dev_analysis_evidence={'11031/T0/2':evidence}, dev_endpoints={'11031/T0/2':dict(receipt=receipt,binding=binding)},
        observed_files=copy.deepcopy(observed))
    blobs = {'ANALYSIS.json':M.canonical(analysis), 'ORIGINAL_POST_RISK.json':M.canonical(risk), 'RISK_REPLAY.json':M.canonical(risk)}
    result = dict(schema='sr-satdose-trained-dev-result-v1', status='TRAINED_DEV_ENDPOINT_COMPLETE',
        permission='NO_TRAINING_OR_MAIN', **EP, release=release, post_risk=dict(path=slot+'/POST_RISK.json',sha256=M.digest(blobs['ORIGINAL_POST_RISK.json'])),
        receipt=receipt, endpoint_binding=binding, global_scientific_hold=hold)
    blobs['ORIGINAL_RESULT.json'] = M.canonical(result)
    claim = dict(endpoint=EP,job_id=JOB,permission=M.AUDIT_PERMISSION,operation_source_sha256=M.AUDITOR_SHA,
                 status='ANALYSIS_ATTEMPT_UNKNOWN_UNTIL_PUBLICATION',submission=submit)
    blobs['CLAIM.json'] = M.canonical(claim)
    blobs['OWN_JOB_STDOUT.log'] = M.canonical(result)
    blobs['OWN_JOB_STDERR.log'] = b'OFFLINE synthetic stderr\n'
    blobs['SCHEDULER.json'] = M.canonical(dict(offline=True))
    directory = M.audit_directory(JOB,EP)
    def package_ref(name):return dict(path=str((directory/name).relative_to(M.ROOT)),sha256=M.digest(blobs[name]))
    resultref = dict(path=slot+'/RESULT.json',sha256=M.digest(blobs['ORIGINAL_RESULT.json']))
    observed.update({M.reference_path(resultref):resultref['sha256'], M.reference_path(result['post_risk']):result['post_risk']['sha256']})
    report = dict(schema='sr-satdose-independent-trained-dev-audit-v1',
        status='TRAINED_DEV_INDEPENDENT_AUDIT_COMPLETE'+('_SCIENTIFIC_HOLD' if hold else ''),
        permission=M.AUDIT_PERMISSION, endpoint=EP,job_id=JOB,next_stage_authorized=False,
        exact_post_risk_object_reproduced=True,scientific_outcome_preserved=True,original_train_release=M.ORIGINAL,
        source_files=84,execution_source_keys=19,responses=108,operation_source_sha256=M.AUDITOR_SHA,submission=submit,
        optimizer_steps=0,scheduler_steps=0,parameter_updates=0,audit_model_calls=0,audit_optimizer_calls=0,
        global_scientific_hold=hold,hold_reasons=risk['hold_reasons'],pending_required_dev=risk['pending_required_dev'],
        analysis=package_ref('ANALYSIS.json'),original_post_risk=package_ref('ORIGINAL_POST_RISK.json'),
        risk_replay=package_ref('RISK_REPLAY.json'),result=resultref,release=release,observed_files=observed)
    blobs['AUDIT_REPORT.json'] = M.canonical(report)
    publication = dict(status='ANALYSIS_ONLY_ARTIFACTS_PUBLISHED',permission=M.AUDIT_PERMISSION,job_id=JOB,endpoint=EP,
        report=package_ref('AUDIT_REPORT.json'),files={n:package_ref(n) for n in blobs},global_scientific_hold=hold)
    blobs['PUBLICATION.json'] = M.canonical(publication)
    manifest = dict(status='ANALYSIS_PUBLICATION_COMPLETE',success=True,job_id=JOB,endpoint=EP,permission=M.AUDIT_PERMISSION,
        files={n:dict(path=str(directory/n),sha256=M.digest(data),bytes=len(data)) for n,data in blobs.items()})
    return blobs,manifest


def bundle(blobs, manifest, manifest_raw):
    return M.canonical(dict(schema='sr-satdose-dev-audit-download-bundle-v1',permission=M.PERMISSION,
        job_id=JOB,endpoint=EP,manifest_sha256=M.digest(manifest_raw),
        files={n:dict(manifest['files'][n],data_base64=base64.b64encode(raw).decode()) for n,raw in blobs.items()}))


class Recovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='satdose-dev-recovery-OFFLINE-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.old = self.directory/'old_attempt';self.old.mkdir()
        self.manifest_path = self.old/'REMOTE_MANIFEST.json'
        self.error = self.old/'LOCAL_ERROR.json';self.error.write_bytes(b'OFFLINE prior failure stays unchanged\n')
        self.blobs,self.manifest = synthetic()
        self.manifest_raw = M.canonical(self.manifest);self.manifest_path.write_bytes(self.manifest_raw)
        self.args = SimpleNamespace(manifest=self.manifest_path,manifest_sha256=M.digest(self.manifest_raw),
            job_id=JOB,start_seed=11031,arm='T0',step=2,local_dir=self.directory/'recovered',execute=True)
    def test_compile(self):compile(OPERATOR.read_bytes(),str(OPERATOR),'exec')
    def test_default_plan_does_not_start_process_or_create_output(self):
        self.args.execute=False
        with patch.object(M.subprocess,'run',side_effect=AssertionError('network called')),redirect_stdout(io.StringIO()):
            self.assertEqual(M.local(self.args),0)
        self.assertFalse(self.args.local_dir.exists())
    def test_manifest_wrong_hash_endpoint_job_missing_or_extra_file_rejected(self):
        for mutate in (lambda d:d.update(job_id=1),lambda d:d['endpoint'].update(arm='T1'),
                       lambda d:d['files'].pop('SCHEDULER.json'),lambda d:d['files'].update({'ERROR.json':{}})):
            value=copy.deepcopy(self.manifest);mutate(value)
            with self.assertRaises(ValueError):M.validate_manifest(value,JOB,EP)
        self.args.manifest_sha256='0'*64
        with self.assertRaises(ValueError):M.local_input(self.args)
    def test_manifest_escape_short_hash_boolean_length_rejected(self):
        for key,val in [('path','/etc/passwd'),('sha256','short'),('bytes',True),('bytes',M.MAX_FILE_BYTES+1)]:
            value=copy.deepcopy(self.manifest);value['files']['ANALYSIS.json'][key]=val
            with self.assertRaises(ValueError):M.validate_manifest(value,JOB,EP)
    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'):
            with self.assertRaises(ValueError):M.loads(raw)
    def test_complete_package_pending_and_unknown_preserved(self):
        self.assertEqual(M.validate_package(self.blobs,self.manifest)['pending_required_dev'],['11031/T1/2'])
        self.assertIsNone(M.loads(self.blobs['ANALYSIS.json'])['rows'][0]['S'])
    def test_scientific_hold_is_recovered_without_authority(self):
        blobs,manifest=synthetic(True);value=M.validate_package(blobs,manifest)
        self.assertTrue(value['global_scientific_hold']);self.assertFalse(value['external_evidence_rehashed'])
    def test_corrupt_bytes_and_base64_rejected(self):
        value=M.loads(bundle(self.blobs,self.manifest,self.manifest_raw));value['files']['RISK_REPLAY.json']['data_base64']='$$$'
        with self.assertRaises(ValueError):M.decode_bundle(M.canonical(value),self.manifest,M.digest(self.manifest_raw))
        value['files']['RISK_REPLAY.json']['data_base64']=base64.b64encode(b'corrupt').decode()
        with self.assertRaises(ValueError):M.decode_bundle(M.canonical(value),self.manifest,M.digest(self.manifest_raw))
    def test_manifest_refs_and_complete_risk_objects_checked_after_hashes(self):
        for name,mutate in [('PUBLICATION.json',lambda v:v['files'].pop('ANALYSIS.json')),
                           ('AUDIT_REPORT.json',lambda v:v.update(next_stage_authorized=True)),
                           ('RISK_REPLAY.json',lambda v:v.update(hold_reasons=['altered']))]:
            blobs=copy.deepcopy(self.blobs);manifest=copy.deepcopy(self.manifest)
            value=M.loads(blobs[name]);mutate(value);blobs[name]=M.canonical(value)
            manifest['files'][name].update(bytes=len(blobs[name]),sha256=M.digest(blobs[name]))
            if name!='PUBLICATION.json':
                publication=M.loads(blobs['PUBLICATION.json'])
                publication['files'][name]['sha256']=M.digest(blobs[name])
                if name=='AUDIT_REPORT.json':publication['report']['sha256']=M.digest(blobs[name])
                blobs['PUBLICATION.json']=M.canonical(publication)
                manifest['files']['PUBLICATION.json'].update(bytes=len(blobs['PUBLICATION.json']),sha256=M.digest(blobs['PUBLICATION.json']))
            with self.assertRaises(ValueError):M.validate_package(blobs,manifest)
    def fake_run(self,argv,**kwargs):
        self.assertEqual(argv[0],'ssh');self.assertEqual(kwargs['input'],OPERATOR.read_bytes())
        self.assertTrue((self.args.local_dir/'RECOVERY_INTENT.json').is_file())
        kwargs['stdout'].write(bundle(self.blobs,self.manifest,self.manifest_raw));kwargs['stderr'].write(b'OFFLINE ssh diagnostic\n')
        return SimpleNamespace(returncode=0)
    def test_single_read_only_call_complete_exact_download(self):
        with patch.object(M.subprocess,'run',side_effect=self.fake_run) as run,redirect_stdout(io.StringIO()):
            self.assertEqual(M.local(self.args),0)
        self.assertEqual(run.call_count,1)
        for name,raw in self.blobs.items():self.assertEqual((self.args.local_dir/name).read_bytes(),raw)
        receipt=M.loads((self.args.local_dir/'RECOVERY_RECEIPT.json').read_bytes())
        self.assertFalse(receipt['next_stage_authorized']);self.assertFalse(receipt['remote_audit_rerun'])
        self.assertEqual(self.error.read_bytes(),b'OFFLINE prior failure stays unchanged\n')
    def test_bad_download_retains_full_raw_output_and_does_not_retry(self):
        bad=b'{"truncated":"'+b'x'*200000
        def fake(argv,**kwargs):
            kwargs['stdout'].write(bad);kwargs['stderr'].write(b'complete bad transfer stderr')
            return SimpleNamespace(returncode=0)
        with patch.object(M.subprocess,'run',side_effect=fake) as run:
            with self.assertRaises(ValueError):M.local(self.args)
        self.assertEqual(run.call_count,1);self.assertEqual((self.args.local_dir/'SSH_STDOUT.raw').read_bytes(),bad)
        self.assertEqual((self.args.local_dir/'SSH_STDERR.raw').read_bytes(),b'complete bad transfer stderr')
        self.assertTrue((self.args.local_dir/'RECOVERY_ERROR.json').is_file())
        self.assertFalse((self.args.local_dir/'RECOVERY_RECEIPT.json').exists())
        with patch.object(M.subprocess,'run',side_effect=AssertionError('must not retry')):
            with self.assertRaises(ValueError):M.local(self.args)
    def test_nonzero_exit_retains_raw_even_when_json_looks_valid(self):
        def fake(argv,**kwargs):self.fake_run(argv,**kwargs);return SimpleNamespace(returncode=255)
        with patch.object(M.subprocess,'run',side_effect=fake):
            with self.assertRaises(ValueError):M.local(self.args)
        self.assertTrue((self.args.local_dir/'SSH_STDOUT.raw').is_file())
        self.assertFalse((self.args.local_dir/'RECOVERY_RECEIPT.json').exists())
    def test_valid_bundle_with_bad_file_hash_retains_all_raw_bytes(self):
        value=M.loads(bundle(self.blobs,self.manifest,self.manifest_raw))
        value['files']['RISK_REPLAY.json']['data_base64']=base64.b64encode(b'bad transferred bytes').decode()
        bad=M.canonical(value)
        def fake(argv,**kwargs):
            kwargs['stdout'].write(bad);kwargs['stderr'].write(b'complete stderr')
            return SimpleNamespace(returncode=0)
        with patch.object(M.subprocess,'run',side_effect=fake) as run:
            with self.assertRaisesRegex(ValueError,'Downloaded bytes differ'):M.local(self.args)
        self.assertEqual(run.call_count,1);self.assertEqual((self.args.local_dir/'SSH_STDOUT.raw').read_bytes(),bad)
        self.assertTrue((self.args.local_dir/'RECOVERY_ERROR.json').is_file())
    def test_semantic_failure_retains_decoded_artifacts(self):
        with patch.object(M.subprocess,'run',side_effect=self.fake_run),patch.object(M,'validate_package',side_effect=ValueError('OFFLINE semantic failure')):
            with self.assertRaisesRegex(ValueError,'semantic failure'):M.local(self.args)
        for name,raw in self.blobs.items():self.assertEqual((self.args.local_dir/name).read_bytes(),raw)
        self.assertTrue((self.args.local_dir/'RECOVERY_ERROR.json').is_file())
    def test_interruption_retains_partial_streams_and_error_without_retry(self):
        def fake(argv,**kwargs):
            kwargs['stdout'].write(b'partial bundle');kwargs['stderr'].write(b'partial stderr')
            raise KeyboardInterrupt('OFFLINE simulated interruption')
        with patch.object(M.subprocess,'run',side_effect=fake) as run:
            with self.assertRaises(KeyboardInterrupt):M.local(self.args)
        self.assertEqual(run.call_count,1);self.assertEqual((self.args.local_dir/'SSH_STDOUT.raw').read_bytes(),b'partial bundle')
        self.assertTrue((self.args.local_dir/'RECOVERY_ERROR.json').is_file())
    def test_local_symlink_or_old_attempt_child_rejected_before_network(self):
        alias=self.directory/'alias';alias.symlink_to(self.directory,target_is_directory=True)
        for dest in (alias/'out',self.old/'new_child'):
            self.args.local_dir=dest
            with patch.object(M.subprocess,'run',side_effect=AssertionError('network called')):
                with self.assertRaises(ValueError):M.local(self.args)
    def test_remote_reader_exact_ten_no_write_or_subprocess(self):
        root=self.directory/'fake_server';root.mkdir();(root/'execution_releases').mkdir()
        with patch.object(M,'ROOT',root),patch.object(M,'PROJECT',self.directory):
            blobs,manifest=synthetic();directory=M.audit_directory(JOB,EP);directory.mkdir()
            for name,raw in blobs.items():(directory/name).write_bytes(raw)
            payload=dict(manifest_base64=base64.b64encode(M.canonical(manifest)).decode(),manifest_sha256=M.digest(M.canonical(manifest)),job_id=JOB,endpoint=EP)
            with patch.object(M.subprocess,'run',side_effect=AssertionError('subprocess in reader')),patch.object(M,'write_once',side_effect=AssertionError('write in reader')):
                got=M.read_remote_bundle(payload)
            self.assertEqual(M.decode_bundle(M.canonical(got),manifest,payload['manifest_sha256']),blobs)
            (directory/'EXTRA').write_bytes(b'x')
            with self.assertRaises(ValueError):M.read_remote_bundle(payload)
    def test_remote_symlink_and_fifo_rejected(self):
        root=self.directory/'fake_server';root.mkdir();(root/'execution_releases').mkdir()
        with patch.object(M,'ROOT',root),patch.object(M,'PROJECT',self.directory):
            blobs,manifest=synthetic();directory=M.audit_directory(JOB,EP);directory.mkdir()
            for name,raw in blobs.items():(directory/name).write_bytes(raw)
            payload=dict(manifest_base64=base64.b64encode(M.canonical(manifest)).decode(),manifest_sha256=M.digest(M.canonical(manifest)),job_id=JOB,endpoint=EP)
            target=directory/'RISK_REPLAY.json';target.unlink();target.symlink_to(directory/'ORIGINAL_POST_RISK.json')
            with self.assertRaises(ValueError):M.read_remote_bundle(payload)
            target.unlink();os.mkfifo(target)
            with self.assertRaises(ValueError):M.read_remote_bundle(payload)


if __name__ == '__main__':
    unittest.main(verbosity=2)
