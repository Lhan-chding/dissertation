"""Offline recovery tests. Positive bundles are explicitly synthetic fixtures.

The original failure receipt and read-only inspection are actual downloaded
evidence; no server, model, native preflight, quota probe or scheduler is invoked.
"""
import base64
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

OPERATIONS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(OPERATIONS))
import satdose_recover_continuation_preparation as r
FAILED=r.PROJECT/'artifacts/sr_satdose_20261112/intake'/(r.LOCAL_PREFIX+'_LOCAL_RECEIPT.json')


def actual_payload():
    return r.local_payload(SimpleNamespace(failed_receipt=FAILED,failed_sha256=r.FAILED_SHA))


def fake_ref(path):return dict(path=path,sha256='a'*64)


def synthetic_fixture():
    """Construct a coherent fake five-file DAG; patch ONLY its publication pins."""
    p=actual_payload();p['recovery_source_sha256']=r.sha(Path(r.__file__))
    trusted={key:[fake_ref(f'training_transactions/start-{key}/step-{n:03d}/COMMIT.json') for n in range(1,9)] for key in r.PATHS}
    dev={f'{seed}/START/0':dict(receipt=fake_ref(f'start_dev/start-{seed}/endpoint/RECEIPT.json'),binding=fake_ref(f'start_dev/start-{seed}/ENDPOINT_BINDING.json')) for seed in (11031,11032)}
    dev.update({f'{key}/{step}':dict(receipt=fake_ref(f'trained_dev/start-{key}/step-{step:03d}/endpoint/RECEIPT.json'),binding=fake_ref(f'trained_dev/start-{key}/step-{step:03d}/ENDPOINT_BINDING.json')) for key in r.PATHS for step in (2,8)})
    prior={key+'/chunk-003-008':fake_ref(f'continuation_control/start-{key}/chunk-003-008/RESULT.json') for key in r.PATHS}
    intent=dict(status='CPU_PREPARATION_INTENT_UNKNOWN_UNTIL_PREFLIGHT',selection=r.SELECTION,preparation_script_sha256=r.PREPARER_SHA,
        model_calls=0,gpu_submission_permission=False,original_train_release=p['original_train_release'],canary_spec=p['summary']['canary_spec'],
        trusted_commits=trusted,dev_endpoints=dev,prior_chunks=prior,created_at_utc='2026-11-19T20:14:23.790385+00:00')
    budget=p['inspected_budget'];raw={'INTENT':r.canonical(intent),'BUDGET':r.canonical(budget)}
    def ref(key):return dict(path=str(r.five_path(key).relative_to(r.ROOT)),sha256=r.digest(raw[key]))
    release=dict(schema='sr-satdose-continuation-release-v1',status='AUDITED_FOR_ONE_FIXED_TRAIN_CHUNK',permission='ONE_FIXED_CHUNK_NO_MAIN',
        selection=r.SELECTION,evidence_root=str(r.ROOT),release_root=str(r.CODE),sources=p['sources'],original_train_release=p['original_train_release'],
        canary_spec=intent['canary_spec'],storage_budget=ref('BUDGET'),trusted_commits=trusted,dev_endpoints=dev,prior_chunks=prior,
        risk_clarification=fake_ref('execution_releases/FAKE_CLARIFICATION.json'),design_freeze_sha256=budget['design_freeze_sha256'],
        config_sha256='b'*64,admission_spec=fake_ref('execution_releases/FAKE_SPEC.json'),admission=fake_ref('execution_releases/FAKE_ADMISSION.json'))
    risk=dict(schema='sr-satdose-scientific-risk-assessment-v1',permission='RISK_ASSESSMENT_ONLY_NO_TRAINING_OR_MAIN_AUTHORITY',
        status='NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED',global_hold=False,hold_reasons=[],pending_required_dev=[],trusted_commits=trusted,
        dev_endpoints=dev,clarification=release['risk_clarification'],verifier=p['execution_sources']['risk_verifier'],paths={key:dict(completed_steps=8) for key in r.PATHS})
    raw['RELEASE']=r.canonical(release);raw['RISK']=r.canonical(risk)
    report=copy.deepcopy(p['inspected_preflight']);report.update(intent=ref('INTENT'),storage_budget=ref('BUDGET'),release=ref('RELEASE'),risk=ref('RISK'))
    raw['PREFLIGHT']=r.canonical(report);p['inspected_preflight']=report
    pins={key:(r.digest(value),len(value)) for key,value in raw.items()}
    documents={key:r.loads(value) for key,value in raw.items()}
    return p,documents,raw,pins


def publication_fixture(p,documents,raw):
    value=r.validate_documents(documents,p)
    evidence=dict(origin=r.ORIGIN,permission=r.PERMISSION,files={key:dict(path=str(r.five_path(key)),sha256=r.digest(value),bytes=len(value),data_b64=base64.b64encode(value).decode()) for key,value in raw.items()},
        failed_receipt=p['failed_receipt'],failed_intent=p['failed_intent'],independent_inspection=p['inspection'],original_producer_source=p['original_source'],
        recovery_source_sha256=p['recovery_source_sha256'],original_ssh_returncode=255,original_stdout_recovered=False,original_stdout_was_empty=True,
        recovered_at_utc='2026-11-20T01:00:00+00:00',original_preflight_checked_at_utc=documents['PREFLIGHT']['checked_at_utc'],
        original_budget_expires_at_utc=documents['BUDGET']['expires_at_utc'])
    evidence.update({key:True for key in ('exclusive_worker_lock_verified','lock_inodes_stable','original_error_absent','target_controller_absent','target_scheduling_absent')})
    evidence.update({key:0 for key in ('prepare_invocations','preflight_invocations','risk_replays','model_calls','optimizer_calls','storage_probes','remote_files_written','gpu_jobs_submitted')})
    value['recovery_evidence']=evidence;return value


class ActualLocalIntake(unittest.TestCase):
    def test_actual_failed_source_raw_intent_and_inspection(self):
        p=actual_payload();self.assertEqual(p['original_failure']['returncode'],255)
        self.assertEqual(p['original_failure']['stdout'],'');self.assertEqual(p['selection'],r.SELECTION)
        self.assertEqual(r.digest(r.canonical(p['inspected_preflight'])),r.FIVE['PREFLIGHT'][0])
        self.assertEqual(r.digest(r.canonical(p['inspected_budget'])),r.FIVE['BUDGET'][0])

    def test_actual_default_plan_has_zero_ssh(self):
        with patch.object(r.subprocess,'run',side_effect=AssertionError('No SSH')),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(r.main(['--failed-receipt',str(FAILED),'--failed-sha256',r.FAILED_SHA]),0)
        self.assertEqual(r.loads(out.getvalue())['status'],'LOCAL_PLAN_ONLY_NO_SSH')

    def test_foreign_failed_sha_or_changed_receipt_rejected(self):
        with self.assertRaises(ValueError):r.local_payload(SimpleNamespace(failed_receipt=FAILED,failed_sha256='f'*64))
        with tempfile.TemporaryDirectory(prefix='satdose-prep-recovery-OFFLINE-') as temp:
            path=Path(temp).resolve()/FAILED.name;path.write_bytes(FAILED.read_bytes()+b' ')
            with self.assertRaises(ValueError):r.local_payload(SimpleNamespace(failed_receipt=path,failed_sha256=r.FAILED_SHA))

    def test_duplicate_json_and_root_escape_rejected(self):
        with self.assertRaises(ValueError):r.loads('{"INTENT":1,"INTENT":2}')
        for path in ('../other','/absolute','a/../b'):
            with self.assertRaises(ValueError):r.rootref(fake_ref(path))


class PublicationValidation(unittest.TestCase):
    def setUp(self):
        self.p,self.docs,self.raw,self.pins=synthetic_fixture()
        self.patch=patch.object(r,'FIVE',self.pins);self.patch.start();self.addCleanup(self.patch.stop)
        self.pub=publication_fixture(self.p,self.docs,self.raw)

    def test_complete_five_file_roundtrip(self):self.assertEqual(r.validate_publication(self.pub,self.p),self.raw)

    def test_changed_bytes_refs_source_or_missing_file_rejected(self):
        for mutation in ('bytes','path','duplicate','source','missing'):
            value=copy.deepcopy(self.pub);files=value['recovery_evidence']['files']
            if mutation=='bytes':files['RISK']['data_b64']=base64.b64encode(b'changed').decode()
            if mutation=='path':files['RISK']['path']='/another/project/RISK.json'
            if mutation=='duplicate':files['RISK']['path']=files['RELEASE']['path']
            if mutation=='source':value['recovery_evidence']['recovery_source_sha256']='f'*64
            if mutation=='missing':files.pop('INTENT')
            with self.assertRaises(ValueError):r.validate_publication(value,self.p)

    def test_expired_budget_cannot_be_refreshed_by_recovery(self):
        for value in ('2026-11-26T21:11:50.016563+00:00','2026-11-19T21:00:00+00:00'):
            pub=copy.deepcopy(self.pub);pub['recovery_evidence']['recovered_at_utc']=value
            with self.assertRaises(ValueError):r.validate_publication(pub,self.p)

    def test_scientific_hold_pending_or_population_mismatch_rejected(self):
        for field,value in [('global_hold',True),('pending_required_dev',['11031/T0/8']),('status','GLOBAL_SCIENTIFIC_HOLD')]:
            docs=copy.deepcopy(self.docs);docs['RISK'][field]=value
            with self.assertRaises(ValueError):r.validate_documents(docs,self.p)
        docs=copy.deepcopy(self.docs);docs['RELEASE']['trusted_commits']['11031/T0'].pop()
        with self.assertRaises(ValueError):r.validate_documents(docs,self.p)

    def test_original_probe_and_preflight_source_not_fabricated(self):
        for field,value in [('preparation_script_sha256','f'*64),('model_calls',1),('final_writability_probe',{})]:
            docs=copy.deepcopy(self.docs);docs['PREFLIGHT'][field]=value
            p=copy.deepcopy(self.p);p['inspected_preflight']=docs['PREFLIGHT']
            with self.assertRaises(ValueError):r.validate_documents(docs,p)

    def test_returned_recovery_cannot_claim_original_stdout_or_new_preflight(self):
        for field,value in [('original_stdout_recovered',True),('original_ssh_returncode',0),('preflight_invocations',1),('original_error_absent',False)]:
            pub=copy.deepcopy(self.pub);pub['recovery_evidence'][field]=value
            with self.assertRaises(ValueError):r.validate_publication(pub,self.p)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='satdose-prep-guards-OFFLINE-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()/'prepared';self.root.mkdir()
        self.patch=patch.object(r,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)

    def test_error_and_actual_target_occupied_rejected(self):
        for relative in ('execution_releases/'+r.PREFIX+'_ERROR.json','continuation_control/start-11031/T0/chunk-009-016'):
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'{}')
            with self.assertRaises(ValueError):r.guard()
            path.unlink()

    def test_scheduling_unknown_target_rejected(self):
        path=self.root.parent/'scheduling/continuation_71018e5/start-11031/T0/chunk-009-016';path.mkdir(parents=True)
        with self.assertRaises(ValueError):r.guard()

    def test_unclosed_begin_and_symlink_rejected(self):
        path=self.root/'continuation_control/start-11031/T0/chunk-003-008/audit';path.mkdir(parents=True)
        (path/'x.BEGIN.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):r.guard()
        (path/'x.CAPTURE.json').write_bytes(b'{}');(path/'x.COMPLETE.json').write_bytes(b'{}');r.guard()
        (path/'foreign').symlink_to(self.root)
        with self.assertRaises(ValueError):r.guard()

    def test_unknown_controller_leaf_population_rejected(self):
        parent=self.root/'trained_dev';leaf=parent/'start-11031/T0/step-002';leaf.mkdir(parents=True)
        r.exact_population(parent,{'start-11031/T0/step-002'})
        (leaf.parent/'step-008').mkdir()
        with self.assertRaises(ValueError):r.exact_population(parent,{'start-11031/T0/step-002'})

    def test_exclusive_lock_rejects_original_shared_preparer_and_replaced_inode(self):
        path=self.root/'DIAGNOSTIC_WORKER.lock';path.write_bytes(b'')
        fd=r.os.open(path,r.os.O_RDONLY)
        try:
            r.fcntl.flock(fd,r.fcntl.LOCK_SH|r.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                with r.existing_lock('DIAGNOSTIC_WORKER.lock',r.fcntl.LOCK_EX):pass
        finally:r.os.close(fd)
        with self.assertRaises(ValueError):
            with r.existing_lock('DIAGNOSTIC_WORKER.lock',r.fcntl.LOCK_EX):
                path.rename(path.with_name('old'));path.write_bytes(b'')

    def test_source84_execution19_requires_exact_bytes_and_no_extra_files(self):
        code=self.root/'train_code_71018e5';code.mkdir();sources={}
        for n in range(84):
            path=code/f'module{n}.py';path.write_bytes(f'# OFFLINE {n}\n'.encode());sources[path.name]=r.sha(path)
        p=dict(sources=sources,execution_sources={f'key{n}':dict(path=str(code/f'module{n}.py'),sha256=sources[f'module{n}.py']) for n in range(19)})
        with patch.object(r,'CODE',code):
            r.source_inventory(p)
            (code/'module0.py').write_bytes(b'# changed\n')
            with self.assertRaises(ValueError):r.source_inventory(p)
            (code/'module0.py').write_bytes(b'# OFFLINE 0\n');(code/'unexpected').write_bytes(b'')
            with self.assertRaises(ValueError):r.source_inventory(p)


class DownloadDriver(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='satdose-recovered-prep-OFFLINE-');self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name).resolve()/'recovery'
        self.p,self.docs,self.raw,self.pins=synthetic_fixture()
        self.patch=patch.object(r,'FIVE',self.pins);self.patch.start();self.addCleanup(self.patch.stop)
        self.pub=publication_fixture(self.p,self.docs,self.raw)

    def argv(self):return ['--failed-receipt',str(FAILED),'--failed-sha256',r.FAILED_SHA,'--local-dir',str(self.directory),'--execute']

    def transport(self,returncode=0,raw=None):
        def run(argv,**kwargs):
            self.assertEqual(argv[0],'ssh');self.assertLess(len(argv[-1]),1500)
            self.assertTrue((self.directory/'RECOVERY_SOURCE.py').is_file())
            kwargs['stdout'].write(raw if raw is not None else r.canonical(self.pub));kwargs['stderr'].write(b'OFFLINE recovery transport\n')
            return SimpleNamespace(returncode=returncode)
        return run

    def test_full_synthetic_recovery_passes_unchanged_legacy_parser_and_preserves_raw_origin(self):
        with patch.object(r,'local_payload',return_value=self.p),patch.object(r.subprocess,'run',side_effect=self.transport()),redirect_stdout(io.StringIO()):
            self.assertEqual(r.main(self.argv()),0)
        receipt_path=self.directory/'RECOVERED_PREPARATION_RECEIPT.json';receipt=r.loads(receipt_path.read_bytes())
        self.assertEqual(receipt['returncode'],0);self.assertEqual(receipt['original_ssh_returncode'],255)
        self.assertEqual(receipt['original_failure']['stdout'],'');self.assertFalse(receipt['original_stdout_recovered'])
        self.assertEqual((self.directory/'RECOVERY_STDOUT.log').read_bytes(),r.canonical(self.pub))
        self.assertEqual(r.loads(receipt['stdout']),receipt['publication'])
        r.load_submitter().payload_from_receipt(receipt_path,r.sha(receipt_path))
        self.assertTrue((self.directory/'RECOVERY_VALIDATION.json').is_file())
        for key,raw in self.raw.items():self.assertEqual((self.directory/(key+'.json')).read_bytes(),raw)

    def test_network_unknown_retains_raw_and_no_usable_receipt(self):
        with patch.object(r,'local_payload',return_value=self.p),patch.object(r.subprocess,'run',side_effect=self.transport(255,b'partial\xff')),redirect_stdout(io.StringIO()):
            self.assertEqual(r.main(self.argv()),2)
            with self.assertRaises(ValueError):r.main(self.argv())
        self.assertEqual((self.directory/'RECOVERY_STDOUT.log').read_bytes(),b'partial\xff')
        self.assertFalse((self.directory/'RECOVERED_PREPARATION_RECEIPT.json').exists())
        self.assertFalse((self.directory/'RECOVERY_VALIDATION.json').exists())

    def test_legacy_format_failure_is_failed_and_final_receipt_absent(self):
        fake=SimpleNamespace(payload_from_receipt=lambda *args: (_ for _ in ()).throw(ValueError('OFFLINE legacy failure')))
        with patch.object(r,'local_payload',return_value=self.p),patch.object(r,'load_submitter',return_value=fake),\
             patch.object(r.subprocess,'run',side_effect=self.transport()),redirect_stdout(io.StringIO()):
            self.assertEqual(r.main(self.argv()),2)
        self.assertFalse((self.directory/'RECOVERED_PREPARATION_RECEIPT.json').exists())
        self.assertFalse((self.directory/'RECOVERY_VALIDATION.json').exists())
        self.assertEqual(r.loads((self.directory/'RECOVERY_ERROR.json').read_bytes())['status'],'READ_ONLY_RECOVERY_FAILED_OR_UNKNOWN_NO_RETRY')


if __name__=='__main__':unittest.main()
