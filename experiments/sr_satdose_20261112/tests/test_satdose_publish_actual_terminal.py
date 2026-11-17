"""Offline operator tests; no SSH, GPU, real terminal audit or future PASS.

Remote authority/auditor fixtures below are explicitly synthetic composition
stubs. The immutable native terminal audit/authority have separate integration
suites; these tests exercise invocation, failure retention, locks and transport.
"""
import argparse
from contextlib import contextmanager
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

PACKAGE=Path(__file__).resolve().parents[1]
SOURCE=PACKAGE/'operations/satdose_publish_actual_terminal.py'
# Exact offline archive copies sit next to PREFIX_SOURCE.py, so the archived
# suite is directly rerunnable against the preserved operator bytes.
ARCHIVED_SOURCE=Path(__file__).with_name(Path(__file__).stem.removesuffix('_TESTS')+'_SOURCE.py')
if ARCHIVED_SOURCE.is_file():SOURCE=ARCHIVED_SOURCE
spec=importlib.util.spec_from_file_location('_satdose_terminal_operator_test',SOURCE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(c.canonical(value))

def args(**options):
    value=dict(execute=False,download=False,ssh_host=None,local_dir=None)
    for name,(path,digest) in c.LOCAL_INPUTS.items():value[name]=path;value[name+'_sha256']=digest
    value.update(options);return argparse.Namespace(**value)


@unittest.skipUnless(all(path.is_file() for path,_ in c.LOCAL_INPUTS.values()),'Recorded original local intake copies required')
class LocalPlanAndTransportTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name).resolve()
    def test_default_plan_actual_fixed_intakes_no_network_no_future_state(self):
        with patch.object(c.subprocess,'run',side_effect=AssertionError('no network')),patch('sys.stdout',new_callable=io.StringIO) as stdout:
            self.assertEqual(c.main([]),0)
        result=json.loads(stdout.getvalue());self.assertEqual(result['actual_terminal_state'],'NOT_OBSERVED_BY_LOCAL_PLAN')
        self.assertEqual(result['required_actual_counts'],dict(COMMIT=96,TRAIN=12288,DEV_endpoints=14,DEV=1512,chunks=12))
        self.assertEqual(result['canary']['sha256'],'44a8db68c655bea987629ea210da1a32e268806f6e0c7b1506d6d77828d4b260')
    def test_changed_local_pin_and_forged_prefix_source_map_rejected(self):
        changed=args(train_intake_sha256='0'*64)
        with self.assertRaisesRegex(PermissionError,'Pinned local'):c.local_payload(changed)
        payload,_=c.local_payload(args());payload['original_release_record']['release']['execution_sources'].pop('continuation_control')
        with self.assertRaisesRegex(PermissionError,'TRAIN source'):c._pin_metadata(payload)
    def test_real_subprocess_default_plan_still_local_only(self):
        result=subprocess.run([sys.executable,'-B',str(SOURCE)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'LOCAL_TERMINAL_PLAN_ONLY_NO_NETWORK_NO_PUBLICATION')
    def test_command_pins_source_hides_cuda_and_has_no_scheduler(self):
        payload,source=c.local_payload(args());command=c._command('CPU.fixture',c.sha_bytes(source),payload)
        remote=shlex.split(command[-1]);self.assertEqual(command[0],'ssh')
        self.assertIn('CUDA_VISIBLE_DEVICES=',remote);self.assertIn('-I',remote);self.assertIn('-B',remote)
        self.assertIn('--remote-worker',remote);self.assertIn(c.sha_bytes(source),command[-1])
        self.assertNotIn('sbatch',remote);self.assertNotIn('srun',remote)
        self.assertIn('--remote-download',shlex.split(c._command('CPU.fixture',c.sha_bytes(source),payload,True)[-1]))
        for host in ('-oProxyCommand=bad','host;bad','host\nother'):
            with self.assertRaises(PermissionError):c._command(host,'0'*64,payload)
    def test_real_child_source_bootstrap_fails_closed_before_server_or_model_access(self):
        payload,source=c.local_payload(args());remote=shlex.split(c._command('CPU.fixture',c.sha_bytes(source),payload)[-1])
        index=remote.index(c.PYTHON)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        # Run the exact pinned-source bootstrap locally, never the SSH command.
        # Its production-interpreter guard must fail before any root or GPU use.
        result=subprocess.run([sys.executable,*remote[index+1:]],input=source,capture_output=True,env=env)
        self.assertEqual(result.returncode,2)
        records=[line for line in result.stdout.splitlines() if line.startswith(c.PREFIX)]
        self.assertEqual(len(records),1);value=json.loads(records[0][len(c.PREFIX):])
        self.assertFalse(value['success']);self.assertEqual(value['files'],{})
        self.assertIn('Wrong production CPU interpreter',value['error'])
    def test_ssh_failure_retains_source_payload_intent_rawlogs_and_no_retry(self):
        directory=self.root/'failed'
        def failed(command,**kw):
            kw['stdout'].write(b'no remote manifest\n');kw['stderr'].write(b'CPU simulated connection failure\n')
            return SimpleNamespace(returncode=255)
        with patch.object(c.subprocess,'run',side_effect=failed) as run:
            with self.assertRaisesRegex(PermissionError,'manifest'):c.local(args(execute=True,ssh_host='CPU.fixture',local_dir=directory))
            self.assertEqual(run.call_count,1)
        for name in ('OPERATOR_SOURCE.py','LOCAL_PAYLOAD.json','LOCAL_INTENT.json','REMOTE_STDOUT.log','REMOTE_STDERR.log','LOCAL_ERROR.json'):
            self.assertTrue((directory/name).is_file(),name)
        self.assertEqual((directory/'OPERATOR_SOURCE.py').read_bytes(),SOURCE.read_bytes())
        with patch.object(c.subprocess,'run') as run:
            with self.assertRaisesRegex(PermissionError,'Existing local'):c.local(args(execute=True,ssh_host='CPU.fixture',local_dir=directory))
            run.assert_not_called()
    def test_download_only_exact_bytes_retains_negative_artifacts(self):
        directory=self.root/'download';data=c.canonical({'status':'CPU_SYNTHETIC_FAILURE','error':'retained'})
        remote=dict(schema='sr-satdose-terminal-operator-manifest-v1',status='HOLD_OR_UNKNOWN_TERMINAL_PUBLICATION_RETAINED',
            publication=str(c.PUBLICATION),success=False,
            files={'ERROR.json':dict(path=str(c.PUBLICATION/'ERROR.json'),sha256=c.sha_bytes(data),bytes=len(data))})
        commands=[]
        def run(command,**kw):
            commands.append(command)
            kw['stdout'].write(data if shlex.split(command[-1])[0]=='cat' else c.PREFIX+c.canonical(remote))
            return SimpleNamespace(returncode=0)
        with patch.object(c.subprocess,'run',side_effect=run),patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(c.local(args(download=True,ssh_host='CPU.fixture',local_dir=directory)),2)
        self.assertIn('--remote-download',shlex.split(commands[0][-1]));self.assertNotIn('--remote-worker',shlex.split(commands[0][-1]))
        self.assertEqual((directory/'ERROR.json').read_bytes(),data)
        result=c.read(directory/'LOCAL_DOWNLOAD_RECEIPT.json');self.assertFalse(result['publication_success'])
        self.assertTrue((directory/'ERROR.json.DOWNLOAD_STDERR').exists())
    def test_mismatched_or_escaped_download_never_becomes_success(self):
        directory=self.root/'download';directory.mkdir()
        remote=dict(schema='sr-satdose-terminal-operator-manifest-v1',publication=str(c.PUBLICATION),success=False,
                    files={'ERROR.json':dict(path='/tmp/escape',sha256='0'*64,bytes=3)})
        with patch.object(c.subprocess,'run') as run:
            with self.assertRaisesRegex(PermissionError,'escapes'):c._download(directory,'CPU.fixture',remote)
            run.assert_not_called()
        remote['files']['ERROR.json']['path']=str(c.PUBLICATION/'ERROR.json')
        def bad(command,**kw):kw['stdout'].write(b'bad');return SimpleNamespace(returncode=0)
        with patch.object(c.subprocess,'run',side_effect=bad):
            with self.assertRaisesRegex(PermissionError,'nonidentical'):c._download(directory,'CPU.fixture',remote)
        self.assertFalse((directory/'ERROR.json').exists());self.assertTrue((directory/'ERROR.json.DOWNLOAD_STDOUT').exists())


class ActualPopulationBoundaryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name).resolve()
        for p in (patch.object(c,'ROOT',self.root),patch.object(c,'PUBLICATION',self.root/'execution_releases/terminal_4190bc3')):
            p.start();self.addCleanup(p.stop)
        self.control=SimpleNamespace(_hold_free=Mock(),_cursor=Mock(),_completed_attempts=Mock())
        self.ctx=dict(modules={'execution.continuation_control':self.control},original={'execution_sources':{'risk_verifier':{'sha256':'a'*64}},'risk_clarification':{}})
        for key in c.PATHS:
            start,arm=key.split('/')
            for step in range(1,25):dump(self.root/f'training_transactions/start-{start}/{arm}/step-{step:03d}/COMMIT.json',{'CPU_FAKE':True,'step':step})
            for step in (2,8,16):dump(self.root/f'trained_dev/start-{start}/{arm}/step-{step:03d}/RESULT.json',
                dict(status='TRAINED_DEV_ENDPOINT_COMPLETE',responses=108,global_scientific_hold=False,receipt={'CPU':True},endpoint_binding={'CPU':True}))
            for a,b in ((3,8),(9,16),(17,24)):dump(self.root/f'continuation_control/start-{start}/{arm}/chunk-{a:03d}-{b:03d}/RESULT.json',
                dict(status='CONTINUATION_CHUNK_COMPLETE',main_authorized=False))
        for seed in (11031,11032):dump(self.root/f'start_dev/start-{seed}/RESULT.json',
            dict(status='START_DEV_ENDPOINT_COMPLETE',responses=108,receipt={'CPU':True},endpoint_binding={'CPU':True}))
        freeze=self.root/'DESIGN_FREEZE.json';dump(freeze,{'artifacts':{'config':{'CPU':True}}})
        self.ctx['original']['design_freeze_sha256']=c.sha(freeze)
        path=self.root/'CPU_CANARY.json';dump(path,{'design_freeze':c.ref(freeze)})
        patcher=patch.object(c,'CANARY',c.ref(path));patcher.start();self.addCleanup(patcher.stop)
    def test_exact_population_constructs96commits14dev12chunks_without_audit_or_write(self):
        control,inputs,chunks=c.terminal_population(self.ctx)
        self.assertEqual(sum(map(len,inputs['trusted_commits'].values())),96)
        self.assertEqual(len(inputs['dev_endpoints']),14);self.assertEqual(len(chunks),12)
        self.control._completed_attempts.assert_called_once();self.assertFalse(c.PUBLICATION.exists())
    def test_step23_or_extra_unknown_transaction_blocks_before_publication(self):
        directory=self.root/'training_transactions/start-11031/T0';last=directory/'step-024/COMMIT.json';last.unlink();last.parent.rmdir()
        with self.assertRaisesRegex(PermissionError,'Exactly24'):c.terminal_population(self.ctx)
        dump(last,{});(directory/'.staging-step-025').mkdir()
        with self.assertRaisesRegex(PermissionError,'Exactly24'):c.terminal_population(self.ctx)
    def test_missing_dev_chunk_or_true_hold_blocks(self):
        path=self.root/'trained_dev/start-11031/T0/step-002/RESULT.json';value=c.read(path)
        value['global_scientific_hold']=True;dump(path,value)
        with self.assertRaisesRegex(PermissionError,'incomplete/HOLD'):c.terminal_population(self.ctx)
        value['global_scientific_hold']=False;dump(path,value)
        (self.root/'continuation_control/start-11032/T1/chunk-017-024/RESULT.json').unlink()
        with self.assertRaises(FileNotFoundError):c.terminal_population(self.ctx)
        self.assertFalse(c.PUBLICATION.exists())
    def test_existing_shared_leases_block_exclusive_worker_and_detect_replacement(self):
        import fcntl
        path=self.root/'DIAGNOSTIC_WORKER.lock';path.touch()
        with c.idle_lock('DIAGNOSTIC_WORKER.lock'):
            fd=os.open(path,os.O_RDONLY)
            try:
                with self.assertRaises(BlockingIOError):fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:os.close(fd)
        with self.assertRaisesRegex(PermissionError,'changed'):
            with c.idle_lock('DIAGNOSTIC_WORKER.lock'):
                path.rename(self.root/'old.lock');path.touch()


class PublicationCompositionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name).resolve()
        self.train=self.root/'train';self.external=self.root/'external';self.train.mkdir();self.external.mkdir()
        for p in (patch.object(c,'ROOT',self.root),patch.object(c,'TRAIN',self.train),patch.object(c,'EXTERNAL',self.external),
                  patch.object(c,'PUBLICATION',self.root/'execution_releases/terminal_4190bc3')):
            p.start();self.addCleanup(p.stop)
        c.PUBLICATION.parent.mkdir()
        for name in ('DIAGNOSTIC_WORKER.lock','FIRST_TWO_LAUNCHER.lock'):(self.root/name).touch()
        self.source=b'# EXPLICIT CPU SYNTHETIC OPERATOR SOURCE\n';self.payload={'publisher_sha256':c.sha_bytes(self.source)}
        self.control=SimpleNamespace(_hold_free=Mock(),_cursor=Mock())
        self.risk=dict(status='NO_SCIENTIFIC_STOP_TRIGGER_OBSERVED',global_hold=False,hold_reasons=[],pending_required_dev=[],CPU_FIXTURE=True)
        self.report=dict(status='CPU_SYNTHETIC_AUDIT',CPU_FIXTURE=True)
        self.candidate=dict(status='TRAINING_TERMINAL_VERIFIED_MAIN_ALLOWED',permission='FIXED_MAIN_MANIFEST_ONLY_NO_TRAINING',CPU_FIXTURE=True)
        self.calls=[];self.fail_prepare=False;self.fail_verify=False
        def audit(spec):self.calls.append('audit');return copy.deepcopy(self.report)
        def prepare(spec):
            self.calls.append('prepare');actual=audit(c.read(c.bound(spec['audit_spec'])))
            if self.fail_prepare:raise PermissionError('CPU synthetic second audit mismatch')
            self.assertEqual(c.read(c.bound(spec['report'])),actual)
            return copy.deepcopy(self.candidate)
        def verify(spec):
            self.calls.append('verify');self.assertEqual(c.read(c.bound(spec['published'])),self.candidate)
            if self.fail_verify:raise PermissionError('CPU synthetic published verification failure')
            return SimpleNamespace(stable=lambda:self.calls.append('stable'))
        self.ctx=dict(original={'sources':{},'admission_spec':{},'admission':{}},original_ref={'CPU_ORIGINAL':True},
            modules={'analysis.risk':SimpleNamespace(assess=lambda *a,**kw:copy.deepcopy(self.risk))},
            external={'sources':{}},authority_ref={'CPU_AUTHORITY':True},auditor_ref={'CPU_AUDITOR':True},
            auditor=SimpleNamespace(audit=audit),authority=SimpleNamespace(prepare=prepare,verify=verify))
        inputs=dict(trusted_commits={key:[{'CPU_COMMIT':n} for n in range(24)] for key in c.PATHS},
                    dev_endpoints={str(n):{} for n in range(14)})
        for p in (patch.object(c,'bootstrap',return_value=self.ctx),
                  patch.object(c,'terminal_population',return_value=(self.control,inputs,{str(n):{} for n in range(12)}))):
            p.start();self.addCleanup(p.stop)
    def run_once(self):return c.remote_worker(self.payload,self.source)
    def test_two_audit_invocations_actual_returned_candidate_and_reverification(self):
        result=self.run_once();self.assertTrue(result['success']);self.assertEqual(self.calls,['audit','prepare','audit','verify','stable'])
        self.assertEqual(c.read(c.PUBLICATION/'TERMINAL_ADMISSION.json'),self.candidate)
        self.assertEqual(c.read(c.PUBLICATION/'INDEPENDENT_TERMINAL_AUDIT.json'),self.report)
        self.assertEqual((c.PUBLICATION/'PUBLISHER_SOURCE.py').read_bytes(),self.source)
        self.assertEqual(c.read(c.PUBLICATION/'PUBLICATION_RECEIPT.json')['independent_terminal_audit_runs'],2)
        self.assertEqual(c.manifest(success=False)['status'],'HOLD_OR_UNKNOWN_TERMINAL_PUBLICATION_RETAINED')
        with self.assertRaisesRegex(PermissionError,'Existing terminal'):self.run_once()
        self.assertEqual(self.calls.count('audit'),2)
    def test_final_risk_hold_preserved_before_any_terminal_audit_or_admission(self):
        self.risk.update(status='GLOBAL_SCIENTIFIC_HOLD',global_hold=True,hold_reasons=['CPU observed hold'])
        result=self.run_once();self.assertFalse(result['success']);self.assertEqual(self.calls,[])
        self.assertEqual(c.read(c.PUBLICATION/'FINAL_RISK.json')['hold_reasons'],['CPU observed hold'])
        self.assertTrue((c.PUBLICATION/'ERROR.json').exists());self.assertFalse((c.PUBLICATION/'TERMINAL_ADMISSION.json').exists())
    def test_failed_second_audit_never_publishes_candidate(self):
        self.fail_prepare=True;result=self.run_once();self.assertFalse(result['success'])
        self.assertTrue((c.PUBLICATION/'INDEPENDENT_TERMINAL_AUDIT.json').exists())
        self.assertFalse((c.PUBLICATION/'TERMINAL_ADMISSION.json').exists());self.assertFalse((c.PUBLICATION/'PUBLICATION_RECEIPT.json').exists())
        with self.assertRaises(PermissionError):self.run_once()
    def test_failed_final_verification_retains_candidate_but_no_success(self):
        self.fail_verify=True;result=self.run_once();self.assertFalse(result['success'])
        self.assertTrue((c.PUBLICATION/'TERMINAL_ADMISSION.json').exists());self.assertTrue((c.PUBLICATION/'ERROR.json').exists())
        self.assertFalse((c.PUBLICATION/'PUBLICATION_RECEIPT.json').exists())
    def test_lease_exit_failure_invalidates_even_previously_written_receipt(self):
        original=c.idle_lock
        @contextmanager
        def failure(name):
            with original(name):yield
            if name=='DIAGNOSTIC_WORKER.lock':raise PermissionError('CPU synthetic lease replaced at exit')
        with patch.object(c,'idle_lock',side_effect=failure):result=self.run_once()
        self.assertFalse(result['success']);self.assertTrue((c.PUBLICATION/'ERROR.json').exists())
        self.assertFalse(c.manifest()['success'])
    def test_no_existing_attempt_or_wrong_source_can_be_retried(self):
        c.PUBLICATION.mkdir();dump(c.PUBLICATION/'CLAIM.json',{'UNKNOWN':True})
        with self.assertRaisesRegex(PermissionError,'Existing terminal'):self.run_once()
        self.assertFalse((c.PUBLICATION/'ERROR.json').exists());self.assertFalse(self.calls)
    def test_truncated_receipt_remains_downloadable_unknown(self):
        self.assertTrue(self.run_once()['success'])
        path=c.PUBLICATION/'PUBLICATION_RECEIPT.json';path.write_bytes(b'{"status":')
        result=c.manifest();self.assertFalse(result['success'])
        self.assertIn('PUBLICATION_RECEIPT.json',result['files']);self.assertIsNotNone(result['completion_validation_error'])
        with self.assertRaises(PermissionError):self.run_once()
    def test_missing_or_replaced_referenced_artifact_is_not_completed_publication(self):
        self.assertTrue(self.run_once()['success']);path=c.PUBLICATION/'FINAL_RISK.json';saved=path.read_bytes()
        path.unlink();result=c.manifest();self.assertFalse(result['success'])
        self.assertIn('PUBLICATION_RECEIPT.json',result['files'])
        path.write_bytes(saved+b' ');result=c.manifest();self.assertFalse(result['success'])
        self.assertIn('FINAL_RISK.json',result['files'])
    def test_downloaded_success_envelope_rechecked_from_local_bytes(self):
        result=self.run_once();destination=self.root/'downloaded';destination.mkdir()
        for name in result['files']:(destination/name).write_bytes((c.PUBLICATION/name).read_bytes())
        receipt=c.validate_publication(destination,result['files'])
        self.assertEqual(receipt['status'],'ACTUAL_TERMINAL_PUBLISHED_AND_INDEPENDENTLY_REVERIFIED')
        (destination/'TERMINAL_SPEC.json').write_bytes(b'{}')
        with self.assertRaisesRegex(PermissionError,'changed'):c.validate_publication(destination,result['files'])
    def test_failed_before_claim_preserves_no_fake_terminal_publication(self):
        with patch.object(c,'terminal_population',side_effect=PermissionError('CPU incomplete96')):
            with self.assertRaisesRegex(PermissionError,'incomplete96'):self.run_once()
        self.assertFalse(c.PUBLICATION.exists());self.assertFalse(self.calls)


if __name__=='__main__':unittest.main()
