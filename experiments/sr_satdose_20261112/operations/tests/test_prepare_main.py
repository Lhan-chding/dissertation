"""Offline MAIN preparer boundary tests; fabricated envelopes are not experiment evidence."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

OPERATOR=Path(__file__).resolve().parents[1]/'satdose_prepare_main.py'
M=runpy.run_path(str(OPERATOR));G=M['main'].__globals__
PUBLISHER=OPERATOR.with_name('satdose_publish_actual_terminal.py')
PUB=runpy.run_path(str(PUBLISHER))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_bytes(value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True)+'\n').encode())
def terminal_fixture(directory):
    """Only envelope shape is valid. Actual terminal tensor/risk evidence is absent."""
    root=M['ROOT'];terminal=M['TERMINAL'];names=PUB['FILES']-{'ERROR.json'}
    for name in names-{'PUBLICATION_RECEIPT.json'}:save(directory/name,dict(OFFLINE_FAKE_ONLY=True))
    save(directory/'PUBLISHER_SOURCE.py',PUBLISHER.read_bytes())
    save(directory/'PAYLOAD.json',dict(publisher_sha256=M['PUBLISHER_SHA']))
    save(directory/'CLAIM.json',dict(publisher_sha256=M['PUBLISHER_SHA']))
    def reference(name):return dict(path=str((terminal/name).relative_to(root)),sha256=digest(directory/name))
    save(directory/'TERMINAL_ADMISSION.json',dict(status='TRAINING_TERMINAL_VERIFIED_MAIN_ALLOWED',permission='FIXED_MAIN_MANIFEST_ONLY_NO_TRAINING'))
    save(directory/'TERMINAL_SPEC.json',dict(published=reference('TERMINAL_ADMISSION.json'),report=reference('INDEPENDENT_TERMINAL_AUDIT.json'),audit_spec=reference('AUDIT_SPEC.json')))
    receipt=dict(status='ACTUAL_TERMINAL_PUBLISHED_AND_INDEPENDENTLY_REVERIFIED',permission='PUBLISHED_TERMINAL_AUTHORITY_ONLY_NO_JOB_SUBMISSION',
      independent_terminal_audit_runs=2,model_calls=0,optimizer_calls=0,publisher_sha256=M['PUBLISHER_SHA'],
      original_train_release=M['ORIGINAL'],canary=M['CANARY'],terminal_spec=reference('TERMINAL_SPEC.json'),
      files={name:reference(name) for name in names-{'PUBLICATION_RECEIPT.json'}})
    save(directory/'PUBLICATION_RECEIPT.json',receipt)
    files={name:dict(path=str(terminal/name),sha256=digest(directory/name),bytes=(directory/name).stat().st_size) for name in names}
    manifest=dict(schema='sr-satdose-terminal-operator-manifest-v1',status='ACTUAL_TERMINAL_PUBLICATION_COMPLETE',success=True,
      publication=str(terminal),files=files,completion_validation_error=None)
    download=dict(status='EXACT_TERMINAL_ARTIFACT_BYTES_DOWNLOADED',publication_success=True,remote_returncode=0,
      publisher_sha256=M['PUBLISHER_SHA'],files={name:dict(value,local_path=str(directory/name)) for name,value in files.items()})
    save(directory/'REMOTE_MANIFEST.json',manifest);save(directory/'LOCAL_DOWNLOAD_RECEIPT.json',download)
    return manifest,download


class PreparationEnvelope(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='satdose-main-prep-OFFLINE-');self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name).resolve();self.manifest,self.download=terminal_fixture(self.directory)
    def validate(self):return M['local_terminal'](self.directory,digest(self.directory/'REMOTE_MANIFEST.json'),digest(self.directory/'LOCAL_DOWNLOAD_RECEIPT.json'))
    def save(self):
        save(self.directory/'REMOTE_MANIFEST.json',self.manifest);save(self.directory/'LOCAL_DOWNLOAD_RECEIPT.json',self.download)
    def test_syntax_and_stable_publisher_pin(self):
        compile(OPERATOR.read_text(),str(OPERATOR),'exec');self.assertEqual(digest(PUBLISHER),M['PUBLISHER_SHA'])
    def test_exact_terminal_download_envelope(self):self.assertEqual(self.validate(),self.manifest)
    def test_partial_spec_admission_never_enough(self):
        self.manifest['files']={k:v for k,v in self.manifest['files'].items() if k in ('TERMINAL_SPEC.json','TERMINAL_ADMISSION.json')}
        self.download['files']={k:v for k,v in self.download['files'].items() if k in self.manifest['files']};self.save()
        with self.assertRaises(PermissionError):self.validate()
    def test_missing_receipt_never_accepted(self):
        (self.directory/'PUBLICATION_RECEIPT.json').unlink()
        with self.assertRaises((PermissionError,FileNotFoundError)):self.validate()
    def test_error_overrides_complete_status(self):
        save(self.directory/'ERROR.json',dict(error='OFFLINE'))
        with self.assertRaisesRegex(PermissionError,'ERROR'):self.validate()
    def test_local_download_error_overrides_complete_status(self):
        save(self.directory/'LOCAL_ERROR.json',dict(error='OFFLINE'))
        with self.assertRaisesRegex(PermissionError,'ERROR'):self.validate()
    def test_manifest_unknown_rejected(self):
        self.manifest['success']=False;self.save()
        with self.assertRaises(PermissionError):self.validate()
    def test_failed_download_rejected(self):
        self.download['remote_returncode']=2;self.save()
        with self.assertRaises(PermissionError):self.validate()
    def test_mutated_artifact_bytes_rejected(self):
        (self.directory/'FINAL_RISK.json').write_bytes(b'tampered')
        with self.assertRaises(PermissionError):self.validate()
    def test_wrong_manifest_size_rejected(self):
        self.manifest['files']['FINAL_RISK.json']['bytes']+=1
        self.download['files']['FINAL_RISK.json']['bytes']+=1;self.save()
        with self.assertRaises(PermissionError):self.validate()
    def test_manifest_hash_mismatch_rejected(self):
        with self.assertRaises(PermissionError):M['local_terminal'](self.directory,'0'*64,digest(self.directory/'LOCAL_DOWNLOAD_RECEIPT.json'))
    def test_fixed_fourteen_order_and_exact_total(self):
        self.assertEqual(len(M['ORDER']),14);self.assertEqual(sum(item[3] for item in M['ORDER']),13760)
        self.assertEqual(M['ORDER'][:2],((11031,'START',0,1952),(11032,'START',0,1952)))
        self.assertEqual(sum(item[3]==1952 for item in M['ORDER']),6);self.assertEqual(sum(item[3]==256 for item in M['ORDER']),8)
    def test_invalid_endpoint_rejected(self):
        for values in ((11031,'START',8),(11031,'T0',0),(11031,'T0',2),(11033,'START',0)):
            with self.assertRaises(PermissionError):M['selection'](*values)
    def test_plan_no_network_or_output_write(self):
        output=self.directory/'NO_OUTPUT.json'
        args=['--terminal-dir',str(self.directory),'--terminal-manifest-sha256',digest(self.directory/'REMOTE_MANIFEST.json'),
          '--terminal-download-receipt-sha256',digest(self.directory/'LOCAL_DOWNLOAD_RECEIPT.json'),
          '--start-seed','11031','--arm','START','--step','0','--receipt',str(output)]
        with patch.object(subprocess,'run',side_effect=AssertionError('Network in plan')),redirect_stdout(io.StringIO()) as text:
            self.assertEqual(M['main'](args),0)
        self.assertEqual(json.loads(text.getvalue())['responses'],1952);self.assertFalse(output.exists())
    def test_network_unknown_preserves_source_and_raw_bytes_no_retry(self):
        output=self.directory/'mock_operator.json';source=OPERATOR.read_bytes();calls=[]
        def fake(argv,**kw):
            calls.append(argv);self.assertEqual(argv[0],'ssh')
            self.assertEqual(output.with_name(output.name+'.source.py').read_bytes(),source)
            self.assertTrue(output.with_name(output.name+'.intent.json').is_file())
            kw['stdout'].write(b'OFFLINE FAILURE\xff\n');kw['stderr'].write(b'OFFLINE STDERR\n');return SimpleNamespace(returncode=255)
        payload=dict(source_sha256=hashlib.sha256(source).hexdigest())
        with patch.object(subprocess,'run',side_effect=fake),redirect_stdout(io.StringIO()):
            self.assertEqual(M['invoke_remote'](payload,source,output,'remote_prepare'),2)
            with self.assertRaises(PermissionError):M['invoke_remote'](payload,source,output,'remote_prepare')
        self.assertEqual(len(calls),1);receipt=json.loads(output.read_text());self.assertEqual(receipt['status'],'MAIN_OPERATOR_UNKNOWN_NO_RETRY')
        self.assertEqual(Path(receipt['raw_logs']['stdout']['path']).read_bytes(),b'OFFLINE FAILURE\xff\n')
    def test_no_implicit_model_or_scheduler_calls(self):
        tree=ast.parse(OPERATOR.read_text());calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
        self.assertFalse(set(calls)&{'execute','build_evaluation_runtime','generate_one','build_training_runtime'})
        self.assertNotIn('sbatch',OPERATOR.read_text())
    def test_prefix_blocks_skip_or_occupied_target(self):
        endpoint,_=M['selection'](11031,'START',0)
        self.assertEqual(M['verify_prefix'](SimpleNamespace(_existing_slots=lambda:{}),endpoint),{})
        with self.assertRaises(PermissionError):M['verify_prefix'](SimpleNamespace(_existing_slots=lambda:{'11031/START/0':self.directory}),endpoint)
        endpoint,_=M['selection'](11032,'START',0)
        with self.assertRaises(PermissionError):M['verify_prefix'](SimpleNamespace(_existing_slots=lambda:{}),endpoint)


if __name__=='__main__':unittest.main(verbosity=2)
