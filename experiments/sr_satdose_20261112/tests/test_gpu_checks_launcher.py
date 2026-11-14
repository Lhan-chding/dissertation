"""CPU-only shell order + isolated stdlib bootstrap with explicit filesystem seams.

No sbatch/srun/GPU/module is executed. Production constants have no override;
only the extracted Python bootstrap in this test is relocated to synthetic files.
"""
from contextlib import redirect_stdout,redirect_stderr
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
SCRIPT=PACKAGE/'scheduling/run_gpu_checks.sh'
ROOT_TEXT='/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1'
ANCESTOR_TEXT='/projects/_ssd/varunssd'
FREEZE='6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
EXPERIMENT='SR-SATDOSE-1-20261112'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,sort_keys=True))


class GPUChecksLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.temp=Path(self.tmp.name).resolve();self.ancestor=self.temp/'ceph';self.root=self.ancestor/'project/prepared_data_v1'
        self.root.mkdir(parents=True);self.release=self.ancestor/'release';self.release.mkdir()
        files=('execution/evaluation_probe.py','execution/evaluation_loader.py','execution/backend_equivalence.py',
            'execution/allzero_check.py','execution/diagnostic_worker.py','execution/shadow_replay.py','execution/production.py',
            'execution/training_state.py','execution/training_transaction.py','execution/diagnostic_inputs.py',
            'scheduling/run_gpu_checks.sh','analysis/calibration.py','supplied_package/config/source_bindings.json',
            'supplied_package/tools/satdose_core.py')
        files+=tuple('runtime/'+p.name for p in (PACKAGE/'runtime').glob('*.py'))
        for name in files:
            path=self.release/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PACKAGE/name,path)
        dump(self.root/'config.json',{'CPU_FAKE':True});dump(self.root/'source_lock.json',{'CPU_FAKE':True})
        self.freeze=self.root/'DESIGN_FREEZE.json'
        dump(self.freeze,dict(schema='sr-satdose-design-freeze-v1',experiment_id=EXPERIMENT,
            artifacts=dict(config=self.ref(self.root/'config.json'),source_lock=self.ref(self.root/'source_lock.json')),
            runtime_sources={p.name:sha(p) for p in (self.release/'runtime').glob('*.py')}))
        self.freeze_sha=sha(self.freeze)
        self.receipt=self.root/'GPU_RELEASE.json'
        self.inventory={str(p.relative_to(self.release)):sha(p) for p in self.release.rglob('*') if p.is_file()}
        dump(self.receipt,dict(schema='sr-satdose-gpu-technical-release-v1',status='AUDITED_FOR_GPU_CHECKS_ONLY',
            permission='NO_TRAINING',experiment_id=EXPERIMENT,evidence_root=str(self.root),
            design_freeze_sha256=self.freeze_sha,config_sha256=sha(self.root/'config.json'),
            release_root=str(self.release),sources=self.inventory))
        (self.root/'prefix_control').mkdir()
        dump(self.root/'prefix_control/RESULT.json',dict(status='PREFIX_METRICS_COMPLETE_GPU_CHECK_STILL_REQUIRED',
            permission='NO_TRAINING',responses=96,prefix_analyses={str(s):dict(
            status='PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED',candidate_gate_passed=True) for s in (11031,11032)}))
        self.budget=self.root/'CPU_STORAGE_BUDGET.json'
        dump(self.budget,dict(schema='sr-satdose-storage-budget-v1',scope='GPU_DIAGNOSTICS_ONLY',
            experiment_id=EXPERIMENT,evidence_root=str(self.root),design_freeze_sha256=self.freeze_sha,
            required_bytes=1024,max_observation_age_seconds=30,created_at_utc='2000-01-01T00:00:00Z',
            expires_at_utc='2099-01-01T00:00:00Z'))
        self.binding=dict(schema='sr-satdose-technical-execution-v1',mode='production',experiment_id=EXPERIMENT,
            experiment_root=str(self.root),max_shadow_optimizer_steps=4,steps_per_slot=1,
            design_freeze=self.ref(self.freeze),config=self.ref(self.root/'config.json'),source_lock=self.ref(self.root/'source_lock.json'),
            gpu_release=self.ref(self.receipt),storage_budget=self.ref(self.budget),prefix_result=self.ref(self.root/'prefix_control/RESULT.json'),
            code={name:dict(path=str(self.release/'execution'/(name+'.py')),sha256=sha(self.release/'execution'/(name+'.py')))
            for name in ('evaluation_probe','evaluation_loader','backend_equivalence','allzero_check','diagnostic_worker',
                         'shadow_replay','production','training_state','training_transaction','diagnostic_inputs')},
            evaluation_loader_slots={str(s):f'evaluation_loader_checks/start-{s}' for s in (11031,11032)},
            backend_equivalence_slots={str(s):f'backend_equivalence/start-{s}' for s in (11031,11032)},
            allzero_slots={str(s):f'allzero_check/start-{s}' for s in (11031,11032)},
            shadow_slots={f'{s}/{t}':{'input_receipt':f'shadow_inputs/start-{s}/trial-{t}/INPUT_RECEIPT.json'}
                          for s in (11031,11032) for t in (0,1)})
        self.binding['code']['gpu_launcher']=dict(path=str(self.release/'scheduling/run_gpu_checks.sh'),sha256=sha(SCRIPT))
        self.binding_path=self.root/'TECHNICAL_EXECUTION_BINDING.json';self.bind()
        self.parent=self.root/'evaluation_loader_checks';self.parent.mkdir()
        self.quotas={self.ancestor:(10000,1000),self.root:(8000,1000),self.parent:(6000,1000)}
        self.space=SimpleNamespace(f_bavail=20000,f_frsize=1)

    def ref(self,path):return dict(path=str(path.relative_to(self.root)),sha256=sha(path))
    def bind(self):dump(self.binding_path,self.binding);self.binding_sha=sha(self.binding_path)

    def xattr(self,path,key,**kwargs):
        path=Path(path)
        if path not in self.quotas:raise OSError(errno.ENODATA,'CPU no local quota')
        maximum,used=self.quotas[path]
        return str(maximum if key=='ceph.quota.max_bytes' else used).encode()

    def bootstrap(self,*,launcher=None,expected=None,start=11031,stage='evaluation_probe'):
        source=SCRIPT.read_text().split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
        # Test-only relocation of literal production constants, not a production flag.
        source=source.replace(ROOT_TEXT,str(self.root)).replace(ANCESTOR_TEXT,str(self.ancestor)).replace(FREEZE,self.freeze_sha)
        args=['-',str(self.root),expected or self.binding_sha,str(launcher or self.release/'scheduling/run_gpu_checks.sh'),str(start),stage]
        output=io.StringIO();errors=io.StringIO()
        with patch.object(sys,'path',list(sys.path)),patch.object(sys,'argv',args),patch.object(os,'getxattr',side_effect=self.xattr,create=True),\
             patch.object(os,'statvfs',return_value=self.space),redirect_stdout(output),redirect_stderr(errors):
            exec(compile(source,'CPU_ONLY_EXTRACTED_GPU_BOOTSTRAP','exec'),{'__name__':'__main__'})
        return output.getvalue().strip(),json.loads(errors.getvalue())

    def waiver_fixture(self):
        # CPU stand-in ensures bootstrap CALLS the bound verifier; no GPU/waiver claim.
        source=self.release/'execution/prefix_waiver.py'
        source.write_text("def verify_completed(root,binding):\n    if not (root/'CPU_COMPLETED').exists():\n        raise ValueError('CPU UNKNOWN continuation')\n")
        for name in ('WAIVER.json','CONTINUATION.json'):dump(self.root/name,{'CPU_ONLY':True})
        self.binding.update(prefix_waiver=self.ref(self.root/'WAIVER.json'),prefix_continuation=self.ref(self.root/'CONTINUATION.json'))
        self.binding['code']['prefix_waiver']=dict(path=str(source),sha256=sha(source))
        receipt=json.loads(self.receipt.read_text());receipt['sources']['execution/prefix_waiver.py']=sha(source)
        dump(self.receipt,receipt);self.binding['gpu_release']=self.ref(self.receipt);self.bind()
        return source

    def test_waiver_requires_actual_completed_verification_and_release_pin(self):
        source=self.waiver_fixture()
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):self.bootstrap()
        (self.root/'CPU_COMPLETED').touch()
        original_path=list(sys.path)
        try:self.assertEqual(self.bootstrap()[0],str(self.release))
        finally:sys.path[:]=original_path
        self.binding['code']['prefix_waiver']['path']=str(self.temp/'foreign.py');self.bind()
        with self.assertRaisesRegex(SystemExit,'audited release'):self.bootstrap()

    def test_waiver_incomplete_binding_and_removed_inventory_refuse(self):
        source=self.waiver_fixture();self.binding.pop('prefix_continuation');self.bind()
        with self.assertRaisesRegex(SystemExit,'Incomplete source-bound'):self.bootstrap()
        self.binding['prefix_continuation']=self.ref(self.root/'CONTINUATION.json');self.bind()
        source.write_text(source.read_text()+'# tamper')
        with self.assertRaisesRegex(SystemExit,'inventory/SHA'):self.bootstrap()

    def test_full_release_spooled_bytes_and_live_all_ancestor_space(self):
        root,observation=self.bootstrap();self.assertEqual(root,str(self.release))
        spool=self.temp/'slurmd/job-1/slurm_script';spool.parent.mkdir(parents=True);shutil.copyfile(SCRIPT,spool)
        self.assertEqual(self.bootstrap(launcher=spool)[0],str(self.release))
        self.assertEqual(observation['required_bytes'],1024)
        self.assertTrue(observation['writable_probe_fsynced_and_removed'])
        self.assertEqual(observation['observations'][0]['effective_available_bytes'],5000)
        paths=[r['path'] for r in observation['observations'][0]['quota_observations']]
        self.assertEqual(paths,[str(self.parent),str(self.root),str(self.root.parent),str(self.ancestor)])
        self.assertFalse(list(self.root.rglob('.gpu-checks-storage-probe-*')))

    def test_tampered_spool_code_nonpython_extra_and_symlink_fail_closed(self):
        spool=self.temp/'spool';shutil.copyfile(SCRIPT,spool);spool.write_bytes(spool.read_bytes()+b'\n# tamper')
        with self.assertRaisesRegex(SystemExit,'spooled launcher'):self.bootstrap(launcher=spool)
        path=self.release/'supplied_package/config/source_bindings.json';original=path.read_bytes()
        path.write_text('{}')
        with self.assertRaisesRegex(SystemExit,'inventory/SHA'):self.bootstrap()
        path.write_bytes(original);extra=self.release/'unexpected.json';extra.write_text('{}')
        with self.assertRaisesRegex(SystemExit,'inventory/SHA'):self.bootstrap()
        extra.unlink();extra.symlink_to(self.budget)
        with self.assertRaisesRegex(SystemExit,'ordinary file'):self.bootstrap()

    def test_required_scoring_helper_cannot_be_omitted_from_new_receipt_inventory(self):
        helper=self.release/'supplied_package/tools/satdose_core.py';helper.unlink()
        receipt=json.loads(self.receipt.read_text());del receipt['sources']['supplied_package/tools/satdose_core.py']
        dump(self.receipt,receipt);self.binding['gpu_release']=self.ref(self.receipt);self.bind()
        with self.assertRaisesRegex(SystemExit,'Incomplete GPU technical release'):self.bootstrap()

    def test_sha_receipt_freeze_budget_and_prefix_are_not_self_authorized(self):
        with self.assertRaisesRegex(SystemExit,'binding SHA'):self.bootstrap(expected='0'*64)
        self.receipt.write_bytes(self.receipt.read_bytes()+b'\n')
        with self.assertRaisesRegex(SystemExit,'artifact SHA'):self.bootstrap()
        self.binding['gpu_release']=self.ref(self.receipt);self.bind()
        self.freeze.write_bytes(self.freeze.read_bytes()+b'\n')
        with self.assertRaisesRegex(SystemExit,'artifact SHA'):self.bootstrap()

    def test_insufficient_child_parent_or_filesystem_and_unverified_xattrs_reject(self):
        for path in (self.parent,self.root,self.ancestor):
            saved=self.quotas[path];self.quotas[path]=(1000,999)
            with self.subTest(path=path),self.assertRaisesRegex(SystemExit,'Insufficient live'):self.bootstrap()
            self.quotas[path]=saved
        self.space.f_bavail=1
        with self.assertRaisesRegex(SystemExit,'Insufficient live'):self.bootstrap()
        self.space.f_bavail=20000;self.quotas[self.root]=(0,0)
        with self.assertRaisesRegex(SystemExit,'zero quota'):self.bootstrap()
        self.quotas[self.root]=(8000,1000);del self.quotas[self.ancestor]
        with self.assertRaisesRegex(SystemExit,'ancestor has no verified quota'):self.bootstrap()

    def test_existing_stage_and_shadow_claim_refuse_retry_and_expired_budget_rejects(self):
        target=self.parent/'start-11031';target.mkdir()
        with self.assertRaisesRegex(SystemExit,'never retry'):self.bootstrap()
        target.rmdir();claim=self.root/'shadow_claims/start-11031/trial-0';claim.mkdir(parents=True)
        with self.assertRaisesRegex(SystemExit,'Existing shadow attempt'):self.bootstrap(stage='diagnostic_worker:0')
        budget=json.loads(self.budget.read_text());budget['expires_at_utc']='2001-01-01T00:00:00Z';dump(self.budget,budget)
        self.binding['storage_budget']=self.ref(self.budget);self.bind()
        with self.assertRaisesRegex(SystemExit,'budget stale'):self.bootstrap()

    def test_prefix_hold_and_unapproved_binding_code_pin_refuse(self):
        path=self.root/'prefix_control/HOLD.json';dump(path,{'CPU_FAKE':True})
        with self.assertRaisesRegex(SystemExit,'PREFIX starts'):self.bootstrap()
        path.unlink();self.binding['code']['allzero_check']['sha256']='0'*64;self.bind()
        with self.assertRaisesRegex(SystemExit,'Code pin'):self.bootstrap()

    def shell(self,*,fail=0,guard_fail='',restart='0'):
        binary=self.temp/'bin';binary.mkdir(exist_ok=True);log=self.temp/'srun.log'
        if log.exists():log.unlink()
        wrapper=binary/'srun';wrapper.write_text('#!'+sys.executable+'\n'+
            'import json,os,pathlib,sys\np=pathlib.Path(os.environ["GPU_TEST_LOG"])\n'+
            'n=len(p.read_text().splitlines()) if p.exists() else 0\n'+
            'with p.open("a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n'+
            'sys.exit(7 if n+1==int(os.environ.get("GPU_TEST_FAIL","0")) else 0)\n');wrapper.chmod(0o700)
        original=SCRIPT.read_text();head=original.split('# Stdlib only:',1)[0];tail=original.split('# Ten independent',1)[1]
        shell=head+'\nguard() { [[ "$1/$2" != "$GPU_TEST_GUARD_FAIL" ]] || return 9; printf "%s\\n" "$GPU_TEST_RELEASE"; }\n# Ten independent'+tail
        script=self.temp/'CPU_SHELL_CONTROL_ONLY.sh';script.write_text(shell)
        env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],SLURM_JOB_ID='123',SLURM_NTASKS='1',SLURM_NNODES='1',
            SLURM_RESTART_COUNT=restart,GPU_TEST_LOG=str(log),GPU_TEST_FAIL=str(fail),GPU_TEST_GUARD_FAIL=guard_fail,
            GPU_TEST_RELEASE=str(self.release))
        result=subprocess.run(['bash',str(script),self.binding_sha],env=env,text=True,capture_output=True)
        return result,[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def test_shell_exact_ten_process_order_resources_and_four_shadow_steps(self):
        result,calls=self.shell();self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(len(calls),10)
        expected=[(start,module,trial) for start in (11031,11032) for module,trial in (
            ('evaluation_probe',None),('backend_equivalence',None),('allzero_check',None),('diagnostic_worker','0'),('diagnostic_worker','1'))]
        for args,(start,module,trial) in zip(calls,expected,strict=True):
            self.assertEqual(args[args.index('-m')+1],'execution.'+module)
            self.assertEqual(args[args.index('--start-seed')+1],str(start))
            self.assertEqual(args[args.index('--binding-sha256')+1],self.binding_sha)
            for flag in ('--exclusive','--exact','--nodes=1','--ntasks=1','--cpus-per-task=8','--gpus-per-task=pro6000:2','--mem=66G'):
                self.assertIn(flag,args)
            if trial is None:self.assertNotIn('--trial',args)
            else:self.assertEqual(args[args.index('--trial')+1],trial)
        self.assertEqual(sum('--trial' in args for args in calls),4)

    def test_shell_nonzero_stage_or_guard_and_requeue_stop_without_retries(self):
        result,calls=self.shell(fail=3);self.assertEqual(result.returncode,7);self.assertEqual(len(calls),3)
        result,calls=self.shell(guard_fail='11031/backend_equivalence');self.assertEqual(result.returncode,9);self.assertEqual(len(calls),1)
        result,calls=self.shell(restart='1');self.assertEqual(result.returncode,2);self.assertFalse(calls)

    def test_shell_syntax_and_finite_cli_flags_match_modules(self):
        result=subprocess.run(['bash','-n',str(SCRIPT)],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for module in ('evaluation_probe','backend_equivalence','allzero_check','diagnostic_worker'):
            code=(PACKAGE/'execution'/(module+'.py')).read_text()
            self.assertIn("'--start-seed'",code);self.assertIn("'--binding-sha256'",code)
        self.assertIn("'--trial'",(PACKAGE/'execution/diagnostic_worker.py').read_text())
        self.assertIn('#SBATCH --no-requeue',SCRIPT.read_text())


if __name__=='__main__':unittest.main()
