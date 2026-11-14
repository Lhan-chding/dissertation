"""Synthetic CPU control/source tests only; no generation or GPU evidence."""
import copy
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from execution import prefix_continuation as p, prefix_control as original
sys.path.pop(0)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.addCleanup(patch.stopall)
        patch.object(p,'BASE',self.root).start()
        self.old=dict(status='HOLD_PREFIX_MEASUREMENT',candidate_gate_passed=False,permission='NO_TRAINING',
                      fixture='SYNTHETIC_CPU_ONLY')
        self.fresh=copy.deepcopy(self.old)
        self.dump('prefix_control/RESULT.json',dict(status='HOLD_PREFIX_MEASUREMENT',stopped_start=11031,prefix_analyses={'11031':self.old}))
        self.dump('prefix_control/PREFIX_11031_ANALYSIS.json',self.old)
        self.dump('runs/PREFIX_CALIB_11031/CALIBRATION_COMPLETE.json',dict(scope='PREFIX_CALIB',start_seed=11031,responses=48))
        self.dump('calibration_allocations/PREFIX_CALIB-11031.json',dict(scope='PREFIX_CALIB',start_seed=11031,
            output=str(self.root/'runs/PREFIX_CALIB_11031'),design_freeze_sha256=p.FREEZE_SHA))
        for start in (11031,11032):
            self.dump(f'runs/DRY_{start}/CALIBRATION_COMPLETE.json',dict(status='CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION',
                scope='DRY',start_seed=start,responses=640))
        self.dump(p.OLD_RELEASE,{});self.dump(p.WAIVER,{})
        self.dump('DESIGN_FREEZE.json',dict(artifacts=dict(config=dict(path='config.json',sha256='c'*64),
                                                        source_lock=dict(path='lock.json',sha256='s'*64))))
        self.authorization=dict(original_result=p.ref(self.root/'prefix_control/RESULT.json'))
        self.pc=SimpleNamespace(core=original.core,prefix_attempts=original.prefix_attempts,
            verify_deployment=Mock(return_value=dict(config=self.root/'config.json',freeze=self.root/'DESIGN_FREEZE.json',native_src=self.root)),
            protocol_identity=Mock(return_value=(self.root/'clarification.json',{})),
            calibration=SimpleNamespace(analyze_dry=Mock(return_value=dict(status='DRY_METRICS_PASS_GPU_CHECK_STILL_REQUIRED',
                failures=[],checks=dict(signal=dict(passed=True)))),analyze_prefix=Mock(side_effect=lambda *a,**k:copy.deepcopy(self.old if a[4]==11031 else self.fresh))),
            command=lambda start:['original-python','-m','runtime.dry_driver','--start-seed',str(start),
                '--scope','PREFIX_CALIB','--output',str(self.root/f'runs/PREFIX_CALIB_{start}')])
        self.waiver=SimpleNamespace(verify_waiver=Mock(return_value=self.authorization),verify_completed=Mock(return_value={}))
        self.release_ref=dict(path='execution_releases/new.json',sha256='e'*64)
        patch.object(p,'_modules',return_value=(self.pc,self.waiver)).start()
        patch.object(p,'_old_release',return_value={}).start()
        patch.object(p,'verify_release',return_value=(self.release_ref,dict(sources={}))).start()
        patch.dict(os.environ,SLURM_JOB_ID='123',SLURM_NTASKS='1',SLURM_NNODES='1',SLURM_RESTART_COUNT='0').start()

    def dump(self,relative,value):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return path

    def test_preflight_rescores_both_dry_and_original_failed48(self):
        result=p.preflight(self.root/'receipt','e'*64,p.WAIVER_SHA)
        self.assertEqual(self.pc.calibration.analyze_dry.call_count,2)
        self.assertEqual(result['original_analysis'],self.old)
        self.assertEqual(result['start_seed'],11032);self.assertEqual(result['responses'],48)
        self.assertEqual(result['permission'],'NO_TRAINING');self.assertFalse((self.root/p.CONTROL).exists())

    def test_partial_dry_rejected_before_any_raw_analysis(self):
        self.dump('runs/DRY_11032/CALIBRATION_COMPLETE.json',dict(status='INCOMPLETE',scope='DRY',start_seed=11032,responses=639))
        with self.assertRaisesRegex(ValueError,'complete DRY'):p.preflight(self.root/'r','e'*64,p.WAIVER_SHA)
        self.pc.calibration.analyze_dry.assert_not_called();self.pc.calibration.analyze_prefix.assert_not_called()

    def test_dry_signal_failure_not_waived(self):
        self.pc.calibration.analyze_dry.return_value=dict(status='HOLD_DRY_SIGNAL',failures=['signal'],checks={})
        with self.assertRaisesRegex(ValueError,'DRY gates'):p.preflight(self.root/'r','e'*64,p.WAIVER_SHA)
        self.pc.calibration.analyze_prefix.assert_not_called()

    def test_existing_11032_allocation_run_or_control_blocks_retry(self):
        for name in ('calibration_allocations/PREFIX_CALIB-11032.json','runs/PREFIX_CALIB_11032','prefix_waiver_control'):
            with self.subTest(name=name):
                path=self.root/name
                if path.suffix:path.write_text('{}')
                else:path.mkdir()
                with self.assertRaises(ValueError):p.preflight(self.root/'r','e'*64,p.WAIVER_SHA)
                if path.is_dir():path.rmdir()
                else:path.unlink()

    def test_gpu_binding_and_every_existing_slot_rejected(self):
        for name in ('TECHNICAL_EXECUTION_BINDING.json','shadow_inputs','shadow_claims','backend_equivalence','allzero_check','evaluation_loader_checks'):
            with self.subTest(name=name):
                path=self.root/name
                if path.suffix:path.write_text('{}')
                else:path.mkdir();(path/'UNKNOWN').write_text('')
                with self.assertRaisesRegex(ValueError,'GPU'):p.no_gpu_attempts()
                if path.is_dir():shutil.rmtree(path)
                else:path.unlink()

    def test_authorization_failure_stops_before_raw(self):
        self.waiver.verify_waiver.side_effect=ValueError('authorization wrong SHA')
        with self.assertRaisesRegex(ValueError,'authorization'):p.preflight(self.root/'r','e'*64,p.WAIVER_SHA)
        self.pc.calibration.analyze_dry.assert_not_called()

    def test_original_failed_measurement_must_remain_exact(self):
        self.pc.calibration.analyze_prefix.side_effect=None
        self.pc.calibration.analyze_prefix.return_value=dict(self.old,status='PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED')
        with self.assertRaisesRegex(ValueError,'Original complete'):p.preflight(self.root/'r','e'*64,p.WAIVER_SHA)

    def test_only11032_launched_lock_held_and_measurement_failure_preserved(self):
        before=p.protected_inventory();calls=[]
        def launch(command,*,cwd,env,check):
            calls.append(command)
            self.assertEqual(command[command.index('--start-seed')+1],'11032');self.assertEqual(cwd,self.root/'code')
            with (self.root/'DIAGNOSTIC_WORKER.lock').open('r+') as stream:
                with self.assertRaises(BlockingIOError):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return SimpleNamespace(returncode=0)
        with patch.object(p.subprocess,'run',side_effect=launch):
            result=p.execute(self.root/'r','e'*64,p.WAIVER_SHA)
        self.assertEqual(len(calls),1);self.assertEqual(result['responses'],96)
        self.assertEqual(result['new_responses'],48);self.assertEqual(result['resampled_responses'],0)
        self.assertEqual(result['gate_decision'],'PASS_USER_OVERRIDE');self.assertEqual(result['permission'],'NO_TRAINING')
        self.assertEqual(result['prefix_analyses'],{'11031':self.old,'11032':self.fresh})
        self.assertEqual(p.protected_inventory(),before)
        self.assertEqual(self.waiver.verify_completed.call_count,1)
        self.assertEqual(self.waiver.verify_completed.call_args.args[1]['prefix_result'],self.authorization['original_result'])

    def test_technical_failure_hold_and_no_retry(self):
        with patch.object(p.subprocess,'run',return_value=SimpleNamespace(returncode=2)) as launch:
            result=p.execute(self.root/'r','e'*64,p.WAIVER_SHA)
            self.assertTrue(result['status'].startswith('HOLD'))
            self.assertTrue((self.root/p.CONTROL/'HOLD.json').exists())
            with self.assertRaisesRegex(ValueError,'Existing continuation'):p.execute(self.root/'r','e'*64,p.WAIVER_SHA)
            self.assertEqual(launch.call_count,1)
        self.waiver.verify_completed.assert_not_called()

    def test_global_lock_conflict_prevents_preflight_and_launch(self):
        lock=self.root/'DIAGNOSTIC_WORKER.lock';lock.touch()
        with lock.open('r+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(p,'preflight') as preflight,patch.object(p.subprocess,'run') as launch:
                with self.assertRaises(BlockingIOError):p.execute(self.root/'r','e'*64,p.WAIVER_SHA)
                preflight.assert_not_called();launch.assert_not_called()

    def test_independent_completed_audit_error_records_hold(self):
        self.waiver.verify_completed.side_effect=ValueError('incomplete native CAPTURE')
        with patch.object(p.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
            result=p.execute(self.root/'r','e'*64,p.WAIVER_SHA)
        self.assertTrue(result['status'].startswith('HOLD'));self.assertTrue((self.root/p.CONTROL/'HOLD.json').exists())

    def test_new_scientific_pass_does_not_rewrite_old_failure(self):
        self.fresh.update(status='PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED',candidate_gate_passed=True)
        with patch.object(p.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
            result=p.execute(self.root/'r','e'*64,p.WAIVER_SHA)
        self.assertFalse(result['prefix_analyses']['11031']['candidate_gate_passed'])
        self.assertTrue(result['prefix_analyses']['11032']['candidate_gate_passed'])
        self.assertEqual(result['gate_decision'],'PASS_USER_OVERRIDE')

    def test_original_command_remains_frozen11032(self):
        with patch.object(original,'BASE',self.root):command=p.command(original)
        self.assertIn('--gpus-per-task=pro6000:2',command)
        self.assertEqual(command[command.index('--start-seed')+1],'11032')
        self.assertEqual(command[command.index('--scope')+1],'PREFIX_CALIB')


class SourceTests(unittest.TestCase):
    def test_controller_requires_independent_release_and_waiver_sha_full_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();release=root/'release';release.mkdir()
            required=['execution/prefix_continuation.py','execution/prefix_waiver.py','execution/prefix_control.py',
                      'scheduling/run_prefix_continuation.sh','analysis/calibration.py','supplied_package/tools/satdose_core.py']
            for name in required:
                path=release/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# SYNTHETIC_CPU_ONLY\n')
            waiver=root/p.WAIVER;waiver.parent.mkdir(parents=True);waiver.write_text('{}');wsha=p.sha(waiver)
            record=dict(schema='sr-satdose-prefix-continuation-release-v1',status='AUDITED_FOR_PREFIX_11032_CONTINUATION_ONLY',
                permission='NO_TRAINING',experiment_id=p.EXPERIMENT,evidence_root=str(root),design_freeze_sha256=p.FREEZE_SHA,
                config_sha256=p.CONFIG_SHA,frozen_code_root=str(root/'code'),release_root=str(release),sources=p.inventory(release),
                waiver=dict(path=p.WAIVER,sha256=wsha),original_release=dict(path=p.OLD_RELEASE,sha256=p.OLD_RELEASE_SHA))
            receipt=root/'release.json';receipt.write_text(json.dumps(record));rsha=p.sha(receipt)
            with patch.object(p,'BASE',root),patch.object(p,'PACKAGE',release),patch.object(p,'WAIVER_SHA',wsha):
                p.verify_release(receipt,rsha,wsha)
                with self.assertRaisesRegex(ValueError,'authorization'):p.verify_release(receipt,rsha,'0'*64)
                with self.assertRaisesRegex(ValueError,'receipt SHA'):p.verify_release(receipt,'0'*64,wsha)
                (release/'pins.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'inventory'):p.verify_release(receipt,rsha,wsha)

    def test_inventory_rejects_symlink_and_covers_noncode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'pins.json').write_text('{}')
            self.assertEqual(set(p.inventory(root)),{'pins.json'})
            (root/'alias').symlink_to(root/'pins.json')
            with self.assertRaises(ValueError):p.inventory(root)

    def test_spool_bootstrap_accepts_same_bytes_rejects_tamper(self):
        script=(PACKAGE/'scheduling/run_prefix_continuation.sh').read_text()
        bootstrap=script.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();release=root/'release';(release/'scheduling').mkdir(parents=True)
            (release/'execution').mkdir();(root/'protocol_clarifications').mkdir();(root/'execution_releases').mkdir()
            launcher=release/'scheduling/run_prefix_continuation.sh';launcher.write_text(script)
            spool=root/'slurm_script';spool.write_bytes(launcher.read_bytes())
            for name in ('prefix_continuation','prefix_waiver','prefix_control'):(release/f'execution/{name}.py').write_text('# synthetic CPU only\n')
            freeze=root/'DESIGN_FREEZE.json';freeze.write_text('{}')
            waiver=root/p.WAIVER;waiver.write_text('{}')
            fsha=p.sha(freeze);wsha=p.sha(waiver)
            record=dict(schema='sr-satdose-prefix-continuation-release-v1',status='AUDITED_FOR_PREFIX_11032_CONTINUATION_ONLY',
                permission='NO_TRAINING',experiment_id=p.EXPERIMENT,evidence_root=str(root),design_freeze_sha256=fsha,
                config_sha256=p.CONFIG_SHA,frozen_code_root=str(root/'code'),release_root=str(release),sources=p.inventory(release),
                waiver=dict(path=p.WAIVER,sha256=wsha),original_release=dict(path=p.OLD_RELEASE,sha256=p.OLD_RELEASE_SHA))
            receipt=root/'execution_releases/new.json';receipt.write_text(json.dumps(record))
            source=bootstrap.replace(str(p.BASE),str(root)).replace(p.FREEZE_SHA,fsha).replace(p.WAIVER_SHA,wsha)
            args=['-',str(receipt),p.sha(receipt),wsha,str(spool)]
            with patch.object(sys,'argv',args),patch.object(sys,'stdout',new_callable=io.StringIO):
                exec(compile(source,'<synthetic-spool-bootstrap>','exec'),{})
            spool.write_text(script+'# tampered\n')
            with patch.object(sys,'argv',args),self.assertRaisesRegex(SystemExit,'spool launcher'):
                exec(compile(source,'<synthetic-spool-bootstrap>','exec'),{})


if __name__ == '__main__':unittest.main()
