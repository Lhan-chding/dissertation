"""CPU control/receipt tests; synthetic captures never certify GPU acceptance."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from execution import prefix_control as p
sys.path.pop(0)
spec = importlib.util.spec_from_file_location('_prefix_fixture_factory', Path(__file__).with_name('test_calibration_analysis.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
core = p.core


@unittest.skipUnless((fixtures.NATIVE/'sr_f6/measurement.py').is_file(), 'Pinned native scorer source required')
class PrefixPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(fixtures.CalibrationTests.doClassCleanups)
        fixtures.CalibrationTests.setUpClass()
        cls.factory = fixtures.CalibrationTests

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()/'fixture'
        shutil.copytree(self.factory.base, self.root)
        (self.root/'runs').mkdir()
        for start in p.STARTS:
            (self.root/f'DRY-{start}').rename(self.root/f'runs/DRY_{start}')
            # Generated synthetic PREFIX material stays outside the evidence root
            # until an injected launch callback completes that start.
            (self.root/f'PREFIX_CALIB-{start}').rename(Path(self.temp.name)/f'prefix-{start}')
        self.config = self.root/'config.json'; self.freeze = self.root/'DESIGN_FREEZE.json'
        self.addCleanup(patch.stopall)
        patch.object(p, 'BASE', self.root).start()
        patch.object(p, 'FREEZE_SHA', core.file_sha(self.freeze)).start()
        patch.object(p, 'CONFIG_SHA', core.file_sha(self.config)).start()
        fake = type('FixtureContext', (), {'root': self.root})()
        source = self.factory.clarification(fake)
        folder = self.root/'protocol_clarifications'; folder.mkdir()
        target = folder/'PREFIX_PROTOCOL_CLARIFICATION_20261113.json'; source.rename(target)
        proof = core.read_json(target)['pre_prefix_confirmation']
        patch.object(p, 'CLARIFICATION_SHA', core.file_sha(target)).start()
        patch.object(p, 'PROOF_SHA', proof['sha256']).start()
        # Deployment/source validation has independent real-file tests below.
        # Only these machine-location seams are mocked for synthetic rescoring.
        patch.object(p, 'verify_release', return_value={'fixture_only': True}).start()
        patch.object(p, 'verify_deployment', return_value=dict(config=self.config, freeze=self.freeze,
            native_src=fixtures.NATIVE, source_lock_sha256='fixture', runtime_sources={})).start()
        self.clarification = target

    def preflight(self):
        return p.preflight(self.root/'not-a-real-release.json', 'synthetic-seam-only')

    def test_both_full_dry_are_rescored_and_never_admit_training(self):
        result = self.preflight()
        self.assertEqual(result['status'], 'READY_FOR_PREFIX_CALIB_ONLY')
        self.assertEqual(result['permission'], 'NO_TRAINING')
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(result['starts'], [11031, 11032])
        self.assertEqual(result['responses_per_start'], 48)
        self.assertEqual([len(r['rescored_rows']) for r in result['dry_analyses'].values()], [640, 640])
        self.assertFalse((self.root/'prefix_control').exists())

    def test_missing_or_partial_second_receipt_stops_before_any_raw(self):
        path = self.root/'runs/DRY_11032/CALIBRATION_COMPLETE.json'
        original = path.read_bytes()
        for missing in (True, False):
            with self.subTest(missing=missing):
                path.write_bytes(original)
                if missing:
                    path.unlink()
                else:
                    receipt = core.read_json(path); receipt['responses'] = 639; fixtures.dump(path, receipt)
                with patch.object(p.calibration, 'analyze_dry', side_effect=AssertionError('must not read partial raw')):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.preflight()

    def test_failed_real_dry_gate_blocks_prefix(self):
        directory = self.root/'runs/DRY_11031'
        rows = list(core.read_jsonl(directory/'raw.jsonl'))
        cases = {r['case_id']: r for r in core.read_jsonl(self.root/'private_cases.jsonl')}
        # Complete, origin-bound synthetic EOS captures: native scorer marks
        # them unknown, so the unchanged scientific DRY gate must reject them.
        for row in rows:
            task = cases[row['case_id']]['task']; row['finish_reason'] = 'eos'
            row['native_score'] = self.factory.scorer(row['full_text'], task['world'], task['query'],
                                                     finished=False, finish_reason='eos')
        fixtures.lines(directory/'raw.jsonl', rows)
        receipt = core.read_json(directory/'CALIBRATION_COMPLETE.json')
        receipt['raw']['sha256'] = core.file_sha(directory/'raw.jsonl')
        receipt['complete_files'] = fixtures.write_fixture_complete(directory, rows)
        fixtures.dump(directory/'CALIBRATION_COMPLETE.json', receipt)
        with self.assertRaisesRegex(ValueError, 'Both complete DRY gates'):
            self.preflight()

    def test_wrong_clarification_binding_is_rejected(self):
        record = core.read_json(self.clarification); record['design_freeze_sha256'] = '0'*64
        fixtures.dump(self.clarification, record)
        with patch.object(p, 'CLARIFICATION_SHA', core.file_sha(self.clarification)):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                self.preflight()

    def test_existing_allocation_run_and_control_refuse_retry(self):
        for relative in ('calibration_allocations/PREFIX_CALIB-11031.json', 'runs/PREFIX_CALIB_11031', 'prefix_control'):
            with self.subTest(relative=relative):
                path = self.root/relative; path.parent.mkdir(exist_ok=True)
                if path.suffix:
                    fixtures.dump(path, dict(scope='PREFIX_CALIB'))
                else:
                    path.mkdir()
                with self.assertRaisesRegex(ValueError, 'Existing PREFIX'):
                    self.preflight()
                if path.is_dir(): path.rmdir()
                else: path.unlink()

    def test_gpu_slots_without_binding_require_audit(self):
        for relative in ('shadow_inputs/start-11031/trial-0/HOLD.json',
                         'shadow_claims/start-11031/trial-0/INTENT.json',
                         'backend_equivalence/start-11031/COMPLETE.json'):
            with self.subTest(relative=relative):
                path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True)
                fixtures.dump(path, dict(fixture_only=True))
                with self.assertRaisesRegex(ValueError, 'audit'):
                    self.preflight()
                top = self.root/Path(relative).parts[0]
                if top.is_dir(): shutil.rmtree(top)
                else: top.unlink()

    def test_invalid_gpu_binding_rejected_before_prefix(self):
        fixtures.dump(self.root/'TECHNICAL_EXECUTION_BINDING.json', dict(fixture_only=True))
        with self.assertRaisesRegex(ValueError, 'binding identity'):
            self.preflight()

    def test_real_prefix_hold_prevents_second_start(self):
        report = self.preflight(); launched = []; written = []
        directory = Path(self.temp.name)/'prefix-11031'
        rows = list(core.read_jsonl(directory/'raw.jsonl'))
        for row in rows:
            row['full_text'] = row['assistant_prefill']+'"selected": ["X"]}'
            row['suffix_text'] = row['full_text'][len(row['assistant_prefill']):]
            row['generated_text'] = row['suffix_text']
        fixtures.lines(directory/'raw.jsonl', rows)
        receipt = core.read_json(directory/'CALIBRATION_COMPLETE.json')
        receipt['raw']['sha256'] = core.file_sha(directory/'raw.jsonl')
        receipt['complete_files'] = fixtures.write_fixture_complete(directory, rows)
        fixtures.dump(directory/'CALIBRATION_COMPLETE.json', receipt)
        def launch(start):
            launched.append(start)
            (Path(self.temp.name)/f'prefix-{start}').rename(self.root/f'runs/PREFIX_CALIB_{start}')
            return 0
        def analyze(start):
            return p.calibration.analyze_prefix(self.config, self.freeze, self.root,
                self.root/f'runs/PREFIX_CALIB_{start}/CALIBRATION_COMPLETE.json', start, self.clarification)
        result = p.run_sequence(report, launch, analyze, lambda start, value: written.append((start, value)))
        self.assertEqual(launched, [11031]); self.assertEqual(len(written), 1)
        self.assertEqual(result['status'], 'HOLD_PREFIX_MEASUREMENT')
        self.assertEqual(result['permission'], 'NO_TRAINING')
        self.assertTrue((Path(self.temp.name)/'prefix-11032').exists())


class PrefixBindingAndSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def release(self):
        release = self.root/'release'; release.mkdir()
        for relative in ('execution/prefix_control.py', 'execution/diagnostic_worker.py',
                         'scheduling/run_prefix.sh', 'analysis/calibration.py',
                         'supplied_package/tools/satdose_core.py', 'supplied_package/config/experiment.json',
                         'supplied_package/config/source_bindings.json'):
            target = release/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PACKAGE/relative, target)
        record = dict(schema='sr-satdose-prefix-technical-release-v1', status='AUDITED_FOR_PREFIX_CALIB_ONLY',
            permission='NO_TRAINING', experiment_id='SR-SATDOSE-1-20261112', evidence_root=str(p.BASE),
            frozen_code_root=str(p.BASE/'code'), design_freeze_sha256=p.FREEZE_SHA, config_sha256=p.CONFIG_SHA,
            release_root=str(release), sources=p.source_inventory(release))
        path = self.root/'release.json'; fixtures.dump(path, record)
        return release, path, core.file_sha(path)

    def verify(self, release, receipt, sha):
        with patch.object(p, 'PACKAGE', release), patch.object(p.calibration, '__file__', str(release/'analysis/calibration.py')), \
             patch.object(core, '__file__', str(release/'supplied_package/tools/satdose_core.py')):
            return p.verify_release(receipt, sha)

    def test_full_release_binds_non_python_scorer_pins_and_extra_files(self):
        release, receipt, sha = self.release()
        self.assertEqual(self.verify(release, receipt, sha)['sha256'], sha)
        with self.assertRaisesRegex(ValueError, 'receipt SHA'):
            self.verify(release, receipt, '0'*64)
        pins = release/'supplied_package/config/source_bindings.json'; original = pins.read_bytes()
        pins.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'inventory/SHA'):
            self.verify(release, receipt, sha)
        pins.write_bytes(original); (release/'unreviewed.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'inventory/SHA'):
            self.verify(release, receipt, sha)

    def test_launcher_bootstrap_binds_itself_and_all_json_before_import(self):
        release, receipt, sha = self.release()
        shell = (release/'scheduling/run_prefix.sh').read_text()
        bootstrap = shell.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        def run(launcher=None):
            return subprocess.run([sys.executable, '-I', '-B', '-', str(receipt), sha,
                str(launcher or release/'scheduling/run_prefix.sh')], input=bootstrap, text=True, capture_output=True)
        result = run(); self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(release))
        spool = self.root/'slurmd/job-123/slurm_script'; spool.parent.mkdir(parents=True)
        shutil.copyfile(release/'scheduling/run_prefix.sh', spool)
        result = run(spool); self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(release))
        spool.write_text(spool.read_text()+'\n# modified spool copy\n')
        self.assertNotEqual(run(spool).returncode, 0)
        (release/'supplied_package/config/source_bindings.json').write_text('{}')
        self.assertNotEqual(run().returncode, 0)

    def worker(self):
        with patch.object(sys, 'path', [str(PACKAGE), *sys.path]):
            from execution import diagnostic_worker as worker
        return worker

    def active_preflight(self, config_sha=None, freeze_sha=None):
        return dict(schema='sr-satdose-prefix-preflight-v1', status='READY_FOR_PREFIX_CALIB_ONLY',
            permission='NO_TRAINING', model_calls=0, starts=[11031, 11032], responses_per_start=48,
            config_sha256=config_sha or p.CONFIG_SHA, design_freeze_sha256=freeze_sha or p.FREEZE_SHA,
            dry_analyses={}, clarification=str(self.root/'clarification.json'))

    def test_lock_conflict_prevents_preflight_claim_and_subprocess(self):
        worker = self.worker()
        with patch.object(p, 'BASE', self.root), patch.object(p, 'verify_release', return_value={}), \
             patch.object(p, 'preflight') as preflight, patch.object(p.subprocess, 'run') as run:
            with worker._worker_lock(self.root):
                with self.assertRaisesRegex(PermissionError, 'global lock'):
                    p.execute(self.root/'release.json', 'synthetic-test-only')
            preflight.assert_not_called(); run.assert_not_called()
            self.assertFalse((self.root/'prefix_control').exists())

    def test_shared_lock_spans_recheck_claim_both_subprocesses_analysis_and_result(self):
        worker = self.worker(); events = []; write_once = p.calibration.write_once
        def locked(event):
            with self.assertRaisesRegex(PermissionError, 'global lock'):
                with worker._worker_lock(self.root):
                    self.fail('PREFIX released the lock too early')
            events.append(event)
        def preflight(*args):
            locked('preflight'); return self.active_preflight()
        def run(argv, **kwargs):
            locked('subprocess-'+argv[argv.index('--start-seed')+1])
            self.assertEqual(kwargs['cwd'], self.root/'code')
            return SimpleNamespace(returncode=0)
        def analyze(*args, **kwargs):
            locked('analysis-'+str(args[4]))
            return dict(status='PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED', permission='NO_TRAINING')
        def write(path, value):
            locked('write-'+path.name); write_once(path, value)
        with patch.object(p, 'BASE', self.root), patch.object(p, 'verify_release', return_value={}), \
             patch.object(p, 'verify_deployment', return_value={}), patch.object(p, 'protocol_identity'), \
             patch.object(p, 'preflight', side_effect=preflight), patch.object(p.subprocess, 'run', side_effect=run), \
             patch.object(p.calibration, 'analyze_prefix', side_effect=analyze), \
             patch.object(p.calibration, 'write_once', side_effect=write), \
             patch.object(p, 'check_gpu_slots', wraps=p.check_gpu_slots) as slots, \
             patch.dict(os.environ, SLURM_JOB_ID='123', SLURM_NTASKS='1', SLURM_NNODES='1'):
            result = p.execute(self.root/'release.json', 'synthetic-test-only')
        self.assertEqual(result['status'], 'PREFIX_METRICS_COMPLETE_GPU_CHECK_STILL_REQUIRED')
        self.assertEqual(events, ['preflight', 'write-PREFLIGHT.json', 'subprocess-11031', 'analysis-11031',
            'write-PREFIX_11031_ANALYSIS.json', 'subprocess-11032', 'analysis-11032',
            'write-PREFIX_11032_ANALYSIS.json', 'write-RESULT.json'])
        self.assertEqual([call.kwargs for call in slots.call_args_list], [{'current_prefix': True}]*2)
        with worker._worker_lock(self.root):
            pass

    def test_internal_active_prefix_exemption_never_ignores_hold(self):
        worker = self.worker()
        control = self.root/'prefix_control'; control.mkdir()
        fixtures.dump(control/'PREFLIGHT.json', self.active_preflight())
        with patch.object(p, 'BASE', self.root), worker._worker_lock(self.root):
            self.assertIsNone(p.check_gpu_slots(self.root, current_prefix=True))
            fixtures.dump(control/'RESULT.json', dict(status='HOLD_PREFIX_CAPTURE', permission='NO_TRAINING'))
            with self.assertRaisesRegex(PermissionError, 'HOLD/incomplete RESULT'):
                p.check_gpu_slots(self.root, current_prefix=True)
            (control/'RESULT.json').unlink()
            fixtures.dump(control/'HOLD.json', dict(status='HOLD_PREFIX_CONTROL'))
            with self.assertRaisesRegex(PermissionError, 'PREFIX HOLD'):
                p.check_gpu_slots(self.root, current_prefix=True)

    def test_subprocess_failure_or_pending_never_starts_second(self):
        for kind in ('exit', 'pending'):
            with self.subTest(kind=kind):
                launched = []
                def launch(start):
                    launched.append(start); return 2 if kind == 'exit' else 0
                result = p.run_sequence({}, launch,
                    lambda start: dict(status='PENDING_PROTOCOL_CLARIFICATION', permission='NO_TRAINING'),
                    lambda start, value: None)
                self.assertEqual(launched, [11031]); self.assertTrue(result['status'].startswith('HOLD'))
                self.assertEqual(result['permission'], 'NO_TRAINING')

    def test_successful_sequence_has_fixed_order_without_gpu_admission(self):
        launched = []
        def launch(start): launched.append(start); return 0
        result = p.run_sequence({}, launch,
            lambda start: dict(status='PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED', permission='NO_TRAINING'),
            lambda start, value: None)
        self.assertEqual(launched, [11031, 11032])
        self.assertEqual(result['status'], 'PREFIX_METRICS_COMPLETE_GPU_CHECK_STILL_REQUIRED')
        self.assertEqual(result['permission'], 'NO_TRAINING')
        self.assertNotIn('GPU_PASS', str(result))

    def test_command_only_frozen_driver_scope_and_exact_dry_resources(self):
        for start in p.STARTS:
            argv = p.command(start)
            self.assertEqual(argv[argv.index('--scope')+1], 'PREFIX_CALIB')
            self.assertEqual(argv[argv.index('--start-seed')+1], str(start))
            self.assertEqual(argv[argv.index('-m')+1], 'runtime.dry_driver')
            for resource in ('--exclusive', '--exact', '--nodes=1', '--ntasks=1',
                             '--cpus-per-task=8', '--gpus-per-task=pro6000:2', '--mem=66G'):
                self.assertIn(resource, argv)
            self.assertEqual(argv[-1], str(p.BASE/f'runs/PREFIX_CALIB_{start}'))
        with self.assertRaises(ValueError): p.command(11033)

    def test_valid_gpu_binding_and_complete_slots_allowed_hold_unknown_blocked(self):
        # The actual shared worker validator reads only these synthetic files.
        with patch.object(sys, 'path', [str(PACKAGE), *sys.path]):
            from execution import diagnostic_worker as worker
        root = self.root
        config = root/'config.json'; lock = root/'source_lock.json'
        fixtures.dump(config, dict(fixture_only=True)); fixtures.dump(lock, dict(fixture_only=True))
        refs = {name: dict(path=path.name, sha256=core.file_sha(path)) for name, path in
                (('config', config), ('source_lock', lock))}
        freeze = root/'DESIGN_FREEZE.json'; fixtures.dump(freeze, dict(artifacts=refs))
        binding = dict(schema='sr-satdose-technical-execution-v1', mode='production',
            experiment_id='SR-SATDOSE-1-20261112', experiment_root=str(root), **refs,
            design_freeze=dict(path=freeze.name, sha256=core.file_sha(freeze)),
            shadow_claims_root='shadow_claims', shadow_slots=worker.shadow.expected_slots(),
            max_shadow_optimizer_steps=4, steps_per_slot=1,
            backend_equivalence_slots={str(s): f'backend_equivalence/start-{s}' for s in p.STARTS},
            max_backend_equivalence_attempts=2,
            code={'diagnostic_worker': dict(path=worker.__file__, sha256=core.file_sha(worker.__file__))})
        binding_path = root/'TECHNICAL_EXECUTION_BINDING.json'; fixtures.dump(binding_path, binding)
        with patch.object(p, 'FREEZE_SHA', core.file_sha(freeze)):
            identity = p.check_gpu_slots(root)
            self.assertEqual(identity['sha256'], core.file_sha(binding_path))
            control = root/'prefix_control'; control.mkdir()
            fixtures.dump(control/'PREFLIGHT.json', self.active_preflight(
                config_sha=refs['config']['sha256'], freeze_sha=core.file_sha(freeze)))
            with self.assertRaisesRegex(PermissionError, 'PREFIX UNKNOWN'):
                p.check_gpu_slots(root)
            with worker._worker_lock(root):
                self.assertEqual(p.check_gpu_slots(root, current_prefix=True), identity)
                fixtures.dump(control/'RESULT.json', dict(status='HOLD_PREFIX_MEASUREMENT', permission='NO_TRAINING'))
                with self.assertRaisesRegex(PermissionError, 'HOLD/incomplete RESULT'):
                    p.check_gpu_slots(root, current_prefix=True)
            shutil.rmtree(control)
            slot = root/'backend_equivalence/start-11031'; slot.mkdir(parents=True)
            with self.assertRaisesRegex(PermissionError, 'HOLD/UNKNOWN'):
                p.check_gpu_slots(root)
            evidence = {}
            for name in ('SELECTION.json', 'BASELINE.pt', 'TWO_DEVICE.pt', 'COMPARISON.json'):
                (slot/name).write_bytes(b'synthetic CPU metadata/hash fixture, no GPU tensors')
                evidence[name] = dict(path=str((slot/name).relative_to(root)), sha256=core.file_sha(slot/name))
            fixtures.dump(slot/'COMPLETE.json', dict(schema='sr-satdose-backend-equivalence-v1',
                status='BACKEND_EQUIVALENCE_CAPTURED_NOT_GPU_ADMISSION', start_seed=11031, bit_exact=True,
                technical_binding_sha256=core.file_sha(binding_path), close=dict(closed=True),
                permission='NO_TRAINING', optimizer_steps=0, new_generations=0, evidence=evidence))
            self.assertEqual(p.check_gpu_slots(root), identity)
            fixtures.dump(slot/'HOLD.json', dict(fixture_only=True))
            with self.assertRaisesRegex(PermissionError, 'HOLD/UNKNOWN'):
                p.check_gpu_slots(root)
            (slot/'HOLD.json').unlink(); (slot/'BASELINE.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(PermissionError, 'evidence changed'):
                p.check_gpu_slots(root)

    def test_frozen_deployment_and_source_inventory_tampering(self):
        base = self.root/'base'; base.mkdir()
        config = base/'code/supplied_package/config/experiment.json'; config.parent.mkdir(parents=True)
        shutil.copyfile(PACKAGE/'supplied_package/config/experiment.json', config)
        runtime = base/'code/runtime'; runtime.mkdir()
        (runtime/'dry_driver.py').write_text('# synthetic CPU import-origin fixture\n')
        for rel in ('supplied_package/tools/satdose_core.py', 'supplied_package/config/source_bindings.json'):
            target = base/'code'/rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(PACKAGE/rel, target)
        native = base/'native'; native.mkdir(); (native/'native.py').write_text('# native fixture\n')
        legacy = base/'legacy'; legacy.mkdir(); (legacy/'legacy.py').write_text('# legacy fixture\n')
        lock = base/'runtime_source_lock.json'
        fixtures.dump(lock, dict(native_src=str(native), native_files={'native.py': core.file_sha(native/'native.py')},
                                legacy_root=str(legacy), legacy_files={'legacy.py': core.file_sha(legacy/'legacy.py')}))
        freeze = base/'DESIGN_FREEZE.json'
        fixtures.dump(freeze, dict(schema='sr-satdose-design-freeze-v1', experiment_id='SR-SATDOSE-1-20261112',
            config_sha256=p.CONFIG_SHA, runtime_sources={'dry_driver.py': core.file_sha(runtime/'dry_driver.py')},
            artifacts={'source_lock': dict(path=lock.name, sha256=core.file_sha(lock)),
                       'runtime': dict(path='code/runtime/dry_driver.py', sha256=core.file_sha(runtime/'dry_driver.py'))}))
        with patch.object(p, 'BASE', base), patch.object(p, 'FREEZE_SHA', core.file_sha(freeze)):
            self.assertEqual(p.verify_deployment()['native_src'], native)
            (native/'extra.py').write_text('# unexpected\n')
            with self.assertRaisesRegex(ValueError, 'Native source inventory'):
                p.verify_deployment()
            (native/'extra.py').unlink(); (legacy/'legacy.py').write_text('# changed\n')
            with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                p.verify_deployment()


if __name__ == '__main__':
    unittest.main()
