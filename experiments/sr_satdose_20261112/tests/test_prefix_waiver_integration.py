"""Real CPU verifier + worker/launcher gates; every calibration/model row is synthetic.

No model, Slurm, GPU, actual user authorization or training admission is exercised.
The fixtures are explicitly source-bound synthetic records, not production evidence.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

TESTS=Path(__file__).resolve().parent
PACKAGE=TESTS.parent
sys.path.insert(0,str(PACKAGE))
from execution import diagnostic_worker as worker


def fixture_module(name):
    spec=importlib.util.spec_from_file_location('_integration_'+name,TESTS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


waiver_fixture=fixture_module('test_prefix_waiver')
launcher_fixture=fixture_module('test_gpu_checks_launcher')


class RealPrefixWaiverGateIntegration(unittest.TestCase):
    def test_worker_preserves_actual_failed_measurement_and_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();binding=waiver_fixture.make_fixture(root)
            source=PACKAGE/'execution/prefix_waiver.py'
            binding['code']={'prefix_waiver':dict(path=str(source),sha256=worker.shadow.file_sha(source))}
            original=root/'prefix_control/RESULT.json';original_bytes=original.read_bytes()
            worker._prefix_gate(root,binding)
            self.assertEqual(original.read_bytes(),original_bytes)
            self.assertEqual(json.loads(original_bytes)['status'],'HOLD_PREFIX_MEASUREMENT')
            with self.assertRaisesRegex(PermissionError,'Active PREFIX'):
                worker._prefix_gate(root,binding,current_prefix=True)
            no_waiver={k:v for k,v in binding.items() if k not in ('prefix_waiver','prefix_continuation','code')}
            with self.assertRaisesRegex(PermissionError,'HOLD/incomplete RESULT'):worker._prefix_gate(root,no_waiver)
            (root/'prefix_waiver_control/CLAIM.json').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):worker._prefix_gate(root,binding)

    def launcher(self):
        fixture=launcher_fixture.GPUChecksLauncherTests('test_full_release_spooled_bytes_and_live_all_ancestor_space')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        verified=waiver_fixture.make_fixture(fixture.root)
        fixture.binding.update(verified);fixture.freeze_sha=verified['design_freeze']['sha256']
        for path in (fixture.root/'code/runtime').glob('*.py'):
            shutil.copyfile(path,fixture.release/'runtime'/path.name)
        source=fixture.release/'execution/prefix_waiver.py'
        shutil.copyfile(PACKAGE/'execution/prefix_waiver.py',source)
        fixture.binding['code']['prefix_waiver']=dict(path=str(source),sha256=launcher_fixture.sha(source))
        receipt=json.loads(fixture.receipt.read_text())
        receipt.update(design_freeze_sha256=fixture.freeze_sha,config_sha256=verified['config']['sha256'],
            sources={str(p.relative_to(fixture.release)):launcher_fixture.sha(p)
                     for p in fixture.release.rglob('*') if p.is_file()})
        launcher_fixture.dump(fixture.receipt,receipt);fixture.binding['gpu_release']=fixture.ref(fixture.receipt)
        budget=json.loads(fixture.budget.read_text());budget['design_freeze_sha256']=fixture.freeze_sha
        launcher_fixture.dump(fixture.budget,budget);fixture.binding['storage_budget']=fixture.ref(fixture.budget)
        fixture.bind();return fixture

    def test_launcher_full_inventory_then_actual96_verifier_not_pass_string(self):
        fixture=self.launcher();original=(fixture.root/'prefix_control/RESULT.json').read_bytes()
        self.assertEqual(fixture.bootstrap()[0],str(fixture.release))
        self.assertEqual((fixture.root/'prefix_control/RESULT.json').read_bytes(),original)
        raw=fixture.root/'runs/PREFIX_CALIB_11032/PREFIX_RAW.jsonl'
        raw.write_bytes(raw.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):fixture.bootstrap()

    def test_launcher_complete_label_cannot_override_missing_claim(self):
        fixture=self.launcher()
        (fixture.root/'prefix_waiver_control/CLAIM.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):fixture.bootstrap()


if __name__=='__main__':unittest.main()
