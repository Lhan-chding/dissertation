"""Check fixed recovery routing and allocator setup without running a model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RecoveryLauncherTests(unittest.TestCase):
    def run_launcher(self, arguments=(), failing_arm='', resources=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = (Path(__file__).parents[1]/'adapter/run_b1_rg_recovery.sh').read_text()
            script = script.replace(
                'root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2',
                'root=' + str(root))
            launcher = root/'launcher.sh'
            launcher.write_text(script)
            stub = root/'srun'
            stub.write_text('#!' + sys.executable + '\n' + '''
import json, os, pathlib, sys, time
args = sys.argv[1:]
arm = args[args.index('--arm')+1]
if os.environ.get('SLURM_TRES_PER_TASK'):
    sys.exit(2)
with pathlib.Path(arm+'.calls').open('a') as stream:
    stream.write('1\\n')
pathlib.Path(arm+'.json').write_text(json.dumps({'argv':args,
    'allocator':os.environ.get('PYTORCH_ALLOC_CONF'),
    'legacy_allocator_present':'PYTORCH_CUDA_ALLOC_CONF' in os.environ,
    'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}))
if arm != os.environ.get('STUB_FAIL_ARM'):
    time.sleep(.02)
pathlib.Path(arm+'.done').touch()
sys.exit(7 if arm == os.environ.get('STUB_FAIL_ARM') else 0)
''')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                       SLURM_MEM_PER_NODE='67584', SLURM_CPUS_ON_NODE='8',
                       SLURM_TRES_PER_TASK='cpu:4', STUB_FAIL_ARM=failing_arm,
                       PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:32',
                       PYTORCH_ALLOC_CONF='expandable_segments:False')
            env.update(resources or {})
            result = subprocess.run(['bash', str(launcher), *arguments], env=env,
                                    capture_output=True, text=True)
            calls = {p.stem:json.loads(p.read_text()) for p in root.glob('*.json')}
            for arm, call in calls.items():
                call['invocations'] = len((root/(arm+'.calls')).read_text().splitlines())
            done = {p.stem for p in root.glob('*.done')}
            return result, calls, done

    def test_fixed_b1_rg_identity_allocator_and_granted_resources(self):
        result, calls, done = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(calls), {'R', 'G'})
        self.assertEqual(done, {'R', 'G'})
        self.assertIn('PYTORCH_ALLOC_CONF=expandable_segments:True', result.stdout)
        self.assertIn('PYTORCH_CUDA_ALLOC_CONF=unset', result.stdout)
        for arm, call in calls.items():
            args = call['argv']
            for flag, value in (('--parent', 'b1'), ('--mode', 'TRAINING'),
                                ('--arm', arm), ('--train-seed', '11031'),
                                ('--binding', 'execution/RUN_BINDING.json')):
                self.assertEqual(args[args.index(flag)+1], value)
            for option in ('--cpus-per-task=4', '--gpus-per-task=1',
                           '--mem=33792M', '--exclusive', '--exact'):
                self.assertIn(option, args)
            self.assertEqual(call['allocator'], 'expandable_segments:True')
            self.assertFalse(call['legacy_allocator_present'])
            self.assertEqual(call['cublas'], ':4096:8')

    def test_any_argument_is_rejected_before_dispatch(self):
        for arguments in (['R'], ['b0', '11031'], ['--help'], ['']):
            with self.subTest(arguments=arguments):
                result, calls, done = self.run_launcher(arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, {})
                self.assertEqual(done, set())

    def test_either_arm_failure_waits_for_both_without_retry(self):
        for arm in ('R', 'G'):
            with self.subTest(arm=arm):
                result, calls, done = self.run_launcher(failing_arm=arm)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(set(calls), {'R', 'G'})
                self.assertEqual(done, {'R', 'G'})
                self.assertTrue(all(call['invocations'] == 1 for call in calls.values()))

    def test_invalid_granted_resources_never_dispatch(self):
        for resources in ({'SLURM_MEM_PER_NODE': ''}, {'SLURM_CPUS_ON_NODE': '1'},
                          {'SLURM_MEM_PER_NODE': 'invalid'}):
            with self.subTest(resources=resources):
                result, calls, done = self.run_launcher(resources=resources)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, {})
                self.assertEqual(done, set())


if __name__ == '__main__':
    unittest.main()
