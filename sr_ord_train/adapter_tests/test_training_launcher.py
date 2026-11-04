"""Validate registered paired training dispatch without model or GPU calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TrainingLauncherTests(unittest.TestCase):
    def run_launcher(self, arguments, failing_parent='', resources=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = (Path(__file__).parents[1]/'adapter/run_training_pair.sh').read_text()
            script = script.replace(
                'root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2',
                'root=' + str(root))
            launcher = root/'launcher.sh'
            launcher.write_text(script)
            stub = root/'srun'
            stub.write_text('#!' + sys.executable + '\n' + '''
import json, os, pathlib, sys, time
args = sys.argv[1:]
parent = args[args.index('--parent')+1]
if os.environ.get('SLURM_TRES_PER_TASK'):
    sys.exit(2)
pathlib.Path(parent+'.json').write_text(json.dumps({'argv':args,
    'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}))
if parent != os.environ.get('STUB_FAIL_PARENT'):
    time.sleep(.02)
pathlib.Path(parent+'.done').touch()
sys.exit(7 if parent == os.environ.get('STUB_FAIL_PARENT') else 0)
''')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                       SLURM_MEM_PER_NODE='67584', SLURM_CPUS_ON_NODE='8',
                       SLURM_TRES_PER_TASK='cpu:4', STUB_FAIL_PARENT=failing_parent)
            env.update(resources or {})
            result = subprocess.run(['bash', str(launcher), *arguments], env=env,
                                    capture_output=True, text=True)
            calls = {p.stem:json.loads(p.read_text()) for p in root.glob('b*.json')}
            done = {p.stem for p in root.glob('b*.done')}
            return result, calls, done

    def test_all_registered_pairs_dispatch_exact_training_identity_and_resources(self):
        for arm in ('R', 'G', 'D'):
            for seed in ('11031', '11032'):
                with self.subTest(arm=arm, seed=seed):
                    result, calls, done = self.run_launcher([arm, seed])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(set(calls), {'b0', 'b1'})
                    self.assertEqual(done, {'b0', 'b1'})
                    for parent, call in calls.items():
                        args = call['argv']
                        for flag, value in (('--parent', parent), ('--mode', 'TRAINING'),
                                            ('--arm', arm), ('--train-seed', seed),
                                            ('--binding', 'execution/RUN_BINDING.json')):
                            self.assertEqual(args[args.index(flag)+1], value)
                        for option in ('--cpus-per-task=4', '--gpus-per-task=1',
                                       '--mem=33792M', '--exclusive', '--exact'):
                            self.assertIn(option, args)
                        self.assertEqual(call['cublas'], ':4096:8')

    def test_invalid_or_extra_arguments_never_dispatch(self):
        for args in ([], ['R'], ['R', '11031', 'extra'], ['M', '11031'],
                     ['R', '11033'], ['r', '11031'], ['D', '011031']):
            with self.subTest(args=args):
                result, calls, done = self.run_launcher(args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, {})
                self.assertEqual(done, set())

    def test_either_worker_failure_waits_for_both_and_fails_job(self):
        for parent in ('b0', 'b1'):
            with self.subTest(parent=parent):
                result, calls, done = self.run_launcher(['D', '11032'], failing_parent=parent)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(set(calls), {'b0', 'b1'})
                self.assertEqual(done, {'b0', 'b1'})

    def test_invalid_granted_resources_never_dispatch(self):
        for resources in ({'SLURM_MEM_PER_NODE': ''}, {'SLURM_CPUS_ON_NODE': '1'},
                          {'SLURM_MEM_PER_NODE': 'invalid'}):
            with self.subTest(resources=resources):
                result, calls, done = self.run_launcher(['R', '11031'], resources=resources)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, {})
                self.assertEqual(done, set())


if __name__ == '__main__':
    unittest.main()
