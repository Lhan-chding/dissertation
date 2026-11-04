"""Exercise mixed registered-path dispatch without model or GPU calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class MixedLauncherTests(unittest.TestCase):
    def run_launcher(self, arguments, failing_path='', resources=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = (Path(__file__).parents[1]/'adapter/run_training_mixed_pair.sh').read_text()
            script = script.replace(
                'root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2',
                'root=' + str(root))
            launcher = root/'launcher.sh'
            launcher.write_text(script)
            stub = root/'srun'
            stub.write_text('#!' + sys.executable + '\n' + '''
import json, os, pathlib, sys, time
args = sys.argv[1:]
identity = '-'.join(args[args.index(flag)+1] for flag in ('--parent', '--arm', '--train-seed'))
if os.environ.get('SLURM_TRES_PER_TASK'):
    sys.exit(2)
with pathlib.Path(identity+'.calls').open('a') as stream:
    stream.write('1\\n')
pathlib.Path(identity+'.json').write_text(json.dumps({'argv':args,
    'allocator':os.environ.get('PYTORCH_ALLOC_CONF'),
    'legacy_allocator_present':'PYTORCH_CUDA_ALLOC_CONF' in os.environ,
    'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}))
if identity != os.environ.get('STUB_FAIL_PATH'):
    time.sleep(.02)
pathlib.Path(identity+'.done').touch()
sys.exit(7 if identity == os.environ.get('STUB_FAIL_PATH') else 0)
''')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                       SLURM_MEM_PER_NODE='67584', SLURM_CPUS_ON_NODE='8',
                       SLURM_TRES_PER_TASK='cpu:4', STUB_FAIL_PATH=failing_path,
                       PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:32',
                       PYTORCH_ALLOC_CONF='expandable_segments:False')
            env.update(resources or {})
            result = subprocess.run(['bash', str(launcher), *arguments], env=env,
                                    capture_output=True, text=True)
            calls = {p.stem:json.loads(p.read_text()) for p in root.glob('*.json')}
            for identity, call in calls.items():
                call['invocations'] = len((root/(identity+'.calls')).read_text().splitlines())
            done = {p.stem for p in root.glob('*.done')}
            return result, calls, done

    def test_distinct_registered_pairs_preserve_identity_allocator_and_resources(self):
        pairs = [(['b0', 'R', '11031'], ['b0', 'D', '11031']),
                 (['b1', 'G', '11032'], ['b0', 'G', '11031']),
                 (['b1', 'R', '11031'], ['b1', 'R', '11032'])]
        for first, second in pairs:
            with self.subTest(first=first, second=second):
                result, calls, done = self.run_launcher(first+second)
                identities = {'-'.join(first), '-'.join(second)}
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(set(calls), identities)
                self.assertEqual(done, identities)
                self.assertIn('/'.join(first)+' '+ '/'.join(second), result.stdout)
                self.assertIn('PYTORCH_ALLOC_CONF=expandable_segments:True', result.stdout)
                self.assertIn('PYTORCH_CUDA_ALLOC_CONF=unset', result.stdout)
                for parent, arm, seed in (first, second):
                    call = calls['-'.join((parent, arm, seed))]
                    args = call['argv']
                    for flag, value in (('--parent', parent), ('--mode', 'TRAINING'),
                                        ('--arm', arm), ('--train-seed', seed),
                                        ('--binding', 'execution/RUN_BINDING.json')):
                        self.assertEqual(args[args.index(flag)+1], value)
                    for option in ('--cpus-per-task=4', '--gpus-per-task=1',
                                   '--mem=33792M', '--exclusive', '--exact'):
                        self.assertIn(option, args)
                    self.assertEqual(call['allocator'], 'expandable_segments:True')
                    self.assertFalse(call['legacy_allocator_present'])
                    self.assertEqual(call['cublas'], ':4096:8')
                    self.assertEqual(call['invocations'], 1)

    def test_invalid_or_duplicate_arguments_rejected_before_dispatch(self):
        valid = ['b0', 'R', '11031', 'b0', 'D', '11031']
        invalid = [[], valid[:-1], valid+['extra'], valid[:3]*2]
        for index, values in ((0, ['b2', '']), (1, ['M', 'r']), (2, ['11033', '011031']),
                              (3, ['b2', '']), (4, ['M', 'd']), (5, ['11033', '011031'])):
            for value in values:
                arguments = valid.copy()
                arguments[index] = value
                invalid.append(arguments)
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                result, calls, done = self.run_launcher(arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, {})
                self.assertEqual(done, set())

    def test_either_worker_failure_waits_for_both_without_retry(self):
        for identity in ('b0-R-11031', 'b0-D-11031'):
            with self.subTest(identity=identity):
                result, calls, done = self.run_launcher(
                    ['b0', 'R', '11031', 'b0', 'D', '11031'], failing_path=identity)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(set(calls), {'b0-R-11031', 'b0-D-11031'})
                self.assertEqual(done, set(calls))
                self.assertTrue(all(call['invocations'] == 1 for call in calls.values()))

    def test_invalid_granted_resources_never_dispatch(self):
        for resources in ({'SLURM_MEM_PER_NODE': ''}, {'SLURM_CPUS_ON_NODE': '1'},
                          {'SLURM_MEM_PER_NODE': 'invalid'}, {'SLURM_MEM_PER_NODE': '1'}):
            with self.subTest(resources=resources):
                result, calls, done = self.run_launcher(
                    ['b0', 'R', '11031', 'b0', 'D', '11031'], resources=resources)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, {})
                self.assertEqual(done, set())


if __name__ == '__main__':
    unittest.main()
