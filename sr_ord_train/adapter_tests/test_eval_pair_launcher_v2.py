"""Synthetic launcher tests: mock Slurm and model Python; never access a GPU."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


class EvaluationPairLauncherV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.calls = self.root / 'calls'
        self.calls.mkdir()
        self.endpoints = [f'{p}-PARENT' for p in ('b0', 'b1')] + [
            f'{p}-{s}-{a}-step-{step:03d}' for p in ('b0', 'b1')
            for s in (11031, 11032) for a in ('R', 'G', 'D') for step in (12, 24)]
        self.lock = self.root / 'ENDPOINT_LOCK.json'
        self.lock.write_text(json.dumps(dict(verified_paths=12, optimizer_updates=288,
            training_responses=36864, endpoints={e: dict(endpoint_id=e) for e in self.endpoints})))
        self.binding = self.root / 'EVAL_BINDING.json'
        self.document = dict(schema='sr-ord-eval-binding-v1', experiment_id='SR-ORD-TRAIN-1',
            package_root=str(self.root), worker_shards=1, endpoint_count=26, registered_responses=26624,
            status='REGISTERED_NOT_GENERATED', optimizer_created=False, parameter_updates=0,
            endpoint_lock=dict(path=str(self.lock), sha256=hashlib.sha256(self.lock.read_bytes()).hexdigest()))
        self.save_binding()
        python = self.bin / 'fake-python'
        python.write_text('#!' + sys.executable + '\n' + '''import json, os, sys, time
from pathlib import Path
if sys.argv[1:3] == ['-B', '-']:
    os.execv(sys.executable, [sys.executable] + sys.argv[1:])
assert sys.argv[1:5] == ['-B', '-m', 'adapter.evaluation', 'worker']
endpoint = sys.argv[sys.argv.index('--endpoint') + 1]
calls = Path(os.environ['PAIR_TEST_CALLS'])
(calls / (endpoint + '.started')).write_text(json.dumps(sys.argv[1:]))
if endpoint != os.environ.get('PAIR_TEST_FAIL'):
    time.sleep(0.15)
(calls / (endpoint + '.finished')).write_text('preserved')
sys.exit(9 if endpoint == os.environ.get('PAIR_TEST_FAIL') else 0)
''')
        python.chmod(0o755)
        srun = self.bin / 'srun'
        srun.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
index = next(i for i, a in enumerate(args) if not a.startswith('--'))
endpoint = args[args.index('--endpoint') + 1]
polluting = ('SLURM_GPUS', 'SLURM_GPUS_PER_NODE', 'SLURM_GPUS_PER_TASK',
    'SLURM_GRES', 'SLURM_STEP_GRES', 'SLURM_GRES_FLAGS', 'SLURM_TRES_PER_TASK',
    'SLURM_GPU_BIND', 'SLURM_TRES_BIND', 'SLURM_CPUS_PER_GPU', 'SLURM_MEM_PER_GPU',
    'SLURM_MEM_PER_CPU', 'SLURM_NTASKS_PER_GPU')
if '--gpus-per-task=pro6000:1' not in args:
    sys.stderr.write('srun: error: Invalid GRES specification (with and without type identification)\\n')
    sys.exit(1)
assert not any(k in os.environ for k in polluting)
assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-a,GPU-b'
assert os.environ['SLURM_JOB_ID'] == '515100'
assert os.environ['SLURM_JOB_GPUS'] == '2,3'
assert os.environ['SLURM_GPUS_ON_NODE'] == '2'
(Path(os.environ['PAIR_TEST_CALLS']) / (endpoint + '.srun')).write_text(json.dumps(args[:index]))
os.execv(args[index], args[index:])
''')
        srun.chmod(0o755)
        source = Path(__file__).resolve().parents[1] / 'evaluation_v2/run_endpoint_pair.sh'
        script = source.read_text().replace(
            'root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2',
            'root=' + shlex.quote(str(self.root))).replace(
            'python_bin=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12',
            'python_bin=' + shlex.quote(str(python)))
        self.script = self.root / 'launcher.sh'
        self.script.write_text(script)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
            SLURM_MEM_PER_NODE='67584', SLURM_CPUS_ON_NODE='8', SLURM_GPUS_ON_NODE='2',
            CUDA_VISIBLE_DEVICES='GPU-a,GPU-b', SLURM_TRES_PER_TASK='cpu=8', PAIR_TEST_CALLS=str(self.calls))
        # Actual typed sbatch shape plus conflicting inherited option variants.
        self.env.update(SLURM_JOB_ID='515100', SLURM_JOB_GPUS='2,3',
            SLURM_GPUS='pro6000:2', SLURM_GPUS_PER_NODE='pro6000:2',
            SLURM_GPUS_PER_TASK='2', SLURM_GRES='gpu:pro6000:2',
            SLURM_STEP_GRES='gpu:2', SLURM_GRES_FLAGS='allow-task-sharing',
            SLURM_TRES_PER_TASK='cpu=8,gres/gpu:pro6000=2',
            SLURM_GPU_BIND='none', SLURM_TRES_BIND='gres/gpu:none',
            SLURM_CPUS_PER_GPU='8', SLURM_MEM_PER_GPU='67584',
            SLURM_MEM_PER_CPU='8448', SLURM_NTASKS_PER_GPU='2')
        self.env.pop('PAIR_TEST_FAIL', None)

    def tearDown(self):
        self.temp.cleanup()

    def save_binding(self):
        self.binding.write_text(json.dumps(self.document))

    def launch(self, a='b0-PARENT', b='b1-PARENT', binding=None):
        return subprocess.run(['bash', str(self.script), str(binding or self.binding), a, b],
            env=self.env, capture_output=True, text=True, timeout=10)

    def assert_denied(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(list(self.calls.iterdir()), [], result.stderr)

    def test_disjoint_endpoint_arguments_and_exclusive_resources(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        for endpoint in ('b0-PARENT', 'b1-PARENT'):
            args = json.loads((self.calls / (endpoint + '.started')).read_text())
            self.assertEqual(args, ['-B', '-m', 'adapter.evaluation', 'worker', '--binding',
                str(self.binding), '--endpoint', endpoint, '--shard-index', '0'])
            resources = json.loads((self.calls / (endpoint + '.srun')).read_text())
            self.assertEqual(resources, ['--exclusive', '--exact', '--nodes=1', '--ntasks=1',
                '--cpus-per-task=4', '--gpus-per-task=pro6000:1', '--mem=33792M'])
            self.assertTrue((self.calls / (endpoint + '.finished')).exists())

    def test_either_failure_waits_for_and_preserves_peer(self):
        for failed in ('b0-PARENT', 'b1-PARENT'):
            with self.subTest(failed=failed):
                for path in self.calls.iterdir():
                    path.unlink()
                self.env['PAIR_TEST_FAIL'] = failed
                result = self.launch()
                self.assertEqual(result.returncode, 1, result.stderr)
                for endpoint in ('b0-PARENT', 'b1-PARENT'):
                    self.assertEqual((self.calls / (endpoint + '.finished')).read_text(), 'preserved')

    def test_same_empty_or_invalid_endpoint_never_launches(self):
        for a, b in [('b0-PARENT', 'b0-PARENT'), ('', 'b1-PARENT'),
                     ('../bad', 'b1-PARENT'), ('b0-PARENT;false', 'b1-PARENT')]:
            with self.subTest(a=a, b=b):
                self.assert_denied(self.launch(a, b))

    def test_existing_either_output_never_launches_and_is_preserved(self):
        for endpoint in ('b0-PARENT', 'b1-PARENT'):
            with self.subTest(endpoint=endpoint):
                stage = self.root / 'run/evaluation' / endpoint
                stage.mkdir(parents=True)
                evidence = stage / 'original'
                evidence.write_text('retain')
                self.assert_denied(self.launch())
                self.assertEqual(evidence.read_text(), 'retain')
                evidence.unlink(); stage.rmdir()

    def test_dangling_output_symlink_never_launches(self):
        stage = self.root / 'run/evaluation/b1-PARENT'
        stage.parent.mkdir(parents=True)
        stage.symlink_to(self.root / 'missing')
        self.assert_denied(self.launch())
        self.assertTrue(stage.is_symlink())

    def test_bad_binding_shards_or_lock_hash_never_launches(self):
        for key, value in [('worker_shards', 2), ('worker_shards', True), ('package_root', '/wrong'),
                           ('registered_responses', 512)]:
            with self.subTest(key=key, value=value):
                original = self.document[key]
                self.document[key] = value
                self.save_binding()
                self.assert_denied(self.launch())
                self.document[key] = original
        self.save_binding()
        self.lock.write_text(self.lock.read_text() + ' ')
        self.assert_denied(self.launch())

    def test_invalid_resources_or_gpu_visibility_never_launches(self):
        for key, value in [('SLURM_CPUS_ON_NODE', '4'), ('SLURM_MEM_PER_NODE', '33000'),
                           ('CUDA_VISIBLE_DEVICES', '0'), ('CUDA_VISIBLE_DEVICES', '0,0'),
                           ('CUDA_VISIBLE_DEVICES', '0,1,2'), ('SLURM_GPUS_ON_NODE', '3')]:
            with self.subTest(key=key, value=value):
                old = self.env[key]
                self.env[key] = value
                self.assert_denied(self.launch())
                self.env[key] = old

    def test_v1_immutable_and_reproduces_typed_untyped_rejection(self):
        source = Path(__file__).resolve().parents[1] / 'evaluation_v1/run_endpoint_pair.sh'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
            'f8e10d573640293b6e1c8a1abcb817802a79a9c1f692bc5c4f0b987d539d9c8e')
        # Retain v2 guards/fixture, replace only its step command by the old
        # untyped request. The Slurm mock rejects it before model Python starts.
        self.script.write_text(self.script.read_text().replace('--gpus-per-task=pro6000:1', '--gpus-per-task=1'))
        result = self.launch()
        self.assert_denied(result)
        self.assertIn('Invalid GRES specification', result.stderr)

    def test_relative_binding_or_wrong_arity_never_launches(self):
        self.assert_denied(self.launch(binding='EVAL_BINDING.json'))
        result = subprocess.run(['bash', str(self.script), str(self.binding)], env=self.env,
            capture_output=True, text=True, timeout=10)
        self.assert_denied(result)


if __name__ == '__main__':
    unittest.main()
