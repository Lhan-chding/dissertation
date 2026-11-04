"""Exercise real shell dispatch without allocating GPUs or loading a model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def run_launcher(self, failing_parent=''):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = (Path(__file__).parents[1]/'adapter/run_smoke_pair.sh').read_text()
            script = script.replace(
                'root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2',
                'root=' + str(root))
            launcher = root/'launcher.sh'
            launcher.write_text(script)
            stub = root/'srun'
            stub.write_text('#!' + sys.executable + '\n' + '''
import json, os, pathlib, sys
args = sys.argv[1:]
parent = args[args.index('--parent')+1]
if os.environ.get('SLURM_TRES_PER_TASK'):
    sys.exit(2)  # Actual Slurm rejects inherited TRES together with --gpus-per-task.
pathlib.Path(parent+'.json').write_text(json.dumps({'argv':args,
    'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}))
sys.exit(7 if parent == os.environ.get('STUB_FAIL_PARENT') else 0)
''')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                       SLURM_MEM_PER_NODE='67584', SLURM_CPUS_ON_NODE='8',
                       SLURM_TRES_PER_TASK='cpu:4', STUB_FAIL_PARENT=failing_parent)
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True)
            calls = {p.stem:json.loads(p.read_text()) for p in root.glob('b*.json')}
            return result, calls

    def test_inherited_tres_does_not_conflict_with_two_explicit_gpu_steps(self):
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(calls), {'b0','b1'})
        for call in calls.values():
            self.assertIn('--cpus-per-task=4', call['argv'])
            self.assertIn('--gpus-per-task=1', call['argv'])
            self.assertIn('--mem=33792M', call['argv'])
            self.assertEqual(call['cublas'], ':4096:8')

    def test_one_failed_worker_fails_job_after_both_workers_return(self):
        result, calls = self.run_launcher('b1')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(set(calls), {'b0','b1'})


if __name__ == '__main__':
    unittest.main()
