import json
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
class GpuFullBatchLauncherTests(unittest.TestCase):
    def run_launcher(self, overrides=None, args=('b0-11031-R',), fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = (ROOT/'diagnostics/run_gpu_full_batch_probe.sh').read_text().replace(
                'root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2', 'root='+str(root))
            (root/'run.sh').write_text(script)
            stub = root/'srun'
            stub.write_text('#!'+sys.executable+'\n'+'''
import json,os,pathlib,sys
assert 'SLURM_TRES_PER_TASK' not in os.environ
assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
pathlib.Path('call.json').write_text(json.dumps(sys.argv[1:]))
sys.exit(7 if os.environ.get('FAIL')=='1' else 0)
''')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'], SLURM_JOB_ID='123',
                       SLURM_MEM_PER_NODE='67584', SLURM_CPUS_ON_NODE='8', SLURM_NTASKS='1',
                       SLURM_NNODES='1', SLURM_TRES_PER_TASK='cpu:8', PYTORCH_CUDA_ALLOC_CONF='legacy',
                       FAIL='1' if fail else '0')
            env.update(overrides or {})
            result = subprocess.run(['bash',str(root/'run.sh'),*args], env=env, capture_output=True, text=True)
            call = json.loads((root/'call.json').read_text()) if (root/'call.json').exists() else None
            return result, call

    def test_exact_two_gpu_single_step_uses_granted_resources(self):
        result, call = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ('--gpus-per-task=pro6000:2','--mem=67584M','--cpus-per-task=8','--ntasks=1',
                      'diagnostics.gpu_full_batch_probe','run/gpu_fullbatch_diag_123_b0-11031-R','--binding','execution/RUN_BINDING.json','--worker-memory-mib','67584','--path','b0-11031-R'):
            self.assertIn(value, call)
        result, _ = self.run_launcher(fail=True)
        self.assertEqual(result.returncode, 7)

    def test_invalid_grants_or_arguments_never_dispatch(self):
        for overrides, args in (({'SLURM_MEM_PER_NODE':''},('b0-11031-R',)), ({'SLURM_CPUS_ON_NODE':'0'},('b0-11031-R',)),
                                ({'SLURM_NTASKS':'2'},('b0-11031-R',)), ({'SLURM_NNODES':'2'},('b0-11031-R',)), ({},()), ({},('extra',))):
            result, call = self.run_launcher(overrides, args)
            self.assertEqual(result.returncode, 2)
            self.assertIsNone(call)
