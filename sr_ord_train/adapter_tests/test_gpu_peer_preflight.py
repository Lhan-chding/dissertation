"""Hardware preflight orchestration tests; no GPU/model calls in these fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.gpu_peer_preflight import probe, require_output


class FakeCuda:
    def __init__(self, count=2, peer=(True, True), names=None):
        self.count, self.peer = count, peer
        self.names = names or ['NVIDIA RTX PRO 6000', 'NVIDIA RTX PRO 6000']
    def is_available(self): return True
    def device_count(self): return self.count
    def get_device_properties(self, index):
        return SimpleNamespace(name=self.names[index], major=12, minor=0, total_memory=96*1024**3,
                               multi_processor_count=188, uuid='fixture-'+str(index))
    def can_device_access_peer(self, a, b): return self.peer[a]
    def mem_get_info(self, index): return (80*1024**3, 96*1024**3)
    def memory_allocated(self, index): return 0
    def memory_reserved(self, index): return 0
    def max_memory_allocated(self, index): return 0
    def max_memory_reserved(self, index): return 0
    def reset_peak_memory_stats(self, index): pass


class GpuPeerPreflightTests(unittest.TestCase):
    def run_probe(self, cuda=None, query_code=0):
        commands, transfers = [], []
        def command(args):
            commands.append(args)
            return {'argv':args, 'returncode':query_code, 'stdout':'original topology output', 'stderr':''}
        def transfer(torch, size):
            transfers.append(size)
            return {'payload_mib':size, 'exact_round_trip':True}
        result = probe(SimpleNamespace(__version__='fixture', version=SimpleNamespace(cuda='fixture'),
                                       cuda=cuda or FakeCuda()), command, transfer)
        return result, commands, transfers

    def test_bidirectional_peer_gate_and_raw_queries_precede_fixed_gpu_transfers(self):
        result, commands, transfers = self.run_probe()
        self.assertEqual(result['status'], 'GPU_PEER_HARDWARE_PASS_NOT_MODEL_EQUIVALENCE')
        self.assertEqual(transfers, [64, 128, 256])
        self.assertEqual(commands, [['nvidia-smi','topo','-m'], ['nvidia-smi','topo','-p2p','r'],
                                    ['nvidia-smi','topo','-p2p','w']])
        self.assertTrue(all(c['stdout'] == 'original topology output' for c in result['topology_commands']))
        self.assertFalse(result['cpu_offload'])

    def test_missing_peer_access_wrong_hardware_or_query_failure_never_transfers(self):
        for cuda, code in ((FakeCuda(peer=(False,True)),0), (FakeCuda(peer=(True,False)),0),
                           (FakeCuda(count=1),0), (FakeCuda(names=['NVIDIA RTX PRO 6000','Other']),0),
                           (FakeCuda(),1)):
            with self.subTest(cuda=cuda, code=code):
                result, _, transfers = self.run_probe(cuda, code)
                self.assertEqual(result['status'], 'BLOCKED')
                self.assertEqual(transfers, [])

    def test_output_is_new_exact_job_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory).resolve()
            (package/'run').mkdir()
            with patch('diagnostics.gpu_peer_preflight.PACKAGE', package):
                expected = package/'run/gpu_peer_123'
                self.assertEqual(require_output(expected, '123'), expected)
                for output, job in ((expected,'123'),(package/'run/paths','123'),(package/'run/gpu_peer_x','x')):
                    with self.assertRaises(PermissionError):
                        require_output(output, job)

    def test_value_mismatch_stops_without_retry_or_larger_transfer(self):
        calls = []
        def command(args):
            return {'argv':args, 'returncode':0, 'stdout':'topology', 'stderr':''}
        def transfer(torch, size):
            calls.append(size)
            return {'payload_mib':size, 'exact_round_trip':False}
        result = probe(SimpleNamespace(__version__='fixture', version=SimpleNamespace(cuda='fixture'),
                                       cuda=FakeCuda()), command, transfer)
        self.assertEqual(result['status'], 'FAILED')
        self.assertEqual(calls, [64])
        self.assertEqual(len(result['transfers']), 1)


class GpuPeerLauncherTests(unittest.TestCase):
    def run_launcher(self, overrides=None, args=(), fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = (ROOT/'diagnostics/run_gpu_peer_preflight.sh').read_text().replace(
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
        for value in ('--gpus-per-task=2','--mem=67584M','--cpus-per-task=8','--ntasks=1',
                      'diagnostics.gpu_peer_preflight','run/gpu_peer_123'):
            self.assertIn(value, call)
        result, _ = self.run_launcher(fail=True)
        self.assertEqual(result.returncode, 7)

    def test_invalid_grants_or_arguments_never_dispatch(self):
        for overrides, args in (({'SLURM_MEM_PER_NODE':''},()), ({'SLURM_CPUS_ON_NODE':'0'},()),
                                ({'SLURM_NTASKS':'2'},()), ({'SLURM_NNODES':'2'},()), ({},('extra',))):
            result, call = self.run_launcher(overrides, args)
            self.assertEqual(result.returncode, 2)
            self.assertIsNone(call)


if __name__ == '__main__':
    unittest.main()
