"""No model calls: verify diagnostic-only dispatch and resource rejection."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class MemoryProbeLauncherTests(unittest.TestCase):
    def run_script(self, memory='33792', tasks='1', failing=''):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (Path(__file__).parents[1] / 'diagnostics/run_memory_probe.sh').read_text()
            source = source.replace('root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2',
                                    'root=' + str(root))
            (root / 'run.sh').write_text(source)
            stub = root / 'srun'
            stub.write_text('#!' + sys.executable + '\n' + '''
import json,os,pathlib,sys
args=sys.argv[1:]
assert 'SLURM_TRES_PER_TASK' not in os.environ
assert os.environ['PYTORCH_ALLOC_CONF']=='expandable_segments:True'
assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ
path=args[args.index('--path')+1]
with pathlib.Path('calls').open('a') as f:f.write(json.dumps(args)+'\\n')
sys.exit(1 if path==os.environ.get('FAIL_PATH') else 0)
''')
            stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],SLURM_JOB_ID='123',
                     SLURM_MEM_PER_NODE=memory,SLURM_CPUS_ON_NODE='4',SLURM_NTASKS=tasks,
                     SLURM_TRES_PER_TASK='cpu:4',PYTORCH_CUDA_ALLOC_CONF='old',FAIL_PATH=failing)
            result=subprocess.run(['bash',str(root/'run.sh')],env=env,capture_output=True,text=True)
            import json
            calls=[json.loads(line) for line in (root/'calls').read_text().splitlines()] if (root/'calls').exists() else []
            return result,calls

    def test_dispatches_each_frozen_probe_once_even_after_failure(self):
        result,calls=self.run_script(failing='b0-11031-R')
        self.assertEqual(result.returncode,1,result.stderr)
        self.assertEqual([args[args.index('--path')+1] for args in calls],
                         ['b0-11031-R','b1-11031-R','b1-11031-G'])
        for args in calls:
            self.assertIn('diagnostics.memory_equivalence',args)
            self.assertIn('--gpus-per-task=1',args)
            self.assertIn('--mem=33792M',args)
            self.assertIn('run/memory_diag_123_'+args[args.index('--path')+1],args)

    def test_rejects_small_memory_and_shared_task_grants(self):
        for memory,tasks in [('16384','1'),('33792','2'),('','1')]:
            result,calls=self.run_script(memory=memory,tasks=tasks)
            self.assertEqual(result.returncode,2)
            self.assertEqual(calls,[])


if __name__=='__main__':
    unittest.main()
