"""Exercise fixed diagnostic dispatch only; never load models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class FullBatchProbeLauncherTests(unittest.TestCase):
    def run_launcher(self, memory='67584', tasks='2', fail=''):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=(Path(__file__).parents[1]/'diagnostics/run_full_batch_memory_probe.sh').read_text()
            source=source.replace('root=/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2','root='+str(root))
            (root/'run.sh').write_text(source)
            stub=root/'srun'
            stub.write_text('#!'+sys.executable+'\n'+'''
import json,os,pathlib,sys,time
args=sys.argv[1:]
assert 'SLURM_TRES_PER_TASK' not in os.environ
assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ
assert os.environ['PYTORCH_ALLOC_CONF']=='expandable_segments:True'
path=args[args.index('--path')+1]
with pathlib.Path(path+'.calls').open('a') as f:f.write(json.dumps(args)+'\\n')
time.sleep(.02)
pathlib.Path(path+'.done').touch()
sys.exit(1 if path==os.environ.get('FAIL_PATH') else 0)
''')
            stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],SLURM_JOB_ID='123',
                     SLURM_MEM_PER_NODE=memory,SLURM_CPUS_ON_NODE='8',SLURM_NTASKS=tasks,
                     SLURM_TRES_PER_TASK='cpu:4',PYTORCH_CUDA_ALLOC_CONF='old',FAIL_PATH=fail)
            result=subprocess.run(['bash',str(root/'run.sh')],env=env,capture_output=True,text=True)
            calls={p.stem:[json.loads(l) for l in p.read_text().splitlines()] for p in root.glob('*.calls')}
            done={p.stem for p in root.glob('*.done')}
            return result,calls,done

    def test_both_lanes_finish_each_path_once_after_failure(self):
        result,calls,done=self.run_launcher(fail='b0-11031-R')
        self.assertEqual(result.returncode,1,result.stderr)
        expected={'b0-11031-R','b0-11031-G','b1-11031-R','b1-11031-G'}
        self.assertEqual(set(calls),expected)
        self.assertEqual(done,expected)
        for path,rows in calls.items():
            self.assertEqual(len(rows),1)
            args=rows[0]
            for item in ('diagnostics.full_batch_memory_probe','--gpus-per-task=1','--mem=33792M','--cpus-per-task=4'):
                self.assertIn(item,args)
            self.assertEqual(args[args.index('--worker-memory-mib')+1],'33792')
            self.assertEqual(args[args.index('--output')+1],'run/fullbatch_memory_diag_123_'+path)

    def test_insufficient_or_unknown_resources_fail_before_dispatch(self):
        for memory,tasks in [('33792','2'),('67584','1'),('','2')]:
            result,calls,done=self.run_launcher(memory=memory,tasks=tasks)
            self.assertEqual(result.returncode,2)
            self.assertFalse(calls or done)


if __name__=='__main__':
    unittest.main()
