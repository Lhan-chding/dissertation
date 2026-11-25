"""Offline transport regression tests. No SSH, cluster, GPU or scheduler calls."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'satdose_trained_dev_detached_transport.py'
spec = importlib.util.spec_from_file_location('transport_test_target', SOURCE)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
BASELINE = t.ARTIFACTS / 'trained_dev_20261124'
PREP_ARGS = ['--start-seed','11031','--arm','T1','--step','16','--publication-tag',t.TAG,
             '--original-train-release',t.TRAIN['path'],'--original-train-sha256',t.TRAIN['sha256'],
             '--canary-spec',t.CANARY['path'],'--canary-spec-sha256',t.CANARY['sha256'],'--execute-preparation']


def remote(op):
    native = SOURCE.parent/f'satdose_{op}_trained_dev.py'
    assert t.digest(native.read_bytes()) == t.SOURCE[op]
    tree = ast.parse(native.read_bytes())
    return next(ast.literal_eval(x.value) for x in tree.body if isinstance(x, ast.Assign)
                and any(isinstance(k, ast.Name) and k.id == 'REMOTE' for k in x.targets))


def prepare_input():
    p = json.loads((BASELINE/'start-11031_T1_step-016/PREPARATION.json.intent.json').read_bytes())['request']
    p['tag'] = t.TAG
    return p


def response(rc, value=None, error=b''):
    return subprocess.CompletedProcess([], rc, t.encoded(value) if value is not None else b'', error)


def finished(claim, stdout=b'test\r\n', stderr=b'\x00\xff', rc=17):
    wait = dict(key=t.digest(t.encoded(claim)), cpu_job_id='3001',returncode=rc, stdout_sha256=t.digest(stdout), stderr_sha256=t.digest(stderr))
    return dict(state='FINISHED', wait=wait, cpu=dict(job_id='3001'),stdout=t.pack(stdout), stderr=t.pack(stderr))


class DetachedTransport(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name).resolve()
        self.argv = t.NATIVE_PREFIX + [remote('prepare')]
        self.raw = t.encoded(prepare_input())
        self.source = SOURCE.read_bytes()
        self.claim = t.validate('prepare', self.argv, self.raw, t.digest(self.source))

    def test_frozen_command_and_input_rules(self):
        for mutation in ('wrong_tag','foreign_endpoint','different_sources','changed_program','env_option'):
            obj = prepare_input()
            args = self.argv.copy()
            if mutation == 'wrong_tag':
                obj['tag'] += '_another'
            elif mutation == 'foreign_endpoint':
                obj['start_seed'] = 11032
            elif mutation == 'different_sources':
                obj['sources'].clear()
            elif mutation == 'changed_program':
                args[-1] += '\n# change'
            else:
                args[1] = 'CUDA_VISIBLE_DEVICES=0'
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                t.validate('prepare', args, t.encoded(obj), t.digest(self.source))

    def test_only_one_mutating_exchange_after_lost_ack_and_subsequent_disconnect(self):
        record = []
        def exchange(action, op, claim, payload, source):
            record.append(action)
            if len(record) in (1, 3):
                return response(255, error=b'network unavailable')
            if len(record) == 2:
                return response(0, dict(state='PENDING'))
            return response(0, finished(claim))
        out, err = io.BytesIO(), io.BytesIO()
        with patch.object(t,'exchange',exchange), patch.object(t.time,'sleep'), contextlib.redirect_stderr(io.StringIO()):
            receipt = self.dir/'NATIVE'
            res = t.relay('prepare',self.argv,self.raw,out,err,receipt,False,self.source)
        self.assertEqual(record,['launch','observe','observe','observe'])
        self.assertEqual((res.returncode,out.getvalue(),err.getvalue()),(17,b'test\r\n',b'\x00\xff'))
        self.assertEqual(json.loads((self.dir/'NATIVE.transport.launch.json').read_bytes())['ssh_returncode'],255)
        self.assertEqual(json.loads((self.dir/'NATIVE.transport.wait.json').read_bytes())['provenance'],'REMOTE_NATIVE_CHILD_WAIT')

    def test_collect_never_launches_and_rejects_changed_result(self):
        record = []
        def exchange(action, op, claim, payload, source):
            record.append((action,payload))
            answer = finished(claim)
            answer['stdout'] = t.pack(b'altered')
            return response(0,answer)
        out, err = io.BytesIO(), io.BytesIO()
        with patch.object(t,'exchange',exchange), self.assertRaisesRegex(RuntimeError,'wait/bytes'):
            t.relay('prepare',self.argv,self.raw,out,err,self.dir/'NATIVE',True,self.source)
        self.assertEqual(record,[('observe',{})])
        self.assertEqual((out.getvalue(),err.getvalue()),(b'',b''))

    def test_absent_after_lost_launch_requires_resolution(self):
        record = []
        def exchange(action, *args):
            record.append(action)
            return response(255) if action == 'launch' else response(0,dict(state='ABSENT'))
        with patch.object(t,'exchange',exchange), patch.object(t.time,'sleep'), \
             patch.object(t.time,'monotonic',side_effect=[0,0,130]), contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError,'No worker acknowledgement'):
            t.relay('prepare',self.argv,self.raw,io.BytesIO(),io.BytesIO(),self.dir/'N',False,self.source)
        self.assertEqual(record,['launch','observe','observe'])

    def test_live_native_prepare_preserves_original_source_and_observed_output(self):
        dest = self.dir/'PREPARATION.json'
        def exchange(action, op, claim, payload, source):
            self.assertEqual(action,'launch')
            self.assertEqual(claim['native_source_sha256'],t.SOURCE['prepare'])
            self.assertEqual(json.loads(t.unpack(payload['stdin'])),prepare_input())
            return response(0,finished(claim, stdout=b'', stderr=b'fixture native error\r\n',rc=13))
        with patch.object(t,'ARTIFACTS',self.dir), patch.object(t,'exchange',exchange), contextlib.redirect_stdout(io.StringIO()):
            rc = t.native('prepare',PREP_ARGS+['--receipt',str(dest)],False)
        receipt=json.loads(dest.read_bytes())
        self.assertNotEqual(rc,0)
        self.assertEqual(receipt['returncode'],13)
        self.assertEqual(Path(receipt['source']['path']).read_bytes(), (SOURCE.parent/'satdose_prepare_trained_dev.py').read_bytes())
        self.assertEqual(Path(receipt['stderr_file']['path']).read_bytes(),b'fixture native error\r\n')

    def test_native_submit_binds_successful_preparer_then_replays_real_child_exit(self):
        # Only files within temp are fixtures. The original submitter checks its
        # own preparation lineage, source84 and execution19 from the real intake.
        previous = json.loads((BASELINE/'start-11031_T0_step-016/PREPARATION.json').read_bytes())['response']
        old = '_71018e5_20261116_start-11031_T0_step-016'
        new = '_'+t.TAG+'_start-11031_T1_step-016'
        synthetic = json.loads(json.dumps(previous).replace(old,new))
        synthetic['result']['arm']='T1'
        output = b'SATDOSE_PREPARE_JSON='+t.encoded(synthetic)+b'\n'
        prepared=self.dir/'P.json'; submitted=self.dir/'S.json'
        def exchange(action,op,claim,payload,source):
            self.assertEqual(action, 'observe' if op=='submit' else 'launch')
            if op=='prepare':
                return response(0,finished(claim,stdout=output,stderr=b'',rc=0))
            self.assertEqual(claim['remote_source_sha256'],t.digest(remote('submit').encode()))
            return response(0,finished(claim,stdout=b'',stderr=b'fixture native submit failure',rc=23))
        with patch.object(t,'ARTIFACTS',self.dir), patch.object(t,'exchange',exchange), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(t.native('prepare',PREP_ARGS+['--receipt',str(prepared)],False),0)
            native_args=['--preparation-receipt',str(prepared),'--preparation-sha256',t.digest(prepared.read_bytes()),
                         '--receipt',str(submitted),'--execute']
            self.assertNotEqual(t.native('submit',native_args,True),0)
        receipt=json.loads(submitted.read_bytes())
        self.assertEqual(receipt['returncode'],23)
        self.assertEqual(receipt['raw_logs']['stderr']['sha256'],t.digest(b'fixture native submit failure'))
        self.assertEqual(receipt['source']['sha256'],t.SOURCE['submit'])

    def test_claim_rejects_foreign_input_and_symlink(self):
        with patch.object(t,'BASE',self.dir):
            d=self.dir/'prepare'; d.mkdir(mode=0o700)
            t.save(d/'CLAIM.json',t.encoded(dict(self.claim,stdin_sha256='0'*64)))
            with self.assertRaisesRegex(RuntimeError,'Different attempt'):
                t.observe(d,self.claim)
            d2=self.dir/'submit';d2.symlink_to(d)
            with self.assertRaisesRegex(RuntimeError,'Noncanonical'):
                t.observe(d2,self.claim)

    def set_up_scheduler(self,op='prepare',stdin=None):
        self.project=self.dir/'project'
        self.project.mkdir()
        base=self.project/'transport'
        self.d=base/op
        self.op=op
        if stdin is not None:
            self.raw=stdin
            self.argv=t.NATIVE_PREFIX+[remote(op)]
        self.claim=t.validate(op,self.argv,self.raw,t.digest(self.source))
        self.key=t.digest(t.encoded(self.claim))
        self.jid='3001'
        self.state='PENDING'
        self.exit='0:0'
        self.visible=True
        self.batch_visible=True
        self.other_ids=[]
        self.submitted=[]
        self.command_sha=None
        self.lose_ack=False
        self.tres='billing=2,cpu=2,mem=6144M,node=1'
        self.wrong_command=False
        self.enter_patch(patch.object(t,'PROJECT',self.project))
        self.enter_patch(patch.object(t,'BASE',base))
        self.enter_patch(patch.object(t,'scheduler',side_effect=self.fake_scheduler))
        self.enter_patch(patch.object(t,'identify',side_effect=lambda pid:dict(pid=pid,hostname=socket.gethostname())))

    def enter_patch(self,patcher):
        obj=patcher.start()
        self.addCleanup(patcher.stop)
        return obj

    def fake_scheduler(self,argv):
        name=t.job_name(self.op,self.key)
        cmd=argv[0]
        if cmd=='sbatch':
            self.submitted.append(argv)
            if self.lose_ack:
                raise OSError('simulated disconnect after submission and before acknowledgement')
            return subprocess.CompletedProcess(argv,0,(self.jid+'\n').encode(),b'')
        if cmd=='squeue' or '-X' in argv:
            ids=[self.jid,*self.other_ids] if self.visible else []
            raw=''.join(f'{j}|{name}\n' for j in ids)
        elif cmd=='scontrol':
            expected=dict(JobId=self.jid,JobName=name,Account='rose',QOS=t.QOS,Partition='cluster02',
                          NumCPUs='2',NumTasks='1',NumNodes='1',TimeLimit='04:00:00',Requeue='0',
                          Command=str(self.d/('WRONG' if self.wrong_command else 'CPU_WRAPPER.sh')),WorkDir=str(self.d),
                          StdOut=str(self.d/'cpu-3001.out'),StdErr=str(self.d/'cpu-3001.err'),
                          UserId='varun024(1)',ReqTRES=self.tres,AllocTRES=self.tres,JobState=self.state)
            raw=' '.join(k+'='+v for k,v in expected.items())+'\n'
        else:
            self.assertEqual(cmd,'sacct')
            if not self.visible:
                raw=''
            else:
                raw='|'.join([self.jid,self.state,self.exit,name,t.USER,'rose',t.QOS,'cluster02',self.tres,
                              self.tres if self.state!='PENDING' else '', '240',str(self.d)])+'\n'
                if self.state!='PENDING' and self.batch_visible:
                    raw+='|'.join([self.jid+'.batch',self.state,self.exit,'batch','','','','','','','',''])+'\n'
        return subprocess.CompletedProcess(argv,0,raw.encode(),b'')

    def launch(self):
        return t.serve('launch',self.op,self.claim,dict(argv=self.argv,stdin=t.pack(self.raw)),self.source)

    def observe(self):
        return t.serve('observe',self.op,self.claim,{},self.source)

    def run_fixture_child(self,stdout=None,rc=0):
        actual=t.run_child
        script=("import os,sys; data=sys.stdin.buffer.read(); assert not any(k.startswith(('SLURM_','SBATCH_','SRUN_')) for k in os.environ); "
                +"sys.stdout.buffer.write("+("data" if stdout is None else repr(stdout))+"); "
                +"sys.stderr.buffer.write(b'unclean\\x00\\xff'); sys.exit("+str(rc)+")")
        def runner(directory,argv,stdin):
            self.assertEqual(argv,self.argv)
            return actual(directory,[sys.executable,'-I','-B','-c',script],stdin)
        self.state='RUNNING'
        with patch.dict(os.environ,SLURM_JOB_ID=self.jid,SBATCH_MEM='bad',SRUN_CPUS_PER_TASK='bad'),patch.object(t,'run_child',runner):
            return t.worker(self.op,self.key)

    def test_lost_sbatch_ack_never_resubmits_and_preserves_true_nonzero_exit(self):
        self.set_up_scheduler()
        self.lose_ack=True
        with self.assertRaises(OSError):
            self.launch()
        self.assertFalse((self.d/'SBATCH_RESPONSE.json').exists())
        self.assertEqual(self.observe()['cpu']['job_id'],self.jid)
        with self.assertRaises(FileExistsError):
            self.launch()
        self.assertEqual(len(self.submitted),1)
        args=self.submitted[0]
        for v in ('--cpus-per-task=2','--mem=6G','--time=04:00:00','--no-requeue','--qos='+t.QOS):
            self.assertIn(v,args)
        self.assertFalse(any(a.startswith(('--gpu','--gres','--gpus')) for a in args))
        self.assertEqual(self.run_fixture_child(rc=23),23)
        self.assertEqual(self.observe()['state'],'PENDING')
        self.state='FAILED';self.exit='23:0';self.batch_visible=False
        self.assertEqual(self.observe()['slurm_state'],'WAIT_FOR_BATCH_ACCOUNTING')
        self.batch_visible=True
        result=self.observe()
        self.assertEqual(result['wait']['returncode'],23)
        self.assertEqual(t.unpack(result['stdout']),self.raw)
        self.assertEqual(t.unpack(result['stderr']),b'unclean\x00\xff')
        self.assertEqual(result['cpu']['accounting'][self.jid],['FAILED','23:0'])
        self.assertIsNone(result['cpu']['sbatch_response'])
        self.assertEqual(self.observe(),result)

    def test_gpu_allocation_or_wrong_command_rejected_before_native_operation(self):
        self.set_up_scheduler();self.launch();self.state='RUNNING'
        with patch.dict(os.environ,SLURM_JOB_ID=self.jid),patch.object(t,'run_child') as child:
            self.tres+=',gres/gpu=1'
            with self.assertRaisesRegex(RuntimeError,'zero GPUs'):
                t.worker('prepare',self.key)
            self.tres='cpu=2,mem=6G,node=1'
            self.wrong_command=True
            with self.assertRaisesRegex(RuntimeError,'identity differs'):
                t.worker('prepare',self.key)
            self.assertFalse(child.called)
        self.assertFalse((self.d/'WORKER.json').exists())

    def test_cancelled_timeout_oom_and_missing_native_wait_remain_unknown(self):
        self.set_up_scheduler();self.launch()
        for self.state in ('CANCELLED by 1','TIMEOUT','OUT_OF_MEMORY','FAILED','COMPLETED'):
            with self.subTest(state=self.state):
                result=self.observe()
                self.assertEqual(result['state'],'CPU_WRAPPER_TERMINATED_WITHOUT_CONFIRMED_NATIVE_WAIT')
                self.assertNotIn('wait',result)
        self.assertEqual(len(self.submitted),1)

    def test_accounting_and_wait_disagreement_never_projects_success(self):
        self.set_up_scheduler();self.launch()
        self.assertEqual(self.run_fixture_child(rc=4),4)
        self.state='COMPLETED'
        with self.assertRaisesRegex(RuntimeError,'disagree'):
            self.observe()
        self.state='FAILED';self.exit='4:0'
        self.assertEqual(self.observe()['wait']['returncode'],4)

    def test_duplicate_or_requeued_wrapper_is_not_run(self):
        self.set_up_scheduler();self.launch()
        self.other_ids=['3002']
        with self.assertRaisesRegex(RuntimeError,'Conflicting CPU wrapper'):
            self.observe()
        self.state='RUNNING'
        with patch.dict(os.environ,SLURM_JOB_ID=self.jid,SLURM_RESTART_COUNT='1'),self.assertRaisesRegex(RuntimeError,'first Slurm'):
            t.worker('prepare',self.key)
        with patch.dict(os.environ,SLURM_JOB_ID=self.jid),self.assertRaisesRegex(RuntimeError,'Another CPU job'):
            t.worker('prepare',self.key)
        self.assertFalse((self.d/'WORKER.json').exists())

    def test_real_native_gpu_and_cpu_job_ids_stay_separate(self):
        p=json.loads((BASELINE/'start-11031_T0_step-016/SUBMISSION.json.intent.json').read_bytes())['payload']
        p.update(start_seed=11031,arm='T1',step=16)
        p['preparation'].update(start_seed=11031,arm='T1',step=16)
        p['preparation_ref']['path']=t.RESULT
        p['preparation']['release']['path']=t.RESULT.replace('_PREPARATION_RESULT.json','_RELEASE.json')
        self.set_up_scheduler('submit',t.encoded(p));self.launch()
        submitted=b'SATDOSE_SUBMIT_JSON={"result":{"job_id":4555}}\n'
        self.assertEqual(self.run_fixture_child(submitted),0)
        self.state='COMPLETED'
        out=self.observe()
        self.assertEqual(out['cpu']['job_id'],'3001')
        self.assertEqual(out['cpu']['native_submitted_gpu_job_id'],4555)
        self.assertEqual(t.unpack(out['stdout']),submitted)
        self.assertEqual(len(self.submitted),1)


if __name__ == '__main__':
    unittest.main()
