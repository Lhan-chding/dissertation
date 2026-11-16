"""Offline MAIN submission boundary tests. No real terminal state or scheduler is used."""
import ast
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

OPERATOR=Path(__file__).resolve().parents[1]/'satdose_submit_main.py'
M=runpy.run_path(str(OPERATOR));P=M['load_prepare'](OPERATOR.with_name('satdose_prepare_main.py'))
TREE=ast.parse(M['REMOTE'])
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalSubmission(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='satdose-main-submit-OFFLINE-');self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name).resolve();self.input=self.directory/'FAKE_preparation.json';self.output=self.directory/'submission.json'
        self.fixture(11031,'START',0)
    def fixture(self,start,arm,step):
        endpoint,count=P.selection(start,arm,step);directory=P.preparation_dir(endpoint)
        files={name:dict(path=str(directory/name),sha256='0'*64,bytes=1) for name in P.FILES}
        files['PREPARER_SOURCE.py']['sha256']=M['PREPARER_SHA']
        report=dict(status='MAIN_CPU_PREFLIGHT_AND_STORAGE_COMPLETE',permission='NO_MODEL_CALLS_OR_GPU_SUBMISSION',
          endpoint=endpoint,responses=count,model_calls=0,optimizer_calls=0,gpu_submissions=0,
          formal_optimizer_steps=96,train_responses=12288,dev_endpoints=14,dev_responses=1512,
          prior_endpoints=[entry[:3] for entry in P.ORDER].index((start,arm,step)),
          release=dict(path=str((directory/'RELEASE.json').relative_to(P.ROOT)),sha256='0'*64))
        self.response=dict(result=report,manifest=dict(status='MAIN_PREPARATION_COMPLETE',endpoint=endpoint,files=files))
        out=self.directory/'FAKE_preparation.stdout';err=self.directory/'FAKE_preparation.stderr'
        out.write_bytes(b'SATDOSE_MAIN_OPERATOR='+json.dumps(self.response).encode()+b'\n');err.write_bytes(b'')
        self.receipt=dict(status='MAIN_OPERATOR_RESPONSE_CAPTURED',returncode=0,response=self.response,
          source=P.localref(OPERATOR.with_name('satdose_prepare_main.py')),raw_logs=dict(stdout=P.localref(out),stderr=P.localref(err)))
        self.save()
    def save(self):self.input.write_text(json.dumps(self.receipt))
    def sync_response(self):
        out=Path(self.receipt['raw_logs']['stdout']['path']);out.write_bytes(b'SATDOSE_MAIN_OPERATOR='+json.dumps(self.response).encode()+b'\n')
        self.receipt['raw_logs']['stdout']=P.localref(out);self.save()
    def argv(self):return ['--preparation-receipt',str(self.input),'--preparation-sha256',digest(self.input),'--receipt',str(self.output)]
    def load(self):return M['local_preparation'](P,self.input,digest(self.input))
    def test_syntax_and_prepare_source_pin(self):
        compile(OPERATOR.read_text(),str(OPERATOR),'exec');compile(M['REMOTE'],'remote','exec')
        self.assertEqual(digest(OPERATOR.with_name('satdose_prepare_main.py')),M['PREPARER_SHA'])
    def test_all_fourteen_plan_locally_and_fixed_walltimes(self):
        for start,arm,step,count in P.ORDER:
            self.fixture(start,arm,step)
            with patch.object(subprocess,'run',side_effect=AssertionError('Network in plan')),redirect_stdout(io.StringIO()) as output:
                self.assertEqual(M['main'](self.argv()),0)
            plan=json.loads(output.getvalue());self.assertEqual(plan['responses'],count)
            self.assertEqual(plan['walltime'],'24:00:00' if count==1952 else '06:00:00')
            self.assertFalse(self.output.exists())
    def test_failed_preparation_not_accepted(self):
        self.receipt['returncode']=2;self.save()
        with self.assertRaises(PermissionError):self.load()
    def test_partial_preparation_manifest_rejected(self):
        del self.response['manifest']['files']['PREPARATION_RECEIPT.json'];self.sync_response()
        with self.assertRaises(PermissionError):self.load()
    def test_wrong_terminal_count_rejected(self):
        self.response['result']['formal_optimizer_steps']=95;self.sync_response()
        with self.assertRaises(PermissionError):self.load()
    def test_wrong_prefix_ordinal_rejected(self):
        self.response['result']['prior_endpoints']=1;self.sync_response()
        with self.assertRaises(PermissionError):self.load()
    def test_extra_draw_budget_rejected(self):
        self.response['result']['responses']=1953;self.sync_response()
        with self.assertRaises(PermissionError):self.load()
    def test_changed_prepare_dependency_rejected_before_execution(self):
        fake=self.directory/'fake_prepare.py';fake.write_text("raise RuntimeError('must not execute')")
        with self.assertRaisesRegex(PermissionError,'source changed'):M['load_prepare'](fake)
    def test_raw_output_mutation_rejected(self):
        Path(self.receipt['raw_logs']['stdout']['path']).write_bytes(b'changed')
        with self.assertRaises(PermissionError):self.load()
    def test_network_unknown_source_preserved_no_retry(self):
        called=[]
        def fake(argv,**kw):
            called.append(argv);self.assertEqual(argv[0],'ssh')
            self.assertEqual(self.output.with_name(self.output.name+'.source.py').read_bytes(),OPERATOR.read_bytes())
            kw['stdout'].write(b'OFFLINE SSH UNKNOWN\xff\n');kw['stderr'].write(b'OFFLINE STDERR\n');return SimpleNamespace(returncode=255)
        with patch.object(subprocess,'run',side_effect=fake),redirect_stdout(io.StringIO()):
            self.assertEqual(M['main'](self.argv()+['--execute']),2)
            with self.assertRaises(PermissionError):M['main'](self.argv()+['--execute'])
        self.assertEqual(len(called),1);receipt=json.loads(self.output.read_text())
        self.assertEqual(receipt['status'],'MAIN_OPERATOR_UNKNOWN_NO_RETRY')
        self.assertEqual(Path(receipt['raw_logs']['stdout']['path']).read_bytes(),b'OFFLINE SSH UNKNOWN\xff\n')


class HeldSubmissionTail(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='satdose-main-tail-OFFLINE-');self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name).resolve();self.code=P.EXTERNAL;self.job=9999999;self.calls=[];self.unlocked=False;self.fault=None;self.shows=0
        self.name='LH-SATDOSE-MAIN-11031-START-s00';self.endpoint=dict(start_seed=11031,arm='START',step=0)
        self.g=dict(os=os,json=json,hashlib=hashlib,re=re,shlex=shlex,Path=Path,ROOT=P.ROOT,CODE=self.code,
          QOS='soujanya-poria-startfund-2026-03',directory=self.directory,observations=[],locks=[],job=None,accepted=None,
          utc=lambda:'OFFLINE_TIME',p=dict(endpoint=self.endpoint,responses=1952,manifest={},source_sha256=digest(OPERATOR)),
          endpoint=self.endpoint,start=11031,arm='START',step=0,name=self.name,
          release_ref=dict(path='OFFLINE_RELEASE.json',sha256='0'*64),report=dict(terminal_publication={}),scheduler={},storage={},
          prepared_inputs=dict(terminal_manifest={}),c=SimpleNamespace(bound=lambda r:P.ROOT/r['path'],_hold_free=self.hold_free),
          P=SimpleNamespace(TERMINAL=P.TERMINAL,validate_terminal=self.terminal))
        definitions=[node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name in ('require','sha','ref','syncdir','write','verify_own_job')]
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'actual-helpers','exec'),self.g)
        self.g.update(run=self.fake,stable_locks=lambda:None,unlock=self.unlock)
        top=next(node for node in TREE.body if isinstance(node,ast.Try))
        index=next(i for i,node in enumerate(top.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='walltime' for t in node.targets))
        isolated=copy.deepcopy(top);isolated.body=isolated.body[index:]
        self.tail=compile(ast.Module(body=[isolated],type_ignores=[]),'actual-submit-tail','exec')
    def terminal(self,*args,**kwargs):
        if self.fault=='terminal_error_before_release':raise RuntimeError('OFFLINE terminal ERROR appeared')
    def hold_free(self):
        if self.fault=='new_hold_before_release':raise RuntimeError('OFFLINE global HOLD appeared')
    def unlock(self):
        if (self.directory/'SUBMISSION.json').exists():self.assertEqual(json.loads((self.directory/'SUBMISSION.json').read_text())['job_id'],self.job)
        self.unlocked=True
    def fields(self):
        fields=dict(JobId=str(self.job),JobName=self.name,Account='rose',QOS=self.g['QOS'],Partition='cluster02',NumCPUs='8',NumTasks='1',
          TimeLimit='1-00:00:00' if self.g['p']['responses']==1952 else '06:00:00',Requeue='0',
          Command=str(self.code/'scheduling/run_main.sh'),WorkDir=str(self.code),StdOut=str(self.directory/f'slurm-{self.job}.out'),
          StdErr=str(self.directory/f'slurm-{self.job}.err'),NumNodes='1-1',UserId='varun024(295800338)',
          ReqTRES='cpu=8,mem=66G,node=1,billing=16,gres/gpu=2,gres/gpu:pro6000=2',JobState='PENDING',Reason='JobHeldUser',Priority='0',ExitCode='0:0')
        if self.shows>=3:fields.update(JobState='RUNNING',Reason='None',Priority='1')
        if self.fault=='wrong_qos':fields['QOS']='rose'
        if self.fault=='wrong_walltime':fields['TimeLimit']='03:00:00'
        if self.fault=='foreign_second_identity' and self.shows==2:fields['JobName']='OTHER_JOB'
        if self.fault=='release_visibility_delay' and self.shows==3:fields.update(JobState='PENDING',Reason='JobHeldUser',Priority='0')
        return fields
    def fake(self,args):
        self.calls.append(args);value=dict(argv=args,returncode=0,stdout='',stderr='',observed_at_utc='OFFLINE')
        if args[0]=='sbatch':
            self.assertTrue((self.directory/'INTENT.json').is_file());self.assertFalse(self.unlocked)
            value['stdout']=str(self.job)+'\n'
            if self.fault=='sbatch_failure':value.update(returncode=1,stdout='',stderr='OFFLINE failure')
            if self.fault=='foreign_cluster':value['stdout']=str(self.job)+';foreign\n'
        elif args[:3]==['scontrol','show','job']:
            self.shows+=1;value['stdout']=' '.join(k+'='+v for k,v in self.fields().items())
        elif args[:2]==['scontrol','release']:
            self.assertTrue(self.unlocked);self.assertEqual(args[2],str(self.job));self.assertTrue((self.directory/'RELEASE_INTENT.json').is_file())
            if self.fault=='release_failure':value.update(returncode=1,stderr='OFFLINE release failure')
        elif args[0]=='sacct':value['stdout']=f'{self.job}|RUNNING|0:0|{self.name}|rose|{self.g["QOS"]}\n'
        else:raise AssertionError('Unexpected command')
        self.g['observations'].append(value);return value
    def execute(self,fault=None):
        self.fault=fault;exec(self.tail,self.g)
        self.assertEqual(sum(args[0]=='sbatch' for args in self.calls),1)
        self.assertLessEqual(sum(args[:2]==['scontrol','release'] for args in self.calls),1)
        return self.g['output']
    def test_full_endpoint_one_submit_correct_five_args_and_handoff(self):
        value=self.execute();self.assertEqual(value['result']['status'],'MAIN_SUBMITTED_RELEASE_CONFIRMED')
        args=next(a for a in self.calls if a[0]=='sbatch');self.assertIn('--time=24:00:00',args)
        self.assertIn('--gpus=pro6000:2',args);self.assertIn('--no-requeue',args)
        self.assertEqual(len(args[args.index(str(self.code/'scheduling/run_main.sh'))+1:]),5)
        self.assertEqual(args[-3:],['11031','START','0'])
    def test_primary_endpoint_six_hour_limit(self):
        self.g['p']['responses']=256;value=self.execute();self.assertIn('result',value)
        self.assertIn('--time=06:00:00',next(a for a in self.calls if a[0]=='sbatch'))
    def test_sbatch_failure_no_release(self):
        self.execute('sbatch_failure');self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_foreign_cluster_no_release(self):
        self.execute('foreign_cluster');self.assertFalse((self.directory/'SUBMISSION.json').exists())
    def test_wrong_qos_no_release(self):
        self.execute('wrong_qos');self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_wrong_walltime_no_release(self):
        self.execute('wrong_walltime');self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_changed_job_identity_no_release(self):
        value=self.execute('foreign_second_identity');self.assertEqual(value['job_id'],self.job)
        self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_terminal_error_after_unlock_keeps_job_held(self):
        value=self.execute('terminal_error_before_release');self.assertEqual(value['job_id'],self.job)
        self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_global_hold_after_unlock_keeps_job_held(self):
        value=self.execute('new_hold_before_release');self.assertEqual(value['job_id'],self.job)
        self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_release_failure_keeps_known_job_no_retry(self):
        value=self.execute('release_failure');self.assertEqual(value['job_id'],self.job);self.assertFalse((self.directory/'RELEASE_RESULT.json').exists())
    def test_release_visibility_delay_preserves_unknown(self):
        value=self.execute('release_visibility_delay');self.assertEqual(value['job_id'],self.job);self.assertNotIn('result',value)
    def test_scheduler_environment_isolation(self):
        function=next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name=='run');seen=[]
        def fake(args,**kw):seen.append(kw['env']);return SimpleNamespace(returncode=0,stdout='',stderr='')
        g=dict(os=SimpleNamespace(environ={'SLURM_CLUSTERS':'foreign','SBATCH_ARRAY_INX':'1-9','CUDA_VISIBLE_DEVICES':'','PATH':'/bin'}),
          subprocess=SimpleNamespace(run=fake),observations=[],utc=lambda:'OFFLINE')
        exec(compile(ast.Module(body=[function],type_ignores=[]),'actual-run','exec'),g)
        g['run'](['sbatch']);self.assertEqual(seen[-1],{'PATH':'/bin'});g['run'](['scontrol']);self.assertNotIn('SLURM_CLUSTERS',seen[-1])


if __name__=='__main__':unittest.main(verbosity=2)
