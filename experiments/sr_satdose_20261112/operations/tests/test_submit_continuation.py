"""Offline-only operator boundary tests. No SSH, scheduler or project preflight.

Fixtures deliberately fabricate local evidence to test parsing and failure
retention. The remote submission tail is executed with a scheduler stub; original
source/admission/checkpoint gates are NOT simulated as production verification.
"""
import ast
from contextlib import redirect_stdout
import copy
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

OPERATOR=Path(__file__).resolve().parents[1]/'satdose_submit_continuation.py'
MODULE=runpy.run_path(str(OPERATOR))
REMOTE_TREE=ast.parse(MODULE['REMOTE'])
PREPARER=OPERATOR.with_name('satdose_prepare_continuation.py')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def localref(path):return dict(path=str(path),sha256=digest(path))
def rootref(name):return dict(path='execution_releases/OFFLINE_FAKE_'+name+'.json',sha256='0'*64)


class LocalDriver(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='satdose-continuation-OFFLINE-')
        self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name).resolve()
        self.input=self.directory/'FAKE_preparation.json';self.output=self.directory/'submission.json'
        self.make_fixture(3)
    def make_fixture(self,first):
        last={3:8,9:16,17:24}[first]
        selection=dict(start_seed=11031,arm='T0',first_step=first,last_step=last)
        report=dict(status='CONTINUATION_CPU_PREFLIGHT_COMPLETE_NO_MODEL_CALLS',permission='NO_GPU_SUBMISSION',
          preparation_script_sha256=MODULE['PREPARER_SHA'],model_calls=0,optimizer_calls=0,gpu_jobs_submitted=0,
          selection=selection,original_train_release=MODULE['ORIGINAL'],completed_dev_endpoints={3:6,9:10,17:14}[first],
          required_bytes=16<<30,**{key:rootref(key) for key in ('intent','release','storage_budget','risk')})
        publication=dict(report=report,preflight=rootref('preflight'),
          release_path=str(Path(MODULE['ROOT'])/report['release']['path']),release_sha256=report['release']['sha256'])
        out=self.directory/'FAKE_preparation.stdout';out.write_text(json.dumps(publication)+'\n')
        err=self.directory/'FAKE_preparation.stderr';err.write_bytes(b'')
        self.receipt=dict(status='CPU_PREPARATION_PUBLICATION_RECEIPT_CAPTURED',returncode=0,publication=publication,
          local_operation_source=localref(PREPARER)|dict(bytes=PREPARER.stat().st_size),
          raw_logs=dict(stdout=localref(out),stderr=localref(err)),stdout=out.read_text(),stderr='',
          summary=dict(mode='WRITE_ONCE_CPU_PREPARATION',selection=selection,original_train_release=MODULE['ORIGINAL'],
            gpu_submission=False,canary_spec=rootref('canary')))
        self.save()
    def save(self):self.input.write_text(json.dumps(self.receipt))
    def argv(self):return ['--preparation-receipt',str(self.input),'--preparation-sha256',digest(self.input),'--receipt',str(self.output)]
    def load(self):return MODULE['payload_from_receipt'](self.input,digest(self.input))
    def test_outer_and_remote_compile(self):
        compile(OPERATOR.read_text(),str(OPERATOR),'exec');compile(MODULE['REMOTE'],'remote','exec')
    def test_all_fixed_chunk_plans_are_local_only(self):
        for first in (3,9,17):
            self.make_fixture(first)
            with patch.object(subprocess,'run',side_effect=AssertionError('Subprocess in plan')),redirect_stdout(io.StringIO()) as out:
                self.assertEqual(MODULE['main'](self.argv()),0)
            plan=json.loads(out.getvalue());self.assertEqual(plan['first_step'],first)
            self.assertEqual(plan['last_step'],{3:8,9:16,17:24}[first]);self.assertEqual(plan['resources']['time'],'24:00:00')
            self.assertFalse(self.output.exists());self.assertFalse(self.output.with_name(self.output.name+'.intent.json').exists())
    def test_failed_preparer_rejected(self):
        self.receipt['returncode']=255;self.save()
        with self.assertRaisesRegex(ValueError,'successfully'):self.load()
    def test_wrong_preparer_source_rejected(self):
        self.receipt['local_operation_source']['sha256']='0'*64;self.save()
        with self.assertRaisesRegex(ValueError,'preparer source'):self.load()
    def test_mutated_raw_output_rejected(self):
        Path(self.receipt['raw_logs']['stdout']['path']).write_text('tampered')
        with self.assertRaises(ValueError):self.load()
    def test_invalid_chunk_rejected(self):
        self.receipt['publication']['report']['selection']['first_step']=4
        out=Path(self.receipt['raw_logs']['stdout']['path']);out.write_text(json.dumps(self.receipt['publication']))
        self.receipt['stdout']=out.read_text();self.receipt['raw_logs']['stdout']=localref(out);self.save()
        with self.assertRaisesRegex(ValueError,'Unknown fixed chunk'):self.load()
    def test_missing_four_path_dev_barrier_rejected(self):
        self.receipt['publication']['report']['completed_dev_endpoints']=5
        out=Path(self.receipt['raw_logs']['stdout']['path']);out.write_text(json.dumps(self.receipt['publication']))
        self.receipt['stdout']=out.read_text();self.receipt['raw_logs']['stdout']=localref(out);self.save()
        with self.assertRaisesRegex(ValueError,'four-path DEV'):self.load()
    def test_original_release_rejected(self):
        self.receipt['summary']['original_train_release']=rootref('foreign');self.save()
        with self.assertRaisesRegex(ValueError,'summary differs'):self.load()
    def test_network_failure_keeps_source_raw_logs_and_blocks_retry(self):
        calls=[]
        def fake(argv,**kwargs):
            calls.append(argv)
            self.assertEqual(argv[0],'ssh')
            self.assertEqual(self.output.with_name(self.output.name+'.source.py').read_bytes(),OPERATOR.read_bytes())
            self.assertTrue(self.output.with_name(self.output.name+'.intent.json').is_file())
            kwargs['stdout'].write(b'OFFLINE FAKE SSH FAILURE\xff\n');kwargs['stderr'].write(b'OFFLINE FAKE STDERR\n')
            return SimpleNamespace(returncode=255)
        with patch.object(subprocess,'run',side_effect=fake),redirect_stdout(io.StringIO()):
            self.assertEqual(MODULE['main'](self.argv()+['--execute']),2)
            with self.assertRaisesRegex(ValueError,'never retry'):MODULE['main'](self.argv()+['--execute'])
        self.assertEqual(len(calls),1);value=json.loads(self.output.read_text())
        self.assertEqual(value['status'],'SUBMISSION_UNKNOWN_NO_RETRY')
        self.assertEqual(Path(value['raw_logs']['stdout']['path']).read_bytes(),b'OFFLINE FAKE SSH FAILURE\xff\n')
        self.assertEqual(Path(value['raw_logs']['stderr']['path']).read_bytes(),b'OFFLINE FAKE STDERR\n')
        self.assertEqual(value['source']['sha256'],digest(OPERATOR))
    def test_cross_root_reference_rejected(self):
        for value in ('../escape','/absolute','execution_releases/../escape'):
            with self.assertRaises(ValueError):MODULE['rootref'](dict(path=value,sha256='0'*64))


class RemoteSubmissionTail(unittest.TestCase):
    """Execute the actual submit/release tail with only local files and fake Slurm."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='satdose-continuation-tail-OFFLINE-')
        self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name).resolve()
        self.code=Path(MODULE['CODE']);self.root=Path(MODULE['ROOT']);self.job=9999999
        self.name='LH-SATDOSE-TRAIN-11031-T0-003-008';self.calls=[];self.unlocked=False;self.fault=None;self.scontrol_count=0
        self.selection=dict(start_seed=11031,arm='T0',first_step=3,last_step=8)
        self.globals=dict(os=os,json=json,hashlib=hashlib,re=re,shlex=shlex,Path=Path,
          ROOT=self.root,CODE=self.code,QOS='soujanya-poria-startfund-2026-03',directory=self.directory,
          observations=[],locks=[],job=None,accepted=None,utc=lambda:'OFFLINE_TEST_TIME',
          p=dict(preparation_ref=rootref('preflight'),source_sha256=digest(OPERATOR)),
          start=11031,arm='T0',first=3,last=8,selection=self.selection,name=self.name,
          release_ref=rootref('release'),release=dict(canary_spec=rootref('canary')),scheduler={},storage={},
          c=SimpleNamespace(bound=lambda reference:self.root/reference['path']))
        keep={'require','sha','ref','syncdir','write','verify_own_job'}
        defs=[node for node in REMOTE_TREE.body if isinstance(node,ast.FunctionDef) and node.name in keep]
        exec(compile(ast.Module(body=defs,type_ignores=[]),'actual-remote-helpers','exec'),self.globals)
        self.globals.update(run=self.fake_run,stable_locks=lambda:None,unlock=self.unlock)
        top=next(node for node in REMOTE_TREE.body if isinstance(node,ast.Try))
        index=next(i for i,node in enumerate(top.body) if isinstance(node,ast.Assign) and
          any(isinstance(target,ast.Name) and target.id=='args' for target in node.targets))
        isolated=copy.deepcopy(top);isolated.body=isolated.body[index:]
        self.tail=compile(ast.Module(body=[isolated],type_ignores=[]),'actual-isolated-submission-tail','exec')
    def unlock(self):
        if (self.directory/'SUBMISSION.json').exists():
            accepted=json.loads((self.directory/'SUBMISSION.json').read_text())
            self.assertEqual(accepted['job_id'],self.job)
        self.unlocked=True
    def fields(self):
        value=dict(JobId=str(self.job),JobName=self.name,Account='rose',QOS=self.globals['QOS'],Partition='cluster02',
          NumCPUs='8',NumTasks='1',TimeLimit='1-00:00:00',Requeue='0',Command=str(self.code/'scheduling/run_continuation.sh'),
          WorkDir=str(self.code),StdOut=str(self.directory/f'slurm-{self.job}.out'),StdErr=str(self.directory/f'slurm-{self.job}.err'),
          NumNodes='1-1',UserId='varun024(295800338)',ReqTRES='cpu=8,mem=66G,node=1,billing=16,gres/gpu=2,gres/gpu:pro6000=2',
          JobState='PENDING',Reason='JobHeldUser',Priority='0',ExitCode='0:0')
        if self.scontrol_count>=3:value.update(JobState='RUNNING',Reason='None',Priority='1',NumNodes='1')
        if self.fault=='wrong_qos':value['QOS']='rose'
        if self.fault=='wrong_gpu':value['ReqTRES']=value['ReqTRES'].replace('pro6000=2','pro6000=1')
        if self.fault=='wrong_walltime':value['TimeLimit']='03:00:00'
        if self.fault=='foreign_second_identity' and self.scontrol_count==2:value['JobName']='OTHER_JOB'
        if self.fault=='release_visibility_delay' and self.scontrol_count==3:value.update(JobState='PENDING',Reason='JobHeldUser',Priority='0')
        if self.fault=='completed_failure' and self.scontrol_count==3:value.update(JobState='COMPLETED',ExitCode='1:0')
        return value
    def fake_run(self,argv):
        self.calls.append(argv);value=dict(argv=argv,returncode=0,stdout='',stderr='',observed_at_utc='OFFLINE')
        if argv[0]=='sbatch':
            self.assertTrue((self.directory/'INTENT.json').is_file());self.assertFalse(self.unlocked)
            value['stdout']=str(self.job)+'\n'
            if self.fault=='sbatch_failure':value.update(returncode=1,stdout='',stderr='OFFLINE simulated failure')
            if self.fault=='foreign_cluster':value['stdout']=str(self.job)+';foreign\n'
        elif argv[:3]==['scontrol','show','job']:
            self.scontrol_count+=1;value['stdout']=' '.join(key+'='+v for key,v in self.fields().items())
        elif argv[:2]==['scontrol','release']:
            self.assertTrue(self.unlocked);self.assertEqual(argv[2],str(self.job))
            self.assertTrue((self.directory/'SUBMISSION.json').is_file());self.assertTrue((self.directory/'RELEASE_INTENT.json').is_file())
            if self.fault=='release_failure':value.update(returncode=1,stderr='OFFLINE release error')
        elif argv[0]=='sacct':value['stdout']=f'{self.job}|RUNNING|0:0|{self.name}|rose|{self.globals["QOS"]}\n'
        else:raise AssertionError('Unexpected scheduler command '+repr(argv))
        self.globals['observations'].append(value);return value
    def execute(self,fault=None):
        self.fault=fault;exec(self.tail,self.globals)
        self.assertEqual(sum(args[0]=='sbatch' for args in self.calls),1)
        self.assertLessEqual(sum(args[:2]==['scontrol','release'] for args in self.calls),1)
        return self.globals['output']
    def test_success_fixed_resources_five_wrapper_args_and_lock_handoff(self):
        value=self.execute();self.assertEqual(value['result']['status'],'CONTINUATION_SUBMITTED_RELEASE_CONFIRMED')
        args=next(args for args in self.calls if args[0]=='sbatch')
        for flag in ('--hold','--qos=soujanya-poria-startfund-2026-03','--account=rose','--partition=cluster02',
          '--gpus=pro6000:2','--cpus-per-task=8','--mem=66G','--time=24:00:00','--no-requeue'):self.assertIn(flag,args)
        launcher=args.index(str(self.code/'scheduling/run_continuation.sh'))
        self.assertEqual(len(args[launcher+1:]),5);self.assertEqual(args[-3:],['11031','T0','3'])
        intent=json.loads((self.directory/'INTENT.json').read_text());self.assertEqual(intent['planned_train_responses'],768)
        self.assertEqual(intent['planned_formal_optimizer_steps'],6)
    def test_sbatch_failure_retained_no_release(self):
        value=self.execute('sbatch_failure');self.assertIn('UNKNOWN_NO_RETRY',value['status'])
        self.assertTrue((self.directory/'RESPONSE.json').is_file());self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_foreign_cluster_response_not_released(self):
        self.execute('foreign_cluster');self.assertFalse((self.directory/'SUBMISSION.json').exists())
    def test_wrong_qos_not_released(self):
        self.execute('wrong_qos');self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_wrong_gpu_not_released(self):
        self.execute('wrong_gpu');self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_wrong_walltime_not_released(self):
        self.execute('wrong_walltime');self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_second_identity_changed_no_release(self):
        value=self.execute('foreign_second_identity');self.assertEqual(value['job_id'],self.job)
        self.assertTrue((self.directory/'SUBMISSION.json').is_file());self.assertFalse((self.directory/'RELEASE_INTENT.json').exists())
    def test_release_failure_preserves_known_job_no_retry(self):
        value=self.execute('release_failure');self.assertEqual(value['job_id'],self.job)
        self.assertTrue((self.directory/'RELEASE_RESPONSE.json').is_file());self.assertFalse((self.directory/'RELEASE_RESULT.json').exists())
    def test_post_release_visibility_delay_preserves_known_job(self):
        value=self.execute('release_visibility_delay');self.assertEqual(value['job_id'],self.job)
        self.assertFalse((self.directory/'RELEASE_RESULT.json').exists());self.assertTrue((self.directory/'ERROR.json').is_file())
    def test_completed_nonzero_exit_not_reported_success(self):
        value=self.execute('completed_failure');self.assertNotIn('result',value)
    def test_real_run_environment_sanitization_with_stub(self):
        function=next(node for node in REMOTE_TREE.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        seen=[]
        def fake(argv,**kwargs):seen.append(kwargs['env']);return SimpleNamespace(returncode=0,stdout='',stderr='')
        g=dict(os=SimpleNamespace(environ={'SLURM_CLUSTERS':'foreign','SBATCH_ARRAY_INX':'1-10','CUDA_VISIBLE_DEVICES':'','PATH':'/bin'}),
          subprocess=SimpleNamespace(run=fake),observations=[],utc=lambda:'OFFLINE')
        exec(compile(ast.Module(body=[function],type_ignores=[]),'actual-run-helper','exec'),g)
        g['run'](['sbatch']);self.assertEqual(seen[-1],{'PATH':'/bin'})
        g['run'](['scontrol']);self.assertNotIn('SLURM_CLUSTERS',seen[-1])


if __name__=='__main__':unittest.main(verbosity=2)
