"""Offline synthetic boundary checks only; never server/GPU certification."""
import copy,importlib.util,io,json,os,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
P=Path('/Users/louis/Documents/ChatGPT/dissertation-ntu/experiments/sr_satdose_20261112/operations/satdose_audit_trained_dev.py')
s=importlib.util.spec_from_file_location('audit_dev_offline',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def ep():return dict(start_seed=11031,arm='T1',step=2,responses=108)
def obs(hold=False):
 d=dict(job_id='123',job_name='LH-SATDOSE-DEV-11031-T1-s02',account='rose',qos=m.QOS,partition='cluster02',
  start='2026-11-16T00:00:00',end='2026-11-16T01:00:00',nodes='gpu-pro6000-8',
  state='FAILED' if hold else 'COMPLETED',exit_code='2:0' if hold else '0:0')
 return dict(records=[d,dict(d,job_id='123.0'),dict(d,job_id='123.extern',state='COMPLETED',exit_code='0:0')])

class Checks(unittest.TestCase):
 def test_successful_normal_scheduler(self):m.completed_scheduler(obs(),123,ep(),False)
 def test_planned_exit2_scientific_hold_is_auditable(self):m.completed_scheduler(obs(True),123,ep(),True)
 def test_unexplained_exit2_rejected(self):
  with self.assertRaises(PermissionError):m.completed_scheduler(obs(True),123,ep(),False)
 def test_scientific_hold_does_not_waive_oom_or_signal(self):
  for state,exitcode in [('OUT_OF_MEMORY','0:125'),('TIMEOUT','0:15'),('FAILED','1:0'),('CANCELLED','0:15')]:
   value=obs(True);value['records'][1].update(state=state,exit_code=exitcode)
   with self.assertRaises(PermissionError):m.completed_scheduler(value,123,ep(),True)
 def test_missing_task_step_rejected(self):
  value=obs();value['records'].pop(1)
  with self.assertRaises(PermissionError):m.completed_scheduler(value,123,ep(),False)
 def test_extra_or_duplicate_model_task_rejected(self):
  for suffix in ('.1','.0','.interactive'):
   value=obs();value['records'].append(dict(value['records'][1],job_id='123'+suffix))
   with self.assertRaises(PermissionError):m.completed_scheduler(value,123,ep(),False)
 def test_scheduler_cannot_be_redirected_by_slurm_clusters(self):
  result=SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
  with patch.dict(m.os.environ,{'SLURM_CLUSTERS':'foreign'}),patch.object(m.subprocess,'run',return_value=result) as run:
   m.scheduler(123)
   self.assertNotIn('SLURM_CLUSTERS',run.call_args.kwargs['env'])
 def test_publication_symlink_rejected_before_mkdir(self):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp).resolve();root=base/'root';root.mkdir();outside=base/'outside';outside.mkdir()
   (root/'execution_releases').symlink_to(outside,target_is_directory=True)
   with patch.object(m,'ROOT',root),patch.dict(m.os.environ,{'CUDA_VISIBLE_DEVICES':''}):
    with self.assertRaisesRegex(PermissionError,'Noncanonical audit publication parent'):
     m.remote_worker(dict(endpoint=ep(),job_id=123))
   self.assertEqual(list(outside.iterdir()),[])
 def test_source_inventory_rejects_fifo_without_opening(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();(root/'execution_releases').mkdir();code=root/'train_code';code.mkdir();os.mkfifo(code/'special')
   with patch.object(m,'ROOT',root),patch.object(m,'CODE',code),patch.dict(m.os.environ,{'CUDA_VISIBLE_DEVICES':''}):
    with self.assertRaisesRegex(PermissionError,'Special file in original code inventory'):
     m.remote_worker(dict(endpoint=ep(),job_id=123))
 def test_foreign_qos_job_or_endpoint_rejected(self):
  for key,value in [('qos','rose'),('job_name','other-user-job'),('job_id','124'),('account','other')]:
   data=obs();data['records'][0][key]=value
   with self.assertRaises(PermissionError):m.completed_scheduler(data,123,ep(),False)
 def test_no_main_or_step24_endpoint(self):
  for key,value in [('step',24),('arm','START'),('responses',107),('start_seed',True)]:
   data=ep();data[key]=value
   with self.assertRaises(PermissionError):m.endpoint(data)
 def test_exact_worker_stdout_bound_even_for_negative(self):
  value=dict(status='TRAINED_DEV_ENDPOINT_COMPLETE',global_scientific_hold=True)
  data=b'native warning\n'+m.canonical(value);m.worker_stdout(data,value)
  with self.assertRaises(PermissionError):m.worker_stdout(data+m.canonical(value),value)
  with self.assertRaises(PermissionError):m.worker_stdout(data,dict(value,global_scientific_hold=False))
 def test_technical_worker_error_not_mislabeled_complete(self):
  with self.assertRaises(PermissionError):m.worker_stdout(m.canonical(dict(status='HOLD_TRAINED_DEV_CONTROL')),dict(status='TRAINED_DEV_ENDPOINT_COMPLETE'))
 def test_duplicate_or_nonfinite_json_rejected(self):
  for text in ['{"x":1,"x":2}','{"x":NaN}']:
   with self.assertRaises(PermissionError):m.loads(text)
 def test_references_cannot_escape_or_use_short_hash(self):
  for p,h in [('../other', 'a'*64),('/absolute','a'*64),('a.json','short')]:
   with self.assertRaises(PermissionError):m.check_ref(dict(path=p,sha256=h))
 def test_writeonce_keeps_original_negative_bytes(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp).resolve()/'negative.json';raw=b'{"global_hold": true, "counts": 0}\n';m.write_once(p,raw)
   with self.assertRaises(FileExistsError):m.write_once(p,b'PASS')
   self.assertEqual(p.read_bytes(),raw)
 def test_default_plan_has_no_subprocess(self):
  payload=dict(endpoint=ep())
  with patch.object(m,'local_payload',return_value=payload),patch.object(m.subprocess,'run') as run,redirect_stdout(io.StringIO()):
   self.assertEqual(m.local(SimpleNamespace(execute=False,job_id=123)),0);run.assert_not_called()
 def test_close_and_zero_optimizer_contract(self):
  r=dict(path='release.json',sha256='a'*64);source=dict(path='/source.py',sha256='b'*64);identity=dict(start_seed=11031,arm='T1',step=2)
  result=dict(schema='sr-satdose-trained-dev-result-v1',status='TRAINED_DEV_ENDPOINT_COMPLETE',permission='NO_TRAINING_OR_MAIN',
   **ep(),release=r,global_scientific_hold=True,process_identity={'pid':1})
  binding=dict(schema='sr-satdose-evaluation-endpoint-binding-v1',scope='DEV',endpoint=identity,requests=108,request_ids_sha256='c'*64,execution_sources_sha256='d'*64)
  claim=dict(schema='sr-satdose-trained-dev-claim-v1',status='ATTEMPTED_UNKNOWN_NO_RETRY',process_identity={'pid':1},release=r,**ep())
  close=dict(closed=True,scope='DEV',attempted_requests=108,request_ids_sha256='c'*64,optimizer_steps=0,scheduler_steps=0,parameter_updates=0,**identity)
  runtime=dict(schema='sr-satdose-evaluation-runtime-v1',endpoint=identity,requests=108,request_ids_sha256='c'*64,execution_sources_sha256='d'*64,
   authority_source=source,optimizer_constructed=False,scheduler_constructed=False,training_rng_restored=False)
  original=dict(execution_sources={'trained_dev_control':source});args=[result,binding,claim,close,runtime,ep(),r,original]
  m.verify_closed(*args)
  for index,key,value in [(3,'optimizer_steps',1),(4,'optimizer_constructed',True),(3,'attempted_requests',107),(2,'process_identity',{'pid':2})]:
   bad=copy.deepcopy(args);bad[index][key]=value
   with self.assertRaises(PermissionError):m.verify_closed(*bad)

if __name__=='__main__':unittest.main(verbosity=2)
