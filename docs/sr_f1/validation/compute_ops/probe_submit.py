import json,os,re,subprocess,sys,importlib.util
from pathlib import Path
r=Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010'); inc=r/'technical_incidents/compute_parallel_20261010'; c=r/'code_engine_compute_candidate_20261010'; py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'; qos='soujanya-poria-startfund-2026-03'
spec=importlib.util.spec_from_file_location('m',c/'scripts/sr_f1/compute_parallel_maintenance.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
phase=sys.argv[1]
if phase=='preserve':
 m.terminal_jobs(r,m.read(inc/'PRE_MAINTENANCE_SNAPSHOT.json'))
 script=inc/'preserve_cpu.sbatch'
 script.write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=""\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\nexec '+py+' '+str(c/'scripts/sr_f1/compute_parallel_maintenance.py')+' --run-root '+str(r)+' --phase preserve-only\n')
 args=['sbatch','--parsable','--account=rose','--qos=override-limits-but-killable','--partition=cluster02','--cpus-per-task=2','--time=30','--job-name=srf11-compute-preserve','--output='+str(inc/'preserve_cpu_%j.log'),str(script)]; stem='PRESERVE_CPU'
elif phase=='probe':
 m.require(m.read(inc/'CPU_STATE_REVIEW.json')['status']=='PASS_PRESERVED_FULL_STATE','CPU_REVIEW_NOT_PASS');m.verify_preservation(r,m.read(inc/'PRESERVATION.json'));m.terminal_jobs(r,m.read(inc/'PRE_MAINTENANCE_SNAPSHOT.json'));m.inventory(c)
 q=m.command(['squeue','-u','varun024','--noheader','--format=%i|%q|%T|%b']);m.require(q['returncode']==0,'QUEUE_UNKNOWN')
 total=0; jobs=[]
 for line in q['stdout'].splitlines():
  job,aq,state,gres=line.strip().split('|');
  if aq!=qos:continue
  obs=m.command(['scontrol','show','job',job,'--oneliner']);f=m.fields(obs);m.require(f['QOS']==qos and f['UserId'].split('(')[0]=='varun024','CAPACITY_OWNER_UNKNOWN')
  tres=dict(x.split('=',1) for x in f['ReqTRES'].split(',')); n=int(tres.get('gres/gpu',0));total+=n;jobs.append(dict(job_id=job,gpus=n,observation=obs))
 m.require(total+4<=5,'TEACHER_CAPACITY_EXCEEDED')
 state=m.read(r/'orchestration/STATE.json')
 m.require(not any(a.get('status')=='SUBMISSION_UNKNOWN' for t in state['tasks'].values() for a in t.get('attempts',[])),'UNKNOWN_SUBMISSION')
 script=inc/'probe_four_gpu.sbatch'
 script.write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH='+str(c/'src')+'\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8\n'+py+' '+str(c/'scripts/sr_f1/validate_compute_parallel.py')+' --gpu-count 4 --output '+str(inc/'SYNTHETIC_GPU_PROBE.json')+'\n'+py+' '+str(c/'scripts/sr_f1/validate_compute_parallel_9b.py')+' --plan '+str(r/'config/SR_F1_1.json')+' --run-root '+str(r)+' --raw-directory '+str(inc/'evidence/engineering/engine/natural/continuous/rollouts')+' --output '+str(inc/'PROBE_9B_20261010')+' --long-tokens 768\n')
 args=['sbatch','--parsable','--hold','--account=rose','--qos='+qos,'--partition=cluster02','--nodes=1','--gres=gpu:pro6000:4','--constraint=highmem','--cpus-per-task=16','--time=180','--job-name=srf11-compute-probe','--comment=srf11-compute-probe-20261010','--output='+str(inc/'probe_four_gpu_%j.log'),str(script)];stem='PROBE_GPU'
 m.save(inc/'PROBE_CAPACITY.json',dict(queue=q,jobs=jobs,gpus_before=total,gpus_requested=4,unknown_reserved=0))
else:raise ValueError(phase)
m.require(not (inc/(stem+'_INTENT.json')).exists(),'SUBMISSION_INTENT_EXISTS_RECONCILE')
m.save(inc/(stem+'_INTENT.json'),dict(args=args));result=m.command(args);m.save(inc/(stem+'_SUBMISSION.json'),result);print(json.dumps(result));m.require(result['returncode']==0,'SUBMISSION_FAILED_OR_UNKNOWN')
