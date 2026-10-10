import pathlib,sys,json,subprocess
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010');c=r/'code';d=r/'technical_incidents/batch_single_20261010';d.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(c/'src'))
from sr_f12.orchestration import Slurm,teacher_usage,save_json,file_hash
s=Slurm();q=s.queue('varun024');assert teacher_usage(q,[])<5
assert not (d/'SUBMISSION_INTENT.json').exists(), 'existing submission must reconcile'
term=s.terminal('196825');assert term['state']=='FAILED' and term['exit_code']=='1:0' and '196825' not in q
save_json(d/'FAILED_TERMINAL.json',term,exclusive=True)
py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'
f=d/'diagnostic.sbatch';f.write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH='+str(c/'src')+'\nexport PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8\nexec '+py+' '+str(d/'srf12_tf_diagnostic.py')+' --run-root '+str(r)+' --output '+str(d/'TF_DIAGNOSTIC.json')+'\n')
a=['sbatch','--hold','--parsable','--account=rose','--partition=cluster02','--qos=soujanya-poria-startfund-2026-03','--gres=gpu:pro6000:1','--constraint=highmem','--cpus-per-task=4','--time=60','--job-name=srf12-tf-diagnostic','--output='+str(d/'diagnostic-%j.log'),str(f)]
save_json(d/'SUBMISSION_INTENT.json',dict(args=a,source_manifest_sha256=file_hash(r/'srf12_source_manifest.json'),script_sha256=file_hash(d/'srf12_tf_diagnostic.py')),exclusive=True)
p=subprocess.run(a,text=True,capture_output=True);save_json(d/'SUBMISSION.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),exclusive=True);assert p.returncode==0,p.stderr
job=p.stdout.strip().split(';')[0];assert job.isdigit();obs=s.show(job)
assert obs['QOS']=='soujanya-poria-startfund-2026-03' and obs['UserId'].split('(')[0]=='varun024' and obs['Command']==str(f)
assert obs['JobState']=='PENDING' and obs['Reason']=='JobHeldUser'
from sr_f12.orchestration import gpu_count,memory_gib
assert gpu_count(obs['ReqTRES'])==1 and memory_gib(obs['ReqTRES'])>=80
assert teacher_usage(s.queue('varun024'),[])<=5
save_json(d/'RELEASE_INTENT.json',dict(job_id=job,observation=obs),exclusive=True)
p=subprocess.run(['scontrol','release',job],text=True,capture_output=True);save_json(d/'RELEASE.json',dict(job_id=job,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),exclusive=True);assert p.returncode==0
print('DIAGNOSTIC_SUBMITTED',job)
