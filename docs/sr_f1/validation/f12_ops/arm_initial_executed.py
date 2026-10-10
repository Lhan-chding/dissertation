import pathlib,json,subprocess,re
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010');d=r/'deployment';c=r/'code'
py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'
m=json.loads((r/'srf12_source_manifest.json').read_text()); model=json.loads((r/'MODEL_ENVIRONMENT_IDENTITY.json').read_text())['model_path']
sub=json.loads((d/'CPU_SUBMISSION.json').read_text());assert sub['returncode']==0
cpu=sub['stdout'].strip().split(';')[0];assert cpu=='196823'
def save(name,value):
 with (d/name).open('x') as f:json.dump(value,f,indent=2)
assert not any((d/n).exists() for n in ['CONTROLLER_INTENT.json','CONTROLLER_SUBMISSION.json','CONTROLLER_RELEASE_INTENT.json','CONTROLLER_RELEASE.json']), 'Existing intent: reconcile without submission'
assert json.loads((d/'CPU_VALIDATION_COMPLETE.json').read_text())['status']=='PASS'
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',cpu,'-o','JobID,State,ExitCode'],text=True)
assert cpu+'|COMPLETED|0:0' in accounting
queue=subprocess.check_output(['squeue','-u','varun024','-h','-o','%i|%j|%k'],text=True)
assert 'srf12-controller' not in queue and not any(x.startswith(cpu+'|') for x in queue.splitlines())
for line in queue.splitlines():
 detail=subprocess.check_output(['scontrol','show','job','-o',line.split('|')[0]],text=True)
 assert 'Command='+str(d/'controller.sbatch')+' ' not in detail
save('CONTROLLER_ABSENCE_CHECK.json',dict(queue=queue,cpu_accounting=accounting,prior_intents_absent=True))
s=d/'controller.sbatch'
with s.open('x') as f:f.write('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=""\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8\nexport PYTHONPATH='+str(c/'src')+'\ntest -f '+str(d/'CPU_VALIDATION_COMPLETE.json')+'\nexec '+py+' '+str(c/'scripts/sr_f12/controller.py')+' --run-root '+str(r)+' --model-path '+model+' --source-manifest '+str(r/'srf12_source_manifest.json')+' --source-commit '+m['git_commit']+' --python '+py+'\n')
a=['sbatch','--hold','--parsable','--account=rose','--partition=cluster02','--qos=override-limits-but-killable','--cpus-per-task=2','--mem=12G','--time=4320','--job-name=srf12-controller','--comment=srf12-controller-3542869','--output='+str(d/'controller-%j.log'),str(s)]
save('CONTROLLER_INTENT.json',dict(args=a,source_commit=m['git_commit'],gpu_qos='soujanya-poria-startfund-2026-03',baseline_shards=3,technical_gpu=1,all_teacher_capacity_limit=5))
p=subprocess.run(a,capture_output=True,text=True);save('CONTROLLER_SUBMISSION.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
assert p.returncode==0,p.stderr
job=p.stdout.strip().split(';')[0];assert job.isdigit()
obs=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
f=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)',obs))
assert f['UserId'].split('(')[0]=='varun024' and f['Command']==str(s)
assert f['QOS']=='override-limits-but-killable' and 'gres/gpu' not in f['ReqTRES']
assert f['JobState']=='PENDING' and f['Reason']=='JobHeldUser'
save('CONTROLLER_RELEASE_INTENT.json',dict(job_id=job,observation=obs))
p=subprocess.run(['scontrol','release',job],capture_output=True,text=True)
save('CONTROLLER_RELEASE.json',dict(job_id=job,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0
print('CPU_CONTROLLER_ARMED',job)
