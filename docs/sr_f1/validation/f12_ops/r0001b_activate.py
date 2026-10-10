import pathlib,sys,json,subprocess,re,shlex
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010');c=r/'code_repair_r0001b';d=r/'deployment/r0001b'
py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'
sys.path.insert(0,str(c/'src'))
from sr_f12.orchestration import Slurm,save_json,file_hash
from sr_f12.evaluation import verify_source_manifest
from sr_f12.protocol import AMENDMENT_ID,object_hash
from sr_f12.source_revision import prepare_technical_repair
from sr_f12.endpoint_orchestration import MODEL_IDS
manifest=r/'source_r0001b.json';m=json.loads(manifest.read_text());source_hash=verify_source_manifest(manifest,c)
assert json.loads((d/'VALIDATION_COMPLETE.json').read_text())['source_sha256']==source_hash
s=Slurm();queue=s.queue('varun024')
for row in queue.values():
 assert row.get('JobName') not in ('srf12-controller-r0001b','srf12-endpoint-r0001b')
assert not any((d/(name+'_INTENT.json')).exists() for name in ('CONTROLLER','ENDPOINT'))
receipt=prepare_technical_repair(r,revision_id='r0001',code_root=c,source_manifest=manifest,source_commit=m['git_commit'],controller_job_id='196824')
a=json.loads((r/'AMENDMENT.json').read_text())
save_json(r/'ENDPOINT_IMPLEMENTATION.json',dict(status='REGISTERED_ENDPOINT_IMPLEMENTATION',amendment_id=AMENDMENT_ID,amendment_sha256=object_hash(a),training_source_commit=a['source_commit'],source_commit=m['git_commit'],source_sha256=source_hash,scientific_settings_unchanged=True,model_ids=list(MODEL_IDS)),exclusive=True)
model=json.loads((r/'MODEL_ENVIRONMENT_IDENTITY.json').read_text())['model_path']
for name,entry in [('CONTROLLER','controller.py'),('ENDPOINT','endpoint_controller.py')]:
 script=d/(name.lower()+'.sbatch')
 args=[py,str(c/'scripts/sr_f12'/entry),'--run-root',str(r),'--model-path',model,'--source-manifest',str(manifest),'--source-commit',m['git_commit'],'--python',py]
 with script.open('x') as f:f.write('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=""\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8\nexport PYTHONPATH='+str(c/'src')+'\nexec '+shlex.join(args)+'\n')
 cmd=['sbatch','--hold','--parsable','--account=rose','--partition=cluster02','--qos=override-limits-but-killable','--cpus-per-task=2','--mem=12G','--time=4320','--job-name=srf12-'+name.lower()+'-r0001b','--output='+str(d/(name.lower()+'-%j.log')),str(script)]
 save_json(d/(name+'_INTENT.json'),dict(args=cmd,script_sha256=file_hash(script),source_commit=m['git_commit']),exclusive=True)
 p=subprocess.run(cmd,text=True,capture_output=True);save_json(d/(name+'_SUBMISSION.json'),dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),exclusive=True);assert p.returncode==0,p.stderr
 job=p.stdout.strip().split(';')[0];assert job.isdigit();obs=s.show(job)
 assert obs['Command']==str(script) and obs['UserId'].split('(')[0]=='varun024' and obs['QOS']=='override-limits-but-killable' and 'gres/gpu' not in obs['ReqTRES']
 assert obs['JobState']=='PENDING' and obs['Reason']=='JobHeldUser'
 save_json(d/(name+'_RELEASE_INTENT.json'),dict(job_id=job,observation=obs),exclusive=True)
 p=subprocess.run(['scontrol','release',job],text=True,capture_output=True);save_json(d/(name+'_RELEASE.json'),dict(job_id=job,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),exclusive=True);assert p.returncode==0
 print(name,'ARMED',job,flush=True)
