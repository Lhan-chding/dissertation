import importlib.util,json
from pathlib import Path
r=Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');i=r/'technical_incidents/compute_parallel_20261010';c=r/'code_engine_compute_candidate_20261010';py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'
s=importlib.util.spec_from_file_location('m',c/'scripts/sr_f1/compute_parallel_maintenance.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(not (i/'HANDOFF_SUBMISSION_INTENT.json').exists(),'INTENT_EXISTS_RECONCILE')
files=['compute_cancel.py','probe_submit.py','probe_release.py','compute_activate.py','compute_resume.py','compute_handoff.py']
for name in files:compile((i/name).read_text(),name,'exec')
m.save(i/'OPERATION_SOURCE_HASHES.json',{name:m.sha(i/name) for name in files})
script=i/'compute_handoff.sbatch';script.write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=""\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\nexec '+py+' '+str(i/'compute_handoff.py')+'\n')
args=['sbatch','--hold','--parsable','--account=rose','--qos=override-limits-but-killable','--partition=cluster02','--cpus-per-task=1','--time=4320','--job-name=srf11-compute-handoff','--comment=srf11-compute-handoff-20261010','--output='+str(i/'handoff_%j.log'),str(script)]
m.save(i/'HANDOFF_SUBMISSION_INTENT.json',dict(args=args,requires_probe_success=True,requires_preflight=True,requires_engine_before_baseline=True));res=m.command(args);m.save(i/'HANDOFF_SUBMISSION.json',res);m.require(res['returncode']==0,'SUBMISSION_UNKNOWN');job=res['stdout'].strip().split(';')[0];m.require(job.isdigit(),'JOB_INVALID')
obs=m.command(['scontrol','show','job',job,'--oneliner']);f=m.fields(obs)
for k,v in dict(JobId=job,Account='rose',QOS='override-limits-but-killable',JobName='srf11-compute-handoff',Comment='srf11-compute-handoff-20261010',Command=str(script),JobState='PENDING',Reason='JobHeldUser').items():m.require(f.get(k)==v,'HANDOFF_IDENTITY:'+k)
m.require(f['UserId'].split('(')[0]=='varun024' and 'gres/gpu' not in f['ReqTRES'],'CPU_IDENTITY')
m.save(i/'HANDOFF_RELEASE_INTENT.json',dict(observation=obs));release=m.command(['scontrol','release',job]);m.save(i/'HANDOFF_RELEASE.json',release);print(json.dumps(dict(job_id=job,release=release)))
