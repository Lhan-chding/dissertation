import importlib.util,json
from pathlib import Path
r=Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');i=r/'technical_incidents/compute_parallel_20261010';c=r/'code_engine_compute_candidate_20261010'
s=importlib.util.spec_from_file_location('m',c/'scripts/sr_f1/compute_parallel_maintenance.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
sub=m.read(i/'PROBE_GPU_SUBMISSION.json');m.require(sub['returncode']==0,'UNKNOWN_SUBMISSION');job=sub['stdout'].strip().split(';')[0];m.require(job.isdigit(),'INVALID_JOB')
obs=m.command(['scontrol','show','job',job,'--oneliner']);f=m.fields(obs)
for k,v in dict(JobId=job,UserId='varun024',Account='rose',QOS=m.TEACHER_QOS,JobName='srf11-compute-probe',Comment='srf11-compute-probe-20261010',Command=str(i/'probe_four_gpu.sbatch'),JobState='PENDING',Reason='JobHeldUser',Features='highmem').items():
 m.require((f.get(k,'').split('(')[0] if k=='UserId' else f.get(k))==v,'IDENTITY:'+k)
t=dict(x.split('=',1) for x in f['ReqTRES'].split(','));m.require(t.get('gres/gpu')=='4' and t.get('gres/gpu:pro6000')=='4' and int(t['cpu'])==16,'RESOURCE_CHANGED');mem=t['mem'];m.require(mem.endswith('G') and float(mem[:-1])>=320,'MEMORY_INSUFFICIENT')
q=m.command(['squeue','-u','varun024','--noheader','--format=%i|%q']);m.require(q['returncode']==0,'QUEUE_UNKNOWN');count=0
for line in q['stdout'].splitlines():
 jid,qos=line.strip().split('|')
 if qos==m.TEACHER_QOS:
  jf=m.fields(m.command(['scontrol','show','job',jid,'--oneliner']));tres=dict(x.split('=',1) for x in jf['ReqTRES'].split(','));count+=int(tres.get('gres/gpu',0))
m.require(count<=5,'CAPACITY_EXCEEDED');m.terminal_jobs(r,m.read(i/'PRE_MAINTENANCE_SNAPSHOT.json'))
m.require(not (i/'PROBE_GPU_RELEASE_INTENT.json').exists(),'RELEASE_UNKNOWN_RECONCILE');m.save(i/'PROBE_GPU_RELEASE_INTENT.json',dict(job_id=job,identity=obs,queue=q,total_teacher_gpus=count));res=m.command(['scontrol','release',job]);m.save(i/'PROBE_GPU_RELEASE.json',res);print(json.dumps(res))
