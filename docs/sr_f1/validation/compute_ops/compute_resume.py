import datetime,fcntl,importlib.util,json,sys
from pathlib import Path
r=Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');i=r/'technical_incidents/compute_parallel_20261010';c=r/'code';py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'
s=importlib.util.spec_from_file_location('m',c/'scripts/sr_f1/compute_parallel_maintenance.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);sys.path.insert(0,str(c/'src'))
from sr_f1.freeze import verify_engine_compute_parallel_repair
with (r.parent/'.sr_f1_project_controller.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 m.require(not (i/'CONTROLLER_SUBMISSION_INTENT.json').exists(),'CONTROLLER_INTENT_EXISTS_RECONCILE')
 j=m.read(i/'VERIFY_CPU_SUBMISSION.json')['stdout'].strip().split(';')[0]
 a=m.command(['sacct','-j',j,'-n','-P','--format=JobIDRaw,State,ExitCode']);m.require(a['returncode']==0 and [j,'COMPLETED','0:0'] in [ln.split('|')[:3] for ln in a['stdout'].splitlines()],'VERIFY_CPU_NOT_COMPLETE')
 m.require(m.read(i/'PREFLIGHT.log')['status']=='PASS','PREFLIGHT_FAILED');v=verify_engine_compute_parallel_repair(r);state=m.read(r/'orchestration/STATE.json');auth=m.read(i/'RESUME_AUTHORIZATION.log')
 m.require(auth==state['engine_compute_parallel_repair_resume'] and auth['consumed_by_attempt_id'] is None,'RESUME_ALREADY_CONSUMED')
 m.require(auth['permitted_next_attempt_id']=='ENGINE_attempt0005' and state['test_sealed'] and state['tasks']['ENGINE']['status']=='RETRYABLE','STATE_NOT_READY')
 m.require(not any(t['attempts'] for k,t in state['tasks'].items() if k not in {'ENGINE','COMMON_START'}),'DOWNSTREAM_STARTED')
 q=m.command(['squeue','-u','varun024','--noheader','--format=%i|%j|%T|%q']);m.require(q['returncode']==0 and not any('srf1-9ef274c9fc6a' in ln or 'srf11-controller' in ln or 'srf11-baseline-handoff' in ln or 'srf11-compute-probe' in ln for ln in q['stdout'].splitlines()),'PROJECT_JOB_ACTIVE')
 args=['sbatch','--hold','--parsable','--account=rose','--qos=override-limits-but-killable','--partition=cluster02','--job-name=srf11-controller-compute','--comment=srf11-compute-controller-20261010','--output='+str(i/'controller_%j.log'),str(c/'scripts/sr_f1/controller.sbatch'),py,str(c),str(r/'config/SR_F1_1.json'),str(r)]
 m.save(i/'CONTROLLER_SUBMISSION_INTENT.json',dict(args=args,repair_sha256=v['repair_sha256'],queue=q));sub=m.command(args);m.save(i/'CONTROLLER_SUBMISSION.json',sub);m.require(sub['returncode']==0,'UNKNOWN_SUBMISSION');job=sub['stdout'].strip().split(';')[0];m.require(job.isdigit(),'JOB_ID_INVALID')
 obs=m.command(['scontrol','show','job',job,'--oneliner']);f=m.fields(obs)
 for key,val in dict(JobId=job,Account='rose',QOS='override-limits-but-killable',JobName='srf11-controller-compute',Comment='srf11-compute-controller-20261010',Command=str(c/'scripts/sr_f1/controller.sbatch'),JobState='PENDING',Reason='JobHeldUser').items():m.require(f.get(key)==val,'CONTROLLER_IDENTITY:'+key)
 m.require(f['UserId'].split('(')[0]=='varun024' and 'gres/gpu' not in f['ReqTRES'],'CPU_RESOURCE_IDENTITY')
 m.save(i/'CONTROLLER_RELEASE_INTENT.json',dict(job_id=job,observation=obs));res=m.command(['scontrol','release',job]);m.save(i/'CONTROLLER_RELEASE.json',res);print(json.dumps(dict(job_id=job,result=res)))
