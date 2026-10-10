import datetime,fcntl,importlib.util,json,os,subprocess,sys
from pathlib import Path
r=Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');inc=r/'technical_incidents/compute_parallel_20261010';c=r/'code_engine_compute_candidate_20261010';old=r/'code';py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12'
s=importlib.util.spec_from_file_location('m',c/'scripts/sr_f1/compute_parallel_maintenance.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(r.resolve()==r and not (r/'STOP').exists(),'ROOT_OR_STOP')
sys.path.insert(0,str(c/'src'))
from sr_f1.freeze import build_engine_compute_parallel_repair,verify_engine_compute_parallel_repair
from sr_f1.prepare import source_identity
with (r.parent/'.sr_f1_project_controller.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 m.require(not (inc/'ACTIVATION_INTENT.json').exists(),'ACTIVATION_INTENT_EXISTS_RECONCILE')
 m.terminal_jobs(r,m.read(inc/'PRE_MAINTENANCE_SNAPSHOT.json'))
 ids=[m.read(inc/(stem+'_SUBMISSION.json'))['stdout'].strip().split(';')[0] for stem in ['CANDIDATE_CPU','PRESERVE_CPU','PROBE_GPU']]
 a=m.command(['sacct','-j',','.join(ids),'-n','-P','--format=JobIDRaw,State,ExitCode']);rows={x[0]:x[1:3] for x in (ln.split('|') for ln in a['stdout'].splitlines())};m.require(a['returncode']==0 and all(rows.get(j)==['COMPLETED','0:0'] for j in ids),'VALIDATION_JOBS_NOT_COMPLETE')
 q=m.command(['squeue','-u','varun024','--noheader','--format=%i|%j|%T|%q']);m.require(q['returncode']==0 and not any(line.split('|')[0] in ids or 'srf1-9ef274c9fc6a' in line or 'srf11-controller' in line or 'srf11-baseline-handoff' in line for line in q['stdout'].splitlines()),'PROJECT_JOB_ACTIVE')
 m.verify_preservation(r,m.read(inc/'PRESERVATION.json'))
 source=source_identity();m.require(source==m.inventory(c),'CANDIDATE_SOURCE_CHANGED');m.require(m.inventory(old)==m.inventory(r/'code_before_engine_compute_parallel_20261010'),'OLD_SOURCE_CHANGED')
 synthetic=m.read(inc/'SYNTHETIC_GPU_PROBE.json');real=m.read(inc/'PROBE_9B_20261010/PROBE_9B.json')
 m.require(synthetic['status']==real['status']=='PASS' and synthetic['slurm_job_id']==ids[-1] and real['slurm']['JobId']==ids[-1],'PROBE_FAILED_OR_JOB_CHANGED')
 realfile=inc/'REAL_MODEL_PROBE.json';m.require(not realfile.exists(),'REAL_RECEIPT_EXISTS_RECONCILE');realfile.write_bytes((inc/'PROBE_9B_20261010/PROBE_9B.json').read_bytes())
 m.save(inc/'GPU_PROBE.json',{**synthetic,'real_model_probe':dict(path=str(realfile.relative_to(r)),sha256=m.sha(realfile))})
 m.require(m.sha(r/'EXECUTION_FREEZE.json')=='c7cf051d3aa6fb45889324e64b08d31a008137668aff2382283e4e19f99ff2b8','FREEZE_CHANGED')
 receipt=build_engine_compute_parallel_repair(r,actual_source=source,gpu_count=4,authorized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorized_user_message='120天太久了，为什么四张卡只有一张在计算？请检查代码和瓶颈，在不改变实验设置的前提下，让多张卡真正并行计算提速，仍然用老师的qos，最多五张卡，不要动无关作业。')
 m.save(inc/'ACTIVATION_INTENT.json',dict(source=source,accounting=a,queue=q,receipt=receipt,old_source_destination=str(r/'code_displaced_engine_compute_20261010')))
 m.save(r/'ENGINE_COMPUTE_PARALLEL_REPAIR.json',receipt)
 verified=verify_engine_compute_parallel_repair(r,actual_source=source)
 old.rename(r/'code_displaced_engine_compute_20261010');c.rename(old)
 m.require(m.inventory(old)==source,'ACTIVE_SOURCE_DIFFERS')
 m.save(inc/'SOURCE_ACTIVATED.json',dict(status='SOURCE_ACTIVATED_PENDING_PREFLIGHT',source_commit=source['source_commit'],repair_sha256=m.sha(r/'ENGINE_COMPUTE_PARALLEL_REPAIR.json'),unrelated_jobs_modified=0))
script=inc/'verify_activation.sbatch'
script.write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=""\nexport PYTHONPATH='+str(old/'src')+'\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'+py+' '+str(old/'scripts/sr_f1/preflight.py')+' --root '+str(r)+' > '+str(inc/'PREFLIGHT.log')+' 2>&1\n'+py+' '+str(old/'scripts/sr_f1/submit_matrix.py')+' --plan '+str(r/'config/SR_F1_1.json')+' --run-root '+str(r)+' --code-root '+str(old)+' --python '+py+' --resume-repaired-engine-compute-parallel > '+str(inc/'RESUME_AUTHORIZATION.log')+' 2>&1\n')
args=['sbatch','--parsable','--account=rose','--qos=override-limits-but-killable','--partition=cluster02','--cpus-per-task=2','--time=30','--job-name=srf11-compute-activation','--output='+str(inc/'verify_%j.log'),str(script)]
m.save(inc/'VERIFY_CPU_INTENT.json',dict(args=args));res=m.command(args);m.save(inc/'VERIFY_CPU_SUBMISSION.json',res);print(json.dumps(res))
