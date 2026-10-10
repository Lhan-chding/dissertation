import datetime,fcntl,hashlib,json,os,pathlib,re,subprocess,sys
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');inc=r/'technical_incidents/multigpu_20261010';code=r/'code';candidate=r/'code_engine_multigpu_candidate_v2'
py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=45)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,time=now())
freeze_sha='c7cf051d3aa6fb45889324e64b08d31a008137668aff2382283e4e19f99ff2b8';parent_sha='ebd17dda5860c292c2529895ab6d5816c04cee8e32c4c3f86230033438b82df0'
assert r.resolve()==r and sha(r/'EXECUTION_FREEZE.json')==freeze_sha and sha(r/'ENGINE_IO_REPAIR.json')==parent_sha
assert not (r/'ENGINE_MULTIGPU_REPAIR.json').exists() and not (r/'STOP').exists()

probe_receipt=read(inc/'GPU_PROBE.json')
probe_job=read(inc/'GPU_PROBE_SUBMISSION.json')['stdout'].strip().split(';')[0]
probe_code=r/'code_engine_multigpu_candidate_v2'
if not probe_code.exists():probe_code=r/'code'
assert probe_receipt['status']=='PASS' and probe_receipt['slurm_job_id']==probe_job and probe_receipt['expected_gpus']==3
assert len(probe_receipt['gpu_identities'])==3 and len({g['uuid'] for g in probe_receipt['gpu_identities']})==3
assert probe_receipt['real_9b_engine_qualified'] is False and probe_receipt['scientific_data_read'] is False
assert probe_receipt['script_sha256']==hashlib.sha256((probe_code/'scripts/sr_f1/validate_multigpu.py').read_bytes()).hexdigest()
assert set(probe_receipt['source_hashes'])=={'scripts/sr_f1/validate_multigpu.py','src/sr_f1/runtime.py','src/sr_f1/training.py','src/mm_dev/runtime.py','src/mm_core/training.py'}
for n,h in probe_receipt['source_hashes'].items():assert hashlib.sha256((probe_code/n).read_bytes()).hexdigest()==h,n

with (r.parent/'.sr_f1_project_controller.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 state=read(r/'orchestration/STATE.json');assert len(state['tasks']['ENGINE']['attempts'])==4 and state['tasks']['ENGINE']['attempts'][-1]['accounting']['terminal_state']=='CANCELLED'
 assert state['test_sealed'] is True and all(t['status']=='WAITING' and not t['attempts'] for k,t in state['tasks'].items() if k not in {'COMMON_START','ENGINE'})
 jobs=[read(inc/n)['stdout'].strip().split(';')[0] for n in ['CANDIDATE_CPU_V2_SUBMISSION.json','GPU_PROBE_SUBMISSION.json','STATE_REVIEW_SUBMISSION.json']]
 a=run(['sacct','-j',','.join(['196225','196227']+jobs),'--format=JobID,State,ExitCode','-n','-P']);rows={s[0]:s[1:3] for s in (line.split('|') for line in a['stdout'].splitlines())}
 assert a['returncode']==0 and all(rows[j][0].startswith('CANCELLED') for j in ['196225','196227']) and all(rows[j]==['COMPLETED','0:0'] for j in jobs),a
 probe=read(inc/'GPU_PROBE.json');assert probe['status']=='PASS' and probe['cleanup_verified'] is True and probe['real_9b_engine_qualified'] is False
 q=run(['squeue','--noheader','--user','varun024','--format=%i|%j|%T|%q'])
 assert q['returncode']==0 and not any('srf1-9ef274c9fc6a' in line or 'srf11-controller' in line for line in q['stdout'].splitlines()),q
 save(inc/'PRE_ACTIVATION_ACCOUNTING.json',dict(accounting=a,queue=q,time=now()))
 before=read(code/'SOURCE_DEPLOYMENT.json');assert before['source_commit']=='b48a638128bfeb6fc41201341df4902b3f2e660a'
 assert before==read(r/'code_before_engine_multigpu_20261010/SOURCE_DEPLOYMENT.json')
 for n,h in before['source_file_hashes'].items():assert sha(code/n)==sha(r/'code_before_engine_multigpu_20261010'/n)==h,n
 preservation=read(inc/'PRESERVATION.json')
 for n,e in preservation['artifact_hashes'].items():assert sha(inc/'evidence'/n)==e['sha256'],n
 assert read(inc/'CPU_STATE_REVIEW.json')['status']=='PASS_ZERO_UPDATE_FULL_STATE'
 review=read(inc/'RECOVERY_REVIEW.json');assert review['status']=='PASS_INTERRUPTED_ENGINE_RESTART'
 for n,h in review['consumed_marker_hashes'].items():assert sha(r/n)==h,n
 inv=read(inc/'inventory_v2.json');after=read(candidate/'SOURCE_DEPLOYMENT.json');assert after['source_commit']==inv['source_commit']
 for n,h in inv['files'].items():assert sha(candidate/n)==h,n
 changed=sorted(k for k in before['source_file_hashes'].keys()|after['source_file_hashes'].keys() if before['source_file_hashes'].get(k)!=after['source_file_hashes'].get(k))
 assert changed==sorted(['src/sr_f1/freeze.py','src/sr_f1/runtime.py','src/sr_f1/orchestration.py','scripts/sr_f1/submit_matrix.py','scripts/sr_f1/run_worker.py','scripts/sr_f1/validate_multigpu.py']),changed
 receipt=dict(repair_id='SR_F1_1_ENGINE_MULTIGPU_20261010',status='AUTHORIZED_TECHNICAL_MULTIGPU_REPAIR',plan_id='SR-F1-20261009',run_root=str(r),original_execution_freeze_sha256=freeze_sha,previous_engine_io_repair_sha256=parent_sha,previous_source_commit=before['source_commit'],previous_source_tree_sha256=before['source_tree_sha256'],preserved_source_relative_path='code_before_engine_multigpu_20261010',registration_sha256=sha(r/'orchestration/REGISTRATION.json'),previous_worker_source_sha256=before['source_file_hashes']['scripts/sr_f1/run_worker.py'],worker_source_sha256=after['source_file_hashes']['scripts/sr_f1/run_worker.py'],scientific_protocol_unchanged=True,original_once_consumed_preserved=True,engine_restart_mode='FULL_REGISTERED_CONTINUOUS4_SPLIT2_PLUS2',maintained_attempt_id='ENGINE_attempt0003',maintained_job_id='196227',gpu_count=4,compute_device_index=0,storage_device_indices=[1,2,3],gpu_activation_budget_bytes=80<<30,cpu_activation_budget_bytes=48<<30,activation_storage='AUXILIARY_GPU_THEN_BOUNDED_CPU_EXACT_EXTERNAL_DISK',gpu_worker_constraint='highmem',minimum_gpu_host_memory_gb_per_gpu=80,minimum_gpu_host_memory_gb=320,cpus_per_gpu=4,qos='soujanya-poria-startfund-2026-03',resource_operations=['engine','train'],quota_root='/projects/_hdd/varunhdd',activation_spill_directory='/projects/_hdd/varunhdd/louis-ssvc/sr_f11_20261010_activation_offload',quota_reserve_bytes=10<<30,authorized_user_message='我说了请你记住，如果能用多张卡提速一定要用多张卡，只要老师的那个qos的卡上限五张都没用完就增加（只是最多五张）我需要速度提升',authorized_at=now(),historical_artifact_hashes={str(p.relative_to(r)):sha(p) for p in [inc/n for n in ['PRESERVATION.json','TERMINAL_JOBS.json','CPU_STATE_REVIEW.json','RECOVERY_REVIEW.json','STATE_RECONCILED.json','PRE_ACTIVATION_ACCOUNTING.json','GPU_PROBE.json','CANDIDATE_SERVER_TESTS_V2.log']]},changed_files=changed,**{k:after[k] for k in ['source_commit','source_tree_sha256','source_file_hashes']})
 save(r/'ENGINE_MULTIGPU_REPAIR.json',receipt)
 sys.path.insert(0,str(candidate/'src'))
 from sr_f1.freeze import verify_engine_multigpu_repair
 check=verify_engine_multigpu_repair(r);assert check['source_commit']==after['source_commit'] and check['gpu_count']==4
 code.rename(r/'code_displaced_engine_multigpu_20261010');candidate.rename(code)
 assert sha(r/'EXECUTION_FREEZE.json')==freeze_sha and sha(r/'ENGINE_IO_REPAIR.json')==parent_sha
 save(inc/'SOURCE_ACTIVATED.json',dict(time=now(),status='BYTE_VERIFIED_PENDING_POST_ACTIVATION_CHECKS',source_commit=after['source_commit'],source_tree_sha256=after['source_tree_sha256'],files=len(inv['files']),repair_sha256=sha(r/'ENGINE_MULTIGPU_REPAIR.json'),execution_freeze_sha256=freeze_sha,gpu_count=4,old_markers_preserved=True,new_generation_count=0,unrelated_jobs_modified=0))
script=inc/'verify_activation.sbatch'
script.write_text(f'''#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH={code}/src
cd {code}
{py} scripts/sr_f1/preflight.py --root {r} > {inc}/PREFLIGHT.log 2>&1
{py} -c 'import json; from sr_f1.freeze import verify_engine_multigpu_repair; print(json.dumps(verify_engine_multigpu_repair("{r}"),indent=2))' > {inc}/MULTIGPU_VERIFY.log 2>&1
{py} scripts/sr_f1/submit_matrix.py --plan {r}/config/SR_F1_1.json --run-root {r} --code-root {code} --python {py} --resume-repaired-engine-multigpu > {inc}/RESUME_AUTHORIZATION.log 2>&1
''')
args=['sbatch','--parsable','--account=rose','--qos=override-limits-but-killable','--partition=cluster02','--cpus-per-task=2','--time=30','--job-name=srf11-multigpu-verify','--output='+str(inc/'verify_%j.log'),str(script)]
save(inc/'VERIFY_SUBMISSION_INTENT.json',dict(command=args,time=now()));x=run(args);save(inc/'VERIFY_SUBMISSION.json',x);assert x['returncode']==0,x
print(json.dumps(dict(status='MULTIGPU_SOURCE_ACTIVATED_CPU_VERIFICATION_SUBMITTED',verify_job=x['stdout'].strip(),source_commit=after['source_commit'],repair_sha256=sha(r/'ENGINE_MULTIGPU_REPAIR.json'))))
