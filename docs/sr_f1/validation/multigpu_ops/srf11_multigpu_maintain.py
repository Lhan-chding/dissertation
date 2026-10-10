import datetime,hashlib,json,os,pathlib,subprocess
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');inc=r/'technical_incidents/multigpu_20261010'
def read(p):return json.loads(p.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
assert r.resolve()==r and not (inc/'MAINTENANCE_CANCEL_INTENT.json').exists()

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

assert read(r/'code/SOURCE_DEPLOYMENT.json')['source_commit']=='b48a638128bfeb6fc41201341df4902b3f2e660a'
state=read(r/'orchestration/STATE.json');assert state['test_sealed'] and len(state['tasks']['ENGINE']['attempts'])==4 and state['tasks']['ENGINE']['attempts'][-1]['job_id']=='196227'
assert not any(t['attempts'] for k,t in state['tasks'].items() if k not in ['COMMON_START','ENGINE'])
assert read(r/'engineering/engine/natural/continuous/checkpoints/LATEST.json')['step']==0
snap=inc/'pre_maintenance';snap.mkdir();files={}
for relative in ['engineering/engine','accounting/ENGINE.jsonl','orchestration/STATE.json','orchestration/journal.jsonl','EXECUTION_FREEZE.json','ENGINE_IO_REPAIR.json','code/SOURCE_DEPLOYMENT.json']:
 base=r/relative
 for p in ([base] if base.is_file() else sorted(base.rglob('*'))):
  if p.is_dir():continue
  assert p.is_file() and not p.is_symlink();b=p.read_bytes();n=str(p.relative_to(r));dst=snap/n;dst.parent.mkdir(parents=True,exist_ok=True)
  with dst.open('xb') as f:f.write(b)
  assert sha(dst.read_bytes())==sha(b);files[n]=dict(bytes=len(b),sha256=sha(b))
queue=run(['squeue','--noheader','--user','varun024','--format=%i|%j|%T|%q|%b|%R'])
ids={line.split('|')[0] for line in queue['stdout'].splitlines()};assert queue['returncode']==0 and {'196225','196227'}.issubset(ids),queue
checks={}
for job in ['196225','196227']:
 x=run(['scontrol','show','job',job,'--oneliner']);assert x['returncode']==0,x
 assert 'UserId=varun024(' in x['stdout'] and 'JobState=RUNNING' in x['stdout'],x
 if job=='196227':
  assert 'JobName=srf1-9ef274c9fc6a-ENGINE_attempt0003 ' in x['stdout'] and 'QOS=soujanya-poria-startfund-2026-03 ' in x['stdout'] and 'Command='+str(r/'orchestration/attempts/ENGINE_attempt0003.sh')+' ' in x['stdout'],x
 else:assert 'JobName=srf11-controller-io ' in x['stdout'] and 'Command='+str(r/'code/scripts/sr_f1/controller.sbatch') in x['stdout'],x
 checks[job]=x
save(inc/'PRE_MAINTENANCE_SNAPSHOT.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_hashes=files,dynamic_files_may_span_observation_times=True,queue=queue,identity_checks=checks,reason='USER_AUTHORIZED_MULTIGPU_PERFORMANCE_ACCELERATION',new_failure_detected=False))
args=['scancel','196225','196227']
save(inc/'MAINTENANCE_CANCEL_INTENT.json',dict(command=args,reason='USER_AUTHORIZED_MULTIGPU_PERFORMANCE_ACCELERATION',authorized_user_message='能用多张卡提速一定要用多张卡，最多五张，我需要速度提升',other_jobs_modified=0))
x=run(args);save(inc/'MAINTENANCE_CANCEL_RESULT.json',x);assert x['returncode']==0,x
print(json.dumps(dict(status='ONLY_VERIFIED_PROJECT_JOBS_MAINTENANCE_CANCEL_REQUESTED',jobs=['196225','196227'],snapshot_files=len(files),terminal_status_not_yet_confirmed=True)))
