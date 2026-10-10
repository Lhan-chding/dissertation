import datetime,fcntl,hashlib,json,os,pathlib,re,subprocess
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010')
inc=r/'technical_incidents/multigpu_20261010';code=r/'code'
py='/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (inc/n).open('x') as f:
        json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=45)
    return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,time=now())
assert r.resolve()==r and not (r/'STOP').exists()
assert not (inc/'CONTROLLER_SUBMISSION_INTENT.json').exists()
verify_job=read(inc/'VERIFY_SUBMISSION.json')['stdout'].strip().split(';')[0]

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
    result=run(['sacct','-j','196225,196227,'+verify_job,'--format=JobID,State,ExitCode','-n','-P'])
    rows={z[0]:z[1:3] for z in (s.split('|') for s in result['stdout'].splitlines())}
    assert result['returncode']==0 and rows['196225'][0].startswith('CANCELLED') and rows['196227'][0].startswith('CANCELLED') and rows[verify_job]==['COMPLETED','0:0'],result
    tests=(inc/'CANDIDATE_SERVER_TESTS_V2.log').read_text();match=re.search(r'(\d+) passed, (?:\d+ skipped, )?(\d+) subtests passed',tests)
    assert match and int(match[1])>=742 and int(match[2])>=6,tests[-4000:]
    assert read(inc/'PREFLIGHT.log')['status']=='PASS'
    v=read(inc/'MULTIGPU_VERIFY.log');repair=read(r/'ENGINE_MULTIGPU_REPAIR.json')
    assert v['repair_sha256']==sha(r/'ENGINE_MULTIGPU_REPAIR.json') and v['source_commit']==read(code/'SOURCE_DEPLOYMENT.json')['source_commit']==repair['source_commit']
    assert v['minimum_gpu_host_memory_gb']==320 and v['gpu_count']==4 and v['gpu_worker_constraint']=='highmem' and v['original_execution_freeze_sha256']==sha(r/'EXECUTION_FREEZE.json')=='c7cf051d3aa6fb45889324e64b08d31a008137668aff2382283e4e19f99ff2b8'
    assert read(r/'orchestration/REGISTRATION.json')==read(inc/'evidence/orchestration/REGISTRATION.json')
    state=read(r/'orchestration/STATE.json');authorization=read(inc/'RESUME_AUTHORIZATION.log')
    assert state['engine_multigpu_repair_resume']==authorization and authorization['consumed_by_attempt_id'] is None
    assert state['test_sealed'] is True and state['tasks']['ENGINE']['status']=='RETRYABLE'
    assert len(state['tasks']['ENGINE']['attempts'])==4
    assert all(t['status']=='WAITING' and not t['attempts'] for k,t in state['tasks'].items() if k not in {'ENGINE','COMMON_START'})
    assert (r/'technical_incidents/engine_memory_20261010/ZERO_UPDATE_REUSE_ACTIVATED.json').exists()
    assert read(inc/'GPU_PROBE.json')['status']=='PASS'
    for n,h in read(inc/'RECOVERY_REVIEW.json')['consumed_marker_hashes'].items():assert sha(r/n)==h,n
    queue=run(['squeue','--noheader','--user','varun024','--format=%i|%j|%T|%q'])
    assert queue['returncode']==0 and not any('srf1-9ef274c9fc6a' in line or 'srf11-controller' in line for line in queue['stdout'].splitlines()),queue
    save('VALIDATION_COMPLETE.json',dict(time=now(),status='PASS',server_tests=int(match[1]),server_subtests=int(match[2]),accounting=result,queue_before_restart=queue,artifact_hashes={str(p.relative_to(r)):sha(p) for p in [inc/'CANDIDATE_SERVER_TESTS_V2.log',inc/'PREFLIGHT.log',inc/'MULTIGPU_VERIFY.log',inc/'RESUME_AUTHORIZATION.log',inc/'GPU_PROBE.json',inc/'RECOVERY_REVIEW.json',inc/'CPU_STATE_REVIEW.json',inc/'TERMINAL_JOBS.json',inc/'PRESERVATION.json',r/'EXECUTION_FREEZE.json',r/'AMENDMENT.json',r/'QOS_SCOPE_REPAIR.json',r/'ENGINE_MULTIGPU_REPAIR.json']},registration_unchanged=True,old_raw_rollouts_preserved=128,consumed_gpu_reuse_not_reset=True,optimizer_updates_before_restart=0,engine_gpu_count=4,real_9b_engine_qualified=False,unrelated_jobs_modified=0))
    args=['sbatch','--hold','--parsable','--account=rose','--qos=override-limits-but-killable','--partition=cluster02','--job-name=srf11-controller-multigpu','--output='+str(inc/'controller_%j.log'),str(code/'scripts/sr_f1/controller.sbatch'),py,str(code),str(r/'config/SR_F1_1.json'),str(r)]
    save('CONTROLLER_SUBMISSION_INTENT.json',dict(command=args,time=now()))
    result=run(args);save('CONTROLLER_SUBMISSION.json',result);assert result['returncode']==0,result
    job=result['stdout'].strip().split(';')[0];assert re.fullmatch('[0-9]+',job)
    result=run(['scontrol','release',job]);save('CONTROLLER_RELEASE.json',result);assert result['returncode']==0,result
print(json.dumps(dict(status='CPU_AND_SYNTHETIC_CUDA_VERIFIED_MULTIGPU_CONTROLLER_SUBMITTED',controller_job=job,source_commit=v['source_commit'],freeze_sha256=v['original_execution_freeze_sha256'],unrelated_jobs_modified=0)))
