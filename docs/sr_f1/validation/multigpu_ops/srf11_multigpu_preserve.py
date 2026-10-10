import datetime,hashlib,json,os,pathlib,shutil,subprocess
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');inc=r/'technical_incidents/multigpu_20261010'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
assert r.resolve()==r and not (inc/'PRESERVATION.json').exists()
assert sha(r/'EXECUTION_FREEZE.json')=='c7cf051d3aa6fb45889324e64b08d31a008137668aff2382283e4e19f99ff2b8'
assert sha(r/'ENGINE_IO_REPAIR.json')=='ebd17dda5860c292c2529895ab6d5816c04cee8e32c4c3f86230033438b82df0'
account=run(['sacct','-j','196225,196227','--format=JobID,State,ExitCode,ElapsedRaw,Start,End,AllocTRES','-n','-P'])
rows={p[0]:p[1:] for p in (s.split('|') for s in account['stdout'].splitlines())}
assert account['returncode']==0 and all(rows[j][0].startswith('CANCELLED') for j in ['196225','196227']),account
queue=run(['squeue','--noheader','--user','varun024','--format=%i|%j|%T|%q|%b|%R'])
assert queue['returncode']==0 and not any(line.split('|')[0] in {'196225','196227'} for line in queue['stdout'].splitlines()),queue
state=read(r/'orchestration/STATE.json');assert state['test_sealed'] is True
assert len(state['tasks']['ENGINE']['attempts'])==4 and state['tasks']['ENGINE']['attempts'][-1]['job_id']=='196227'
assert not any(t.get('attempts') for k,t in state['tasks'].items() if k not in ['COMMON_START','ENGINE'])
checkpoint=r/'engineering/engine/natural/continuous/checkpoints/LATEST.json'
assert read(checkpoint)['step']==0 and not list((r/'engineering/engine/natural/continuous/steps').glob('*.json'))
inc.mkdir(parents=True,exist_ok=True)
paths=['engineering/engine','engineering/ENGINE_CONFIG.json','manifests/ENGINE_SCHEDULE.json','accounting/ENGINE.jsonl','orchestration','EXECUTION_FREEZE.json','AMENDMENT.json','QOS_SCOPE_REPAIR.json','ENGINE_MEMORY_REPAIR.json','ENGINE_STORAGE_REPAIR.json','ENGINE_IO_REPAIR.json','COMMON_START.json','COMMON_ZERO_LORA.json','FORMAT_AND_BRIDGE_RECEIPT.json','technical_incidents/engine_memory_20261010/ZERO_UPDATE_REUSE_ACTIVATED.json','technical_incidents/engine_memory_20261010/ZERO_UPDATE_REUSE_PROCESS.json']
files={}
for relative in paths:
 base=r/relative
 for p in ([base] if base.is_file() else sorted(base.rglob('*'))):
  if p.is_dir():continue
  assert p.is_file() and not p.is_symlink(),p
  n=str(p.relative_to(r));dst=inc/'evidence'/n;dst.parent.mkdir(parents=True,exist_ok=True)
  assert not dst.exists();before=sha(p);shutil.copyfile(p,dst)
  assert sha(dst)==sha(p)==before,n;files[n]=dict(bytes=p.stat().st_size,sha256=before)
source=read(r/'code/SOURCE_DEPLOYMENT.json');assert source['source_commit']=='b48a638128bfeb6fc41201341df4902b3f2e660a'
for n,h in source['source_file_hashes'].items():assert sha(r/'code'/n)==h,n
shutil.copytree(r/'code',r/'code_before_engine_multigpu_20261010')
spill=pathlib.Path('/projects/_hdd/varunhdd/louis-ssvc/sr_f11_20261010_activation_offload')
save(inc/'TERMINAL_JOBS.json',dict(accounting=account,queue=queue,controller_terminal=True,gpu_terminal=True,controller_job_id='196225',gpu_job_id='196227',controller_terminal_state='CANCELLED',gpu_terminal_state='CANCELLED',controller_exit_code=rows['196225'][1],gpu_exit_code=rows['196227'][1],maintenance_reason='USER_AUTHORIZED_MULTIGPU_PERFORMANCE_ACCELERATION',unrelated_jobs_modified=0))
save(inc/'PRESERVATION.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='BYTE_VERIFIED_PRESERVED',files=len(files),bytes=sum(x['bytes'] for x in files.values()),artifact_hashes=files,source_commit=source['source_commit'],source_tree_sha256=source['source_tree_sha256'],checkpoint_step=0,raw_rollouts=len(list((r/'engineering/engine/natural/continuous/rollouts').glob('*.json'))),science_attempts=0,unrelated_jobs_modified=0,derived_scratch_metadata=[dict(name=p.name,bytes=p.stat().st_size,inode=p.stat().st_ino,mtime_ns=p.stat().st_mtime_ns) for p in spill.glob('*.bin')],derived_scratch_not_model_evidence=True))
print(json.dumps(dict(status='MULTIGPU_MAINTENANCE_EVIDENCE_PRESERVED',files=len(files),bytes=sum(x['bytes'] for x in files.values()),incident=str(inc),checkpoint_step=0,source_commit=source['source_commit'])))
