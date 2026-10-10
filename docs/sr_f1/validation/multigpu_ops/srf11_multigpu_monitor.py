import collections,datetime,hashlib,json,pathlib,subprocess
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');inc=r/'technical_incidents/engine_memory_20261010'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
x=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_root=str(r))
s=read(r/'orchestration/STATE.json');x['state']={k:s.get(k) for k in ['phase','test_sealed','technical_blockers','engine_memory_repair_resume','engine_storage_repair_resume','engine_io_repair_resume']}
x['tasks']={k:{z:v.get(z) for z in ['status','blocker','attempts']} for k,v in s['tasks'].items() if v['status']!='WAITING'}
ids=[a['job_id'] for t in s['tasks'].values() for a in t['attempts'] if a.get('job_id')]
for incident in [inc,r/'technical_incidents/host_memory_20261010',r/'technical_incidents/io_exit120_20261010']:
 for n in ['VERIFY_SUBMISSION','CONTROLLER_SUBMISSION']:
  p=incident/(n+'.json')
  if p.exists():ids.append(read(p)['stdout'].strip().split(';')[0])
x['accounting']=run(['sacct','-j',','.join(sorted(set(ids+['196046','196123']))),'--format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS','-n','-P'])
x['queue']=run(['squeue','--noheader','--user','varun024','--format=%i|%j|%T|%q|%R|%M|%m|%b'])
for n in ['CPU_STATE_REVIEW.json','SOURCE_ACTIVATED.json','VALIDATION_COMPLETE.json','ZERO_UPDATE_REUSE_ACTIVATED.json','ZERO_UPDATE_REUSE_PROCESS.json']:
 p=inc/n
 if p.exists():x[n]=dict(sha256=sha(p),data={k:v for k,v in read(p).items() if k not in ['raw_file_hashes','checkpoint_field_hashes','accounting','queue_before_restart']})
for n in ['EXECUTION_FREEZE.json','AMENDMENT.json','QOS_SCOPE_REPAIR.json','ENGINE_MEMORY_REPAIR.json','ENGINE_STORAGE_REPAIR.json','ENGINE_IO_REPAIR.json','ENGINE_PROBABILITY_GRADIENT_RESUME.json','FORMAT_AND_BRIDGE_RECEIPT.json']:
 p=r/n
 if p.exists():x[n]=dict(sha256=sha(p),status=read(p).get('status'))
x['steps']={str(p.relative_to(r)):read(p)['step'] for p in (r/'engineering/engine').rglob('LATEST.json')}
logs=list((r/'technical_incidents/io_exit120_20261010').glob('*.log'))+list((r/'technical_incidents/host_memory_20261010').glob('*.log'))+list(inc.glob('controller_*.log'))+list(inc.glob('verify_*.log'))+list((r/'engineering/engine').rglob('*.log'))+list((r/'orchestration/attempts').glob('ENGINE*.log'))
x['logs']={str(p.relative_to(r)):dict(bytes=p.stat().st_size,tail=p.read_text(errors='replace')[-2200:]) for p in sorted(logs)}
ledger=r/'accounting/ENGINE.jsonl';events=[]
if ledger.exists():events=[json.loads(line) for line in ledger.read_text().splitlines()]
x['accounting_event_counts']=dict(collections.Counter(z['kind'] for z in events))
x['physical_optimizer_updates']=sum(z['count'] for z in events if z['kind']=='physical_optimizer_updates')
x['recent_events']=events[-8:]
repair=read(r/'ENGINE_MEMORY_REPAIR.json');raw={n:h for n,h in repair['zero_update_recovery']['artifact_hashes'].items() if '/rollouts/' in n}
x['original_raw_128_hashes_unchanged']=len(raw)==128 and all((r/n).exists() and sha(r/n)==h for n,h in raw.items())
x['raw_count_by_track']={str(d.relative_to(r)):len(list(d.glob('*.json'))) for d in (r/'engineering/engine').rglob('rollouts')}
x['science_attempts']=sum(len(v['attempts']) for k,v in s['tasks'].items() if k not in ['COMMON_START','ENGINE'])
x['scratch_files']=[dict(name=p.name,bytes=p.stat().st_size) for p in (r/'technical_scratch/activation_offload').glob('*.bin')]
x['host_memory_observations']={str(p.relative_to(r)):read(p) for p in sorted((r/'orchestration/observations/host_memory').rglob('*.json'))[-3:]}


io=r/'technical_incidents/io_exit120_20261010'
x['current_track_original_raw_hashes_match']=x.pop('original_raw_128_hashes_unchanged')
x['original_raw_128_preserved_hashes_unchanged']=len(raw)==128 and all(sha(io/'evidence'/n)==h for n,h in raw.items())
x['interrupted_tracks']={str(d.relative_to(r)):len(list(d.glob('*.json'))) for d in (r/'engineering/engine_interrupted').rglob('rollouts')}
x['event_counts_by_attempt']={a:dict(collections.Counter(v['kind'] for v in events if v.get('attempt_id')==a)) for a in sorted({str(v.get('attempt_id')) for v in events})}
x['io_recovery']={n:read(io/n) for n in ['SOURCE_ACTIVATED.json','VALIDATION_COMPLETE.json','CONTROLLER_SUBMISSION.json','EXTERNAL_IO_PROBE.json'] if (io/n).exists()}
x['actual_source_identity']=read(r/'code/SOURCE_DEPLOYMENT.json')
x['actual_source_files_verified']=all(sha(r/'code'/n)==h for n,h in x['actual_source_identity']['source_file_hashes'].items())
spill=pathlib.Path('/projects/_hdd/varunhdd/louis-ssvc/sr_f11_20261010_activation_offload')
x['external_scratch_files']=[dict(name=p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in spill.glob('*.bin')]
x['external_io_failures']={p.name:read(p) for p in sorted(spill.glob('failure-*.json'))[-5:]}


mg=r/'technical_incidents/multigpu_20261010'
x['state']['engine_multigpu_repair_resume']=s.get('engine_multigpu_repair_resume')
x['multigpu_receipts']={}
for p in sorted(mg.glob('*.json')):
 if p.name.startswith(('inventory','OPERATION_SCRIPT','PRESERVATION','PRE_MAINTENANCE')):continue
 value=read(p)
 x['multigpu_receipts'][p.name]=dict(sha256=sha(p),data=value)
 if p.name.endswith('SUBMISSION.json') and value.get('returncode')==0:
  job=value.get('stdout','').strip().split(';')[0]
  if job.isdigit():ids.append(job)
p=r/'ENGINE_MULTIGPU_REPAIR.json'
if p.exists():x[p.name]=dict(sha256=sha(p),data=read(p))
x['multigpu_accounting']=run(['sacct','-j',','.join(sorted(set(ids))),'--format=JobID,State,ExitCode,Elapsed,AllocTRES','-n','-P'])
for p in sorted(mg.glob('*.log')):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-4000));tail=f.read().decode('utf-8',errors='replace')
 x['logs'][str(p.relative_to(r))]=dict(bytes=p.stat().st_size,tail=tail)
p=mg/'PRESERVATION.json'
if p.exists():
 preserved=read(p);x['multigpu_evidence']={k:preserved.get(k) for k in ['files','bytes','raw_rollouts','checkpoint_step','science_attempts']}
 x['multigpu_evidence']['all_hashes_verified']=all(sha(mg/'evidence'/n)==v['sha256'] for n,v in preserved['artifact_hashes'].items())
active={line.split('|')[0] for line in x['queue']['stdout'].splitlines()}
current=[a['job_id'] for t in s['tasks'].values() for a in t['attempts'] if a.get('job_id') in active]
x['active_project_job_details']={j:run(['scontrol','show','job',j,'--oneliner']) for j in current}
print(json.dumps(x,indent=2))
