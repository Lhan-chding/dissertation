import pathlib,sys,json,os,subprocess
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010');inc=r/'technical_incidents/multigpu_20261010'
sys.path.insert(0,str(r/'code/src'))
from sr_f1.contract import file_hash
from sr_f1.orchestration import digest,Scheduler,process_lease

def read(p):return json.loads(p.read_text())
def save(name,value):
 with (inc/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
manifest=read(inc/'PRESERVATION.json')
assert read(inc/'CPU_STATE_REVIEW.json')['status']=='PASS_ZERO_UPDATE_FULL_STATE'
a=run(['sacct','-j','196225,196227','--format=JobID,State,ExitCode,ElapsedRaw,Start,End','-n','-P'])
rows={x[0]:x[1:] for x in (line.split('|') for line in a['stdout'].splitlines())}
assert a['returncode']==0 and all(rows[j][0].startswith('CANCELLED') for j in ['196225','196227']),a
q=run(['squeue','--noheader','--user','varun024','--format=%i|%j|%T|%q'])
assert q['returncode']==0 and not any('srf1-9ef274c9fc6a' in line or 'srf11-controller' in line for line in q['stdout'].splitlines()),q
original=read(r/'ENGINE_MEMORY_REPAIR.json')['zero_update_recovery']['artifact_hashes']
old_evidence=r/'technical_incidents/io_exit120_20261010/evidence'
assert all(file_hash(old_evidence/n)==h for n,h in original.items())
markers=['technical_incidents/engine_memory_20261010/ZERO_UPDATE_REUSE_ACTIVATED.json','technical_incidents/engine_memory_20261010/ZERO_UPDATE_REUSE_PROCESS.json']
assert all(file_hash(r/n)==manifest['artifact_hashes'][n]['sha256'] for n in markers)
oldstate=read(inc/'evidence/orchestration/STATE.json')
assert oldstate['test_sealed'] and all(not x['attempts'] for k,x in oldstate['tasks'].items() if k not in ['ENGINE','COMMON_START'])
previous=None
for index,line in enumerate((inc/'evidence/orchestration/journal.jsonl').read_bytes().splitlines(keepends=True),1):
 assert line.endswith(b'\n');row=json.loads(line);h=row.pop('sha256');assert digest(row)==h and row['previous']==previous and row['sequence']==index;previous=h
assert row['state']==oldstate
with process_lease(r.parent/'.sr_f1_project_controller.lock'),process_lease(r/'orchestration/controller.lock'):
 scheduler=Scheduler(r/'config/SR_F1_1.json',r,code_root=r/'code',python=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'))
 scheduler._load();assert len(scheduler.state['tasks']['ENGINE']['attempts'])==4
 scheduler._observe('ENGINE');scheduler._phase();scheduler._save('MULTIGPU_MAINTENANCE_RECONCILED_NO_SUBMISSION')
 assert scheduler.state['tasks']['ENGINE']['attempts'][-1]['accounting']['terminal_state']=='CANCELLED'
 save('STATE_RECONCILED.json',scheduler.state)
review=dict(status='PASS_INTERRUPTED_ENGINE_RESTART',science_attempts=0,test_sealed=True,once_activation_consumed=True,once_process_consumed=True,journal_integrity=True,journal_lines=index,original_artifact_hashes_unchanged=True,full_engine_restart=True,no_partial_gradient_reuse=True,no_old_once_reuse=True,old_original_artifact_hashes=original,old_original_artifact_location=str(old_evidence),consumed_marker_hashes={n:file_hash(r/n) for n in markers},source_commit=read(r/'code/SOURCE_DEPLOYMENT.json')['source_commit'],gpu_calls=0,optimizer_updates=0,maintenance_reason='USER_AUTHORIZED_MULTIGPU_PERFORMANCE_ACCELERATION')
save('RECOVERY_REVIEW.json',review)
print(json.dumps(review))
