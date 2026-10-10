import datetime,hashlib,json,os,pathlib,shutil,subprocess
base=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc')
old=[base/'sr_f11_20261010',base/'sr_f1_20261009',pathlib.Path('/projects/_hdd/varunhdd/louis-ssvc/sr_f11_20261010_activation_offload')]
new=base/'sr_f12_20261010'; audit=new/'cleanup'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 with (audit/name).open('x') as f:json.dump(obj,f,indent=2)
def cmd(args):return subprocess.check_output(args,text=True)
assert base.resolve()==base and new.resolve()==new
queue=cmd(['squeue','-u','varun024','-h','-o','%i'])
observed=[]
for j in queue.split():
 s=cmd(['scontrol','show','job','-o',j]);observed.append(s)
 assert not any(str(r)+'/' in s for r in old),s
terminal=cmd(['sacct','-j','196798,196802','-X','-n','-P','-o','JobID,State,ExitCode'])
for j in ['196798','196802']:
 assert any(x.startswith(j+'|CANCELLED') for x in terminal.splitlines())
save('OLD_TERMINAL.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accounting=terminal,active_unrelated_jobs=observed))
source=old[0]
allow=['images','external','manifests/MODEL_INPUTS.jsonl','manifests/TASKS_GOLD_AUDIT_ONLY.jsonl','manifests/DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl','manifests/ROOTS_AUDIT_ONLY.jsonl','manifests/TRAIN_FIT_QIDS.json','manifests/ENGINE_SCHEDULE.json','manifests/schedules','MODEL_ENVIRONMENT_IDENTITY.json']
files=[]
for rel in allow:
 p=source/rel
 files.extend(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else files.append(p)
assert all(not p.is_symlink() for p in source.rglob('*'))
manifest=[]
for p in files:
 rel=p.relative_to(source);q=new/rel
 assert not q.exists(),str(q)
 q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 h=sha(p);assert sha(q)==h
 manifest.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=h))
(new/'config').mkdir(exist_ok=True)
shutil.copy2(source/'config/SR_F1_1.json',new/'config/PARENT_INPUT_CONFIG.json')
save('INPUTS_RETAINED.json',dict(kind='INPUT_ASSETS_AND_MODEL_IDENTITY_ONLY_NOT_OLD_RESULTS',files=manifest,bytes=sum(x['bytes'] for x in manifest)))

print("INPUT_COPY_VERIFIED",len(manifest))
