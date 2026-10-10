import os,json,pathlib,subprocess,hashlib,datetime
b=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc'); new=b/'sr_f12_20261010'; audit=new/'cleanup'
roots=[b/'sr_f1_20261009',b/'sr_f11_20261010']
hdd=pathlib.Path('/projects/_hdd/varunhdd/louis-ssvc/sr_f11_20261010_activation_offload')
assert (audit/'INPUTS_RETAINED.json').exists()
def save(n,o):
 with (audit/n).open('x') as f:json.dump(o,f,indent=2)
queue=subprocess.check_output(['squeue','-u','varun024','-h','-o','%i'],text=True)
for j in queue.split():
 s=subprocess.check_output(['scontrol','show','job','-o',j],text=True)
 assert not any(str(r)+'/' in s for r in roots)
terminal=subprocess.check_output(['sacct','-j','196798,196802','-X','-n','-P','-o','JobID,State,ExitCode'],text=True)
assert all(any(s.startswith(j+'|CANCELLED') for s in terminal.splitlines()) for j in ['196798','196802'])
selected=[]
for r in roots:
 assert r.resolve()==r and r.stat().st_uid==os.geteuid()
 for p in r.rglob('*'):
  assert not p.is_symlink(),str(p)
  if not p.is_file():continue
  rel=p.relative_to(r);parts=rel.parts
  # Retain ALL code, inputs, config, deployment and scheduler/audit records.
  if any(x.startswith('code') for x in parts) or parts[0] in {'config','amendment','deployment','manifests','images','external','private_assets','orchestration','accounting','validation','logs','qa'}:continue
  if p.suffix in {'.py','.sh','.sbatch','.log','.gz','.zip','.txt','.md'}:continue
  reason=None
  if parts[0] in {'states','recovery_checkpoints'}:reason='old_model_optimizer_checkpoint'
  elif p.suffix in {'.pt','.pth','.safetensors','.bin'}:reason='old_model_optimizer_or_activation_binary'
  elif p.suffix in {'.json','.jsonl'}:
   if any(x in {'rollouts','raw','steps','adapters','checkpoints','before','after','confirm','confirmation'} for x in parts):reason='old_raw_responses_or_training_outputs'
   else:
    try:
     v=json.loads(p.read_text())
     if isinstance(v,dict) and any(k in v for k in ['generated_token_ids','generated_tokens','old_logprobs','sampler_logprobs']):reason='old_raw_response_schema'
    except (ValueError,UnicodeError):pass
  if reason:selected.append((p,reason))
assert hdd.resolve()==hdd and hdd.stat().st_uid==os.geteuid()
for p in hdd.rglob('*'):
 assert not p.is_symlink()
 if p.is_file() and p.suffix=='.bin':selected.append((p,'abandoned_old_activation_scratch'))
manifest=[dict(path=str(p),bytes=p.stat().st_size,allocated_bytes=p.stat().st_blocks*512,reason=reason) for p,reason in selected]
assert len({x['path'] for x in manifest})==len(manifest)
save('RESULT_DELETE_ALLOWLIST.json',{'files':manifest,'authorization':'Delete old experimental results and unused scratch only; retain code configs and operation audit','old_results_backed_up':False})
for p,_ in selected:p.unlink()
assert all(not p.exists() for p,_ in selected)
result=dict(status='DELETED_OLD_RESULT_FILES_AND_ACTIVATION_SCRATCH',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=len(manifest),logical_bytes=sum(x['bytes'] for x in manifest),allocated_bytes=sum(x['allocated_bytes'] for x in manifest),old_results_backed_up=False,code_configs_inputs_and_audits_retained=True)
save('RESULT_DELETE_COMPLETE.json',result)
print(json.dumps(result))
