import datetime,json,math,pathlib,sys
import torch
r=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010')
inc=r/'technical_incidents/multigpu_20261010';torch.set_num_threads(1)
sys.path.insert(0,str(r/'code/src'))
from mm_core.training import state_hash
from mm_core.vl_runtime import hash_json
from sr_f1.contract import PLAN_ID,digest,file_hash
from sr_f1.engine import _read_state,engine_schedule,engine_run
from sr_f1.data import load_inputs,load_tasks
from sr_f1.json_protocol import AMENDMENT_ID,validate_record
from sr_f1.training import CHECKPOINT_FIELDS
def read(p):return json.loads(p.read_text())
d=inc/'evidence/engineering/engine/natural/continuous'
manifest=read(inc/'PRESERVATION.json')
for n,e in manifest['artifact_hashes'].items():assert file_hash(inc/'evidence'/n)==e['sha256'],n
state=_read_state(d,0);latest=read(d/'checkpoints/LATEST.json');commit=read(d/'checkpoints/commit-00.json')
assert latest==commit and latest['step']==0 and state['committed_logical_step']==0
assert set(state)==CHECKPOINT_FIELDS and not state['optimizer']['state']
assert not list((d/'steps').glob('*.json')) and not list((d/'update_attempts').glob('*.json'))
common=read(r/'COMMON_START.json');plan=read(r/'config/SR_F1_1.json')
assert state_hash(state['parameters'])==state_hash(state['reference'])==common['trainable_state_hash']
assert state['plan_id']==PLAN_ID and state['reference_hash']==state_hash(state['reference'])
identity=state['run_identity'];assert all(identity[k]==v for k,v in engine_run('natural').items())
assert identity['training_identity']['learning_rate']==1e-4
assert identity['training_identity']['protocol_amendment']==plan['protocol_amendment']
assert all(g['lr']==g['initial_lr']==1e-4 for g in state['optimizer']['param_groups'])
assert state['scheduler']['base_lrs']==state['scheduler']['_last_lr']==[1e-4]*len(state['optimizer']['param_groups'])
assert (r/'manifests/ENGINE_SCHEDULE.json').is_file();schedule=engine_schedule(r)
assert state['input_stream_hash']==digest(schedule)
run_manifest=read(d/'RUN_MANIFEST.json');assert run_manifest['run_identity']==identity and run_manifest['sampling_hash']==state['sampling_hash'] and run_manifest['stream_hash']==state['input_stream_hash']
account=[json.loads(s) for s in (inc/'evidence/accounting/ENGINE.jsonl').read_text().splitlines()]
updates=sum(x['count'] for x in account if x['kind']=='physical_optimizer_updates');assert updates==0
inputs,tasks=load_inputs(r),load_tasks(r);slots={s['slot']:s for s in schedule if s['step']==1};seen=set();raw_hashes={}
for p in sorted((d/'rollouts').glob('*.json')):
 x=read(p);slot=x['slot'];i=x['sample_index'];s=slots[slot];q=s['qid'];row=inputs[q]
 expected=dict(request_id=digest([PLAN_ID,identity['run_id'],1,slot,i,common['trainable_state_hash']]),run_id=identity['run_id'],logical_step=1,slot=slot,sample_index=i,qid=q,root_id=tasks[q]['root_id'],seed=s['rollout_seeds'][i],policy_hash=common['trainable_state_hash'],sampling_hash=state['sampling_hash'],image_sha256=file_hash(r/row['image_file']),input_hash=digest({k:row[k] for k in ['qid','image_file','text']}))
 assert all(x.get(k)==v for k,v in expected.items()),p.name
 assert x['record_hash']==digest({k:v for k,v in x.items() if k!='record_hash'})
 assert x['generation_status']=='COMPLETE' and not x['technical_validation_errors']
 assert x['old_logprobs']==x['sampler_logprobs'] and all(math.isfinite(v) for v in x['old_logprobs'])
 assert validate_record(x,AMENDMENT_ID) is True
 assert (slot,i) not in seen;seen.add((slot,i));raw_hashes[p.name]=file_hash(p)
assert seen=={(s,i) for s in range(16) for i in range(8)}
review=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_ZERO_UPDATE_FULL_STATE',committed_logical_step=0,physical_optimizer_updates=updates,optimizer_empty=True,raw_rollouts=len(seen),raw_record_hashes_verified=True,raw_file_hashes=raw_hashes,learning_rate=1e-4,checkpoint_state_hash=latest['state_hash'],checkpoint_file_sha256=latest['sha256'],checkpoint_field_hashes=latest['field_hashes'],run_identity_hash=digest(identity),sampling_hash=state['sampling_hash'],source_commit=read(r/'code/SOURCE_DEPLOYMENT.json')['source_commit'],model_calls=0,cuda_context_initialized=torch.cuda.is_initialized())
assert review['cuda_context_initialized'] is False
with (inc/'CPU_STATE_REVIEW.json').open('x') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in review.items() if k not in ['raw_file_hashes','checkpoint_field_hashes']}))
