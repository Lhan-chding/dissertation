from pathlib import Path
import json,sys,statistics,datetime,hashlib,subprocess
from collections import Counter,defaultdict
r=Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010');sys.path.insert(0,str(r/'code_repair_r0001b/src'))
from sr_f1.contract import digest
from sr_f1.data import load_tasks
from sr_f12.runner import scored
from sr_f12.protocol import iter_model_slots
load=lambda p:json.loads(p.read_text())
tasks=load_tasks(r);fit=load(r/'manifests/TRAIN_FIT_QIDS.json')
metrics=['A','E','p_read','P','J']
def aggregate(rows):
 q=defaultdict(list)
 for x in rows:q[(x['family'],x['root_id'],x['qid'])].append(x)
 roots=defaultdict(list)
 for (f,root,qid),rr in q.items():roots[(f,root)].append({m:statistics.mean(x['score'][m] for x in rr) for m in metrics})
 fs=defaultdict(list)
 for (f,root),rr in roots.items():fs[f].append({m:statistics.mean(x[m] for x in rr) for m in metrics})
 fm={f:{m:statistics.mean(x[m] for x in rr) for m in metrics} for f,rr in fs.items()}
 return dict(n=len(rows),questions=len(q),roots=len(roots),family_means=fm,balanced_means={m:statistics.mean(x[m] for x in fm.values()) for m in metrics},draw_means={m:statistics.mean(x['score'][m] for x in rows) for m in metrics},format_failure_count=sum(x['format_failure'] for x in rows),reasons=dict(Counter(x['score'].get('reason') for x in rows)),rows=rows)
o={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'seed71002 existing TRAIN_FIT only; no TEST accessed; descriptive single seed training-fit analysis','arms':{}}
for arm in ['A','J','GATE','DEC']:
 model=f'SRF1_2_{arm}_s71002';d=r/'runs'/model;panels={}
 for step in [32,64,96]:
  p=d/'train_fit'/f'step-{step:02d}';slots=list(iter_model_slots(tasks,fit,model,stage='train_fit',step=step));rr=[];raw=[]
  for i in range(32):
   g=load(p/f'{i:02d}.json');assert g['slots']==slots[i*4:i*4+4];assert digest(g['records'])==g['sha256'];assert len(g['records'])==4
   for slot,row in zip(g['slots'],g['records']):
    s=scored(row,tasks[slot['qid']]);assert s==row['score'];assert s['J']==s['A']*s['P']
    rr.append({k:slot[k] for k in ['qid','root_id','family','draw']}|{'score':s,'format_failure':bool(row.get('format_protocol_error')) or s.get('reason')!='scored'})
   raw.extend(g['records'])
  assert load(p/'COMPLETE.json')==dict(status='COMPLETE',count=128,records_sha256=digest(raw))
  panels[step]=aggregate(rr)
 ss=[load(d/'steps'/f'{i:02d}.json') for i in range(1,97)]
 def train_stats(ss):
  return {k:statistics.mean(x[k] for x in ss) for k in ['format_failure_rate','kl','step_seconds','sampling_seconds','training_seconds','total_gradient_norm']}|{'nonzero_pg_steps':sum(x['policy']!=0 for x in ss)}
 o['arms'][arm]={'fit':panels,'training':{'all96':train_stats(ss),'first16':train_stats(ss[:16]),'last16':train_stats(ss[-16:]),'zero_advantage_steps':sum(all(v==0 for row in x['reward_advantage_audit']['final_advantage'] for v in row) for x in ss),'last16_zero_advantage_steps':sum(all(v==0 for row in x['reward_advantage_audit']['final_advantage'] for v in row) for x in ss[-16:]),'last':{k:ss[-1][k] for k in ['kl','finite','format_failure_rate','sampler_train_mean_absolute_difference','tis_truncated_fraction']}},'complete_sha256':hashlib.sha256((d/'COMPLETE.json').read_bytes()).hexdigest()}
print(json.dumps(o,ensure_ascii=False,indent=2))
