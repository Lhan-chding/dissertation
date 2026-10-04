#!/usr/bin/env python3
"""Read existing committed train records only; no model imports or fitting.
Run: python3 support_cost.py --raw /path/to/extracted/raw --output /path/to/output
Evaluation correspondence is added from opportunity_by_stage_group.csv if available.
"""
import argparse,csv,json,collections,pathlib,subprocess,hashlib
p=argparse.ArgumentParser();p.add_argument('--raw',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,default=pathlib.Path(__file__).parent);a=p.parse_args();R=a.raw;O=a.output;O.mkdir(parents=True,exist_ok=True)
inputs={};missing=[]
def track(p,kind):
 inputs[str(p.resolve())]=kind
 return p
def read(p,kind='json'):return json.loads(track(p,kind).read_text())
def local(s):return R/('campaign/'+s.split('/campaign/',1)[1])
def write(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fields,lineterminator="\n");w.writeheader();w.writerows(rows)
prepared=read(R/'inputs/prepared_first_four_ONLY.json','train prompt metadata')
meta={r['prompt_id']:r for k in ['source_prompts','continuation_prompts'] for r in prepared[k]}
protocol=read(R/'campaign/protocol.json','registered source recipes and GPU allocation')
sources={r['seed']:r['source_recipe'] for r in protocol['origins']['development']}
for code in ['code_e59b9ee/ssvc_flow/src/prospective_selection/branch_runtime.py','code_e59b9ee/ssvc_flow/src/modeling_v3/vlm_observation.py','code_e59b9ee/ssvc_flow/src/decision_modeling/reward_recipes.py']:
 track(R/code,'historical timer definition')
 # Read for reproducibility receipt; these files are never imported/executed.
 (R/code).read_text()
groups=[];updates=[];segments=[]
completes=sorted((R/'campaign/branches').glob('6100[1-4]_t*/*/repeat_1/COMPLETE.json'))+sorted((R/'campaign/sources').glob('6100[1-4]/COMPLETE.json'))
assert len(completes)==92,len(completes)
for completepath in completes:
 c=read(completepath,'successful training COMPLETE');identity=c['identity'];branch='origin_id' in identity;scope='branch_training' if branch else 'source_training';lin=identity['lineage_id'];origin=identity.get('origin_id',str(lin)+'_source');source_step=int(origin.split('_t')[-1]) if branch else 0;stage=('early' if source_step==32 else 'late') if branch else 'source';recipe=identity.get('recipe_id',identity.get('source_recipe'))
 common=dict(scope=scope,lineage_id=lin,source_recipe=sources[lin],source_step=source_step,source_stage=stage,origin_id=origin,branch_recipe=recipe)
 for cp in c['authoritative_segments']:
  cp=local(cp);commit=read(cp,'authoritative COMMIT');samplefile=local(commit['samples']['path']);rows={}
  for line in track(samplefile,'committed training samples for missing semantic metadata and counters').open():
   x=json.loads(line);rows[x['sample_key']]=x
  segments.append({**common,'step_bin':f"{commit['start']:02d}-{commit['stop']:02d}",'elapsed_seconds':commit['elapsed_seconds'],'sampling_seconds':commit['sampling_seconds'],'update_seconds':commit['update_seconds'],'commit_path':str(cp)})
  for up in commit['updates']:
   up=local(up['path']);u=read(up,'committed actual advantage and update timer');s=u['step'];rs=u['reward_statistics'];adv=rs['advantages'];rewards=rs['reward_vectors'];assert len(adv)==4 and all(len(v)==8 for v in adv)
   ordered=[rows[r['sample_key']] for r in u['ratio_records']];assert len(ordered)==32
   zero=all(v==0 for g in adv for v in g);nonzero=[abs(v) for g in adv for v in g if v!=0]
   cost=u['sample_cost'];updates.append({**common,'source_step':source_step if branch else s,'step':s,'step_bin':f'{((s-1)//8)*8+1:02d}-{((s-1)//8+1)*8:02d}','exact_zero_advantage':int(zero),'grad_exact_zero':int(u['grad_norm_preclip']==0),'grad_norm_preclip':u['grad_norm_preclip'],'min_abs_nonzero_advantage':min(nonzero) if nonzero else '', 'max_abs_advantage':max(abs(v) for g in adv for v in g),'sampling_seconds':u['sampling_seconds'],'generation_measured_seconds':cost['elapsed_seconds'],'update_seconds':u['update_seconds'],'scored_sequences':cost['scored_sequences'],'generated_sequences':cost['generated_sequences'],'backward_calls':u['backward_calls'],'update_path':str(up)})
   for gidx in range(4):
    rr=ordered[gidx*8:(gidx+1)*8];assert len({r['prompt_id'] for r in rr})==1;assert [r['draw_index'] for r in rr]==list(range(8));first=rr[0];sem=[r['semantic'] for r in rr];cat=[r['category'] for r in rr];nX=cat.count('X');noX=nX==0;vF=[x['F'] for x in sem if x['F'] is not None];vB=[x['B'] for x in sem if x['B'] is not None];vM=[x['M'] for x in sem if x['M'] is not None];vC=[r['reward_vector'][3] for r in rr];av=adv[gidx];vec=rewards[gidx]
    row={**common,'source_step':source_step if branch else s,'step':s,'step_bin':f'{((s-1)//8)*8+1:02d}-{((s-1)//8+1)*8:02d}','H':s if branch else '', 'panel':'train','base_scene_id':first['base_scene_id'],'prompt_id':first['prompt_id'],'interface':first['interface'],'family':first['family'],'operation':meta[first['prompt_id']]['operation'],'draws':8,'groups_n':1,'all_X_n':int(nX==8),'no_X_n':int(noX),'mixed_X_n':int(0<nX<8),'all_valid_n':int(all(x['valid'] for x in sem)),'contains_S_n':int('S' in cat),'identical_reward_vectors_n':int(all(v==vec[0] for v in vec)),'identical_original_four_rewards_n':int(all(r['reward_vector']==rr[0]['reward_vector'] for r in rr)),'exact_zero_advantage_groups_n':int(all(v==0 for v in av)),'no_X_with_F1_n':int(noX and 1 in vF),'no_X_with_F_observed_n':int(noX and bool(vF)),'F_different_n':int(len(set(vF))>1),'F_comparable_groups_n':int(len(vF)>=2),'B_different_n':int(len(set(vB))>1),'B_comparable_groups_n':int(len(vB)>=2),'C_different_n':int(len(set(vC))>1),'C_comparable_groups_n':1,'no_X_F_different_n':int(noX and len(set(vF))>1),'no_X_F_comparable_n':int(noX and len(vF)>=2),'no_X_B_different_n':int(noX and len(set(vB))>1),'no_X_B_comparable_n':int(noX and len(vB)>=2),'no_X_C_different_n':int(noX and len(set(vC))>1),'no_X_C_comparable_n':int(noX),'X_samples_n':nX,'samples_n':8,'F_sum':sum(vF),'F_valid_samples_n':len(vF),'B_sum':sum(vB),'B_valid_samples_n':len(vB),'M_sum':sum(vM),'M_valid_samples_n':len(vM),'I_samples_n':cat.count('I'),'W_samples_n':cat.count('W'),'S_samples_n':cat.count('S'),'W_positive_advantage_n':sum(c=='W' and v>0 for c,v in zip(cat,av)),'S_positive_advantage_n':sum(c=='S' and v>0 for c,v in zip(cat,av)),'W_positive_advantage_groups_n':int(any(c=='W' and v>0 for c,v in zip(cat,av))),'S_positive_advantage_groups_n':int(any(c=='S' and v>0 for c,v in zip(cat,av))),'no_X_W_positive_advantage_n':sum(noX and c=='W' and v>0 for c,v in zip(cat,av)),'no_X_W_samples_n':sum(noX and c=='W' for c in cat),'no_X_S_positive_advantage_n':sum(noX and c=='S' and v>0 for c,v in zip(cat,av)),'no_X_S_samples_n':sum(noX and c=='S' for c in cat),'sample_path':str(samplefile),'update_path':str(up)}
    groups.append(row)
assert len(updates)==3200 and len(groups)==12800,(len(updates),len(groups))
assert all(g['all_X_n']+g['no_X_n']+g['mixed_X_n']==1 for g in groups)
assert sum(g['samples_n'] for g in groups)==102400
pilot=pathlib.Path(__file__).resolve().parents[2]/'ssvc_flow/docs/prospective_selection/pilot_four_20261004/training_updates_3200.csv'
if pilot.exists():
 old=list(csv.DictReader(track(pilot,'reused prior update index; path and timer consistency check').open()))
 assert {r['path'] for r in old}=={str(pathlib.Path(u['update_path']).relative_to(R)) for u in updates}
 assert abs(sum(float(r['update_seconds']) for r in old)-sum(u['update_seconds'] for u in updates))<1e-6
write('training_group_detail_CF.csv',groups);write('training_update_detail_CF.csv',updates)
branchgroups=[g for g in groups if g['scope']=='branch_training'];branchupdates=[u for u in updates if u['scope']=='branch_training']
metrics=[k for k in groups[0] if k.endswith('_n') or k.endswith('_sum')]
def agg(rows,keys):
 acc={}
 for row in rows:
  key=tuple(row[k] for k in keys)
  if key not in acc:acc[key]={**dict(zip(keys,key)),**{m:0 for m in metrics}}
  for m in metrics:acc[key][m]+=row[m]
 out=list(acc.values())
 for row in out:
  for m in ['all_X','no_X','mixed_X','all_valid','contains_S','identical_reward_vectors','identical_original_four_rewards','exact_zero_advantage_groups']:
   row[m+'_denominator']=row['groups_n'];row[m+'_rate']=row[m+'_n']/row['groups_n']
  for m,den in [('no_X_with_F1','no_X_n'),('F_different','F_comparable_groups_n'),('B_different','B_comparable_groups_n'),('C_different','C_comparable_groups_n'),('no_X_F_different','no_X_F_comparable_n'),('no_X_B_different','no_X_B_comparable_n'),('no_X_C_different','no_X_C_comparable_n'),('W_positive_advantage','W_samples_n'),('S_positive_advantage','S_samples_n'),('no_X_W_positive_advantage','no_X_W_samples_n'),('no_X_S_positive_advantage','no_X_S_samples_n')]:
   row[m+'_denominator']=row[den];row[m+'_rate']=row[m+'_n']/row[den] if row[den] else ''
  for field in ['F','B','M']:
   row[field+'_mean_valid']=row[field+'_sum']/row[field+'_valid_samples_n'] if row[field+'_valid_samples_n'] else ''
 return out
keys=['lineage_id','source_recipe','source_step','source_stage','origin_id','branch_recipe','step_bin'];timeline=agg(branchgroups,keys)
for row in timeline:
 us=[u for u in branchupdates if all(u[k]==row[k] for k in keys)];row.update(update_steps_n=len(us),grad_exact_zero_steps_n=sum(u['grad_exact_zero'] for u in us),exact_zero_advantage_steps_n=sum(u['exact_zero_advantage'] for u in us))
write('support_timeline.csv',timeline)
response=[]
for level,keys in [('stage',['source_stage']),('stage_family_interface',['source_stage','family','interface']),('stage_recipe',['source_stage','branch_recipe']),('stage_lineage',['source_stage','lineage_id']),('stage_family_interface_recipe',['source_stage','family','interface','branch_recipe']),('stage_time',['source_stage','step_bin']),('stage_family_interface_recipe_time',['source_stage','family','interface','branch_recipe','step_bin'])]:
 for row in agg(branchgroups,keys):row['aggregation_level']=level;response.append(row)
opportunity=O/'opportunity_by_stage_group.csv'
if opportunity.exists():
 endpoint=[r for r in csv.DictReader(track(opportunity,'existing H32 family/interface/stage response; separate evaluation pool').open()) if r['aggregation']=='stage' and r['H']=='32']
 sums=['X_n','draws','valid_n','C_sum','copy_sum_valid','coordinate_correct_sum','coordinate_denominator','F_sum_valid','B_sum_valid','M_sum_valid','relation_full_wrong_n']
 for row in response:
  if row['aggregation_level'] not in ['stage','stage_family_interface','stage_recipe','stage_family_interface_recipe']:continue
  er=[r for r in endpoint if r['stage']==row['source_stage'] and r['family']==row.get('family','ALL') and r['interface']==row.get('interface','ALL') and (not row.get('branch_recipe') or r['branch_recipe']==row['branch_recipe'])]
  if not er:continue
  for k in sums:
   if all(k in r for r in er):row['eval_H32_'+k]=sum(float(r[k]) for r in er)
  row['eval_H32_pX']=row['eval_H32_X_n']/row['eval_H32_draws']
  row['pool_correspondence']='same family/interface/source_stage only; distinct train/eval prompt identities; no causal pairing'
write('support_response_summary.csv',response)
# Allocation-time totals are sums of successful segments on one GPU; not campaign wall-clock.
costs=[]
for scope,stage,recipe in sorted({(u['scope'],u['source_stage'],u['branch_recipe']) for u in updates}):
 us=[u for u in updates if (u['scope'],u['source_stage'],u['branch_recipe'])==(scope,stage,recipe)];ss=[s for s in segments if (s['scope'],s['source_stage'],s['branch_recipe'])==(scope,stage,recipe)]
 def costsrow(us,ss,scope,stage,recipe):
  sampling=sum(u['sampling_seconds'] for u in us);generation=sum(u['generation_measured_seconds'] for u in us);update=sum(u['update_seconds'] for u in us);elapsed=sum(s['elapsed_seconds'] for s in ss);zero=[u for u in us if u['exact_zero_advantage']];tiny=[u for u in us if 0<u['max_abs_advantage']<1e-6]
  return dict(scope=scope,source_stage=stage,branch_recipe=recipe,update_steps_n=len(us),exact_zero_advantage_steps_n=len(zero),grad_exact_zero_steps_n=sum(u['grad_exact_zero'] for u in us),tiny_nonzero_steps_lt_1e_6_n=len(tiny),generation_measured_hours=generation/3600,sampling_total_hours=sampling/3600,update_hours=update/3600,sampling_overhead_unseparated_hours=(sampling-generation)/3600,segment_other_hours=(elapsed-sampling-update)/3600,successful_segment_gpu_allocation_work_hours=elapsed/3600,exact_zero_advantage_update_hours=sum(u['update_seconds'] for u in zero)/3600,exact_zero_advantage_sampling_hours=sum(u['sampling_seconds'] for u in zero)/3600,tiny_nonzero_update_hours=sum(u['update_seconds'] for u in tiny)/3600,min_abs_nonzero_advantage=min([u['min_abs_nonzero_advantage'] for u in us if u['min_abs_nonzero_advantage']!=''],default=''),extra_scored_sequences=sum(u['scored_sequences'] for u in us),semantic_scoring_hours='',model_loading_hours='',parallel_campaign_wall_hours='',provenance='successful COMPLETE->COMMIT->UPDATE; single GPU/job; no utilization telemetry',limitations='semantic scoring+preparation+state checks+ledger bundled in sampling overhead; pre-segment model loads/retries/queue and parallel wall unavailable; zero gradient does not imply unchanged parameters')
 costs.append(costsrow(us,ss,scope,stage,recipe))
for scope in ['branch_training','source_training','all_training']:
 us=[u for u in updates if scope=='all_training' or u['scope']==scope];ss=[s for s in segments if scope=='all_training' or s['scope']==scope];costs.append(costsrow(us,ss,scope,'ALL','ALL'))
# Existing evaluation timers include sampling/chunk overhead; no raw responses needed.
eval_files=sorted((R/'campaign/branches').glob('6100[1-4]_t*/*/repeat_1/evaluations/H*/COMPLETE.json'))+sorted((R/'campaign/prestate').glob('6100[1-4]_t*/t*/COMPLETE.json'))+sorted((R/'campaign/historical_E').glob('*/COMPLETE.json'))
ev=collections.defaultdict(list)
for f in eval_files:
 d=read(f,'existing evaluation COMPLETE timer');scope='endpoint_'+f.parent.name if '/branches/' in str(f) else ('prestate_P' if '/prestate/' in str(f) else 'historical_E');ev[scope].append(d)
for scope,ds in sorted(ev.items()):
 costs.append(dict(scope=scope,source_stage='ALL',branch_recipe='ALL',completed_evaluations_n=len(ds),sampling_total_hours=sum(d['elapsed_sampling_seconds'] for d in ds)/3600,provenance='existing COMPLETE.elapsed_sampling_seconds',limitations='sum of successful prompt chunk timers; generation/scoring/loading split and parallel wall clock not separately identifiable'))
write('cost_existing_only.csv',costs)
summary={'branch_groups':len(branchgroups),'all_training_updates':len(updates),'branch_group_by_stage':[{k:v for k,v in r.items() if k in ['source_stage','groups_n','no_X_n','no_X_with_F1_n','exact_zero_advantage_groups_n','F_different_n','B_different_n','C_different_n','W_positive_advantage_n','W_samples_n','S_positive_advantage_n','S_samples_n','I_samples_n']} for r in response if r['aggregation_level']=='stage'],'cost_totals':[c for c in costs if c['branch_recipe']=='ALL']}
(O/'support_cost_summary_CF.json').write_text(json.dumps(summary,indent=2));(O/'inputs_CF.json').write_text(json.dumps({'actual_inputs':[{'path':k,'purpose':v} for k,v in sorted(inputs.items())],'missing_fields':['semantic_scoring_seconds','model_loading_seconds','complete single-clock successful job start/end intervals for parallel wall clock'],'command':'python3 support_cost.py --raw '+str(R)+' --output '+str(O),'code_version':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'new_model_calls':0,'new_samples':0,'selector_refits':0,'statistical_resamples':0},indent=2))
print(json.dumps(summary,indent=2))
