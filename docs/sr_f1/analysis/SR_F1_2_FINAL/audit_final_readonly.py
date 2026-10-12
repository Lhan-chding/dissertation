#!/usr/bin/env python3
"""Independent final SR-F1.2 audit; runs on ordinary CPU, never releases or mutates science."""
import datetime, hashlib, json, os, pathlib, subprocess, sys, traceback
ROOT = pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010')
OUT = pathlib.Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT/'deployment/early_sealed_20261011/code/src'))
from sr_f12.protocol import object_hash, BASELINE, scientific_matrix
from sr_f12.endpoint import verify_endpoint_authorization
from sr_f12.evaluation import score_record, verify_source_manifest, collect_scored_shards
from sr_f1.chartqa import load_verified_manifest, score_chartqa
from sr_f1.evaluation import sha_file, read_jsonl

def rd(p): return json.loads(pathlib.Path(p).read_text())
def save(name, data):
 p=OUT/name; p.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)+'\n'); return sha_file(p)
def run(args): return subprocess.run(args, capture_output=True, text=True, check=True).stdout

audit={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id':os.environ.get('SLURM_JOB_ID'), 'root':str(ROOT), 'script_sha256':sha_file(__file__), 'checks':{}}
try:
 final=rd(ROOT/'endpoint_orchestration/COMPLETE.json'); state=rd(ROOT/'endpoint_orchestration/STATE.json')
 assert final['status']=='EVALUATIONS_COMPLETE_REPORT_PENDING' and final['endpoint_models']==13 and final['endpoint_records']==227700
 assert state['scores_sealed'] is False and state['endpoints_complete']==13
 assert not state['failures'] and not state['unknown_submissions']
 audit['checks']['final_receipt']={'sha256':sha_file(ROOT/'endpoint_orchestration/COMPLETE.json'),'object_sha256':object_hash(final),'state_at':state['at']}
 registration=rd(ROOT/'EARLY_ENDPOINT_IMPLEMENTATION.json')
 audit['checks']['sources']={}
 for label,prefix in [('training','training_'),('evaluation','')]:
  manifest=registration[prefix+'source_manifest']; code=registration[prefix+'code_root']; actual=verify_source_manifest(manifest,code)
  assert actual==registration[prefix+'source_manifest_sha256']
  audit['checks']['sources'][label]={'files':len(rd(manifest)['files']),'manifest_object_sha256':actual,'git_commit':rd(manifest)['git_commit']}
 audit['checks']['frozen']={}
 for rel,expected in registration['frozen_files'].items():
  actual=sha_file(ROOT/rel); assert actual==expected; audit['checks']['frozen'][rel]=actual
 applied=rd(ROOT/'deployment/checkpoint_storage_20261010/APPLIED.json'); mapping=[]
 for source,target in applied['mappings'].items():
  p=pathlib.Path(source); assert p.is_symlink() and p.resolve(strict=True)==pathlib.Path(target).resolve(strict=True)
  mapping.append({'source':source,'target':str(p.resolve()),'latest':rd(p/'LATEST.json'),'commits':len(list(p.resolve().glob('commit-*.json')))})
 assert len(mapping)==8; audit['checks']['hdd_mappings']=mapping
 # This revalidates live Slurm training terminals, all12 complete Adam/state/cursor/RNG,
 # adapter bytes and FIT32/64/96, explicitly following the HDD checkpoint links.
 authorization=verify_endpoint_authorization(ROOT)
 acceptance=rd(ROOT/'endpoint_orchestration/MAIN_ACCEPTANCE.json')
 assert object_hash(authorization)==acceptance['main_completion_sha256']
 audit['checks']['full_training_state']={'status':'PASS','paths':authorization['paths'],'authorization_object_sha256':object_hash(authorization)}
 eval_jobs=[v['job_id'] for v in state['tasks'].values()]; train_jobs=[v['terminal']['job_id'] for v in acceptance['terminals']]
 jobs=train_jobs+eval_jobs+[state['scheduler_job_id']]
 assert len(set(train_jobs+eval_jobs))==25
 queue=run(['squeue','-h','-j',','.join(jobs),'-o','%i|%T']); assert not queue.strip()
 accounting=run(['sacct','-X','-n','-P','-j',','.join(jobs),'--format=JobID,State,ExitCode,Start,End,Elapsed,QOS,AllocTRES'])
 parsed=[line.split('|') for line in accounting.splitlines() if line.strip()]
 assert {row[0] for row in parsed}==set(jobs)
 assert all(row[1:3]==['COMPLETED','0:0'] for row in parsed)
 audit['checks']['slurm']={'queue':queue,'accounting':accounting,'training_jobs':train_jobs,'evaluation_jobs':eval_jobs}
 tasks={v['qid']:v for v in read_jsonl(ROOT/'manifests/TASKS_GOLD_AUDIT_ONLY.jsonl')}
 chart={v['qid']:v for v in load_verified_manifest(ROOT)}
 save('audit_gold_tasks.json',list(tasks.values()))
 save('audit_chartqa_manifest.json',[{k:v[k] for k in ('qid','subset','image_id','image_sha256')} for v in chart.values()])
 models=[]; indices=[]; total=0
 fields=('slot_id','model_id','step','pool','protocol','view','qid','root_id','family','chart','variant','draw')
 meta_fields=fields+('completion_token_count','prompt_token_count','finish_reason','truncated','format_protocol_error','generation_status','generation_seconds','input_hash','image_hash')
 with (OUT/'audit_raw_metadata.jsonl').open('w') as metadata:
  for release_ref in final['releases']:
   model=release_ref['model_id']; release_path=ROOT/'released/evaluation'/f'{model}.json'; release=rd(release_path)
   assert object_hash(release)==release_ref['release_sha256']
   assert release['count']==len(release['records'])==release_ref['records']
   assert object_hash(release['records'])==release['records_sha256']
   assert release['authorization_sha256']==object_hash(authorization)
   directory=ROOT/'sealed/evaluation'/model; complete=rd(directory/'COMPLETE.json')
   assert release['evaluation_sha256']==object_hash(complete)
   if complete.get('generation_gate')=='MODEL_TERMINAL_SEALED':
    generation=rd(directory/'GENERATION_AUTHORIZATION.json'); assert generation==rd(ROOT/'endpoint_orchestration/model_acceptance'/f'{model}.json')
    assert object_hash(generation)==complete['generation_authorization_sha256']
    assert generation['implementation_sha256']==object_hash(registration)
    selected=generation['model_completion']; assert object_hash(selected)==generation['model_completion_sha256']
    assert selected['paths']==[v for v in authorization['paths'] if v['model_id']==model]
    assert all(selected[k]==authorization[k] for k in ('science_freeze_sha256','config_sha256'))
   else: assert complete['main_completion_sha256']==object_hash(authorization)
   recomputed=[]; seen=set(); panel_audits=[]
   for panel in complete['panels']:
    d=directory/panel['panel']; done=rd(d/'COMPLETE.json'); plan=rd(d/'PLAN.json')
    assert sha_file(d/'COMPLETE.json')==panel['completion_sha256']; assert done['plan_sha256']==object_hash(plan)
    assert len(done['groups'])==len(plan['assigned_groups']) and {g['group_id'] for g in done['groups']}==set(plan['assigned_groups'])
    count=0
    for group in done['groups']:
     p=d/'groups'/(group['group_id']+'.json'); value=rd(p)
     assert sha_file(p)==group['raw_sha256']; assert object_hash(value['records'])==value['records_sha256']; assert len(value['records'])==group['records']
     indices.append({'path':str(p.relative_to(ROOT)),'sha256':group['raw_sha256'],'records_sha256':value['records_sha256'],'records':len(value['records'])})
     for row in value['records']:
      assert row['slot_id'] not in seen; seen.add(row['slot_id']); assert row['model_id']==model
      if row['pool']=='CHARTQA':
       item=chart[row['qid']]; assert row['root_id']==item['image_id'] and row['variant']==item['subset']
       score=score_chartqa(item['answer'],row['raw_text'])
      else: score=score_record(row,tasks[row['qid']])
      recomputed.append({k:row[k] for k in fields}|{'score':score})
      met={k:row.get(k) for k in meta_fields}; met['score']=score
      metadata.write(json.dumps(met,ensure_ascii=False,sort_keys=True)+'\n'); count+=1
    assert count==panel['records']==done['generated_records']==plan['expected_records']
    panel_audits.append({'panel':panel['panel'],'records':count,'complete_path':str((d/'COMPLETE.json').relative_to(ROOT)),'complete_sha256':sha_file(d/'COMPLETE.json'),'plan_sha256':sha_file(d/'PLAN.json'),'groups':len(done['groups'])})
   assert recomputed==release['records']; total+=len(recomputed)
   models.append({'model_id':model,'records':len(recomputed),'release_file_sha256':sha_file(release_path),'release_object_sha256':object_hash(release),'recomputed_records_sha256':object_hash(recomputed),'score_mismatches':0,'panels':panel_audits})
   print(model,'PASS',len(recomputed),flush=True)
 assert total==227700 and len(models)==13
 audit['checks']['endpoint_rescore']={'models':models,'records':total,'score_mismatches':0,'method':'Recomputed all scores with unchanged pinned scorer, compared every score and metadata value with released rows; independent traversal and hash checks.'}
 save('audit_raw_group_index.json',indices)
 baseline=collect_scored_shards(ROOT/'raw/evaluation'/f'{BASELINE}_step0_baseline',expected_shards=3)
 base_meta=[]
 for row in baseline:
  assert score_record(row,tasks[row['qid']])==row['score']
  base_meta.append({k:row.get(k) for k in meta_fields}|{'score':row['score']})
 save('audit_baseline_scored_rows.json',base_meta)
 audit['checks']['baseline_rescore']={'records':len(baseline),'score_mismatches':0}
 audit['checks']['quota']={}
 for p in ('/projects/_ssd/varunssd','/projects/_hdd/varunhdd'):
  quota=int(os.getxattr(p,'ceph.quota.max_bytes'));used=int(os.getxattr(p,'ceph.dir.rbytes'))
  audit['checks']['quota'][p]={'quota_bytes':quota,'used_bytes':used,'free_bytes':quota-used}
 audit['status']='PASS'
except BaseException:
 audit['status']='FAIL';audit['error']=traceback.format_exc();raise
finally:
 audit['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 audit['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha_file(p)} for p in OUT.iterdir() if p.is_file()}
 save('audit_result.json',audit)
