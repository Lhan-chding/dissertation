#!/usr/bin/env python3
"""Post-release strict answer normalization and frozen ChartQA whitespace diagnosis."""
import collections, datetime, json, os, pathlib, sys, traceback
ROOT=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010')
OUT=pathlib.Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT/'deployment/early_sealed_20261011/code/src'))
from sr_f12.protocol import object_hash
from sr_f12.evaluation import score_record
from sr_f1.evaluation import sha_file, read_jsonl
from sr_f1.contract import load_json_strict, fraction_of
from sr_f1.chartqa import load_verified_manifest,score_chartqa

def rd(p):return json.loads(p.read_text())
def normalize(value):
 if value is None:return {'kind':'null','value':None}
 f=fraction_of(value,text=True);return {'kind':'rational','numerator':f.numerator,'denominator':f.denominator}
def save(name,obj): (OUT/name).write_text(json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
res={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ.get('SLURM_JOB_ID'),'script_sha256':sha_file(__file__)}
try:
 complete=rd(ROOT/'endpoint_orchestration/COMPLETE.json');assert complete['status']=='EVALUATIONS_COMPLETE_REPORT_PENDING' and complete['endpoint_records']==227700
 tasks={t['qid']:t for t in read_jsonl(ROOT/'manifests/TASKS_GOLD_AUDIT_ONLY.jsonl')};chart={t['qid']:t for t in load_verified_manifest(ROOT)}
 meta=[];chart_summary={};endpoint_count=0;baseline_count=0
 sources=[(ROOT/'sealed/evaluation'/ref['model_id'],ref['model_id'],'endpoint') for ref in complete['releases']]
 sources.extend((ROOT/'raw/evaluation/SRF1_2_COMMON_ZERO_step0_baseline'/f'shard{i:03d}','SRF1_2_COMMON_ZERO','baseline') for i in range(3))
 with (OUT/'audit_normalized_answers.jsonl').open('w') as out:
  for directory,model,stage in sources:
   panels=[directory/p['panel'] for p in rd(directory/'COMPLETE.json')['panels']] if stage=='endpoint' else [directory]
   for panel in panels:
    for group in rd(panel/'COMPLETE.json')['groups']:
     path=panel/'groups'/(group['group_id']+'.json');assert sha_file(path)==group['raw_sha256']; data=rd(path);assert object_hash(data['records'])==data['records_sha256']
     for row in data['records']:
      if row['pool']=='CHARTQA':
       d=chart_summary.setdefault(model,{'counts':collections.Counter(),'examples':[]})
       gold=chart[row['qid']]['answer'];pred=row['raw_text'];score=score_chartqa(gold,pred);c=d['counts'];c['records']+=1;c['exact_match']+=score['exact_match'];c['relaxed_accuracy']+=score['relaxed_accuracy'];c['trailing_newline']+=int(pred.endswith('\n'));c['leading_or_trailing_whitespace']+=int(pred!=pred.strip());c['literal_exact_after_all_whitespace_strip_diagnostic_only']+=int(str(gold).strip().lower()==pred.strip().lower())
       try: float(str(gold));float(pred); numeric=True
       except (ValueError,OverflowError):numeric=False
       c['both_float_parseable']+=numeric;c['numeric_relaxed_correct']+=int(numeric and score['relaxed_accuracy']);c['nonnumeric_relaxed_correct']+=int(not numeric and score['relaxed_accuracy'])
       if len(d['examples'])<3 and not score['exact_match'] and str(gold).strip().lower()==pred.strip().lower():d['examples'].append({'qid':row['qid'],'gold':gold,'raw_prediction':pred,'raw_group':str(path.relative_to(ROOT)),'raw_group_sha256':group['raw_sha256'],'frozen_score':score})
       continue
      score=score_record(row,tasks[row['qid']]);norm=None;reason=None
      if score.get('L_answer')==1:
       try:norm=normalize(load_json_strict(row['raw_text'])['answer'])
       except (ValueError,TypeError,KeyError,OverflowError,RecursionError,ZeroDivisionError) as e: reason=type(e).__name__
      else:reason='L_answer_not_1'
      gold_norm=normalize(tasks[row['qid']]['answer'])
      record={k:row.get(k) for k in ('slot_id','model_id','pool','protocol','view','qid','root_id','family','chart','variant','draw')}
      record.update(stage=stage,normalized_answer=norm,answer_parse_reason=reason,gold_normalized=gold_norm,gold_normalized_sha256=object_hash(gold_norm),score_L_answer=score.get('L_answer'),score_A=score.get('A'))
      out.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+'\n')
      if stage=='endpoint':endpoint_count+=1
      else:baseline_count+=1
 res.update(status='PASS',endpoint_structured_rows=endpoint_count,baseline_structured_rows=baseline_count,chartqa=chart_summary,diagnostic_caveat='Frozen exact_match and relaxed_accuracy unchanged. strip-all-whitespace counts are explanatory sensitivity only, not replacement scientific scores.')
 assert endpoint_count==195200 and baseline_count==6656 and sum(d['counts']['records'] for d in chart_summary.values())==32500
except BaseException:
 res.update(status='FAIL',error=traceback.format_exc());raise
finally:
 res['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();res['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha_file(p)} for p in OUT.iterdir() if p.is_file()};save('audit_structure_result.json',res)
