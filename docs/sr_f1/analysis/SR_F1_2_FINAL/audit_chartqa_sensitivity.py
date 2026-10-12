#!/usr/bin/env python3
"""Explicit POST_HOC_WHITESPACE_SENSITIVITY; never replaces frozen scores."""
import collections,datetime,json,os,pathlib,sys
R=pathlib.Path('/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010');O=pathlib.Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(R/'deployment/early_sealed_20261011/code/src'))
from sr_f1.chartqa import load_verified_manifest,score_chartqa
from sr_f1.evaluation import sha_file
from sr_f12.protocol import object_hash
rd=lambda p:json.loads(p.read_text())
final=rd(R/'endpoint_orchestration/COMPLETE.json');assert final['endpoint_records']==227700 and final['status']=='EVALUATIONS_COMPLETE_REPORT_PENDING'
manifest={r['qid']:r for r in load_verified_manifest(R)}; summary={};n=0
with (O/'audit_chartqa_sensitivity_rows.jsonl').open('w') as f:
 for ref in final['releases']:
  model=ref['model_id'];d=R/'sealed/evaluation'/model/'chartqa'
  for g in rd(d/'COMPLETE.json')['groups']:
   p=d/'groups'/(g['group_id']+'.json');assert sha_file(p)==g['raw_sha256'];v=rd(p);assert object_hash(v['records'])==v['records_sha256']
   for r in v['records']:
    gold=manifest[r['qid']]['answer'];pred=r['raw_text'];frozen=score_chartqa(gold,pred);sensitivity=score_chartqa(str(gold).strip(),pred.strip())
    out={k:r[k] for k in ('slot_id','model_id','qid','root_id','variant')};out.update(label='POST_HOC_WHITESPACE_SENSITIVITY',frozen=frozen,whitespace_stripped=sensitivity,trailing_lf=pred.endswith('\n'),leading_or_trailing_whitespace=pred!=pred.strip())
    f.write(json.dumps(out,sort_keys=True)+'\n');n+=1
    for label in ('ALL',r['variant']):
     c=summary.setdefault(model,{}).setdefault(label,collections.Counter());c['n']+=1;c['trailing_lf']+=out['trailing_lf'];c['leading_or_trailing_whitespace']+=out['leading_or_trailing_whitespace']
     for metric in ('exact_match','relaxed_accuracy'):
      c['frozen_'+metric]+=frozen[metric];c['sensitivity_'+metric]+=sensitivity[metric];c['changed_'+metric]+=frozen[metric]!=sensitivity[metric]
assert n==32500
result={'status':'PASS','label':'POST_HOC_WHITESPACE_SENSITIVITY','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'script_sha256':sha_file(__file__),'records':n,'summary':summary,'rows_sha256':sha_file(O/'audit_chartqa_sensitivity_rows.jsonl'),'boundary':'Only first/last whitespace stripped. No generation, number extraction, answer repair, official-score replacement or scientific contract change.'}
(O/'audit_chartqa_sensitivity_result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
