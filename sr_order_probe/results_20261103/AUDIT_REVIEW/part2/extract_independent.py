import json, hashlib
from pathlib import Path
from datetime import datetime,timezone
rows=[json.loads(x) for x in Path('BLIND.jsonl').read_text().splitlines()]
out=[]
for n,row in enumerate(rows,1):
 duplicates=[]
 def pairs(p):
  d={}
  for k,v in p:
   if k in d: duplicates.append(k)
   d[k]=v
  return d
 error=None; obj=None
 try:
  obj=json.loads(row['full_text'],object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite '+x)))
  if duplicates: raise ValueError('Duplicate keys: '+str(duplicates))
  if not isinstance(obj,dict): raise ValueError('Root not object')
 except (ValueError,TypeError) as exc: error=str(exc); obj=None
 alpha=[v for v in obj.get('evidence',[]) if isinstance(v,dict) and v.get('series')=='Alpha'] if obj else None
 out.append(dict(ordinal=n,request_id=row['request_id'],raw_text_sha256=hashlib.sha256(row['full_text'].encode()).hexdigest(),finish_reason=row['finish_reason'],strict_single_json_object=obj is not None,duplicate_keys_detected=duplicates,parse_error=error,complete=obj is not None and row['finish_reason'] in ['balanced','eos','stop'],answer=obj.get('answer') if obj else None,answer_type=type(obj.get('answer')).__name__ if obj else None,selected=obj.get('selected') if obj else None,selected_shape_valid=isinstance(obj.get('selected'),list) and all(isinstance(x,str) for x in obj.get('selected')) if obj else False,bindings=obj.get('bindings') if obj else None,alpha_evidence=alpha,alpha_order=[x.get('category') for x in alpha] if alpha is not None else None,alpha_values=[x.get('value') for x in alpha] if alpha is not None else None,raw_read_by_agent=True,raw_observation='Whole source read; missing final root closing brace; no repair or field salvage.' if 7<=n<=12 else ('Only 9 Alpha evidence rows.' if n==2 else 'Whole source read; final answer only extracted; work not substituted.')))
Path('EXTRACTION.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in out))
h=hashlib.sha256(Path('EXTRACTION.jsonl').read_bytes()).hexdigest()
Path('EXTRACTION.sha256').write_text(h+'  EXTRACTION.jsonl\n')
Path('STAGE1_RECEIPT.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),'source':'BLIND.jsonl','source_sha256':hashlib.sha256(Path('BLIND.jsonl').read_bytes()).hexdigest(),'extraction_sha256':h,'rows_read':24,'private_or_scores_read':False,'production_scorer_read_or_called':False},indent=2)+'\n')
print('extraction sha256',h)
for r in out: print(r['ordinal'],r['strict_single_json_object'],r['parse_error'],r['answer'])
