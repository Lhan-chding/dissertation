import json,hashlib,pathlib,datetime
p=pathlib.Path(__file__).parent
rows=[json.loads(x) for x in (p/'BLIND.jsonl').read_text().splitlines()]
out=[]
for i,r in enumerate(rows,1):
 duplicates=[]
 def pairs(ps):
  d={}
  for k,v in ps:
   if k in d: duplicates.append(k)
   d[k]=v
  return d
 def badconst(s): raise ValueError('nonfinite '+s)
 err=None;o=None
 try:
  o=json.loads(r['full_text'],object_pairs_hook=pairs,parse_constant=badconst)
  if duplicates: raise ValueError('duplicate keys: '+str(duplicates))
  if not isinstance(o,dict): raise ValueError('not object')
 except (ValueError,TypeError) as e:err=str(e);o=None
 ev=o.get('evidence') if o else None
 alpha=[x for x in ev if isinstance(x,dict) and x.get('series')=='Alpha'] if isinstance(ev,list) else None
 out.append(dict(index=i,request_id=r['request_id'],raw_sha256=hashlib.sha256(r['full_text'].encode()).hexdigest(),finish_reason=r['finish_reason'],strict_single_json_object=o is not None,parse_error=err,duplicate_keys=duplicates,complete=o is not None and r['finish_reason'] not in ['length','max_tokens'],answer=o.get('answer') if o else None,answer_type=type(o.get('answer')).__name__ if o else None,selected=o.get('selected') if o else None,selected_present='selected' in o if o else False,selected_string_array=isinstance(o.get('selected'),list) and all(isinstance(x,str) for x in o['selected']) if o else False,bindings=o.get('bindings') if o else None,alpha_evidence=alpha,alpha_order=[x.get('category') for x in alpha] if alpha is not None else None,alpha_values=[x.get('value') for x in alpha] if alpha is not None else None,manual_raw_read=True,observation='Raw lacks final closing brace; visible fields are not accepted or repaired.' if i in [16,17] else ('selected absent; work text is not substituted.' if i==18 else ('Alpha repeats Bowl/Table/Shelf/Frame/Tray/Clock; no deduplication.' if i==21 else ('Alpha evidence only four categories; work not substituted.' if i in [22,23] else ('Work self-correction does not replace final answer.' if i==14 else 'Fields read directly from original JSON.'))))))
(p/'EXTRACTION.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
h=hashlib.sha256((p/'EXTRACTION.jsonl').read_bytes()).hexdigest()
(p/'EXTRACTION.sha256').write_text(h+'  EXTRACTION.jsonl\n')
(p/'STAGE1_RECEIPT.json').write_text(json.dumps(dict(stage=1,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=len(rows),extraction_sha256=h,blind_sha256=hashlib.sha256((p/'BLIND.jsonl').read_bytes()).hexdigest(),private_or_scores_read=False,production_scorer_used=False,raw_read_count=24),indent=2)+'\n')
print('sealed',len(out),h)
