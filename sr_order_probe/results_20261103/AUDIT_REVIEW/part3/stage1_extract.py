# -*- coding: utf-8 -*-
import json,hashlib,pathlib,datetime
p=pathlib.Path(__file__).parent
notes={3:'末尾缺少对象闭合大括号；不可采纳可见answer=43.5。',6:'末尾缺少对象闭合大括号；不可采纳可见answer=132。',7:'末尾缺少对象闭合大括号；不可采纳可见answer=168。',8:'末尾缺少对象闭合大括号；不可采纳可见answer=198。',9:'末尾缺少对象闭合大括号；不可采纳可见answer=44。',10:'末尾缺少对象闭合大括号；不可采纳可见answer=138。',11:'末尾缺少对象闭合大括号；不可采纳可见answer=114。',12:'末尾缺少对象闭合大括号；不可采纳可见answer=168。',13:'末尾缺少对象闭合大括号；不可采纳可见answer字符串46.5。',14:'末尾缺少对象闭合大括号；不可采纳可见answer=48。',15:'末尾缺少对象闭合大括号；不可采纳可见answer=54。',16:'末尾缺少对象闭合大括号；不可采纳可见answer=120。',17:'evidence之后出现非法**标记。',18:'length终止；evidence重复类别且中途截断，无最终字段。',19:'末尾缺少对象闭合大括号；不可采纳可见answer=198。',22:'末尾缺少对象闭合大括号；不可采纳可见answer=198。',23:'work字符串内有未转义换行与引号且未正确闭合；work自我纠正不能修复JSON。',24:'末尾缺少对象闭合大括号；不可采纳可见answer=44。'}
rows=[]
for i,line in enumerate((p/'BLIND.jsonl').read_text().splitlines(),1):
 r=json.loads(line); duplicates=[]
 def pairs(kvs):
  d={}
  for k,v in kvs:
   if k in d: duplicates.append(k)
   d[k]=v
  return d
 def bad(s): raise ValueError('nonfinite '+s)
 obj=None; err=None
 try:
  obj=json.loads(r['full_text'],object_pairs_hook=pairs,parse_constant=bad)
  if not isinstance(obj,dict): raise ValueError('not object')
  if duplicates: raise ValueError('duplicate keys')
 except (ValueError,TypeError) as e: err=str(e);obj=None
 alpha=[x for x in obj.get('evidence',[]) if isinstance(x,dict) and x.get('series')=='Alpha'] if obj and isinstance(obj.get('evidence'),list) else None
 x={'row':i,'request_id':r['request_id'],'raw_read_by_agent':True,'raw_sha256':hashlib.sha256(r['full_text'].encode()).hexdigest(),'finish_reason':r['finish_reason'],'strict_single_json':obj is not None,'duplicate_keys_detected_before_error':duplicates,'parse_error':err,'complete':obj is not None and r['finish_reason'] not in ('length','max_tokens'),'answer':obj.get('answer') if obj else None,'answer_type':type(obj.get('answer')).__name__ if obj and 'answer'in obj else 'unavailable','selected':obj.get('selected') if obj else None,'selected_type':type(obj.get('selected')).__name__ if obj and 'selected'in obj else 'unavailable','bindings':obj.get('bindings') if obj else None,'alpha_evidence':alpha,'alpha_order':[a.get('category') for a in alpha] if alpha is not None else None,'notes':notes.get(i,'完整单JSON；未从work或其他字段补出缺失selected。')}
 rows.append(x)
(p/'EXTRACTION.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
digest=hashlib.sha256((p/'EXTRACTION.jsonl').read_bytes()).hexdigest()
(p/'STAGE1_RECEIPT.json').write_text(json.dumps({'stage':'BLIND-only extraction frozen before PRIVATE/FROZEN read','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(rows),'raw_rows_actually_read':24,'extraction_sha256':digest,'blind_sha256':hashlib.sha256((p/'BLIND.jsonl').read_bytes()).hexdigest(),'production_scorer_read_or_called':False},indent=2))
print('EXTRACTION SHA256',digest)
print([(r['row'],r['strict_single_json']) for r in rows])
