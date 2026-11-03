import json,hashlib,re
from fractions import Fraction
from pathlib import Path
from datetime import datetime,timezone
E=[json.loads(l) for l in Path('EXTRACTION.jsonl').read_text().splitlines()]
P={r['request_id']:r['task'] for r in map(json.loads,Path('PRIVATE.jsonl').read_text().splitlines())}
def num(x):
 if isinstance(x,bool) or not isinstance(x,(int,float,str)): return None
 s=str(x).strip()
 if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?\d+/[+-]?\d+',s):return None
 try:return Fraction(s)
 except (ValueError,ZeroDivisionError):return None
out=[]
for e in E:
 t=P[e['request_id']];w=t['world'];q=t['query'];cats=w['categories']; alpha=w['series'][q['selection_series']]; beta=w['series'][q['value']]
 ranking=sorted(range(len(cats)),key=lambda i: (alpha[i] if q['direction']=='smallest' else -alpha[i],i))
 idx=sorted(ranking[:q['k']]); selected=[cats[i] for i in idx]; operands={cats[i]:num(beta[i]) for i in idx}
 ans=sum(operands.values(),Fraction())/(len(idx) if q['aggregate']=='mean' else 1)
 g=t['gold'];gold_checks={'indices':idx==g['indices'],'selected':selected==g['selected'],'operands':[str(operands[x]) for x in selected]==[str(num(x)) for x in g['operands']],'answer':ans==num(g['answer'])}
 A=int(e['strict_single_json_object'] and num(e['answer'])==ans)
 s=e['selected'];S='UNKNOWN' if not e['complete'] or not e['selected_shape_valid'] else ('PASS' if len(s)==len(set(s)) and set(s)==set(selected) else 'FAIL')
 binding=e['bindings'];bm={};bad=[]
 if isinstance(binding,list):
  for v in binding:
   if not isinstance(v,dict) or not isinstance(v.get('category'),str) or v['category'] in bm:bad.append(v);continue
   bm[v['category']]=num(v.get('operand'))
 B=int(e['strict_single_json_object'] and isinstance(binding,list) and not bad and set(bm)==set(selected) and all(bm.get(k)==v for k,v in operands.items()))
 order=e['alpha_order']; known=isinstance(order,list) and len(order)==len(cats) and len(set(order))==len(order) and set(order)==set(cats)
 truth=all(num(v.get('value'))==num(alpha[cats.index(v['category'])]) for v in e['alpha_evidence']) if known else None
 printed=order==cats if known else None; alphabetical=order==sorted(cats) if known else None
 sorted_true=all((alpha[cats.index(order[i])]<=alpha[cats.index(order[i+1])]) if q['direction']=='smallest' else (alpha[cats.index(order[i])]>=alpha[cats.index(order[i+1])]) for i in range(len(order)-1)) if known else None
 cond=e['request_id'].split('/')[2]; compliance=printed if cond=='C_PRINT' else alphabetical if cond=='C_ALT' else None
 wrong=[{'category':k,'expected':str(v),'observed':str(bm.get(k))} for k,v in operands.items() if bm.get(k)!=v] if e['strict_single_json_object'] else []
 reason=f"World/query {q['direction']} {q['k']} by Alpha with original category index tie-break gives {selected}; {q['aggregate']} Beta {list(map(str,operands.values()))} = {ans}. "
 reason+=f"Final answer={e['answer']!r}; S={S}; B={B}. " if e['strict_single_json_object'] else 'Missing final root brace: invalid JSON; A=0, S=UNKNOWN, B=0, order metrics UNKNOWN. '
 if wrong:reason+='Binding defects: '+json.dumps(wrong)+'. '
 if known:reason+=f'All 10 Alpha names unique; values_true={truth}; print={printed}; alphabetical={alphabetical}.'
 elif e['strict_single_json_object']:reason+=f'Only {len(order or [])} Alpha names: order/value completeness UNKNOWN.'
 out.append({'ordinal':e['ordinal'],'request_id':e['request_id'],'derived_gold':{'indices':idx,'selected':selected,'operands':{k:str(v) for k,v in operands.items()},'answer':str(ans),'full_rank_order':[cats[i] for i in ranking]},'provided_gold_checks':gold_checks,'A':A,'S':S,'B':B,'rank_order_known':known,'values_true':truth,'print':printed,'alphabetical':alphabetical,'value_sorted':sorted_true,'order_compliance':compliance,'evidence_forced':cond.startswith('P_'),'facts':reason})
Path('INDEPENDENT_FINDINGS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in out))
h=hashlib.sha256(Path('INDEPENDENT_FINDINGS.jsonl').read_bytes()).hexdigest()
Path('INDEPENDENT_FINDINGS.sha256').write_text(h+'  INDEPENDENT_FINDINGS.jsonl\n')
Path('STAGE2_RECEIPT.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),'independent_findings_sha256':h,'frozen_scores_read':False,'production_scorer_read_or_called':False,'gold_checks_all_pass':all(all(r['provided_gold_checks'].values()) for r in out)},indent=2)+'\n')
for r in out: print(r['ordinal'],r['A'],r['S'],r['B'],r['rank_order_known'],r['values_true'],r['print'],r['alphabetical'],r['order_compliance'],r['facts'])
print('SHA',h)
