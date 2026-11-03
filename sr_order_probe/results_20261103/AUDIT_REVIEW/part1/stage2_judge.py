import json,pathlib,hashlib,datetime
from fractions import Fraction
p=pathlib.Path(__file__).parent
e=[json.loads(x) for x in (p/'EXTRACTION.jsonl').read_text().splitlines()]
tasks={x['request_id']:x['task'] for x in map(json.loads,(p/'PRIVATE.jsonl').read_text().splitlines())}
def num(v):
 if isinstance(v,bool) or not isinstance(v,(int,float,str)): return None
 try:return Fraction(str(v))
 except (ValueError,ZeroDivisionError):return None
out=[]
for x in e:
 t=tasks[x['request_id']];w=t['world'];q=t['query'];cs=w['categories'];av=list(map(num,w['series'][q['selection_series']]));bv=list(map(num,w['series'][q['value']]))
 ordered=sorted(range(len(cs)),key=lambda i:((-av[i] if q['direction']=='largest' else av[i]),i));ix=sorted(ordered[:q['k']]);goldsel=[cs[i] for i in ix];operands={cs[i]:str(bv[i]) for i in ix};ans=sum(bv[i] for i in ix)/(q['k'] if q['aggregate']=='mean' else 1)
 gold=t['gold'];goldmatch={'selected':gold['selected']==goldsel,'indices':gold['indices']==ix,'operands':list(map(num,gold['operands']))==[bv[i] for i in ix],'answer':num(gold['answer'])==ans}
 A=int(x['strict_single_json_object'] and num(x['answer'])==ans)
 sel=x['selected'];S='UNKNOWN' if not x['complete'] or not x['selected_string_array'] else ('PASS' if len(sel)==len(set(sel)) and set(sel)==set(goldsel) else 'FAIL')
 bd=x['bindings'];bmap={};valid=isinstance(bd,list)
 if valid:
  for b in bd:
   if not isinstance(b,dict) or not isinstance(b.get('category'),str) or b['category'] in bmap or num(b.get('operand')) is None:valid=False;break
   bmap[b['category']]=num(b['operand'])
 B=int(valid and set(bmap)==set(goldsel) and all(bmap.get(cs[i])==bv[i] for i in ix))
 ao=x['alpha_order'];ae=x['alpha_evidence'];known=bool(x['complete'] and ao is not None and len(ao)==len(cs) and len(set(ao))==len(cs) and set(ao)==set(cs))
 vals=all(num(z.get('value'))==av[cs.index(z['category'])] for z in ae) if known else None
 pr=ao==cs if known else None;al=ao==sorted(cs) if known else None
 numeric=all((av[cs.index(ao[j])] >= av[cs.index(ao[j+1])] if q['direction']=='largest' else av[cs.index(ao[j])] <= av[cs.index(ao[j+1])]) for j in range(len(cs)-1)) if known else None
 condition=x['request_id'].split('/')[2];comp=pr if condition=='C_PRINT' else al if condition=='C_ALT' else None
 bad_binding=[c for c in goldsel if bmap.get(c)!=num(operands[c])]
 reason=f"Sort Alpha {q['direction']} with original plot position as tie-break; required {goldsel}, Beta {operands}, {q['aggregate']}={ans}. Final answer={x['answer']!r}; selected={sel!r}; missing/wrong gold bindings={bad_binding}. Alpha category count={len(ao) if ao is not None else 'unparsed'}, complete unique coverage={known}."
 if not x['strict_single_json_object']: reason+=' Whole raw JSON invalid (missing closing brace); no repair/field salvage.'
 if x['index']==18:reason+=' Missing selected remains UNKNOWN despite work text.'
 if x['index']==14:reason+=' Work mentions 174 before self-correction; final answer 216 alone determines A.'
 out.append(dict(request_id=x['request_id'],index=x['index'],A=A,S=S,S_hit=int(S=='PASS'),B=B,rank_order_known=known,values_true=vals,print=pr,alphabetical=al,numeric_sorted=numeric,order_compliance=comp,derived_gold={'indices':ix,'selected':goldsel,'operands':operands,'answer':str(ans)},provided_gold_verified=goldmatch,alpha_evidence_forced=condition.startswith('P_'),reason=reason))
(p/'INDEPENDENT_FINDINGS.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
h=hashlib.sha256((p/'INDEPENDENT_FINDINGS.jsonl').read_bytes()).hexdigest()
(p/'INDEPENDENT_FINDINGS.sha256').write_text(h+'  INDEPENDENT_FINDINGS.jsonl\n')
(p/'STAGE2_RECEIPT.json').write_text(json.dumps(dict(stage=2,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=len(out),findings_sha256=h,frozen_scores_read=False,production_scorer_used=False,all_provided_gold_verified=all(all(x['provided_gold_verified'].values()) for x in out)),indent=2)+'\n')
print('sealed findings',len(out),h)
for x in out: print(x['index'],x['A'],x['S'],x['B'],x['rank_order_known'],x['values_true'],x['print'],x['alphabetical'],x['order_compliance'])
