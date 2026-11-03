# -*- coding: utf-8 -*-
import json,pathlib,hashlib,datetime
from fractions import Fraction
p=pathlib.Path(__file__).parent
extract=[json.loads(x) for x in (p/'EXTRACTION.jsonl').read_text().splitlines()]
priv={r['request_id']:r['task'] for r in map(json.loads,(p/'PRIVATE.jsonl').read_text().splitlines())}
def num(x):
 if isinstance(x,bool) or not isinstance(x,(int,float,str)): return None
 try: return Fraction(str(x))
 except (ValueError,ZeroDivisionError):return None
out=[]
for e in extract:
 t=priv[e['request_id']];w=t['world'];q=t['query'];cats=w['categories'];av=w['series']['Alpha'];bv=w['series']['Beta'];sign=1 if q['direction']=='smallest' else -1
 rank=sorted(range(len(cats)),key=lambda j:(sign*av[j],j));picked=rank[:q['k']];selected=[cats[j] for j in picked];bind={cats[j]:str(Fraction(bv[j])) for j in picked};a=sum((Fraction(bv[j]) for j in picked),Fraction(0));a=a/q['k'] if q['aggregate']=='mean' else a
 gold=t['gold'];gold_checks={'answer':num(gold['answer'])==a,'selected':set(gold['selected'])==set(selected) and len(gold['selected'])==len(selected),'indices':set(gold['indices'])==set(picked),'operands_by_category':dict(zip(gold['selected'],map(num,gold['operands'])))=={c:num(v) for c,v in bind.items()}}
 s=e['selected'];known=e['complete'] and isinstance(s,list) and all(isinstance(c,str) for c in s)
 S=('PASS' if len(s)==len(set(s)) and set(s)==set(selected) else 'FAIL') if known else 'UNKNOWN'
 A=int(e['strict_single_json'] and num(e['answer'])==a)
 bindings=e['bindings'];bmap={};bvalid=isinstance(bindings,list)
 if bvalid:
  for b in bindings:
   if not isinstance(b,dict) or not isinstance(b.get('category'),str) or b['category'] in bmap or num(b.get('operand')) is None: bvalid=False;break
   bmap[b['category']]=num(b['operand'])
 B=int(bvalid and set(bmap)==set(bind) and all(bmap[c]==num(v) for c,v in bind.items()))
 order=e['alpha_order'];rk=bool(e['strict_single_json'] and isinstance(order,list) and len(order)==10 and len(set(order))==10 and set(order)==set(cats))
 values=all(num(item['value'])==Fraction(av[cats.index(item['category'])]) for item in e['alpha_evidence']) if rk else None
 pr=order==cats if rk else None;al=order==sorted(cats) if rk else None
 vr=all(sign*av[cats.index(order[j])]<=sign*av[cats.index(order[j+1])] for j in range(9)) if rk else None
 cond=e['request_id'].split('/')[2];compliance=pr if cond=='C_PRINT' else al if cond=='C_ALT' else None
 threshold=av[picked[-1]];block=[cats[j] for j in range(10) if av[j]==threshold];observed_block=[c for c in order if c in block] if rk else None
 reason=f"From {q['direction']} Alpha with printed-position tie-break: {selected}; Beta={bind}; {q['aggregate']}={a}. "
 if not e['strict_single_json']:reason+='Strict raw JSON invalid: '+e['notes']+' A=0, S=UNKNOWN, B=0; all evidence-order metrics unknown.'
 else:
  reason+=f"Final answer={e['answer']!r}: A={A}; selected={s!r}: S={S}; bindings on task-correct set: B={B}. Alpha has ten unique categories; values_true={values}, print={pr}, alphabetical={al}, true_value_ranked={vr}."
 out.append({'row':e['row'],'request_id':e['request_id'],'derived_gold':{'selected_rank_order':selected,'indices_rank_order':picked,'bindings':bind,'answer':str(a)},'provided_gold_checks':gold_checks,'A':A,'S':S,'S_hit':int(S=='PASS'),'B':B,'rank_order_known':rk,'rank_values_true':values,'rank_is_print':pr,'rank_is_alphabetical':al,'rank_is_value_sorted':vr,'order_compliance':compliance,'critical_tie_print_order':observed_block==block if rk else None,'evidence_forced':cond.startswith('P_'),'reason':reason})
(p/'INDEPENDENT_FINDINGS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
digest=hashlib.sha256((p/'INDEPENDENT_FINDINGS.jsonl').read_bytes()).hexdigest()
(p/'STAGE2_RECEIPT.json').write_text(json.dumps({'stage':'independent world/query judgment frozen before FROZEN_SCORES read','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(out),'findings_sha256':digest,'all_provided_gold_checks_pass':all(all(r['provided_gold_checks'].values()) for r in out),'production_scorer_read_or_called':False},indent=2))
print('FINDINGS SHA256',digest)
for r in out: print(r['row'],r['A'],r['S'],r['B'],r['rank_order_known'],r['derived_gold']['answer'])
