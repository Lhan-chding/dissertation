"""No inference. Exercise all frozen evaluation identities with deterministic artificial outputs."""
from pathlib import Path
import sys,json,collections,argparse
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import score_eval as se, analyze as an
def read(f):return list(an.load_rows(f))
def main(path):
    inp=se.ids(read(path/'inputs.jsonl'),'input_id',('image_file','prompt','assistant_prefill'))
    cs=se.ids(read(path/'private_cases.jsonl'),'case_id',('input_id','task','audit')); er=read(path/'evaluation_manifest.jsonl'); dev=read(path/'clarification_manifest.jsonl')
    def raw(m):
        c=cs[m['case_id']];p=inp[m['input_id']];v=c['task'];a=c['audit'];w=v['world'];g=v['gold'];names=w['categories'];order=a.get('expected_report_order') or names
        obj={'evidence':[dict(series=k,category=n,value=col[names.index(n)]) for k,col in w['series'].items() for n in order], 'selected':g['selected'],'bindings':[dict(category=n,operand=x) for n,x in zip(g['selected'],g['operands'])],'aggregate':v['query']['aggregate'],'answer':g['answer']}
        if p['assistant_prefill']=='{':txt=json.dumps(obj,separators=(',',':'))
        else:del obj['evidence'];txt=p['assistant_prefill']+json.dumps(obj,separators=(',',':'))[1:]
        return {**m,'finish_reason':'balanced','full_text':txt,'assistant_prefill':p['assistant_prefill'],'suffix_text':txt[len(p['assistant_prefill']):]}
    cache={}; scores=[]
    for m in er:
        k=m['case_id']
        if k not in cache:
            q=se.score_one(m,raw(m),cs[k],inp[m['input_id']])
            if not (q['A']==q['S_hit']==q['B']==1 and q['U'] in (1,None) and q['J_native'] in (1,None)):raise ValueError('gold failed '+k)
            cache[k]=q
        scores.append({**cache[k],**m})
    a=an.run(er,scores)
    if a['primary']['estimate']!=0:raise AssertionError(a['primary'])
    se.validate_dev(dev)
    for m in dev:
        c=cs[m['case_id']];q=se.score_one(m,raw(m),c,inp[m['input_id']]);assert q['A']==q['S_hit']==1
    return {'status':'SYNTHETIC_CPU_INTEGRATION_ONLY','real_generations':0,'parameter_updates':0,'cases_scored_with_constructed_gold_text':len(cache),'formal_eval_rows_joined_and_analyzed':len(er),'clarification_rows_checked':len(dev),'artificial_primary_D_minus_G':a['primary']['estimate'],'note':'This is a scoring/oracle/manifest consistency test; synthetic repeated outputs give no scientific evidence.'}
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--frozen',type=Path,default=BASE/'frozen');a=ap.parse_args();print(json.dumps(main(a.frozen),indent=2))
