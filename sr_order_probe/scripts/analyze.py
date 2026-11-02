#!/usr/bin/env python3
import argparse,csv,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from order_probe.common import read_jsonl,write_json
from order_probe.statistics import PRIMARY,SECONDARY,root_effects,paired_bootstrap,holm,descriptive

def main():
    p=argparse.ArgumentParser();p.add_argument('--scores',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()): raise SystemExit('Do not overwrite prior analyses')
    a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((R/'config/plan.json').read_text());rows=read_jsonl(a.scores)
    if len(rows)!=4320 or len({r['request_id'] for r in rows})!=4320:raise SystemExit('Need all 4320 main responses; incomplete is not zero effect')
    if len({r['root_id'] for r in rows})!=48:raise SystemExit('Wrong root denominator')
    if set(r['parent'] for r in rows)!=set(cfg['model']['parents']):raise SystemExit('Do not drop parents')
    result=[];root_rows=[]
    panels=[('POOLED_FIXED_PARENTS',list(cfg['model']['parents']))]+[(p,[p]) for p in cfg['model']['parents']]
    for panel,parents in panels:
        for metric in ('A','S_hit'):
            for contrast in PRIMARY+SECONDARY:
                roots,strata,values=root_effects(rows,metric,contrast,parents)
                stat=paired_bootstrap(values,strata,B=cfg['statistics']['bootstrap_replicates'],seed=cfg['statistics']['bootstrap_seed'])
                primary=panel=='POOLED_FIXED_PARENTS' and contrast in PRIMARY
                result.append({'panel':panel,'metric':metric,'contrast':contrast,'primary':primary,**stat})
                root_rows += [{'panel':panel,'metric':metric,'contrast':contrast,'root_id':r,'structure':s,'effect':float(v)} for r,s,v in zip(roots,strata,values)]
    primary=[r for r in result if r['primary']]
    adjusted=holm([r['p_approx_centered_bootstrap'] for r in primary])
    for r,padj in zip(primary,adjusted):r['p_holm_8']=padj
    write_json(a.out/'CONTRASTS.json',result);write_json(a.out/'DESCRIPTIVE.json',descriptive(rows))
    with (a.out/'ROOT_EFFECTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(root_rows[0]));writer.writeheader();writer.writerows(root_rows)
    write_json(a.out/'ANALYSIS_SCOPE.json',{'status':'DESCRIPTIVE_AND_PAIRED_ROOT_INFERENCE','optimizer_updates':0,
        'model_training_variance_estimated':False,'natural_mediation_identified':False,
        'primary_tests':8,'native_J_not_replaced':True,
        'note':'S_hit means confirmed correct/all assigned outputs. UNKNOWN is not assigned a semantic truth value.'})
    print('Analysis complete. No training authorization issued.')
if __name__=='__main__':main()
