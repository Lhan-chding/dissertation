#!/usr/bin/env python3
"""Read saved semantic fields in committed D records; no scoring or fitting."""
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict,Counter

READS=set()
def read_json(p):
    READS.add(str(p.resolve()));return json.loads(p.read_text())
def rows_file(p):
    READS.add(str(p.resolve()));out=[]
    for line in p.open():
        if line.strip():
            r=json.loads(line);out.append({k:r[k] for k in ['semantic','prompt_id','family','interface','draw_index']})
    return out
def write_csv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)
def local(raw,p):
    return raw/p.split('prospective_selection_v2_20260924/',1)[1]
def metric(rows):
    n=len(rows);valid=[r for r in rows if r['semantic']['event']!='I']
    z={'draws':n,'valid_n':len(valid)}
    for e in ['X','W','S','I']:
        z[e+'_n']=sum(r['semantic']['event']==e for r in rows);z['p'+e]=z[e+'_n']/n
    for k in ['F','B','M','copy']:
        z[k+'_sum_valid']=sum(r['semantic'][k] for r in valid)
        z[k+'_given_valid']=z[k+'_sum_valid']/len(valid) if valid else None
    z['C_sum']=sum(r['semantic']['relation_score'] for r in rows);z['C_mean']=z['C_sum']/n
    z['coordinate_correct_sum']=sum(r['semantic']['coord_accuracy']*4 for r in rows)
    z['coordinate_denominator']=4*n;z['coord_accuracy']=z['coordinate_correct_sum']/(4*n)
    z['relation_full_wrong_n']=sum(r['semantic']['relation_score']==1 and r['semantic']['event']!='X' for r in rows)
    z['relation_full_wrong']=z['relation_full_wrong_n']/n
    return z

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--out',type=Path,default=Path(__file__).parent);a=ap.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    docs=a.repo/'ssvc_flow/docs/prospective_selection';compact=read_json(docs/'first_four_20261004/compact_input.json');prepared=read_json(a.raw/'inputs/prepared_first_four_ONLY.json')
    promptmeta={p['prompt_id']:p for pool in [prepared['source_prompts'],prepared['continuation_prompts'],*prepared['panels'].values()] for p in pool}
    promptrows=[];grouprows=[];endpoint={};halves={};by_stage_prompt=defaultdict(list);inventory=[];index=[]
    for b in compact['branches']:
        pay=b['task']['payload'];origin=pay['origin_id'];rec=pay['recipe_id'];stage='early' if pay['origin_step']==32 else 'late'
        meta={'lineage_id':pay['lineage_id'],'source_recipe':pay['source_recipe'],'source_step':pay['origin_step'],'origin_id':origin,'stage':stage,'branch_recipe':rec}
        for h in [8,32]:
            cp=a.raw/f'campaign/branches/{origin}/{rec}/repeat_1/evaluations/H{h:02d}/COMPLETE.json';c=read_json(cp)
            assert c['status']=='EVALUATION_COMPLETE';assert c['identity']['panel_id'] in ['D','D_H8']
            rs=[]
            for chunk in c['chunks']:
                rr=rows_file(local(a.raw,chunk['samples']['path']));assert len(rr)==chunk['count'];rs.extend(rr)
            grouped=defaultdict(list)
            for r in rs:grouped[r['prompt_id']].append(r)
            expected=(24,8) if h==8 else (144,16);assert len(grouped)==expected[0];assert all(len(v)==expected[1] for v in grouped.values())
            m={**meta,'H':h,'panel':c['identity']['panel_id']}
            inventory.append({**m,'artifact':'endpoint','available':True,'prompts':len(grouped),'draws':len(rs),'path':str(cp),'authoritative_chunks':len(c['chunks'])})
            endpoint[origin,rec,h]=grouped
            for pid,rr in grouped.items():
                pr=promptmeta[pid];mt={**m,'base_scene_id':pr['base_scene_id'],'prompt_id':pid,'interface':pr['interface'],'family':pr['family'],'operation':pr['operation']}
                st=metric(rr);promptrows.append({**mt,**st});index.append({**mt,'draws':len(rr),'authoritative_file':'COMPLETE.chunks','complete':str(cp)})
                if h==32:by_stage_prompt[stage,pid].extend(r['semantic']['event']=='X' for r in rr)
            ss=defaultdict(list)
            for r in rs:ss[r['family'],r['interface']].append(r)
            for key,items in [('ALL',rs),*ss.items()]:
                fam,intf=('ALL','ALL') if key=='ALL' else key
                grouprows.append({**m,'aggregation':'origin','family':fam,'interface':intf,**metric(items)})
            assert abs(metric(rs)['pX']-b['evaluations'][str(h)]['summary']['J'])<1e-12
            if h==32:
                for half in [0,1]:
                    subset=[r for rr in grouped.values() for r in sorted(rr,key=lambda r:r['draw_index'])[half*8:(half+1)*8]]
                    halves[origin,rec,half]=metric(subset)['pX']
    classes={k:('unseen_X' if sum(v)==0 else 'all_X' if sum(v)==len(v) else 'mixed') for k,v in by_stage_prompt.items()}
    effects=[];promptmap={(r['origin_id'],r['branch_recipe'],r['H'],r['prompt_id']):r for r in promptrows}
    for r in promptrows:
        if r['H']!=32:continue
        ref=promptmap[r['origin_id'],'R0',32,r['prompt_id']];out={**r,'posthoc_H32_class':classes[r['stage'],r['prompt_id']],'fixed_weight':1/144,'R0_pX':ref['pX'],'delta_pX':r['pX']-ref['pX'],'contribution':(r['pX']-ref['pX'])/144}
        for k in ['C_mean','copy_given_valid','coord_accuracy','relation_full_wrong','F_given_valid','B_given_valid','M_given_valid']:
            out['delta_'+k]=r[k]-ref[k] if r[k] is not None and ref[k] is not None else None
        effects.append(out)
    by_action=defaultdict(list)
    for r in effects:by_action[r['origin_id'],r['branch_recipe']].append(r)
    concentration=[]
    for (origin,rec),rr in by_action.items():
        mag=sum(abs(x['contribution']) for x in rr);ordered=sorted(rr,key=lambda x:(-abs(x['contribution']),x['prompt_id']));cum=0
        for rank,r in enumerate(ordered,1):
            prior=cum;cum+=abs(r['contribution']);r['absolute_contribution_rank']=rank;r['cumulative_abs_share']=cum/mag if mag else 0;r['in_top_80pct_absolute_mass']=bool(mag and prior<.8*mag)
        concentration.append({'origin_id':origin,'stage':rr[0]['stage'],'branch_recipe':rec,'delta_pX':sum(x['contribution'] for x in rr),'absolute_mass':mag,'prompts_to_80pct_absolute_mass':sum(x['in_top_80pct_absolute_mass'] for x in rr),'positive_mass':sum(max(0,x['contribution']) for x in rr),'negative_mass':sum(min(0,x['contribution']) for x in rr),**{cl+'_contribution':sum(x['contribution'] for x in rr if x['posthoc_H32_class']==cl) for cl in ['unseen_X','mixed','all_X']}})
    # Stage pools retain numerator/denominator; each origin and prompt has equal design weight.
    pooled=defaultdict(list)
    for (origin,rec,h),groups in endpoint.items():
        stage='early' if origin.endswith('t32') else 'late'
        for rr in groups.values():
            r=rr[0];pooled[stage,rec,h,r['family'],r['interface']].extend(rr);pooled[stage,rec,h,'ALL','ALL'].extend(rr)
    for (stage,rec,h,fam,intf),rs in pooled.items():
        grouprows.append({'lineage_id':'ALL','source_recipe':'ALL','source_step':32 if stage=='early' else 96,'origin_id':'ALL','stage':stage,'branch_recipe':rec,'H':h,'panel':'D' if h==32 else 'D_H8','aggregation':'stage','family':fam,'interface':intf,**metric(rs)})
    baseline={(r['aggregation'],r['origin_id'],r['stage'],r['H'],r['family'],r['interface']):r for r in grouprows if r['branch_recipe']=='R0'}
    for r in grouprows:
        ref=baseline[r['aggregation'],r['origin_id'],r['stage'],r['H'],r['family'],r['interface']]
        for k in ['pX','C_mean','copy_given_valid','coord_accuracy','relation_full_wrong','F_given_valid','B_given_valid','M_given_valid']:r['delta_'+k]=r[k]-ref[k] if r[k] is not None and ref[k] is not None else None
    matched=[];overlaps=[]
    for b in compact['branches']:
        p=b['task']['payload'];o=p['origin_id'];rec=p['recipe_id'];g8=endpoint[o,rec,8];g32=endpoint[o,rec,32];common=sorted(g8.keys()&g32.keys());assert len(common)==24
        overlaps.extend({'origin_id':o,'branch_recipe':rec,'prompt_id':pid,'H8_draws':len(g8[pid]),'H32_draws':len(g32[pid])} for pid in common)
        for label,pids in [('ALL',common),*[(f+'/'+i,[pid for pid in common if promptmeta[pid]['family']==f and promptmeta[pid]['interface']==i]) for f,i in sorted({(promptmeta[x]['family'],promptmeta[x]['interface']) for x in common})]]:
            r8=metric([r for pid in pids for r in g8[pid]]);r32=metric([r for pid in pids for r in g32[pid]])
            matched.append({'origin_id':o,'lineage_id':p['lineage_id'],'stage':'early' if p['origin_step']==32 else 'late','source_step':p['origin_step'],'branch_recipe':rec,'group':label,'matched_prompts':len(pids),'n_H8':r8['draws'],'n_H32':r32['draws'],'pX_H8_matched':r8['pX'],'pX_H32_matched':r32['pX'],'delta_H32_minus_H8':r32['pX']-r8['pX'],'independent_replicate':False})
    # Fixed halves are a deterministic repartition, not new samples or train seeds.
    split=[];cands=['R'+str(i) for i in range(8)];recipes=sorted({p['task']['payload']['recipe_id'] for p in compact['branches']});origins=sorted({p['task']['payload']['origin_id'] for p in compact['branches']})
    for o in origins:
        for name,acts in [('R0_R7',cands),('all_11',recipes)]:
            best=[]
            for half in [0,1]:
                scores={r:halves[o,r,half] for r in acts};maxscore=max(scores.values());best.append('|'.join(r for r in acts if scores[r]==maxscore))
            for rec in acts:split.append({'origin_id':o,'stage':'early' if o.endswith('t32') else 'late','candidate_set':name,'branch_recipe':rec,'half1_pX':halves[o,rec,0],'half2_pX':halves[o,rec,1],'half1_delta_R0':halves[o,rec,0]-halves[o,'R0',0],'half2_delta_R0':halves[o,rec,1]-halves[o,'R0',1],'half1_maximizers':best[0],'half2_maximizers':best[1],'argmax_sets_equal':best[0]==best[1],'sign_reversal':(halves[o,rec,0]-halves[o,'R0',0])*(halves[o,rec,1]-halves[o,'R0',1])<0})
    J={(r['origin_id'],r['branch_recipe']):r['pX'] for r in grouprows if r['aggregation']=='origin' and r['H']==32 and r['family']=='ALL'}
    contrasts=[]
    for name,acts in [('R0_R7',cands),('all_11',recipes)]:
        hindsight=max(acts,key=lambda rec:sum(J[o,rec] for o in origins));
        for o in origins:
            lineage=o.split('_')[0];train=[z for z in origins if z.split('_')[0]!=lineage];fold=max(cands,key=lambda rec:(sum(J[z,rec] for z in train),-cands.index(rec)));oracle=max(acts,key=lambda rec:J[o,rec]);
            contrasts.append({'origin_id':o,'stage':'early' if o.endswith('t32') else 'late','candidate_set':name,'fold_BestStatic':fold,'fold_BestStatic_pX':J[o,fold],'hindsight_fixed':hindsight,'hindsight_fixed_pX':J[o,hindsight],'noisy_origin_max':oracle,'noisy_origin_max_pX':J[o,oracle],'R0_pX':J[o,'R0'],'noisy_opportunity_over_fold':J[o,oracle]-J[o,fold],'hindsight_over_R0':J[o,hindsight]-J[o,'R0'],'scope':'descriptive_external_baselines_not_added_to_saved_selector' if name=='all_11' else 'original_candidates'})
    classtable=[]
    for (stage,pid),values in sorted(by_stage_prompt.items()):
        pr=promptmeta[pid];classtable.append({'stage':stage,'prompt_id':pid,'base_scene_id':pr['base_scene_id'],'family':pr['family'],'interface':pr['interface'],'operation':pr['operation'],'posthoc_H32_class':classes[stage,pid],'X_n':sum(values),'draws':len(values),'nonX_n':len(values)-sum(values),'endpoints':44})
    # Metadata-only P and historical E index; do not reread their sample texts.
    for pre in compact['prestates']:
        o=pre['origin'];step=int(o.split('_t')[1]);pay=next(b['task']['payload'] for b in compact['branches'] if b['task']['payload']['origin_id']==o)
        for when,t in [('history',step-8),('current',step)]:
            for p in prepared['panels']['P']:index.append({'lineage_id':pay['lineage_id'],'source_recipe':pay['source_recipe'],'source_step':step,'observation_source_step':t,'origin_id':o,'stage':'early' if step==32 else 'late','branch_recipe':'PRESTATE','H':'','panel':'P','base_scene_id':p['base_scene_id'],'prompt_id':p['prompt_id'],'interface':p['interface'],'family':p['family'],'operation':p['operation'],'draws':32,'authoritative_file':'compact.prestates.packets.'+when,'complete':'reused_prior_verified_packet'})
    for x in compact['historical_E']:
        identity=x['evaluation']['identity'];task=x['task']['payload']
        inventory.append({'artifact':'E_old','available':True,'prompts':72,'draws':2304,'path':'compact_input.historical_E','identity':json.dumps(identity,sort_keys=True)})
        for p in prepared['panels']['E_old']:index.append({'lineage_id':identity.get('lineage_id','historical'),'source_recipe':'historical','source_step':'','origin_id':identity.get('origin_id',task.get('origin_id','')),'branch_recipe':task.get('recipe_id',task.get('recipe','')),'H':identity.get('horizon',''),'panel':'E_old','base_scene_id':p['base_scene_id'],'prompt_id':p['prompt_id'],'interface':p['interface'],'family':p['family'],'operation':p['operation'],'draws':32,'authoritative_file':'compact.historical_E','complete':'reused_prior_verified_summary'})
    for model in sorted((docs/'pilot_four_20261004/models').glob('*.json')):inventory.append({'artifact':'saved_selector','available':True,'path':str(model),'model':model.stem})
    for fname,rr in [('opportunity_by_stage_group.csv',grouprows),('prompt_effect_contributions.csv',effects),('prompt_effect_concentration.csv',concentration),('endpoint_prompt_metrics.csv',promptrows),('matched_panel_time.csv',matched),('matched_panel_ids.csv',overlaps),('existing_output_split_stability.csv',split),('action_opportunity_comparison.csv',contrasts),('prompt_posthoc_classes.csv',classtable),('unified_observation_index.csv',index),('artifact_inventory.csv',inventory)]:write_csv(a.out/fname,rr)
    (a.out/'inputs_B.json').write_text(json.dumps({'files':sorted(READS),'statistical_repartition':{'count':1,'method':'fixed draw_index 0..7 vs 8..15; no random resampling','scope':'88 H32 endpoints x 144 prompts x 16 saved outputs'},'missing':['Independent analysis package and its previous 8/8 split were not found; fixed split calculated here']},indent=2))
    print(json.dumps({'prompt_rows':len(promptrows),'effect_rows':len(effects),'inputs':len(READS),'classification':dict(Counter(r['stage']+'/'+r['posthoc_H32_class'] for r in classtable))},default=str))
if __name__=='__main__':main()
