#!/usr/bin/env python3
"""Conditional resampling and fixed-half diagnostics of existing H32 outputs.
No new model observations, no fitted selector, no features/weights/alpha tuning.
Binomial(16, saved X_n/16) is the exact sufficient-statistic implementation of
resampling 16 binary X indicators with replacement from each existing prompt.
"""
import argparse,csv,json,hashlib,subprocess,sys
from pathlib import Path
from collections import defaultdict
import numpy as np

P=Path(__file__).resolve().parent;PREV=P.parent

def read(name):return list(csv.DictReader((PREV/name).open()))
def write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repetitions',type=int,default=10000);parser.add_argument('--seed',type=int,default=20261004);a=parser.parse_args()
    assert a.repetitions>=1000
    data=[r for r in read('endpoint_prompt_metrics.csv') if r['H']=='32']
    baseline={r['origin_id']:r['fold_BestStatic'] for r in read('action_opportunity_comparison.csv') if r['candidate_set']=='R0_R7'}
    class_by={(r['stage'],r['prompt_id']):r['posthoc_H32_class'] for r in read('prompt_posthoc_classes.csv')}
    core=['R'+str(i) for i in range(8)];actions=core+['DIRECT_REPAIR_R4','GDPO_R4','SAW_R4'];origins=sorted(baseline)
    rng=np.random.default_rng(a.seed);output=[];summaries=[];margin_rows=[];checks={};delta_error=0
    for origin in origins:
        rs=[r for r in data if r['origin_id']==origin];stage=rs[0]['stage'];pids=sorted({r['prompt_id'] for r in rs});assert len(pids)==144
        ix={(r['branch_recipe'],r['prompt_id']):r for r in rs};assert len(ix)==11*144
        successes=np.array([[int(ix[act,p]['X_n']) for p in pids] for act in actions]);assert all(r['draws']=='16' for r in rs)
        point=successes.sum(axis=1)/(144*16);ref=actions.index(baseline[origin]);r0=actions.index('R0')
        boot=np.empty((a.repetitions,11))
        for start in range(0,a.repetitions,500):
            b=min(500,a.repetitions-start)
            boot[start:start+b]=rng.binomial(16,successes[None,:,:]/16,size=(b,11,144)).sum(axis=2)/(144*16)
        for label,subset in [('R0_R7',np.arange(8)),('all_11',np.arange(11))]:
            vals=boot[:,subset];maxima=vals.max(axis=1);ties=vals==maxima[:,None];fraction=ties/ties.sum(axis=1)[:,None]
            pmax=point[subset].max();observed_tops=[actions[j] for j in subset if point[j]==pmax]
            sorted_idx=sorted(subset,key=lambda j:(-point[j],actions[j]));top,runner=sorted_idx[:2]
            contrast=boot[:,top]-boot[:,runner];mean_delta=point[top]-point[runner]
            variance_unbiased=np.sum((successes[top]/16)*(1-successes[top]/16)/15+(successes[runner]/16)*(1-successes[runner]/16)/15)/(144**2)
            summary={'origin_id':origin,'stage':stage,'candidate_set':label,'original_BestStatic':baseline[origin],'observed_maximizers':'|'.join(observed_tops),'lexical_top_for_margin':actions[top],'lexical_runner_for_margin':actions[runner],'observed_margin':mean_delta,'fixed_panel_MC_SE_top_runner':np.sqrt(variance_unbiased),'resampled_top_runner_p025':np.quantile(contrast,.025),'resampled_top_runner_p975':np.quantile(contrast,.975),'resampled_top_outranks_runner_fraction':np.mean(contrast>0),'resampled_top_runner_tie_fraction':np.mean(contrast==0),'observed_max_set_retained_any_fraction':np.mean(np.any(ties[:,[k for k,j in enumerate(subset) if actions[j] in observed_tops]],axis=1)),'resampling_repetitions':a.repetitions}
            summaries.append(summary)
            for k,j in enumerate(subset):
                row={'origin_id':origin,'stage':stage,'candidate_set':label,'branch_recipe':actions[j],'observed_pX':point[j],'original_BestStatic':baseline[origin],'observed_delta_R0':point[j]-point[r0],'observed_delta_BestStatic':point[j]-point[ref],'top_fraction_ties_split':float(fraction[:,k].mean()),'top_membership_fraction':float(ties[:,k].mean()),'resampling_repetitions':a.repetitions}
                for tag,reference in [('R0',r0),('BestStatic',ref)]:
                    d=boot[:,j]-boot[:,reference];row.update({f'delta_{tag}_p025':np.quantile(d,.025),f'delta_{tag}_p975':np.quantile(d,.975),f'delta_{tag}_positive_fraction':np.mean(d>0),f'delta_{tag}_zero_fraction':np.mean(d==0),f'delta_{tag}_negative_fraction':np.mean(d<0)})
                output.append(row)
            # Exact existing top-runner margin contribution bookkeeping, no altered
            # prompt panel evaluated and no posthoc feature fed into any selector.
            contributions=(successes[top]-successes[runner])/(16*144);assert abs(contributions.sum()-mean_delta)<1e-12
            abs_total=float(np.abs(contributions).sum());order=np.argsort(-np.abs(contributions),kind='stable');cumulative=0.
            for rank,i in enumerate(order,1):
                cumulative+=abs(contributions[i]);r=ix[actions[top],pids[i]]
                margin_rows.append({'origin_id':origin,'stage':stage,'candidate_set':label,'top':actions[top],'runner':actions[runner],'prompt_id':pids[i],'base_scene_id':r['base_scene_id'],'family':r['family'],'interface':r['interface'],'operation':r['operation'],'posthoc_class':class_by[stage,pids[i]],'X_top':int(successes[top,i]),'X_runner':int(successes[runner,i]),'draws_each':16,'fixed_weight':1/144,'margin_contribution':contributions[i],'observed_margin':mean_delta,'absolute_contribution_rank':rank,'abs_mass_fraction':abs(contributions[i])/abs_total if abs_total else 0.,'cumulative_abs_mass_fraction':cumulative/abs_total if abs_total else 0.,'individual_abs_term_exceeds_net_margin':bool(abs(contributions[i])>abs(mean_delta)+1e-14)})
            summary['positive_margin_mass']=float(np.maximum(contributions,0).sum());summary['negative_margin_mass']=float(np.minimum(contributions,0).sum());summary['nonzero_margin_prompts']=int(np.count_nonzero(contributions));summary['top_two_abs_mass_fraction']=float(np.abs(contributions)[order[:2]].sum()/abs_total) if abs_total else 0.;summary['individual_abs_term_exceeds_net_margin_n']=int(sum(abs(contributions)>abs(mean_delta)+1e-14))
    write('conditional_ranking_stability.csv',output);write('ranking_margin_summary.csv',summaries);write('ranking_margin_prompt_contributions.csv',margin_rows)
    # Existing half-table reuse: a descriptive split winner, evaluated on the
    # opposite saved half; this neither refits nor changes any deployed selector.
    split=read('existing_output_split_stability.csv');splitout=[]
    for origin in origins:
        for label in ['R0_R7','all_11']:
            rs=[r for r in split if r['origin_id']==origin and r['candidate_set']==label];d={r['branch_recipe']:r for r in rs};default=baseline[origin]
            for select,test in [(1,2),(2,1)]:
                peak=max(float(r[f'half{select}_pX']) for r in rs);winners=[r['branch_recipe'] for r in rs if float(r[f'half{select}_pX'])==peak]
                evaluation=sum(float(d[w][f'half{test}_pX']) for w in winners)/len(winners);def_train=float(d[default][f'half{select}_pX']);def_test=float(d[default][f'half{test}_pX'])
                splitout.append({'origin_id':origin,'stage':rs[0]['stage'],'candidate_set':label,'select_half':select,'evaluate_half':test,'posthoc_half_winners':'|'.join(winners),'tie_policy':'equal average over all tied maxima; no new samples or action executed','original_BestStatic':default,'winner_selection_half_pX':peak,'same_winners_other_half_pX':evaluation,'selection_half_delta_static':peak-def_train,'other_half_delta_static':evaluation-def_test,'selection_minus_other_half_delta':(peak-def_train)-(evaluation-def_test)})
    write('existing_half_winner_transfer.csv',splitout)
    for o in origins:
        for s in ['R0_R7','all_11']:assert abs(sum(r['top_fraction_ties_split'] for r in output if r['origin_id']==o and r['candidate_set']==s)-1)<1e-12
    receipt={'status':'PASS','actual_inputs':[str(PREV/n) for n in ['endpoint_prompt_metrics.csv','action_opportunity_comparison.csv','prompt_posthoc_classes.csv','existing_output_split_stability.csv']],'command':f'{sys.executable} {Path(__file__).resolve()} --repetitions {a.repetitions} --seed {a.seed}','source_analysis_commit':'2898c6f3f7874cc045a1421145eb9f5c485e249f','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy':np.__version__,'statistical_resampling':{'method':'within endpoint/prompt empirical binary bootstrap with replacement, implemented exactly via Binomial(16,X_n/16)','seed':a.seed,'joint_replicates_per_origin':a.repetitions,'origins':8,'total_origin_replicates':a.repetitions*8,'candidate_sets_share_same_draw_resamples':True,'resampled_saved_H32_records':202752,'independence_assumption':'draws within each fixed prompt/endpoint and across endpoints; trained states/panel held fixed','observed_zero_or_one_probabilities_remain_degenerate':True},'checks':{'margin_decomposition':'PASS','tie_split_probability_conservation':'PASS','core_rows':64,'all11_rows':88,'split_transfer_rows':32},'new_training_runs':0,'new_model_calls':0,'new_samples':0,'selector_refits':0,'hyperparameter_searches':0,'gpu_jobs_submitted':0,'final_T_records_read':0,'limits':['conditional resampling frequencies are not population confidence or posterior probabilities','no unseen successes can be created for X_n=0; no uncertainty about trained states or training seeds quantified','original saved fold BestStatic remains fixed; no refit on bootstrap','winner transfer is a posthoc diagnostic on original 8/8 outputs, not a newly validated selection policy','prompt contribution bookkeeping does not recompute a changed evaluation target']}
    (P/'ranking_scope.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt['checks']))
if __name__=='__main__':main()
