#!/usr/bin/env python3
"""Acceptance checks on actual saved/derived evidence, no synthetic experiments."""
import csv,json,math,re,collections
from pathlib import Path
P=Path(__file__).resolve().parent

def rows(n):return list(csv.DictReader((P/n).open()))
def main():
 checks={};o=rows('opportunity_by_stage_group.csv');effects=rows('prompt_effect_contributions.csv');idx=rows('unified_observation_index.csv')
 assert len(effects)==8*11*144
 by=collections.defaultdict(float)
 for r in effects:
  assert float(r['fixed_weight'])==1/144;by[r['origin_id'],r['branch_recipe']]+=float(r['contribution'])
 err=max(abs(by[r['origin_id'],r['branch_recipe']]-float(r['delta_pX'])) for r in o if r['aggregation']=='origin' and r['H']=='32' and r['family']=='ALL');assert err<1e-12;checks['contribution_sum_max_error']=err
 assert all('C_mean' in r and 'pC' not in r for r in o)
 for r in o:
  n=int(r['draws']);assert sum(int(r[e+'_n']) for e in ['X','S','W','I'])==n
  assert abs(float(r['C_sum'])/n-float(r['C_mean']))<1e-12
  assert int(r['coordinate_denominator'])==4*n
  for k in ['F','B','M','copy']:
   assert (r[k+'_given_valid']=='')==(int(r['valid_n'])==0)
   if r[k+'_given_valid']:assert abs(float(r[k+'_sum_valid'])/int(r['valid_n'])-float(r[k+'_given_valid']))<1e-12
 checks['metric_numerators_denominators']='PASS'
 cl=rows('prompt_posthoc_classes.csv');counts=collections.Counter(r['stage']+'/'+r['posthoc_H32_class'] for r in cl);assert counts=={'early/all_X':43,'early/unseen_X':20,'early/mixed':81,'late/all_X':108,'late/unseen_X':28,'late/mixed':8};checks['posthoc_classes']=dict(counts)
 assert all(float(r['contribution'])==0 for r in effects if r['posthoc_H32_class']!='mixed')
 matched=rows('matched_panel_ids.csv');assert len(matched)==88*24;assert all((r['H8_draws'],r['H32_draws'])==('8','16') for r in matched)
 checks['H8_H32_intersection']='88 x 24 prompt IDs; 8 vs 16 saved draws'
 pred=rows('saved_prediction_decomposition.csv');assert len(pred)==256
 for r in pred:assert abs(float(r['prediction_delta_R0'])-float(r['mean_delta_R0'])-float(r['state_correction']))<1e-12
 checks['saved_model_replay']=json.loads((P/'inputs_D.json').read_text())['checks']
 q=rows('action_opportunity_comparison.csv');core={r['origin_id']:r for r in q if r['candidate_set']=='R0_R7'};assert len(core)==8
 old=list(csv.DictReader((P.parents[1]/'ssvc_flow/docs/prospective_selection/pilot_four_20261004/selected_outcomes.csv').open()))
 assert all(core[r['origin']]['fold_BestStatic']==r['best_static'] for r in old)
 assert all(r['fold_BestStatic']==core[r['origin_id']]['fold_BestStatic'] for r in q);checks['external_baselines_not_in_original_selector']='PASS'
 g=rows('training_group_detail_CF.csv');branch=[r for r in g if r['scope']=='branch_training'];assert len(branch)==11264
 for r in g:assert int(r['all_X_n'])+int(r['no_X_n'])+int(r['mixed_X_n'])==1
 for st,no,f1 in [('early',897,125),('late',845,0)]:
  rr=[r for r in branch if r['source_stage']==st];assert len(rr)==5632;assert sum(int(r['no_X_n']) for r in rr)==no;assert sum(int(r['no_X_with_F1_n']) for r in rr)==f1
 late=[r for r in branch if r['source_stage']=='late' and r['no_X_n']=='1'];assert sum(r['no_X_with_F_observed_n']=='0' for r in late)==39
 checks['training_support']='11264 branch groups; early/late no-X and F1 counts; 39 all-I groups verified'
 resp=rows('support_response_summary.csv');ev=[r for r in resp if r.get('eval_H32_pX','')!=''];assert len(ev)==168;checks['train_eval_common_strata_rows']=len(ev)
 pre=rows('prestate_information_vs_response.csv');assert len(pre)==8;assert len({r['origin_id'] for r in pre})==8
 cond=rows('prestate_reward_conditional_repair.csv');assert len(cond)==2960
 checks['prestate_point_and_atom_coverage']={'origins':8,'atom_rows':2960}
 scope=json.loads((P/'EXECUTION_SCOPE.json').read_text());assert all(scope[k]==0 for k in ['new_training_runs','new_model_calls','new_samples','selector_refits','hyperparameter_searches','gpu_jobs_submitted','final_T_records_read'])
 required=['ANALYSIS_REPORT_zh.md','FINDINGS_FOR_DISCUSSION_zh.md','input_scope.json','artifact_inventory.csv','opportunity_by_stage_group.csv','prompt_effect_contributions.csv','support_timeline.csv','support_response_summary.csv','saved_prediction_decomposition.csv','kernel_geometry.csv','existing_tuning_diagnostics.csv','prestate_reward_conditional_repair.csv','prestate_information_vs_response.csv','matched_panel_time.csv','cost_existing_only.csv']
 assert all((P/n).is_file() for n in required);checks['required_deliverables']=len(required)
 for name in ['ANALYSIS_REPORT_zh.md','FINDINGS_FOR_DISCUSSION_zh.md','REPRODUCE_zh.md']:
  for link in re.findall(r'\]\(([^)]+)\)',(P/name).read_text()):
   if link!='VALIDATION.json' and not link.startswith(('http','#')):assert (P/link).exists(),(name,link)
 checks['local_report_links']='PASS'
 (P/'VALIDATION.json').write_text(json.dumps({'status':'PASS','scope':'actual existing and derived tables; no new experiments','checks':checks},ensure_ascii=False,indent=2)+'\n');print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
