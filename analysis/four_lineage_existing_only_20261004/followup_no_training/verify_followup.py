#!/usr/bin/env python3
"""Validate actual existing-data follow-up outputs without synthetic experiments."""
import csv,json,re,hashlib,ast,subprocess
from pathlib import Path
from collections import defaultdict
P=Path(__file__).resolve().parent

def read(name):return list(csv.DictReader((P/name).open()))
def main():
    checks={}
    for p in P.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    failure=read('failure_decomposition.csv');assert len(failure)==22935
    for r in failure:
        n=lambda k:int(r[k])
        assert min(n(k) for k in ['draws','valid_n','X_n','I_n','F1_n','copy_n','F0_noncopy_n','F1_damage_n','nonX_n'])>=0
        assert n('I_n')+n('copy_n')+n('F0_noncopy_n')+n('F1_damage_n')==n('nonX_n')==n('draws')-n('X_n')
        assert n('X_n')==n('F1_n')-n('F1_damage_n')
    delta=read('failure_delta_decomposition.csv');assert len(delta)==22935
    for r in delta:
        assert int(r['draws'])==int(r['R0_draws'])
        assert int(r['delta_X_n'])==int(r['delta_F1_n'])-int(r['delta_F1_damage_n'])
        assert abs(float(r['delta_X_pp'])-100*int(r['delta_X_n'])/int(r['draws']))<1e-12
    pooled=read('failure_pooled_summary.csv')
    late=[r for r in pooled if r['stage']=='late' and r['posthoc_H32_class']=='unseen_X' and r['family']=='ALL' and r['branch_recipe']=='ALL'];assert len(late)==1
    r=late[0];assert [int(r[k]) for k in ['draws','valid_n','I_n','copy_n','F0_noncopy_n','F1_n','F1_damage_n']]==[19712,17529,2183,9150,8379,0,0]
    checks['failure_count_identities']={'rows':len(failure),'delta_rows':len(delta),'late_unseen_X_F1':0,'late_unseen_X_valid':17529,'status':'PASS'}
    attrib=read('saved_Z3_Z2_attribution.csv');assert len(attrib)==64;maxerr=0.
    for r in attrib:
        for base in ['R0','BestStatic']:
            d=float(r['delta_prediction_'+base])
            for route in ['forward','reverse']:
                parts=float(r['mean_change_'+base])+sum(float(r[route+'_'+comp+'_'+base]) for comp in ['current','history','repair','repair_history','nuisance','beta_change'])
                maxerr=max(maxerr,abs(d-parts))
    assert maxerr<1e-15
    sums=read('saved_Z3_Z2_attribution_summary.csv');rr=next(r for r in sums if r['origin']=='ALL' and r['baseline']=='BestStatic' and r['path']=='forward')
    assert int(rr['opposite_sign_actions'])==int(rr['n_nonzero_actions'])==56
    assert abs(float(rr['aggregate_cancellation_fraction'])-.95820645)<1e-8
    checks['saved_prediction_attribution']={'rows':64,'replay':json.loads((P/'attribution_scope.json').read_text())['checks'],'independent_identity_residual':maxerr}
    stability=read('conditional_ranking_stability.csv');assert len(stability)==152
    groups=defaultdict(list)
    for r in stability:
        groups[r['origin_id'],r['candidate_set']].append(r)
        for ref in ['R0','BestStatic']:
            assert abs(sum(float(r['delta_'+ref+'_'+s+'_fraction']) for s in ['positive','negative','zero'])-1)<1e-12
            assert float(r['delta_'+ref+'_p025'])<=float(r['delta_'+ref+'_p975'])
            if r['branch_recipe']==('R0' if ref=='R0' else r['original_BestStatic']):assert float(r['delta_'+ref+'_zero_fraction'])==1
    assert len(groups)==16
    for rr in groups.values():assert abs(sum(float(r['top_fraction_ties_split']) for r in rr)-1)<1e-12
    margins=read('ranking_margin_prompt_contributions.csv');totals=defaultdict(float)
    for r in margins:totals[r['origin_id'],r['candidate_set']]+=float(r['margin_contribution'])
    for r in read('ranking_margin_summary.csv'):assert abs(totals[r['origin_id'],r['candidate_set']]-float(r['observed_margin']))<1e-12
    transfer=read('existing_half_winner_transfer.csv');assert len(transfer)==32
    checks['conditional_ranking']={'action_rows':152,'origin_set_groups':16,'half_transfer_rows':32,'counts_and_margins':'PASS'}
    for name in ['FOLLOWUP_REPORT_zh.md']:
        for dest in re.findall(r'\]\(([^)]+)\)',(P/name).read_text()):
            if dest=='VALIDATION.json':continue
            assert (P/dest).exists(),dest
    for py,manifest in [('failure_algebra.py','failure_scope.json'),('prediction_attribution.py','attribution_scope.json'),('ranking_stability.py','ranking_scope.json')]:
        saved=json.loads((P/manifest).read_text());assert saved['script_sha256']==hashlib.sha256((P/py).read_bytes()).hexdigest()
    scope=json.loads((P/'EXECUTION_SCOPE.json').read_text())
    for k in ['new_training_runs','new_model_calls','new_samples','selector_refits','hyperparameter_searches','gpu_jobs_submitted','final_T_records_read']:assert scope[k]==0
    checks['links_script_hashes_scope']='PASS'
    result={'status':'PASS','code_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),'scope':'Actual saved and derived outputs only; no model execution','checks':checks}
    (P/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
