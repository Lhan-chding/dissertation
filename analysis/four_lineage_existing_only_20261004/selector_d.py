#!/usr/bin/env python3
"""Existing saved selector algebra only: never fit, solve, tune, or access raw T."""
import argparse,csv,json,sys,subprocess,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=ap.parse_args();root=args.root.resolve();out=Path(__file__).resolve().parent
 pilot=root/'ssvc_flow/docs/prospective_selection/pilot_four_20261004'; inputs=[]
 def read(p):
  inputs.append(str(p));return json.loads(p.read_text())
 def rows(p):
  inputs.append(str(p));return list(csv.DictReader(p.open()))
 def write(name,rr):
  keys=list(dict.fromkeys(k for r in rr for k in r))
  with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rr)
 code=read(pilot/'CODE_VERSION.json');commit=code['git_commit'];version={}
 for rel in ['features.py','kernels.py','semantics.py','selectors.py']:
  p=root/'ssvc_flow/src/prospective_selection'/rel;inputs.append(str(p));old=subprocess.check_output(['git','show',f'{commit}:ssvc_flow/src/prospective_selection/{rel}'],cwd=root)
  assert old==p.read_bytes(),f'experiment source differs: {rel}'
  version[rel]={'experiment_commit':commit,'byte_identical':True,'sha256':hashlib.sha256(old).hexdigest()}
 sys.path.insert(0,str(root/'ssvc_flow'))
 from src.prospective_selection.features import PreDecisionPacket,LEVELS
 from src.prospective_selection.kernels import FeatureKernel,_dot
 compact=read(root/'ssvc_flow/docs/prospective_selection/first_four_20261004/compact_input.json')
 packets={(e['origin'],l):p for e in compact['prestates'] for l,p in e['packets'].items()}
 truth={}
 for b in compact['branches']:
  p=b['task']['payload'];truth[p['origin_id'],p['recipe_id']]=b['evaluations']['32']['summary']['J']
 existing={(r['level'],r['origin'],r['recipe']):float(r['predicted_delta_R0']) for r in rows(pilot/'all_candidate_predictions.csv')}
 inner=rows(pilot/'inner_tuning_288.csv');sens=rows(pilot/'alpha_sensitivity_POSTHOC.csv');dec=read(pilot/'decisions_before_scoring.json')
 predictions=[];geometry=[];tuning=[];pair_rows=[];maxerr=0.;maxgram=0.;maxmean=0.;maxresidual=0.
 def rank(p):return sorted(p,key=lambda a:(-p[a],a))
 def spectrum(k,alpha):
  n=len(k);h=np.eye(n)-np.ones((n,n))/n;ev=np.linalg.eigvalsh(k);ce=np.linalg.eigvalsh(h@k@h);pos=np.maximum(ev,0);cp=np.maximum(ce,0)
  return dict(eigenvalues=json.dumps(ev.tolist()),centered_eigenvalues=json.dumps(ce.tolist()),trace=float(np.trace(k)),centered_trace=float(np.trace(h@k@h)),lambda_max=float(ev[-1]),centered_lambda_max=float(ce[-1]),edf=float(sum(pos/(pos+n*alpha))),max_mode_retention=float(pos[-1]/(pos[-1]+n*alpha)),centered_edf_geometry_only=float(sum(cp/(cp+n*alpha))),centered_max_mode_retention_geometry_only=float(cp[-1]/(cp[-1]+n*alpha)))
 def x_embedding(packet,component):
  # Split the current embedding into X-derived mass and other mass. For marginals,
  # e_X = sqrt(weight/N) * p_X/sqrt(p_atom), which sums to e over event classes.
  s=packet['current'];res={};N=len(s.get('reward_histograms',{}))
  if component=='current':
   for p,h in s['reward_histograms'].items():
    for a,c in h['counts'].items():
     if a.startswith('X:'):res[f'reward:{p}:{a}']=math.sqrt(c/h['n']/N)
  if component=='repair':
   for p,h in s['repair_histograms'].items():
    nx=sum(c for a,c in h['counts'].items() if a.startswith('X:'))
    for a,c in h['counts'].items():
     if a.startswith('X:'):res[f'joint:{p}:{a}']=math.sqrt(.25*c/h['n']/N)
    for coord,atom in [('F','1'),('B','0'),('M','1')]:
     mh=s['repair_marginals'][p][coord];c=mh['counts'].get(atom,0)
     if c:res[f'{coord}:{p}:{atom}']=math.sqrt(.25/N)*nx/math.sqrt(h['n']*c)
  return res
 for file in sorted((pilot/'models').glob('*.json')):
  m=read(file);fold=read(pilot/'folds'/file.name);level=m['feature_level'];lid=fold['held_lineage'];alpha=m['alpha'];n=len(m['training_packets']);default=m['best_static'];acts=[f'R{i}' for i in range(8)]
  kernel=FeatureKernel.from_dict(m['kernel']);train=[PreDecisionPacket.from_dict(p) for p in m['training_packets']];emb=[kernel.embedding(p) for p in train];weights=kernel.weights();K=kernel.gram(train)
  maxgram=max(maxgram,float(np.max(np.abs(K-np.array(fold['gram'])))))
  for packet in m['training_packets']:assert packet==packets[packet['origin_id'],level]
  targets=np.array([[truth[p.origin_id,a]-truth[p.origin_id,'R0'] for a in acts] for p in train])
  maxmean=max(maxmean,float(np.max(np.abs(targets.mean(axis=0)-np.array(m['mean'])))))
  maxresidual=max(maxresidual,float(np.max(np.abs((K+n*alpha*np.eye(n))@np.array(m['beta'])-(targets-np.array(m['mean']))))))
  for o in fold['held_origins']:
   q=PreDecisionPacket.from_dict(packets[o,level]);cross=kernel.gram([q],train)[0];state=cross@np.array(m['beta']);mu=np.array(m['mean']);pred=mu+state;pd=dict(zip(acts,pred));top=rank(pd)[0];didx=acts.index(default)
   for i,a in enumerate(acts):
    err=float(pred[i])-existing[level,o,a];maxerr=max(maxerr,abs(err))
    predictions.append(dict(level=level,lineage=lid,origin=o,stage='early' if o.endswith('t32') else 'late',action=a,alpha=alpha,best_static=default,mean_delta_R0=mu[i],state_correction=state[i],prediction_delta_R0=pred[i],actual_delta_R0=truth[o,a]-truth[o,'R0'],prediction_delta_BestStatic=pred[i]-pred[didx],actual_delta_BestStatic=truth[o,a]-truth[o,default],state_contrast_BestStatic=state[i]-state[didx],predicted_best=top,mean_rank=rank(dict(zip(acts,mu))).index(a)+1,prediction_rank=rank(pd).index(a)+1,chosen=default if pred[acts.index(top)]-pred[didx]<=m['practical_tie'] else top,tau=m['practical_tie'],tau_override=(top!=default and pred[acts.index(top)]-pred[didx]<=m['practical_tie']),replay_abs_error=abs(err)))
  mat={'TOTAL':K}
  for c,w in weights.items():mat[c]=np.array([[_dot(a[c],b[c])*w/kernel.scales[c] for b in emb] for a in emb])
  for c,k in mat.items():
   geometry.append(dict(level=level,lineage=lid,component=c,stratum='ALL',n_fit=n,alpha=alpha,ridge_n_alpha=n*alpha,component_weight=weights.get(c,''),saved_raw_scale=kernel.scales.get(c,''),normalized_mean_diagonal=float(np.trace(k)/n),offdiagonal_sum=float(k.sum()-np.trace(k)),**spectrum(k,alpha)))
  if level not in LEVELS[2:]:continue
  strata=sorted(set(kernel.panel.values()));xe={c:[x_embedding(p,c) for p in m['training_packets']] for c in ['current']+(['repair'] if level==LEVELS[3] else [])}
  for c in weights:
   if c=='nuisance':continue
   def stratum_for(key):return kernel.panel[key.split(':')[1]]
   for st in strata:
    es=[{key:v for key,v in e[c].items() if stratum_for(key)==st} for e in emb]
    k=np.array([[_dot(a,b)*weights[c]/kernel.scales[c] for b in es] for a in es]);dist=np.diag(k)[:,None]+np.diag(k)[None,:]-2*k
    xx=np.array([[_dot({key:v for key,v in a.items() if stratum_for(key)==st},b)*weights[c]/kernel.scales[c] for b in xe[c]] for a in xe[c]]) if c in xe else None
    off=float(k.sum()-np.trace(k));xoff=float(xx.sum()-np.trace(xx)) if xx is not None else None
    geometry.append(dict(level=level,lineage=lid,component=c,stratum=st,n_fit=n,alpha=alpha,ridge_n_alpha=n*alpha,component_weight=weights[c],saved_raw_scale=kernel.scales[c],normalized_mean_diagonal=float(np.trace(k)/n),offdiagonal_sum=off,pair_distance_sum=float(np.triu(dist,1).sum()),X_X_offdiagonal_sum=xoff if xoff is not None else '',X_X_similarity_share=xoff/off if xoff is not None and off>0 else '',**spectrum(k,alpha)))
    for i in range(n):
     for j in range(i+1,n):pair_rows.append(dict(level=level,lineage=lid,component=c,stratum=st,left=m['training_packets'][i]['origin_id'],right=m['training_packets'][j]['origin_id'],weighted_similarity=k[i,j],weighted_squared_distance=dist[i,j],X_X_weighted_similarity=xx[i,j] if xx is not None else ''))
  scores={float(s['alpha']):s['selected_action_utility'] for s in m['tuning_scores']};peak=max(scores.values());eligible=[a for a,s in scores.items() if peak-s<.001+1e-12]
  tuning.append(dict(row_type='outer_tuning',level=level,lineage=lid,alpha=alpha,n_alphas=len(scores),exact_utility_tie_pairs=sum(abs(x-y)<1e-14 for i,x in enumerate(scores.values()) for y in list(scores.values())[i+1:]),all_utility_tied=max(scores.values())-min(scores.values())<1e-14,n_eligible=len(eligible),tie_rule_selects_largest=(alpha==max(eligible)),tie_rule_has_multiple_eligible=len(eligible)>1,utility_range=max(scores.values())-min(scores.values()),utility_scores=json.dumps(scores)))
 # Tuning includes all levels; the loop above exits moments early, so collect their outer summaries here.
 for file in sorted((pilot/'models').glob('*Z[01]*.json')):
  m=read(file);scores={float(s['alpha']):s['selected_action_utility'] for s in m['tuning_scores']};peak=max(scores.values());eligible=[a for a,s in scores.items() if peak-s<.001+1e-12]
  tuning.append(dict(row_type='outer_tuning',level=m['feature_level'],lineage=file.name.split('_')[0],alpha=m['alpha'],n_alphas=len(scores),exact_utility_tie_pairs=sum(abs(x-y)<1e-14 for i,x in enumerate(scores.values()) for y in list(scores.values())[i+1:]),all_utility_tied=max(scores.values())-min(scores.values())<1e-14,n_eligible=len(eligible),tie_rule_selects_largest=(m['alpha']==max(eligible)),tie_rule_has_multiple_eligible=len(eligible)>1,utility_range=max(scores.values())-min(scores.values()),utility_scores=json.dumps(scores)))
 for file in sorted((pilot/'folds').glob('*.json')):
  f=read(file);l=file.stem.split('_',1)[1];lid=f['held_lineage'];grid=f['alpha_grid'];base=grid[str(float(f['alpha']))]
  for aa,ds in grid.items():
   for o,d,b in zip(f['held_origins'],ds,base):
    r=rank(d['predictions']);br=rank(b['predictions']);inversions=sum((r.index(a)-r.index(z))*(br.index(a)-br.index(z))<0 for i,a in enumerate(r) for z in r[i+1:])
    tuning.append(dict(row_type='saved_alpha_sensitivity',level=l,lineage=lid,origin=o,alpha=float(aa),saved_selected_alpha=f['alpha'],best_static=f['best_static'],predicted_best=d['predicted_best'],chosen=d['recipe'],margin=d['margin_to_default'],ranking='>'.join(r),rank_pair_inversions_vs_saved=inversions,top_changed_vs_saved=d['predicted_best']!=b['predicted_best'],chosen_changed_vs_saved=d['recipe']!=b['recipe'],actual_J=truth[o,d['recipe']],actual_delta_BestStatic=truth[o,d['recipe']]-truth[o,f['best_static']]))
 grouped=defaultdict(list)
 for r in inner:grouped[r['level'],r['outer_held_lineage'],r['inner_validation_lineage']].append(r)
 for (l,ol,il),rr in grouped.items():
  vs=[float(r['selected_J']) for r in rr];tuning.append(dict(row_type='inner_validation',level=l,lineage=ol,inner_validation_lineage=il,n_alphas=len(rr),all_utility_tied=max(vs)-min(vs)<1e-14,exact_utility_tie_pairs=sum(abs(x-y)<1e-14 for i,x in enumerate(vs) for y in vs[i+1:]),utility_range=max(vs)-min(vs)))
 write('saved_prediction_decomposition.csv',predictions);write('kernel_geometry.csv',geometry);write('existing_tuning_diagnostics.csv',tuning);write('kernel_pair_geometry_D.csv',pair_rows)
 assert maxerr<1e-12 and maxgram<1e-12 and maxmean<1e-12 and maxresidual<1e-12
 receipt=dict(actual_input_files=sorted(set(inputs)),experiment_code_version=version,current_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),command=f'{sys.executable} {Path(__file__).resolve()} --root {root}',numpy_version=np.__version__,checks=dict(prediction_rows=len(predictions),replay_max_abs_error=maxerr,gram_max_abs_error=maxgram,target_mean_max_abs_error=maxmean,saved_beta_equation_max_abs_residual=maxresidual),execution_scope=dict(new_training_runs=0,new_model_calls=0,new_samples=0,selector_refits=0,hyperparameter_searches=0,gpu_jobs_submitted=0,statistical_resamples=0),missing_fields=[],notes=['Only saved alpha grid read; no new solve.','Centered spectra are descriptive geometry, not centered model predictions.','X-X repair marginal attribution splits observed marginal embeddings by X mass; includes no X-other cross attribution in numerator.','No raw outputs or final T accessed.'])
 (out/'inputs_D.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt['checks']))
if __name__=='__main__':main()
