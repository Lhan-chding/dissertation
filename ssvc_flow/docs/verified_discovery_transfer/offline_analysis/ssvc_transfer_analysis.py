#!/usr/bin/env python3
"""Offline, post-release analysis. Writes only transfer-prefixed outputs."""
import argparse,csv,importlib.util,json,math,sys
from pathlib import Path
from collections import defaultdict,Counter
from statistics import mean
SRC=Path(__file__).resolve().parent/'statistics_frozen.py'
if not SRC.exists():
 SRC=Path('/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/verified-discovery-transfer-20261006/ssvc_flow/src/verified_discovery_transfer/statistics.py')
spec=importlib.util.spec_from_file_location('registered_statistics',SRC); stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
SCOPE='Conditional on executed checkpoints; family-stratified base-scene bootstrap, all parents/repeats moved together; not training-seed-population inference. Exploratory unless explicitly primary. No multiplicity correction.'
def read(p): return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def boot(rows):return stats.paired_cluster_bootstrap(rows,replicates=5000,seed=106069)
def fmean(rows,field):
 g=defaultdict(list)
 for r in rows:
  if r.get(field) is not None:g[r['family']].append(r[field])
 return mean(mean(v) for v in g.values()) if g else None

def selftest():
 rows=[dict(family=f,base_instance_id=f+str(i),unit=u,left=.7,right=.4) for f in ['a','b'] for i in range(4) for u in ['p0','p1']]
 b=boot(rows);assert abs(b['difference']-.3)<1e-12 and all(abs(v-.3)<1e-12 for v in b['interval_95'])
 try:boot(rows[:-1]);raise AssertionError('missing panel not rejected')
 except ValueError:pass
 same=[dict(r,right=r['left']) for r in rows];assert boot(same)['interval_95']==[0.,0.]
 assert abs(stats.pass_at_k(1,2,2)-1)<1e-12
 print('SYNTHETIC_PASS: constant paired differences, identity, incomplete panel rejection, finite-stream pass@2')

def main(root):
 release=read(root/'TEST_RELEASE.json');assert release['status']=='RELEASED'
 endpoints=read(root/'FINAL_ENDPOINTS.json');parents=read(root/'PARENT_ENDPOINTS.json');existing=read(root/'FINAL_COMPARISONS.json');teachers=read(root/'TEACHER_TWO_CALL_PORTFOLIOS.json')
 out=root/'final_analysis';out.mkdir(exist_ok=True)
 cells=[]; tables={};semantics={};mapping={}
 for name,e in sorted(endpoints.items()):
  if '.E_test.' not in name:continue
  _,parent,repeat,arm,split,step=name.split('.')
  sem=read(root/'final_semantics'/(name+'.json'));semantics[name]=sem
  mt={r['task_id']:r for r in sem['tasks']}; mapping.update({tid:r['base_instance_id'] for tid,r in mt.items()})
  tables[name]={r['task_id']:r for r in e['tasks']}
  assert set(tables[name])==set(mt)
  cells.append(dict(endpoint=name,parent=parent,repeat=int(repeat),arm=arm,step=int(step),task_count=len(mt),pX=e['pX_family_equal'],parent_pX=parents[parent+':E_test']['pX_family_equal'],parent_delta=e['parent_delta'],families=e['families'],contrast=e['contrast_state_counts']))
 csvout(out/'transfer_endpoints.csv',cells)
 aggregates=[]
 for arm in sorted({r['arm'] for r in cells}):
  for parent in ['S96','REP96','BOTH']:
   c=[r for r in cells if r['arm']==arm and (parent=='BOTH' or r['parent']==parent)]
   if c:aggregates.append(dict(parent=parent,arm=arm,cells=len(c),pX=mean(r['pX'] for r in c),parent_delta=mean(r['parent_delta'] for r in c)))
 csvout(out/'transfer_arm_averages.csv',aggregates)
 comparisons={}
 for comparator in ['SELF_O0','SELF_SINGLE']:
  key='SELF_MIX - '+comparator; frozen=existing[key]
  comparisons[key]={'registered':frozen,'parent_averages':{}}
  for parent in ['S96','REP96']:
   panel=[]
   for repeat in [0,1]:
    a=tables[f'eval.{parent}.{repeat}.SELF_MIX.E_test.256'];b=tables[f'eval.{parent}.{repeat}.{comparator}.E_test.256']
    for tid,x in a.items():panel.append(dict(family=x['family'],base_instance_id=mapping[tid],unit=f'{parent}:{repeat}',left=x['pX'],right=b[tid]['pX']))
   comparisons[key]['parent_averages'][parent]=boot(panel)
 # Every arm vs matched parent: repeat units share the same parent stream; scene bootstrap keeps that dependency.
 parent_ci=[]
 for arm in sorted({r['arm'] for r in cells}):
  for parent_scope in ['S96','REP96','BOTH']:
   sel=[r for r in cells if r['arm']==arm and (parent_scope=='BOTH' or r['parent']==parent_scope)]
   if not sel:continue
   panel=[]
   for cell in sel:
    p={r['task_id']:r for r in parents[cell['parent']+':E_test']['tasks']}
    for tid,x in tables[cell['endpoint']].items():panel.append(dict(family=x['family'],base_instance_id=mapping[tid],unit=f"{cell['parent']}:{cell['repeat']}",left=x['pX'],right=p[tid]['pX']))
   b=boot(panel);parent_ci.append(dict(arm=arm,parent=parent_scope,classification='exploratory',**b))
 write(out/'transfer_comparisons.json',{'scope':SCOPE,'main':comparisons,'arm_vs_parent':parent_ci})
 csvout(out/'transfer_parent_comparisons.csv',parent_ci)
 strata=[]
 for name,sem in semantics.items():
  groups=defaultdict(list)
  for t in sem['tasks']:
   for key in ['all','PRE_ZERO16','family','corrupted_index','DPE1_O0','truth_all_in_0_49']:
    groups[(key,'ALL' if key=='all' else str(t[key]))].append(t)
  for (stratum,value),ts in groups.items():
   rr=[]
   for t in ts:
    x=dict(t)
    for field in ['partial_repair','copy','all_original_relations_wrong']:
     x[field+'_rate']=t[field+'_count']/t['n']
    x['noncopy_nontruth_ptlc_rate']=t['noncopy_nontruth_ptlc_count']/t['n']
    x['K8_partial_present']=float(t['contrast']['has_partial_repair']);x['K8_C_orig_varies']=float(t['contrast']['C_orig_varies'])
    rr.append(x)
   row=dict(endpoint=name,stratum=stratum,value=value,task_count=len(ts),families=dict(Counter(t['family'] for t in ts)),draw_count=sum(t['n'] for t in ts),K8_contrast_counts=dict(Counter(t['contrast']['state'] for t in ts)))
   for field in ['pX','pS','pW','pI','C_orig_mean','F_mean','B_mean','M_mean','original_anchor_preservation','partial_repair_rate','copy_rate','all_original_relations_wrong_rate','noncopy_nontruth_ptlc_rate','K8_partial_present','K8_C_orig_varies']:
    row[field]=fmean(rr,field)
   for f in ['F','B','M']:row[f+'_available_draws']=sum(t[f+'_count'] for t in ts)
   strata.append(row)
 csvout(out/'transfer_semantic_strata.csv',strata)
 sem_groups=defaultdict(list)
 for r in strata:
  arm=r['endpoint'].split('.')[3]
  sem_groups[(arm,r['stratum'],r['value'])].append(r)
 sem_aggregates=[]
 for (arm,stratum,value),rr in sorted(sem_groups.items()):
  vals=dict(arm=arm,stratum=stratum,value=value,endpoint_count=len(rr),task_instances=sum(r['task_count'] for r in rr),unique_task_count='see endpoint stratum tables; repeated task instances are not independent')
  for field in ['pX','pS','pW','pI','C_orig_mean','F_mean','B_mean','M_mean','original_anchor_preservation','partial_repair_rate','copy_rate','all_original_relations_wrong_rate','noncopy_nontruth_ptlc_rate','K8_partial_present','K8_C_orig_varies']:
   present=[r[field] for r in rr if r[field] is not None]
   vals[field]=mean(present) if present else None
  sem_aggregates.append(vals)
 csvout(out/'transfer_semantic_strata_averages.csv',sem_aggregates)
 # Frozen teacher policies: all parents move together within each base scene. No E-based policy selection.
 teacher_rows=[r for t in teachers.values() for r in t['task_results']]
 tidx={(r['parent'],r['task_id'],r['policy']):r for r in teacher_rows}
 tcomparisons=[];taggregates=[]
 for policy in sorted({r['policy'] for r in teacher_rows}):
  ps=[r for r in teacher_rows if r['policy']==policy]
  for scope in ['S96','REP96','BOTH']:
   rr=[r for r in ps if scope=='BOTH' or r['parent']==scope]
   if rr:taggregates.append(dict(policy=policy,parent=scope,rows=len(rr),families=sorted({r['family'] for r in rr}),**{field:fmean(rr,field) for field in ['unbiased_success','offline_trial_success','early_stop_expected_calls','offline_trial_calls']}))
  for baseline in ['O0_REPEAT','V_BEST_SINGLE']:
   for metric in ['unbiased_success','offline_trial_success']:
    panel=[]
    for r in ps:
     b=tidx[(r['parent'],r['task_id'],baseline)]
     panel.append(dict(family=r['family'],base_instance_id=r['base_instance_id'],unit=r['parent'],left=r[metric],right=b[metric]))
    tcomparisons.append(dict(policy=policy,baseline=baseline,metric=metric,families=sorted({r['family'] for r in ps}),classification='frozen B2 exploratory paired contrast',**boot(panel)))
 csvout(out/'transfer_teacher_averages.csv',taggregates)
 csvout(out/'transfer_teacher_comparisons.csv',tcomparisons)
 write(out/'transfer_teacher_comparisons.json',{'scope':SCOPE,'warning':'V_BEST_SINGLE is V-selected frozen best single, never E-selected. B1_REPEAT is cross-only; do not compare its marginal directly with two-family scores. Offline replay does not measure wall time.','comparisons':tcomparisons,'averages':taggregates})
 md=['# 迁移与固定两调用参照','',SCOPE,'','所有数字为已统一揭晓的现有数据分析；无新增生成或训练。主终点为SFT step256，R0为step32且计算预算不同。','', '## E端点（百分比；括号为相对父模型百分点）','', '|Parent|Repeat|Arm|pX|Δparent|','|---|---:|---|---:|---:|']
 for r in cells:md.append(f"|{r['parent']}|{r['repeat']}|{r['arm']}|{100*r['pX']:.3f}|{100*r['parent_delta']:+.3f}|")
 md+=['','## 预登记主要/关键次要比较','']
 for key,x in comparisons.items():
  c=x['registered'];mc=c.get('fixed_panel_mc',{}).get('mcse');md.append(f"- {key}: {100*c['difference']:+.3f} pp; 95% scene CI [{100*c['interval_95'][0]:+.3f}, {100*c['interval_95'][1]:+.3f}] pp; fixed-panel MCSE {100*mc:.3f} pp.")
  for p,b in x['parent_averages'].items():md.append(f"  - {p}: {100*b['difference']:+.3f} pp; exploratory scene CI [{100*b['interval_95'][0]:+.3f}, {100*b['interval_95'][1]:+.3f}] pp.")
 md+=['','## 语义与分层解释边界','','完整分层表为 `transfer_semantic_strata.csv`。每个子组按存在的家族等权、家族内题等权；跨不同子组比较不是因果效应。F/B/M只对字段可用的回答，分别报告可用回答数；不可解析/非法不能静默当作合法零。PRE_ZERO16是父模型预训练O0固定16抽无成功，不能解释为真实成功概率为零。copy/PTLC/partial等可重叠，不能相加当互斥机制。K8是训练后E八条回答的题组对比状态，不是训练时contrast。','','## 冻结两调用策略','','完整无偏估计、固定8-trial离线回放及配对区间分别见 `transfer_teacher_averages.csv`、`transfer_teacher_comparisons.csv`；共享O0流及所有父模型随基础场景整组移动，区间是条件于已执行模型的场景区间，不是独立调用bootstrap或种子总体区间；B1_REPEAT仅cross家族。最佳单协议由V冻结，未在E重新挑选。','']
 for r in taggregates:
  if r['parent']=='BOTH' and r['policy'] in ['O0_REPEAT','FIXED_MIX','V_BEST_SINGLE','V_BEST_PAIR']:
   md.append(f"- {r['policy']}: 无偏成功率{100*r['unbiased_success']:.3f}%；离线回放{100*r['offline_trial_success']:.3f}%；理论早停调用{r['early_stop_expected_calls']:.3f}，回放调用{r['offline_trial_calls']:.3f}。")
 md+=['','## 各臂语义与分层摘要','','下表跨实际登记parent/repeat单元等权，家族内题等权；REPLAY_ONLY只有S96，不能把其跨单元均值视为两个父模型的同一总体。百分数均乘100；F/B/M保持原尺度。','', '|Arm|Stratum|Value|Cells|pX%|pS%|pW%|pI%|C_orig|F|B|M|copy%|partial%|noncopy nontruth PTLC%|', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 def fmt(v,scale=1):return 'NA' if v is None else f'{v*scale:.3f}'
 for r in sem_aggregates:
  md.append('|'+ '|'.join([r['arm'],r['stratum'],r['value'],str(r['endpoint_count'])]+[fmt(r[k],100) for k in ['pX','pS','pW','pI']]+[fmt(r[k]) for k in ['C_orig_mean','F_mean','B_mean','M_mean']]+[fmt(r[k],100) for k in ['copy_rate','partial_repair_rate','noncopy_nontruth_ptlc_rate']])+'|')
 md+=['','G保持性另表报告；本分析不以E收益替代保持性结论。没有显著差异不等于等效，2pp只作解释尺度；未进行事后选步、追加预算或扩展矩阵。']
 (out/'transfer_report_zh.md').write_text('\n'.join(md)+'\n')
 scientific_readout(out,semantics)
 print(json.dumps({'status':'COMPLETE','E_endpoints':len(cells),'stratum_rows':len(strata),'teacher_contrasts':len(tcomparisons),'output':str(out)}))
def scientific_readout(out,semantics):
 c=read(out/'transfer_comparisons.json');te=read(out/'transfer_teacher_comparisons.json')
 sem=list(csv.DictReader((out/'transfer_semantic_strata_averages.csv').open()))
 def val(arm,st,v,f='pX'):
  return float(next(r for r in sem if (r['arm'],r['stratum'],r['value'])==(arm,st,v))[f])
 def percent(v):return f'{100*v:.3f}%'
 main=c['main']['SELF_MIX - SELF_O0']['registered'];sec=c['main']['SELF_MIX - SELF_SINGLE']['registered']
 text=['# 迁移科学读出：正收益、代价与不确定性','',f"主要MIX−O0点估计{100*main['difference']:+.3f}pp，95%场景区间[{100*main['interval_95'][0]:+.3f}, {100*main['interval_95'][1]:+.3f}]pp；关键次要MIX−SINGLE为{100*sec['difference']:+.3f}pp，区间[{100*sec['interval_95'][0]:+.3f}, {100*sec['interval_95'][1]:+.3f}]pp。两个区间都跨零，不能声称已确证优于对应SELF对照，也不能解释为等效。",'',f"主比较固定面板MCSE为{100*main['fixed_panel_mc']['mcse']:.4f}pp；它描述固定题与模型的采样不确定性，不能替代跨基础场景的区间，更不涵盖新训练种子总体。",'','## 成功率的变化伴随得失','']
 for st,v,label in [('PRE_ZERO16','True','父模型预先16条无成功组'),('PRE_ZERO16','False','父模型预先16条有成功组'),('DPE1_O0','False','DPE1_O0=False'),('DPE1_O0','True','DPE1_O0=True'),('family','cross_series','cross家族'),('family','trend','trend家族'),('corrupted_index','0','损坏位置0'),('corrupted_index','1','损坏位置1'),('corrupted_index','2','损坏位置2'),('corrupted_index','3','损坏位置3')]:
  a,b=val('SELF_MIX',st,v),val('SELF_O0',st,v)
  text.append(f'- {label}：MIX {percent(a)}，O0 {percent(b)}，差{100*(a-b):+.3f}pp。')
 text+=['','以上为固定端点的描述性分层，未经多重检验校正，也不能独立识别训练机制。PRE_ZERO16按各父模型预训练固定流定义，绝不等于真实成功概率为零。两个组各自重新按家族等权，不能用组间数字直接加总恢复主终点。','']
 for f,label in [('pI','非法事件I'),('copy_rate','copy'),('noncopy_nontruth_ptlc_rate','合法非copy非truth PTLC匹配'),('partial_repair_rate','合法部分修复F=1且B>0')]:
  text.append(f"- {label}：MIX {percent(val('SELF_MIX','all','ALL',f))}，O0 {percent(val('SELF_O0','all','ALL',f))}。这些事件不是全部互斥，也不能据它们单独识别内部程序。")
 text+=['','## 旧数值范围子组','']
 any_sem=next(iter(semantics.values()))['tasks'];old=[r for r in any_sem if r['truth_all_in_0_49']];counts=dict(Counter(r['family'] for r in old))
 text.append(f'旧0–49范围E子组只有{len(old)}个基础任务，家族分布为{counts}。这里家族等权会给极少数任务较大权重；这些数据不能支持旧ID分布精确复现结论。完整子组数字保留于分层表。')
 text+=['','## 固定teacher两调用：与监督迁移不同的问题','']
 for metric in ['unbiased_success','offline_trial_success']:
  r=next(x for x in te['comparisons'] if x['policy']=='FIXED_MIX' and x['baseline']=='O0_REPEAT' and x['metric']==metric)
  text.append(f"- {metric}: MIX−O0_REPEAT {100*r['difference']:+.3f}pp，条件场景95%区间[{100*r['interval_95'][0]:+.3f}, {100*r['interval_95'][1]:+.3f}]pp。")
 text+=['','V_BEST_PAIR与FIXED_MIX在本次两父冻结选择中相同；V_BEST_SINGLE与O0_REPEAT相同，故这些同值不是独立重复证据。共享O0原子样本及所有父模型随基础场景整体重采样。两调用组合的明确场景收益不自动证明16调用发现集的SFT优越性；全样本无偏估计与固定8次回放分别保留，回放不是新增部署试验或真实墙钟加速。B1_REPEAT仅cross，必须和cross上的O0_REPEAT配对，不能拿它和两家族O0宏平均直接相减。','', '## K8固定输出组','', '|Arm|题×模型实例数|all-X|mixed-X|no-X|有部分修复组|C_orig变化组|','|---|---:|---:|---:|---:|---:|---:|']
 grouped=defaultdict(list)
 for name,sv in semantics.items():grouped[name.split('.')[3]].extend(sv['tasks'])
 krows=[]
 for arm,ts in sorted(grouped.items()):
  counts=Counter(t['contrast']['state'] for t in ts);partial=sum(t['contrast']['has_partial_repair'] for t in ts);vary=sum(t['contrast']['C_orig_varies'] for t in ts)
  krows.append(dict(arm=arm,task_model_instances=len(ts),contrast=dict(counts),partial_groups=partial,C_orig_varies_groups=vary))
  text.append(f"|{arm}|{len(ts)}|{counts['all-X']}|{counts['mixed-X']}|{counts['no-X']}|{partial}|{vary}|")
 text+=['','表中计数是同一基础题在不同模型/重复上的实例数，不是互相独立的新题。K8全部来自训练后E的八条固定回答；C_orig变化不等于有正确输出，也不等于训练优势。G保持性须另行检查后才能形成完整结论。']
 write(out/'transfer_K8_summary.json',krows)
 (out/'transfer_scientific_readout_zh.md').write_text('\n'.join(text)+'\n')
 for row in sem:
  assert abs(sum(float(row[k]) for k in ['pX','pS','pW','pI'])-1)<1e-12
 assert all(sum(row['contrast'].values())==row['task_model_instances'] for row in krows)
 for comp,field in [('SELF_MIX - SELF_O0','SELF_O0'),('SELF_MIX - SELF_SINGLE','SELF_SINGLE')]:
  assert abs((val('SELF_MIX','all','ALL')-val(field,'all','ALL'))-c['main'][comp]['registered']['difference'])<1e-12
 write(out/'transfer_validation.json',{'status':'PASS','semantic_aggregate_rows':len(sem),'checks':['X/S/W/I sum to one in every stratum aggregate','K8 counts equal task-model instance counts','semantic endpoint differences reproduce frozen primary and secondary differences','synthetic constant paired differences and identity and incomplete-panel rejection'],'scope':'offline supplemental analysis; not new experiment or scientific superiority certificate'})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('snapshot',nargs='?');ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
 if a.self_test:selftest()
 if a.snapshot:main(Path(a.snapshot))
