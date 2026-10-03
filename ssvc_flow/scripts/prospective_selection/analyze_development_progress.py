"""Descriptive, reproducible first-four analysis without fitting/selecting tests."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
from src.prospective_selection.features import PreDecisionPacket

ACTIONS=[f'R{i}' for i in range(8)]
RECIPES=ACTIONS+['SAW_R4','GDPO_R4','DIRECT_REPAIR_R4']
ORIGINS=[f'{i}_t{t}' for i in range(61001,61005) for t in (32,96)]

def write_csv(out,name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)

def information_audit(pre):
    packets=pre['packets']
    assert len(packets)==4 and pre['complete']['four_levels_same_samples']
    for p in packets.values():PreDecisionPacket(json.dumps(p))
    z3=packets['Z3_REPAIR_STRUCTURE'];rows=[]
    for when in ('history','current'):
        s=z3[when]
        for p in packets.values():
            assert p['origin_id']==pre['origin'] and p[when]['n']==s['n']==2304
            for key,value in p[when].items():assert value==s[key]
        buckets=[];mixed_prompts=set();max_identity_error=0.0
        for prompt,h in s['repair_histograms'].items():
            by_reward=defaultdict(Counter)
            for atom,n in h['counts'].items():
                if atom=='INVALID':continue
                reward,f,b,m=atom.split('|');by_reward[reward][(f,b,m)]+=n
            for reward,counts in by_reward.items():
                n=sum(counts.values());buckets.append((prompt,n,counts))
                if len(counts)>1:mixed_prompts.add(prompt)
        repeated=[b for b in buckets if b[1]>=2]
        mixed=[b for b in repeated if len(b[2])>1]
        pairs=sum(n*(n-1)//2 for _,n,_ in repeated)
        different=pairs-sum(k*(k-1)//2 for _,_,counts in repeated for k in counts.values())
        events=Counter()
        for h in s['reward_histograms'].values():
            for atom,n in h['counts'].items():events[atom.split(':')[0]]+=n
        mu=s['moments_global'];expected=[mu[0],mu[1]-mu[0],mu[2]-mu[1],1-mu[2]]
        max_identity_error=max(abs(events[e]/s['n']-v) for e,v in zip('XSWI',expected))
        assert max_identity_error<1e-12
        rows.append(dict(origin=pre['origin'],snapshot=z3['history_step'] if when=='history' else z3['step'],
            raw_outputs=s['n'],reward_event_reconstruction_error=max_identity_error,
            valid_reward_buckets=len(buckets),repeated_reward_buckets=len(repeated),
            mixed_repair_buckets=len(mixed),mixed_n_ge_8=sum(n>=8 for _,n,_ in mixed),
            prompts_with_mixed_repair=len(mixed_prompts),same_reward_pairs=pairs,
            different_repair_pairs=different,pair_disagreement=different/pairs if pairs else None))
    return rows

def candidate_comparison(index, complete):
    means={r:statistics.mean(index[o,r] for o in complete) for r in ACTIONS}
    best=max(means,key=means.get)
    oracle=statistics.mean(max(index[o,r] for r in ACTIONS) for o in complete)
    return dict(candidate_recipes=ACTIONS,descriptive_best_static_recipe=best,
                descriptive_best_static_mean=means[best],
                descriptive_hindsight_oracle_mean=oracle,
                descriptive_oracle_gap_pp=100*(oracle-means[best]))


def build(data,out,baseline_timing=None):
    out.mkdir(parents=True,exist_ok=True)
    index={};branches=[];strata=[];updates=[]
    for b in data['branches']:
        q=b['task']['payload'];o,r=q['origin_id'],q['recipe_id'];s=b['evaluations']['32']['summary']
        assert o in ORIGINS and r in RECIPES and (o,r) not in index
        assert q['repeat']==1 and q['role']=='development'
        assert s['samples']==2304 and s['prompts']==144
        assert abs(s['J']-b['audit']['J'])<1e-12
        if b['updates']:assert sorted(u['step'] for u in b['updates'])==list(range(1,33))
        assert all(math.isfinite(u[k]) for u in b['updates'] for k in ('loss','grad_norm_preclip'))
        assert abs(statistics.mean(v['pX'] for v in s['strata'].values())-s['J'])<1e-12
        index[o,r]=s['J'];audit=b['audit'];worker=audit.get('worker_elapsed_seconds')
        # Incomplete historical timing is left missing rather than mixing units.
        timing=audit.get('sacct') or (baseline_timing or {}).get('jobs',{}).get(str(audit.get('job_id')))
        if worker is None and isinstance(timing,list):
            assert timing[0]==str(audit['job_id']) and timing[1:3]==['COMPLETED','0:0']
            duration=timing[3];days,hms=(duration.split('-') if '-' in duration else ('0',duration))
            h,m,sec=map(int,hms.split(':'));worker=int(days)*86400+h*3600+m*60+sec
        row=dict(origin=o,recipe=r,J=s['J'],H8_J=b['evaluations']['8']['summary']['J'],
            image_pX=s['interfaces']['IMAGE_CUE_FRESH'],symbolic_pX=s['interfaces']['SYMBOLIC_FRESH'],
            invalid_rate=statistics.mean(x['pI'] for x in s['strata'].values()),
            worker_hours=None if worker is None else worker/3600,
            training_sampling_hours=sum(u['sampling_seconds'] for u in b['updates'])/3600 if b['updates'] else None,
            optimizer_hours=sum(u['update_seconds'] for u in b['updates'])/3600 if b['updates'] else None,
            H32_sampling_hours=b['evaluations']['32']['elapsed_sampling_seconds']/3600,
            zero_gradient_steps=sum(u['grad_norm_preclip']==0 for u in b['updates']) if b['updates'] else None)
        branches.append(row)
        for st,v in s['strata'].items():strata.append(dict(origin=o,recipe=r,stratum=st,**v))
        updates.extend(dict(origin=o,recipe=r,**u) for u in b['updates'])
    matrix=[dict(origin=o,**{r:index.get((o,r)) for r in RECIPES}) for o in ORIGINS]
    complete=[o for o in ORIGINS if all((o,r) in index for r in RECIPES)]
    origins=[]
    for o in ORIGINS:
        scores={r:j for (origin,r),j in index.items() if origin==o}
        top=max(scores.values());bottom=min(scores.values())
        origins.append(dict(origin=o,completed=len(scores),best_observed_J=top,
            best_observed_recipes=';'.join(r for r,j in scores.items() if abs(j-top)<1e-12),
            worst_observed_J=bottom,observed_spread_pp=100*(top-bottom),complete=len(scores)==11))
    static=[dict(recipe=r,complete_origins=len(complete),mean_J=statistics.mean(index[o,r] for o in complete)) for r in RECIPES] if complete else []
    comparison=candidate_comparison(index,complete) if complete else {}
    assert len(data['prestates'])==8 and {p['origin'] for p in data['prestates']}==set(ORIGINS)
    info=[row for p in data['prestates'] for row in information_audit(p)]
    hist=[]
    for e in data['historical_E']:
        a=e['audit'];s=e['evaluation']['summary'];assert abs(a['J']-s['J'])<1e-12
        hist.append(dict(origin=a['origin_id'],recipe=a['recipe'],J=s['J'],outputs=e['evaluation']['generated_outputs']))
    for name,rows in [('response_matrix.csv',matrix),('branch_summary.csv',branches),('origin_summary.csv',origins),
                      ('six_strata.csv',strata),('training_updates.csv',updates),('information_audit.csv',info),
                      ('historical_E.csv',hist),('static_complete_origins.csv',static)]:
        if rows:write_csv(out,name,rows)
    assert len(hist)==16 and len({(h['origin'],h['recipe']) for h in hist})==16
    status='FIRST_FOUR_ANALYSIS_COMPLETE_AWAITING_DELIVERY' if len(index)==88 and not data['missing'] else 'PARTIAL_DEVELOPMENT_ANALYSIS_NOT_DELIVERY'
    facts=dict(extracted_at=data['extracted_at'],branches=len(branches),expected_branches=88,
        independent_lineages=4,origins=origins,complete_origins=complete,historical_E=len(hist),
        missing=[t['key'] for t in data['missing']],information_audit=info,
        **comparison,
        static=static,timing_rows=sum(b['worker_hours'] is not None for b in branches),
        available_worker_mean_hours=statistics.mean(b['worker_hours'] for b in branches if b['worker_hours'] is not None),
        status=status,selector_fitted=False,final_test_run=False)
    (out/'summary.json').write_text(json.dumps(facts,indent=2,ensure_ascii=False)+'\n')
    missing='、'.join('/'.join(map(str,t['key'][:2])) for t in data['missing']) or '无'
    origin_table='\n'.join(f"|{x['origin']}|{x['completed']}/11|{x['best_observed_recipes']}|{100*x['best_observed_J']:.3f}%|{x['observed_spread_pp']:.3f}|" for x in origins)
    intro=f"# 首四条开发数据阶段分析\n\n数据快照UTC：{data['extracted_at']}。状态：`{status}`。\n\n已核验{len(branches)}/88分支、{len(hist)}/16旧E、8个前置原点。缺失：{missing}。\n\n仅分析已通过服务器原始核验的任务，本次重查任务/完成回执哈希，并复核导出汇总与信息包；没有重新逐条复核全部原始生成文本。完整输入文件哈希及服务器小文件索引见INPUT_SHA256.txt和compact_input.json。\n"
    subset_note='全部8原点已齐全；以下比较仍仅限首4条开发lineage。' if len(complete)==8 else '这一子集的来源/阶段组成可能不均衡，不能冒充完整8原点结论。'
    development=intro+f"\n## 当前响应表\n\n|原点|完成|当前最高配方|H32 J|已观测极差（百分点）|\n|---|---:|---|---:|---:|\n{origin_table}\n\n完整矩阵见response_matrix.csv，分层结果见six_strata.csv。缺失单元格为空，不计作0。\n\n只有{len(complete)}个配方齐全的原点参与static_complete_origins.csv中的均值比较；{subset_note}11列均值表包括三个外部固定对照；四层选择器的可选动作严格限定R0–R7。在R0–R7内，当前事后最佳静态配方为{comparison['descriptive_best_static_recipe']}，均值为{100*comparison['descriptive_best_static_mean']:.3f}%；事后逐原点最高分均值为{100*comparison['descriptive_hindsight_oracle_mean']:.3f}%，两者之差为{facts['descriptive_oracle_gap_pp']:.3f}个百分点。这是使用已知结果的描述性可选空间，存在事后乐观偏差，不是任何实际选择器收益或独立验证上界。\n\nH8和H32使用不同评估面板，branch_summary.csv列出的两个分数不能直接相减解释成训练提升。本轮没有逐题重采样置信区间、选择器拟合、最终测试或显著性宣称。原点之间的同配方差值也同时受来源和训练阶段影响。\n\n统计独立单位仍为4条lineage，88个配方/阶段分支属于嵌套重复测量；每分支只有repeat1，不能由这些数值估计同原点同配方的训练seed方差。\n"
    (out/'DEVELOPMENT_RESULTS_zh.md').write_text(development)
    info_table='\n'.join(f"|{x['origin']}|{x['snapshot']}|{x['mixed_repair_buckets']}|{x['mixed_n_ge_8']}|{x['prompts_with_mixed_repair']}|{100*x['pair_disagreement']:.2f}%|" for x in info)
    audit=intro+f"\n## 信息冗余与新增结构\n\npX=μX、pS=μA−μX、pW=μV−μA、pI=1−μV在16份前置快照中的最大重建误差为{max(x['reward_event_reconstruction_error'] for x in info):.3g}。四事件不是奖励均值以外的独立新增信息。\n\n32份信息包通过字段权限、有限矩、直方图计数、修复联合分布向同一奖励分布投影以及各层共用样本的校验。合法修复原子的X iff F=1且B=0及F/B/M可行性由原信息包验证器检查；非法输出使用独立INVALID原子。\n\n|原点|快照步|同奖励多修复结构桶|其中n≥8|涉及提示数|同桶样本对结构不同率|\n|---|---:|---:|---:|---:|---:|\n{info_table}\n\n桶严格按(prompt,event,精确关系奖励)分组。样本对不同率来自固定观测样本，不是稳定条件互信息估计或总体概率保证。存在同奖励而修复结构不同，支持逐样本不可完全恢复性；这并不证明该结构有助于选择奖励。完整数字见information_audit.csv。\n"
    hist_table='\n'.join(f"|{x['origin']}|{x['recipe']}|{100*x['J']:.3f}%|{x['outputs']}|" for x in sorted(hist,key=lambda x:(x['origin'],-x['J'])))
    (out/'HISTORICAL_E_zh.md').write_text(intro+f"\n## 旧端点固定E面板复核\n\n|历史原点|配方|E J|输出数|\n|---|---|---:|---:|\n{hist_table}\n\n16项均复用已保存H32端点，未重训。O1和O2的已注册臂集合不同，不能把缺失臂填0或跨原点直接汇总排名。该复核使用旧端点与E面板，既不是本次新D原点的公平横向比较，也不是冻结后未见原点的最终测试；本表不用于宣称Z3选择收益。\n")
    (out/'STATE_VARIABLE_AUDIT_zh.md').write_text(audit)
    hours=[b['worker_hours'] for b in branches if b['worker_hours'] is not None]
    runtime=intro+f"\n## 实测计算量\n\n本快照覆盖{len(branches)*1024}个训练输出、{len(branches)*32}次有限优化器更新、{len(branches)*192}个H8评估输出、{len(branches)*2304}个H32评估输出；16个旧E共有{sum(x['outputs'] for x in hist)}输出。8原点×历史/当前×2304=36864个前置观测输出，四层信息包复用这些样本。\n\n可直接恢复完整作业/worker计时的{len(hours)}个分支合计{sum(hours):.3f} GPU小时、均值{statistics.mean(hours):.3f}小时；其中旧作业使用单分支Slurm分配时长，流水内分支使用worker时长；旧计时补充见baseline_timing.json。它们覆盖成功分支的运行区间，仍不是包含全部阶段与失败成本的项目总成本。\n\n本次实际导出逐步优化器元数据的分支有{sum(b['training_sampling_hours'] is not None for b in branches)}个；该子集采样计时和为{sum(b['training_sampling_hours'] or 0 for b in branches):.3f}小时，更新计时和为{sum(b['optimizer_hours'] or 0 for b in branches):.3f}小时，H32独立评估采样计时和为{sum(b['H32_sampling_hours'] for b in branches):.3f}小时。计时口径不同，不相互相加冒充总墙钟；没有计入源轨迹、前置观测、旧E、失败attempt、排队或空闲。\n\n有逐步元数据时才生成training_updates.csv；本次缺失逐步计时的格子为空，计时子集为空时其求和0并不表示训练没有成本。branch_summary.csv记录分支汇总。损失或梯度为0的步数保留为诊断，不能跨GDPO与其它算法直接解释成有效学习信号强弱。\n"
    (out/'RUNTIME_AND_RESOURCE_zh.md').write_text(runtime)
    (out/'README_zh.md').write_text(intro+"\n## 已完成的提前分析\n\n- [开发结果与响应矩阵](DEVELOPMENT_RESULTS_zh.md)\n- [信息冗余审计](STATE_VARIABLE_AUDIT_zh.md)\n- [实测计算量](RUNTIME_AND_RESOURCE_zh.md)\n- [16个历史E端点复核](HISTORICAL_E_zh.md)\n- [响应矩阵图](response_matrix.png)\n\n## 收齐后的后续步骤\n\n1. 加入最后缺失分支，核验88个唯一分支和16个旧E的覆盖与原件回执。\n2. 更新最后两项计时与全批分析，补全首四条报告并交付，写交付回执。\n3. 停用当前固定清单入口后，按既定协议推进剩余开发/调参；不因当前分数修改配方、seed或目标。\n4. 开发/调参完成后拟合四层相同候选选择器，冻结N/m与规则，再先写测试decision.json，仅执行选择与固定对照去重并集。\n\n本目录不创建FIRST_FOUR_DELIVERED.json、freeze.json或测试decision.json。数据和图仅保存在本地，原GitHub外发授权限制仍有效。\n" )
    return facts,branches

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--baseline-timing',type=Path);args=p.parse_args()
    data=json.loads(args.input.read_text());facts,rows=build(data,args.out,json.loads(args.baseline_timing.read_text()) if args.baseline_timing else None)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/ssvc-development-progress-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=np.full((8,11),np.nan)
    for b in rows:a[ORIGINS.index(b['origin']),RECIPES.index(b['recipe'])]=100*b['J']
    fig,ax=plt.subplots(figsize=(13,5));im=ax.imshow(np.ma.masked_invalid(a),cmap='viridis',aspect='auto',vmin=74,vmax=80)
    ax.set_xticks(range(11),RECIPES,rotation=35,ha='right');ax.set_yticks(range(8),ORIGINS)
    for i in range(8):
        for j in range(11):ax.text(j,i,'pending' if np.isnan(a[i,j]) else f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8,color='black' if np.isnan(a[i,j]) or a[i,j]>77 else 'white')
    ax.set_title(f"Development H32 J (%) — {len(rows)}/88 audited branches; one continuation seed")
    fig.colorbar(im,ax=ax,label='J (%)');fig.tight_layout()
    fig.savefig(args.out/'response_matrix.png',dpi=180);fig.savefig(args.out/'response_matrix.pdf');plt.close(fig)
    (args.out/'INPUT_SHA256.txt').write_text(hashlib.sha256(args.input.read_bytes()).hexdigest()+'  '+args.input.name+'\n')
    if args.baseline_timing:
        with (args.out/'INPUT_SHA256.txt').open('a') as f:f.write(hashlib.sha256(args.baseline_timing.read_bytes()).hexdigest()+'  '+args.baseline_timing.name+'\n')
    print(json.dumps({k:v for k,v in facts.items() if k not in ('information_audit','static')},ensure_ascii=False))

if __name__=='__main__':main()
