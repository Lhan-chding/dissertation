#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact partition of existing counts; no raw output parsing, scoring or fitting."""
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
VERSION = '7af92dff42e9c0432716550f6611dc244af23ecb'
SOURCE = 'ssvc_flow/src/prospective_selection/semantics.py'
INPUTS = ['endpoint_prompt_metrics.csv', 'prompt_posthoc_classes.csv', 'support_response_summary.csv']
META = ['level','stage','branch_recipe','H','panel','origin_id','lineage_id','family','interface','prompt_id','posthoc_H32_class']
COUNT = ['draws','valid_n','X_n','I_n','F1_n','copy_n','F0_noncopy_n','F1_damage_n','nonX_n','B_sum_valid','M_sum_valid','coordinate_correct_sum','coordinate_denominator','relation_full_wrong_n','C_sum']
PARTS = ['I_n','copy_n','F0_noncopy_n','F1_damage_n']

def read(name):
    return list(csv.DictReader((BASE / name).open()))

def write(name, rows):
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

def enrich(r):
    assert all(r[k] >= 0 for k in COUNT), r
    assert r['draws'] == r['valid_n'] + r['I_n']
    assert r['X_n'] == r['F1_n'] - r['F1_damage_n']
    assert r['nonX_n'] == sum(r[k] for k in PARTS)
    assert r['draws'] == r['X_n'] + r['nonX_n']
    for k in ['X_n','F1_n',*PARTS]:
        r[k.removesuffix('_n')+'_rate_all_draws'] = r[k] / r['draws']
    for k in PARTS:
        r[k.removesuffix('_n')+'_share_nonX'] = r[k] / r['nonX_n'] if r['nonX_n'] else ''
    r['F1_rate_valid'] = r['F1_n']/r['valid_n'] if r['valid_n'] else ''
    r['B_mean_valid'] = r['B_sum_valid']/r['valid_n'] if r['valid_n'] else ''
    r['M_mean_valid'] = r['M_sum_valid']/r['valid_n'] if r['valid_n'] else ''
    r['C_mean'] = r['C_sum']/r['draws']
    r['coordinate_accuracy'] = r['coordinate_correct_sum']/r['coordinate_denominator']
    r['relation_full_wrong_rate'] = r['relation_full_wrong_n']/r['draws']
    return r

def main():
    # Read the pinned experiment source as text only. No parser is rerun or changed.
    source = subprocess.check_output(['git','show',f'{VERSION}:{SOURCE}'],cwd=ROOT,text=True)
    for marker in ['if len(changed) != 1:','f = int(prediction[j] == truth[j])','copy=int(m == 0)','F=None','B=None','M=None','(row["event"] == "X") != (f == 1 and b == 0)']:
        assert marker in source, marker
    classes = {(r['stage'],r['prompt_id']):r['posthoc_H32_class'] for r in read(INPUTS[1])}
    rows = []
    for r in read(INPUTS[0]):
        z = {k:r.get(k, '') for k in META}
        z.update(level='endpoint_prompt', posthoc_H32_class=classes[r['stage'],r['prompt_id']])
        for k in COUNT:
            if k in r:
                v = float(r[k]); z[k] = int(v) if v.is_integer() else v
        z.update(F1_n=int(r['F_sum_valid']),copy_n=int(r['copy_sum_valid']))
        z['F1_damage_n'] = z['F1_n']-z['X_n']
        z['F0_noncopy_n'] = z['valid_n']-z['F1_n']-z['copy_n']
        z['nonX_n'] = z['draws']-z['X_n']
        rows.append(enrich(z))
    endpoint_n = len(rows)
    specifications = {
        'origin': ['origin_id','lineage_id'],
        'origin_group': ['origin_id','lineage_id','family','interface'],
        'stage': [],
        'stage_group': ['family','interface'],
        'stage_prompt': ['family','interface','prompt_id'],
    }
    original = list(rows)
    for level, dimensions in specifications.items():
        keys = ['stage','branch_recipe','H','panel',*dimensions]
        for by_class in [False,True]:
            if level == 'stage_prompt' and not by_class:
                continue
            group_keys = keys + (['posthoc_H32_class'] if by_class else [])
            buckets = defaultdict(list)
            for r in original:
                buckets[tuple(r[k] for k in group_keys)].append(r)
            for key, rr in sorted(buckets.items()):
                z = dict.fromkeys(META,'ALL')
                z.update(level=level, **dict(zip(group_keys,key)))
                z.update({k:sum(r[k] for r in rr) for k in COUNT})
                rows.append(enrich(z))
    rows = sorted(rows, key=lambda r:tuple(str(r[k]) for k in META))
    write('failure_decomposition.csv', rows)
    refkeys = [k for k in META if k != 'branch_recipe']
    r0 = {tuple(r[k] for k in refkeys):r for r in rows if r['branch_recipe']=='R0'}
    deltas = []
    for r in rows:
        b = r0[tuple(r[k] for k in refkeys)]
        assert b['draws'] == r['draws']
        d = {k:r[k] for k in META}
        d.update(draws=r['draws'],R0_draws=b['draws'],X_n=r['X_n'],R0_X_n=b['X_n'],F1_n=r['F1_n'],R0_F1_n=b['F1_n'],F1_damage_n=r['F1_damage_n'],R0_F1_damage_n=b['F1_damage_n'])
        for key in ['valid_n','B_sum_valid','M_sum_valid','C_sum','coordinate_correct_sum','coordinate_denominator','relation_full_wrong_n']:
            d[key] = r[key]
            d['R0_'+key] = b[key]
        for key in ['X_n','F1_n','F1_damage_n',*PARTS[:-1]]:
            d['delta_'+key] = r[key]-b[key]
            d['delta_'+key.removesuffix('_n')+'_pp'] = 100*(r[key]-b[key])/r['draws']
        assert d['delta_X_n'] == d['delta_F1_n']-d['delta_F1_damage_n']
        for metric in ['B_mean_valid','M_mean_valid','C_mean','coordinate_accuracy','relation_full_wrong_rate']:
            d['delta_'+metric] = r[metric]-b[metric] if r[metric]!='' and b[metric]!='' else ''
        deltas.append(d)
    write('failure_delta_decomposition.csv',deltas)
    # Pool recipes only for a descriptive census; this is not a deployed mixture policy.
    pooled = []
    for stage in ['early','late']:
        for label in ['ALL','unseen_X','mixed','all_X']:
            rr = [r for r in original if r['stage']==stage and r['H']=='32' and (label=='ALL' or r['posthoc_H32_class']==label)]
            z = dict.fromkeys(META,'ALL')
            z.update(level='stage_all_recipes',stage=stage,H='32',panel='D',posthoc_H32_class=label)
            z.update({k:sum(r[k] for r in rr) for k in COUNT})
            z.update(prompt_count=len({r['prompt_id'] for r in rr}),endpoint_count=len({(r['origin_id'],r['branch_recipe']) for r in rr}))
            pooled.append(enrich(z))
    write('failure_pooled_summary.csv',pooled)
    delta_stages = [r for r in deltas if r['level']=='stage' and r['H']=='32' and r['posthoc_H32_class']=='ALL']
    write('failure_action_tradeoffs.csv',delta_stages)
    support = read(INPUTS[2])
    late_support = next(r for r in support if r['aggregation_level']=='stage' and r['source_stage']=='late')
    scope = {
        'command':f'python3 {Path(__file__).relative_to(ROOT)}',
        'git_commit_at_execution':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs':[{'path':str(BASE/n),'sha256':hashlib.sha256((BASE/n).read_bytes()).hexdigest()} for n in INPUTS],
        'semantic_source':{'git_commit':VERSION,'path':SOURCE,'sha256':hashlib.sha256(source.encode()).hexdigest()},
        'checks':{'endpoint_prompt_rows':endpoint_n,'failure_rows':len(rows),'delta_rows':len(deltas),'all_count_partitions_exact':True,'all_delta_count_identities_exact':True,'all_counts_nonnegative':True,'same_R0_denominators':True},
        'execution_scope':dict(new_training_runs=0,new_model_calls=0,new_samples=0,selector_refits=0,hyperparameter_searches=0,gpu_jobs_submitted=0,statistical_resamples=0,new_scoring_calls=0,final_T_reads=0),
        'interpretation':['F is binary on valid draws only; I repair coordinates remain missing. F1_n/draws means observed valid F1 mass, not imputing F=0 for I.', 'X=F1-F1damage and nonX=I+copy+F0noncopy+F1damage are exact observed count identities.', 'F1damage means F1 with B>0; it is not a measured recoverable gain or a new policy.', 'All_H32 prompt classes are posthoc; H8 inherits H32 class solely for diagnosis.', '4 independent lineages; recipes and prompts are repeated measurements, not new independent states.', 'Training/evaluation pools correspond only by stage/family/interface, never by prompt identity or causal mapping.'],
    }
    (HERE/'failure_scope.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2)+'\n')
    text = ['# 不新增训练即可完成的失败结构定位','', '本分析仅复用已交付端点逐题计数、事后分类与训练支持摘要；未读取原始回答或最终T，未运行parser/评分、模型、训练或拟合。', '', '## 1. 原问题已分清：晚期28题并非修好原坐标后又破坏其他坐标', '', '从原实验固定版本的定义核对：合法输出的F仅为0/1，X等价于F=1且B=0；copy表示原样复制被污染输入，因此F=0；I的F/B/M保持缺失。由此直接得到：', '', '`nonX = I + copy + (valid − F1 − copy) + (F1 − X)`', '', '`X = F1 − F1damage`，其中 `F1damage = F1 − X`。这些是已有输出计数恒等式，不是干预、潜在收益或反事实效果估计。', '', '|源阶段 / H32事后类|题数|输出分母|nonX|I|copy|F0非copy|F1但B>0|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in pooled:
        text.append('|'+ '|'.join(str(x) for x in [r['stage']+' / '+r['posthoc_H32_class'],r['prompt_count'],r['draws'],r['nonX_n'],*[r[k] for k in PARTS]])+'|')
    text += ['', '表中每阶段均为4原点×11已运行动作×144题×16输出；类别子集每题704条。晚期28题合计19,712条输出，F1=0/17,529合法输出，另2,183条I不可定义F。这直接排除了“这些题主要已修好原错误坐标，只是同时破坏其他坐标”的解释。它支持现有记录缺少对原错误坐标的正确修复，但不说明未来永远不可能修好，也不等同于所有输出没有修改。9,150次copy与8,379次修改后仍F0须分开。', '', '晚期全部F1damage的175条均来自8道mixed题，占所有21,672条错误的0.8074935%。这些175条不能被当成可自动兑现的收益；尚未测试的干预可能同时改变F与B。', '', f"训练支持摘要复核：晚期无X组{late_support['no_X_n']}组，其中含F1={late_support['no_X_with_F1_n']}组、具有可定义F观测={late_support['no_X_with_F_observed_n']}组。训练池与评估池只有家族/接口/阶段层面的对应，不可声称某训练组导致这28题的失败。", '', '## 2. 11动作相对R0的pX差来自什么', '', '以下按阶段合并4原点，分母均为9,216输出/动作；百分点变化严格使用同一总输出分母。ΔpX = Δ观测F1质量 − ΔF1damage质量；F1质量的分母包含I，仅为合法F1事件占总输出的质量，并未将I的F补零。', '', '|阶段|动作|ΔX计数|ΔF1计数|ΔF1damage计数|ΔpX (pp)|ΔF1 (pp)|ΔF1damage (pp)|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in delta_stages:
        text.append('|'+ '|'.join(str(x) for x in [r['stage'],r['branch_recipe'],r['delta_X_n'],r['delta_F1_n'],r['delta_F1_damage_n'],*[f'{r[k]:.6f}' for k in ['delta_X_pp','delta_F1_pp','delta_F1_damage_pp']]])+'|')
    text += ['', '例如晚期R3相对R0，正确修复原坐标增加13条，但F1damage也增加13条，pX差恰为0；晚期R4则F1增加8条、F1damage增加18条，最终X减少10条。早期R4的F1增加79条、F1damage增加29条，最终X增加50条。以上均为9,216条/动作的已观测比较，不能视为一一配对回答的状态转换。', '', '原有指标也存在方向不一致：早期R3的pX增加0.509983个百分点，坐标准确率却下降1.703559个百分点、全关系成立但答案错误率增加1.312934个百分点；晚期R2的坐标准确率增加2.145725个百分点、copy增加2.560764个百分点，但pX下降0.108507个百分点。不能用这些辅助指标替换原pX目标判胜。', '', '配方差可能包含正确修复增多、修好后破坏减少，或二者互相抵消，不能只看F的单指标变化。`failure_action_tradeoffs.csv`同时列出相对R0的copy/I/F0非copy、C、坐标准确率、全关系成立错误率、有效B/M变化。此表不新增综合分；DIRECT_REPAIR_R4、GDPO_R4、SAW_R4保持外部基线身份，不并入原R0–R7选择器。', '', '## 3. 当前可解决与仍不可解决的问题', '', '- 已解决：主要剩余错误属于未正确修复原坐标，还是正确修复后额外破坏；并已按逐题、原点、早晚阶段、11配方、6家族×接口群体与H32事后类拆分。', '- 已解决：各动作pX变化在代数上由多少F1增加与F1damage改变构成，并同步保留其他原有指标供权衡。', '- 现有数据仍不能证明：换训练支持、换动作、去掉额外破坏或降低收缩后会实现多少新收益；观察计数不是干预效果，也不能确定未见F1的模型/数据/优化唯一成因。', '', '完整行组：`failure_decomposition.csv`；同分母R0差分：`failure_delta_decomposition.csv`；跨动作描述汇总：`failure_pooled_summary.csv`。所有14,784逐题端点行以及汇总行通过非负计数、分区计数恒等式和R0差分恒等式验证。4条lineage是独立来源单位；表中大量输出不构成额外独立训练seed。', '']
    (HERE/'FAILURE_FINDINGS_zh.md').write_text('\n'.join(text))
    print(json.dumps(scope['checks']))

if __name__ == '__main__':
    main()
