"""Readable facts and diagnostic tables for the exploratory first-four replay."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.prospective_selection.pilot_four import table


def main():
    p=Path('docs/prospective_selection/pilot_four_20261004');s=json.loads((p/'summary.json').read_text())
    rows=list(csv.DictReader((p/'selected_outcomes.csv').open()));pred=list(csv.DictReader((p/'all_candidate_predictions.csv').open()))
    tuning=[];diagnostics=[]
    for f in sorted((p/'folds').glob('*.json')):
        d=json.loads(f.read_text());level=d['model']['feature_level'] if 'model' in d else d['kernel']['feature_level']
        for x in d['inner']:
            tuning.append(dict(level=level,outer_held_lineage=d['held_lineage'],inner_validation_lineage=x['validation_lineage'],
                inner_fit_lineages=';'.join(x['fit_lineages']),alpha=x['alpha'],best_static=x['best_static'],
                selected_J=x['selected_J'],selected_recipes=';'.join(y['recipe'] for y in x['decisions'])))
        for c in d['decisions']:
            vals=c['predictions'];default=d['best_static'];others=[v for a,v in vals.items() if a!=default]
            diagnostics.append(dict(level=level,lineage=d['held_lineage'],origin=c['origin_id'],alpha=d['alpha'],
                default=default,predicted_best=c['predicted_best'],selected=c['recipe'],
                best_nondefault_minus_default_pp=100*(max(others)-vals[default]),
                predicted_action_range_pp=100*(max(vals.values())-min(vals.values())),
                kernel_smallest_eigenvalue=min(d['eigenvalues']),kernel_largest_eigenvalue=max(d['eigenvalues'])))
    table(p/'inner_tuning_288.csv',tuning);table(p/'decision_diagnostics.csv',diagnostics)
    z3=[r for r in rows if r['level'].startswith('Z3')]
    delta=np.array([[100*(float(r['J'])-float(r[k])) for k in ['best_static_J','GDPO_J','SAW_J','DIRECT_REPAIR_J']] for r in z3])
    fig,ax=plt.subplots(figsize=(8.2,5.6));lim=max(abs(delta.min()),abs(delta.max()));im=ax.imshow(delta,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(4),['Fold\nBestStatic','GDPO','SAW','Direct\nrepair']);ax.set_yticks(range(8),[r['origin'] for r in z3])
    for i in range(8):
        for j in range(4):ax.text(j,i,f'{delta[i,j]:+.3f}',ha='center',va='center',color='white' if abs(delta[i,j])>lim*.55 else 'black')
    ax.set_title('Exploratory held-lineage replay: Z3 minus comparator\nAll four information levels selected the same actions')
    fig.colorbar(im,ax=ax,label='H32 J difference (percentage points)');fig.tight_layout()
    fig.savefig(p/'paired_origin_differences.png',dpi=180);fig.savefig(p/'paired_origin_differences.pdf');plt.close(fig)
    method=s['methods'][3]
    methods='\n'.join(f"|{m['level'].split('_')[0]}|{100*m['J']:.4f}%|{m['delta_static_pp']:+.4f}|{m['static_choices']}/8|{m['tie_overrides']}/8|" for m in s['methods'])
    baselines='\n'.join(f"|{label}|{100*method[k]:.4f}%|{100*(method['J']-method[k]):+.4f}|" for k,label in [('best_static_J','折内 BestStatic'),('GDPO_J','GDPO_R4'),('SAW_J','SAW_R4'),('DIRECT_REPAIR_J','DIRECT_REPAIR_R4')])
    picks='\n'.join(f"|{r['origin']}|{r['recipe']}|{100*float(r['J']):.4f}%|{float(r['observed_regret_pp']):.4f}|" for r in z3)
    text=f'''# 首四条小实验：完整探索性离线闭环

**完成状态：EXPLORATORY_FIRST_FOUR_NESTED_LOLO_COMPLETE。** 4条lineage、8个原点、88个真实分支；外层4折整条留出，内层3折调参；4种信息层，16个拟合模型，32个最终选择。没有新增GPU训练、没有使用后续8条开发/4条调参、没有读取最终T结果、没有写主实验freeze.json。

这是已经查看过端点结果的开发数据回放。折内实现隔离标签，仍不能把整轮称为全新独立前瞻确认。它完成的是“前置信息→拟合/调参→对留出lineage选动作→查已观测真实端点→比较”的小实验闭环，适合分析方法和完善idea。

## 主要结果

|信息层|留出H32 J均值|相对折内BestStatic（百分点）|选择等于BestStatic|因tau改写预测首选|
|---|---:|---:|---:|---:|
{methods}

**Z3−Z2=0.0000个百分点，四条lineage各自的配对差也全部为0。** 32个最终决策均有recipe=predicted_best=fold BestStatic。并非预测首选其它动作却被0.005回退阈值强行改回；当前模型本身就预测默认动作最高。61001留出时默认R4，其余留出时默认R3。

|对照|H32 J均值|Z3−对照（百分点）|
|---|---:|---:|
{baselines}

以上为观察到的均值差，不是显著优越结论。全部平均先在同lineage两个anchor内平均，再四lineage等权；只有4个独立训练来源，外层拟合集重叠。SAW/GDPO/DIRECT为固定外部对照，不进入R0–R7可选集合。

|留出原点|四层共同选择|所选J|与同原点R0–R7已观测最高的差（百分点）|
|---|---|---:|---:|
{picks}

平均已观测候选regret={method['observed_regret_pp']:.4f}个百分点。这里最大值使用带噪的已观测开发端点，仅作描述。之前全8原点事后选R3得到77.648%，本次折内BestStatic仅77.539%，原因是每折默认只用另外3条选择；不能拿全数据事后最优当无泄漏基线。

## 调参与决策诊断

Z0、Z2、Z3四折均选alpha=10；Z1在留出61001时选1，其余选10。alpha按内层三个验证lineage的所选动作实际J均值选择，0.001内优先较大值，使用既定六项网格。完整288条内层记录见inner_tuning_288.csv，模型与训练包在models/，每折输入分组、预测、核矩阵和特征规模在folds/。

alpha_sensitivity_POSTHOC.csv保留全部6alpha的外层结果，仅用于诊断，不能从中再挑最好alpha来改写本轮主结果。弱正则下有部分动作变化，但Z2/Z3仍使用相同的动作序列；这项观察不代表所有算法或新数据都不会利用修复结构。

margin_to_default在默认自身最高时按定义为0，不能用它证明各动作预测完全相同。decision_diagnostics.csv另列“最高非默认预测−默认预测”和八动作预测范围。all_candidate_predictions.csv包含32×8=256条预测及对应真实相对R0差，可检查排序、误差及具体失败。

## 可供下一轮idea/算法设计分析的问题

这些是后续可研究的问题，**尚未修改原算法或挑选更好结果**：

1. 内层哪些候选alpha的效用几乎相同，强正则偏好是否削弱了状态依赖？用288条调参记录和核谱定位，不单凭alpha=10下结论。
2. 修复结构的新增差异是否对应动作响应的差异？结合32份packet、原始逐样本结构、88列响应和逐候选预测，区分“结构不同”与“可以预测最佳动作”。
3. Z3当前加入repair时同时将reward核权重由1改成0.5；因此Z3−Z2是完整表示方案对比，不能严格归因于纯新增信息。以后设计消融时需要控制这一变化。
4. D面板有限采样和每分支仅repeat1造成哪些标签不确定性？原始逐draw可研究固定端点评估噪声，但不能当作新增训练seed或独立lineage。
5. GDPO在当前均值上更高，候选奖励配置与当前H32目标之间的关系应结合分阶段/六群体结果检查，而不是先增加计算量。

本轮证据支持的结论仅是：**在当前四条、既定表示/核/调参/选择规则下，尚未观察到Z3的选择增益。** 不支持“修复信息无用”“所有状态选择算法无效”或“扩大样本必然成功”。

## 数据阅读入口

- PILOT_PLAN_zh.md：计算结果前固定的小实验规则。
- selected_outcomes.csv / lineage_scores.csv / method_summary.csv：原点、lineage、方法三个层级结果。
- decisions_before_scoring.json：先落盘的32个选择与全候选预测；summary.json记录其哈希。
- inner_tuning_288.csv、alpha_sensitivity_POSTHOC.csv、decision_diagnostics.csv：调参、敏感性、决策边际。
- models/、folds/：16模型和16折完整诊断。
- paired_origin_differences.png/pdf：按原点与固定对照比较。
- REPLAY_VERIFICATION.json：独立新目录复跑与原输出逐字节一致。
- 上一级first_four_20261004/：88分支完整矩阵、六群体、历史E、信息冗余、吞吐与原件验证。
- 数据包的RAW_MANIFEST.json及覆盖说明：逐样本原件、输入、训练代码与服务器保留张量索引。

所有新GPU任务仍被STOP拦截。此前已启动的4项source可完成并保存，不能进入本轮拟合或评分，不能自动扩展完整矩阵。GitHub数据推送限制仍在，本地分析包不代表已公开发布。
'''
    (p/'PILOT_REPORT_zh.md').write_text(text)

if __name__=='__main__':main()
