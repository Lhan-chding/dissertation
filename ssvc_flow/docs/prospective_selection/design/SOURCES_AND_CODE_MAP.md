# 来源与代码接线

核读日期：2026-09-24。下列结论来自原始论文或仓库；本包的方法设计是待验证建议，不是新颖性认证。

## 1. 直接研究依据

- 用户上传的D2实验包：`2c53dccb-21c8-4f82-8f9d-9ddf5143ffe0.zip`。
- 已交付独立分析：`SSVC_D2_Independent_Review_20260924/REPORT_zh.md`。
- 本包只复制摘要和冗余/精度检查，不再次声称重放了94,720条回答或GPU训练。
- 已知问题：关系奖励收益可能位于错误世界；四事件边际可由X/A/V均值恢复；旧决策读出使用已观测端点排序，尚非前瞻选择；旧同时Hoeffding门槛与采样量不匹配。

## 2. 本次在线核读的仓库

仓库`Lhan-chding/dissertation`，分支`codex/decision-modeling-v1-20260919`，读取提交`b52f55993791a9954be07a0195eb656441b53651`。

| 文件 | 已确认的行为 | 新实现的接法 |
|---|---|---|
| `src/decision_modeling/reward_recipes.py` | R0–R7，GDPO_R4，SAW_R4；不同标准差约定 | 原有配方保持；新直接修复惩罚单独命名，不改旧R4 |
| `src/decision_modeling/block_runner.py` | 同一原点完整状态恢复、实际on-policy PPO/Adam；source sampler继承源seed顺序 | 复用`update_batch`等，增加显式共同branch schedule；不再由origin seed隐式改变比较题序 |
| `src/decision_modeling/panels.py` | P/E各36场景、72prompt；元数据选择；E延后读取 | 保留旧P/E；新增开发结果D和封存测试T面板 |
| `src/decision_modeling/decision_readout.py` | 当前严格可行性/区间读出；不是学习型前瞻selector | 保留为历史统计；新selector与旧safe门禁分开 |
| `src/decision_modeling/score_cache.py` | 评分缓存 | 可继续用；主性能评估不要求全部输出重新评分 |
| `src/decision_modeling/semantic_schema.py` | 语义映射入口（本次核对目录，实现需Codex展开） | 增加repair字段，旧event/parser不变 |

入口不是旧`main`，不要在那里重建一套训练器。首次开发时核对上述函数签名和完整测试；文档里的新增CLI是实现契约，当前不宣称已存在。

## 3. 原始相关工作（控制引用范围）

### R1. GDPO
Liu et al., *GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization*, arXiv:2601.05242, 2026。
原文：`https://arxiv.org/html/2601.05242`
核读§3.1–3.2：先按奖励通道做组内标准化，再聚合并做batch归一化。它不是“根据学习状态自动选择奖励”的算法。保留其完整归一化和priority约定。

### R2. SAW
He et al., *SAW: Stage-Aware Dynamic Weighting for Multi-Objective Reinforcement Learning in Large Language Models*, arXiv:2606.07705, 2026。
原文：`https://arxiv.org/html/2606.07705v1`；记录页：`https://arxiv.org/abs/2606.07705`
核读：基于batch相对变异度动态权重。与本包语义观测反馈非常接近，必须保留。理论最小值、两处delta、标准差修正、priority位置按原文及仓库mapping复核一次。

### R3. Constrained GRPO
Girgis et al., *Constrained Group Relative Policy Optimization*, arXiv:2602.05863, 2026。
原文：`https://arxiv.org/abs/2602.05863`
动态约束/乘子已有直接相关工作。本轮不主张已建立约束控制算法；后续真正进入SSVC在线控制时必须加入等指标约束对照。

### R4. 动态多奖励权重
Lu et al., *Learning to Optimize Multi-Objective Alignment Through Dynamic Reward Weighting*, arXiv:2509.11452。
原文：`https://arxiv.org/abs/2509.11452`
超体积和梯度引导权重已有先行工作。不能以“权重随训练变化”作为本包贡献。后续扩展多目标Pareto实验时纳入，当前核心检验是信息价值。

### R5. 决策导向学习
Elmachtoub and Grigas, *Smart “Predict, then Optimize”*, arXiv:1710.08005。
原文：`https://arxiv.org/abs/1710.08005`
启发：评估最终选择的价值，而非要求所有坐标预测精确。我们采用简单有限动作utility回归，不将这种通用思想认领为新理论。

### R6. 顺序区间
Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*, arXiv:1810.08240。
原文：`https://arxiv.org/abs/1810.08240`
适用边界：反复查看和可选停止需要额外统计处理。本包主检验固定样本量、固定H32；不把固定样本区间反复使用为随时有效保证。

## 4. 本轮可能产生的研究增量

只有当“完整奖励分布/分层历史”已经得到相同证据和公平调参后，额外修复结构仍改善未见原点的实际选择，才能支持语义状态的信息增益。概率云是表示，不是仅凭名称成立的新方法。直接修复奖励基线用于排查：收益是否仅来自增加一项语义监督，而无需状态选择。
