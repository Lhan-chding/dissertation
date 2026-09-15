# 来源与核读范围

核读日期：2026-09-15。下列外部文献使用一手来源；本次为本建模阶段补充核验，不是对全部相关领域的穷尽新颖性认证。

## A. 本项目来源

### [P1] 上一版建模设计

`SSVC_Modeling_Design_20260915/MODELING_AND_OBSERVATION_zh.md`，本会话实际附件。

本轮沿用：X/S/W/I、(v,q,s) 解释坐标、身份保留、响应维数与条件误差界。

本轮新增/细化：默认线性 Helmert 拟合坐标；局部校准的成本和范围；正交残差范数不足以识别方向的反例；逐步净位移界与路径界同时比较；只做建模、暂缓主动检测与控制。

### [G1] 仓库分支快照

通过连接的 GitHub 读取：

https://api.github.com/repos/Lhan-chding/dissertation/branches?per_page=100

后续分支当时为 `97b7fd646046cfd58c45e0083f0e8fe6f37d6f1e`，与旧文档的 `7e405faa...` 不同。此次只读，没有提交代码。

### [G2] 本地实现报告

https://github.com/Lhan-chding/dissertation/blob/97b7fd646046cfd58c45e0083f0e8fe6f37d6f1e/ssvc_flow/docs/mechanism_followup/IMPLEMENTATION_REPORT_zh.md

核读完整报告：当前 followup 模块职责、CPU 验收计数、代码/数据绑定、停止边界。报告中的历史 CPU 状态不能代替最新服务器状态；本次没有重新执行其测试，也没有访问服务器确认 GPU。

### [G3] 最新执行文档提交

https://github.com/Lhan-chding/dissertation/commit/97b7fd646046cfd58c45e0083f0e8fe6f37d6f1e

核读 commit diff：记录 S1 五个原 bank 完成、composite 和部分历史 R2 在对应快照中执行中。这只用于避免沿用过时的“完全没有非零响应”判断。是否已最终完成，应在 Codex 实施时按完成产物再次核实。

### [G4] 数值场景生成代码

https://github.com/Lhan-chding/dissertation/blob/97b7fd646046cfd58c45e0083f0e8fe6f37d6f1e/ssvc_flow/src/generate_worlds.py

本次核读生成规则、家族、运算、split 容量约束、唯一解和 `render` 分支。Codex 必须查看实际入口签名，不得把本包拟实施 CLI 当成既有函数。

## B. 外部相关工作

### [R1] Predictive Representations of State

Littman, Sutton, Singh，NeurIPS 2001。

https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html

本次核读官方摘要/条目。作用：预测式状态表达是已有思想。本包不声称首次用概率表示隐藏状态，也不将局部线性响应等同于完整 PSR。

### [R2] AutoJudger: An Agent-Driven Framework for Efficient Benchmarking of MLLMs

Ding et al.，ACL 2026。

https://aclanthology.org/2026.acl-long.685/

本次核读官方摘要及发表信息：能力分解、能力估计、主动题目选择。作用：能力画像和主动评估不能单独作为本项目创新。本轮不复现其 MLLM agent；不同任务目标也不应声称直接超过它。

### [R3] TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins

Luo et al.，arXiv:2606.17660v1，2026-06-16。

https://arxiv.org/html/2606.17660v1

本次核读摘要、§4 及特征/评价部分：使用静态数据描述与短训练动态特征预测微调表现。作用：普通训练信息预测基线必须保留。我们的 B1 是任务适配的日志 baseline，不叫 TuneAhead 精确复现；本包不引用其图表或实验数值作本项目证据。

### [R4] Computing Active Subspaces with Monte Carlo

Constantine, Gleich，arXiv:1408.0545。

https://arxiv.org/abs/1408.0545

本次核读一手摘要：基于梯度乘积矩阵估计重要方向，研究样本数与估计精度，涉及近似梯度误差。

补充一手说明：
https://epubs.siam.org/doi/10.1137/1.9781611973860.ch6

作用：梯度/响应方向降维及其样本误差不是空白。本包的 B4/秩分析使用这一类标准思想，不能仅重新命名后认领为原创。

### [R5] LESS: Selecting Influential Data for Targeted Instruction Tuning

Xia et al.，ICML 2024。

https://proceedings.mlr.press/v235/xia24c.html

本次核读官方摘要/发表信息：Adam-aware 影响估计与低维梯度特征。作用：实际优化器方向与低维影响分析已有先行工作。本轮比较 actual d 与 SGD proxy 是资格诊断，不称首次考虑 Adam。

### [R6] Active Testing: Sample-Efficient Model Evaluation

Kossen et al.，ICML 2021。

https://proceedings.mlr.press/v139/kossen21a.html

本次核读官方摘要。作用：主动评估和有限测试预算已有系统研究。本轮先使用固定面板，主动选题与选择偏差修正留到下一阶段。

### [R7] Optimizing Neural Networks via Koopman Operator Theory

Dogra, Redman，arXiv:2006.02361。

https://arxiv.org/abs/2006.02361

本次核读一手摘要。作用：训练动态表示与低维/线性化预测已有工作。本包不宣称概率坐标满足 Koopman 闭合，也不声称第一次建模训练动态。

## C. 本轮研究边界

当前目标是检验：在明确的语义事件和有限观测预算下，什么表示和多少响应维数足以预测局部行为变化，在哪些条件下失效。经典表示/回归/降维若已经足够，必须如实保留该结论。

Fisher 恒等式、秩截断和 Taylor 不等式属于基础推导。数值测试只是核验实现，既不是一般数学证明，也不是新颖性证明。需要新理论/方法时，应针对资格实验中确实剩余的问题继续设计。
