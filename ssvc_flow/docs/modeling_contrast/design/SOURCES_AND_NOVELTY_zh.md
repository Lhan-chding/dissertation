# 来源、证据层级与新颖性边界

核对日期：2026-09-15。

## A. 研究事实来源

1. 用户本轮完整分析附件：`d48b4ebd-5316-46e8-abf3-cc8a0231a764.zip`，来源提交 `c71c60b4dad2d4432869eb195bb82069ebde374f`。
2. 先前独立复核：`SSVC_Modeling_Independent_Review_20260915/REPORT_zh.md`，总体/候选差异、噪声、方向、退化消融与成本表。该报告中的复算是本轮设计依据，本交接没有重新执行全部原研究。
3. GitHub读：`https://api.github.com/repos/Lhan-chding/dissertation/branches?per_page=100`。
4. 已核读源码：`https://github.com/Lhan-chding/dissertation/blob/c71c60b4dad2d4432869eb195bb82069ebde374f/ssvc_flow/src/modeling_qualification/models.py`；附件同提交的 `toy.py`、`collection.py`。前者确认当前普通ridge和低秩规则，后两者确认有限候选、精确概率计算及共享观测路径。

本包 `evidence/` 是小范围证据摘录，不是全部原研究raw数据。`parent_module_inventory.json` 是附件源码的文件大小和字节hash。不要把Git blob SHA和文件SHA256混用。

## B. 外部理论与最近邻工作

以下仅将本轮采用的工具连接到相应文献；不是完成全部相关工作审计，也不声称已经证明新颖性。

### B1. Monte Carlo采样与共同密度观测

Eric Veach, Leonidas J. Guibas. *Optimally Combining Sampling Techniques for Monte Carlo Rendering*. SIGGRAPH 1995.

- 作者页面：https://graphics.stanford.edu/papers/combine/
- 论文DOI：https://doi.org/10.1145/218380.218498
- 本次核读作者页面摘要与出版记录。混合分布、多重重要性采样与减方差是既有方法；本计划的单proposal差异恒等式由明确公式直接推导，不声称该论文研究了我们的RL任务。
- 拟访问Art Owen个人Monte Carlo页面时返回403；本包不把未成功取得的内容作为已核读证据。

Ivo Kondapaneni et al. *Optimal Multiple Importance Sampling*. 2019.

- 作者页面：https://cgg.mff.cuni.cz/~jaroslav/papers/2019-optimal-mis/index.html
- 本次核读摘要。其方法已经联系MIS与control variates，因此共同样本/控制变量本身不能认领为新贡献。

### B2. 降秩回归与类别模型

Alan Julian Izenman. *Reduced-rank regression for the multivariate linear model*. Journal of Multivariate Analysis, 1975, 5(2):248–264.

- 出版社摘要：https://www.sciencedirect.com/science/article/pii/0047259X75900421
- DOI：10.1016/0047-259X(75)90042-1。
- 本次核读摘要；降秩回归及秩选择具有长期基础，不以C5/C6的SVD步骤宣称首创。

Thomas W. Yee, Trevor J. Hastie. *Reduced-rank vector generalized linear models*. Statistical Modelling, 2003.

- 出版社：https://journals.sagepub.com/doi/10.1191/1471082X03st045oa
- 本次核读摘要。类别/多项响应的降秩建模已有先行工作；本轮没有自动切换到新的广义线性模型来追求更复杂架构。

### B3. 训练动态、主动能力评估与优化器信息

Mengzhou Xia et al. *LESS: Selecting Influential Data for Targeted Instruction Tuning*. 2024.

- https://arxiv.org/abs/2402.04333
- 摘要明确使用optimizer-aware、低维梯度特征和Adam适配。实际更新/低维梯度本身不是我们的独立创新。

Yuxiang Luo et al. *TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins*. 2026-06-16.

- https://arxiv.org/abs/2606.17660
- 本次核读摘要与版本信息。它已经利用短探测的动态特征预测完整微调表现；不能以“用小预算预测训练效果”作为宽泛区别。

Xuanwen Ding et al. *AutoJudger: An Agent-Driven Framework for Efficient Benchmarking of MLLMs*. ACL 2026.

- https://aclanthology.org/2026.acl-long.685/
- 本次核读ACL正式摘要。能力分解、能力估计与主动问题选择已有直接框架；概率云画像与自适应评估不天然新颖。

## C. 本轮允许与不允许的贡献表述

允许：

- 以本系统数据检验总体变化与候选差异是否需要不同表示；
- 比较不同观测权限与成本下，目标相关方向的可辨识性；
- 展示某估计器在预先固定的有限预算、独立CPU训练随机性下的性能及失效范围；
- 依据实际结果决定是否需要低维预测器。

不允许：

- 把直接相减、GLS、RRR、CRN、IS或混合proposal命名成全新理论；
- 宣称本轮已经发现模型内部唯一概率运输或已证明安全训练；
- 只因算法组合不同就认定不与他人重合；
- 在当前16动作toy中忽略可直接评分的X/I事件，却声称大幅节约计算。

更窄的待检验目标：建立任务目标、候选更新激励、观测相关性和成本共同决定的“可辨识响应维数”资格范围。是否足以成为论文增量，需要独立结果和下一轮针对性文献比较后再判断。
