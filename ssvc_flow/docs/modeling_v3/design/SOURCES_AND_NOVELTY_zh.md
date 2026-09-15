# 来源、核读范围与新颖性边界

检索/仓库核对日：2026-09-15。此处记录本轮针对性核对，不声称完成了该领域所有论文的排查。

## A. 用户实验与仓库（设计的主要依据）

- 用户上传 `e2d70224-ff15-4cf0-a0fb-ebaebec385d2.zip`：第二轮观测与预测结果。
- 前次独立分析 `SSVC_Contrast_V2_Independent_Review_20260915`：本包只复制摘要和关键表至 `evidence/`。这些结果在此前已公开，仅作开发依据。
- 当前 GitHub 分支 `codex/ssvc-modeling-contrast-v2-20260915`，HEAD `474601e020184776265fbfc0ffea343bb56eb21e`。
- 已重新读取 `src/modeling_contrast/observations.py` 与 `estimators.py`：确认主观测服务保存概率贡献、GLS按prompt内联合协方差白化，以及FULL回归/rank的实际构造。
- 所有新实验、种子、128步轨迹、pilot修正、方向选择和两卡方案是本轮建议，不是附件声称已做过的工作。

## B. 本轮核对的一手公开来源

### R1. Safe and Effective Importance Sampling

Art Owen, Yi Zhou. Journal of the American Statistical Association 95(449):135–143, 2000.
DOI: 10.1080/01621459.2000.10473909。
https://www.tandfonline.com/doi/abs/10.1080/01621459.2000.10473909

本轮核对范围：出版方题名、作者、摘要与元数据。其摘要明确讨论混合重要性采样、控制变量和带符号被积函数。不能把“差值可正可负＋混合proposal＋控制变量”视为无人研究的新统计机制。

### R2. Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization

Shanika L. Wickramasuriya, George Athanasopoulos, Rob J. Hyndman. JASA 114(526):804–819, 2019.
DOI: 10.1080/01621459.2018.1448825。
https://www.tandfonline.com/doi/abs/10.1080/01621459.2018.1448825
https://southwest.robjhyndman.com/publications/mint/

本轮核对范围：出版方及作者公开页面的摘要、元数据。以协方差信息使估计满足线性一致性约束属于既有思想。本包的零和控制变量公式是对当前事件差的具体应用；没有宣称与MinT逐项实现等价，也没有宣称新的通用最优性。

### R3. Active Learning for Identification of Linear Dynamical Systems

Andrew Wagenmaker, Kevin Jamieson. COLT / PMLR 125:3487–3582, 2020.
https://proceedings.mlr.press/v125/wagenmaker20a.html

本轮核对范围：正式会议页面、摘要。主动设计输入以辨识响应已有理论；QR、D-opt或“增加激励”不自动构成新颖性。其线性动态系统保证不能直接移植到神经网络训练。

### R4. LESS: Selecting Influential Data for Targeted Instruction Tuning

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. ICML / PMLR 235:54104–54132, 2024.
https://proceedings.mlr.press/v235/xia24c.html
https://arxiv.org/abs/2402.04333

本轮核对范围：正式会议页面与论文摘要。该工作已有Adam感知的低维梯度特征与数据影响估计。我们不能将“低维＋优化器”本身认作贡献。

### R5. AutoJudger: An Agent-Driven Framework for Efficient Benchmarking of MLLMs

Xuanwen Ding 等，ACL 2026，15009–15034。
https://aclanthology.org/2026.acl-long.685/

本轮核对范围：正式会议页面、摘要与元数据。已包含能力分解、能力估计和自适应问题选择。概率画像和主动评估本身不是本轮方法的新颖性。

### R6. The Dark Room in the Reward Channel

Yu Wang，arXiv:2607.21273，2026。
https://arxiv.org/abs/2607.21273

本轮核对范围：arXiv摘要页。全失败组中归一化抵消辅助权重等现象属于重要近邻；本轮不重新认领该机制。版本之间题名和实验规模可能变化，若在论文中引用具体表或预警算法，必须按对应版本全文核验，不用摘要填补细节。

### R7. GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization

Shih-Yang Liu 等，arXiv:2601.05242，2026。
https://arxiv.org/abs/2601.05242

本轮核对范围：论文摘要及官方NeMo GRPO文档中对应算法段落。
https://docs.nvidia.com/nemo/rl/nightly/guides/grpo.html

当前实验目标是观测/预测，不将“加一个动态reward规则”作为已完成方法。若以后进入训练控制，GDPO等方法需要在相同任务/目标下实现为基线；本轮不因为缺该训练臂而冒称胜过GDPO。

### R8. Game-Theoretic Statistics and Safe Anytime-Valid Inference

Aaditya Ramdas, Peter Grünwald, Vladimir Vovk, Glenn Shafer. Statistical Science 38(4):576–601, 2023.
DOI: 10.1214/23-STS894。
https://doi.org/10.1214/23-sts894

本轮核对范围：正式出版页面、摘要。顺序监测和可选停止需要相应推断工具。模型不断改变时的瞬时概率不能直接套用固定均值区间。

## C. 本轮可争取的增量

不是：概率云图、Helmert、GLS、控制变量、D-opt、低秩、Adam历史或No-X门控的单独出现。

待验证：对于多模态RL中的嵌套语义事件，**如何区分可测量的差值、校准可覆盖的差值和能够可靠预测的差值；如何在真实更新与有噪声序列观测下形成可检验的适用范围**。

新增算力用于排除低样本/低方向覆盖的混淆，并建立真实模型证据，不用于用更大样本掩盖与经典方法等价的事实。

## D. 文献审查交付要求

Codex在实现时建立 `RELATED_WORK_IMPLEMENTATION_MAP.md`：每个基线给出其实际公式、来源版本、对应代码、相同信息权限、是否完整复现。未核对全文时只用“受该思想启发的基线”，不要给出完整复现标签。

论文创新的确认仍需后续针对最终算法的全文比对，本包不是新颖性证书。
