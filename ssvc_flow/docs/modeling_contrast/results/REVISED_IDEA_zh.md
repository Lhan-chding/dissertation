# 修订后的建模对象

本轮把总体响应T、主响应B及候选差异C分别命名，直接用e拟合C，同时保留父d→T后相减、零基线及同权限直接测量。候选差异共享起点噪声严格消去，但共同baseline和测量packet相关项仍需联合协方差。

O-IND、CRN、共同样本LR和候选混合是既有统计工具；C5/C6为透明两阶段投影，不将IS/CRN/GLS/RRR/PCA认领为新理论。总体低秩好不保证低能量关键差异好；216是观测坐标数，k/r由固定fit bank激励决定。

本轮实际开发决定 `RESOURCE_REVIEW_REQUIRED`；N4 `NOT_RUN`。方法是否值得部署依据MODELING_DECISION_V2_zh.md及N3锁，不预设需要更复杂的预测器。

来源：[MODEL_SELECTION_LOCK.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N3/MODEL_SELECTION_LOCK.json)；SHA-256 `559a1b513d49c1683e19155c4183cfc0c34daf92d0892c7fbfa46f6709c88645`
