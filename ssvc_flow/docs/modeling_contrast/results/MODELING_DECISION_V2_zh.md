# 建模第二轮决定

**N3决定：`RESOURCE_REVIEW_REQUIRED`。工程总状态：`BLOCKED`。冻结锁完整性：`PASS`；N4实际状态：`NOT_RUN`。**

原件完整轨迹 60/60；N1 状态 `COMPLETE`，轨迹 12；N2 状态 `COMPLETE`，exact 轨迹 60，finite 轨迹 20。所有旧seed均为legacy_diagnostic，旧locked-test只作公开开发/复现，原role保留。

| 门禁 | 状态 | 原因 |
| --- | --- | --- |
| science_gate | PASS | 无失败原因 |
| cost_gate | PASS | 无失败原因 |
| resource_gate | FAIL | UNVERIFIED_OR_EXCEEDED_ADDED_OUTPUT_GIB |
| input_identity_gate | PASS | 无失败原因 |
| data_gate | PASS | 无失败原因 |

来源：[MODEL_SELECTION_LOCK.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N3/MODEL_SELECTION_LOCK.json)；SHA-256 `559a1b513d49c1683e19155c4183cfc0c34daf92d0892c7fbfa46f6709c88645`；[DECISION_INDICATORS.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N3/DECISION_INDICATORS.json)；SHA-256 `bc05ed28bd4fb9e5706b3bf085aa5fbd20e176336c3717f907cbee66f2fc169c`

## 决定摘要与停止原因

推荐直接针对候选差异 C=joint_1−joint_0 测量，并同时保留 no_x_off_1−joint_0 和两者一致性。当前16动作toy中，优先使用已知事件评分：每策略每题最多3个标量logp，生成采样量 n=0，无需代理拟合维数 k/r。216维是72题的Helmert响应表示；不据此预设动力学秩。

完整N4预计总新增结果 **3.962937 GiB**，超过 **3 GiB**；其中新验证增量为 2.013480 GiB。时间预计 1.152 小时、峰值内存预计 2.791 GiB，这两项在上限内。N4因此未运行，新训练、Qwen/GPU调用与在线SSVC均为0。

N3保留的两个候选仅通过开发集门禁，并未获准部署或独立验证：

| 冻结模型/观测 | n | fit banks | rank cap | k范围 | pX NRMSE | v NRMSE | pX误差q95 | v误差q95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C6 / O_LR_MIX | 64 | 2 | 4 | 0–4 | 0.644504 | 0.416279 | 0.000368048 | 0.001069733 |
| C5 / O_LR_ORIGIN | 64 | 2 | 4 | 0–4 | 0.644633 | 0.416834 | 0.000360942 | 0.001064203 |

两个候选的NRMSE均小于0.75，但误差q95均高于主门槛0.000125，故分辨率不通过。每个配置24个selection锚点中有6个无激励锚点；保留这些真实零秩节点，没有把非零消融整体替换为r0。

评分/生成=0.25、标签/生成=0.05的操作数情景下，按60个实际锚点单位汇总：

| 冻结配置 | 校准成本 | 4个bank直接测量成本 | 已知事件评分成本 | 相对直接测量的平衡查询数 |
| --- | --- | --- | --- | --- |
| C6 / O_LR_MIX / fit2 | 553832.85 | 1068105.05 | 25758.00 | 3 |
| C5 / O_LR_ORIGIN / fit2 | 434071.30 | 835434.35 | 25758.00 | 3 |

这些是含付费校准观测的条件操作数比较；查询计算按乐观下界记账，实际拟合/查询CPU组件另有实测。两个候选均没有已知事件评分的成本优势，不主张实际CPU或真实VLM节省。sample-only两种仪器未形成合格的主差异测量建议。

失效范围：旧数据均为事后开发资料；未完成新seed验证，不验证跨初始化、数据分布、真实VLM或在线跟踪。已知事件评分只适用于X/I动作集合已知且可完整枚举的当前toy。

工程证据：新模块213项测试及Ruff通过；N1/N2独立数值复算与N3独立bootstrap/门禁复算均通过。全仓库旧回归的历史fixture缺失、早期失败smoke记账缺口和未分离的部分CPU/I/O时间保留为明确限制，未标全绿。

来源：[冻结锁](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N3/MODEL_SELECTION_LOCK.json)；[完整新验证资源投影](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N3/FRESH_RESOURCE_PROJECTION.json)；[成本前沿](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N2/COST_FRONTIER.csv)。

## 推荐与有限条件

**没有获准部署的局部surrogate配置；不将best_nonzero_diagnostic当成合格推荐。**

在允许已知完整动作归一化logp评分的当前toy，保留直接 `D-KNOWN-EVENT-LOGP` 作为pX/v测量建议：每策略每题最多3个标量评分，同一次16动作forward可复用，生成样本n=0、无需拟合k/r。实际检查600块，最大pX/v绝对误差1.94289e-16。此建议只适用于已知X/I动作集合的有限toy，不扩展为真实VLM事件概率等式。

最优非零开发诊断（不是部署推荐）：配置 `C5|O_LR_ORIGIN|256|4|1e-05|0.01|8`，模型 `C5`，观测 `O_LR_ORIGIN`，n=256，fit banks=8，r=4，k范围=1-14，pX/v NRMSE=0.315287/0.206402。

## 固定目标、语义及计费范围

主目标 `joint_1_minus_joint_0`；次目标 `no_x_off_1_minus_joint_0`；一致性目标是两者之差。
72个固定题目、X/S/W/I四事件存储，拟合216个Helmert响应坐标。
参数空间737维；局部输入子空间维数k及所选r分别记录，不能将3或216解释为动态秩。
同bank的输入为候选终点减joint_0终点的e；T、B、C分别记录。
共同起点计数在C中消去；共享baseline、跨bank历史alias和同packet相关项仍保留。

fit banks 0-7、diagnostic 8-9、evaluation 10-13绑定所有候选。
主n=64，16/256是冻结稳健性；8个noise replicas不增加训练seed数。
O-IND历史fit计数已有5份原副本，其余3份为新增观测；evaluation使用独立新观测。
其余测量的RNG按seed/arm/anchor/bank/n/replica/purpose固定分离。
训练bank样本与control观测分开，eval语义概率在预测文件封存后才用于评分。

成本列包含旧观测购置成本、生成、标签、指定动作评分、缓存/alias复用与拟合；
scenario中评分/生成比为0.05/0.25/1，标签/生成比为0.05。CPU时间另列，不等于真实VLM时间。
surrogate新增查询成本按0计是乐观下界；收支平衡仅在已验证的4个evaluation bank内评价。

## 四种观测的实际比较

主目标、Helmert视图，按误差/真值平方和汇总群体NRMSE；原始事件LR另保留在完整表。

| 观测 | n | pX NRMSE | v NRMSE | 理论方差均值 | 经验方差均值 | 经验/理论 | 未观测X次数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O_CRN | 16 | 8.6915 | 2.06614 | 0.000126467 | 0.000126441 | 0.999801 | 57023 |
| O_CRN | 256 | 2.17148 | 0.521196 | 7.90417e-06 | 7.90602e-06 | 1.00023 | 0 |
| O_CRN | 64 | 4.32751 | 1.03303 | 3.16167e-05 | 3.1486e-05 | 0.995868 | 2522 |
| O_IND | 16 | 84.6253 | 15.0564 | 0.0102086 | 0.0101787 | 0.997074 | 19342 |
| O_IND | 256 | 21.3638 | 3.79707 | 0.000638035 | 0.00063895 | 1.00143 | 0 |
| O_IND | 64 | 42.5313 | 7.56432 | 0.00255214 | 0.00255566 | 1.00138 | 84 |
| O_LR_MIX | 16 | 0.609535 | 0.196498 | 6.03753e-07 | 6.04074e-07 | 1.00053 | 57843 |
| O_LR_MIX | 256 | 0.153925 | 0.0493562 | 3.77345e-08 | 3.76476e-08 | 0.997697 | 0 |
| O_LR_MIX | 64 | 0.305176 | 0.0988342 | 1.50938e-07 | 1.51268e-07 | 1.00218 | 2629 |
| O_LR_ORIGIN | 16 | 0.594887 | 0.192173 | 5.87553e-07 | 5.85581e-07 | 0.996643 | 58572 |
| O_LR_ORIGIN | 256 | 0.149019 | 0.048621 | 3.67221e-08 | 3.67932e-08 | 1.00194 | 0 |
| O_LR_ORIGIN | 64 | 0.296175 | 0.0961917 | 1.46888e-07 | 1.46918e-07 | 1.0002 | 2562 |

来源：[OBSERVATION_COMPARISON.csv](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N1/OBSERVATION_COMPARISON.csv)；SHA-256 `adb33a74beecbab63083803ba33b45da33fccc87d7961cf82471a564f99dd248`

## 目标与方向模型的实际比较

展示每模型/观测/n在公开开发资料中max(pX,v NRMSE)最小的代表行；这张诊断表不改变N3冻结选择。所有被淘汰配置仍在完整表，r_min/r_max包含无激励状态。

| 模型 | 观测 | n | fit banks | rank cap | r_min | r_max | k_min | k_max | pX NRMSE | v NRMSE | pX q95 | v q95 | 配置ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | EXACT | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0.000554442 | 0.00257845 | C0\|EXACT\|0\|0\|0.01\|0.01\|8 |
| C0 | O_IND | 16 | 8 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0.000558146 | 0.00258238 | C0\|O_IND\|16\|0\|0.01\|0.01\|8 |
| C0 | O_IND | 256 | 8 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0.000558146 | 0.00258238 | C0\|O_IND\|256\|0\|0.01\|0.01\|8 |
| C0 | O_IND | 64 | 8 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0.000558146 | 0.00258238 | C0\|O_IND\|64\|0\|0.01\|0.01\|8 |
| C1_LEGACY_EXACT_RULE | EXACT | 0 | 8 | 1 | 1 | 1 | 6 | 22 | 0.896437 | 1.20893 | 0.000527139 | 0.00341049 | C1_LEGACY_EXACT_RULE\|EXACT\|0\|1\|0.01\|0.01\|8 |
| C1_LEGACY_EXACT_RULE | O_IND | 16 | 8 | 1 | 1 | 1 | 6 | 22 | 26.8519 | 6.41054 | 0.0115816 | 0.0121946 | C1_LEGACY_EXACT_RULE\|O_IND\|16\|1\|0.01\|0.01\|8 |
| C1_LEGACY_EXACT_RULE | O_IND | 256 | 8 | 1 | 1 | 1 | 6 | 22 | 6.37633 | 1.915 | 0.00285182 | 0.00414989 | C1_LEGACY_EXACT_RULE\|O_IND\|256\|1\|0.01\|0.01\|8 |
| C1_LEGACY_EXACT_RULE | O_IND | 64 | 8 | 1 | 1 | 1 | 6 | 22 | 12.8951 | 3.43159 | 0.00582692 | 0.00698783 | C1_LEGACY_EXACT_RULE\|O_IND\|64\|1\|0.01\|0.01\|8 |
| C1_LEGACY_FROZEN | EXACT | 0 | 8 | 0 | 1 | 1 | 6 | 22 | 0.896437 | 1.20893 | 0.000527139 | 0.00341049 | C1_LEGACY_FROZEN\|EXACT\|0\|0\|0.01\|0.01\|8 |
| C1_LEGACY_FROZEN | O_IND | 16 | 8 | 0 | 0 | 0 | 6 | 22 | 1 | 1 | 0.000558146 | 0.00258238 | C1_LEGACY_FROZEN\|O_IND\|16\|0\|0.01\|0.01\|8 |
| C1_LEGACY_FROZEN | O_IND | 256 | 8 | 0 | 0 | 0 | 6 | 22 | 1 | 1 | 0.000558146 | 0.00258238 | C1_LEGACY_FROZEN\|O_IND\|256\|0\|0.01\|0.01\|8 |
| C1_LEGACY_FROZEN | O_IND | 64 | 8 | 0 | 0 | 0 | 6 | 22 | 1 | 1 | 0.000558146 | 0.00258238 | C1_LEGACY_FROZEN\|O_IND\|64\|0\|0.01\|0.01\|8 |
| C2 | EXACT | 0 | 8 | FULL | 1 | 14 | 1 | 14 | 0.322291 | 0.221604 | 0.000177819 | 0.000549957 | C2\|EXACT\|0\|FULL\|1e-05\|0.01\|8 |
| C2 | O_CRN | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 3.07504 | 0.922509 | 0.00168089 | 0.00227029 | C2\|O_CRN\|64\|FULL\|1e-05\|0.01\|8 |
| C2 | O_IND | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 52.3323 | 8.6941 | 0.0273726 | 0.0218838 | C2\|O_IND\|64\|FULL\|1e-05\|0.01\|8 |
| C2 | O_LR_MIX | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 0.369076 | 0.225551 | 0.000205632 | 0.000585471 | C2\|O_LR_MIX\|64\|FULL\|0.01\|0.01\|8 |
| C2 | O_LR_ORIGIN | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 0.362787 | 0.221665 | 0.000202845 | 0.000581084 | C2\|O_LR_ORIGIN\|64\|FULL\|0.01\|0.01\|8 |
| C3 | EXACT | 0 | 8 | FULL | 1 | 14 | 1 | 14 | 0.322291 | 0.221604 | 0.000177819 | 0.000549957 | C3\|EXACT\|0\|FULL\|1e-05\|0.01\|8 |
| C3 | O_LR_MIX | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 0.36856 | 0.224628 | 0.00020673 | 0.000586104 | C3\|O_LR_MIX\|64\|FULL\|0.01\|0.1\|8 |
| C3 | O_LR_ORIGIN | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 0.362199 | 0.221342 | 0.000202971 | 0.000584376 | C3\|O_LR_ORIGIN\|64\|FULL\|0.01\|0.1\|8 |
| C4_PCA | EXACT | 0 | 8 | FULL | 1 | 14 | 1 | 14 | 0.322291 | 0.221604 | 0.000177819 | 0.000549957 | C4_PCA\|EXACT\|0\|FULL\|1e-05\|0.01\|8 |
| C4_PCA | O_LR_MIX | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 0.369076 | 0.225551 | 0.000205632 | 0.000585471 | C4_PCA\|O_LR_MIX\|64\|FULL\|0.01\|0.01\|8 |
| C4_PCA | O_LR_ORIGIN | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 0.362787 | 0.221665 | 0.000202845 | 0.000581084 | C4_PCA\|O_LR_ORIGIN\|64\|FULL\|0.01\|0.01\|8 |
| C4_RANDOM | EXACT | 0 | 8 | FULL | 1 | 14 | 1 | 14 | 0.322291 | 0.221604 | 0.000177819 | 0.000549957 | C4_RANDOM\|EXACT\|0\|FULL\|1e-05\|0.01\|8 |
| C4_RANDOM | O_LR_MIX | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 0.369076 | 0.225551 | 0.000205632 | 0.000585471 | C4_RANDOM\|O_LR_MIX\|64\|FULL\|0.01\|0.01\|8 |
| C4_RANDOM | O_LR_ORIGIN | 64 | 8 | FULL | 1 | 14 | 1 | 14 | 0.362787 | 0.221665 | 0.000202845 | 0.000581084 | C4_RANDOM\|O_LR_ORIGIN\|64\|FULL\|0.01\|0.01\|8 |
| C5 | EXACT | 0 | 8 | 4 | 1 | 4 | 1 | 14 | 0.319065 | 0.195731 | 0.000178837 | 0.000531835 | C5\|EXACT\|0\|4\|1e-05\|0.01\|8 |
| C5 | O_LR_MIX | 64 | 8 | 4 | 1 | 4 | 1 | 14 | 0.341029 | 0.22324 | 0.000190738 | 0.000600279 | C5\|O_LR_MIX\|64\|4\|1e-05\|0.01\|8 |
| C5 | O_LR_ORIGIN | 16 | 8 | 4 | 1 | 4 | 1 | 14 | 0.484067 | 0.282794 | 0.000258901 | 0.000748786 | C5\|O_LR_ORIGIN\|16\|4\|1e-05\|0.01\|8 |
| C5 | O_LR_ORIGIN | 256 | 8 | 4 | 1 | 4 | 1 | 14 | 0.315287 | 0.206402 | 0.000179863 | 0.00055799 | C5\|O_LR_ORIGIN\|256\|4\|1e-05\|0.01\|8 |
| C5 | O_LR_ORIGIN | 64 | 8 | 4 | 1 | 4 | 1 | 14 | 0.330383 | 0.225873 | 0.000183185 | 0.000593915 | C5\|O_LR_ORIGIN\|64\|4\|1e-05\|0.01\|8 |
| C6 | EXACT | 0 | 8 | 4 | 1 | 4 | 1 | 14 | 0.319217 | 0.21052 | 0.00017971 | 0.000544874 | C6\|EXACT\|0\|4\|1e-05\|0.01\|8 |
| C6 | O_LR_MIX | 16 | 8 | 4 | 1 | 4 | 1 | 14 | 0.540975 | 0.273986 | 0.000271483 | 0.000727348 | C6\|O_LR_MIX\|16\|4\|1e-05\|0.01\|8 |
| C6 | O_LR_MIX | 256 | 8 | 4 | 1 | 4 | 1 | 14 | 0.324487 | 0.217113 | 0.000179195 | 0.000553302 | C6\|O_LR_MIX\|256\|4\|1e-05\|0.01\|8 |
| C6 | O_LR_MIX | 64 | 8 | 4 | 1 | 4 | 1 | 14 | 0.339918 | 0.224204 | 0.000191836 | 0.000591669 | C6\|O_LR_MIX\|64\|4\|1e-05\|0.01\|8 |
| C6 | O_LR_ORIGIN | 64 | 8 | 4 | 1 | 4 | 1 | 14 | 0.332949 | 0.220786 | 0.000185286 | 0.000582681 | C6\|O_LR_ORIGIN\|64\|4\|1e-05\|0.01\|8 |
| D_DIRECT | O_CRN | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 4.28206 | 1.17149 | 0.00244739 | 0.00306517 | D_DIRECT\|O_CRN\|64\|0\|0.0\|0.0\|0 |
| D_DIRECT | O_IND | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 89.3853 | 17.2177 | 0.0470296 | 0.0449917 | D_DIRECT\|O_IND\|16\|0\|0.0\|0.0\|0 |
| D_DIRECT | O_IND | 256 | 0 | 0 | 0 | 0 | 0 | 0 | 23.1192 | 4.34804 | 0.0125301 | 0.0113882 | D_DIRECT\|O_IND\|256\|0\|0.0\|0.0\|0 |
| D_DIRECT | O_IND | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 44.4441 | 8.53726 | 0.024 | 0.0224214 | D_DIRECT\|O_IND\|64\|0\|0.0\|0.0\|0 |
| D_DIRECT | O_LR_MIX | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0.597885 | 0.221269 | 0.000340368 | 0.000587342 | D_DIRECT\|O_LR_MIX\|16\|0\|0.0\|0.0\|0 |
| D_DIRECT | O_LR_MIX | 256 | 0 | 0 | 0 | 0 | 0 | 0 | 0.145644 | 0.0543137 | 8.20687e-05 | 0.000148554 | D_DIRECT\|O_LR_MIX\|256\|0\|0.0\|0.0\|0 |
| D_DIRECT | O_LR_MIX | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0.301206 | 0.10786 | 0.000171877 | 0.000289293 | D_DIRECT\|O_LR_MIX\|64\|0\|0.0\|0.0\|0 |
| D_DIRECT | O_LR_ORIGIN | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0.576915 | 0.210503 | 0.000324941 | 0.000553346 | D_DIRECT\|O_LR_ORIGIN\|16\|0\|0.0\|0.0\|0 |
| D_DIRECT | O_LR_ORIGIN | 256 | 0 | 0 | 0 | 0 | 0 | 0 | 0.140492 | 0.0540817 | 7.85855e-05 | 0.000146833 | D_DIRECT\|O_LR_ORIGIN\|256\|0\|0.0\|0.0\|0 |
| D_DIRECT | O_LR_ORIGIN | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0.294137 | 0.107368 | 0.000164184 | 0.000286773 | D_DIRECT\|O_LR_ORIGIN\|64\|0\|0.0\|0.0\|0 |

来源：[CONTRAST_SELECTION_POOLED.csv](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N3/CONTRAST_SELECTION_POOLED.csv)；SHA-256 `faf2ea795792c6406569bbb4f1e60686410686eaa2e35a1260afb83f2d469686`；逐seed：[CONTRAST_PREDICTION_BY_SEED.csv](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N2/CONTRAST_PREDICTION_BY_SEED.csv)；SHA-256 `2d301a5956484db876483895a1d9e2c97b4a4865a1a32848a1d8f9d193cb32a8`

## 收支平衡与权限

仅展示冻结候选及best_nonzero诊断；完整成本前沿保留全部配置。

| 配置 | 权限 | 评分/生成 | 直接基线 | 校准成本 | 4查询直接成本 | 平衡查询数 | 受测域 | 在域内 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C5\|O_LR_ORIGIN\|256\|4\|1e-05\|0.01\|8 | SAMPLE_AND_LOGP | 0.05 | D_DIRECT | 6.07443e+06 | 2.93753e+06 | 9 | 4 | False |
| C5\|O_LR_ORIGIN\|256\|4\|1e-05\|0.01\|8 | SAMPLE_AND_LOGP | 0.05 | D_KNOWN_EVENT_LOGP | 6.07443e+06 | 5670 | 4286 | 4 | False |
| C5\|O_LR_ORIGIN\|256\|4\|1e-05\|0.01\|8 | SAMPLE_AND_LOGP | 0.25 | D_DIRECT | 6.24999e+06 | 3.02581e+06 | 9 | 4 | False |
| C5\|O_LR_ORIGIN\|256\|4\|1e-05\|0.01\|8 | SAMPLE_AND_LOGP | 0.25 | D_KNOWN_EVENT_LOGP | 6.24999e+06 | 25758 | 971 | 4 | False |
| C5\|O_LR_ORIGIN\|256\|4\|1e-05\|0.01\|8 | SAMPLE_AND_LOGP | 1 | D_DIRECT | 6.90835e+06 | 3.35685e+06 | 9 | 4 | False |
| C5\|O_LR_ORIGIN\|256\|4\|1e-05\|0.01\|8 | SAMPLE_AND_LOGP | 1 | D_KNOWN_EVENT_LOGP | 6.90835e+06 | 101088 | 274 | 4 | False |

来源：[COST_FRONTIER.csv](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N2/COST_FRONTIER.csv)；SHA-256 `460c797e69f47c7c1cfbf0df0de312f2b660ed057820f60e3c2aca6c75366918`；[COST_LEDGER.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N2/COST_LEDGER.json)；SHA-256 `d23acd642086674a770abcc9a70b6b8aad84953893e34d7e8d21d16799cf9627`

上述成本前沿按操作计数、给定评分/生成价格比及零增量查询成本下界比较；它们是有条件的收支情景，不能直接称为实际CPU或费用节约。实际同条件CPU节约证据通过核验：False。全阶段成本审计 `BLOCKED`。

来源：[TOTAL_COST_AUDIT.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/implementation/TOTAL_COST_AUDIT.json)；SHA-256 `9546e9d5f84d8e7d6a4d706f711c682b62c6adab2b6bab3765c5f468ab6b7c89`

### 已记录CPU组件的实际比较

实测审计 `PASS`，selection单元 24，冷拟合次数 2880，比较行 17136，冷拟合CPU合计 19.1944 秒，首次query CPU合计 0.0593262 秒。以下仅列所选及best_nonzero配置；单位为秒，全部是737参数toy。缺少历史O_IND获取CPU及部分packet构建调度成本，不能据此主张完整CPU节省。

| 配置 | 单元 | 冷拟合CPU均值 | 4查询首次CPU均值 | 4查询warm CPU中位数 |
| --- | --- | --- | --- | --- |
| C5\|O_LR_ORIGIN\|256\|4\|1e-05\|0.01\|8 | 24 | 0.0112693 | 2.53107e-05 | 1.70313e-05 |

来源：[CPU_COST_SUMMARY.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/cpu_cost_audit/CPU_COST_SUMMARY.json)；SHA-256 `0ac8b97009c69dd73a4132c780b11952cf51cf216c8b0ae6715275a096336809`

## 方向、区间与独立验证

N4 `NOT_RUN`；许可为False，许可不代表实验已执行。旧结果不能充当新独立验证。方向曲线、bootstrap与分辨率表不用于事后更换测试规则。

| 独立验证项目 | 实际状态 |
| --- | --- |
| 回执/预测/评分文件hash及selected配置覆盖 | NOT_RUN |
| 独立预测门禁 | NOT_RUN |
| 独立分辨率门禁 | NOT_RUN |
| 满足独立推荐条件 | 0 |

[PAIRED_BOOTSTRAP.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N3/PAIRED_BOOTSTRAP.json)；SHA-256 `47d1efbf78b0601344f960b432be8060d0f4b7e20b910c1f64fd9da21cbbba43`；[RESOLUTION.csv](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N3/RESOLUTION.csv)；SHA-256 `3e3b7372dfe0827af4c8c7339f25c980cb257e5f2f35f87c91e444432975dc03`；[DIRECTION_CURVES.csv](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N3/DIRECTION_CURVES.csv)；SHA-256 `0ce44a59dc1a232039f205861032cf4af7bb0769d0c841267184104fb77f22e5`

## 观测、方向、系数与局部非线性的隔离

下表仅使用旧development数据的已封存预测，分别置换精确方向、精确系数与精确协方差。它们属于oracle诊断，不能作为有限观测配置的成绩或实际运行推荐。

| 观测 | 模型 | r | 方向/系数来源 | 协方差来源 | pX NRMSE | v NRMSE | 投影矩阵距离均值 | 训练seed数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O_LR_MIX | C5 | FULL | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.344181 | 0.202911 | 2.96782e-08 | 6 |
| O_LR_MIX | C5 | FULL | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.344173 | 0.202957 | 3.33197e-08 | 6 |
| O_LR_MIX | C5 | FULL | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.294173 | 0.19372 | 3.52711e-08 | 6 |
| O_LR_MIX | C5 | FULL | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.29417 | 0.193726 | 3.56047e-08 | 6 |
| O_LR_MIX | C5 | FULL | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.344181 | 0.202911 | 3.52711e-08 | 6 |
| O_LR_MIX | C5 | FULL | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.344173 | 0.202957 | 3.56047e-08 | 6 |
| O_LR_MIX | C5 | 1 | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.404025 | 0.355452 | 1.59375e-08 | 6 |
| O_LR_MIX | C5 | 1 | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.372877 | 0.360186 | 1.5262e-08 | 6 |
| O_LR_MIX | C5 | 1 | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.39748 | 0.353747 | 0.0782067 | 6 |
| O_LR_MIX | C5 | 1 | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.394276 | 0.350611 | 0.0780859 | 6 |
| O_LR_MIX | C5 | 1 | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.406345 | 0.355291 | 0.0782067 | 6 |
| O_LR_MIX | C5 | 1 | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.375882 | 0.359702 | 0.0780859 | 6 |
| O_LR_MIX | C5 | 2 | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.313978 | 0.195933 | 1.57995e-08 | 6 |
| O_LR_MIX | C5 | 2 | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.306549 | 0.230869 | 1.93362e-08 | 6 |
| O_LR_MIX | C5 | 2 | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.306311 | 0.197481 | 0.556368 | 6 |
| O_LR_MIX | C5 | 2 | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.306001 | 0.19722 | 0.556295 | 6 |
| O_LR_MIX | C5 | 2 | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.326495 | 0.202566 | 0.556368 | 6 |
| O_LR_MIX | C5 | 2 | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.315541 | 0.230683 | 0.556295 | 6 |
| O_LR_MIX | C6 | FULL | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.344181 | 0.202911 | 3.48443e-08 | 6 |
| O_LR_MIX | C6 | FULL | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.344173 | 0.202957 | 3.80338e-08 | 6 |
| O_LR_MIX | C6 | FULL | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.294173 | 0.19372 | 3.54505e-08 | 6 |
| O_LR_MIX | C6 | FULL | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.29417 | 0.193726 | 3.67213e-08 | 6 |
| O_LR_MIX | C6 | FULL | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.344181 | 0.202911 | 3.54505e-08 | 6 |
| O_LR_MIX | C6 | FULL | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.344173 | 0.202957 | 3.67213e-08 | 6 |
| O_LR_MIX | C6 | 1 | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.443349 | 0.330958 | 1.82004e-08 | 6 |
| O_LR_MIX | C6 | 1 | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.398135 | 0.340743 | 1.56927e-08 | 6 |
| O_LR_MIX | C6 | 1 | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.4368 | 0.32796 | 0.0809848 | 6 |
| O_LR_MIX | C6 | 1 | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.432536 | 0.325367 | 0.0808618 | 6 |
| O_LR_MIX | C6 | 1 | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.444458 | 0.329601 | 0.0809848 | 6 |
| O_LR_MIX | C6 | 1 | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.399885 | 0.339325 | 0.0808618 | 6 |
| O_LR_MIX | C6 | 2 | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.313561 | 0.196426 | 2.27062e-08 | 6 |
| O_LR_MIX | C6 | 2 | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.306181 | 0.231788 | 2.07575e-08 | 6 |
| O_LR_MIX | C6 | 2 | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.299822 | 0.19281 | 0.494763 | 6 |
| O_LR_MIX | C6 | 2 | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.299449 | 0.192704 | 0.494724 | 6 |
| O_LR_MIX | C6 | 2 | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.320904 | 0.198018 | 0.494763 | 6 |
| O_LR_MIX | C6 | 2 | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.310503 | 0.22934 | 0.494724 | 6 |
| O_LR_ORIGIN | C5 | FULL | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.337008 | 0.202853 | 4.28254e-08 | 6 |
| O_LR_ORIGIN | C5 | FULL | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.337015 | 0.202899 | 3.35188e-08 | 6 |
| O_LR_ORIGIN | C5 | FULL | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.294142 | 0.193724 | 3.57524e-08 | 6 |
| O_LR_ORIGIN | C5 | FULL | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.294139 | 0.193728 | 3.34128e-08 | 6 |
| O_LR_ORIGIN | C5 | FULL | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.337008 | 0.202853 | 3.57524e-08 | 6 |
| O_LR_ORIGIN | C5 | FULL | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.337015 | 0.202899 | 3.34128e-08 | 6 |
| O_LR_ORIGIN | C5 | 1 | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.44857 | 0.342691 | 1.82171e-08 | 6 |
| O_LR_ORIGIN | C5 | 1 | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.452492 | 0.358269 | 1.62762e-08 | 6 |
| O_LR_ORIGIN | C5 | 1 | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.439136 | 0.339912 | 0.0580241 | 6 |
| O_LR_ORIGIN | C5 | 1 | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.432174 | 0.339146 | 0.0579637 | 6 |
| O_LR_ORIGIN | C5 | 1 | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.450344 | 0.341509 | 0.0580241 | 6 |
| O_LR_ORIGIN | C5 | 1 | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.455822 | 0.356878 | 0.0579637 | 6 |
| O_LR_ORIGIN | C5 | 2 | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.327217 | 0.1946 | 2.30309e-08 | 6 |
| O_LR_ORIGIN | C5 | 2 | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.307242 | 0.227733 | 1.75158e-08 | 6 |
| O_LR_ORIGIN | C5 | 2 | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.302122 | 0.192554 | 0.521219 | 6 |
| O_LR_ORIGIN | C5 | 2 | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.299119 | 0.192416 | 0.521158 | 6 |
| O_LR_ORIGIN | C5 | 2 | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.323061 | 0.197091 | 0.521219 | 6 |
| O_LR_ORIGIN | C5 | 2 | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.310236 | 0.226261 | 0.521158 | 6 |
| O_LR_ORIGIN | C6 | FULL | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.337008 | 0.202853 | 3.88177e-08 | 6 |
| O_LR_ORIGIN | C6 | FULL | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.337015 | 0.202899 | 3.17952e-08 | 6 |
| O_LR_ORIGIN | C6 | FULL | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.294142 | 0.193724 | 3.77706e-08 | 6 |
| O_LR_ORIGIN | C6 | FULL | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.294139 | 0.193728 | 3.6762e-08 | 6 |
| O_LR_ORIGIN | C6 | FULL | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.337008 | 0.202853 | 3.77706e-08 | 6 |
| O_LR_ORIGIN | C6 | FULL | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.337015 | 0.202899 | 3.6762e-08 | 6 |
| O_LR_ORIGIN | C6 | 1 | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.422338 | 0.319912 | 1.37583e-08 | 6 |
| O_LR_ORIGIN | C6 | 1 | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.413954 | 0.336197 | 1.28289e-08 | 6 |
| O_LR_ORIGIN | C6 | 1 | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.409602 | 0.317766 | 0.0599205 | 6 |
| O_LR_ORIGIN | C6 | 1 | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.409102 | 0.316807 | 0.0598733 | 6 |
| O_LR_ORIGIN | C6 | 1 | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.422615 | 0.319453 | 0.0599205 | 6 |
| O_LR_ORIGIN | C6 | 1 | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.415752 | 0.33549 | 0.0598733 | 6 |
| O_LR_ORIGIN | C6 | 2 | EXACT_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.326807 | 0.194523 | 1.93982e-08 | 6 |
| O_LR_ORIGIN | C6 | 2 | EXACT_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.3069 | 0.228399 | 1.79264e-08 | 6 |
| O_LR_ORIGIN | C6 | 2 | FINITE_U_EXACT_COEFFICIENTS | EXACT_ORACLE | 0.305984 | 0.189662 | 0.455314 | 6 |
| O_LR_ORIGIN | C6 | 2 | FINITE_U_EXACT_COEFFICIENTS | FINITE_PAID | 0.302879 | 0.189895 | 0.455281 | 6 |
| O_LR_ORIGIN | C6 | 2 | FINITE_U_FINITE_COEFFICIENTS | EXACT_ORACLE | 0.327414 | 0.194387 | 0.455314 | 6 |
| O_LR_ORIGIN | C6 | 2 | FINITE_U_FINITE_COEFFICIENTS | FINITE_PAID | 0.309374 | 0.225143 | 0.455281 | 6 |

派生表：[DIAGNOSTIC_ISOLATION_SUMMARY.csv](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/docs/modeling_contrast/results/DIAGNOSTIC_ISOLATION_SUMMARY.csv)；SHA-256 `88c1726cfe84138da7c16a1742786315b177ea9a06296e873d4b995187a08dda`。每个输入diagnostics.json及raw预测的路径/hash见REPORT_SOURCE_BINDING.json。

完整Jacobian在起点和baseline终点的一阶预测只评估局部非线性，均为oracle。

| 位置 | pX NRMSE | v NRMSE | 训练seed数 |
| --- | --- | --- | --- |
| O_J_ORIGIN | 0.049284 | 0.0233269 | 6 |
| O_J_BASELINE | 0.0276286 | 0.0202903 | 6 |

来源：[ORACLE_JACOBIAN_POOLED.csv](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/oracle_diagnostics/ORACLE_JACOBIAN_POOLED.csv)；SHA-256 `0f8257eb06da1c38cf6aa6d441a1f4b5bd5e6f5c5ff6bd362f87e9a25cac3f50`

## 失效范围与不能推出的结论

- 当前证据固定数据seed20260915、初始化7001、737参数、16动作、两个奖励臂；
  只覆盖旧训练RNG与固定bank查询范围。
- SAMPLE_ONLY中CRN额外要求可控制采样随机数；普通黑盒API不自动具备该权限。
  共同随机数可能提高方差，必须看实测行。
- LR依赖完整归一化动作概率、正确proposal和支持覆盖；
  未观察到X、经验零方差及大权重不能证明事件无变化。
- 无fit差异激励、未覆盖更新方向、局部非线性及低信号保留失效状态；active-only结果不替代all-bank结果。
- n16/256及fit-bank消融只改冻结的一项；所有非零消融保留实际rank/norm/hash；r0仅为C0基线。
- N3选模仅有4个旧selection训练seed，方向门槛要求至少5个seed/20个anchor-bank单元；
  不足则为UNKNOWN/INSUFFICIENT。
- 新校准规划仅6个seed，不支持95%分布无关seed级预测覆盖；
  noise replicas、题目、锚点均不能补充训练独立性。
- 点估计、点态正态近似区间、群体误差分位、同时区间、预测区间和工程门禁分别命名；
  没有安全认证或在线SSVC结论。
