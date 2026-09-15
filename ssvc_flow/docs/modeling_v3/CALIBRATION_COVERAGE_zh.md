# V3 校准覆盖与维数：当前实测证据

状态：`DEVELOPMENT_ONLY`。本报告对应服务器 Q2 作业 155844 及原件复算 155896，尚无独立 Q3 确认或真实 9B 响应证据。

## 实际执行范围

固定初始化 7001；training seed 101/201 × X_BASE/X_VALID × anchor 8/24/40，共 12 个原点。每个原点有 24 个校准 bank 和 12 个独立 query bank，完整执行 1,108 个设计，共 13,296 次拟合。不同噪声、设计 seed、arm、anchor 不能增加独立 training seed 数。

Q2 核心耗时 1,572.82 秒，作业用时 27 分 33 秒，最高 RSS 6,769,008 KiB。分析从原件重新计算全部设计，用时 3 分 18 秒。申请 32 GiB 内存，调度器实际分配 11 GiB；上述是实际用量。完整逐原点指标和预测、真值、几何、计费、参数原件留在服务器，已回收汇总与哈希清单。

## 覆盖与误差

主对比为 `joint_1_minus_joint_0`，表中 MSE 使用完整四事件，q95 为绝对残差分位数，不是置信区间。固定 m=8、n=1024、PRESERVE_XI、alpha=1e-5：

| 方法 | MSE | q95 绝对误差 |
|---|---:|---:|
| STRATIFIED_RANDOM + FULL_RIDGE | 3.48417e-7 | 0.00115236 |
| BLOCK_PIVOT_QR + FULL_RIDGE | 2.82857e-7 | 0.00112376 |
| 同 n 的 DIRECT_MEASURE | 1.03191e-8 | 0.000227651 |

当前 35.42% 主对比接受率全部来自相同策略的精确零差异。非零 query 尚没有通过独立校准容差的接受结论。有限预测、较低误差和相同策略控制的“接受”均不能替代这项证据。

有限动作模型的 KNOWN_EVENT_SCORE 可以枚举完整事件，因此精确误差为零；这一结论不适用于仅知道一个正确字符串的真实语言模型。DIRECT 使用同预算完整测量，保留在每个方法和 n 的结果中。

## 维数与选择边界

QR 的 rank cap 1/2 在全部 12 个原点均是真正的 `0 < r < k`；STRAT 的 cap 1 为 10/12，cap 2/4 为 8/12。cap 16 的全部原点达到 r=k，属于完整已观测子空间回归，不能称为低秩发现。RESPONSE_SVD cap 2 没有在任何真正降维原点击败其匹配 FULL 对照。

两种观测 PRESERVE_XI/PILOT_SHRINK_ZERO_SUM、两种选择 STRATIFIED_RANDOM/BLOCK_PIVOT_QR、alpha=1e-5、cap 2/FULL 是后续冻结的候选；冻结前仍需核对 Q1 的通道代价及完整预登记比较接口。阈值 rho=0.05、leverage=10 是已有 development 的操作设置，不是从边际分位数推导出的联合风险保证。Q3 校准不能根据测试误差重选这些阈值。

## 下一层证据

Q3 保留 20 个区间校准 seed 和 30 个锁定测试 seed，每个 seed 的全部 arm、anchor、bank、prompt、测量重复共同聚类；40 条不同初始化轨迹独立报告。校准计算包含几何拒判及未解析预测，不能删掉高误差 query 来缩短区间。首次真实 GPU 执行仍待命令、成本交接及用户明确授权。

完整数字、CSV 行号和来源哈希见 `Q2_SELECTION_EVIDENCE_zh.md`、`Q2_SELECTION_EVIDENCE.json` 及 `verification/server_Q2_analysis_155896/`。本报告不认证在线 SSVC 有效。
