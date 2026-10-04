# D：保存选择器的状态信息与强收缩诊断

本节仅读取 16 个保存模型、16 个保存 fold、compact_input 的前置 packet 与既有 H32 汇总、原预测/调参/敏感性表。没有重新拟合、求解 beta、新增 alpha、生成样本、模型调用或 T 访问。代码使用实验 commit `7af92dff42e9c0432716550f6611dc244af23ecb` 的语义：对本次实际导入的 features/kernels/semantics/selectors 四文件逐一与该 commit 字节比对一致。输入、命令及 SHA 只记于 `inputs_D.json`；不做全树哈希。

## 1. 实际用了多少状态修正？

`saved_prediction_decomposition.csv` 有 8 原点 × 4 层 × 8 动作 = **256 行**。`prediction_delta_R0 = mean_delta_R0 + state_correction`，逐行与保存预测的最大绝对误差 **0**；由保存 kernel 参数重构的 16 个 K 与 fold.gram 的最大误差也是 **0**。`actual_delta_R0` 与 `actual_delta_BestStatic` 均来自既有开发端点。BestStatic 是各外折训练数据选出的固定动作，不是全数据后见最佳或逐原点 oracle。

|层|保存 alpha（61001/2/3/4）|最大绝对状态修正（百分点）|相对 BestStatic 最大绝对状态修正（百分点）|均值排序与最终排序不同的动作行 / 64|
|---|---|---:|---:|---:|
|Z0|10/10/10/10|0.047423|0.054729|8|
|Z1|1/10/10/10|0.176072|0.248862|10|
|Z2|10/10/10/10|0.020994|0.020994|4|
|Z3|10/10/10/10|0.020925|0.020925|4|

上述行组为 `level=各层`，最大值分别取 `abs(state_correction)`、`abs(state_contrast_BestStatic)` ×100；排序变化计 `mean_rank != prediction_rank`。状态修正并非严格零，但保存模型的 32 个原点×层选择，其 **predicted_best 本身已等于 BestStatic，tau_override=0/32**。因此取消 tau=0.005 本身不会改变这 32 个原选择；这是复核一致。不能把“小修正”写成“前置数据没有状态信息”。

## 2. 原核谱、中心化几何与尺度

`kernel_geometry.csv` 的 `component=TOTAL,stratum=ALL` 为 16 个原训练核；每折 n=6，Z2/Z3 的 ridge=n×alpha=60。所有 eigenvalues 是原 K 的谱；centered_eigenvalues 来自 H K H，仅为几何描述，未训练任何中心化模型。edf 按 Σ λ/(λ+nα) 计算，不加独立拟合截距的 1。

|层|原 K 的 edf 范围|最大模式保留比例范围|中心化几何 edf 范围|中心化最大模式保留比例范围|
|---|---:|---:|---:|---:|
|Z0|0.137468–0.138021|0.102138–0.104201|0.122945–0.126936|0.101827–0.103820|
|Z1|0.138527–0.911234|0.089950–0.507376|0.128826–0.850408|0.089716–0.506365|
|Z2|0.140241–0.140491|0.088519–0.089788|0.050682–0.052141|0.025165–0.025575|
|Z3|0.140204–0.140467|0.088698–0.090029|0.050399–0.051929|0.024923–0.025321|

Z1 较大的端点是 61001 折 alpha=1，其余三折为 10。Z2/Z3 的原 K 绝大多数模态被抑制；中心化后的状态变化模态最大只保留约 **2.49%–2.56%**（几何描述）。这支持“当前学习器设置强收缩”，但不能据此声称弱 alpha 会稳定改善选择。

相同 alpha 并不表示完全相同的几何收缩。每个组件先除以训练均值对角线尺度，再乘冻结权重；各层 TOTAL 的 mean diagonal 都是 1.5，避免直接把原始特征维度当正则尺度。然而谱集中度不同：Z2 current 的平均中心化 trace 约 0.43，而 Z0/Z1 标准化 current 的中心化 trace=6。保存 raw scale：Z0 current=36；Z1 current=243/252；Z2/Z3 current=1、Z3 repair=1；Z2 reward history=0.04356–0.04547、Z3 repair history=0.04201–0.04407；nuisance=8。`component!=TOTAL,stratum=ALL` 给出每组件实际权重、raw scale、谱与中心化 trace。组件 edf 是以同一 ridge 单独描述该组件的几何，**不可加起来当总 edf，也不是新拟合结果**。

Z3 的 reward current 权重由 Z2 的 1 降为 0.5，reward history 从 0.25 降为 0.125，同时新增 repair=0.5、repair_history=0.125，nuisance=0.25 不变。**Z3−Z2 不是纯“增加修复信息”对照，而混入已有 reward 信息减半。** 原核参数可直接验证，不能把两层零收益差当作修复信息无用的干净识别。

## 3. 相似度由全对质量主导，但状态差异集中在少数固定层

以下统计采用四个外折各自 6 个训练原点的 15 个无序对，按原冻结权重聚合，共 60 个“折内原点对”；外折重叠，**不是 60 个独立观测**。`kernel_geometry.csv` 中 `stratum!=ALL` 的 offdiagonal_sum 使用对称的 30 个有序非对角项；比例不受两倍重复影响。`kernel_pair_geometry_D.csv` 提供逐对、逐层、逐组件明细。

Z2/Z3 reward current 的 X–X 部分占正的非对角相似度 **81.800%**；Z3 repair current 的可归于 X–X 部分占 **80.331%**。reward 是 X 原子精确分解；repair 联合项精确分解，边缘项将 embedding 按该 bin 中来自 X 的质量拆分，分子只取 X–X 项，不含 X–非 X 交叉项。此比例是已观察 X 质量的相似度贡献，**不是“H32 全 X 题的占比”，也不是条件修复信息量**。历史差分与中心化元信息可有负相似度，不把其有符号和称为概率占比。

|固定层|reward current 的 X–X 相似度占比|repair current 的 X–X 相似度占比|
|---|---:|---:|
|cross_series × IMAGE_CUE_FRESH|99.145%|97.540%|
|cross_series × SYMBOLIC_FRESH|30.997%|27.865%|
|duplicate_encoding × IMAGE_CUE_FRESH|100.000%|99.886%|
|duplicate_encoding × SYMBOLIC_FRESH|98.562%|96.621%|
|trend × IMAGE_CUE_FRESH|98.474%|96.905%|
|trend × SYMBOLIC_FRESH|48.089%|47.460%|

行过滤：Z2/current 或 Z3/repair、`stratum=表中家族|接口`，比值为该组 `sum(X_X_offdiagonal_sum)/sum(offdiagonal_sum)`。在 `pair_distance_sum` 的全固定层总和中，**trend 与 cross_series 两个 SYMBOLIC_FRESH 层合计占 reward current 距离的 80.591%、repair current 距离的 83.905%**。所以多数相似度来自全对质量，区分状态的差异则集中在这两个符号层；单凭总体 kernel 或总体 pX 会掩盖这个结构。这是前置 P 的描述，未使用 H32 标签筛选或修改特征。

## 4. 调参为何偏强？原有弱 alpha 有没有改排序？

`existing_tuning_diagnostics.csv`：

- `row_type=outer_tuning` 共 16 行，6 个候选 alpha 的效用完全平局为 **7/16**；按层为 Z0 2/4、Z1 1/4、Z2 2/4、Z3 2/4。共有 **15/16** 外折的“距峰值 <0.001”合格集合包含多个 alpha，强 alpha 优先规则作用于这些集合；16/16 保存 alpha 等于合格集合最大 alpha。Z1/61002 仅一个合格 alpha，因此不能把它也称为平局驱动。
- `row_type=inner_validation` 共 48 行，对应 4 层×4 外折×3 内验证 lineage；6 个 alpha 效用全平局 **38/48**，分层 10/12、8/12、10/12、10/12。`exact_utility_tie_pairs` 另记录每组六 alpha 的 15 对中平局多少，不把对数当独立样本。
- `row_type=saved_alpha_sensitivity` 共 192 行，全部读取原 fold.alpha_grid，不求新 beta。对 Z2 或 Z3，原 alpha=0.0001 时：8/8 原点的完整动作排序与保存 alpha 排序不同，5/8 的 predicted_best 变化，只有 **1/8** 通过 tau 改变最终动作；8 原点平均相对 BestStatic 的已观测 J 差为 **−0.005425 个百分点**。alpha=0.001 同样；0.01、0.1 各仍只变 1/8 动作且差相同。alpha=1 有 1/8 top 变化、0/8 最终动作变化。

其他层的既有最弱 alpha：Z0 改 3/8 最终动作，均值差 −0.021701 pp；Z1 改 1/8，均值差 +0.010851 pp。它们是既有事后敏感性结果，不是新的验证胜出，也不是建议挑出“最佳 alpha”。

## 5. 能排除、被支持、仍不可识别

- 能排除：“保存预测没有正确重放”“原选择全是 tau 把不同 top 强制拉回 BestStatic”“状态项严格为零”。前两项可直接检验，后一项被非零 state correction 否定。
- 被支持：Z2/Z3 在当前尺度及 alpha 下强收缩；选择效用的许多平局/近平局配合偏强规则；全对质量主导总体相似度，而修复差异集中于两个 SYMBOLIC_FRESH 层；Z3 对照有 reward 权重下降混杂。
- 仍不可识别：强收缩、有限独立状态及目标噪声各自造成多少零增益；去除全对主质量是否提高收益；保持 reward 权重后修复信息的增量作用。没有重拟合，因此未测这些反事实。已有弱 alpha 改了排序却没有稳定改善原决策，故不能把“只减小 alpha”当已证实修复。

下一次讨论对 D 的首要问题应是：在保留 pX 目标与现有候选的前提下，**估计/学习器的选择效用平局与 kernel 几何尺度如何共同压低状态修正，以及表示比较如何避免 Z3 权重混杂**。需结合 B/C/E 的证据决定总体优先层；本节不执行新方案。
