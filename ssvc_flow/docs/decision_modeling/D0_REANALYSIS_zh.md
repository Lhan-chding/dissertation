# D0 历史预览重分析

状态：D0_REANALYZED；科学结论仍为 NOT_CERTIFIED。无模型调用、无新训练、无预测器测试。

读取 2,304 条生成、3,072 条评分；6 题/3 场景，一个历史原点、两个被评分推理策略。严格解析、完整 token/EOS/length、评分关联复核未发现不一致。
完整评分键 446 个；重复评分行 2626 条，潜在复用比例 85.481771%。这是计算复用潜力，未实测新缓存墙钟加速。2,304 个抽样观察均保留，重复 token 不从统计样本去重。

| 接口 | 家族 | 端点 | n | X | S | W | I |
|---|---|---|---:|---:|---:|---:|---:|
| SYMBOLIC_FRESH | trend | calibration_001_joint_0 | 64 | 0 | 2 | 62 | 0 |
| SYMBOLIC_FRESH | trend | calibration_001_joint_1 | 64 | 0 | 2 | 61 | 1 |
| SYMBOLIC_FRESH | duplicate_encoding | calibration_001_joint_0 | 64 | 60 | 0 | 4 | 0 |
| SYMBOLIC_FRESH | duplicate_encoding | calibration_001_joint_1 | 64 | 63 | 0 | 1 | 0 |
| IMAGE_CUE_FRESH | duplicate_encoding | calibration_001_joint_0 | 64 | 64 | 0 | 0 | 0 |
| IMAGE_CUE_FRESH | duplicate_encoding | calibration_001_joint_1 | 64 | 64 | 0 | 0 | 0 |
| IMAGE_CUE_FRESH | trend | calibration_001_joint_0 | 64 | 10 | 0 | 54 | 0 |
| IMAGE_CUE_FRESH | trend | calibration_001_joint_1 | 64 | 6 | 0 | 58 | 0 |
| IMAGE_CUE_FRESH | cross_series | calibration_001_joint_0 | 64 | 64 | 0 | 0 | 0 |
| IMAGE_CUE_FRESH | cross_series | calibration_001_joint_1 | 64 | 64 | 0 | 0 | 0 |
| SYMBOLIC_FRESH | cross_series | calibration_001_joint_0 | 64 | 43 | 0 | 21 | 0 |
| SYMBOLIC_FRESH | cross_series | calibration_001_joint_1 | 64 | 37 | 0 | 27 | 0 |

ORIGIN/MIX 的联合均值及协方差独立复算与历史 DIAGNOSTICS 最大绝对差 3.47e-17。完整非对角协方差保存在 lr_moments.json；ORIGIN 没有由样本最大值推断总体权重上界，MIX 贡献范围为 [-2,2]。

Z0–Z4 由同一组样本生成，逐题、流、角色分开保存。Z3 保留 event×关系分子/分母及 event×single_edit；不同关系家族分母没有阈值化。state_table.parquet 为逐观察事实，不是只含均值的汇总。

| 题目序号 | 策略指纹前缀 | 已知质量 | 未知尾部 | 无 X 支持 |
|---|---|---:|---:|---|
| 6 | 5861350246fa | 0.6366316395 | 0.3633683605 | True |
| 6 | 85103c267b41 | 0.6375955915 | 0.3624044085 | True |
| 4 | 5861350246fa | 0.9591692599 | 0.0408307401 | False |
| 4 | 85103c267b41 | 0.9593154613 | 0.0406845387 | False |
| 3 | 5861350246fa | 0.9983391578 | 0.0016608422 | False |
| 3 | 85103c267b41 | 0.9986414997 | 0.0013585003 | False |
| 5 | 5861350246fa | 0.9524120509 | 0.0475879491 | False |
| 5 | 85103c267b41 | 0.9575957213 | 0.0424042787 | False |
| 1 | 5861350246fa | 0.9991409097 | 0.0008590903 | False |
| 1 | 85103c267b41 | 0.9991574121 | 0.0008425879 | False |
| 2 | 5861350246fa | 0.8910385217 | 0.1089614783 | False |
| 2 | 85103c267b41 | 0.8903301241 | 0.1096698759 | False |

质量界名称为 conditional_mass_bounds，全部标记 CONDITIONAL_ON_SCORER。未界定评分数值误差，不属于统计置信区间，不产生 CERTIFIED_SAFE。没有 X 支持不等于 X 不存在；qX 分母区间含 0 时返回未识别。未知尾部没有分配为确定事件。

历史生成记录 elapsed_seconds 合计 5018.667410 秒，生成 token 35,458 个，记录的生成 vision_forward_calls 17,584。此处是行级生成耗时求和，不是任务墙钟。原始评分表未单列评分秒数，保留未记录；逐题当次尝试耗时见 summary.json，不能据此分解评分/加载成本。

inference_aliases.json 保留推理别名与训练/Adam 身份；共享推理指纹不表示可交换未来训练状态。六题只作历史调试，不进入新 P/E 面板。未执行 D1/D2/D3/D4。

产物：state_table.parquet、scored_observations.parquet、states_Z0_Z4.json、conditional_mass_bounds.json、lr_moments.json、inference_aliases.json、score_reuse_summary.json、summary.json；COMPLETE.json 记录 CSV 输入哈希与所有输出 SHA-256。
