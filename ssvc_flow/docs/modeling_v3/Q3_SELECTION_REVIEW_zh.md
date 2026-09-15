# Q3 冻结选择审查与执行草案

状态：**DRAFT_NOT_A_SELECTION_LOCK**。配套 [Q3_SELECTION_DRAFT.json](Q3_SELECTION_DRAFT.json) 是 `freeze --selection` 所需的 selected 对象，已用实际设计生成器和冻结比较校验器验证；没有生成锁、提交作业或读取新 calibration/test 结果。执行前仍须完成当前源码版本的完整 development 验证和冻结审查。

## 1. 具体草案与依据

| 字段 | 草案 | 理由及边界 |
|---|---|---|
| observation_methods | PRESERVE_XI、PILOT_SHRINK_ZERO_SUM | 保留简单直接基线及独立 pilot 修正；最终观测选择须同时核对 Q1 结果 |
| selection_rules | STRATIFIED_RANDOM、BLOCK_PIVOT_QR | 保留随机对照与实际几何选择，使用同 m、n、α 比较 |
| models | ZERO、FULL_RIDGE、FULL_GLS、PCA、RANDOM_Q、RESPONSE_SVD、GROUP_WEIGHTED_RESPONSE、RBF_RIDGE、DIRECT_MEASURE、KNOWN_EVENT_SCORE | 全部原基线保留；DIRECT/KNOWN 由共用测量/评估单独产生，不增加拟合 design |
| rank_caps | `[2, "FULL"]` | 保留一个实测有意义的低维点与完整维数；不重扫 1/2/4/8/16 |
| alpha | `1e-5` | Q2 的 0/1e−8/1e−5 几乎同平台；沿已有维数实验设置，非唯一最优声明 |
| rho_threshold | `0.05` | 沿 Q2 已执行的 development 操作阈值；不是从 calibration/test 学习，也不是边际 q95 的反推值 |
| leverage_threshold | `10.0` | 同上，作为预声明拒判条件；不是经验置信半径 |
| primary_comparison_family | 四个 RQ 的完整不变模板 | 前三个在 CPU 计算；RQ4 预声明真实 VLM 关系，等待 development 绑定合法设计和独立测试 |

配置另行固定 m=8、bank-design seed=2026091500、n=[64,256,1024,4096]；这些不从 selected 中的同名扩展键读取。PILOT_SHRINK 的实际 shrink 默认为 0.1（`observation.selected_shrink` 未配置时），不能只向 selected 添加一个 shrink 值并假定算法会改变。

Q2 的证据见 [Q2_SELECTION_EVIDENCE_zh.md](Q2_SELECTION_EVIDENCE_zh.md)：在 QR/m8 下，r=2 的 12/12 个 development origin 均满足 r<k；随机选择下为 8/12，其余属于 r=k。RESPONSE_SVD r2 仍比 FULL 有误差增加；保留它用于衡量压缩损失，不能称为低维已获胜。Q2 原件全部保留，删减后续 rank 网格不删除原结果。

**源码版本边界：** candidate05 的 Q1/Q2 数字仍是真实历史 development 证据。本轮新增冻结主比较实现改变 CPU 依赖哈希，现有严格门禁不能直接把旧源码原件当作新锁的匹配输入。必须用最终源码快照重新完成匹配 development，或另行制定明确版本变更协议；本草案不放宽当前哈希校验。

## 2. ρ/leverage 与 calibration 的准确分工

原方案第 203 行要求 rho 阈值只能在 development 选择；第 209 行要求新 CPU 数据前冻结全部阈值。`schema.freeze_selection` 要求两个几何阈值均为显式有限非负数；实际 `classify_coverage` 还要求 rho 在 [0,1]。本草案 0.05/10 均满足。来源：[计划](design/CODEX_EXPERIMENT_PLAN_zh.md)、[schema.py](../../src/modeling_v3/schema.py)、[coverage.py](../../src/modeling_v3/coverage.py)。

`cpu_results.analyze_calibration` 的实际操作是：

1. 对每个冻结 design、每个 calibration training seed，取**两臂 × 三 anchor × 20 观测重复 × 12 heldout bank × 三目标 × 72 prompt × 四事件**的最大绝对误差。
2. 此最大值包含被几何规则拒判的查询；不会先按 rho/leverage/accepted mask 过滤。冻结几何阈值影响后续路由，不影响这一全体 seed-max 的定义。
3. 对 20 个独立 seed 最大值用有限样本顺序统计量 `ceil((20+1)*0.95)=20`，即取最大值作为 radius。任一必需预测非有限，radius 保持 null/UNKNOWN。
4. 固定 tolerance 为配置的 `old_toy_q95_threshold=0.000125`，实际作用对象为 **MAX_ABSOLUTE_FOUR_CHANNEL_ERROR**，比旧 pooled q95 门槛更严格。这里的名字不能改变实际统计对象。
5. locked_test 的非零查询只有在 radius≤tolerance 且 target 有激励、rho≤0.05、leverage≤10 等条件共同满足时才可能接受。相同策略独立标记，ZERO 仍是统计基线。

因此 calibration 学习的是每个冻结 design 的**误差半径**，不是 rho、leverage、模型/rank、tolerance 或筛样规则。缩紧 rho 不能删除 calibration 大误差以“改善”全体 radius。测试风险—覆盖曲线的其他倍率属于预定义诊断，不是自动改变冻结操作点。

Q2 默认 QR/m8 的主对比接受率 35.4167% 来自 51/144 个相同策略；非零查询尚无已校准容差资格。上述阈值是保守操作预声明，不宣称已从 Q2 学到良好的非零覆盖。新 calibration 可能给出不可用半径或 UNKNOWN；这是允许结果，不改阈值追求通过。

Conformal 对象是同固定数据/初始化、交换的一个未来训练 seed 的全体最大误差；不是单个 prompt 的 95% 置信区间，也不是 192 个设计同时获得的 95% 保证。额外初始化移位不能继承该保证。

## 3. 精确工作量：缩小模型搜索，不削独立 seed 或重复次数

当前 `development_designs(selected=...)` 必然构造统一笛卡尔；它不支持“n1024 跑所有模型、其他 n 只跑 FULL”的稀疏矩阵，也不读取 selected.n_grid。不能在文档写少量运行，却让代码悄悄执行另一张矩阵。

草案的每个 observation×selector×n 有：

- 4 个固定规则：ZERO、FULL_RIDGE、FULL_GLS、RBF_RIDGE，各只生成 FULL。
- 4 个方向模型 × 2 个 cap（2/FULL）=8 个规则。
- 合计 12 × 2 observation × 2 selector × 4 n = **192 个拟合 design**。
- DIRECT 是 2 observation × 4 n =8 个独立报告基线；KNOWN_EVENT_SCORE 另有1个；两者不乘 selector/model/rank 重新测量。

| 阶段 | training seed | 轨迹（含 arm/init） | origin | 每 origin 观测重复 | measurement unit | 拟合数 |
|---|---:|---:|---:|---:|---:|---:|
| interval_calibration | 21001–21020，共20 | 40 | 120 | 20 | 2,400 | 460,800 |
| locked_test | 22001–22030，共30 | 60 | 180 | 20 | 3,600 | 691,200 |
| 主验证合计 | 50 | 100 | 300 | 20 | 6,000 | **1,152,000** |
| orthogonal_generalization（另报） | 23001–23010 × init7101/7201 | 40 | 120 | 20 | 2,400 | 460,800 |

主验证 source update=100×64=6,400；真实 scratch fork update=300×36bank×3candidate=32,400。20次是测量噪声重复，不是训练轨迹重复；不能把 1,152,000 次拟合当作独立样本数。

相比 Q2 1,108 designs，草案不再扫描五种 m、固定总预算切片、十种随机 design seed、多 α 或六种 rank。若沿 `[1,2,4,FULL]` 则每 unit 320 designs、主验证1,920,000次拟合；本草案减少40%，保留完整独立 seed/repeat 和全部基线。受当前统一矩阵 API 限制，FULL 方向等价与 ZERO 仍会在各组合重复执行；不得把这称作已经实现的稀疏调度优化。

测量缓存按 n 共用所有候选的同一 origin packet，两观测共享生成动作。主验证中四 n 总和5,440，calibration/direct 两个独立 packet ×72 prompt×6,000 measurement unit，约 **4,700,160,000 个有限动作抽样**。这是实际采样计数公式，不是新增GPU forward数或运行小时数；CPU使用已收集的有限概率表。整个24-bank pool的评分原件仍保存，不能按最终m8反向缩减事实成本。

以上计数由实际生成器计算。**尚无该192-design完整确认阶段的实测运行时/RSS/磁盘估计**，不能按Q2时间简单线性外推。先在最终锁下以一个完整 training seed、两臂、全部三anchor和20 repeats启动一个可续跑 shard，记录实际时间/RSS/产物大小；不能看其误差再改变方法。单seed shard为120 measurement units、23,040次拟合。当前 seed collision inventory 按seed检查，建议按完整seed分片，避免同seed两臂分拆后被当作已使用seed。

## 4. 四RQ、正式统计要求与已补接口

原计划四个RQ（第41/43/45/47行）是观测几何、校准覆盖、有效维数和**真实Qwen迁移**。第224行要求：训练seed聚类的5,000次配对bootstrap；四个主问题的正式检验采用预登记Holm。CPU验收清单E第70行要求所有p值/主比较/校正与冻结选择一致。原文没有唯一指定四个可执行零假设、端点、标量统计量和侧别。

此前无spec返回NOT_RUN是诚实行为，但不能作为全项目“正式统计完成”。此前 `_paired_comparisons` 只计算 model vs 同obs/n DIRECT，无法执行自然的RQ1观测规则相比较、RQ2 selector配对及RQ3真实r<k配对；用四个CPU model-vs-DIRECT冒充四RQ不合规。本轮已补以下具体冻结关系（是对原研究问题的预声明操作化，不宣称涵盖该RQ全部子问题）：

| RQ | 左端 − 右端 | 主比较范围 |
|---|---|---|
| RQ1 CPU | DIRECT/PILOT_SHRINK − DIRECT/PRESERVE，n1024 | 主joint1−joint0，全部查询；观察简单基线与pilot改进 |
| RQ2 CPU | FULL_RIDGE/QR − FULL_RIDGE/STRAT | 同PRESERVE、n1024、m8、α1e−5、design seed；全部查询 |
| RQ3 CPU | RESPONSE_SVD/cap2 − FULL_RIDGE/FULL | 同QR/PRESERVE/n1024/m8/α；same-origin实际低端0<r<k、FULL端r=k |
| RQ4 VLM | RESPONSE_SVD/cap2 − FULL_RIDGE/FULL | 固定PRESERVE/QR/n1024/m8/α1e−5/design seed2026091500；同独立reference、实际r<k；X_BASE32/96为主，X_VALID96另报 |

每个比较固定：`target_index=0`、`channels=[group_pX,group_v]`、`metric=mse`。逐seed先从原件得到每通道SSE/count，再将两通道MSE等权平均；最后比较seed均值。保留每通道效果、原分母、全部/合格origin与measurement-unit数。RAW4及其他目标仍单独报告，不把多通道/多目标重复扩成独立训练seed。

正式检验为双侧 `PAIRED_SEED_SIGN_FLIP_MEAN`，5,000次符号随机化，随机种子2026091503；其假设是零效应下**独立训练seed差值符号对称**，不是无假设的分布无关检验。至少6个独立seed；5,000次bootstrap单独报告区间。双侧检验不显著不能证明等价或非劣，本文不臆造非劣界值。

RQ3先核对同一原点的Q、query_e、selected_bank_indices、k及实际r，再读取误差。r=k单元仍保存在原件和rank_audit，但不进入低维比较；任何必需比较单元预测/参考未解析，或任一预定seed没有合格origin，整项保持UNKNOWN，无p值，不删除seed。接受mask不参与主比较筛选。

跨阶段 family 模板保持同一hash。VLM development只绑定满足固定RQ4关系的具体design_id；若没有该配对，保持UNAVAILABLE，不能换成development/test更好看的赢家。CPU完成只有RQ1–3结果时family为PENDING；只有实际RQ1–4都AVAILABLE才由最终汇总应用四项Holm。缺失或UNKNOWN不能填p=1/0或宣称正式family完成。

相关接口：

- `frozen_comparisons.validate_primary_comparison_family(config, selected)`：冻结时验证真实CPU endpoint与固定RQ4模板。
- `paired_seed_inference(spec, seed_totals, *, bootstrap_seed)`：计算等权通道、整seed推断和真实原分母。
- `cpu_results.verify_cpu_primary_comparisons(config, lock, analysis_report, *, fixture=False)`：接受完整TEST_ANALYSIS路径或path/hash绑定，从实际response原件重算指定主比较；拒绝手写p值。
- `summarize_primary_family(family, results)`：没有真实第四项时不应用完整family Holm；真实文件门禁由 `primary_family.finalize_primary_family` 承担。

## 5. 当前CLI的正确顺序

以下命令在**服务器Slurm CPU allocation**内通过既有request wrapper执行。路径变量须指最终同源码/hash的完整原件和新输出目录；这是操作顺序，不是已提交授权或已创建路径。

```bash
"$PY" -m src.modeling_v3.cli freeze --config "$CFG" \
  --inputs "$Q1_COMPLETE_ROOT" "$Q2_COMPLETE_ROOT" \
  --selection docs/modeling_v3/Q3_SELECTION_DRAFT.json --out "$Q3_LOCK_ROOT"

"$PY" -m src.modeling_v3.cli validate-cpu --config "$CFG" \
  --lock "$Q3_LOCK_ROOT" --role interval_calibration \
  --existing-run-roots "$EXISTING_RUN_ROOT" --out "$Q3_CAL_ROOT"

"$PY" -m src.modeling_v3.cli calibrate-cpu --config "$CFG" \
  --lock "$Q3_LOCK_ROOT" --inputs "$Q3_CAL_ROOT/interval_calibration/response" \
  --out "$Q3_CAL_ANALYSIS_ROOT"

"$PY" -m src.modeling_v3.cli validate-cpu --config "$CFG" \
  --lock "$Q3_LOCK_ROOT" --role locked_test \
  --calibration-receipt "$Q3_CAL_ANALYSIS_ROOT/CALIBRATION_RECEIPT.json" \
  --existing-run-roots "$EXISTING_RUN_ROOT" --out "$Q3_TEST_ROOT"

"$PY" -m src.modeling_v3.cli analyze-cpu --config "$CFG" \
  --lock "$Q3_LOCK_ROOT" --inputs "$Q3_TEST_ROOT/locked_test/response" \
  --calibration-receipt "$Q3_CAL_ANALYSIS_ROOT/CALIBRATION_RECEIPT.json" \
  --out "$Q3_TEST_ANALYSIS_ROOT"
```

`freeze --inputs`要求完整Q1/Q2原运行根及匹配source/config，不能用单独摘要CSV或Q2_ANALYSIS目录替代。分seed调度使用 `--seeds` 且所有分片拥有同一锁；最终calibrate/analyze的 `--inputs` 必须包含该role所有完整response根，省seed/重复不能通过生产完整矩阵校验。`validate-cpu`省略role时实际默认只执行interval_calibration，不会自动完成测试。校准后即使没有模型通过容差，后续如按冻结计划测量测试，其结果也不得事后换方法以追求通过。

## 6. 本次验证与未完成工作

本次只执行微型原件/冻结接口测试：14项通过（2.47秒），包括完整manifest重算p值、拒绝伪改分析p值、实际r=k高误差不混入低维子集、未解析seed保留及错bank拒绝；Ruff检查/格式通过。未提交Slurm/GPU、未运行大CPU、未创建selection lock。整体candidate06验收、development复跑、真正冻结、Q3实测运行估计与阶段执行由根任务继续。
