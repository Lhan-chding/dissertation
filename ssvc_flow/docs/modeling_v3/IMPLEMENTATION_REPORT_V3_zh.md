# V3 实现与验收映射（阶段记录）

更新：2026-09-16。本文登记当前源码、已有小型工程检查和待执行阶段；不是最终建模决定。主执行任务后续应以同一代码快照的服务器回执更新实测字段。

## 1. 当前结果与执行边界

当前工作区新增独立 `src/modeling_v3/`，保留旧 `modeling_contrast/` 的定义与历史输出。观测几何、校准选择、响应模型、原件审计、CPU 采集和统计分析已有代码入口。真实 9B 的运行时、源训练、候选分支和序列观测提供了接入代码与小型替身测试；真实 Q4–Q6 尚不能登记为完成。

本稿核读了本地留存的 CPU pilot 回执 `runs/modeling_v3_dev/server_inventory/CPU_PILOT_155686.json`。该回执明确标记 `pilot: true`、`scientific_status: PILOT_NOT_CONFIRMATION`。**155686 只提供该次小型源轨迹采集的时序与工程记录，不能替代 Q1 观测比较、Q2 非退化维数实验或 Q3 独立确认。** 服务器状态与本轮回收的原件见 `RUN_MANIFEST.json` 和 `verification/server_CPU_155797/`。

此前两次上传被自动审批拒绝；用户对已明确列出的上传目录与服务器 CPU 验收范围答复“继续”后，candidate03 增量上传与部署已成功。服务器端核验 37 个增量文件、997 个快照文件，归档 SHA256 为 `de96a3e5e95538c6cd2e80eeb7e844a794d18d91cd6eb629aeb20fe6ab4f2cfe`。Q0 作业 155775 已完成并回收核验。首次 CPU 工程验收 155776 在收集时发现同名测试模块冲突；修正启动器后，candidate04 的重跑作业 155797 完成：1,837 passed、4 failed、2 skipped、1 deselected。4 个失败均来自旧 R1 supplement 需要真实 Git 提交原件；哈希快照没有 `.git`。独立 Git 对照作业 155873 在同源代码上以真实新快照提交 `ebfd4e322025825f55dfda50dc6308be72ae23f4` 重跑该模块，12 passed；这是服务器本地快照提交，不是伪称上游历史提交。最新修改尚需下一版完整验收。未新增 GPU 提交；首次 GPU 确认单独记录在新的不可变 Q4 执行绑定内，不能复用原 pending 阶段文件。

本稿不将计划计数、代码存在、替身测试通过或 `--dry-run` 的成功解释为真实实验完成，也不生成最终 `MODELING_DECISION_V3_zh.md`。

## 2. 按阶段的实现、测试与尚缺证据

代码路径以下均相对于 `ssvc_flow/`。

| 阶段 | 真实代码/API 与 CLI | 已有工程覆盖 | 完成阶段仍需的实际证据 |
|---|---|---|---|
| Q0 原件审计 | `source_audit.audit_parent(config, parent, out)`；CLI `audit-parent --parent ... --legacy-parent ...` | 强制从原始样本重建 packet；原件哈希、路径、别名；raw/Helmert；all/active 平方和；匹配 C3/C5/C6 的 `r=k`；缺样本不能回退到摘要 | 作业 155775 已完成，完整性错误 0；实际缺失清单见 verification/server_Q0_155775。未保留的 Q/U/C、逐动作或 pilot 角色不能补写为旧原件 |
| Q1 旧固定策略接入 | `historical_observation.prepare_historical_observation_collection(parent_raw_root, out, config)`；CLI `prepare-historical` | 身份哈希选择与 arm×anchor 分层；旧参数重新评分；原文件与 Adam 引用；与 `observe_toy` 的小型集成；缺原件时不发 COMPLETE | 在旧开发原件上生成可核验的固定策略集合，并单列新增评分成本。新采集的 V3 development 不能替代这项旧策略实验 |
| Q1 观测几何 | `observation_geometry`、`covariance_pilot`；`cpu_campaign.observe_toy`；CLI `observe-toy` | RAW4 无静默投影；质量残差可逆；ORIGIN 与 IID MIX；独立 pilot；L1/fallback；共享/别名协方差；交叉拟合方差边界；边界 witness | 旧固定策略的全部冻结 n 网格及 200 次重复；主 n1024 原始贡献；实际策略边界覆盖、系数方差/fallback、分事件和群体误差、分项耗时。解析 witness 单列 |
| Q2 校准方向 | `bank_selection.select_banks`、`coverage.coverage_diagnostics/classify_coverage`；`cpu_campaign.coverage_study` | FIRST、分层随机、范数、整 bank QR/logdet；实际增量；48/24 训练 prompt 来源分区；整 bank 选择；隐藏 query 响应隔离 | 24 个候选校准 bank、12 个 query bank 的完整采集与同支出/每 bank 固定 n 曲线；设计 seed 单列；真实 query 覆盖、拒判和直接测量比较 |
| Q2 维数与模型 | `response_models.fit_response_model`、`error_decomposition`；`cpu_campaign.persist_error_decomposition` | ZERO、FULL_RIDGE、FULL_GLS、PCA、随机方向、响应 SVD、群体加权、RBF 诊断；`0<r<k` 与 `r=k` 分离；联合 GLS；显式输出约束；四项误差及交叉项 | 在真实 CPU 原点得到非退化单元、谱/条件数/rho/leverage、完整四事件与群体误差。存在低秩代码不代表已观察到低秩收益 |
| Q3 独立确认采集 | `schema.freeze_selection/verify_selection_lock`、`cpu_campaign.validate_cpu/scan_cpu_seed_inventory`；CLI `freeze`、`validate-cpu` | 冻结 Q1/Q2 development 证据和源码；V3 seed 身份扫描；精确恢复文件例外；锁定测试需真实校准回执；按真实 e 区分目标激励 | 固定初始化 100 条轨迹、300 锚点；20 个区间校准 seed、30 个锁定测试 seed；额外初始化 7101/7201 的 40 条轨迹另报。没有这些全量原件不能称 Q3 通过 |
| Q3 统计分析 | `cpu_results.analyze_calibration/analyze_test`；CLI `calibrate-cpu`、`analyze-cpu`；`statistics` | seed-max 校准、有限样本次序统计量、20-seed 验证；主/次目标分开；UNKNOWN 保留；风险/覆盖；配对训练 seed bootstrap 与 Holm 入口；输入矩阵及原件篡改拒绝 | 实际校准回执必须在测试开放前冻结；用完整独立测试原件计算区间覆盖、误差范围、拒判覆盖与预登记比较。统计 fixture 不能授权真实测试 |
| Q4 双卡真实观测桥接 | `server_preparation.prepare_gpu`；`vlm_campaign.build_q4_bridge_plan/vlm_smoke/finalize_q4_bridge`；CLI `prepare-gpu`、`vlm-smoke`、`merge-q4` | CPU 技术回执与源码绑定；原模型/依赖兼容核对；两 worker 冻结计划；相同动作跨卡评分、IDENTICAL_POLICY、独立带噪参考与实际调用计费的替身测试 | 实际两张 PRO 6000 的分配信息、驱动/显存、相同原件的跨卡 logp 误差、真实64-draw桥接、参考精度和吞吐。现有 CPU/替身结果没有 GPU 精度含义 |
| Q5 源轨迹与候选响应 | `vlm_campaign.build_training_schedule/train_source/run_source_trajectory/build_bank_plan/make_forks/run_candidate_banks`；`vlm_observation.observe_vlm` | 128 步调度构造；真实 Adam 驱动的 CPU 替身；完整状态恢复；候选真实 e；LoRA 初始化身份；授权先于加载；请求角色、worker、去重、异常/部分分片恢复 | 已实现 `prepare-source/forks/observation/geometry/fit/evaluation`、`freeze-predictions`、`prepare-reference-extension` 和 `finalize-stage`；替身观测到冻结预测与独立参考评估已有工程测试。实际新 9B 源轨迹、候选观测、选择/校准/锁定测试仍未运行，不能将 Q5 标 COMPLETE |
| Q6 离线窗口 | `tracking_offline.fixed_basis_tracking/evaluate_tracking/track_offline`；CLI `track-offline` | 每步固定基；中间更新不可省略；先冻结预测后读参考；未获点预测资格时拒绝；真实参考须有不确定性；无源训练反馈 | 来自已冻结、合格真实 Q5 轨迹的窗口与密集参考；按窗口长度报告中间最大误差、覆盖、未覆盖路径与失效。未执行实际 Q6 回放 |

Q4 的进入条件是 Q1/Q2 技术链正确，不能额外要求复杂估计器胜过简单基线。Q5 锁定预测验证需要实际方法与阈值冻结；Q6 只回放已冻结轨迹。

## 3. 核心统计与信息权限

- 主指标统一保存为概率单位：`delta_pX` 与 `delta_v = -delta_pI`；raw 的 `X+S+W` 另作诊断，差别由质量残差解释。旧 N1 的 raw-v 历史复算保留当时的 `X+S+W` 定义，不重写旧表。
- 简单约束估计使用全部总样本，独立 pilot 方法只用与系数独立的 main 样本；保留相同 main 子集比较。二折交叉拟合不自动获得独立折标准误。
- 固定策略测量、未评分候选预测和离线跟踪分别报告。估计器输入只含所选付费响应与真实增量；完整 finite-action law、exact Jacobian 和 query 真值属于评估层。
- 完整保留直接差值测量与有限动作可精确枚举的已知事件评分。真实 VLM 的单个正确字符串只是事件子集，除非有完备枚举证明。
- 校准响应为零、没有目标激励、相同策略、超出 span、高 leverage、测量未分辨和达到已验证误差界是不同状态。不能把有限预测或几何 rho 小直接标成精度合格。
- `r=k` 只检验同观测、同正则、同校准输入下的 full 模型等价；方向收益主比较限于 `0<r<k`。
- 实际模型采集、缓存表读取与假设部署工作量分开计费；训练 seed 是顶层独立单位，bank、题目和噪声重复不能充当新增独立 seed。

## 4. 已核对的小型工程执行证据

candidate03 代码快照的小型工程回归（本机，不替代服务器验收）：

```text
python -m pytest tests/modeling_v3 docs/modeling_v3/design/reference/test_math_contracts.py -q
206 passed in 6.64s
```

原始日志、JUnit 与运行前后源码哈希保存在 `verification/local_20260916_final/`，源码在该次运行期间未改变，执行类别为 `LOCAL_SMALL_ENGINEERING_TESTS`、科学状态为 `NOT_EVALUATED`。此前 200 项版本的回执独立保留在 `verification/local_20260916/`。新增回归确认非服务器 CPU 回执不能准备 Q4、pending GPU 授权不能到达模型加载。全部 V3 源码、测试与 Python 运行脚本的 Ruff 检查通过。

以下是统一小型工程回归覆盖的主要行为；真实实验结果仍须使用服务器阶段回执：

| 测试文件 | 核对的行为范围 | 不能替代的实验 |
|---|---|---|
| `test_observation_geometry.py` | 概率贡献、约束、独立 pilot、协方差、crossfit、支持/别名、质量残差 | Q1 实际误差分布与真实 VLM 概率一致性 |
| `test_coverage_models.py` | 未中心化几何、整体 bank 选择、GLS、显式 RAW4、r<k/r=k、误差分解 | Q2 真实更新上的维数结论 |
| `test_cpu_campaign.py` | 小型实际 Adam 分支、恢复、冻结、原件、身份扫描、解析边界、完整 query 顺序 | Q1–Q3 大型服务器矩阵 |
| `test_cpu_results.py` | 合成 seed 级数据的校准、测试统计和拒判完整性 | 实际 20/30 seed 的区间覆盖与效应 |
| `test_vlm_campaign.py` | CPU 替身上的源训练/分支驱动和恢复；计划、成本外推输入校验 | Qwen3.5-9B 的 CUDA 训练和实际吞吐 |
| `test_vlm_observation.py` | 替身生成/评分、EOS/截断、独立角色、参考精度规则、worker/去重/恢复 | PRO 6000 跨卡精度、语义事件支持及尾部行为 |
| `test_tracking_offline.py` | 固定基、逐步路径、拒判、参考不确定性和窗口评估 | Q6 真实轨迹回放与任何闭环收益 |
| `test_cli.py`、`test_infrastructure.py` | 调用参数、授权先行、冻结来源、原件与完成回执 | 服务器发布成功、作业执行成功或科学效果 |

统一服务器工程验收入口是 `scripts/verify_modeling_v3_cpu.py`，它记录实际命令、Python/依赖、Slurm job、源文件与测试哈希、运行前后源码一致性、`pytest.log`、`junit.xml` 和 `result.json`。`prepare_gpu` 必须验证真实日志及测试条目，而不是读取一个手写 PASS。

candidate04 完整服务器验收已取得并保留失败记录；后续新增接口尚待 candidate05 完整服务器复验，因此不能把旧回执算作最新源码通过。当前 `scripts/modeling_v3_cpu.sbatch` 的 test 分支还显式 deselect 一项旧测试：`tests/test_audit_r0_remaining.py::test_cross_split_passes_generated_dataset`；最终报告须保留该排除及其独立验证状态，不能称未限定的全仓全部测试通过。

## 5. 真实服务器 pilot 155686 的事实范围

来源是本地留存的 `runs/modeling_v3_dev/server_inventory/CPU_PILOT_155686.json`，本稿已实际读取。

| 字段 | 回执值 |
|---|---:|
| 采集状态 | `COLLECTED` |
| 科学状态 | `PILOT_NOT_CONFIRMATION` |
| seed / arm / 初始化 | `1999001 / X_BASE / 7001` |
| 源更新 | 8 |
| 候选分支更新 | 12 |
| 完整原点恢复检查 | 12 |
| 锚点 / 轨迹数 | 1 / 1 |
| 训练抽样动作 / bank 抽样动作 | 256 / 128 |
| 独立候选参数策略 / 别名 | 10 / 2 |
| 采集总 wall seconds | 6.117956385016441 |
| 单轨迹 wall seconds | 5.936060121981427 |
| 记录的输出字节数 | 2,003,460 |
| 新 GPU 调用 | 0 |

该 pilot 没有测量 Q1 的 200 次重复、Q2 的完整方向与维数矩阵，也没有测量真实 9B 生成、prefix 重算评分或训练。不能用其约 6.12 秒直接外推整个 V3 或 GPU 工期。其运行源码快照与后续修正后的源码还需分别绑定。

## 6. 原件保留、重放和重新评分入口

### Q0：历史 packet 重放

`audit-parent` 接受 N4 run root，自动定位 `N4/raw` 和 `N4_validation`；旧 N1 需要另外指定真正的 M2 parent。输出：

- `Q0_SOURCE_AUDIT.json`：实测检查计数、错误、缺失和耗时。
- `PARENT_RECORD_INDEX.jsonl`：原文件绝对路径、字节数、哈希、角色及原点身份。
- `Q0_RAW_HELMERT.csv`、`Q0_ALL_ACTIVE_SQUARED_SUMS.csv`、`Q0_R_EQUALS_K.csv`：仅由实际存在的原始样本/预测生成。
- `MISSING_INPUTS_V3.json` 与 `Q0_REPRODUCTION_zh.md`：保留缺失、不完整和不能认领的范围。

此入口不训练、不新采样、不运行模型评分。找不到原始 candidate logp 时，存有 raw/Helmert 汇总数组也不能代替 primitive replay。

### Q1：旧参数上的新增评分

`prepare-historical` 读取真实旧 `manifest.json`、`resolved_config.json`、dataset/probe/layout、完整 observations NPZ 与 RNG 原件。身份哈希按 arm×anchor 分层，默认选 12 个旧原点及每原点一个 bank；该数量是登记的固定策略子集，不是把旧小预算限制带回 V3。选择不读取误差、奖励或响应大小。

输出 `HISTORICAL_SELECTION.json`、`ORIGINAL_SOURCE_BINDINGS.json`、`SCORING_COST.json` 及 `observe_toy` 可读取的不可覆盖 collection。每个 view 保存旧 source/branch 参数、旧 trajectory/anchor/bank 身份和 Adam 数组引用。`trajectory_count` 明确表示 Q1 原点 view 数，另报唯一旧轨迹数；新增源轨迹数始终为零。

原 737 参数 evaluator 在选定旧参数上重新计算归一化 16 动作概率，属于 **新增评分**，不是原始 packet 重放。成本报告新增 forward、每题 forward、动作评分数量、精确参数别名缓存命中与时间。缺少任一所需原件时发布 `MISSING_INPUTS`，不发布供后续观测消费的 COMPLETE。

### 新 CPU/VLM 原件与恢复

CPU collection 保存逐步参数、真实 d/g/Adam 状态、训练与 bank 动作/类别/logp、原始 RNG 和可恢复状态树/张量。测量保存原始贡献、系数、角色身份、主 n1024 完整评分及模型预测。query 参考开放前先保存预测锁；完成回执绑定嵌套回执和精确文件集合。

真实 VLM 路径保存原始 token/text、EOS/截断、事件标签、逐 token 与序列 logp、采样/评分身份、调用账本和未完成分片。`REFERENCE_UNRESOLVED` 是允许的最终测量状态，不能改写为精确参考。是否完整保留真实 Q5 源状态与候选响应，仍须服务器实际执行后按 manifest 核查。

## 7. 尚未满足的交付项

| 交付/阶段证据 | 本稿状态 |
|---|---|
| 最新源码在服务器发布并完成实际工程验收 | candidate04 完整验收 1,837 passed / 4 failed；独立 Git 对照模块 12 passed，待新版完整重跑 |
| 服务器完整旧原件 Q0 审计与旧策略 Q1 比较 | Q0 完整性错误 0，历史缺失保留 `MISSING_INPUTS`；Q1 历史接入 155783、计时 pilot 155792 已完成，完整网格 155812 已提交 |
| Q2 完整发展矩阵及非退化 rank 结果 | 完整 source collection 155837 已完成；24-fit timing pilot 155841 已完成；完整矩阵 155844 已以 exit 0 完成，Slurm wall 27:33，原件分析待完成 |
| 方法选择锁、Q3 实际区间校准/锁定测试与外部初始化结果 | `NOT_COMPLETED` |
| 首次 GPU 精确启动命令与实测时长估计 | 待最新技术验收、完整来源绑定、实测 GPU 吞吐及主执行任务交付 |
| 两张 PRO 6000 的真实 Q4 桥接 | `NOT_RUN` |
| Q5 9B 完整源轨迹、候选响应、独立参考和阶段 helper 验收 | `NOT_COMPLETED` |
| Q6 真实轨迹离线窗口 | `NOT_RUN` |
| 最终观测/校准/维数/误差范围/失效条件决定 | 待逐阶段实际证据审阅；本稿不代写最终决定 |

上线 SSVC、自动改变源训练奖励和在线检测控制均未取得有效性证据。论文方法或经典基线的来源核读边界见 `RELATED_WORK_IMPLEMENTATION_MAP.md`；实现通过不能替代全文级复现或新颖性判断。

## 8. 服务器续跑修正与阶段记录

- candidate04 以 4 文件增量修正 pytest 同名模块收集、Q4 换卡 null 评分和 known-event 原件复用；服务器代码快照 998 个文件已核验。归档 SHA256：`ec1d0777cd1fc437272723291ea71512fc691b00544196d50614089091344b88`。
- Q4 换卡修正保留首次原件；每次真实 UUID、硬件、评分、成本和容差比较分别存档。完成 worker 引用本次评分回执；最终跨卡门禁仍要求两个真实不同 UUID。没有完整 hash/request 回执的残缺 known-event 文件拒绝复用。小型 RED 13 failed → GREEN 13 passed；受影响回归 61 passed / 3.12s。真实跨卡续跑尚未运行。证据见 `verification/q4_resume_20260916/`。
- Q1 历史接入选择 12 个原点 view／12 个 bank，来自 11 条既有轨迹。新增 35 次批量 forward、2,520 个 prompt forward、40,320 个动作概率评分；无新采样、无 Adam，准备过程实测 1.1084s。
- Q1 计时 pilot 为 48 单元 × 2 次重复、n64，观测链实测 8.2540s，整个进程 9.00s、最大 RSS 49,484 KiB；它不代表完整网格的耗时或科学结论。完整运行保留 n64/256/1024/4096 和每单元 200 次重复。
- 分区查询显示内存上限字段为 UNLIMITED，但实际 Slurm 作业由调度规则分配 1 CPU / 11 GiB；资源判断以实际分配为准。没有沿用旧研究输出上限。


## 9. 本轮实际 CPU 结果与新增接口

- 完整 development 采集 155837：4 条轨迹、256 次 source Adam、1,296 次 fork Adam；核心 wall 11.834826s；输出 113,990,151 bytes；53 个原件清单项已核验。不是 pilot。
- 覆盖计时 155841：完整 12 原点上的 24 次小设计拟合，核心 wall 15.197812s，进程 15.98s，最大 RSS 528,424 KiB；标记 `PILOT_NOT_CONFIRMATION`，未覆盖最大 n、GLS 和全部方向模型。原件见 `verification/server_Q2_155837_155841/`。
- 完整 Q2 155844：12 原点 × 1,108 设计＝13,296 fits。申请 32 GiB 后 Slurm job-submit 明确强制为 11 GiB；这属于实测调度规则，不能报告已获得 32 GiB。作业已完成，Slurm wall 27:33、exit 0；实际峰值内存以原始 time 日志为准。
- CPU `validate-cpu` / `analyze-cpu` 保留校准回执原路径，由下层重新核对 receipt 所属完整清单，避免 JSON 字典传递跳过回执自身原件验证。
- 首次 GPU 准备现在同时要求：最新源码/测试/脚本的完整服务器工程验收（保留封存数据的显式排除项）、实际非 pilot 的完整 Q1/Q2 development 原件。部分测试 nodeid 不能授权进入 Q4。方法不必优于强直接测量基线。
- `freeze-vlm` 在真实 VLM development 原件上冻结完整设计和显式误差/覆盖规则；`calibrate-vlm`、`analyze-vlm` 校验 3/6 条独立训练 seed 的实际测量矩阵。经验点预测资格与 95% 跨 seed 认证分开；后者始终 `NOT_CERTIFIED`。
- `freeze-prediction-set` 在共享参考开放前冻结全部设计的预测；测试接收 mask 仅来自冻结校准与几何。参考未分辨的已接受单元保留分母，不能用测试误差改 mask 或以已分辨子集风险冒充全接受风险。
- VLM STRATIFIED_RANDOM 使用 bank 的原始 family/interface composition；GROUP_WEIGHTED_RESPONSE 使用原 probe 分组的 X 与 v=-I 映射，不能退化成单一 strata 或每 probe 一组。
- 正交初始化分析 `analyze-cpu-generalization` 按 7101/7201 分层，核验完整 40 条轨迹和 2,400 个 origin/repeat 单元；初始化迁移不继承固定初始化的覆盖保证。
- Q6 已接通真实 step64 起点校准及 64–80 状态 DIRECT 绝对事件观测的生产入口和小型替身测试。它只允许已具有固定线性回放资格的独立测试设计；新增 step64 校准成本与 dense 参考成本分别登记，无新 source training、无在线反馈。真实 Q6 未运行。

## 10. Candidate05 冻结前统一检查

372 项 V3/原交接数学小型工程测试通过（31.87s），运行前后 src/tests/scripts Python 哈希一致；原始日志、JUnit 和收据见 `verification/local_20260916_candidate05/`。Ruff 全 V3 源码、测试及新增运行脚本检查通过。此结果不替代服务器全仓验收或科学确认。

同一节点两张 PRO6000 的 Q4 pair 启动器保留原 Slurm GPU token，每个 worker 使用其单独设备的 cuda:0；启动前验证已记录授权、完整阶段绑定和技术原件。启动器不提交作业、不创建授权。真实 UUID 的不同性仍由桥接最终验收核对。

最终 candidate05 候选在两项集成修复和恢复 UUID 检查后统一复测：388 passed / 31.65s，源代码与测试在运行期间未改变；`verification/local_20260916_candidate05_final/` 保留完整小型工程日志与哈希。所有 V3 源码/测试和新增 Python 运行脚本 Ruff 检查与格式检查通过。真实服务器全仓复验将使用该冻结代码。
