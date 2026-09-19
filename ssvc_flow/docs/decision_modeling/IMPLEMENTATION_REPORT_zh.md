# decision-modeling-v1 本地实施报告

日期：2026-09-19。范围：按收到的方案完成本地代码、D0 复算、输入冻结和 CPU 验证，停止在需要服务器执行真实 9B 的 D1 之前。

## 实际完成

新增 `src/decision_modeling/` 15 个 Python 文件，未修改旧实验模块：

| 文件 | 已实现行为 |
|---|---|
| semantic_schema / state_table | 严格完整输出 Phi、精确关系分子/分母、派生量嵌套、逐题及分群体 Z0–Z4、保留重复抽样频数 |
| reanalysis | 读取真实原包 CSV/记录，核对完整 tokens、EOS/length、source role、输入和评分关联；生成 Parquet、别名、复用键、质量界和非对角协方差 |
| score_cache / measurement | 进程内和 SQLite 持久评分缓存，真推理 fingerprint、输入/完整动作/路径键；相同候选下批量评分；端点/ORIGIN/随机 MIX、RAW4/PRESERVE_XI、支持质量、固定 look 与前缀恢复 |
| panels / runtime | 元数据确定性 P/E、旧 R3/debug 排除、原数据及图像一次校验；继承已验证环境；导入原始完整 checkpoint |
| reward_recipes / block_runner | R0–R7、官方 GDPO 定义及论文 SAW 模式；真实 on-policy B4/K8、同完整起点、H8/H32、恢复状态、失败保留、评估 RNG 隔离 |
| intervals / decision_readout | 条件质量界和统计区间分离，固定 look 多重误差预算，群体非劣/伤害/等效/未判定，UNKNOWN 候选保留在 regret 中 |
| evaluation / reporting | 场景成组且家族分层 bootstrap 5000、固定面板 MC、独立参考及其区间、同信息 Z0–Z4 回放、六群体及成本报告；E 不反向选规则 |
| cli / __init__ | plan/reanalyze/prepare-runtime/bridge/block/observe/report；默认 GPU 命令只做 dry-run；E/D4 必须依赖全部 16 个开发分支 H32 和冻结规则 |

`protocol.json` 冻结研究设置；`panels.json` 固定 P/E 和输入 hash；真实路径只由未提交的 `runtime.json` 提供。D2 登记 512 更新、16,384 训练输出；D2+D4 上限 768 更新、24,576 输出。没有新源轨迹矩阵或自动后继作业。

## 实际本地数据结果

- D0：2,304 条生成、3,072 条评分、6 prompts/3 base_scene/1 历史原点；446 个核心唯一评分键。
- 严格 parser、完整 token/终止条件、评分关联不一致均为 0。
- ORIGIN/MIX 均值及协方差相对历史诊断最大绝对差 `3.47e-17`。
- 2 个 prompt-policy 情形没有 X 支持，未知尾部保留；未据此断言 X 不存在。
- 2,626 行可复用评分计算，占评分行约 85.481771%；这是复用潜力，尚无新缓存的真实墙钟加速结论。
- P/E：原 control 144 场景，排除后118可选；各36场景、72 prompts；72 张图像 hash 全部核验；缺额0。

原始 D0 Parquet、逐流 Z0–Z4、质量界、LR 矩与完整性收据位于本地 `ssvc_flow/runs/decision_modeling_ready/D0/`。报告输出位于同级 `report/`；`COMPLETE.json` 绑定三个输入 CSV、实施源码和每项输出的 SHA-256，真实 Parquet 已读回核验。

## 验证证据

阶段集成回归：**163 passed in 4.92s**。覆盖新增模块和实际 D0 原包复算，以及相关历史 followup 更新、V4 采集/preview、V3 观测接口；没有重复运行全仓巨型 CPU 实验。

```bash
DECISION_D0_ARCHIVE=/path/to/SSVC_V4_EXPERIMENT_DATA_COMPACT_20260919.zip \
python -m pytest tests/decision_modeling tests/test_panels.py tests/test_rewards_blocks.py \
  tests/test_followup_updates.py tests/modeling_v4/test_gpu_collect.py \
  tests/modeling_v4/test_measurement_preview.py tests/modeling_v3/test_vlm_observation.py -q
python -m ruff check src/decision_modeling tests/decision_modeling \
  tests/test_panels.py tests/test_rewards_blocks.py
python -m compileall -q src/decision_modeling
```

Ruff 和 compileall 通过。最后一次 LR 启用状态调整后，6 项完整观测集成测试再次通过。plan、真实 reanalyze、report 命令已实际执行；GPU 命令的默认不加载模型行为有 CLI 测试。完整日志在本地 `runs/decision_modeling_ready/validation.txt`。

重点测试包括：同 token 重复观察不去重、持久缓存复用与 namespace 隔离、输入/策略/路径变更分离、非有限评分拒绝、32→128不重采旧前缀、角色/策略 RNG 分离、固定24条路径比较、真实 tiny Torch 的 ORIGIN/MIX与质量界、故障记录和恢复、H8→H32与连续32步完全一致、完整Adam/RNG/sampler恢复、评估失败后不重复更新、服务器图像字节变化拒绝、无X未知、零经验方差不冒充确定性、UNKNOWN候选regret、独立E不参与选择。

## 尚需服务器验证及明确未完成的实验

| 项目 | 当前事实 |
|---|---|
| D1 真9B prefix/full/chunk/token、梯度路径 | NOT_RUN；生产保留prefix，不声称修复了full teacher或梯度一致性 |
| GPU 新缓存吞吐、显存与forward成本 | NOT_MEASURED；CPU缓存测试不是墙钟加速证据 |
| 原seed41001/41002 step32/96完整.pt恢复 | 代码已接入，真实原件仅服务器可读；未在本地导入或替代构造 |
| D2 16条分支H8/H32、全部多奖励结果 | NOT_RUN |
| D3 同证据选择收益、独立参考/E | 无新实验数据，UNRESOLVED / NOT_ESTABLISHED；均值目标不预设联合信息胜出 |
| D4 step96阶段迁移 | NOT_RUN；需全部D2结果及冻结选择；不是新训练seed |
| CROSSFIT / tail MC | 旧full-refit观测模块仍可用，但新正式CLI默认RAW4/PRESERVE_XI；未把同一发现样本当独立tail MC，也未使用未经校准的经验区间作保证 |
| DVAO / CGRPO | 完成paper→code映射；按方案不实现/运行后续控制基线 |
| 在线控制、额外seed | 未实现为本轮自动流程，未启动 |

条件质量界一律 `CONDITIONAL_ON_SCORER`，尚无评分数值误差界，科学状态仍 `NOT_CERTIFIED`。五份阶段报告是本地交接时点快照：D0实际完成，新D1/D2/D4明确未运行；服务器运行后用 report 命令生成新目录，不覆盖这些历史事实。

下一必要动作及完整命令见 [服务器交接](README_zh.md)：绑定服务器现存原件，在单卡分配内首先运行 D1 bridge，然后读取实际输出再安排 D2。
