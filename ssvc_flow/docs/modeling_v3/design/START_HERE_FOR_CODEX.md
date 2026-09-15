# Codex 入口：语义概率建模 V3（两张 PRO 6000）

## 用户目标与本轮边界

用户要求继续完善概率状态／变化建模，明确提供两张 PRO 6000，不再把上一轮 4 小时 CPU、3/6 GiB 输出量、禁止新 Qwen 调用或“成本不赢就不准做真实验证”作为本轮限制。资源用于提高证据质量：观测噪声、校准覆盖、实际低秩与真实 9B 的响应验证。

本文件允许规划、实现和准备 CPU/GPU 实验；**仅收到本交接文件不等于已经提交服务器作业**。完成当前阶段后，向用户提供测得的运行估计与准确启动命令；已有服务器执行授权只能用于其明确覆盖的工作。新 GPU 作业首次提交前确认授权范围。整个项目最多占用两张 GPU，不取消其他正在运行的作业。

## 阅读顺序

1. `CODEX_EXPERIMENT_PLAN_zh.md`：科学问题、实验矩阵和每个阶段的验收。
2. `THEORY_AND_ESTIMATORS_zh.md`：精确数学定义；尤其区分有噪声观测投影与无损坐标变换。
3. `GPU_RUNBOOK_zh.md`：双卡调度、真实序列概率、原件恢复与运行交接。
4. `CPU_ACCEPTANCE_CHECKLIST_zh.md`、`protocol.json`。
5. `evidence/` 中上一轮结论与对照；这些是已公开历史数据，不是新确认集。
6. `SOURCES_AND_NOVELTY_zh.md`：经典控制变量、校准设计、低秩工具必须保留为基线，不宣称它们首次出现。

## 代码起点

仓库 `Lhan-chding/dissertation`，工作目录 `ssvc_flow/`。
本次核对的分支：`codex/ssvc-modeling-contrast-v2-20260915`。
本次核对的 HEAD：`474601e020184776265fbfc0ffea343bb56eb21e`。
建议新分支：`codex/ssvc-modeling-v3-observation-coverage`，使用独立 worktree。

开始前重新读取远端和本地 HEAD、dirty state、已有作业。若 HEAD 更新，先做源码差异审查，不回退、不 reset、不覆盖已有工作。`main` 不是当前建模代码起点。禁止覆盖旧 `modeling_contrast/` 的实验锁与输出。

## 实施顺序

Q0 原件审计 → Q1 观测噪声几何 → Q2 校准方向与非退化模型 → Q3 新 CPU 确认。
在 Q1/Q2 的技术验收完成后，可以并行准备 Q4 双 GPU 真实观测桥接与 Q5 9B 轨迹收集；**不要求新算法先胜过所有基线，才允许收集真实现象**。
Q5 的锁定测试数据必须在算法、阈值和选择规则冻结后才开放。
Q6 仅离线回放短窗口，不让检测器实时改变源训练。
Q7 汇总分层建模决定。

## 必须完成的具体工作

- 新增 `src/modeling_v3/`，实现原始4维测量保留、至少5个简单/经典约束估计器、独立 pilot 修正、校准选择与覆盖诊断。
- 恢复旧 all/active 及 raw/Helmert 对照；不得从汇总表反造原始数据。
- 新实现与旧 joint / no_x_off、真实 Adam、检查点恢复及 logprob 路径做回归。
- 对零向量、别名、r=k、没有激励、越界更新进行显式分类；它们不能被计为“安全”。
- 固定种子、数据、方法、阈值、采样拆分、完整计费账本；执行真实 CPU 测试并记录失败/修复过程。
- 编写 `plan / audit / test / observe / select / fit / evaluate / train-source / make-forks / track-offline / summarize` 对应 CLI。主文档中的命令是**待实现的接口契约**，不是声称仓库当前已有全部入口。
- 高精度 VLM 参考也是估计量，必须保留其误差；没有精度时返回 `REFERENCE_UNRESOLVED`。

## 最终交付

`IMPLEMENTATION_REPORT_V3_zh.md`、`CPU_TEST_RESULTS.json`、`SOURCE_BINDINGS.json`、`OBSERVATION_GEOMETRY_zh.md`、`CALIBRATION_COVERAGE_zh.md`、`MODELING_DECISION_V3_zh.md`、`SERVER_HANDOFF_V3_zh.md`、`RUN_MANIFEST.json`。

报告分别说明：数学测试、固定策略观测、小模型预测、9B 观测、9B 建模、离线跟踪各自做到哪里。不要用一个 `PASS` 混合工程完成与方法有效。
