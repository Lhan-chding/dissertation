# Codex 从这里开始

## 任务

实现并执行`CODEX_EXECUTION_PLAN_zh.md`规定的真实Qwen3.5-9B冻结推理。目标是检验**输出顺序、等价关系表达及当前模型状态**怎样影响修复行为，而不是重跑奖励训练。

阅读顺序：
1. `OVERVIEW_zh.md`；
2. 主实施规范，尤其第3–9节的定义、矩阵与统计；
3. `PROMPT_EXAMPLES_zh.md`、`protocol.json`和`manifests/`；
4. `GPU_RUNBOOK_zh.md`与`CPU_ACCEPTANCE_CHECKLIST_zh.md`。

## 已有与待实现

本包提供实际题目编译后的509个case、预期解析规则、审计标签、PTLC程序预测和70,656条去重后计划回答的任务清单；这些都不是新模型输出。`reference/`已用于CPU契约检查，不是GPU runner。

你需要新增`src/protocol_state_probes`并接入仓库实际推理与checkpoint loader。不要直接调用原评价器把反序输出当正序，也不要用fixture开关绕过旧面板限制。

## 工作边界

- 允许最多5个单卡并行任务，卡型/显存以实际分配为准。
- 使用已保存的6个检查点；缺少文件只记录缺项，**不自动补训练**。
- 不调用backward、optimizer.step，不拟合新选择器，不新增似然重评分。
- 不改旧训练脚本、旧原件或旧统计结果。
- 保留原始回答、token、停止原因、完整公开提示及固定逆排列。
- 旧数据只用于来源和历史对比；O0在本轮必须重新采样。
- 别名共享的是同一份证据，不是独立重复。
- 一次source/version记录、一次导入核对、每个checkpoint一次加载验证；不要逐样本全模型hash或每个worker重复全仓大测试。

## 最短执行路径

1. 在独立worktree确认父版本；读取主规范并完成编译、解析和PTLC测试。
2. 核实旧checkpoint COMMIT及路径，产出`CHECKPOINT_AVAILABILITY.json`。
3. 实现raw-only后端、seed隔离、8-draw原子提交、恢复和报表。
4. 用执行对照做技术smoke；确认主数据没有truth/j/DPE注入。
5. 先完成S96的D48核心和完整对照，输出第一批报告，同时其他卡运行其余预定检查点。
6. 完成T1、T2、2×2因子、U22及REP矩阵。阶段可并行，但不能按结果删臂。
7. 输出`FINAL_MODELING_DECISION_zh.md`和紧凑结果包，停止。新的训练和路由需要下一轮单独设计。

## 命令接口

主代码尚未实现，因此下列是**要求你实现的CLI合同**，不是声称仓库已有的命令：

```bash
python -m src.protocol_state_probes.cli prepare --protocol protocol.json --prepared /path/to/prepared.json --out /path/to/run
python -m src.protocol_state_probes.cli check-checkpoints --run /path/to/run
python -m src.protocol_state_probes.cli smoke --run /path/to/run --allow-gpu
python -m src.protocol_state_probes.cli worker --run /path/to/run --checkpoint S96 --allow-gpu --resume
python -m src.protocol_state_probes.cli summarize --run /path/to/run
```

`prepare/check-checkpoints/summarize`不得实例化模型；`worker`不得开启训练。服务器Slurm提交命令必须依据实际账户/QoS、分区和已分配资源生成，不在文档中猜测。

请交付真实代码、执行记录和数据分析，不停留在伪代码；技术缺项只局部阻塞，不以某方法效果差为理由停止完整预定对照。
