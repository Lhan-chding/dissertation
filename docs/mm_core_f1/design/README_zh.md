# 给 Codex 的使用说明

**先阅读 `CODEX_FINAL_PLAN_MM_AUDIT_ENGINE_zh.md`。它是本包的执行权威文件。**

本包落实最后三轮多模态讨论：回到 Qwen2.5-VL-3B 的真实图表任务，首先确认可测量的语义行为及训练接口。当前允许 MM-AUDIT 和条件触发的工程／桥接，完成后返回报告，**不得自动执行 MM-DEV**。

## 文件

| 文件 | 用途 |
|---|---|
| `CODEX_FINAL_PLAN_MM_AUDIT_ENGINE_zh.md` | 完整执行契约：阶段、数据、评分、预算、权限、验收、后续冻结骨架 |
| `MM_CORE_F1.json` | 机器可读限制与默认审计原型；null 为必须真实核验的字段 |
| `FINAL_DECISIONS_zh.md` | 一页主要决定与对最后 Claude 三问的结论 |
| `reference/contracts.py` | CPU 参考 scorer、代数核对、稳定性噪声对照、配额和权限预算函数 |
| `tests/test_contracts.py` | 55 项构造例子与契约测试；不是模型实验 |
| `run_reference_checks.py` | 在本包根目录运行测试并生成验证回执 |
| `templates/` | 阶段冻结、题目侧车、审计报告和下一阶段补充表模板 |
| `sources/` | 本次直接依据的三份项目文档；保留原文，不把被撤回提案重新执行 |
| `validation/` | 本地 CPU 测试日志与范围说明 |
| `MANIFEST.json` | 交付文件字节数与 SHA-256（不含自身） |

## 本地复核

只需 Python 标准库，在包根目录运行：

```bash
python run_reference_checks.py
```

这条命令不会调用模型、连接服务器、修改队列或执行训练。它仅复核数学、解析、分母、预算及阶段阻断。

## 给 Codex 的指令

请按主计划在当前项目仓库落地任务。先核验 SER-J23 停止状态与真实资产，不假设本机路径就是服务器路径。不要用旧执行器答案拼造双字段数据，不允许通过改提示／容差追求非零补偿事件。所有 GPU 任务必须先登记真实模型、processor、数据、预算和阶段哈希。完成本轮审计及必要工程后，输出主计划要求的报告与 `DEV_FREEZE_PROPOSAL`，然后停止。

## 未执行事项

本包没有运行 Qwen、绘制新图表并完成其视觉审查、恢复旧 VL 原始回答、测试 CUDA／optimizer、测量吞吐或停止服务器作业。CPU参考代码不是生产模型训练器。未来 1088 次准备／续训更新仅为后续结构的算术，不是本轮任务或已完成记录。
