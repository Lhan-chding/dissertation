# 交付文件

`OVERVIEW_zh.md`是研究阅读入口；`START_HERE_FOR_CODEX.md`是Codex执行入口。

- `CODEX_EXECUTION_PLAN_zh.md`：完整实验规范。
- `protocol.json`：机器可读的科学与运行配置。
- `THEORY_AND_STATISTICS_zh.md`：推导、估计与边界。
- `DATA_CONTRACTS_zh.md`：输入、标签、缓存、训练view与评价schema。
- `GPU_RUNBOOK_zh.md`：待实现CLI与多卡运行。
- `CPU_ACCEPTANCE_CHECKLIST_zh.md`：实际接入验收项。
- `SOURCES_AND_SCOPE_zh.md`：旧证据、外部资料、新设计的区分。
- `reference/`：纯CPU数学与配置参考，并非真实模型训练实现。
- `templates/`：服务器配置占位示例及未启动作业表。
- `verification/`：本次实际参考测试记录与范围。
- `sources/`：必要的既有资料摘录，不含权重。

配置或工作量改变后，在`reference/`中运行`python compile_plan.py`与`python -m unittest -v test_contracts`。测试中的精确数量是本版本的冻结约定，变更科学设计须同步记录，不能为使测试通过而静默改数字。

本包没有执行新的真实模型实验，也没有承诺结果或耗时。
