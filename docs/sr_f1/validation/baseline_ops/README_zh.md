# 基线并行部署操作

- `stage_candidate.py`：只向独立候选目录展开已逐文件校验的源码包，并提交CPU测试。不修改生产源码或GPU作业。
- `arm_handoff.py`：确认服务器CPU测试作业COMPLETED/0:0和候选源码身份，保存用户授权及旧CPU控制器真实身份，复制不可变的交接程序到本实验技术目录，提交并验证CPU交接作业后释放。
- 交接程序的生产源码为 `scripts/sr_f1/baseline_handoff.py`，稳定执行副本为服务器 `technical_incidents/baseline_parallel_20261010/baseline_handoff.py`。后者跨源码目录切换和Slurm requeue保持可用。

两份操作脚本均只适用于已明确的独立运行根。提交意图、提交结果和释放检查不可覆盖；出现未知提交时先核实Slurm与回执，不重复运行提交脚本。新CPU交接器只替换身份核实后的本实验旧CPU控制器196501，保持GPU196508运行。源码仍为原ENGINE版本，直到ENGINE完整验收成功并且已终态退出。

实际作业、测试与哈希见本目录上级的部署核验JSON及 `BASELINE_PARALLEL_REPAIR_zh.md`。原科学矩阵、注册、冻结和失败证据保持。
