# C 局部信用试验：部署依赖遗漏恢复

2026-10-24 09:30:06 UTC，双卡作业 223622 获得资源；09:31:08 以 FAILED/1:0 退出。24 个初始可执行任务（12 条训练、12 个父模型评测分片）均在导入 `sr_f3_ce.training → sr_f12.training → sr_f1.contract` 时因部署目录缺少 `docs/sr_f1/package/PACKAGE_SHA256.json` 失败。这发生在 Runtime 构造之前；现场无 accounting、runtime_identity、评测回答或 checkpoint，训练目录只有初始 run_identity.json。不能将获得 GPU 资源称为实际训练。

根因是新试验部署绕过已有 `sr_f4.deployment.build_archive` 的完整资源清单。此次使用该既有归档器构建新的独立代码目录，补齐原版本的历史契约资源。原目录及失败日志保留；运行时、判定器、训练目标、数据和评测算法不修改。`c_local_controller.sbatch` 在准备数据和租赁 GPU 之前运行已有 CPU 导入检查；已有 READY 继续跳过数据准备。该检查不加载模型、不调用 GPU、不生成回答，不进行额外哈希扫描。

`sr_f4.c_local_recovery` 只接受本池已离队、sacct 确认 FAILED/1:0 且用户/QoS/account 匹配的初次故障。核对全部 24 个原始 attempt 的同一精确 traceback、无模型输出、无活跃/未知预约及已释放任务锁，写 PREPARED 回执后将这些任务恢复为 PENDING。保留原 attempt、日志和技术事件，增加 recoveries；保留原作业终态。其他错误、新 attempt 或模型输出存在时拒绝恢复，不能泛化重试。

恢复记录为运行根 `RECOVERY_RUNTIME_RESOURCES.json`，含原池完整快照、部署版本、终态与 CPU 导入结果。FROZEN/CONFIG/TASKS/PRELAUNCH_GATE 及初始代码身份保持原样；新部署位置另写启动登记。使用原两卡/余量一卡调度和全账号五作业、老师 QoS 八卡限制。只有一个名额时优先 GPU，不占用唯一名额创建 CPU 控制器。

验证：46 项相关 CPU 回归通过（恢复边界、已有部署归档器、任务池）；bash 语法检查和 Ruff 检查通过。无新增训练、评测、GPU 烟测或科学预算。
