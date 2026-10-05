# 运行状态与继续入口

本记录区分 CPU 验证、调度器接受、真实模型输出和科学结论。当前尚未完成任何新增 GPU 回答，不能报告干预效应。

## 已完成与保留的证据

- 初始实现 `d15c015e87fbd30580c93f4ed0a80f097e26e771` 已提交并核实远端分支。
- 初始 CPU 联合测试 192 passed；显存与首块资源记录补充后的针对性测试 52 passed，新增模块全部 88 项测试通过，Ruff、Slurm shell 语法与 diff whitespace 检查通过。
- 原包、逐字段重编译、六个 checkpoint 元数据可读性与 U22 污染检查见同目录 README、IMPLEMENTATION_REPORT 与 U22_EXPOSURE_SUMMARY。U22 整组保留并降级为开发诊断，不称独立未用数据。
- 旧服务器运行目录：`/projects/varunssd/louis-ssvc/protocol_state_probes_v1_20261005/run`。该目录没有已接受 GPU 作业；`scheduler/lane-1.intent.json` 与 `lane-1.submission.json` 保留首次拒绝。
- 首次 `sbatch` 返回 1、stdout 为空，明确报告 `QOSMaxSubmitJobPerUserLimit`。该失败不能计作模型失败或实验输出。2026-10-05 重新连接后查询队列，没有 `psp-v1-` 活动作业。
- 调度器强制每张 PRO6000 配 4 CPU 和 33792 MiB 内存，提交脚本已对应改为 33 GiB。原 QoS 保持 `soujanya-poria-startfund-2026-03`，不切换 QoS 绕过限制，不停止其他人的作业。

## 接下来固定执行

显存记录修订后重新冻结代码和 `run_v2` 身份；保留旧空运行及其调度拒绝记录。先提交 lane-1 S96，核实加载、16 条独立 smoke 和首 8 条正式输出；S96 首个交付为完整 D48 核心及三个执行对照。其余登记矩阵保持不变，不按早期效果删臂。

五条 lane 的 checkpoint 顺序为 S96、S32、R4_128、DIRECT_128、REP96→REP32，每条单卡，同时最多五卡。全部正式输出为 70,656 条；smoke 最多 80 条，单独计数。首次加载和每次输出读取 CUDA allocator 峰值计数，不额外同步、不改变采样；`first_block_receipts/<checkpoint>/<role>.json` 保存各角色首八条实际资源数据。

服务器根目录：`/projects/varunssd/louis-ssvc/protocol_state_probes_v1_20261005`。原环境：`/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python`。原 runtime 和 prepared 数据：`/projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/`。新生成原始输出和私有 runtime 仅留在服务器及本地忽略目录，Git 只保存实现与必要审计摘要。

最终须依据完整真实表格回答结构与状态信息的解释力、反例以及是否需要新训练或状态路由；当前没有足够新增数据作出这些判断。本任务不授权新增训练、选择器拟合或状态路由实验。
