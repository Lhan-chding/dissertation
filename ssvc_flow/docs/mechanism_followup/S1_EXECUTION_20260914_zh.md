# 2026-09-14 S1 双 PRO 6000 执行记录

用户在 CPU 和单卡 smoke 复核通过后授权进入下一步，并明确要求两个卡一起跑、可以逐个申请。本次采用两个独立单卡 Slurm 作业并行，覆盖旧文档按 bank 串行的执行安排；冻结设计、算法、原件、样本和预算没有改变。

## 已核验的实际运行状态

以下是 2026-09-14 14:02:30 UTC（新加坡时间 22:02:30）的快照，不是最终结果。

| bank | 实际作业号 | 状态 | UTC 开始 | GPU 节点 / 物理索引 | GPU 数量 |
| --- | --- | --- | --- | --- | --- |
| bank00 | 154220 | RUNNING | 14:00:27 | gpu-pro6000-5 / 0 | 1 |
| bank03 | 154221 | RUNNING | 14:00:29 | gpu-pro6000-5 / 5 | 1 |

作业分别于 14:00:25 和 14:00:26 UTC 提交，约 2 秒和 3 秒后启动。`scontrol show job -dd` 确认分配的是两张不同的 PRO 6000。两份启动回执存在，stdout/stderr 当时为空，无退出回执。尚无 bank 完成结论；S2 正式训练未启动。

两作业使用特殊 QOS `soujanya-poria-startfund-2026-03`、partition `cluster02`、account `rose`，各申请 `gpu:pro6000:1`，24 小时墙钟上限，no-requeue。CPU 和内存由调度器分配，提交脚本未手填 `--mem` 或 `--cpus-per-task`。24 小时是资源上限，不能由 smoke 耗时推断 S1 必定在此时间内完成。

## 身份与提交验证

执行源码仍冻结于 `154599ced9a0826997bdc69b2ed6ac4943e863bd`。提交前在原服务器环境中重新核对：73 个源码哈希与完整计划和真实 smoke 原件一致；`_smoke_gate` 通过；bank00、bank03 均执行了不加载模型的 dry-run；两份启动脚本本地及服务器 `bash -n` 均通过。没有重跑已完成 CPU 准备或 smoke 作业，各 S1 命令自身的完整前置门禁保持执行。

- 完整计划 canonical SHA-256：`eddfb5a469807cd2761208ffc39d78908ca204cd8b1d6ae4bf1ebcdf0163bcb5`。
- 已通过 smoke 文件 SHA-256：`c623957dd596614acf358e443ce335d4085341b7be6f0b8fe83b35324703dbb8`。
- bank00 提交回执 SHA-256：`b22d94c03bfe5eaadcb97ddd1c90f10c273633b1fae3cab22a46bbf84dbefa99`。
- bank03 提交回执 SHA-256：`00ce7931c1b86692abe10becf998b5b1a63678b81182bfedd6933c3c11656770`。
- 两作业运行快照回执 SHA-256：`d93db92fa0c82f83f611cb8f3d7834fc515e21295d93b7fa8ca4f82dedaf97bd`。

脚本、提交 intent/回执及运行快照保存在服务器新结果根的 `receipts/` 和本地忽略目录 `runs/mechanism_followup_server/20260914/retry_154599c/s1_launch/`。脚本调用已有 `scripts/followup_s1.sbatch`；服务器冻结源码及环境未修改。输出分别是新结果根的 `S1/bank00`、`S1/bank03`，未覆盖旧结果。

## 并行隔离、预算及推进条件

只读复核确认两 bank 无跨 bank 数据依赖：每个进程独立加载同一只读 X_BASE step64，独立维护 adapter、Adam、origin、输出锁和预算，随机身份包含 bank ID。共享 `revalidation_audits` 使用内容哈希路径与幂等原子发布，允许并发写入相同内容。

| 固定计数 | bank00 | bank03 |
| --- | --- | --- |
| 候选 + replay | 2 + 1 | 9 + 1 |
| 最大可能 Adam 调用 | 3 | 10 |
| backward sequences | 96 | 320 |
| 最大新增输出 | 1,536 | 3,072 |

每个唯一策略为 48 prompts × 16 samples = 768 条。只在完整策略指纹相同时共享样本，不增加样本量。更新中断可能永久消耗 reservation；不能删除记录、换目录或增加预算补跑。合法 resume 使用原身份、原目录和完整账本，先核对是否可以复用已完成候选。

后续待执行 bank 顺序为 bank04、bank05、bank11、composite_no_x_plus_three_level。保持最多两个 S1 作业排队或运行。每个完成后先核对 completed/status/manifest、来源身份、预算、candidate、baseline replay、group statistics、geometry、aliases、原始计数及 execution、paired/all-candidate response、runtime profile 和日志，再决定是否补充下一独立 bank；不按退出码或 PASS 自动跳过分析。

既有自动任务已改为 S1 双卡执行与结果检查。六个 bank 全部完成后，生成并复核实测 S1 技术链，回传分析结果并暂停；不会因两张卡可用而自动进入 S2。结果保留 `NOT_CERTIFIED` 和所有不确定性，不把作业启动记为实验完成。
