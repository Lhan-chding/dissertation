# 2026-09-14 S1 双 PRO 6000 执行记录

用户在 CPU 和单卡 smoke 复核通过后授权进入下一步，并明确要求两个卡一起跑、可以逐个申请。本次采用两个独立单卡 Slurm 作业并行，覆盖旧文档按 bank 串行的执行安排；冻结设计、算法、原件、样本和预算没有改变。

## 已核验的实际运行状态

截至 2026-09-14 15:47:49 UTC（新加坡时间 23:47:49），bank00 已完成并复核，bank03 和 bank04 分别在两张 PRO 6000 上运行。以下保留初始启动快照，最新结果及补位回执见本文末尾。

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

## bank00 完成复核与 bank04 补位

bank00 作业 154220 于 15:34:25 UTC 正常退出，Slurm 状态 `COMPLETED`、退出码 `0:0`，墙钟耗时 5,638 秒（1 小时 33 分 58 秒）。最终状态为 `MEASURED / REAL_CUDA_FOLLOWUP`，`training_started=false`、`permanent_training_commit=false`。服务器核验了 manifest 全部 52 个文件（含 checkpoint）、73 个冻结源码以及计划、模型、数据、parser、origin 身份；3 次更新均已确认，未决更新为 0，共 96 次 backward。三个候选均从 step64 独立更新至 step65，最终原状态恢复门禁通过，无 fault。

bank00 训练 bank 的四组计数合计为 X27/W5/S0/I0。固定评估 panel 有 24 场景、48 prompts，每题 16 条，共 768 条独立样本，评估计数为 X537/S7/W186/I38。原始 sample 请求与 seed、record hash、execution checks、prepared input 绑定、语义重新解析、旧 logprob 与逐题计数均核验通过。训练 bank 与评估 panel 的计数属于不同对象。

joint1 与 joint0 的完整 forward 策略指纹相同，joint1 复用 joint0 的样本，新增独立样本为 0；两个候选各自的更新和 checkpoint 均保留。七项 paired 指标在七个报告范围内的差值均为 0。5,000 次共享场景 bootstrap 抽样、逐题汇总和固定 panel 区间经独立重算一致；退化的 paired bootstrap 标记为 `UNINFORMATIVE`，不能改写为 `[0, 0]` 置信区间。全候选 union M=35，pair 报告 M=70 是重复标签的保守计数，不能算成 1,536 条独立样本。整体 pX=0.69921875、v=0.9505208333333334、qX=0.7356164383561643。

baseline replay 的参数、Adam、优化器结构及 RNG/sampler 检查一致，但 `state_bitwise_equal=false` 保留。geometry 的 `alias_authorized=false` 也保留；样本复用依据另行核验的完整 forward 指纹。结果继续为 `NOT_CERTIFIED`，不能据此认定训练收益、安全性或跨 seed 泛化。runtime profile 记录峰值 CUDA 已分配内存 24,173,737,472 bytes、完成调用耗时 4,577.839056 秒、13,268 次 forward；其范围不含整个 Slurm 前置准备和模型加载时间。

本地紧凑证据包有 54 个普通文件，下载 SHA-256、路径安全和 46 个包含在 bank manifest 中的非 checkpoint 文件均通过；6 个 manifest checkpoint 路径未下载，其哈希已在服务器验证。六份统计原件与独立复核报告引用的哈希一致。证据位于本地忽略目录 `runs/mechanism_followup_server/20260914/retry_154599c/bank00_review/`，服务器完整原件继续保留。

| 证据 | SHA-256 |
| --- | --- |
| bank00 最终 manifest | `b3e0bf6bb778e9196e6bde1ee609406bdb8174fe3867aeb7a4e79d96d6f48893` |
| bank00 紧凑证据包 | `b8fb4daf4f2919b8e8f2140eb4c21f393ab08543bfdf8b3649988bb06d764310` |
| bank00 完整复核与推进决策 | `a9d410e4c1b123c307866a0876183105726b0449ac2c0f469af673d2a7a72cab` |
| bank04 单卡启动脚本 | `d5505a84da0588d73bf8035d7ee159ddc2e353a4fffbca123fef5c4ea1e6dc3b` |
| bank04 实际提交回执 | `0361ecaa8d8998162e4c2b9c91ae68f313f91b6bba3b87628ce981dcec87cb8d` |
| bank03/bank04 运行快照 | `b6b5feffd567489cec83bd94f4630227304ca60478b3b15bfa830cd5ba1c1181` |

完成上述复核后，重新检查冻结来源与 smoke 门禁，并对 bank04 启动脚本执行本地和服务器 `bash -n`、无模型 dry-run。确认仅 bank03 占用一个 S1 名额、bank04 输出和提交 intent 不存在后，以独占 intent 和提交回执防止重复申请。bank04 作业 154346 于 15:47:14 UTC 提交、15:47:15 启动，仍使用相同特殊 QOS、单张 PRO 6000、24 小时上限。冻结源码及运行环境未改动。

| bank | 作业号 | 15:47:49 UTC 实查状态 | GPU 节点 / 物理索引 |
| --- | --- | --- | --- |
| bank00 | 154220 | COMPLETED；结果复核完成 | 已释放 |
| bank03 | 154221 | RUNNING；7 个候选标记，尚无 bank 完成 | gpu-pro6000-5 / 5 |
| bank04 | 154346 | RUNNING；启动回执已确认 | gpu-pro6000-7 / 4 |

后续未提交顺序更新为 bank05、bank11、composite_no_x_plus_three_level。自动任务已同步 bank00 完成事实和两个当前作业号，继续遵守每个完成 bank 先分析、最多两个排队或运行作业、S1 完成后暂停的规则。
