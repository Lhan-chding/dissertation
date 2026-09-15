# 2026-09-14 S1 双 PRO 6000 执行记录

用户在 CPU 和单卡 smoke 复核通过后授权进入下一步，并明确要求两个卡一起跑、可以逐个申请。本次采用两个独立单卡 Slurm 作业并行，覆盖旧文档按 bank 串行的执行安排；冻结设计、算法、原件、样本和预算没有改变。

## 已核验的实际运行状态

用户于2026-09-15要求：当前两项任务完成后不再开展新实验，只整理本轮全部已完成实验的详细结果文档。该要求覆盖本文较早的补位安排。最后实查时间为05:59:06 UTC：composite bank（154797）仍为RUNNING；历史R2 X_BASE（154821）已于05:09:03 UTC正常完成、exit0，完整原件和独立统计复核已完成。此前五个S1 bank均已完成复核。X_VALID尚未提交，现标记NOT_RUN_CANCELLED_BEFORE_SUBMISSION_BY_USER；S2及其他新增实验不执行。每小时仅监控当前任务，完成已有证据复核和六bank CPU整体分析，生成、核验并交付一份详细中文文档后暂停自动任务。以下较早记录均为历史快照，最新边界见本文末尾。

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

## bank03、bank04 完成复核与后续双卡补位

2026-09-14 18:55:47 UTC（新加坡时间 2026-09-15 02:55:47）检查时，两作业均已 `COMPLETED`、退出码 `0:0`，最终状态 `MEASURED / REAL_CUDA_FOLLOWUP`。bank03 于 18:12:39 UTC 结束，墙钟 15,130 秒（4 小时 12 分 10 秒）；bank04 于 18:24:51 UTC 结束，墙钟 9,456 秒（2 小时 37 分 36 秒）。两者 `training_started=false`、`permanent_training_commit=false`。

下表为每个独立候选在固定评估 panel 上的原始计数：24 场景、48 prompts、每题 16 条。pX 与 v 为概率，qX 为有效回答中的 X 比例。不同 bank 分别保留自己的采样身份，不跨 bank 合并成同一次策略比较。

| bank | 候选 | n | X | S | W | I | pX | v | qX |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bank03 | joint_0 | 768 | 540 | 9 | 181 | 38 | 0.70312500 | 0.95052083 | 0.73972603 |
| bank03 | joint_1 | 768 | 535 | 9 | 183 | 41 | 0.69661458 | 0.94661458 | 0.73590096 |
| bank03 | joint_1em2 | 768 | 537 | 9 | 183 | 39 | 0.69921875 | 0.94921875 | 0.73662551 |
| bank04 | joint_0 | 768 | 536 | 8 | 195 | 29 | 0.69791667 | 0.96223958 | 0.72530447 |
| bank04 | joint_1 | 768 | 534 | 9 | 193 | 32 | 0.69531250 | 0.95833333 | 0.72554348 |

bank03 有 3 个独立采样策略、2,304 条独立样本；bank04 有 2 个、1,536 条。两者的 `no_x_off_1` 都按完整 forward 策略指纹复用各自 `joint_0` 的样本，新增样本为 0，各自候选更新和 checkpoint 仍单独保留。两项 joint1 相比 joint0 均多 3 条 I；pX、v 均下降，qX 在 bank03 下降、bank04 略升。

服务器独立核验覆盖 bank03/04 的完整 manifest 137/133 个文件路径（各含 20 个 checkpoint 路径），各 10 个实际候选 checkpoint 均安全加载到 CPU、全部有限，从原 X_BASE step64 独立更新至 Adam step65。每 bank 10 reserved=10 confirmed、0 uncertain、320 backward；全部样本重新构建请求、seed、原始 control 输入绑定并执行语义重解析，逐题计数与账本一致。73 个源码、计划、模型/data/parser/origin 身份均与冻结来源匹配。未重新加载模型或生成 GPU 样本；模型文件本轮未重读哈希，身份依据既有冻结计划、smoke、checkpoint 绑定。prepared tensors 未重新生成，保存的输入哈希逐条与原 control 行比对。

重放实际完整状态的唯一差异已从 checkpoint 证明为 `metadata.candidate_spec.id`（joint_0 / joint_0_replay）；其余状态逐字节一致。保留原始 `state_bitwise_equal=false`，不改写为完整状态一致。

本地紧凑包分别含 123/119 个普通文件，下载 SHA、路径安全和包含的 117/113 个非 checkpoint manifest 文件均通过；各 20 个 checkpoint 路径仅保留在服务器，其文件哈希及实际状态已单独验证。统计报告引用的所有源文件 SHA 与下载原件一致。完成证据与独立复核没有发现阻止下一独立 S1 bank 的完整性、执行或统计一致性问题；后续按预先设计继续，不因点估计方向改变样本或阈值。

两 bank 的逐题原始计数、固定权重、所有群体与候选的 5,000 次共享场景 bootstrap、全候选联合固定 panel 界、比率与差分边界已用独立实现重算，未调用生产统计模块重跑。重算与原文件一致；bootstrap 最大绝对差 2.22e−16，百分点换算最大差 1.11e−14。bank03/04 全候选 union M 分别为 105/70，pair M 均为 70。

以下均为 joint1−joint0，单位是百分点；场景区间为逐项 95% 区间，固定 panel 界按 bank 全候选联合范围计算。

| bank | 指标 | 差值 pp | 场景 95% 区间 pp | 全候选固定 panel 95% 界 pp |
| --- | --- | ---: | --- | --- |
| bank03 | pX | -0.651042 | [-1.302083, -0.130208] | [-15.390833, 14.088750] |
| bank03 | v | -0.390625 | [-0.911458, 0.000000] | [-12.708437, 12.317812] |
| bank03 | qX | -0.382506 | [-1.160729, 0.419083] | [-26.303850, 25.303444] |
| bank04 | pX | -0.260417 | [-0.781250, 0.000000] | [-14.637567, 14.116734] |
| bank04 | v | -0.390625 | [-1.171875, 0.000000] | [-11.355242, 10.964617] |
| bank04 | qX | 0.023901 | [-0.785340, 0.930203] | [-24.117606, 23.944413] |

bank03 的 pX 场景区间排除 0；两 bank 的 qX 场景区间均跨 0，表内所有联合固定 panel 界均跨 0。两个区间回答不同不确定性问题，不能替换为安全证书。零差 alias 的退化 bootstrap 继续保留 `UNINFORMATIVE/None`。两 bank 仍只有 seed17，结果为 `NOT_CERTIFIED`，不宣称训练收益或跨 seed 泛化。bank04 的 trend/SYMBOLIC_FRESH 中 qX 上升 0.745342 pp，同时 pX 不变、v 下降 2.343750 pp；不将条件概率单独上升解释成普遍改善。

训练 bank 计数分别为 bank03 X24/S0/W4/I4、bank04 X24/S0/W7/I1，与上表评估输出分开。baseline replay 的参数、Adam、优化器结构和 RNG/sampler 一致，`state_bitwise_equal=false` 仍按原值保留。

两项峰值 CUDA 已分配内存均为 24,173,737,472 bytes。bank03 完成调用耗时 14,074.849609 秒、40,124 次 forward；bank04 为 8,662.318898 秒、28,390 次。candidate elapsed 中位数分别为 685.112280 秒和 498.492096 秒。这些计时范围不同，bank 与采样工作量也不同，不能仅据耗时将差异归因于 GPU 节点。

完成上述复核后，两份推进决策分别绑定 bank03→bank05、bank04→bank11。bank05/11 脚本本地及服务器 `bash -n`、冻结 smoke/source 门禁与无模型 dry-run 均通过。确认没有活动 S1 作业、输出目录和提交 intent 不存在后，分别以独占 intent、实际 sbatch 回执提交；第二项提交前再次检查已有作业数，最多保持两个 S1 作业。

| bank | 新作业号 | Slurm UTC 开始 | 19:10:10 UTC 实查状态 | GPU 节点 / 物理索引 |
| --- | --- | --- | --- | --- |
| bank05 | 154573 | 19:09:36 | RUNNING | gpu-pro6000-7 / 0 |
| bank11 | 154574 | 19:09:38 | RUNNING | gpu-pro6000-13 / 2 |

两项均在 19:09:34 UTC 分别提交，约 2 秒和 4 秒后启动。bank05 启动脚本回执为 19:09:37，比 Slurm 分配开始晚 1 秒；bank11 回执为 19:09:38。相同特殊 QOS、各单张 PRO 6000、24 小时上限、no-requeue，未手填 CPU/内存；源码和环境仍冻结于 154599c。

截至该快照，已复核完成 bank00/03/04，累计 23 次 Adam、736 backward、4,608 条独立样本；bank05/11 运行中，唯一未提交项为 composite_no_x_plus_three_level。监控按用户要求保持每小时一次，有完成 bank 后先复核再提交 composite，全部 S1 完成后分析并暂停，S2 未启动。

| 证据 | SHA-256 |
| --- | --- |
| bank03 紧凑包 | `d612ddffbcff56a74f65587ed627392b7222b5a69d3ebf2796ec022756212142` |
| bank04 紧凑包 | `e818c2123ae4ae8e1c07ca8e9c6d0b785ea8680d2a4e5f58a6bb104e3845dc87` |
| bank03 原件及账本复核 | `63f0f5700988361ab57c9e3e0911720fbfb86e4a7b19576ae244e654d3f90b51` |
| bank04 原件及账本复核 | `be037376429b7efc331b48e929ce000510febef5364f91aa26afb22f3efa182c` |
| bank03 独立统计复核 | `a76b6bbf8c21511191fec7ec1969b470543d3b013c556f2df2158be858d2ff87` |
| bank04 独立统计复核 | `f772a25bc314ccbfaff38e55b325d28facb3b1fbeec0a7afdf73a8ef99eb2972` |
| bank03 推进决策 | `8e3c21ae4d7aeea6717fc51dfcce985d83ac4cba300550a96c0922c325140a1d` |
| bank04 推进决策 | `92a0de46aaddaf8ed5ad9d063cd0549a1f77f099473be2dd4f2cb8b124d4d704` |
| bank05 提交回执 | `394249e81b0cee890a19bd6768a5db95a9edfeb5c87c920c33441c7858ec8ac7` |
| bank11 提交回执 | `f01bd9c035ba9de4a841c2a3ffa47086933a72961fb4e50825086d4617a8db9f` |
| bank05/11 运行快照 | `7cc6bed711b65e59446e513ff5c9f24b56a8ffa58ebb4ddc38df0c377c8b3da9` |

## bank05 完成及连接中断记录

bank05 作业 154573 于 2026-09-14 21:47:49 UTC（新加坡时间 2026-09-15 05:47:49）正常结束，Slurm `COMPLETED`、退出码 `0:0`，墙钟 9,493 秒（2 小时 38 分 13 秒）。最终状态 `MEASURED / REAL_CUDA_FOLLOWUP`，`training_started=false`、`permanent_training_commit=false`。

服务器独立核验已完成并写入耐久回执：133 个 manifest 路径（20 个 checkpoint 路径），10 个实际 CPU checkpoint 均有限，Adam step64→65；10 reserved=10 confirmed、0 uncertain、320 backward。原始 bank 组装、输入绑定、请求与 seed、1,536 条 raw 语义重解析、逐题计数及完整策略指纹均通过。replay 完整状态仅候选名称元数据不同，其余状态一致；`state_bitwise_equal=false` 保留。模型身份沿用冻结绑定，本轮未重读模型权重或重新生成 prepared tensors。

本地紧凑包的 119 个普通文件及包含的 113 个非 checkpoint manifest 文件已经校验；20 个 checkpoint 路径仅在服务器保留并完成核验。随后下载完整服务器复核回执时发生 `Network is unreachable`，有界只读重试得到 `Operation timed out`，因此当时该回执的完整本地副本为 `PENDING_NETWORK`；2026-09-15 的恢复及核验记录见下节。本地空下载占位不是核验回执，不算成功传输。服务器已完成的核验与传输阻断分别记录。

| 策略 | 独立样本 | X | S | W | I | pX % | v % | qX % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_0 | 768 | 532 | 12 | 193 | 31 | 69.270833 | 95.963542 | 72.184532 |
| joint_1 | 768 | 529 | 12 | 196 | 31 | 68.880208 | 95.963542 | 71.777476 |

训练 bank 总计 X16/S0/W15/I1，与评估输出分开。每个独立候选仍为 48 prompts × 16；no_x_off_1 完整指纹匹配 joint0，复用样本且新增独立样本为 0。joint1 相比 joint0 的 X 减少 3，I 和 v 不变。

独立统计实现复算全部群体、5,000 次共享场景 bootstrap 和联合固定 panel 边界，结果 `RECOMPUTED_MATCH`、无不一致，最大差 2.22e−16。全候选和 pair 均为 M70。以下差值为 joint1−joint0，单位百分点。

| 指标 | 差值 pp | 场景 95% 区间 pp | 全候选固定 panel 95% 界 pp |
| --- | ---: | --- | --- |
| pX | -0.390625 | [-0.781250, 0.000000] | [-14.767776, 13.986526] |
| v | 0.000000 | UNINFORMATIVE / None | [-11.225034, 11.225034] |
| qX | -0.407056 | [-0.829876, 0.000000] | [-24.435585, 23.604943] |

pX、qX 的场景区间均包含 0，固定 panel 界跨 0；零差 v 保留退化 `UNINFORMATIVE/None`。结果为 `NOT_CERTIFIED`，不据此认定训练收益或跨 seed 泛化。

runtime profile：峰值 CUDA 已分配内存 24,173,737,472 bytes，完成调用耗时 8765.054710 秒，28,509 次 forward；范围与 Slurm 全墙钟不同。

该次连接中断前的最后快照为 21:57:41 UTC：bank11 作业 154574 仍在 gpu-pro6000-13 运行，已有 10 个候选标记、joint0 ledger 768 行，尚无 bank 完成或退出回执。随后当前状态因网络不可达而未知，未认定作业失败，也未停止或修改该作业。

composite_no_x_plus_three_level 的本地单卡脚本已生成并通过 bash -n，但远端准备 SSH 在连接阶段失败，未取得远端 dry-run 通过证据，也没有执行 sbatch。该 bank 当时为 NOT_SUBMITTED，不能记作第二张卡已申请。网络恢复后先取回并校验 bank05 已有完整回执，再完成最终推进决策、source/smoke/script/dry-run 和队列/intent 检查，单独申请一张 PRO 6000。无需重跑已完成 bank05 或重复昂贵验证，除非发现新的证据不一致。

截至该次连接中断时，累计四个 bank 已完成服务器实测：33 次 Adam、1,056 backward、6,144 条独立样本。监控仍为每小时一次；S1 尚未全部完成，S2 未启动。

| 证据 | SHA-256 |
| --- | --- |
| bank05 最终 manifest | `2211bef459c685afc727d5f0c561cd90b9367a34ec592357cbe2b20659ec8822` |
| bank05 紧凑包（已下载核验） | `291f34320ccd6413724d5ee66452fcf6a291cb1b8b62f15702b3de795f3273c3` |
| bank05 服务器核验回执（当时待下载；现已补齐） | `15a042b7987f33ae388ea1d1adaca3dd99308db8f74c5a189b722561a7c3f668` |
| bank05 独立统计复核（本地） | `7d64e7cff2cf3ae408997f20fe904a3cdcdd979b6b8c38a019e4f7ffbc9c2b14` |
| composite 本地单卡脚本 | `7b4e7a6908e12c3321e352536b78cd4912e2024a4687148c219b645203942970` |


## 2026-09-15 网络恢复、bank11 复核及最后一组启动

03:38:27 UTC（新加坡时间11:38:27）重新连接成功。bank11 作业154574已于2026-09-14 22:39:32 UTC（新加坡时间2026-09-15 06:39:32）结束，Slurm `COMPLETED`、退出码 `0:0`，墙钟12,594秒（3小时29分54秒）。恢复连接时无其他S1作业排队或运行。

bank05 原服务器完整复核回执成功下载，SHA-256与网络中断前记录一致。重新核对其本地统计来源和113个已下载manifest文件后，完成最终推进决策。没有重跑bank05，也没有重复昂贵checkpoint核验。

bank11完成原件及统计两项独立复核：133个manifest路径含20个checkpoint路径全部在服务器核验；10个实际CPU checkpoint均有限、Adam step64→65；10 reserved=10 confirmed、0 uncertain、320 backward。紧凑包119个普通文件通过安全及哈希检查，其中113个非checkpoint manifest文件本地匹配。1,536条原始样本的请求、seed、原始输入绑定、语义重解析、逐题计数及实际checkpoint完整策略指纹一致。joint0及其重放的完整状态仅 `metadata.candidate_spec.id` 不同，保留 `state_bitwise_equal=false`；参数、Adam和RNG分别一致。模型权重未重读、prepared tensors未重新生成，身份仍由原件/冻结计划/smoke和原控制输入绑定验证。

| 策略 | 独立样本 | X | S | W | I | pX % | v % | qX % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_0 | 768 | 529 | 12 | 186 | 41 | 68.880208 | 94.661458 | 72.764787 |
| joint_1 | 768 | 530 | 11 | 188 | 39 | 69.010417 | 94.921875 | 72.702332 |

训练bank为X20/S0/W9/I3，共32条，与评估样本分开。bank11 的 `no_x_off_1` 实际完整策略指纹匹配 **joint_1**，复用joint1的768条样本，新增独立样本为0；不能套用此前bank指向joint0的别名记录。

独立统计复算全部群体、5,000次共享场景bootstrap与联合固定panel界：`RECOMPUTED_MATCH`、无不一致；全候选与pair均M70。概率/界最大比较误差1.11e−16，百分点差值最大误差1.11e−14。下表为joint1−joint0，单位百分点。

| 指标 | 差值 pp | 场景95%区间 pp | 全候选固定panel 95%界 pp |
| --- | ---: | --- | --- |
| pX | +0.130208 | [0.000000, 0.390625] | [-14.246942, 14.507359] |
| v | +0.260417 | [≈0, 0.781250] | [-12.266700, 12.527117] |
| qX | -0.062455 | [-0.605889, 0.401070] | [-25.140862, 25.161355] |

v场景区间下界原值为−1.11e−16（概率单位），表中记为≈0；保留原始浮点值。pX和v上升、qX略降；场景区间达或跨0，联合固定界均跨0。no_x_off_1与joint1复用同一组输出，三项差值全0，其退化bootstrap保留 `UNINFORMATIVE/None`。结果仍为 `NOT_CERTIFIED`，不据此宣称训练收益或跨seed泛化。

几何记录显示joint_1em6、joint_1em5相对joint0的参数差为0；joint_1em2与joint1更新向量cosine=0.19299857882930452、relative_vector_difference=13.794437498897304，不能将前几bank的近似相同更新描述套用到此处。这里报告产物中的几何值，不将它等同于模型输出变化或因果结论。

runtime profile峰值CUDA已分配内存24,173,737,472 bytes，完成调用耗时11121.330464秒、28,596次forward；Slurm墙钟为12,594秒。日志保留generation min-length警告，stdout为 `MEASURED`，未发现Traceback、CUDA OOM、RuntimeError或AssertionError。

### 最后一个composite bank

bank05最终复核完成后，重新核对73个冻结源码、计划和真实smoke；远端脚本 `bash -n` 与不加载模型的dry-run均通过。确认目标输出、submission intent和提交回执不存在后，先独占写入intent，再单次提交。作业 **154797** 于2026-09-15 03:41:41 UTC（新加坡时间11:41:41）开始，03:42:19 UTC实查 `RUNNING`，节点gpu-pro6000-5、物理GPU索引5。使用老师QOS `soujanya-poria-startfund-2026-03`、partition cluster02、account rose；一张PRO6000、no-requeue、24小时上限。CPU/内存由调度器自动分配，未手填资源值。初次启动快照尚未出现bank输出目录或退出回执，该快照仅证明Slurm运行与启动回执，不能作为测量完成证据。 后续03:49:53 UTC检查仍为RUNNING，已运行8分12秒，stdout/stderr均0 bytes，仍无bank输出目录和退出回执；未据此认定失败或测量完成。

本bank固定3个候选+1次replay，最多4次Adam、128 backward、2,304条新输出。目前仅剩这一独立bank，未额外申请闲置GPU。五个已完成bank累计43次Adam、1,376 backward、7,680条独立样本；全S1原上限47/1,504/13,824不变。

下一步按每小时检查作业154797；完成后复核其真实结果，再运行冻结 `followup_cli analyze` 核验六bank整体技术测量链。S1整体尚未完成，S2正式训练未启动。

| 证据 | SHA-256 |
| --- | --- |
| bank05最终推进决策 | `54f826d37f154a82666f7500fa53c4becff47251c28287881b8e87ac296022d9` |
| bank11最终manifest | `ac6382e4c3bfacce61de2e38e7c53f1bedbd80a15d7cda8a60c0e632ede9e06b` |
| bank11紧凑包 | `c78ca0818df6eda0febbdf173291bf4456ca858969bf37238a44117b31663f7e` |
| bank11完整原件及账本复核 | `61d69392da026ca664114fd9b429e6eab669b354a9bf1d7e5d34c2e8c7cf5e45` |
| bank11独立统计复核 | `c80e3c7ea17c20f3666030bbd4f1224031928db9b10a52b7998b0c4340f57904` |
| bank11最终复核决策 | `20efea1a78b73cfa46a53572869e3e4aadef65387e1f718270e0f69f1ccbb698` |
| composite远端准备回执 | `7ba8fd65b95242025b2de13f93342d097b85b1f45c0218e0dd78f6b16ea94506` |
| composite提交回执 | `9326414331150bafdf9933858cc4c6f8e8b7af552499d3f6693ae342313eb038` |
| composite初次RUNNING快照 | `febf3800407100bf9e10134d4cc60fa4553c0d70a4c5d1a706cf8e40a165f8b1` |
| composite后续启动检查03:49:53 UTC | `519fe6d9ebc17c48379075ea9e1f29217102670ab352ab75d758fdfd12cc428f` |


## 2026-09-15 独立历史 R2 并行补测

用户进一步要求有可同步进行的工作就现在并行安排。逐项核对冻结154599c的CLI、训练器、backend和设计：七个新S2 run全部要求完整六bank S1 measurement chain；当前composite尚未完成，没有可提前启动的S2训练例外。代码会重建各bank的manifest与共同身份，不能用部分S1报告或手写PASS替代。

计划内旧seed17、step64端点的 `eval-historical-r2` 经同一exact smoke gate后直接进入backend，不读取新S1 chain。它只读原R4 X_BASE/X_VALID检查点并做R2诊断；每臂72 scenes×3条件×4=864条输出，两个端点合计最多1,728条，0次Adam、0条训练输出。该预算原已列入设计，未增加seed、模型或样本量。两个历史端点彼此独立，输出也不依赖新composite；各自使用独立进程、单GPU和新目录。

03:54 UTC只读确认当前ROOT尚无 `historical_R2/`、对应submission intent/回执或新S2目录。提交前核验73源码/exact plan/smoke、原R4 checkpoint manifest及两臂step64 checkpoint字节哈希；两份独立脚本本地和远端 `bash -n`、无模型dry-run均通过。dry-run不代表模型测量；实际CLI仍执行完整source/data/model/env/parent重验证、成熟Adam和参数/状态/采样器绑定检查。

先写独占intent，再单次提交X_BASE。作业 **154821** 于2026-09-15 03:57:33 UTC（新加坡时间11:57:33）开始，03:58:01 UTC实查与原S1作业同时为RUNNING：

| 工作 | 作业号 | 状态 | GPU节点 / 物理索引 | GPU数 |
| --- | --- | --- | --- | ---: |
| S1 composite_no_x_plus_three_level | 154797 | RUNNING | gpu-pro6000-5 / 5 | 1 |
| 历史R2 seed17 X_BASE step64 | 154821 | RUNNING | gpu-pro6000-2 / 0 | 1 |

两者均使用老师QOS `soujanya-poria-startfund-2026-03`、partition cluster02、account rose，24小时墙钟上限、no-requeue；未手填CPU/内存。新作业启动回执03:57:34 UTC；初次并行快照两项目标输出目录尚未出现，尚不能声称已产出测量或完成训练。

历史输出为私有ROOT下 `historical_R2/X_BASE`；旧R4 checkpoint和当前S1源码/脚本未修改。X_VALID的独立单卡wrapper曾生成并核验，但尚未提交。此前计划在空位出现后提交X_VALID；该计划已被用户随后要求停止新实验的指令覆盖，现不再有效。准备脚本不属于已执行实验。

历史R2完成验收要求 `PASS / REAL_CUDA_HISTORICAL_R2`、864个唯一输出、0 optimizer_updates/0 training_outputs、training_started=false、full_endpoint_state_unchanged=true；同时核对completion manifest/binding、identity/request manifest、raw ledger、原端点哈希、三条件及family×condition计数、完整状态恢复和日志。结果继续保留NOT_CERTIFIED，与N/OOD主指标分开，不能把补测称为新增S2训练。

最新收尾安排：每小时检查154797和154821的状态，仅完成现有作业和已有结果复核；S1完成后运行CPU整体分析。X_VALID不提交，正式S2训练不启动。详细结果文档核验交付后暂停监控。

| 证据 | SHA-256 |
| --- | --- |
| 并行依赖审查 | `f8f0c1cf312dc95f9e364383fd729610eceabdc980e00248e1036bbd2233395f` |
| 历史R2准备回执 | `e06a62e267ed202a2699c20cffced0144eec9c6a7fa5060feb241330a4978bf8` |
| X_BASE单卡脚本 | `8cfae42480a9a3c5bc46533c0efb76fa48782e53e3a22938fdcf542a2027e482` |
| X_VALID单卡脚本（未提交） | `11b755abad76e70ffccdf4ae874a884b9702c3bb254c5581abf2c06c1dae4925` |
| X_BASE提交回执 | `e5ef3865af7acec1f47c87561587e0cd14967136ac5b0b0336ca59ae30a59815` |
| 两作业并行快照 | `87d25ec3bee9053702383d6e31bcb9c66a7397bf9d2068ef0b9a3f33d3a5c3cf` |

## 2026-09-15 用户收尾边界与历史 X_BASE 完成状态

用户明确要求“这两个文件任务完成之后就不再做新的实验了，而是将这次完成的所有实验的结果详细信息整理成一个文档给我即可”。当前两项对应154797、154821，不取消它们已提交的工作；不再新增GPU实验、历史X_VALID补测、S2、seed或模型。空闲卡不再补位。该指令优先于旧设计中的后续执行模板和本文较早安排。

05:59:06 UTC实查154797仍RUNNING、尚无completed；154821已COMPLETED、exit0，03:57:33至05:09:03 UTC的Slurm时长4,290秒，脚本开始回执03:57:34 UTC至退出的时长4,289秒。其保存结果为PASS / REAL_CUDA_HISTORICAL_R2、864条评估输出、0次更新、0条训练输出，training_started=false、full_endpoint_state_unchanged=true，保留NOT_CERTIFIED；后者是GPU运行保存的检查结果，尚须区分CPU独立复核可以确认的事实。

已取回紧凑包并独立复算：SYM_ORIGINAL的X/S/W/I为170/1/103/14；IMAGE_CUE为278/2/7/1；IMAGE_ONLY为288/0/0/0。每条件288条。原计数全部一致；未新增区间或M。历史R2与S1及N/OOD统计分开。原始输入、parser、checkpoint和完整绑定由另一路只读CPU复核。

本地与服务器同名控制回执 `execution_scope_stop_after_current_20260915.json` 已核对相同SHA-256：`a13f753b9a0ccab3590d397903bb2c3d65ce3575bdcbee510836f408ede678e1`。既有每小时自动任务ssvc-5090已改为“SSVC现有任务收尾与完整报告”，明确禁止新实验。它在所有现有结果核验、单份详细中文文档生成并交付后暂停；当前仍有154797未完成，不能宣称最终文档已交付。

最终文档范围是本轮六个S1 bank的实际状态与结果、已执行的历史X_BASE R2、CPU/真实smoke及失败尝试记录；旧R3/R4仅作输入背景。文档保留所有分组、策略别名、样本/更新预算、区间与不确定性、几何及恢复检查、运行时间和来源哈希，不补写未执行数据或因果结论。

### 历史 X_BASE 独立复核完成

只读CPU复核已完成：73个冻结源码、5个completion manifest文件及3项完成binding一致；原R4实际step64 checkpoint的文件/state/参数/Adam哈希、成熟step64与2,048个训练sample keys吻合，状态全部有限。基于原R4 core state和已有S1 forward metadata在CPU重建本次origin与policy hash，与保存请求身份吻合；未重新捕获GPU模型状态。

从原calibration和72张图像核对216 prompts及864条固定seed/sample key。每prompt全部5条旧R2输入支持记录（合计1,080条）的final prompt、token、tensor/pixel哈希一致。864条原始记录全部经CPU tokenizer decode、EOS/有限logprob和semantic reparse复核；停止原因均为EOS，共13,499个behavior tokens。prepared tensors未重新生成，模型权重分片未重新加载或逐片重算哈希。

| 条件 | n | X / S / W / I | pX % | v % | qX % |
| --- | ---: | --- | ---: | ---: | ---: |
| SYM_ORIGINAL | 288 | 170 / 1 / 103 / 14 | 59.027778 | 95.138889 | 62.043796 |
| IMAGE_CUE | 288 | 278 / 2 / 7 / 1 | 96.527778 | 99.652778 | 96.864111 |
| IMAGE_ONLY | 288 | 288 / 0 / 0 / 0 | 100 | 100 | 100 |

上表概率为已有计数的独立派生点值，pX=X/n、v=(X+S+W)/n、qX=X/(X+S+W)；原程序只保存family×condition计数，不包含这些新派生点值的置信区间。本次未产生新CI、M或跨臂效果。IMAGE_ONLY的全X仅为本次观察，保留NOT_CERTIFIED。三种条件分开，不并入N/OOD或S1。样本elapsed字段合计1,922.3449183967896秒，与Slurm4,290秒和shell4,289秒分别报告。

原raw的864条phase均为S2，这是共用request builder的标签；identity.phase为S2_HISTORICAL_R2、role为historical_endpoint/R2、执行类型为REAL_CUDA_HISTORICAL_R2。保存结果为0更新/0训练输出，不能据phase标签称为新增S2训练。full_endpoint_state_unchanged=true由保存的完整结果及冻结代码检查支持；没有单独的post-state artifact，CPU复核未独立重测评估后的GPU状态。

归档含12个普通文件、488,037 bytes，全部5个completion manifest文件安全下载并核验；原端点PT不在本地归档，已在服务器CPU核验。下列回执将其字节身份与结论绑定：

| 文件 | SHA-256 |
| --- | --- |
| historical_r2_X_BASE_review_bundle_154821.tar.gz | `1000107dc54f310804f0c10845fc6ce99cd293b9cc0ef87b132e491a1de60936` |
| completion_manifest.json | `b1082fdecd35b51806af65b6c866cc05db109ec0444a9155749b7ba8fc0a7982` |
| archive_verification.json | `710f1cedc9f07355ef4e48de42510515a0b0055330390f9e385a96c716b90b37` |
| historical_r2_X_BASE_integrity_154821.json | `ca5b9bc5c28d0bcff3b8cb1bf43ee85691003377efb169c593fda7c43dd819bc` |
| statistics_review.json | `86b61afbb06ed4c580ae4bef85beee79fe218b40c12e43deb4f08422ebc95de8` |
| final_review_decision.json | `af0b16dddc18bd77589df2847daf4ad073fd8c24cedf32989c1a5440f5f85442` |

最终复核状态为COMPLETED_REVIEWED_NO_SUCCESSOR_BY_USER。仍仅等待现有composite154797，完成已有结果CPU分析和最终文档；不提交任何新实验。
