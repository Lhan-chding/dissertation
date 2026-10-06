# 最新恢复状态：2026-10-06 17:06，续跑已产生增量

网络恢复后已完成原五项的显式retry及五个单PRO6000 worker提交：187958–187962。17:06:33核验为1 RUNNING、4 PENDING（Resources/Priority）；S96 teacher原子回答从故障时533条增至575条，chunk从13增至14。队列67 COMPLETE、1 RUNNING、226 PENDING；已完成项没有重做。R0第27步完整状态哈希再次通过，但R0尚未获得worker续跑。E/G仍封存，故障原因UNKNOWN。详见 [恢复与续跑核验](RECOVERY_20261006_zh.md)。

三小时监控已更新为新worker；五卡是并发上限，目前实际一张。恢复后15–24小时的规划估计以五卡资源到位为条件，当前排队和资源不足会延长用时。

---

# 恢复状态：2026-10-06 16:39，等待 VPN 登录

用户已授权原五个中断项恢复；旧进程、独占锁、冻结身份和存储探针均已核验。随后本机VPN断开，恢复SSH在建连前失败，因此尚未retry或提交新作业。GlobalProtect已打开密码登录页等待用户完成认证；现有三小时监控已保留本次恢复授权。详见 [恢复记录与续行范围](RECOVERY_20261006_zh.md)。

---

# 监控：2026-10-06 16:24，五个worker已停止

五个Slurm作业于16:18:35–36以FAILED/1:0结束；原因尚未确认。67个工作项完成，5个残留RUNNING不代表仍在计算。6个SFT臂已完成，R0第27步恢复状态已核验。未重提，E/G未揭晓。详见 [故障与恢复依据](MONITOR_20261006_1624_zh.md)。以下为此前快照。

---

# 当前进度：正确 QOS 下已并行启动

截至 2026-10-06 12:37:24（新加坡），已使用账户授权的 `soujanya-poria-startfund-2026-03`，其 MaxSubmitJobsPerUser=5（运行＋排队）、MaxJobsPerUser 未设置。此前只查默认 rose 导致误报配额阻塞，此处更正。

- `187023`：错误默认 QOS 下排队，零运行时间取消。
- `187045`：正确 QOS 下完成 v1 两父真实桥接，5分41秒，Slurm 0:0。
- `187112`：完成 v2 门禁复核与固定训练题数值定位，5分39秒，Slurm 0:0。
- 正式 worker `187115 / 187116 / 187154 / 187155` 已运行；`187156` 等待 Resources。已提交五个单 GPU worker，当前实际四张 PRO6000，不超过五张上限。
- 两个 teacher 与两个 S96 GOLD_ALL repeat 正在并行。快照记录 782 条 V 原子回答、51 次正式 SFT 更新；294 个工作项中 4 RUNNING、290 PENDING，无 BLOCKED_TECHNICAL 项。快照之后计数会继续增加。
- 原始 BF16 差异报告保留；两父因果/PAD与精确恢复通过，受控精度诊断及独立审阅后接受固定共同 SFT 路径。详见 [数值审阅](NUMERICAL_REVIEW_zh.md)。不声称旧 prefix 等价或科学效果成立。

当前正式运行目录为服务器 `verified_discovery_transfer_20261006/run_v2`，代码 `code_v2/ssvc_flow`，代码提交 `ed1d248`。新旧 cohort 清单字节一致；旧 run_v1 没有正式科学模型调用，只保留技术桥接证据。v2 在任何 teacher 回答前已重新冻结源码和同一预登记矩阵。

[实时阶段快照](evidence/run_v2_snapshot/SCHEDULER_RECEIPT.json)、[执行范围](evidence/run_v2_snapshot/EXECUTION_SCOPE.json)、[训练登记表](evidence/run_v2_snapshot/TRAINING_REGISTRY.json)、[生成与训练成本](evidence/run_v2_snapshot/EXECUTION_COSTS.json)。E/G 仍封存，实验未完成，尚未生成最终发现/迁移/保持性结论；未新增定时监控。

以下保留较早的准备与排队快照，属于历史记录。

---

# QOS 更正及当前进度

用户指出 soujanya QOS 后，实时核实账户具备 `soujanya-poria-startfund-2026-03` 权限；其 MaxSubmitJobsPerUser=5，MaxJobsPerUser 未设置。集群禁止原地更改 QOS，故将尚未启动的 187023 取消，并重提同一桥接为 **187045**。新作业已在 `gpu-pro6000-6` 使用 1 张 PRO6000 完成，Slurm 0:0，耗时 5分41秒。两父均 future/PAD delta=0、resume_exact=true，但数值门槛为 NUMERICAL_REVIEW_REQUIRED；正式训练未放行。原始回执见 [bridge_v1](evidence/bridge_v1/bridge.json)。以下保留先前排队阶段的历史快照，不能当作当前调度状态。

# 第一阶段事实记录：等待 GPU 调度

截至 2026-10-06 12:10（新加坡时间），本轮尚未产生真实 Qwen bridge、teacher、SFT、R0 或 E/G 结果。Slurm 作业 **187023** 已提交，状态 **PENDING / QOSMaxJobsPerUserLimit**。`rose` QOS 每用户最多同时运行 2 个作业、最多提交 5 个作业；当前两个运行名额被既有作业占用，未修改它们。排队时间不计作 GPU 执行或吞吐。

## 已完成且有证据的工作

- 从给定 `START_HERE_FOR_CODEX.md` 开始，原 ZIP 的 24 个文件按字节保留于 `design/`；隔离工作区、旧流水线 STOP 和用户原有改动保留。
- 清理三个旧 pytest fixture 根下可重新生成的白名单文件：103,723 个，逻辑大小 5,632,339,970 字节（5.63 GB）。删除前逐文件哈希登记，删除后剩余目标为 0。真实实验原始结果、唯一检查点、失败日志、验收汇总、源码和共享 inode 保留。完整删除清单在服务器项目的 `evidence/`，本地见 [清理回执](evidence/CLEANUP_RECEIPT.json)。
- S96、REP96 均核对 COMMIT/manifest、checkpoint 文件 SHA-256 与完整 CPU 状态哈希；每父 192 个有限、名称/形状/dtype 相符的 LoRA 张量。此项不替代 GPU 恢复及前向验证，见 [父状态核验](evidence/PARENT_VERIFICATION.json)。
- 新 T384/V96/E384 与 G96、R64 已生成并冻结。T/V/E 来自 0–99 新数值域；G 含 64 个 E 配对图像接口和 32 个独立 duplicate；R 为历史真实 O0 成功训练输出的 32/16/16 分层选择。
- 历史排除读取 84 份现存元数据，登记 3,323 个真值轨道、3,822 个 scene ID；29 个数据根绑定归并为 6 个现存根。已删除、外部或只在未审计归档内的来源不能被这一核验覆盖，范围见 [历史输入说明](evidence/historical_inputs/README_zh.md)。
- 48 项针对性 CPU 测试通过，附带参考脚本 53 项测试通过；新增/修改代码 Ruff 与 Slurm 脚本语法检查通过。mask、shift、梯度累积和崩溃恢复的 CPU 结果不称作真实 Qwen 验收。
- 冻结注册 294 个工作项，其中 SFT 20 个、R0_RESET32 4 个。当前全部 PENDING；矩阵登记不是 294 个 Slurm 提交。见 [注册矩阵](evidence/REGISTERED_MATRIX.json)、[执行范围](evidence/EXECUTION_SCOPE.json)。

## 已实现的实验边界

SELF_O0 / SELF_MIX / SELF_SINGLE 每题 16 次调用，同一 completion-only SFT 设置、256 次更新、12 focus + 4 replay。SELF 只接受原始回答经公开验证得到的目标，失败不补求解器标签；GOLD 权限单列。相同 parent/repeat/完整设置下严格相同训练 view 才别名复用；空发现集合返回父模型并保留结果。SFT 不使用旧 PPO ratio。

角色加载器约束 public、gold、trainer、selector、final_eval 的读取；E/G 在登记矩阵终态与完整性核验后才统一揭晓。该软件边界不是同一 Unix 用户下抵御恶意代码的操作系统沙箱。训练、发现、原始负结果、物理重算和技术失败均保留。最终分析分别报告发现、拟合、O0 迁移和 G 保持性。

## 真实 GPU 验收及未完成项

作业 187023 申请 1 张 PRO6000，调度器实际资源为 4 CPU / 33 GiB。它按顺序对 S96、REP96 做 8 个固定训练题的目标 mask、EOS、因果性、PAD、单例/批量 loss、microbatch 梯度、prefix/full 梯度和两步精确恢复验证。BF16 差异必须依据真实诊断审阅，不能用 CPU PASS 或任意放宽阈值替代。

目前 GPU 恢复、bridge、teacher 发现、GOLD/SELF SFT、R0、V 诊断及 E/G 评估均未完成，吞吐未知，无科学结论。正式 worker 尚未提交；bridge 完成也不自动放行数值审阅。最多 5 卡是上限，不是已获得资源。当前 QOS 限制下不能保证五个单 GPU 作业并行。

继续点：读取 187023 的 Slurm 状态、输出和 `run_v1/bridge.json`，先处理真实技术诊断，再启动已登记队列。不要重建新数据、修改冻结源代码或重复提交已有 bridge。若需技术修复，保留旧 run 与失败证据，明确新代码版本及身份绑定。未增加定时监控。

## 代码及路径

代码提交：`6902fe8fa7634a819ac9dd9a4cfdb537d27559b7`，远端分支 `codex/verified-discovery-transfer-20261006` 已核验。

服务器项目：`/projects/varunssd/louis-ssvc/verified_discovery_transfer_20261006`。
正式源码：`code_v1/ssvc_flow`；冻结运行目录：`run_v1`；配置：`machine.server.json`。
运行身份：`4de8c0c0358cf73617414b03da6e15304c6d03629057ff3a24893c43ef5e44fd`。
新 cohort digest：`ad1d1a7dc193ed0ae9e35d5a9c17e473c35f565d8e6e9a33e34eded3176e15af`。
矩阵身份：`717f161def098ac587140c4578bfc790544dccc76af8f7d3da2411170c599e5d`。

实际命令见 [SERVER_COMMANDS.md](SERVER_COMMANDS.md)。未生成声称完整实验完成的 FINAL_FINDINGS。
