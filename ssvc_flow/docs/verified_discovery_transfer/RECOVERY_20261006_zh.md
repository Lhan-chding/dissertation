# 2026-10-06 17:06：恢复已执行，核验到新增回答

用户确认网络恢复后，再次核验原五个Slurm作业FAILED/1:0、原五个队列残留RUNNING、无既有retry事件、五lane锁可用、冻结身份与桥接接受有效。重复1 MiB写读/fsync/原子重命名探针通过。R0 step027再次通过完整状态哈希；文件SHA256为 `5f1a87001974c4ad0f3a332caab27cbc0c1888e548c7a020c934cd0fd55ff939`。

17:04:37开始执行已授权恢复：备份旧queue.sqlite，复制保留15个零字节临时文件；用已有Queue.retry为原五个中断项写入EXPLICIT_RETRY与原始行。67个COMPLETE逐行比对不变，重置后227 PENDING。没有更改源码、科学设置、矩阵或E/G揭晓状态。

| lane | 新Slurm ID | 17:06:33状态 |
|---|---:|---|
| v2-lane0 | 187958 | RUNNING，gpu-pro6000-10 |
| v2-lane1 | 187959 | PENDING，Resources |
| v2-lane2 | 187960 | PENDING，Priority |
| v2-lane3 | 187961 | PENDING，Priority |
| v2-lane4 | 187962 | PENDING，Priority |

全部使用soujanya-poria-startfund-2026-03、单PRO6000、4 CPU、33G及原3天时限。其他项目的rose作业187922未改动。现有三小时监控已切换新ID；本次恢复授权已执行，不能重复提交。

17:06:33队列为67 COMPLETE、1 RUNNING、226 PENDING。S96 teacher.T_train.0.96已发布原子回答从533增至575，chunk从13增至14，确认实际续跑；新worker日志未匹配到Traceback/Error/Exception/Killed/No space。REP96 teacher、两个中断评估和R0尚待worker领取；不得将已提交五个worker写为五卡都在计算。R0将按冻结运行器选取step027继续，当前尚未证明第28步已执行。

原同时失败原因仍为UNKNOWN；新worker再次由调度器分到gpu-pro6000-10，本次观察到写入成功不证明根因已解决。E/G仅统计文件数量，科学结果未读取。实验尚未完成，恢复不是科学结论。

回执：[显式恢复与提交](evidence/recovery_20261006T090437Z.json)、[实际续跑核验](evidence/resume_check_20261006T090633Z.json)。服务器完整操作证据在 `run_v2/recovery_evidence/20261006T090437Z/`；旧队列备份和临时文件副本保留在服务器。本地恢复回执SHA256与服务器核验值为 `4c7a69c1fb37a6ba1fb4558f31627e93eef477c92b4e89c217a7ac52df5bf0b1`。

---

# 2026-10-06 16:39：已授权恢复，等待 VPN 登录（历史）

用户在五个 worker 故障报告后要求想办法恢复。此次授权仅用于原冻结矩阵的五个中断项续跑，不改变数据、预算、SFT 设置或科学源码。

## 已完成的恢复前核查

08:31:54 UTC（新加坡16:31:54）通过 SSH 核实：旧作业187115、187116、187154、187155、187156全部 FAILED / 1:0；squeue 中没有该用户活动作业；v2-lane0至4的五把独占锁均可取得并释放。冻结源码/协议/机器配置与桥接接受门禁核查通过。

在 run_v2/tmp 下执行唯一命名的1 MiB临时探针：写入、fsync、读取、SHA校验、原子重命名及再次读取校验通过；仅清除该探针自身文件。共享盘约198 TB可用。此测试不证明用户配额或存储长期稳定；旧故障原因仍为 UNKNOWN。

此前核验的R0 step027完整状态哈希、原子回答及67个已完成工作项证据见 [故障监控记录](MONITOR_20261006_1624_zh.md)。不读取E/G科学结果。

## 本次恢复尚未执行

随后准备执行的 SSH 连接在建连阶段返回 `Network is unreachable`，两次只读连接验证返回 `Operation timed out`。因此远端恢复脚本没有开始，尚未执行 Queue.retry、备份操作或 sbatch，没有新的Slurm作业ID。不能将发送过命令当作恢复成功。

本机GlobalProtect显示未连接；集群IP走en0普通网络路由。已点击现有NTU门户的连接按钮并选择已有NTU帐户，界面停在Microsoft密码登录页，等待用户在GlobalProtect中自行输入密码并完成可能的MFA。未读取或收集密码，未更改网络安全配置。

## 连接恢复后的唯一续行范围

1. 重新核实旧作业终止、无已提交恢复作业、原五个残留RUNNING、五lane锁可用、存储及冻结身份有效。
2. 保留故障与临时文件证据，使用已有Queue.retry或cli retry，为下列五项写入显式原因；不手工更新SQLite，不重做67个已完成项：
   - teacher.S96.T_train.0.96
   - teacher.REP96.T_train.0.144
   - r0.S96.0
   - eval.S96.1.GOLD_ALL.E_test.256
   - eval.S96.1.REPLAY_ONLY.V_selection.128
3. 用同一code_v2/run_v2/machine.v2.json及soujanya-poria-startfund-2026-03 QOS恢复最多五个单PRO6000 worker。复用已发布原子回答，R0从已核验的step027继续，不能因LAST_COMMIT指向26而重复第27步。
4. 实际核验续跑增量后记录新Slurm ID，并更新现有三小时监控。新故障不属于自动盲目重提授权。

现有ssvc三小时heartbeat已保留此一次恢复授权和网络阻塞状态；无需重复询问恢复许可。此前恢复后15–24小时估计仍为条件估计，当前停机与排队另算。
