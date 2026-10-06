# 2026-10-06 16:39：已授权恢复，等待 VPN 登录

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
