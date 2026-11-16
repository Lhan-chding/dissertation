# 11031/T0/step2 DEV 实际核验

作业1741834已COMPLETED/0:0，所有子步骤退出0，耗时01:05:08，108条BEGIN/CAPTURE/COMPLETE齐全。2026-11-16 13:10 UTC恢复VPN后实测。原TRAIN84文件及19键身份、真实step2 checkpoint、实际canary、原始生成、进程关闭及零优化器更新经独立CPU审核；完整POST_RISK复算一致，global_hold=false，仍待另外三组DEV2。该结果不授权step3或MAIN，也不是训练有效性的科学结论。

| 指标 | 本次DEV2 | 同起点START DEV |
|---|---:|---:|
| 冲突S_hit |15/32|16/32|
| 对齐与无列表S_hit |60/64|59/64|
| 保留S_hit |12/12|12/12|
| 全部S_hit |87/108|87/108|

本次A=85/108，native J总分81/108；108条均balanced/json_valid/S_known，零UNKNOWN。逐行记录及所有分组保留在ANALYSIS.json，不据单个检查点宣称增益。

远端审核于13:13:26 UTC完成，AUDIT_REPORT SHA eb989d200905a8ccd5647c494456c4374d063affe0ca00b334af41dda0d32f22；ANALYSIS SHA a6a6697511b82a0095f68d2aaab222bbc32878615bb43699e94481e90c903f5c；原POST_RISK与RISK_REPLAY均833c87194cb5c810932c4165dba1082200ff40c0ee3ad329528d6099a042baf4。审核没有新增模型调用、训练或生成。

首次下载在RISK_REPLAY处出现大小/哈希不匹配而停止，原LOCAL_ERROR及已成功下载的8文件完整保留在independent_audit。首次不一致的具体原因未确定，当时不匹配的传输字节未被旧下载实现保存，不能据此断言远端文件损坏。后续只读重查确认10份远端文件全部与原manifest一致。

使用独立审核后的只读恢复工具，从原manifest SHA faadc50d39edaf4d08199df8e8007cee9aea46b10adad6f0823bdacb0fd4e25b恢复全部10文件；新目录independent_audit_download_recovery留存完整SSH原始输出、精确文件、交叉引用及恢复回执。RECOVERY_RECEIPT SHA 08c35c26038a4b290363a13a0adeb42c72c48e4272bb967187631c13e0214f32。所有字节/哈希、完整JSON及风险对象核验通过；恢复未重跑审核或复算外部证据，仅恢复已经完成的原审核包。

后续仍为11031/T1/2、11032/T0/2、11032/T1/2，每组108条；各自实际核验后再判定下一阶段。
