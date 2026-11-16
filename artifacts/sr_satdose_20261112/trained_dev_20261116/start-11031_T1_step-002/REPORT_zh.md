# 11031/T1/step2 DEV 实际核验

作业1742113已COMPLETED/0:0，全部子步骤退出0，耗时00:33:08，108条BEGIN/CAPTURE/COMPLETE齐全。2026-11-16 15:58 UTC小时巡检确认完成。原TRAIN84/19身份、实际checkpoint和canary、108条捕获、进程关闭及零更新经独立CPU审核；完整POST_RISK复算一致，未触发HOLD。仍须完成11032的两组DEV2，不能从本次核验直接进入step3或MAIN。

| 指标 | 本次T1 DEV2 | 同起点START DEV |
|---|---:|---:|
| 冲突S_hit |18/32|16/32|
| 对齐及无列表S_hit |61/64|59/64|
| 保留S_hit |12/12|12/12|
| 全部S_hit |91/108|87/108|

本次A=90/108，native J总分81/108；108条均balanced/json_valid/S_known，零UNKNOWN。这是一个起点、一个固定检查点的观测，不代表整体训练有效性。全部108行及分组结果保留。

AUDIT_REPORT SHA 5eafcb8d911f4eb94efc893bfbbacbd5c7a1443888b9f453eaeb3eb36ad19c3c；ANALYSIS SHA d616d641c9c70bb5efa9cc358b333cc8c3a199b549654a36da06f1ed64977348；原POST_RISK及RISK_REPLAY均093caebcfe280423d2bcc8145ac7d058c767bcfc91271d76a457dc89cca70a14。

首次逐文件下载再次在RISK_REPLAY处出现哈希/大小不匹配，旧LOCAL_ERROR和8份已匹配文件保留。使用已审c168dcfc只读工具，从原manifest SHA 79703bb3aed27557dd1c17168cc8d96a228a476de11852165065a047e37753c8，一次整包恢复全部10文件。新independent_audit_download_recovery目录保存完整传输字节及RECOVERY_RECEIPT（SHA80c92a1176cfae42193c0495800a5b625d033ee4a1f9e391e9e5da1e778385a2），所有文件与原manifest匹配、两份风险文件逐字节一致。恢复不重跑审核、模型或风险计算；旧传输错误的具体原因仍未确定，首次坏字节未被旧下载实现保存。

下一步是11032/T0/step2的独立实际准备；四路径风险门禁保持原口径。
