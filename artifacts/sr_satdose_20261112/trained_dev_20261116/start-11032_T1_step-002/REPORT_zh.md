# 11032/T1/step2 DEV 实际核验

2026-11-17 15:14（新加坡时间）小时巡检确认作业1742662 COMPLETED/0:0，所有Slurm子步骤退出0，耗时00:48:18。108条BEGIN/CAPTURE/COMPLETE齐全。原审计器7c29df29于07:18:02UTC完成实际CPU独立核验，原TRAIN84/19、实际checkpoint/canary、捕获完整性、关闭和零参数更新均经检查，完整POST_RISK复算一致，无HOLD，pending_required_dev为空。

| 指标 | 本次T1 DEV2 | 同起点START DEV |
|---|---:|---:|
| 冲突S_hit |20/32|14/32|
| 对齐及无列表S_hit |63/64|63/64|
| 保留S_hit |12/12|12/12|
| 全部S_hit |95/108|89/108|

本次A=94/108，native J总分89/108；108条均balanced/json_valid/S_known，零UNKNOWN。四组DEV2全部完成是后续固定训练段的前置条件，不代表最终有效性，也不准许使用DEV选择检查点或改变科学参数。11032/T0的一条长度截断UNKNOWN继续保留在其原始报告和固定分母中。

驱动a9c4dae3只执行一次原始审计，再进行一次c168只读整包下载。全部10份原始文件与manifest大小及SHA一致；ORIGINAL_POST_RISK与RISK_REPLAY逐字节相同。源代码、原始SSH字节、收据及完整逐行评分保存在`independent_audit_bundle`。

- AUDIT_REPORT SHA：7fcd7b2fd57af415b487c99297457142af7ccb9d5417489ddf3292ded30a782b
- ANALYSIS SHA：ca2de1511949d5c5b77ee3593cc07f9a149dc926d2f0b9d1feca4ba71cfc6057
- ORIGINAL_POST_RISK及RISK_REPLAY SHA：677563946fe7bade508973a9849f6660e90c326047786a63af3a77669e316152
- 原始manifest SHA：b41db3f6e955fd2709bef1f838f353ac44a245b8889a1a7791d62c870104598d
- DRIVER_RECEIPT SHA：f1eb826324478c7107660d9977f8581c5d15ef64d35a7c8553033999ddc0e4e5
- RECOVERY_RECEIPT SHA：9a570861bbe68ea3b2ef4636dba78491d4c952fae5ffe53deaae12f3a24bcde2

下一步为11031/T0的第3–8步实际准备和固定门禁；每段完成均先独立验收，再推进下一条路径。全部四路径达到8后完成各自DEV8，才进入9–16。MAIN保持封存。
