# 11031/T0 第8步 DEV 独立核验

作业1746695于2026-11-19 14:18:01–15:21:40（新加坡）运行，耗时01:03:39，父作业及batch/extern/0均COMPLETED、0:0。使用老师QoS soujanya-poria-startfund-2026-03，2张PRO6000、8CPU、66G、3小时时限、禁止requeue。此端点完成固定108条DEV，无训练或MAIN模型调用。

原7c29审计于08:30:30UTC完成，a9c4驱动及c168单次整包下载于08:30:41UTC完成。完整10文件、原始单次JSON/base64传输、来源SHA、发布及本地收据已复核；POST_RISK原始文件与重算文件逐字节一致，也与08:19:40UTC终态巡检的完整对象一致。远程审计观察16034个文件，核对原84文件/19执行源码键。admission_review独立二次只读复核通过，未发现阻碍下一组完整CPU准备的问题。

| 分组 | 固定回答数 | S命中 | A正确 | 原生J合计 |
| --- | ---: | ---: | ---: | ---: |
| list8 aligned | 32 | 31 | 30 | 29 |
| list8 conflict | 32 | 17 | 16 | 16 |
| list8 no_list | 32 | 32 | 32 | 30 |
| retention8 | 12 | 12 | 12 | 12 |
| 合计 | 108 | 92 | 90 | 87 |

108条均完成、JSON有效、balanced结束，S/B已知108条，UNKNOWN与length均0。分组分母未筛选，冲突S为17/32，aligned+no_list为63/64，retention为12/12。相对于本起点START的aligned+no_list 59/64，本次下降值为-0.0625；第2步60/64及本次均未触发严重退化。global_hold=false，仍需11031/T1、11032/T0、11032/T1的DEV8；四路径训练进度仍为32/96更新，禁止提前第9步。

这些是固定DEV检查结果，不证明最终训练效果、饱和或统计显著改善，不用于选checkpoint/调参。原PREFIX的13/16及用户豁免保持不变。下载核验确认原审计证据精确传回；166个外部引用仅核对包内一致性，未在本地重新读取远端原始大文件，不应将下载核验说成第二次服务器审计。无新增GPU重放、训练更新或调度修改；每个后继阶段仍须各自准入，next_stage_authorized=false。

关键SHA-256：

- AUDIT_REPORT：581b34bcfec638641bd04af6ba005f28dcd3bf9b61107b2685ece4472da1e488
- ANALYSIS：c8a89ac3607cce1d7aa5fcfb33f6f27f9023e844a4909b6ffd1f026440fe9cb6
- 原始manifest：40e9151449aebd916a2ae92b58b33266f17c6b84f7ccbde3ab2ebdb9d43f6d3c
- POST_RISK／RISK_REPLAY：94999b7c5fda9bfbf331ca1cf90916e8658767b8d96d5aba7b09c09f263d7027
- ORIGINAL_RESULT：04cc1880bd1fe95850b4de08213c78effc2eec71adb8c387a08b32d0f840efda
- PUBLICATION：eaa7e04d38a78b601218d5cec15e33b48c49a50d74db30b1d26dd1d1ae57fac6
- DRIVER_RECEIPT：8b9efbd6f9847014513bf3c1c46838d405a7f39ef98370d1c442f46e8a63982f
- RECOVERY_RECEIPT：1f67672e6878620bc4ae8215aa9c4de1a0227d3afa3c768f8b5e9e7c7cd532fe
