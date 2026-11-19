# 11031/T1 第8步 DEV 独立核验

作业1747421于2026-11-19 17:29:10–18:29:25（新加坡）运行，耗时01:00:15，父作业及batch/extern/0均COMPLETED、0:0。使用老师QoS soujanya-poria-startfund-2026-03，2张PRO6000、8CPU、66G、3小时时限、禁止requeue。固定108条DEV完成；此前VPN中断没有改变作业或触发重提。

原7c29审计、a9c4驱动与c168单次JSON/base64整包下载已完成。root复核完整10文件、原始传输逐字节、源码pins及发布/收据；POST_RISK原始文件与重算文件逐字节相同，也与11:04:31UTC终态巡检的完整对象一致。远程审计观察16362文件，原84文件/19执行源码键不变。admission_review独立二次只读复核通过，未发现阻碍下一组完整CPU准备的问题。

| 分组 | 固定回答数 | S命中 | A正确 | 原生J合计 |
| --- | ---: | ---: | ---: | ---: |
| list8 aligned | 32 | 32 | 31 | 29 |
| list8 conflict | 32 | 26 | 25 | 24 |
| list8 no_list | 32 | 29 | 29 | 28 |
| retention8 | 12 | 12 | 12 | 12 |
| 合计 | 108 | 99 | 97 | 93 |

108条均完成、JSON有效、balanced结束，S/B已知108条，UNKNOWN及length均0。冲突S为26/32，aligned+no_list61/64，retention12/12；分母未筛选。相对于本起点START的aligned+no_list59/64，第2步与第8步均为61/64，下降值-0.03125，均非严重退化。

T1的1–4、5–8窗口平均d分别1.8083395272396081、2.1976671516460957，均未满足d<0.75；exact authorised分别207/384、256/384，训练UNKNOWN分别2、1，固定384分母完整保留。最新DEV冲突S=26/32=0.8125；整套停止规则未触发，global_hold=false。仍待11032/T0、11032/T1的DEV8；训练32/96更新不变，全部DEV8完成且无HOLD前禁止第9步。

这是固定DEV检查，不证明最终效应、饱和或统计显著改善，不用于调参或选checkpoint。原PREFIX13/16及用户豁免不变。下载核验只确认原审计精确传回；168外部引用仅包内交叉验证，未在本地重新读取远端原始capture/model大文件或重新评分。审计无模型/优化器调用、参数更新或调度修改，后继仍须各自准入，next_stage_authorized=false。

关键SHA-256：

- download/AUDIT_REPORT.json：2d3fe065d74e13a9900739a0c4446af6df0831ed1106a57c05f4cc37c4cf2959
- download/ANALYSIS.json：6506743d26e9caa6293fbe7c136908539f09d038cc5be96ff87e22fac65b250a
- audit/REMOTE_MANIFEST.json：f03dc17be4363a695b1cf9c3f1e72c7c96efd16c929171a8e740c9f4922cca70
- download/ORIGINAL_POST_RISK.json：2408f2af61262edea5cf14f1af65fa5856d87067416c91492f404a4b32cc212f
- download/ORIGINAL_RESULT.json：a364426a7d6058cfbcfbf3980a18c896330c3aa1126b982ab7d090a1b19dd62a
- download/PUBLICATION.json：9b3648e8f6bfbfe00b7dbbb2c83abd517d43ece8a8e8418a6542d53301df42af
- DRIVER_RECEIPT.json：6c951a12bf788f90dd0a8352a8849a40a15d5adfc7e8875aadc447721fcb6bb7
- download/RECOVERY_RECEIPT.json：c8a6cc824381631e75c2cb6d26cf2c8f04b6bd95dccd3f5fe5ead21f3103fbf2
