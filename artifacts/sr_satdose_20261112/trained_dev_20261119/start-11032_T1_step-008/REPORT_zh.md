# 11032/T1 第8步 DEV 独立核验

作业1748313于2026-11-20 02:01:23–03:15:15（新加坡）运行，耗时01:13:52，父作业及batch/extern/0均COMPLETED、0:0。老师QoS soujanya-poria-startfund-2026-03，2张PRO6000、8CPU、66G、3小时时限、禁止requeue。固定108条DEV均完成，未重提或修改其他作业。

原7c29审计于20:12:29UTC完成，a9c4驱动+c168精确JSON/base64传输于20:12:34UTC完成。root复核10文件、传输逐字节、22本地引用、三来源pins和发布关系；全POST_RISK原件与重算相同，且与20:02:19UTC终态monitor全对象一致。审计观察17018文件，风险观察17003，原84源码/19执行键不变。admission_review二次只读复核通过，包含实际submission源payload重建、固定request顺序及完整风险规则；未发现阻止第9–16步完整CPU准备的问题。

| 分组 | 固定回答数 | S命中 | A正确 | 原生J合计 | S/B已知 |
| --- | ---: | ---: | ---: | ---: | ---: |
| list8 aligned | 32 | 32 | 32 | 31 | 32 |
| list8 conflict | 32 | 27 | 27 | 26 | 32 |
| list8 no_list | 32 | 32 | 32 | 31 | 32 |
| retention8 | 12 | 12 | 12 | 12 | 12 |
| 合计 | 108 | 103 | 103 | 100 | 108 |

JSON有效108/108，UNKNOWN0，length0，balanced108。冲突S27/32=0.84375；aligned+no_list64/64，相对同起点START63/64及DEV2的63/64未严重退化。T1的1–4、5–8两个训练窗mean d分别2.548582671954303、2.14654819629857；exact authorised185/384、275/384，UNKNOWN0、2保留，两窗均非低d且低authorised。因此未触发双低窗口加最新DEV冲突S<0.80的停止规则。其他三路径复算也无HOLD，pendingDEV为空；四路径仍各8步，训练32/96不变。

这是固定DEV检查，不证明最终效应或统计显著改善，不用于调参/选checkpoint；PREFIX原始13/16与用户豁免保留。172外部引用仅包内交叉验证，未本地重新读取远端capture/model大文件或重评分。审计模型/优化器调用、参数更新、调度修改均0；next_stage_authorized=false，第9–16步仍需自己的完整CPU/存储/调度准入。

关键SHA-256：

- download/AUDIT_REPORT.json：a10c7bffd850b71f46ac5fec38d6959bfd00b3d43ca4a1cd52ef680e52ffb4c7
- download/ANALYSIS.json：1c1c632b44cf7088e1942445073c3b2b669b490ca8f270ba1b5374c9aed6cbd8
- audit/REMOTE_MANIFEST.json：d866c0a4cd12d5d3405018fd9e0c57cece0410dd54848948dcdeca75a6b52815
- download/ORIGINAL_POST_RISK.json：c2461f211994ddd2a694610783e8d361201b55e51de5ece39d00445ed9d332a2
- download/ORIGINAL_RESULT.json：5d0451f4691ac8aa6fe8ea2e36da7260823f2b871e6dfcc2593fe1b135ef0830
- download/PUBLICATION.json：b17107fbd957b3ce6c283ec6f029c9461cffd2ddeca93075e8bdd1fcd2127694
- DRIVER_RECEIPT.json：f75577f376a1b0d42b08a0ede84fa41a7d35b381c2e7c5472f152eea6c0a5bc1
- download/RECOVERY_RECEIPT.json：9940ad66ca7c92f0350d8fc82149cbc4ed272c3c51e1fa66639501f961f6f4ea
