# 11032/T0 第8步 DEV 独立核验

作业1747963于2026-11-19 22:14:55–23:26:22（新加坡）运行，耗时01:11:27，父作业及batch/extern/0均COMPLETED、0:0。老师QoS soujanya-poria-startfund-2026-03，2张PRO6000、8CPU、66G、3小时时限、禁止requeue。固定108条DEV均完成；排队和VPN中断未导致重提。

原7c29审计于15:41:49UTC完成，a9c4驱动+c168单次JSON/base64包下载于15:41:55UTC完成。root复核全10文件、原始字节、来源pins和发布/本地收据；原始POST_RISK与重算逐字节相同，且与15:30:31UTC终态monitor全对象一致。远程audit观察16690文件，原84源码/19键不变。admission_review二次只读复核通过，未发现阻碍最后一组DEV8完整CPU准备的问题。

| 分组 | 固定回答数 | S命中 | A正确 | 原生J合计 | S/B已知 |
| --- | ---: | ---: | ---: | ---: | ---: |
| list8 aligned | 32 | 32 | 31 | 30 | 32 |
| list8 conflict | 32 | 16 | 16 | 14 | 31 |
| list8 no_list | 32 | 32 | 32 | 30 | 32 |
| retention8 | 12 | 12 | 12 | 12 | 12 |
| 合计 | 108 | 92 | 91 | 86 | 107 |

全部108条balanced结束，length0，但balanced不代表JSON有效。有效JSON107/108，UNKNOWN1。异常请求为SAT1/DEV/11032/T0/s08/list8/OT1_SATDOSE_DEV_LIST_n8_00019_m2r1_sm_2026111201/conflict/d00；原始解析错误为Expecting ',' delimiter: line 17 column 3 (char 1547)。该条S/B=null、known=0、S_hit=0、A=0、原生J=0，保留固定分母，未修补/重试/补样本。与第2步的length UNKNOWN不是同一结果。

冲突S16/32，aligned+no_list64/64，retention12/12。相对于11032 START aligned+no_list63/64，第2步63/64与本次64/64均非严重退化。global_hold=false，pending仅11032/T1/8；训练32/96更新不变，全部DEV8通过且无HOLD前不得第9步。

这是固定DEV检查，不证明最终效应或统计显著改善，不用于调参/选checkpoint；PREFIX原始13/16与用户豁免仍保留。下载核验确认原审计精确传回，170外部引用只包内交叉验证，未本地重读远端原始capture/model大文件或重评分。审计模型/优化器调用、参数更新、调度修改均0；next_stage_authorized=false，后继仍须自己的完整准入。

关键SHA-256：

- download/AUDIT_REPORT.json：6370a6a2b5259da5917b623ab496fe95c893ff3041c6f7aede7c44b7b2ba3172
- download/ANALYSIS.json：5cf25ad36c6a5dd092abe8545f01efdabcb2ab4e0bd2a0ebb76f23ada8784222
- audit/REMOTE_MANIFEST.json：8cc8ece3feb679d44111b0d313db8cfe280a7deaac5bdf2aaba58fd4c5cd281d
- download/ORIGINAL_POST_RISK.json：832a059579833422dc2815ef1f16bc39af4b95008c53a87660fdb8e2449fcf4f
- download/ORIGINAL_RESULT.json：824b31555df66b4a4f5b5930a28c1c99abdf05e673ccacde4a39e80cff750fc6
- download/PUBLICATION.json：280de802c56f7714794ec333a1c8200ec9012cdb382944cc897f4355a177b655
- DRIVER_RECEIPT.json：0a48aee0bc22e25e1b9f6b8ce069e8ac0ca8da279ab2bd97f60ac799826bfc49
- download/RECOVERY_RECEIPT.json：cd06a2cd555d6001588a6a4365606f363a584a9953dbe9a884d50d27e8edce1e
