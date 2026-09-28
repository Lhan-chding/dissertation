# 服务器执行入口

后续调度使用e59b9ee，用户新增授权上限7卡。当前开发分支与前置观测使用e59b9ee；已完成source使用bb23e9b，smoke使用414df78，各原快照保留。以下命令在 `ssh eee-cluster` 后执行；不修改历史D2目录。单次 submit-ready 只提交当下依赖满足的任务，没有后台循环。提交数量以包含pending的项目队列及持久化intent共同限流。

```bash
cd /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/code_e59b9ee/ssvc_flow
/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python -m src.prospective_selection.cli submit-ready \
  --root /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/campaign \
  --runtime /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/runtime.json \
  --max-gpu-jobs 7 --available-gpus 5 --qos soujanya-poria-startfund-2026-03 \
  --python /projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python \
  --project-root /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/code_e59b9ee/ssvc_flow \
  --cache-root /projects/varunssd/louis-ssvc/cache
```

按实际可用卡数降低两处GPU数量，绝不超过7；pending和未知提交也占槽。不要改B4/K8或题序。smoke169593已COMPLETED并核验全部门禁。首次开发已提交169616–169620，均单PRO6000；后续运行上面的命令自动核对槽位并保留提交意图。

```bash
squeue -u varun024 -o '%i %j %T %M %R %b'
sacct -j 169616,169617,169618,169619,169906 --format=JobID,State,ExitCode,Elapsed,AllocTRES,NodeList -P
/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python -m src.prospective_selection.cli analyze \
  --phase first-four \
  --root /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/campaign \
  --out /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/first_four_report
```

只有所有4个source、8个P前置观测、88个H32分支及其独立D评估完整后，first-four报告才能称完成。先交付给用户，记录 `campaign/FIRST_FOUR_DELIVERED.json`（交付文件、时间和内容标识），然后 `register-development --expand-after-first-four` 才允许扩展。

E复核使用保存的16个H32 checkpoint和原E面板；不要重训旧分支。每个实际端点只测一次72×32，注册historical_e任务并与新开发共享5卡上限。

最终测试尚未获执行门禁：必须先完成全部开发/调参、真实吞吐报告、四选择器拟合、嵌套lineage精度规划，锁定N/m、代码和T，再为每个测试原点写decision。后续代码快照需记录阶段版本，不覆盖正在运行的414df78文件；本地已补强冻结与最终证据验证，P3前必须部署经过验证的最终版本。

当前轮转记录：169620（O2_R0旧E）已完成并重算核验，继任169684为O1_R3旧E。后续须从campaign/submissions及recoveries读取最新job ID；不要固定沿用示例ID判断以后进度。

2026-09-24T14:18Z：169684（O1_R3旧E）完成并重算核验；4条source均已有H32，下一槽按优先级给前置观测61001_t32，作业169906。其它旧E与其余prestate继续由同一5卡队列轮转。

用户授权7卡的完整说明见RESOURCE_OVERRIDE_zh.md。原设计包和protocol.json中5卡是历史资源设置，已被本次显式授权覆盖；不要为修改并发而改动原协议文件身份。

实际站点门限：2026-09-24核实教师QOS MaxSubmitJobsPU=5，7卡尝试被拒绝且无新JobID。当前示例available-gpus=5反映有效容量；保留用户授权上限7。管理员调整并核实后可改7。拒绝提交已归档并完成无作业核对，不留下永久未知占位。


2026-09-24T14:45Z后续决定（覆盖上文扩卡执行安排）：现场确认账户允许官方可抢占QOS `override-limits-but-killable`，所以并非只能等待管理员增加教师配额。CLI提交e59b9ee已新增显式QOS选择，可抢占任务启用requeue并追加日志，54项针对性测试、ruff、diffcheck通过，代码已push。新快照为 `code_e59b9ee/ssvc_flow`，已有运行快照不变。

两个新增PRO6000前置观测169962/169963成功提交，但Slurm在14:45:31Z给出的预计启动为次日00:17:23Z/00:31:39Z，需等待约9小时32分/9小时46分。用户随后要求排队太久就放弃，因此取消此次扩卡尝试。保留取消终态及从未分配GPU的核验，归档其intent/submission后让原任务回到正常五卡队列；不是删除科学任务或更换seed。最新证据见current_evidence/EXTRA_GPU_CANCELLED_20260924.json。

后续只使用教师QOS，`--max-gpu-jobs 7 --available-gpus 5 --qos soujanya-poria-startfund-2026-03`；不再自动重试额外可抢占资源，也不混用其它型号。每小时续跑已同步此决定。原5个作业169616–169619和169906保持运行；授权上限7保留，但本次扩卡安排已放弃。


2026-09-25网络恢复后的最新队列：四条source及169906均已完成并核验；新增170518/170519/170520/170521为61001_t32的R4/GDPO_R4/SAW_R4/R5开发分支，170522为61001_t96前置观测。不要再按上文旧jobID判断当前运行状态，必须读取submissions/recoveries并查Slurm。见SOURCE4_PROGRESS_zh.md。


2026-09-25T07:46Z：170521(R5)/170519(GDPO)完整分支已核验并用于MEASURED_RUNTIME_zh.md；继任170853=61001_t96/R1、170863=61001_t96/R5。170518(R4)/170520(SAW)尚在评估；前置观测继任为170771=61003_t32。提交回执服务器LAUNCH_20260925_0745.json和LAUNCH_20260925_0748.json。


2026-09-25T08:42Z最新队列为170853/170863/170929/170930（61001_t96的R1/R5/R7/R4），以及170771（61003_t32前置观测）。170518–170521已全部完成且原始证据核验通过。提交回执服务器LAUNCH_20260925_0842.json。


2026-09-25T11:46Z：170771=61003_t32已完成并通过原始证据与四层信息包核验；继任171054=61002_t32前置观测（R1 source），现场RUNNING。当前其它作业为170853/170863/170929/170930，仍为五卡教师QOS。提交回执LAUNCH_20260925_1146.json；前置观测3/8、branch4/88、旧E2/16，首四条交付门禁尚未满足。


2026-09-25T15:52Z：170853/170863/170929/170930均完成并核验；继任171305/171306/171307/171308为61003_t32的R3/R1/GDPO_R4/R7，现场全部RUNNING。第五项仍为171054=61002_t32前置观测（137/144提示，未完成）。当前branch8/88、prestate3/8、旧E2/16。回执LAUNCH_20260925_1552.json，保持五卡教师QOS与原首四条门禁。


2026-09-25T16:50Z：171054=61002_t32已完成并核验；继任171346=61004_t96前置观测（source R1），任务bbfc472877f533c7a8c846a88b76bb6f5708b04cf9ae5e56743823299beb1f36。初查PENDING（ReqNodeNotAvail, Reserved for maintenance），不得重复提交。四个训练分支171305–171308正常RUNNING，项目共4运行+1等待。当前prestate4/8、branch8/88、旧E2/16，首四条门禁不变。回执LAUNCH_20260925_1650.json。


2026-09-26T05:40Z：网络恢复后已核验171305–171308全部正常完成，branch12/88、prestate4/8、旧E2/16。当前等待171346=61004_t96前置观测，以及171705/171706/171707/171708=61002_t32的R0/R2/R7/R1；5个PENDING、0运行，均为维护预留。UTC维护09-26 16:00至09-27 13:00；原24小时申请无法在维护前容纳，站点拒绝对171346修改TimeLimit（8小时尝试失败，未生效）。保留原提交，不取消或重提，不切换QOS。回执LAUNCH_20260926_0540.json；维护证据current_evidence/MAINTENANCE_QUEUE_20260926.json。


2026-09-27T16:20Z：171346=61004_t96前置观测正常完成，4608输出与四层共同信息包核验通过；prestate5/8、branch12/88、旧E2/16。继任173523=61002_t96前置观测，现场RUNNING。其它171705/171706/171707/171708为61002_t32的R0/R2/R7/R1，均已提交24步恢复段并RUNNING；当前五卡教师QOS。新回执LAUNCH_20260927_1620.json及current_evidence/PRESTATE_61004_t96_VERIFIED.json。每小时自动续跑已恢复；996b610及后续Git历史推送仍受此前数据外发审批拒绝约束，未重试推送，不影响授权范围内实验推进。


2026-09-27T18:23Z：171705–171708正常完成且原始证据核验通过，branch16/88、prestate5/8、旧E2/16。当前作业173667/173668/173669/173670分别为61004_t96的R2/DIRECT_REPAIR_R4/R4/R5，均RUNNING；第五项173523为61002_t96前置观测。新回执LAUNCH_20260927_1823.json；分支回执current_evidence/BRANCH_61002_T32_FIRST4_VERIFIED_20260927.json。仍未满足首四条交付门禁，保持原配方与样本，不重试受阻Git推送。


2026-09-27T20:21Z：173523=61002_t96完成并核验，prestate6/8；继任173741=61004_t32前置观测，首次PENDING。当前另4个分支173667–173670（61004_t96 R2/DIRECT_REPAIR_R4/R4/R5）RUNNING且均有8步权威恢复段。branch16/88、旧E2/16，原首四条门禁和Git推送审批边界保持。回执LAUNCH_20260927_2021.json及current_evidence/PRESTATE_61002_t96_VERIFIED.json。


2026-09-28T02:06Z：网络恢复，173667–173669三个branch及173741前置观测已完成并核验；branch19/88、prestate7/8、旧E2/16。当前173670=61004_t96/R5仍运行（H32评估130/144），新提交173821=61002_t96/R3、173822=61004_t32/R5、173823=61002_t96/DIRECT_REPAIR_R4、173824=61003_t96前置观测，首次PENDING；总计5个项目单PRO6000作业。服务器提交回执LAUNCH_20260928_0206.json。本地核验回执BRANCH_61004_T96_FIRST3_VERIFIED_20260928.json和PRESTATE_61004_t32_VERIFIED.json。原首四条交付、最终测试及Git推送边界不变。

补位后02:08Z现场复核：173821–173824全部RUNNING，加上173670共5个单PRO6000项目作业；新提交不再处于等待状态。


2026-09-28T02:28Z接续：173670已COMPLETED/0:0且原始证据核验通过，branch累计20/88，prestate7/8、旧E2/16。最新项目作业为173821=61002_t96/R3、173822=61004_t32/R5、173823=61002_t96/DIRECT_REPAIR_R4、173824=61003_t96前置观测（以上RUNNING），以及新补位173845=61004_t32/R1（首次PENDING）。新提交持久回执LAUNCH_20260928_0227.json，任务425b460406d9cc9eecbf4e99f586442ce826889489566ed2f558ad534919dd55。仍按max-gpu-jobs=7、available-gpus=5、教师QOS运行，未知提交不得重提。首四条未交付，不能扩展开发矩阵或最终测试；996b610及后续历史推送继续等待特定数据外发授权，不重试。

补位后02:30Z复核：173845已RUNNING，173821–173824仍RUNNING，5个项目作业均为单PRO6000教师QOS。


2026-09-28T03:23–03:31Z：出现新的执行阻塞。173821/173822/173823/173824/173845均FAILED/1:0，分别运行00:35:15/00:36:06/00:40:05/00:34:21/00:19:46，于02:40–02:48Z退出，涉及4个节点；当前项目GPU作业0个。stdout为空，stderr只有生成警告，没有异常堆栈或time结束记录，不能仅凭日志断言退出根因。独立几十字节写入探针明确返回OSError(122, Disk quota exceeded)，留下零字节IO_PROBE_20260928_0327.txt。文件系统整体281T、已用78T、可用204T（28%）；quota/lfs工具不可用，目录及两级父目录getfattr没有返回属性，因此具体配额类型和上限尚未知。证据见current_evidence/STORAGE_QUOTA_BLOCK_20260928.json。

四个失败branch均保留H00和原attempt，无8步COMMIT；R3/R5/DIRECT_REPAIR/R1分别保留3/3/4/0条未提交更新，已有更新数值有限，不能当完整恢复段或完成结果。61003_t96前置观测保留t88的23/72个提示COMMIT，t96为0。已核验总数仍为source4/4、prestate7/8、branch20/88、旧E2/16。未删除原件、改seed或提交替代任务。

恢复前置条件：先解除实验目录写入配额阻塞，确认一次小文件写入成功，再实时核对旧JobID终态、原始绑定及H00完整恢复状态，按现有TaskRegistry.resubmit_terminal保留原intent/submission并登记一次显式恢复。禁止因squeue为空直接submit-ready或擦除失败提交；禁止把未提交更新当Adam恢复点。当前仍按5卡有效容量与教师QOS；待用户/管理员处理存储配额。自动化保持每小时检查；不扩展、冻结或运行最终测试。Git仍仅本地保存，未重试既有受限push。

2026-09-28T04:46Z：用户授权且限定仅自己目录内的旧结果清理已完成，candidate04的Q1/Q2 units共约59.9 GB已删除；当前campaign、D2/V4、runs、模型、环境、数据均保留。1MiB写入+fsync成功，配额阻塞解除。清单及恢复输入见current_evidence/OLD_RESULTS_CLEANUP_20260928_PLAN.json、OLD_RESULTS_CLEANUP_20260928_RECEIPT.json、RECOVERY_INPUTS_20260928.json。

当前恢复ID全部RUNNING：174109=61002_t96/R3，174110=61004_t32/R5，174111=61002_t96/DIRECT_REPAIR_R4，174112=61003_t96前置观测，174113=61004_t32/R1，分别接续173821/173822/173823/173824/173845。使用TaskRegistry.resubmit_terminal登记attempt_1，RECOVERY_LAUNCH_20260928.json及campaign/recoveries/<task>/attempt_1/submission.json是当前权威提交。旧失败attempt及原submission保留；再次失败或状态未知时须单独诊断，不能重复恢复。

仍为5卡教师QOS、e59b9ee生产代码；已核验20/88分支、7/8前置观测、2/16旧E。清理完成后不扩大删除范围，不扩展开发范围或最终测试，保持每小时检查。服务器清理授权不解除GitHub数据推送限制，仍不push包含996b610的历史。
