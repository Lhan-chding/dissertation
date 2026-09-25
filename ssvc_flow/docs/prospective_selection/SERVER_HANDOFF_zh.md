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
