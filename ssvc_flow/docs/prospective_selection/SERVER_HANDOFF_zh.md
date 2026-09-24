# 服务器执行入口

后续调度使用afd3119，用户新增授权上限7卡。既有5个作业仍使用bb23e9b，smoke使用414df78；各原快照保留。以下命令在 `ssh eee-cluster` 后执行；不修改历史D2目录。单次 submit-ready 只提交当下依赖满足的任务，没有后台循环。提交数量以包含pending的项目队列及持久化intent共同限流。

```bash
cd /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/code_afd3119/ssvc_flow
/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python -m src.prospective_selection.cli submit-ready \
  --root /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/campaign \
  --runtime /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/runtime.json \
  --max-gpu-jobs 7 --available-gpus 5 \
  --python /projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python \
  --project-root /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/code_afd3119/ssvc_flow \
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
