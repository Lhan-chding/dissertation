# 服务器执行入口

当前开发真实GPU快照为bb23e9b（smoke使用414df78）。以下命令在 `ssh eee-cluster` 后执行；不修改历史D2目录。单次 submit-ready 只提交当下依赖满足的任务，没有后台循环。提交数量以包含pending的项目队列及持久化intent共同限流。

```bash
cd /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/code_bb23e9b/ssvc_flow
/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python -m src.prospective_selection.cli submit-ready \
  --root /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/campaign \
  --runtime /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/runtime.json \
  --max-gpu-jobs 5 --available-gpus 5 \
  --python /projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python \
  --project-root /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/code_bb23e9b/ssvc_flow \
  --cache-root /projects/varunssd/louis-ssvc/cache
```

实际只允许4个worker时将两处GPU数量改4。不要改B4/K8或题序。smoke169593已COMPLETED并核验全部门禁。首次开发已提交169616–169620，均单PRO6000；后续运行上面的命令自动核对槽位并保留提交意图。

```bash
squeue -u varun024 -o '%i %j %T %M %R %b'
sacct -j 169616,169617,169618,169619,169620 --format=JobID,State,ExitCode,Elapsed,AllocTRES,NodeList -P
/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python -m src.prospective_selection.cli analyze \
  --phase first-four \
  --root /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/campaign \
  --out /projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/first_four_report
```

只有所有4个source、8个P前置观测、88个H32分支及其独立D评估完整后，first-four报告才能称完成。先交付给用户，记录 `campaign/FIRST_FOUR_DELIVERED.json`（交付文件、时间和内容标识），然后 `register-development --expand-after-first-four` 才允许扩展。

E复核使用保存的16个H32 checkpoint和原E面板；不要重训旧分支。每个实际端点只测一次72×32，注册historical_e任务并与新开发共享5卡上限。

最终测试尚未获执行门禁：必须先完成全部开发/调参、真实吞吐报告、四选择器拟合、嵌套lineage精度规划，锁定N/m、代码和T，再为每个测试原点写decision。后续代码快照需记录阶段版本，不覆盖正在运行的414df78文件；本地已补强冻结与最终证据验证，P3前必须部署经过验证的最终版本。
