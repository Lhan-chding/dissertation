# V4 源训练补启与测量修正（2026-09-17）

## 执行范围

用户本轮明确授权：完成测量流程修正，保留已有源训练继续，补启另一路 GPU。
已有 job 157057 / worker 1 继续；补启 worker 0，使用既有 B/C 任务表。
该表包括 seed 41001/41002 的 X_BASE 128 步源训练及 step32/96 四个原点。
补启跳过已完成 B 原件；没有加入 D 多种子扩展、性能探针或在线 SSVC。

## 准确补启命令

在服务器执行。提交器查询实际 Slurm 状态，复用活动 worker，仅提交已退出且未完成的 worker；
保留提交意图和返回 job ID，若提交结果不确定则拒绝重复提交。

```bash
cd /projects/varunssd/louis-ssvc/modeling_v4_20260916/code_161c591/ssvc_flow
/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python scripts/submit_modeling_v4.py \
  --config configs/modeling_v4/protocol.json \
  --project-root /projects/varunssd/louis-ssvc/modeling_v4_20260916/code_161c591/ssvc_flow \
  --campaign-root /projects/varunssd/louis-ssvc/modeling_v4_20260916/campaign \
  --tasks /projects/varunssd/louis-ssvc/modeling_v4_20260916/tasks_initial.json \
  --workers 2 --stage gpu --phase bridge --resume --submit --execute-gpu
```

GPU 配置：`cluster02`，账户 `rose`，QoS `soujanya-poria-startfund-2026-03`，
每个 worker 一张 `gpu:pro6000:1`，72 小时作业时限。提交器沿用原有 CPU/内存请求；
实际分配以 Slurm 为准。预期本 campaign 同时两卡，上限三卡。

## 已有计时依据

恢复连接后，seed41002 的 step1 至 step10 已提交时间相隔 6186.17 秒，
对应 9 个完整步间隔，平均 687.35 秒/步；间隔范围约 647–723 秒。
按当前源训练实测间隔线性换算，新一路 128 步约 24.4 小时，
不包含初始加载、排队和后续 C 响应测量，也不是保证完成时间。

对比此前依据：两个已完成 B 原点的 12 次实际 Adam 更新，分别耗时 6090.79 秒和 6111.65 秒，
平均约 508 秒/次。按这个已测更新成本线性换算，128 次约 18 小时。
这只是源训练的量级参考：新源策略的生成长度、存储及运行环境可能改变每步耗时，
不是源训练实测完成时间，也不包含后续 C 响应测量或排队。
已完成 B 的响应采样与评分分别耗时约 16 小时 51 分和 17 小时 42 分；
C 的题数、bank 和样本预算更大，不能照搬 B 的结束时间。
本轮不为估时增加模型调用或性能探针。

## 采集与分析版本

两个 GPU worker 都使用既有 `code_161c591` 采集快照及原始任务身份，避免在进行中的
campaign 内混入不同训练/采样版本。旧快照不原地修改，已有原始结果不覆盖。
新测量代码修正统计报告，并提供独立服务器 CPU 入口读取冻结原件；
报告分别记录采集源码和实际分析源码，不能伪装成同一个版本。

15 分钟监控保持只读，只在阶段完成、失败、配置不符或需要决定时通知。
具体提交结果、实际分配、CPU 验证及部署版本写入本次运行收据后再报告完成。

## 本轮连接与验证状态

本轮首次 SSH 核验成功：157057 正在运行，seed41002 最近保存到 9/128 步；
之后连接返回 `Operation timed out` / `Network is unreachable`。
用户随后确认网络恢复；重新核验 157057 仍为 RUNNING，源训练推进到 10/128 步。
执行上述命令后，提交器记录 worker0 新作业 **158482**，worker1 为
`REUSED_ACTIVE`，继续原作业 **157057**。没有取消、重启或修改已有 GPU 进程。
提交明细在服务器 `campaign/SLURM_SUBMISSIONS.json`；实际运行状态另行核验。
修正后真实原件的服务器 CPU 验证在独立分析部署完成后执行。

本地七个受影响测试文件共 73 项通过（4.60 秒），覆盖稳定协方差、零方差、
计数区间、身份错配拒绝、采集恢复与响应拟合接口。这个结果属于小型代码回归，
不能代替真实原件验收，也不表示 V4 科学有效性已通过。随后仅复验新增的
有界缓存及版本检查：3 项通过（0.59 秒）；未重复运行全仓测试。
