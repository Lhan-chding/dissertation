# MM-DEV F2 执行记录与入口

本目录实现用户于 2026-10-09 明确授权的完整 MM-DEV F2。
原始合同见 [CODEX1_MM_DEV_FINAL_PLAN_zh.md](design/CODEX1_MM_DEV_FINAL_PLAN_zh.md)，
授权范围见 [EXECUTION_AUTHORIZATION_20261009_zh.md](EXECUTION_AUTHORIZATION_20261009_zh.md)。
`design/` 保留签收包字节；执行实现位于 `src/mm_dev/` 与 `scripts/mm_dev/`。
`src/mm_core/` 作为冻结依赖复用，未改变旧实验。

服务器规范运行根为
`/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos`。
不带 `_teacherqos` 的旧根保留原冻结、被取消的排队记录及迁移回执。
本地独立工作树为
`/Users/louis/.codex/worktrees/mm-dev-f2-20261009/dissertation-ntu`。

## 科学范围和资源

模型固定为 `Qwen/Qwen3.5-9B`，revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`。保持原生视觉处理、非 thinking
模板、BF16 base、FP32 LoRA 和固定软件环境。

先完成 4 对 2+2 的真实 GRPO ENGINE 恢复验收，再执行 4 条 PREP、5 起点
PROBE、30 条 CONTINUE 和 35 个模型身份的 DEV_EVAL。科学训练共 1088 步、
208896 次生成；PROBE 30720、DEV_EVAL 430080，科学生成合计 669696。
ENGINE、失败、未知尝试、重做和额外前向单独计数。

每个 worker 一张实际 Pro6000，本计划最多五张同时占用。每次分配显式请求
4 CPU、64 GiB 主机内存；集群会按实际 CPU/GPU 配比覆盖 RAM 请求，最终以
Slurm 分配回执为准（CPU 每核 3 GiB）。GPU 小时只记账，没有累计 GPU 或研究墙钟停止门槛。
集群单次分配的三天上限是可续租约；正常抢占/到期用新 attempt 恢复同一完整状态。
禁止 Slurm 隐式 requeue；技术错误、身份不符及未知提交必须先保留并核对证据。
按用户明确要求，全部实验作业使用老师的 `soujanya-poria-startfund-2026-03` QoS，
account `rose`。当前该 QoS 每用户最多提交五个作业，CPU 控制器、依赖后继和其他
同用户同 QoS 作业均计入；调度器查询实际限额和队列，满额或无法核实时等待。
五张 GPU 是本计划上限，实际并行数还受这些提交名额约束。

## 冻结与执行

1. `materialize_data.py` 固化数据；在无 CUDA 的 CPU 作业中完成实际 processor
   检查，再复核固定首尾根的原图和处理图。保存独立 `VISUAL_REVIEW.json`。
2. 记录当前 Slurm 用户、account、已授权 QoS 和本计划资源许可，保存
   `ALLOCATION_PERMISSION.json`。代码必须已 commit，按该 commit 独立部署。
3. `validate_freeze.py --create` 重算源文件、数据、模型每文件与环境身份；
   `F2_FREEZE.json` 写入后不可覆盖。
4. `submit_matrix.py --watch` 通过不可变 registration、提交 intent、job ID、
   实际 Slurm 分配和 hash-chain journal 推进依赖。ENGINE 不通过则不推进科学任务。
5. 全部科学路径及测量完成、所有 GPU attempt 已有真实终态记账后，才执行
   `score_and_analyze.py` 和 `verify_release.py`。独立复算通过不等于科学效果成功。

调度进程在独立 CPU Slurm 作业中运行。运维入口
`scripts/operations/mm_dev_f2_controller.py` 与 `ops/CONTROLLER_CODE.json` 单独绑定源码
哈希。每个控制作业先核验登记和日志，再登记一个 afterany 后继，之后才提交科学任务；
这样后继的提交名额不会被首轮调度占用。
正常租约信号在完整调度 tick 后交接，未知提交不盲目重提。后继发现 RELEASED 或
全部 worker 停止后的技术阻断即退出，不再生成新后继。
这样不依赖登录连接存续，符合集群的
[登录节点说明](https://github.com/NTUEEECluster/docs/blob/main/quickstart.md)与
[Slurm 资源规则](https://github.com/NTUEEECluster/docs/blob/main/slurm-guide.md)。

典型服务器命令（`CODE` 指向已经校验的单个 commit 部署目录）：

```bash
PY=/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python
RUN=/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos
"$PY" "$CODE/scripts/mm_dev/submit_matrix.py" \
  --plan "$CODE/docs/mm_dev_f2/design/config/MM_DEV_F2.json" \
  --run-root "$RUN" --code-root "$CODE" --python "$PY" \
  --measurement-shards 1 --engine-lease-minutes 720 --lease-minutes 4320 \
  --watch --poll-seconds 30
```

## 验证与交付边界

本地回归使用 importlib 模式，避免两个历史测试目录的同名文件导入冲突：

```bash
PYTHONPATH=src python -m pytest --import-mode=importlib tests/mm_core tests/mm_dev -q
ruff check src/mm_dev scripts/mm_dev tests/mm_dev
```

CPU 测试包含真实 tiny Qwen 视觉采样、192 序列等权梯度、恢复/信号、完整矩阵
门禁模拟和篡改拒绝。它们不替代真实 9B CUDA 的采样概率与恢复验收。
真实 ENGINE 还需独立 CPU 进程按字段与元素比较 step2/4 的 checkpoint，
覆盖参数、Adam、scheduler、reference、RNG 和游标。

最终 `verification/RELEASE_MANIFEST.json` 是证据清单，尚不等同压缩包。
完整 raw、失败记录及保留的 checkpoint 留在服务器；汇报包需校验 CRC/EOF、
安全路径、逐文件 SHA256、入口链接，并明确服务器专有权重和计划轮换的中间
checkpoint 缺省项。不得把轻量结果包称为包含完整权重的重训包。
本轮结束后不自动启动 MM-LOCK、MM-CAL、MM-ONLINE、奖励搜索或新实验。

## 已发生的技术事件

- 用户指定老师 QoS 后，旧 ENGINE 194700 与控制器后继 194701 均在未启动时取消，
  两者 ElapsedRaw=0 且无 AllocTRES。旧 CPU 控制器 194699 经租约信号正常退出
  （COMPLETED 0:0）。旧根未产生 GPU 结果；4692 个数据/QA/材料化与视觉复核文件
  逐项验证原冻结 SHA256 后原样复制到新根。只重新冻结资源许可和调度实现，
  科学计划、数据、模型、seed 和训练配方不变。两根均留存 QOS_MIGRATION_RECEIPT.json。
- CPU 预处理首次提交 194635 在 `rose` 的并发作业限制下始终 PENDING。
  集群拒绝原 job 修改 QoS；确认未启动后取消此唯一作业，在相同脚本和输入下
  用已允许的 QoS 重提为 194644。194644 以 `COMPLETED 0:0` 完成，耗时 4分03秒，
  处理 2336 图和 7008 题，无 GPU 或模型生成。两次提交和变更回执保留于运行根。
- 初次上传被自动审批要求进一步证明文件范围；核验全部 synthetic-chart/data
  allowlist 与实际哈希后，同一目标和同一载荷的重试获准。没有未解决的授权阻碍。
- 20 张全分辨率处理图的多源 SFTP 传输遇到 channel 失败；改用单 SSH 连接下载
  同一固定清单，原图与数据未改动，也未重跑 processor。
