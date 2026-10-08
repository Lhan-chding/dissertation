# SER-J23 一次性 CPU 运维控制器

本目录提供独立的运维脚本；运行回执应写入 Git 忽略的私有目录，不属于冻结的科学源码。`cpu_controller.py` 只使用 Python 标准库，不导入模型、PyTorch、任务生成器或评分器；不会读取任务或 truth 清单。控制器独立校验自身脚本哈希，不修改科学源码或既有验证回执。

## 启动

由调用者选择 CPU-only 环境，显式提供新 run、此次四个独立 worker 的 **顶层数字 allocation ID**、唯一 session ID、私有回执目录。控制器不申请资源、不调用 `sbatch`、不占 GPU。不支持 job ranges、array 字符串或 `.batch/.extern` ID。

```bash
PYTHON=python3
OPS=/path/to/operations
RUN=/path/to/registered/run

"$PYTHON" "$OPS/cpu_controller.py" \
  --run "$RUN" \
  --worker-job-ids 123001 123002 123003 123004 \
  --session-id controller-20261008-01 \
  --receipts-root "$OPS/controller_receipts" \
  --poll-seconds 60
```

上面的 job IDs 是示例，必须替换为本轮实际 worker allocation IDs。若使用 Slurm 启动控制器，应申请单个 CPU、少量内存和适当时限，**不请求 GPU，也不继承 GPU allocation 的环境**。脚本会拒绝明确的 GPU allocation 环境变量，包括 `SLURM_JOB_GPUS=0`（这是 GPU 编号 0，不是零张卡）。应在四个 worker 提交后启动控制器，不能依赖它们全部结束后才启动。

可以加 `--once` 只做一轮元数据观察；除完整矩阵候选外返回 exit 2，不能将一轮观察成功当实验完成。重复使用相同 session ID 会报错，绝不覆盖已有配置、查询证据或回执。控制器异常退出后，如需再次观察，必须使用新的 session ID，旧会话保留。

## 行为与边界

- `queue.sqlite` 通过 `mode=ro` 和 `PRAGMA query_only=ON` 读取；核对任务顺序、payload、状态以及注册 identity。启动时固定 `FROZEN_PLAN.json`、`REGISTERED_MATRIX.json` 字节哈希，后续变化立即中止本控制会话。
- 每轮只查询显式指定 IDs 的 `sacct`、`squeue`，保存命令、原始 stdout/stderr、退出码、耗时和哈希。数据库在查询前后分别取快照，以免正常完工与调度查询的竞争造成错误 STOP。
- 只有 `sacct` 明确终态、完整 ExitCode/ElapsedRaw/End 记录，且 `squeue` 成功查询后不存在该 allocation，才认定 `TERMINAL_VERIFIED`。查询失败、缺行、重复行、格式异常都只是 UNKNOWN **观测**，不会据此判进程死亡、写科学 UNKNOWN、重试或重提。
- 任一科学任务为 `BLOCKED_TECHNICAL/UNKNOWN`，或已核验终态的本轮 worker 仍持有科学 `RUNNING` 任务，原子创建新 run 的 `STOP`。STOP 包含具体科学任务、原因与本轮原始证据目录；现有 STOP 永不覆盖。
- STOP 后继续观察 worker allocation 是否终止。现有科学 worker 仅在任务边界检查 STOP；正在进行的任务可以完成。控制器不会 `scancel`、发信号、改队列、释放锁、删除文件或自动重试。
- 仅当 SQLite **全部注册任务 COMPLETE** 时，输出 `SCIENTIFIC_MATRIX_COMPLETE_CANDIDATE` 并退出。它不是 release、不是科学结论、不是 `STAGE_COMPLETE`，也不认证 `SCHEDULER_ACCOUNTING`。
- 若 STOP 后全部指定 allocations 已核验终止，输出 `STOPPED_WORKER_ALLOCATIONS_TERMINAL`。若没有 STOP、仍有未完成科学任务、但所有指定 allocations 已核验终止，输出 `INCOMPLETE_NO_LIVE_WORKER_ALLOCATIONS`。这两种状态均退出 2。
- 普通等待期间每 60 秒再观察一次，不占 GPU。若控制器本身达到墙钟时限或失去调度访问，不能推断实验失败或成功，也不会增加科学预算。

生产 worker 的默认身份 `host:SlurmJobID:PID` 可直接关联 allocation；自定义 worker 身份通过 `worker_attempts/*/STARTED.json` 的 `slurm_job_id` 关联。无法唯一关联时保留未知映射，不推断该科学任务的进程死亡。

## 输出

每个私有会话目录包含：

```text
CONFIG.json
polls/000000/
  SQLITE_SNAPSHOT_BEFORE_QUERIES.json
  SQLITE_SNAPSHOT.json
  sacct.json  sacct.stdout.txt  sacct.stderr.txt
  squeue.json squeue.stdout.txt squeue.stderr.txt
  OBSERVATION.json
...
FINAL.json                         # 正常到达某个控制终态
CONTROLLER_INTERRUPTED.json        # 控制器异常/中断时尽力保留
```

只允许额外创建 `<run>/STOP`。旧 raw、模型 checkpoint、队列和 release 状态不修改。查询证据属于观察回执，不是完整计费认证；收账和最终 release 仍须由独立流程核对全部科学证据、真实终态 allocation、失败历史、NLL 与报告。

## 本地验证

```bash
python3 /path/to/operations/test_cpu_controller.py -v
```

测试仅创建临时 SQLite 元数据、mock scheduler 文本和运维 JSON；不生成任何四元组 truth，不调用 `sacct/squeue`、SSH、调度器或模型。部署前运行这些元数据回归及 Ruff 检查，并单独保存脚本哈希与本机验证回执。
