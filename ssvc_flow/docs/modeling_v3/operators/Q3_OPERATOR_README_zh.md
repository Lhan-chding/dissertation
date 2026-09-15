# Q3 服务器 CPU 准备器

`prepare_q3_operator.py` 只生成校准请求与提交意向，不运行实验、不调用 SSH 或 sbatch。等 candidate06（或最终选定 candidate）的 CPU 验收、开发证据分析和配对分析冻结完成后再使用。实际选择锁文件名是 `Q3_SELECTION/SELECTION_LOCK.json`。

## 先准备并审核

把准备器与 `docs/modeling_v3/CPU_SEED_INVENTORY_BASIS.json` 的原件送至服务器由操作人执行。以实际部署回执填写下面占位值；不沿用 candidate05 的快照哈希：

```sh
/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python prepare_q3_operator.py \
  --mode prepare \
  --candidate-root /projects/varunssd/louis-ssvc/modeling_v3_20260915/candidate06 \
  --checkout /projects/varunssd/louis-ssvc/modeling_v3_20260915/candidate06/checkout \
  --snapshot-sha256 VERIFIED_FULL_SNAPSHOT_MANIFEST_SHA256 \
  --inventory-basis /ABSOLUTE/SERVER/PATH/CPU_SEED_INVENTORY_BASIS.json \
  --inventory-basis-sha256 VERIFIED_BASIS_FILE_SHA256
```

准备器逐一核验完整代码清单、配置、选择锁及其开发证据、四个历史窄根的 identity 文件哈希。仅扫描训练身份与文件路径，不打开测试响应。新 Q3 根改为所选 candidate 的 `Q3_campaign`，并在准备回执中明确记录它与 basis 文档原 candidate05 路径的对应关系。

新目录 `Q3_OPERATOR_INTERVAL_CALIBRATION` 保存全部 20 个请求、唯一有序 `REQUEST_LIST.json`、`CALIBRATION_MATRIX.json`、`SUBMISSION_INTENTS.json` 和 `PREPARATION_MANIFEST.json`。记录打印出的 preparation 和 list 两个 SHA256 后，审核完整清单。每个请求含一个校准 seed、两个 arm、三个 anchor、20 次观测重复；模型、秩和选择器来自冻结锁，不在准备器中另行选择。原件输出位于 `Q3_campaign/interval_calibration_seed<SEED>/interval_calibration/{collection,response}`。

已有非空 Q3 根或准备目录会被拒绝。失败留下的准备原件也不自动删除、覆盖或续写。若原实验需要恢复，另行审核精确 resume 请求，不能修改已封存清单或重新采样。

## 分批提交由操作人完成

每次提交前，用同样参数改为 `--mode verify` 并附 `--prepared-sha256 RECORDED_PREPARATION_SHA256`，重新验证代码、选择锁、开发证据、历史 identity 和整份请求清单。verify 不重扫已产生的校准 seed；每个实际请求仍会重新扫描自己的 seed 冲突。

`SUBMISSION_INTENTS.json` 内的命令仅供审核后逐批执行。固定 account 是 `rose`，QOS 是 **soujanya-poria-startfund-2026-03**；CPU partition/constraint 和环境由现有 CPU array 模板执行。GPU 可见性为空，数学库线程数为 1，离线模式开启。没有手工指定 CPU 数或内存，也没有 GPU 请求。

完整列表一次封存，提交顺序是 `[0]`，随后 `[1–5]`、`[6–10]`、`[11–15]`、`[16–19]`。先完成首个真实校准 seed、核验原件并获取完整运行时间，再推进余下批次。MaxSubmitPU=5 包括排队任务：必须先检查该 QOS 当前已提交任务数量，确保本批全部 indices 放得下；不能把所有批次用依赖一次性塞入队列。准备器不会自动检查队列或提交。

每个 seed 有 2 条轨迹、128 次源更新、648 次候选 Adam 更新、6 个 origin、120 个 origin/repeat 单元。设锁展开为 D 个设计，则每 seed 运行 120D 个响应拟合，完整 20 seeds 是 40 条轨迹和 2400D 次拟合。直接测量与真实参考由现有数值实现保留。实际端到端耗时与磁盘量须等待首 seed；已有采集或 Q2 小矩阵 timing 不能替代这一测量。命令里的 `1-00:00:00` 是调度 wall limit，不是实测估计或实验预算。

所有 20 seeds 完成并通过原件核验后，才能用真实完整响应集合生成校准回执。30 个 locked-test seeds 和 40 条正交泛化轨迹必须依赖该真实回执；本准备器不会生成这些后继请求，也不会创建 GPU 执行授权。
