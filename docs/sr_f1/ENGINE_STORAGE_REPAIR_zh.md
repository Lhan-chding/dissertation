# SR-F1.1 集群主机内存与暂存修复

## 实际调度结果

显存修复源码 `6d764aaf568649eceb7766f12c51cf8ef2c7bb7b` 已部署，CPU复核196130完成：417项测试、6项子测试以及冻结身份检查通过。控制器196138随后提交ENGINE196139（attempt0001），请求单张PRO6000、老师QoS和384GiB主机内存。

实际提交回执明确记录：请求393216MiB，集群 `slurm_job_submit` 强制改成33792MiB，即33GiB。`scontrol` 的 `ReqTRES` 与此一致。[集群官方Slurm指南](https://github.com/NTUEEECluster/docs/blob/main/slurm-guide.md#rules-checked-at-submission)说明GPU绑定固定CPU及主机内存，`--mem`会被覆盖；PRO6000可用 `highmem` 特征获得更大主机内存，资源变更需取消并重提。

为避免低内存作业启动，仅取消本项目CPU控制器196138和仍未启动的GPU196139。GPU最终为 `CANCELLED`、ElapsedRaw=0、Start=None；无worker进程、无恢复激活标记、无优化器更新。132项预置恢复文件（128条回答、RUN_MANIFEST、完整第0步检查点及两份提交指针）逐项哈希保持一致，服务器 `HOST_STATE_REVIEW.json` 为 `PASS_UNCONSUMED_ZERO_UPDATE_STATE`。339项原始调度与恢复文件和原源码另行保存在 `technical_incidents/host_memory_20261010/evidence/` 与 `code_before_engine_storage_20261010/`。第一次审计脚本只接受Start=Unknown，遇到实际None后未修改状态；保留原脚本，第二版接受两种未启动表示法后完成核验。

## 当前技术实现

- 单GPU继续使用老师QoS `soujanya-poria-startfund-2026-03`，添加 `--constraint=highmem`，不再传入无效的GPU `--mem`。每次释放已提交的挂起作业前，核验实际ReqTRES至少80GiB、highmem特征、单张PRO6000、账号/QoS及作业身份。失败或无法确认则保持挂起并保留原始调度响应，未知提交不重提。
- CPU保存的反向激活限制为48GiB；超出部分按原dtype和原storage布局，无损写入运行根下私有临时文件。每次梯度forward最多一个追加文件，32MiB分块传输，64MiB同步并请求回收文件缓存。反向恢复检查SHA256；短读、短写、磁盘不足或校验失败均作为技术错误停止。参数storage仍驻留GPU并检查版本。
- 原形状、stride（包括零stride和重叠视图）、完整可微缓存及两次反向流程保持。第一次保留图的反向后临时文件仍有效；最后一个保存记录释放后删除该派生暂存文件。清理仅针对本次新建暂存，不触及原回答、检查点或故障证据。
- 独立 `ENGINE_STORAGE_REPAIR.json` 追加绑定前次 `ENGINE_MEMORY_REPAIR.json`、源码快照、未启动取消证据以及新源码。原冻结、QoS修订、第一次内存修订、旧失败均不改写。ENGINE恢复授权只允许新attempt0002，之前两次attempt和已消费的调度授权继续保留；GPU进程层的第0步复用授权尚未消费。

五臂×三seed×96步、lr=1e-4、768-token上限、格式评分、奖励、数据及TEST释放条件不变。格式确认已完成500/512、PASS_95；ENGINE仍需真实GPU连续4步对独立2+2验收，科学训练尚未开始。实际部署及调度进展以独立验证回执和实时注册为准。
