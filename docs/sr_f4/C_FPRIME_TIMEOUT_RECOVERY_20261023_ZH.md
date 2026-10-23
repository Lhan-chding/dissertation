# F′六小时时限退出：补齐调度恢复标志

2026-10-23 22:15 UTC检查发现：GPU租约222207、222208分别在21:47 UTC到达六小时时限，Slurm记为TIMEOUT。控制器222206仍ACTIVE，已自动补交222584等待资源；222209上的两条训练继续执行。

受影响的五项工作为b1 SELECT评测s0/s1、b1/b2 SELECT_SHUF训练和b1父模型评测s0。控制器已确认原租约离队并释放任务lease，却将这些任务记为FAILED，原因为 `lease ended: TIMEOUT`。根因是F′准备程序漏填 `resume_supported=True`；训练/评测执行器本身已有checkpoint及回答缓存恢复逻辑。此次是调度注册缺陷，不是新的科学失败。

修复为准备程序使用独立 `pool_task_matrix()`：给12训练和60评测登记可恢复能力，科研 `task_matrix()`、TASKS.json和报告任务不变。现有运行用独立恢复程序 `sr_f4.c_pilot_recovery`，先验证冻结和原池spec，仅在pool/STATE.json中补该调度字段；记录 `RECOVERY_TIMEOUT_RESUMPTION.json` 和原始spec。FROZEN.json、CONFIG.json中的初始版本、checkpoint身份、数据、随机种子及回答预算都不变。

本次FAILED恢复只接受原调度器已明确记录的TIMEOUT：重新查Slurm，核对job_id、教师QoS、account rose、池登记用户，确认作业已离队、终态TIMEOUT、任务lease释放、状态和attempt没有被其他进程改变。保留原FAILEDattempt及终态记录，追加recoveries后置PENDING。不得自动恢复OOM、其他退出码、取消或未知状态。已有提交步的训练若缺少latest.pt则拒绝恢复，不从头重训。

原控制器已经支持该标志，会从共享状态读取；因此无需重启控制器、取消正在训练的GPU或修改既有任务脚本，也不额外提交作业。训练从latest.pt中的参数、优化器和RNG继续，评测复用已完成group。恢复代码commit并push后，单独部署并运行：

```bash
PYTHONPATH=/path/to/fixed-code/src python -m sr_f4.c_pilot_recovery \
  --root /projects/_ssd/varunssd/louis-ssvc/sr_f4_c_fprime_pilot_20261023 \
  --revision FIX_COMMIT
```

必要CPU回归覆盖注册标志实际进入pool、科学任务不变、已完成回答/checkpoint/失败attempt保留、排队或未释放lease不恢复、非TIMEOUT和用户不符拒绝，以及修复后原控制器在下一次TIMEOUT自动置PENDING。没有额外GPU烟测或哈希扫描。
