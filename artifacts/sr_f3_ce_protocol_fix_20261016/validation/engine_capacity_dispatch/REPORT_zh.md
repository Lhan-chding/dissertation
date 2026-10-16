# ENGINE 空位派发本地验证

辅助 dispatcher 已补齐 ENGINE 阶段：真实 P0 完成后，通过冻结 Controller.submit("engine") 补交尚未登记的四个固定 ENGINE 任务。每次提交仍走原有 hold、真实资源与脚本核验、重新核定配额、release 链；每轮最多尝试四次，遇到无容量即等待，遇未知、held 或失败即停止。

独立审查发现旧 Controller 允许 LOCAL_P0。现已在整个无科学冻结的共同入口补严：唯一真实 COMPLETE P0 attempt 必须通过原 Controller.verify_job 全链核验，并且现场 Slurm accounting 必须与已认证的成功终态逐字段相同。该检查先于任何 ENGINE 补交及 FREEZE_INTENT 写入。LOCAL_P0、账目缺失、失败、非零退出、用户身份变化均在两个阶段阻断。

本地两套相关测试共 132 项通过（5.59 秒），Ruff、格式检查、git diff --check 通过。独立审查对同一源 SHA 复跑 132 项通过（5.05 秒），原样重跑此前失败的 LOCAL_P0 复现也通过：零 ENGINE intent、零 fake sbatch，并明确拒绝。完整命令、SHA 和边界清单见同目录 LOCAL_VALIDATION.json。

本轮仅修改 dispatch_registered.py 和其测试；src 对科学冻结 652e52d530520a276b6420ff0f4ee479dcde6db1 无差异。未操作服务器、未运行真实模型或 GPU、未提交或推送 Git。最终辅助提交、部署到新的 operations/<commit> 目录以及现场验证由主代理负责；不得覆盖原冻结科学源码或旧 operations 目录。
