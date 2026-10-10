# 真正计算并行修订操作记录

对应源码 be406da245c7acb39692265d6819291406a917cb，运行根 sr_f11_20261010。

这些脚本是已执行/已部署操作的归档，不是可重复一键重跑脚本。所有 INTENT、SUBMISSION、RELEASE、ACTIVATION 回执不可覆盖；有意图但结果未知时先核实 Slurm 和文件状态。

- compute_cancel.py：在 CPU 验证和维护前快照通过后，只取消身份匹配的旧 CPU196654/GPU196508。已消费。
- probe_submit.py：preserve 已提交 CPU196791并通过；probe 已提交老师QoS四卡196798。两项意图已消费。
- probe_release.py：核查GPU型号、真实主机内存、教师QoS五卡容量，再释放196798。已消费。
- compute_arm.py：只申请CPU的交接作业196802，已放行。不可再次提交。
- compute_handoff.py：等待已登记GPU探针成功且退出队列，依次调用后两项；失败/UNKNOWN不伪造通过、不重复提交。
- compute_activate.py：双GPU探针、全部源码/冻结/证据和终态核实通过后，绑定新修订并保留旧源码，再启动CPU预检与一次性恢复授权。
- compute_resume.py：CPU预检通过且恢复令牌未消费后，启动新普通CPU控制器。ENGINE之后的基线并行包含在此源码，旧731候选不得再激活。

2026-10-10 16:58新加坡时间：196798 PENDING(Resources)，196802 RUNNING且WAITING_FOR_VALIDATION。尚未执行新生产激活，尚无真实GPU数值通过或加速倍数。源码61文件和1668保留证据逐项哈希已复核；未改无关作业。
