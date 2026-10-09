# ENGINE 概率一致性技术失败与修复

2026-10-09 13:48:17（新加坡时间），老师 QoS 下的 ENGINE 作业194796以
`FAILED 1:0`结束，GPU分配468秒。控制器194794及后继194795按技术阻断规则退出，
76个后续科学任务均未启动。此前13:41的RUNNING记录是有效的历史快照。

第一步已保存192条完整回答、4957个token；参数更新次数为0，最新checkpoint仍为step0。
采样raw logprob与带梯度teacher forcing的平均绝对差0.0029789611 nat，
最大绝对差0.5841155052 nat；26条回答中的63个token超过最大差门槛。
原合同要求平均≤0.005、最大≤0.05；平均通过而最大失败。
没有完成自然非零梯度/恢复验收，也没有可发布的科学结果。

## 技术定位与范围

原训练采样按冻结配置使用`use_cache=True`，经过原生prefill和逐token缓存解码；
原teacher forcing继承`mm_core`，整段completion一次前向且`use_cache=False`。
这使Qwen3.5的原生linear attention分别走递推与分块计算，构成首要待实测验证的原因。
位置、token切片和EOS的代码审阅未发现错位。

按合同7.4修复F2训练current/reference的复算路径，使其使用原生generation的位置与
缓存接口逐token读取已知答案。训练cache只改变中间状态的存储所有权，保留完整历史
计算图，避免原生推理cache的原位覆盖；不detach早期状态、不更改模型内核。
PROBE沿用原审计的整段teacher forcing，`mm_core`保持原样。
每序列另保存实际prefill/decode/model/vision调用成本，异常也记录。

不改变模型、BF16/FP32精度、奖励、采样seed、`use_cache=True`、学习率、步数、数据或
0.005/0.05验收阈值。该改动需要重新冻结代码、重新完成整个受影响ENGINE；tiny模型
概率/梯度回归不能替代真实9B GPU验收。

## 证据与重跑边界

- 原失败根：`/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos`。
  原代码、freeze、192条raw回答、step0权重、Slurm日志和账本全部保留。
- 原失败数据包：`ops/numeric_gate_incident/ENGINE_FAILED_194796_DATA_ONLY.tar.gz`，
  SHA256 `822db16b2432960c165b0e58c5ac4c300854063a030d22321c11e6555400e7d9`。
  不含checkpoint张量正文；张量仍在原根保存。它不是恢复验收通过或完整实验交付包。
- 独立诊断194828使用同一老师QoS，将192条原回答的token强制回放，并对比原生缓存
  raw logits、原整段复算与原old logprob。没有新采样答案、没有optimizer更新。
  诊断代码commit `b1908c5`；结果尚需以实际SUMMARY和Slurm终态为准。
- 修复运行根预定为
  `/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos_cachefix`。
  重新核对并复制同一4692个数据/QA文件，独立freeze/registration，不覆盖失败根。

最终报告必须分别保留原取消队列、失败ENGINE、只读诊断和修复运行的物理成本；
不能只把最后成功根的GPU记账当作全过程费用。旧排队194700为0 GPU秒，失败194796
为468 GPU秒，诊断194828的实际费用应从终态Slurm回执补齐。所有不利记录随最终
证据包交付，不根据正确性选择重跑样本。
