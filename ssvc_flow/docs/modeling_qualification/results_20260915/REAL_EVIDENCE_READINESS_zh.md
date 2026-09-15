# M5 真实 checkpoint CPU 审计最终补充

**实际完成：** CPU Slurm 作业 `155077` 为 `COMPLETED`、退出码 `0:0`；程序用时 17.675852 秒，峰 RSS 752287744 字节（0.701 GiB）。请求及实际 TRES 均只有 CPU、内存和节点，没有 GPU。11 GiB 是调度器默认分配的内存；脚本没有设置 `--mem` 或 `--cpus-per-task`，实测内存低于本次 2 GiB 预算。

本次在独立 checkout 的 CPU allocation 内读取 15 个既有 checkpoint：一个 R4 X_BASE step64 原点、14 个已完成 S1 终态候选。文件大小合计 2,273,190,231 字节。逐文件字节 SHA-256、完整 state、候选参数和 Adam 哈希均已实际复算；没有构造 Qwen 模型、调用生成、产生新训练更新或操作已有实验队列。没有下载 checkpoint。

## 实际参数与几何

实际 LoRA 参数为 **12,582,912** 个、192 个张量，dtype 为 `torch.float32`。五个 bank 的参数名称、顺序、shape 和 dtype 一致；布局 SHA-256：`02fc946aa4fcca575abbad4c0f3f9265f01349276aa12d4b08f8beb8db9c76f2`。

| bank | 已选实际更新的秩 | 相对 joint_0 的辅助位移秩 | joint_0 / joint_1 余弦 | 夹角（度） |
|---|---:|---:|---:|---:|
| bank00 | 1 | 0 | 1.000000000000 | 0.000000000 |
| bank03 | 2 | 1 | 0.893213528504 | 26.720122775 |
| bank04 | 2 | 1 | 0.870265416130 | 29.510502776 |
| bank05 | 2 | 1 | 0.952614918967 | 17.708772673 |
| bank11 | 2 | 1 | 0.997970943850 | 3.650548513 |

秩由实际参数差构成的矩阵经逐张量 TSQR 和小矩阵 SVD 计算，阈值为 `max(1e-14, 1e-10 × 最大奇异值)`；未通过 Gram 小特征值估计近共线秩。14 个候选的实际更新范数均与原记录在 `rtol=1e-6, atol=1e-10` 内一致。重复或 alias 位移未增加数值秩。

bank00 只存在 joint_0/joint_1 两个选定终态；其 no_x_off_1 并非原设计候选。本次未加入其它 lambda 或 replay。bank03 已有 joint_1em2 的额外直接观测仍保留在原始统一表中，但未包含在此次 15-checkpoint 向量预算内。

## 能补齐和不能推出的内容

原本“本机紧凑包没有实际向量”的缺项已得到部分补充：服务器 checkpoint 中的真实参数可在 CPU 重建，现已核验参数布局、实际 delta hash、范数、Gram、夹角及上述子集的更新空间秩。大向量保留在原 checkpoint；统一表提供完整重建配方，`update_vector_path` 保持 null 并标明没有单独向量文件。

新统一表 [real_evidence_table_with_vectors.jsonl](tables/M5_real_evidence_table_with_vectors.jsonl) 保留 748 行，其中 672 行补入实际向量哈希与布局；其余候选保留原有缺项。样本数仍为原来的 7,680，没有因别名或 CPU 读取而增加。

**这些秩不是语义响应维数，也不是 J 的秩。** 本次只有一个 warm 原点及 train seed17，且仅核验指定候选子集；没有原点上匹配的概率分布、多个独立锚点/seed 的验证或全 Jacobian。不能据此推荐真实 Qwen 的全局维数、观测间隔，或声称跨 checkpoint 泛化。

候选 raw/count 的 joint_1 − joint_0 比较仍是候选之间的差；joint_0 自身已经执行一次 Adam，不能把它当作未更新原点。

R4 旧原点的完整 state hash 已复算；S1 recapture 增加字段，其不同的完整 origin hash 仍是记录值。两者的对应通过实际原点参数和 Adam 哈希与 S1 候选 audit before-hashes 相等来核验，没有把两种完整 hash 混为一谈。

## 交付与完整性

- [M5_public_vector_summary.json](tables/M5_public_vector_summary.json)：公开机器摘要、CPU/TRES 证据、各 bank 数值和剩余缺项。
- [real_update_geometry.json](tables/M5_real_update_geometry.json)：实际参数布局、delta hashes、Gram、奇异值、角度和逐 checkpoint 验证。
- [retrieval_verification.json](tables/M5_retrieval_verification.json)：11 个小结果/日志/回执共 659,108 字节，远端与本地逐文件 SHA-256 全部一致。
- [bank_vector_summary.csv](tables/M5_bank_vector_summary.csv)：表格原始数值。

本补充覆盖先前本机 M5 报告中参数布局和实际向量尚未导入的对应缺项；其余单锚点、原点分布和 composite 完成证据限制继续保留。


服务器核验前的本机历史快照单独保留于 [旧准备度快照](tables/M5_local_before_server_zh.md)；其中向量/布局/夹角缺项已由本报告对应结果更新。
