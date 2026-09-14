# 2026-09-14 服务器验证记录

本记录追加服务器阶段事实，不替代 `CPU_ACCEPTANCE_RESULTS.json` 对原实现冻结版本的验收。

## 已完成的首次验证

| 作业 | 设备 | 结果 | 用时 |
| --- | --- | --- | --- |
| 153851 | cpu-2，无 GPU | 原始 parent、checkpoint、数据、模型快照及源码核验通过；退出码 0 | 22 分 31 秒 |
| 153868 | 单张 NVIDIA GeForce RTX 5090 | 计划重验指纹不一致；退出码 2 | 12 分 52 秒 |

两次作业使用冻结提交 `e1ce594591453f3970e96fa1dfcd6997fdc0ee05`。GPU 作业实际设备可见数为 1、设备总显存为 33,668,726,784 字节；PyTorch peak allocated 与 peak reserved 均为 0。该次失败发生于模型分配之前，没有 `smoke_evidence.json`，没有启动三卡训练，也不能据此判定显存是否足够。

CPU `validated_plan.json` 的 SHA-256 为 `4e49135b672c0a2482a181d48eaadd0664d4c1b44c2b370ec2e9b1d8cc10c64a`。完整计划及私有失败回执保存在本地忽略目录 `runs/mechanism_followup_server/20260914/`。

## 复核发现及修复范围

旧 `followup_backend._revalidate_plan` 将整个重新生成的计划与冻结计划逐字节等价地比较。计划包含上游报告审计的两个 observer 字段：

- `warm_binding.response_roundoff`
- `warm_binding.direct_validation.roundoff`

`r3_report_gate._compare_response` 已明确允许同版本 NumPy 在不同 CPU 数学内核下产生的有限舍入差，并要求 observer 记录不进入稳定身份。`r3_warm_gate` 返回这些观测供审计。首次 CPU 计划中，前一字段包含 1,075 条差异，最大绝对差为 `1.1368683772161603e-13`；后一字段包含 260 条差异，最大绝对差为 `1.1362438767648086e-16`，全部位于原有允许界限内。

首次 GPU 作业没有保存重建计划，因此不能把代码审查发现表述为已取得该次 CPU/GPU 全部逐字段差异。后续验证必须保留完整重建计划，便于核对真实差异。

修复仅针对上述两个 observer 子树的身份比较。每条 observer 必须符合原格式、允许字段及原有 `64 × float64 epsilon` 界限，且原始核验仍完整执行。父级原件哈希、模型、源码、设计、预算及研究门禁保持不变；重验通过后返回冻结计划的深拷贝，确保 CLI、smoke 和后续运行继续引用同一完整计划哈希。

完整原计划、重建计划和比较回执保存在新结果根的 `revalidation_audits/`，包含文件哈希、身份投影哈希、主机和 Slurm 作业号。写入采用原子发布；并发写入相同内容可以复用，不同内容、符号链接或写入失败均被拒绝。身份不一致时也保留比较证据。

## 修复验证

- 新增回归先在旧实现下得到 16 项失败；最终后端套件 39 项通过。
- 集成执行全部 followup 测试及原 R3 report/warm gate 测试：297 项通过，0 失败、跳过或错误；运行期间源码与测试哈希未变。
- Ruff 与 scoped diff 检查通过，独立只读复核未报告具体缺陷。
- 63 个继承的生产源码文件与原 parent 记录逐一哈希一致。
- 实际取回的 CPU 计划通过新 observer 格式校验；输入未被修改。该检查不构成 CPU/GPU 重建计划的一致性证明。

集成证据位于 `runs/mechanism_followup_cpu/roundoff_fix_integration/`；其 `pytest.log` SHA-256 为 `766f9e8f97570ea17cf026f329ac1ef1438a5c897fc1ea1eff190280c3698662`。修复源码 `src/followup_backend.py` SHA-256 为 `8132e142fc73bc52e42d6e25bc9549a345081386a9c1d0868da7123b4390e90e`，对应测试文件 SHA-256 为 `cc01597de6ddee25506e2018d32689ae5e4fcd5d7f08ccfcc0aec59497d31f7f`。

服务器重试应在修复提交的独立冻结 checkout 和新的结果根中执行：先重新生成完整 CPU 计划，再提交依赖其成功的单卡 RTX 5090 smoke。首次成功 CPU 与失败 GPU 的原件保留；重试的实际作业号、退出码和显存回执写入私有运行证据，不能提前记为通过。

## 容量结论的边界

单卡 smoke 检查 warm bank03 的旧／新 joint 路径及恢复，不能替代全部 S2 训练的峰值验证。历史 R4 已保存的 PyTorch 峰值为 allocated 28.761 GiB、reserved 29.947 GiB。是否采用三张 5090，应以实际单卡路径通过后的显存证据和各进程独立用卡安排为依据；当前尚无单卡 smoke 通过结论。
