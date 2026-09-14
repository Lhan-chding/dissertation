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

服务器重试使用修复提交的独立冻结 checkout 和新的结果根。重新生成完整 CPU 计划后，按用户选择更早可用资源的授权，将尚未运行的 5090 替代作业取消，改用老师特殊 QOS 的单卡 PRO 6000 验证。首次成功 CPU 与失败 GPU 的原件保留。

## 修复后的实际服务器结果

冻结源码提交为 `154599ced9a0826997bdc69b2ed6ac4943e863bd`，服务器部署的 73 个生产源码文件哈希全部匹配。

| 作业 | 设备 | 最终结果 | UTC 起止时间 | 用时 |
| --- | --- | --- | --- | --- |
| 153965 | cpu-2，无 GPU | 完整原件核验通过，COMPLETED，退出码 0 | 11:55:20–12:17:38 | 22 分 18 秒 |
| 153966 | 原申请单张 RTX 5090 | 开始前取消，未执行 | 未开始 | 0 |
| 154001 | 单张 RTX PRO 6000 Blackwell Max-Q Workstation Edition | smoke PASS，COMPLETED，退出码 0 | 12:17:39–13:34:15 | 1 小时 16 分 36 秒 |

GPU 作业在 CPU 核验结束后 1 秒开始。实际仅可见一张 GPU；设备总显存为 101,971,460,096 字节（94.968 GiB）。PyTorch peak allocated 为 24,173,737,984 字节（22.514 GiB），peak reserved 为 25,421,676,544 字节（23.676 GiB）。这里记录的是该次 smoke 的进程峰值。

CPU 冻结计划与本次 GPU 重建的完整计划相同，canonical SHA-256 均为 `eddfb5a469807cd2761208ffc39d78908ca204cd8b1d6ae4bf1ebcdf0163bcb5`，文件 SHA-256 均为 `8599e971587c1673edf9a037836ea07239d30ab32a60697082fe75f80a08c95d`。原始数值门禁已重新执行，未放宽阈值。比较回执 SHA-256 为 `214e39cf46adc2ec9c5724f4ab01de5b708865cd489947268fa865841cdd0311`。

真实 smoke 覆盖 warm bank03 的 joint λ=0 和 λ=1，共 4 次 scratch Adam，新增采样为 0，未启动永久训练。两组参数、Adam 的旧／新哈希均相同，最大数值差均为 0；optimizer structure 和 RNG/scheduler/metadata 比较通过。完整 origin 恢复及冻结基座未变检查通过。两组完整候选状态的 `bitwise_equal` 均为 false；该字段包含更多状态，现有证据未给出其逐字段差异，不能表述为完整候选状态逐位一致。本地对取回原件重新调用 `_smoke_gate`，核对准确计划与全部 73 个源码文件后通过。

| 完整私有原件 | SHA-256 |
| --- | --- |
| `smoke_evidence.json` | `c623957dd596614acf358e443ce335d4085341b7be6f0b8fe83b35324703dbb8` |
| `smoke_pro6000_memory_154001.json` | `562cfec24ffa017f57b4e98d16c5a9ebdce60d412b4ac19bf088a6ee562c4b3e` |
| `smoke_exit_154001.txt` | `cb9128843df621f6188ab46894c0332802a45df5b8191e925dbd5ffa6790e68c` |

原件、stdout/stderr、调度结果和紧凑复核记录已保存至本地忽略目录 `runs/mechanism_followup_server/20260914/retry_154599c/pro6000_route/`。日志包含已知的 gated-delta 快速路径缺失、使用 torch 实现的提示；未修改冻结运行环境。该提示及过程中的瞬时利用率不能单独确定总耗时的主要原因。

## 资源选择与下一阶段

2026-09-14 13:41:54 UTC 刷新 Slurm `--test-only`，同为单节点、24 小时申请：特殊 QOS 双 PRO 6000 预计 14:06:54 UTC 开始，rose 三 RTX 5090 预计 23:10:54 UTC 开始，前者早 9 小时 4 分。这些是可变化的调度预测，不是已提交作业或完成时间预测；没有同负载的双 PRO／三 RTX 吞吐实测。结合已完成的 PRO 验证，保留特殊 QOS 的 PRO 路线。

截至上述 13:41:54 UTC 资源复核时，仅完成 CPU 和单卡 smoke，未提交双卡实验、S1 或 S2。用户随后授权进入 S1 并使用两个独立单卡作业并行；实际启动记录见 [S1_EXECUTION_20260914_zh.md](S1_EXECUTION_20260914_zh.md)。每个进程只可见一张 GPU；不为等待后续门禁而闲占第二张卡。下一门禁仍是完成六个 S1 bank、检查各自原始结果并生成实测技术链；随后才审核 S2 技术链、预算与训练授权。smoke PASS 不替代这些门禁。

## 容量结论的边界

本次 PRO 6000 单卡 smoke 已通过，但只检查 warm bank03 的旧／新 joint 路径及恢复，不能替代全部 S2 训练的峰值验证或 RTX 5090 上的实际验证。历史 R4 已保存的 PyTorch 峰值为 allocated 28.761 GiB、reserved 29.947 GiB。多卡显存不会因启动多个独立进程而合并；不能据本次峰值声称三张 5090 已完成全流程容量认证。证据中的 `safety_status` 仍为 `NOT_CERTIFIED`，本次没有新增研究结论。
