# 既有真实证据只读适配（M5）

状态：`PARTIAL_REAL_EVIDENCE_READY`。本轮只读本机已取回产物；不读取 sealed confirm，不加载 checkpoint，不调用 Qwen 或 GPU。

本机可复核完成 bank：bank00, bank03, bank04, bank05, bank11。独立样本 7680；统一表 748 行；本轮语义重新计算 7680 条。
相同策略 alias 与 direct/flat ledger 副本均不增加独立样本量。每个缺失字段使用 null + reason。

| bank | 独立样本 | 语义复算 | ΔpX (joint_1 − joint_0) | Δv | ΔqX |
|---|---:|---:|---:|---:|---:|
| bank00 | 768 | 768 | 0.0 | 0.0 | 0.0 |
| bank03 | 2304 | 2304 | -0.00651041666666663 | -0.00390625 | -0.0038250645361873614 |
| bank04 | 1536 | 1536 | -0.0026041666666666297 | -0.00390625 | 0.00023901276695881268 |
| bank05 | 1536 | 1536 | -0.00390625 | 0.0 | -0.004070556309362372 |
| bank11 | 1536 | 1536 | 0.0013020833333332593 | 0.0026041666666666297 | -0.0006245483345692637 |

## 可支持的结论

上述为已完成候选间的实测比例差，qX 按所有有效样本加权后取比。joint_0 自身执行了一次 Adam 更新，不能把它当作原点分布。点估计没有添加未计算的显著性结论。
历史 R3-cold/R3-warm 的 bank 0/6 与新 S1 bank00/03/04/05/11 属于不同实验；历史 R4 是单独训练证据。历史数据在本次仅导入状态和清单元数据，未声称重算其原始输出。

## 缺项与边界

本机紧凑包没有导入绑定参数布局的实际更新向量；现有范数、梯度哈希、参数哈希和几何摘要均不能代替向量，故真实 J、夹角和响应维数保持未知。服务器上可能保有 checkpoint；本次本机审计不能推断服务器原件缺失。
原点/候选/Adam 哈希仅从已校验文件读取；未加载原始张量，因此没有把张量哈希元数据称为独立重算。未核验 LoRA 参数顺序或 tensor dtype。
新 S1 仍只有一个 warm 原点（step64）及 seed17；不支持拆分独立训练 seed 的拟合/验证，也不支持跨 checkpoint 泛化。
本机 composite 最新快照仅为状态参考；没有本机完成原件就不导入部分输出，不以 RUNNING、样本计数或提交回执作为完成。

## 最小已有原件导出清单

- Ordered LoRA parameter names/shapes/dtypes and layout hash
- Actual realized update vectors or lossless parameter deltas bound to origin/candidate/optimizer hashes
- Matched origin distribution on identical probe panel and sample budget
- Completed composite evidence with manifest and alias-aware per-prompt raw/count exports
- Multiple independent training seeds and checkpoints with preassigned fit/selection/test roles

上述是所需数据导出清单，不授权新 GPU 测量、训练、采样或控制。

文件和字段来源见 `real_evidence_table.jsonl`；字节哈希、各 bank 完整性检查、遗漏清单、状态快照和时间见 `audit_summary.json`。相对 evidence_id 是来源标识，不包含私有绝对路径。
