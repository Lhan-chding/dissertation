# 固定 BF16 SFT 路径的数值审阅

本决定适用 code `ed1d248`、`run_v2`，不表示旧 prefix 路径与新 full-sequence 路径等价。原始 bridge 顶层 `BLOCKED_TECHNICAL` 和两父 `NUMERICAL_REVIEW_REQUIRED` 均保留；另以哈希绑定的 [接受记录](evidence/bridge_v2/BRIDGE_NUMERICAL_ACCEPTANCE.json) 放行共同训练实现。

## 直接证据

两个父模型各 8 个固定训练题的目标位置及单 EOS 检查通过；实际存在不同序列长度的 PAD 干预不是空操作。future/PAD 干预差值与同路径重复基线均为 0。LoRA 梯度与两次更新的参数变化有限且非零，更新入口检查 base/vision 无梯度。第二次更新的完整状态在断点恢复后完全相同。v1/v2 两次桥接的逐 token、PAD、梯度及更新诊断完全一致。

新增 CPU 门禁回归防止遗漏 prefix 差异、缺失诊断、非有限梯度或因果/恢复失败被旧 PASS 标记或人工接受覆盖；68 项测试和 35 个子测试通过，Ruff 通过。

## 保留的数值差异

| 父模型 | 对照 | BF16 梯度相对 L2 | 同一舍入权重上转 FP32 的梯度相对 L2 |
|---|---|---:|---:|
| S96 | micro4 / micro1 | 0.13573517 | 0.0000290683 |
| REP96 | micro4 / micro1 | 0.18215611 | 0.0000391695 |
| S96 | full / prefix | 0.07649574 | 0.00000261254 |
| REP96 | full / prefix | 0.05164205 | 0.00000295498 |

精度诊断只使用原 8 道桥接训练题：forward 投影覆盖首个 cross 和首个 trend，梯度累积覆盖原前 4 题，full/prefix 梯度覆盖原首题。没有新生成，没有 optimizer 更新。FP32 运算禁止 TF32、采用 highest matmul precision；逐父保存/恢复设置。上转不会恢复原始未量化的 FP32 权重。

固定相同 hidden 的 BF16 全行/单行 LM-head 投影 NLL 差不超过约 4.77e-7，不能解释原来的完整路径差。整体上转 FP32 后，完整路径与 microbatch 差异大幅下降，支持上游低精度运算与序列/batch形状的关联；**单个 kernel 根因未唯一定位**。forward 投影是 no-grad 诊断，另有 train-mode autograd full/prefix 与 microbatch 梯度对照，二者不混称。

## 接受范围

按照给定执行计划 §8.4，接受统一、固定的 BF16 base / FP32 LoRA、full-sequence causal、microbatch=4 作为本轮新 SFT 目标实现。依据是精确目标语义、真实因果/状态/更新证据、跨执行重复性、受控精度诊断和所有臂固定同一设置，而不是任意数值容差或高 cosine。

正式臂均重新加载原 BF16 父模型；不会使用诊断的 FP32 模型、不投影或重标定梯度，也不修改 optimizer、步数、采样或 E/G 评价。较大的 BF16 差异照实保留，不声称误差无害、训练稳定或性能保持。所有发现、拟合、迁移和保持性结论仍等待本轮登记实验。

原始资料：[v1 bridge](evidence/bridge_v1/bridge.json)、[v2 bridge](evidence/bridge_v2/bridge.json)、[逐项审计](evidence/bridge_v2/NUMERICAL_AUDIT.json)、[S96 精度诊断](evidence/bridge_v2/numerical_probe/S96.json)、[REP96 精度诊断](evidence/bridge_v2/numerical_probe/REP96.json)。累计技术桥接生成 8 条回答；精度诊断新增生成 0 条，均低于 128 条技术桥接上限。
