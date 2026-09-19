# P/E 面板本地 CPU 核验

状态：`READY`。选择 seed 为 `20260919`；没有调用模型，没有读取 sealed confirm。

- 原 control：144 个 base_scene，每个 family×chart×operation 单元 8 个。
- 重放原 R3 元数据选择器，排除 24 个旧 base_scene；另外排除 preview 六题对应的 3 个 debug 场景，其中 1 个与 R3 重合。
- 最终可选 118 个场景，各单元余下 6–7 个；两面板合计需每单元 4 个，实际缺额 0。
- P、E 各 36 个 base_scene × 2 个配对接口 = 72 prompts。每面板 3 家族 × 2 图类型 × 3 operation × 2 场景，两面板场景互斥。
- train 576 个 base_scene 的 ID 与 control 无交集。所选 72 个图像全部存在，SHA-256 与原场景记录一致。
- 未追加生成场景；未使用模型输出、难度或效果选题。D1 在 P 上的 discovery 是独立 RNG 的输出角色，不得与 endpoint MC 合并。E 仅在开发 recipe/规则冻结后评估。

冻结文件：`configs/decision_modeling/panels.json`。仅含相对图像路径；`data_root` 留空，部署时从 runtime 提供。`config_hash` 绑定完整 `configs/decision_modeling/protocol.json` 的规范化内容，`protocol_binding` 绑定其文件字节；协议若变化必须核验绑定。`inputs_hash` 绑定除自身字段外的完整冻结内容。

输入 SHA-256：

| 文件 | SHA-256 |
|---|---|
| control.jsonl | `bbf1a6200e9814dfa5b2421ff0a2913fce86df6dc95368812302b5f76db75491` |
| train.jsonl | `d47bc1e7c8a9796f5f2fcfeec8fd0f14e18be507f89a0fcaaf65472bb7a33954` |

面板冻结内容哈希：`fa7f3e424a70405b3ddf16b282437e77aa96df5a657defdfe8c972528cf324ce`。

代码测试覆盖确定性/输入顺序不变、单元平衡、P/E/R3/train/debug 分离、输入不可变、缺额报告、禁止读取错误 split、重复身份与来源文件哈希不符。验证命令：`python -m pytest tests/test_panels.py -q`，9 passed。

本结果仅证明面板输入可用；未运行 D1 GPU 观测。服务器需提供同一 train/control 字节与所选图像，并在运行前验证这些冻结身份。
