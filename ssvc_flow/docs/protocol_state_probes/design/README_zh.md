# SSVC 协议与状态探针交接包

先阅读 `OVERVIEW_zh.md`；执行以 `CODEX_EXECUTION_PLAN_zh.md` 和 `protocol.json` 为准。

- 主计划：6个旧检查点，最多5张GPU，冻结推理；不训练。
- 实际生成任务：70,656条回答，另有最多80条技术smoke；不是已完成结果。
- 精确提示、结构/程序预测、审计字段与逻辑任务已提前编译。
- `reference` 为CPU确定性参考，不包含已完成的GPU适配器。
- 主内容、统计、GPU接入、验收、来源分别成文，避免把算法假设与运行约定混写。

测试：`cd reference && python -m unittest -v test_contracts`。
具体已执行范围见 `verification/EXECUTION_SCOPE.json`。

请勿把真值、污染位置或DPE审计标签发送给模型；请勿因某个旧端点缺失而默认新训练。
