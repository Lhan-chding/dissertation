# D1 可观测性

逐题状态、频数、reward 协方差及精确有理关系联合计数见 endpoint_states.json。未知尾部未被当作确定事件；零 X 样本不能证明 pX=0。

NOT_RUN：无新端点观测。

完整性问题：`[]`。
评分不一致只禁用对应 LR/质量解释；合法端点采样仍可单列分析。质量界未覆盖数值误差时仍为 CONDITIONAL_ON_SCORER / NOT_CERTIFIED。
