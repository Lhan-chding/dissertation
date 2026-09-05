# P2

Status: `PARTIAL`

```json
{
  "scope": "toy_mathematical_validation",
  "rows": 420,
  "config_hash": "40c723c1bced1a5c5153416ec30149cb86a52393c8e4306daa49864de19db9e7",
  "formula_provenance": "self_derived_explicit_definitions_theory_review_source_unavailable",
  "H_K_definition": "E[U], U=(N/K)*group_zscore",
  "Xi_K_definition": "E[U U^T] (not covariance)",
  "B_eta_K_definition": "E[softmax(log p+eta U)]-p",
  "q_aggregation": "expected_q_next=E[q(p_next)]; pooled_q_next=q(E[p_next])",
  "ode_comparison": "first-order local ODE Euler prediction; second-order one-step stochastic Taylor",
  "meanfield_interpretation": "O(1/K) is asymptotic order, no required finite-scan slope",
  "normalization": "sequence-score group average; training eta must account for fixed Lnorm=64",
  "pending": [
    "exact THEORY_REVIEW nonclosure construction source",
    "all real-model GPU experiments"
  ]
}
```

## 数学口径与未完成项

这些结果仅为 CPU toy 数学与实现验证。

本地自推定义: U=(N/K)A, H_K=E[U], Xi_K=E[UUᵀ]; Xi 不是协方差。有限步计算 E[softmax(log p+ηU)], 不是 softmax(log p+ηE[U])。同时报告 E[q(p⁺)] 与 q(E[p⁺]), 两者不能互换。

一阶比较为 ODE 的局部 Euler 项; 二阶为随机单步更新的 Taylor 项。K≤32 使用完整枚举; 64/128 使用独立 Monte Carlo 并保存种子、样本数和标准误。O(1/K) 是渐近量级, 未强制拟合斜率等于 -1。

THEORY_REVIEW 未提供, H_K/Xi/B 的符号对应与其指定 θ=±1 构造尚未核对。另附同一非线性共享策略的自推非闭合构造, 不能称为已验证原文反例。此缺失验收项使 P2 保持 PARTIAL。所有模型实验待 NTU GPU 环境。
