# Q1 观测统计汇总

状态：DEVELOPMENT_ONLY_NOT_CERTIFIED。

已读取 192 个完整观测单元；缺失指标记录 960 条。原件未修改，未生成新样本。

## 指标定义

- 逐事件与 delta_v=-delta_pI 的偏差、MSE、MAE和分位数均从原始重复估计减去精确真值重算。
- pooled 方差含不同题目的偏差差异；测量方差逐题在重复间计算，再平均。二者分别报告。
- 跨单元分位数从原始残差拼接重算；残差分位数不是置信区间。
- 系数方差、L1、fallback 与修正前后质量残差保存在各单元 SUMMARY.json；crossfit 两折系数分别保留。
- 计时按实际保存的组合块汇总，不拆分推测的生成、评分或标签时间。缺少原始回执时标 MISSING_INPUTS。

## 原始记录

Q1_POOLED_ERRORS.csv：按方法、样本量、proposal、策略情形分别汇总。

units/*/EXACT_RECORDS.npz：完整 repeat x prompt x event 估计、真值、残差、系数及质量残差。
units/*/ORIGINALS.json：被读取的原始文件与 SHA-256。

本报告不认证在线 SSVC 有效，也不将工程完成等同于统计有效。
