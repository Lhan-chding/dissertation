# Q0 原件审计与复算

状态：`MISSING_INPUTS`。

核验文件 4675 个；原点 96 个；从原始付费数组重放 packet 1296 个。模型指标复算 86016 行，已对照原始指标 110208 行。

raw 的 v 为 X/S/W 三项之和；Helmert 还原后的 v 为 -I。两种公式只对零和差值等价。all/active 的活跃判据来自原始 fit 增量；inactive 单列为 all 减 active 的平方和。r=k 比较只与同观测、n、校准量、正则及噪声副本的 C3 比较，不作为真实截断收益。

原始来源及哈希见 PARENT_RECORD_INDEX.jsonl。缺失项见 MISSING_INPUTS_V3.json；任何缺失原件均未从摘要或新采样反造。旧公开 seed 仅属 development。本次没有训练、概率评分或采样；自包含 Q1 数学验收可继续。

完整性错误 0 项；缺失分类 3 项。Q0 不能认证在线 SSVC 有效。
