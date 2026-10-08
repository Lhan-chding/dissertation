# MM-DEV草案：Pro审阅版v2

状态：`READY_WITH_OPEN_DEV_FIELDS`；`authorized=false`。本版只补工程证据，不选择尚未规定的科学参数。
对应[机器可读草案](DEV_FREEZE_PROPOSAL.pro_review_v2.json)。原[事实补录v1](DEV_FREEZE_PROPOSAL.factfill_v1.json)保持原样。

原答案奖励GRPO的恢复比较通过，但梯度全零。新授权的非零监督SFT恢复比较与独立checkpoint复核均通过，
详见[最终工程报告](../NONZERO_RECOVERY_RESULT_20261009_zh.md)。这个新增结果不验证非零GRPO或训练收益，
不替换未经训练的公共起点，不给MM-DEV自动放行。v1中“非零尚未验证”是当时记录，当前范围以本版为准。

原文已经固定Hp/H=32/32、4准备、5起点、30响应、1088次逻辑更新及三种动作配额。
这些科学训练均未执行。用户条件继续为Qwen3.5-9B、最多五张Pro6000并发、GPU小时只记账。

仍须Pro给出或确认：完整reward公式/精度与准备β、KL/G/batch、LoRA与优化器/lr/loss配方、
新独立数据根和seed矩阵、槽位余数策略、probe/eval规模与K、目标/保护权重及成本单位。
K=4存在源稿依据，但面板配置需按最终计划明确签收；不能把ENGINE参数直接升级成科学默认值。

请结合[参数出处与待决表](NEXT_STAGE_DECISION_TABLE_zh.md)和[给Pro的审阅请求](PRO_REVIEW_REQUEST_zh.md)，
返回可签收的冻结补充表。32步无信号只能报告本时长证据不足，不自动延期。
用户拿到审阅意见后再决定是否授权下一阶段执行。
