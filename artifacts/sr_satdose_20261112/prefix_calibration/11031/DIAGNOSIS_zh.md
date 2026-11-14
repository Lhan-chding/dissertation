# PREFIX 校准暂停诊断（2026-11-14）

结论：完整48条已生成，存档/拼接/评分与冻结规则一致；未发现可修复后将本轮判为通过的实现错误。状态保持 HOLD_PREFIX_MEASUREMENT / NO_TRAINING。

## 已完成核验

- 服务器原734e921发布清单、原冻结部署和来源SHA核验通过；使用原评分器对完整48条CPU复算，全部报告字段与原控制器结果完全相同。
- 下载153成员的原始证据包，大小2,296,093字节，SHA256为55e24dc67f0bdf2e2f49f877b5038c8993186a6b37b42f58a34737777ffadd1c。原raw SHA为e577eae1a4ab138ec885806822ead19d545bdef0dfda762fe8ddbe2475f89204。
- 独立标准库严格JSON解析（拒绝重复键/非有限常量）逐条复算48条，与strict/sensitivity两套原始评分逐字段一致；原始BEGIN/CAPTURE/COMPLETE链、48条token/sampler长度、冻结输入/种子及prefill+suffix拼接一致。
- 独立审查重算原生括号扫描，对照四个相关native/legacy源文件SHA；3项已有解析器测试通过。未执行新模型生成、GPU回放、参数更新或重提。

## 指标

| 口径 | STABLE可解析/正确 | REVTIE可解析/正确 | PLACEBO可解析/正确 |
| --- | --- | --- | --- |
| 严格解析 | 8/16；8/16 | 8/16；0/16 | 7/16；7/16 |
| 限定EOS补一个右括号 | 15/16；15/16 | 13/16；1/16 | 15/16；15/16 |

限定口径下STABLE正确数15≥12，STABLE−REVTIE正确数差14≥4；REVTIE可解析13<14，因此整套门禁仍失败。两套口径均未通过，没有混选指标。22条EOS、26条括号平衡停止、0条长度截断。

## 剩余5条未知回答的原因

1. root0000 / REVTIE：99 tokens，EOS。追加唯一一个`}`后JSON有效，但没有`selected`字段。bindings或work中出现类别不能替代预登记selected字段。
2. root0011 / REVTIE：1234 tokens，EOS。work文本含未转义引号/换行并继续嵌入另一段JSON；追加一个`}`仍不能严格解析。不是1536-token上限截断。
3. root0013 / 三个视图：每条原始序列都只有1 token，即`}`。与末尾带逗号的固定前缀拼接成`],}`，括号已平衡但JSON语法不合法。原生规则据括号平衡停止，未丢弃后续token；三个prompt哈希不同，共用该root的冻结种子符合原manifest。

6条REVTIE回答可通过补括号恢复JSON语法，其中root0000仍缺selected，故可解析selected为8+5=13，不能计成8+6=14。

## 对后续执行的影响

原计划CODEX_EXPERIMENT_PLAN_zh.md第190–200行限定唯一补括号和每view至少14/16；第242行规定HOLD后只读诊断/技术恢复。本次未找到可恢复的实现偏差，现有预登记下维持暂停，11032尚未生成，GPU技术绑定未创建，TRAIN/DEV/MAIN未开始。

若继续推进，需另行明确并登记PREFIX测量方案的修订及适用范围，保留本次失败和原始回答；当前诊断不授权重采样、替换样本、补selected、改变停止规则或放宽门槛。此结论仅说明本次存档与固定规则一致，不证明模型/模板一般不存在格式脆弱性，也不构成机制确认性结论。

证据：PREFIX_REANALYSIS_20261114.json、PREFIX_READONLY_DIAGNOSIS_20261114.json、PREFIX_CONTROL_RESULT_20261114.json、PREFIX_EVIDENCE_20261114.zip。
