# 独立复核方法与裁决材料的冻结合同

本文件只规定新独立复核的方法，不包含真实回答、旧审核个案、程序标签或审核结果。生产者保持
`6e5d36351b8c1f6f18c61661246191772ba6a6dd`，回答仍为同一批9,216条
`SCHEMA_CONFIRM_V3 / DEV_SCHEMA_CONFIRM`。沿用原seed `2026102701`、360条分层抽样及去重、
根上限、来源绑定规则；不重采、不补题、不改评分，不据新旧结果改变样本。既有审核及其失败
保持原件。新审核必须登记独立来源和运行根，不能覆盖或重签既有审核。

## 唯一发布来源

`src/sr_f6_followup/review_contract.py`提供两个纯函数，不读取回答、运行目录或旧审核：

- `payloads()`返回确定性UTF-8字节：`REVIEW_METHOD.md`、`ADJUDICATION_METHOD.md`、`CONTRACT.json`。
- `specification()`返回版本、生产者身份、冻结政策和三文件`payload_sha256`。`CONTRACT.json`
  只含两个方法文件的`method_sha256`，不包含自身SHA，避免递归身份定义。

新消费者应在任何真实审核输出前，把`specification()`完整写入SPEC并冻结；按原字节发布到
`REVIEW/PACKAGE/method/`，验证每一项SHA。实际审阅与裁决必须绑定并读到相应方法；不能只在
manifest声明一个从未提供给审阅者的方法文件。方法身份与数据阶段输入需分别如实记录。
本说明不是第二份可替代的方法；实际执行以冻结三文件为准。

## 从实际生产调用链确定标签

`confirm_census.annotate`复用`census.annotate_group`。生产者的
`src/sr_f6/census.py:188–194`实际调用为`finished=(raw finish_reason == "balanced")`。
不能脱离入口，只引用`measurement.score_response`的默认参数或内部`balanced/eos`列表。

|维度|必须保留的含义|
|---|---|
|finish|原始停止为balanced才为1；EOS和length为0。它独立于JSON合法性；不能把原始EOS改成balanced，也不能用评分器归一化的length覆盖原始原因。|
|A|一份完整无歧义JSON对象中的answer是有界精确数值并等于独立oracle；不要求V、finish或新增声明正确。|
|V|共同五字段及可选字符串work的格式、数值约束；不检查真值、非空、去重、world成员资格或aggregate是否符合题意。|
|S|完成且对象合法时，独立检查selected；可读则按无重复精确字符串集合判0/1；缺失或不可读为UNKNOWN。|
|B|完成且对象合法时，独立检查bindings；可读则与**真实最终集合**及任务操作数映射比，不跟随错误selected；任何操作数不可读则B为UNKNOWN。|
|J|F6 V与冻结F5 `J_M3`相交。实际F5公式是`F5_V ∧ native_finish ∧ native_value_correct ∧ certificate`，不加入S/B真值或aggregate语义门。|

只允许S/B用null，并逐字段写UNKNOWN原因。finish=0强制S/B为null，但不抹掉独立A/V。
非法/重复键/多对象/带外文本的整个JSON使A=V=J=0、S=B=null，不从片段补救。合法对象中某个
字段格式错误仅使V/J失败及该字段UNKNOWN；其它可读S/B仍单独测量。空数组、重复或错误但
完整可读的声明是语义失败；UNKNOWN不是失败分数，也不能删除分母。

**实际F5包装器没有把legacy `presentation`再作为J门。**原生底层记录的16,000字符work限制
属于该诊断；不得把它额外加到新方法。F5投影本身的64,000字符解析上限仍适用。该细节来自
`src/sr_f5/measurement.py:120–155`，不是新评分放宽；合成回归覆盖了该区别。

## 精确数值及证据

`src/srf6_ref/core.py:18–34`的数值规则同时用于**每个evidence.value、每个bindings.operand
和answer**；不只检查最终答案。布尔值、非有限数、表达式、非法类型及零分母均拒绝。引号
数值strip后最多64字符，完整匹配冻结grammar；数字字符串指数最多两位，且去ASCII空格后的
精确分数转换必须成功。JSON小数另有`abs(Decimal.adjusted()) <= 24`限制，包括零。
所有表示都在约分后要求`abs(numerator) <= 10^12`、`denominator <= 10^9`，边界含等号。
不能用浮点误差、截断小数或统一“位数太长”规则替代约分与类型相关限制。answer:null只在
V语法上合法，本登记非空答案库中A=0；operand/value:null不可读。

原生J证据还要求最多32条、真实world标识、每个cell只报一次、值为world.domain内整数且
逐cell真实；错误额外证据也失败。TOPK要求全部类别的排序列，再要求按这些证据确定的最终
集合的任务取值列。INTERVAL按两端点规则先定范围；positive_only时必须保留过滤前范围
全部Alpha/Beta，包括被排除类别。diff需要两份原始cell，不能用计算后差值替代。
bindings/work里的数值不能补成evidence。原生numeric string还有strip前64字符及原生ASCII
grammar要求，完整区别写入方法；不得用F6 V=1直接推断原生J=1。

## 独立性与裁决可见材料

新盲审核实例只得到冻结方法与匿名原文/协议/原停止；封存完整盲提取后才得到world/query。
提取字段必须给原文精确span，不能让gold指导抽取。新盲审核者不能继承旧审核、实施审计、
生产或程序标签上下文，不调用生产parser/scorer/oracle生成判定。字段、oracle、六标签和
原因由真实独立阅读产生；机械校验只核字节、span、schema和顺序。审阅类型如实为MODEL_TEXT
或HUMAN，平台未暴露的模型名记NOT_EXPOSED，不能伪造。

360条独立judgment全部封存以后，才允许比较程序标签并向独立新实例释放裁决材料。
裁决必须包含全部原claims、blind文本、world/query、封存extraction/judgment、finding，
以及**每个关联匿名回答的完整PROGRAM六标签和answer/selected/bindings oracle**，其来源
逐项绑定冻结ledger/task。这个PROGRAM投影也必须提供给只有REVIEW_AMBIGUITY或
REVIEW_CONTRACT_VIOLATION的finding，不能假设“没列分歧就与程序一致”。没有answer ID的
过程finding使用其完整过程claims与trace，不虚构答案标签。

每个finding_id分别给出实际程序声明、审核声明、适用规则、决定性原文/算术/过程依据和
结论。PROGRAM材料缺失时不能推断或默认PRODUCTION_CORRECT，保留UNRESOLVED或
REVIEW_PROCESS_INVALID。所有决定引用真实可见输入。裁决实例必须不同于每个原审核实例；
原判断、分歧及负决定不改写。无新增一致率门；全量普查90%可观察性门不移植到覆盖样本。

## 验证与认识边界

测试`tests/sr_f6/test_followup_review_contract.py`仅使用手写合成world/原文，不读取任何真实
回答或旧审核。对来源文件核SHA，并通过实际`census.annotate_group`入口检验balanced/EOS、
字段独立UNKNOWN、全字段数值边界、原生证据、tie/端点/有符号值、legacy presentation
区别及三payload身份。测试通过只证明这些实现边界，无独立真实审核或科学PASS效力。

哈希只绑定字节，无法认证隐藏上下文、真实身份或实际独立性。MODEL_TEXT不冒充人工，也不
凭方法文件获得签科学门权限。出现未覆盖的解释、源绑定冲突或任务非空不变量违反，保存
歧义和原件，不能另造有利评分/任务/采样。新审核终态仍须由完整实际材料与既有门核验。
