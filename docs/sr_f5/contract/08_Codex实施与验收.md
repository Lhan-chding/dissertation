# 08｜Codex实施清单：模块、输入输出与依赖

## 8.1 交付参考代码的地位
`srf5_ref/`是已测试的数学/数据/信用参考，**不是已经接通Qwen训练的完整系统**。Codex应把它接入原生runtime，保留gold-blind输入与原始token路径；不能把CPU测试通过写成GPU训练已验证。

原包源码可定位在 `05_SOURCE_REFERENCE/src/`：
- `sr_f4/task_contract.py`：真实充分证据与扩展TOPK/INTERVAL oracle。
- `sr_f3_ce/contract.py`、`prompts.py`：历史精确标量、证据模板、v2协议。
- `sr_f4/c_local_credit.py`：旧P/S抽取、共同门、局部mask与目标。
- `sr_f4/c_local_training.py`：128条累积、保存/恢复。
- `sr_f12/training.py`：实际single-update TIS损失与概率门。
- `sr_f4/runtime.py`：独立draw RNG、模型加载、父adapter身份。

不要覆盖这些旧模块；新增 `sr_f5/`，旧测试与旧数据仍用旧版本。

## 8.2 P0：建分支与身份映射
输出 `IMPLEMENTATION_MAP.json`：本包参考模块→服务器实际新模块；现有调用者→新增协议分支；原commit→新commit；未实现事项。保存机器实际env，不自行pip升级。

保存 `BASELINE_COMPONENTS.json`：J各分量的字段、函数、原条件、v3新增结构条件、从旧score继承/重新实现的范围。明确运行根与旧root隔离。

验收：旧J和旧协议回归全过；原配置路径存在；没有read-gold进入prepare_input；新版参数清单与继承父结构一致。

## 8.3 P1：题组与数据
输入：configs、split seed、family构造器。
输出：`TASKS_PRIVATE.jsonl`、`MODEL_INPUTS.jsonl`、`IMAGES.jsonl`、`ROOT_RELATIONS.jsonl`、`FACTOR_COVERAGE.json`、`SPLIT_AUDIT.json`、`ORACLE_A_B_DIFF.json`、`RENDER_GEOMETRY.jsonl`。

private任务表可被scorer读取，不得被模型输入函数读取。模型输入只包含qid、当前图像、当前prompt和protocol；qid不拼进模型可见文字。提示根据公共题意生成，不用gold_selected/gold_answer。

必须有独立oracle的差分测试，至少包括两列反绑、方向、同值、端点全组合、筛选后集合、负差、mean的分母。未知query字段报错而非忽略。兼容时复用旧renderer，但固定其具体style/字体/图像处理revision；本包demo renderer不是自动授权替换整个生产渲染链。

## 8.4 P2：新协议与J包装
给runtime增加 `structured_v3` 分支，不修改旧`structured_v2/natural_work`行为。

先验证原始对象结构，记录selected字段跨度；再给旧scorer生成**删除selected字段的保真投影**。投影不能修复原本非法JSON、删除重复key或四舍五入Decimal。推荐按顶层原始字符成员区间删除selected及相邻逗号，保留所有其他数值字面和work原文。

J_M3 = V_M3 × finish × A_legacy × certificate_legacy，其中V_M3只含本轮结构要求。P、certificate_truth/sufficient、A与旧scorer定义绑定。`measurement.py`里的sound template只是独立核验子集，不默默替换旧scorer的更广接受范围。

输出 `LABELS.jsonl`，每行包含提取版本、raw text摘要/定位、S/原因、selected span、P/证据真假与充分性、A/V/finish/J、gold来源。

验收：selected从正确改错误且结构不变时，J不变；HARD改变。空/重复/错误标签分别属于已知FAIL，不因格式检查提前被剔除。自然工作面板S=NOT_APPLICABLE。

## 8.5 P3：token与目标
在真实Qwen tokenizer上运行跨度夹具。保存original sampled IDs、prefill及decode设置，映射完成才算`mask_ok`。DENSE与LOCAL使用同K。

为每组保存：组身份、八条row_id、native reward/advantage、S/P/K、c、c_shuffle、permutation及其seed、实际arm系数、mask、各类信号计数。对每一个生成组离线重算并与训练实际传入系数逐值一致。

把 `sequence_loss` 与原生sequence_objective做零辅助前向/梯度等价。HARD须改native reward而非在J之上再加硬奖。DENSE/LOCAL/SHUF都保留原J项且KL一次。

验收：128条等权；mask外辅助logprob差分梯度0；λ只用一次；SHUF层计数不变；全J1/全J0的新增S信号可正确出现；不依据最终A推S。

## 8.6 P4：G2/G3开发执行
按docs/05生成真实回执。每次允许的开发修订都有完整config、原因、累计模型调用数和旧版本归档。不从正式test的分数倒推选题。

新增 `NUMERICAL_TOKEN_AUDIT.jsonl`，至少保留异常top10位置的三套logprob及tokenid。所有概率门在optimizer.step前，异常不提交半个step。开发先导验证恢复和连续信息更新。

## 8.7 P5：冻结与正式调度
`FROZEN.json`登记所有数据/规则/统计小文件hash，外部模型沿用snapshot/parent身份。`RUN_MATRIX.json`包含45个parent-repeat-arm任务、相同parent-repeat内的计划hash、每路径9216回答。

推荐任务DAG：各run生成/更新→最后checkpoint发布→独立评测分片→完整性验收→统计。可以端点完成即评测，但汇总者不得利用中途分数改剩余训练。

主训练末步未到的路径不能标为COMPLETE。正式评测父模型不重复伪装训练seed。缺失端点保持显式缺失，报告不能把实际完成8条当登记9条。

## 8.8 P6：报告和归档
必须交付以下文件：
```
REPORT_ZH.md
SUMMARY.json
FROZEN.json / ENVIRONMENT_LOCK.json / RUN_MATRIX.json
GATES/G0.json ... G4.json
DATA/{TASKS_PRIVATE,MODEL_INPUTS,ROOT_RELATIONS,IMAGES}.jsonl
DATA/{SPLIT_AUDIT,FACTOR_COVERAGE,ORACLE_CHECK}.json
RUNS/<block>/<repeat>/<arm>/{updates,credit_groups,rollouts,index}.jsonl
EVAL/{answers,scores,coverage}.jsonl
STATS/{PRIMARY,TRAIN_REPEAT_SENSITIVITY,RETENTION,RELATIONS,ALL_ARMS}.json
FAILURES.jsonl
SOURCE_MANIFEST.json
```

原文可gzip/zstd但保留逐row索引；至少把总体/分家族/分父模型/9重复效应、非零/可交换/实际改变的时间轨迹及外部保持放轻量报告。最终报告应直接回答原始科学目标，不以“GPU运行成功”结束。

## 8.9 最容易遗漏的集成点
1. 新prompt不能交给只认识旧contract后缀的`question_from_prompt`强行截取。
2. 图像缓存键包含正确world/render/processor身份，不只qid文本。
3. 所有派生问题跨split检查基于root和事实形状，不只文件名。
4. 新selected不改变旧外部ChartQA输入/评分。
5. 两个同时存在的J版本必须命名区分；代码里不允许一个`J`字段混装历史J与HARD奖励。
6. 报告中的“训练seed”指新repeat，不是三个历史父adapter的seed。
7. 诊断梯度、概率重算和重试都计资源，不能污染正式optimizer/RNG。
