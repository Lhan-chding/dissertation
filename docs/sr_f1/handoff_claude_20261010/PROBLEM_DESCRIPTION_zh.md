# 问题与已记录数据

## 1. 原本要做什么

用户提供SEMANTIC_REWARD_CODEX_NEXT_20261009.zip，要求实施固定反馈RL实验。模型为Qwen/Qwen3.5-9B，固定revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。五臂A/J/PART/DEC/GATE，各三个seed（71001/71002/71003）、96步。正式RL前须通过统一输出协议、GPU概率/梯度和完整状态恢复验收。

任务是带图表的可执行语义题。模型按原生视觉聊天接口接收一张1024×768图和注册自然语言问题，要求输出一个JSON对象：`evidence`列出series/category/value，`answer`给数值/有理数等答案。程序有世界与查询真值，模型输入不含隐藏世界、qid或文件路径。

格式覆盖条件为 `L_json && L_answer && L_evidence`。它要求整段回答能被严格解析、两字段可评分、entry类型合法；不要求答案正确、实体选对或证据数值正确。解释文字、代码围栏、非法顶层字段、重复键等会导致整段JSON失败。该判据与答案正确率分别计算。

采样固定temperature=1、top_p=1、top_k=0、max_new_tokens=768，按模型实际EOS停止。原生chat template使用`enable_thinking=False`。

## 2. 两次已记录的问题

### 已修复的实现错误

原零输出LoRA与基础模型logits逐位一致；原FORMAT32题×8生成全部完成，但覆盖93/256=36.328125%，22条达到768上限。旧代码在存在任何截断时直接返回`FORMAT_TRUNCATION_REQUIRES_TECHNICAL_REVIEW`，没有在完成技术排除后进入合同允许的一次桥接。该guard触发后GPU195415退出1、控制器195409退出2。

随后进行了只读原生CPU审计：32题路由、256条原始token/slot/seed/概率/身份、parser和停止行为，以及FORMAT/CONFIRM/BRIDGE的全部192个gold completion。最长gold为118 tokens（含EOS），均能在768内表达；即使把全部22条截断假设为成功也只有115/256=44.921875%。原始记录和失败均保留。

修复提交`e61e9de86e6954c94c751b6383b53da68718185b`引入被证据绑定的一次性恢复。旧代码在确认/复测之后才发布新adapter身份；修复后先发布`SRF1_FORMAT_BRIDGED`实际身份，再确认/复测。原冻结未改写，修复通过独立`TECHNICAL_REPAIR.json`绑定旧/新源码及审计。见[修复差异](history/format_guard_repair.patch)和[原修复说明](history/project_docs/FORMAT_GUARD_REPAIR_zh.md)。

### 当前未解决的协议未达标

唯一共同SFT桥接已执行16次实际更新，128个程序gold completion恰用一次。每序列对completion token的NLL取均值，再对8条序列等权；包含EOS，不训练prompt/padding，不加PPO比值、reward或KL。LoRA为full-attention层q_proj/v_proj、r8/alpha16/dropout0，FP32 adapter参数；AdamW lr1e-5、零weight decay、常数学习率。

独立FORMAT_CONFIRM已生成256/256，覆盖100/256=39.0625%，低于95%。其中155条整段JSON/顶层协议无效，1条证据字段不可评分；17条长度截断、239条EOS结束。即使全部截断算成功，上界也只有117/256=45.703125%。正式分母为256，包含17条截断回答。

取证时正在完成合同要求的原FORMAT同题复测，包内只包含当时已落盘的113/256条。before与confirm来自不同面板；完整after面板在该时点尚未生成。

代码将在同题复测完成后因独立确认低于95%返回`ONE_BRIDGE_CONFIRMATION_BELOW_95_PERCENT`。取证时最终回执尚不存在；下游ENGINE、MONITOR、15条科学路径和最终评价仍未启动，TEST封存。当时RUNNING对应原题复测生成阶段。

## 3. 已执行核查与未完成项目

- 256条确认记录的槽位、seed、输入、原始记录hash、冻结采样参数、原生视觉输入路由和encoder调用核验通过；全部generation_status为COMPLETE，technical_validation_errors为空。
- 实际adapter参数hash和发布身份对应SRF1_FORMAT_BRIDGED；独立重评分为100/256。
- 16个桥接更新记录中的损失和梯度范数均有限，参数hash改变，检查点提交标记存在。包内收录128条gold completion、每序列损失/长度、每步梯度范数与参数hash。
- 原before256条及原冻结字节核验通过。此次网络不可达期间没有取消或重复提交作业。
- 本地修复检查记录为276 passed、3 subtests passed；原任务包76个哈希检查通过。小型CPU Qwen测试覆盖FP32/BF16采样概率、cached/full-prefix LoRA梯度、prompt梯度回传及detach反例，fixture使用F2Runtime。
- 当前SRRuntime真实9B/CUDA的ENGINE阶段尚未执行。包内没有该阶段的通过回执；没有真实9B/CUDA逐层cached/full-prefix梯度对照、有限差分或固定gold验证集的桥接前后NLL记录。
- 同题after仅113/256条，完整桥接前后同题统计和最终协议阻塞回执在取证时尚未产生。

## 4. 时间线与状态边界

CPU预检195408完成；原GPU195415/控制器195409因guard阻塞退出；原生审计195666和修复激活195668均成功；恢复控制器195677和GPU195679运行共同桥接及确认/复测。中途本机网络不可达只造成监控盲区，没有证据表明服务器训练中断。

服务器源码为e61e9de；问题说明前的本地证据文档提交为3b02dbc。包内来源身份另见`PROVENANCE.json`和`evidence/CAPTURE_MANIFEST.json`。原冻结SHA256：`b84d6d1ad26761cb07e3db706be0b8ca2b4e4a8d436473b0ad3e964a81f463f9`。

用户此次请求是将问题、数据、错误信息及代码打包交给Claude生成解决方案文件。
