# Before RL Can Repair Vision
## Qwen2.5-VL 约束吸收、自然状态修正与支持注入式强化学习研究计划 v4.0

**用途**：本文件是交给 Codex 的唯一有效实施规范。  
**固定基础模型**：`Qwen2.5-VL-3B-Instruct`  
**固定本地路径**：`/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct`  
**固定模型快照 SHA-256**：`e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87`  
**服务器项目根目录**：`/cloud/cloud-ssd1/dissertation`  
**状态**：重新定义科学问题；禁止直接启动原 Pilot A/Pilot B；先完成根因定位，再进行支持注入与 RL。

---

# 0. Codex 总指令

从现在开始，不再把当前失败概括为“Qwen 不会补偿视觉错误”，也不把两阶段文本重放称为真实 natural-state fork。

本项目需要回答的核心问题改为：

> **当外部约束已经使一个视觉错误在数学上唯一可恢复时，Qwen2.5-VL 为什么仍然无法完成恢复？断裂发生在事实语义、约束联合、错误定位、逆向求解、自由生成、自然状态接口、策略支持，还是奖励可识别性？在正确恢复进入策略支持后，RL 能否增强真正的恢复，而不是只提高最终答案？**

Codex 必须遵守以下顺序：

1. 归档并统一旧实验口径；
2. 实现精确约束求解和候选空间；
3. 分解能力链；
4. 计算候选概率与层间约束吸收曲线；
5. 构建 Qwen 架构一致的自然状态 continuation；
6. 比较自然视觉修正与纯文本符号修复；
7. 仅在完成上述诊断后实施语言侧 LoRA 支持注入；
8. 最后才比较 outcome-only RL；
9. 所有简化任务只能用于定位瓶颈，不能代替最终自由恢复任务；
10. 所有外部 solver 只能用于数据生成、验证和评分，不能在主模型推理时替模型完成恢复。

---

# 1. 已锁定的事实与证据边界

## 1.1 小型神经网络机制证据

`small_neural_natural_replay_v2` 已经证明：在一个接口明确、自然状态可精确 fork 的受控系统中，人工错误状态不能替代自然错误状态。五类错误的 `c_syn-c_fork` 约为 `-0.0711` 至 `-0.0734`，配对区间均不含 0。

允许声明：

> 受控小模型中，natural mediator 与 synthetic mediator 的下游补偿性质不同。

禁止声明：

> Qwen 已经复现同一机制。

## 1.2 Qwen 的干净行为证据

当前最干净的 Qwen 结论来自 world-only 实验，而不是 Phase-C v3 的严格 DSL 结果：

- `cross_series + trend` 的 valid-cue 世界恢复为 `0/50`；
- 格式与语义解析均为 `50/50`；
- `41/50` 直接复制 observed values；
- 另外 `9/50` 修改了观测，但没有恢复真值；
- 尚未执行任何针对恢复任务的 SFT、LoRA 或 RL。

允许声明：

> 冻结 Qwen2.5-VL-3B 在当前 hard-text、zero-shot、free-generation 接口下，没有表现出可靠的关系约束驱动世界恢复；其主行为是复制观测。

禁止声明：

> Qwen 的完整自然视觉状态中不存在可恢复信息；
> Qwen 在任何接口下都不能修正错误；
> RL 已经失败。

## 1.3 Phase-C v3 的地位

`0/27,840` 首先表示严格 ResultProgram 测量接口整体失效。它是 measurement-interface failure，不是因果修复率为零的证据。事后语义提取只能用于定位，不能覆盖正式预注册结果。

## 1.4 Stage-2 v2 的意义

Stage-2 v2 的 `24/24` 说明：给定可信变量，Qwen 可以生成受限正向程序，确定性执行器可以稳定执行。它证明正向计算链可工作，但不证明模型具备逆向错误恢复能力。

---

# 2. 研究问题的精确重构

原问题把以下五层混成了一个“recoverable”：

1. 外部设计是否唯一可解；
2. 模型接口是否与自然状态一致；
3. 恢复算法是否可计算；
4. 当前策略是否给正确恢复非忽略概率；
5. RL 是否能利用该支持。

新版建立五层恢复层级。

## R0：Design Identifiability

给定观测 `o` 与可靠事实 `F`，外部规范求解器是否得到唯一真值 `x*`。

## R1：Interface-Conditional Recoverability

不同接口 `I` 下，模型实际接收到的状态是否适合执行恢复：

- hard textual report；
- hard report + soft alternatives；
- same-conversation image-retained state；
- exact cached natural continuation。

R1 不是单纯的信息论问题，还包括表示形式和模型归纳偏置。

## R2：Algorithmic Decodability

是否存在确定算法从 `(o,F)` 恢复 `x*`。外部 solver 成功只证明外部算法可解，不代表 Qwen 内部已经实现该算法。

## R3：Policy Accessibility

当前冻结模型是否给正确恢复序列分配可访问概率，而不仅是 greedy 是否成功。

## R4：RL Learnability

在当前策略支持和 reward 下，RL 是否能增加 genuine recovery，而不是增加复制、猜测、视觉重读或最终答案 shortcut。

---

# 3. 核心理论

## 3.1 约束系统

真实世界：

\[
x^*=(x_0,x_1,x_2,x_3)\in\mathbb Z^4.
\]

自然观测最多一个位置错误：

\[
o=x^*+\delta e_j.
\]

可靠事实统一写为：

\[
A_Fx=b_F.
\]

三类事实对应：

- `known_value(i,v)`：\(x_i=v\)；
- `pair_sum(i,j,t)`：\(x_i+x_j=t\)；
- `arithmetic_progression(l,m,r)`：\(x_l-2x_m+x_r=0\)。

定义允许的一次编辑候选集合：

\[
\mathcal C(o)=\{x:\operatorname{Ham}(x,o)\le 1,\ x_k\in\mathcal V\}.
\]

其中 `V` 是数据生成器预先定义的整数取值域。

Design-recoverable 条件：

\[
\left|\{x\in\mathcal C(o):A_Fx=b_F\}\right|=1.
\]

唯一元素为外部约束投影：

\[
P_F(o)=x^*.
\]

## 3.2 一个必须纠正的理论点

在 eligible scenes 中，`o+F` 已经唯一确定 `x*`，因此：

\[
H(X\mid O,F)=0.
\]

所以不能简单声称：

> hard text 因为丢失信息而使恢复在数学上不可能。

对于这些场景，hard text 在任务信息意义上已经充分。真正的问题是：

> **同一份充分信息是否处于 Qwen 已学习、可访问的计算表示与策略区域中。**

自然 hidden state 可能提供额外捷径、置信度或视觉残差信息，但 hard-text 失败首先说明的是计算可访问性和归纳偏置失败，而不是 Shannon 信息缺失。

## 3.3 逆向恢复不是正向算术

把错误观测代入事实：

\[
s=b_F-A_Fo.
\]

由 `o=x*+δe_j` 得：

\[
s=-\delta A_{F,j}.
\]

恢复至少需要：

1. 识别事实语义；
2. 计算所有事实残差；
3. 找到唯一错误列 `j`；
4. 解出误差 `δ`；
5. 只修改位置 `j`；
6. 对全部事实做全局复核。

Stage-2 v2 的正向运算成功不能推出上述逆向链成功。

## 3.4 约束吸收与观测锚定

令模型在接口 `I` 下对世界候选的分布为：

\[
q_\theta(x\mid o,F,I).
\]

定义真值对观测的 log-odds margin：

\[
M_\theta(F,I)
=
\log q_\theta(x^*\mid o,F,I)
-
\log q_\theta(o\mid o,F,I).
\]

定义事实吸收增量：

\[
\Delta_F(I)=M_\theta(F,I)-M_\theta(\varnothing,I).
\]

解释：

- `Δ≈0`：facts 基本没有进入模型决策；
- `Δ>0, M<0`：facts 被吸收，但仍未推翻 observed-world 锚点；
- `M>0` 但 free generation 失败：搜索、解码或格式实现瓶颈；
- `Δ<0`：facts 被误解或形成干扰。

## 3.5 自由生成差距

定义候选选择任务 T5 和自由恢复任务 T6。

\[
G_{search}=Acc(T5)-Acc(T6).
\]

若 T5 明显优于 T6，不能说模型已经会完整恢复；只能说候选验证能力高于自由构造能力。

## 3.6 错误定位差距

定义：

\[
G_{loc}=Acc(T4\mid \text{given error index})-Acc(T3\mid \text{infer index}).
\]

该差距定位是否主要卡在错误位置识别。

## 3.7 Qwen 自然状态与 hard-text 接口不是同一机制

Qwen2.5-VL 将视觉编码器输出经 patch merger 映射到语言隐藏维度，并把视觉 token 与文本 token 放入同一自回归序列。后续语言层能够在同一上下文中继续访问视觉 token 信息。

因此：

- 新开一个 text-only call 测的是 **symbolic downstream recovery**；
- 在同一 cache / conversation 中保留 image tokens 测的是 **natural visual revision**；
- 二者不能互相替代。

定义接口差距：

\[
G_{interface}
=
Acc(I_{cache})-Acc(I_{hardtext}).
\]

若 `I_cache` 成功而 `I_hardtext` 失败，说明实际修正依赖视觉状态保留或重新访问；不能把它称为纯文本 downstream compensation。

## 3.8 RL 的策略支持

对样本 `i`，令 genuine recovery 的单 rollout 概率为 `p_i`。一个 K-rollout 组同时包含成功和失败、从而具有非零组内 reward variance 的概率为：

\[
g_i(K)=1-p_i^K-(1-p_i)^K.
\]

总体 informative-group rate：

\[
G_K=\frac1N\sum_i g_i(K).
\]

若一个 prompt 的全部 rollout reward 相同，标准 group-relative update 对该 prompt 不提供区分方向。新版不使用主观“成功几次才允许训练”，而是直接报告：

- `p_i`；
- `G_K`；
- 计划训练预算中的预期 informative groups；
- 实际 group reward variance。

## 3.9 Reward 可识别性

在原始 chart answer task 中，下列轨迹都可能获得相同 `R=1`：

- 真正世界恢复；
- operator invariance；
- 错误抵消；
- 视觉重新读取；
- 猜测与答案先验。

因此 answer-only reward 无法从目标层面区分成功来源。

新版将两类 RL 分开：

1. **Recovery-outcome RL**：最终 reward 检查完整世界 `x*`；用于确认 RL 是否能放大恢复策略；
2. **Answer-only RL**：只检查最终图表答案；用于检验它是否转向 shortcut。

Recovery-outcome RL 是机制正控制，不替代 answer-only 主问题。

---

# 4. Qwen2.5-VL 特定设计原则

## 4.1 架构事实必须运行时核验

官方配置通常包含：

- 约 36 个语言层；
- 约 32 个视觉块；
- ViT 输出经 patch merger 映射到语言隐藏维度；
- 图像 token 使用多模态位置编码；
- 动态分辨率输入。

Codex 不得硬编码这些数字。必须从本地固定快照读取：

```python
model.config
model.config.vision_config
model.named_modules()
```

并生成：

```text
artifacts/v4/model_introspection.json
artifacts/v4/module_manifest.txt
```

## 4.2 固定视觉输入预算

为避免图像 token 数成为实验变量：

- 所有主实验使用固定 `resized_height` 与 `resized_width`；
- 尺寸必须为 28 的整数倍；
- train/dev/test 使用同一处理器和同一 resize 路径；
- OOD 风格变化不得改变像素总数；
- 保存 `image_grid_thw` 和实际视觉 token 数。

## 4.3 最小输出接口

主 world-recovery 输出固定为：

```text
a,b,c,d
```

不要求：

- DSL；
- 解释；
- 独立最终答案；
- Markdown；
- 自由 CoT。

DSL 仅用于独立的正向运算控制，不进入主恢复率。

## 4.4 自然状态 continuation

必须实现 Qwen 的 exact cached continuation：

1. 输入图像和 Stage-1 observation prompt；
2. 使用手动或可审计 greedy generation 生成 observation；
3. 保存完整 `past_key_values`、token IDs、attention mask、position IDs、image token 位置；
4. 通过 chat template 生成新增 user turn 的 suffix token；
5. 只将 suffix token 送入已有 cache；
6. 继续生成 corrected world。

必须做 parity test：

- cache continuation；
- 完整对话重新编码；

在 greedy 条件下输出应一致或逐 token 解释差异。未经 parity 验证，不得把 cache runner 用作主结果。

## 4.5 不得将 image-retained correction 称为纯 reasoning repair

如果 image tokens 仍在 KV 中，模型可能重新读取视觉证据。该条件应命名为：

```text
natural_visual_revision
```

只有新 text-only call 才命名为：

```text
symbolic_downstream_recovery
```

---

# 5. 可证伪假设

## H1：事实语义不是唯一瓶颈

预测：单事实验证 T1 高于完整恢复 T6。

若 T1 本身失败，则事实语义是基础瓶颈。

## H2：主要困难之一是逆向错误定位

预测：给定错误 index 的 T4 优于要求模型定位 index 的 T3。

若 T4 仍失败，瓶颈不止定位。

## H3：facts 能改变候选概率但不足以推翻观测锚点

预测：

\[
\Delta_F>0,\qquad M_F<0.
\]

若 `Δ≈0`，facts 没有被有效吸收。

## H4：候选验证能力高于自由恢复能力

预测：`T5 > T6`。

若二者均失败，不能归因于搜索。

## H5：Qwen 的自然视觉修正优于 hard-text 符号修复

预测：

\[
Acc(I_{cache})>Acc(I_{hardtext}).
\]

若成立，说明原两阶段 image-cut 低估了架构自然修正能力；但这类修正包含视觉重访。

## H6：语言侧恢复程序可以通过同模型 LoRA 注入

预测：只有 constraint-recovery LoRA 显著改善 T3–T6；format-only 与 forward-arithmetic 控制不能产生同等改善。

## H7：RL 受初始策略支持限制

预测：base outcome RL 的 informative-group rate 较低；support-seeded 模型具有更高 `G_K`，其 RL 才可能稳定增加恢复率。

## H8：answer-only RL 与 genuine recovery 可以脱钩

预测：answer-only accuracy 可以增加，而 exact world recovery、`M_F` 或 counterfactual compliance 不增加。

---

# 6. 数据治理与实验隔离

## 6.1 旧 580 场景的地位

现有 580 个 eligible scenes 已参与多轮诊断、prompt qualification 和事后分析。

它们只能用于：

- 工程调试；
- 能力分解；
- SFT 训练池；
- 生成示例；
- 不作为最终 confirmatory test。

## 6.2 新数据版本

创建：

```text
CVA-Constraint-Recovery-v4
```

至少区分：

```text
legacy_diagnostic
symbolic_support_train
natural_error_support_train
support_dev
confirm_iid
confirm_style_ood
confirm_constraint_ood
```

## 6.3 最终确认集

最终确认集必须：

- 使用新 seed；
- semantic scene、数值表、constraint graph 均与训练和 legacy 数据不重叠；
- 在提示、LoRA 和 RL 配置冻结后一次性生成；
- 固定语义场景数，不根据恢复结果继续补采样；
- 对全部自然 Stage-1 错误按预定义规则纳入，不只选择成功或接近成功样本。

## 6.4 duplicate_encoding 的地位

`duplicate_encoding` 仅作为可信状态复述与 instruction-following 正控制。

论文主结论必须来自：

- `cross_series`；
- `trend`。

---

# 7. Phase 0：旧证据统一审计

## 7.1 目标

在任何新模型调用前，统一旧结果口径。

## 7.2 必做事项

1. 所有 no-cue / valid-cue 使用同一个 hidden truth 评分；
2. observed-world consistency 单独命名；
3. 严格 DSL、事后 semantic extraction、world-only 结果不混合；
4. Phase-C v3 标记为 interface failure；
5. Qwen 两调用文本 replay 全部改名为 `text_replay`，不得叫 `c_fork`；
6. 恢复缺失的 100-call no-cue 汇总；
7. 生成 claim-evidence matrix。

## 7.3 输出

```text
artifacts/v4/audit/legacy_experiment_registry.csv
artifacts/v4/audit/claim_evidence_matrix.csv
artifacts/v4/audit/scoring_contract.md
artifacts/v4/audit/legacy_hash_manifest.json
```

---

# 8. Phase 1：能力链分解

本阶段使用 legacy 诊断场景，不作为最终假设检验。

## T1：单事实语义验证

输入一个候选世界和一条 fact，只回答：

```text
YES
```

或：

```text
NO
```

分别覆盖：

- known value；
- pair sum；
- arithmetic progression。

## T2：整体冲突检测

输入 observed world 与全部 facts，只回答：

```text
CONFLICT
```

或：

```text
CONSISTENT
```

## T3：错误位置定位

只回答：

```text
0
1
2
3
```

不要求新值。

## T4：给定位置后的真值恢复

明确给出唯一错误位置，只输出该位置真实值。

## T5：候选世界选择

外部 solver 构造：

- observed world；
- true world；
- 若干 matched one-edit distractors。

候选随机映射到单 token 标签。模型只输出标签。

## T6：完整自由恢复

只输出：

```text
a,b,c,d
```

## 注意

T1–T5 仅用于瓶颈定位。最终能力结论仍以 T6 为准。不得因为 T5 成功而声称模型具备完整恢复能力。

## 输出

```text
artifacts/v4/capability_chain/per_scene.csv
artifacts/v4/capability_chain/summary_by_family.csv
artifacts/v4/capability_chain/paired_gaps.json
```

---

# 9. Phase 2：候选概率与层间约束吸收

## 9.1 目的

区分：

- facts 未被理解；
- facts 提高了真值概率但未克服 observation anchor；
- 真值已是高概率但自由解码失败。

## 9.2 候选标签前置检查

候选标签必须在当前 tokenizer 中是稳定的单 token。Codex 必须自动搜索并保存：

```text
artifacts/v4/tokenizer/candidate_labels.json
```

候选顺序必须随机且平衡。

## 9.3 条件

每个场景至少评分：

- no-cue；
- valid-cue；
- sham-cue；
- counterfactual-cue。

不调用生成，使用 teacher-forced next-token logits。

## 9.4 指标

- `logp_true`；
- `logp_observed`；
- `M_F`；
- `Delta_F`；
- true rank；
- observed rank；
- family-stratified effects。

## 9.5 Layerwise Constraint Assimilation Profile

Qwen 文本层数由本地 config 读取。对每层 `l` 的最后 prompt token hidden state：

1. 应用与最终输出兼容的 norm / lm head；
2. 读取候选标签 logits；
3. 计算：

\[
M_F^{(l)},\qquad \Delta_F^{(l)}.
\]

生成四类模式：

- no assimilation；
- transient assimilation；
- persistent but insufficient assimilation；
- successful revision。

该分析使用任务定义的 logit margin，不使用 raw latent Euclidean distance。

## 9.6 有效性检查

- 最后一层 candidate logits 必须与标准 forward 一致；
- 标签必须单 token；
- 候选世界必须唯一、无重复；
- no-cue / valid-cue 仅 facts 不同；
- 任何 mismatch 都应使脚本退出非零状态。

---

# 10. Phase 3：Qwen 架构一致的接口阶梯

## I0：Hard-Text Symbolic Recovery

新 text-only call：

```text
observed_values + facts -> corrected world
```

这是当前已失败的接口。

## I1：Soft-Report Diagnostic

由 Stage-1 生成 logits 构造每个位置的 top-k 数值候选与相对分数。

用途：判断 argmax hardening 是否丢失有用不确定性。

它是干预式诊断，不是自然主结果。

## I2：Candidate-World Diagnostic

列出 one-edit 候选，模型选标签。

用途：隔离自由搜索，不作为最终恢复能力。

## I3：Same-Conversation Visual Revision

第一轮：图像 -> natural observed values。  
第二轮：同一多轮上下文加入 facts，图像仍在历史中。  
要求自由输出 corrected world。

该接口符合 Qwen 的正常多模态使用，但允许重新访问视觉证据。

## I4：Exact Cached Natural Continuation

保留第一轮的 image-token KV 和自然生成状态，在 cache 上追加 facts 并继续。

这是最接近 natural-state continuation 的接口。

## 10.1 分解自然修正来源

对 I3/I4 分别运行：

- no-cue；
- valid-cue；
- sham-cue；
- counterfactual-cue。

定义：

### Spontaneous visual revision

\[
R_{vision}=Acc(\text{no-cue},I4)-Acc(\text{no-cue},I0).
\]

### Fact-conditioned revision

\[
R_{fact}=Acc(\text{valid},I4)-Acc(\text{no-cue},I4).
\]

### Counterfactual compliance

facts 改为另一个合法世界时，输出按理论方向改变的比例。

## 10.2 解释纪律

- I4 改善不自动等于 downstream reasoning repair；
- I0 改善才支持 pure symbolic recovery；
- I4 > I0 表示视觉状态保留/重访贡献；
- counterfactual facts 生效才支持 facts 被使用。

---

# 11. Phase 4：支持注入式 LoRA

## 11.1 目的

不是把 LoRA 当最终解决方案，而是建立恢复轨迹的策略支持，测试“当前失败是否主要来自 base policy 没学会逆向恢复程序”。

## 11.2 参数范围

第一轮仅训练语言侧 LoRA：

- vision tower frozen；
- patch merger/projector frozen；
- base language weights frozen；
- language attention/MLP LoRA trainable。

Codex 必须通过 `named_modules()` 生成精确 target list，禁止凭模块名猜测。

保存：

```text
artifacts/v4/training/trainable_parameter_manifest.json
artifacts/v4/training/frozen_hashes.json
```

## 11.3 三个控制模型

### C0：Format-only LoRA

只学习最小四整数输出，不含事实驱动修复。

### C1：Forward-Arithmetic LoRA

学习给定正确变量后的 sum/difference/max-minus-min 或 fact verification，不学习错误恢复。

### T：Constraint-Recovery LoRA

学习完整逆向链：

1. fact verification；
2. conflict detection；
3. error index；
4. replacement value；
5. global fact verification；
6. full free world recovery。

## 11.4 训练课程不是最终测试替代

允许在训练早期使用 T1–T4 子任务建立能力，但最后训练阶段必须只使用与主测试一致的自由恢复输出：

```text
a,b,c,d
```

最终评估禁止提供：

- error index；
- candidate worlds；
- external solver output；
- gold values；
- retry。

## 11.5 训练数据

### symbolic_support_train

程序生成的约束恢复样本；数字、约束图与测试集隔离。

### natural_error_support_train

使用冻结 base Qwen 在独立 train scenes 上自然产生的单位置错误，配合可靠事实和真值作为监督。

严禁用最终确认集的自然错误训练。

## 11.6 Qwen 配置

初始建议：

```yaml
precision: bf16
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0
gradient_checkpointing: true
vision_frozen: true
merger_frozen: true
```

学习率、batch、epoch 必须通过 support_dev 预先冻结；不得在最终 confirm test 上挑选。

---

# 12. Phase 5：策略支持测量

对 Base、C0、C1、T 四个 checkpoint：

1. greedy T6；
2. candidate scoring；
3. 固定温度采样；
4. pass@K；
5. 估计 `p_i` 与 `G_K`；
6. 记录 observation-copy rate。

输出：

```text
artifacts/v4/support/policy_support_by_scene.parquet
artifacts/v4/support/informative_group_rate.json
artifacts/v4/support/pass_at_k.csv
```

关键解释：

- C0 改善只能说明格式；
- C1 改善只能说明普通算术/事实判断；
- T 在 held-out natural errors 上改善，才说明逆向恢复程序可学习；
- T 仍失败，则该模型规模/训练方案可能不足，或自然接口存在更深瓶颈。

---

# 13. Phase 6：RL 实验

## 13.1 模型组

至少比较：

1. `Base`；
2. `Base + Answer-Only RL`；
3. `Recovery LoRA`；
4. `Recovery LoRA + Recovery-Outcome RL`；
5. `Recovery LoRA + Answer-Only RL`。

可选机制控制：

6. `Recovery LoRA + Constraint-Aware RL`。

## 13.2 两种主 reward

### Recovery-Outcome Reward

输出完整世界等于 `x*` 才奖励 1。

它消除 operator invariance 和最终答案猜测，是“RL 能否放大恢复策略”的正控制。

### Answer-Only Reward

只检查原 chart question 的最终答案。

它保留原始研究问题，但成功来源不唯一。

## 13.3 GRPO 信号诊断

每步记录：

- group rewards；
- reward variance；
- all-zero group 比例；
- all-one group 比例；
- non-degenerate group 比例；
- KL；
- entropy；
- exact world recovery；
- observation-copy rate。

不得只报告训练 loss 和 answer accuracy。

## 13.4 主效应

### Support hypothesis

比较：

\[
\Delta_{RL}^{seeded}
\quad\text{vs}\quad
\Delta_{RL}^{base}.
\]

若只有 Recovery LoRA 后 RL 有效，支持“RL 依赖先验策略支持”。

### Reward-identifiability hypothesis

比较：

- Recovery-Outcome RL 的 exact world recovery；
- Answer-Only RL 的 answer accuracy 与 world recovery。

若 Answer-Only RL 只提高答案、不提高世界恢复，支持 reward-source ambiguity。

---

# 14. Phase 7：回到真实多模态问题

前述 world recovery 是机制任务。最终必须回到 chart QA，避免研究变成纯文本 CSP。

对每个最终 checkpoint，执行完整链：

```text
image -> natural observation -> revision/recovery -> chart operation -> final answer
```

同时报告：

- Stage-1 visual exact；
- post-revision world exact；
- reasoning operator exact；
- final answer exact；
- operator-invariant correct；
- genuine recovery；
- error cancellation；
- trace mismatch；
- OOD error-mechanism shift。

主结论必须区分：

1. 模型看得更准；
2. 模型从 facts 纠正世界；
3. 模型重新看图；
4. 模型只提高最终答案。

---

# 15. OOD 设计

至少包含：

## Style OOD

改变字体、线宽、marker、legend 位置，但保持语义、分辨率和 token 数。

## Constraint-Graph OOD

训练使用某些 pair-sum / trend 图结构，测试使用未见约束连接模式。

## Error-Mechanism OOD

训练自然错误以低读 1 为主，测试通过渲染变化诱导：

- 高读；
- 不同位置错误；
- legend binding；
- trend point confusion。

如果 answer-only RL 依赖特定补偿模式，OOD 应更脆弱。

---

# 16. 统计分析

不设置事后“成功至少多少”的主观门槛。所有结果报告：

- 点估计；
- scene-clustered bootstrap 95% CI；
- paired difference；
- family-stratified effect；
- seed-level variability；
- multiple comparison Holm correction；
- equivalence control 使用预注册 TOST margin；
- rollout 不作为独立 scene。

主要配对比较：

- valid vs no-cue；
- valid vs sham；
- counterfactual compliance；
- I4 vs I0；
- T vs C0/C1；
- seeded RL vs base RL；
- recovery-reward RL vs answer-only RL。

---

# 17. 结果解释矩阵

| 观察结果 | 允许解释 |
|---|---|
| T1 失败 | 事实语义未掌握或提示编码失败 |
| T1 成功、T2 失败 | 多事实联合/注意失败 |
| T2 成功、T3 失败 | 逆向定位失败 |
| T3 成功、T4 失败 | 误差量求解失败 |
| T5 成功、T6 失败 | 候选验证存在，但自由搜索/生成失败 |
| `Δ_F≈0` | facts 未改变候选偏好 |
| `Δ_F>0, M_F<0` | 事实已吸收但 observation anchor 仍占优 |
| I4 成功、I0 失败 | 自然修正依赖视觉状态保留/重访 |
| T LoRA 成功、C0/C1 失败 | 逆向恢复程序可学习，不是格式/正向算术效应 |
| Base RL 无效、seeded RL 有效 | 支持受限的 RL 假设得到支持 |
| Answer RL 提高答案但不提高世界恢复 | reward shortcut / source ambiguity |
| Recovery RL 提高世界恢复 | RL 能放大已存在的恢复策略 |
| 所有接口和训练均失败 | Qwen2.5-VL-3B 在该恢复任务上存在强容量或优化限制 |

---

# 18. 新颖性边界

不得将以下内容声明为新颖：

- 最终答案不代表视觉正确；
- VLM 可以忽略图像；
- LLM 可能不遵守约束；
- 外部执行器比自由算术稳定；
- 自我纠正可能失败；
- 视觉重访可以提升多模态推理。

当前可能形成贡献的是以下联合结构：

1. **五层 recoverability hierarchy**：design、interface、algorithm、policy、RL；
2. **constraint assimilation profile**：事实如何在 Qwen 的层间 logits 中改变 true-vs-observed preference；
3. **natural visual revision vs symbolic downstream recovery** 的架构一致区分；
4. **support-seeded RL test**：先建立同任务恢复支持，再检验 outcome RL 是否放大；
5. **reward-source comparison**：完整世界 reward 与最终答案 reward 导致不同学习结果；
6. **将所有结论重新落回自然视觉错误和 OOD 机制变化**。

---

# 19. 仓库实现结构

在现有 repository 中新增：

```text
src/compensability_v4/
├── theory/
│   ├── constraint_system.py
│   ├── candidate_space.py
│   ├── recoverability_hierarchy.py
│   └── policy_support.py
├── data/
│   ├── v4_generator.py
│   ├── splits.py
│   ├── natural_error_pool.py
│   └── ood_generator.py
├── qwen/
│   ├── introspect_model.py
│   ├── model_loader.py
│   ├── candidate_scoring.py
│   ├── layerwise_assimilation.py
│   ├── manual_generation.py
│   ├── cache_continuation.py
│   └── interface_runner.py
├── diagnostics/
│   ├── capability_chain.py
│   ├── observation_anchor.py
│   └── interface_ladder.py
├── training/
│   ├── build_support_sft.py
│   ├── lora_manifest.py
│   ├── train_format_control.py
│   ├── train_forward_control.py
│   ├── train_recovery_lora.py
│   ├── train_recovery_rl.py
│   └── train_answer_rl.py
├── eval/
│   ├── world_recovery.py
│   ├── answer_source.py
│   ├── support_metrics.py
│   ├── counterfactual.py
│   ├── ood.py
│   └── statistics.py
└── schemas/
    ├── scene.py
    ├── observation.py
    └── record.py
```

脚本：

```text
scripts/v4/
├── 00_audit_legacy.py
├── 01_introspect_qwen.py
├── 02_run_capability_chain.py
├── 03_score_candidates.py
├── 04_layerwise_assimilation.py
├── 05_validate_cache_runner.py
├── 06_run_interface_ladder.py
├── 07_build_support_data.py
├── 08_train_controls.py
├── 09_train_recovery_lora.py
├── 10_measure_policy_support.py
├── 11_train_recovery_rl.py
├── 12_train_answer_rl.py
└── 13_final_multimodal_eval.py
```

---

# 20. 关键 API

```python
def enumerate_one_edit_candidates(
    observed: tuple[int, int, int, int],
    value_domain: range,
) -> list[tuple[int, int, int, int]]: ...


def satisfies_all_facts(
    world: tuple[int, int, int, int],
    facts: list[dict],
) -> bool: ...


def unique_constraint_projection(
    observed: tuple[int, int, int, int],
    facts: list[dict],
    value_domain: range,
) -> tuple[int, int, int, int]: ...


def score_candidate_labels(
    model,
    processor,
    prompt,
    candidate_labels: list[str],
) -> dict[str, float]: ...


def layerwise_candidate_logits(
    model,
    batch,
    label_token_ids: list[int],
) -> list[dict[str, float]]: ...


def generate_observation_with_cache(
    model,
    processor,
    image,
    prompt,
) -> dict: ...


def append_turn_and_continue(
    model,
    cached_state,
    new_user_text: str,
) -> dict: ...


def informative_group_probability(p: float, k: int) -> float:
    return 1.0 - p**k - (1.0 - p)**k
```

---

# 21. 必须编写的测试

## Theory

- 每种事实与矩阵表示一致；
- unique projection 与暴力枚举一致；
- 不唯一场景必须抛错；
- `s=-δA_j` 随机实例验证；
- informative group 公式与 Monte Carlo 一致。

## Candidate scoring

- 标签均为单 token；
- candidate order 改变不应改变世界分数映射；
- final-layer logits 与标准 forward 一致；
- no-cue 与 valid-cue 除 facts 外完全相同。

## Cache continuation

- 完整历史重编码与 cache continuation greedy 输出对齐；
- image token 数和位置保存；
- suffix token 拼接精确；
- cache state 不被意外复用到其他样本；
- RNG 和 generation config 记录。

## Freeze

- vision hash 不变；
- merger hash 不变；
- 只有指定 language LoRA 可训练；
- adapter disable 后恢复 base 行为。

## Evaluation

- copy、single-edit、overedit、true recovery 分类互斥且完备；
- counterfactual world 合法；
- scene 为统计单位；
- 旧数据不会进入 confirm split。

---

# 22. 执行顺序与停止纪律

## 第一步

只执行 Phase 0–3，不训练。

## 第二步

根据诊断结果选择 LoRA 课程，但选择规则必须由能力链结果预先写入 amendment，不能看最终 test。

## 第三步

训练 C0、C1、T，冻结配置，测 policy support。

## 第四步

运行两类 RL。不得跳过 Base + Answer-Only RL 负控制，也不得只报告表现最好的一组。

## 第五步

在全新 confirm set 一次性评估，并锁定 hashes。

## 禁止事项

- 不再在 legacy 580 上调 prompt 后当正式结果；
- 不因某个接口成功就删除其他接口；
- 不把 candidate selection 当完整恢复；
- 不把 image-retained correction 叫纯 reasoning compensation；
- 不用 external solver 在主推理中替模型纠错；
- 不因 RL answer accuracy 提升就声称 perception/recovery 提升；
- 不根据中途成功率扩大样本；
- 不覆盖原失败报告。

---

# 23. Codex 每阶段报告模板

```text
Phase:
Git commit:
Model snapshot hash:
Config hash:
Dataset manifest hash:
Prompt hash:
Tokenizer/version:
Interface:
Number of semantic scenes:
Number of model calls:
Primary estimands:
Point estimates:
95% confidence intervals:
Observation-copy rate:
Exact world recovery:
Candidate true-vs-observed margin:
Informative-group rate:
Parser failures:
Unresolved confounds:
Claims allowed:
Claims prohibited:
Next authorized action:
```

---

# 24. 最终论文主线

## 暂定标题

**Before RL Can Repair Vision: Constraint Assimilation, Natural-State Revision, and Policy Support in Qwen2.5-VL**

## 核心叙事

1. 外部可唯一求解不等于模型策略可访问；
2. Qwen 在 hard-text world recovery 中表现出强 observation copying；
3. 层间候选概率揭示 facts 是未吸收、被晚层覆盖，还是不足以越过锚点；
4. Qwen 的架构自然允许视觉 token 在推理中继续参与，因此 hard-text repair 与 natural visual revision 必须分开；
5. 逆向恢复 LoRA 用于建立策略支持，而不是替代主任务；
6. 只有在支持建立后，才能严谨比较 outcome RL；
7. 完整世界 reward 与 answer-only reward 揭示 RL 是增强 genuine recovery，还是只增强最终成功。

## 最强可证伪结果

### 结果 A

facts 在中层提高 true-world logits，但末层重新偏向 observed world。

结论：存在 late-layer observation anchoring。

### 结果 B

cached visual continuation 成功，hard-text 失败。

结论：Qwen 的自然修正依赖视觉状态保留/重访，而非纯符号 downstream compensation。

### 结果 C

Recovery LoRA 建立非零支持后，RL 才提高 exact recovery。

结论：RL 对恢复能力存在 support dependence。

### 结果 D

Answer-only RL 提高答案但不提高 exact world recovery。

结论：outcome gain 与 genuine correction 脱钩。

### 结果 E

所有接口、LoRA 和 RL 均失败。

结论：在 Qwen2.5-VL-3B 和当前任务族中，design-recoverable inverse correction 不是一个稳定可访问或可学习的能力；该负结论仍然成立。

---

# 25. 相关工作与声明边界

实施和写作时至少核对：

- Qwen2.5-VL Technical Report，arXiv:2502.13923；
- Qwen2.5-VL 官方 Transformers 实现与本地固定快照配置；
- The Reversal Curse，arXiv:2309.12288；
- Towards a Theoretical Understanding of the Reversal Curse，arXiv:2405.04669；
- Don’t Look Only Once: Selective Visual Revisitation，arXiv:2505.18842；
- Correction-Oriented Policy Optimization，arXiv:2605.14539；
- SCOPE-RL，arXiv:2607.11506；
- 现有 multimodal reward hacking、perception/reasoning decomposition 与 sensitivity/invariance 工作。

本项目不声称首次发现 LLM 自我修正困难、视觉重访有效或 outcome reward 稀疏。新颖性必须落在本文件第 18 节的联合问题与实证闭环。

---

# 26. Definition of Done

只有完成以下全部内容，v4 主实验才算结束：

1. 旧证据统一审计；
2. T1–T6 能力链；
3. candidate log-prob 与 layerwise assimilation；
4. cache continuation parity；
5. I0–I4 接口对照；
6. C0/C1/T LoRA 控制；
7. policy support 与 `G_K`；
8. Base 与 seeded RL 对照；
9. recovery reward 与 answer reward 对照；
10. 新 confirm set；
11. IID、style OOD、constraint OOD、error-mechanism OOD；
12. answer-source decomposition；
13. 全部配置、数据、prompt、代码和结果 hash；
14. 原失败实验完整保留。

