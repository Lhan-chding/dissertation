# SSVC 下一轮 Codex 执行规范：可验证协议发现与原协议监督回迁

**版本：2026-10-06；新设计，尚未执行本轮模型实验。**  
**主模型：现有 Qwen3.5-9B；最多并行使用5张PRO6000。**  
**本轮包含新的冻结推理、监督微调，以及小规模R0训练参照。旧的“本轮禁止一切训练”限制属于上一轮，不适用于本文件；但本轮不做off-policy RL、动态奖励控制或新选择器拟合。**

## 0. 本轮必须回答的问题

现有开发结果支持协议之间可能有可利用的成功互补，但尚未完成清洁新实例上的确认，也没有证明这些成功能够变成原协议中的模型改进。本轮分开回答：

1. **独立互补**：在新修复实例上，同样的回答调用次数，多协议验证组合是否优于重复O0，也是否优于验证集选出的最佳固定单协议？
2. **发现覆盖**：同样16次调用，多协议能为哪些训练题获得至少一个通过验证的正确世界？相对O0，新增和丢失的题分别是什么？
3. **监督回迁**：把这些成功世界规范化后，用原始O0提示监督训练，是否提高新题上的O0单次精确恢复率，而不是仅提高带变换提示的系统成功率？
4. **瓶颈定位**：若没有收益，主要是成功样本没有多发现、发现集合没有额外训练价值、训练未拟合、迁移不足，还是造成遗忘？

**主要终点：新E面板上，256步SFT后的O0精确恢复率；主要比较SELF_MIX−SELF_O0。SELF_MIX−SELF_SINGLE是必须同时报告的强基线比较。**

本轮不是检验“PTLC是模型内部算法”，也不是宣称实现了最终SSVC。PTLC、DPE₁和错误结构只作为冻结的诊断读数。核心线路是：

> 语义行为概率与协议互补 → 获得可验证的训练样本 → 原协议参数学习 → 新实例上的语义表现与训练对比。

## 1. 先固定已有结论，避免重走弯路

| 已有依据 | 本轮落实 | 不可写成 |
|---|---|---|
| D48上的O0/L11成功互补；U22 cross上组合可能不如重复 | 新题上保留两家族、全体题、单协议和组合 | 多协议必然提高所有任务 |
| D/P/U22已经被多轮查看，U22存在历史执行 | 全新任务划分；旧面板只用既有结果作背景 | 旧面板是独立确认 |
| 原始数组与请求排列可能不一致 | 严格按请求逆排列一次，再验证 | 用真值搜索最优排列 |
| PTLC与copy/truth有别名，38条已修正为22条 | 保存所有匹配位及别名；不通过优先级归类声称机制 | 匹配就证明内部执行了程序 |
| 唯一解、同样规范目标下solver-J和self-J相同 | 合并相同训练视图，不重复训练 | 标签来源本身产生差异 |
| 多对一规范化＋成功筛选改变样本分布 | 使用定义明确的SFT，不套旧PPO比率 | 当前方法是无偏off-policy GRPO |
| 复杂选择器尚无增益 | 先比较可执行的固定组合；不训练高维路由器 | 描述更丰富就有决策收益 |

上表来源为`sources/LATEST_DISCUSSION_zh.md`及本包来源说明；不是本轮预先保证的结果。下文所有数量、阈值和模块是本次新设计。

## 2. 代码起点与不可混用的运行路径

仓库：`Lhan-chding/dissertation`。本次实际核读的分支为：

```text
codex/protocol-state-probes-20261005
2f8c6b19970f49542b009f2816f1c1a91eb9974c
```

建议从相应已完成实现建立独立worktree：`codex/verified-discovery-transfer-20261006`。本文件没有修改GitHub。Codex开始时记录实际HEAD和相关工作区差异一次，不覆盖用户已有修改。

已检查的接口：

- `src/protocol_state_probes/inference.py`：已有冻结、raw-only生成；恢复以后会把所有参数冻结，且旧验证器明确禁止训练。**不能直接复用它当训练初始化后忘记重新开放精确的LoRA白名单。**
- `src/protocol_state_probes/cases.py`、`semantics.py`：复用已验证的O0/L11/B1提示和坐标映射。新数据使用新命名空间，不绕过旧面板注册。
- `src/model_adapters/base.py`：历史`logprobs(require_grad=True)`逐token重算前缀。它适合旧on-policy一致性要求，但不应未经评估地被当成全量SFT的高效实现。
- `src/generate_worlds.py`：历史ID真值主要在0–49，trend唯一世界容量有限。不要简单修改旧尺寸参数后无限拒绝采样。
- `src/constraint_solver.py`、`src/verifiers.py`：复用任务契约；新生成器和独立验证器各自做确定性检查。
- `src/prospective_selection/branch_runtime.py`及旧GRPO loss：仅供R0参照复用，不能给反序成功目标套旧behavior log-prob。

建议新增`src/verified_discovery_transfer/`，具体接口见第14节。路径是待实现建议，不是声称现在已有这些CLI。

## 3. 模型与两类起点

### 3.1 固定模型

- 基座：`Qwen/Qwen3.5-9B`，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- BF16冻结基座；现有LoRA：r=8、alpha=16、dropout=0，96个语言MLP模块的gate/up/down投影；视觉模块保持冻结。
- 不升级环境、不换模型、不重新初始化LoRA、不新增adapter、不将两个adapter意外叠加。
- 父检查点：`S96`（61001 t96）与`REP96`（61003 t96）。二者均属于已有R0源训练历史。不是两个模型尺寸，也不代表广泛模型泛化。
- 历史服务器位置与COMMIT记录在`sources/checkpoints_historical.json`。当前可读性仍需服务器检查。缺失不自动重跑源轨迹。

### 3.2 冻结teacher与训练student分开

teacher是父检查点的冻结副本，只负责发现。不得随着任何学生更新；不得用后训练后的更强模型补齐原teacher没找到的题。

每个学生从相同父权重重新恢复，并使用**新的SFT AdamW状态**。旧Adam动量不带入SFT；所有SFT臂同样处理。这是新的训练干预，不是原RL轨迹的精确延续。

R0参照同样明确使用fresh-AdamW版本，名称为`R0_RESET32`，不用它代表历史动量连续训练。恢复断点时则必须恢复本轮已经产生的Adam、scheduler、sampler和RNG，不能再次reset。

## 4. 新数据：先解决清洁性与有限数值域

### 4.1 主任务不改变

四整数世界，观察恰好一处污染，可靠关系，0–99合法输出；cross采用star，trend采用原等差关系；模型仍输出四个数，不输出推理、不输出下游答案。保留sum4/difference_pairs/range4，只在评价时由规范世界执行。

### 4.2 明确的新数值分布

本轮主数据定义为`FRESH_VALID_0_99_v1`：**真值和观察都在0–99**。这保持输出合法域和结构，但真值生成范围宽于历史ID的0–49，不能称为历史ID的精确复现。理由是历史0–49非恒定trend仅有784个有序世界，去掉反转后只有392个世界轨道，继续要求数百个与旧数据不重合的新实例会产生容量问题。

在0–99中非恒定trend有3234个有序世界，按反转合并后1617个。新T/V/E共需432个trend世界轨道；启动前先减去历史已暴露轨道和预留轨道，报告真实剩余容量。不足时报告具体缺项，不偷偷扩到100以上或放松唯一性。

**这是有意登记的数值范围扩展，不是免费的OOD鲁棒性结论。** 所有结果同时报告`truth_all_in_0_49`与其补集；某组太小就如实报告，不制造等数量。新数据本身也可能改变协议效果，这是本轮要测的而不是需要隐藏的问题。

### 4.3 数据表

| 集合 | 数量 | 用途 | 可否用于模型参数更新 |
|---|---:|---|---|
| T_train | 384个基础符号实例，cross/trend各192 | 固定teacher发现、SFT目标、R0参照 | 可以 |
| V_selection | 96个基础符号实例，各48 | 选择固定单协议；运行诊断 | 不可以 |
| E_test | 384个基础符号实例，各192 | 冻结设计后的最终互补和回迁检验 | 不可以 |
| G_guard | 96个提示 | 保持性：E中32个cross和32个trend的图像接口＋32个独立duplicate符号题 | 不可以 |
| R_replay | 64道历史train-only、已有验证正确O0输出 | 各SFT臂共同的保持性回放 | 可以 |

G中的图像与E的符号题共享基础实例，是有意的同场景接口对照，不是独立样本；bootstrap按基础实例捆绑。主pX不混入这些容易图像题。

R优先32个符号duplicate、16个符号cross、16个符号trend。只从历史真实训练输出中挑选已通过公开验证的X；不得从P/D/U22、V/E或旧评估回答中拿“便宜的正确标签”。按固定scene ID排序取样；不足时使用同一个预先登记的可得组成，所有臂相同，记录缺项；不能根据新测试效果挑回放题。

### 4.4 分层与独立性

T/V/E内各家族×污染位置均衡；cross内中心位置×污染位置16格均衡（T/E每格12题，V每格3题）。任务操作按格内轮转，避免与污染位置固定绑定；chart metadata同样只作预定轮转。模型提示不得包含j、DPE、真值、失败标签。

生成只按元数据、合法域和唯一性拒绝，不按任何模型输出拒绝。每个新实例独立求解四个可能污染位置，每位置枚举0–99候选，必须恰有一个与观察相差一处的合法解。

去重采用保守的`truth_orbit_key=sorted(true_world)`，在历史已训练/生成过回答的场景和新T/V/E/独立G之间全局排除；这比仅比较prompt ID严格，也会排除某些并非完全等价但具有相同数值集合的实例，需记录该选择。R属于历史训练回放，不作为新独立集。

同一数学实例的变量置换、关系行重排、B1和输出排列必须保留同一个基础实例ID。不能让原题在T、其重命名版本在E。另一方面，**不按任意仿射变换把全部trend当成一个模板剔除**：本轮检验模板内数值泛化，不是新关系类型泛化。

历史排除索引只需遍历小元数据与已完成任务清单一次；不需要重新解析全部40万回答，不需要反复模型哈希。来源不明但已经被研究人员看过结果的题，一律当开发暴露。

### 4.5 新测试的冻结规则

在任何E模型输出产生前，冻结：任务集合、模板、teacher、协议预算、训练臂、SFT步数、LR、主比较和报告规则。V可用于选择固定单协议；不得用于选择主训练步数或按臂调LR。

E可以在teacher推理阶段生成并封存，与学生训练并行，但E标签/结果不供发现或训练代码读取。所有已登记模型在E完成以后一次性揭晓。若看E后修改方法，原E转开发，新确认必须另建，不能继续称其未见。

## 5. 生成和验证：最容易出错的合同

### 5.1 冻结协议集合

- trend：O0、L11。
- cross：O0、L11、B1。
- 固定主组合：trend用O0＋L11，cross用O0＋B1。
- 只复用上一轮已完成的确切system/user模板和关系行规则。不要同时引入草稿、提示位置、变量重标号、temperature扫描或新的Q奖励。
- L11保持变量含义，命名输入按d,c,b,a列举，要求同样次序输出；B1的首叶按固定原输出顺序确定，不读取污染位置。

### 5.2 坐标映射

若请求输出顺序为sigma，严格使用：

```python
canonical[sigma[k]] = emitted[k]
```

只执行一次。拒绝bool、float、额外解释、非四项数组。合法域仍为0–99。无法解析、越域、截断均保留原始记录；如果长度上限结束但整个输出已经是合法数组，按旧评分约定评价，不因无EOS自动判错。

不能搜索24种排列选出能通过的那一个；数值集合正确只是诊断，不能补救主标签。

### 5.3 方法实际使用的验证器

对规范输出y，只检查公开输入：

`V_public(y) = legal(y) AND all_original_relations(y) AND edits(y, observed)==1`。

在该唯一解任务中它等于X。B1样本仍用**原关系**验证。只在隔离的审计代码中把它与solver真值比较；不从solver给失败生成补答案。如果出现公开验证通过但不是唯一真值，先停该数据单元并修复契约，不能把错误标签继续送进SFT。

同时保存X/S/W/I、C_orig/C_display、F/B/M、copy、原锚点保留、PTLC V1/V2、全部别名。F/B/j/真值属于审计权限；不会成为本轮teacher提示或动态路由输入。PTLC各匹配可以重叠，不相加成“潜在程序权重”。

### 5.4 推理设置与复用

T=1、top_p=1、top_k=0、max_new_tokens=64、thinking=false、无grammar约束，先复用已经验证的uncached-prefix生成。每个draw的种子由(parent,split,task,protocol,role,repeat,draw)确定，**协议必须参与种子键**，避免把不同协议意外设成相关的“独立抽样”。

生成时已有chosen-token log-prob可以保留；不再额外给全部候选做完整似然评分。不同协议不能因规范输出相同就共用生成；只允许完全相同prompt和相同请求ID作严格缓存。

## 6. 发现实验与可执行验证组合

### 6.1 原子样本库

T中每个父teacher做两个独立发现repeat，每个协议每题16次；V和E只做一次每协议16次。分协议固定运行16次，不看到某题成功后再追加或删掉该题。失败、无终答也计入预算。

原子样本库是为了复用多个已冻结策略的比较。**物理总开销是所有实际生成的和；模拟某个部署策略只消耗其登记的16次子集。两种成本都报告。** 不把研究阶段同时测所有协议的额外费用藏掉。

### 6.2 T上的三个发现集合

- `J_O0`：O0的16次内至少一次通过。
- `J_MIX`：O0的前8次或固定家族companion的前8次内至少一次通过。不是O0的16次再加8次。
- `J_SINGLE`：V上按家族预先选定的最佳固定单协议，其16次内至少一次通过。

V上的最佳单协议按平均pass@16/发现覆盖选择，精确并列优先O0，再L11，再B1。每个parent各自选择、两个T发现repeat共享该选择。不要根据T或E选择协议。

三个集合不必嵌套。必须报告交集、`J_MIX−J_O0`及`J_O0−J_MIX`，并按固定家族/位置/DPE风险/数值范围分层。原协议16次没成功只能叫“本预算未发现”，不叫零支持。

每题保留一个规范正确JSON，不按成功次数重复加权。保存所有成功来源，但训练view只含O0输入和同一个规范目标。

### 6.3 E上的两调用验证组合

预先比较：O0重复2次、每个可用固定协议重复2次、固定O0＋家族companion，以及V选择的最佳固定单协议和最佳固定二协议（允许重复）。

V的两调用最佳者按pass@2效用选择；**它与16调用发现的最佳单协议不是同一个选择问题，分别保存两个配置。**

每题可构造8个独立trial：同协议使用第2r、2r+1次；异协议使用各自第r次，或报告利用全部独立样本的无偏估计。所有策略共享的原子样本作为同一随机变量，统计不能假装不同策略结果相互独立。

真实可执行规则：先执行首协议，公开验证通过则返回；否则执行第二协议，验证通过则返回；仍失败则标记无已验证解，并保留两条回答，不调用solver。离线按照预先固定规则回放原子样本，是该算法效用的估计，不能谎称额外真实部署调用已经执行。

报告固定两调用预算效用与early-stop期望调用数。O0成功后跳过第二次的节省可离线计算，但实际wall-time节省需实测，不能仅用计数推成GPU加速。

### 6.4 验证组合的强基线

不能只与一次O0或两次O0比较。必须同时给出：最佳固定单协议重复、最佳固定组合及成本。E上的事后最佳只作为描述性oracle，不能当已实现路由器。

直接符号求解器对这个任务可给出完整解：将其作为任务可解性/验证契约参照，不把模型方法宣称比求解器更实用，也不把solver当未知任务上免费的通用组件。

## 7. 训练数据：以题为单位规范化，合并严格相同训练臂

对每个(parent,repeat)，构造：

| 臂 | 焦点数据 | 信息来源 | 核心作用 |
|---|---|---|---|
| SELF_O0 | J_O0 | 原协议生成＋公开验证 | 同预算筛选自训练基线 |
| SELF_MIX | J_MIX | 固定多协议生成＋公开验证 | 主方法候选 |
| SELF_SINGLE | J_SINGLE | V选定固定单协议生成＋公开验证 | 强单协议发现基线 |
| GOLD_ALL | T中全部384题 | solver标签，显式特权 | 指定训练设置的可训练性/覆盖参照，不是性能上界 |
| GOLD_MATCH_MIX | T中的随机匹配子集 | solver标签，显式特权 | 分离数据量与发现分布；仅S96执行 |
| REPLAY_ONLY | 无焦点数据，仅共同R | 历史已验证训练输出 | 分离共同回放本身的贡献；仅S96执行 |

GOLD_MATCH_MIX按family×污染位置精确匹配J_MIX各格的题数，从该格T中用独立固定种子均匀不放回抽取；不要求排除J_MIX，允许真实重合并报告。它控制了样本量和部分结构组成，但不能控制全部难度，不据此宣称唯一识别了发现偏差。

每个集合内按题等权，不按某题成功频率、难度分数或PTLC匹配重新加权。规范目标使用同一个序列化函数（无空白JSON）及一个终止token。

**solver-J与self-J不另开臂。** 若题集、O0输入、目标token、权重相同，它们的训练损失完全相同。若SELF_SINGLE选到O0，或其他数据集完全相同且初始状态、种子、训练配置相同，记录alias并只训练一次；不要因名字不同重复作业。不同parent或不同训练随机seed不得alias。

空集合记为`NO_VERIFIED_TARGETS_RETURN_PARENT`：该流程返回父模型、不执行该学生的SFT，不填solver标签，不丢弃该parent/repeat。最终效用复用父O0在E的前8条与G的前4条作为同预算评价别名，发现失败属于端到端结果；另外报告有数据条件下的SFT分析。极小非空集合允许按预定规则训练，但报告重复暴露和过拟合风险，不把没有数据/低多样性混为不可学。

## 8. SFT：准确的目标、初始化与采样

### 8.1 损失

每条规范目标的completion NLL为：

`ell_i(theta) = -(1/L_i) sum_{t=1}^{L_i} log pi_theta(y_t | O0(i), y_<t)`。

包括恰好一个EOS，prompt、assistant header、padding不计损失。每个序列先除自己的目标长度，再按题平均，避免数字分词长度改变隐含题权重。

每步16个有效槽：12个焦点例、4个共同回放例，目标为：

`L = 0.75 * mean(ell_focus) + 0.25 * mean(ell_replay)`。

REPLAY_ONLY仍使用`0.25 * mean(ell_replay)`，**不能把它重新归一化成1.0**；不执行12个空槽的forward。这样它具有与其他臂相同的回放贡献与步数。

焦点数据每轮shuffle后顺序遍历，用完再shuffle；不为某个arm额外提高hard例权重。回放题序按(parent,repeat,update)固定，跨臂相同。焦点集不同，题序当然不能逐题相同；共同题、重复次数和unique exposure均输出。

### 8.2 优化设置

- 每臂256步；保存0/32/64/128/256，主终点固定256，不按E或各臂V结果选最好step。
- AdamW：lr=1e-5、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、grad clip=1.0；前8步线性warmup，此后constant。
- dropout保持0；BF16基座；可训练LoRA张量保留已验证dtype，不偷偷量化。
- microbatch最大4，梯度累积得到总16个槽；OOM时只减microbatch、增加累积，不改学习率或数据权重。
- 两个repeat使用73101、73102；每个repeat同时对应自己的T发现采样。它们是**全流程重复（发现＋优化随机性）**，不是固定数据下纯优化方差实验。
- S96执行6臂×2repeat；REP96执行前4臂×2repeat；最多20次SFT。

所有结论只覆盖两个已有源轨迹的这组起点。四个parent×repeat单元不能当四个独立源模型种子，更不能把每道题视为独立训练重复。

### 8.3 训练实现必须处理的细节

1. 从parent恢复精确的96模块LoRA权重后，重新开放原白名单的requires_grad；基座和vision不变。记录trainable名、形状、dtype及计数一次。不得再次调用随机adapter初始化覆盖权重。
2. 保留原O0 chat template、thinking=false和assistant起始token。拼接prompt token与独立编码的规范目标token，明确第一目标token由哪一行logit预测；不要按字符长度估计mask。
3. EOS采用实际tokenizer约定的唯一主终止token，写入配置。推理可以接受已有多个EOS，但监督不能把两个都拼上，也不能把EOS当padding去掉。
4. 所有prompt和目标预tokenize一次；max_total_tokens=1024只作完整性上限，超长时报告并统一处理，禁止截断可靠关系或正确目标。
5. 不做sequence packing；不同样本不能共享attention历史。右padding及attention mask必须实际通过Qwen hybrid层测试。
6. 每次forward/reset清除跨请求position/cache状态；训练不复用上一个样本的KV或线性attention状态。
7. loss采用FP32 log-softmax / CE accumulation；检查真实非零LoRA梯度、更新后的参数变化、基座无梯度；不能只看loss降低。

### 8.4 高效teacher forcing与旧路径不同，怎样验收

本轮SFT是显式的新目标，不需要与旧GRPO的behavior概率形成ratio=1。允许使用模型正式的全序列causal teacher-forcing，在一次forward里计算目标段loss，而不是每token重跑前缀。

但历史Qwen实现有hybrid层及BF16路径差异，因此不能只换一个`SFTTrainer`就认为接入成功。小样本桥接必须检查：目标shift、prompt mask、padding、EOS、future-suffix独立性、一次完整梯度与分microbatch梯度、单例与batch loss。

量纲分开：
- 整数token/位置/mask必须精确一致；future信息泄漏、使用错误位置logit、空梯度直接阻塞SFT。
- CPU FP32小模型契约采用明确allclose；真实BF16先测重复同路径误差，再报告每token及每sequence logp差、loss差、梯度夹角和相对范数。**不能用一个任意1e-12阈值阻塞所有GPU，也不能用投影或重标定掩盖系统误差。**
- prefix与full路径不逐位相同不自动否定新SFT；关键是正式causal目标正确、误差稳定可解释、训练路径在所有臂一致、终点评价固定。若出现大于同路径数值波动且显著改变梯度方向的差异，先定位kernel、dtype、mode和state，再批量开SFT；冻结发现任务可继续，不跟着停。
- 至少保存8个固定train-only例的逐token诊断和两个更新后的生成。失败不退回全矩阵逐token慢路径作为默认；提交具体诊断，限定在小例上排查。

## 9. R0_RESET32：小规模RL上下文参照

每parent×repeat一臂，共4次；每次32步、B=4、K=8、reward=2*1[X]，旧组归一化epsilon=1e-4、Lnorm=64、PPO clip=0.2、grad clip=1。

从同父权重恢复后新建AdamW，参数同SFT，前8步线性warmup后constant；不带入旧动量。每个repeat的128道训练prompt按T的固定分层shuffle选定，两parent同顺序。每步从当前策略真实采样，不复用teacher样本；保留行为logp、真实优势和旧PPO目标。

精确全零优势可跳过无意义的反向图，但仍按本轮优化器语义设置零梯度并执行相应Adam步，不能跳过历史动量作用。实现只需一次小例等价测试，不每步重复完整校验。

**它不是等GPU时长或等步数的SFT公平胜负基线。** 它只提供现有主奖励在32步内的上下文响应；报告生成tokens、优化processed sequences和GPU秒。主因果比较仍是使用相同SFT目标/预算的发现策略之间。

本轮不直接实现SA-MRPO/AMRP新算法，不因为SFT赢R0就宣称胜过多奖励方法。后续若恢复多奖励机制主张，需要相应正式对照。

## 10. 评价：原协议是主目标，错误结构与覆盖是解释

### 10.1 冻结teacher的E评价

每协议每题16条，用于验证组合、pass@k和固定协议选择的独立检验。不得回流到J。parent的O0基线可从这个合法既有单元复用，不为不同学生重复抽同一个父模型。

### 10.2 SFT评价

- step64、128：V96，每题4次；仅作运行曲线，主终点和超参数不因看到V而改变。
- step256：E384的O0，每题8次；G96每提示4次。
- 固定16个训练sentinel在64/128/256各greedy一次，用于区分是否拟合；它们是训练题结果，绝不能与E合并。
- 日志每步保存focus/replay loss及分家族loss；需要看loss曲线，不必每步生成回答。
- R0在step8观察V，step32观察E/G；不执行其他大协议矩阵。
- 默认不为所有学生重新跑O0/L11/B1全集；本轮首要检验是原协议回迁，避免评价再次占满十天。

### 10.3 固定语义指标

主：O0 pX，cross/trend各1/2，家族内题等权。其次：pass@1/2/4/8、题级成功分布、pS/pW/pI、原关系全成立错误、copy、修改数、F/B、原固定锚点保留、非copy非truth的PTLC匹配。

非法但可解析越域的整数数组：主类别I，另记录算术关系和程序匹配；这些诊断不能给其主指标加分。协议专属参考与共同语义事件分开输出。

“以前未见X的题”的子集只能由训练前O0固定16次样本形成并冻结，命名`PRE_ZERO16`。同时报告预先由元数据定义的DPE₁正负和全体；不能把训后失败集合倒填到前置选择。PRE_ZERO16不等于pX=0。

对post-SFT E中每题8次回答，可直接构成一组K=8：记录all-X、mixed-X、no-X、有无部分修复、C是否变化。它是**固定评估面板上的对比状态**，不冒充实际训练抽题分布；mixed组比例下降若伴随all-X上升并非坏事。

### 10.4 保持性与代价

G的图像配对检查防止纯符号SFT明显损害原多模态表现，duplicate检查简单任务。总体G不与E加权成一个可掩盖困难题的高分。

主SFT比较步数、batch和focus/replay配比完全一致；但不同J导致每题重复次数不同，必须报告unique tasks与每题曝光分布。发现代价、SFT代价、最终推理代价各自记录，不用solver标签臂的零模型发现成本冒充自训练效率。

## 11. 统计、阈值与结论分层

### 11.1 本轮不使用旧的极窄统一门槛

不再要求每个群体都以同时Hoeffding界证明下降不超过1pp，不要求全部概率估到0.0125pp。不把“区间跨零”改写成两方法等效。

- **主检验只有一项**：E上SELF_MIX−SELF_O0的O0 pX差。
- SELF_MIX−SELF_SINGLE、冻结组合vs固定重复、GOLD/REPLAY/R0及各子组都明确标为关键次要或探索。
- 实际意义参考2pp只帮助解读，不作为“继续/终止”的门槛。样本量设计关注约5pp的效应，而不是保证检出任意微小优势。
- E固定384题；根据V可提前报告预计MCSE/MDE，但不按E结果加题加seed直到显著。

### 11.2 独立单位

两parent是两条已有源轨迹；每parent两个发现＋优化repeat嵌套。主报告给4个单元各自的差和两parent平均，再给共同任务面板上的描述性配对区间。不能以4个pipeline重复声称有4个独立基座，也不能让回答总数替代训练随机性的样本量。

按基础scene在家族内bootstrap5000次，保留该scene的全部协议、parent和repeat对齐，得到**条件于当前检查点和已执行运行**的场景区间。固定面板的采样误差另报；0/n和n/n采用95% Wilson二项区间，不把MCSE=0当确定无误差。涉及共用O0原子样本时，重采样同一原子记录一次再同时计算所有策略。

各pipeline训练差不足以提供稳定的种子总体CI；如实列出，不从2个seed拟合“普遍算法增益”。

### 11.3 结果解释矩阵

| 发现/训练结果 | 可以说 | 下一步讨论，不自动执行 |
|---|---|---|
| 新题互补不成立 | 历史组合没有在该新分布复现 | 核对数值/结构条件，保留简单单协议 |
| 新题互补成立、J_MIX没有更多题 | 两调用与16调用覆盖不是同一目标，或O0已在16次赶上 | 讨论调用预算而非强推训练收益 |
| J_MIX扩展，SELF_MIX不优于SELF_O0 | 更多发现尚未转为更好的SFT结果 | 看新增题的结构、拟合、遗忘和样本权重 |
| GOLD_ALL训练题也不拟合 | 当前数据/目标/优化实现或强度不足 | 小规模排查，不写“正向不可学” |
| GOLD_ALL拟合但E不提高 | 当前监督主要局限于训练实例或存在干扰 | 新结构/规范鲁棒性诊断 |
| SELF_MIX与SELF_SINGLE都提高且相近 | 更有效的固定协议发现已足够 | 不强行加观测器 |
| SELF_MIX在E和保持性上表现更好 | 验证后的多协议发现能够支持原协议回迁，在本设置成立 | 再讨论失败条件路由或RL采样/奖励协同 |

精确恢复提高不自动证明内部机制更“真实”；需要按覆盖、前置难题、错误结构和保持性共同报告。

## 12. 运行阶段与并行

### Stage A：接入与新数据（CPU＋少量GPU技术检查）

1. 读取当前代码及历史checkpoint元数据；生成新T/V/E/G、R。
2. 冻结protocol、cohort和角色权限；编译O0/L11/B1；完成CPU契约。
3. 在train-only小例上跑raw生成与SFT shift/gradient桥接；不在E上调格式。

### Stage B：teacher发现与求解器参照并行

- GPU1/2：两teacher的V及T发现；余卡可分repeat或题块。
- 验证集选择固定single后立即冻结；T不同发现策略不再改预算。
- GOLD_ALL可以在代码验收后与发现并行，不要求所有发现先结束；其主设置不按其他臂效果选择。
- teacher的E任务可排入空闲卡，结果封存不用于开发；不要等SFT结束再重复加载全部parent。

### Stage C：学生训练

以S96 repeat0的SELF_O0、SELF_MIX、SELF_SINGLE、GOLD_ALL作为第一批完整可解释组合，不能只先跑预期赢家。随后按冻结矩阵跑其余，遇负结果不删臂。第一批阶段报告只展示V和训练sentinel的诊断，不提前揭晓E/G结果来改变后续臂。

建议每卡一个单独模型任务；同GPU在任务结束后显式卸载前一adapter和optimizer，不在多个活跃训练作业之间复用可变模型对象。四卡时缩减并发，不改变实验参数。

### Stage D：final评估与分析

final E/G在每个学生结束后排队，配置不可变；最终统一揭晓。优先交付数据清单、发现覆盖、训练曲线和技术问题，不承诺未经吞吐测量的完成天数。

完整配置最多20个SFT作业＋4个R0短作业，不是旧88分支扫描。可通过数据集精确alias进一步减少；不得因为一个小集合恰好同名却内容不同而合并。

## 13. 成本与实施效率

本包`workload.json`由`protocol.json`计算。默认上限（尚未扣严格alias）：

- 冻结teacher T/V/E：99,840次回答；
- SFT最终E/G：69,120次回答；V中间诊断15,360次；训练sentinel960次；
- parent保持性：768次；R0训练4,096次、评价15,360次；
- 合计：**205,504次登记模型回答**，技术检查另列，默认最多128次。
- SFT共5,120个optimizer step，R0共128步；SFT实际目标序列75,776条（REPLAY_ONLY没有12个焦点forward）；这与生成回答数不同。

这是上限矩阵，不是对wall-time的承诺。先实测代表性任务的输入/输出token、GPU秒与峰值显存，按symbolic/image、generation/SFT/RL分别估计。若历史逐token梯度路径会把SFT拖回数天，优先修复全序列causal实现，不把大量重复慢评分当作科学要求。

必要检查：首次导入小清单与恢复一次、关键数据角色、token/mask和断点状态。取消每回答/每step全模型哈希、重复全仓数千测试、重复解析历史全包、给全部候选重新算likelihood。任务按几十个scene分片，不为每条回答提交Slurm。

## 14. 新模块、伪代码与数据合同

建议新增：

```text
src/verified_discovery_transfer/
  config.py                 # protocol读取、派生数量与合法配置
  fresh_cohort.py            # 有限容量、orbit去重、T/V/E/G与R
  public_tasks.py            # 模型可见输入与solver审计字段隔离
  portfolio.py               # 原子stream、预算分配、verify-first执行
  discover.py                # 固定teacher、J集合、实际来源
  canonical_targets.py       # 唯一规范目标、去重、EOS、alias
  sft_runtime.py             # 新SFT可训练恢复、fresh AdamW
  sft_loss.py                # completion-only、每序列均值及replay权重
  sft_runner.py              # checkpoint、sampler、resume
  r0_reference.py            # 小规模fresh-state on-policy参照
  evaluate.py               # O0固定面板与对比状态
  statistics.py             # 题等权、共享原子样本、paired区间
  reports.py                # 事实、解释、缺项、成本
  cli.py
```

发现核心：

```python
for parent in parents:
    teacher = load_frozen(parent)
    fixed_single = choose_on_V_only(teacher_sample_bank_V)
    for repeat in (0, 1):
        bank = generate_T_all_registered_protocols(teacher, repeat)
        for policy in ('O0_16', 'FAMILY_MIX_8_8', 'V_SINGLE_16'):
            J = {}
            for task in T:
                for sample in policy_budget_view(bank, task, policy):
                    y = parse_then_inverse_permute_once(sample)
                    if public_verifier(task.public, y):
                        J[task.id] = canonical_json(y)
                        break
            export_training_view(O0_prompts(T), J)  # 不使用solver补空项
```

训练核心：

```python
student = restore_parent_trainable_lora(parent)  # 恢复而不是重初始化
optimizer = new_AdamW(student.lora_parameters, protocol.sft.optimizer)
for step in range(1, 257):
    focus = focus_sampler.next(12)  # REPLAY_ONLY则无
    replay = shared_replay_sampler.next(4)
    loss = .75 * focus_mean_nll(focus) + .25 * replay_mean_nll(replay)
    # REPLAY_ONLY第一项严格为0，第二项不重新归一化
    backward_accumulated(loss)
    clip_grad_once(); optimizer.step(); scheduler.step()
    save_or_evaluate_at_registered_steps(step)
```

`max_microbatch`只影响实现，不能让某个microbatch中的短序列或回放被重复除batch；累积后梯度必须等于完整定义目标。

## 15. 交付内容与停止边界

必须交付：

1. `FIRST_STAGE_REPORT_zh.md`：新数据分布、历史排除范围、teacher和SFT接入、实测吞吐、具体风险。
2. `discovery_summary.json`与逐题表：所有J、交集/新增/丢失、预算曲线、V固定选择。
3. `TRAINING_REGISTRY.json`：每个学生的parent、数据view、repeat、Adam初始化策略、alias、checkpoint路径；原始训练日志。
4. `FINAL_FINDINGS_zh.md`：回答第0节四问，不仅给排行榜。
5. `EXECUTION_SCOPE.json`：本轮到底生成多少、训练多少、用过哪些角色数据、未执行项。
6. `evidence/`：逐题计数、所有原始失败回答、规范标签、训练曲线、保持性及成本。真实checkpoint在服务器保留，可复现的实际位置写入清单，不强行塞进交接包。
7. `DISCUSSION_FOR_CLAUDE_zh.md`：事实、候选解释和仍无法区分的问题分开。

停止只针对数据泄漏、验证器错误、模型/断点恢复失败、非因果训练、非有限梯度等技术问题。没有收益、PTLC不符、样本置信范围宽，不是删结果或自动扩大实验的理由。

本轮结束以前不自动启动：失败条件路由器、更多奖励权重搜索、加入Q或损坏新奖励、LoRA rank变更、teacher刷新、多轮自训练、off-policy重要性修正、SFT后再加长RL。需要后续数据讨论后另立协议。

## 16. 对最终研究定位的限定

本轮有望回答的是：**可验证的协议互补是否改变成功训练数据的覆盖，并在原协议上留下可测量的参数学习收益。**

若成立，它为“概率观测→训练支持设计→后训练”建立一个连接；仍未单独证明复杂概率观测器优于简单固定策略，也未解决所有多奖励调权。若不成立，发现、拟合、迁移、遗忘被分开，仍能确定下一步应修改哪一层。

不要把SFT包装为新的GRPO，不把监督成功称为一般能力上限，不把失败称为架构不可学，不把测试过的实例重新声明为未见。保留这个边界，才能使后续SSVC建立在可靠的证据上。
