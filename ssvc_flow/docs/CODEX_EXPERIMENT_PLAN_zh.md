# 面向 Codex 的实施规范：SSVC 跨规模概率流诊断

版本：2026-09-05。

## 0. 任务边界与完成定义

这是新实验的实现文档。不要直接重写原始 Research Plan，也不要默认其全部理论已在真实模型上成立。先交付能够被复核的概率测量和更新干预工具，再根据结果决定 SSVC。

**本轮首要输出**：新模型在旧协议和新受控协议下的 X/S/W/I 分布；辅助奖励引起的局部和短训练变化；这种变化是否能被联合归一化、采样支持和共享更新解释；哪些原有问题改善，哪些仍在，哪些变成新问题。

**不允许的完成方式**：只给平均准确率；只跑一个长训练；为了找到负面现象反复换模型/改任务；用不同起点或不同 rollout 比较 λ；把未观测类别记成理论零概率；把虚拟步、采样重加权预测当作真实训练结果。

必须分阶段执行，并提供每阶段的 `status.json`、`report.md` 和 `manifest.json`。没有研究者授权，不得自动进入多种子确认训练或完整 SSVC 训练。

---

## 1. 来源与不可更改的历史事实

### 1.1 原计划

研究者的 `Research Plan.pdf` 日期为 2026-08-25，标题为 *When “Valid” Does Not Mean “True”*。[P1]

历史核心分层：X=精确世界；S=合法且执行答案正确但世界错误；W=合法且执行答案错误；I=非法/不完整。qX=P(X|V)，不是 P(X|答案正确)。C2 的 26.17%→44.39% 是答案正确集合内的纯度，不能与 C3 的 P(X|V) 混用。[P1, §2.5]

### 1.2 找到的历史 C3 配置

历史项目材料明确记录：[P2]


| 项目        | 已知值                              |
| --------- | -------------------------------- |
| 模型        | Qwen2.5-VL-3B-Instruct           |
| 训练 prompt | 192                              |
| 组大小 K     | 8                                |
| 优化步       | 192                              |
| 训练重复      | 同一 seed；具体 seed 值须查旧文件           |
| 动作        | 四整数                              |
| A-BIN     | (1,1,0,0)                        |
| X-BIN     | (1,0,0,0)                        |
| A-LEX     | (3,3,1,0)                        |
| X-LEX     | (3,1,1,0)                        |
| 评估        | 88 个配对语义单元、两种匹配条件、每场景 16 rollout |
| 解码        | 自由采样及有限状态约束；包括 free-48 对照        |


**未知**：每更新 prompt 数、确切 LR/LoRA 参数、旧标准差约定、具体系统 prompt、旧 parser 和接口实现。必须从文件审计获取；不能从上述摘要推断。

历史数据存在时只读导入，找不到时，将 `legacy_exact_reproduction=false` 和缺失清单写入报告；可以做新协议实验，但不得称为 C3 精确复现。

---



## 2. 模型选择、定位与设备预算



### 2.1 主模型

使用官方后训练检查点 `Qwen/Qwen3.5-9B`，不是 `-Base`，不是社区蒸馏版，也不是名字里带 VL 的自造模型 ID。[W1–W3]

选择依据：官方发布记录为 2026-03-02；有视觉编码器；语言模型标称 9B；32 个语言层采用 24 个 Gated DeltaNet/linear-attention 层与 8 个 full-attention 层的混合布局。[W1–W3] 它满足“比 3B 大且较新”的需求，不必为了绝对最新改用 27B。

官方 Qwen3.8-27B 更近期，但本轮不作为默认训练模型；仅其 27B 语言权重按 2 字节估算就约 54 GB，尚未计入视觉权重、激活、缓存及优化器。这个数字是容量算术，不是实测显存。[W4]

### 2.2 对照矩阵


| 角色     | 模型                          | 默认任务                          |
| ------ | --------------------------- | ----------------------------- |
| 主实验    | Qwen/Qwen3.5-9B             | 全部冻结诊断；短训练；满足阶段条件后确认训练        |
| 历史锚点   | Qwen/Qwen2.5-VL-3B-Instruct | 统一评估复测；X_BASE/X_VALID 两个短训练对照 |
| 同代规模桥接 | Qwen/Qwen2.5-VL-7B-Instruct | 首先只做冻结评估；必要时两个短训练             |
| 技术后备   | Qwen/Qwen3-VL-8B-Instruct   | 只有 9B 兼容性确实阻塞时启用              |


3B→9B 同时改变代际、架构、预训练和后训练，不能把所有变化归因于参数量。7B 桥接只能帮助分离解释，仍不是严格单变量规模因果实验。没有 7B 奖励干预，不能仅从冻结准确率推出“规模修复了奖励动态”。

后备切换必须写明错误、依赖版本和失败测试，不得因为 9B 没有出现预期问题而换模型。主矩阵不扩成模型排行榜。

### 2.3 显存方案：估算，不是保证

首选单张 **48 GB GPU**，BF16 基座、FP32 LoRA/优化器状态、小 microbatch、梯度检查点、分批生成 K 个样本。主训练不需要同时驻留独立 reference 模型：KL 对照可以顺序禁用 adapter 计算 reference log-prob，或加载只读 reference adapter。

32 GB 可先尝试同样方案的短上下文和 microbatch=1；不保证峰值足够。24 GB 优先考虑单独的 4-bit QLoRA 分支，但必须先证明该模型/所选库的反向传播可用，并对全部比较 arms 使用相同精度。量化结果与 BF16 主结果分开，不能当成同一策略分布。

CPU RAM 建议按 64 GB 规划，单主模型工作区磁盘预算先预留 100 GB 以上，最终由实际下载和 checkpoint 大小报告。不要同时运行占满显存的 vLLM 服务和 Hugging Face 训练进程。

P1 必须实测并保存：权重大小、加载峰值、单条生成峰值、K=8 分批生成峰值、一次反向传播峰值、token/s、update 秒数、峰值 CPU 内存。由实测推算每阶段 GPU 小时；未测前不得承诺某张卡“肯定能跑”或固定耗时。

---



## 3. 仓库与命令契约

Codex 应实现如下功能结构；名称可保持，内部代码可调整：

```text
ssvc_flow/
  configs/
  data/{legacy,generated,manifests}/
  src/
    audit_legacy.py
    generate_worlds.py
    render_charts.py
    constraint_solver.py
    verifiers.py
    model_adapters/{base,qwen25vl,qwen35,qwen3vl}.py
    rollout.py
    likelihood.py
    grpo_update.py
    optimizer_fork.py
    tabular_flow.py
    sensitivity.py
    kernel_probe.py
    decoding_audit.py
    natural_interfaces.py
    ssvc_audit.py
    statistics.py
    plotting.py
    report.py
  tests/
  scripts/
  runs/
```

需要的入口：

```text
python -m src.audit_legacy --root <EXISTING_WORKSPACE> --out runs/P0
python -m src.tabular_flow --config <LOCKED_CONFIG> --out runs/P2
python -m src.rollout --phase smoke --model qwen35_9b --out runs/P1
python -m src.rollout --phase frozen --split dev --out runs/P3
python -m src.grpo_update --phase pilot --arm X_BASE --seed 17 --allow-training
python -m src.report --run-root runs --out reports
```

这些是**需要实现的接口**，不是本设计包已经提供的可运行命令。

`--dry-run` 必须列出 prompt 数、rollout 数、上限 token 数、forward/backward 次数和预算。`--resume` 必须检查模型/数据/配置 hash；不一致时拒绝静默续跑。每条生成要有稳定 sample key，避免崩溃后重复计数。

依赖版本由 P1 解析并锁定，保存 `pip freeze`、源码 commit、模型 revision、tokenizer/chat template/processor hash。不能在正式实验里持续跟随 `main`。配置中的 revision 未解析时只允许 smoke，不允许 confirm。

---



## 4. 数据与可执行真值



### 4.1 两条数据轨道

**L：legacy 轨道。** 原数据、原 prompt、原动作语法、原长度限制均保持，测新模型时只做官方模型输入适配。输入适配必须保存最终 tokenized prompt，注明不能保证跨 tokenizer token 数相同。新模型的性能改善或没有改善都保留。

**N：新受控轨道。** 以下为本次拟议的新设置；不冒充旧实验。

基本真实世界 x*=(a,b,c,d)，取值域 D={0,…,99}。训练和 ID 主集真值在 {0,…,49}；数值 OOD 真值在 {50,…,99}。观察值 o 恰有一处与 x* 不同，仍落在 D；更改位置和幅度由生成器随机种子决定。

主要动作是一个长度恰为 4 的 JSON 整数数组，例如 `[12,18,9,15]`。允许标准 JSON 空白；禁止布尔值、浮点数、字符串数字、额外解释、markdown fence 和五个数字。解析器必须使用完整字符串解析，不使用“正则截取第一个数组”作为主 parser。

合法性 V 只检查语法、长度和整数域，不检查关系约束是否满足。关系正确性是单独的语义验证器，不能暗中并入 V。

### 4.2 唯一解检查

生成完成后，必须求解

C(o,cue)={x∈D⁴：Hamming(x,o)=1 且 x 满足 cue}。

只将 C(o,cue)={x*} 的样本纳入主修复任务。因为恰一处变化，仅需枚举 4×99 个替代值，不需要枚举 100⁴ 个世界。求解器是生成器之外的独立实现。

主集不能包含无解/多解题后仍要求模型恢复指定真值。无解、多解、无 cue 的案例单独成为诊断集，记录可辨识性；适当拒答不能简单当成能力失败。

### 4.3 三种关系家族

**duplicate_encoding**：给一个位置的可信真实值，错误恰在该位置；其余三个数保留。用于检查最简单的修复与格式能力，不用于代表复杂逆推。

**cross_series**：将四个变量作为图节点，提供边两端的和。ID 使用覆盖四个节点的星形图，中心位置做随机化；至少三条独立可审计关系。例如 a+b=30、a+c=21、a+d=27。完整关系与“恰一处错误”共同确定唯一世界。

**trend**：四个值构成等差数列，等价于 a−2b+c=0、b−2c+d=0；允许正/负非零公差，保持数值域。观察中恰一处错误。每题仍用独立求解器核验唯一性。

星形图→路径图/环图作为 cross_series 的约束图 OOD，不能把 duplicate_encoding 也标成图结构 OOD。难度因素分别记录关系数、错误幅度、数字 token 数和拓扑；不要把多个因素合并成一个未经解释的“hard”标签。

### 4.4 图像

保持两个既有视觉域：grouped bar chart、line chart。一个图对应同一个四元组，提供清楚的标签顺序。定义 a,b 为第一个 x 位置的两个系列，c,d 为第二个 x 位置的两个系列；所有 prompt、渲染器和 evaluator 使用同一映射。

用确定性程序渲染，不用图像生成模型。保存原图和 processor 处理后的分辨率/视觉 token 数。默认图片为 768×512，若 processor 自动缩放，保存实际尺寸；视觉 token 上限初拟 768，需在 P1 确认该参数如何正确映射到当前 processor。

不能通过删除关键图例、使数值不可读或缩到极低分辨率来制造模型错误。独立人工 spot-check 至少 36 张，覆盖所有关系×图形×运算单元。

### 4.5 三种答案映射

N 轨道固定为：

- `sum4`：h(x)=a+b+c+d；
- `difference_pairs`：h(x)=(a+b)−(c+d)；
- `range4`：h(x)=max(x)−min(x)。

模型主动作只输出世界，答案由确定性 executor 计算。L 轨道的 h 必须按旧文件，不得被本定义覆盖。

答案纤维大小指 D⁴ 中使 h(x)=h(x*) 的世界数，不是唯一修复候选集合 C 的大小。sum 用整数卷积/DP；difference 用二元和频数的相关；range=r>0 用 `(100-r)*((r+1)^4-2*r^4+(r-1)^4)`，r=0 时为 100。小域 D={0,…,4} 上用暴力枚举交叉验证这些计数。

### 4.6 主接口与自然接口分开

N 主协议提供两种可控输入：

- `SYMBOLIC_FRESH`：新调用，仅有错误观察、可信关系、修复指令，无图像。
- `IMAGE_CUE_FRESH`：新调用，含真实图像、同一错误观察、同一可信关系；图像和 cue 对同一 x* 一致。

两种都是可控输入；`IMAGE_CUE_FRESH` 不得冒充模型自己产生错误后的 I3/I4 自然继续。自然错误接口见 P8。

主训练一半样本用 SYMBOLIC_FRESH，一半用 IMAGE_CUE_FRESH，按关系×图形×运算单元精确平衡。确认评估对同一 base_scene_id 同时测两个接口，从而可以观察跨接口差异。训练中不提供正确数组或正确最终答案。

### 4.7 新协议的固定 prompt 模板

下列是 N 轨道需要锁定的英文模板；L 轨道继续使用原模板。字段值由机器生成，不能让语言模型自行改写。

System:

```text
You are repairing a four-integer chart record. Return only one JSON array
containing four integers from 0 to 99. Do not include an explanation.
```

User（两个接口共用主体）:

```text
The record order is [a,b,c,d]. For the chart, a and b are the two series
at the first x position; c and d are the same two series at the second position.
The observed record is {observed_json}.
Exactly one value in this observed record is wrong.
The following relationships are reliable:
{cue_text}
The downstream calculation is: {operation_expression}.
Recover the correct record. Return [a,b,c,d], not the downstream answer.
```

IMAGE_CUE_FRESH 在同一 user message 放入真实图像，并在主体前增加固定句：`The attached chart also shows the true record.`。SYMBOLIC_FRESH 不插入图像或该句。错误观察和关系均为任务输入，系统不得拼入 truth_world、正确答案或 solver 解。

`cue_text` 用固定字符串格式：known-value 写 `b = 18` 等位置锚点；cross-series 按索引排序输出 `a + b = 30` 等边关系；trend 写 `b - a = c - b = d - c`。`operation_expression` 分别为 `a+b+c+d`、`(a+b)-(c+d)`、`max(a,b,c,d)-min(a,b,c,d)`。输出要求保持一致。

下游运算对模型可见，但正确运算结果不可见。这让答案验证器的目标明确，而不泄漏答案。若 calibration 发现模板本身被普遍误解，只能在 confirm 封存且不查看奖励对比结果的前提下做一次版本化修订。

### 4.8 固定划分

每个 base scene 的所有渲染、cue、接口、反事实和错误变体只能在同一 split 内。按 base_scene_id/真值结构 hash 去重，不按单条 prompt 随机切分。


| Split        | base scene 数 | 用途                             |
| ------------ | ------------ | ------------------------------ |
| calibration  | 144          | 模板、长度、显存、可辨识性和效应精度试测           |
| train        | 576          | 新协议 RL；每场景固定一个主接口，两个接口各 288    |
| control      | 144          | 更新敏感度和 SSVC 候选控制；不作为训练 rollout |
| dev          | 144          | 短训练与诊断；两接口共 288 prompt         |
| confirm      | 288          | 冻结协议后的确认；两接口共 576 prompt       |
| ood          | 288          | 图结构、数值平移、渲染/表达扰动，各 96 场景       |
| natural_pool | 最多 2048      | 先读图产生自然错误；不按想要的结果无限扩池          |


calibration/train/control/dev/confirm 在 3 家族×2 图形×3 运算的 18 个单元中平衡。OOD 图结构子集仅 cross_series；另两个子集覆盖所有家族，报告其不同组成，不强求所有 OOD 格子同样解释。

所有数量均为拟议默认值。只能在查看 confirm 前根据预算/精度审计形成版本化修改，并记录理由。不能见到结果后删掉“没出问题”的单元。

---



## 5. 事件标注与数据格式



### 5.1 唯一主分类函数

```python
parsed = strict_parse(raw_completion)
if parsed is None:
    category = "I"
elif parsed == truth_world:
    category = "X"
elif executor(parsed, operation) == executor(truth_world, operation):
    category = "S"
else:
    category = "W"
```

必须验证四类互斥且完备。格式不合法即使文本里含正确答案，主分类仍为 I；可额外记录容错解析结果，但不能替换主指标。

达到 max_new_tokens 不自动等于 I：若实际字符串已经完整且符合主语法，按 parser 标注并另记 `stop_reason=length`。未闭合数组才是不完整输出。改变此规则必须在 L/N 分别写明，不能训练和评估不一致。

有模型自行生成最终答案的扩展任务时，另记 `model_answer_correct` 和 `answer_consistency`。四类主标签仍依据世界和确定性 h；不能把模型心算错误硬塞进原本的嵌套关系。

### 5.2 每条样本至少保存

`run_id, phase, model_id, model_revision, dtype, adapter_hash, optimizer_step, train_seed, sample_seed, base_scene_id, prompt_id, split, interface, constraint_family, chart_type, operation, graph_topology, truth_world, observed_world, changed_index, cue, solution_count, fiber_size, image_path, image_hash, prompt_hash, prompt_token_count, image_token_count, generation_config_hash, raw_completion, token_ids, completion_length, stop_reason, parsed_world, category, syntax_valid, constraint_satisfaction, executor_answer, model_answer(optional), logprob_behavior_sum, logprob_base_sum, elapsed_seconds`。

训练额外保存：`group_id, K, raw_rewards_by_channel, reward_sum, reward_mean, reward_std, epsilon_convention, advantages, zero_variance_flag, contains_X, contains_S, contains_I, loss_reduction, old_logprobs, actual_step_norm, grad_norm_preclip, grad_norm_postclip, clip_fraction, reference_kl`。

保存逐 token log-prob 的压缩数组即可，不保存所有词表 logits。例外是少量 cache/数值一致性测试。不要在报告中暴露服务器凭据或访问 token。

---



## 6. Qwen3.5-9B 专项实现规范



### 6.1 官方类与模板

P1 检查当前固定版本是否支持 `Qwen3_5ForConditionalGeneration`，使用官方 processor/chat template。[W2–W3] 模型卡中的最低/开发版本信息不能替代实测。发现不同版本的 AutoModel 名称变动时，优先使用明确的模型类并锁定已通过的版本。

主训练设置 `enable_thinking=False`，检查这一开关实际作用在最终模板上。模板插入的控制 token 不是模型生成输出，不计入动作格式错误；模型自己生成的解释/think 内容不能被主 parser 静默删掉。

Thinking 只在独立诊断分支启用，不与主短数组任务混合；给独立总 token 预算并报告 reasoning/final 分段和截断。该分支只回答推理预算是否改变现象，不用于在 64-token 主动作协议下把 think 截断制造成 I。

### 6.2 采样分布

N 主训练和概率流测量：temperature=1.0、top_p=1.0、top_k=0、min_p=0、repetition_penalty=1.0，无频率/存在惩罚，无 beam search，无 speculative/MTP，无隐藏格式修复。显式覆盖 generation_config 中的默认值。

理由：用于生成的 π 必须与 teacher-forcing 计算 score/likelihood 的 π 相同。若用 temperature=0.7 或 top-p 截断生成，却对未变换的模型 softmax 求 score，就不是本理论的 on-policy 更新。

官方建议采样参数可作为单独的使用场景诊断，但不用于主动力学检验。Greedy 只作行为补充，不能从 greedy 输出估计类别概率。

N 主 max_new_tokens=64；长度诊断为 48/64/128；L 保持原 free-48。若 calibration 显示 64 存在显著未完成数组，按预先定义的规则统一改到 128 后冻结：不是只给某个 arm 更长输出。

### 6.3 混合架构与 LoRA

主方案：冻结视觉编码器、视觉投影和 LM 基座，只训练语言层 MLP 的 `gate_proj/up_proj/down_proj` 的 LoRA。rank=8，alpha=16，dropout=0，bias=none；这些是新实验默认值。

必须用实际 `named_modules()` 选出语言层 MLP，不用宽泛后缀匹配误选视觉模块。对 9B 预期覆盖 32 层×3 个 MLP 矩阵；实际不符则中止并审计。输出可训练参数数、层名清单和权重 dtype。

MLP-only 是为了控制首轮成本并避免在混合架构上只适配 full-attention 子集，却误以为适配了全部层。它并不保证最优训练效果。后续可加一个 attention+MLP 适配范围敏感性对照，不能把 MLP-only 的失败直接推广为整个基础模型不具备能力。

### 6.4 cache 与似然一致性

关闭 dropout。训练 forward 不使用带梯度 cache；生成可使用 cache。P1 必测：完整 forward、分段 prefix continuation、逐 token continuation 的 log-prob/greedy 一致性，覆盖纯文本与图像输入。

Qwen3.5 含 linear-attention/卷积状态，不能只复制普通 attention KV 并称为完整缓存。[W2] I4 必须复制框架实际需要的所有状态、position/cache indices 及相关输入元数据。若无法可靠分支复制，则 I4 标为 unsupported；保留 I3 的完整 transcript 重算作为主自然接口。近似 cache 不能伪装成 exact cache。

数值一致性阈值先在 FP32 tiny fixture 上设 1e-5 级误差目标；真实 BF16 使用测得的重算误差基线，默认 top-1 token 一致率至少 99.9%、平均选中 token log-prob 差不超过 0.02 作为报警值。报警不自动表示理论失败，需定位 padding/position/cache/内核差异；阈值不是普适数学界。

---



## 7. 主训练算法的精确定义



### 7.1 新协议奖励：固定主权重，避免奖励尺度混淆

N 轨道使用主权重 α=2，validity 权重 λ=1 作固定辅助对照：


| Arm     | (X,S,W,I) reward | 解释       |
| ------- | ---------------- | -------- |
| A_BASE  | (2,2,0,0)        | 答案主验证器   |
| X_BASE  | (2,0,0,0)        | 精确世界主验证器 |
| A_VALID | (3,3,1,0)        | 答案+有效性   |
| X_VALID | (3,1,1,0)        | 精确世界+有效性 |


这与历史 A-BIN/X-BIN 的主权重 1 不完全相同，尤其 ε 固定时不能声称精确等价。历史四组只在 L 轨道按原值复现。N 的两个 VALID 组恰好与历史 LEX 向量相同，名称仍明确区分协议。

λ 敏感度固定 α=2，扫描 λ∈{0,0.1,0.25,0.5,1,2,4}。报告 λ/α。主轮不混合 αX 和 αA 同时非零，以免一次增加太多变量。

### 7.2 联合标准化

每个 prompt 单独采 K 个动作：

`reward_i = alpha_X*is_X + alpha_A*is_A + lambda*is_valid`

`adv_i = (reward_i - group_mean) / sqrt(mean((reward-group_mean)^2) + epsilon^2)`

新协议 epsilon=1e-4，方差分母 K，计算用 FP32 或 FP64；无奖励差异的组 advantage=0，不重采到有差异为止。不要对四类或奖励渠道分别归一化，除非运行明确标出的 GDPO 对照。

### 7.3 loss reduction 与理论对应

使用逐 token PPO surrogate、固定长度常数归一化：

L_reward = −(B K Lnorm)⁻¹ Σprompt,i Σgenerated_token min(ratio_token*A_i, clip(ratio_token,1−e,1+e)*A_i)。

`Lnorm=64`，在 N 主 max token 统一改成 128 时，Lnorm 仍固定 64 以免长度诊断同时改变步长；配置必须明确。EOS 是否进入 mask、PAD 如何屏蔽要固定并测试。

在起点 θ=θold、一次 on-policy update、clip 不活动时，其梯度是 sequence-score 组梯度的常数倍 1/Lnorm。由此能与理论对接。按每条序列实际长度再平均会引入样本相关因子，不能套用同一精确式。

这应在报告中称为“联合组 z-score + 固定 token 常数归一化的 GRPO 实现”，不能偷偷当作历史实现或所有框架的默认 GRPO。TRL 当前默认 reduction 可能不同，必须显式配置或写最小可审计实现。[W5]

### 7.4 优化与预算

拟议默认：AdamW，LR=1e-5，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；grad_clip=1.0；LoRA 初始化在同 seed 下完全一致；无熵奖励；主 βKL=0；1 次 rollout batch 只做 1 次 optimizer update。

每更新 B=4 个不同 prompt，K=8，每次更新 32 个 completion。GPU microbatch=1 或 2，通过梯度累积实现，不得把 microbatch 数误当作不同 optimizer steps。采样 K 个动作时 θ 不变。

主训练不丢弃全对/全错/零方差组；记录其占比。若要加重采样、SFT warm-start、动态难度或 entropy bonus，必须另开干预组，不能悄悄作为 baseline 的“工程修复”。

P1 可以做最多 24 步主奖励稳定性测试。只有数值爆炸或完全没有可测参数变化等实现性问题，才允许在 {5e-6,1e-5,2e-5} 中选择稳定 LR 并冻结；不得根据哪个 LR 更容易产生 validity–truth reversal 来选。

KL 对照 β=0.01、clip 活动对照、额外 PPO epochs 都放到 P6 后续，分别报告。主 KL 为零不表示不控制更新：仍监测实际 KL/步长，出现预注册异常阈值时中止并报告，而不是筛掉异常 batch 后续训。

### 7.5 checkpoint

Pilot：step 0/16/32/64。Confirm：0/32/64/128/192。保存 LoRA、优化器状态、scheduler、所有随机状态、已处理 sample keys 和数据 sampler state。不同 arm 共用 prompt 顺序与数据生成种子，但不能强迫后续 on-policy 输出相同；每个 arm 必须从自己的当前策略采样。

---



## 8. 分阶段实验



### P0：旧资料与数据审计（CPU，必做）

找出历史脚本、四组 reward、parser、数据、日志、checkpoint；生成差异表 `legacy_vs_new_protocol.md`。旧研究目录只读。建立数据 schema 和 hash manifest，运行完整 parser/constraint solver 单元测试。

验收：知道哪些是历史精确复现，哪些是新实验；所有主样本独立验证唯一解；split 没有 base-scene 泄漏。

### P1：模型兼容性与显存 smoke（GPU，必做）

用 18 个 calibration 场景的两接口测试；确认主模型、官方模板、非 thinking、LoRA 命中、生成/teacher-forcing 分布一致、图像未丢失、梯度非空且基座冻结。做 2–4 个真 optimizer update，并验证 save/resume 一致性。

针对混合缓存做 §6.4 测试；输出吞吐与预算估计。P1 未通过前禁止正式训练。

### P2：有限组数学基准（CPU，必做，可与 P1 并行）

在四状态 softmax 中精确枚举 N∼Multinomial(K,p)。K∈{2,4,8,16,32}；任选 (0.02,0.18,0.30,0.50)、(0.15,0.20,0.35,0.30)、(0.60,0.10,0.20,0.10) 三个事前支持状态，并补边界状态。

计算 HK、ΞK、有限步 Bη,K、qX 的真实有限步变化。K=32 只有 C(35,3)=6545 种 count vectors，可以枚举。mean-field 收敛趋势另扩 K 至 64/128，按运行时间决定枚举或独立 Monte Carlo，方法必须标清。

η∈{0.001,0.01,0.05}。比较一阶 ODE、二阶 Taylor 与精确递推。O(1/K) 是渐近上界/量级预测，不要求有限扫描的拟合斜率必为 −1；不得将偶然更快收敛视作理论错误。

测试 K=2、ε=0、固定严格 reward ordering 时 λ 间距不敏感；再单独测试 λ=0 奖励并列边界和 ε>0 平滑化。测试所有输出合法时加入 λ 为常数、更新应完全不变。

用同一非线性共享参数策略在 θ=±1 的构造验证 `THEORY_REVIEW` 的非闭合反例。检验简单合法性奖励下 qX 导数公式。

**P2 的 toy 数值只能标为数学/实现验证，不能当成新 Qwen 模型实验。**

### P3：冻结模型的跨规模诊断（GPU，第一批核心结果）

顺序：3B 锚点、9B 主模型、7B 桥接，不并行占用多个模型副本。

L：使用旧 manifest 中实际的全部评估场景，每场景 16 个自由 rollout，并额外保存一次 greedy。历史摘要记为 88 个配对语义单元、两种条件；只有审计确认每个单元恰有两个 scene 时才按 176 场景计数，不从摘要强制推断实际 manifest 的行数。

N：dev 的 144 场景×2 接口=288 prompt，每 prompt 16 rollout，另加一次 greedy。先固定上述样本，不因 X 少、S 少或 I 少而过滤。

必须报告：pX/pS/pW/pI、v、qX、qS、pA；P(X|A) 单独命名 `truth_given_answer`；copy_observation_rate；关系满足比例；每次生成实际 token 数；完整/部分截断；各六个家族×接口群体。

按每 prompt 估计：`P(no X in K)=(1-pX)^K`；对 X 二元奖励的 informative-group 概率 `1-pX^K-(1-pX)^K`。这是模型概率下的公式，代入有限样本估计需带区间；不把单个 plug-in 数字当精确值。

直接用实际组统计验证零方差比例、含 X/S/I 的比例。比较模型整体以及在相近初始支持/有效率区间中的结果；匹配支持属于辅助分析，不得丢弃原始全样本结果。

**冻结阶段不能回答奖励训练是否安全**，只能判断初始支持、现象可观测性和后续实验适用性。

#### P3 分支规则

- 若旧任务在 9B 上近乎全对且全合法：明确报告任务饱和，继续固定的新 ID/OOD 集，不降低图像质量制造错误。
- 若新集也接近饱和：有效性奖励的作用范围可能已消失，保留结果；转测语义一致性奖励的必要性，而不是继续无限加难度。
- 若某群体 X/S/I 零计数：报告计数及上界，不说概率为零。可按事先记录的扩样规则增至每 prompt 32/64，扩样只用于 control/dev，不按 confirm 的结果触发。
- 若答案与状态奖励在当前支持上没有足够分歧：结论是这个对照当前不可辨识，不是两个验证器理论等价。



### P4：同起点短训练（GPU，问题诊断，不先上 SSVC）

9B 跑 A_BASE/X_BASE/A_VALID/X_VALID 四组，seed=17，64 steps；每组 64×4×8=2048 训练 rollout，总 8192。

3B 在 N 轨道至少跑 X_BASE/X_VALID 两组相同短训练，增加 4096 rollout。7B 两组短训练是预算允许的后续桥接；若未运行，规模结论必须相应缩小。

使用预先确定的 dev panel：36 场景×2 接口=72 prompt，每 prompt 8 rollout，step 0/32/64；step 64 再评估完整 dev，每 prompt 8 rollout。panel 的样本可以作为完整 dev 的子样本复用，避免重复计数。

同时检验两个主对比：

1. X_BASE−A_BASE：更细语义验证器是否仍能改变世界恢复；
2. X_VALID−X_BASE：辅助有效性是否改善 v、却损害 pX/qX，或者这种损害已经减弱。

A_VALID/A_BASE 提供完整 2×2 因子对照及交互项。所有结果先报告变化量和区间，不只报告 p 值。

P4 不把单训练 seed 的结果称为稳健跨种子结论。没有出现旧反转也属于可解释输出。

### P5：确认训练（GPU，单独授权）

在协议、parser、模型 revision、主统计和长度冻结后，9B 四组各用 seeds={17,29,43} 运行 192 steps。总训练 rollout：4×3×192×4×8=73728；不包括评估。

每组保存所有确认 checkpoint；最终在 confirm 288 场景×2 接口、每 prompt 16 rollout 上评估。中间完整评估使用 dev；confirm 不用来改超参或选最好 checkpoint，主要 endpoint 固定 step 192。

可复用完全同协议的 seed17 pilot 继续训练，但必须标明 discovery 的数据仅来自 dev，且 confirm 未被查看。另报告每个 seed 的曲线，避免 bootstrap 掩盖三个训练重复本身的差异。

若资源不足，先完成四组×一个 seed 的诊断并注明；不要通过减少不利 arm 的种子数来制造结果平衡。

### P6：同批 rollout 的 λ 敏感度与真实更新干预（机制核心）



#### P6a：数值导数

固定一个 checkpoint 和 rollout bank。用公式、autograd、中心有限差分比较 ∂A/∂λ，使用 FP64 reward arithmetic，差分步长 {1e-2,1e-3,1e-4}，避开 ε=0 的 reward-tie 点。FP64 toy 上相对误差目标 <1e-5，接近零的量用绝对误差判据。

依次比较联合归一化梯度和“每奖励分别归一化后相加”的梯度；后者应明确标成另一个算法，不是联合梯度的精确分解。

#### P6b：actual optimizer fork

核心先取 9B X_BASE 的 step64 checkpoint，固定 8 个独立训练 batch；每 batch 4 prompt、K=8。对 λ∈{0,0.25,1,2} 分别从完全相同的模型+Adam+RNG 状态做一个更新；保存候选 θ⁺、实际 Δ(λ)、optimizer-state hash、grad clip 状态。

必须每个候选独立 restore，不能连续执行多个 λ。对同 λ 重复恢复两次，应得到数值一致的候选。候选没有 commit 前不能改变真实训练 run。

额外给一个固定预条件器/SGD 的审计模式，用于分离梯度核解释和 Adam 非线性；不能假设 Adam 恰好是该模式。

#### P6c：概率响应估计

在与训练 batch 独立的 control rollout bank 上，用 teacher-forcing 计算候选与起点的 sequence likelihood ratio，得到一阶 score 预测与 importance-reweighting 预测。记录 ESS、最大权重、权重方差和 overlap；这些结果名为 `predicted_delta`，不是 `observed_delta`。

无截断的普通 importance mean 可估计类概率；self-normalized ratio 有有限样本偏差。权重裁剪改变估计量，只能作为另报的稳健性诊断。若无 X 样本，重加权不会凭空获得关于新 X 轨迹的可靠证据。

再对事前由 batch ID 选定的两个 batch、λ={0,1,2} 共六个候选，用同一 72-prompt control panel、每 prompt 16 个全新 rollout 直接测量概率；不是选择效果最大的两个 batch。小步变化的 CI 可能很宽，允许结论为“当前精度不能验证”。

报告 qX 候选与主 baseline 的差、pX 差、v 差、预测残差、方向命中率、KL、参数步长。方向命中率应同时报告全体和高精度子集；不能把大量不确定符号硬算成命中/失败。

#### P6d：有限 K 与 KL/clip

优先在冻结 bank 上比较 K={2,4,8,16,32} 的实际 advantage 与梯度估计；注明重复抽组是条件于经验 bank 的重采样，不是新的独立模型生成。

少量实际 fork 使用 K={2,8,16}。同时给固定 B 和固定 BK=32 的口径：对应 B={16,4,2}；不能把额外 rollout 成本隐藏为 K 的优势。

βKL=0/0.01 和 clip-active 对照在有预算时增补。训练主流程一次 batch 一个更新起点 ratio=1，clip 通常不活动；若要研究 clip 必须明确额外 epoch/较大偏移来源。

### P7：共享参数的局部核与跨群体影响（机制扩展）

从 control 中事前选择 24 prompt，覆盖三家族×两接口；每 prompt 最初 16 个动作。计算 LoRA 子空间的类别 score 均值或未归一化类概率梯度；记录每个类别样本数，缺失类别标 NA。

目标不是存储 9B 全参数 Fisher。只在可训练 LoRA 子空间，用 streaming 梯度、Fisher-vector products 或低维 sketch。sketch 维数先 128，并用 16 对实际梯度 dot products 校验近似误差；误差过大时不宣称 kernel prediction 有效。

事前定义三种预测器：

- M0：仅 pooled 四类概率+λ/K/步长；
- M1：prompt/群体条件概率、样本数及归一化统计；
- M2：M1 + 经验梯度耦合特征。

在未用于拟合的场景/constraint family 上比较单步 Δp、Δq 的 MAE、带不确定度的方向预测和相对误差；不能只比 M0 和 M2，因为提升可能仅来自 M1 的异质性信息，而不是 kernel。

分家族训练 batch 作 fork，评估所有其他家族：得到“来源群体更新→目标群体指标变化”的干预矩阵。它是局部更新响应矩阵，不是类别间真实物理运输矩阵。至少与随机打乱来源标签和 norm-only 特征比较。

若 M2 没有稳定提升，报告该低秩近似不足；不要以拟合训练点的高 R² 宣称闭合了真实动力学。

### P8：解码与自然视觉接口（排除替代解释）



#### P8a：解码独立审计

固定同一 θ 和同一 prompt，在 control panel 比较 free、精确序列级 rejection、FSA。

rejection 从完整自由序列提议，只按 V 接受。每 prompt 目标接受数 16，上限 proposal 数 256；未达到目标时记不足，不能用 FSA 输出补齐。报告接受率和成本。按逐 prompt 比较条件纯度，并用适当原策略有效率权重重建 pooled 指标；详见 THEORY_REVIEW §7。

FSA 只实现相同整数数组语法，不访问 x* 或 cue 解。逐 token mask 后的 log-prob 是受限分布的 log-prob，不能与原始模型 log-prob 混同。FSA 允许的字符串集合须与主 parser 一致；语法合法不保证在有限 token 预算内必然完成。

必须验证小词表/短序列 toy 上的解析枚举：rejection=π(·|V)，FSA 一般可不同。实际结果若 rejection 条件纯度系统性偏离 free conditional purity，先审计实现和聚合权重，不把偏差直接归因于“模型不真实”。

MCMC/draft-conditioned decoder 不作为第一轮必需功能；引入时要报告 target、混合诊断、初值和有效样本数。

#### P8b：自然错误

从独立 natural_pool 对每个模型做一次固定协议的初始读图，记录所有结果；不只保存错误。将模型自己产生的错误观察与人工注入错误分开。天然错误不足时保留上限 2048，不无限筛到想要的数量。

建立两种分析：共同固定图像集合上的无条件错误率，以及每个模型自身自然错误条件下的恢复率。后者因选择条件不同，不能直接当成模型整体能力排名。

主自然恢复 cohort 先取“可解析且恰一个数值错误、cue 下唯一可辨识”的样本，报告其占全部初始读图场景的比例。多错误和不可解析输出作为另外的 cohort 保存，不偷换为人工单错误。对主 cohort 比较：

- I0：仅错误文字报告与 cue 的新调用，无图像；
- I3：保留原图和模型原始回答的同一对话完整 transcript，再请求修正；
- I4：同一 token transcript 的完整 cache continuation，仅在 P1 验证通过时启用；
- fresh-image：只给原图和任务重新回答，测重新读图可达到什么水平；
- no-image fresh：明确重建无图像上下文，作为不同信息条件，不称为“同一自然状态纯删除视觉”。

不要删除 image tokens 的部分 KV 后声称移除了所有视觉信息：后续文本/循环状态可能仍携带图像信息。I3/I4 成功可以来自仍可访问图像，不能直接叫 language-only compensation。

反事实 cue 单独构造并由 solver 验证唯一新世界；对图像与 cue 冲突的情况明确可信来源规则，不能把合理不服从错误 cue 记成失败。无解/多解和拒答单独分析。

#### P8c：真正能力与 OOD

用新模型与关键训练 checkpoint 测：未经 cue 修复的初始读图、固定 OOD、图像内容改变时的响应、文字观察不变但正确世界改变的配对。四类恢复任务与 initial perception 指标分开。

至少 X_BASE/X_VALID 要做；后续有 SSVC 也同步做。不能只因四整数世界更准确就宣称所有视觉/推理能力提高。

### P9：SSVC 离线可行性，随后才是算法比较

第一轮只实现 actual-update-family 的离线候选审计，不强制训练完整控制器。

候选 λ 使用固定网格 {0,0.1,0.25,0.5,1,2,4}；所有候选共享训练 rollout 起点；评估数据独立。保护六个家族×接口群体的 pX 和 qX；qS 为预注册可选约束。优先改善 v，同时限制辅助偏移的 Fisher/KL 二次型。

`lambda=0` 必须返回相同 baseline θ⁺，相对指标严格为 0，而不是因为独立采样噪声被判断不安全。

区分三种候选状态：SAFE / UNSAFE / UNKNOWN。UNKNOWN 不得当成 SAFE。按经验 proxy 选出的 λ 只能称 empirical choice。记录 λ=0 回退率、非零可行覆盖率、有效率收益、真实性风险、每步额外 forward/backward 与采样成本。

如果只用近似曲率/gradient error，从报告和代码命名中删除 `certified=true`。若使用独立 rollout 的有限候选审计，按群体、候选和时刻分配总 error budget；区间足够窄才可认证。

后续最小算法比较：X_BASE、在 dev 上调好的最佳固定 λ、GDPO、Empirical SSVC，各至少三个相同 seeds；已经跑过的 X_BASE 可以复用。固定 λ 至少允许与 SSVC 相同的候选集合，不能只拿一个故意差的 λ=1 作为对手。

GDPO 按官方算法实现，包括其聚合后的 batch 归一化约定，保存 revision。[W6] GD²PO 是相关的可选追加对照。[W7] 不把 PCGrad 写成 CPO，也不把 CPO 名字挂在未实现的安全投影上。本轮推迟 cone correction。

除等训练 rollout 预算，还要报告等总算力/等 wall-clock 预算。SSVC 的 control rollout 和候选 forward 都计成本；不能隐藏控制器开销。

若主数据 v 已饱和，可只加一个扩展奖励：`R_consistency = satisfied_relations / total_relations`，且不更改语法 V。这个 reward 在原四类内部可能变化，必须使用逐序列 reward/score 公式；不再套用四类常数奖励的精确 HK。

---



## 9. 统计分析、可观测性与安全口径



### 9.1 主要估计量

按固定 prompt 分布估计 pooled pX/pS/pW/pI；qX 定义为 pooled pX/pooled v，而不是未经说明的逐 prompt qX 平均。另报 macro mean per-prompt qX，缺少合法样本的 prompt 标 NA 并报告占比。

接口与家族固定权重。不同 decoder 的拒绝次数、不同模型的自然错误筛选不能静默改变这些权重。

统计单位以 base_scene 为主。每个场景的所有接口、cue、图像和 rollout 一起成 cluster；训练 seed 为外层重复。确认训练先报告每个 seed 的效果，再给分层/嵌套 bootstrap；bootstrap 10,000 次。三个训练 seed 仍然较少，不能把数万个 rollout 说成数万个独立训练重复。

主要两类对比为 §8 P4 中指定的比较，主 endpoint=step192；对六个受保护群体采用 simultaneous CI 或 Holm 校正。中间 checkpoint、纤维大小、稀有错误亚型是次要/探索性分析，标明。

### 9.2 效果判据（提议值，查看 confirm 前冻结）

“Validity–truth reversal”需要 X_VALID−X_BASE 的 Δv 正向，ΔqX 负向，联合区间支持对应方向。再检验 ΔpX 是否也负向；不能只看 purity 比值的波动。

“改善”要求最终未见数据上的 pX/OOD 或有效率–真值 trade-off 改善，而不是只有训练 reward 增加。

SSVC 的主要非劣界初拟为 qX 和 pX 各 −0.01（1 个百分点），同时需要有效率正收益。这是实践阈值，不是原理论给定常数。先在 calibration/control 上估计达到该精度所需样本，再判断本轮预算是否足够。

“未观察到显著损害”不等于“已证明问题解决”。只有 CI 已排除预定义的有意义损害范围，才能写“本测试范围内排除了超过该阈值的下降”。否则写证据不足。

### 9.3 零计数与稀有支持

按群体报告 X/S/I 计数、有效样本数和独立场景数。独立同分布 Bernoulli 条件下可用精确或 Wilson 区间；固定分层多 rollout 场景用 cluster bootstrap/适当分层模型，不能直接给所有样本套独立 Binomial。

rule-of-three 只能作为独立同分布近似的说明，不作为复杂分层数据的正式零概率上界。重采样扩池前后使用不同 data_version。

### 9.4 一个可实现但可能保守的独立审计方案

若需要真正的有限样本覆盖率，控制候选更新先固定，再在每个有限的受保护群体从指定经验 prompt 分布**有放回独立抽 prompt，并每次独立生成一个输出**。X/V 都是 [0,1] 有界变量。对两个均值建立同时 Hoeffding 区间，v 的下界>0 时可得 qX 的保守比值区间；再构造候选减 baseline 的上下界。

这种证书保护的是所审计 prompt 分布与更新，不是未知总体或所有未来时间。每候选/群体/时刻用联合 error budget；若多轮自适应训练反复复用同一审计集合，不能仍称独立新审计。λ=0 使用代数相等而非两个独立宽区间。

这可能需要很多样本才能判断 1pp 甚至更小单步变化。因此 P9 要报告 certificate coverage–compute 曲线，而不是预先保证 SSVC 非零权重总能被认证。

### 9.5 长期与短期不同

单步比 X_BASE 更新安全，并不推出最终训练轨迹优于独立 X_BASE。确认训练必须保留该独立对照。不能沿路径把局部非劣界简单相加后得到没有额外稳定性假设的全局保证。

---



## 10. 预注册的结果分支与下一步解释


| 观察                                    | 允许的解释                     | 不允许的解释                   |
| ------------------------------------- | ------------------------- | ------------------------ |
| 新模型旧集和新集都更好，辅助奖励无可测损害                 | 在给定模型/数据/精度下原风险减弱         | 所有大模型已经解决 reward hacking |
| v≈1，加入 validity 的梯度不变                 | validity 变成近常数，原控制通道无作用空间 | SSVC 必须提高 v 才算有效         |
| X 很少、无 X 组很多                          | 真值支持/信号稀疏，需要研究采样或验证器      | 多跑 λ 就肯定能学会不存在的轨迹        |
| S 很少，答案与状态奖励几乎一致                      | 当前对照缺乏分辨力                 | 答案奖励天然等价于世界奖励            |
| 平均改善，某约束家族下降                          | 有群体异质性，值得研究保护最差群体         | 平均值足够高所以语义安全             |
| I0 差，I3/fresh-image 好                 | 信息接口、重读图像或多模态上下文重要        | 已证明语言推理修复了自然视觉隐状态        |
| FSA 差，rejection 保持 conditional purity | 合法序列分布扭曲是候选解释             | 所有受限解码必然有害               |
| 实际 optimizer fork 与固定 M 预测不符          | Adam/clip/有限步或测量误差需要建模    | 联合归一化敏感度公式必然错误           |
| SSVC 总回退 λ=0                          | 无收益空间、测量不足或确实不安全需区分       | 控制器已证明成功提升能力             |
| ID 好、图结构 OOD 差                        | 局部策略或结构泛化仍不足              | 训练集高恢复率就是普适纠错规则          |


---



## 11. 必须生成的图表和汇总文件

1. `probability_trajectories`：四类概率及 v/qX/qS/pX 的曲线；保持独立图，明确 uncertainty 类型。
2. `validity_truth_plane`：v–qX 平面，另图 pX；按群体显示局部响应和置信椭圆/区间；不能把置信区间称为内部真实云。
3. `lambda_response`：同起点真实更新、score 预测、importance 预测的差异。
4. `group_support`：no-X、zero-variance、X/S/I 共现与 K、B 的关系。
5. `prediction_comparison`：M0/M1/M2 在 held-out 数据上的误差。
6. `decoder_audit`：逐 prompt 条件纯度、正确加权 pooled 纯度和 proposal 成本。
7. `interface_comparison`：人工/自然错误、I0/I3/I4/fresh-image 的独立结果。
8. `ssvc_feasibility`：可行 λ 范围、UNKNOWN/回退比例、非零覆盖与算力成本。

不要求“看起来像云”的动画。所有图必须可以追溯到表格。若绘制类别 Sankey，标题和 caption 必须显式写“指定采样耦合下的配对”，不得称为唯一真实流量。

最终最少输出：`run_manifest.json`、`data_manifest.json`、`model_audit.json`、`phase_status.json`、`metrics_by_scene.parquet`、`metrics_by_group.csv`、`training_steps.parquet`、`paired_effects.csv`、`flow_forks.parquet`、`failures.jsonl`、`results_report_zh.md` 和图表目录。训练未运行的表保留状态说明，不生成假的数字占位。

---



## 12. 验收测试（不得跳过）



### 数学与验证器

- 四类互斥完备；X⊂A⊂V。
- fixture 中 world-wrong/answer-correct 判 S，不判 X。
- 拒绝 bool/float/fence/额外文本，空白处理一致。
- 每个主场景独立 solver 恰有一个解，且等于真值。
- h/fiber 在小域穷举对照通过。
- 多项分布 count vector 权重和为 1；Σc UK,c=0；Σc Δpc=0。
- K=2 排序不变测试、reward ties 测试、all-valid constant-shift 测试通过。
- 有限步精确递推与 ODE 误差在减小 η 时按预期缩小。
- ∂A/∂λ 公式与 autograd/finite difference 一致。



### 模型与训练

- image 输入真的进入 forward，替换图像会改变对应输入 hash/token 表征。
- 没有真值进入 prompt、FSA、文件名可见文本。
- 主采样和 likelihood 分布一致；completion mask 不含 prompt/padding。
- LoRA 之外梯度/权重不变；模块命中清单符合规范。
- fork 的所有 λ 起点参数、Adam 和 RNG hash 相同；两次同 λ 可复现。
- λ=0 的辅助参数差严格为零。
- save/resume 与不中断运行在容许数值误差内一致。
- K 分批生成期间没有 optimizer update。
- 若用 GPU nondeterministic 内核，记录来源和实测差异，不伪称 bitwise determinism。



### 统计与报告

- confirm 未用于调参或选择 λ；记录每次访问。
- pooled 与 macro 指标名称不同。
- 缺失类别/低 ESS/没有足够有效样本标 UNKNOWN/NA。
- 训练 seed 与 rollout sample seed 分开，不把采样次数当独立训练次数。
- 真实模型结果、toy 数学结果、虚拟步预测和未完成阶段明确分开。
- 无显著差异、不支持假设、OOM、unsupported cache 都保留在报告中。

---



## 13. 第一批应交给研究者的结论页面

不写泛泛的“模型更强了”。用固定顺序回答：

**原来的 3B 问题是什么？** 按旧协议/新协议分别给真实计数。

**9B 的初始支持改变了什么？** X/S/W/I、合法率和零方差组如何改变。

**同一辅助奖励在 9B 上造成什么变化？** 给同起点短训练和 optimizer-fork 证据，明确误差范围。

**变化来自哪里？** 哪些证据支持归一化、有限组、群体耦合、解码或接口解释；哪些尚无法区分。

**SSVC 当前有无必要且有无可测空间？** 非零安全候选、精度和成本是多少；不能测清的地方是什么。

**下一阶段应只补哪一个关键缺口？** 根据事前结果分支给出最有判别力的一个实验，不自动扩大完整矩阵。

这份页面才是本轮问题分析的主要交付物。完整 SSVC 方法比较应在这些事实清楚后展开。