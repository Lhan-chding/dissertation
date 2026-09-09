# 面向 Codex：Qwen3.5-9B P3 后续诊断、局部更新与短训练实施规范

**版本：2026-09-09 / next-stage-v1**

## 0. 指令优先级与范围

本文件建立在用户已经完成的 P1/P3 上，不从头复建项目。先阅读现有仓库、配置、manifest 与日志；在现有模块中增量实现，不擅自更换算法框架、模型、parser、数据生成器或目录。

当前上传报告见 `sources/P1_P3_REPORT_2026-09-09.md`。报告只证明冻结推理完成，不能替代原始产物验收。先前方案中“拟议”的参数必须与服务器实际配置区分。

**本轮固定：**

| 项目 | 决定 |
|---|---|
| 主模型 | `Qwen/Qwen3.5-9B` |
| 模型 revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，必须在本地缓存核验 |
| 精度 | 保持 BF16；不引入量化 |
| Transformers | 报告为 5.14.1；导出实际 wheel/source 信息，不自动升级 |
| 模型范围 | 仅 9B；不启动 3B、7B、其他模型或替代检查点 |
| 主要训练协议 | N；原 N 主动作与 parser 不变 |
| L 的地位 | legacy-derived 冻结诊断及迁移 endpoint，不是 C3 精确复现；本轮不训练 L |
| 主短训练 arms | X_BASE 与 X_VALID，各 64 steps、seed=17 |
| SSVC | 先离线可测性/可控性，完整在线控制默认禁用 |
| 数据安全 | 旧产物只读；新产物进新运行目录，不覆盖 P1/P3 |

采用 **R0–R5** 作为这轮修订阶段名，避免把原 P4/P6 执行状态写错。原 P4 的短训练对应 R4；原 P6 的机制实验前移并分为 R0 组统计与 R3 更新响应。R5 是将来的离线 SSVC，可实现但不默认启动。

**不得做的事：**不把一条 greedy 输出当概率；不把虚拟重加权当新采样；不把旧 P3 输出用于正式训练；不悄悄重采零奖励组；不使用 SFT/拒绝采样/FSA/额外思考作为主 baseline 修复；不因没看到预期伤害而换模型、提高 λ 或选择最差样本。

本包的 `reference/` 只有可运行的 CPU 数学参考，不含服务器模型 runner。下述命令属于需要接入现有仓库的接口契约，不是已实现或已运行的 GPU 命令。

## 1. 阶段、默认开关与先后关系

| 阶段 | 核心问题 | 默认执行 | GPU 权重更新 |
|---|---|---|---|
| R0 | 原结果、数据、估计量与奖励组是否可信？ | 是，先做 | 否 |
| R1 | 采样/似然一致吗？训练路径真能更新吗？吞吐瓶颈在哪？ | 是 | 仅隔离 smoke，随后丢弃 |
| R2 | 纯符号失败主要与什么可观测因素相关？ | 是 | 否 |
| R3-cold | 奖励组与 λ 如何改变实际更新？冷启动优化器是否有特殊性？ | 是，R0/R1 通过后 | 隔离 fork，不 commit |
| R4 | X_VALID 相对 X_BASE 的短训练效果与迁移如何？ | `--allow-training` 后 | 是 |
| R3-warm | 成熟 Adam 状态下上述响应是否保持？ | R4 后，复用 X_BASE step64 | 隔离 fork，不 commit |
| R5 | λ 候选能否被独立数据以足够精度筛选？ | 默认否 | 只生成候选，不启动在线 SSVC |

可在 R0/R1 通过后并行准备 R2 与 R3 的代码；不要同时加载多个 GPU 模型副本。缓存提速失败不等于研究失败：保留经过数值验证的慢路径，更新预算并降低并发，而非静默改变分布。

### 1.1 输出状态

每个阶段必须有 `status.json`、`manifest.json`、`report_zh.md`。

`status` 使用 `NOT_STARTED/RUNNING/PASS/FAIL/BLOCKED/INCONCLUSIVE`；`execution_kind` 使用 `CPU_AUDIT/CPU_MATH/REAL_CUDA_INFERENCE/REAL_CUDA_TRAINING_SMOKE/REAL_CUDA_FORK/REAL_CUDA_TRAINING`。不能让 dry-run、toy 或假模型输出 `REAL_CUDA_*`。

P1 旧状态保留；新增 `inference_pass` 和 `training_smoke_pass` 两个字段，不追改旧作业结果。

### 1.2 服务器只读入口

```text
/projects/varunssd/louis-ssvc/runs/P3/qwen35_9b/N/
/projects/varunssd/louis-ssvc/runs/P3/qwen35_9b/L/
/projects/varunssd/louis-ssvc/runs/slurm-145423-attempt-0/runs/P1/
/projects/varunssd/louis-ssvc/cache/huggingface/
```

这些位置来自报告，存在性由 Codex 检查。逐条 rollout 不一定就在报告文件旁；从 manifest 跟随精确路径，若缺失则记 `MISSING_ARTIFACT`，不要从表格重建并冒充原始输出。

## 2. R0：不新增模型生成的完整审计

### 2.1 文件与版本绑定

定位并哈希：数据 manifest、每题真实世界和观察、cue、图像、最终序列化 prompt、tokenizer/chat template/processor、generation config、模型 revision、已安装源码、原始 completion/token ids/log-prob、分组指标生成脚本。

为 N/L 分别建立 `baseline_lock.json`：包括模型、权重、输入、采样、长度、parser、verifier、executor、分层、统计权重。任何未知必填项填 null 并列为阻塞项，不得用旧方案的拟议值补齐。

N 原计划的预期采样：T=1、top_p=1、top_k=0、min_p=0、repetition_penalty=1、非 thinking、max_new_tokens=64。它们不是当前汇总已经证明的已执行值。L 只读继承实际旧配置，不能假定等于 N，也不能只因历史写 free-48 就填 48。

若 N 已执行参数与原计划不同：保留 P3-N 为实际策略分布；明确差异，针对训练所用新 lock 建立必要的 frozen bridge。不要把旧 rollout 当成新策略的 on-policy 数据。

### 2.2 数量、重复和事件审计

检查 sampled N=4,608、L=2,816；greedy N=288、L=176。每题 sampled 恰 16 条，不存在条件性丢样、报错样本被排除、重跑重复或 greedy 混入。每条 `sample_key` 唯一。

独立重算 X/S/W/I：

```python
if strict_parse(raw_action) is invalid:
    category = 'I'
elif parsed_world == truth_world:
    category = 'X'
elif execute(parsed_world, operation) == execute(truth_world, operation):
    category = 'S'
else:
    category = 'W'
```

N 的 V 只检查整个动作的 JSON 语法、恰四个 Python 整数、域 0..99；布尔值不算整数，浮点数、字符串数字、额外文本、markdown fence 均非法。不得把约束满足、只改一处或答案正确偷放进 V。L 用其实际锁定 parser。

仅移除明确属于聊天模板/终止的结构 token。保存原 token ids；不能用 `skip_special_tokens=True` 删除模型生成的 think 等内容后，把原本不符合格式的输出变合法。对原 P3 的处理保持可追溯，重标注只是新审计列，不覆盖旧数据。

长度达到上限与非法分开：完整合法数组即使 `stop_reason=length` 仍按旧锁定 parser；未闭合数组才记未完成。无 EOS 的截断序列不能在似然中补一个不存在的 EOS。

从原始整数计数重算所有比例；报告舍入允许约 5.1e-5 误差。检查 pX+pS+pW+pI=1、v=1−pI、pA=pX+pS、qX=pX/v、truth_given_answer=pX/pA。

本包 `SUMMARY_ARITHMETIC.json` 的反推题数仅为线索：N 六组各48；L duplicate两组各16，其余各36。若原数据不同，解释原因，原始数据优先。

### 2.3 数据可辨识性与图像审计

以独立实现枚举每题至多 4×99 个一处替换候选：

```text
C(o,cue) = {x in {0,...,99}^4 : Hamming(x,o)=1 and all_constraints(x,cue)}
```

N 每个正式修复样本必须 C={truth}。逐项验证真值、观察恰一错、错误索引、所有可靠 cue、索引映射与图表 a/b/c/d 顺序。

独立求解器不得 import 生成器的判定函数；允许共享数据 schema，但要用独立约束计算，且在小域暴力枚举交叉验证。检查 `parsed in C` 当且仅当 `category==X`（对合法且已验证唯一解的主修复题）。

查是否错误地把像素图、truth JSON、正确最终答案、solver 解或 changed_index 拼进了 SYMBOLIC prompt。changed_index 只允许在 R2_ORACLE_INDEX 诊断中出现。公开 cue 对真值的约束不是泄漏；额外未声明的真值标签才是问题。

按哈希预选36张图做人工可视检查，覆盖家族×图形×operation；不得只看成功例。核对原图、processor 输出尺寸与实际 `image_grid_thw`/视觉 token 数。请求上限不是执行值。

N 两接口与所有变体必须按 base_scene_id 配对。L 查 collision/separating 是否共享同一语义 base unit，究竟改变了哪些因素。找不到匹配键，不得伪造 paired CI。

### 2.4 逐题附加指标

每个 prompt 输出以下字段，采样与 greedy 分开：

```text
n_X,n_S,n_W,n_I,n_total,n_valid
p_X,p_S,p_W,p_I,p_A,v,qX_pool_component
copy_observation_count
hamming_to_observed, hamming_to_truth
corrected_original_error_count
new_errors_on_originally_correct_coordinates
all_constraints_satisfied_count, constraint_satisfaction_fraction
feasible_repair_count
operation, observed_answer_collision, changed_index, is_extremal_changed_index
completion_length, stop_reason, malformed_type
```

`malformed_type` 至少拆为空输出、未闭合/截断、额外文字或思考、错长度、非整数、越界、其他。关系满足但修改了多处不是 X；“不再复制原观察”也不是精确恢复。

为缺字段单独报告覆盖率。无 raw completion 时不能声称完成 parser 审计；无 token ids 时不能声称完成概率/模板审计。

### 2.5 估计量与不确定性

主 p 指标为固定 prompt 权重 wi 下的每题采样比例平均。每题输出数可能扩样时，必须先算每题比例，再按既定 wi 聚合，不能让扩样题权重变大。

```text
pX_g = sum_i w_i * nX_i/n_i
v_g  = sum_i w_i * nV_i/n_i
qX_pool_g = pX_g / v_g
qS_pool_g = pS_g / v_g
```

主 N 群体为三家族×两接口，权重各1/6；原 N manifest 若不平衡，报告原始组成与标准化组成，两者不要混写。L 首先按实际固定题分布报告，再给家族等权重敏感性。

场景外推的主 CI：按 base_scene 分层 cluster bootstrap，5,000次，固定seed=20260909；两接口、collision/separating、各 arm/checkpoint 的同一 base unit 一起抽。每次重算分子分母及其比值，不平均已算好的 q。缺乏共同支持的某组输出 NA。

这一主 bootstrap 保留每个 cluster 的原始完整输出集合，不再叠加一次逐输出 bootstrap。固定题库的 Monte Carlo 区间可另外做每题内重采样并明确 `conditional_on_panel`；不得混用两种 estimand，也不把它们堆成过窄或错误重复计数的区间。

训练只有一个 seed 时，所有 CI 仅针对该训练实现的评估不确定性，不能宣称覆盖训练随机性。未来多 seed 必须先给逐 seed 配对效果，不能把大量 rollout 当成很多训练重复。

### 2.6 用已有 bank 测奖励组构成

对每题16条 IID 输出，K∈{2,4,8,16}，用无放回组合平均计算：

- no_X：无正确世界；
- X_informative：同时含 X 与非 X；
- validity_mixed：同时含合法和非法；
- aux_only_activation：没有 X，但同时含 S/W 与 I；
- three_level_X_B_I：同时含 X、S/W、I；
- 每种奖励的 zero_variance；A 与 X 两套归一化优势的差异。

参考 `reference/finite_group_reference.py`。重叠子集不算独立重复，CI 仍按场景；K>16 需要新真实生成或只标经验分布模拟。

对标准化优势比较，按多元超几何概率枚举类别计数，计算期望 `mean((A_lambda−A_0)^2)`、`mean((dA/dlambda)^2)` 与 A/X 差异；这是**固定经验 bank 的条件统计**，不是参数更新，更不是新训练结果。

CPU λ 网格：0、1e-6、1e-5、1e-4、1e-3、1e-2、0.1、0.25、1、2。用来覆盖 ε 附近的激活边界，不只扫描大权重。

epsilon 取实际新训练 lock 的值；拟议为1e-4，方差分母 K、分母 sqrt(var+epsilon²)。不要误用 std+epsilon 或 correction=1。

### 2.7 R0 验收

`PASS` 要求：原始输出可追溯、计数一致、分类与真值独立验证通过、采样/解析配置绑定、无 split 泄漏。

缺少区间但原数据齐全：完成重算后继续。缺少关键原始文件或发现标签/数据错误：`BLOCKED`；保留原报告，输出受影响列表和修复计划。不得为保持 PASS 删除失败样本。

## 3. R1：正确性、吞吐与训练 smoke

### 3.1 先定位 forward 计数

旧报告 N 的151,776/75,888与 L 的70,338/35,169都为2。这只能说明计数比为2。

给顶层模型 forward 加带原因的计数：prefill、decode、teacher_forcing、reward/scoring、backward_recompute、audit。另报真实处理的 token 数、vision encoder 调用、padding token、设备同步时间。避免顶层与内部层 hook 重复计数。

只在确认冗余后优化：生成时缓存已使用的选中 token log-prob；一次 teacher forcing 得到整条输出的 old log-prob；CPU 先缓存渲染/processor；冻结视觉模块时可在证明等价后缓存视觉特征；普通生成使用经过校验的完整缓存。

不默认引入 vLLM、量化、MTP、speculative、torch.compile 或新内核；它们可以成为后续独立加速分支，但不能未校验即进入主实验。

### 3.2 固定数值测试

从 calibration 按 hash 选12个 base scenes、两接口共24 prompts，覆盖六群体。比较：原 prefix recompute、teacher forcing、普通 cache 生成、batch size1/2/4的输入处理与 likelihood。

保存固定 prefix 上所有选中 token 的 log-prob，而不只比较最终类别。覆盖短/长数组、EOS、截断、左右 padding、两个不同图像的顺序、同图多样本、不同生成长度及重复调用。

先确认原参考路径自身稳定。BF16 起始工程报警阈值：平均选中 token log-prob 绝对差0.02、p99差0.1、top1一致率99.9%；这些不是分布等价证明。完整序列 log-ratio 也要单独报告，不能因 token 平均小就忽略累计误差。超阈值不许静默接受。

每题先4 sampled+1 greedy，共120条参考输出；其它路径尽量重放固定序列而非重复大量随机生成。若替换生产生成路径，额外用36场景×两接口×8 sampled=576条 bridge，与参考路径按场景比较，并结合固定序列数值证据决定是否建立新 baseline lock。没有显著差异不能单独作为等价证明。

Qwen3.5 的普通 cache 也包含混合层状态；复制、重复 batch 或不同 prompt 之间不能共用可变状态。若只能安全逐样本调用，先保留逐样本。普通生成 cache 验证通过不等于 I4 自然状态 fork 已完成；本轮不做 I4。

### 3.3 采样与训练 likelihood 必须对应同一分布

新 N 主流：T=1、top_p=1、top_k=0、min_p=0、repetition_penalty=1；无额外 logits mask、无最小输出长度强制、无隐式坏词过滤或不同 EOS。显式覆盖模型 generation_config 默认值。

teacher forcing 只对真实生成 token（包括真实 EOS）计算 score，不计 prompt、padding 或未生成的尾部。对完整与截断输出使用相同有限 horizon 定义。old log-prob 必须在更新前缓存，更新后不能重算并冒充 old。

随机流以 model/adapter/prompt/sample_index/stage 为键，重启幂等；不同 sample_index 不得重置为同一 seed。相同文本输出可以是合理概率集中，不能仅按文本相同删重复。只按 sample_key 去掉作业重跑重复。

### 3.4 LoRA 训练 smoke 必做

保留原方案的 MLP-only LoRA：语言层 gate_proj/up_proj/down_proj，rank8、alpha16、dropout0、bias none；视觉模块和基座冻结。实际 named_modules 清单写入产物，不用宽泛正则误匹配视觉模块。9B预计32×3个目标矩阵；不符先审计。

使用隔离 scratch adapter 和优化器，完成2–4个真实更新、保存/恢复一致性、finite gradient 与参数变化检查、一次完整梯度累积。smoke 数据只能来自 calibration，smoke adapter 不得作为主训练起点。

LoRA 常见零初始化使部分因子的首步梯度为零；不要要求每个 A/B 因子首步都非零。要求理论上应更新的可训练参数确有有限非零更新，基座/视觉参数不变，并验证多个步骤后链路工作。

可用明确标记的 teacher-forced calibration 样例测试梯度数值，但这不是 RL 成果；formal training 不能注入正确 completion。完成后重建原始 LoRA 初始化、全新 Adam 与 RNG 状态。

记录 `max_memory_allocated/reserved` 分阶段峰值、设备实际 memory.total、训练步秒数、forward/backward/update调用次数。加载峰值不代替训练峰值。

### 3.5 实现不变量

- 一组全部合法，λ0与λ1优势、梯度和同起点实际更新应一致。
- 固定 rollout/状态重复一次同 λ，候选更新一致至声明精度。
- baseline λ0 的辅助增量严格为0，不能靠两组独立采样判断。
- microbatch=1与等效大 batch 的梯度在数值容差内一致。
- 奖励计算完全 detach，不通过 reward/advantage 反向传播。
- 全零 reward gradient 时，不随意跳过 Adam 的 step、动量衰减或 scheduler；`.grad=None` 与显式零梯度可能不同，必须锁定统一语义。
- 无 NaN/Inf、无 silent OOM fallback、无 adapter/图像缓存跨 candidate 污染。

## 4. R2：有限规模的失败来源诊断

### 4.1 样本与原则

从 calibration 选72个独立 base scenes，三家族各24，家族内图形×operation六格各4。按稳定 hash 选择，不能按 P3 成败筛题。每题的所有诊断变体共用 base_scene_id，不进入 train/control/dev/confirm。

若现有 calibration 不足或已泄漏，独立生成补足，先过 solver，再固定 manifest。不能通过查看模型成绩过滤样本。

### 4.2 六个预注册条件

| condition | 输入变化 | 可回答什么 | 不能宣称什么 |
|---|---|---|---|
| SYM_ORIGINAL | 原 N SYMBOLIC_FRESH | 诊断 panel 的同协议基准 | 不能把新 panel 数值替代完整 P3 |
| SYM_CLEAR | 同一 o、同一信息量的 cue，关系逐行写清并强调恰替换一处 | 对表达/程序化提示是否敏感 | 不等于底层能力变强 |
| SYM_NO_OPERATION | 仅移除下游运算那一句，h仍在 evaluator 保留 | 无关/竞争性答案目标提示是否影响恢复 | 不是模型在全部操作上天然等价 |
| IMAGE_CUE | 原 N IMAGE_CUE_FRESH | 图像可用条件下的表现 | 不自动归为推理补偿 |
| IMAGE_ONLY | 保留图像、变量映射、操作说明，删除错误观察与 cue；要求直接读出世界 | 直接读图是否已足够 | 不与修复任务混称单一干预因果效应 |
| ORACLE_INDEX | SYM_ORIGINAL + 错误索引，但不告诉正确数值 | 已知定位后恢复是否容易 | 不进入训练或正式主测试，不是同信息任务 |

`SYM_CLEAR` 只允许一次固定模板：cross_series用逐条等式；trend用 `b-a=c-b` 与 `c-b=d-c`；明确其他三项正确。不得写入 solver 推出的值或额外示例。保存原始和修订模板 diff/hash。

每条件4 sampled+1 greedy，统一主采样配置、64 token；共72×6×5=2,160条输出。greedy与sampled单独统计。

R2 以 pX、copy、正确修复错误坐标、破坏原正确坐标、完整约束满足为主，不仅比较 qX。把干预是否改变信息量、格式或 token预算写进图表标题。

### 4.3 小型长度/thinking诊断

按 hash 从上述 panel 选12个场景，cross_series/trend各6、每家族覆盖图形×operation六格；只用纯符号接口。每题2 sampled+1 greedy，新增两个条件：

1. 原非 thinking 模式、max_new_tokens=256；
2. 正式开启 thinking、max_new_tokens=1024，使用该模型正确的模板与输出分段。

共12×2×3=72条新输出。原64-token结果复用，不重复计算。此分支用于区分截断/解题预算，不用于改主 baseline。

thinking 的 reasoning 与 final 原文和 token 数分开保存；只有通过官方分隔符明确识别的最终答案才计算 `diagnostic_final_X`。若缺少可可靠识别的 final，标 `unresolved_final`，不从思考中抓取真值数组。不要将这个诊断指标并入原来的严格动作 pX/pI。

1024是总新增token上限，不承诺会完成推理；截断需单独报告。主非 thinking 的失败并不等于 thinking 条件下没有能力。

### 4.4 R2 决策

- 发现无解/多解/索引错误/图像被漏传：阻塞受影响的正式训练，版本化修复。
- 仅 SYM_CLEAR 提高恢复：保留原 N 协议；记录策略/表达敏感性。若研究者将来选择 N-v2，必须给它独立 baseline 与 split，不重写本轮结果。
- IMAGE_ONLY 已很强：后续避免把 IMAGE_CUE 高分说成关系推理修复；继续记录 symbolic vs image 迁移。
- ORACLE_INDEX 很强但原版弱：错误定位是可检验的瓶颈候选；不是定位能力的唯一机制证明。
- 长度改善只减少 I、不改善合法错误：先去除“纯截断解释”，继续奖励组与训练分析。
- 所有条件仍弱且 solver正确：这是当前策略的可靠失败证据，不自动启用 SFT、答案注入或持续加难。

R2 不因结果出乎预期而失败；只因数据/执行错误而 FAIL。

## 5. R3：从“概率快照”走到“真实奖励响应”

### 5.1 核心估计对象

固定 checkpoint θ、Adam 状态 s、同一 rollout D：

```text
(theta_plus(lambda), state_plus(lambda)) = Update(theta,s,D,lambda)
Delta_f(lambda) = f(theta_plus(lambda)) - f(theta_plus(0))
```

f至少包括六群体的pX、v、qX_pool、qS_pool。先比较优势变化，再比较梯度，再比较实际参数更新，再比较概率响应；四层不能互相替代。

冷启动 fork 只用于实现及起始状态诊断。机制主解释必须补 R3-warm，因为 Adam 首步与成熟动量状态下的响应可能不同。

### 5.2 rollout bank

从 train split 事前按hash选48 prompts，六家族×接口群体各8；不与 control/dev/confirm重叠。不按类别筛选生成结果。

组成12个 B=4 batch，每prompt K=8，共384条真实 on-policy输出。组内 θ 不变；不同 λ 共用同一 bank。

使用新训练 lock 的原分布，不从P3 dev bank做正式更新，不“采到X才结束”。未观测到某类的经验梯度项可为零，但真实类别概率/条件均值标不可估计。

主梯度与 Adam候选 λ={0,0.01,0.25,1,2}；CPU优势层使用R0更密的ε附近网格。另在同一bank上算A_BASE/A_VALID的优势和梯度作为verifier分辨率诊断，不为它们启动长期训练。

### 5.3 精确复用固定 bank 的梯度

在on-policy起点ratio=1、clip未活动、固定token常数归一化、无额外正则时，同一prompt内同一类别有相同优势。因此可缓存：

```text
G[prompt,c] = sum_{sample in c} grad_theta log pi_theta(sequence | prompt)
g(lambda) = 1/(B*K*Lnorm) * sum_{prompt,c} A[prompt,c](lambda) * G[prompt,c]
```

这不是把奖励渠道独立梯度相加，而是对**联合优势**做精确线性汇总。每个λ仍要重新算完整组均值、组方差和联合优势。

只在LoRA子空间保存/流式累加类别score sum，禁止保存9B全参数Fisher。不得对每条不同长度的输出额外做1/length归一化。missing类别不除以零估计条件score均值。

随机指定两个bank、λ0和1，将此复用梯度与直接完整loss反向传播对比；误差超过FP32/BF16已注册容差则禁用复用，保留直接路径。写明这是起点单步等价，不能直接用于多epoch/clip-active或改变θ后的梯度。

### 5.4 实际优化器 fork

每候选独立恢复：模型/LoRA、Adam一二阶状态、step计数、scheduler、所有RNG、grad scaler（如有）、数据sampler。先清除残留gradient和可变cache，再加载同一个初始状态。

将真实梯度按优化器符号约定写入，运行同样的global grad clipping、AdamW.step和scheduler。不能仅把“理想梯度×LR”当作真实参数更新，也不能使用候选之间共享的已更新Adam状态。

每bank5个候选，12bank共60个隔离更新，均不commit。保存adapter候选、optimizer-state hash、pre/postclip norm、参数差、同起点基线增量、候选间距离、实际backward/update次数。

必须验证λ0重复恢复完全一致；全部组合法的bank各λ实际更新一致；两个奖励层的优势近似不变时不预设Adam也完全不变，实测并报告数值/ε影响。

提供一份固定预条件器/SGD参考响应，仅作为数学对照，不能替代Adam结果。冷启动LoRA的零梯度结构单独记录。

### 5.5 组构成与响应的连接

对每bank记录：noX比例、validity_mixed比例、aux_only_activation比例、three_level比例、优势差范数、导数能量、梯度差范数、Adam候选差范数。

对ε=0、Var(r)>0的toy测试：

```text
mean((dA/dlambda)^2) = 4*x*b*i / Var(r)^2
```

真实训练ε=1e-4时用完整导数公式，不用极限式代替。λ=0、无X且有B/I是并列边界，不能用右侧区间零导数说“λ从0变正没有影响”。

先检验这些量能否解释“哪些bank更新有变化”，而非直接拟合一个四类总概率的巨大流量矩阵。全部为局部响应，不宣称恢复模型内部真实类别运输。

### 5.6 control bank 与低方差配对重加权

从control split按hash选24个base scenes，三家族各8，两个接口共48 prompts；按metadata尽量平衡operation/chart，不按模型表现选。该panel从未用于产生训练梯度。

在当前θ下每prompt16条新采样，共768条作为control proposal bank；旧bank若同checkpoint、同lock且属于control，可按sample key复用。

为控制预算，只对事前指定的bank index0和6的5个候选（合计10个候选）计算完整control bank上的likelihood ratio。其他bank只交付优势、梯度与实际参数更新，不虚构其概率响应：

```text
w_lambda(y) = pi_theta_plus(lambda)(y|prompt) / pi_theta(y|prompt)
Delta_pX_IS = average_prompt[ mean_sample( (w_lambda - w_0)*1_X ) ]
```

直接用同一proposal上的权重差估计candidate与baseline之差，避免把两个独立高方差均值相减。qX候选由分别加权的pX/v求比值；每次cluster bootstrap重算全部比值及差。

主值为未裁剪普通importance estimator；self-normalized和裁剪权重只能另报，注明偏差。EOS、截断与action定义一致，proposal必须对目标支持覆盖；原top-p截断bank不能当成完整softmax的支持充分proposal。

记录ESS、ESS/n、最大归一化权重、mean(w)、权重分位数、每类ESS/计数与序列log-ratio。工程警戒线：ESS/n<0.5或最大归一化权重>0.05时，标`OVERLAP_WARNING`并停止用它做安全判断。通过这些诊断也不构成有限样本覆盖率保证。

无X样本的题，其重加权X贡献为经验零，但要标`NO_OBSERVED_X_SUPPORT`；不能据此认证该题安全。

可另外teacher-force canonical truth/observed的log-prob margin，但它只是两个固定字符串的相对偏好，不是所有合法写法合并后的pX。

### 5.7 真实重新采样验证

事前指定bank index0和6；每bank只验证λ0与1，四个候选各48prompt×16输出，总3,072条新输出。默认仅在R3-warm执行这批重新采样；R3-cold只做control重加权及实现审计，`direct_resample_cold=false`。候选选择不能依据哪个bank效应最大。

对proposal预测与直接采样分别命名 `predicted_delta`、`observed_delta`。四个候选的采样可采用相同random-number coupling降低方差，但必须保存coupling方法；“相同seed”并不证明两条输出有自然身份或真实流向。

small panel可能无法分辨很小的单步变化。报告CI半宽、与预期效应的比例；允许`INCONCLUSIVE`，不通过增加LR只为了把结果做显著。需要扩样时优先增加独立base scenes，扩样规则在查看新结果前锁定，不按最差群体结果逐个追样。

所有fork比较都有三列：相对θ的绝对变化、相对θ⁺(0)的辅助变化、预测残差。只有第二列对应SSVC相对安全对象。

### 5.8 冷/热状态安排

R3-cold在正式训练前执行，主要证实实现与初始组构成。R4完成后，在X_BASE step64的完整Adam状态上，重新从其当前策略生成同一prompt ID bank，重复R3的组/梯度/fork/control流程，并首次执行默认的3,072条直接重新采样验证，输出 `R3_warm/`。不能直接复用cold策略的rollout称为warm on-policy。

单步fork的训练prompt不改；control ID保持一致，但proposal rollout必须对应warm θ。不同checkpoint下的相同ID便于比较，不等于样本序列相同。

## 6. R4：两个主arm的可审计短训练

### 6.1 奖励与算法

| arm | r(X,S,W,I) |
|---|---|
| X_BASE | (2,0,0,0) |
| X_VALID | (3,1,1,0) |

即r=2·1X+λ·1V，λ分别0、1。α固定，避免主奖励尺度随arm改变。

每个prompt内：

```text
A_i = (r_i - mean(r)) / sqrt(mean((r-mean(r))^2) + (1e-4)^2)
```

方差分母K，reward arithmetic FP32/FP64；全相同奖励A=0。主流程不丢弃全对/全错/零方差组。

逐token PPO surrogate，固定常数Lnorm=64：

```text
L = -(1/(B*K*64)) * sum_generated_tokens min(ratio*A, clip(ratio,0.8,1.2)*A)
```

A按序列广播到其generated token，真实EOS计入、PAD和prompt屏蔽。`old_logprob`在更新前冻结。每个rollout batch只做一次optimizer update，因此起点ratio≈1、clip通常不活动；不能把这个实验写成研究clip-active机制。

βKL=0，无entropy bonus、无额外SFT、无关系一致性reward；仍记录诊断KL与更新幅度。未来正则/多epoch另开arm。

### 6.2 超参数与初始化

```text
train_seed=17
steps=64
B=4 distinct prompts per optimizer update
K=8 independent samples per prompt
microbatch=1（通过gradient accumulation保持等效B*K）
AdamW: lr=1e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0
max_grad_norm=1.0
LoRA: rank8, alpha16, dropout0, language MLP only
sampling: T1/top_p1/top_k0/min_p0/repetition_penalty1
max_new_tokens=64, enable_thinking=False
```

两个arm从相同全新adapter与Adam状态开始，不能从R1 smoke或R3某候选继续。相同训练seed包括LoRA初始化、prompt顺序与数据sampler；每个arm后续必须由自己的当前策略on-policy采样，不能强行共用另一arm已生成的轨迹。

训练池沿用通过审计的N train manifest（原方案576场景、每场景固定一个接口）。不将dev/P3、control或L评估题混入。若原train manifest不存在，依原生成规范先建立再锁定。

按六群体轮转或分层随机化，尽量保持均衡；每6步的24个prompt slots中各群体4个，64步末余数确定性分配，两个arm完全一致。不按奖励决定抽题、不按模型错误率加权。

正式LR不根据哪个值更容易出现反转来选。只有R1数值失败时，在旧拟议范围5e-6/1e-5/2e-5做隔离技术修复，并在看arm效果前锁定选择理由。

### 6.3 检查点与异常处理

保存step0/16/32/64：adapter、完整optimizer/scheduler/RNG/sampler状态、模型与数据lock、已完成sample keys。

任一NaN/Inf、基座被更新、输入/输出hash不匹配立即FAIL并保留证据。监测已固定control probe上的policy KL；工程停止线采用mean token KL>0.1或sequence log-ratio p99绝对值>2时暂停诊断，阈值不是安全定理。数值估计噪声与方向说明写明；不删除导致超线的batch后悄悄继续。

KL口径：固定step0 control proposal的prefix分布，记d=logπ_candidate(token|prefix)−logπ_step0(token|prefix)，对step0生成的token计算 `exp(d)−d−1`，先逐题平均再按固定权重汇总。这是step0状态分布下的条件token KL诊断，既不是完整轨迹KL，也不是当前策略状态占用的KL；分母/方向和序列log-ratio同时保存。计算溢出视为异常，不能裁剪后掩盖。

正式诊断β=0不保证小步安全。出现明显政策跳变时先检查实际optimizer、loss缩放与old logprob；两个arm不得采用不同恢复规则。

### 6.4 评估预算

| endpoint | panel | sampled / prompt | 新输出数（两arm合计） |
|---|---|---:|---:|
| step0 | dev 36base×两接口=72prompt | 8 | 共576，两个arm共享；满足lock与sample key条件可复用P3 |
| step32 | 同72prompt | 8 | 1,152 |
| step64 N-ID | 完整dev 288prompt | 8 | 4,608 |
| step64 L | 完整L 176prompt | 8 | 2,816 |
| step64 graph-OOD | cross_series 36新base×两接口=72prompt | 8 | 1,152 |

step64完整N包含panel时，按sample key复用，禁止双算。表内仅sampled，不强制给所有endpoint再加greedy；如记录greedy，单列成本和指标。

graph-OOD只测cross_series，36base按2拓扑（path/cycle）×2图形×3operation×3实例构成；全部先过唯一解检查，且不与ID共享真值结构/图像变体。它不是全家族OOD，不能宣称所有关系类型都有泛化证据。除图结构外，其余生成条件保持ID范围。

L保留自己的lock和长度。只能比较同一L协议下X_VALID−X_BASE及各自对起点的变化，不把N与L的不同绝对值当作受控训练差。

### 6.5 主分析与结果命名

主contrast为step64的X_VALID−X_BASE，六群体与固定权重整体分别报告 ΔpX、Δv、ΔqX、ΔqS。另报每个arm相对于同一初始化的变化，以及图结构OOD变化。

显示点估计、cluster CI、实际计数，不只给p值。用预指定六群体共同区间（如bootstrap max-stat）分析最差群体；普通逐组95%区间不得解释为六组同时95%。小样本/退化bootstrap时标不稳定，不能平滑后强行给正结论。

可将“validity–truth reversal candidate”定义为Δv>0且ΔpX或ΔqX<0，分别列效应及区间；只有方向的点估计称趋势。无显著下降不等于问题解决；需给可以排除的实际下降幅度。

单seed为探索性训练结果。不要自动选择最有利checkpoint，也不要用L/OOD来挑λ、LR或模板。

### 6.6 暂缓的训练

A_BASE/A_VALID、多seed192步、完整SSVC、GDPO/PCGrad/CPO比较、关系一致性奖励、LoRA范围扩展均默认禁用。R3的A/X梯度诊断用于决定是否值得追加完整2×2，不因省略这些arm就声称已完成主验证器交互结论。

## 7. R5：下一轮的离线 SSVC 进入条件与接口

本轮交付代码接口和决策报告，不默认运行在线控制。

只有完成R0–R4/R3-warm，且有可辨别的非零候选响应或已明确的测量改进方案，才启动R5。没有观察到伤害并不阻止研究安全区域，但不能为证明方法必要而制造负面结果。

### 7.1 候选与约束

候选仍由相同θ、Adam、rollout独立fork产生，λ有限网格。主保护f为六群体的pX和qX_pool；v为优化目标。qS是附加诊断，初版不把大量约束无节制叠加。

τpX=0、τqX=0为“不降低”的主定义；非零容忍度必须独立注册，不得因难以通过而临时放宽。λ0增量为0，逻辑上可行；不要用独立采样噪声把完全相同更新判为不安全。

分离控制proposal、候选选择、最终审计；不能在同一noisy bank挑最安全候选再用原区间报告安全。有限网格共同区间要覆盖候选×群体×指标；多步自适应选择需额外time-wise控制，本轮不声称已解决。

### 7.2 三态而非二态

- `EMPIRICAL_SAFE`：每个保护量相对baseline的共同下界均≥−τ，且目标/信任域要求通过；明确是经验筛选，不是数学证书。
- `UNSAFE`：至少一个保护量的共同上界低于−τ。
- `UNKNOWN`：其余情况，特别是稀少X、ESS低、区间过宽、候选很小但测不准。

UNKNOWN不commit，回退λ0。没有可信曲率/梯度误差界或具有证明覆盖率的独立有限候选审计时，禁止写`CERTIFIED_SAFE`。

### 7.3 控制分辨率审计

报告不同λ实际产生多少个可区分的更新、非零λ接受率、回退原因、额外GPU成本、分层安全精度。若λ0→正值主要是组激活且正区间近似平坦，继续加密λ网格不是解决方案。

需要另开新算法时，明确其候选族。例如对“完整联合优势相对于主优势的增量”增加独立缩放，是不同于原r(λ)族的新干预；不能伪称只是调λ，也不能假设Adam下参数步长按系数线性缩放。须重新执行同起点实际更新和独立审计。

## 8. 统计和理论的统一边界

1. 四类概率变化是边缘净变化，不是唯一的X→S→W→I运输矩阵。
2. 相同seed生成前后类别配对只定义了一种耦合；不证明内部路径身份。
3. rollout bank的重采样是条件经验分析；新模型生成必须标实际CUDA。
4. pX可因合法率或条件纯度变化而改变，必须联合报告pX=v*qX。
5. prompt群体内全合法不代表它不会受到别的群体更新的迁移。
6. 由canonical真值字符串的likelihood不能代替包含所有合法写法的事件概率。
7. 一步SSVC相对安全不保证独立长期X_BASE轨迹比较安全，更不保证主基线本身不退化。
8. 增大K会改变组信息，但固定BK时减少独立prompt数；本轮只CPU审计K，不自动扩大训练K。
9. α、ε、归一化、loss、clip、Adam状态都是机制的一部分，不是可以静默忽略的实现细节。

## 9. 最小代码结构和命令契约

优先扩展现有repo；缺少模块时增加等效功能：

```text
src/audit_p3_artifacts.py
src/audit_sampling_and_runtime.py
src/diagnose_repair_inputs.py
src/group_support_audit.py
src/joint_advantage_reference.py
src/optimizer_fork.py
src/paired_response_estimator.py
src/train_two_arm_pilot.py
src/statistics_scene_cluster.py
src/report_next_stage.py
```

建议入口（需实现/映射）：

```bash
python -m src.audit_p3_artifacts --config configs/next_stage.yaml --phase R0 --read-only
python -m src.audit_sampling_and_runtime --config configs/next_stage.yaml --phase R1
python -m src.diagnose_repair_inputs --config configs/next_stage.yaml --phase R2
python -m src.optimizer_fork --config configs/next_stage.yaml --state cold --no-commit
python -m src.train_two_arm_pilot --config configs/next_stage.yaml --arm X_BASE --allow-training
python -m src.train_two_arm_pilot --config configs/next_stage.yaml --arm X_VALID --allow-training
python -m src.optimizer_fork --config configs/next_stage.yaml --state warm --checkpoint <X_BASE_step64> --no-commit
python -m src.report_next_stage --run-root <NEW_RUN_ROOT>
```

`--dry-run`必须只打印阶段、数量、token上限和可测成本；不产生REAL_CUDA结果。`--resume`必须核对模型/数据/配置/优化器状态hash，不匹配则拒绝续跑。SLURM脚本沿用用户已验证的QoS/分区模板，从已有成功脚本读取，不猜GPU资源字符串或新账号权限。

## 10. 逐条日志、阶段产物与验收

### 10.1 rollout schema

```text
run_id, phase, execution_kind, protocol_version, model_id, model_revision
adapter_hash, optimizer_state_hash, checkpoint_step, train_seed
base_scene_id, prompt_id, split, family, interface, chart_type, operation
truth_world, observed_world, changed_index, cue, solution_count
image_hash, final_prompt_hash, input_ids_hash, image_grid_thw, actual_image_tokens
generation_config_hash, sample_key, sample_index, sample_rng_key
raw_token_ids, raw_text, extracted_action, parse_result, category
stop_reason, n_generated_tokens, per_token_logprob_behavior, logprob_sequence
copy_observation, hamming_to_observed, hamming_to_truth, constraint_results
runtime_forward_by_reason, elapsed, peak_memory
```

### 10.2 update/fork schema

```text
bank_id, group_id, prompt_id, K, category_counts, reward_vector
lambda, alpha, epsilon, variance_convention, group_reward_mean/std, advantages
dA_dlambda, no_X, validity_mixed, aux_only_activation, three_level_X_B_I
grad_norm_preclip/postclip, baseline_grad_hash, candidate_grad_hash
restored_model_hash, restored_optimizer_hash, candidate_model_hash
actual_delta_norm, actual_aux_delta_norm, optimizer_updates, backward_calls
proposal_bank_id, importance_diagnostics, predicted_delta, observed_delta
CI_scope, CI_method, joint_coverage_scope, support_warning, decision_status
```

### 10.3 每阶段必交文件

| 阶段 | 产物 |
|---|---|
| R0 | audit_findings.csv、protocol_diff.md、baseline_lock.json、per_prompt_counts.parquet、group_support_K.csv、recomputed_metrics.json、paired_CI.json、data_solver_checks.json |
| R1 | environment_lock.json、runtime_profile.json、likelihood_parity.json、lora_module_manifest.json、training_smoke.json、budget_projection.json |
| R2 | panel_manifest.json、prompt_diffs.md、diagnostic_rollouts、paired_condition_effects.csv、invalid_taxonomy.csv、diagnosis_report.md |
| R3 | bank_manifest.json、joint_advantage_checks.json、gradients_summary.parquet、candidate_manifest.json、paired_response.csv、support_control_report.md |
| R4 | two_arm_training_config.json、checkpoint_manifest.json、learning_curves.csv、endpoint_metrics.json、N_L_OOD_effects.csv、pilot_report.md |
| 汇总 | NEXT_RESULTS_SUMMARY_zh.md、unresolved_items.json、run_costs.csv、all_hashes.json |

每份报告必须含“实际执行了什么/没执行什么”“与锁定配置的差异”“原始证据路径”“统计单位”“异常与失败记录”。不得只返回一个PASS和几个总准确率。

## 11. 可直接转为测试的验收清单

### CPU 必测

- parser完整字符串检查、bool/float/越界/额外文本反例；四类互斥完备。
- 独立solver唯一解，真值与观察的一处变化，图像/变量索引一致。
- step/arm/protocol/raw hash绑定，sample_key去重，sampled/greedy不混。
- pooled ratio与prompt宏平均在构造的异质性样例上确实不同。
- 全合法组λ不变；B/I组正λ缩放抵消；λ0并列边界。
- dA/dλ与中心差分/自动微分一致，ε=0边界不输出假导数。
- 三层敏感度能量公式；K2三层出现率为0。
- bank组合概率与逐子集枚举一致；原bank IID时无偏；K>n拒绝。
- scene-cluster配对保存；增加某题采样数不会改变其主分析权重。

### GPU 必测

- 图像确实进入模型，替换图像改变输入张量，跨prompt无cache污染。
- 生成与teacher forcing对应同一π，EOS/截断/padding正确。
- adapter命中语言层；基座/视觉冻结；真实梯度和Adam更新有限。
- 同状态fork可复现；相同λ不依赖候选执行顺序。
- 梯度复用与完整loss等价；microbatch与等效batch一致。
- 全合法bank候选相同；λ0辅助增量为0。
- save/resume与不中断运行在约定数值精度内一致。
- smoke adapter不流入主训练；cold输出不冒充warm on-policy。

## 12. 本轮停止点

默认最终交付R0–R4及R3-warm的证据链，不自动进入多seed确认、在线SSVC、其他模型或高成本核建模。

最后的决策报告必须回答：

1. 哪些P3现象通过原始审计，哪些受到协议/实现影响？
2. 困难群体是否主要产生“无X的辅助激活组”，还是存在足够三层可调组？
3. λ到底改变了优势、梯度、Adam更新和输出分布中的哪几层？
4. X_VALID相对X_BASE是否改善有效率，是否改变绝对/条件真值及图结构迁移？
5. 下一轮的主要瓶颈是数据、推理接口、策略支持、归一化控制分辨率、统计精度，还是确有可控制的奖励风险？

允许其中部分结论是INCONCLUSIVE；不允许用未经验证的解释填补这些空白。
