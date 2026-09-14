# SSVC 下一阶段实验与代码接续规范

**版本：2026-09-14 / mechanism-followup-v2**  
**执行对象：Codex；仓库：`Lhan-chding/dissertation`；项目目录：`ssvc_flow/`**  
**本次工作终点：实现、CPU 测试、只读证据核验、生成服务器运行交接；服务器模型加载、GPU 推理及训练均单独授权。**

---

## 0. 先执行什么

请按以下顺序工作，不要从重新跑训练开始：

1. 核对 Git 分支、现有实现和本地未提交修改。
2. 阅读本文件、`CPU_ACCEPTANCE_CHECKLIST_zh.md` 和 `configs/followup_design.json`。
3. 在实验分支上实现 S0/S1/S2 所需代码；保留已有 R0–R4 行为与历史结果。
4. 运行新增 CPU 单元测试、已有相关回归测试、fake-adapter 端到端测试。
5. 使用用户提供的旧结果包做只读元数据审计；有服务器原始文件时再核验完整 checkpoint、raw rollout 和图片绑定。
6. 输出 `IMPLEMENTATION_REPORT_zh.md` 与 `SERVER_HANDOFF_zh.md`，列出真实通过的测试、剩余阻塞和准确启动命令。
7. 停在服务器启动之前。不要自动提交 Slurm，不要下载或加载 9B，不要将 CPU fixture 记录写成真实模型结果。

本文件中的新模块名、CLI 和状态名是**实现契约**。代码完成前，它们不是已有可运行入口。交付时必须逐一核实。

## 1. 实际仓库起点

### 1.1 已核实的分支

2026-09-14 通过 GitHub 连接读取到：

| 用途 | 分支 | HEAD |
|---|---|---|
| 默认分支，较早代码 | `main` | `653e3b96a9847f92e32ee9b262e11f2c16cb2033` |
| 已完成 9B 实验的实现分支 | `codex/ssvc-flow-ntu` | `7e405faa241e0085f701cdc858ba423ea76ba5a8` |
| 已创建的后续工作分支 | `codex/ssvc-mechanism-followup-20260914` | `7e405faa241e0085f701cdc858ba423ea76ba5a8` |

**后续分支已创建，但尚未包含已提交的新功能。** 对该分支读取 `ssvc_flow/src/followup_protocol.py` 返回 404。此前对话中的 `followup_*` 草稿没有形成可执行、已验证的分支实现。曾建立的 tree 对象不是 commit，不要把其存在当作功能已落地。

启动时重新读取远端状态；若 HEAD 已推进，应检查新增提交并保留用户/Codex 的工作。本文件的起点是审计锚点，不要求将较新的正确代码回退到该提交。

### 1.2 安全接续命令

```bash
git status --short --branch
git remote -v
git fetch origin
git branch -a
git log -5 --oneline --decorate origin/codex/ssvc-mechanism-followup-20260914
```

同时检查根目录及 `ssvc_flow/` 中适用的 `AGENTS.md`，读取后再修改代码。

工作区干净且无同名本地分支时：

```bash
git switch --track origin/codex/ssvc-mechanism-followup-20260914
```

已有本地分支时直接切换并检查差异。存在未提交内容时，先记录，使用独立 worktree 或保留当前工作区；禁止 `reset --hard`、`clean -fd`、强制覆盖、强制 push。不要修改 `main`，也不要合并 PR。

### 1.3 优先阅读的已有代码

路径均相对 `ssvc_flow/`：

| 文件 | 需复用或核对的能力 |
|---|---|
| `src/grpo_update.py` | `grouped_advantages`、`torch_ppo_loss`、`reward_channels` |
| `src/fork_gradients.py` | `direct_loss_gradients`、`apply_gradient_update`、梯度及样本绑定检查 |
| `src/optimizer_fork.py` | `capture_state`、`restore_state`、`state_hash`、checkpoint 读写 |
| `src/r3_updates.py` | 真 Adam 分支、主更新重放、旧候选比较与复用门禁 |
| `src/r3_runtime.py` | 原始 bank 装配、输入绑定、原子执行单元、候选加载 |
| `src/r3_warm_runtime.py` | step64 warm 起点加载、直接采样；旧实现固定 bank 0/6 |
| `src/r3_response.py`、`src/r3_warm_response.py` | 重要性采样、直接响应和现有统计 |
| `src/r4_inputs.py` | 训练排序、dev 面板、图结构 OOD；旧实现固定 seed17 |
| `src/r4_runtime.py`、`src/r4_continuation.py` | 两组真实训练、checkpoint 链、已审查警告续跑 |
| `src/next_stage_runtime.py`、`src/r1_reference_smoke.py` | 数据/运行环境/模型证书门禁 |
| `src/model_adapters/base.py` | 真实输入准备、采样、log-prob、位置状态重置 |
| `configs/next_stage.yaml` | 历史设计配置；实际执行以各 run 的 runtime lock 为准 |

已有证书和历史配置仍然约束旧实验。新增 S 阶段应有独立版本、配置、结果目录和入口，不能通过改写旧 JSON 来获得 PASS。

## 2. 研究目标与证据起点

本轮要补齐：

> **辅助奖励 → 联合归一化优势 → 真实优化器增量 → 独立评估上的四类输出概率变化。**

原始交付包根目录是 `SSVC_GPT_PRO_DELIVERY_20260914/`。以下是已完成实验的观察，不是本次新实验结果：

| 观察 | 对新实验的要求 |
|---|---|
| N 终点中 X_VALID 相对 X_BASE：`pX -1.35 pp`、`v +1.78 pp`、`qX -2.79 pp` | 检验辅助信号的边际效果，保留所有方向的结果 |
| 纯符号 trend：`pX -9.11 pp`、`v +7.29 pp`、`qX -13.42 pp` | 该分组列为预先指定重点；其余五组仍完整报告 |
| X_VALID 的 256 组中，214 组全合法；19 组无 X 且 V/I 混合 | 记录实际组构成；区分辅助信号的有无和强度 |
| warm bank3 中 lambda 0.01 与 1 的辅助增量范数接近 | 比较完整向量差、夹角和概率响应，不能只比较范数 |
| 原 warm 直接验证仅覆盖 bank0/6，二者内部候选参数相同 | 本轮必须覆盖已确认有辅助参数变化的 bank |
| 部分训练步当前梯度为零，Adam 仍更新 | 保留完整 optimizer 状态和显式零梯度语义 |
| 当前只有训练 seed17；L 协议效果方向不同 | 不将单 seed 或单分组结果推广到所有任务 |

具体来源见第 16 节及 `reference/REPOSITORY_AND_EVIDENCE_SNAPSHOT.json`。历史数值的区间和证据强度沿用原分析；不得补造显著性。

### 2.1 核心假设

- **H1：权重—更新不成比例。** 在无 X、V/I 混合的组中，小的正 lambda 可能已进入优势近饱和区；参数响应还受 Adam 历史状态影响。
- **H2：无 X 组的辅助信号具有可隔离的作用。** 对同一 bank，关闭这些组的辅助系数可能改变真值与合法率；方向由实验决定。
- **H3：作用具有群体差异。** 同一更新对不同关系家族/图像接口的影响可能不同。
- **H4：可区分的更新未必可精确测量。** 参数变化、分布重叠、目标事件支持与统计精度应分别报告。

四个假设允许不成立。S1 的技术通过条件不包含“必须降低 pX”或“必须使新方案更好”。

## 3. 共同数学与训练契约

### 3.1 事件与指标

`X`：输出的四整数世界完全正确。  
`S`：世界错误、输出合法，但确定性执行器得到正确最终答案。  
`W`：输出合法，但最终答案错误。  
`I`：输出非法或不完整。

`V = X ∪ S ∪ W`。这些是可观察输出事件，不是直接读取模型内部感知/推理隐状态。

```text
pX = P(X)
v = P(V)
qX = P(X | V) = pX / v
qS = P(S | V)
```

代码使用原始家族名 `duplicate_encoding`，不要将显示标签 `duplicate` 写回数据。报告变化使用“百分点（pp）”；概率差乘 100 后才是 pp。

### 3.2 两种辅助策略

对一个含 K 条输出的组 G：

\[
N_X(G)=\sum_i\mathbf 1[X_i],\qquad
r_i=2\mathbf 1[X_i]+\lambda_G\mathbf 1[V_i].
\]

`joint`：`lambda_G = lambda`。  
`no_x_off`：`lambda_G = lambda` 当 `N_X(G)>0`；否则 `lambda_G=0`。

随后只做一次联合标准化：

\[
\bar r=K^{-1}\sum_i r_i,\quad
s=\sqrt{K^{-1}\sum_i(r_i-\bar r)^2+\epsilon_r^2},\quad
A_i=(r_i-\bar r)/s,\qquad \epsilon_r=10^{-4}.
\]

零方差组所有优势为 0。实现 `no_x_off` 的位置是**辅助系数进入联合奖励之前**。不要把独立归一化的 validity advantage 从总优势中减掉；那不等价。

组不删除、样本不重采、主奖励不修改、总 token 损失分母不改变。`N_X=0` 只表示此次组内没有采到 X，不意味着策略对 X 的概率为零。

### 3.3 损失与优化器

保持历史目标：

\[
\mathcal L=-\frac{1}{BK L_{\rm norm}}\sum_{g,i,t\in\text{真实生成 token}}
\min\{\rho_{git} A_{gi},\operatorname{clip}(\rho_{git},0.8,1.2)A_{gi}\},
\]

其中 `B=4`、`K=8`、`Lnorm=64`；包含真实采到的 EOS，排除 padding；不补 EOS，不因输出属于 I 而删样本。old log-prob 是当前 bank 的真实行为策略记录，不允许重写为新模型概率来制造 ratio=1。

| 项目 | 固定值/规则 |
|---|---|
| 模型 | `Qwen/Qwen3.5-9B` |
| revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| 基座 / 可训练参数 | 基座 BF16；旧运行记录 LoRA 为 FP32，须核验实际 dtype |
| LoRA | language MLP：gate/up/down；96 个目标矩阵；r=8、alpha=16、dropout=0、无 bias |
| 冻结范围 | 基座和视觉编码器冻结 |
| AdamW | lr=1e-5，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| 梯度 | 全局 norm clip=1；所有可训练参数显式梯度张量，包含零 |
| 更新频率 | 每个完整 B×K bank 一次 Adam；microbatch=1 |
| 其他项 | beta_KL=0；entropy_coef=0；无 scheduler/grad scaler |
| N 生成 | temperature=1、top_p=1、top_k=0、min_p=0、repetition_penalty=1；sample；thinking=false；64 token |
| 图像与路径 | 沿用原 processor、图像限制和已验证 prefix-recompute 路径 |
| L 评估 | 完整继承 L 自己的 parser、数值域、模板和 token 预算；不套用 N 的 64 规则 |

历史 warm lock 记录 Python 3.12.14、torch 2.13.0+cu130、transformers 5.14.1、peft 0.19.1。它们是**原运行记录**，不是本文件要求重新安装的“最新版本”。先使用原环境，不自动升级，也不混用根目录较早 GPU lock。

Adam 的 `grad=None` 与显式 0 可能产生不同 step 行为。禁止用“零奖励就跳过 optimizer.step”替换历史实现。[W1,W2]

## 4. 执行阶段与优先级

| 阶段 | 任务 | 本地默认执行 | GPU |
|---|---|---|---|
| S0 | 代码、旧证据、配置、组构成审计；计划冻结 | 是 | 无 |
| S1A | 奖励/优势/小模型 Adam 的 CPU 性质与回归测试 | 是 | 无 |
| S1B | 原 warm bank 的非零更新和直接概率响应 | 只实现、fixture 和 dry-run | 单独授权 |
| S1C | 新混合 bank 的 No-X 隔离机制实验 | 只实现、fixture 和 dry-run | 单独授权 |
| S2 | 多 seed 三组训练及固定评估 | 只实现、fixture 和 dry-run | S1 测量链通过后单独授权 |
| S3 | SSVC 离线筛选可行性研究 | 留接口和报告字段 | 本轮默认关闭 |

S1 结果为零、不显著或与旧现象相反，仍可通过测量验收；只要技术链可靠，允许完整报告并进入预先批准的 S2。不要按效果好坏决定是否补种子。

## 5. S0：核对与冻结

### 5.1 只读来源

至少核对：

```text
02_data/R3_warm/runtime_lock.json
02_data/R3_warm/bank_manifest.json
02_data/R3_warm/candidate_manifest.json
02_data/R3_warm/gradients_summary.parquet
02_data/R4/runtime_lock.json
02_data/R4/learning_curves.csv
02_data/R4/continuation_decision.json
```

前三个文件的 SHA-256 已从当前上传 ZIP 计算，写在本包 snapshot 中。正式原始文件需进一步定位：`origin.pt`、候选 checkpoint、`samples.jsonl`、数据 split 文件和图片。

上传的 compact 包可以审计 JSON/表格，不能替代完整模型张量、raw rollout 和图片。缺少服务器原件时：继续完成本地代码和 fixture；把真实数据门禁标为 `BLOCKED_MISSING_PARENT_RAW`，不要合成真实样本。

### 5.2 来源身份

旧 warm runtime 记录的运行源码提交为 `9cb60cd8e4e26f881e83f80f20b55999f8b8ecde`，与 GitHub 当前实验分支 HEAD 不同。需比较 `runtime_lock.source.source_files` 的逐文件 SHA-256，重点是梯度、Adam、parser、adapter 和输入构造模块。

Git blob SHA、文件 SHA-256、模型参数 hash、完整 state hash 分别记录，禁止混用。路径可迁移，但内容与身份绑定必须核验；路径别名不能导致读写落入同一旧目录。

### 5.3 必须输出

`repository_audit.json`、`parent_evidence_audit.json`、`source_diff_review.md`、`parent_bank_composition.json`、`resolved_runtime_plan.json`。

每项标注 `CHECKED / NOT_AVAILABLE / FAILED`。`metadata checked` 不能升级为 `GPU_READY`。

## 6. S1B：原有 warm bank 的局部干预

### 6.1 固定来源与选择

起点：原 `X_BASE / seed17 / step64` 的完整状态。使用原 R3-warm 在这个状态采出的 training rollouts；禁止用 step65 策略重新生成后仍称同一个 bank。

预先指定 bank：`0、3、4、5、11`。这是基于已完成研究发现制定的**后续诊断设计**，不是原 R3 的事前随机抽样。

只使用 train-bank 组构成与参数响应选 bank，不使用新的 control/dev 结果筛选；来源不匹配时停止核对，不能换一个“效果更好”的 bank。

### 6.2 已核对的组构成及代数预期

下表中 X/W/I 数量按每组 K=8；所有这些组的 S 均为 0。

| bank | 四个 group（按原顺序） | 功能 |
|---|---|---|
| 0 | 8X；8X；4X+4W；7X+1W | 全合法的零辅助处理对照 |
| 3 | 8X；8X；4W+4I；8X | 纯无 X 混合组激活 |
| 4 | 7W+1I；8X；8X；8X | 不平衡 V/I 的无 X 激活 |
| 5 | 8W；8X；7W+1I；8X | 全错误合法组加无 X 混合组 |
| 11 | 8W；8X；4X+1W+3I；8X | X/合法错误/I 三奖励层 |

在当前损失契约下，代数上应有：

- bank0：joint0 与所有 joint 正权重具有相同优势。
- bank3/4/5：`no_x_off_1` 与 joint0 具有相同优势。
- bank11：`no_x_off_1` 与 joint1 具有相同优势。

这是必须验证的优势/更新一致性，不是新方法有效性的结果。尤其不能只在 bank3/4/5 比较 no_x_off 与主奖励，然后将预期等价包装成独立算法收益。

### 6.3 候选

bank3/4/5/11：

```text
joint lambda ∈ {0, 1e-6, 1e-5, 1e-4, 2e-4, 1e-3, 0.01, 1}
no_x_off lambda = 1
```

bank0：joint0、joint1。

新网格覆盖 `epsilon_r / sigma(V)` 附近的变化尺度。没有使用负 lambda；不改变 epsilon；不在此阶段搜索学习率。

先算全部候选的优势和真实参数更新；直接生成只覆盖：

| bank | 直接评估候选 |
|---|---|
| 0 | joint0、joint1 |
| 3 | joint0、joint0.01、joint1、no_x_off1 |
| 4 / 5 / 11 | joint0、joint1、no_x_off1 |

### 6.4 完整状态分支

对每个 bank，流程必须是：

```text
加载并核验 parent origin、原始 bank 与模型输入
→ 保存/检查完整起点
→ 恢复起点
→ 计算候选的联合优势与实际 clipped PPO 梯度
→ 显式赋全部梯度 → global clip → 一次真实 Adam
→ 保存候选参数/Adam/RNG/身份
→ 恢复起点并核验
→ 下一个候选
```

完整状态至少包括：可训练参数、Adam 所有状态/param_groups、Python/NumPy/CPU/CUDA RNG、sampler 位置、模块模式，以及影响 forward 的 buffer/位置状态。当前路径不引入缓存继续；使用现有 `_reset_positions` 并测试可变状态恢复。

一个 bank 至少重放一次 joint0，用于候选顺序不变性核验。GPU 上浮点一致性采用先冻结的容差，不得事后调大以通过测试；精确相等另报 hash。

`origin_checkpoint_step=64`、`candidate_optimizer_step=65`、`permanent_training_commit=false` 分开记录。候选不能覆盖原 step64，也不能直接作为 S2 的训练起点。

### 6.5 比较完整更新向量

定义：

\[
d_b(\lambda)=\theta_b^+(\lambda)-\theta_b^+(0).
\]

每对候选报告：`aux_l2`、`vector_difference_l2`、`max_abs_difference`、`cosine`、`relative_vector_difference`、参数 hash 和 optimizer hash。点积/范数累计使用 CPU FP64；实现须能逐张量流式处理。

**两个范数相同不代表两个向量相同。** 推理去重只能依据经过核验的精确 forward 状态相等，不能依据范数、allclose 或 cosine 接近 1。

同一推理策略但 Adam 状态不同的候选，可以复用推理测量；未来训练状态不得合并。

## 7. S1C：新增混合 bank，避免 No-X 消融退化为已有候选

### 7.1 为什么需要

原 bank3/4/5 的有效当前梯度都来自无 X 混合组，关掉后就是主更新；bank11 的无 X 组又全合法，关掉不改变优势。它们可以定位两类更新，但缺少“同一 batch 内二者同时存在”的隔离对照。

因此，新增一个组合 bank `composite_no_x_plus_three_level`。**只重组相同 warm 策略下已有的自然 sampled groups，不改 token、标签、old log-prob 或 prepared input。** 它是结构化诊断 batch，不是原训练 sampler 的一次自然 batch。

### 7.2 固定组成（group position 从 0 开始）

```text
source bank3 / group2  : 4W + 4I
source bank11 / group2 : 4X + 1W + 3I
source bank0 / group0  : 8X
source bank0 / group1  : 8X
```

以 parent `bank_manifest` 和 `candidate_manifest` 双重定位完整 prompt ID；本包 snapshot 保存了对应 ID。核验四题不重复、来源策略和数据身份相同、每组8条齐全、与 control 场景不重叠。保留 source bank/group ID；新 composite ID 单独生成。

### 7.3 三组候选

- joint0：只有三奖励层组中的 X 相对信号。
- joint1：同时包含无 X 混合组和三奖励层组的合法性影响。
- no_x_off1：仅移除无 X 混合组的辅助系数；其余组仍按 joint1。

仍然用完整 B4×K8、Lnorm64，一次 Adam；不删除被关掉的组。三个候选都纳入直接评估。保留一次 joint0 重放。

比较 `joint1 - no_x_off1`，得到**在这个固定 batch 和固定完整起点下，启用无 X 组辅助通道的增量作用**。Adam 和 global clip 非线性，因此不能把它等同于一条独立梯度经固定线性矩阵变换后的效果。

该结果不能独自推导长期退化的中介比例；长期影响由 S2 检验。

## 8. S1 的直接评估、预算与续跑

### 8.1 固定 control 面板

继承原 R3 的 24 个 control base scenes、每景两个接口，共48 prompts；每题16次 sampled；不混 greedy。六个群体均保留。`train/control` 原题与 truth-structure 排重沿用已经通过的旧证据，新增组装再检查 ID。

所有候选使用同一组 control prompts、解析器、模板、图片、生成配置与输出预算。随机 key 基于 `protocol/origin/bank/prompt/sample_index/seed_root`，同 bank 的候选不把 lambda/arm 放进随机 key；样本记录身份仍包含候选指纹。

同随机 seed 是 common-random-number 设计，不代表找到了自然的事件配对或 `I→W` 运输路径。

### 8.2 样本保存与正确分类

逐条保存 raw text、真实 token IDs、行为 token log-prob、stop reason、解析结果、分类、prompt/scene/candidate hash、seed 与执行检查。即使是非法回答 I，也是一条有效观测，不得因为语义错误判为执行失败。

真正的执行故障（非有限概率、输入 hash 错、没有调用视觉但应当有等）保留返回数据和故障后停止该单元。禁止吞错后补采“好样本”。

### 8.3 去重与资源上限

完整逻辑预算：

| 项目 | 数量 |
|---|---:|
| 原 bank 候选 | 38 |
| composite 候选 | 3 |
| 合计候选 Adam 更新 | 41 |
| joint0 重放 | 每个 bank 1 次，共6次 |
| 最大 Adam 调用总数（不含独立环境 smoke） | 47 |
| 每次梯度计算序列数 | 32 |
| 对应反向序列调用上限 | 1,504 |
| 直接评估逻辑候选 | 18 |
| 每逻辑候选输出 | 48×16=768 |
| 直接输出上限（去重前） | **13,824** |

只有同一 bank、相同完整推理指纹、相同 sample seed 设计时才去重；alias 记录引用已有 raw ledger，不能被统计为额外独立样本。按代数预期可能减少至约13个独立候选面板，但预算仍按未去重上限。

禁止一次把所有 bank 的候选模型/Adam 状态留在 GPU。候选逐个执行，checkpoint 存 CPU/磁盘；计算成对距离时按张量读取。时间与峰值内存依据实际计量填写，不根据加载显存推算训练峰值。

### 8.4 测量精度与可选扩展

16 次/题是小预算 screen，不承诺能识别单步的微小真值变化。默认不因为“不显著”或“方向不对”自动增加样本。

可选扩展需单独批准：对预先固定的候选使用全新的评估随机命名空间，每题64个新样本。pilot 与新 block 分开报告；不得在反复查看后把累计样本当固定样本实验。全18个候选扩展上限为55,296条，未批准时为0。

### 8.5 断点续跑

以 `(origin, bank, candidate, prompt, sample_index, protocol)` 作为唯一身份；每条输出完成后原子写入/flush。已有完整 candidate 或 sample 只校验后复用。

SIGTERM/OOM/异常不得覆盖旧 ledger、忽略已保存错误或把“完成”标记写早。测试随机中断后续跑与不中断执行的一致性。原始 parent 全程只读，新运行不得使用 parent 目录作为 output。

## 9. S2：多 seed 三组训练

### 9.1 训练组

| arm | 奖励/组策略 |
|---|---|
| `X_BASE` | joint，lambda=0 |
| `X_VALID` | joint，lambda=1 |
| `X_VALID_NO_X_OFF` | no_x_off，lambda=1 |

先完成所有三组的可运行实现和 CPU 测试；GPU 执行在 S1 的测量链技术验收与预算批准之后。

### 9.2 种子与历史结果处理

第一批计划：`17、29、41`。

- seed17 的 X_BASE/X_VALID：保留已完成结果，作为已观察的探索锚点。
- seed17 的新 arm：新增一次；必须绑定原初始 adapter 与原训练 prompt 顺序。
- seed29、41：各自三组，从该 seed 的相同初始 adapter 和同一 prompt 顺序开始。

共7个新训练 run：`1 + 2×3`；每 run64步×4题×8条=2,048条；总新增训练输出 **14,336**。

这是“1个探索 seed + 2个前瞻 seed”，不能写成3个独立事前确认 seed。需要3个全新前瞻 seed时，另行批准 seed53 的三个 run，增加6,144条训练输出。

源代码/输入实现发生影响结果的变化且不能通过 bridge 时，不把旧 seed17 与新 seed17 的差异解释为纯 reward 效应。保留旧结果，新建单独命名的重跑计划；不得静默重跑并替换历史值。

### 9.3 训练顺序与随机性

`r4_inputs.build_train_schedule` 的旧版本使用固定 seed17。新 S2 必须显式参数化 sampler seed；seed17 要能逐 prompt 复现旧顺序 hash。

同一 seed 的三个 arm 使用同一初始参数、数据顺序、训练步数、B/K、损失、解码与优化器配置。每个 arm 从自己的当前策略采样；禁止共享另一条训练轨迹的输出。

将“实验随机身份”和“代码执行身份”分别保存。相同实验重启不得因输出目录、日志时间或非语义日志修改而重置样本流；跨版本接续必须核验明示的允许差异。不同 arm 的同 seed 不保证训练后同 token。

### 9.4 固定评估

| 项目 | 计划 |
|---|---|
| checkpoint | 0、16、32、48、64 |
| 72 prompt dev 面板 | 每题8次，以上所有步骤；终点可从完整 N 评估读取同题结果 |
| 终点 N | 原144 base scenes×2接口=288 prompts；每题8次 |
| 终点 L | 原176 prompts；每题8次；独立协议报告 |
| 图结构 OOD | 原36 base scenes×2接口=72 prompts；起点与终点每题8次 |
| 接口诊断 | 在0/64步复测 R2 的 SYM_ORIGINAL、IMAGE_CUE、IMAGE_ONLY；原72景、每条件4次 |

N、L、OOD 不合并为单一“总正确率”。R2 接口诊断是独立诊断面板，不能把其结果塞入 N/OOD 主结果。

同一 seed 的初始状态完全相同且接口/采样锁一致时，可复用一份 step0 评估并保留 alias；不同 seed 或配置不一致不得假定相同。历史没有某步 checkpoint 时标记不可用，不通过新训练伪造旧中间点。

主要长期对比：`X_VALID - X_BASE` 与 `X_VALID_NO_X_OFF - X_VALID`。预先指定重点群体 `trend/SYMBOLIC_FRESH`；总体和六个群体均完整报告，保留其他方向和无差异结果。

评估也占用推理预算：按上述面板与接口诊断，单个新 run 的评估输出上限为8,896条（step64面板从完整N读取），7个新 run 合计62,272条，尚未扣除同seed相同初始状态的复用。另对两个历史seed17终点补接口诊断最多1,728条。S2的14,336条只是训练输出，不是总生成量。所有时间/显存预算应结合这些评估工作量，先由服务器实测计量。

### 9.5 诊断警告政策

读取原 `continuation_decision.json` 并在任何新训练前冻结统一政策。已审查的有限累计 KL/log-ratio 警告与真正非有限数值故障区分处理；不可按 arm 或效果差异临时放宽。

非有限梯度/参数/概率、原件 hash 变化、训练/评估串扰、冻结参数变动、证书不匹配必须停止。有效的 I 输出本身不触发停止。

`X_VALID_NO_X_OFF` 是机制干预，不称为已证明安全的 SSVC；如果它损失合法率、没有改善真值或伤害其他群体，照实报告。

## 10. 统计与概率响应契约

### 10.1 点估计

对预先固定的 prompt 权重 w（总和为1）：

\[
\hat p_{X,g}=\sum_{j\in g}w_j\frac{n_{X,j}}{n_j},\quad
\hat v_g=\sum_{j\in g}w_j\frac{n_{V,j}}{n_j},\quad
\hat q_{X,g}=\hat p_{X,g}/\hat v_g.
\]

w 在所报群体内归一。qX 是“加权 pX 与 v 的比”，不是逐题 qX 的平均。v=0 时 qX 为 undefined；不能填0。

每个 bank 单独报告 `candidate - same-bank joint0`。composite 另报 `joint1 - no_x_off1`。不得把不同 bank 的 joint0 混成一个基线，也不得把挑选的5个 bank 当作完整策略的无条件平均更新。

### 10.2 两类误差分开

1. **场景泛化不确定性**：配对 bootstrap base_scene；同一景的两个接口、所有候选一起重采；在关系家族内分层；5,000 replicates；无组内二次重采。
2. **固定面板的采样不确定性**：单独报告每题事件计数、未观察 X 的题数，以及固定面板 Monte Carlo 区间/保守边界。

两类区间目标不同，不混为一个“安全置信度”。单个 bank/少量场景、边界概率、全零事件导致 bootstrap 退化时标记 `UNINFORMATIVE`，不能给出“[0,0] 因而确定安全”的结论。

CPU 可实现一个不依赖渐近近似的固定面板保守边界，用于诊断：对事件指示变量 Y∈[0,1]，

\[
\hat p=\sum_j\frac{w_j}{n_j}\sum_kY_{jk},\qquad
r=\sqrt{\frac12\Big(\sum_j w_j^2/n_j\Big)\log(2M/\alpha)}.
\]

在不同 j,k 抽样独立、权重与候选集合固定时，对 M 个事件概率各报 `[max(0,p_hat-r),min(1,p_hat+r)]`；用 union bound 保证这组区间的覆盖。候选间共享 seed 允许相关，不影响 union bound。

qX 使用 pX/v 的区间传播；v 下界为0时仅给出平凡范围[0,1]并标记 UNINFORMATIVE；点估计 v=0 时 qX 点估计为 undefined。差值可由右候选下界减左上界、右上界减左下界构成。该保守边界可能很宽，作用是避免稀有事件零计数带来的虚假精确性。

这个边界只处理固定候选/面板的生成采样误差，不处理模型选择、训练 seed、场景分布迁移或无限次自适应选择，也不是本轮 SSVC 的整体安全证明。

### 10.3 重要性采样

旧重要性采样结果可以作为对比，但必须与新直接采样独立核验：

- OIS、SNIS、直接采样分别命名；不要让估计器静默切换。
- 每题 log-space 计算，报告 mean weight、ESS、ESS/n、`n*max(normalized_weight)`。
- `n=16` 时均匀权重的 max 为0.0625，不能使用0.05作为逐题硬门槛。
- 不默默截断权重；OIS 的有限样本估计超出[0,1]时保留估计器诊断，不能 clip 后伪装为无偏结果。
- 分别报告 X/S/W/I 观察次数与事件 ESS。未观察到 X 不等于 pX=0；高总 ESS 也不意味着目标事件估计可靠。
- 旧 proposal 只能提供 estimator 诊断，不能用自己给候选打分再宣称独立验证通过。

本轮保留 `NOT_CERTIFIED`。`ESS/n<0.5` 和 `n*maxw>8` 可作为预先固定的告警，不作为充分安全/不安全条件。

### 10.4 多 seed

先报告每个 seed、每个 arm 的结果与配对差，再汇总均值/标准差和所有 seed 的方向。场景 bootstrap 不代表训练随机性。

seed29/41 与已看过的 seed17 分开标注。只有少量训练 seed 时，不用成千上万条 completion 制造虚假的大样本显著性。群体、指标、候选网格探索的多重比较需单独标注；点态95%区间不叫“所有群体联合安全”。

## 11. 代码实现组织

### 11.1 建议新增模块（允许等价合并，功能不可缺失）

| 新模块，均在 `ssvc_flow/src/` | 责任 |
|---|---|
| `followup_protocol.py` | 强类型设计配置、candidate ID、reward policy、预算、phase 边界 |
| `followup_parent.py` | 原件读取、ZIP/目录元数据审计、版本/哈希/scene/样本绑定 |
| `followup_inputs.py` | 原 bank 选择、composite 装配、S2 seed 排序；无 model import |
| `followup_updates.py` | no_x_off 接入、完整状态 fork、向量比较、默认 joint 回归 |
| `followup_runtime.py` | S1 候选持久化、逐样本 ledger、resume、推理 alias、真实 adapter 接口 |
| `followup_train.py` | S2 多 seed/arm 驱动，复用 R4 原子 step 与断点链 |
| `followup_statistics.py` | 点估计、场景配对 bootstrap、固定面板边界、重叠/事件支持 |
| `followup_cli.py` | plan/audit/analyze；独立真实执行子命令有明确门禁 |

可新增 `configs/mechanism_followup.yaml`，由本包 `followup_design.json` 转成已校验的运行配置。不要将设计 JSON 直接传给旧 R3/R4 入口。

### 11.2 最小修改原则

避免另写一套基本 PPO/Adam 数学。优先为已有梯度计算增加显式、默认不变的 group policy 接口，或抽出共享纯函数；旧默认值必须通过完整回归。

不能靠 module-global monkeypatch 修改 `LAMBDAS`、`TRAIN_SEED`、`ARMS` 或旧 gate 的 PASS 条件。新的 runtime 应调用可复用底层组件，明确继承和改变的协议字段。

若必须改变旧模块，逐文件记录目的、默认行为是否改变、对旧证书和 source hash 的影响；保留旧 run 的代码绑定并做 CPU/GPU bridge。不能把“新源码哈希不同”当作自动忽略旧来源检查的理由。

### 11.3 核心接口契约

```python
# 类型名称为接口约定，Codex 可结合现有类型实现。
def group_reward_statistics(categories, *, auxiliary_weight, policy, epsilon): ...
def audit_parent_evidence(parent_path, *, required_origin, readonly=True): ...
def build_followup_bank(parent_manifest, raw_rows, bank_spec): ...
def fork_one_candidate(adapter, optimizer, origin, groups, candidate_spec): ...
def compare_update_vectors(baseline_state, left_state, right_state): ...
def run_followup_bank(validated_plan, *, output_root, resume=False): ...
def build_training_schedule(train_scenes, *, seed, preserve_legacy17=True): ...
def run_training_arm(validated_plan, *, seed, arm, output_root, resume=False): ...
def summarize_direct_response(left_rows, right_rows, *, fixed_weights): ...
```

真实 runtime 必须调用已有 adapter 的实际 `generate` 和 `logprobs`，使用原 `annotate_diagnostic`/parser 分类。不能以“接受任意 callback 并信任它返回 PASS”替代来源、模型、概率路径和输入核验。

### 11.4 候选更新伪代码

```python
restore_complete_origin()
for group in groups:
    has_x = any(row.category == 'X' for row in group)
    effective_lambda = requested_lambda
    if group_policy == 'no_x_off' and not has_x:
        effective_lambda = 0.0
    rewards = [2.0 * (r.category == 'X')
               + effective_lambda * (r.category != 'I') for r in group]
    advantages = historical_joint_zscore(rewards, epsilon=1e-4)
    # 完整 B*K*64 分母；包括优势为零的样本。
    accumulate_real_ppo_gradients(group, advantages, denominator=B*K*64)
assign_explicit_gradients_for_every_trainable_parameter()
clip_global_grad_norm(1.0)
optimizer.step()
save_candidate_with_optimizer_rng_and_hashes()
restore_complete_origin_and_verify()
```

本轮不启用 category-score 梯度复用、vLLM、新 cache path、量化、新优化器、reward-wise 独立标准化或在线 SSVC。

## 12. 产物、失败与恢复

### 12.1 每个 S1 bank

```text
identity.json
source_binding.json
bank_manifest.json
group_statistics.json
origin_binding.json
candidates/<id>/checkpoint.pt
candidates/<id>/candidate.json
candidates/<id>/samples.jsonl
candidates/<id>/counts_by_prompt.json
parameter_geometry.json
policy_aliases.json
paired_response.json
runtime_profile.json
manifest.json
status.json
```

公共仓库只提交代码、测试、去标识规范和必要的紧凑摘要。模型、Adam/RNG、原始输出、图片、服务器身份信息、绝对私有路径放在忽略目录，不自动推送。

### 12.2 每条样本最少字段

```text
protocol_version, execution_kind, run_id, seed_root
origin_state_hash, origin_checkpoint_step
bank_id, source_bank_ids, candidate_id, candidate_policy_fingerprint
candidate_optimizer_state_hash, candidate_optimizer_step
base_scene_id, prompt_id, family, interface, operation
prompt_hash, scene_hash, image_hash_or_null
sample_index, sample_seed, sample_key
raw_completion, token_ids, behavior_token_logprobs, stop_reason
parsed_world_or_null, category, parser_version_hash
execution_checks, record_hash
```

原始空文本+EOS可以是 I，但 token ledger 和执行检查必须完整。不要将 `I` 与运行异常混为一类。

### 12.3 状态与文件安全

采用 `CPU_TESTED`、`BLOCKED_MISSING_PARENT_RAW`、`READY_FOR_SERVER_REVIEW`、`RUNNING`、`MEASURED`、`MEASUREMENT_FAULT` 等显式状态。没有实际采样时不能写 `MEASURED`。

路径必须在新 run root 下，拒绝 `..`、越界符号链接与重复身份；已完成结果 no-clobber。对外来 ZIP 只读受限 JSON，先限制单文件大小/总量；不要无条件解压或执行压缩包代码。

checkpoint 先验 file hash 和来源，再 `torch.load(..., weights_only=True, map_location='cpu')`；该设置减少反序列化风险，不把不可信文件变成绝对安全。禁止 `weights_only=False` 作为自动兜底。[W3]

异常时保留 fault 记录，恢复完整 scratch origin；恢复失败须单独保留并使 adapter 不可继续使用。故障后的“恢复成功”不能把本次测量写成 PASS。

## 13. CPU 验收与完成定义

逐项测试见 `CPU_ACCEPTANCE_CHECKLIST_zh.md`。至少覆盖：

- 旧 joint 损失/梯度/更新回归；no_x_off 前归一化语义；全合法与零方差组。
- 小模型上明确零梯度 vs None 的 Adam 行为；相同范数不同向量。
- 六个 bank 的装配、预期组构成、composite 来源；wrong origin/old score 拒绝。
- 每个候选参数/Adam/RNG恢复；候选顺序不变；可变 forward 状态恢复。
- 假模型完整 train/fork/evaluate/resume；覆盖 I 与真正异常。
- parser/权重/比值指标/置信区间边界；16样本重叠门槛；零事件不产生安全结论。
- 隔离多个 seed、arm、protocol、输出目录；无默认 GPU/network/sbatch。

复用项目已有 CPU 环境；记录实际 Python/库版本。跑过的测试给出命令、退出码、通过/失败/skip 数与完整日志路径。若依赖或资源不够，列出具体失败及缺项，不能用 mock 全部通过代替关键真实 PyTorch CPU 测试。

本包 `reference_math.py` 是标准库数学参考，不是生产更新器。其通过只验证本文件的组奖励公式和预算约束。

## 14. CLI 与交接命令

### 14.1 实现完成后应支持的入口

下列命令是**验收目标**，Codex 须实现后再运行。工作目录为仓库 `ssvc_flow/`：

```bash
"$PY" -m src.followup_cli plan \
  --design configs/mechanism_followup.yaml \
  --out "$NEW_ROOT/plan.json"

"$PY" -m src.followup_cli audit-parent \
  --design configs/mechanism_followup.yaml \
  --parent "$PARENT_WARM" \
  --out "$NEW_ROOT/parent_audit.json"

"$PY" -m src.followup_cli analyze \
  --run-root "$NEW_ROOT/S1" \
  --out "$NEW_ROOT/S1_ANALYSIS"
```

`$PY` 是已核验的 Python 路径；`$NEW_ROOT` 是新结果目录；`$PARENT_WARM` 是目录或可做紧凑审计的 ZIP。缺少必填环境变量时退出，不落到当前目录默认执行。

真实入口要与 plan 分离，例如：

```bash
# 仅放入服务器交接文档；本次 Codex 不执行。
"$PY" -m src.followup_cli run-s1 \
  --validated-plan "$NEW_ROOT/validated_plan.json" \
  --bank bank03 --out "$NEW_ROOT/S1/bank03" \
  --allow-gpu-execution

"$PY" -m src.followup_cli run-s2 \
  --validated-plan "$NEW_ROOT/validated_plan.json" \
  --seed 29 --arm X_BASE \
  --out "$NEW_ROOT/S2/seed29/X_BASE" \
  --allow-gpu-execution --allow-training
```

命令中的 bank ID 列表包括 `bank00/03/04/05/11/composite_no_x_plus_three_level`。GPU 标志不是充分授权；必须同时通过实际 source/data/model/runtime/预算门禁。

### 14.2 `SERVER_HANDOFF_zh.md` 必须提供

最终实际 commit、分支、工作目录；经过实现验证的全部命令；私有路径如何绑定；需要的原件清单；完整 CPU 结果；服务器 smoke 步骤；S1 各 bank 顺序与预算；每个 S2 run；resume 与停止操作；应回传的紧凑分析文件。

使用仓库现有可用 Slurm 模板与获批资源；不要猜 GPU 节点、partition、QOS 或墙钟限额，也不要沿用旧聊天中的过期排队信息。先生成脚本并 `bash -n`，实际 `sbatch` 留给用户。

### 14.3 Codex 最终必须交付

```text
新增/修改源码与完整 diff
configs/mechanism_followup.yaml
新增 CPU 测试与相关旧测试的运行日志
IMPLEMENTATION_REPORT_zh.md
SERVER_HANDOFF_zh.md
machine_readable_readiness.json
```

`machine_readable_readiness.json` 分别报告 `code_implemented`、`cpu_tests_passed`、`parent_metadata_verified`、`raw_tensors_verified`、`gpu_smoke_passed`、`gpu_started`、`training_started`。本次默认最后三项为 false/未测，而不是由配置自动置真。

## 15. 结果如何影响后续 idea

| 新结果 | 下一步研究解释 |
|---|---|
| 小 lambda 与大 lambda 更新接近，微小尺度出现平滑变化 | 控制需基于实际更新分辨率；数值奖励权重不等于干预幅度 |
| 无 X 激活组的真实响应损害 pX/qX | 提供局部机制证据；再看 S2 长期可重复性 |
| no_x_off 改善真值但明显降低合法率 | 确认权衡，不能称为已经解决；继续设计有约束控制 |
| composite 中 no_x_off 与 joint1 不同，但各原 bank 中出现预期 alias | 表明组合后的条件消融具有区分力；注意非线性优化器影响 |
| 梯度/参数确有差异，直接采样无法辨认概率变化 | 保留测量不足；调整后续预算或研究可测性，不宣布安全 |
| 新 seed 没有旧趋势或方向相反 | 收窄现象适用范围，分析组构成与训练状态；完整保留结果 |
| 图像读取高、修复接口变化明显 | 进一步研究信息源冲突与正确坐标保持，勿直接等同感知增强 |

SSVC 后续按“候选更新是否可区分 → 影响是否可测 → 是否满足语义约束”组织。但本阶段只为该控制器建立证据，不将固定 no_x_off 门控改名为 SSVC，也不在尚无测量精度时自动上线。

## 16. 来源与本次设计的区分

**实验证据：**用户上传的 `SSVC_GPT_PRO_DELIVERY_20260914/02_data/`，以及此前独立分析 E04/E05。snapshot 仅重新核对了所列 JSON 身份、组构成、环境记录和 SHA-256；没有重跑模型。

**仓库依据（2026-09-14 实时读取，具体代码固定到以下提交）：**

```text
https://github.com/Lhan-chding/dissertation/tree/7e405faa241e0085f701cdc858ba423ea76ba5a8/ssvc_flow
https://github.com/Lhan-chding/dissertation/blob/7e405faa241e0085f701cdc858ba423ea76ba5a8/ssvc_flow/src/fork_gradients.py
https://github.com/Lhan-chding/dissertation/blob/7e405faa241e0085f701cdc858ba423ea76ba5a8/ssvc_flow/src/optimizer_fork.py
https://github.com/Lhan-chding/dissertation/blob/7e405faa241e0085f701cdc858ba423ea76ba5a8/ssvc_flow/src/r3_updates.py
https://github.com/Lhan-chding/dissertation/blob/7e405faa241e0085f701cdc858ba423ea76ba5a8/ssvc_flow/src/r3_warm_runtime.py
https://github.com/Lhan-chding/dissertation/blob/7e405faa241e0085f701cdc858ba423ea76ba5a8/ssvc_flow/src/r4_inputs.py
```

**技术实现核对，官方文档（访问日期2026-09-14）：**

```text
[W1] https://docs.pytorch.org/docs/2.13/generated/torch.optim.Optimizer.zero_grad.html
[W2] https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html
[W3] https://docs.pytorch.org/docs/2.13/notes/serialization.html
[W4] https://huggingface.co/docs/trl/grpo_trainer
```

W4 仅用于说明 GRPO 实现存在不同损失分母和标准化选项；本项目以实际锁定代码为准，不迁移到 TRL 默认实现。

**本次新设计：**S阶段命名、微小 lambda 网格、composite bank、前归一化 no_x_off 消融、两新 seed 的执行矩阵、固定面板保守边界与代码模块契约。它们是下一步计划，不是已完成结果。
