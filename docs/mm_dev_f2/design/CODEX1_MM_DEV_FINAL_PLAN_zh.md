# Codex1 执行计划：Qwen3.5-9B 多模态训练响应实验（MM-DEV F2）

**版本：2026-10-09 / MM-DEV-F2-QWEN35-9B-20261009**  
**性质：下一阶段科学实验执行合同；不是继续讨论稿，也不是已运行结果。**  
**执行范围：必要的新增 GRPO 实现验收 → 4 条准备训练 → 5 个起点的训练前测量 → 30 条响应训练 → 固定终点评价与开发分析。**  
**模型：仅 Qwen/Qwen3.5-9B。GPU 运行小时只记账，不设累计时长或研究墙钟停止阈值。**

本文件与 `config/MM_DEV_F2.json`、数值源清单、槽位表和运行矩阵共同交给 Codex1。用户将本包交给 Codex1 执行后，技术验收通过就直接进入上述科学实验；不要又只完成审计然后等待一轮研究讨论。只有明确技术阻塞、输入身份冲突或需要改变科学配方时才暂停。

本包不包含已经实现、在 GPU 上验证过的完整 MM-DEV 训练器。`reference/` 是可核验的 CPU 参考合同与清单构造代码。Codex1 需要在现有 `mm_core` 工程上实现独立的 `mm_dev` 执行层，完成本文件规定的技术验收，再运行科学路径。

---

## 0. 总决定与不可漂移的研究问题

本阶段检验的问题是：

> **对同一 Qwen3.5-9B 起点，只改变下一段训练的数据曝光，能否测出各项答案表现和报告读数忠实性的不同变化？这些响应是否随起点而变，为以后使用训练前语义信息预测、选择训练安排提供真实的数据基础？**

研究主线仍是：

**可验证语义行为测量 → 未见训练响应预测 → 收益与跨能力干扰约束下的训练选择 → 多模态 RL 验证。**

本轮已经处在真实多模态 RL 中，不再以符号实验为前提。SER-J2 只作已有动机；SER-J23 不恢复。不增加位置、输出顺序、提示搜索、几何机制、额外奖励算法等科学分支。

本轮确实运行科学训练，而不是再次测一个公共起点。但 **5 个开发起点不能证明新起点泛化或控制器有效**。本轮的强交付是完整的“起点 × 动作 × 未来重复 × 能力”响应表、状态与动作的交互读数，以及下一次独立预测验证所需的数据接口。MM-LOCK、MM-ONLINE、MM-CAL 不在本包执行范围内。

---

## 1. 先读懂此次结果，而不是把 ENGINE 的 PASS 当成学习收益

### 1.1 证据来源与本次核查范围

本计划的事实依据是用户提供的 `MM_CORE_PRO_REVIEW_20261009.zip`，不是旧讨论中的预期。

本次在本地执行了该包的完整性验证器，并从原始 `raw_text` 与题目真值侧表独立重算全部 **1,536 条 MEASUREMENT_AUDIT 回答**的 P/C/A 和精确误差分解，逐条对齐原评分；未发现不一致。另核对了原 ENGINE 的 8 条更新记录。

**没有在本地重新运行 Qwen、重算服务器 checkpoint 张量，或重新测量 GPU 吞吐。** 非零 SFT 的逐元素恢复结论来自包内服务器独立核验回执；本地仅核验其文件身份和报告内容。范围详见 `EVIDENCE_REVIEW_zh.md` 和 `validation/INDEPENDENT_AUDIT_RECOUNT.json`。

| 已有事实 | 原始计数/结果 | 对下一步的实际含义 |
|---|---:|---|
| 两个格式面板 | 各 384/384 合格 | 不再做格式 SFT；公共起点仍是本轮未训练的 9B |
| 报告读数全部正确 P | 1,509/1,536 = 98.2422% | 当前原型读数基线很高；不能预设还有很大提升空间 |
| 模型答案正确 A | 1,465/1,536 = 95.3776% | 有错误，但不等于每个采样组都有可学习奖励对比 |
| P 与 A 同时正确 | 1,451/1,536 = 94.4661% | 需要保留联合事件，不能只报宏答案分 |
| 精确六格 111/100/011/010/001/000 | 1,451 / 58 / 14 / 13 / 0 / 0 | 58 条“读数对、答案错”全部在 range；001 未观察到不等于不可能 |
| K=4 中答案奖励非恒定的题 | 13/384 | 原审计采样通道下的奖励对比较稀疏 |
| S 有定义的错误回答 | 27 条、支持 6/16 根 | S 不是全体回答上的无缺失状态变量 |
| 原 GRPO ENGINE | 16 组零奖励对比；8 次梯度为 0 | 恢复比对通过不等于非零 RL 更新已经验证 |
| 后补 NONZERO_SFT_RESUME | 8 次真实非零 SFT 物理更新，连续4与新进程2+2一致 | 证明对应 SFT 路径的恢复；不把该 adapter 或结论挪给科学 GRPO |

来源：原包 `START_HERE_FOR_GPT_PRO_zh.md`、`audit/report/MEASUREMENT_FINDINGS_zh.md`、`documents/mm_core_f1/NONZERO_RECOVERY_RESULT_20261009_zh.md`、原始回答/评分/更新记录。

### 1.2 本次复算新增的有限观察

27 条读数错误中，**26 条仅一个坐标有 ±1 误差；1 条有两个同号、幅度为 1 的非零坐标**。对应 S 分别为：2 个读数时 1/2，3 个读数且单坐标错时 1/3，两坐标错时 2/3。

因此，大部分现有 S 值已由“操作数个数＋单坐标错误”约束，只有极少观测展示额外形状差异。不能因为 S 被算出来，就宣称它已经包含可用于训练选择的信息。该观察只针对这批已揭晓审计回答，不是未来状态的结论。

### 1.3 对准备奖励还必须做一个检查

若同一 rollout 组的读数得分都是常数 c，则

\[
 r^{AP}=0.5r^A+0.5c.
\]

去均值并按组内标准差归一化后，rA 与 rAP 的优势几乎相同；差异仅来自固定的数值 epsilon。**给奖励增加读数项，不保证产生不同的训练方向。** 高读数基线使这一风险尤其值得记录。

所以必须记录两种奖励在同一组实际轨迹上的优势差异，而不是先假定 4 条准备路径会制造出明显不同的语义状态。不按结果挑选或替换准备状态。

---

## 2. 哪些继承、哪些是本次正式决定

### 2.1 继承已收敛的科学骨架

- Hp=32，H=32；4 条准备，5 个起点，3 个续训动作，每动作 2 个未来重复，共 30 条响应。
- 同次生成 `readings` 与 `answer`，读数在前；`enable_thinking=false`；真实图像输入。
- 续训动作只改数据曝光，三动作使用同算法、同答案奖励、同调度、fresh optimizer。
- V×D 四格顺序为 **LL、HL、LH、HH**，第一字母是视觉复杂度 V，第二字母是数值配置 D。
- a0=(1/4,1/4,1/4,1/4)，aP=(1/8,3/8,1/8,3/8)，aC=(1/8,1/8,3/8,3/8)。每个图表×运算层内执行。
- 起点锚定的收益/损害及 no-train 参照；BNum 对 BGeo-S 是未来主要信息比较；context 稳定性仍单列。
- MM-DEV 为开发证据，不把 30 条路径或 180 个能力单元当成独立训练起点。

### 2.2 本次补齐的全部科学参数

下表是本次明确选择，不声称这些值早已在审计阶段冻结，亦不声称最优。

| 待决项 | 本次确定值 | 理由/范围 |
|---|---|---|
| 数据域 | 沿用已审计的 chart-v1 原型；使用全新数值源 | 先测已可解释环境的响应，不根据已有错误临时改难度制造效果 |
| 准备 reward | rA=A_full；rAP=0.5A_full+0.5逐所需读数正确比例 | β=0.5、凸组合、精确评分；明确允许该配方未产生足够状态差异 |
| Bp/Br | 都为每更新 24 个 prompt | 扩大每步题目覆盖；每两步恰好48槽，无配额余数 |
| G | 每 prompt 8 条新 on-policy 轨迹 | 两类准备、三个续训动作完全一致 |
| 梯度 microbatch | 1 条 completion；192 条累积后一次更新 | 保留序列等权，避免显存需求决定样本权重 |
| RL 采样 | temperature=1，top_p=1，top_k=0 | 使实际采样与 raw-policy 概率比对应同一数学策略；不是按效果调温度 |
| probe/eval 采样 | K=4，temperature=0.7，top_p=0.9，top_k=0 | 保持原审计评价通道，训练采样与评价采样分别登记 |
| KL 系数 | 0.02，固定起点 reference，采样 token KL 估计 | 有限更新正则，不把它称为读数安全保证；不同于旧 ENGINE 的 KL=0 |
| ratio clip / 复用 | 0.2；每组轨迹只作1次优化更新，不做多epoch | 固定单次训练语义；不可用复用次数增加未登记训练量 |
| optimizer | AdamW lr=1e-5，betas=(0.9,0.999)，eps=1e-8，wd=0，clip-norm=1；constant，无warmup | 明确签收原工程稳定实现，不按32步结果调学习率 |
| LoRA/backend | 完整继承当前9B原生语言 full-attention q/v；r8/alpha16/dropout0；BF16 base、FP32 adapter、eager、原生未融合线性核 | 不借机扩大模型可训练范围或切换高吞吐后端 |
| PREP/CONTINUE 根 | 32 / 64 | 准备全池等权一次；续训按固定前缀选样，无单路径题目重复 |
| PROBE/DEV_EVAL 根 | 64 / 128 | 增加稀疏错误测量覆盖；不承诺1pp功效，不替代训练重复 |
| 权重/损害惩罚 | 六层各1/6；λ=1；每保护层容忍度2pp | 在任何新响应揭晓前固定 |
| 成本效用 | η=0；所有成本另报 | 首轮隔离同更新/rollout机会下的信息问题；不再引入GPU小时门槛 |
| MM-LOCK | 本轮不实例化、不执行 | 独立准备来源数与预测器应由开发结果另行冻结 |

训练采样改变意味着旧 K4 支持频率不能被当成本轮 T=1、top_p=1 下的真实混合组概率。本轮记录实际每组奖励，不能拿插件估计替代。

### 2.3 用户覆盖优先级

最新用户决定 > 本 F2 文件及 JSON > 原 F1 未被覆盖条款 > 最后收敛讨论 > 更早候选。

取消的是 GPU 时长阈值，不是取消实验设计。32步、固定题量和重复次数定义本次比较；它们不能因资源充足自动延长。原审计的4096次生成/64工程更新等限额属于旧 run，**不得误加载进新的 MM-DEV ledger**。也不能通过关闭全部账本来规避这一问题。

---

## 3. 执行身份与历史证据保护

### 3.1 模型与公共起点

使用原包记录的同一快照，执行前重新在服务器核对实际文件：

```text
model_id: Qwen/Qwen3.5-9B
revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
model_weights_hash（原清单复合身份）:
921fb7dbfce93fbcf3ab0ccc4fe63bdc41269e7b6ce8f8d6fc315c042fee5511
recorded_snapshot_path:
/projects/varunssd/louis-ssvc/cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

该复合身份不是某一个 safetensors 文件的 SHA。逐文件 identities 仍须逐项核对。权重若移动，只允许定位相同字节的快照，不允许下载 `main` 的更新版本替换。

S0 是 `COMMON_START.json` 中的未训练9B。SFT工程adapter、原零梯度ENGINE adapter都不得作为S0。

为统一训练参数化，给 S0 初始化一次固定的零输出 LoRA：常规随机 A、零 B；使用 `seed_registry.json` 的公共初始化seed。它在数学上不改变基座输出，需用CPU/短前向数值比对确认装载实现；把这个完整初始adapter存成不可变公共文件。所有准备路径从同一份完整adapter开始，而不是各自随机重初始化A。

### 3.2 当前运行环境

原证据记录：Python 3.12.14、torch 2.13.0+cu130、transformers 5.14.1、peft 0.19.1、accelerate 1.14.0、Pillow 12.3.0；TRL 当时未安装，实际使用自定义原生训练代码。完整环境和核函数源身份见 `evidence/E06_PROCESSOR_ENVIRONMENT_LOCK.json`。

这些是原报告中的环境事实，不是本次聊天重新核验的服务器状态。Codex1应定位该环境并逐项核对；**不得以安装 pyproject 的所有 optional dependencies、升级 Transformers/TRL/vLLM 作为默认步骤**。

输出名称中 `model_hash`、tokenizer hash、processor hash、chat-template文件hash、模板参数hash和协议hash是不同对象，按各自定义记录，不能把它们合并成一个“模型没变”。新执行代码有新commit和source hash；原审计冻结目录保持字节不变。

### 3.3 运行根、并发和权限

建议独立运行根：

```text
/projects/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009/
```

这是一项新输出路径建议，Codex1须先确认没有既有同名run冲突。使用已有项目工作区的新代码目录/commit，保留用户未提交修改，不执行破坏性 reset/clean。

最多5个GPU worker，每worker1张已授权Pro6000，原则上一个训练路径一张卡。先核对当前账户/QoS权限；原证据中的 `rose / override-limits-but-killable` 只是定位线索，不代替当前分配许可。记录显卡完整型号、驱动与设备编号。不得切换3B、别的9B版本或量化版来求快。

Slurm要求的单作业walltime是租约，不是科学停止条件：在租约/抢占前保存并精确续跑。无“超过8/24/26 GPU小时就停止研究”的分支，不把延长累计GPU时长作为再次科学审批门槛。

---

## 4. 实施顺序：一次进入真实科学训练

| 阶段 | 操作 | 放行条件 |
|---|---|---|
| F2-0 | 原包只读核验；新清单渲染；实现新训练器；CPU/损失测试 | 数据身份、权限、评分、配额、代码均通过 |
| F2-ENGINE | 按第7节完成新增实际GRPO路径与恢复核对 | 新概率/奖励/批次/参考策略实现可靠；不要求工程题必有学习收益 |
| PREP | rA与rAP各2条，32步 | 四条均达到终态；不按表现择优 |
| PROBE | S0与四个准备终点的训练前测量 | 5个起点全部固定且完整测量；缺支持照实记录 |
| CONTINUE | 每起点a0/aP/aC×2未来seed，32步 | 30条全部达到终态；中途不按结果改动作 |
| DEV_EVAL | 5起点+30终点，固定新面板 | 全部预登记槽位完成或明确列为技术不完整 |
| RELEASE | 响应表、误差/支持/成本、开发机会分析、独立复算 | 一次统一揭晓、完整交付后停止 |

允许工程上并行：S0 probe可与4条准备一起占用5张卡；不同完成时间不能决定样本、起点或续训动作。预先提交固定运行矩阵。为了防止不自觉地按结果改配方，DEV_EVAL的科学聚合结果应在34条科学路径和5个probe均固定之后统一释放。

---

## 5. 数据：已有可读原型，全新可核验的科学根

### 5.1 数据域正式签收

本次正式签收如下原型参数，不自动把这些条件外推为真实世界图表：

- grouped_bar、line；sum和有序difference各2个所需值，range=max−min为3个所需值。
- 真值整数10–89，轴0–100，单位count，δ=1；difference的第一个真值大于第二个。
- 目标值两两间隔至少5；896×672画布；原生processor后每单位4.88像素，整数网格最小栅格间隔4像素。
- V低/高分别显示2/4系列，同一完整数值源派生；目标系列及所需类别沿用原查询契约，不改成新的寻物任务。
- D在同一数值源上共同满足三种运算的个位进位/借位低或高条件；D不是被证实的纯计算难度，也不由模型准确率定义。
- 不添加真值标签，不降低分辨率制造读数错误，不更改容差或输出顺序，不筛选“模型恰好不会”的题。

当前高正确率可能使实验在32步下只得到小响应。这是允许的结果，不能用它否定广义语义状态，也不能通过临时加难题挽救预期。

### 5.2 完整池规模

每个根家系含2个D数值源，每数值源有2种V×2种图表，每张图3个运算提示。因此每根有 **8张图、24题**。

| 池 | 根家系 | 数值源 | 原始图像 | 题数 | 用途 |
|---|---:|---:|---:|---:|---|
| ENGINE_F2 | 4 | 8 | 32 | 96 | 新实现的隔离工程检查 |
| PREP | 32 | 64 | 256 | 768 | 四条准备路径共享的固定池 |
| CONTINUE | 64 | 128 | 512 | 1,536 | 三种曝光动作共用候选池 |
| PROBE | 64 | 128 | 512 | 1,536 | 每个起点的事前特征 |
| DEV_EVAL | 128 | 256 | 1,024 | 3,072 | 每个起点及训练终点固定评价 |
| 合计 | 292 | 584 | 2,336 | 7,008 | 不含未来LOCK |

`config/numeric_sources.jsonl` 已在本地用原审计生成器的数值构造函数产生这584个数值源，**未调用模型，未渲染图像**。数值源SHA和构造记录见 `MANIFEST_BUILD_RECEIPT.json`。不要在服务器重新随机找一批“更合适”的根。

Codex1须从这些源渲染出2,336张新图、完成同一原生processor路由，生成题目/侧表/模型输入清单，并把实际图像、处理结果、提示、token和代码hash写入新的 `F2_FREEZE.json`。这些hash待产生是执行身份字段，不是尚未决定的数据规模或科学配方。

### 5.3 隔离与去重

完整根家系是最小切分单位。一个家系的两个D源、所有V/图表/运算提示不得跨池。共享读数的sum/difference必须同池；不能把一种运算当训练、另一种当独立测试。

本次额外预先采用数值内容去重：在各D内全局不重复有序前两个目标值对；全局不重复排序后的三个range操作数集合，并排除原审计全部面板中已经出现的对应内容。另核对完整源和图像的重复。不要因标签相同就删除所有等答案题；答案碰撞不是输入泄漏。

这是本次新增的明确构造条件，不是原审计已经实施的事实。新面板是同原型下、带上述有限总体无重复约束的独立数据集，不能声称与原16根完全同分布独立有放回。

构造记录只因定义上的数值冲突重抽：本次274次有序对冲突、4次range集合冲突；没有使用模型输出来选样。失败候选也在构造规则与receipt中留痕。

### 5.4 真值权限与可读性

模型只能接收图像及可见提示。`true_values`、gold answer、D标签、root ID、候选序号、reward不能进入token prompt；图像路径也不能被模板转成可读提示。通过负测试确认注入假的sidecar值不会改变准备给模型的输入张量。

训练奖励可在生成之后使用训练真值；PROBE真值用于双方共有的离线测量；DEV_EVAL真值只评分，不参与选择训练数据/起点/长度。

渲染与processor在新图上复核几何、截断和目标可辨性。在任何模型结果揭晓前，从固定的首尾根/各视觉格抽样看原图与处理图。只因渲染或路由错误修复整个规则，不因模型答错换图。图像从源到processor的两套hash都保存。

---

## 6. 固定起点与曝光动作

### 6.1 五个起点

| 起点 | 来源 | 准备输入/顺序 |
|---|---|---|
| S0 | 本轮未训练9B＋固定零输出LoRA参数化 | 无准备更新 |
| S_A_0 | S0以rA训练32步 | PREP_p0 |
| S_AP_0 | S0以rAP训练32步 | 与S_A_0完全同输入/顺序/配对采样seed |
| S_A_1 | S0以rA训练32步 | PREP_p1 |
| S_AP_1 | S0以rAP训练32步 | 与S_A_1完全同输入/顺序/配对采样seed |

两种准备seed使用同一PREP题目多重集，只改变预登记顺序与采样随机性；每条准备路径使用PREP的768个题各一次，每题生成G8轨迹。两配方的实际轨迹随策略分叉各自on-policy，不共享另一模型产生的训练答案。

准备是改变参数、梯度及优化历史的干预，不是“只改变μ”。保留两种配方的所有结果，即使适配器完全相同。报告独特参数hash数和可测行为差异；不得按补偿率选择“最漂亮”的状态。

### 6.2 每条响应路径的动作

每条路径32更新×24prompt=768个输入槽位。6个图表×运算层各128槽。

| 动作 | 每层LL | 每层HL | 每层LH | 每层HH | 所有层合计 |
|---|---:|---:|---:|---:|---:|
| a0 | 32 | 32 | 32 | 32 | 768 |
| aP | 16 | 48 | 16 | 48 | 768 |
| aC | 16 | 16 | 48 | 48 | 768 |

所以aP仅把V高的边际提升到3/4，D仍各1/2；aC反之。三动作每层权重均1/6、操作数个数不变。它们分别称“视觉复杂配置侧重”和“数值配置侧重”，不称纯感知训练/纯计算训练。

### 6.3 确定性配额与共同样本

每连续两次更新形成48槽块；每层8槽配额分别为(2,2,2,2)、(1,3,1,3)、(1,1,3,3)。每一步仍含每层4题；两步合起来精确配额，不声称每一步都精确3/4。

实现见 `reference/plan_contract.py::slots` 和9份 `config/schedules/*.jsonl`：两个PREP流、六个CONTINUE流、一个ENGINE流。

同一未来重复下，所有起点使用相同的对应动作输入流。三动作在各层各格使用同一确定性根排序的不同长度前缀，避免动作之外再随机换整批供体。高曝光格取48根，均衡取32，低曝光取16；64根池足够且单路径无题目重复。

不需要余数策略：32×24=768=16×48。禁止把24改回工程的2题，再声称六层精确配额。

---

## 7. 最小必要技术补充：验收新科学训练器，而不是再开科研校准

旧 SFT 非零恢复的价值保留，但新配方增加了连续奖励、实际采样概率记录、KL、192序列累积和一般化checkpoint游标。这些路径必须先验收。

### 7.1 CPU/不更新核对

至少覆盖：

1. 原评分器精确六格、容差八格、缺字段、重复键、NaN/Inf、字段顺序、域外值；原1,536条记录复算零不一致。
2. rAP的1/3、2/3等分数奖励可进入normalizer，删除旧ENGINE“只允许0/1奖励”的硬编码。总体标准差，不用样本标准差。
3. all-zero/all-one优势为0；loss标量为0不代表梯度为0；有常数读数项时归一化近等价。
4. 概率ratio、clipping、KL符号、token mask、192序列均权；microbatch与不累积的玩具梯度结果一致。
5. checkpoint恢复不再只接受step2；流hash、逻辑步、rollout游标、reference、RNG、optimizer状态完整。
6. 数据泄漏、配额、seed、run matrix、资源账本、重复执行和未知作业状态的负测试。

本包66项CPU测试是参考底线，不等于服务器训练器已经通过这些验收。

### 7.2 自然奖励的新 ENGINE_F2

固定4个ENGINE_F2根的96题，a0均衡顺序，B24/G8；rAP、KL=0.02、T1/top_p1，完全使用第8节新训练器。

连续4步与独立进程2步保存、退出、新进程再2步比较：8个物理更新、1,536条真实生成。工程adapter全部隔离并丢弃出科学起点集合。

验收：

- 输入槽位、采样seed、token、原始回答、实际奖励、优势、old/ref/current logprob、损失、梯度、参数变化记录齐全。
- 同一硬件/后端下，第2、第4步可用张量精确比较；参数、Adam一二阶矩、scheduler、RNG、reference、游标及后续真实生成一致。
- actual sampler选择token的raw logprob与teacher-forcing同前缀的raw logprob核对：预登记工程容差为平均绝对差≤0.005 nat、最大绝对差≤0.05 nat。它们是本次技术标准，不是已获得的实测PASS；不按本次结果放宽。
- 当前模型logprob必须带梯度，old/ref必须detach；reference hash始终不变；base/vision/projector/linear attention不改变。
- 若实际奖励出现非恒定组，必须确认对应policy-gradient路径确实有有限非零梯度并产生可见参数变化。不能用KL梯度或“优化器调用了”替代。

### 7.3 若固定工程题仍无奖励对比

不得重抽题、改真值、加奖励、扫描更多题直到出现想要的结果。

若**全部实际组均为恒定奖励**，先核验真实组归一化和policy-gradient确实按数学应为零。然后允许一次已预登记的隔离实现压力测试：重置到工程初始状态，用同样真实模型采样的序列，在policy surrogate中暂设每序列系数为+1；保留实际奖励/实际优势旁路日志但不把它们用于压力目标。仍连续4对新进程2+2，共额外8物理更新、1,536生成。

这个有意设置的系数不是自然GRPO优势，不是新奖励算法，也不是科学训练。其用途只是在同一ratio/KL/累积/恢复代码路径上验收非零参数更新；结果只能叫 `PASS_NONZERO_SURROGATE_KERNEL_RESUME`，不能叫“非零答案奖励GRPO已验证”。负优势和组归一化由CPU及自然轨迹核对共同覆盖。

若有真实非恒定奖励却始终没有对应的梯度/变化，这是 `GRPO_GRADIENT_MISMATCH` 技术阻塞，**不能通过压力测试绕过**。

自然路径数学正确、恢复一致，且自然或隔离压力分支至少一种非零实现检查通过，即可进入固定的科学矩阵。真实科学任务是否有奖励对比由MM-DEV本身测量，不再变成无止境的工程等待。

### 7.4 技术失败的处理

数据/概率/梯度/恢复不合格：记录明确失败，修复代码后重做受影响工程检查，重新冻结代码身份。不得在失败期间偷偷开始科学路径。技术验证不使用PROBE/DEV_EVAL来选超参数，不做MM-CAL，不据ENGINE收益决定H或lr。

---

## 8. 唯一科学训练配方

### 8.1 输出与评分保持原契约

模型一次回答：

```json
{"readings": [number, ...], "answer": number}
```

各字段均由模型生成；不是统计独立，不由执行器填答案。只接受冻结解析器认可的数值，精确比较用Fraction/Decimal，不四舍五入到gold。字段缺失和可评分字段分别处理：缺readings不删除一个可评分的answer；逆序/额外字段若数值明确，保留能力评分并另报格式违例。不可解析整条记录不是000。

P_full：完整所需读数向量正确；缺读数字段为0。  
A_full：答案等于真值运算结果；缺答案为0。  
C：答案与模型自己的读数运算一致，只在联合可评分集合L定义。  
J_full：P与A同时正确，联合字段缺失为0。

### 8.2 奖励公式

令所需读数个数为n，逐值正确比例为

\[
 q_{\rm read}(q,y)=
 \begin{cases}
 \frac1n\sum_{i=1}^n\mathbf1[r_i=v_i],&L_P=1,\\
 0,&L_P=0.
 \end{cases}
\]

四条准备：

\[
 r^A=A_{\rm full},\qquad
 r^{AP}=\tfrac12 A_{\rm full}+\tfrac12q_{\rm read}.
\]

三十条响应全部使用 rA。无格式奖励、长度奖励、per-token读数奖励或SFT混入；rAP不是整向量P的简单加分。两种准备均使用同样精确规则；δ/2容差只用于评价诊断。

### 8.3 实际采样与策略概率

准备和响应都按当前策略以T=1、top_p=1、top_k=0采样，所有其他logits过滤/重复惩罚关闭，`enable_thinking=false`，max_new_tokens=192。完整有效生成序列含实际EOS/停止token；不把padding或prompt/image token计入completion loss。

保存实际sampling过程的选中token raw logprob作为old logprob。避免用一个未说明的库默认generation_config覆盖这些条件。改变采样通道是本次科学配方选择，不能把原0.7/0.9审计的rollout混进训练。

每个更新先在冻结的当前策略下生成24×8条序列；这一批生成期间不做参数更新。然后一次性计算优势和累计梯度，最后做一次optimizer step。各分支之后自行重新生成下一批on-policy轨迹。

### 8.4 GRPO式目标完整展开

对每题G=8个奖励，

\[
 \bar r_q=\frac18\sum_i r_{qi},\qquad
 \sigma_q=\sqrt{\frac18\sum_i(r_{qi}-\bar r_q)^2},\qquad
 \hat A_{qi}=\frac{r_{qi}-\bar r_q}{\sigma_q+10^{-8}}.
\]

若所有奖励相同，明确输出零优势。禁止丢弃这类组后补采，禁止只按非恒定组归一化训练batch。

对实际token t，

\[
 \rho_{qit}=\exp(\ell_\theta-\ell_{\rm old}),\quad
 d_{qit}=\ell_{\rm ref}-\ell_\theta,\quad
 k_{qit}=\operatorname{expm1}(d_{qit})-d_{qit}.
\]

\[
 \mathcal L=
 \frac1{24\cdot8}\sum_{q,i}\frac1{T_{qi}}
 \sum_{t=1}^{T_{qi}}
 \left[-\min\{\rho_{qit}\hat A_{qi},
 \operatorname{clip}(\rho_{qit},0.8,1.2)\hat A_{qi}\}
 +0.02k_{qit}\right].
\]

logprob和exp计算至少FP32；不偷偷clip logratio来掩盖溢出。非有限loss/梯度按技术失败处理，不能把该组丢掉后继续。

该KL是采样token估计项，不是完整序列KL的精确值，也不是读数不退化保证。单次轨迹集只进行一次梯度更新：在理想同策略数值下初始ratio为1，clipping通常不活跃。因此不能将收益归给“多轮PPO clipping”，也不能未登记地增大epoch使clip发挥作用。

### 8.5 参考策略、LoRA与优化器

准备reference是S0；每个响应起点拥有其精确冻结副本，供该起点所有动作和两个未来重复共享。准备后的reference不能通过 `disable_adapter()` 返回未训练基座来替代。

可用独立冻结模型或经验证的adapter安全切换实现reference前向；必须在当前策略带梯度前向前恢复正确参数，且不破坏optimizer参数引用。ref cache要按起点权重、输入、生成token共同键控。

LoRA仅在层3/7/11/15/19/23/27/31的q_proj/v_proj，共16个原生模块，r8/alpha16/dropout0；q_proj同时含query和gate通道。视觉、projector、线性注意力、其他base参数冻结。base BF16，adapter FP32，eval/dropout0但policy前向启用autograd。

AdamW：lr1e-5，betas(0.9,0.999)，eps1e-8，weight_decay0，global grad norm clip1；constant schedule，warmup0。新路径fresh optimizer；抢占恢复则继承完整optimizer/scheduler/RNG，不可fresh重新开始。

即使某次policy优势为0，KL或此前Adam动量仍可能改变参数。必须把“zero-contrast组”“policy-gradient为零”“总梯度为零”“参数不变”四项区分记录。

---

## 9. Seed、配对和完整运行矩阵

实际34条run ID和依赖见 `config/run_matrix.json`；无需Codex1自行补seed。

固定主要seed：

```text
公共LoRA初始化：6836382373189252249
准备主seed p0：3509815867821782651
准备主seed p1：2808912230178417534
未来主seed f0：691346289789482254
未来主seed f1：5685518181638526340
工程seed：8932404210382745746
bootstrap：4311636612181726188
```

派生方式统一是 `plan_contract.seed` 的canonical JSON → SHA-256 → 前8字节big-endian → 63bit，禁止Python内置hash、进程PID、worker号或Slurm job号。

- 准备：采样seed由(phase=PREP,repeat,question_id,sample_index)决定，不含reward recipe，使A/AP在同一准备重复内配对。
- 响应：由(phase=CONTINUE,future_repeat,question_id,sample_index)决定，不含起点/动作；共享题采用相同随机数安排，但输出必须由各自当前策略真实生成。
- 评价：由(panel,question_id,sample_index)决定，不含模型ID；同题K次种子不同。
- request ID必须包含run/model身份，避免不同模型的配对seed被误当成可复用同一回答。

共享随机数只能减少某些比较噪声，不能保证逐token输出相同，也不能增加独立训练重复数。

---

## 10. 测量：分清训练前特征、终点标签和条件缺失

### 10.1 固定调用量

PROBE：每个起点1,536题×4次=6,144条，5起点共30,720条。

DEV_EVAL：每个模型3,072题×4次=12,288条；5起点＋30响应终点，共430,080条。不得把no-train再训练一遍；它引用相应起点的同面板评价。

主分析只比较step32。step8/16/24保留checkpoint，但本轮不新增这些时点的科学评价，也不从中挑选“最好”的checkpoint。

所有已完成输出都计入预定分母。缺字段能力指标按各自full规则为0；技术上尚未生成的槽位不是模型答错，单列未完成。不能用删掉失败题后的分母冒充完整评价。

### 10.2 必交字段

每个PROBE回答记录原文/token、来源、解析覆盖、P/C/A、精确六格、容差八格、e、近失、S及有效性、εP/εC/εO、域外、截断、图表/运算/V/D和完整根身份。

同批PROBE额外记录：

- 实际自生成前缀下readings/answer的NLL和token entropy：每条回答一次raw teacher-forcing。
- gold completion的readings/answer NLL：每题每起点只算一次，4次采样共用；明确其answer前缀含gold readings，不等于自由生成表现。
- 源准备配方、预算、分层曝光、近期训练支持/梯度摘要，作为所有基线共同可见历史。

DEV_EVAL保存原文、token和事件/数值评分；本轮不为430,080条回答再做字段teacher-forcing。不能把终点特征输入事前预测器。

### 10.3 S与非冗余检查

\[
 e_i=(r_i-v_i)/\delta,\qquad
 S(e)=\frac{(n^{-1}\sum_i e_i)^2}{n^{-1}\sum_i e_i^2}
\]

只在n≥2且e非零时定义。零误差为missing，不是S=0。

按每根固定题型权重汇总；条件S只对有支持的根定义，报告支持根/全部根、有效回答数、幅度和缺失数。不能把条件根均值写成全部根固定权重均值。任意bootstrap副本无S支持时保留undefined，不补0。

同时记录非零误差坐标数和arity，在新状态上复核S是否几乎只由单坐标误差决定。BNum和BGeo共同看到错误支持、arity、幅度及missing指示；不能把“BGeo知道哪些题有错”算成S形状的贡献。

### 10.4 信息阶梯接口

交付完整、无标签泄漏的以下特征表，不以这5个起点直接认证方法：

| 档 | 共有信息/新增信息 |
|---|---|
| B0 | 分层P/A/C、L_P/L_A/L、NLL/entropy、结构、可得准备与曝光摘要 |
| BJ | 加联合正确率j |
| BTab | 加z等价的完整精确联合事件表；明确代数依赖 |
| BNum | 加逐角色bias/MAE/RMSE、整向量近失、错误支持/arity等常规数值信息 |
| BGeo-S | 在完整BNum之上，仅增加低维跨坐标S摘要；同样分母和支持条件 |

同根sum/difference稳定性只作context审计：跨提示collision与两个同提示collision一起报告；负的噪声校正差值不截0。不自动并入BGeo-S，不据此识别“先读后算”等内部机制。

**本轮不强制拟合高维预测器或控制器。**需要输出可直接用于独立验证的接口、固定低维候选登记和信息冗余检查。如Codex1做附加开发拟合，必须另列为探索且不得取代完整BNum比较；不自动升级为正式结果，也不据此运行新的动作。

---

## 11. 科学训练中必须记录的支持诊断

每个prompt的G8组保存：真实A奖励、q_read、实际使用reward、rA与rAP的反事实重评分（只算数，不另训练）、均值/总体std、实际优势、零对比标记、格式/截断比例。

对于四条准备，固定记录：

\[
 d_{\rm adv}(q)=\sqrt{\tfrac18\sum_i
 (\hat A^{AP}_{qi}-\hat A^A_{qi})^2}.
\]

报告 `d_adv > 1e-6` 的组比例及其分层值；这只是数值区分读数，不作采样筛选条件。两种counterfactual reward使用该条路径自己真实生成的轨迹，不当成另一个配方的真实训练结果。

每步另记录policy项/KL项、old/reference/current logprob摘要、ratio clip率、token/序列数、更新前后参数hash和最大变化、梯度范数、实际reference身份、各V/D格有效组数及稀疏性。

all-correct和all-wrong都保留；尤其不能只保存“产生梯度”的题，不能按未来轨迹再定义动作内容。

---

## 12. 固定统计分析与效用

### 12.1 响应定义

对起点s、动作a、未来重复f、六个能力层r，主指标A和P均用full概率：

\[
 D^A_{s,a,f,r}=A_r(\theta^{s,a,f}_{32})-A_r(\theta_s),\qquad
 D^P_{s,a,f,r}=P_r(\theta^{s,a,f}_{32})-P_r(\theta_s).
\]

同起点、同未来重复的相对动作效应：

\[
 \Delta^A_{s,a,f,r}=D^A_{s,a,f,r}-D^A_{s,a0,f,r},\quad
 \Delta^P_{s,a,f,r}=D^P_{s,a,f,r}-D^P_{s,a0,f,r}.
\]

同时保存J、六格和数值误差的变化，但不要把它们都宣布为独立主要终点。D描述相对起点的绝对变化，Δ描述相对于均衡续训的替换效应。

先题内K均值，再根内4个V/D等权，再六层各1/6；宏指标相当于24格等权。展示每个起点、两个未来重复的全部效应，而不只汇总30条路径平均。

### 12.2 主要比较与区间

本轮预先指定四个总体描述性主比较：在5个固定起点和2个配对未来重复上等权平均的

1. aP−a0的宏A；2. aC−a0的宏A；3. aP−a0的宏P；4. aC−a0的宏P。

给95%根簇区间，并额外给这四项对应的98.75%区间。5000次root bootstrap，使用seed_registry中的seed。每次重采一个完整DEV_EVAL根，连同该根所有24变体、K次回答、所有起点/动作/未来重复一起取出。

这些区间条件于本次实际训练路径，反映所测数值根的不确定性；并非模型/训练总体区间。新根采用有限总体去重构造，区间应称经验根簇描述区间，不写成任意自然图表总体的保证。

状态×动作差异、六层/24格、两种未来seed的差别完整描述。只有两个未来重复，不能把它们的std包装成可靠训练方差估计或安全界。预登记1pp作为实际关注量级，不是科学PASS门槛，不保证本面板能区分1pp。

### 12.3 起点锚定的效用与保护

以概率为单位，不以百分点混入公式：

\[
 G_s(a)=\frac16\sum_r D^A_{s,a,r},\qquad
 I_s(a)=\frac16\sum_r[-D^P_{s,a,r}]_+,\qquad U_s(a)=G_s(a)-I_s(a).
\]

λ=1，η=0；两未来重复先取平均并保留各自结果。正部逐层取后再平均，不能让一层读数提升抵消另一层损害。主保护条件逐层为 `D_P >= -0.02`。

no-train/延期：D_A=D_P=0、U=0、优化更新成本0；共同probe开销不是0。均衡a0不自动安全。若采用a0参照，no-train的Δ=−D(a0)，不能设为0。

完整报告各项原始读数、成本及1pp效应量级，不能事后调λ或ε让动作排名翻转。

有限样本的负部惩罚与max选择会产生乐观/悲观估计偏差，需承认；bootstrap是描述这种不确定性，不消除全部选择偏差。

### 12.4 是否有状态选择空间

先报告：动作响应是否可辨、哪些状态的动作次序不同、排序在两个未来重复间是否一致。不得从“aP平均比a0好”直接推出需要状态控制器。

提供两个标明为探索的机会读数：

- 不带约束：状态内最佳U的平均，减去所有状态上最佳固定动作的平均U；包含no-train。
- 带约束：状态内可行动作最佳U，与在所有开发状态均可行的最佳固定动作比较；两边都有no-train，避免可行集定义错误。

不能在同一带噪声表上挑oracle又宣布收益可靠。另做预先固定的双重交叉读数：用 `rank(range(128), "crossfit-root-half")` 将128个评价根确定性排序后分为前64/后64（完整索引在JSON中）；用未来f0的第一半根排序，在未来f1的第二半根评价，交换future与根半后平均。对固定动作基线用相同训练/评价拆分。

此读数仍是利用开发响应挑动作的事后上界/诊断，不是训练前语义选择器，更不是已实现策略。约束区间若跨边界，标为不确定；不能声称已保证安全。

### 12.5 不把同一准备来源拆成训练和测试

未来预测数据的行ID必须保留 `origin → preparation recipe/repeat → start_state → action → future_repeat → evaluation root` 家系。准备p0的A与AP是配对家系，不应拆一条到训练一条到独立验证；同一base公共起点更不是独立基座样本。

本轮5个开发起点最多支持开发趋势、可行性和噪声评估。正式BNum/BGeo比较必须以后使用新准备来源，先提交预测/动作再揭晓。无需这次凭5个状态填写MM-LOCK样本数或安全校准系数。

---

## 13. 成本：没有GPU时长门槛，但每个调用必须有身份

### 13.1 科学部分的固定工作量

| 项目 | 算式 | 数量 |
|---|---|---:|
| 准备逻辑更新 | 4×32 | 128 |
| 响应逻辑更新 | 30×32 | 960 |
| 总科学更新 | 128+960 | **1,088** |
| 输入prompt槽位 | 1,088×24 | 26,112 |
| 准备rollout | 4×32×24×8 | 24,576 |
| 响应rollout | 30×32×24×8 | 184,320 |
| 训练rollout合计 | 1,088×24×8 | **208,896** |
| 起点PROBE回答 | 5×64×24×4 | **30,720** |
| DEV_EVAL回答 | 35×128×24×4 | **430,080** |
| 科学生成合计 | 上三项 | **669,696** |

另有PROBE的30,720次self前向、7,680次缓存gold前向；训练的208,896次reference和208,896次带梯度sequence前向。在不重复计算old logprob的规定实现下，合计456,192次非generation sequence前向。必须继续记录实际vision encoder调用、prefill/decode token与反向成本；“一次生成”不是“一次Transformer前向”。

生成max_new_tokens=192是输出协议上限，不把它当作实际平均token。新T=1可能与原审计平均耗时不同，不能按旧2.018GPU小时线性承诺本轮耗时。

### 13.2 工程、失败和重做另列

自然ENGINE_F2：8物理更新＋1,536生成。若按注册条件执行一次压力分支，再加8更新＋1,536生成。它们不进入1,088条科学逻辑更新或响应表。

正常科学任务不能凭空多采；技术失败恢复可以重做未提交的更新，计入physical cost，不增加logical dataset。计数超过预期必须能由明确失败/恢复事件解释，不能悄悄扩大科学采样。

不设GPU小时限制。用设备数×实际分配时间记账，包含加载、空闲、失败和重做；排队、wrapper墙钟、GPU分配时间分开。不因达到某个小时数判失败、停任务或要求另选模型。

---

## 14. 检查点、可恢复性和长任务执行

每条路径有独占lease和不可变run manifest，防止重复提交。每次提交的逻辑更新保存完整状态；保留两份滚动安全checkpoint及0/8/16/24/32不可变checkpoint。保存参数、optimizer、scheduler、reference、Python/NumPy/torch/CUDA RNG、输入流hash、逻辑步/slot/sample游标、采样通道hash。

逐回答先落盘原始token/text，再完成该step的评分与更新；日志append和checkpoint提交要原子化。中断后：

- 从最后一次已提交更新恢复；未提交step的已完成rollout仅在策略hash、输入、seed都一致时复用，继续补齐未完成槽位。
- 若optimizer.step已执行但checkpoint/commit状态不明，回到上一可证明提交点重做，保留重复物理成本；不能猜测更新成功。
- 同一逻辑slot重做使用相同seed，旧失败原文不覆盖；canonical结果必须由事前技术重试规则确定，不能按正确性选择。
- 只有抢占/租约结束可在相同身份下自动续跑；软件bug、NaN、哈希冲突、梯度断裂暂停受影响任务并显式修复。修复若改变科学行为，需要版本化重跑受影响可比块，而不是混拼结果。
- 网络失联记录UNKNOWN，先核对Slurm与lease，不根据失联重复提交或擅自取消其他作业。

worker没有明确分配许可时不得加载GPU。全部停止/取消操作只针对本plan ID注册的任务；不删除历史唯一证据、其他项目数据或共享缓存。

---

## 15. 必须实现的程序入口与验收顺序

以下是Codex1应新增的接口目标，**不是声称这些脚本目前已存在**：

```text
scripts/mm_dev/materialize_data.py
scripts/mm_dev/validate_freeze.py
scripts/mm_dev/run_engine_f2.py
scripts/mm_dev/submit_matrix.py
scripts/mm_dev/run_train_path.py
scripts/mm_dev/run_probe.py
scripts/mm_dev/run_eval.py
scripts/mm_dev/score_and_analyze.py
scripts/mm_dev/verify_release.py
```

CLI都要求显式 `--plan` 与 `--run-root`，禁止从旧F1默认目录隐式取预算/模型/seed。训练入口只接受运行矩阵中的run ID；评价入口验证模型hash和固定panel；状态机不能因为ENGINE_PASS就启动MM-LOCK。

建议阶段状态：

```text
NEW -> DATA_MATERIALIZED -> F2_FROZEN -> ENGINE_VERIFIED
-> PREP_COMPLETE -> PROBE_COMPLETE -> RESPONSES_COMPLETE
-> EVAL_COMPLETE -> ANALYZED -> RELEASED
```

允许执行调度依赖重叠，但科学配置在F2_FROZEN后不可改。对外结果统一揭晓。技术失败状态单独记录，不能覆盖已完成原始文件。

启动前至少验收：

- `python -m unittest discover -s tests -v` 的本包参考测试；已有mm_core测试也继续通过。
- 数据清单、34run矩阵、9输入流、全路径768slot、每两步配额、真实图像路由、真值隔离。
- 双字段评分、连续奖励、fresh/reference、实际概率、梯度累积和恢复。
- CPU模拟调度能遍历完整状态机且不会因GPU小时字段为null出错。
- 可估算工作量、充分磁盘和分配可用性；不把估算GPU小时作为研究门槛。

---

## 16. 必交结果文件

至少交付：

```text
summary/FINAL_MM_DEV_REPORT_zh.md
summary/RUN_COMPLETENESS.json
summary/PRIMARY_EFFECTS.json
summary/RESPONSE_BY_STATE_ACTION_FUTURE_STRATUM.jsonl
summary/ABSOLUTE_AND_RELATIVE_CHANGES.jsonl
summary/PREPARATION_STATE_DIFFERENCES.json
summary/REWARD_AND_ADVANTAGE_SUPPORT.json
summary/JOINT_EVENT_TABLES.json
summary/NUMERIC_AND_GEOMETRY_SUPPORT.json
summary/CONTEXT_AUDIT.json
summary/UTILITY_AND_FEASIBILITY.json
summary/EXPLORATORY_OPPORTUNITY_CROSSFIT.json
summary/ACTUAL_COST_ACCOUNTING.json
summary/NEXT_STAGE_RECOMMENDATION_zh.md
manifests/F2_FREEZE.json
manifests/MODEL_AND_ENVIRONMENT_IDENTITY.json
manifests/DATA_AND_IMAGE_HASHES.json
manifests/RUN_MATRIX.json
manifests/TRAINING_STREAM_HASHES.json
manifests/CHECKPOINT_INDEX.json
raw/PROBE/...
raw/DEV_EVAL/...
training/<run_id>/ROLLOUTS.jsonl
training/<run_id>/UPDATES.jsonl
features/PRETRAIN_FEATURES_B0_BJ_BTAB_BNUM_BGEO.jsonl
verification/INDEPENDENT_RECOUNT.json
verification/RELEASE_MANIFEST.json
```

完整raw、训练轨迹与checkpoint留服务器；汇报包可分raw包与轻量结果包，但必须提供逐文件hash和明确排除项。原始失败、零信号、不利结果均保留。

最终报告顺序：研究想回答什么 → 实际训练做了什么 → 起点是否真的不同 → 哪些动作产生怎样的绝对/相对响应 → 奖励/语义支持与两未来重复的限制 → 有无状态相关的选择空间 → 下一步独立预测检验的具体建议。

不得只交工程PASS、loss曲线、宏平均或一张挑选过的热图。所有5起点×3动作×2未来重复必须可追溯。

---

## 17. 允许的结论与停止规则

| 实际结果 | 允许结论 | 不允许的替代解释 |
|---|---|---|
| 技术/数据/身份失败 | 本轮技术不完整，列明受影响路径 | 用缺失样本补0，或把它写成算法无效 |
| 准备归一化优势差很小、起点近乎相同 | 当前配方没有产生足够状态多样性 | 5个文件名等于5种独立能力状态 |
| 奖励对比稀疏、32步响应区间宽 | 本采样通道/剂量/精度下证据不足 | 语义信息普遍无用；自动延长到64步 |
| 一个固定动作跨状态占优 | 当前动作集下自适应选择空间有限 | 必须加新动作/改效用直到排名翻转 |
| 状态间排序不同，未来重复不稳定 | 存在开发线索，但训练噪声大 | 只报告有利seed、直接训练安全控制器 |
| 稳定的状态×动作差异 | 值得设计新准备来源上的事前预测验证 | 已证明BGeo有效或已取得部署决策收益 |
| S低支持或主要由单坐标错/arity决定 | 当前几何信息增量不足或难以测量 | missing补0；把arity/错误率当形状优势 |
| BNum足够而形状无增量 | 常规语义/数值信息可能更适合 | 为保住复杂术语继续添加未注册特征 |

全部34条科学路径及预定评价完成后，本阶段自然结束。**这不是GPU运行时间阈值，而是实验合同完成。** 不自动启动MM-LOCK、在线控制、奖励搜索、硬题重构或新的符号实验。

---

## 18. Codex1 接收时的简短执行指令

> 在现有已核验的MM-CORE工程上实现并执行本MM-DEV F2合同。沿用指定Qwen3.5-9B快照与原生视觉路径，取消所有GPU时长停止逻辑，最多5张已授权Pro6000并行。首先用所给数值源和9份输入流固化新数据；完成有限的新GRPO实现/恢复验收后，不再停留在审计阶段，直接完成4条准备、5起点probe、30条响应、35模型终点评价及完整开发分析。科学参数以本文件及JSON为准，不能按结果调题、调reward、调H、换模型或挑checkpoint。遇到技术阻塞具体报告；成功则完成整个矩阵再一次性交付，不自动进入下一研究阶段。

---

## 19. 来源与本轮新增决策的界限

**原始实验事实：**用户ZIP，具体出处与SHA见 `evidence/SOURCE_MAP.json`、`EVIDENCE_REVIEW_zh.md`。

**本次独立复算：**`reference/review_evidence.py` 及 `validation/INDEPENDENT_AUDIT_RECOUNT.json`；仅原始文本/真值/日志级别。

**本次新增设计：**β、B/G、KL、raw训练采样、数据池/去重、效用数值、额外工程分支、seed、分析与资源合同。它们不是已经证明最优的值。

**外部核对（不覆盖本地快照或历史结果）：**Qwen官方模型卡确认9B视觉语言模型及其32层混合布局；DeepSeekMath原文用于核对GRPO式目标的来源。本文具体实现仍按展开式和本地固定版本，不按公开页面的安装建议升级环境。

```text
https://huggingface.co/Qwen/Qwen3.5-9B
https://arxiv.org/html/2402.03300
```

**最终提醒：这轮真正要得到的是可比较的多模态训练响应，而不是一个新的几何名字、一个重复的格式PASS，或提前宣布一个数据agent已经成立。**
