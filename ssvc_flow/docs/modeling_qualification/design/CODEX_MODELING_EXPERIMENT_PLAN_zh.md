# SSVC 建模资格实验：面向 Codex 的实施规范

**版本：modeling-qualification-v1 · 2026-09-15**  
**本轮目标：选择可测的语义表示、验证局部变化模型、确定需要保留的响应维数及适用范围。**  
**执行范围：独立 CPU 实现与实验；既有真实实验的只读结构审计。**

---

## 0. Codex 的任务与停止边界

完成本文件规定的 M0–M6：实现、运行、保存原始结果、进行模型比较，最后提交一份明确的建模决策。不能只交伪代码或计划。

本轮不运行新的 Qwen 推理/训练，不自动提交 Slurm，不实施在线 SSVC，不搜索最优奖励策略，不训练另一个大模型作为 judge。允许训练本文件定义的微型 CPU 概率模型。这是研究模型表示所需的实验，不是新一轮 9B 训练。

**不要等待真实服务器数据才能开始。** M0–M4 是完全自包含的 CPU 建模实验；缺少真实张量只阻塞 M5 对应审计项。也不要因模型预测不佳而无限调参、换数据或扩大实验；预测失败同样需要完整报告。

### 0.1 本轮必须回答的六个问题

1. X/S/W/I 的哪些坐标适合存储、解释和拟合？边界/稀有事件如何处理？
2. 总体平均、六群体平均与保留题目身份的表示，分别遗漏什么？
3. 预测参数更新后的语义概率变化，至少需要多少独立方向？这个数怎样从数据估计？
4. 压缩误差、有限样本误差和局部非线性误差，能否分开测量？
5. 不重新生成全部评估输出时，实际更新摘要可以支持多长的局部跟踪？
6. 当前真实 Qwen 产物是否具备将来验证上述建模所需的信息？

### 0.2 最终交付

- 可运行代码、增量测试、配置及命令记录。
- `MODELING_DECISION_zh.md`：建议保留的表示与维数范围、证据、失败条件、下一步最小数据需求。
- `MATH_AUDIT_zh.md`：公式、适用前提、独立数值核验。
- `RESULTS_zh.md`：所有锁定基线及分组结果，不只展示最好的模型。
- `REAL_EVIDENCE_READINESS_zh.md`：真实产物有哪些、缺哪些；没有远程原件时明确注明。
- `machine_readable_readiness.json`：分开报告代码通过、建模通过、真实模型验证三个状态。

本轮应停在“建模资格结论”处。GPU 后续只提供数据采集契约，不提供自动启动动作。

---

## 1. 仓库起点与现有工作的保护

### 1.1 本次实际核读到的仓库状态

仓库：`Lhan-chding/dissertation`。工作子目录：`ssvc_flow/`。

2026-09-15 核读时：

| 分支 | 提交 |
|---|---|
| `main` | `653e3b96a9847f92e32ee9b262e11f2c16cb2033` |
| `codex/ssvc-flow-ntu` | `7e405faa241e0085f701cdc858ba423ea76ba5a8` |
| `codex/ssvc-mechanism-followup-20260914` | `97b7fd646046cfd58c45e0083f0e8fe6f37d6f1e` |

**后续分支已更新，不能继续使用“新功能尚未提交”的旧判断。** 该分支已有 `followup_*` 实现、CPU 验收记录和 S1 执行记录。最新文档快照记载五个原 bank 已完成，composite 与部分历史 R2 补测当时仍在执行；这是仓库文档快照，不是本文件对服务器实时状态的确认。[G1–G3]

### 1.2 工作方式

- 开始前执行只读 `git status`、分支/HEAD 检查，查看是否存在 `AGENTS.md` 及仓库规范。
- 建议从实际最新的 followup 分支建立**独立 worktree**，分支名 `codex/ssvc-modeling-qualification-20260915`。
- 若分支已存在，检查其变更，不覆盖、不 hard reset。
- 不在正在被服务器作业使用的源码目录修改文件，不改变旧实验的源代码指纹和配置。
- 本次不暂停、取消、续接或扩容任何已运行的作业。旧 S1/S2 授权与本 CPU 建模阶段分开。
- 建模输出写到新的忽略目录 `ssvc_flow/runs/modeling_qualification/<run_id>/`。
- 本文件及 `protocol.json` 是新的研究规范；不改写历史 `next_stage.yaml`、followup 配置或旧结果。

### 1.3 复用位置

本次核读/历史代码中已存在：

- `src/grpo_update.py`：联合总体方差标准化、PPO loss。
- `src/optimizer_fork.py`：模型/Adam/RNG 状态与哈希。
- `src/fork_gradients.py`、`src/r3_updates.py`：同起点真实优化器分支与向量比较。
- `src/followup_protocol.py`、`src/followup_updates.py`、`src/followup_statistics.py`：后续奖励、恢复、统计；执行前核对当前接口。
- `src/generate_worlds.py`、`src/constraint_solver.py`、`src/verifiers.py`：受控世界生成、唯一修复与执行器。

**复用数学定义，不绕过旧生产门禁。** 微型模型使用独立适配器与命名空间；不得用 fake adapter 标注为 Qwen 实验。旧源码不应为了适应新 toy 而被大规模重构。

---

## 2. 来源、研究假设与新增设计必须分开

| 内容 | 身份 |
|---|---|
| X/S/W/I、联合归一化、实际 Adam 分支 | 原研究计划与已执行协议 |
| 每题身份、(v,q,s)、局部响应与检测间误差界 | 上一版 `MODELING_AND_OBSERVATION_zh.md` 的设计基础 |
| Helmert 线性坐标、具体 toy、模型比较、数据划分、预算与门槛 | **本轮新增资格实验设计** |
| Fisher 重参数化、SVD 秩界、Taylor 界 | 标准数学工具，不作为已经确立的新颖性 |
| “低维响应足够”“能够廉价跟踪” | 待检验假设，不是运行必须达到的结果 |

本轮最重要的修订是：**不预先断言 (v,q,s) 最适合回归，也不预先指定 3/18 维就是动态维数。** 保留它作为解释坐标；使用无边界除法的线性概率坐标作为默认回归基线，并通过实验决定。

---

## 3. 测量对象与维度契约

### 3.1 固定输出事件

对于题目 i 和固定解码/接口，模型输出 a：

- X：输出世界完全等于真值。
- S：合法、世界错误，确定性执行答案正确。
- W：合法、确定性执行答案错误。
- I：不合法或不完整。

四类互斥且穷尽。本轮没有“未判定”第五类；**执行故障/记录损坏不是 I**，应独立报错。没有真实标签时标为不可分类，不推断类别。

`p(i,theta)` 是模型在题目 i 上的四事件分布，不是神经元状态或内部感知/推理正误的直接测量。事件顺序固定为 `[X,S,W,I]`。

### 3.2 五种维度不要混用

1. 每题事件自由度：3。
2. 固定探针数 N：覆盖量。
3. 语义群体数 G：本 toy 为 3 家族 × 2 接口 = 6。
4. 参数更新空间的局部秩 k：校准更新真正张成多少方向。
5. 保留响应维数 r：为预测语义变化保留多少方向，`r <= k`。

历史长度、置信区间、熵、qX 等不能随意计入“独立事件维数”。3 是完整四事件单纯形的自由度上限；若某类别结构上不可能出现，局部可变自由度可能更低。有限采样计数为零不能据此减少结构维数。`3G=18` 是六个完整群体分布的自由度上限，不是模型内部维数。

### 3.3 三层表示

**存储层：** 每个 `base_scene_id/prompt_id/interface/checkpoint` 的 `[nX,nS,nW,nI]`、n、固定权重、来源与采样种子。toy oracle 概率另外存，不混入可用观测。

**解释层：**

\[
v=1-p_I,\quad q=\frac{p_X}{v},\quad s=\frac{p_S}{p_S+p_W}.
\]

\[
p_X=vq,\ p_S=v(1-q)s,\ p_W=v(1-q)(1-s),\ p_I=1-v.
\]

`v=0` 时 q 未定义；`pS+pW=0` 时 s 未定义。JSON 使用 `null` 并带状态，不填 0。只有 nV=0/nB=0 而 oracle 概率非零时，标为 `UNOBSERVED_CONDITION`，不宣称真实概率为零。

**默认回归层：** 使用固定正交对比矩阵

\[
H=\begin{pmatrix}
1/\sqrt2&1/\sqrt6&1/\sqrt{12}\\
-1/\sqrt2&1/\sqrt6&1/\sqrt{12}\\
0&-2/\sqrt6&1/\sqrt{12}\\
0&0&-3/\sqrt{12}
\end{pmatrix},\quad H^TH=I,\ H^T\mathbf1=0.
\]

对列向量 p，`y=H^T p`，`p=1/4*1 + H y`。按固定 prompt 顺序拼接，72 个探针对应 `y in R^216`。

它只是四维概率的无冗余线性表达，保留欧氏距离。它不自带语义解释，也不提供新信息。

比较表示：原始四概率、Helmert 三坐标、解释坐标 `(v,q,s)`。前两者的等价结果用作实现自检；条件坐标的动态回归只在整个校准/评价窗口对应条件概率均大于预设 `0.05` 的 oracle 诊断子集比较，并报告覆盖率。这种筛选不是主结果；全样本结果仍以原概率为准。

### 3.4 指标聚合

固定每题权重，先计算各题分布再聚合。群体 qX 使用：

\[
q_{X,g}=\frac{\sum_iw_i p_X(i)}{\sum_iw_i v(i)}.
\]

不是逐题 qX 的简单平均。三个家族及两个接口主结果等权；`base_scene_id` 将两个接口配对。另保存逐题误差，防止群体平均掩盖问题。

### 3.5 预测合法性

局部线性预测可能暂时越出 simplex。必须保存**原始预测**、越界比例及负概率幅度。可同时给欧氏 simplex 投影后的结果，但不能只保留投影后结果，也不能将归一化/裁剪后的误差用于证明原 Taylor 模型准确。

---

## 4. 待验证的数学对象

### 4.1 似然、信息量与边界

对固定策略上的 n 次独立评估，`nV=nX+nS+nW`、`nB=nS+nW`：

\[
L\propto v^{n_V}(1-v)^{n_I}q^{n_X}(1-q)^{n_B}s^{n_S}(1-s)^{n_W}.
\]

内部点的 Fisher 信息：

\[
I(v,q,s)=\mathrm{diag}\left(\frac1{v(1-v)},\frac{v}{q(1-q)},\frac{v(1-q)}{s(1-s)}\right).
\]

核验 `D^T diag(1/p)D=I`。不要把坐标变化本身当成统计效率提高：相同无约束 MLE 映回 p 后必须等于频率估计。

区间主实现：每类比例的固定样本 Clopper–Pearson；条件 q/s 使用各自 nV/nB。当分母为零，点估计 null、可能集合 `[0,1]`。若声称同一面板多坐标同时覆盖，用显式 Bonferroni 分配；单指标区间默认只称点态。禁止把不同 checkpoint 的样本合并为固定策略样本。

### 4.2 局部响应与压缩

\[
y(\theta+d)=y(\theta)+J_\theta d+\mathcal R(d).
\]

令校准实际更新张成的正交基为 Q，`d=Qa+d_perp`。响应矩阵：

\[
R=J_\theta Q.
\]

固定 theta、输出表与更新子空间时，精确线性编码/解码维数至少为 `rank(R)`；最优秩 r 截断谱误差 `sigma_(r+1)(R)`。测试该恒等式，不宣称是新的内在维数定理。

**需增加的一项盲区测试：**校准只覆盖 Q，不能据此断言 `J(I-QQ^T)=0`。只有更新的正交残差范数，不足以判断其语义影响方向。构造两个投影和残差范数相同、真实响应相反的更新，验证任何只看该摘要的预测器都无法同时预测正确。

### 4.3 误差分解必须有 oracle 对照

在 toy 精确概率已知的单步更新中，分别计算：

1. `Jd`：全方向一阶 oracle；
2. `JU U^T d`：低维一阶 oracle；
3. `Bhat U^T d`：从有限校准数据拟合的低维模型；
4. `Delta_y_true`：更新后精确概率差。

由此分别观测：方向遗漏、拟合/采样误差、有限步非线性。范数各项一般不会直接相加为总误差；保存向量残差与三角界，不能把误差范数强行相减作“贡献比例”。

### 4.4 检测间模型本轮仅作离线适用范围验证

在校准点 tau 固定 U，实际更新 `d_t` 已发生，预测：

\[
\hat f_t=\hat f_\tau+\hat b_\tau^T U^T D_t,\quad D_t=\sum_{k=\tau}^{t-1}d_k.
\]

假设起点误差 e、子空间系数误差 epsilon、遗漏梯度界 kappa、梯度 Lipschitz 常数 L，则：

\[
E_{net}(t)=e+\epsilon\|U^TD_t\|+\kappa\|(I-UU^T)D_t\|+\tfrac L2\|D_t\|^2.
\]

累计路径版本：

\[
E_{path}(t)=e+\epsilon\sum\|U^Td_k\|+\kappa\sum\|(I-UU^T)d_k\|+\tfrac L2(\sum\|d_k\|)^2.
\]

**对上一版的细化：**每一步保存/累计净位移并计算 `E_net(t)`，同样能够检查中途变化；不是只有路径长度才能发现中途退化。只看最后端点才会遗漏。路径界通常更保守，但当没有保留完整正交位移向量时更方便计算。本轮要比较二者，不能人为让路径方案占优。

该公式是已发生更新的概率跟踪，不是无需试算就能预测未来 Adam 更新。固定参数不同 Adam 的情况，对“未来操作预测”很重要；对于已知实际 d 的跟踪，动量的影响已进入 d，不能重复声称加入 Adam 摘要必然再提高准确率。

---

## 5. 实验总览与执行顺序

| 阶段 | 内容 | 资源 | 主要产物 |
|---|---|---|---|
| M0 | 代码/协议快照、维度/统计契约、数值自检 | CPU，标准小矩阵 | 审计与单元测试 |
| M1 | 6 类具有解析参考的可识别性/遗漏反例 | CPU，小模型 | 反例与理论校验 |
| M2 | 共享非线性有限动作模型，生成真实 CPU RL 轨迹和局部分支 | CPU | 原始 p、更新、计数、身份 |
| M3 | 同预算表示/方向基/维数比较 | CPU | 独立测试误差和选择结果 |
| M4 | 固定观测间隔下的轨迹跟踪及误差分解 | CPU，仅离线回放 | 适用半径/时间范围、成本 |
| M5 | 现有 Qwen/S1/R4 产物只读适配与可用性审计 | CPU，可部分缺文件 | 真实数据准备状态 |
| M6 | 建模决策与最小后续实验需求 | 文档 | `MODELING_DECISION_zh.md` |

M0/M1 不依赖 M2；M2 主实验前先 smoke。M3/M4 复用同一份冻结轨迹，不为每个表示重训模型。M5 可独立进行，不得因缺少数据阻塞 CPU 部分。

---

## 6. M0：数学、统计与工程契约

### 6.1 必须通过的数值测试

- 4 类概率和为 1；全部边界点与内部点的嵌套坐标处理正确。
- Helmert 往返、正交性、距离保持；类别列和权重顺序错误必须失败。
- 四类 MLE 与嵌套 MLE 映回等价；零分母不制造信息。
- Fisher 恒等式及有限差分 Jacobian，FP64 内部点误差小于 `1e-9`。
- 全合法组 lambda 平移不改变优势；无 X 混合组的尺度饱和。
- epsilon 严格为 `sqrt(population_variance + epsilon^2)`，不是 `std + epsilon`。
- 同一 theta 的参数梯度/Adam 状态恢复；零张量梯度与 None 的行为分开。
- SVD 不能将复制列计成额外维数；固定 r 无关的真秩容差。
- 参数更新相同范数但方向不同，必须在向量几何中区分。
- 单纯形投影前后的指标分别存储。
- 本文件 §4.4 的两个条件界在具有已知全局 L 的模型中成立。
- 偷看 oracle、未来 checkpoint、test 标签的特征会被契约拒绝。

### 6.2 随机性分离

分别使用本地 RNG：数据、模型初始化、训练题目、训练动作、分支题目/动作、有限评估采样、随机方向、bootstrap。所有命名空间记录到 manifest。绘图和评估不能消耗训练 RNG。

### 6.3 命名与安全

`execution_kind` 区分 `ANALYTIC_FIXTURE`、`REAL_CPU_TOY`、`READONLY_REAL_METADATA`。不能使用 `REAL_CUDA`。不 import 会加载 VLM 的入口；默认空 `CUDA_VISIBLE_DEVICES`。创建新 run 目录，不覆盖旧成功/失败记录。

---

## 7. M1：必须具备的解析反例与阳性对照

这些是资格测试，不是“真实大模型自发出现”的发现。

### W1：同样总体云、不同题目移动

构造两个题目的分布互换，总体均值不变。验证 pooled 表示误差为零而逐题/分组变化非零。使用原始点身份对齐，不用最优传输重新匹配题目。

### W2：同样当前 p、不同响应

两个可微策略在 theta=0 均为四类均匀分布，但至少一个 logit 的参数系数符号相反。施加相同 d，精确计算响应。证明只看当前 p 不能决定下一步；不从这个反例推断真实模型必然需要某个固定维数。

### W3：同参数、不同 Adam 历史

同 theta、相同当前梯度、不同合法一阶/二阶历史，执行一步 AdamW。比较未来 d 及精确 p。随后将实际 d 作为输入，检查一阶响应预测恢复了哪些信息。

### W4：相同投影和残差范数、不同语义后果

令 logits 为 `[theta1+theta2,0,0,0]`，U=e1；比较 d+=(0,h)、d-=(0,-h)，`h in {0.01,0.1,0.5}`。摘要 `(U^T d,||d_perp||)` 完全相同，pX 响应符号相反。报告两种真响应差的一半作为这组相同观测下的最坏情形点预测误差下界。

### W5：可控局部秩与低能量重要方向

线性响应 R 使用已知奇异值。第一组设精确 rank=4，冗余校准列人为复制；第二组为第 5 个小奇异值方向只影响一个预设群体的 pX。

比较按 95% 总能量选维与按最坏群体误差选维。方向/群体/幅度在测试前固定。允许 95% 在部分设置表现足够好；不得仅选失败例子作为整个方法结果。

### W6：中途退化与表示失效

以 `f(theta)=softmax(A theta+b)` 的事件概率为观测，构造一段先走出、再返回的参数路径，以及先在 U 内、后转入 U 外的路径。

对于选定事件（或系数绝对值不超过 1 的线性事件对比），可使用保守全局界 `L=3||A||_2^2`：单类 softmax Hessian 的外积项范数不超过 2，协方差项范数不超过 1；按概率加权后得到该界。Codex 在 `MATH_AUDIT` 写出推导。

同时比较逐步净位移界、路径界、只看端点、遗漏方向设为零的错误模型。错误模型应在反例中被击穿；其失败是软件/理论自检。

---

## 8. M2：共享非线性有限动作 CPU 系统

### 8.1 为什么选择这个系统

需要一个能精确得到语义概率、又存在共享参数和真实优化器历史的系统。只用四个自由 logit 会把跨题耦合和类内差异人为消除；直接上 9B 则无法低成本得到密集真值轨迹。

因此使用“题目—候选输出”打分网络。动作数有限，可以精确求和得到 X/S/W/I。它是概率建模试验台，不代表视觉能力或开放生成能力。

### 8.2 数据与任务

优先复用 `generate_worlds.py` 的实际可调用生成 API，在新目录、`render=False` 下生成新 toy 数据；Codex 先检查函数名，不能假设本文件给出了现有命令。

- 数据种子 `20260915`。
- train：72 个 base scenes，沿原规则各自固定一种接口。
- probe：36 个 base scenes，每个构造两个接口，共 72 prompts。
- 保留三家族：`duplicate_encoding/cross_series/trend`。
- 保留运算：`sum4/difference_pairs/range4`；图表类型只作为生成平衡元信息，本轮不渲染图像。
- 全部数据验证唯一修复、真值去重、train/probe 无 base_scene/truth 重叠。
- 不读取真实研究的 sealed confirm split。

toy 接口重命名为 `SYMBOLIC_PROXY` 与 `EVIDENCE_PROXY`。后者提供额外数值证据向量，用于模拟信息访问差异，**不声称它是图像编码器**。

### 8.3 固定有限动作集

每题 16 个动作：14 个不同合法四整数世界、2 个显式非法符号。合法候选包括：真值、原错误观察；若答案纤维允许，再纳入至多两个答案相同但世界不同的候选，其余以确定性枚举/采样补齐。

- 由真值/执行器构建候选集属于该有限试验台的设计；不能称为自然模型支持分布。
- 动作位置固定打乱；打分网络不能读取候选索引或 category。
- 改变候选顺序，精确 p 与单步梯度必须不变（浮点容差内）。
- 不同合法动作即使属于同一事件，也具有不同特征/梯度，不能用一个事件 logit 取代。
- S 候选用确定性执行器验算；若某题在规定数值域内没有其他同答案世界，记录 `STRUCTURALLY_EMPTY_S`，不删题、不重采题、不将 W 改标 S。结构上存在但构造程序失败，属于数据构造错误。
- 数值域、原错误修复规则沿实际 generator；特征按 99/198 缩放，不自行缩小数据域。

### 8.4 明确模型特征

题目特征 37 维：

1. observed_world/99：4；
2. 家族 one-hot：3；
3. known_value mask：4，known_value/99：4；
4. 六种无序坐标对的边存在 mask：6，pair_sum/198：6；
5. 接口 one-hot：2；
6. evidence_world/99：4；符号接口填 0，额外 availability mask：1；
7. operation one-hot：3。

候选特征 7 维：合法世界值/99（4）与 `[valid,invalid_a,invalid_b]` one-hot（3）；非法动作数值填 0。

合计 44 维。网络：`Linear(44,16) -> tanh -> Linear(16,1)`，总参数 **737**。对 16 个候选分数 softmax。无 dropout、无 BatchNorm、无位置编码。EVIDENCE_PROXY 的 evidence_world 使用真值作为受控额外观测；SYMBOLIC_PROXY 不得接收该字段。

不能输入 truth_world 的其他副本、changed_index、是否满足约束、是否等于真值、类别或最终是否正确等答案标签。

同一参数网络服务全部题目和候选，形成跨题耦合；精确事件概率为同类动作 softmax 概率之和。

### 8.5 训练设置

- 所有轨迹共用初始化 seed=7001、同一初始参数：避免不同初始化的参数坐标对齐问题。本轮研究训练随机性，不声称覆盖模型初始化随机性。
- float64，CPU，threads=1。
- AdamW：lr=0.01，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，global grad clip=1。
- 每步 B=4、每题 K=8，组内总体方差加 epsilon²，epsilon=1e-4。
- 64 steps；每个 rollout batch 只更新一次；无 scheduler/KL/entropy bonus。
- 两条固定训练轨迹规则：`X_BASE=(2,0,0,0)`、`X_VALID=(3,1,1,0)`。
- finite action 是一个决策单元，toy 的 loss 分母为 `B*K`（Lnorm=1）。**不是历史 Qwen 的 64-token loss**，不声称数值训练轨迹完全一致。
- 所有动作先采完，再计算优势和梯度；优势 detach。
- 零优势组保留，显式零梯度照常执行 Adam；不筛选组、不补采到出现 X。

不同训练 seed 改变训练题序和动作采样；同 seed 两臂使用相同题序，各自从当前策略采样。真实例子不能被挑到“必须出现真值下降”。

### 8.6 轨迹划分

默认 30 个训练随机种子、每个两臂，共 60 条轨迹：

| 用途 | seeds |
|---|---|
| development-fit | 101–106（6） |
| dimension/model selection | 201–204（4） |
| empirical interval calibration | 251–260（10） |
| locked evaluation | 301–310（10） |

所有同 seed 的臂、分支和时间点属于同一顶层统计簇。不能将其拆到不同全局划分。

每条轨迹保存 0–64 的 exact probe p 作为**离线 oracle 标签**、每步实际 d、完整 toy Adam/checkpoint、训练组构成。预测算法只通过受限观测 API 访问允许测量的节点，不能直接读取密集 oracle 文件。

### 8.7 同起点局部分支设计

每条轨迹在 t=8,24,40 保存锚点。每个锚点固定 14 个独立训练 bank：

- bank 0–7：local fit；
- bank 8–9：local diagnostic（不用于 locked-test 的模型调参）；
- bank 10–13：response evaluation。

每 bank B4/K8；同 bank 的三种候选共用相同动作样本：

- joint lambda=0；
- joint lambda=1；
- `no_x_off`：若整组无 X，在**联合标准化前**设该组辅助 lambda=0。

每候选均从相同参数、Adam、RNG 起点产生真实 CPU 更新。候选结束恢复原锚点，主轨迹不被分支污染。当前选这些操作只为提供不同更新，不以验证 No-X 方法有效为本轮目标。

局部拟合可使用目标锚点的 8 个 fit bank，即 24 条校准响应；这是**付费的局部校准**，不是零样本预测。test bank 的输出/梯度方向不能提前进入 Q 或 B。全局 r、ridge 等只在 selection seeds 上选择。

主要目标是原始更新相对锚点的 `Delta_y`；另比较同 bank 辅助更新相对主更新的差异。不要把两种量混成一张“流量表”。

### 8.8 精确与有限采样视图

对同一冻结概率轨迹生成评估计数，n∈{16,64,256}，每设置 5 个独立测量噪声 seed；无需重跑训练。

- 同一锚点原点计数在相关差值中共用，保留差值间相关性。
- 参数和完整推理状态精确相同的候选可使用 alias；复用输出不能计作新增独立样本。
- toy exact 概率只供 oracle 基线/最终评分；拟合有限样本模型只能访问计数。
- 原始概率、采样计数、拟合输入分别存储并有不同对象类型。
- MLE 主比较不加伪计数；需要平滑的可选 baseline 独立命名、固定先验并报告敏感性。

## 9. M3：表示、方向基与维数选择

### 9.1 先固定信息访问等级

| 等级 | 可以读取 | 不可以读取 |
|---|---|---|
| `COUNTS_ONLY` | 已付费节点计数、题目元信息、操作名 | 参数方向、精确概率/J |
| `UPDATE_AWARE` | 前项＋已发生实际 d、训练已有日志 | 未观察节点真概率/J |
| `EXACT_J_ORACLE` | 精确 J、精确 p、实际 d | 不作为可部署低成本方法 |

本轮重点是 `UPDATE_AWARE`。它在优化器完成后、额外行为评估前输出预测，是**事后一步跟踪**；不能叫“未经执行就预测未来训练结果”。局部分支中可在提交前拿到 d，但分支计算成本要计入。

### 9.2 校准更新空间 Q

把 local fit 的实际更新按列组成 `D in R^(P x M)`，这里 P=737、M≤24（alias 后可能更少）。不减均值：一个共同的动量方向也可能有语义作用；强行中心化会把它删除。

对 D 作 thin SVD，截断仅依据固定数值秩容差 `max(1e-14,1e-10*sigma_max)`，得到正交基 Q。保存全部奇异值、秩、alias 数和条件数。

- 该 k 是**观测到的更新张成维数**，不是全模型梯度秩。
- 列重复、缩放共线、近零更新不增加可靠新方向。
- Q 不能使用 response-evaluation bank 的更新；即使更新 d 比输出标签便宜，也会造成使用测试设计的泄漏。
- 测试时单独报告 `||d_perp||/||d||`，分为 `<=0.05`、`0.05–0.2`、`>0.2`；阈值预设为诊断，不用于删除难样本。

### 9.3 每个锚点的响应拟合

令 `A = [Q^T d_j]^T`，`Y=[y(theta+d_j)-y(theta)]^T`。约束零截距：`d=0 -> Delta_y=0`。

采用：

\[
\hat B=Y^TA(A^TA+\gamma I)^{-1},\quad
\gamma=\alpha\,\sigma_{max}(A)^2.
\]

`alpha in {0,1e-8,1e-5,1e-2}`，alpha=0 使用明确的伪逆容差，不求不稳定逆矩阵。超参数只在 selection seeds 选，之后冻结。全局选择冻结的是 `(r_cap,alpha)` 规则，而不是根据 locked-test 响应逐锚点重新选 r。

有限采样时起点 yhat 的噪声进入所有 Y 行；不能把这些行视为完全独立。显式保存同一个 `anchor_measurement_id`，用重复测量模拟和 seed/block bootstrap 反映相关性。

### 9.4 必须比较的模型

| ID | 模型 | 目的 |
|---|---|---|
| B0 | 保持上次测量不变（persistence） | 小变化场景的必要强基线 |
| B1 | 普通训练日志＋当前概率概要的 ridge 预测器 | 检查廉价常规信息是否已经够用 |
| B2 | 固定随机正交方向＋同一局部 ridge | 随机投影基线 |
| B3 | 校准更新 PCA 的前 r 方向＋局部 ridge | 仅保留参数移动能量 |
| B4 | 校准响应拟合 Bhat 的右奇异方向＋局部 ridge | 经典监督式低秩响应基线 |
| B5 | B4 方向下按最坏语义群体误差选择 r | 检验风险相关选维是否有增量 |
| B6 | 使用全部观测更新子空间 Q 的局部 ridge | 比较压缩究竟损失什么 |
| O0 | 精确全 Jd | 一阶模型上限；仍有有限步曲率误差 |
| O1 | 精确 JQ 的秩 r SVD 截断 | 区分理想低秩结构与估计困难 |

B4/B5 与 active subspaces / reduced-rank regression 有直接联系，是**资格候选与已有方法基线**，不能称为原创算法。[R4]

B1 的固定输入：锚点六群体三维概率、实际更新范数、梯度裁剪前后范数、当前 optimizer step、四事件训练计数比例、全零优势组比例、无 X 混合组比例、操作类别。只使用过去和当前已获得的字段。全局 fit 使用 development-fit seeds，提供固定 ridge 网格；不可用未来概率误差作输入。可以追加“同样允许读取全部逐题锚点概率”的 B1-full 作为公平性诊断。

B2/B3/B4/B6 校准预算相同，局部响应数据相同。B5 是选择规则变化，不多拿校准样本。随机基种子在测试前固定。

### 9.5 两种方向空间基线

主比较 B2 的随机方向限制在 Q 内，确保与 B3/B4 相同可激励空间。另提供“全参数空间随机 sketch”成本诊断，使用其自己的投影/拟合误差，不能沿用正交投影的误差界。

全局比较标签为 `r_cap={0,1,2,4,8,16,FULL}`。每个锚点实际 `r=min(r_cap,k)`，FULL 对应 r=k；必须同时报告 cap 与实际 r，局部重复模型不计为独立算法。`k=0` 时只有 persistence/zero-response，明确 `NO_IDENTIFIED_UPDATE_SUBSPACE`。

参数坐标固定为当前有序参数名、shape、dtype。不能跨不同初始化/不同 LoRA gauge 直接拼方向基；本 toy 通过共用初始化规避该问题。真实模型未来需单独审计参数坐标一致性。

### 9.6 维数选择规则

B4-energy 作为对照：取拟合响应谱累计平方能量≥95%的最小 r。

B5 主规则：在 selection seeds 的独立 bank 响应上，选满足以下设计目标的最小 r（同一预算与 ridge 配置）：

- 各六群体 `Delta pX` 的绝对预测误差 95 分位数≤0.01；
- 各群体 `Delta v` 的对应误差≤0.01；
- 逐题四事件概率的平均 absolute error≤0.02；
- 在非零响应的总体上，响应 NRMSE≤0.75；
- 投影前不合法概率预测比例≤5%。

这些数值是本轮**资格目标**，不是统计安全阈值或已证实性能。若没有任何 r 达标，输出 `NO_ACCEPTABLE_R`，不得偷偷增加 r/样本量或降低门槛。继续报告完整误差—维数—成本曲线，并分析瓶颈。

NRMSE 使用 `sqrt(sum||error||^2 / sum||true_delta||^2)`。分母低于预设数值阈值时输出 `UNINFORMATIVE_RESPONSE_SCALE`，不要加任意常数让结果显得良好。

冻结选择的 r 规则必须同时报告 exact 与 n=16/64/256；全网格按 §9.7 分阶段执行。维度随噪声变大而失效，是有效结果。

### 9.7 额外但有界的消融

仅三项，复用同一轨迹：

1. `pooled / six-group / per-prompt` 输出表示，检查谁掩盖哪类变化。
2. `actual d / norm-only / SGD proxy(-lr*g)` 输入；第三项仅诊断 Adam 状态作用，不能冒充真实更新。
3. local fit bank 数 `1 / 2 / 4 / 8`，检验识别维数与校准样本量关系。只取固定前缀 bank，不增加分支计算；低预算不够激励时输出不可识别。

不做完整笛卡尔组合搜索。主表示固定 Helmert：先在 exact 和 n=64（前三个测量噪声 seed）上比较模型/r；在 selection split 冻结选择后，再对选中配置与 B0/B6 做 n=16/64/256、5个噪声 seed 的稳健性检查。fit bank 数和表示消融分别在 n=64 上进行。不得对每个 test 噪声实现重新挑 r。

### 9.8 评价指标与统计单位

同时保存两种评分对象：

- 概率水平预测：`p_hat_anchor + Delta_p_hat` 对比候选 exact p，包含起点测量误差。
- 响应预测：`Delta_p_hat` 对比 exact `p_candidate-p_anchor`，用于分析响应模型本身。

所有方法的可用起点观测相同；只有明确标注的 oracle 版本读取 exact 起点。不能用水平误差和响应误差混算增益。

主要：四概率 RMSE/MAE、逐群体 pX/v 误差、逐题尾部误差、响应 NRMSE、无效 simplex 预测率。

辅助：当真实 `|Delta pX| >= delta_min` 时的方向识别，`delta_min={0.002,0.005,0.01}` 均报告；每个门槛同时给合格分支数，不按结果选择有利门槛。近零响应单独报告。

方法比较使用相同 seed/anchor/bank/测量噪声的配对结果。主 uncertainty 以**训练 RNG seed**为顶层簇，保留该 seed 的两臂、所有 anchor/bank；评估噪声重复先汇总为 seed 内指标，不把 5 次噪声重复或 72 道题当成 5/72 个训练重复。

locked test 有 10 个 seed，报告每 seed 数值、配对均值、中位数及探索性 cluster bootstrap（5000次）。这是固定数据集和固定初始化下的轨迹随机性结果，不是跨初始化/数据集总体保证。

---

## 10. M4：固定观测间隔下的模型适用范围

**本阶段不实现自适应检测器或控制器。** 只验证建模能否支持以后设计检测，使用预先固定的窗口回放。

### 10.1 固定窗口

每个 anchor t=8,24,40，跟踪 H∈{1,2,4,8,16} 步。模型只能获得锚点付费测量与之后的实际 d/廉价日志，密集 exact p 由独立 evaluator 保管。

基 U、响应 Bhat 在窗口内固定。不得每一步用真实 J 重估，再声称是“低成本长窗口跟踪”。每步保存误差和是否超阈值，不能只比较最后一点。

### 10.2 三种遥测量

T1：仅每步更新范数。

T2：每步投影 x=U^Td、正交残差范数、累计路径长度。

T3：T2 加累计净位移 D 的精确范数/残差范数；在 toy 上允许保留完整 D，计入 O(P) 存储与计算。

比较拟合误差、方向遗漏、曲率与成本，不预设 T2 最优。

### 10.3 条件界的两种身份

- **解析资格系统：**使用已知 L、精确梯度残差；检查数学界。每个时刻必须覆盖对应 exact f；若不覆盖，属于实现或推导错误。
- **共享非线性 toy：**Hessian/J 可用于 oracle 诊断，但有限网格/有限路径上的最大曲率不能叫全局 L 上界。主输出使用 `EMPIRICAL_CALIBRATED_ENVELOPE`，从 interval-calibration seeds 固定残差分位数，再在 test 测覆盖、宽度与失效。

经验误差带按“完整 anchor 16步窗口 × 六群体”最大误差校准，再按 seed 成块报告覆盖。10 个校准 seed 的尾部信息有限，因此**不宣称 95% 分布无关保证**，也不输出 `SAFE`。本轮只使用 `WITHIN_MODEL_TOLERANCE / EXCEEDS_MODEL_TOLERANCE / UNKNOWN`。

### 10.4 三类噪声分开

1. 固定同一个 checkpoint 只改变评估 RNG：测量噪声。
2. 固定初始化、改变训练 RNG：训练路径随机性。
3. 固定路径和精确概率，只改变压缩基/维数：模型逼近误差。

报告它们各自实验和交互。不要把总残差中不能解释的部分统一命名为“扩散”。

### 10.5 能支持后续检测的输出

形成表：`(representation,r,n,fit_budget,H) -> exact_error,finite_sample_error,worst_group_error,cost`。

同时报告：

- `largest_tested_H_meeting_tolerance`：只能从冻结测试网格定义适用范围，不能再据 test 反向修改方法。
- `subspace_exit_failure_rate`：误差大时是否伴随离开 Q/U。
- `rare_event_failure_rate`：q/s 信息不足带来的失败比例。
- `curvature_failure_rate`：全 Jd 也失效的比例。

这张表才是以后共同设计维度与检测间隔的依据。本轮不能据此直接宣称已经节约在线训练成本。

### 10.6 必须核算校准的摊销条件

主结构实验使用 8 bank × 3候选=24个校准响应面板，而最大跟踪窗口只有16步。若每个面板都需要独立昂贵生成，这种配置在单条16步轨迹上**不会因为省去逐步评估而自动节约成本**。

必须计算：

\[
C_{model}=C_{calibration}+H C_{telemetry}+C_{refresh},\quad C_{dense}=H C_{panel}.
\]

报告实测 CPU 时间与抽象评估单位两种成本。toy 的多项采样不是 Qwen 生成，不能用微秒级采样时间证明 VLM 评估便宜。对真实模型成本只能列待测参数，不虚构秒数。

低预算 1/2/4 bank、利用已有历史校准、或更长可验证有效范围，才可能改变这项收支。复用历史数据仅在其预测准确性经独立检验时才成立，不能将其默认免费且有效。本轮输出每种配置的盈亏平衡窗口；若所需窗口大于已验证有效窗口，明确 `NOT_COST_FEASIBLE_AT_TESTED_HORIZON`。

---

## 11. M5：既有真实数据只读适配

### 11.1 先查新产物，不回到过时缺口

本次仓库文档已记录 followup S1 正在产生/完成新的非零处理响应。Codex 应读取实际已有完成记录，区分旧 R3 的无作用 bank 0/6 与新 S1 的指定 bank/composite，不能继续默认所有真实非零响应缺失。[G3]

对未完成结果只列状态与缺项；不调用训练、补采样或排队工具。不将“作业提交”“状态RUNNING”计为测量完成。

### 11.2 检查清单

- 父模型、LoRA 参数顺序、dtype、parser、生成配置、数据/场景/接口身份是否固定。
- raw output -> X/S/W/I 是否可重新计算；仅摘要可读时明确标注层级。
- 各候选是否同起点模型和 Adam；实际更新向量是否存在。
- 相同推理策略是否被 alias；是否误计独立样本。
- 原点及候选是否有相同 probe 面板、相同样本预算。
- pX/v/qX 能否正确配对；缺乏原点分布时只允许候选之间响应，不反推绝对起点变化。
- 跨多个锚点/训练 seed 的覆盖是否足以拆分拟合和验证。
- 仅有向量范数或哈希时，**不能计算夹角、J、响应子空间秩**。
- 仅有一个 warm 锚点、少数 bank 时，不拟合高维全局动态模型，也不声称跨 checkpoint 泛化。

### 11.3 统一真实数据表

每行至少：

```text
run_id, source_commit, origin_state_hash, candidate_state_hash,
origin_optimizer_hash, candidate_optimizer_hash,
parameter_layout_hash, inference_fingerprint,
seed, checkpoint_step, bank_id, candidate_id, intervention,
base_scene_id, prompt_id, family, interface, operation,
nX,nS,nW,nI,n, sample_ledger_hash,
update_vector_path, update_vector_hash, update_norm,
source_files, evidence_level, alias_of
```

缺失值 `null + reason`。不要生成占位随机向量。私有路径/原始权重不提交公开仓库。

如果原件已在本机，只执行只读导入与 CPU 分析；如果仅在服务器，本轮交付一个所需导出清单，不启动远程计算。该导出清单不是新 GPU 运行授权。

---

## 12. M6：怎样做出建模决策

报告必须按下面问题给出结论，而不是写“全部通过，准备训练”。

### 12.1 表示决定

- 原始四计数保留：是/否，理由。
- `(v,q,s)` 适合解释的范围和不可识别比例。
- Helmert/原概率与条件坐标在 finite-n 拟合中的差异。
- 哪一聚合层足以支持当前目标，哪一层会漏掉重要题目/群体变化。

### 12.2 维数决定

每个访问等级、观测预算和预测窗口分别给出最小可接受 r。

可能结论包括：

- 低维模型在指定局部范围可用；
- 低维结构存在，但有限样本无法稳定识别；
- 局部线性模型本身误差较大；
- 校准未覆盖重要方向；
- 全维/普通日志基线已足够，复杂选维没有增量；
- 目前证据不足以作选择。

这些都是允许的研究结果。工程正确性测试仍需通过。

### 12.3 向下一阶段的最小交接

只确定下一步需补的最小东西，例如：新增多少个不同 checkpoint 的已发生更新向量、多少个独立语义探针、哪个方向的响应验证。不得直接展开新多模型/多奖励大训练网格。

本轮不会得出“概率云恢复了内部机制”或“SSVC 已被证明安全”的结论。

---

## 13. 代码组织与函数契约

建议新增 `ssvc_flow/src/modeling_qualification/`，不覆盖旧 followup 模块。

| 文件 | 主要函数/职责 |
|---|---|
| `schema.py` | `EventCounts`, `ProbeIdentity`, `TrajectoryIdentity`, `MeasurementView`, source/type checks |
| `coordinates.py` | `to_nested`, `from_nested`, `to_helmert`, `from_helmert`, boundary masks |
| `likelihood.py` | 原始多项计数、条件/单类 CP 区间、Fisher 检查 |
| `witnesses.py` | W1–W6，解析真值与断言 |
| `toy_data.py` | 数值场景适配、候选集、44维特征、数据隔离 |
| `toy_policy.py` | 737参数共享网络、exact category probabilities |
| `toy_training.py` | 联合奖励训练、实际 Adam、状态恢复、局部分支 |
| `views.py` | `ObservedView` 与 `OracleView` 权限隔离 |
| `response.py` | Q、ridge、PCA/SVD/随机基、oracle JVP |
| `dimension.py` | 候选 r、dev 选择、无合适 r 的状态 |
| `tracking.py` | 固定窗口跟踪、净位移/路径界、经验误差带 |
| `metrics.py` | 同种子配对、最坏群体、方向阈值、成本 |
| `real_import.py` | 完成产物只读导入、缺项报告 |
| `report.py` / `cli.py` | 运行记录、图表数据与最终建模决定 |

函数至少有类型检查、非有限值检查、shape/identity 检查。共用哈希工具时明确 raw bytes hash 与 canonical object hash 的区别。

### 13.1 数组布局

```text
p_exact:       [checkpoint, probe, 4]
counts:        [noise_replica, checkpoint_or_candidate, probe, 4]
y:             [probe*3]
delta_theta:   [parameter]
D_fit:         [parameter, fit_candidate]
Q:             [parameter, k]
Bhat:          [probe*3, k]
U:             [parameter, r]
y_pred:        [candidate_or_step, probe*3]
```

参数名/shape 列表记录到 `parameter_layout.json`。数据有向量形状但没有身份绑定时拒绝拼接。

### 13.2 禁止信息泄漏

- predictor 只能接收 ObservedView；oracle 路径不出现在其对象中。
- 不以未来 exact p 选择 U、r、lambda、测试样本或停止时刻。
- 相同 anchor 的所有分支保持在同一全局 split。
- test local fit 是预先定义、计入成本的在线校准角色；test response bank 与其分开。
- 测试结果在 `selection.json`、模型超参数与 manifest 冻结后才揭盲。

### 13.3 恢复与故障

每步 toy 完整状态可保存；分支必须 finally 恢复参数/Adam/RNG。保留 failure 原件，成功结果 no-clobber。缺失数据与数值失败不能当成零响应。

本轮不复制整个 VLM 生产系统的所有门禁到 toy；保留必要的身份、恢复、数据隔离与成本记录即可。

---

## 14. 测试、预算及停止策略

### 14.1 配置文件

包内 `protocol.json` 是机器可读方案，需在代码中实施 schema 验证，不是“已有训练命令”。运行前生成冻结 resolved config，记录当前 commit 和源码哈希。软件版本读取本机已核验 CPU 环境，不为了符合历史版本字符串自动升级依赖。

### 14.2 默认 core 工作量

- 主轨迹：30 seeds × 2 arms × 64 steps = **3,840 CPU optimizer steps**。
- 主训练动作：3,840 × 4 × 8 = **122,880 sampled finite actions**。
- 局部锚点：30 × 2 × 3 = **180**。
- 局部候选：180 × 14 banks × 3 operations = **7,560 scratch CPU optimizer steps**。
- 分支基础采样：180 × 14 × 4 × 8 = **80,640 finite actions**，三个候选复用同 bank，不能重复计采样。
- 总 optimizer steps = **11,400**；总训练/分支动作 = **203,520**。
- counts 的 n=16/64/256 和 5次测量噪声来自 exact p 的多项采样，不增加神经模型训练。

这是 CPU 微型模型预算，**不是 9B 工作量**。exact 概率/JVP/拟合次数另记，不因为样本来自 toy 就计为免费。

### 14.3 先 smoke，再 core

smoke 使用 2 个 seed × 2 arms × 8 steps，anchor=4，6 banks、3 ops。通过正确性和耗时检查后才能 core。

单线程默认；先估计 core 墙钟、峰值内存与磁盘。默认自动运行上限：4小时估计墙钟、8GiB RAM、2GiB输出。超过时停止在 `RESOURCE_REVIEW_REQUIRED`，给测得吞吐和分阶段建议；不静默减少 test seed/改变配置。

可用逐 anchor 缓存/JVP、共享 exact 轨迹、批量评分来节约成本。不得为“加速”改奖励、省略对照或混用 test 信息。

### 14.4 仓库回归

执行新模块测试、实际 smoke、既有相关数学/恢复/followup 测试。相关回归通过后，在资源允许下执行既有 CPU 全套。记录哪些实际运行、哪些未运行。测试数从日志提取，不写预期通过数。

包内 `reference/` 仅是独立的小矩阵规范参考；其测试通过不替代仓库实现或 full toy 实验。

### 14.5 工程与研究状态分离

```json
{
  "engineering_status": "PASS|FAIL|BLOCKED",
  "modeling_status": "SUPPORTED_LOCAL|NO_ACCEPTABLE_R|UNIDENTIFIED|INSUFFICIENT_DATA",
  "observation_scope": "KNOWN_REALIZED_UPDATES",
  "real_vlm_validation": "NOT_RUN|READONLY_PARTIAL|READONLY_READY",
  "new_gpu_started": false,
  "online_ssvc_started": false,
  "safety_certified": false
}
```

研究结果不好不导致伪造工程 PASS；工程通过也不让 modeling_status 自动变为成功。

---

## 15. 结果文件及必须制作的图

每个阶段保存配置、命令、版本、时间、资源、原始指标、退出码和路径哈希。

存储不能按所有方法×r×n×噪声×时刻×题目无上限展开：精确轨迹、计数、分支更新与拟合参数保留一次；完整逐题预测至少保留冻结最终配置的 test 输出。全网格以逐 seed/群体充分汇总和可复算模型参数保存，不保存冗余中间张量。预测未揭盲身份记录不能省略。

主表：

1. `coordinate_reconstruction.csv`：表示误差、边界覆盖、n 对 q/s 精度的影响。
2. `identifiability_witnesses.json`：反例参数、观测相同项、响应差异。
3. `response_dimensions.csv`：k、r、谱、别名、条件数、方向遗漏。
4. `prediction_by_seed_group.csv`：所有方法、所有群体和预算。
5. `error_decomposition.csv`：全 J、低维 J、拟合与有限步真响应。
6. `tracking_windows.csv`：每个中间时刻及 H，不只终点。
7. `cost_ledger.csv`：数据生成、训练、校准、JVP、拟合、跟踪、输出 I/O 分开。
8. `real_artifact_inventory.csv`：来源及缺项，不制造数据。

必要图仅五张：

- 误差—维数曲线，按 n 分开，包含最弱群体。
- 实际响应与预测响应散点，标注近零区域与测试数。
- 同一窗口的真值/预测/误差带，选择预先固定的轨迹，不按最漂亮曲线挑选。
- 维数—窗口 H 的误差热图，缺数据处留空。
- 相同计算预算的平均误差与最坏群体误差。

所有图链接源表；不使用 Sankey 声称识别了唯一 X→W 运输。不同题目移动用固定 identity 连接，不能重新匹配点获得平滑流线。

---

## 16. 相关工作与本阶段可声明的范围

下列关联已在本次通过一手来源核读；核读深度见 `SOURCES_AND_REPO_SNAPSHOT_zh.md`。

| 来源 | 必须尊重的已知内容 | 本轮安排 |
|---|---|---|
| PSR [R1] | 用预测描述隐藏状态已存在 | 作为思想基础，不声称首次概率状态表示 |
| AutoJudger [R2] | 多维能力估计＋主动 MLLM 评估 | 本轮不将能力画像或主动检验器作新颖性 |
| TuneAhead [R3] | 短训练日志/梯度等特征预测微调表现 | 包含普通训练信息预测基线 |
| Active subspaces [R4] | 梯度/响应方向降维、估计误差已存在 | B4 与维数估计直接比较，不换名认领 |
| LESS [R5] | Adam-aware、低维梯度影响分析 | actual d 与梯度代理比较，但不把前者作首次 |
| Active Testing [R6] | 有限评估预算与主动选择偏差 | 本轮固定面板/预算，主动抽题延后 |
| Koopman training [R7] | 训练动态低维建模已有工作 | 不因加线性状态方程声称新方向 |

本轮可能提供的是：**在严格定义的嵌套语义事件上，对表示、响应维数、采样量与局部外推范围作可反驳的资格测量。** 它首先服务建模选择，尚不是完整创新性证明。

若 B6/经典低秩方法已解决问题、风险选维没有增量，报告该结果；不因论文需要而隐藏对照。把自动检测、反馈控制留到模型通过资格之后。

---

## 17. Codex 完成后的汇报格式

请按以下顺序写 `MODELING_DECISION_zh.md`：

1. 本轮实际执行的范围与版本。
2. 语义概率的存储、解释、拟合表示分别选什么。
3. 哪些任务/更新范围存在可用低维结构，给 r/n/H，而不是单个“最佳维数”。
4. 主要失败来源：遗漏方向、曲率、测量还是不可识别。
5. 和 persistence、日志模型、随机投影、经典低秩响应的对比。
6. 真实 Qwen 数据是否足够，缺项精确到字段/产物。
7. 下一步最小实验建议；停止，不自动进入检测/控制或 GPU 训练。

所有数值以真实日志为准。不能把参考测试结果写成完整实验，也不能把 toy 建模结论直接转述为 Qwen 模型内部机制。

