# 语义状态驱动的前瞻多奖励选择：Codex执行规范

**日期：2026-09-24｜工作名：prospective-selection-v2｜资源：最多5张PRO6000**

本文件规定待执行的研究，不代表已取得方法收益。主线：**在训练前观察语义状态，选择未来32步的奖励配置，用独立结果评价该选择。**

---

## 1. 本轮必须解决的研究问题

### 1.1 主问题

在所有方法都获得相同的前置样本、X/A/V/C标签和任务身份时，额外的**修复位置、原正确坐标破坏及其联合结构**，能否改善未来训练操作的选择？

主要对比：

`Z3_REPAIR_STRUCTURE selector` 对 `Z2_REWARD_DISTRIBUTION selector`。

后者不是只看accuracy的弱基线，而是获得**逐题完整奖励分布、任务分层、过去8步变化和相同的已知训练元信息**。

### 1.2 本轮产出

1. 一套有明确字段、冗余关系和统计单位的语义概率表示。
2. 一个轻量、可复算的有限动作选择器；只预测各奖励配方的后续标量效用，不预测每题每一步完整概率。
3. “前置观测 → 冻结选择 → 独立训练 → 独立评估”的真实9B证据。
4. 对固定最优配方、GDPO、SAW和直接修复奖励的公平比较。
5. 明确收益是否来自额外信息、额外监督、学习器容量或更大搜索预算。

### 1.3 本轮不强求的东西

不要求低维、完整Jacobian、每步安全认证或所有候选都测到0.1个百分点；不要求先恢复完整黑盒状态；不以运行成本胜出为开工条件。仍记录时间和样本成本，使后续实用性判断有依据。

---

## 2. 本次设计依据与需修复的旧问题

最新D2已有真实16分支、512次更新。它提供了关系奖励在错误输出上改善的现象，但尚未验证前瞻选择。原点O1/O2的训练顺序不同，只有两个原点，不足以学习高容量选择器后声称泛化。

必须针对以下问题设计：

| 旧问题 | 本轮处理 |
|---|---|
| 读完端点pX后按pX排序，所有信息层一样 | 选择器只读取操作前状态；在测试候选训练启动前写出选择 |
| X/S/W/I与完整奖励均值重复 | 加入代数冗余测试和逐题完整奖励分布基线 |
| 不同原点训练顺序不同 | 新源轨迹及未来训练块使用共同的元数据预选题序 |
| 每题32次却要求所有群体同时1pp非劣 | 主检验改为固定H32、独立lineage上的选择收益；群体风险另报告 |
| all-UNKNOWN阻塞全部实验 | 不确定性状态与研究性操作选择分开，不伪造safe标签 |
| 两个原点训练出大网络 | 新增独立source lineage；默认核岭回归，无深度选择器前置要求 |
| 图像接口近饱和 | 图像/纯符号分别报告，保留固定全域主指标与预设视觉OOD检查 |
| 加更多语义监督可能本身有效 | 添加不做状态选择的DIRECT_REPAIR_R4对照 |

**不能把D2已读面板再称为新确认集，也不能把更多采样当成更多训练seed。**

---

## 3. 阶段图与并行顺序

### P0：旧端点独立面板复核＋信息审计

使用已保存的16个H32端点，对原E面板做每题32次采样；如果E已被读过，记录为外部面板复核而非未见确认。重点重算pX、C=1且非X、坐标破坏、修复局部性。无需重训旧16臂。已有同协议E结果可直接复用，不再次生成。

从旧原始数据验证信息恒等式和同奖励向量下的结构差别；输出变量目录。

### P1：实现与小规模真实smoke

完成新sampler、state builder、选择器和数据权限隔离。运行针对性CPU测试，再以真实9B运行2步、两配方和恢复测试。复用已验证概率执行路径；失败的teacher/cache优化不是主实验前置条件。

### P2：开发与调参矩阵

12个开发source lineage，4个调参lineage；各取step32/96两个原点。每个原点运行R0–R7、GDPO_R4、SAW_R4、DIRECT_REPAIR_R4，共11个分支，各32步。

先交付前4个开发lineage的完整结果和吞吐；继续剩余开发与调参任务。此阶段允许看结果、修正实现和选择超参数，但必须记录版本变化，不能混用不同训练协议而当同一试验。

### P3：冻结

选定表示、选择器参数、主目标、最终N及评估采样量。用开发/调参数据重拟合最终选择器。写出`freeze.json`。此后测试输出不得用于重新选择超参数。

### P4：前瞻测试

至少12个新测试lineage，每个只使用一个预定原点。先采集前置状态，生成四个selector的选择，再训练实际被选择的配方及固定对照。每个原点两次独立的未来训练重复。同一原点/配方/重复只运行一次，结果可被多个selector引用。

### P5：统计与建模决定

以独立source lineage为主单位，评价信息增量和最终选择收益。区分描述、选择性能、等效和非劣，不合并为一个PASS。

### P6：后续在线闭环

不是本次默认任务。即使P4有收益，连续多次调权的累积效应仍需另一轮设计。不得把一次32步选择直接写成在线SSVC已验证。

---

## 4. 训练原点：数量、独立性与来源

### 4.1 Lineage定义

一个lineage是一条从同一固定基座和初始化协议出发、具有独立采样随机性的source训练轨迹。来自同一lineage的step32和step96**不是两个独立训练seed**。

主研究固定同一Qwen3.5-9B checkpoint和LoRA初始化seed17；source随机seed变化。source配方在R0/R1之间平衡，形成有目的但不按结果筛选的训练状态差异。所有基线都知道source配方和训练步数。

### 4.2 注册

- 开发：61001–61012；12 lineage，每个step32与96。
- 调参：62001–62004；4 lineage，每个step32与96。
- 测试池：63001–63048；最终使用前N个，N在12/16/20/24/32/48中、在看任何测试结果前决定。
- 测试按四cell循环平衡：R0/step32，R0/step96，R1/step32，R1/step96。

开发/调参source训练到96；测试source只训练到其指定原点。每条source保存0、24、32、88、96中实际经过的节点。t−8与t用于观测源训练过去8步的变化。

旧41001/41002和D2端点仅用于历史探索，不混入新测试lineage。

### 4.3 为什么不只复用O1/O2

当前需要学习“某类前置状态适合哪种奖励”，独立上下文数量由lineage和原点决定，不由每题的回答数量决定。4–5张卡优先并行不同lineage和奖励分支，而不是重复生成同一原点上极多回答。

---

## 5. 数据面板与训练题序

### 5.1 新训练池划分

从现有576个train scene按family×interface分为六层，每层96个；仅按固定元数据排序，分为：

- source池384：每层64个；
- continuation池192：每层32个。

source共96步×4prompt，无放回。每个future block取continuation池128prompt，32步×4prompt，无放回。两次重复使用两个预先固定的分层题序。

这样，未来分支不会重新使用source已经优化过的题。source池和continuation池只在train中划分，不打开control/dev/confirm补齐。

**同一重复的未来题序跨配方、跨原点保持相同。** 两次重复使用不同的固定题序，所有方法共同承受这项变化。source配方和源随机性不会悄悄改变未来题目组合。

同一原点的所有配方从完全相同的参数和Adam开始；新的branch sampler是明确的共同实验设置。保留parent source sampler，不把主动换成新分支题序描述成原source题序的原样续跑。

### 5.2 面板

| 名称 | 场景/提示数 | 用途 | 默认采样 |
|---|---:|---|---:|
| P | 原D2的36场景/72prompt | 所有selector的前置状态，t−8与t | 每点32/题 |
| D | dev中72场景/144prompt | 开发/调参分支H32效用标签 | 16/题 |
| T | confirm中144场景/288prompt | 冻结后的最终H32评估 | 先规划16/32/64之一 |
| T8 | T中固定12场景/24prompt | H8次要轨迹诊断 | 8/题 |
| E_old | 原36场景/72prompt | 旧端点独立场景复核 | 32/题 |

D、T在family×chart×operation的18个cell内平衡，分别每cell4/8场景。两个接口始终保留同一base_scene_id。所有选择只用元数据，不按难度或方法效果选题。

P、D、T与训练真值/场景保持分离。T读权限到冻结后才打开；P可跨所有原点复用，因为选择器必须使用可比探针。D可用于新原点的开发效用，即使其历史上有别的模型评估记录，也不声称D本身全新；最终推广依赖新lineage和T。

如果confirm此前被使用，先明确暴露情况，再选择未暴露的合法子集。不要无限追加新trend世界：原数值域的可用等差数列有限。确需新数据时，改为单独版本和数值域，重新定义外推范围；不能通过换ID伪装真值不重复。

### 5.3 保留视觉意义

主指标固定六层等权，不因图像任务接近饱和而改权重。单独报告图像pX、纯符号pX和三个关系家族。

可从现有OOD池预选36个场景的图像检查集（graph/numeric/render各12），只在最终选中策略及固定基线的H32上评分。各subtype分开，按base_scene聚类。它是次要视觉泛化检查，不改变selector、主指标或数据生成难度。文件不足时报告缺项，不阻塞主研究。

若收益仅在纯符号接口，应准确写成关系修复/奖励选择结果，不声称视觉感知能力已提升。

---

## 6. 概率状态的字段与信息权限

### 6.1 基础奖励变量

`r=[X,A,V,C]`：X精确世界正确；A由确定执行器得到答案正确；V严格合法；C可靠关系满足比例。I输出的C为0。沿用旧parser，不因预测器需要而宽松修复输出。

必须测试：

`pS=E[A]−E[X]`，`pW=E[V]−E[A]`，`pI=1−E[V]`。

trend的C∈{0,1/2,1}时还需测试：`P(C=1, not X)=2E[C²]−E[C]−pX`。这些指标用于解释，但不作为额外信息的证据。

### 6.2 新修复结构

对合法四整数输出z、观测o、真值x及唯一损坏坐标j：

- `F=1[z_j=x_j]`：损坏坐标是否恢复；
- `B=sum_{k≠j}1[z_k≠x_k]`：原本正确的三个坐标中改坏几个，0..3；
- `M=sum_k1[z_k≠o_k]`：相对观测修改几个位置，0..4；
- `copy=1[M=0]`；`single_edit=1[M=1]`；
- `coord_accuracy=(F+3−B)/4`；
- `damage_any=1[B>0]`。

对合法输出有`X=1[F=1 and B=0]`。因此F/B也不是彼此完全独立的维度，要保留恒等式并报告冗余。对非法输出F/B/M记为缺失，稀疏直方图用单独INVALID atom；不要把非法的B=0解释为保留了信息。

计算表现指标时，无效输出的坐标正确率按0计；训练DIRECT_REPAIR使用的Bdamage另按第7节定义，不混入合法条件统计。

### 6.3 四层选择器

全部使用同一批P输出、同一标签器和同一过去窗口；不追加高层专属样本：

1. **Z0_MOMENTS_GLOBAL**：所有四通道全局均值、完整二阶矩/协方差，以及t−8到t的变化。
2. **Z1_MOMENTS_STRATIFIED**：上述统计按六个family×interface分层。
3. **Z2_REWARD_DISTRIBUTION**：Z1＋逐题完整`event×C`经验分布，等价保留当前可用的完整奖励向量分布；保留逐题身份和时间变化。
4. **Z3_REPAIR_STRUCTURE**：Z2＋逐题`reward_bin×F×B×M`联合稀疏分布及修复边际。

C用满足数/关系数的精确有理数键，不把0.999等浮点阈值当新语义状态。所有方法还得到相同的source配方、checkpoint步数、过去8步已产生的梯度范数/有效组比例等训练日志摘要。不给任何方法lineage ID、未来题目结果或测试收益。

**原始输出可统一落盘，但selector只接收其层次允许的feature packet。** 不能让Z2通过隐藏路径访问完整输出和真值坐标，再自行恢复B/F。

### 6.4 维度与噪声

不强制PCA或低维embedding。高维稀疏分布通过Gram矩阵建模，不需要巨大深度网络。经验零仅表示未观察到，不能解释成真实概率为零。

条件统计没有样本时输出mask，不填成确定零。主核使用经验分布的平方根表示，不给上千个格子各加0.5而使先验质量压过32条样本。修复边际与联合分布分别保留；信息熵等探索量不得替代实际选择收益。

---

## 7. 奖励操作与训练公式

### 7.1 可选动作：R0–R7

沿用旧配方：

| 操作 | [X,A,V,C]权重 |
|---|---|
| R0 | [2,0,0,0] |
| R1 | [0,2,0,0] |
| R2 | [2,0,1,0] |
| R3 | [2,0,0,1] |
| R4 | [2,0,0.5,0.5] |
| R5 | [0,2,1,0] |
| R6 | [0,2,0,1] |
| R7 | [0,2,0.5,0.5] |

每个selector从同样八个动作中选一个，未来32步内固定该配方。此阶段检验离散配方选择，不宣称解决了连续任意权重优化。

### 7.2 外部比较

- **BestStatic**：只用开发/调参数据选出的最优固定R0–R7，在所有测试原点使用同一动作。
- **R0**：精确主奖励参考。
- **GDPO_R4**：按通道组内归一化、再batch归一化；它不是动态阶段选择器。
- **SAW_R4**：沿用已核验CV算法和R4 priorities，训练时逐batch动态权重。
- **DIRECT_REPAIR_R4**：`2X+0.5V+0.5C−0.5Bdamage`，合法Bdamage=B/3，非法Bdamage=1。

DIRECT_REPAIR只是固定的额外监督基线，不加入可选动作集合，避免改变四信息层候选集合。它测试“直接奖励保留正确信息”是否已足够，无需状态选择。

GDPO/SAW公式从已测`reward_recipes.py`继承，核对原文一次；不得把单次方差标准化写成GDPO、不得把SAW的理论最小值替换成经验最小值。paper-default和旧实现有差异时分别命名，不静默改写历史。

### 7.3 共同训练细节

- Qwen/Qwen3.5-9B，已锁revision；BF16基座；LoRA target/rank/alpha/dropout按D2实际配置继承。
- B=4，K=8，32个序列/步；每个batch仅一次PPO/GRPO更新。
- rollout max_new_tokens=64，thinking=false；采样温度及过滤按已验证路径核对，不静默改变。
- 主联合归一化：population variance（ddof=0），`sqrt(var+1e-8)`，即epsilon=1e-4在平方根内。
- PPO clip=0.2；固定Lnorm=64；每序列token损失分母为`B*K*64`。不能换成每序列长度、有效非零样本数或单个microbatch大小。
- AdamW lr=1e-5，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；global grad clip=1；无新增scheduler/scaler/KL项。若实际D2不同，先报告差异并桥接。
- 所有训练输出都用当前分支策略产生；相同配方/原点/重复可复用完整run，不同策略不能复用rollout训练。
- 零优势保留显式零梯度并执行Adam；不得通过跳步消除历史动量。
- 评估前后恢复训练RNG和模型mode；评估不能改变下一个训练batch。
- EOS计入实际生成的动作得分；padding与prompt不进loss；长度截断按旧合法性规则判定，不丢掉难样本。

### 7.4 随机性

未来训练题序来自固定branch_schedule_seed，与origin_id/recipe无关。采样seed由`(experiment_seed,lineage,repeat,step,prompt,draw)`确定，recipe不进入seed键，便于共同随机数配对。

各分支仍独立调用自己的当前策略。共同seed不是逐回答身份相同，也不是事件运输证据。dropout/模型随机性使用同样的起点恢复。评估采用单独随机role，默认各实际策略独立流，不暗设相同token配对。

---

## 8. 具体选择器：不等待大型响应模型

### 8.1 学习目标

对开发原点o、动作a：

`Y[o,a] = J(o,a,H32) − J(o,R0,H32)`。

`J`为六个family×interface层的pX等权均值。所有方法使用同一个目标；不把损坏坐标惩罚加入主J来天然偏向语义表示。纯符号pX、坐标质量和关系错误作为次要结果。

R0的差值恒为0。共享R0造成不同动作标签相关；拟合和bootstrap以完整原点/lineage块处理，不把8个动作标签当8个独立上下文。

### 8.2 表示核

对离散经验分布h使用`phi(h)=sqrt(h)`。逐题内积后按固定prompt权重平均，得到PSD的Hellinger亲和核。保留raw counts和n；同样的预处理对全部原点固定。

- K_R：完整奖励分布的核。
- K_E：修复联合分布及修复边际的核。
- 语义核：`K_S=0.5*K_R+0.5*K_E`，不通过测试结果调整0.5。
- 历史：另计算`phi_t−phi_(t−8)`的线性核，固定权重0.25。
- 相同训练元信息使用共同核，权重0.25。
- 各核仅用fit数据的平均对角线作尺度归一化；保存该比例，测试不重新计算。

Z0/Z1使用标准化矩向量的线性核；标准化只在开发fit数据上拟合，常数列删除。Z2/Z3使用上述分布核。对每层都报告输入维数/非零数，但不把它们称为真实内部状态维数。

### 8.3 核岭回归与选择

每个动作分别拟合（同一个K，可共用分解）：

`beta_a = solve(K + n_fit*alpha*I, Y_a − mean_fit(Y_a))`。

`pred_a = mean_fit(Y_a) + k(z,new)^T beta_a`。

alpha只取`[1e-4,1e-3,1e-2,0.1,1,10]`。每层同样六个alpha，避免给语义模型远多于基线的调参预算。用调参lineage上的**所选动作实际J**决定alpha，而非只看回归R²；性能差小于0.001时优先更大的alpha。

若BestStatic的预测值距离最大预测值≤0.005，回退BestStatic；否则取预测最大者，完全相同时按固定recipe ID。0.005是工程上的近似平局范围，**不是证明真效用等价的统计区间**。

所有选择器记录预测值、备选动作、margin、历史/当前状态；可用200次开发lineage bootstrap拟合小型selector记录选择频率（不追加9B训练）。后者只是稳定性描述，不写成95%安全概率。

当前状态完全没有有效观测或packet损坏时，回退BestStatic并标记原因，不用未来结果填补。合法但少见/陌生状态可以标记EXTRAPOLATION，同时仍按冻结的研究性规则选择；不要设置任意参数距离0.05的拒绝门槛。

### 8.4 开发、调参、冻结

开发12 lineage×2原点=24个fit上下文。调参4×2=8个上下文。多个draw、bootstrap副本、同源step不能伪装成更多fit原点。

alpha和所有可调选择只使用开发/调参数据。最终冻结后可用两者合并的32个原点重拟合系数，但不再改规则。做leave-lineage-out诊断，不随机拆散同一source的两个checkpoint。

主比较是Z3对Z2，非Z3对只看pX的Z0。Z0/Z1说明平均与分层信息的价值。另做now-only去历史、修复特征置乱等开发消融；不为每个消融重新训练9B，复用开发动作收益表。

### 8.5 三个结论等级

- **有诊断信息**：修复分布不同，未证明有选择收益。
- **有开发选择收益**：只在已见lineage的重采样/开发验证中有效。
- **有前瞻增量价值**：未见lineage的冻结Z3优于同权限Z2，且没有被静态、动态或直接语义奖励对照解释掉。

新增信息不一定改变最佳动作。若所有状态都适合同一动作，选择器可能没有任何可利用的收益空间，应报告，而不是故意制造失败。

---

## 9. 前瞻测试的执行防泄漏协议

### 9.1 冻结文件

`freeze.json`必须包含：源码版本、配方、模型、四层特征schema、缩放系数、alpha、BestStatic、T面板、最终N/m、主比较/H32、次要分析、选择规则。

### 9.2 每个测试原点

1. 从注册source构造指定原点；读取t−8/t的P样本。
2. 所有selector只接收`PreDecisionPacket`。
3. 写`decision.json`，记录四个选择和冻结模型ID。此时该原点未来32步端点不存在。
4. 汇总四个选择、BestStatic、R0、GDPO_R4、SAW_R4、DIRECT_REPAIR_R4；按实际recipe去重。
5. 对去重后的每个recipe运行repeat1/2，共同题序、各自on-policy采样。
6. H8只记诊断；不据其升降重选、停掉差臂或删除结果。
7. H32在T评估，再做统计。

同一recipe被多个selector选中，只产生一份该原点/重复的训练与评估数据；不同lineage/重复仍使用独立评估抽样流，不能复用计数后增加独立N；它们因此可能有完全相同结果，这是有效结论。未来训练结果不能反馈进本轮已经冻结的selector。

### 9.3 不是测试时穷举搜索

开发阶段完整动作矩阵用于学习。测试阶段只训练冻结选择和预定基线的并集，不看所有未来结果再选最优。需要“oracle”参考时，在开发数据上用独立评估流给出描述性上界；不要用测试噪声最大的动作作oracle再声称regret准确。

### 9.4 测试重复

每个source lineage有两个未来训练重复，包含不同branch schedule和rollout stream；两个重复对同一selector使用同一个提前选好的recipe。先平均两个重复，再进行lineage层统计。

它们降低未来训练随机性，但仍只有N个独立source lineage，不是2N个。

---

## 10. 精度、阈值与样本量：执行前算清楚

详见`THEORY_STATISTICS_zh.md`。这里给出操作规定。

### 10.1 主指标与规模

- 主指标：T上六层等权的pX，H32。
- 主信息比较：Z3−Z2。
- 目标有意义效应：1个百分点（0.01），这是本次研究目标尺度，不是预期保证。
- 选择近似平局：0.5个百分点（0.005）。
- 群体风险参考：相对R0的−2个百分点，仅用于报告风险区间和敏感性，不作为所有训练分支必须先证明的启动门槛。

主选择任务不执行旧版“所有群体同时1pp非劣”的筛选；旧判定保留在历史报告，不能事后改写为已通过。

### 10.2 最终N

最终测试先注册至少12 lineage。使用开发/调参leave-lineage-out选择收益差规划；每个lineage仅选一个元数据预定阶段（32或96），使四cell均衡，不能先平均同源两阶段而低估单原点方差。按四cell内方差得到等权有效SD，bootstrap其75%分位作为保守规划值s_plan。按

`N_req ≈ ((1.96+0.8416)*s_plan/0.01)^2`

映射到12/16/20/24/32/48。N只在测试结果出现前确定。公式是正态近似功效规划，不是严格有限样本定理。

若48仍不足，则预先报告本轮实际可检测效应，执行已锁规模；不能看最终p值后补seed直到显著。旧D2两原点不能用于声称s_plan已被精准估计。如果有效独立lineage少于8，或差异全因选择同一动作而恒零，标记SD_UNIDENTIFIED；默认在测试前按s=0.02选择32个lineage，同时报告这是保守规划而非实测功效。

### 10.3 每题样本量

P固定32；开发D固定16/题用于初始矩阵，系数拟合记录标签Monte Carlo方差。16条不是为了精确估计每题尾部概率，而是形成144prompt宏观J。需要开发精度复核时，在预先指定4个开发原点、所有11臂统一追加到32；不能只补看起来会赢的配方。

最终T从16/32/64中一次性规划，使**全N个lineage、两重复平均后的Monte Carlo标准误**≤目标效应的四分之一，即0.0025。使用调参数据/最坏0.25 Bernoulli方差规划。不能把这个全实验精度要求变成每题都要0.25pp。

训练seed方差与场景差异不会随同题重复抽样无限减少。MC已足够小以后，把资源用于独立lineage，不盲目每题增加到4096。

### 10.4 主统计

每lineage先对重复平均，得`D_l=J_l(Z3)-J_l(Z2)`。按source配方×原点阶段四个预定cell等权汇总；主CI使用分层配对均值的Welch–Satterthwaite t区间，bootstrap在每个cell内以lineage为块重采样（10,000次）。主推断条件于固定T和冻结选择规则。

Bootstrap必须把该lineage的配方、重复和指标作为整体重采样；主分析不再内层重采样token/draw，以免重复计入已含的测量随机性。场景推广另做scene-cluster/multiway敏感性，明确这是另一个估计对象。

N少、重尾或方差为零时，不输出虚假的精确结论。所有观察差都为0时报告NO_OBSERVED_DIFFERENCE；需要真实概率等效声明则另外给出固定面板抽样界。

### 10.5 多重比较

仅Z3−Z2为一个预定主检验。Z3相对BestStatic、GDPO、SAW、DIRECT_REPAIR的四项比较用Holm，属于次要家族。

H8、三个纯符号群体、三个图像群体、pA/qX/坐标质量/关系全满足错误均为次要或探索分析，标清区间种类。不要对所有prompt×配方×时刻做一个巨大的Bonferroni，再要求主选择实验全部通过。

### 10.6 决定与安全不同

点预测选择可以被执行并接受真实检验，不代表已获安全保证。CI包含0不等于等效；只有相应区间完整落入预先指定±1pp时，才能在该对象/置信水平上说近似等效。

数值故障是真停机条件；语义效果不佳是需要保留的实验结果。不能以“不符合预期”为理由换seed、删配方或停差臂。

---

## 11. 4–5卡训练与数据复用

默认5个独立单卡worker。一张卡跑一个source/分支/评估任务；不用DDP复制小batch或改变B/K。分配4卡时并发降到4，不修改实验配置。

阶段优先级：

1. 一张卡smoke，其他可做无需该smoke的新数据整理/旧端点E只读测量。
2. smoke后3–4卡source/分支，1卡评估；任务短缺时可重分配。
3. 开发按lineage完整小块交付，避免跑完所有source才开始分支。
4. 测试在冻结后并行，不能因空卡绕过freeze。

源t=32完成后可以分支，source进程继续到96；读取的是不可变已完成checkpoint，两者不共享可写adapter。

保存一个基座快照、每分支LoRA/Adam/RNG等可训练状态；不要复制9B到每个分支目录。P/D/T原始样本一次落盘，可供所有信息层重分析。Score cache键至少包含策略/输入/token序列/终止条件/概率路径；同文不同token不得误合并。

主状态与最终性能优先采用真实生成计数，不为每一条输出额外执行两个策略的似然评分。LR/支持质量界只是特定诊断的可选补充，不是所有结果必须走的慢链路。

详细任务表、续跑和Slurm要求见`GPU_RUNBOOK_zh.md`。

---

## 12. 代码实现目录与数据类型

新目录：`ssvc_flow/src/prospective_selection/`。不要直接改历史结果的决策状态。

建议模块：

- `protocol.py`：配置、阶段、独立lineage注册和派生工作量；
- `dataset.py`：source/continuation池及P/D/T；
- `sampler.py`：跨原点共同题序、按role随机数；
- `semantics.py`：F/B/M、奖励冗余、INVALID atom；
- `features.py`：四层信息权限和概率结构；
- `kernels.py`：完整分布与历史核，不强制降维；
- `selectors.py`：核岭回归、BestStatic、近似平局；
- `origin_runtime.py`：真实source训练；
- `branch_runtime.py`：复用D2训练损失/状态恢复；原10配方回归一致，新DIRECT_REPAIR通过独立的advantage回调接入相同loss reducer，不全局monkey-patch旧函数；
- `evaluation.py`：固定面板计数、独立RNG、配对聚合；
- `precision.py`：冻结前N/m规划；
- `analysis.py`：信息冗余、主比较、多重比较、错误结构；
- `jobs.py`：不重复训练的DAG和最多5卡调度；
- `cli.py`、`reporting.py`。

三种类型必须分开：

```python
PreDecisionPacket  # origin及其过去的观测；无未来端点字段
ActionDecision    # selector version, recipe, margin, made_before_run
OutcomeRecord     # origin/recipe/repeat/H、P/D/T与指标
```

`choose_action(packet)`不能接收OutcomeStore/任意文件路径。开发拟合才可接收已标注的D OutcomeRecord；测试推理只能加载冻结模型。

`OutcomeRecord`需有`lineage_id, origin_step, source_recipe, continuation_repeat, recipe_id, schedule_id, panel_id, horizon, n_draws, execution_kind`。禁止把origin_id替代lineage_id用于随机划分。

### CLI契约（需要Codex实现）

```bash
python -m src.prospective_selection.cli plan --config ... --out ...
python -m src.prospective_selection.cli audit-history --config ... --out ...
python -m src.prospective_selection.cli prepare-data --config ... --runtime ... --out ...
python -m src.prospective_selection.cli smoke --config ... --runtime ... --allow-gpu --out ...
python -m src.prospective_selection.cli source --lineage 61001 ...
python -m src.prospective_selection.cli observe-state --origin ... --role predecision ...
python -m src.prospective_selection.cli branch --origin ... --recipe R3 --repeat 1 ...
python -m src.prospective_selection.cli fit-selectors --role development ...
python -m src.prospective_selection.cli plan-precision ...
python -m src.prospective_selection.cli freeze ...
python -m src.prospective_selection.cli decide --origin ... --freeze ...
python -m src.prospective_selection.cli submit-ready --max-gpu-jobs 5 ...
python -m src.prospective_selection.cli worker --task ... --runtime ...
python -m src.prospective_selection.cli analyze --phase final ...
```

`...`由Codex替换为实测路径，最终手册不得留下无法执行的省略号。`plan`/`prepare-data`不能加载9B或偷偷提交作业。

---

## 13. 科学停止点与失败解释

- E独立复核没有重现：保留结果，降低相关机制叙述强度；新前瞻比较仍可继续。
- Z3不能胜过Z2：可能额外信息不帮助选择、上下文不足、奖励动作缺少收益空间或学习器未学会。分别检查，不把所有失败归为“模型黑盒”。
- Z3与Z2总选同动作：这是有效的零增量结果，不重复训练以制造差异。
- Z3只胜过Z0、不胜过Z2：收益来自更完整/分层奖励信息，不支持额外修复结构的优势。
- DIRECT_REPAIR更好：额外语义监督的简单使用已有效；不能声称复杂状态选择必要。
- 只在已见lineage或H8有效：不能外推到H32和独立训练。
- 所有候选效果接近：固定配方可能已经足够；不要为了选出赢家去放大
action强度或筛难题。
- 最弱群体有损害而总体改善：如实报告权衡，不用总均值掩盖。

---

## 14. 必须交付的文件

1. `IMPLEMENTATION_REPORT_zh.md`：新增模块、实际测试、未完成内容。
2. `DATA_AND_ORIGIN_REGISTRY.json`：来源、角色、题序和独立单位。
3. `STATE_VARIABLE_AUDIT_zh.md`：冗余关系、同奖励不同修复结构、数值稳定性。
4. `DEVELOPMENT_RESULTS_zh.md`：所有配方/原点、H32主结果、H8诊断、选择器开发曲线。
5. `freeze.json`、四个frozen selector、precision planning报告。
6. `decisions.jsonl`：每个测试原点在未来训练前的选择。
7. `test_lineage_outcomes.csv`、`primary_comparison.csv`、`secondary_holm.csv`。
8. `RUNTIME_AND_RESOURCE_zh.md`：实际GPU小时、生成/评分/反向数量、缓存/去重节省；不把逻辑操作数当物理吞吐。
9. `FINAL_DECISION_zh.md`：表示是否增加可用信息、是否改善选择、适用范围、仍未解决的问题。
10. 共享原始样本、必要checkpoint索引、可复算脚本和测试日志。报告与原件分开，不把TB级权重塞进每个交付ZIP。

本轮最终主张必须与证据对应：**前瞻奖励选择的信息价值**，而不是已经恢复了内部概率流或证明任意在线训练安全。
