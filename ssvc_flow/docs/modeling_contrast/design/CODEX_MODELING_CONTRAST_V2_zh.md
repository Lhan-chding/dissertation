# SSVC 建模第二轮：候选差异、观测噪声与有效维数

版本：`modeling-contrast-v2-20260915`  
对象：`Lhan-chding/dissertation` 的 `ssvc_flow/`  
实施范围：旧实验只读复算、737 参数 CPU 估计与建模实验、必要的独立 CPU 验证。  
交付目标：确定在何种观测方式、采样量、校准预算下，哪些候选差异可辨识，以及所需的最小有效预测维数。

> 先执行 `START_HERE_FOR_CODEX.md`。本文件中的新估计器和新目标都是待检验方案，不是已取得的实验结果。新模块不得改变历史结果、原奖励、原解析器或现有服务器作业。全篇以概率为计算单位：0.001 = 0.1 个百分点。

## 0. 本轮必须产生的决定

完成后要给出下面五项答案，允许答案为“在受测范围内没有合格方案”。

1. 总体响应的低秩近似，能否在相同预算下识别 `joint_1 − joint_0` 的关键群体差异？
2. 差异估计的主要瓶颈来自独立计数、共享观测协方差、方向估计，还是局部非线性？
3. 直接差异建模、协方差处理、共同样本似然差，分别带来了多少增益？
4. 最小合格维数是否会随目标和观测机制而变？不要求预先证明一维、二维或任何特定秩。
5. 相对于直接测量候选差异，建立局部预测器是否有实际的成本收支平衡点？

本轮不实现在线触发器，不优化 SSVC，不更换大模型，不增加真实 GPU 调用，不把候选优劣的离线诊断称为已完成在线控制。历史 S1/R2/S2 若有外部作业，既不取消也不自动续接；本工作独立执行。

## 1. 依据与版本核对

### 1.1 已有结果：作为动机，不作为本轮成功标准

依据附件 `d48b4ebd-5316-46e8-abf3-cc8a0231a764.zip` 及独立复核报告：

| 已观察结果 | 直接设计影响 |
|---|---|
| exact 局部一维的总体响应 NRMSE = 0.2334 | 保留原预测器作为强基线，不否定其总体跟踪价值 |
| 同一规则在六群体 `joint_1−joint_0` 的 pX/v NRMSE = 0.9652/1.0169 | 把候选差异设为单独主目标，不能沿用总体误差资格 |
| exact 响应平方能量中 X 约占 2.65% | 完整四事件拟合和关键真值目标同时评价，不能只看总能量 |
| n16/64/256 下受测有限观测规则未合格；方向与 exact 方向夹角中位数约 82–83° | 先解决观测与辨识，不先增大模型或盲目增加秩 |
| 多个候选差分共享起点计数 | 显式构造联合协方差；直接差异中应当代数消去共同起点 |
| 七项消融用了 r=0 | 所有消融另设非零固定规则；未通过选模不是 r=0 方法获胜 |
| H4 聚合合格，但约 25% 窗口至少一次超标；误差带宽 | 本轮不重新宣传 H4 检测周期，也不把宽带覆盖称为可用 |
| 附件只含一条完整轨迹，其他原始文件须补齐 | 先取原件，不悄悄重训替代缺失文件 |

这些数字来自既有复核，不是本交接包新跑出来的效果。`evidence/` 保留部分摘录和表供核对。

### 1.2 本次仓库实查

2026-09-15 读取 GitHub：

- `codex/ssvc-modeling-qualification-20260915`：`c71c60b4dad2d4432869eb195bb82069ebde374f`，与附件来源一致；本轮优先以此为父提交。
- `codex/ssvc-mechanism-followup-20260914`：`7b0da74cca8f20bf1a816b50ba13750c111044f9`，是另一条继续推进的工作线，不应覆盖或自动合并。
- `main` 仍指向旧研究线，不能从 main 直接假定已有 modeling 模块。
- 建议新分支：`codex/ssvc-modeling-contrast-v2-20260915`，独立 worktree；已有同名分支时先检查实际内容，不重置。

Codex 启动时重新读取 HEAD、dirty state、分支和父文件哈希；若父分支进展，保留该状态并输出差异审计。不要 force checkout、reset、force push 或修改正在运行的源码。当前任务默认不需要远程修改权限。

### 1.3 必须先读的代码

`src/modeling_qualification/{toy,collection,math_contracts,models,evaluation,tracking,protocol,io,real_audit,real_vectors}.py`，以及 `docs/modeling_qualification/results_20260915/` 下的报告。特别检查：

- `collection.py` 的 `theta/d/branch_theta/branch_d/branch_alias/anchors` 及 paid-observation/oracle 数据边界。
- `models.py::build_subspace, ridge_map, fit_local_model, log_features`：当前基于实际 d，普通 ridge，低秩方向由拟合系数决定。
- `toy.py` 的精确有限动作概率、状态恢复、Adam 与 seed 规则。
- 现有拟合数据、测试 bank 和 seed 角色的真实绑定。

保留旧模块的公开接口和源码。新代码放 `src/modeling_contrast/`，薄适配器复用已有数据结构。只有确定的实现 bug 才允许独立补丁，必须有失败用例、修复记录和兼容性测试；本协议改变不是旧 bug。

## 2. 核心改进：分别建模共同运动和候选额外变化

### 2.1 语义状态不变

仍记录逐题固定身份、四类计数、n、权重、checkpoint、推理指纹。解释坐标：

\[
v=p_X+p_S+p_W,\quad q=p_X/v,\quad s=p_S/(p_S+p_W).
\]

分母为零返回 `null + reason`；缺样本与结构不可能分开。拟合用原正交 Helmert 矩阵 H：

\[
z_i=p_iH,\quad HH^\top=I_4-\tfrac14\mathbf1\mathbf1^\top.
\]

72 道探针组成 216 个响应坐标。不得改动作集合、事件定义、题目权重来扩大信号。对比之前必须确认相同 prompt、相同答案映射和解码规则。

### 2.2 三类响应逐一命名

在锚点 \(\theta\) 和训练 bank b 上，主更新后参数 \(\theta_b^0\)，其他候选 \(\theta_b^u\)。

\[
T_{b,u}=y(\theta_b^u)-y(\theta),\quad
B_b=y(\theta_b^0)-y(\theta),\quad
C_{b,u}=y(\theta_b^u)-y(\theta_b^0).
\]

因此 \(T_{b,u}=B_b+C_{b,u}\)。

- **主目标**：`C_joint = joint_1 − joint_0`。
- **预设次目标**：`C_gate = no_x_off_1 − joint_0`。
- **一致性目标**：`joint_1 − no_x_off_1 = C_joint − C_gate`。
- **辅助诊断**：原始 T、主响应 B、事件水平 y，分别报告，禁止互相代替。

相应参数输入：

\[
d_b^0=\theta_b^0-\theta,\qquad e_{b,u}=\theta_b^u-\theta_b^0.
\]

本轮对 C 主要拟合 `e → C`，保留原 `d → T → 相减` 作为基线。不让“模型已经知道所有测试候选输出”伪装成预测；测试 e 可见，但测试行为概率或似然不能在预测封存前用于预测器。

### 2.3 为什么局部差异值得单独建模：明确的理论条件

对梯度 L-Lipschitz 的标量 f，有：

\[
f(\theta+d^0+e)-f(\theta+d^0)
=\int_0^1\nabla f(\theta+d^0+te)^\top e\,dt.
\]

因而：

\[
|C-\nabla f(\theta)^\top e|
\le L\|d^0\|\|e\|+\tfrac L2\|e\|^2.
\]

若能在主更新终点获得局部梯度：

\[
|C-\nabla f(\theta+d^0)^\top e|\le\tfrac L2\|e\|^2.
\]

这说明共同运动项在差异中抵消，剩余误差与 e 相联系；不意味着所有 bank 共用的拟合算子都精确，也不意味着 e 方向容易估计。完整 Jacobian 和 L 只可用于 oracle/解析见证，不能偷渡给有限观测预测器。

当完整推理状态相同、e=0 时，C=0。参数相同但 Adam 不同，只能共享当前推理，不能合并未来训练状态。两个范数相同不代表向量相同。

## 3. 新的观测对象：联合测量包，不是互相独立的差分行

### 3.1 先消去共同起点，再谈统计改善

旧计数响应为 \(\widehat T_j=\widehat p_j-\widehat p_*\)。候选独立采样时：

\[
\operatorname{Cov}(\widehat T_j,\widehat T_k)=\Sigma_*\quad(j\ne k).
\]

但同一 bank 的差异：

\[
\widehat C=(\widehat p_1-\widehat p_*)-(\widehat p_0-\widehat p_*)
=\widehat p_1-\widehat p_0.
\]

共同锚点噪声严格消去。此事实必须数值核验，不能把“共享起点噪声”继续列为 C 的直接残差。C 仍有两个候选端点噪声、跨差异共享 baseline 的相关性以及拟合误差。

对唯一推理策略的概率估计堆叠成 P，候选/锚点别名用固定 incidence 矩阵处理；差异矩阵 L 的每行和为零：

\[
C=LP,\qquad \Sigma_C=(L\otimes I)\Sigma_P(L\otimes I)^\top.
\]

独立计数时，单策略、单题的四类比例协方差是 \((\operatorname{diag}(p)-pp^\top)/n\)。原始多项协方差奇异是正常约束；先转换 Helmert。CRN 或共同样本法有额外非对角块，不能按独立计数公式处理。

### 3.2 区分三种方差，不要混算

1. 固定策略和固定题目下的观测噪声。
2. 不同训练 bank 的真实响应异质性。
3. 不同训练 RNG 轨迹产生的训练状态差异。

`noise_replica` 只估计第1种，不算新的训练种子。多个题目、多个候选、多个锚点共享同一个训练种子时，不能当作独立训练重复。

## 4. 先比较观测方式：不默认只是把 n 加大

每个估计器必须在同一固定候选对、同一题目上估计 C。以下均是经典统计工具的应用，不能单独认领新算法。

### O-IND：候选独立计数差

每个唯一候选各生成 n 个动作，计算 \(\widehat p_u-\widehat p_0\)。别名去重；不同 contrast 共享同一 baseline 计数时记录协方差。主要复用旧计数，不重新模拟历史已存在样本。固定 n=64 为主，16/256 为稳健性。

### O-CRN：固定动作顺序、共同随机数

737 参数有限动作系统中，每个重复使用同一个 U~Uniform(0,1)，对两个候选分别做 inverse-CDF sampling。动作顺序由原 action_id 固定，禁止按 X/S/W/I 或真值排序。每个候选的边缘必须严格等于自己的策略。

共同随机数不是保证降方差，必须报告配对事件协方差、discordance 和相对 O-IND 的真实方差。相同 seed 不是语义运输路径，也不意味着真实 VLM 的变长采样器有同样耦合效果。

对某一事件的配对差 Z=1[C_u]-1[C_0]，令 \(\eta=E[Z^2]\)：

\[
\operatorname{Var}(\bar Z)=(\eta-C^2)/n.
\]

### O-LR-ORIGIN：共同独立探针样本上的似然差

在训练 bank 之外、冻结候选后，从锚点策略 \(\rho=\pi_\theta\) 采样 \(a_j\)。对同一动作，分别查询两个候选的完整动作概率：

\[
\widehat C_f=\frac1n\sum_j
\frac{\pi_u(a_j)-\pi_0(a_j)}{\rho(a_j)}f(a_j).
\]

f 可以是事件指示、Helmert 行或群体线性评分。要求 rho 覆盖候选差异的支持、概率为实际采样分布的归一化概率，且存在相应二阶矩。条件于固定候选与训练 bank，上式无偏。

固定候选接近时，先作概率/权重差再求均值，可利用共同随机性。这里的“先”是数值与数据耦合要求；分别求两个均值但使用完全同一批样本、同样分母，代数上相同。

源样本必须是独立 control，不准用生成候选更新的训练样本作独立精度证明。fit 和 held-out bank 的观测 packet 使用独立 RNG。旧 finite-count 文件没有动作级概率时，不能从四类计数反推权重；缺项就记录。

在 toy 中，采样服务可以用完整16动作分布实现采样，但预测器只获得付费查询到的样本、标签和该动作的 logp。不得把16动作概率整表交给 finite-observation 拟合器。

### O-LR-MIX：候选对混合采样的稳健对照

对每一 contrast 使用 \(\rho=(\pi_0+\pi_u)/2\)，每个样本先公平选择源策略，再从该策略生成。每个动作对两候选评分：

\[
w(a)=\frac{\pi_u(a)-\pi_0(a)}{(\pi_u(a)+\pi_0(a))/2},\qquad |w(a)|\le2.
\]

该界是全支持的解析结果，不是从有限样本的最大权重推断。固定每源各取 n/2 是另一种分层估计，方差公式不同；主实现使用 iid 混合，固定配额法只作明确命名的扩展。

混合采样需要候选生成和交叉评分，可能并不便宜。其价值和成本均须实测。多个 bank 不自动共享同一个混合分布；每包记录 proposal_id、源策略和其抽样机制。

### 4.1 数值与表示约定

- 直接事件 LR 估计的四项和在有限样本中可能不为零，其期望为零。报告 `mass_residual = mean(w)`。
- 拟合主视图使用 `w * H[category,:]`。还原的四事件差相当于 `w*(onehot(category)-1/4)`，是显式的已知零均值控制变量，而不是后处理偷偷“修正”观测。
- 同时输出 raw event-LR 与 Helmert-LR 的方差；Helmert 变换不保证每个事件方差都降低。
- 不截断、丢弃或自归一化权重作为主方法；这些操作会改变偏差。log-space 使用 expm1 等稳定差值算法，非有限值直接失败并保留记录。
- 常数/别名对比应精确为零，非零但未采到 X 的结果不能被当作真值不变证据。即使控制变量使估计不为零，也仍标记 `UNOBSERVED_X`。
- 支持缺失、过大权重、或经验协方差退化：返回观测状态而非默认通过。

### 4.2 为什么这种比较具有理论针对性

若 \(\pi_u-\pi_0=\alpha h\) 且固定 rho 满足支持与二阶矩条件，则共同样本 LR 的方差为：

\[
\operatorname{Var}(\widehat C_f)
=\frac{\alpha^2}{n}\operatorname{Var}_\rho[h(a)f(a)/\rho(a)].
\]

独立计数差的方差一般不会随 \(\alpha\to0\) 同样趋于零。因而“差异很小”对不同观测方式意味着不同困难。不要把一次独立计数失败认定为所有检测机制的信息下界。

这不是无条件优势：稀有事件、支持失配和打分成本可能抵消收益。O-LR-MIX、O-CRN 和 O-IND 正是为验证边界，而不是只保留有利估计器。

### 4.3 访问权限与有限动作的强基线：不得忽略

把比较分成两种访问条件：

- `SAMPLE_ONLY`：只获得按策略生成的输出和验证标签，比较独立/共同随机数计数。CRN还需明确可控制采样随机数，不能套给只有普通API的黑盒。
- `SAMPLE_AND_LOGP`：允许对指定完整动作求归一化概率，比较LR方法与直接评分。由额外logp信息带来的改善，不能全部归因于新估计器设计。

**当前16动作toy存在更强的基线。** 它有唯一正确候选以及两个明确非法候选。因此，若允许对已知动作评分，pX可由一次正确动作评分精确得到，v可由两个非法动作概率的补集精确得到。每策略每题最多3个标量logp查询就得到这两个主指标。原toy通常一次完整16动作forward即可完成softmax归一化；若该forward可复用，物理计算只记一次，不能把3个标量查询硬算成3次完整forward。Codex必须读取实际动作身份验证这一事实，并实现 `D-KNOWN-EVENT-LOGP`，计入完整成本。

这不是向预测器偷渡完整Jacobian，而是当前有限动作设定真正允许的直接方法。不得人为禁止正确动作评分以显示LR更有效。整个16动作概率表的直接枚举也保留为CPU诊断成本参考。若该基线胜出，本轮仍能用于分析受限采样的辨识问题，但不能声称在该toy获得实际计算节省。

对真实VLM，X通常是经过parser映射为真世界的多个token序列，单个规范字符串的概率不一定等于pX；本轮不得把toy这条等式直接迁移。将来要验证多序列表达的事件集合，另立协议，不修改本轮原动作空间。

## 5. 局部模型：完整概率存储 + 差异子空间 + 协方差意识

### 5.1 只用拟合 bank 构建方向

主校准沿用 fit banks 0..7；diagnostic 8/9；held-out 10..13。所有同一 bank 的候选绑定在一起，不能拆入 fit 与 test。

用拟合数据构造两种子空间并分别记录：

- `Q_track = span(d_b^u)`；
- `Q_contrast = span(e_{b,u})`。

未中心化 SVD，沿用 rtol=1e-10/atol=1e-14。报告奇异值、有效 k、别名、数值秩和测试 e 的出空间比例。不能从测试响应或测试更新构建 Q；测试更新只用于预测与 OOS 诊断。

若某锚点拟合 e 全为零，返回 `NO_CONTRAST_EXCITATION`。它不是算法预测成功，也不能通过选择非零 test bank 来补救。冻结 all-bank 结果和 active-only 结果并列。

### 5.2 需要实现的核心比较

| ID | 方法 | 目的 |
|---|---|---|
| C0 | C 恒为零 | 必需基线 |
| C1 | 原 `d→T` 规则预测两个候选再相减 | 判断目标重设是否有价值 |
| C2 | `e→C`，全 Q_contrast 普通 ridge | 最简单的直接差异模型 |
| C3 | `e→C`，全 Q_contrast 协方差感知 ridge | 判断观测协方差处理的价值 |
| C4 | 更新 PCA/随机方向的非零 r + ridge | 方向选择基线 |
| C5 | 协方差感知拟合后，未加权响应谱取 r | 经典低秩基线 |
| C6 | 同 C5，但方向谱同时保留群体 pX/v 响应 | 检验目标相关方向的增量 |
| D-DIRECT | 直接测 held-out 候选差异，不拟合 surrogate | 同访问权限下的成本与精度基线；不能省略 |
| D-KNOWN-EVENT-LOGP | 在原16动作toy中对正确/两个非法动作直接评分 | SAMPLE_AND_LOGP下更强的精确pX/v基线 |
| O-J | 完整 Jacobian at origin / at baseline 两种 | 非线性与一阶可预测性参考 |

C5/C6 是透明的两阶段投影估计器，不宣称全局最优的广义低秩回归解。经典 RRR、GLS、PCA、importance sampling、CRN 都不是新颖性主张。

### 5.3 C3 的精确定义

对单题 i，将 m 个 fit contrast 的三维 Helmert 响应 row-major 展开为 \(c_i\in\mathbb R^{3m}\)，A 的行是 \(e^\top Q\)。令：

\[
X_i=A\otimes I_3,
\qquad
\widehat\beta_i=\arg\min_\beta
(c_i-X_i\beta)^\top\widetilde\Omega_i^{-1}(c_i-X_i\beta)
+\lambda_{\rm ridge}\|\beta\|^2.
\]

用 whitened QR/SVD 求解，禁止直接求大矩阵逆。\(\widetilde\Omega\) 为 shrinkage 后协方差。跨题独立抽样时按题分块；群体指标由固定权重汇总，不额外假定不同接口语义独立。

- 原 O-IND 协方差由付费计数估计。只为协方差稳定化使用 Jeffreys 伪计数 1/2；响应点估计不加伪计数。
- LR/CRN 从同一 packet 的联合贡献向量估计协方差。不能把各 candidate 的 marginal SE 拼成独立对角矩阵。
- shrinkage 固定候选 eta=[0.01,0.1]，对角+谱地板来自观测尺度；只能在开发数据选。exact 协方差仅在 oracle 消融使用。
- 先合并确定相同推理策略与冗余对比；经验奇异零方向不能被解释为已知零噪声。
- ridge 使用 whitened 设计最大奇异值平方进行尺度化，alpha=[1e-5,1e-2]。OLS/ridge 原比较保留旧尺度，报告差别。

### 5.4 C5/C6 方向选择与维数

将每题全 Q 拟合系数拼成 \(\widehat B\in\mathbb R^{216\times k}\)。

C5：对 \(\widehat B\) 的右奇异方向取前 r 个。C6：构造固定目标矩阵

\[
\mathcal T(\widehat B)=
\begin{bmatrix}
\widehat B/\sqrt{216}\\
L_{XV}\widehat B/\sqrt{12}
\end{bmatrix},
\]

其中 \(L_{XV}\) 是把逐题 Helmert 差变成六群体 pX 和 v 差的 12×216 线性矩阵，群体内固定题目权重归一化。对 \(\mathcal T(\widehat B)\) 的右奇异向量取 r。完整与群体目标的两个块各有固定整体权重，不用测试真值的方差来放大某类结果。

在所选参数方向上重新用**同一种** OLS/GLS 目标拟合响应系数。r=[1,2,4,FULL]，effective_r=min(r,k)。始终保留 r0 为 C0 基线，不能让无合格方法自动使其他消融也变成 r0。

语义目标权重、超参数、r 和 observation method 全部由开发集冻结。C6 若不优于 C3/C5，就不要包装成有效新表示。

## 6. 新增理论核验与诊断，必须先于核心试跑

以下必须在无模型或小解析例子上验证：

1. T=B+C 的向量恒等式；同一组变量相减严格取消共同锚点噪声。
2. 候选对比的联合 covariance，包括共用 baseline、exact alias 与共同 packet。
3. 主事件概率三自由度、Helmert 回转；条件量的边界状态。
4. contrast Taylor 界；e=0；同范数但相反向量不能判为同策略。
5. 小差异 LR 无偏和二阶矩；mixture 的 |w|≤2；缺支持时拒绝。
6. CRN 两侧边缘正确；固定排序；类别重排敏感度只作仪器诊断，不修改主数据排序。
7. 两个高能量共同变化、一个低能量关键差异的人工例子：总体 r1 好不代表 C 好。
8. 协方差不确定性：样本协方差为零不应变成置信区间零宽。
9. GLS 在球形协方差下等价普通 ridge；数值秩与 shrinkage 不应制造新信息。
10. 同一测试 packet 不得用于 fit、选维、控制变量系数估计和性能判定。

包内 `reference/` 是独立公式参考，不替代真实代码集成。必须自己加入回归测试。

## 7. 阶段安排与数据预算

### N0：源码、原件和旧结果桥接——立即执行，零新增训练

- 核验附件 manifest、父提交、原配置；补取其余59条 M2 原轨迹与缺失预测文件，逐文件核对原哈希。
- 本地无原件时输出 `MISSING_INPUTS.json`，继续独立数学和有数据的开发诊断；不得把缺59条改成只用seed301却宣称全量。
- 旧 locked-test 已经被分析：本轮全部旧数据都是公开开发/复现资料，不作为新的独立泛化证据。保留原 seed role 字段，再增加 `v2_role=legacy_diagnostic`。
- 修复七项退化消融：固定非零 r1/r2/FULL，先 exact 后 n64，报告 predicted_difference_norm、actual_rank、prediction_hash，确认处理真正改变了结果。
- 重算 C 目标、各事件信号、alias 和 q 的分母，不预先挑好看的 group。

输出 `N0_AUDIT_zh.md`、缺项表、paired_contrast.csv、nondegenerate_ablation.csv、source_binding.json。

### N1：观测估计器资格——不新增主训练

主比较只做原开发 fit seeds 101..106，两臂、三个锚点。smoke 从101/一个臂/anchor8开始，文件缺失则使用独立解析见证，不假造真实轨迹。

- 每个固定候选：O-IND、O-CRN、O-LR-ORIGIN、O-LR-MIX。
- 主 n64，观测噪声副本8个（独立 seed），两种主/次 contrast。
- n16/256 只对冻结的主协议做稳健性，不以更多观测“追显著”。
- 另在同一候选精确16动作表上计算各估计器的理论均值和 covariance，放隔离 oracle 报告；这能低成本验证噪声模型，不能提供给 finite fitting。
- 若原 npz 只存四类概率但有模型参数，用旧 toy 的冻结 forward 重算动作概率；记录为新增 CPU forward 计费，不是新的训练；禁止由四类p任意拼造16动作分布。

评价：偏差、RMSE、精确方差/经验方差比、事件支持、joint covariance、估计区间宽度、完整调用成本。先看估计器，再看模型，避免把观测失败归罪于 rank。

### N2：精确目标与有限观测建模——逐步，不做完整笛卡尔积

2A：exact 目标上比较 C0..C6，r1/2/4/FULL，fit-bank预算1/2/4/8。用途是识别本数据是否有可用的差异结构；不推荐实际运行时使用exact。

2B：所有 N1 合法估计器先只接 C2 的 FULL-ridge，n64。按旧 selection seeds 201..204 的 pX/v 双目标NRMSE与成本选最多2个观察方法。淘汰的也保留完整结果。

2C：对最多2个方法比较 C2..C6 的r网格和固定小alpha/eta网格。C1原规则、C0、D-DIRECT全程保留。

2D：隔离误差——仅开发集，三种诊断：

- exact U + finite 系数；
- finite U + exact 系数；
- finite U + finite 系数。

另加有限观测协方差、exact 协方差对照。报告子空间夹角/投影矩阵距离和响应误差，不以某根奇异向量符号作为方向差异。

2E：数据预算消融使用非零冻结规则。n16/256与fit1/2/4只改一项；无激励节点保留，不用手选银行替代。没有合格模型仍输出best_nonzero_diagnostic，不能自动把消融降成r0。

### N3：冻结选择和独立性——未通过则到此结束

在旧开发资料上选择最多2个可部署候选，加 C0、C1、C3/FULL、D-DIRECT 等必要基线。写出 `MODEL_SELECTION_LOCK.json`、确切哈希、成本与精度预期。

进入新CPU独立验证的必要条件：

- n64 或预设 n256 中至少一个**有限观测**配置在 pX、v 候选差异上各自 NRMSE<0.75；不是 exact 或某个单群体达标。
- 有非零可辨识的fit方向，且非alias评估覆盖；预测不只是输出零。
- 相对C0/C1有稳定开发集改进；无法分辨时记 `INCONCLUSIVE`。允许继续研究SAMPLE_ONLY的观测资格，但SAMPLE_AND_LOGP若不能胜过D-KNOWN-EVENT-LOGP则不得进入“实际节省成本”结论。
- 计入fit+score+label成本，在声明的访问条件下，存在至少一种明确计费比例下与直接测量有交集；如果只有特定比例有利，明确标记条件。
- 新数据投影资源不超过本轮总预算。

以上是开发门禁而非正式统计结论。方法、rank和阈值在新测试生成前冻结；未过门禁不自动扩大样本量或运行更大模型。

### N4：新的 CPU 验证——有条件执行，固定新增规模

保留原初始化7001、数据seed20260915、两主奖励臂、64步、B4/K8、anchors8/24/40、14banks×3候选。

- 独立误差校准 RNG seeds：501..506（6个）。
- 新 locked-test RNG seeds：601..610（10个）。
- 检查这些seed与全部现有本地运行清单是否冲突，冲突则停止并报告，不自行换成更好看的seed。
- 16seeds×2arms=32条CPU轨迹；2,048主更新+4,032分支更新=6,080次Adam；108,544次训练/分支动作采样。
- 这些变化只验证训练RNG泛化，仍不验证跨数据、初始化或VLM。
- 每个 test 锚点允许付费 fit-bank 校准，因为这是局部模型使用协议；test evaluation banks10..13的语义结果必须在预测封存后才用于评分。
- 测试只运行冻结的方法、rank/alpha/eta、n64主及16/256冻结稳健性。不再在test选择最好r。
- CPU实施保留历史明确零优势的Adam行为，不按伪目标变更奖励。

6个校准seed不足以声称95% seed级 distribution-free conformal覆盖：max分位所能支撑的分辨率有限。这里只报告独立经验覆盖与校准不足；严格95%保证需要另行扩大校准单元数，当前不授权。不得把众多同seed锚点当独立校准样本凑数。

### N5：有限短窗口与成本——仅当差异模型通过后

主报告到 N4 已足够决定观察模型。可选但冻结的小补充：对已完成主轨迹只读回放H=[1,2,4]，分别使用主跟踪头和差异头；差异头不可直接外推成时间跟踪器。

窗口所有中间点评分；不能只看终点抵消。没有可验证的新path误差带，不重复画PATH/NET相同预测来宣称触发器不同。估计L等仍是oracle或经验项，不能给真实VLM安全认证。

输出若显示付费校准不能摊薄，则结论 `NO_COST_FEASIBLE_SURROGATE`；直接差异测量胜出也属于有效研究决定。

### N6：最终决定与真实资料只读映射

汇总每个目标在各预算下的 `(estimator, bank budget, n, k, r, noise, prediction error, cost, domain)`。旧 Qwen 资料只读盘点所需字段；没有新GPU模型调用，没有补原点采样，没有启用旧S2/在线SSVC。

## 8. 统计指标与验收尺度

### 8.1 主指标同时看三个层次

A. 完整逐题四事件 contrast MAE/NRMSE。  
B. 六群体pX与v各自的 pooled NRMSE、MAE、误差q95和最差群体。  
C. 主/次候选对一致性、alias零响应与出空间比例。

不得把pX与v合成一个可能被v掩盖的主指标。所有NRMSE用原始真实差异能量作分母，floor只防0除，不能加大floor制造达标。能量为零返回 `NO_SIGNAL`。

### 8.2 分辨率而不是笼统1个百分点

- 预设工程分辨率 `delta_ref=0.00025`（0.025个百分点）；它依据既有问题尺度设置，是本轮测量设计目标，不是任务风险阈值。
- 同时完整报告 delta=[0.0001,0.00025,0.0005,0.001] 的分辨率曲线；只有主delta用于指定判断，不能挑一个通过的secondary后称主结果成功。
- 主工程目标：六群体pX/v各自NRMSE≤0.75，绝对误差q95≤delta_ref/2。
- 对真差异|C|≥delta_ref的方向判别，最低20个不同anchor-bank单元且覆盖至少5个test seed才作描述；支持不足返回 `INSUFFICIENT_EFFECTIVE_CASES`。不为凑数筛题/放大奖励。
- 对近零差异，使用 `MEANINGFUL_POSITIVE / MEANINGFUL_NEGATIVE / WITHIN_TOLERANCE / UNKNOWN`。点估计为0不代表无效应；区间完全在[-delta,delta]内才可称区间支持的近等价，且须说明区间类型。
- NRMSE达标但区间太宽，分别标 `PREDICTION_PASS` 与 `RESOLUTION_FAIL`，不合并成安全通过。

### 8.3 推断单位与选择

5000次训练seed聚类bootstrap；同seed两臂、全部锚点、候选和噪声重复一起采样。显示每seed结果与配对方法差值。选择前开发比较标posthoc；新测试只给冻结选择比较。若有多个正式比较，固定比较族并做Holm，不能把数千个群体行当独立p值来源。

观测估计器的固定策略方差实验可按独立观测replica分析，但不要与跨训练seed泛化混称。单题区间、群体区间、多群体同时区间、prediction interval分别命名。

对alias推理结果只付一次成本、只算一次观测；不能复制同一结果来缩窄区间。不同proposal和不同采样模式的结果不得未经加权混合。

## 9. 信息权限与泄漏防护

实现三层接口：

1. `ToyWorldService`：持有完整策略/精确16动作概率，仅提供采样与指定动作logp，计费；可以使用精确概率生成模拟观测，但不返回隐含整表。
2. `ObservationPacket`：模型真正能够获得的样本、标签、logp、proposal与协方差估计。
3. `OracleAudit`：完整p、J、真实noise covariance、action总枚举，仅供冻结预测后的评分与明确oracle诊断。

Predictor接收 `FitView` 和查询e，不能持有模型对象、文件路径到oracle、或通过闭包再次调用评估。保存feature schema/fit sample IDs/packet IDs/selected rank/预测哈希。测试构造一个trap oracle对象：访问就抛错。

从候选完整动作密度计算权重是允许的付费**标量logp查询**，不是直接泄漏事件总概率。仅CPU模型的有限动作小使该查询很便宜，真实VLM需对整段序列评分，迁移时不得忽略其成本。

## 10. 成本定义

每次物理操作记录：

- generated_actions、generated_sequences（真实VLM本轮为0）；
- sample_policy_evaluations、cross_policy_action_scores、verified_labels；
- exact_probability_calls、Jacobian/VJP_calls（oracle单列）；
- optimizer_updates、fit_seconds、score_seconds、sampling_seconds、I/O、RAM峰值和新增磁盘字节。

同样本、同题、同推理指纹的重复评分可以cache；它节约物理成本但不增加样本信息量。不同模型参数对应相同action label可复用标签，但logp不能复用。

提供三种成本视图：

1. 737参数CPU真实时间（不能称为VLM节省）；
2. 生成/评分/标签向量计数；
3. 场景化成本曲线 `score_cost/generate_cost ∈ {0.05,0.25,1.0}`，明确假设而非实测。

D-DIRECT按同一任务、相同目标精度和同样观测选择计算成本。把24个校准策略全部测过再预测4个测试策略，未必节约；必须算break-even，不能只报推理阶段成本。

限制：单线程CPU，RAM≤8GiB，新结果≤3GiB，其中临时文件≤1GiB；全部新CPU阶段预测总walltime≤4小时。先smoke计时外推，超过则 `RESOURCE_REVIEW_REQUIRED`，不偷减科学样本、不重跑失败job直到成功。拆分测试scratch，每批清理仅本次已登记临时目录；保留失败摘要与哈希。原件只读不算新增预算，但禁止复制大模型权重。

## 11. 代码清单和拟实现CLI

新增包 `src/modeling_contrast/`：

| 模块 | 职责 |
|---|---|
| `protocol.py` | 严格配置、seed与预算、禁止GPU和在线状态 |
| `parent.py` | 旧原件与alias/schema/hash只读验证 |
| `targets.py` | T/B/C、对比矩阵L、Helmert/群体线性映射 |
| `observations.py` | 四类观测估计器、独立RNG与标量logp计费 |
| `covariance.py` | multinomial、paired与LR joint covariance、shrinkage |
| `estimators.py` | GLS/ridge、非零低秩投影、参数方向与谱 |
| `selection.py` | 非退化gate、rank/alpha锁定、无信号状态 |
| `evaluation.py` | 双目标指标、成对seed推断、分辨率和覆盖 |
| `cost.py` | cache、物理账本、oracle与实际成本分离 |
| `collect_fresh.py` | 条件执行新CPU验证，完整原始文件输出 |
| `report.py` | 事实/设计/推断分开输出 |
| `cli.py` | 以下命令及不可写旧数据的路径检查 |

CLI是Codex需要实现并实测的契约，不是现已存在命令。执行目录为 `ssvc_flow/`，使用经核验CPU环境的 `$PY`：

```bash
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${PY:?请设置经核验CPU环境的python}"
"$PY" -m src.modeling_contrast.cli audit-parent --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/N0
"$PY" -m src.modeling_contrast.cli smoke --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/smoke
"$PY" -m src.modeling_contrast.cli observation-study --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/N1
"$PY" -m src.modeling_contrast.cli fit-study --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/N2
"$PY" -m src.modeling_contrast.cli freeze-selection --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/N3
# 仅前述科学、原件与预算门禁通过后：
"$PY" -m src.modeling_contrast.cli verify-fresh --config configs/modeling_contrast/protocol.json --selection runs/modeling_contrast_v2/N3/MODEL_SELECTION_LOCK.json --out runs/modeling_contrast_v2/N4
"$PY" -m src.modeling_contrast.cli report --config configs/modeling_contrast/protocol.json --root runs/modeling_contrast_v2 --out docs/modeling_contrast/results
```

`--resume`须绑定source/config/data/selector/packet hash，no-clobber。并发writer互斥；中断后不把新随机样本覆盖旧样本。暂缺原件的命令必须输出明确BLOCKED理由，但数学与无依赖阶段继续。

## 12. 输出契约

至少输出以下文件：

- `REVISED_IDEA_zh.md`：目标、表示、观测与维数的新关系；尚未验证的内容明确标记。
- `N0_AUDIT_zh.md` 和 `MISSING_INPUTS.json`。
- `OBSERVATION_COMPARISON.csv`：estimator、n、support、bias、variance、成本。
- `CONTRAST_PREDICTION_BY_SEED.csv`、`NONDEGENERATE_ABLATIONS.csv`。
- `MODEL_SELECTION_LOCK.json`，测试前生成。
- `MODELING_DECISION_V2_zh.md`：每种目标的最终选择、失败范围与停止建议。
- `COST_FRONTIER.csv`、`MEASUREMENT_COVARIANCE_AUDIT.json`。
- `REAL_DATA_REQUIREMENTS_zh.md`：只列将来最小真实验证需求，不启动。
- `IMPLEMENTATION_REPORT_zh.md`、`machine_readable_readiness.json`、测试原始记录及源码/数据manifest。

status维度分开：`engineering_status`、`observation_status`、`prediction_status`、`resolution_status`、`cost_status`、`independent_validation_status`。不能以一个PASS覆盖全部。全套旧回归存在缺fixture时，保留missing-fixture错误与新模块测试结果，不删用例或谎称完整PASS。

原始结果保存压缩NPZ、JSONL与源码快照，全部finite arrays `allow_pickle=False`。完整60条旧轨迹与新32条的可用性分别核对；打包时用coverage manifest精确标明未打包大文件，不再把汇总包称作含全部原件的重跑包。

## 13. 实验后如何作研究判断

- **直接差异建模优于总体相减，且有限观测有效**：支持目标相关的概率响应建模，但还不证明内部机制被唯一识别。
- **只有改善观测器有效，低秩不增加收益**：优先采用直接差异测量，承认当前surrogate没有必要。
- **GLS与简单全Q已经足够**：它们是经典方法；研究增量需另行证明，不给新缩写包装。
- **只在exact或很高打分预算有效**：保留结构诊断，停止进入在线控制。
- **真实差异普遍小于预设工程分辨率**：记录任务对当前控制问题信号不足，不通过筛题或放大奖励追求负面现象。
- **部分方向在预算下不可辨识**：输出目标/状态/预算依赖的可辨识范围，不声称模型动态本质维数已确定。

最终要回答“用哪些观测，能以多少成本分清哪些变化”，而不是必须交付一个更复杂的网络或预先指定的秩。
