# M0 数学审计

事件顺序固定 X/S/W/I。存储原始四计数；Helmert 是正交线性坐标，四概率往返与距离保持；
原概率和 Helmert 对同一更新矩阵的线性回归数值等价。v/q/s 仅作解释。
条件分母为零返回 null：精确概率为零标为 UNDEFINED_ZERO_MASS，
有限样本零分母标为 UNOBSERVED_CONDITION。零计数不证明结构零。

固定策略的多项似然可写为 v^nV (1-v)^nI q^nX (1-q)^nB s^nS (1-s)^nW。
内部点 Fisher 由 D^T diag(1/p) D 得到所给对角式。CP 区间按各自实际分母计算；逐个区间只称点态。
同一四计数向量的四类别加 v/q/s 共七区间使用 alpha/7 的 Bonferroni 分配；
这不是跨 checkpoint 或跨题面板的自动同时覆盖。

对 affine-softmax，令行向量 a_j、均值 μ=Σ p_j a_j、事件对比系数 |w_j|≤1。
单类 Hessian 为 p_j A^T[(e_j-p)(e_j-p)^T−(diag(p)−pp^T)]A。
第一矩阵谱范数≤2，第二矩阵谱范数≤1，因此按 Σ |w_j|p_j≤1 得到 ||∇²f||≤3||A||²。
Taylor 余项≤L||D||²/2，再将起点梯度分成 U 内系数误差及 U 外遗漏，得到 E_net；
应用三角不等式逐项放大得到 E_path。因此每步净位移也能检查中途退化，路径界通常更保守。
M1 用已知 L 同时验证两个界；该全局保证不转移给共享 tanh toy 的经验带。

rank(JQ) 只涉及固定起点、输出表和校准子空间，最优秩 r 的谱误差为下一奇异值。
复制列不增加数值秩。仅给投影和正交残差范数不能识别未覆盖方向的语义响应。
误差向量应分别保存，范数不能相减解释为贡献。

数值结果与固定种子采样原件见 math_results.json、coordinate_reconstruction.csv、
coordinate_counts.npz。
工程测试、Adam 状态恢复与不可用观测访问的测试由完整仓库测试记录共同证明。
