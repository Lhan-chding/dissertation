# V3 基线、公式与来源核验范围

核验日期：2026-09-15。这里登记实际实现、访问权限与核读范围，不构成新颖性证书。

## 来源状态

| 来源 | 本轮实际核验 | 使用边界 |
|---|---|---|
| [Wickramasuriya, Athanasopoulos, Hyndman, JASA 2019, 114(526):804–819](https://southwest.robjhyndman.com/publications/mint/) | 作者公开页面的作者、出版信息、摘要；未核对全文 | 协方差约束协调的近邻；V3 的零和修正受其思想启发，不是 MinT 完整复现 |
| [Wagenmaker & Jamieson, COLT/PMLR 125, 2020:3487–3582](https://proceedings.mlr.press/v125/wagenmaker20a.html) | 正式会议页面的作者、题名、摘要、出版信息；未核对全文 | 主动选择激励以辨识响应的近邻；不移植其线性动态系统理论保证 |
| [Xia et al., LESS, ICML/PMLR 235, 2024:54104–54132](https://proceedings.mlr.press/v235/xia24c.html) | 正式会议页面的作者、题名、摘要、出版信息；未核对全文或实现 | 低维梯度、Adam 感知影响估计的近邻；本轮没有复现 LESS，不将“低维 + 优化器”认作新机制 |
| [Owen & Zhou, JASA 2000, DOI 10.1080/01621459.2000.10473909](https://www.tandfonline.com/doi/abs/10.1080/01621459.2000.10473909) | 本轮未打开；仅从交接包转录文献身份 | 混合重要性采样与控制变量的来源线索；不可写为本轮已核读或完整复现 |
| AutoJudger / Dark Room / GDPO / Ramdas et al. anytime inference | 交接包 `design/SOURCES_AND_NOVELTY_zh.md` 的待核对清单；本轮未打开其原文 | 本轮未实现其完整算法，不宣称超越；真实控制或顺序推断声明前需按引用版本核对全文 |

## 统一符号与权限

四事件顺序为 X/S/W/I。`z` 是一次原始带符号四维贡献，`m = 1ᵀz`；
`E` 的行是真实参数增量，`Q` 是其未中心化行空间基，`A = E Q`；`Y` 是付费校准响应。
所有 baseline 的数据生成、评分、标签、候选更新、校准、拟合与 I/O 成本应分别计费。
原始四维输出始终保存。`HHᵀ` 对真实零和差值无损，对含质量残差的观测是统计投影。

| 实现基线 | 实际公式或选择目标 | 对应代码 | 信息权限与完整复现状态 |
|---|---|---|---|
| RAW4 | `mean(z)` | `src/modeling_v3/observation_geometry.py:estimate_geometry` | 付费原始贡献；无 oracle；当前公式直接实现 |
| EQUAL_ZERO_SUM | `z - (1/4) 1 m` | 同上 `correction`、`fixed_coefficient` | 同一原始贡献；当前公式直接实现；不宣称 MinT 复现 |
| DROP_W | `z - (0,0,1,0) m` | 同上 | 同一原始贡献；X/S/I 保留，W 吸收残差 |
| PRESERVE_XI | `z - (0,1/2,1/2,0) m` | 同上 | 同一原始贡献；X/I 保留，S/W 分担残差 |
| 独立 pilot 协方差修正 | `b = Σ1/(1ᵀΣ1)`，main 使用 `z - bm` | `src/modeling_v3/covariance_pilot.py`；`observation_geometry.py` | 系数只从独立 pilot 取得；简单基线使用同总支出样本，另列相同 main 子集；控制变量思想的具体应用 |
| 独立 pilot 收缩修正 | 先以预先固定收缩量稳定 pilot 协方差，再计算 `b` | 同上 | 系数不得读取 main/query 真值；边界触发简单 fallback；不是全文级复现某个收缩估计器 |
| 二折交叉拟合 | 每折系数在另一折拟合，合并两折估计 | 同上 | 两折样本身份分离；不把交叉拟合后的方差误作独立两项之和 |
| KNOWN_COV_ORACLE | 已知协方差代入 `Σ1/(1ᵀΣ1)` | `observation_geometry.py` | 仅评估 oracle；不能进入付费主排名 |
| ORIGIN 差值重要性采样 | `(πu−πb)/ρ`；带符号稳定指数差 | `observation_geometry.py:stable_origin_weight` | 原点采样动作及该动作各 policy logp；全支持须由采集端证明 |
| 两端点等权 MIX | `2 tanh((logπu−logπb)/2)` | `observation_geometry.py:stable_pair_weight` | 原始动作、混合分量 RNG、两端点 logp；保持界 2，不裁剪概率比 |
| 直接测量 / 精确已知事件评分 | 付费差值；事件可完整枚举时对全部合法事件动作求质量 | 旧 `modeling_contrast/observations.py:measure,known_event_logp`；V3 `vlm_observation.py` | 强测量基线；VLM 单一规范字符串不等于完整语义事件，枚举完备性未证明时不得标精确 |
| FIRST / 分层随机 | 原顺序；只按冻结身份分层随机抽取完整 bank | `src/modeling_v3/bank_selection.py:select_banks` | 无 query 响应、无 fit 标签；随机设计 seed 不是训练 seed |
| LARGEST_NORM | 选 `||E_bank||²_F` 最大块 | 同上 | 读取全部候选真实增量，候选池几何成本必报 |
| BLOCK_PIVOT_QR | 贪心最大化 `||E_bank−E_bank Q Qᵀ||²_F` | 同上 | 一次选 bank 内所有对照；受激励/设计思想启发，不是 Wagenmaker–Jamieson 算法完整复现 |
| BLOCK_LOGDET | 贪心最大化 `logdet(M+A_bankᵀ A_bank)−logdet(M)`，`M` 含冻结正则 | 同上 | 相同候选池几何；绝不根据未见 query 响应选 bank；D-opt 思想基线 |
| ZERO | 恒零预测，统计对照 | `src/modeling_v3/response_models.py:fit_response_model` | 无响应信息；零向量/没有激励不自动获得精度认证 |
| FULL_RIDGE | 最小化 `||A B−Y||² + λ||B||²`；相对 `alpha` 乘设计最大奇异值平方得到实际 `λ` | 同上 `_ridge` | 只读 fit 实际增量和付费响应；无截断 `r=k` |
| FULL_GLS | 在 prompt 内完整联合协方差白化后 ridge；显式协方差特征值下界 | 同上 `_fit_coefficients`、`_whiten` | 保留共享基线/对照的协方差；不把联合协方差替成各项独立方差 |
| PCA / RANDOM_Q | `Q` 前 `r` 个未中心化更新方向；或 span 内冻结随机正交方向 | 同上 | 方向只读取 fit 更新几何；主方向比较须 `0<r<k` |
| RESPONSE_SVD | 对全模型系数 `B` 做 SVD，取左奇异方向，再在其上拟合 | 同上 | 只读取付费 fit 响应；`r=k` 与同目标 full 模型等价 |
| GROUP_WEIGHTED_RESPONSE | 对 `B Gᵀ` 做 SVD；`G` 为冻结群体 X/v 映射 | 同上、`group_xv_map` | 群体映射事先确定；不读取 query 标签调整权重 |
| RBF_RIDGE | 原点锚定核 `exp(−γ||a−b||²)−exp(−γ||a||²)−exp(−γ||b||²)+1` 上 ridge | 同上 | 仅非线性诊断，不混入低秩方向的单机制归因 |
| 覆盖/拒判 | `ρ = ||e−QQᵀe||/||e||`；杠杆值 `aᵀ(AᵀΩ⁻¹A+αI)⁻¹a` | `src/modeling_v3/coverage.py` | 几何量不是概率区间；阈值须 development 冻结，测量未分辨或无激励保留 UNKNOWN |

## Q0 与经典等价关系的检验

`src/modeling_v3/source_audit.py` 只从哈希绑定的原始 packet 与预测数组重放。它复用旧独立复算中的群体指标公式，强制从原始付费样本重建 raw/Helmert/covariance，列出 all/active/未激励的平方和，比较匹配 C3/C5/C6 的 `r=k` 单元。

旧文件未保存的 Q/U/C、逐样本动作、独立 pilot/main/ref 角色保持 `MISSING_INPUTS`；不从旧摘要反造，也不把新重拟合矩阵标作旧原件。N1 若有不同 parent，必须明确提供 `q0.legacy_parent_root`。正文中的数学实现与原件审计都不证明在线 SSVC 有效。
