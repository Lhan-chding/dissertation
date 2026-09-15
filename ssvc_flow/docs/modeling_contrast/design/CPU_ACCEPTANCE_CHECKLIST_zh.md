# CPU 验收清单

对每项记录命令、输入/源码hash、通过/失败/阻塞原因。不能用“代码存在”替代实际执行，也不能以新模块通过覆盖旧回归缺项。

## A. 原件与范围

- [ ] 父提交、附件sha256、原protocol和数据seed一致。
- [ ] 60条历史轨迹逐项检查；只有1条完整时输出精确缺失清单。
- [ ] 不自动重跑缺失历史轨迹，不改其run ID或哈希。
- [ ] 独立worktree；不改现有followup线上源码、计划、队列或状态。
- [ ] 全新输出目录，no-clobber，路径不能越过指定结果根目录。
- [ ] 对新模块封禁GPU/模型下载/网络训练入口。

## B. 语义表示与目标

- [ ] X/S/W/I严格互斥，映射来自原定义；保持action顺序与身份。
- [ ] Helmert往返和零和contrast保持正确；条件分母为零返回未知。
- [ ] T=B+C逐元素成立；C使用同bank两候选，不串bank。
- [ ] 共同锚点噪声严格抵消；共享baseline仍引入相应covariance。
- [ ] joint0不是未更新原点；绝对响应和候选差异分别命名。
- [ ] alias只共享推理结果，不能合并不同Adam历史。
- [ ] 相同范数反向向量被识别为不同；完整状态fingerprint与参数hash分别记录。

## C. 观测与统计

- [ ] O-IND在单策略独立计数下的均值/方差匹配解析值。
- [ ] O-CRN配对分布两侧边缘正确；不根据类别排序动作。
- [ ] CRN存在降方差和无改善的见证，不能实现成固定返回有益。
- [ ] 同策略CRN差异为零；标签不同但共享随机数不叫自然事件运输。
- [ ] LR在足够支持时均值等于真差异；缺支持明确拒绝。
- [ ] LR将两端点概率按同一proposal和同一draw相减，不能混用不同packet。
- [ ] mixture源选择是iid coin；|w|≤2的见证成立。
- [ ] LR原始事件估计的质量残差与Helmert控制变量视图区分。
- [ ] 非有限权重失败，不偷偷clip、winsorize或删行。
- [ ] 联合sample covariance使用所有contrast贡献，不能仅拼marginal SE。
- [ ] exact covariance仅oracle；响应不加伪计数，covariance平滑有单独配置。
- [ ] 未观测X、经验方差0、不充分支持都不产生“已证安全”。
- [ ] known-event logp基线能在原toy用正确动作及两个非法动作得到精确pX/v。
- [ ] 同一访问权限下公平比较；真值评分不能被人为排除。
- [ ] 标签复用、动作logp缓存和完整forward成本分别记账。

## D. 模型和维数

- [ ] Q只用fit e；查询e不参与Q/SVD/超参数选择。
- [ ] OLS/GLS球形特例一致；row-major vec与kron契约有显式测试。
- [ ] covariance奇异、alias冗余、ridge、shrinkage均测试。
- [ ] 非零r消融不会因主选择失败而自动变r0。
- [ ] C0明确为零基线；无激励节点返回NO_CONTRAST_EXCITATION。
- [ ] C1沿用原规则，不在本轮测试上重新“修好”。
- [ ] C5/C6矩阵维度、右奇异方向、重新拟合及群体线性映射正确。
- [ ] 原216响应坐标与k/r分开；高总体拟合与低差异拟合反例能重现。
- [ ] exact U/finite系数与finite U/exact系数仅诊断，不参与有限观测主排名。
- [ ] raw预测/投影预测分别保留；不靠投影掩盖负概率或误差。

## E. 数据分割与推断

- [ ] 旧所有seed均标legacy_diagnostic，原role保留。
- [ ] 新501..506/601..610未被先前使用；冲突不自行改号。
- [ ] 同seed所有臂/锚点/bank共同bootstrap，noise replica不算新轨迹。
- [ ] 新测试之前封存方法与rank/alpha/eta/阈值/代码hash。
- [ ] 测试语义响应在预测落盘前不可访问；trap oracle测试实际运行。
- [ ] n16/256不重新用test选规则。
- [ ] near-zero信号报告unknown/equivalence边界，不任意扩大NRMSE floor。
- [ ] 方向指标最小独立单元与seed数不足时不输出高置信准确率。
- [ ] 校准seed仅6个时拒绝声称95%分布无关seed级prediction interval。
- [ ] 点态、群体、同时区间、预测区间和工程门槛明确区分。

## F. 工程与成本

- [ ] smoke实测walltime/RAM/磁盘并外推，超预算停止。
- [ ] 所有策略查询、生成、标签、评分、校准、oracle、拟合均有计数。
- [ ] 32条新CPU轨迹预算=6080次Adam、108544动作采样；不包含额外测量动作，额外测量单独估算。
- [ ] old-training evaluator/model状态恢复不被新测量污染。
- [ ] 随机数源：训练/fit测量/test测量/noise独立，并有可复现索引。
- [ ] 故障恢复保留已有结果，身份变化拒绝resume。
- [ ] 全suite用分批临时目录，不能因磁盘满不断无说明重跑。
- [ ] 不删旧test来获得green；缺fixture列清单并限定engineering status。
- [ ] 打包coverage manifest列出收录和未收录原件；所有npz allow_pickle=False。
- [ ] 最终结果通过重算脚本独立从raw数据恢复，表格指向具体源文件hash。

交付 `CPU_ACCEPTANCE_RESULTS_zh.md`，逐条填入 evidence_path 与实际状态。包内参考32项通过不替代此清单。
