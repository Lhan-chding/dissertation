# CPU 与服务器接入验收清单

每项给出对应测试、命令、退出码、输入/源码hash；没有执行的项标NOT_RUN。参考包的数学测试不替代仓库测试。

## A. 事件与观测（必须）

- [ ] X/S/W/I互斥且穷尽；动作截断/EOS/解析失败不丢样本。
- [ ] 精确零和向量的Helmert正逆无损。
- [ ] noisy raw的质量残差单独保存；无静默等权投影。
- [ ] EQUAL总L2不增，但单事件可以更差的确定性见证。
- [ ] DROP_W、PRESERVE_XI逐次零和，X/I保持不变。
- [ ] stable MIX权重与概率直接公式一致；大正/负log差、近零差、相等策略均通过。
- [ ] 两策略MIX |w|≤2；三策略MIX≤3，两个公式不能混用。
- [ ] ORIGIN支持不足/非有限权重不静默clip。
- [ ] exact统计仅评估器可读，估计器不能持有全概率表。
- [ ] 每prompt每candidate每token的身份一致，alias有记录且不是新独立样本。

## B. 协方差与pilot（必须）

- [ ] b*和oracle协方差闭式与枚举一致。
- [ ] 已知b下Cov公式、额外v_M(b−b*)项一致。
- [ ] pilot与main RNG/样本IDs完全分离。
- [ ] 同成本简单baseline使用pilot+main全部样本，配对子集比较另列。
- [ ] finite pilot的Oracle误差项导致优势可能消失的见证。
- [ ] covariance均值/每draw区别清楚，不多除或少除n。
- [ ] v_M=0、空X、空S、单样本/不够估协方差的路径明确。
- [ ] shrink、L1限制与fallback只使用pilot，不依据heldout结果。
- [ ] 全部候选联合相关性和共享baseline/alias按贡献账本计算。
- [ ] 二折均值可无偏但折间相关，区间必须重做完整cross-fit。
- [ ] 修正前raw总质量残差单独审计，持续偏离0不被投影掩盖。
- [ ] 无负概率差裁剪、SNIS冒充OIS或掩盖支持问题。

## C. 方向覆盖与建模（必须）

- [ ] e由同完整原点两次真实Adam后参数相减，不是梯度替代。
- [ ] 相同范数、相反方向的见证；hash不同不自动说明差异有意义。
- [ ] Q未中心化、正交；rho/e_perp精确重建，e=0不除零。
- [ ] 低rho/高leverage见证；不能只用rho宣称可靠。
- [ ] 相同fit响应、不同未覆盖query响应的不可辨识见证。
- [ ] FIRST/分层随机/最大范数/QR/LOGDET都真正改变至少部分设计。
- [ ] 一bank的全部对比一起选择；query语义标签不能进入selector。
- [ ] 选择池参数差允许可见但其生成/优化器费用有账。
- [ ] r=k旋转等价回归；r<k实际截断见证。
- [ ] k=0时输出UNKNOWN；零预测baseline独立保留，不自动所有模型退回0。
- [ ] 所有方法使用相同packet、alpha、样本和target做干净消融。
- [ ] nonlinearity、span遗漏、exact拟合、finite拟合向量分解保留交叉项。

## D. 实际优化器与数据（必须）

- [ ] 同seed两个arm初始化、prompt顺序一致，后续各自on-policy。
- [ ] Python/NumPy/Torch/CUDA RNG、Adam状态、sampler、mode、buffer完整恢复。
- [ ] 全零优势显式零梯度后Adam仍step；没有把它当跳过更新。
- [ ] 所有冻结参数无grad且字节/版本不改变。
- [ ] 异常时恢复origin；恢复失败则对象不可继续。
- [ ] 新128步source调度512个prompt位置，验证集/confirm从未用于训练。
- [ ] 数据和LoRA初始化跨seed的变动范围按协议记录。
- [ ] 旧raw/summary/source不会被修改，旧确认集不再用于新确认。

## E. 统计（必须）

- [ ] NRMSE按平方和聚合后开方，非逐行NRMSE均值。
- [ ] 真信号零的NRMSE返回undefined；不加大epsilon制造通过。
- [ ] q95由原始残差重算，不能平均逐seed分位数。
- [ ] 训练seed顶层cluster，臂/anchor/bank/题目共同bootstrap。
- [ ] UNKNOWN既不算安全也不从全体指标删除；risk-coverage曲线真实。
- [ ] 区间覆盖和宽度共同报告；误差分位数不写成置信区间。
- [ ] 有限校准seed数的conformal分位数正确；不足95%时不硬标95%。
- [ ] 多次精度追加有预注册规则及多look控制，不按方法排名停。
- [ ] 参考是估计量，MSE扣噪按聚合处理、不逐单元clip负值。
- [ ] 所有p值/主比较/多重校正与冻结选择一致。

## F. GPU接入与工程（真实执行前必须）

- [ ] 实际GPU型号、容量、驱动和可见设备记录，不猜卡容量。
- [ ] 两卡相同策略评分零处理一致性与数值误差测得。
- [ ] 同路径generation logp与teacher-forcing/重评分语义一致；不通过保持原认证路径。
- [ ] terminal EOS/64截断动作概率符合实际采样分布。
- [ ] 新CLI dry-run不导入/加载真实模型，不调用sbatch。
- [ ] main/test标签在预测和选择锁发布前不可访问。
- [ ] worker分片独占写入、request去重、原子完成和resume测试。
- [ ] 对OOM/中断/重复提交/配置变化/数据篡改有实测故障注入。
- [ ] 大张量hash引用而非重复复制，存储配额与预计I/O明确。
- [ ] 全新测试与相关旧回归通过；历史fixture缺失不能改成假PASS。
- [ ] 公开产物不包含token、GPU账号凭据、原始权重和不必要的私有路径。

## G. 科学验收（不得预填PASS）

- [ ] 新观测是否胜过最强简单baseline，且关键事件不被总误差掩盖？
- [ ] 增加方向是否真的改善校准未覆盖部分，而非只改善easy群体？
- [ ] r<k上的收益能否在新seed/原点复现？
- [ ] 拒判是否能够识别错误，同时有非零可用覆盖？
- [ ] 真实VLM的参考精度是否足够比较当前方法？
- [ ] 若直接测量已足够，是否如实接受它而非制造代理胜利？
- [ ] 大预算验证与未来低成本部署的成本结论是否分开？
