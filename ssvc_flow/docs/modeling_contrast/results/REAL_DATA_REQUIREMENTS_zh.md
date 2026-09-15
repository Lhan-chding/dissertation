# 下一阶段最小真实资料需求

当前没有发起新Qwen/GPU调用、训练、原点补采样或在线SSVC。以下是资料需求，不是执行授权。

1. 固定prompt/场景/接口/答案映射/parser/解码配置与版本hash，以及逐题权重、
   完整动作身份与X/S/W/I标签规则。
2. 同一起点、同训练bank下joint_0/joint_1/no_x_off_1的真实参数增量e与d，
   以及参数布局、参数hash、Adam/gradient/RNG完整状态指纹或明确缺项。
3. 独立于训练bank的control样本，完整输出序列、实际采样分布、proposal_id、draw_id、
   source_policy、normalized sequence logp、候选交叉评分和支持检查。
4. 明确普通采样、可控制共同随机数、指定序列logp三种访问权限；不能从四类计数反推出动作似然，
   也不能假定真实VLM支持toy的逆CDF耦合。
5. 真实X/I事件通常含多个输出序列：先核验事件集合是否已知且可枚举，
   规范答案字符串的单一概率不能直接当pX。
6. 原始训练seed/arm/anchor/bank角色、测量replica与fit/eval packet独立性；
   测试行为标签在预测封存后才可评分。
7. 实测生成、交叉评分、标签验证、候选更新、归一化forward及缓存成本；
   给出新bank查询域，不把同一packet重复读取当独立信息。

新数据、初始化和真实模型的泛化不在本轮已验证范围内。任何后续真实模型调用须另行授权并制定独立协议。
