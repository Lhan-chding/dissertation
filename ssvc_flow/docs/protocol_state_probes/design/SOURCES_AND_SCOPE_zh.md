# 来源、证据范围与相关工作

## 私有项目材料

S1. 用户上传`Claude第五轮回应：SSVC四条开发讨论.md`：PTLC原定义、DPE₁、A1/B1/I1、T1/T2的讨论提案。副本在`sources/CLAUDE_ROUND5_zh.md`。其中若干强推论已在S2修正，不能把S1全部当作已达成结论。

S2. 用户提供的第六轮讨论与证据包`SSVC_Claude_Round6_20261004.zip`：PTLC字面核验、前缀与完整程序分离、B1风险修正、checkpoint历史指针。讨论全文及重要表格在本包sources中。

S3. `SSVC_Claude_Round5_20261004.zip/sources/prepared.json`与S2的task_DPE_registry：本次仅用于提取已存任务、构建D48/U22和执行对照、登记程序预测。本包保留所需原record；没有重新运行全部原始输出分析。

S4. `SSVC_FOUR_LINEAGE_FULL_DATA_20261004.zip/raw/RAW_MANIFEST.json`：查询导出时存在的6个checkpoint路径。未读到服务器现有模型张量，未判断当前实际可加载性。

S5. GitHub当前读取：
- `codex/prospective-reward-selection-20260924`，`76ff2eab62d7fa6d902be972853fe73ca33dee9e`；
- `codex/four-lineage-existing-only-20261004`，`eb58174c6992e8a82e04ca75cdf06a682f5e9a46`；
- `ssvc_flow/src/prospective_selection/evaluation.py`：旧面板数量锁定及原评价入口；
- 同版本原交付source：`branch_runtime.py`、`modeling_v3/vlm_observation.py`、`model_adapters/base.py`、`prompts.py`、`verifiers.py`。

本轮没有向GitHub写入代码或提交，不代表新runner已存在。

## 外部来源（本次查证于2026-10-05）

R1. Bachmann & Nagarajan, *The pitfalls of next-token prediction*, arXiv:2403.06963. 本次核读作者摘要；用于区分teacher-forcing学习与自回归推理，不从其反向训练结果推导本模型冻结反序提示必然成功。
https://arxiv.org/abs/2403.06963

R2. Yu et al., *Thinking Out of Order: When Output Order Stops Reflecting Reasoning Order in Diffusion Language Models*, arXiv:2601.22035. 本次核读作者摘要；已讨论输出结构与推理顺序冲突及order robustness。本项目不能认领一般“输出顺序重要”。没有据此更换模型为扩散模型。
https://arxiv.org/abs/2601.22035

R3. Zheng et al., *Make Sparse Rewards Count: Density-Aware Reward Aggregation for Multi-Reward RL*, arXiv:2610.00574（DARA）. 作者摘要确认按active-group density校准奖励贡献；对比可用性本身不是空白。本轮不训练DARA，也不按摘要拼一个同名实现。
https://arxiv.org/abs/2610.00574

R4. Wang et al., *Learn What's Left, Not What's Mastered: Saturation Aware Advantage Reweighting for Multi-Reward Policy Optimization*, arXiv:2608.16072. 作者摘要确认独立标准化与batch级饱和度调权；它不是已跑过的SAW。本文不填未经核实的默认gamma或完整公式。
https://arxiv.org/abs/2608.16072

## 新颖性定位

已知：一般顺序效应、多奖励统计调权、活跃组密度都已有近邻。待检验：本受控任务的结构条件与具体错误向量是否预测真实协议响应；相同结构下的训练状态是否改变有用的操作排序。

只要统一协议已经最佳，复杂状态路由可能没有增益。若只有新提示下有效、原协议未改善，结论是条件推理表现，不是后训练能力已提高。完整方法创新仍待后续证据，本文不作唯一性或首创性声明。

## 本次实际完成

- 读取现存讨论、任务元数据、运行源码及checkpoint导出记录。
- 按元数据编译509个case、35个完全重复提示别名、1944个逻辑单元。
- 形成完整矩阵70,656条回答的计划；这不是已生成回答数。
- 运行46项CPU契约测试，包含19,008个程序级断言。
- 新模型生成、似然评分、梯度、优化器更新、选择器拟合、服务器作业均为0。

因此文档中的协议效果、T1/T2和后续操作排序全部是待执行目标。不能把本包测试通过写成PTLC已由真实干预验证。
