# 来源、核读程度与本次交付范围

## 1. 用户材料与项目代码

### 最近讨论与已完成结果

- `sources/LATEST_DISCUSSION_zh.md`：上一轮GPT–Claude协议讨论第三轮，2026-10-06。用于solver-J/self-J等价、U22曝光、PTLC别名与域混杂等边界。
- `sources/historical_portfolio_estimates.csv`：此前开发结果上的组合估计。本次只作为设计依据，没有重新执行其模型实验。
- `sources/unused64_current_exposure.json`、`U22_EXPOSURE_AUDIT.json`：旧确认候选池已被使用的记录。
- `sources/PREVIOUS_EXPERIMENT_FACTS_zh.md`：原始包的既有实验范围说明。
- `sources/checkpoints_historical.json`：本次从已提供清单摘取的S96/REP96；当前服务器文件存在性未在此会话核实。

### 实际核对的GitHub状态

仓库`Lhan-chding/dissertation`，分支`codex/protocol-state-probes-20261005`，HEAD `2f8c6b19970f49542b009f2816f1c1a91eb9974c`。

已通过连接读取当前branch和目录、`protocol_state_probes/inference.py`及`generate_worlds.py`相关部分；从用户已提供的QWEN35原始包读取了相关adapter、runtime、协议源码快照进行设计。没有提交代码、开PR或启动服务器。

涉及实际代码边界的依据：冻结inference loader会关闭梯度；历史模型revision及LoRA拓扑被固定；旧生成器的ID真值域为0–49且有trend有限容量检查。这些不能被新SFT流程不加区分地复用。

## 2. 外部原始工作：为什么不把工具组件当创新

本次查看原作者页面/摘要，并对ReST^EM的正文相关部分作了补充阅读。以下仅用于定位工具与对照，不是已完成全面新颖性审计。

| 工作 | 核读范围 | 与本方案的关系 |
|---|---|---|
| STaR: Bootstrapping Reasoning With Reasoning | 原作者arXiv摘要 | 已采用生成、选择成功推理、微调的自改进流程；本轮没有宣称筛选自训练本身新颖，也不复现其rationale设置。 |
| Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models（ReST^EM） | 原作者摘要及HTML正文的方法部分 | 生成、二元反馈筛选、微调已有直接基础；SELF_O0是我们任务内的简化固定一轮参照，不叫完整复现ReST^EM。 |
| Reverse Training to Nurse the Reversal Curse | 原作者摘要 | 正逆训练增强已有先例；其训练字符串反转/保留子串与我们的坐标规范化不同，但“换方向提供监督”不是空白。 |
| Off-Context GRPO | 原作者摘要 | 处理有提示采样与无提示目标错配；本方案还有多对一输出映射及成功筛选，不直接套用其比率。 |
| Hugging Face TRL SFT Trainer | 官方在线文档 | completion-only/assistant-only训练已有实现路径；本轮实际依赖现有Qwen环境，不强制升级或假定所有mask默认正确。 |

可追溯地址：

```text
https://arxiv.org/abs/2203.14465
https://arxiv.org/abs/2312.06585
https://arxiv.org/html/2312.06585v4
https://arxiv.org/abs/2403.13799
https://arxiv.org/abs/2607.19313
https://huggingface.co/docs/trl/sft_trainer
```

SA-MRPO/AMRP仍是多奖励主张的重要对照。本次未完成其完整算法与官方代码核验，也没有设计同名近似实现。SFT比R0好并不等于胜过它们。若以后恢复在线多奖励控制贡献，需另行准确实施。

## 3. 本次实际执行

本次是计划编制与设计契约验证：读取上述资料，编写新方案、配置与标准库参考函数，运行53项确定性测试，生成工作量和待执行作业表。

**没有**：生成正式T/V/E新数据、调用新Qwen回答、评分、GPU加载、模型反向传播、SFT或RL更新、拟合新选择器、修改远程仓库。

测试中的数组、少量数学示例、趋势容量枚举，不是新的模型实验证据。`templates/run_matrix.json`只是计划队列；数据view未生成前不能做真实alias判定。`workload.json`是上限算术，不是实测耗时。

## 4. 本方案相对上一轮的新增设计

显式0–99真值域的新cohort；固定预算下三种发现方式；公开验证后的唯一规范目标；fresh-AdamW的SFT主对照；两个父状态的pipeline重复；新E主终点与保持性guard；小规模R0_RESET32上下文参照。这些是本次建议，不是从旧文档中转述出来的已执行事实。

任何实现调整须说明它改变的是工程实现、测量协议还是科学目标。看完E后新添方法，属于下一轮开发，不能继续把同一E作为未见确认。
