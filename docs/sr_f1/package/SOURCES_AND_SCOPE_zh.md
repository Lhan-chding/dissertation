# 来源、文献实施核对与实际范围

## 用户材料

[S1] 最近综合讨论的 `04_STUDY_DESIGN_zh.md`，本包副本 `sources/PRIOR_STUDY_DESIGN_zh.md`：四类诊断思想、证据契约、五固定反馈、原始任务和迁移。

[S2] 最近给Claude的 `01_REPLY_TO_CLAUDE_zh.md`，本包副本 `sources/PRIOR_DISCUSSION_zh.md`：保留有意义的目标，不把门控率当创新终点；有界精化的实现和反例；GDPO完整接口；支持训练与奖励对照分开。

[S3] `CODEX1_MM_DEV_FINAL_PLAN_zh.md`，本包副本 `sources/PRIOR_MM_DEV_F2_zh.md`：历史9B快照、LoRA、采样概率与恢复要求。**这是文稿而非本轮核验的服务器状态。**本轮不借其实验身份推出新题组支持情况。

## 本轮外部原始来源

[R1] Liu等，GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization，arXiv:2601.05242v1，2026。核对§3.1公式(4)–(7)、§3.2优先级与附录A。采用分组件组归一化＋全批次归一化，不把单个microbatch标准化称为GDPO。人口/样本std和本实验权重明确记为本任务适配，不复制论文实验分数。

[R2] Shao等，DeepSeekMath，arXiv:2402.03300，§4.1与相关推导：GRPO组相对优势、序列平均policy surrogate、token KL。这里的自定义loss具体实施按主合同，不自动下载一个库最新版代替已核验数学对象。

[R3] Qwen官方9B模型卡与官方模型文档：确认该模型是视觉语言模型、32层混合布局。网页main可能变动，实际实验继续以原固定revision为准。

[R4] Masry等，ChartQA作者仓库README，官方链接至作者Hugging Face数据集；核对测试数据来源和需隔离的底层tables/annotations。尚未在本地下载ChartQA图像，完整数目/版本/评分代码待执行端验证。

原文入口（供Codex定位，不覆盖已固定权重）：

```text
R1 https://arxiv.org/html/2601.05242v1
R2 https://arxiv.org/html/2402.03300
R3 https://huggingface.co/Qwen/Qwen3.5-9B
R3 https://huggingface.co/docs/transformers/en/model_doc/qwen3_5
R4 https://github.com/vis-nlp/ChartQA
R4 https://huggingface.co/datasets/ahmed-masry/ChartQA
```

本轮没有重新声称把八篇近邻全部逐页重读。最近讨论的正文级文献比较仍作背景；本次外部核查聚焦实际要实现的GDPO/GRPO与模型、外部数据接口。不能据此写成“已完成完整新颖性排查”或“本算法首次”。

## 本地实际完成

- 读取最近讨论和原F2执行稿，明确继承/修改项。
- 构造496个新根、4,800题的程序世界/AST/文本/真值清单、3个配对输入流和15路径矩阵。
- 独立第二执行器核算全部4,800个答案，与主执行器逐项一致；gold输出全部获得J=1。
- 核验544组完整图表变体家系的保持/变化条件、AST留出和输入流配额。
- 数学/评分/奖励测试以 `validation/PYTEST_LOG.txt` 的真实结果为准。
- 本地Pillow渲染36张不同例图；查看联系图的8个原题样例。没有运行Qwen processor，未逐张认证所有变体可读性。
- 所有工作都在本地CPU完成。没有对新题组作模型采样，没有训练、没有新F2结果、没有操作服务器作业。

尝试直接从本地网络取得外部原始测试JSON时DNS不可用，未产生数据副本。网页核查成功不等于本地已下载ChartQA。外部资产留给执行端按作者源核验，不能虚构其字节hash。

## 权限与诚实边界

图像例图不是模型输出；gold答案不是模型成功。CPU测试通过不等于GPU学习有效。生成文件的本地hash不构成服务器预登记身份。未加入字体/模型权重/外部数据私有副本。
