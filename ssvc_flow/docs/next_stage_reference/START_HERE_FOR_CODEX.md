# 给 Codex 的起始指令

你正在接手已有的SSVC Qwen3.5-9B项目。用户已经在NTU服务器跑完P1和P3-N/P3-L冻结评估。不要重新搭一套仓库或默认重跑P3。

先完整阅读本包`CODEX_NEXT_STAGE_zh.md`和`configs/next_stage.yaml`，再阅读已有仓库README、配置、成功作业脚本、P1/P3 manifest与原始输出。最新报告只证明冻结推理完成，训练smoke必须单独验证。

依次完成：

**R0** 只读审计已有产物，重算逐题四类、场景配对CI、唯一解、parser和采样参数，利用原16条bank计算K≤16的奖励组构成。

**R1** 核查forward计数、似然/采样/缓存一致性与MLP-LoRA训练路径。仅在数值验证通过后优化吞吐。smoke adapter和Adam状态必须丢弃。

**R2** 固定、不按表现筛选的72场景六条件诊断，以及小型长度/thinking诊断。所有诊断变体保留独立protocol tag，不能污染主训练。

**R3-cold** 从48个固定训练prompt生成384条on-policy bank，对λ0/0.01/0.25/1/2做同起点联合优势、梯度和真实Adam分支。只对预指定两个bank做完整control概率重加权，不默认做cold直接重采样。

**R4** 在显式`--allow-training`且门禁通过后，从同一全新初始化运行X_BASE和X_VALID，各64步、B4、K8、seed17；评估N/L和预注册graph-OOD。不能启动其他模型或A奖励训练arm。

**R3-warm** 在X_BASE step64的完整Adam状态重复局部审计，并对预指定候选做3,072条真实重新采样，检验预测和观测的差异。

不要自动启动在线SSVC、三seed确认、其他模型、FSA、I4、SFT warm-start或奖励重采样。R5先交付离线决策接口和进入条件。

所有新代码要有测试、dry-run、hash绑定resume和独立stage status。旧目录只读。无法访问服务器时先实现/测试CPU部分并列出待运行命令，不能输出伪造的REAL_CUDA结果或PASS。

最终返回`NEXT_RESULTS_SUMMARY_zh.md`，逐项区分实际结果、计算、推断和未知。回答规范第12节的五个决策问题，并附原始证据路径、配置hash、逐题统计、训练状态和实际成本。
