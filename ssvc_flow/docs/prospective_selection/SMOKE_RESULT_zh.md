# 真实smoke结果

2026-09-24核验。Qwen3.5-9B，单张NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition；Slurm169593，代码414df78，终态COMPLETED，ExitCode0:0，墙钟31分08秒。

- 模型加载与初始化：175.152707秒。
- R0：2次Adam更新、64条训练输出，训练831.649045秒；采样140.909193秒，更新690.064790秒。
- R1：2次Adam更新、64条训练输出，训练838.619126秒；采样142.237586秒，更新695.735352秒。
- 两种输入界面各2条parity样本；行为logp与独立prefix评分最大token/sequence差值均为0。
- 两配方完整checkpoint恢复通过，包含Adam状态；独立parity评估前后完整训练状态不变。
- 任务完成登记与smoke/COMPLETE.json一致，两训练权威COMMIT存在，所绑定checkpoint文件存在。原始训练输出和tensor状态留在服务器campaign/smoke；小型原件见current_evidence/smoke_169593。

这是运行技术门禁通过，不是奖励选择有效性的结果。完整source和至少2个branch尚未完成，不据4个更新宣称首批或全矩阵实测工期。
