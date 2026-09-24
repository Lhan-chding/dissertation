# 当前执行状态

本文件是续作入口；科学方案见 design/START_HERE_FOR_CODEX.md。

- 本地独立worktree：`/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/prospective-reward-selection-20260924`。
- 远程分支：`codex/prospective-reward-selection-20260924`，首个实现提交 `414df78` 已push。
- 当前实际GPU代码快照：`/projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/code_414df78/ssvc_flow`。本地后续报告/冻结接线修改尚不影响该运行快照。
- 服务器运行根：`/projects/varunssd/louis-ssvc/prospective_selection_v2_20260924`；队列根为同目录 `campaign`；Python `/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python`。
- 首个GPU smoke：Slurm `169593`，任务键 `ba84fb3d931915804cfac154024181573d6fac6f9e9206fb95177bff7c6a6cfa`。2026-09-24T09:25Z后核验：COMPLETED，elapsed 00:31:08，ExitCode0:0；R0/R1各2次真实更新、完整checkpoint恢复、独立评估状态不变、4条logp parity均通过。见SMOKE_RESULT_zh.md及current_evidence/smoke_169593原件。
- GPU实测：NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition；显存101971460096 bytes；初始化及加载175.152707秒。
- 已注册前4条开发lineage：61001–61004，共4个source、8个前置状态、88个分支。smoke完整门禁已通过，source待本轮正式提交登记。
- P0已重读36,864条旧P输出；16个旧E端点评估尚未启动。实际端点列表由 history.HISTORICAL_ENDPOINTS 提供（O1十配方，O2六配方）。
- 完整T元数据准备已完成，排除2个历史CPU代理真值别名。`prepared_full.json` 与在用 `prepared_development.json` 的 source/continuation/P/D/题序完全一致；当前smoke仍绑定后者，不覆盖运行中的绑定。
- 首4条完整结果、实测工期、完整开发/调参、冻结、最终测试尚未完成。没有最终科学结论。

用户已明确回复“授权持久自动续跑”。每20分钟当前任务 heartbeat 已创建：automation ID `ssvc`，状态 ACTIVE。范围是最多5张PRO6000、门禁通过后推进、工程故障修复与commit/push、首4条先交付、冻结后最终测试，直到报告全部交付后暂停。状态无变化时不通知。此前因缺少明确持久调度授权的自动审批拒绝已由本次授权解决；没有创建替代daemon或cron。

下一步：按主方案启动最多5个项目单PRO6000作业，并先交付首4条lineage及E复核/实测工期再扩展。任何实际故障先核对权威完成段、日志和Slurm终态，保留失败attempt；未知提交不重提，数值非有限不能盲目重启。

最近本地验证：188 passed, 1 skipped（23.53秒）；ruff及git diff --check通过。对应测试为tests/test_prospective_*.py、tests/prospective_selection、tests/decision_modeling。跳过项是既有可选原件检查，不能据此宣称GPU完成。首个真实更新采样56.28秒、更新330.99秒；完整source与至少两个branch实测尚缺，不能将单步外推冒充主方案工期交付。

续跑注意：当前服务器414df78按task哈希顺序选ready，smoke后16个historical_e与4个source同时ready。首次提交前检查ready顺序，按runbook优先保留3–4个训练worker及1个评估worker；若需改调度，部署新的独立版本并保留持久intent，不直接sbatch绕过登记。代码新版本须在最终冻结前部署并生成CODE_VERSION.json（commit与implementation_id），不可改写正在运行的代码快照。最终测试的prepared_full.json已在本地runs/prospective_selection_setup准备，尚未上传，不得覆盖运行中的开发数据绑定。
