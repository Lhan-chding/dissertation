# 当前执行状态

本文件是续作入口；科学方案见 design/START_HERE_FOR_CODEX.md。

- 本地独立worktree：`/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/prospective-reward-selection-20260924`。
- 远程分支：`codex/prospective-reward-selection-20260924`，当前GPU代码提交 `bb23e9b` 已push（此前414df78、1bb5888保留）。
- 当前实际开发GPU代码快照：`/projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/code_bb23e9b/ssvc_flow`，含已核验CODE_VERSION.json；smoke原快照414df78不动。训练、采样、评估、Adam模块相对smoke没有修改；后续修改为调度、报告、冻结门禁。
- 服务器运行根：`/projects/varunssd/louis-ssvc/prospective_selection_v2_20260924`；队列根为同目录 `campaign`；Python `/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python`。
- 首个GPU smoke：Slurm `169593`，任务键 `ba84fb3d931915804cfac154024181573d6fac6f9e9206fb95177bff7c6a6cfa`。2026-09-24T09:25Z后核验：COMPLETED，elapsed 00:31:08，ExitCode0:0；R0/R1各2次真实更新、完整checkpoint恢复、独立评估状态不变、4条logp parity均通过。见SMOKE_RESULT_zh.md及current_evidence/smoke_169593原件。
- GPU实测：NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition；显存101971460096 bytes；初始化及加载175.152707秒。
- 已注册前4条开发lineage：61001–61004，共4个source、8个前置状态、88个分支。smoke完整门禁已通过；source作业61001=169616、61003=169617、61002=169618、61004=169619，首次队列核验全部RUNNING。
- P0已重读36,864条旧P输出；16个旧E端点评估：O2_R0作业169620已COMPLETED（01:47:04，ExitCode0:0），72×32原始输出重算与完成回执一致；O1_R3作业169684已COMPLETED（01:53:14，ExitCode0:0），原始2304条输出已重算核验；其余14个尚未提交。实际端点列表由 history.HISTORICAL_ENDPOINTS 提供（O1十配方，O2六配方）。
- 完整T元数据准备已完成，排除2个历史CPU代理真值别名。`prepared_full.json` 与在用 `prepared_development.json` 的 source/continuation/P/D/题序完全一致；当前smoke仍绑定后者，不覆盖运行中的绑定。
- 首4条完整结果、实测工期、完整开发/调参、冻结、最终测试尚未完成。没有最终科学结论。

用户已明确回复“授权持久自动续跑”。每小时当前任务 heartbeat 已创建：automation ID `ssvc`，状态 ACTIVE。范围是最多5张PRO6000、门禁通过后推进、工程故障修复与commit/push、首4条先交付、冻结后最终测试，直到报告全部交付后暂停。状态无变化时不通知。此前因缺少明确持久调度授权的自动审批拒绝已由本次授权解决；没有创建替代daemon或cron。

下一步：核验source169616–169619及前置观测169906（61001_t32）的当前进度；满5个项目单PRO6000作业时不再提交。完成/释放槽位后按持久队列继续，并先交付首4条lineage及E复核/实测工期再扩展。任何实际故障先核对权威完成段、日志和Slurm终态，保留失败attempt；未知提交不重提，数值非有限不能盲目重启。

最近本地验证：188 passed, 1 skipped（23.53秒）；ruff及git diff --check通过。对应测试为tests/test_prospective_*.py、tests/prospective_selection、tests/decision_modeling。跳过项是既有可选原件检查，不能据此宣称GPU完成。首个真实更新采样56.28秒、更新330.99秒；完整source与至少两个branch实测尚缺，不能将单步外推冒充主方案工期交付。

续跑注意：服务器bb23e9b已支持4训练+1观测/评估（4卡时3+1），优先prestate，分支跨原点轮转；单队列时利用空槽，未知提交仍占槽。新增5个调度回归用例，jobs针对集47 passed，ruff及diffcheck通过，不重复全仓测试。当前5个作业的启动快照见current_evidence/FIRST_FOUR_LAUNCH.json；不能把此快照当未来实时状态。

当前CODE_VERSION.json已绑定commit与implementation_id；每次使用新代码必须新建独立版本，不改写运行中的快照。最终测试prepared_full.json已在本地runs/prospective_selection_setup准备，尚未上传，不得覆盖运行中的开发数据绑定。所有source运行96步，t32权威checkpoint后可注册队列中的P观测；观测和branch均已有登记，不要重复注册/直接sbatch绕过intent。当前未交付首4条，不得写FIRST_FOUR_DELIVERED或扩展后续lineage。

2026-09-24T11:25Z检查：source61001/61002/61003/61004已见更新14/15/14/15步，权威恢复段均到8步，均未完成96步。当前无已完成P观测或branch；不能据此生成首4条完整报告。O2_R0旧E结果J=0.8268229167，IMAGE_CUE_FRESH=1.0、SYMBOLIC_FRESH=0.6536458333；只是旧端点评估，不能当未见原点的前瞻选择结果。完整原始文本留服务器，小型回执见current_evidence/historical_E/O2_R0。

2026-09-24T14:17:48Z：用户告知网络恢复后已重新连接；先前Network is unreachable没有作为作业失败或重提依据。4个source观察更新分别36/38/36/37步，权威段与H32均已完成32步。第二个旧E端点O1_R3的J=0.8302951389，72×32原始输出与回执一致；保留为历史审计结果。已通过队列启动prestate61001_t32，Slurm169906，任务键2da3a7a59cb5d3b5f1134ecc8fc38a9222916093bdd199cb3a3a49d7937b9c75，首次检查RUNNING，当前共5个单PRO6000作业。该任务从t24/t32 checkpoint采集共同P样本，4层选择器共享；不能把正在生成的前置观测称为已完成branch。
