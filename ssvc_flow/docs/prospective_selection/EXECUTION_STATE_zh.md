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

用户已明确回复“授权持久自动续跑”。每小时当前任务 heartbeat 已创建：automation ID `ssvc`，状态 ACTIVE。范围是最多7张PRO6000（用户随后明确增加2张）、门禁通过后推进、工程故障修复与commit/push、首4条先交付、冻结后最终测试，直到报告全部交付后暂停。状态无变化时不通知。此前因缺少明确持久调度授权的自动审批拒绝已由本次授权解决；没有创建替代daemon或cron。

下一步：核验source169616–169619及前置观测169906（61001_t32）的当前进度；满7个项目单PRO6000作业时不再提交。完成/释放槽位后按持久队列继续，并先交付首4条lineage及E复核/实测工期再扩展。任何实际故障先核对权威完成段、日志和Slurm终态，保留失败attempt；未知提交不重提，数值非有限不能盲目重启。

最近本地验证：188 passed, 1 skipped（23.53秒）；ruff及git diff --check通过。对应测试为tests/test_prospective_*.py、tests/prospective_selection、tests/decision_modeling。跳过项是既有可选原件检查，不能据此宣称GPU完成。首个真实更新采样56.28秒、更新330.99秒；完整source与至少两个branch实测尚缺，不能将单步外推冒充主方案工期交付。

续跑注意：服务器bb23e9b已支持4训练+1观测/评估（4卡时3+1），优先prestate，分支跨原点轮转；单队列时利用空槽，未知提交仍占槽。新增5个调度回归用例，jobs针对集47 passed，ruff及diffcheck通过，不重复全仓测试。当前5个作业的启动快照见current_evidence/FIRST_FOUR_LAUNCH.json；不能把此快照当未来实时状态。

当前CODE_VERSION.json已绑定commit与implementation_id；每次使用新代码必须新建独立版本，不改写运行中的快照。最终测试prepared_full.json已在本地runs/prospective_selection_setup准备，尚未上传，不得覆盖运行中的开发数据绑定。所有source运行96步，t32权威checkpoint后可注册队列中的P观测；观测和branch均已有登记，不要重复注册/直接sbatch绕过intent。当前未交付首4条，不得写FIRST_FOUR_DELIVERED或扩展后续lineage。

2026-09-24T11:25Z检查：source61001/61002/61003/61004已见更新14/15/14/15步，权威恢复段均到8步，均未完成96步。当前无已完成P观测或branch；不能据此生成首4条完整报告。O2_R0旧E结果J=0.8268229167，IMAGE_CUE_FRESH=1.0、SYMBOLIC_FRESH=0.6536458333；只是旧端点评估，不能当未见原点的前瞻选择结果。完整原始文本留服务器，小型回执见current_evidence/historical_E/O2_R0。

2026-09-24T14:17:48Z：用户告知网络恢复后已重新连接；先前Network is unreachable没有作为作业失败或重提依据。4个source观察更新分别36/38/36/37步，权威段与H32均已完成32步。第二个旧E端点O1_R3的J=0.8302951389，72×32原始输出与回执一致；保留为历史审计结果。已通过队列启动prestate61001_t32，Slurm169906，任务键2da3a7a59cb5d3b5f1134ecc8fc38a9222916093bdd199cb3a3a49d7937b9c75，首次检查RUNNING，当前共5个单PRO6000作业。该任务从t24/t32 checkpoint采集共同P样本，4层选择器共享；不能把正在生成的前置观测称为已完成branch。

用户新增授权：上限提高到7张PRO6000，见RESOURCE_OVERRIDE_zh.md。原protocol.json不改，避免改变已经生成的科学证据身份；后续使用更新调度器与显式--max-gpu-jobs 7 --available-gpus 7。现有5个作业持续运行，不需要重启。自动续跑提示已同步为7卡，每小时频率不变。

7卡调度代码afd3119已commit/push；针对集51 passed、ruff和diffcheck通过。后续单次submit-ready须从code_afd3119/ssvc_flow运行并显式传7；此前5个运行任务继续读code_bb23e9b，不改写原目录。

资源实际阻塞（2026-09-24T14:34Z）：用户授权7卡，代码afd3119已部署并核验，但教师QOS soujanya-poria-startfund-2026-03 的 MaxSubmitJobsPU=5。新增prestate61003_t32的sbatch明确以QOSMaxSubmitJobPerUserLimit拒绝，没有创建JobID；squeue/sacct核对无对应名称，原intent和submission_error已移入campaign/rejected_submissions/3f36a48b3a603c5a04fd05745d52d0599a79081f51cc370a23f3ab3c60d8d854/attempt_1，RECONCILIATION.json保留依据。不是未知提交盲重试，也没有取消其它作业。当前仍为原5个作业运行，尚未启用新增2张卡。后续调度使用--max-gpu-jobs 7 --available-gpus 5；正常槽位释放后继续队列。管理员提高配额且重新核实后才把available-gpus改7，不切换QOS绕过上限。


2026-09-24T14:45Z后续决定（覆盖上文扩卡执行安排）：现场确认账户允许官方可抢占QOS `override-limits-but-killable`，所以并非只能等待管理员增加教师配额。CLI提交e59b9ee已新增显式QOS选择，可抢占任务启用requeue并追加日志，54项针对性测试、ruff、diffcheck通过，代码已push。新快照为 `code_e59b9ee/ssvc_flow`，已有运行快照不变。

两个新增PRO6000前置观测169962/169963成功提交，但Slurm在14:45:31Z给出的预计启动为次日00:17:23Z/00:31:39Z，需等待约9小时32分/9小时46分。用户随后要求排队太久就放弃，因此取消此次扩卡尝试。保留取消终态及从未分配GPU的核验，归档其intent/submission后让原任务回到正常五卡队列；不是删除科学任务或更换seed。最新证据见current_evidence/EXTRA_GPU_CANCELLED_20260924.json。

后续只使用教师QOS，`--max-gpu-jobs 7 --available-gpus 5 --qos soujanya-poria-startfund-2026-03`；不再自动重试额外可抢占资源，也不混用其它型号。每小时续跑已同步此决定。原5个作业169616–169619和169906保持运行；授权上限7保留，但本次扩卡安排已放弃。


2026-09-24T19:29Z小时检查：两次有界SSH连接eee-cluster（10.97.216.128:22）均Operation timed out，当前服务器状态无法核验；不是实验失败证据。未提交、重提或取消任务。最近成功现场检查为18:30:01Z：169616/169618/169617/169619对应61001/61002/61003/61004最新更新66/71/67/69，均已提交恢复段至64；169906前置观测t24为72/72、t32为66/72，尚无COMPLETE；五个作业当时均RUNNING，未知提交错误0。连接恢复后先读取当前提交ID、完成回执、原始观测与Slurm终态；不能把上述历史快照当当前进度，不能因连接超时重跑。每小时自动检查保留。


2026-09-25T01:05Z网络已恢复并完成现场验证：source61001–61004全部96步完成，Slurm169616/169618/169617/169619均COMPLETED、ExitCode0，耗时12:56:39/12:08:15/12:42:26/12:27:35。共12,288条原始训练输出、384次有限更新及48个权威恢复段核验通过。前置观测169906完成2×72×32条输出（04:20:39）；四层信息包重建核对通过，P J24=0.6501736111、J32=0.6905381944，仅作前置状态事实。详见SOURCE4_PROGRESS_zh.md及current_evidence/SOURCE4_PRESTATE1_VERIFIED_20260925.json。

已按五卡队列提交170518–170521（61001_t32的R4/GDPO_R4/SAW_R4/R5，repeat1）和170522（61001_t96前置观测），代码e59b9ee。当前任务ID以current_evidence/RESUMED_LAUNCH_20260925.json及服务器实时记录为准；旧source与169906不再是运行任务。首四条完整响应表尚未交付，没有写FIRST_FOUR_DELIVERED、没有扩展或最终测试。

提交后现场复核：170518–170522全部RUNNING，分别分配gpu-pro6000-2（三个）、gpu-pro6000-14和gpu-pro6000-11，均为单PRO6000作业；当前并发5。


2026-09-25T06:36Z：170522（61001_t96）COMPLETED，ExitCode0，耗时05:06:48。两窗口共4608条原始输出及四层共同信息包复核通过（回执current_evidence/PRESTATE_61001_t96_VERIFIED.json）；P J88=0.8346354167、J96=0.8342013889，仅为前置状态描述，不构成选择器比较。当前前置观测2/8完成。170518–170521均已完成32步训练并进行H32独立D评估，06:35Z分别54/58/53/77个提示（目标144），尚无完整branch完成回执。释放槽位已提交170771=61003_t32前置观测，首次状态PENDING；后续须读实时状态。原五卡容量继续，不扩展首四条范围。


2026-09-25T07:46Z首两个完整开发branch核验通过：170521=61001_t32/R5，Slurm06:19:35，D J=0.7565104167；170519=61001_t32/GDPO_R4，06:36:53，D J=0.7734375。每个分支32个有限更新、1024训练输出、H8的192及H32的2304独立评估原始输出，身份和完成绑定、小文件哈希、结构化记录复算均通过。证据见current_evidence/FIRST_BRANCHES_VERIFIED_20260925_0746.json。仅为同一开发原点的描述性结果，不能推断未见原点选择器优劣；继续所有登记配方。

已完成“source加至少两个branch”实测工期更新，见MEASURED_RUNTIME_zh.md。当前完整branch均值6.471 GPU小时；以五卡80%有效利用率假设外推，首四条剩余约6.4天、全部开发与调参剩余约26.9天（均包含在运行任务的保守全额，非最终测试工期承诺）。N/m未冻结，最终测试不运行。

释放的两个已核验槽位分别补入170853=61001_t96/R1、170863=61001_t96/R5（repeat1），教师QOS；仍为5卡容量。07:45Z 170853 RUNNING、170863刚提交PENDING，170518(R4)/170520(SAW)仍在H32评估，170771=61003_t32前置观测运行。无关pi05-vfd-goal作业使用rose QOS，不计入项目并发也未触碰。后续以实时队列为准，未写FIRST_FOUR_DELIVERED或扩展lineage。
