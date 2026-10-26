# G3 开发先导实施边界

`src/sr_f5/pilot.py` 负责开发先导，不启动正式训练。只有实际 G2 4,608 条完整普查通过、服务器 CPU 回执通过、当前配置与普查冻结一致时，才能创建 G3 冻结。每父模型五臂各 12 步，每步 16 组 × G8，共 23,040 条主先导回答。问题计划和 draw seed 在臂间相同；每臂都从自己的已提交策略重新生成。

原文 05/07 要求各臂恢复和连续 12 步对照，但没有要求 15 个父模型×臂组合全部重复采样。本次在任何 G3 输出前登记 **b0 的五臂**做“连续 12 步”与“新进程读取第 4 步状态后继续至 12 步”的恢复对照。额外 5,120 条回答记为恢复诊断；b1/b2 使用相同恢复实现，并核对各自 checkpoint 身份及 cursor，但不宣称它们各自完成了独立的重采样等价实验。恢复不能只比较文本：实际比较参数、Adam、scheduler、CPU/CUDA/NumPy/Python RNG、数据 cursor、步骤 5–12 原始 token/采样概率/训练与 reference 概率、信用、数值与更新记录，均采用事前冻结的精确相等规则。未通过不得放宽容差。

不新增末端探针。每路径第 12 步已有回答按 row_id 排序，取 24 个唯一 `(qid, full_text)`，合计 360 个独立结构/标签/跨度程序审核；不足或审核失败阻止 G3 通过。所有臂分别检查 S 可观测比例 ≥90%，不允许合并平均掩盖单臂崩塌。HARD 必须报告实际 J 与 `J × 1[S=PASS]` 改变的行、组及 native advantage 改变组；没有真实奖励差异不能标为该检查通过。梯度与更新证据来自实际已提交更新，通道诊断沿登记步 1/12、固定每组首行的 16 行子集进行；没有观察到非零辅助梯度时明确阻止该项通过，不把“代码支持梯度”当成实测。

每个 checkpoint 包含原生完整状态和 cursor，在临时目录写完并同步后，以目录 rename 一次发布包含 `COMMITTED.json` 的目录。旧完整状态只保留第 4 步、第 12 步和最新状态，逐步更新/信用/原文始终保留。未提交更新失败后从上次已提交状态恢复；已完整保存的同策略 G8 可重用，未完成的采样组停止诊断，不悄悄补采或换 seed。已完成路径重复调用不会加载模型或重写原始连续运行证据。

每次加载拥有独立的 lane/execution task_id，保留其硬件身份和调用账本；科学 run_id 仍用于同路径恢复比对。主控制路径必须真实连续完成 12 步；发生中断后的一般恢复不冒充这项连续对照。b0 对照若受中断影响，将有明确未通过记录，需诊断而非伪造 PASS。

调用顺序：

```bash
python -m sr_f5.pilot --root "$RUN_ROOT" --freeze-revision "$COMMITTED_REVISION"
python -m sr_f5.pilot --root "$RUN_ROOT" --parent b0 --arm LOCAL
python -m sr_f5.pilot --root "$RUN_ROOT" --parent b0 --arm LOCAL --lane recovery
python -m sr_f5.pilot --root "$RUN_ROOT" --parent b0 --arm LOCAL --compare-recovery
python -m sr_f5.pilot --root "$RUN_ROOT" --aggregate
```

调度器对全部 15 条主路径执行第一种运行，对 b0 五臂分别启动新进程恢复并比较。G3 聚合读取全部原始组、冻结 schedule、adapter step/seed、实际信用、已提交更新、最终 checkpoint 的 mmap 元数据与绑定恢复回执；不扫描基模权重，不新增烟测采样。CPU fixture 验证的是门控、事务和恢复逻辑，不能替代实际 GPU G3 结果。

验证：13 项针对性测试通过，含真实小型 CPU Adam/checkpoint 的第 5 步中断、同策略完整批缓存恢复和完成路径零模型调用；Ruff 检查通过。实际 GPU 数值和门结果须由服务器回执确认。

## 已运行作业的时限边界（2026-10-25）

普通第 2 步实测：慢节点 J/SHUF 约 1,120/1,108 秒，快节点 HARD/LOCAL 约 522/517 秒。慢 worker 的 52–56 步超过 11.5 小时时限；集群拒绝修改 TimeLimit。新增外部运维脚本 `scripts/sr_f5/drain_pair.py`，不替换 G3 冻结科学代码、池顺序或采样参数。

控制器由持久 CPU 作业 `226455/LH-SRF5-G3-DRAIN` 承载，在 226282 allocation 内启动 CPU-only overlapping step；当前全用户共四个作业，不增加 GPU 预约。等两个 worker 的第一条 b0 路径进入恢复阶段，使用经过身份验证的 Linux pidfd 仅暂停负责启动下一阶段的 coordinator；正在执行的恢复子进程继续运行。子进程必须自行正常退出且 COMPLETE 绑定原 PID/start_ticks、从第 4 步恢复至第 12 步，随后用冻结部署做完整精确恢复比较。两侧全部 PASS 后记录计划性边界并取消本项目 allocation。其后仅在 squeue 消失和 sacct 终态已证实后，使用同一冻结代码和池计划提交新的双卡作业；已完成路径零模型调用复用，剩余路径照原顺序执行。

任何身份歧义、阶段已越过、非零退出、比较失败或控制器期限到达，都解除已验证 coordinator 的暂停并保留错误回执；不会把部分恢复或计划性 CANCELLED 写为实验 PASS。控制器异常退出需要检查 `CONTINUATION/DRAIN_226282.json` 的 coordinator PID/start_ticks 与控制器存活；不得直接重提或盲发信号。科学子进程不收暂停信号，冻结代码目录保持原样。下一阶段仍以真实 G3 门为准。

服务器运维部署 `sr_f5_operations_139709d`，revision `139709df6fe381b93a3d43cc92d1fdfb7a5f205c`。本地 7 项通过、1 项 Linux 专属测试跳过；服务器 8 项全部通过（0.38 秒），实测 coordinator 暂停时子进程仍正常退出。服务器 portable Python 未导出 pidfd API，但宿主 libc 提供同一接口，已使用并验证，未退化为有 PID 复用风险的裸信号。最初直接后台启动的子步骤 226282.0 随登录会话退出而 CANCELLED，events 为空，没有向训练进程发送信号，原始回执保留。22:25 核实持久控制作业 RUNNING、DRAIN 回执 ARMED，尚未执行暂停或边界取消。

## 用户授权的夜间加卡（2026-10-25）

用户授权再增加一至两张卡。当前四张 GPU，按五张总预约限制新增一张；两个双卡主作业保持原样，新增 `LH-SRF5-G3-EXTRA`。追加卡使用原冻结科学部署 `ea50c51`，提前执行原 worker 0/1 队列末尾的 b2-DENSE、b2-LOCAL 各 12 步，共 3,072 条原登记主回答；不增加实验臂、步数、采样预算或恢复回答，不按结果挑任务。排除已观测较慢的 gpu-pro6000-12；仍需实际启动后确认吞吐。

这是用户授权的资源分配补充，原 FROZEN 与 POOL_PLAN 不改写。启动前要求两条目标尚无运行目录、原 pair0 仍在首条 b0 任务且边界控制器健康。`CONTINUATION/SUPPLEMENTAL_G3.json` 保存资源补充和作业身份。pair0 边界退出后，续跑必须同时等待追加作业进入终态、两条路径 COMPLETE，再运行原冻结池；完成路径以已有 checkpoint 零模型调用跳过。不得让原池与追加卡同时抢跑同一条路径；追加失败须先诊断并保留原文与状态，不能盲重提。

## 第二版分支生效（2026-10-26 03:32）

本节取代上文第一版的后续池重提安排。第一版因真实概率门失败，经同 token 全序列/缓存、批量/单条与精度诊断后，事前登记唯一 recipe_v2。旧科学代码和全部原始证据保留；现存 b0 J/SHUF 连续恢复由原控制器完成，之后不再执行第一版未完成池。

第二版科学版本 `47d5e85fb0c9b39d8f996465045d21956825f59d`，统一全模型和浮点输入 FP32；新开发/先导数据使用登记的派生种子，全部臂、预算、数值报警门和精确恢复标准不变。新根 `/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_recipe_v2_20261026` 已完成 CPU 准备（226894 COMPLETED/0:0）。普查双卡 pair0 为 226900，pair1 等待资源；回执分别为新根 `G2_SUBMISSION_PAIR0.json` / `G2_SUBMISSION_PAIR1.json`，不得冒用第一版合并回执。先验证两项退出队列及成功终态，再从第二版冻结源码聚合。已有 G3 冻结之后不得重新写入带时间戳的 G2 gate，避免破坏冻结身份。第二版仍失败即保留诊断、停止正式扩展，不允许第三版。

第二版衔接使用独立运维脚本 `scripts/sr_f5/advance_recipe_v2.py`，不能调用旧 `advance_development.py`（旧版本既不识别两个独立 pair 回执，且会重复写 G2 gate）。运维脚本通过 `--code` 和 `--revision` 导入已登记科学部署，核对两项原始提交身份、退出队列和成功终态，复用实际 CPU PASS，随后仅一次聚合。`CONTINUATION/RECIPE_V2_G2_AGGREGATION.json` 保存聚合开始/完成及绑定摘要；不完整聚合需诊断，不能盲覆盖。`CONTINUATION/RECIPE_V2_ADVANCE.json` 保存当前决定。已有 G3 冻结只验证已有证据；已有提交直接停止重复操作。G2 第二版失败产生 `STOP_RECIPE_V2_FEASIBILITY`，不提交 G3。容量不足且从未提交时可在后续重入。

`advance_recipe_v2.sbatch` 是 CPU 决策入口，`SRF5_CODE` / `SRF5_REVISION` 固定科学版本，`SRF5_OPERATIONS` 指向单独提交的运维部署。只有两个普查作业身份齐备且不存在活动/未知控制器时才提交后续 CPU 作业；读真实阶段决定后继续推进，禁止把提交成功当作科学通过。

桥接验证：19 项小型操作层回归通过（0.16 秒），包含缺失/未知/错误 pair、活动/未知/失败作业、第二版门失败、部分聚合保留、已提交去重及冻结后的容量重试不改 gate 字节。Ruff 和 CPU 启动脚本语法检查通过；未重新运行既有 332 项服务器检查，没有模型调用。

04:04 实际接续：旧 J/SHUF 恢复精确 PASS 后 226282 已计划性终止且退出队列，226455 COMPLETED。第二版 pair0/pair1 为 226900/226918，依赖 CPU 决策为 226919；完整原始提交回执在新根两个 G2_SUBMISSION_PAIR*.json 和 CONTINUATION/RECIPE_V2_DECISION_SUBMISSION.json。当前控制器存在，禁止与其抢跑聚合或另交控制器。

## 第二版G3实际运行与双pair恢复边界（2026-10-26 09:58）

真实第二版G2已全量PASS，226919完成一次汇总并冻结提交227050/227051。不得重新聚合G2。科学仍47d5e85，边界运维独立fdec062；新增显式`--pair`，验证allocation.pair与[0,1]/[2,3]完全匹配，仍要求每worker第一条b0恢复，不适用于任意后续臂。两个CPU控制器227163/227164分别保护pair0的J/SHUF和pair1的LOCAL/HARD；回执为新根CONTINUATION/DRAIN_<target>.json及DRAIN_CPU_SUBMISSION_<target>.json。控制器max_seconds=20000、CPU作业6小时，均已实际运行。

首批恢复边界后仅在双方比较PASS、目标GPU作业退出squeue且sacct终态时重入对应pair，不能重提完整两个池或改冻结计划。使用科学部署原pilot.sbatch和SRF5_ROOT/SRF5_CODE/SRF5_REVISION/SRF5_PAIR，名称仍LH-SRF5-G3-P0/P1。新job回执必须单独保存并核实全用户作业和GPU预约。原primary/recovery COMPLETE与原子checkpoint由冻结代码零模型调用复用；已通过RECOVERY证据保留，不能把中断连续主路径重新拼接成连续对照。第二版没有supplemental；旧版追加卡屏障不适用。

按当前步速，首b0恢复后pair0最长剩36步、pair1剩36步，预计各约8–9小时计算，可进入新11h30作业；仍须以实际速度检查时限。聚合需完整15主路径及b0五臂恢复PASS，正式训练仍禁止。当前部署代码的14项服务器针对性CPU检查通过，包含Linux pidfd身份固定；不重跑科学CPU门。
