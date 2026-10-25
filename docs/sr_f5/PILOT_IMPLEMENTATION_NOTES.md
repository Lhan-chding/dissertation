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
