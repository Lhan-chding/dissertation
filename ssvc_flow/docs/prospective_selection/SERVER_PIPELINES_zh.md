# 当前固定流水：2026-10-04扩展阶段

控制目录：`/projects/varunssd/louis-ssvc/prospective_selection_v2_20260924/fixed_pipelines_20261004`。旧20260930目录保留且STOP；login-3 cron已改为SSVC_FIXED_PIPELINES_20261004。

代码88ed250，训练仍用原code_e59b9ee。新任务300项按整条lineage分成五份75/75/50/50/50项，同lane依次source、两个prestate、22分支；验证退出码、原件和身份后立即执行下一项。注册总数417，旧117为不可变审计baseline。配置绑定首批交付回执SHA256及完整固定矩阵，拒绝最终测试任务。

教师QOS所有用户作业合计最多5（含pending、无关同QOS作业及未知提交）；用户项目总上限7，本阶段available_gpus5。初次182756–182759启动四lane，182577占第五槽位。每五分钟cron只交接首次空槽和72小时时限后的同清单接续，GPU内不轮询领取。缺失/异常/未知提交停止新项，不自动重提。

source核验96有限更新/3072样本，prestate复算两时点4608样本及四层包。300项全部核验后状态DEVELOPMENT_TUNING_COLLECTED_AWAITING_ANALYSIS；先分析、拟合、冻结，再按decision.json选择测试。详细事实与验证见[执行记录](EXECUTION_STATE_zh.md)最后一节及current_evidence/EXPANDED_PIPELINES_20261004/。

以下为历史首四条部署记录，不再是当前入口。

---

# 五份固定任务清单连续执行

用户于2026-09-30明确要求：预先分好五份任务，每份脚本完成一个实验后直接执行下一个，不再等本地网络恢复或定时补交。

## 执行方式

- 五条固定、互不重复的清单，只包含首四条lineage剩余开发分支及旧E复核。按预计耗时分配只改变执行顺序，不改变任何样本、配方、seed或科学门禁。已经提交的五个作业继续完成，不加入新清单。
- 每条清单由一个单PRO6000的Slurm脚本顺序执行。每项调用原始`code_e59b9ee` worker，进程退出0后核验原始输出、身份、有限更新、恢复段/检查点绑定和评估汇总，写出该项的描述性分数比较，然后立即执行下一项。
- 清单索引和完成回执持久保存。没有动态领取或跨清单偷取任务。全局登记锁及提交意图用于防止重复启动；失败或未知状态会写BLOCKED并阻止后续新实验，保留所有原件。
- 首四条88分支和16旧E均核验后停在`FIRST_FOUR_COLLECTED_AWAITING_ANALYSIS`。这不等于完成首批交付，不写`FIRST_FOUR_DELIVERED.json`，不扩展开发/调参，不冻结或提前测试。

## 首次交接与Slurm时限

教师QOS实测MaxSubmitPU=5，包括pending。不能在原五个作业外再排五个GPU作业，也不额外占一个该QOS的CPU控制作业。

用户级定时入口每五分钟只检查队列并提交已经写好的固定清单脚本，负责首次交接和作业时限接续。它不运行模型、不扫描原始数据、不在每个实验之间调度任务。GPU脚本内的实验连续运行，不等待该入口。首次空位的交接可能仍有最多一个检查周期的延迟及Slurm排队。

Slurm单次最多72小时。脚本在运行46小时后不再开始新实验，给原worker的24小时时限留足空间；正常退出后由入口重新提交同一清单，跳过已核验项。异常退出不自动重试。每次仍使用教师QOS、最多五个单PRO6000项目作业；历史扩卡限制不变。

## 证据与操作

服务器独立`fixed_pipelines_20260930/`目录保存配置、五份清单、分配/提交意图、逐项核验、Slurm日志和STATUS。原代码快照不修改。STOP文件阻止新实验及新清单作业；正在执行的一项仍可完成。BLOCKED需要审查原因，不能盲目删除或重新提交。

本地每三小时自动检查用于读取这些回执、分析与通知，不再调用原`submit-ready`抢占清单任务。断开本地SSH不影响已提交的Slurm脚本；服务器本身故障、维护、配额和作业失败仍会造成中断。

针对性验证：固定清单归属、重复执行拒绝、连续执行顺序、原始文件损坏拒绝、汇总错误拒绝、失败停止、STOP、未知提交以及GPU容量计数；并运行原任务登记调度测试。


## 2026-09-30部署清单

当前已有5项独立作业178090–178094，保持原提交。尚未提交47项固定分配如下：

| 脚本 | 分支 | 旧E | 总项数 |
|---|---:|---:|---:|
| 1 | 9 | 0 | 9 |
| 2 | 9 | 0 | 9 |
| 3 | 8 | 2 | 10 |
| 4 | 8 | 2 | 10 |
| 5 | 8 | 1 | 9 |

入口安装回执在current_evidence/FIXED_PIPELINES_20260930_INSTALL.json，配置与版本绑定在FIXED_PIPELINES_20260930_CONFIG.json。五份逐项清单为lane_1_tasks.json至lane_5_tasks.json。代码本地提交893f504，服务器原训练快照未改；GitHub推送仍受原数据外发审批限制。


2026-09-30T13:25:01Z: cron independently ran after installation SSH disconnected; five original GPU jobs were still active, no new lane submitted, no BLOCKED marker. GPU pipeline execution awaits a free slot. See current_evidence/FIXED_PIPELINES_20260930_CRON_VERIFIED.json.


2026-10-03补充：R6转空卡请求到达时，原lane2已完成GDPO并自动开始R6，交接前检拒绝重复，未迁移任务、未新提交、未更改五清单。辅助parallel_tail_handoff.py只供审计本次预备流程，现在不要运行。前检错误BLOCKED已归档HANDOFF_PREFLIGHT_REFUSAL_20261003.json，STOP不存在，两个原作业继续。已提前分析86项，见analysis_progress_20261003/README_zh.md。
