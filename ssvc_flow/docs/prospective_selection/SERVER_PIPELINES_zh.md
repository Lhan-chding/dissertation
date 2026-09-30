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
