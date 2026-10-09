# MM-DEV F2 执行状态（用户取消，结果已删除）

**最终状态：`USER_CANCELLED_RESULTS_DELETED`。** 用户于2026-10-09明确要求
“停止这次的实验并删除结果”；此前“完整执行”的授权已撤销，不得恢复或重新提交。

2026-10-09 16:21（新加坡时间）已确认：控制器194848、后继194849、ENGINE194850
均为`CANCELLED`并已全部退出队列。三个本次服务器运行根已按逐文件哈希白名单删除，
共16,110文件、768,962,449逻辑字节；29个受保护相邻目录身份保持不变。
本地八个运行/结果副本目录的2,477文件，以及75个临时复核文件也已删除。
未发现本次实验的Codex定时自动化；其他实验作业193374、193375未操作，继续运行。

保留源代码、原始输入文档包、旧MM-CORE结果、共享模型/环境及不含实验payload的删除回执。
删除属于用户主动终止，不构成恢复资格通过、科学矩阵完成或实验结果交付。
详细回执见 [USER_CANCELLATION_20261009_zh.md](USER_CANCELLATION_20261009_zh.md)。

## 以下为取消前的历史快照（不是当前状态或继续授权）

最新状态核验：2026-10-09 14:14（新加坡时间）。原ENGINE194796概率一致性失败，
未提交参数更新。修复已完成本地验证、独立审阅与服务器重新冻结，重跑ENGINE194850
已提交，当前为`PENDING / Priority`。实际9B恢复验收尚未通过，正式科学矩阵尚未开始。
详细原因和失败证据见 [PROBABILITY_GATE_INCIDENT_20261009_zh.md](PROBABILITY_GATE_INCIDENT_20261009_zh.md)。

- 完整合同已获用户“完整执行吧”授权；固定的 77 个调度任务已登记。
- 修复版ENGINE **194850** 在老师QoS下排队；控制器 **194848** 正在CPU运行，
  后继 **194849** 为`PENDING / Dependency`。当前根76个后续任务等待、技术blocker为空。
  状态投影中的`ACTIVE`包含已排队作业，不能解释为GPU已运行。
- ENGINE **194796**：`FAILED 1:0`，GPU468秒，13:48:17结束。已保存192条回答、
  4957token、0次更新。平均概率差0.002979≤0.005，最大0.584116>0.05。
  Slurm确认account `rose`、QoS `soujanya-poria-startfund-2026-03`。
- CPU控制器 **194794** 与后继 **194795** 均按技术阻断退出（Slurm `FAILED 2:0`）；
  没有残留的原ENGINE控制链。76个后续科学任务均等待。
- 诊断 **194828** 仅强制回放192条已存答案，当前也为`PENDING / Priority`。
- 旧 ENGINE **194700** 已 `CANCELLED`，
  `ElapsedRaw=0`、无 `AllocTRES`，此前没有 GPU 结果或模型生成。
- 旧 CPU 控制器 **194699** 已经信号正常退出：`COMPLETED 0:0`，2068秒；
  后继 **194701** 已 `CANCELLED`，`ElapsedRaw=0`。三者均已离开队列。
- 76 个后续任务尚未执行。真实 9B GRPO 非零/恢复验收、34 条科学训练、5 个 PROBE、
  35 个 DEV_EVAL、完整分析、独立复算和汇报 ZIP 均未完成。
- 旧排队尝试 GPU 秒数已核实为0；这不是整个实验最终费用。
- 原冻结和原调度链完整保留。4692个数据/QA/材料化与视觉复核文件逐项比对旧冻结
  SHA256后复制到新根；没有重新生成数据，也没有改动科学配方。

## 已完成的验证

原包36项文件哈希和66项参考测试通过；原图与实际CPU处理器完成2336图/7008题，
固定10张接触表（原图和处理图各80样本）及20对全分辨率图片已实际复核。
原始来源、图像、处理张量、token/路由、模型13文件、依赖环境均核验。

概率路径修复后完整回归 **719 tests + 175 subtests PASS**，耗时141.13秒；
Ruff检查和格式检查38文件通过，独立只读审阅未发现阻断缺陷。
核心ENGINE包含独立CPU进程的step2/4张量、Adam、
scheduler、reference与RNG逐元素复算；真实9B ENGINE门禁尚未通过。

原服务器CPU冻结任务 **194681**：`COMPLETED 0:0`，32秒；
35个执行文件、4694个输入文件。
冻结SHA256：`d19d8f99398b4fcf4295ad3156cf94186744a4d010b2f321dc55951c16fcaed9`。

老师QoS的新CPU冻结任务 **194792**：`COMPLETED 0:0`，36秒；
35个执行文件、4694个输入文件。仅冻结执行源码中的`orchestration.py`因提交容量
保护发生改变，训练/模型/数据/分析执行源码未变。
新冻结SHA256：`b4279f3f1fcf46ef4e3e52326b15e74e89633dc2d8de4ec7f20d13d14435c3c9`。
源码与验证见`validation/TEACHER_QOS_VALIDATION.json`，实际调度回执见
`validation/TEACHER_QOS_LAUNCH_SNAPSHOT.json`。

修复版CPU冻结 **194841**：`COMPLETED 0:0`，32秒。35个执行文件、4695个输入；
多出的输入是`qa/TECHNICAL_REPAIR_PROVENANCE.json`，绑定被保留的失败来源与费用边界。
修复冻结SHA256：`f0c1a97676579058e09c2d126a1e2a78e51932df60dbc417da4b1325f6437c35`。
实际启动记录见`validation/CACHEFIX_LAUNCH_SNAPSHOT.json`。

## 身份与操作入口

- 当前修复运行根：`/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos_cachefix`
- 失败运行根：`/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos`。
- 原服务器根：`/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009`，完整保留。
- SSH别名：`eee-cluster`
- 原老师QoS部署：失败根下 `code`，运行代码commit
  `3557e66748cf2e8778d0e389a2c2fe0446ad7e86`，已push并核验远端SHA。
- 当前修复部署：修复根下 `code`，运行代码commit
  `d3334ca12de7dfd49df0df75126b17f147e18540`，已push并核验远端SHA。
- 原科学代码commit：`c65b406204e052aa35b2c0e9915c8470d5bfb413`。
- 原运维控制器commit：`8ee5e66169d75613650e907feda7ef739efdfdf1`。
- Git分支：`codex/mm-dev-f2-20261009`。
- 模型：Qwen/Qwen3.5-9B，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- GPU许可：account rose，用户指定的 `soujanya-poria-startfund-2026-03` QoS，单worker一张
  Pro6000，本计划最多五张并行。GPU小时仅记账，无累计GPU/研究墙钟门槛。
- 该QoS的MaxSubmitJobsPU=5，CPU控制器和依赖后继均占槽；满额时等待，不换QoS。
- 修复版容量查询剔除Slurm缓存保留的明确终态行，仍保留未知/活动作业名额；
  不改变科学任务或GPU小时规则。
- 集群覆盖内存请求：首个GPU请求实际为4 CPU/33 GiB主机内存/1 Pro6000；
  CPU控制器为2 CPU/6 GiB。后续实际分配继续以Slurm证据为准。

继续时先读当前 `orchestration/STATE.json`、`REGISTRATION.json`、
`manifests/ENGINE_GATE.json`（若存在）及该任务的Slurm/原始日志。
不要把未知提交、SSH失联、排队或部分产物视为失败并重复提交。
不改动其他任务、旧MM-CORE或SER-J23；不新增奖励、种子、数据或研究阶段。

## 自动跟进的授权边界

已启动的是本实验的固定调度器。另行尝试创建“Codex每30分钟自动回来检查、技术修复、
打包交付”的持续跟进被自动审批拒绝；审批理由是持续自主操作需用户额外明确授权。
该自动化**未创建**，已通过界面向用户询问，当前等待答复。
不得用另建自动化或间接执行绕过此拒绝。现有获准的实验调度继续运行。

最终交付仍需完整验证、实际汇报/原始包、CRC/EOF/逐文件hash及本地下载。
`verification/RELEASE_MANIFEST.json`本身不是已交付压缩包。
