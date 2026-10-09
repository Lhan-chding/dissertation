# MM-DEV F2 执行状态（未完成）

最新运行证据时间：2026-10-09T05:41:32.427929+00:00。用户指定的老师 QoS 已生效，
原排队链已停止，新 ENGINE 已实际运行。这不是实验结果或最终交付。

- 完整合同已获用户“完整执行吧”授权；固定的 77 个调度任务已登记。
- 新 ENGINE **194796** 已在 `gpu-pro6000-2` 上 `RUNNING`，实际分配一张 Pro6000、
  4 CPU、33 GiB 主机内存。Slurm 明确记录 account `rose`、QoS
  `soujanya-poria-startfund-2026-03`。Qwen3.5-9B 的760项权重已加载。
- 新 CPU 控制器 **194794** 正常运行；后继 **194795** 为 `PENDING / Dependency`。
  固定77任务已登记，ENGINE为`ACTIVE`，76个后续任务等待，未发现技术blocker。
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

本次修改后完整回归 **706 tests + 175 subtests PASS**，耗时142.16秒；
Ruff检查和格式检查36文件通过。核心ENGINE包含独立CPU进程的step2/4张量、Adam、
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

## 身份与操作入口

- 当前服务器运行根：`/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos`
- 原服务器根：`/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009`，完整保留。
- SSH别名：`eee-cluster`
- 新部署位置：当前根下 `code`，运行代码commit
  `3557e66748cf2e8778d0e389a2c2fe0446ad7e86`，已push并核验远端SHA。
- 原科学代码commit：`c65b406204e052aa35b2c0e9915c8470d5bfb413`。
- 原运维控制器commit：`8ee5e66169d75613650e907feda7ef739efdfdf1`。
- Git分支：`codex/mm-dev-f2-20261009`。
- 模型：Qwen/Qwen3.5-9B，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- GPU许可：account rose，用户指定的 `soujanya-poria-startfund-2026-03` QoS，单worker一张
  Pro6000，本计划最多五张并行。GPU小时仅记账，无累计GPU/研究墙钟门槛。
- 该QoS的MaxSubmitJobsPU=5，CPU控制器和依赖后继均占槽；满额时等待，不换QoS。
- 容量查询保守计入`squeue --states=all`短暂保留的已终止行，因此实际空槽可能稍晚
  才放行；这些行从Slurm缓存移除后下一tick自动推进，不改变科学任务或GPU小时规则。
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
