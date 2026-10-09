# MM-DEV F2 执行状态（未完成）

证据时间：2026-10-09T04:59:47.657115+00:00。这是启动状态记录，不是实验结果或最终交付。

- 完整合同已获用户“完整执行吧”授权；固定的 77 个调度任务已登记。
- 科学阶段仍为 `F2_FROZEN`。ENGINE 作业 **194700** 是 `PENDING / Priority`；
  控制状态里的 `ACTIVE` 包括已登记的排队作业，不能解释成 GPU 已运行。
- CPU 控制器 **194699** 在 cpu-2 运行，后继 **194701** 以 afterany 依赖排队。
  控制器会在工程验收通过后推进固定矩阵，技术异常会保留证据并阻断相应任务。
- 76 个后续任务尚未执行。真实 9B GRPO 非零/恢复验收、34 条科学训练、5 个 PROBE、
  35 个 DEV_EVAL、完整分析、独立复算和汇报 ZIP 均未完成。
- 本任务未观测到技术 blocker。当前 GPU 未获分配，记账未知项保留；不能把摘要中的
  `known_actual_gpu_seconds=0` 当作整个实验最终费用。

## 已完成的验证

原包36项文件哈希和66项参考测试通过；原图与实际CPU处理器完成2336图/7008题，
固定10张接触表（原图和处理图各80样本）及20对全分辨率图片已实际复核。
原始来源、图像、处理张量、token/路由、模型13文件、依赖环境均核验。

本地完整回归 **696 tests + 163 subtests PASS**，Ruff检查和格式检查通过。
额外CPU控制器边界回归1项通过。核心ENGINE包含独立CPU进程的step2/4张量、Adam、
scheduler、reference与RNG逐元素复算；真实GPU产物尚不存在。

服务器最终CPU冻结任务 **194681**：`COMPLETED 0:0`，32秒；
35个执行文件、4694个输入文件。
冻结SHA256：`d19d8f99398b4fcf4295ad3156cf94186744a4d010b2f321dc55951c16fcaed9`。

## 身份与操作入口

- 服务器运行根：`/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009`
- SSH别名：`eee-cluster`
- 科学部署：运行根下 `code_c65b406204e0`
- 科学代码commit：`c65b406204e052aa35b2c0e9915c8470d5bfb413`
- 运维控制器commit：`8ee5e66169d75613650e907feda7ef739efdfdf1`
- Git分支：`codex/mm-dev-f2-20261009`，两次commit均已push并核验远端SHA。
- 模型：Qwen/Qwen3.5-9B，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- GPU许可：account rose，已允许的 `override-limits-but-killable` QoS，单worker一张
  Pro6000，本计划最多五张并行。GPU小时仅记账，无累计GPU/研究墙钟门槛。
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
