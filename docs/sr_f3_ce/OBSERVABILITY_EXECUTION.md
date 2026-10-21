# SR-F3-CE 前瞻观测修订与真实执行身份

此扩展在正式 science 前补充实际 image encoder、prefill/decode 与 teacher-forcing 观测。原科学部署 `71483fbf31d57213fdc7e6f3c881848397b45329`、`ROOT/code`、`RUN_IDENTITY.json`、原 source manifest、科学配置和已经执行的 P0/ENGINE 全部保留。不会重复 ENGINE，也不会改变训练矩阵、种子、模型、优化步数或采样预算。旧 ENGINE 未收集的细分成本明确为 `NOT_COLLECTED_DO_NOT_IMPUTE_OR_REPEAT_ENGINE`，不补成零或假定值。

`scripts/sr_f3_ce/instrumented_execution.py` 是显式启动入口；观测行为及计量边界见 [OBSERVABILITY.md](OBSERVABILITY.md)。不使用 `sitecustomize`，不向共享环境注入补丁，不将真实 wrapper 命令伪装成原 `run_worker.py`。

## 静态身份与部署

只从 `ROOT/native/observability/<40位扩展commit>/SOURCE_MANIFEST.json` 指定的目录执行。禁止搜索 latest 或任意额外模块。manifest 的相对路径允许安全子目录，例如 `tests/test_observability.py`、`docs/OBSERVABILITY.md`；禁止绝对路径、`..` 和解析后逃出扩展目录。只有 `executed_sources` 中的两个文件作为扩展运行源码加载，测试和文档作为验证原件保留。

```json
{
  "schema_version": 1,
  "experiment_id": "SR-F3-CE-20261016",
  "status": "FROZEN_OBSERVABILITY_EXTENSION",
  "extension_commit": "<40位真实扩展commit>",
  "base_code_commit": "71483fbf31d57213fdc7e6f3c881848397b45329",
  "run_identity_sha256": "<原RUN_IDENTITY文件SHA256>",
  "base_source_manifest_sha256": "<原deployment/SOURCE_MANIFEST文件SHA256>",
  "sources": [
    {"path": "instrumented_execution.py", "sha256": "<SHA256>", "bytes": 0},
    {"path": "observe_runtime.py", "sha256": "<SHA256>", "bytes": 0}
  ],
  "executed_sources": ["instrumented_execution.py", "observe_runtime.py"],
  "cpu_validation": {"path": "CPU_VALIDATION.json", "sha256": "<SHA256>", "bytes": 0}
}
```

上例大小和哈希是占位符，部署必须填写实际值。实际 sources 应同时保留相应测试与文档原件。`SOURCE_MANIFEST.json`、`CPU_VALIDATION.json`、`AMENDMENT.json` 不放入 sources，避免循环引用。

CPU_VALIDATION 至少包含 `status: PASS`、`scope: CPU_TOY_TRANSPARENCY_AND_ORCHESTRATION`、实际正整数 `tests_passed`、`source_sha256` 精确覆盖 sources 的每个路径，并明确 `gpu_validation_performed: false`、`new_model_generations: 0`、`new_optimizer_steps: 0`。这里后两项指本实验新增真实科学模型生成和优化，不把 CPU toy 透明性测试写成真实科学调用。保留实际测试命令、Python/torch 环境、退出码及原始输出证据；本地 CPU 与服务器实际 torch 的 CPU 验证应分别如实记录，不能声明 GPU 等价。

## API 与冻结顺序

- `verify_extension(root, manifest_path, science=False)` 验证静态源、CPU 原件、原 run/source identity；返回路径为绝对路径。`science=True` 还要求 amendment 与最终 freeze 的完整绑定。
- `install(root, manifest_path)` 验证旧源码和扩展后，返回 `InstrumentedController` class，并显式替换该进程内的 `sr_f3_ce.orchestration.Controller`。同一进程只允许相同 root、manifest 与 hash，CPU dispatcher 和交付构建器不安装模型 hooks。
- `seal_amendment(root, manifest_path, typed_verified=...)` 在全部 5 个原工程任务真实终态成功、无 science 目录和科学提交、无最终 freeze 时，封存原 ENGINE/P0 回执哈希、原 CONFIG/科学配置/固定矩阵、双盲发布及扩展身份。`typed_verified` 必须来自 dispatcher 的完整双盲证据验证。

顺序为原 P0/4 ENGINE 真正完成 → 完整 typed_v1 证据核验 → 封存 AMENDMENT → 原 `freeze_science` → 验证扩展静态链被 freeze 绑定 → 固定 train/eval/release。typed 已发布的准备代码保持 `4146e156ffa9a476b360daeb67503703faca138f`，脚本 SHA256 `1a75df6bf5fa1c0ef22a23ee7fd766a8e22f5a1c6c1fe7d579c4ab6135106aaf`；新 helper commit 不重新发布或重做双盲审阅。

原 `immutable_files` 直接纳入 native 下 JSON/Markdown 等，未直接收录 `.py`。因此最终 freeze 必须直接绑定扩展 SOURCE_MANIFEST、CPU_VALIDATION、AMENDMENT 三个文件；manifest 再绑定实际 `.py`、测试和文档原件。每次扩展进入时验证完整链，不把间接哈希称作原科学 source manifest 的变化。

两个扩展模块与原 worker 脚本采用读取源码 bytes → 对同一 bytes 检查已绑定 SHA/大小 → compile 同一 bytes 的显式加载流程，禁止 loader 再读文件或选择未登记 `.pyc`。首次导入原科学包前，driver、wrapper、builder 拒绝 `ROOT/code` 下实际存在的 `.pyc`/`.pyo`，并关闭字节缓存写入；发现缓存只拒绝，不删除原件。原 package 继续使用正常导入及其冻结文件/来源验证，不注入共享环境 importer。该防护针对已复现的静态未绑定缓存和显式加载竞争，不声明能防御恶意进程内存改写、所有文件系统竞争或替代第三方原生依赖的原环境审计。

## 实际提交、恢复和端点验证

所有未来 science 使用单 GPU、已登记任务和原资源上限。INTENT 除原字段外保存 `execution_entry_schema: 1`、绝对 `extension_manifest_path`、manifest/amendment SHA256、真实 `job_entry_command` 和每个 `worker_entry_commands`。claim_id 包含原 run identity、job key、manifest、amendment。batch script 的逐字节正文、sbatch 参数和每个资源字段均核验。

真实 batch 入口为 `instrumented_execution.py --mode job --root ROOT --manifest MANIFEST --job-key KEY`；真实 child LAUNCH 为 `--mode worker --root ROOT --manifest MANIFEST --task-id TASK`。job runner 保留原 held allocation、release intent、真实 GPU 可见性、唯一 task claim、resume chain、子进程 exit 和 task receipt 校验。原 P0/ENGINE verification 完整委托旧 Controller；未来 science 的真实 LAUNCH 单独校验，不修改证据来调用旧验证器。

worker 先验证原 scientific freeze、扩展与绑定，显式安装同一 Controller，使 endpoint 内部 `_completed()` 也验证新入口；随后安装纯 observer 并调用原 worker.main 一次。任何异常都保留失败，先尽力调用幂等 `observer.seal` 保存末态；事件日志缺失或不可读单独记为 `events_sha256: UNKNOWN` 和 `events_hash_error`，不覆盖原科学异常。异常回执本身不能写入时将该原因附在原异常上；不会因观测失败重跑 main。正常完成也 seal，任何未闭合 scope、未知 coverage、写入错误或未 sealed 状态都阻止完整观测回执。

TIMEOUT 仅沿原单 GPU science 精确恢复接口，通过原 CPU checkpoint/cache verifier 后才提交新 attempt。新 attempt 的真实 wrapper 和 amendment 不能丢失，旧目录和未完成观测永久保留。ENGINE 不自动重跑；COMPLETE 科学回执与失败作业并存时仍进入 receipt-only 核对，不重新采样。

## 观测原件与交付

每个 worker 的固定目录为 `accounting/observability/<job_key>/<task_id>/`，不写到 native 冻结树：

- `events.jsonl`：observer 事件加不可替换的 `execution_binding`，逐条 flush/fsync。
- `INSTALLATION.json`：`schema_version: 1`、`status: INSTALLED`、绑定及 observer.install 返回的原始安装身份，含实际原方法来源哈希。
- `FINAL.json`：只有原 worker/task 校验完成、完整 scope 逐项匹配、覆盖 COMPLETE、snapshot 已 sealed 且无 active/write_errors 才写 `status: COMPLETE`；绑定 INSTALLATION、任务回执、事件 SHA/bytes/rows/SCOPE_BEGIN/SCOPE_END 和末态 snapshot。
- `FAILURE.json`：异常类型/原异常、事件哈希、可取得的 sealed snapshot 或 snapshot 错误；缺少 FINAL 的轨迹不是完整观测。

`execution_binding` 固定包含 experiment_id、run_identity_sha256、execution_freeze_sha256、extension_manifest_sha256、amendment_sha256、job_key/job_id/task_id/claim_id。正常处理过的 OOM、`CHECKPOINT_EARLY_STOP` 仍保留实际成本；不能仅因存在 RAISED 就抹掉已恢复的科学路径。缺 END 或 UNKNOWN coverage 仍然拒绝完整观测。耗时是 host perf-counter 时间，不声称 CUDA kernel 时间；父子 scope 不相加冒充成本。

最终 builder 从 freeze 唯一解析扩展 manifest，导入前验证全部静态源/CPU 链，安装同一 Controller 后核验全部任务与原件。成功外部回执新增 `observability_extension`，其 manifest_path 为 run 相对路径，其余绑定 manifest_sha256、extension_commit、base_code_commit、observer_source_sha256、amendment_sha256、cpu_validation_sha256。dispatcher 逐字段核验后才能 DONE。

本地验证使用 `test_instrumented_execution.py`、`test_dispatch_registered.py` 和 `test_observability.py`，均无真实 GPU 或模型采样。这些测试不代替实际 CPU allocation、ENGINE 成功或最终科学交付证据。


## 2026-10-16 已授权的协议修订运行

以上固定 `71483fb...` 与“不重复 ENGINE”叙述描述原 v1 根。原 STOP 根、旧回执和旧原文保持不变。用户后续修复文件授权的 v2/C2 只能在独立新根执行；提示消融先在另一个独立根完成，不能添加到现有 33 个科学任务 ID 中。

运维入口仍对 v1 强制原提交。v2/C2 先逐文件校验新根的 SOURCE_MANIFEST，禁止未绑定字节缓存，再调用该已验证源码中的 `protocol.verify_run_source_revision`。它要求 `RUN_IDENTITY.run_stage=SELECTED_PROTOCOL_REVISION`，请求文件 SHA256 为 `83f3688707b09d69d84fb4f7ce3a9a7f2706b26c978176d97a6fb9637189c220`，根内 PROTOCOL_SELECTION 与独立消融根的完整原始结果逐字节相同，并重新验证九个消融任务的原文、终态和固定选择规则。源码提交必须同时被本根 RUN_IDENTITY 与 SOURCE_MANIFEST 精确绑定。

新版扩展 SOURCE_MANIFEST、AMENDMENT、OPERATION_SOURCES 和 OPERATION_IDENTITY 增加 `protocol_source_guard`，值为上述函数的完整返回对象（code_commit、protocol_revision、protocol_selection_sha256、request_sha256、ablation_root）。旧 v1 不新增该字段。观测扩展仍只在新根 P0/ENGINE 全部真实终态成功之后封存，之后用于 train/eval/release；“repeat_engine=false”指观测扩展自身不重复本根 ENGINE，不否定修复文件要求在新根重新生成 ENGINE/恢复/MONITOR。安装前未记录的阶段成本仍 UNKNOWN。

最终 builder 按选定 revision 重建 ENGINE、MONITOR 和端点 slot/prefill 身份。原 v1 完整压缩包与回执保存在 `protocol_revision/history_v1/`；消融原件保存在 `protocol_revision/ablation/`，路径镜像其独立运行根。2,304 条消融和 3,904 条历史 v1 原文分别统计，绝不替代新根的 2,880 条 ENGINE 诊断、512 条恢复缓存或 26,112 条 MONITOR。新根可用原 4146 finalizer 对未改变的外部双审原件重新发布，并用明确的 TYPE_REVIEW_REUSE_RECEIPT 说明来源；这不构成新增人工审核或模型评估。

`WORK_OCCURRENCE_TRAJECTORY.json` 从既有原文复算 768 个训练步的 work 出现率并核对训练回执，同时呈现 51 个固定 MONITOR 面板的全部回答与可解析回答两种分母；保留无效回答和非空 work 字符串计数，不改变奖励或路由。


## 已冻结科学运行的提交修复边界

2026-10-17 的明确用户授权及完整运维修复规则见 [PROTOCOL_REVISION.md](PROTOCOL_REVISION.md#2026-10-17-已授权的接续器与账本修复)。修复不替换本文件规定的 frozen InstrumentedController、job/worker 命令或观测方法。只有未分配的明确拒绝可沿原 INTENT/script/claim 在独立运维日志中重提；一旦得到真实 job_id，继续走原 held → allocation/script核验 → release → worker/observer → 完成回执链。已完成任务只做完整验证后的账本结清，既有失败或OOM不删除。

紧邻 sbatch/release 的新容量守卫位于单独运维 Slurm 适配器。重试前的源码、回执及原生恢复验证仍使用冻结代码；它们结束后再取队列，避免哈希耗时使容量判断过期。队列竞态导致的明确QoS拒绝追加原件并等待容量变化；结果未知则停止补交。服务器SOURCE_MANIFEST、AMENDMENT和科学冻结身份不因这次运维修订改变。
# 2026-10-17 接续器轮询开销修订

用户授权查明长期不补交的原因、修复并利用空闲额度，同时明确要求省去多余哈希与身份检查。
本次只新增独立的 CPU 运维版本；冻结科学源码、原观测扩展、训练设置和历史回执不变。

复现原因是 `Driver.act → status → tick → status/submit/status` 反复扫描完整历史，
而 `attempts` 对每项作业反复调用完整 `identity`。原身份验证还会重放历史协议验证。
在满额的真实 Controller CPU fixture 中，一轮触发 3 次 status、96 次 identity、
100 次 protocol.verify_run；其队列在耗时检查前读取，空位出现后仍可能按旧满额状态返回。

新的 `dispatch_efficiency.py` 只在一个已冻结的 CPU Controller 实例上工作：

- 每次 CPU 分配启动时执行原完整门禁；之后身份调用仅检查小型冻结配置及授权文件、路径和 STOP。
  不再在日常轮询中重放历史协议或扫描全部科学负载；不替换 worker 或全局 Controller 类。
- 先获取当前队列、有效预留与轻量任务账本。满额直接等待，有空位才调用原 `submit(kind)`。
  固定矩阵提示仅用于减少无效工作，任务选择、认领、完整依赖核验仍由原 submit 执行。
- 原 submit 因验证期间的旧队列返回 None 后，只要仍有可用额度，下轮即使签名相同也再次检查。
  `GuardedSlurm` 仍在真正 sbatch/release 前重读队列；未知提交继续占位，明确未分配拒绝才可安全重提。
- 新完成任务仍通过原 verifier 后结账，恢复和最终交付仍使用完整门禁。没有已完成科学结果缓存，
  不将文件数当作优化更新数。CPU release 只要求提交名额，不要求空闲 GPU。
- `EFFICIENCY_STARTED.json` 记录启动时耗时，`PHASE.json` 与独立 phase 回执标明当前慢操作，
  `QUEUE_SNAPSHOT.json.queue_observed_at` 区分读队列时间和写回执时间。

更换正在运行的 CPU 接续器时，必须先取得原 orchestration 租锁，确认不存在未确认的提交/放行，
只结束明确指定的旧 CPU 作业，保留全部原件并核实队列消失与终态 accounting。
新的 `EFFICIENCY_AUTHORIZATION.json` 只允许该精确 CANCELLED 前驱交接，不授权科学作业重试。
旧 CPU 没有 END 时保持缺失，不补造成功回执。旧失败和旧运维部署仍完整保留。


## 固定任务池的独立运维适配（2026-10-17）

已授权的 `pooled_execution.py` 只在独立 operation 目录执行。它保留原 InstrumentedController 类对象，向该对象安装显式 schema2 的 submit/verify/run 分支，并为原 native `_verify_intent` 增加 schema2 分支；schema1 的历史验证继续委托原实现。这样新建 Controller 的 endpoint_runner、原 worker 和最终 builder 读取同一份真实账本，不修改冻结源码或其清单。

operation 注册接受 `--pool-authorization`。授权约定按实际调度额度组合固定任务、每任务一张卡、原实验设置不变、无新增任务且完整保留历史。父作业和每个子进程均从该 operation identity 加载适配；每个子进程仍执行原 native worker 并写原 observer 记录。N卡作业用N GPU、4N CPU、90N GiB主存（N按实时QoS、关联额度、单节点资源和ready任务数决定），每个worker用4 CPU线程；CPU release仍0 GPU、4 CPU、12 GiB。设备必须来自本次真实 Slurm 分配且互异。

新池的一个任务完成后仍保留独立完成原件，父进程逐个收集 EXIT；作业终态不成功时不把整池结账为 COMPLETE，也不自动重投任何成员。`POOL_HANDOFF_REQUIRED.json` 阻止把科学失败误当成 CPU 租期交接。未知提交或未知释放仍占用实际声明的全部GPU和一个提交名额。

builder 的 `--operation-identity` 显式绑定新运维目录，独立核验 helper manifest 并在自己的原 native Controller 上安装相同适配；最终回执增加 `task_pool_operation`。dispatcher须核对该字段后才能声明最终交付成功。未提供池身份的旧builder不能接受schema2账本。


2026-10-17 用户对实际QoS额度的更正优先于本文较早阶段的五卡运维限制。
新 `scheduler_capacity.py` 每次实际 sbatch/release 前读取 `sacctmgr` 的QoS及可见用户关联额度、
`sinfo` 的单节点GPU/CPU/内存容量，再读取最新队列。共享QoS GPU使用量包含其他用户与未知预留；
本用户提交名额独立计算。日常查询短暂缓存，不读取模型或checkpoint。隐藏的祖先关联、billing分钟和
运行数限制仍由Slurm执行，提交拒绝保留并由既有安全接续逻辑处理。
OPERATION_IDENTITY绑定 `LIVE_SLURM_QOS_AND_ASSOCIATION`，不记录人为固定GPU上限。
每个新intent保留当次不可变容量快照；历史完成验证只读该快照，不因后来的额度下降改写历史。
原CONFIG中的历史数值保持原件，运维适配覆盖提交、释放与旧单卡恢复入口的调度判断；核心科学常量不改。


## CPU 接续器独立普通 QoS（2026-10-17）

用户授权把 CPU 接续器迁出老师 QoS。独立 operation 的 `dispatcher_qos=rose`
必须与本次 EFFICIENCY_AUTHORIZATION 中的显式字段一致；旧版本缺省仍使用老师 QoS。
该字段只作用于 dispatcher 的提交、放行和租期接续，不改变 `qos` 科学任务字段、
原 CONFIG.cpu_policy、训练/评价/最终交付的原科学入口。科学任务仍按老师 QoS 真实额度运行。

CPU 提交及放行前先读取普通 QoS 与用户关联的实际限制，再读新鲜队列；
普通 QoS 的本用户提交数与老师 QoS 分开统计，存在有限用户关联额度时跨 QoS 合计。
运行作业上限只决定何时获配，不冒充提交上限。不能因老师 GPU/提交额度已满而阻塞普通 CPU。
历史 CPU 的 terminal QoS 始终核对其原 INTENT；安全交接保留原回执与缺失 END，
只退役指定旧 CPU，未知提交/释放不重投，不取消科学或其他作业。

## 2026-10-18 滚动端点池运维回执

授权的schema3端点池将固定任务成员数与真实GPU数分开记录；训练和CPU release继续schema2。单设备复用必须由前任成功EXIT及原任务/observer核验触发，每个任务只写一次LAUNCH。最终verify_job核对slot_index、设备、predecessor_task、有限数值时间戳以及退出不晚于后继启动；再走原冻结产物核验。提交/释放仍采集实时容量证据，未知分配始终占N卡和一个提交名额。

新运维验证旧schema2时，从其不可变pool_operation_identity_path及OPERATION_SOURCES加载原Pool作为只读验证器，不安装第二个Controller，不替换科学源码。旧源码/身份发生变化则拒绝验收。CPU接续器仅在共享提交锁内交接，旧作业终态和原件保留；正在运行的科学作业不取消、不重投。


## 2026-10-18 未启动七卡端点池拆分

用户明确授权“七卡拆成两个小作业吧”。仅针对仍处于 PENDING 且没有任何 STARTED、LAUNCH、observer 或端点产物的原七卡池，先在共享锁内暂挂并再次核对状态，再以仅取消 PENDING 作业的命令退役。必须取得真实 CANCELLED/0:0、从未运行的 accounting、零已分配资源与队列消失证据；任何启动竞态或未知终态均停止拆分。保留原 INTENT、STATE、提交/放行和取消回执，不覆盖或删除历史。

新运维授权绑定独立 supersession 回执及原 intent/state/terminal 哈希；只在有效任务视图中排除该精确前任。15项原端点按固定顺序拆成4卡8项、3卡7项两组，每组仍逐卡成功验收后连续补位。每组整体认领一次，按实时 QoS/关联/节点容量分别提交；不足一组资源时等待，不拆成额外任务或越过限制。新 worker、接续器及最终验收均验证同一绑定，旧训练与七卡池原件不改。

实际提交和放行仍必须通过紧邻调用的新鲜容量守卫。此次已发生的 CPU 共享容量异常退出保留 FAILED 回执；只允许同时绑定本次已核验拆分、具体故障与真实终态的下一 CPU 租期交接，不扩展为任意失败自动重试。原超额报错规则保持不变。科学设置、冻结文件与其他作业不变。


## 2026-10-20 原 CPU release 作业的节点例外

用户明确授权在测试通过后，把仍未启动的原 release 作业 `215520`
（`job-164520339dd44ae6`）从 `cpu-2` 迁至 `cpu-1`，保留科学设置、job_id、
claim、原脚本及全部历史；如果已启动则停止迁移。此次只部署新的独立
`operations/<commit>/migrate_cpu_node.py` 与 `cpu_dispatcher_pause.py`，
不替换旧 operation、科学源码、原 CONFIG.cpu_policy 或原脚本中的 cpu-2。
历史请求仍是 cpu-2，独立例外回执记载实际生效的 cpu-1，不能把两者描述成同一原始配置。

`MIGRATION_AUTHORIZATION.json` 精确绑定原 INTENT、run.sbatch、ALLOCATION_VERIFIED、
RELEASE_INTENT、RELEASE、CONFIG、EXECUTION_FREEZE、RUN_IDENTITY 和旧 OPERATION_IDENTITY
九份原件的 SHA256，以及新提交、两个 helper 的源码哈希与用户原始授权。
仅允许原作业 `scontrol hold` → 复核未启动 → `scontrol update JobId=215520 ReqNodeList=cpu-1`
→ 验证资源/原脚本/原件 → `scontrol release`。不调用 sbatch、scancel 或 requeue。
每个调用前后均追加不可覆盖回执；未知结果、已有失败尝试或身份变化不自动重试。
运行/分配/STARTED/LAUNCH/observer/任务完成的任何证据都会阻止待启动迁移。

旧接续器在每次轮询中非阻塞获取同一 SUBMIT.lock，不能让它与迁移争锁而异常退出。
操作可通过 `srun --jobid=211333 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1`
在原 CPU 分配内运行独立辅助步骤。监督器仅在 cpu-2 核对原 dispatcher 的 UID、
完整 argv、SLURM_JOB_ID、可执行文件和 /proc startticks 后暂停该单一 Python 进程，
再由受监督子进程非阻塞获取原 SUBMIT.lock。锁被占则停止操作并恢复进程。
监督器最长暂停 120 秒；异常或超时先终止并回收迁移子进程及其子命令，再恢复原 PID。
原 CPU 作业、原 dispatcher 源码和租期不变，PAUSED/RESUMED 等记录全部保留。

迁移后的在途监控只用既有轻量队列/任务账本路径；不要手动调用旧 status/tick/refresh
对 cpu-1 活跃作业进行原 cpu-2 分配验收。旧 worker、最终 builder 与完成结账仍沿原绑定
执行。旧历史验收本身没有证明节点例外，因此旧 builder 的 PASS 或 dispatcher DONE
只是最终交付的必要条件。

完整交付还必须由新 helper 的 `--mode verify-final` 写出
`delivery/FINAL_MIGRATION_VERIFIED.json`：核对真实 215520 在 cpu-1 的 COMPLETED/0:0、
4 CPU/12 GiB/0 GPU、原 claim/LAUNCH/任务与 observer、原 builder 的全部检查及 ZIP 哈希、
dispatcher 的交付绑定与 DONE，并逐字节核验 ZIP 已归档原九份文件、完整迁移链、
授权、新源码和暂停/恢复原件。补充验收不持共享提交锁，不修改原 ZIP；其回执单独交付并
绑定该 ZIP 与原 builder 回执。缺少任一部分，监控不得宣布完整交付或自行删除。
新的 helper、授权和审计文件名不含 `pending`，避免旧 builder 对临时 pending 文件的排除规则。

## 2026-10-21 高内存 CPU 交付恢复

原 dispatcher `211333` 在最终 builder 阶段被 Slurm 以 OUT_OF_MEMORY 终止，
原 DELIVERY_PROCESS=-9、END=FAILED 原件保持不变。训练12项、端点15项和
release `215520` 已成功完成；用户仅授权重做打包和验收，不重跑这些任务。
独立恢复不改变 CONFIG、EXECUTION_FREEZE、旧 operation、原 builder、原 claim 或迁移原件。

三个新 helper 独立部署到 `operations/<commit>/`。`RECOVERY_AUTHORIZATION.json`
绑定用户授权、原失败、冻结身份、原 builder 命令及源码、release/observer 完成、
迁移授权和源码、三个新 helper、原科学与观测 source manifest，以及唯一恢复脚本。
`submit_delivery_recovery.py` 在独立锁内核验新鲜 rose QoS/用户关联/队列和 cpu-1 容量，
为打包恢复申请 22CPU/66GiB/0GPU/72h，先落盘 MAY_HAVE_SUBMITTED，再唯一 held sbatch。
提交后核验未启动、资源及实际 batch script，重新核验容量后释放同一恢复 job_id。
任何不明提交或释放均停止，不自动重投；不触碰其他作业。66GiB 是运维资源例外，
不是科学设置变更，也不能据此保证不会再次 OOM。

`recover_delivery.py --mode run` 在恢复分配内核验真实 Slurm 身份及原件后，仅执行
原 DELIVERY_INTENT 中完全相同的 builder argv，流式保存其日志。STARTED 前不得存在
历史恢复尝试或正式 ZIP。Slurm 放行与 RELEASE 回执之间允许最多60秒等待，未取得
完整绑定则停止。原 builder 的七项检查、冻结身份、233340条记录等完整覆盖及 ZIP 哈希
通过后，写恢复目录自己的 DELIVERY_VERIFIED 和 END=DONE，不改写旧 FAILED。

恢复 job 真正 COMPLETED/0:0 后，`recover_delivery.py --mode verify-final` 核验
原 builder、实际 cpu-1/22CPU/66GiB/0GPU、提交/放行/启动链、旧失败绑定及 ZIP 中的
全部恢复前置证据，写 `delivery/FINAL_RECOVERY_VERIFIED.json`。随后
`verify_recovered_migration.py` 复用哈希固定的原迁移链验证，保留原215520的终态、
claim、LAUNCH、task、observer、迁移和暂停恢复归档检查；仅以本次明确授权的恢复
成功链替代原 dispatcher 已不可能取得的 DONE。它写独立的
`delivery/FINAL_MIGRATION_VERIFIED.json`，同时绑定原 FAILED、新恢复、原 builder 和 ZIP。

交付必须同时提供原 ZIP、原 `.verification.json`、恢复与迁移两份补充回执。
后置验收回执在 ZIP 建成之后产生，以外部哈希绑定补充，不反向修改已验收 ZIP。
三重验收与本地文件传输哈希均通过后才删除监控。只完成数值发布、恢复END或单项PASS
均不得宣布完整交付。恢复期间不重复启动昂贵 builder/全量验收轮询。

按用户节省空间要求，失败大型临时打包产物仅在确认所属失败进程已经终止、没有活跃
写入者且不是唯一科学证据后清理，保留小型失败回执与删除清单。2026-10-21 01:48 SGT
预检原 delivery 目录为空，本次 OOM 尚未留下可删除的大 ZIP；不虚报释放空间。

2026-10-21 02:03 SGT 首个独立恢复申请216128被集群提交插件按每核3GiB
强制改为4CPU/12GiB，资源门禁在hold状态拒绝放行，没有运行builder。
按[集群CPU规则](https://github.com/NTUEEECluster/docs/blob/main/slurm-guide.md)，
达到至少64GiB须申请22CPU/66GiB，不能修改已提交作业的资源。02:11:13 SGT
经原脚本、PENDING/JobHeldUser、零运行和零分配复核后，只对216128执行一次
`scancel --state=PENDING 216128`，终态CANCELLED/0:0、Start=None、ElapsedRaw=0、
AllocTRES为空、NodeList=None assigned，队列已消失；原申请、拒绝和首轮
节点空值检查回执保留。后续独立恢复授权必须哈希绑定该精确SUPERSESSION证明，
原83e授权、源码、脚本、SUBMITTED和取消命令回执；运行及最终验收再读真实sacct，
并验证ZIP逐字节归档上述证据。此例外不允许任意失败自动重试或重跑215520。

### 216142 同作业记账延迟接续

249539a 独立部署通过服务器210项测试后，02:17:39 SGT唯一提交216142，
held资源已准确为22CPU/66GiB/0GPU。紧接提交的第008次sacct命令返回0但stdout为空，
原门禁停止放行。02:20真实记账已是PENDING、Start=Unknown、ElapsedRaw=0、
AllocTRES为空，原脚本和job_id未变。这是提交后的记账传播延迟，不能当作builder失败，
也不能重新sbatch。

独立 `resume_held_delivery.py` 只允许此精确216142和249539a授权。新授权绑定
全部原16份命令文件、原源码三件、脚本、SUBMITTED、MAY_HAVE_SUBMITTED和容量证据。
原第008次空结果保持原样；前7对实际执行过的命令回执逐字节复制至新命令链，
另存复制谱系，明确这些是历史命令副本而非重新执行。补齐新第008次记账检查后，
继续原格式的第009–016次新鲜容量、held、脚本、零运行复核及一次release。
全程持原SUBMIT.lock；已有STARTED/RELEASE/接续意图或未知操作结果拒绝盲目重试。
不再调用sbatch、scancel或requeue，不修改249539a源码、授权、原脚本或原8对命令。

所有须归档的侧车授权、源码、复制谱系和接续结果先完成，再发布原RELEASE.json，
避免worker提前构建ZIP遗漏审计。运行后沿用原worker和原builder。最终侧车
`--mode verify-final` 先调用原恢复和迁移验收，再逐字节检查ZIP中的侧车证据，
写 `delivery/FINAL_HELD_RECOVERY_VERIFIED.json` 绑定同一ZIP及另外三份验收回执。
此附加门禁也通过且本地传输哈希一致，才可宣布交付并关闭监控。


### 216142 容量等待后的精确接续

ab7 同作业记账恢复在第012次节点容量查询处停止：cpu-1当时已满，
未发布CAPACITY_BEFORE_RELEASE、RELEASE_INTENT或RELEASE，216142仍held且零运行。
原001–012、ALLOCATION_VERIFIED及ab7部分审计原件保持不变；不得直接重跑
249提交入口或ab7一次性恢复入口，也不得重新sbatch。

独立 `continue_held_delivery.py` 仅处理这一已知未执行release的精确断点。
新授权绑定原249授权和脚本、ab7授权及源码、原提交链、ab7部分命令与审计。
容量不足时只在独立时间戳目录保存只读查询并返回WAIT_CAPACITY，可以重新检查；
未知命令结果、身份变化、启动证据或已经写出的放行意图不获得自动重试权限。
节点状态、22个空闲有效CPU、67584MiB调度可用内存及rose/关联额度均须通过。

成功接续另建命令链，前八对回执复制并记录来源，不重执行历史sbatch；
第009–016使用新鲜容量、未启动、原脚本和唯一release回执。
原ALLOCATION_VERIFIED逐字节保留。完整新审计先落盘，再原子发布RELEASE，
原249恢复worker与原builder继续使用同一216142、原脚本和科学设置。

最终独立接续验收生成 `delivery/FINAL_HELD_CAPACITY_VERIFIED.json`，
保留原builder、恢复、迁移门禁，补充核验ZIP中的所有历史
容量中断、复制谱系、只读等待和成功放行证据。它明确替代未成功完成的ab7
FINAL_HELD_RECOVERY_VERIFIED路径，不声称旧ab7门禁已经通过，也不改写旧失败。
只有真实恢复COMPLETED/0:0、所有适用验收及下载后的本地哈希均通过，才能
给用户交付原ZIP和外部绑定回执并关闭监控。


### 2026-10-21 老师QoS唯一替代打包作业

用户明确允许确认216142从未启动、零分配后精确取消，保留审计，再使用
`soujanya-poria-startfund-2026-03`提交唯一替代打包作业。此授权替代此前
保留216142 job_id及原脚本作为执行入口的要求；原文件继续作为历史保留。
科学源码、12条训练、15个端点、215520、原builder命令和冻结设置均不变。

独立 `teacher_delivery/` 运维源部署在新commit目录，原249、ab7、b77源码
和授权不覆盖。新授权绑定216142原提交链、所有已知容量等待及最后缺RESULT
的完整查询原件；不补写旧RESULT，不声称旧b77最终门禁已成功。先核验
held身份、保存脚本和零运行/零分配，再唯一精确取消PENDING216142；
只有真实CANCELLED/0:0及队列消失后才可提交替代作业。未知操作不能重试。

新作业仅用CPU，Account=rose，老师QoS，22CPU/66GiB/0GPU、72h，
仍使用cpu-1和cpu_ok约束。提交额度按当前QoS和用户关联核算；held登记
与实际放行分开。记账传播延迟和已知容量不足仅重复独立只读查询，
放行仍要求IDLE或MIXED、足够空闲核和调度内存；COMPLETING/PLANNED
不绕过。原一次性sbatch或release结果未知则停止，不新增重复作业。

最终须保留原builder七项PASS及233340条覆盖，验证老师QoS恢复终态、
旧211333失败、原215520迁移、216128退役及216142退役和全部历史查询
逐字节归档。独立老师QoS补充回执绑定唯一ZIP和适用恢复/迁移回执，
明确替代未完成的旧rose接续验收路径。ZIP不反向修改；下载后SHA256
一致并交付可点击文件，才删除监控。该段是授权边界，不代表作业已提交
或验收已通过；实际job_id和阶段以真实回执为准。

## 2026-10-21 低内存打包修订与唯一4核恢复

用户明确授权：定位打包内存热点并流式/分批改造，科学结果与原验收标准不变；
确认216897未启动、零分配后可精确取消，提交唯一4CPU/12GiB/0GPU老师QoS替代作业。
12GiB是首次全量实测上限，不是已测得的最低需求。原215520、12训练与15端点不重跑。

`build_delivery.py`逐面板处理原文，内存基准与边界见`DELIVERY.md`。
独立`lowmem_delivery/`保持原恢复、迁移和全部退役历史，并把新builder与原358运维
通过`DELIVERY_REVISION.json`显式绑定。旧358/3a7/249/ab7/b77/4401源码与授权不改。
原211333 OOM/END FAILED保留；未成功的旧补充门禁不伪造PASS。

新脚本使用老师QoS、account rose、单节点、4CPU/12GiB、`cpu_ok`，不请求GPU，也不
指定固定nodelist；实际节点必须是x86_64的cpu-1或cpu-2。容量查询同时核对两节点，
只有精确IDLE/MIXED且至少4空闲有效核与12288MiB调度可用内存的节点才合格。
某候选的COMPLETING/PLANNED/DOWN/DRAIN等状态不使另一合格节点失效；全部不合格时
保存完整WAIT_CAPACITY回执，不消耗一次性放行机会。缺失/未知命令结果或放行意图
不允许盲重试。旧216897的完整历史、原脚本、未启动证明及精确取消终态/队列消失
必须通过，之后才能唯一sbatch；MAY_HAVE_SUBMITTED出现后永不重复submit。

新worker仅通过`profile_delivery_memory.py --mode real-build`运行修订builder的原打包验收。
MEMORY_PROFILE记录builder子进程RSS与源哈希、成功回执hash；PROCESS另记录可用的
当前worker cgroup-v2峰值，不能把某step读数冒充整个Slurm作业总峰值。
最终要求原七项PASS与233340覆盖、FINAL_RECOVERY_VERIFIED、FINAL_MIGRATION_VERIFIED、
FINAL_LOWMEM_RECOVERY_VERIFIED绑定同一ZIP、修订身份及内存回执。后者明确替代此前未成功
的held/teacher补充门禁。原ZIP不反向改写；下载ZIP与验收/内存回执并核对SHA之后才结束监控。

完整真实打包尚需计算节点实测；热点基准成功、单元测试通过或新作业已提交均不等于交付完成。

## 216983 重复CPU约束的同作业规范化

唯一4核替代216983首次接续读取到`Features=cpu_ok&cpu_ok`，原538门禁精确要求
`cpu_ok`，因此在第一条held查询后停止，未放行。原poll只有完整001 intent/result，
不补写RESULT、不删除原件，不重跑原continue。原worker运行时也校验该字段，
不能只放宽接续器或改写查询回执。

独立`normalize_lowmem_constraint.py`仅绑定216983、原538授权和这一个已知断点。
它在原共享提交锁内再次核验held身份、原存储脚本、零运行和零分配，唯一执行
`scontrol update JobId=216983 Features=cpu_ok`；这只是同一约束重复AND的等价规范化，
不移除cpu_ok、不改变4CPU/12GiB、QoS、job_id、脚本或科学设置。只有调度器真实
返回cpu_ok、原脚本及零分配再次通过才接受；未知结果、命令失败、再次重复追加
或任何启动/放行证据都停止，不盲重试、不提交另一作业。

新授权全文、完整源码UTF8与SHA、前后查询和唯一修改命令均保存在独立
`commands/poll_constraint_normalization_216983` JSON审计。规范化之后的已知容量
不足仅保存完整WAIT回执，可重复只读检查；原严格节点、CPU、内存、QoS门禁不变。
条件具备时新建真实16命令链，历史提交只复制并记录谱系，原538
`prior_release`必须能回放接受；原缺RESULT查询仍逐字节纳入全部teacher审计。

最终侧车先执行原538完整原交付、恢复、迁移和低内存验收，然后核对同ZIP内的
规范化源码、授权、全部命令和等待链，输出外置
`FINAL_CONSTRAINT_NORMALIZATION_VERIFIED.json`。该补充验收及本地下载SHA也通过
才算完整交付；不反向修改ZIP、不伪造历史门禁、不把规范化成功当作worker已启动。

## 2026-10-21 216983未启动精确替代

集群明确拒绝修改216983已提交Features；e184规范化命令返回1，STOP原件保留。
用户随后明确授权：确认216983未启动、零分配后精确取消，再提交唯一同规格
4CPU/12GiB老师QoS替代作业，用修复后的程序完成内存实测和打包验收。
独立`lowmem_replacement/`绑定原538授权、脚本、唯一不完整查询poll及e184拒绝
审计，旧源和旧回执均不改。只认可这个已知未放行断点，不泛化失败重试。

新held和live分配门禁仅接受`cpu_ok`或集群重复AND的`cpu_ok&cpu_ok`，保留
查询原文；不接受其他逻辑表达式。退役必须取得CANCELLED/0:0、零运行、零分配
和队列消失证明，然后才允许唯一sbatch。新工作目录为
`operations/delivery_recovery/20261021_delivery_lowmem_replacement`。
原cpu-1/cpu-2候选、4CPU/12GiB/0GPU、72h、节点及QoS容量守卫保持不变。

原builder七项检查、233340条覆盖、内存PROFILE成功、恢复及215520迁移验收
仍须全部通过。`FINAL_LOWMEM_REPLACEMENT_VERIFIED`补充回执必须绑定同一ZIP
及216983退役、原未完成查询和被拒绝规范化的完整归档；明确取代未成功的旧
低内存/规范化补充门禁，不声称它们PASS。12训练、15端点、215520不重跑，
冻结科学设置不变。下载ZIP及全部适用回执、本地SHA一致后才交付和关闭监控。


## 2026-10-21 取消217029后的正常调度队列交付

用户明确要求取消217029并尽快用合法方式完成完整打包。已在共享提交锁内核验
原job、脚本、PENDING/JobHeldUser、Start Unknown、Elapsed0及空AllocTRES，精确
`scancel --state=PENDING 217029`；终态CANCELLED/0:0、队列消失。原件不改，
审计在`operations/delivery_recovery_handoff/217029_user_cancel`。

独立`queued_delivery/`使用唯一4CPU/12GiB/0GPU、老师QoS、account rose、12小时
上限的CPU批作业。科学652e52d、12训练/15端点/215520及已验证流式builder不变。
授权明确绑定`queue_policy=SLURM_MANAGED_CPU_QUEUE`。提交先hold核验身份、脚本及
零启动分配，再唯一release进入正常Slurm队列；不再以两节点当下空闲或IDLE/MIXED
作为release必要条件。物理资源和节点可用性由Slurm负责，不强制分配COMPLETING/
PLANNED节点、不修改节点状态。完整新鲜QoS/关联提交容量、两候选节点配置、CPU架构
和cpu_ok约束仍必须核验；worker实际启动时照常验真分配4CPU/12GiB/0GPU、节点及脚本。
旧人工hold等待可能额外错过资源释放窗口，新队列由调度器持续考虑；这不保证立即启动。

本次显式退役旧217029，不再调用其接续器。新流程只有一次sbatch机会，未知提交/放行
结果禁止盲重试，完整WAIT_ACCOUNTING/WAIT_CAPACITY可按新入口续接同一job。
新worker仍只运行原打包命令的流式修订并记录真实子进程峰值；12GiB是首次全量上限，
不是已证明最低需求。12小时是新操作上限，不是交付耗时承诺。

最终必须原ZIP.verification七项PASS、233340覆盖、FINAL_RECOVERY_VERIFIED、
FINAL_MIGRATION_VERIFIED、FINAL_QUEUED_RECOVERY_VERIFIED及MEMORY_PROFILE绑定同ZIP。
新补充验收明确替代未成功的先前恢复补充路径，并逐字节覆盖原失败、全部退役、迁移、
217029取消和新提交放行执行链；不伪造旧PASS。下载唯一ZIP与回执并分块SHA核验后
交付、删除监控。未完成前不声称交付，也不重跑科学任务。
