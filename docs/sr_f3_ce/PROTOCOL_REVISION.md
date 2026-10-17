# SR-F3-CE 2026-10-16 协议修订与固定消融

本次修订的授权是用户明确要求执行 `CLAUDE_SR_F3_CE_PROTOCOL_FIX_FOR_CODEX_zh.md`，文件 SHA-256 为 `83f3688707b09d69d84fb4f7ce3a9a7f2706b26c978176d97a6fb9637189c220`。这是原 v1 停止之后的新授权。原目录的 STOP、原文、失败日志和已交付证据包保持原样；该文件中的机制判断不作为已经验证的科学结论。

## 精确范围

`prompts.CONTRACT` 默认仍是 v1；新入口通过显式 `protocol_revision` 选择 v2 或 c2。v2 使用用户文件中的完整英文代码块，包括其对事实充分性和可选 work 句的实际改写。它不仅修改了原文声称的最后两句。c2 保留 v1 提示文本，仅对 structured 使用 `{\n  "evidence": [` 预填。answer_only 的预填仍为 `{`，natural_work 无预填，外部任务不受影响。

两套候选的所有教师目标均使用 `json.dumps(obj, indent=2, ensure_ascii=False)`；GOLD 没有 work，PREFIX/EXEC 完整文本和 token 序列相同，只有损失掩码不同。所有目标重新计算完整文本字符跨度和实际 native tokenizer 的完整序列偏移。C2 预填属于上下文；即使空证据的 ` [],` 发生 token 合并，也不能给跨越上下文边界的 token 加损失。新格式若无法严格对齐就报错，不能退回其他格式。

候选工具先认证原计划的固定哈希，嵌入原始 48 个文件和用户修订文件，实际执行参考生成器两次并比较。4,096 题的题目、世界、答案、证书、根划分、运行矩阵和日程不变；只允许 prompt、teacher 文本与 spans 改动。`contract.py` 不改。模型输入加载仅打开输入清单、原始输入清单、配置、授权文件和图片清单，不能借助 gold sidecar 构造输入。

## 固定选择

先运行独立 PROTOCOL_ABLATION 根：三父模型 × 64 ENGINE 题 × K4 × P1/P2/P3，共 2,304 条。P1 为 v1+`{`，P2 为 v2+`{`，P3 为 v1+C2 预填。此阶段不创建 optimizer，不做训练、教师 forward 或 MONITOR。它保持原 ENGINE 的采样参数及 group/draw seed；变体和缓存身份分开，不能相互复用输出。

每个父模型每变体的分母严格为 256。P1 的 b0 和 b2 必须均低于 231，否则标记 `STACK_REPRODUCTION_FAILURE`。在复现条件成立后，P2 三块均至少 231 则选 v2；否则只有 P3 三块均至少 231 才选 c2；两者不满足则 `STOP_PROTOCOL_FAILURE`。选择只使用原评分器的 `value_parseable`。正确率、证书率、呈现率、完成率、首键、work 和 token 数只作描述。

`ABLATION_MANIFEST.json` 在输出前冻结源代码、原 STOP 和身份、候选计划、实际 native 上下文、父模型、全部输入与 2,304 个采样位置。所有九个真实 Slurm 作业均有 `COMPLETED/0:0`、完整原文、allocation 和终态证据后，才能创建不可覆盖的 `PROTOCOL_SELECTION.json`。该回执的只读验证重新计算完整计数和决定。

## 选中后的独立运行

新选中根必须与旧根、消融根互不包含。其 `RUN_IDENTITY.json` 显式绑定授权文件、选择回执、新源代码及与消融所测完全相同的候选计划。旧 v1 运维入口继续锁定原科学 commit；新版只有 `verify_run_source_revision` 通过才能使用新源代码。

选中后完整重做 ENGINE，包括三诊断视图、三父块各 512 条 step0 MONITOR，以及新生成的 512 条恢复缓存和 4 次逻辑更新的 2+2 恢复测试。不能用消融输出替代新的 ENGINE，也不能把 v1 的缓存当作新缓存。仍保留四臂、三个父模型、lr=1e-5、64 步、G=8、gamma=0.1、固定未来种子和 90% 门槛。微批可行性与协议门禁分别记录。

选择回执先于新 ENGINE；最终 `EXECUTION_FREEZE.json` 仍要等本根所有 CPU/ENGINE 门禁和真实 Slurm 终态通过后才生成，并嵌入选择全文、v1→候选差异及源身份。正式 12 条路径只能在此冻结之后开始。执行冻结还认证完整旧证据包及其核验回执；交付独立认证 2,304 条消融镜像，旧、新记录分母不能混合。

所有服务器作业名均以 `-LH` 结尾。继续使用登记的 teacher QoS、rose account、cluster02 和最多五张实际/待分配/未知占用物理 GPU；必须同时遵守实际提交数限制。网络超时表示状态未知，不能据此重投。

## 证据中的事实界限

对旧 2,880 条原文独立复算重现了主结构化首键表，但并未确认用户文件的全部说明：549 条可解析回答均换行，其中首层恰好两空格为 530 条；诊断文本中含 `"series"` 字面量的是 true_table，不是 selected_operands；在相同 16 个诊断 qid 上，原基线 b0/b1/b2 的 series 首键为 18/0/17（各分母 64），不是主 64 题总体百分比。work 率须按注册 parseable 分母重新计算。完整核验见本次 artifact 的 `evidence_review/REVIEW_zh.md` 和机器回执。

这些差异不改变选择规则。提示、预填与首键的关系仍是待验证机制，不能仅凭格式变化或一次技术门禁通过声称能力或科学结果改善。报告保留不利结果和全部技术事件，并从已有原文描述 work 随训练变化，不修改奖励。

## 固定消融的调度接续

科学代码冻结在 `652e52d530520a276b6420ff0f4ee479dcde6db1`。辅助入口 `scripts/sr_f3_ce/dispatch_protocol_ablation.py` 另行提交，部署到 `ROOT/operations/ablation_driver/<辅助 commit>/`，以独立 SOURCE_MANIFEST 和 OPERATION_IDENTITY 绑定；不得覆盖已经产生输出的 `ROOT/code` 或原 RUN_IDENTITY、ABLATION_MANIFEST。

接续只处理已经冻结的九个任务，读取初始人工提交账本并验证实际作业身份。每个新任务先 hold，核验真实 Slurm allocation 和保存的 batch script，确认共享 teacher QoS 容量后才 release。单实例锁与原入口共用 `ablation/submissions/SUBMIT.lock`。CPU 调度作业自身也占一个本用户 teacher 提交名额；GPU 实际分配、排队和未知预留另行计入五张卡的限制。

提交或释放结果不明、真实失败、未登记的部分原文都会停止补交；队列已退出但 accounting 尚未可读时，仅保留额度并等待查询，不重新生成。只有九个任务均有真实 `COMPLETED/0:0`、完整完成回执与原文，才调用冻结代码生成选择回执。调度器到选型结束，不启动 SCIENCE；选定的新运行仍须完整重做 ENGINE、MONITOR 和恢复门禁。


## 2026-10-16 实测消融与选型

九项固定消融均已真实 `COMPLETED/0:0`；CPU 接续器 207281 亦正常结束。完整 2,304 条原文、576 个组及 REQUEST/pending 原件经服务器验证与本地逐成员哈希验证，独立冻结评分器复算无差异。

| 变体 | b0 / 256 | b1 / 256 | b2 / 256 |
|---|---:|---:|---:|
| P1 原 v1 | 157 | 236 | 156 |
| P2 v2 | 245 | 250 | 247 |
| P3 C2 | 231 | 253 | 242 |

按预先登记的 value_parseable ≥231 和 P2 优先规则，采用 v2，保留 `{` 预填。P1 b0/b2 复现失败；P1 全部 768 条 token、文本、随机种子及 sampler log probability 与旧原文精确一致。正确率、证书率、呈现率、完成率、work 和长度仅作描述，未参与选择；P3 首 evidence 由预填提供，不是自主生成结构的独立证明。

不可变 `PROTOCOL_SELECTION.json` SHA256 为 `6bd7ddca72b5a7a941c59b85eaf7b18ec3de10e0d2a296b96e26af35124130d4`；完整消融归档 SHA256 为 `806ee061514a71478a4f0c3cf802ec1a435fcf0eaea04e3399d0d3dfef237388`，53,326,462 bytes。九项单卡 Slurm 实际累计 6,692 GPU 秒，包含加载与作业开销，不能称作纯 kernel 时间或文档估计的约20分钟。

原 v1 STOP、原证据和不利结果保持不变。该选择只准许在独立 `selected_v2` 根重做初始化与工程门禁；不是 SCIENCE 完成或放行。科学源码仍固定 `652e52d530520a276b6420ff0f4ee479dcde6db1`。本次单次 CPU 初始化辅助源码及必要精确复制清单保存在 `artifacts/sr_f3_ce_protocol_fix_20261016/validation/selected_run_preparation/`、`validation/selected_data_preparation/` 和 `deployment/selected_cpu_launcher/`；其中模板及本地验证是准备证据，不是服务器成功回执。整个 CPU 初始化真实成功后才允许新 P0；新 P0、三个父块 ENGINE/MONITOR 与恢复 ENGINE 全部成功后，原科学调度器才能封存并推进固定12条训练路径。


## 选中根初始化与受限容量接续

2026-10-16 UTC 的实际验收中，selected_v2 初始化作业 207359 为 `COMPLETED/0:0`，耗时 38:13；随后 P0 作业 207422 为 `COMPLETED/0:0`，耗时 37 秒。新 native 覆盖 816 个目标 mask、48 张图像与144条协议路由，未加载模型权重、初始化 CUDA 或生成新回答。新 RUN_IDENTITY 的 SHA256 为 `81dea1b8746f72509cb99a39b1498ddca8917227dbd98fbf1cdb8b84ff85ef05`。

历史重新导出45,452个文件、64,944条记录，108 COMPLETE / 1 MISSING 保持原样；父模型 adapter 字节、历史 RECORD_INDEX 与原图身份均核对一致。2,756个外部输入在逐项身份等价后复用原盲审：85 resolved / 2,671 unresolved，明确 `new_human_review=false`。这不是新的人工审查，也不补齐原有缺失项。

共享 teacher QoS 同时受五张 GPU 和本用户五个 submitted 作业限制。等待依赖的已提交 GPU 作业仍占额度，CPU 接续器也占一个提交名额。新辅助 dispatcher 在真实 P0 完成后，通过原冻结 `Controller.submit("engine")` 提交原四个 ENGINE 任务，每次都重新认证状态和容量；未提交的固定任务可在空位释放后继续提交。没有容量就等待，已认领、未知、held 或失败状态不能触发重复提交；失败 ENGINE 不能使用仅对训练/评价 TIMEOUT 开放的恢复规则。只有四个 ENGINE 的完整证据与真实终态都通过后，才允许原有 freeze 和 SCIENCE 路径。

该调整只部署到新的 `operations/<辅助commit>/` 并单独绑定源码身份。已经执行的 `code/`、科学提交652、原操作目录、RUN_IDENTITY 和 native extension 保持其已登记字节。单个 CPU 接续器仍有72小时租期；若退出为 `HANDOFF_REQUIRED`，须按既有顺序登记下一次CPU租期，不能将它视为无限期自续。


## 2026-10-17 已授权的接续器与账本修复

用户明确授权：“把接续器和账本阻塞修好，验证通过后继续剩下的实验，保持冻结的实验设置不变，保留历史记录，记得用好可用的GPU额度。”修复仅注册于新的 `operations/<commit>/`，不覆盖科学源码652、原运维6170、native观测扩展、EXECUTION_FREEZE、旧v1 STOP或已生成科学原文。

独立 `reconcile_submissions.py` 由新 OPERATION_SOURCES 和 OPERATION_IDENTITY 绑定，并要求保留的 REPAIR_AUTHORIZATION 绑定原运行/执行冻结和失败CPU前任的原始记录。CPU接续仍使用 `operations/dispatcher/attempts/` 的下一个序号；跨运维版本只允许这个已明确授权、已核验不在队列且真实 `FAILED/1:0` 的前任。普通失败、未知提交及不明会计状态不会获得通用重试权限。

已完成任务必须以真实 `COMPLETED/0:0`、原 allocation/脚本/进程、完整任务产物和观测回执经冻结 verifier 验证，才把可变 STATE 索引结为 COMPLETE；原 STATE 字节先保存为 `operations/submission_reconciliation/<key>/completion/BEFORE_STATE.raw.json`，另写不可覆盖的 RECONCILIATION。没有新模型调用或科学重采样。

同 key 重提严格限于尚无 job_id、allocation、release、STARTED、worker/观测或任务输出，并有原始精确 `QOSMaxSubmitJobPerUserLimit` 拒绝回执，或新守卫明确未调用 sbatch 的本地容量等待。还要查询实时队列与 accounting，拒绝匹配到的实际分配。原 INTENT、脚本、claim 和首次 SUBMISSION 原件不改；每次尝试在 `operations/submission_reconciliation/<key>/attempts/NNNN/` 记录 BEFORE_STATE.raw、INTENT、紧前队列、MAY_HAVE_SUBMITTED、RESULT 和成功时的 ACCEPTED。发出sbatch前先持久化可能已发出的标记；超时、异常、缺少结果、已获配或释放结果未知均阻塞，不能再引用更早的拒绝回执重投。实际已获配的 TIMEOUT 恢复继续使用原冻结恢复接口与新 attempt key，不复用原分配。

完整源码、任务和检查点验证之后，sbatch及release调用前再次读取实际队列，再用轻量账本核算。候选提交在可变STATE中已占一个预留，只计一次，投影达到5可以提交，超过5必须等待。实际/待分配/未知GPU总预留与本用户teacher提交数分别限制为5；CPU接续器占提交名额。明确未分配的拒绝可以从投影额度排除，原失败记录仍保留；真正未知的提交继续预留。固定任务仍每作业单卡、pool_size=1，按空出的提交及GPU额度补交，不能擅自修改科学矩阵或启用未验证的多卡执行。

交付原科学任务门禁不变。新增运维摘要明确区分提交拒绝、本地未调用等待、实际接受和未知结果；INITIAL日志与原SUBMISSION按显式身份去重，拒绝不算GPU分配失败、零科学效果或新增科学采样。旧失败CPU租期、原拒绝、STATE原字节和所有重试记录随证据保留。
# 2026-10-17 调度效率修订（不改变科学协议）

用户明确授权消除多余哈希与身份检查、修复接续器长期不补交。独立运维修订采用容量优先轮询、
每个 CPU 分配一次完整启动验证和 Controller 实例内轻量身份检查，取消 status/tick 的重复历史扫描。
原科学 submit、任务完成核验、恢复、端点评价和最终交付门禁仍保留；不增加实验臂、种子或采样。
具体边界及 CPU 安全交接见 `OBSERVABILITY_EXECUTION.md` 的同日调度开销修订。
提交额度和 GPU 额度仍分别计算，单任务每作业一张 GPU、pool_size=1；所有新作业名仍以 `-LH` 结尾。


## 2026-10-17 固定任务按额度合并为多GPU作业的授权

用户新增明确授权：“把多个固定实验任务放进同一个多GPU作业里并行跑，把剩余GPU额度用起来；保持冻结实验设置不变，保留历史记录，别重复提交，也别影响其他作业。”该授权允许独立运维入口把额度内尚未认领的同阶段固定任务合并到一个真实 Slurm allocation；每个任务仍独占一张 GPU，使用原科学 worker、原种子、原训练批次、原采样与评价日程。没有数据并行、模型并行、共享 optimizer 或改变科学预算。

新入口为 `operations/<新commit>/pooled_execution.py`，通过该版本的 OPERATION_SOURCES、OPERATION_IDENTITY.task_pool 和 POOL_AUTHORIZATION 显式绑定。冻结的 `orchestration/CONFIG.json` 仍为 pool_size=1；原 code/native/EXECUTION_FREEZE 和所有历史 intent/STATE/失败保持原件。新 execution_entry_schema=2 记载实际 operation 命令、实际成员及实际 GPU 数；历史 schema=1 继续使用原验证逻辑，不能伪装成旧单任务 intent。

CPU 接续器仍占一个提交名额。提交前在共享租锁内认领成员，实际 sbatch 和 release 前再读取新鲜队列，分别核算实际QoS的GPU额度与本用户提交额度。新池选择至多 min(实际GPU余量, 实际每作业限制, 单节点资源容量, ready任务数) 个固定任务；没有提交名额则等待。已有 GPU 作业不中断，不取消其他项目的作业来制造余量。所有新作业名以 `-LH` 结尾。

每个子进程只见一个不同的 CUDA_VISIBLE_DEVICES，保留独立 task、observer、EXIT 和 checkpoint。一个成员失败时，其余已启动成员可以写完自己的真实结果；父作业保留失败终态与全部子进程回执。任何新池的失败、TIMEOUT 或不明分配都不能触发整池自动重放。接续器写 POOL_HANDOFF_REQUIRED 并停止补交，等待有据可查的逐成员处理；这种交接不是 CPU 租期到期，不得仅登记下一CPU序号绕过。

端点评价仍需全部12条训练路径完成，最终 release 仍需全部端点完成。新的训练、评价和 CPU release worker，以及独立 builder，都加载同一已绑定运维适配层；每个任务的原冻结产物与观测门禁不变。成本按一个真实 Slurm job 的实际分配 GPU 数与耗时计算，不把同一 allocation 重复算成两个作业。常规轮询继续只检查小型绑定与队列，不重新扫描模型、checkpoint 和历史科学大文件。

CPU 合成测试只能证明调度与隔离行为；实际多GPU运行必须以真实allocation、父/子STARTED与独立CUDA设备回执确认，不能把运维部署成功称为多卡训练已经启动。

用户随后明确纠正：“qos的限制就是你的限制而不是什么五张卡限制”。实际查询该QoS为共享PRO6000额度8张、本用户最多5个提交；highmem PRO6000节点支持8张卡、64 CPU、760832 MiB，满足每任务4 CPU和90 GiB的资源要求。任务池按实际额度动态选择N，不保留人为五卡上限；未知池按真实N张完整占位。额度不等于即时空闲物理卡，仍须服从Slurm实际分配。


2026-10-17 用户对实际QoS额度的更正优先于本文较早阶段的五卡运维限制。
新 `scheduler_capacity.py` 每次实际 sbatch/release 前读取 `sacctmgr` 的QoS及可见用户关联额度、
`sinfo` 的单节点GPU/CPU/内存容量，再读取最新队列。共享QoS GPU使用量包含其他用户与未知预留；
本用户提交名额独立计算。日常查询短暂缓存，不读取模型或checkpoint。隐藏的祖先关联、billing分钟和
运行数限制仍由Slurm执行，提交拒绝保留并由既有安全接续逻辑处理。
OPERATION_IDENTITY绑定 `LIVE_SLURM_QOS_AND_ASSOCIATION`，不记录人为固定GPU上限。
每个新intent保留当次不可变容量快照；历史完成验证只读该快照，不因后来的额度下降改写历史。
原CONFIG中的历史数值保持原件，运维适配覆盖提交、释放与旧单卡恢复入口的调度判断；核心科学常量不改。
