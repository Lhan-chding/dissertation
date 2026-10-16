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
