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
