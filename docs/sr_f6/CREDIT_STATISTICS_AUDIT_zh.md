# SR-F6 信用、目标与统计审计

本次结论仅覆盖源代码、CPU 数学与本地回归。没有加载真实 Qwen、没有运行 GPU 生成/更新、没有核验服务器父 adapter，也没有把本文件当作任何服务器门的 PASS。

## 已核对的科学定义

原包 `docs/04` 的 COND、GLOBAL、SHUF 与参考实现一致：COND 在同题、同 J、同 S/B 可判模式内居中；GLOBAL 只移除 J 分层；SHUF 对完成的 COND 系数进行一次层内均匀置换，保留未改变的置换。UNKNOWN 不等于 FAIL，单例层归零，缺失分量不重新分配权重。两分量抵消属于公式结果，不能通过换权重修饰。

整序列目标包含 native、aux 与一次 KL。current-old 为当前前向的 detach，因此 ratio 的前向值为 1；该目标不提供多 epoch PPO 的解释。条件化信用是已登记代理目标，不是 J 加语义奖励的无偏策略梯度，也没有共享参数下不干扰 J 的保证。

## 已修复的移植风险与本地边界

1. **F5 辅信用上限不能沿用。** 全 J=0 的 G8 中，七条 S=B=0、一条 S=B=1 时，F6 合法信用为七个 −0.0625 与一个 +0.4375。旧 `src/srf5_ref/objective.py` 会拒绝绝对值大于 0.25 的信用。新 `src/sr_f6/objective.py` 使用 F6 G8 的精确上界 7/16，未修改旧 F5。
2. **KL 必须只算一次，诊断分量必须独立。** 旧 F5 的 `native` 返回值已含 KL。新目标分别返回纯 native、aux 和加权 KL 的计算图，同时提供未加权 KL 诊断；总目标仅相加一次。
3. **参考信用函数不能单独证明样本身份。** 新 `build_credit_group` 拒绝非 G8、重复 draw/row、混合 qid/policy/parent/factor/root/family，以及非有限或非二值标签。按 draw 规范化，保留全部八条与原始分母、可判掩码、层内均值、原始及实际信用、置换索引和 row_id 映射。它核验 hash 的格式与组内一致性，真实 policy hash 仍需生产 runtime 从参数内容产生并复核。
4. **剂量不可借调用参数漂移。** 信用适配器固定 λ=0.5、S/B=0.5；J 臂辅助项为零。零辅助回归是目标函数校验，不是允许正式训练改 λ 的入口。
5. **概率数值契约前置。** 目标仅接受同设备、同长度、非空、无 padding 的 FP32 概率向量，要求 sampler/reference detach，拒绝非有限值。TIS 使用 `exp(min(delta, log(2)))` 稳定计算登记的 `min(2, exp(delta))`，没有截断 reference/current 位移。实际 sampler/train 阈值必须在完整 128 条的更新前另行核验；本模块没有 optimizer.step。

## 原包合同问题与落实要求

下列为审计发现的原包缺口；本轮前置澄清及其最终登记以 `src/sr_f6/contract.py` 的 `AMENDMENTS` 和正式冻结证据为准，不将已明确登记的解释重复称为未决问题。

- 原包 `docs/04:51` 列诊断步 1/12/36/72，`docs/06:28` 列 1/4/8/12/36/72。采用后者全集应明确记入实施澄清；诊断不得改变 RNG、梯度累积或优化次数。
- 原包 `planning.root_schedule` 的随机流只含 repeat，三个父会共享随机顺序；`statistics.fixed_parent_interval` 的主方差公式未计父间协方差。若顺序随机性属于推断目标，这一共享来源可能使主标准误失准。示例：三个父均有重复差值 [−1,0,+1]pp 时，忽略父间协方差给出 SE=0.333pp，按共享重复块计算为 0.577pp。该例说明可能性，不是实测方差。正式冻结前需明确顺序随机流；主实施可加入 parent、保持同父同重复六臂共享，从源头匹配主方差假设。不能在效果已揭盲后选择更有利的方差算法。
- `planning.paths()` 只枚举 54 路，未实现块内随机执行顺序。生产 scheduler 必须另有冻结的父×重复块内六臂顺序，不能直接按配置数组的先后执行。
- 原包 `gates.launch_check` 只核对 PASS、配置 hash、来源字符串与证据文件 hash，没有检验阶段对应、实际命令/返回码、code/data 身份和各门事实。它是参考一致性检查，不足以作为生产启动授权；需要严格生产门。五个文件重复引用同一任意文本不能成为五阶段证据。
- 非劣条款给出 LEGACY 1pp、ChartQA 2pp 边界及两项家族错误率要求，但未完全指定比较对象与声明家族，例如 U11 对父、六臂各对父，或其他对比。应在正式揭盲前写明估计对象和具体多重校正；本次没有自行挑选正式保持声明。

## 统计核对与解释限制

四项主对比符号、固定父模型等权估计、父内方差、Satterthwaite 自由度以及 Holm 实现与 `docs/07` 一致。零经验训练方差返回 NA，不能解释成绝对确定。最终汇总仍须保留四项确认家族，将 NA 的 p 以 1 占位参与校正，并调用 α=0.0125 获得 98.75% 主区间；不能仅用函数默认 95% 区间冒充登记区间。

交叉 bootstrap 保持各臂/父共享 root 重采样，避免把回答 draw 当独立训练重复。但只有每父三个重复，尾部近似脆弱；应与主 t 分析并列，不能挑选较显著的方法。三个固定父的结论不能外推为任意父模型总体。C_GLOBAL 与 COND 的区别同时改变层结构及系数幅度；P1 上的 SHUF 不能证明 P0 独有效应的语义对应机制。

## 实际验证记录

本地测试入口：`tests/sr_f6/test_objective.py`、`tests/sr_f6/test_credit.py`。零辅助 loss/gradient 直接与真实仓库 `sr_f12.training.sequence_objective` 对照，覆盖 G8 各成功数、正负优势以及 1/7/1536 token 长度；另外检查合法 0.4375 信用、三个梯度分量相加、FP32/非有限/对齐/梯度分离契约、UNKNOWN、单例层、一次置换、完整分母和身份拒绝。

测试日志与实际运行结果以本轮 validation 回执为准。以上只构成 T03 的 CPU 可证部分。

## 补充：完整单步更新适配

`src/sr_f6/training.py` 实现真实生产调用接口 `update(runtime, optimizer, scheduler, records, group_sizes, reference, credits, diagnostic_row_indices=...)`，没有采样、先导调度或自动 COMMIT。它要求 4 roots×4 views 的 16 个独立 G8、128 条完整序列，以 1/128 累加每条原始 completion 的 token 均值；只在全部检查通过后进行一次 AdamW 和一次 scheduler.step。

接口现场计算可训练参数 tensor hash，与每条回答的 `policy_hash` 比较；`reference_hash` 也使用 tensor 内容身份，必须对应 `reference_parent`，与父模型文件 inventory hash 分开保存。它检查冻结 source、run、目标步与源 adapter 步、原 tokens/raw_tokens、实际 sampler 来源、原生 prompt tensor/token/mask/grid、输入图像文件内容及采样配置。`trajectory_hash(row)` 绑定原 token/probability 路径，不重新编码文本。

runtime identity 必须具有 `run_id, parent_id, step, run_root, source_sha256, reference_hash, reference_parent`；登记诊断步另需冻结的 `diagnostic_row_indices`。回答的 `step` 是待执行步，`adapter_identity.step` 是上一提交步；回答另须保存 `prompt_sha256` 和 `token_path_sha256`。信用构建函数接受 flat 或 nested `score`，但拒绝二者冲突。

实际概率门直接调用继承的 mean/TIS 检查，并另核验逐 token 最大差异；均在 optimizer.step 之前。失败保留原始 128 条路径及已完成的实际 forward trace；非有限值以字符串留档，不写成 0。若 optimizer 已尝试或 forward 异常修改了参数，明确标为必须恢复上次 checkpoint。成功返回 `UPDATED_UNCOMMITTED`，不自行推进已提交步号。

登记步的诊断要求 64 条、8 个完整 G8，复用相同 forward 图分别获得 native/aux/weighted-KL 参数向量；总向量由同参数顺序相加，记录范数、点积与 cosine。诊断不写入 `.grad`，正式梯度仍仅回传总目标；保存并实际复核恢复后的 Python/NumPy/CPU/CUDA RNG 身份。

新增 `tests/sr_f6/test_training.py` 的 CPU 小模型 fixture 覆盖完整更新流程。测试没有替换关键概率门，分别构造仅 mean、仅最大尾部、仅 TIS 违规，证明 Adam 前阻断；另外核验 1536 tokens、全六臂、零组保留、等序列权重、全局裁剪一次、单 Adam、实际内容身份、同子集三通道梯度无污染、RNG恢复、backend 异常及更新后异常的恢复标记。三组测试合计 **105 passed in 2.55s**；新更新模块及测试 Ruff 通过。

这些是生产更新代码的 **CPU 行为回归**，不是真实 Qwen 训练或 GPU 门通过。真实 tokenizer/EOS/图像路径、GPU sampler/current/reference 数值、9B 模型参数位移和事务恢复仍需服务器实测；正式结果仍为 NOT_RUN。
