# Task F：SR-ORD-TRAIN 训练目标 / 采样 / checkpoint 代码事实（只读审计，2026-11-12）

路径缩写（全部只读）：
- `EXP` = `/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/sr-ord-train-20261103/sr_ord_train`
- `NAT` = `EXP/execution/historical_readonly/native_source/src`（归档 native 源码）。我核对过：本地 166 个文件的 sha256 与生产 `BUN/provenance/RUN_BINDING.json` `source_files` 中的对应条目 **166/166 一致**，另有 404 个 RUN_BINDING 文件不在归档里；`sr_f6/objective.py` 的 sha `7be5b90a…` 与 `EXP/runtime_contract.py:36` PINS 一致。
- `F1` = `/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/sr-f1-20261009/src`：其中 `mm_core/training.py`、`mm_core/vl_runtime.py`、`mm_dev/runtime.py`、`sr_f12/runtime.py`、`sr_f12/training.py`、`sr_f1/runtime.py`、`sr_f1/json_protocol.py` 的 sha 与 RUN_BINDING 完全一致。
- `BUN` = `EXP/execution/final_delivery_20261111/bundle`；`REV` = `EXP/execution/gpu_review_2026110x`（本地审计副本）。
- 本地 `EXP/adapter/{backend,driver,training,transactions,evaluation}.py` 的 sha 前缀与 RUN_BINDING `implementation_files` 一致（例如 training.py `aac309a9cb1f`）。

---
## 1. Loss 形式（`objective.sequence_loss`）

- 来源：`EXP/adapter/training.py:189` 里是 `import sr_f6.objective as objective`；运行时会按 PINS 校验指纹（`training.py:194-195`）。对应文件是 `NAT/sr_f6/objective.py:34-101`（`EXP/vendor/native_src/sr_f6` 里**没有** objective.py）。
- 常量：`KL_COEFFICIENT=0.02`、`TIS_CAP=2.0`（`objective.py:13-14`）。
- 每个 token 的计算（输入都是 FP32 一维向量，长度 = 该回答的 completion token 数）：
  - `old = current.detach()`；`ratio = exp(current - old)`，**数值上恒为 1**（`objective.py:63-64`）。
  - `delta = old - sampler`；`weights = exp(clamp(delta, max=log 2)).detach()`，即 `min(2, π_train/π_sampler)`，只截上界（`objective.py:65-68`）。
  - `policy(A) = -weights * minimum(A*ratio, A*clamp(ratio,0.8,1.2))`（`objective.py:70-72`）。
  - `kl_token = expm1(ref - current) - (ref - current)`，即 k3 估计（`objective.py:76-77`）。
- 每条序列：`loss_i = mean_t(policy(A_i)) + mean_t(policy(aux=0)) + 0.02*mean_t(kl_token)`（`objective.py:78-81`）。也就是**先在每条序列内部取 token 均值**做长度归一化，不按全局 token 数归一化，也不是 sum。
- 128 条回答的合并方式：逐条执行 `(result["loss"]/128).backward()`，梯度在 `.grad` 中累加（`training.py:229-251`）。合计等于 `(1/128)·Σ_i loss_i`，每条序列等权，与长度无关；**不另外除以组数**（16 组 × 8 = 128 一起等权）。回执字段 `loss_reduction="mean of all 128 original completion token means"`（`training.py:283`）。配方名 `task_reduction="mean_of_128_sequence_token_means"`（`NAT/sr_f3_ce/training.py:49`）。
- 等效梯度（因为 ratio≡1，clamp 在 1.0 处不生效）：`g = (1/128)Σ_i (1/T_i)Σ_t [ -A_i·w_it·∇logπ(y_it) + 0.02·(1-exp(r_it-c_it))·∇c_it ]`。没有 entropy 项。

## 2. Advantage

- 公式：`A_i = (J_i - mean J)/sqrt(Σ(J-mean)²/8)`，用总体 std、不加 epsilon，std=0 时整组为 0.0（`EXP/adapter/training.py:46-48`）。只接受完整的二值 G8，否则抛 ValueError（`training.py:44`）。它与 native `sr_f6.credit.build_credit_group` 的计算一致（`NAT/sr_f6/credit.py:170-172`），并有测试对照（`EXP/adapter_tests/test_training.py:25-30`）。
- J 的定义：J、V、finished 都必须是 0/1，且 J=1 要求 V=1 且 finished=1（`training.py:162-165`）。`finished = finish_reason=='balanced'`（`EXP/gpu_production_v2/backend.py:481`），所以被截断（length）或先遇到 EOS 的回答一律 J=0。
- aux：`credit_groups` 写入 `"aux":[0.]*8`（`training.py:169`）；`sequence_loss(..., 0.)`（`training.py:248`）；回执里 `aux_nonzero_rows=0, prefix_rows=0`（`training.py:277`）；契约要求 `aux_enabled is False`（`EXP/runtime_contract.py:137`）。
- 回执复核（我在 `BUN/training/path_receipts/*.json` 上的计算）：4608 组全部满足上述公式（不一致 0 组），aux 非零 0 组，max|A| = √7 = 2.6458（与 `objective.py:61` 的界一致）。**零方差组 3038/4608 = 65.9%**，这些组的 policy-gradient 为 0，但仍计入 1/128 分母，并且仍贡献 KL 梯度。

## 3. Clipping / TIS / KL / reference

- PPO 式 clip：有 ε=0.2 的写法（`objective.py:72`；配方 `clip_epsilon=0.2` 在 `NAT/sr_f3_ce/training.py:53`、`BUN/provenance/RUN_BINDING.json` recipe）。但 `old=current.detach()`，ratio 数值恒为 1，所以 **clip 永远不生效**。这是单 epoch、严格 on-policy（`policy_version_sampled = step-1`，`runtime_contract.py:175`）。
- TIS：权重 = `min(2, exp(train_logprob - sampler_logprob))`，按 token 计算、detach，只乘在 policy 项上，不乘 KL（`objective.py:65-74`）。`sampler_logprobs` 取自生成时的原始 logits：`log_softmax(raw_logits)[token]`，并检查 raw 与 processed scores 相同，即温度和 top-p 没有变换 logits（`NAT/sr_f6/runtime.py:700-708`，`training.py:155-156`）。
  - `tis_truncated_fraction` = 全批 token 合并后 `delta>log 2` 的比例（`NAT/sr_f5/training.py:175`）。`sampler_train_mean_absolute_difference` = 全部 token 上 |train-sampler| 的均值（`sr_f5/training.py:137-139,170`）。
  - 门限：mean ≤ 0.01、max ≤ 5、截断比例 ≤ 0.01（`runtime_contract.py:119-120`）。实测 288 次更新中截断比例**全部为 0**，mean abs diff 最大 5.0e-7，max abs diff 最大 2.2e-3（`BUN/training/training_updates_288.csv`）。
- KL：**在 loss 里**（不只是记录），系数 0.02（`objective.py:80-81`），梯度经过 current，k3 形式。CSV 的 `kl_unweighted` 是 128 条序列各自 token 均值 k3 的等权平均，不乘 0.02（`training.py:250-251,261`）；seal 侧用同一公式重算（`EXP/gpu_production_v2/seal_path.py:123-129`）。step 1 时 policy 与 reference 相同，KL 及其梯度都为 0。
- Reference：路径启动时父 LoRA 张量的 CPU FP32 clone（`gpu_production_v2/backend.py:392`；`adapter/backend.py:387`），**整条路径固定不动**。b0 的 reference_sha256 始终是 `0d7979e1…`（`BUN/training/path_receipts/b0-11031-R.json`：reference_tensor_sha256 = parent_tensor_sha256）。前向时用 `reference_parameters` 把 LoRA 原地临时换成父权重，在 no_grad 下计算（`F1/mm_dev/runtime.py:126-137`，`F1/sr_f12/runtime.py:248-250`，`training.py:236-237`）。

## 4. Token mask

- 计入 loss 的 token：只有生成的 completion。prompt（含图像 token）和 assistant prefill `{` 都是上下文：prefill 在 processor 层拼进 prompt（`NAT/sr_f3_ce/runtime.py:387-456`），routing 标记为 `prefill_is_context_not_loss_target=True`（`gpu_production_v2/backend.py:240-241`）。训练只允许 prefill=`{`（`backend.py:93-94`，`training.py:128`）。
- 截断点：`n = min(首个 balanced-JSON 边界 token, 首个 EOS 的位置+1, 生成长度)`，`tokens = backend_tokens[:n]`（`NAT/sr_f6/runtime.py:684-700`）。所以 EOS 若被采到会**包含**在内，objective 文档也写明"including a sampled EOS"（`objective.py:5`）。balanced 停止时不含 EOS。停止后的尾部是 pad，被排除，若出现非 pad 就报错（`runtime.py:724-726`）。上限 1536 token（`training.py:149-150`）。EOS id = pad id = 248044（`REV…06/B0R_GPU_BRIDGE_AUDIT.json` loaded.generation_config_expanded）。
- 对齐：`logits_to_keep = total - completion_start + 1`；`logits[row,:len]` 与 `input_ids[completion_start : completion_start+len]` 做 gather（`NAT/sr_f3_ce/runtime.py:539-563`）。microbatch=1，所以训练前向**没有 padding**（`backend.py:370-384`，`training.py:239`）。

## 5. Optimizer / LR / clip / 参数 / 精度

- AdamW：lr=1e-5，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，foreach=False（`NAT/sr_f3_ce/training.py:33,71-78`）。schedule 是常数 LambdaLR（`:60-61,79`），warmup=0。每条路径新建 optimizer，不继承父 Adam（`backend.py:416`；`training.py:212-213`）。
- 梯度裁剪：对全部 LoRA 参数做全局 `clip_grad_norm_(params, 1.0, error_if_nonfinite=True)`（`training.py:258`）。before_clip 由 torch 计算，after_clip 用 float64 重算（`training.py:259-260`）。**288 次更新中 before_clip 最大 0.375，从未超过 1，裁剪实际未触发**（`BUN/training/training_updates_288.csv`）。
- 每次更新只调用 1 次 `optimizer.step()` 和 1 次 `scheduler.step()`（`training.py:267-268`；GPU 版用计数器强制，`gpu_production_v2/update.py:179-180`）。microbatch=1，对 128 条做梯度累积；`forward_calls=256`（128 次 ref no_grad + 128 次 current grad），`backward_calls=128`（`training.py:279,284`）。
- 可训练参数：只有父 J6 LoRA，r=8、α=16、dropout=0、bias=none（`NAT/sr_f3_ce/runtime.py:205`）。范围是 language_model.layers 中的 lora_A/lora_B；vision、merger、lm_head、embed 都被禁止（`:233-256`）。共 **496 个张量**（`update.py:78`；B0R bridge 中 trainable_names=496），target_modules 为 q/k/v/o_proj、gate/up/down_proj、linear_attn 的 in_proj_{a,b,qkv,z} 和 out_proj。
- 精度：从 bf16 cast 成全模型 FP32（`backend.py:390`；loaded.model_cast_from_dtype="bfloat16"）。attention 为 eager。启用 deterministic algorithms，关闭 TF32，`CUBLAS_WORKSPACE_CONFIG=:4096:8`（`F1/mm_dev/runtime.py:474-491`，`gpu_production_v2/run_training.sh:12`）。训练前向处于 `model.eval()`，并禁止 dropout（`training.py:206,227`；`NAT/sr_f3_ce/runtime.py:547`）。

## 6. 采样设置与评测 draw 语义

- 训练 rollout：temperature=1、top_p=1、top_k=0、do_sample=True、max_new_tokens=1536（`NAT/sr_f6/runtime.py:98-100` 覆盖 `NAT/sr_f3_ce/runtime.py:61-74`；`training.py:159-160` 逐行校验；loaded.generation_config_expanded 中 repetition_penalty=1、num_beams=1）。每次调用只生成一个序列。
- 评测：`EndpointRuntime._set_recipe` 使用同一个 `native.generation_recipe()`（`EXP/adapter/evaluation.py:358-360`），同一个生成 kernel，同样 FP32、单 GPU，不创建 optimizer（`evaluation.py:450-457,377-384`）。`evaluation_v2/run_endpoint_pair.sh:80-82` 直接调用未改动的 `adapter.evaluation worker`（`evaluation_v2/README.md:3-6`）。长度上限同为 1536：评测 26624 行中 138 行 finish=length，且恰好都是 1536 token（`BUN/results/eval_raw.jsonl`）。
- draw 的随机性：`seed_all(seed)`（`NAT/sr_f6/runtime.py:669`），再用 `independent_draw_rng` 把 multinomial 替换为每个 draw 各自的 `torch.Generator(device).manual_seed(seed)`（`NAT/sr_f4/runtime.py:15-44`；`evaluation.py:455`）。字段写为 `seed_semantics='independent_per_draw_torch_generator'`（`evaluation.py:476`；eval_raw 26624/26624 行都是此值）。
- 评测 seed 公式：`seed_for(2026110301,'eval',parent,panel,root_id,draw)`（`EXP/build_plan.py:20,128`）。**不含 arm、train_seed、step、view。**
  - transfer8/swap8 中同一单元的 draw 0/1：`input_id` 相同（prompt 与图像相同），seed 不同，因此在同一 endpoint 策略下是条件独立的两次采样（我在 frozen manifest 的 8320 个单元上验证过）。每个 view 2 个 draw（`build_plan.py:107,110`）。
  - **endpoint 之间是 CRN**：同一 parent 的 13 个 endpoint（PARENT + 12 个训练 endpoint）在相同 (panel,root,view,draw) 上 seed 完全相同。同一 root 和 draw 下的不同 view（aligned/conflict、swap 的 4 个 view）也共用 seed。b0 与 b1 的 seed 不同（我在 `EXP/frozen/evaluation_manifest.jsonl` 和 eval_raw 上都验证过）。CRN 只是共享 RNG 流；策略不同时 token 级的耦合**没有保证**（未确定程度）。
- 训练 draw seed：`seed_for(group_seed,'draw',i)`（`EXP/export_runtime_plans.py:58`），`group_seed = seed('G8',p,s,step,slot)`（`build_plan.py:91`），**在 R/G/D 三个 arm 之间共享**，root 也相同（我在 `EXP/frozen/training_groups.jsonl` 的 1536 个槽位上验证过）。

## 7. step-24 checkpoint 内容（依据收据，辅以代码 schema）

- 收据中有：`evidence.checkpoint_24 = {path:"checkpoints/step-024/state.pt", sha256:"af187606…"}`，`updates[23].evidence = {commit: step-024/COMMIT.json, native_update: step-024/UPDATE.json, raw, token_trace}`，`evaluation_policy_sha256["24"]="bf542369…"`，`seal_status=PATH_NATIVE_REPLAY_AND_RECEIPT_VERIFIED`（`BUN/training/path_receipts/b0-11031-R.json`）。b0-11031-D 的结构相同，但没有 `gpu_execution_bridge`（`BUN/training/path_receipts/b0-11031-D.json`）。`BUN/checkpoint_summary.json` 写明 `large_tensors_included=false`，scope 为 "original tensor files remain at bound server paths"，即 state.pt **不在 bundle 中**。
- 收据级的状态证据：bridge 审计的每个 checkpoint 都列出 `optimizer_sha256, scheduler_sha256, rng_sha256, state_sha256, policy_sha256`（`REV…06/B0R_GPU_BRIDGE_AUDIT.json` checkpoints[]，覆盖 step 5-24）。说明 COMMIT 对 optimizer、scheduler、RNG 各自做了 hash 绑定。
- 代码 schema：`state.pt = {schema, identity, step, policy(仅 496 个 LoRA 的 FP32 CPU 张量), reference_sha256(只有 hash，不含 reference 张量), optimizer=AdamW.state_dict(), scheduler=LambdaLR.state_dict(), rng={python,numpy,cpu,cuda[1 或 2 卡]}, sampler_state}`（`EXP/adapter/transactions.py:374-382`；`F1/mm_core/training.py:142-171`）。COMMIT 的字段见 `transactions.py:385-394`。UPDATE.json 就是 update 回执，含 128 行 `probability_vectors`（tokens/sampler/train/reference logprobs）、groups（J/native/aux）、numerics、totals、`parameter_update_norm`（`training.py:275-284`），GPU 版还附带 `gpu_pre_adam_check/gpu_update_accounting/gpu_execution_provenance`（`update.py:181,206`；`driver.py:200-204`）。
- Adam 状态：**有保存**。代码强制每个参数都有 `step==committed step` 的 Adam 状态（`transactions.py:372-373`），seal 时也复核过（`gpu_production_v2/seal_path.py:308-312`）。按 torch AdamW 标准，state 含 `exp_avg/exp_avg_sq`，但这一点是**推断**：收据里只有 hash，键名未确定（不打开 state.pt 无法确认）。RNG：**有保存**；v2 checkpoint 保存 2 卡 CUDA RNG，原单卡 checkpoint 只有 1 卡（`seal_path.py:184-186,321-331`）。评测加载 endpoint 时只恢复 policy，"Optimizer/RNG payloads are never restored"（`evaluation.py:425`）。

## 8. Dry-run 梯度诊断方案（只是计划，未执行）

目标：在 step s 的 128 条原始 rollout 上**不做更新**，计算 (a) J-advantage 梯度和 (b) R_source 经同样 G8 标准化后的梯度，并比较两者的 cosine 和幅度。

需要复用的函数：
1. 加载：`gpu_production_v2.backend.NativeBackend.__init__`（`backend.py:268-421`），包括 audited backend、FP32 cast、`configure_parent_training`（`NAT/sr_f3_ce/runtime.py:193-273`）、获取父 reference、`fresh_optimizer`（`NAT/sr_f3_ce/training.py:64-81`）。
2. 恢复 policy s-1，以及 optimizer/scheduler/RNG：原单卡 checkpoint 用 `PathTransactions.restore_latest`（`adapter/transactions.py:404-445`）；v1/v2 lineage 用 `gpu_production_v2.state.restore_latest_gpu`（`state.py:169+`）。rollout 来自 s-1，必须配对使用 s-1 的 policy 和**父** reference。
3. 数据：step s 的原始 128 行（raw-step 与 token-trace 证据都在服务器上），先经过 `validate_records`（`training.py:88-170`）和 `_verify_prepared`（`:173-183`）。
4. Advantage：(a) 用 `native_j_advantages`（`training.py:38-48`）。(b) 该函数只接受二值输入（`:44`），需要**按完全相同的公式另写**：总体 std 除以 8、无 epsilon、std=0 时为 0；|A| 不超过 √7+1e-7（`objective.py:61`，n=8 时数学上 ≤√7 也成立）。
5. Loss：直接调用 pinned 的 `objective.sequence_loss(current, sampler, ref, A, 0.)`（`objective.py:34`），可以分别对 `native` 和 `kl` 两张图 backward（`objective.py:3-4,87-92`）。
6. 循环：照抄 `training.py:226-251`（逐行 ref no_grad、current grad、`(loss/128).backward()`），以及 `:258` 的裁剪。
7. 捕获梯度且不调用 Adam：复用 `diagnostics/gpu_capacity_full_batch_v2.py:131-236` 中的 `probe_update`。它在 optimizer pre-hook 里抓取**已裁剪**的梯度，然后抛 `StopBeforeAdam`（`:35,176-194`），再由 finally 清空梯度、恢复 RNG。长序列要配合 `GPUSavedTensors`（`diagnostics/gpu_capacity_v2.py:102-235`）。先用 (a) 跑一遍，确认与生产 UPDATE.json 的 numerics 以及 b0-G step 9 已接受的 496 个梯度 `torch.equal` 一致（`update.py:41-55`），再换成 (b)。

朴素重写容易与正式训练产生的偏差：
- 长度归一化：正式训练是每条序列取 token 均值、再对 128 条等权；若改成全局 token 均值或求和，长回答的权重就变了。
- TIS：w=min(2,exp(train-sampler)) 必须使用**原始 sampler logprob**。实测 w≈1、截断 0，影响小，但重新前向得到的 sampler 或省略 w 都会失去 bit-exact。
- ratio：old 必须是 `current.detach()`。若把 sampler 当 old，ratio≠1，clip 和缩放就可能生效。
- KL：在 loss 中、系数 0.02、不依赖 advantage，对 65.9% 的零方差组也有梯度，在 step 1 为 0。(a)/(b) 共享同一个 KL 梯度，比较 PG 方向时应把 native 和 kl 拆开。
- 零方差组：仍计入 1/128，但 PG 为 0。R_source 是连续值时零方差组会少很多，有效样本数不同，幅度不可直接比较。
- 全局裁剪（max_norm=1，作用于 496 个张量整体）：只改尺度，cosine 不变，但幅度被封顶。正式训练从未触发；(b) 可能触发，比较幅度时应取裁剪前的梯度（现有 probe 抓的是裁剪后梯度）。
- 精度和内核：必须 FP32、eager、deterministic、关 TF32。bf16、TF32、SDPA/flash 都会改变 logprob 和梯度。
- LoRA 范围：只有 496 个 LoRA A/B 张量；cosine 必须在同一参数集合上计算，不能包含 base 或 vision。
- 梯度不等于更新：真实 Δθ 经过 Adam 预条件（需要 s-1 的 exp_avg/exp_avg_sq 和 bias correction）；step 1 时 Δθ≈lr·sign(g)。若要与 `parameter_update_norm` 比较，必须恢复 Adam 状态。
- token 路径：必须用原始 token id，不能重新 tokenize；prefill `{` 只作上下文；EOS 若被采到要计入；`logits_to_keep` 要正确对齐。
- 累加顺序：microbatch=1，按记录顺序（组序、draw 序）逐条 backward，浮点求和顺序影响 bit 级结果。
- 显存：1536-token 的回答在单卡上会 OOM；生产用 cuda:1 做 48 GiB 无损 snapshot（`gpu_production_v2/README.md:3-18`）。原始长行在单卡上的梯度等价性是 **UNVERIFIED**。
- 副作用：`update()` 会写 `NUMERICAL_TOKEN_AUDIT.jsonl`（`training.py:264`）并调用 `optimizer.step`，没有挂 hook 时绝不能直接调用。`check_numerics` 的 grad>1000 哨兵（`runtime_contract.py:122`）可能在 (b) 时拦截，这属于门禁，与梯度本身无关。

## 9. gpu_production 与 gpu_production_v2 的差异，以及各路径实际用了哪一版

- 目标相关部分：**没有差异**。三个执行版本都调用同一个未修改的 `adapter.training.update`：`adapter/driver.py:184`、`gpu_production/update.py:171`、`gpu_production_v2/update.py:178`。`backend.py` 在 v1 与 v2 之间只差两行 import（diff 结果）。与原 `adapter/backend.py` 相比只多了 production binding 校验和"2 卡"要求。
- 非目标差异：saved-tensor 快照容量 16 GiB 与 48 GiB（`diagnostics/gpu_saved_tensors.py:21` 对 `gpu_capacity_v2.py:21`），快照是 `copy=True` 且不改 dtype 的无损拷贝（`:151,208`）。首次恢复的 gate 对象不同：v1 是 b0-R step 5，v2 是 b0-G step 9（`update.py:58-81`）。MODE 字符串不同（两者的 `binding.py:16`）。
- 各路径各 step 的实际版本（依据 `REV/*_GPU_BRIDGE_AUDIT.json` 的 checkpoints[].gpu_execution_provenance.mode，以及 preupdate_passes 中的 job 目录）：
  - b0-11031-D：24 步全部为原单卡 `adapter/`，没有 bridge（`BUN/checkpoint_summary.json` training_path_seals 中无 gpu_bridge_sha256；`gpu_production_v2/README.md:73-74`）。
  - b0-11031-R：1-4 原单卡（job 247321），5-11 v1（16 GiB），12-24 v2（48 GiB）。
  - b0-11031-G：1-8 原单卡（247322），9-24 v2。
  - b1-11031-R：1-2 原单卡（253011），3-24 v1。
  - b1-11031-G：1-2 原单卡，3-24 v2。
  - b1-11031-D 以及全部 6 条 11032 路径：v2，从 step 0 起就是两卡新父模型。
- 科学身份：所有路径的 identity.code_commit 都是 `822f0f8a…`，RUN_BINDING sha 为 `814a197e…`（`BUN/checkpoint_summary.json` endpoint_lock.original_run_binding）。执行版本单独绑定在 GPU_EXECUTION_BINDING 中（`gpu_production_v2/state.py:68-98`）。
- 未确定：单卡与两卡之间的 bit 级梯度等价，只在 b0-R step 5 和 b0-G step 9 这两个恢复点做过 `torch.equal` 对比；长行的单卡基线是 UNVERIFIED（`gpu_production_v2/README.md:11-18`）。
