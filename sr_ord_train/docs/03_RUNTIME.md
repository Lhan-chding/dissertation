# 03｜从历史 F6 接入 R/G/D：训练运行合同

**交付状态：本地 CPU 评分夹具和回执校验器；不是已在集群运行的训练器。** 用包内的实验清单决定数据、臂、预算、终点。下文给 Codex 列出远端需要实现并验证的环节；不把 11 月 3 日的只读探查烟测当成本轮训练许可或数值通过记录。

## 1. 已定位的源码、模型和数据通路

历史包 `SR_ORDER_PROBE_COMPLETE_20261103.zip` 内 `SOURCE_DIFF/sr_order_probe/sources/historical_measurement.py` 与 `NATIVE_SOURCE_20261103.tar.gz:src/sr_f6/measurement.py` SHA256 都是 `90c26410af3f5526bce5b697138c72e0bf2d0cb58f3557d67fe42cf35db29a04`。历史 `RUNTIME_CONFIG.json` 指向 `/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_b7e7b06`；`historical_source_binding` 为 `/projects/_ssd/varunssd/louis-ssvc/sr_f6_semantic_factorial_20261027/SOURCE_BINDING.json`。源 tar SHA256：`1c5901061d996987b9683ddfb6919abc41724004a5f43f70693200f574b1abd3`。这些是只读历史定位，不是新任务写入位置。

- **模型**：Qwen/Qwen3.5-9B，snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。b0=SRF2_J6_s71001；b1=s71002；历史父 adapter 在 `sr_f3_ce_protocol_fix_20261016/selected_v2/parents/<id>/adapter` 中。`SOURCE_BINDING.json` 给两份文件的指纹，`runtime_contract.PARENT_FILES` 已逐项列出。加载后另算父 adapter **张量**哈希；它与 safetensors **文件**哈希是不同对象，不能混用。只继承父 LoRA；每路径新建 AdamW，不继承父 optimizer；reference 永远是该路径启动时对应父模型，不随策略移动。
- **11/03 的实际推理**：`RUNTIME_BINDING.json`：eager、模型参数和浮点缓冲、浮点输入、概率均为 FP32，历史权重源从 BF16 转为 FP32；实测 torch `2.13.0+cu130`、transformers `5.14.1`。自然请求仅图像＋公开题面/输出要求，普通 assistant prefill 为 `{`。原始行的后端标签可以是 `protocol=structured, protocol_revision=v2`；F6 逻辑 wire 名为 `structured_v4`。两层标签原样保留，不能用字符串相等代替请求/输入/输出比对。
- **模型回答**：一个 JSON 对象，必需顶层 `evidence,selected,bindings,aggregate,answer`，`work` 可选、如出现须为字符串。`evidence` 是 `series,category,value` 记录，`selected` 是类别名列表，`bindings` 是 `category,operand` 列表，`aggregate` 为 `sum`/`mean`。与模板保持共同顺序；不把字段顺序或报告顺序充当额外奖励。模板、图像、用户消息、assistant prefill、解码 token/截断和停止原因要分开保存。
- **评分**：在独立 scorer 进程对原始完整 `full_text` 调用 `sr_f6.measurement.score_response(text, world, query, finished=..., finish_reason=...)`。`J = V_F6 × J_F5_projection`：先验证整个 F6 结构，原字节投影去 `bindings,aggregate`；F5 的有效 `selected` 接口再无损投影到已有答案/证据判定。F5 的 `J_M3=V_F5×finish×value_correct×certificate`；证书要真实且充分。`S/B` 的**语义正误**不乘入 J；结构是否合法则可能影响 V。不得用 `sr_f4.task_contract.score` 直接评分五字段，不得把 `V_probe`、`S_hit`、外加顺序服从率 U 等带入更新。无效或截断保留原分母。冻结原生评分不对正文补 `}`，不接受重复键、不作答案修补。
- **题目可用域**：F6 先调用 `srf5_ref.oracle.evaluate_a`，与 `srf6_ref.core.oracle_a` 的答案及选集一致才继续；F5 只接受 6 或 8 类、域 `[10,90]`、Alpha/Beta 整数、白名单 `TOPK/INTERVAL`、`sum/mean`。本轮正式训练用 8 类。TOPK 同值按图上位置。新报告顺序是无关干扰，不是金标准的变更。10 类不得求本轮 native J，也不得把异常换成 J=0，单列任务 A/S。若另做“真正改变合法破平局规则”的评估，用明确的新 oracle 单列，不把新字段偷偷传给旧评分器。

本地结果：同一组内数值完全正确但 `selected/bindings` 错的 8 类夹具，F6 返回 `A=V=J=1, S=B=0`；缺关键排序事实、错误答案、无效结构或不完整输出均 J=0；翻转证据列表不改 J；10 类及未知 tie rule 字段拒绝。`EXECUTION/LOCAL_PREPARATION/PREFLIGHT_CPU_v2/SCORER_FIXTURES.json` 和 `EXECUTION/SMOKE_RECEIPT.json` 是**旧探查**的独立证据；旧 smoke 为 216 请求、0 更新，新计划为 192 请求、0 更新。不得移用回执。

## 2. 集群移植范围与参数冻结

先只读旧根的 `CONFIG.json`、`SOURCE_BINDING.json`、`PHASE_RUNS/PILOT_V3/PILOT_FROZEN/PILOT_SPEC_FROZEN.json`，核对对应历史已提交 J-only 路径的实际 `update` 与 `commit` 记录；比较代码与配置，导出到新根 `NATIVE_BINDING.json`、`TRAIN_RECIPE.json`。11/03 包中的 `SOURCE_DIFF/sr_order_probe/sources/historical_sr_f6.json` 是可查的设计配置，并非代替实际执行凭据。**读不到、指纹不同、超参或版本冲突时停在 BLOCKED，并精确报缺少哪个引用；不改用“差不多”的默认值。** 应冻结模型目录、adapter 配置/文件/张量、processor/tokenizer/已处理图像摘要、chat template、eager/backend、torch/cuda/transformers、RNG 随机算法/种子映射、训练源及所有新 wrapper 指纹。

| 项 | 绑定的历史 F6 配方与新实验限制 |
|---|---|
| 模型与算术 | 参数、参考、浮点 buffer、图像浮点输入、logprob 和损失为 FP32。核查实际 forward 的 dtype；关闭混精与未登记的近似内核，核查 TF32/确定性等实际开关按真实绑定锁定。禁止内存不足时自动改 BF16、换模型或删回答。 |
| 学习与采样 | AdamW lr `1e-5`, betas `(0.9,0.999)`, eps `1e-8`, weight decay `0`, constant schedule；grad clip norm `1`；温度 1、top_p 1、top_k 0、max_new_tokens 1536，每条回答独立 draw RNG；同一 model/version 每批只更新一次。microbatch=1 个完整序列。若实测与历史配置不一致，不静默选其一。 |
| 正式训练 | b0,b1 × 11031,11032 × R,G,D，共12路径；每路径24次更新，每步16组×8=128条，每路径3,072条；总288次更新、36,864条正式训练回答。保存0、12、24；12诊断、24主终点；另留每个已提交步的恢复信息。评测预算只取上层清单，不另加试跑、覆盖或坏回答重抽。 |
| 优势/目标 | 取 `sr_f6.credit` 的 J-only 模式（历史 U00/U10 的数值定义）。`A_i=(J_i−mean J)/popstd J`；若组常数返回8个0，无 epsilon 臆加。保留128序列，aux优势恒0、无选择/顺序局部项。`sr_f6.objective.sequence_loss`：一次全生成序列 KL，系数0.02；TIS cap=2，clip epsilon=0.2；全序列原 token 均值再对128条等权。前置证据文本仅用于指定前缀评测，不作为训练生成 token。 |
| 架构分界 | `src/sr_f6/{runtime,rollouts,credit,objective,training,transactions,pilot}.py` 提供真实内核、逐token轨迹和事务参考。历史 `_validate_records` 明确要求4个root各含4个variant；历史事务/调度仅注册旧phase、旧arm、旧路径。**不可把新R/G/D伪装成U10，不可绕过历史权限**。在新实验命名空间登记与真实清单逐行一致的身份、数据、接入和事务层；数值内核优先原样复用。若须移植训练包层，编写同一批输入 native 与新 wrapper 的 scalar/gradient、前向及资格等价检查，而非将整个旧收集驱动直接套上去。历史 `pilot_v3_kernel_binding` 明示旧 T03 核验范围，不能当作新 prompts、更新或恢复已跑通。 |

## 3. 最小集成顺序

1. **CPU准备**：用 `runtime_contract.py check-native --src vendor/native_src` 重放评分夹具；真实部署上对原始 `.../src` 再加 `--historical-kernel`。将新请求清单和每条日志路径做内容绑定，检查所有 `root/qid/view/draw/seed`、非平局/平局排程和图片 hash；保持私有 world/gold/反事实/arm 标签/资格原因远离模型输入。用真正的 processor 过图，核对边界组刻度、左右标签、重命名；不能用渲染成功替代看见标签。
2. **先合成，再做一次固定 DEV smoke**：CPU 可对合成小模型做一次独立 optimizer 夹具，比较带 aux=0 的目标、梯度及事务恢复；这不是9B训练。GPU smoke 严格来自另册24个预注册G8=192回答，总数是全计划，不是每父、每臂或每seed各192。保存未通过、全J0、全J1回答；对同批生成仅做原 token 前向重放和必要隔离 backward（复制模型/快照、检查前后父权重和 RNG 一致、清除梯度，**不 step、不提交、不继承状态**）。新烟测侧重真实图像路由、答案数值/证书/协议是否能分别计量、轨迹配对和显存；不得凭某臂正确率或混合J比例高低选择更换题组、延长步数或挑seed。多条件差异只以原计划固定槽位报告。
3. **收集真正的 on-policy 回答**：对一条路径的第 s 步，先锁定已提交 policy(s−1) 张量哈希、父 reference 哈希、16条冻结提示和每组8个 draw seeds；各臂用配对 seeds、题目排程，生成来自各自当前策略。不得给 D 复用 R 的后续生成。请求只发给登记模型适配器（公开 prompt＋图），返回 row_id、真实 completion token IDs、采样 logprob、prompt token/张量与图像指纹、停止原因、生成器版本、full_text。无论答案无效与否，落账。
4. **原生评分与逐token对齐**：将 gold 私有侧评分写入 sidecar；固定评分版本，以 raw text、finish 复算 J。原 token、真实生成 EOS（若有）、prefill 边界、有效长度、padding 排除情况、图像 processor 的实际张量必须一致；不得重新 tokenize 文本猜路径。用同一记录分别算 current（有梯度）和 frozen reference（无梯度）logprob，按行写入 `tokens,sampler_logprobs,train_logprobs,reference_logprobs`；保存 token_path/prompt_tensor/source/policy/reference/调用计数哈希。任何错位在更新前中止。
5. **数值门（必须在 optimizer.step 之前调用）**：沿用历史检查：无非有限数；mean absolute sampler/train logprob ≤0.01 nat，max ≤5 nat，TIS 截断比例≤1%。增加两项**新工程哨兵**：当步未加权 KL 代理的全序列等权均值>1，或全局裁剪前范数>1,000，则技术挂起并封存原记录；裁剪后范数≤1+1e−5。这些不是效果或资源停止阈值。10/25 旧 BF16 的 b1 共同峰值说明仅查均值不够，不据此断定 F6 已有同样问题。原式 `expm1(ref-current)−(ref-current)` 只在全序列计一次，区分逐token极值、未加权 KL、加权 KL、sampler/current 最大差、TIS截断与总梯度；它们不是同一 quantity。异常时保存 row_id、top异常 token 位置、三种概率、所有原始向量、模型/适配器/processor/硬件与 RNG 信息。不得额外截断 KL、删样本或偷偷降低学习率。真实 native 驱动在 `training.update` 内部 step；新增门应移植到新 wrapper 的**反传后、Adam 前**，不能先调用会 step 的完整函数再做门控。
6. **原子提交/恢复**：写入 `PREUPDATE_PASS` 和原日志后只进行一次 optimizer.step＋scheduler.step。检查参数与 Adam 状态有限，保留 policy 前后与 reference 哈希、参数位移、裁剪前后范数、step/采样/RNG 游标、路径身份、checkpoint bytes/hash。持锁原子发布 `COMMIT_s` 后才转下步。TIMEOUT 仅恢复同一路径最近已提交 checkpoint；丢失记录不得重造，半写/未知结算先审计。可复用原已完成的真实 rollout 仅限同一 step、policy、父模型、计划、请求、seed及 token 路径一致；不得复用到别臂。在“调用已发生但结果或记账不确定”时保留 attempted/complete/unknown 账并停该路径核证，不把重复调用冒充零成本、不为凑齐效果再采。数值异常保留失败记录；若修复改变算法或题目，则新版本另登记，不覆盖原结果。
7. **调度与计费**：新独立 run root，不写旧快照或改变原门控。历史环境 `.../envs/ssvc-py312/bin/python3.12`、Slurm `account=rose`, `partition=cluster02`, `qos=soujanya-poria-startfund-2026-03`；派发前核对当时实际授权、共享队列及可用 PRO6000，按用户/课题组当前配额，不假设能同时申请12张。独立一GPU worker，可跨节点；为长任务用可续接的批作业。单作业 walltime 属调度约束，没有累计 GPU 小时或运行时长的科学停止条件。逐路径列 attempts、完整/截断/丢失的生成数、completion tokens、模型前向与反向/诊断调用、真实GPU型号/分配时段/显存、GPU小时/有效step、失败/空闲成本。重复、烟测、端点评测单列。不得把历史17.8 GPU小时简单线性当作训练承诺。

## 4. 可调用的 CPU 回执检查器与适配器接口

```sh
python -B runtime_contract.py check-native --src vendor/native_src
python -B runtime_contract.py check-native --src /absolute/path/to/archive/src --historical-kernel
python -B runtime_contract.py check-smoke receipts/smoke.json --slots frozen/receipt_plans/smoke_slots.json --manifest-sha256 <已冻结实验清单SHA> --plan-sha256 <已登记smoke_slots文件SHA>
python -B runtime_contract.py check-path receipts/b0-11031-R/path.json --manifest-sha256 <同一SHA> --plan-sha256 <该路径预登记计划SHA>
```

这是只读的**结构、覆盖与被引用文件字节**检查器，不能替代服务器端模型绑定、原评分重放、scheduler 对账或独立审查；“某个 JSON 写了 PASS”不是成功证据。`receipt` 内的相对文件路径以该 receipt 文件目录为根，禁止越界；保留原生回执，另导出标准化 `order-runtime-v1` 视图。`native_binding` 要包含 bound manifest、原生依赖 SHA、实际 argv/环境和政策/父张量身份。路径使用下列字段：

- 所有范围：`schema,scope,manifest_sha256,snapshot,dtype='float32',reward='native_srf6_J_only',protocol='structured_v4',aux_enabled=false,prefix_training=false,has_gpu_hour_stopping=false`。`evidence.<name> = {"path":"相对路径","sha256":"64位摘要"}`。
- `DEV_SMOKE`：`optimizer_steps=parameter_updates=0, generated_responses=192`。`groups[24]` 对照从上层冻结清单**预先**导出的 `smoke_slots.json`（并核对导出文件的注册digest），每组有 `group_id,parent,qid,root_id,seeds[8],request_ids[8]` 与 `policy_before_sha256,policy_after_sha256`。`evidence` 包括 `raw,scorer_receipt,token_replay,native_binding,processor_checks`。不提权为已完成训练。
- `TRAIN_PATH`：`parent,arm,seed,adapter_config_sha256,adapter_file_sha256,parent_tensor_sha256,reference_tensor_sha256,total_optimizer_steps,total_generated_responses,evaluation_policy_sha256`（字符串键 `12`,`24`），`evidence` 包括 `native_binding,plan,recipe,zero_checkpoint,checkpoint_12,checkpoint_24`。`plan` 是从不可变科学清单按路径导出的24行 JSON 数组 `[{"step":1,"groups":[...16个...]}]`；其文件字节及 SHA 预先登记于上层清单，不允许以实际结果回填 expected。每个预期组带 `group_id,qid,root_id,seeds[8],request_ids[8]`；非平局背景题亦须有可追踪的源root。
- `updates[24]`：`step,policy_version_sampled,policy_before_sha256,policy_after_sha256,reference_sha256,optimizer_steps=1,responses=128,aux_nonzero_rows=0,prefix_rows=0,precommit_gate='PASS_BEFORE_ADAM',groups,numerics,evidence`。`groups` 等于预期组字段另加 `J[8]`；`numerics` 使用 `check_numerics` 的六个字段名，`kl_unweighted` 不可传乘0.02结果；`evidence` 指向本步 `raw,token_trace,native_update,commit`。整路径匹配后再独立核对 step12/24 端点从对应 commit 加载，不能拿最新权重代替 step12。

**集群适配接口**：`prepare(public_request, image) → traced_input`；`generate(policy_hash, traced_input, draw_seeds) → original_trajectories`；`score_response(...) → F6 sidecars`；`native_J_advantages(G8) → coeff[8]`；`prepare_step_behind_new_registration(...) → FP32 precommit numerics/receipts`；`check_numerics(...)` 后 `commit_once(...)`；`load_verified_endpoint(...)` 单独对接冻结评测器。完整组内优势与每个轨迹身份须由新 wrapper 独立复算并比对历史J-only内核；训练不使用 S/U/B/报告正确性，也不使用评测的裸 A；native training / access / transactions 旧身份判断需**为新实验另注册**，不可直接调用旧CLI去跑新的世界。历史事实、此次要实现的适配、已跑的CPU夹具、未来服务器实测须在交付报告中分别列明。
