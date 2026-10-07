# 数据与输出接口

## 1. 现成清单

- `common_targets.jsonl`：182条旧T共同题，每条含公开task、规范target、来源。
- `common_audit.jsonl`：对应真值／结构，仅供设计与审计。
- `replay_targets.jsonl`：64条原回放，保留source。
- `donors_public.jsonl`：96条新供体任务，三个臂同根对应。
- `donors_audit.jsonl`、`donor_targets.jsonl`：结构审计和供体金标分开。
- `E_DIAG/`、`E_CONFIRM/`：各自公开task与audit_only。
- `new_prompts_public.jsonl`：944条新任务的O0纯system/user文本。
- `schedule_108701/2/3.jsonl`：三个明确step/slot表，每表含A/B/C，12288行。注意实际文件名是`schedule_108701.jsonl`等。
- `training_jobs.jsonl`：18项计划；`evaluation_jobs.jsonl`：95个批量评估计划。
- `new_root_registry_AUDIT_ONLY.jsonl`：根、真值、各位置污染替代值；不得发送给生成模型。

## 2. 执行身份

每个训练作业键：`experiment,parent,block,arm`。checkpoint键再加`step`。父模型基线键不含block，三个block共用同一父基线。

每个生成键：`checkpoint,panel,task,draw,role`。保留root_id以便配对统计。每个key只允许一个正式完成记录；失败尝试另外保存，不能把多次重试中最好的答案当正式结果。

## 3. 原始生成最少字段

```
experiment_id, checkpoint_id, parent, block, arm, step,
panel, task_id, root_id, draw_index, sample_seed,
public_prompt_version, raw_text, token_ids,
stop_reason, generated_length, elapsed_seconds
```

已有生成器产生的chosen-token logp可记录，但不新增teacher强制重评分。缺失logp不由外部近似值填充。

解析输出另存：`parsed_vector, parse_ok, domain_ok, X/S/W/I, verifier_pass, edit_mask, M, B, L, F, value_present, mixed_anchor_public, mixed_anchor_audit, projection_success`等。audit相关字段由评分端生成，不能回送训练。

## 4. 训练逐step

必须有16个有序槽位，记录：

```
update, slot, role(common/donor/replay), task_id, root_id,
target_token_count_including_EOS, sequence_mean_NLL,
coefficient=1/16, microbatch_id
```

每step还包括：实际lr、clip前norm、clip比例、参数变化摘要、更新时间、完整/失败状态。NLL不能称作梯度贡献；相同slot共同题的后期loss不同不代表data错配。

## 5. 分层结果

每一行都保留parent/block/arm/step/panel/family/center/corrupted_index/n_tasks/n_roots/n_draws。c×j和family是独立字段，不用一个“hard”列替代全部信息。

中心correct-but-extra既给全体分母，也给可解析／合法子分母；越界单列。归因比例的分子分母都输出。0分母用null，不空填0。

## 6. 泄漏与权限

- 新生成器可以知道真值来构造正确的因果诊断语料；这项金标权限在所有三臂中相同并明确声明。
- 训练器可以读训练目标，但不能读E_CONFIRM审计/结果。
- 渲染器只取公开任务关系与观察；不能把arm、j、truth加到prompt。
- 评价器按task_id取审计标签，不根据输出好坏选择解析方式。
- 公开删改投影不读取audit。投影成功结果本轮不回流训练。
- E_DIAG参与本轮报告，不参与改配置。若人为用它改设计，另登记版本并重新锁定最终确认；不能静默继续使用原确认声明。

## 7. 共享根与去重

本轮故意在一个root下创建不同c/j的任务，或三个供体条件。它们允许共享规范目标，不能被普通“每truth只留一条”去重删除。

但一个root不能跨DONOR_TRAIN/E_DIAG/E_CONFIRM。最小统计簇也是root，不是生成字符串。计算pass@K必须在单checkpoint单task中完成，不能混合不同臂的抽样。
