# 10 生产接口与实现验收

以下是本包新定义的适配器接口，不是假装旧仓库已存在这些函数。Codex应在新实验模块实现并连接旧代码。

## Adapter API（概念签名）
```
load_parent(block_id, resolved_contract) -> ModelHandle
make_native_prompt(task, protocol_revision='v2') -> Prompt
rollout(model, assignments, generation_seed_map) -> list[RawRecord]
score_native(task, raw_record) -> NativeScore
verify_semantics(task, raw_record, native_score, verifier_version) -> SemanticState
native_advantages(group_native_scores, resolved_contract) -> list[float]
update_with_advantages(model, raw_records, final_advantages, frozen_parent, contract) -> UpdateReceipt
save_resume_state(run_id, step, model, optimizer, scheduler, rng_states) -> CheckpointReceipt
```
生产代码入口建议统一 `python -m <new_package>.run --stage ... --freeze ...`；真实模块名在F1填写。不得在文档里给不存在的旧模块传一串未经核验的CLI flags。

## 关键接点
A只替换选择(qid,G)的位置；奖励、优势和更新均使用旧原生实现。G不同的组必须有独立边界，不能把128回答全局当一个组。
C固定G8；在native advantage算完、进入原policy surrogate前，加本包credit residual。若原trainer在函数内还会归一化advantage，应新增明确的`advantages_are_final`入口，并用J臂梯度等价测试验证；不得删除所有臂原有步骤来方便接入。
每个token应用哪一个sequence advantage、logp masking与原生一致；额外项不作用于prompt/image token。

## 不允许替代实现
不能以参考population_advantages直接替换原库，除非逐组golden tests通过并明确签名/epsilon/precision一致；不能自行换KL estimator；不能新增reward clipping抹平新增信号；不能读取CONFIRM决定采样。
本包没有通用GPU launcher，因为集群队列、分配、模型路径、仓库入口必须从用户真实环境解析。Codex可生成该环境专用sbatch，但只使用已有授权QOS和GPU配额。

## 最小GPU工程验收（不计科学路径）
从父模型副本跑1次更新/恢复回放，验证J臂与旧路径等价、SELECT residual注入、128条预算、恢复后下一组seed一致。使用专用ENGINE题，不使用确认题。只做数值一致性和恢复测试，不根据这个step效果选择β或训练臂；验收后丢弃工程权重，科学路径从原父模型重启。把额外ENGINE回答数单独记录。

## 规范化字段映射（必须在适配器中显式做）
本包示例`Observation.j`←原生score.J或实际training reward（需先证明一致）；`finished`←原生finish且generation_status无技术失败；`protocol_valid`←原生合法性规则（不能简单 reason==ok）；`sufficient`←certificate_sufficient；`perception`←非空证据的certificate_truth审核；`selection`←新验证器S；`defined`←任务oracle定义状态。row_id包含model/block/repeat/step/group/draw，不得只用qid。
原始score字段原封不动保存，规范化层不得覆盖它们。Schema中semantic dimension用P/S/O/E；CPU信用参考使用perception/selection的字符串枚举，这个映射不是两个不同标签系统。
