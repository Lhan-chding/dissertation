# 数据合同与接口细节

本文件是新模块的数据格式建议。实例内容只是schema示例，不是已生成的新训练/测试实例。

## 1. 任务公开区与审计区

`tasks_public.jsonl`每行：

```json
{
  "task_id":"opaque-id",
  "base_instance_id":"opaque-id",
  "split":"T_train",
  "family":"cross_series",
  "observed":[3,46,47,32],
  "H_original":[[1,0,0,1],[0,1,0,1],[0,0,1,1]],
  "b_original":[35,78,55],
  "legal_domain":[0,99],
  "operation":"difference_pairs",
  "template_version":"inherit-O0-L11-B1-plus-new-cohort-v1"
}
```

公开task对象可以包含family/结构，不包含j、DPE、gold、历史是否失败。prompt编译器只接收公开对象。train/V/E的role约束由loader执行，不能接受随意拼接的目录。

`audit_only.jsonl`：task_id、true_world、corrupted_index、truth_orbit_key、DPE1各协议、center位置、独立solver结果等。GOLD臂明确允许读取T审计标签；SELF构造器不允许。E审计标签只用于最终评价，teacher/训练selector不得读取。

新数据渲染图像时使用true_world是生成器定义的一部分，图像guard本来就是信息条件对照；该图不能泄漏给本轮符号训练任务。

## 2. 协议实例

`protocol_cases.jsonl`：task_id、protocol_id、system/user完整文本、output_order、H_display/b_display、公共图像路径（若有）、compiled_input_token_count。

output_order采用0-based。例如L11=[3,2,1,0]。B1是固定原顺序上的行变换，不跟据j挑选更有利的first leaf。所有变换映回同一base_instance_id，评价使用H_original。

检查输出示例仅用于数学/工程单测，不能作为few-shot样例意外写进正式提示。

## 3. 原子生成记录

每条`raw_generations.jsonl`至少包含：

- request_id、parent_id、model_revision、runtime_version；
- task_id、base_instance_id、split、protocol_id、role、pipeline_repeat、draw_index、sample_seed；
- raw_completion、raw_token_ids、completion_length、stop_reason、elapsed_seconds；
- emitted_vector（可能null）、canonical_vector（可能null）、strict_parse_status、domain_status；
- public_verifier_pass；
- 独立audit事件与指标在派生表中，不用作生成输入；
- 生成时原本已有的logp可以附带，禁止为凑齐这个字段新增评分。

固定元数据输入、模型和seed下同一请求只保留一条已接受记录。异常中断可以重试相同请求，但不能丢弃正常返回的错误回答再抽一次。回答错误不等于运行失败。

## 4. 固定策略与选择记录

`fixed_policies.json`按parent和family保存三个独立对象：

1. best_single_B16：V上每题16次至少成功一次的均值；
2. best_single_B2：V上无偏pass@2均值；
3. best_pair_B2：允许重复协议的全部固定二元组合，按同样两调用效用选择。

精确并列优先O0、L11、B1，组合按该顺序字典序。实现可用Fraction保证本配置有限计数下的稳定比较，或只使用1e-12数值tie容差；该容差不是实际效果阈值。

固定FAMILY_MIX先调用O0，再调用trend-L11或cross-B1。best_pair_B2的调用先后，可在V上按单次成功率高者优先来减少预期调用，精确tie按固定顺序。调用次序同样在读E前冻结。

## 5. 数据view

`training_view.jsonl`：task_id、O0_prompt_id、canonical_target_string、target_token_ids（仅target、不含prompt）、EOS_id、weight、label_source_provenance、discovery_request_ids。

weight在集合内按题等权。success_count不进入weight。label_source_provenance只审计，不进入model或sampler seed。源标签不允许相同task出现两个不同canonical world。

GOLD_MATCH_MIX按family×j抽样，保存每格N与配额和rng；允许与J_MIX有交集，报告Jaccard，不能偷改为完全不相交而引入新偏差。

`alias_registry.json`只合并parent、repeat、view、replayview、LoRA初始状态、目标token、batch/optimizer/scheduler全部一致的作业。两个不同teacher或两次独立pipeline重复不合并。空J返回父模型的evaluation别名单独标注，父E取前8条、G取前4条。

## 6. 训练与评价记录

`updates.jsonl`：step、lr_used、focus/replaytask IDs、unique_seen、sequence_lengths、各loss、gradnorm_preclip、processed_tokens、GPU秒、峰值显存。不得把固定target的teacher log-prob当作PPO旧概率；SFT没有ratio。

`checkpoint.json`：parent、arm、repeat、step、view ID、optimizer state origin=fresh/resume、checkpoint位置、代码版本。参数tensor、optimizer、scheduler、sampler、RNG一起提交。

`per_task_metrics.csv`：parent、arm、repeat、step、split、task_id、n、nX/nS/nW/nI、passK、各语义计数；proto-specific字段单列。epoch/step不同不能混成一个策略。

`comparisons.csv`：effect_name、conditioning_population、parents、repeats、taskN、effect_pp、CI_type、CI_bounds、missing_or_alias_notes。CI明确是scene条件区间、MC区间还是仅重复结果范围。

## 7. 因果与信息边界

同一学生的loss与输出字段变化，不能直接命名为“推理模块改变”。发现集合更大也不能直接当作最终模型收益。每项分析要指向：哪个输入数据view、哪次参数训练、哪个固定评价协议。

主终点不是verifier挑过后的E成功率，而是每个训练后模型**一次O0回答的pX**。推理组合在独立小节报告，不能混入主SFT收益。
