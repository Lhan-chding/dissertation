# E05｜局部响应验证的关键缺口
## warm直接验证没有非零辅助处理
warm阶段12banks×5lambda有真实候选fork，但控制bank打分与直接重采样只在bank0和6执行。两个bank各自所有lambda对应的参数hash完全相同，梯度差与辅助参数增量也为0。

因此，在同一bank内比较lambda0与1，比较的是同一个参数策略，而不是两种具有不同处理的候选更新。56个辅助响应条目均为0。3072条直接输出仍可用作零效应/采样噪声及主更新校准，但不能验证非零辅助干预方向、效应大小或SSVC安全选择。

有实际作用的warm bank3、4、5、11缺少对应控制响应和直接重采样。应按组构成和实际更新差选择这些处理，不按评估结果正负挑选；保留bank0作负对照。

## “predicted_delta”不是已验证的概率流理论
所存响应来自普通、未截断importance sampling有限候选重打分，不是已经验证的ODE、共享核或扩散预测。cold bank6有非零候选，但没有同等直接重采样完成验证。不能将IS预测与自身评分一致当成独立机制验证。

## 测量门槛与采样数不相容
控制bank每prompt16输出；使用归一化权重时max_i wtilde_i至少为1/16=0.0625。若逐题要求该值≤0.05，连完美均匀权重都不能通过。因此逐题OVERLAP_WARNING不能全部解释为策略漂移过大。源文件已保留该scope注释。

修复应明确采用随样本量归一的诊断，如n·max(wtilde)、相对ESS，并报告事件支持和真实覆盖率；或提高n。不能简单放宽阈值就改称“安全”。

## 总体ESS掩盖事件缺失
代表性候选总体ESS约占样本数98%以上，但cold控制集中11/48题、warm中12/48题没有X。若某题无X，普通IS估计的该题X概率仍为0，不管新策略的真实X概率如何。高ESS不能弥补未观察到关键事件。

这需要区分：处理是否非零、候选是否有足够重叠、目标事件是否被观察到、估计误差是否小于安全容忍度。任一环节不足均应保留UNKNOWN。


## 原始证据位置

以下路径相对于原始交付包 `SSVC_GPT_PRO_DELIVERY_20260914/`；是本次审计所读的文件，不代表本报告重新执行了 GPU 实验。

- `02_data/R3_warm/candidate_manifest.json`
- `02_data/R3_warm/response_bank_00.json`
- `02_data/R3_warm/response_bank_06.json`
- `02_data/R3_warm/direct_validation_effects.csv`
- `02_data/R3_warm/direct_validation_bank_00.json`
- `02_data/R3_warm/direct_validation_bank_06.json`
- `02_data/R3_cold/response_bank_06.json`
