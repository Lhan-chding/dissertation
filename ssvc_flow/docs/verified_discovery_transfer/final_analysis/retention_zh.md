# G保持性：已揭晓输出的配对分析

分析状态：COMPLETE；缺失项目：0。

图像64题（cross/trend各32）与独立duplicate32题分开。图像和E共享的64个基础场景在bootstrap中联合移动。各区间为条件于已运行检查点的边际95%场景区间，不是种子总体区间，也不是同时置信带。

|训练臂|分层|相对父模型变化 pp|条件场景95%区间 pp|
|---|---|---:|---|
|GOLD_ALL|image64_family_equal|-0.684|[-2.051, 0.000]|
|GOLD_ALL|image_cross32|-1.367|[-4.102, 0.000]|
|GOLD_ALL|image_trend32|0.000|[0.000, 0.000]|
|GOLD_ALL|duplicate32|0.391|[-4.688, 5.859]|
|GOLD_MATCH_MIX|image64_family_equal|0.000|[0.000, 0.000]|
|GOLD_MATCH_MIX|image_cross32|0.000|[0.000, 0.000]|
|GOLD_MATCH_MIX|image_trend32|0.000|[0.000, 0.000]|
|GOLD_MATCH_MIX|duplicate32|-2.734|[-9.375, 1.172]|
|R0_RESET32|image64_family_equal|0.000|[0.000, 0.000]|
|R0_RESET32|image_cross32|0.000|[0.000, 0.000]|
|R0_RESET32|image_trend32|0.000|[0.000, 0.000]|
|R0_RESET32|duplicate32|-0.391|[-1.172, 0.000]|
|REPLAY_ONLY|image64_family_equal|0.000|[0.000, 0.000]|
|REPLAY_ONLY|image_cross32|0.000|[0.000, 0.000]|
|REPLAY_ONLY|image_trend32|0.000|[0.000, 0.000]|
|REPLAY_ONLY|duplicate32|2.344|[0.000, 7.031]|
|SELF_MIX|image64_family_equal|0.000|[0.000, 0.000]|
|SELF_MIX|image_cross32|0.000|[0.000, 0.000]|
|SELF_MIX|image_trend32|0.000|[0.000, 0.000]|
|SELF_MIX|duplicate32|-2.344|[-6.836, 0.000]|
|SELF_O0|image64_family_equal|0.000|[0.000, 0.000]|
|SELF_O0|image_cross32|0.000|[0.000, 0.000]|
|SELF_O0|image_trend32|0.000|[0.000, 0.000]|
|SELF_O0|duplicate32|1.953|[0.000, 5.859]|
|SELF_SINGLE|image64_family_equal|0.000|[0.000, 0.000]|
|SELF_SINGLE|image_cross32|0.000|[0.000, 0.000]|
|SELF_SINGLE|image_trend32|0.000|[0.000, 0.000]|
|SELF_SINGLE|duplicate32|1.758|[0.000, 5.273]|

全部parent×repeat×arm单元、各parent重复均值及shared64的G/E联合变化见JSON/CSV。仅S96登记的臂，其汇总只覆盖S96。没有将G与E混合成排行榜；没有收益或负向变化均保留。区间包含0不等于无损或等效，未预登记非劣门槛，因此本表不输出“保持性认证”。

shared64的图像接口提供额外图像线索，接口差和变化差为描述性配对诊断，不能单独归因于模态。R0为32步，SFT为256步；计算预算不相等。


## 图像与duplicate的实际变化

- SELF_O0 / image64_family_equal：父模型100.000%，学生100.000%，变化+0.000 pp，95%场景区间[0.000, 0.000] pp。
- SELF_O0 / duplicate32：父模型98.047%，学生100.000%，变化+1.953 pp，95%场景区间[0.000, 5.859] pp。
- SELF_MIX / image64_family_equal：父模型100.000%，学生100.000%，变化+0.000 pp，95%场景区间[0.000, 0.000] pp。
- SELF_MIX / duplicate32：父模型98.047%，学生95.703%，变化-2.344 pp，95%场景区间[-6.836, 0.000] pp。
- SELF_SINGLE / image64_family_equal：父模型100.000%，学生100.000%，变化+0.000 pp，95%场景区间[0.000, 0.000] pp。
- SELF_SINGLE / duplicate32：父模型98.047%，学生99.805%，变化+1.758 pp，95%场景区间[0.000, 5.273] pp。
- GOLD_ALL / image64_family_equal：父模型100.000%，学生99.316%，变化-0.684 pp，95%场景区间[-2.051, 0.000] pp。
- GOLD_ALL / duplicate32：父模型98.047%，学生98.438%，变化+0.391 pp，95%场景区间[-4.688, 5.859] pp。
- GOLD_MATCH_MIX / image64_family_equal：父模型100.000%，学生100.000%，变化+0.000 pp，95%场景区间[0.000, 0.000] pp。
- GOLD_MATCH_MIX / duplicate32：父模型97.656%，学生94.922%，变化-2.734 pp，95%场景区间[-9.375, 1.172] pp。
- REPLAY_ONLY / image64_family_equal：父模型100.000%，学生100.000%，变化+0.000 pp，95%场景区间[0.000, 0.000] pp。
- REPLAY_ONLY / duplicate32：父模型97.656%，学生100.000%，变化+2.344 pp，95%场景区间[0.000, 7.031] pp。
- R0_RESET32 / image64_family_equal：父模型100.000%，学生100.000%，变化+0.000 pp，95%场景区间[0.000, 0.000] pp。
- R0_RESET32 / duplicate32：父模型98.047%，学生97.656%，变化-0.391 pp，95%场景区间[-1.172, 0.000] pp。

## 共享64场景的E/G联合配对

|训练臂|符号E相对父模型 pp|图像G相对父模型 pp|图像增益减符号增益 pp|后者95%场景区间 pp|
|---|---:|---:|---:|---|
|GOLD_ALL|+24.756|-0.684|-25.439|[-36.914, -14.600]|
|GOLD_MATCH_MIX|+16.406|+0.000|-16.406|[-26.270, -6.641]|
|R0_RESET32|+3.662|+0.000|-3.662|[-6.836, -1.074]|
|REPLAY_ONLY|+2.148|+0.000|-2.148|[-6.641, 1.855]|
|SELF_MIX|+6.494|+0.000|-6.494|[-14.014, 0.879]|
|SELF_O0|+6.152|+0.000|-6.152|[-11.035, -2.148]|
|SELF_SINGLE|+5.859|+0.000|-5.859|[-12.842, 0.684]|

变化差为（学生G−父G）−（学生E−父E）。负值表示该面板上符号端改善更多，不能单独解释为图像退化。图像端接近/达到天花板时，须结合G本身变化阅读。所有指标使用同一批场景bootstrap索引，各边际区间不等于同时覆盖。

观测为100%时，场景bootstrap可能给出[0,0]变化区间；这不含未观测回答的固定面板采样不确定性，不能认定真实失败率为0。原FINAL_ENDPOINTS逐题Wilson区间必须一并保留。
