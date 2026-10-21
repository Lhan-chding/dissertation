# 08 任务卡、验收和交付结构

## 任务卡
| ID | 任务 | 依赖 | 可验收产物 |
|---|---|---|---|
| T00 | 输入与旧代码定位 |无|INPUT_MANIFEST、环境/代码快照|
| T01 | native contract与golden tests |T00|J/adv/loss完整定义，测试逐项差|
| T02 | 世界生成和分割 |T00|8个split清单、gold程序、去重证明|
| T03 | P/S/O/E验证器 |T01,T02|schema、规则、源片段、hard fixtures|
| T04 | 冻结父模型B0 |T01|3,840回答与p初值|
| T05 | B2开发+C训练探针 |T02,T03|4,800+1,152回答与语义标签|
| T06 | 人工盲核 |T03,T05|200及最多100补充、可追溯人工来源|
| T07 | LZ采样与恢复 |T01,T04|DP/MC/边界测试与A冻结|
| T08 | A六路径 |T07|6×64更新、完整rollout/分配/检查点|
| T09 | 原生和有界优势重放 |T05,T06|逐组重放、条件shuffle、C_GATE|
| T10 | C生产集成和冻结 |T01,T09通过|J臂等价、残差界、无二次归一化|
| T11 | C24路径 |T10|24×32更新、四臂固定曝光|
| T12 | 最终确认与统计 |T08+T09/T11结束|冻结后开放确认、逐块/seed/世界报告|
| T13 | 完整打包 |T12|交付manifest、原始不利结果、中文报告|

T08和T05/T06/T09可并行。无人工审核时T06/T09输出等待原因，不冒充完成；T08不需要等待T06。T09未通过时T10/T11明确SKIPPED_BY_PREREG_GATE，T12仍交付A/B结果。

## 输出目录
```
results/SRF4_CONSOLIDATED_20261022/
  INPUT_MANIFEST.json
  ENVIRONMENT.json
  CODE_SNAPSHOT/  # commit、diff、实际依赖版本
  FREEZE_F0.json
  FREEZE_A_F1.json
  FREEZE_C_F1.json  # blocked时仍发布含原因
  RUN_REGISTRY.jsonl
  DATA/{B_DEV,B_CONFIRM,C_TRAIN,C_DEV,C_CONFIRM_ID,C_CONFIRM_TRANSFER,OLD_TRAIN,OLD_REGRESSION}/
  PROBES/{B0,B2_DEV,C_TRAIN_PROBE}/
  VERIFIER/{rules,labels,audit,coverage,fixtures}/
  REPLAY/{native,scalar,bounded,shuffle}/
  C_GATE_REPORT.json
  RUNS/{track}/{block}/{repeat}/{arm}/
    run_identity.json
    assignments.jsonl
    rollouts/  # 完整文本、token/logp、原始score、语义标签
    updates/  # native/final advantage与mask、采样层级、资源
    checkpoints/
    technical_events.jsonl
  EVAL/{DEV,REGRESSION,CONFIRM}/
  ANALYSIS/{paired,root_bootstrap,seed_sensitivity,retention}/
  REPORT_ZH.md
  RESULT_MANIFEST.json
```

## 每次更新必须记录
checkpoint_before/after；step与attempt；group_id及其唯一题目；G与slot；raw J及全部原生分量；P/S/O/E与UNKNOWN依据；eligible mask、s、mean_F、residual、A_J、A_final；shuffle种子和映射；loss各项及分母；KL、ratio/TIS clipping、梯度范数；生成/训练token；截断与finish；参数/优化器RNG状态；实际显存/用时/GPU小时。
step只有成功完成更新与保存可恢复状态后才提交；基础设施重试不能重复累计已提交step。若生成成功而更新失败，是否复用该step rollout按原生on-policy版本校验，参数未改变才允许；否则丢弃原因保留并重采，不能当免费额外探索。

## 最终报告的固定问题
1. 哪些数值是旧报告、本次重算或新实验？
2. P/S可验证覆盖与错误率如何，UNKNOWN在哪里？
3. 原J已经包含什么，新增S增加了什么？
4. 哪些组实际优势改变、改变方向是否可靠？
5. SELECT分别是否优于J、EVID、SHUF？各block/seed是否一致？
6. 新世界、迁移层、旧任务保持、协议和长度副作用怎样？
7. A是否有效，且它的结论为什么不替代C？
8. 未通过门禁或技术阻断是否完整保留？

## 退出条件
本轮按照预登记样本量/更新步数结束，不按照GPU小时或验证分数结束。只有明确技术完整性问题可阻断；所有阻断有可复现错误和受影响范围。不要提交“已运行”却只有配置、不要把脚本输出当原始训练证据。

本包的`configs/RUN_MATRIX.csv`是计划矩阵，所有行状态均为PLANNED_NOT_RUN；它不能代替实际执行后的RUN_REGISTRY。参考测试使用的合成group、人工审核空模板与机器门禁空模板都不是实测证据。
