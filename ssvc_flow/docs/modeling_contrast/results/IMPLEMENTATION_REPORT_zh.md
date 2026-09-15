# 实现与验证记录

新模块测试状态 `PASS`，通过数 213，失败 0，错误 0。工程总状态 `BLOCKED`。

父建模/数学回归状态 `PASS`，总测试 226。其他旧回归：通过 884，失败 52，错误 163；没有把旧失败删掉后声称全套通过。

来源：[final_tests.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/implementation/final_tests.json)；SHA-256 `53ed594752514bdb66c853b1b8e8b2c99753e4ad60aac2b820e04addc870773a`；[statistics_regression_receipt.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/implementation/regression/statistics_regression_receipt.json)；SHA-256 `ac578a8f3b8f3f75fa792816dc6158977e73b39128ce8c624530df7a87b4b23b`；[SUMMARY.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/implementation/regression_remaining/SUMMARY.json)；SHA-256 `41e69be3493ee57e00b30c2961fe9cd4123f6ba2f08d77c13ae3c540cd869ff1`

## 明确排除和缺项

实际排除文件/理由：`{"test_model_official_tiny.py": "EXCLUDED_USER_NO_NEW_QWEN_CALLS", "test_frozen_qwen25_tiny.py": "EXCLUDED_USER_NO_NEW_QWEN_CALLS", "test_data_verifiers.py": "DELEGATED_OTHER_AGENT", "test_data_generation.py": "DELEGATED_OTHER_AGENT", "test_math_flow.py": "DELEGATED_OTHER_AGENT", "test_contract.py": "DELEGATED_OTHER_AGENT"}`。

实际deselected节点：`["tests/test_audit_r0_remaining.py::test_cross_split_passes_generated_dataset"]`。两个tiny Qwen测试涉及真实新模型调用，sealed-confirm节点不读封存真实测试响应；其他委派回归由独立回执说明。

旧回归sealed-confirm实际读取：False。错误原因须以原始日志为准；缺fixture不是本轮方法失败，也不是工程全绿。

## CPU实际运行及外推

| 项目 | 实际回执值 |
| --- | --- |
| smoke观测秒 | 20.1084 |
| smoke拟合秒 | 2.4585 |
| smoke packet bytes | 24633331 |
| smoke fit bytes | 12024137 |
| N1 wall seconds | 629.215 |
| N2 wall seconds | 778.93 |
| N2新增optimizer updates | 0 |

资源外推（不是全程实测）：`{"wall_seconds": 2676.942099974258, "peak_ram_gib": 1.8118438720703125, "added_output_bytes": 3115227730, "temporary_bytes": 367001600, "basis": "real seed101 X_BASE anchor8 all three n measured; 60 finite anchor states; 110 equivalent model sweeps including oracle isolation and robustness; 1200s overhead"}`。

来源：[smoke_result.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/smoke/smoke_result.json)；SHA-256 `63ec4d17309890d07e870b5fa6664ec55e5e8e9056f347b03f5992362a23128a`；[stage_result.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N1/stage_result.json)；SHA-256 `6c08b84f93e901f1466e3dd46ac7695ae24e5d6560fbe5ad6829bb8ce89d2261`；[stage_result.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N2/stage_result.json)；SHA-256 `50ec408b27f0edbc744efeccda0e8bb51170e45a97b6f32764b56ad1069b7355`

## 数据与代码边界

原件完整轨迹 60/60；N1 状态 `COMPLETE`，轨迹 12；N2 状态 `COMPLETE`，exact 轨迹 60，finite 轨迹 20。所有旧seed均为legacy_diagnostic，旧locked-test只作公开开发/复现，原role保留。

父来源：[parent_integrity_audit.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/runs/modeling_contrast_v2/N0/parent_integrity_audit.json)；SHA-256 `dd978b55c0816e62f91f8ac09fb7af1b1820114cce079cf9882588f9ce648cae`

parent服务、有限packet、拟合输入和oracle评分分开。联合packet保留共享baseline、alias跨bank与CRN/LR共同贡献协方差。原5份O-IND计数不改写；补充副本和evaluation独立观测保留来源字段。旧源码保留，完整状态hash不可得时输出缺项，不把参数hash代替Adam/RNG状态。

## N1元数据勘误及C1辅助修正

协方差勘误核验 `VERIFIED_AND_SUPERSEDED`；N1 oracle/covariance_audit.json的bank_independence:true声明过宽。旧O_IND noise 0-4的fit计数曾按全局参数alias复用，跨bank协方差一般非零；N2联合观测保留这些cross-bank项。新evaluation采用独立新packet，不能将旧历史计数的相关性或该笼统元数据直接移用到新eval。 C1辅助修正 `PASS`，验证overlay数 108。N2/C1_auxiliary只替换C1拟合的辅助total levels。按各自bank的计数补充；主contrast观测和N1原件不变。保存5份原历史副本及origin counts；新fit/eval独立性由bank-local修正回执及独立测试核验。 N1原始hash保持，逐文件及103 X_VALID anchor24 bank0/1的NPZ证据见[MEASUREMENT_COVARIANCE_AUDIT.json](/Users/louis/Documents/ChatGPT/dissertation-ntu/.worktrees/ssvc-modeling-contrast-v2-20260915/ssvc_flow/docs/modeling_contrast/results/MEASUREMENT_COVARIANCE_AUDIT.json)。

所有本报告读取文件及实际SHA-256见REPORT_SOURCE_BINDING.json。报告文件manifest仅包含报告自产文件；源码commit、push和源码总manifest由最终交付步骤提供。

## 复算源码清单的范围

独立raw复算后仅目录provenance发生变化的已知未执行报告/编排模块数：3。若非零，readiness逐项保留SOURCE_CATALOG_CHANGED_AFTER_REPLAY及旧/新hash；recompute、packet、parent、observations、targets等真实依赖与所有raw文件仍严格比对原hash，这项例外不扩大科学复算覆盖范围。
