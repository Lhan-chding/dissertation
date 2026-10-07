# 最终交付入口

本轮冻结294项全部完成，0技术缺失。先阅读[四部分最终结果](FINAL_FINDINGS_zh.md)，再读[事实与解释边界](DISCUSSION_FOR_CLAUDE_zh.md)。

## 关键证据

- [发现覆盖与拟合](final_analysis/discovery_fit_findings_zh.md)：四个J、重合/新增/丢失、预算前缀、匹配差异、曝光和sentinel内外分解；CSV保留逐题和逐步记录。
- [迁移解释](final_analysis/transfer_scientific_readout_zh.md)与[完整比较表](final_analysis/transfer_report_zh.md)：主/次比较、父内repeat均值、对父变化、两调用固定策略、风险和语义分层。
- [保持性](final_analysis/retention_zh.md)：图像64/duplicate32分开，64个共享scene联合E/G配对；每臂/父/重复及区间在JSON/CSV。
- [统一揭晓](evidence/final_release/TEST_RELEASE.json)、[执行范围](evidence/final_release/EXECUTION_SCOPE.json)、[训练登记摘要](evidence/final_release/TRAINING_REGISTRY_COMPACT.json)、[成本账本](evidence/final_release/EXECUTION_COSTS.json)。
- [训练与非封存完整性](evidence/final_release/ssvc_training_prerelease_audit.json)、[揭晓后E/G原子响应核验](evidence/final_release/postrelease_atomic_audit.json)、[R0采样核验](evidence/final_release/r0_rollout_audit.json)、[服务器到本地核验](evidence/final_release/SERVER_TO_LOCAL_VERIFICATION.json)、[最终报告传输核验](evidence/final_release/FINAL_REPORTS_TRANSFER_VERIFICATION.json)。
- [原始数值审阅](NUMERICAL_REVIEW_zh.md)、[中断和恢复](RECOVERY_20261006_zh.md)保留，不能被最终PASS覆盖。
- [离线分析代码和重算说明](offline_analysis/README.md)：标准库，无模型调用；与冻结训练代码分离。

## 完整证据包

本地：`artifacts/reports/verified_discovery_transfer_20261007/SSVC_VERIFIED_DISCOVERY_TRANSFER_RESULTS_20261007.zip`。

包含所有201408条正式teacher/评估回答的19392个chunk、4096条R0采样、失败回答、规范标签、T/V/E/G/R清单、全部更新与attempt日志、恢复证据和Slurm日志、最终报告/分层表、冻结源码以及离线分析脚本。包内PAYLOAD_SHA256.json与VERIFY_PACKAGE.py支持解压后逐文件验证；ZIP的外部SHA256/CRC/安全路径核验见evidence/final_release/PACKAGE_VERIFICATION.json。

不含权重、113个.pt检查点、依赖环境、重复atomic包装、临时文件和锁，因此不是独立训练重跑包。完整检查点实际路径与SHA256保存在训练审计JSON；它们和原子封装继续保留于服务器`/projects/varunssd/louis-ssvc/verified_discovery_transfer_20261006/run_v2`，未删除任何当前实验数据。服务器源代码为`code_v2/ssvc_flow`，科学提交`ed1d24839dc6bd374d17a92bd29678b7185c7605`。

## 验证与范围

34,509个下载证据文件和73个最终报告/队列文件均与服务器SHA256逐字节一致。独立审计核验113个完整检查点、5120 SFT/128 R0更新、全部正式原始回答。离线统计通过B16集合重建、X/S/W/I闭合、K8闭合、冻结主比较再现、场景配对恒等式及合成bootstrap检查。科学源码未修改、没有追加训练或测试抽样。

保留审计工具自身的过程记录：初次广泛预揭晓读取被自动审批阻止，随后改用不访问E/G的审计；揭晓后E/G审计初次使用错误目录导致计数断言失败，纠正为sealed目录后通过；无实验数据被改写。审计脚本两次初始checkpoint预期修正记在训练审计回执。它们不记为模型失败，也不隐藏。

主比较区间跨零、V不利值、duplicate下降与全生命周期成本未计量部分均保留。没有使用不利结果作为扩矩阵或改变参数的理由。

Git内训练登记为明确标注省略逐步metrics/逐题exposures的摘要，原文件哈希绑定；完整TRAINING_REGISTRY.json在ZIP的run_v2中。CSV采用标准CRLF，Git空白检查以cr-at-eol识别行尾；无数值转换。

最终核验和交付完成后，ssvc半小时监控已删除。
