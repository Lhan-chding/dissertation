# SR-ORDER-PROBE 最终状态（2026-11-03）

实验与审核已完成：216条烟测、4320条主推理、72条独立代理复核，0训练更新。主作业242982 COMPLETED 0:0；新加坡06:18:03开始、10:44:21结束。后续不重复提交任何已完成请求，不新增训练、种子或正式TEST。

完整结果报告见 ../results_20261103/RESULTS_REPORT_zh.md；每父×条件×命名54组见FULL_54_GROUPS.csv。主要结论包含前缀UNKNOWN与父差异限制，不能只引用显著差值。72审核28处B状态分歧均为代理初判将不可判记0，初判和裁定完整保留；未改生产评分。

原始输入/suffix token和概率全部保存，4536条请求各1次START/SAVED。完整本地重评分及4份统计产物字节一致；完整性与复现回执见results_20261103。此前95项实现测试与启动定向回归保持历史记录，最终全套98项测试通过，详见results_20261103/FINAL_VALIDATION.json。

最终交付目录位于工作区outputs/sr_order_probe_final/SR_ORDER_PROBE_COMPLETE_20261103，包含完整原文/token、只读源码、统计及审核。压缩包校验结果另见DELIVERY_VERIFICATION.json，压缩包和全量原文不提交git。

旧RESUME与EXECUTION_STATE中的等待状态仅为历史快照，以下前置状态索引优先。不再开监控或下一项实验。
