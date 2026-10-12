# 独立最终证据审计

2026-10-12 23:03 新加坡时间实核，三个新建普通 CPU 只读审计作业 200452、200456、200457 均真实 `COMPLETED 0:0` 并已离队。核查、评分复算和结构解析全部在 CPU 分配内执行，没有新生成、重新释放、改动训练或评价文件，也没有使用 GPU。

## 核验结果

- 全 12 训练与全 13 评价 GPU 作业、评价控制器 CPU198239 均成功终态并已离队。
- 重新加载全部 12 条真实 step96 检查点，复核完整参数、Adam96步、调度器、cursor、RNG、状态分字段哈希、adapter 字节以及 TRAIN_FIT32/64/96。八条 HDD 目录链接均显式解析到原登记目标。
- 458 个训练源文件与 466 个评价源文件逐一哈希通过；四份冻结文件字节哈希通过。
- 最终 `COMPLETE.json` 的 13 个 release 对象哈希、每个 release 的记录对象哈希、授权/模型验收绑定全部通过。另记录 release 文件字节 SHA；对象哈希与文件字节 SHA 分开使用。
- 227700 条终点评价逐行重算，和统一释放的每个分数及槽位元字段完全一致，差异 0；原始组字节、记录对象、面板覆盖、唯一槽位和计划绑定全部通过。起点 6656 条已评分输出也逐行复算，差异 0。
- 复算复用冻结评分函数，但由独立只读脚本遍历、校验和比较，不宣称另行实现了一套独立科学评分器。

小型核验回执为 `audit_result.json`、`audit_cpu_terminals.json`。大型逐行无解释正文元数据、原始组索引、gold 结构和规范化答案放在工作树的 `artifacts/sr_f12_final_20261012/audit/`；`audit_artifact_locations.json` 给出精确位置、大小和 SHA256，下载后已逐文件复核。服务器独立审计输出根为 `/projects/_ssd/varunssd/louis-ssvc/sr_f12_final_audit_20261012_2310`，实验根未改动。

首次提交误用不存在的 `cpu` 分区，被调度器明确拒绝且没有创建作业；随后核对既有 CPU 配置和分区清单，以 `cluster02`、普通 QoS、4 CPU、12 GiB 唯一提交。该辅助提交失误保留于 `audit_submission.json`，不是科学任务失败或重跑。

## ChartQA 冻结评分口径诊断

全部 32500 条 ChartQA 中，32497 条生成文本以 LF 换行结尾。冻结 `score_chartqa` 仅 `.strip(" ")` 去空格，未去尾换行：精确字符串比较因而全部为 0；数值 relaxed accuracy 中 `float()` 能接受尾换行，数值答案可以得分，非数值字串则受到换行惩罚。例如基线 `chartqa-0001` 的 gold 为 `0.57`，生成字符串为 `0.57\n`，冻结 relaxed=1 而 exact=0；原始组路径与 SHA 见 `audit_structure_result.json`。

原冻结 exact 与 relaxed 分数全部保留。另以 `POST_HOC_WHITESPACE_SENSITIVITY` 明确标注的诊断，仅剥除字符串首尾空白，不提取数字、不修复答案、不再生成、不覆盖正式成绩。基线冻结 relaxed 为 1670/2500（66.8%），敏感性为 2148/2500（85.92%）；敏感性 exact 为 1953/2500（78.12%）。各模型 relaxed 受此单一口径影响增加 474–480 条，约 19 个百分点。每题正式/敏感性分数和 human/augmented 分层计数分别在 `audit_chartqa_sensitivity_rows.jsonl` 与 `audit_chartqa_sensitivity_result.json`。

因此 exact=0 无法解释为完全没有外部答题能力，亦无跨模型鉴别力；正式 relaxed 的外部能力解释必须同时交代非数值行的格式惩罚。事后敏感性不替换预登记正式终点，不用于选择模型或调整科学结论。
