# SR-F1.2 最终交付入口

先读 [最终中文报告](FINAL_REPORT_zh.md)。本轮已完成12条96步训练、13个端点评价及统一释放，并重新核验227,700条端点评分、6,656条起点评分。没有新增科学实验。

## 主要文件

| 文件 | 用途 |
|---|---|
| FINAL_REPORT_zh.md | 中文结论、全部不利结果、统计边界、成本与技术事件 |
| PRIMARY_CONTRASTS.json / PRIMARY_FIGURE.svg | 四项注册主对比，三种子及95%/98.75%区间 |
| ANALYSIS.json / RESULTS_LONG.csv | 全部模型、面板、题族、变体、协议及条件分母 |
| TRANSFER_RESULTS.json | 独立纯答案协议和三类偏移的次要对比 |
| SEMANTIC_PREDICTION_RESULTS.json | 冻结P1–P3的检验，含反向和不确定结果 |
| DIAGNOSTIC_VIEWS.json / STRUCTURAL_RESULTS.json | 五视图及变体配对；无终点MONITOR原图 |
| REWARD_ADVANTAGE_AUDIT.json | 固定起点MONITOR块上的四臂离线信用审计 |
| TRAINING_SUPPORT_AND_FAILURES.json / GPU_ACCOUNTING.json | 1152步数值/成本/机制、31个GPU作业计账 |
| CHARTQA_RESULTS.json | 冻结正式ChartQA指标，保留空白口径问题 |
| CHARTQA_WHITESPACE_SENSITIVITY.json | 明确事后的首尾空白敏感性；不替换正式成绩 |
| audit_result.json / audit_note_zh.md | 全量只读独立遍历与复算、完整状态、来源与HDD验证 |
| INDEPENDENT_ANALYSIS_REVIEW.json | 另一实现对统计点值及区间的独立核验 |
| RUN_MATRIX_FINAL.json / RAW_OUTPUT_INDEX.json | 最终作业、训练身份及原始输出定位 |
| VALIDATION.json | 实际执行的本地检查及已解决的检查工具问题 |

## 可验证证据包

本地ZIP：工作树 `artifacts/sr_f12_final_20261012/SR_F1_2_FINAL_EVIDENCE_20261012.zip`。

解压后的目录保留仓库相对结构：

- `docs/sr_f1/analysis/SR_F1_2_FINAL/`：本报告、统计脚本和小型审计回执。
- `docs/sr_f1/analysis/SR_F1_2_THREE_SEED_TRAIN_FIT/`：先前已核验的全部拟合分数及分析。
- `artifacts/sr_f12_final_20261012/evidence/`：13个逐条释放分数文件、冻结文件、1152步日志、最终训练/评价验收和固定评价源码。
- `artifacts/sr_f12_final_20261012/audit/`：无解释正文的原始槽位元数据、规范化答案、原始组路径/hash索引、起点评分、ChartQA每题敏感性。
- 根目录 `SHA256SUMS.json`：各内容文件的相对路径、字节数与SHA256；`verify_package.py`逐文件复核。

包外 `PACKAGE_VALIDATION.json` 记录ZIP字节SHA、唯一安全路径、CRC和全部内容逐项SHA核验。清单不递归列自身；外层ZIP SHA不嵌入ZIP内，避免自引用。

## 本地复算

在仓库根或解压包根执行，使用已安装NumPy、Matplotlib的Python3.12环境；没有模型加载或GPU生成。以 `python` 表示该环境：

```bash
export PYTHONPATH=artifacts/sr_f12_final_20261012/evidence/deployment/early_sealed_20261011/code/src
python docs/sr_f1/analysis/SR_F1_2_FINAL/analyze_released.py artifacts/sr_f12_final_20261012/evidence docs/sr_f1/analysis/SR_F1_2_FINAL
python docs/sr_f1/analysis/SR_F1_2_FINAL/training_analysis.py --evidence artifacts/sr_f12_final_20261012/evidence --output docs/sr_f1/analysis/SR_F1_2_FINAL
python docs/sr_f1/analysis/SR_F1_2_FINAL/companion_analysis.py artifacts/sr_f12_final_20261012/evidence docs/sr_f1/analysis/SR_F1_2_FINAL
python docs/sr_f1/analysis/SR_F1_2_FINAL/sensitivity_analysis.py artifacts/sr_f12_final_20261012/audit/audit_chartqa_sensitivity_rows.jsonl
python docs/sr_f1/analysis/SR_F1_2_FINAL/write_report.py artifacts/sr_f12_final_20261012/evidence
```

先校验包，再复算；复算会更新分析文件，因此不要把复算后的文件拿原包清单误判为传输损坏。SVG包含绘图库生成元数据；科学数值以JSON和CSV为准。各服务器`audit_*.py`用于原始数据与完整checkpoint复验，只能在原服务器有权限的普通CPU分配内运行；不属于上述离线复算所需步骤，不应重新运行任何部署/释放/提交脚本。

## 服务器保留与明确省略

原实验根：`/projects/_ssd/varunssd/louis-ssvc/sr_f12_20261010`。

八条后两种子checkpoint目标：`/projects/_hdd/varunhdd/louis-ssvc/sr_f12_20261010_checkpoints/<model>`。具体映射、完整state/adapter/源及冻结哈希见 `audit_result.json`，原部署映射亦包含在包中。

没有把原始解释全文、全部训练rollout文本/token张量、模型权重、图像、全部大型checkpoint放入本地包。这些仍在服务器原位，原始组索引提供路径与hash；本地包足以复算已释放分数的统计结果，**不是自包含重训练包，也不是完整原始数据副本**。独立审计复用冻结评分器进行逐条重算；统计核验另行实现，不把二者混称为一套全新评分定义。

ChartQA尾换行问题属于正式评分口径的技术限制；正式分数、事后敏感性和所有不利结果并存。未重新采样、未覆盖冻结回执、未选择最好种子，也未新增PART或其他实验。
