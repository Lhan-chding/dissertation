# MM-CORE：审计与非零梯度恢复汇报包

日期：2026-10-09。模型 **Qwen/Qwen3.5-9B**。最多五张Pro6000并发，GPU小时只记账、无累计门槛。
本轮工作已完成，**MM-DEV没有开始**。请独立审阅证据并确定下一阶段尚未冻结的配方；不要把工程PASS当作科学收益。

## 建议阅读顺序

1. [给GPT Pro的审阅请求](documents/mm_core_f1/dev_review/PRO_REVIEW_REQUEST_zh.md)：可以直接作为用户提示词使用。
2. [非零梯度恢复最终报告](documents/mm_core_f1/NONZERO_RECOVERY_RESULT_20261009_zh.md)：新完成工程验证的结果、成本和适用边界。
3. [原公共起点测量结论](audit/report/MEASUREMENT_FINDINGS_zh.md)与[完整量化报告](audit/report/SCIENTIFIC_RESULTS_zh.md)。
4. [下一阶段参数出处与待决表](documents/mm_core_f1/dev_review/NEXT_STAGE_DECISION_TABLE_zh.md)：区分原文已固定、仅工程使用、仍待决。
5. [当前DEV草案v2](documents/mm_core_f1/dev_review/DEV_FREEZE_PROPOSAL.pro_review_v2_zh.md)与[JSON](documents/mm_core_f1/dev_review/DEV_FREEZE_PROPOSAL.pro_review_v2.json)。
6. [原始执行计划](documents/mm_core_f1/design/CODEX_FINAL_PLAN_MM_AUDIT_ENGINE_zh.md)、[用户覆盖条件](documents/mm_core_f1/USER_AMENDMENT_20261008_zh.md)、[新增工程授权](documents/mm_core_f1/NONZERO_RECOVERY_AUTHORIZATION_20261009_zh.md)。

## 当前事实

| 项目 | 已得到的事实 | 不能据此声称 |
|---|---|---|
| 格式 | 两个面板各384/384通过；SFT桥接未触发 | 格式通过等于能力正确 |
| 正式测量 | 384题×4回答=1536条；读数正确1509，答案正确1465，联合正确1451 | 1536个独立训练状态；训练收益 |
| 支持限制 | 001格为0；S仅27条有定义、6/16根有支持；13/384题在K4中见非恒定答案奖励 | 001总体不可能；missing可补0；奖励诊断是未来收益 |
| 原答案奖励GRPO ENGINE | 连续/恢复比较通过，但16组零奖励对比、8次梯度为0 | 已验证非零GRPO恢复 |
| 新非零SFT工程验证 | 8次实际非零更新，连续4与新进程2+2精确一致，独立checkpoint复核通过 | 非零GRPO学习效果、长期稳定性、能力提升 |
| 下一阶段 | 原文固定32/32、4准备/5起点/30响应框架；完整科学配方仍开放 | 1088次逻辑更新已执行或已自动获准 |

新验证只用固定ENGINE_TEST输入和明确披露的gold监督，工程adapter不替换公共起点。
原始结果与补充工程结果分别存放，不改原审计的评分、奖励或冻结材料。

## 成本与技术事件

新作业193946：COMPLETED、0:0，单卡339秒=0.094166666667 GPU小时。
含原审计合计2.018333333333 GPU小时，生成尝试2441、物理更新16、额外前向4881。
GPU小时不是限制或门槛。wrapper墙钟与实际Slurm分配时分别保存。
旧抢占、唯一明确技术修复、一条未保存的生成尝试和额外前向收费记录全部保留；
“保存的failed回答为0”不能读成“没有技术事件”。本次新增验证没有重跑。

## 数据与证据索引

| 目录/文件 | 内容 |
|---|---|
| `audit/raw/` | 原格式/正式测量原始回答、token、请求与字段评分旁证 |
| `audit/scoring/` | 三面板逐回答评分JSONL，保留原评分版本 |
| `audit/tables/`、`audit/report/` | 全部聚合表、分母、按根区间、上下文/奖励诊断与原报告 |
| `audit/data/` | 原始及processor图像、问题侧表、split/根家系与生成器记录 |
| `audit/engineering/`、`audit/accounting/`、`audit/manifests/` | 原ENGINE轨迹、技术失败、成本和冻结身份 |
| `audit/code_863cac0/src/mm_core/` | 原11个冻结运行模块，独立保存，不能用新增模块目录替代旧source-root |
| `nonzero/engineering/nonzero/` | 两条新轨迹、真实采样、每步更新、checkpoint身份清单与三段进程回执 |
| `nonzero/report/INDEPENDENT_NONZERO_VERIFICATION.json` | 独立CPU逐元素核验的完整报告，含实际scheduler原始回执 |
| `nonzero/report/independent_nonzero_verification.py` | 本次独立核验的代码；重新运行需要服务器上的checkpoint和实际运行环境 |
| `nonzero/manifests/`、`nonzero/accounting/` | 新冻结、实际单卡分配、物理成本和提交日志 |
| `nonzero/data/` | 与原工程池一致的固定输入副本；实际使用范围由新freeze限定8题 |
| `src/mm_core/`、`scripts/mm_core/`、`tests/mm_core/` | 可审阅的运行、报告、打包工具与测试；runtime/package工具提交65aad213 |
| `provenance/` | 服务器/本地传输校验、旧冻结清单覆盖与CPU检查记录 |

原最终审计清单863项中，827项在本地逐SHA匹配；其余36项是明确未装入的checkpoint或锁文件，
没有未声明的数据缺口。最初导出目录未覆盖的3份scoring JSONL已补齐并核对原快照哈希。
新增运行698个导出文件及导出清单也逐项匹配服务器哈希。

**不含模型大权重、7份工程checkpoint张量正文、安装环境二进制和临时锁/cache。**
具体列表见[排除清单](PACKAGE_EXCLUSIONS.json)。这是一份审阅证据包，不是包含权重的完整重训环境。
独立checkpoint复核在服务器上实际执行；ZIP验证只验证包内文件完整性，不能在缺少张量时重做该比较。
原始文档/原报告/v1草案保留历史措辞；新增授权和v2报告明确区分后来完成的SFT验证。

## 完整性核验

解压后，可用标准库运行：

```bash
python -I verify_package.py /path/to/MM_CORE_PRO_REVIEW_20261009.zip
```

`MANIFEST.json`记录文件大小和SHA-256，明确不hash自身及SHA256SUMS以避免循环；
`SHA256SUMS.txt`覆盖MANIFEST和其余文件，仅不hash自身。
验证器检查ZIP CRC/读到EOF、安全唯一相对路径、文件集合与逐文件哈希。VERIFIED不是科学结论。

请按审阅请求返回明确的下一阶段配方、依据、仍开放项及是否需要额外工程验证。
保持DEV草案`authorized=false`，由用户拿到审阅意见后另行决定执行。
