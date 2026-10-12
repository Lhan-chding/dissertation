# SR-F2 既有证据审计入口与解释边界

`scripts/sr_f2/audit_prior.py` 是 CPU 只读工具，不加载推理/训练运行时、不提交作业、不修改历史发布成绩。每次执行须给新的输出目录，不能位于 SR-F1.2 原根、旧源码根或其登记的 HDD 检查点目录内部。不能以已有目录覆盖重跑；失败回执和部分 JSONL 保留。

按以下顺序执行，三个目录均位于新 SR-F2 根。`OLD` 是原 SR-F1.2 canonical root，`NEW` 是本轮独立根，Python 使用经过项目校验的环境。

```sh
python scripts/sr_f2/audit_prior.py --root "$OLD" --output "$NEW/prior_audit/inventory" --phase inventory
python scripts/sr_f2/audit_prior.py --root "$OLD" --output "$NEW/prior_audit/blind" --phase blind-classify
python scripts/sr_f2/audit_prior.py --root "$OLD" --output "$NEW/prior_audit/rescore" --phase rescore --blind-receipt "$NEW/prior_audit/blind"
```

每阶段写 `SUMMARY.json`、`SOURCE_MANIFEST.json`、`AUDIT.jsonl` 和简短 `README_zh.md`。`SUMMARY.status=PASS` 仅表示该阶段实现的检查成功；不能代表新实验完成、科学预测成立或所有未知身份均已解决。`INCOMPLETE_RESOURCES` 表示检查点缺失/损坏，`FAIL` 附原始异常栈。退出码分别为 0、2 或异常非零。源 SHA 清单区分原文件路径和解析后的真实路径；源文件完成后再核对大小、mtime、ctime、inode及链接目标，发现审计期间变更即失败。

资源阶段核对旧源清单及冻结文件、历史发布授权链、八个登记 HDD 目录映射。每条路径对 0/8/.../96 的13份 checkpoint commit 检查逻辑步、run identity字段哈希、完整文件 SHA-256，终点再对 `LATEST`、`COMPLETE` 和最终授权。总计12路径、156份检查点，约52GB字节顺序读取。这里不反序列化 pickle、不重新计算 tensor state hash；保存原始完整状态/字段哈希并以原提交的字节哈希绑定。恢复加载时仍需训练/评测引擎执行自己的完整状态校验。

训练原文按96步×16组×8条遍历，每组 identity、记录对象哈希及每步 raw-records 哈希均相符；1152步、147456条原 rollout全部核对。由原组件分数和认证的原语义合同重新计算每步原优势，必须逐项等于保存的审计日志。两组GATE异常固定为71002 step39 slot11和step77 slot4，保存原文/token、原分量分数、p_read分子/分母、float64 h、二元J优势、中心化u、scale、旧分支和原实际优势。中间量明确标记为由原组件只读派生；不使用修复后的GATE重写历史。

终点遍历全部13模型的已发布227700条，校验COMPLETE→panel PLAN/COMPLETE→group文件SHA→records对象SHA，以及逐条发布slot身份与唯一覆盖。重评分进一步逐条复算旧v1并与正式发布值精确相等，再并列保存v2。未解析回答、技术错误字段和负面结果不丢弃。ChartQA的v2仅两侧`strip()`后调用原评分器；无数值抽取、单位修复或近似答案搜索。

盲分类阶段不读任务金标和ChartQA金标；函数仅接收回答原文和协议，保存全部学生组合面板回答与分类，再以原`L_answer`标示旧不可解析子集。重评分要求先有同根成功盲分类回执，分类函数及其模块SHA须保持不变，原文SHA也须与盲分类相符。该阶段读取旧发布解析标记用于分层，不将A/J正确性或金标传入分类器。

A臂TOPK在TEST_ID证据协议上保存全量缺项键、额外键、重复键和选中/未选中类别模式。原文按三种子×缺项0/2/4×原A0/1固定分层，每层按原冻结遍历顺序取前3条（CLI可在1至20之间事先固定），不按是否符合机制解释选例。缺两项不自动算六值证书；E6/P6/J6由独立预登记评分器判断。

`audit_prior.py` 的通用重评分不自行推断指定 ChartQA 丢失题集合，故其 `requested_chartqa_cohort.status=UNRESOLVED`。后续独立工具 `verify_chartqa_cohort.py` 接收私有题号/规则声明，与原敏感性 sidecar 和原文逐项核验；成功回执才解决集合身份。候选丢失集合只供诊断，不替代指定集合。去空白敏感性选择属于事后诊断口径，不能回写历史正式分数。

`rescore/AUDIT.jsonl` 的主要记录为 `kind=endpoint_score`，字段包括原slot/model/pool/protocol/view/qid/root/family/chart/variant/draw、`source={path,sha256,row_index}`、`score_v1`、`score_v2`。额外记录为 `gate_original_group` 和 `chartqa_loss_candidate`，终点分析须排除这些额外记录后检查227700条。无原文选择标签的终点只保留元数据和分数，原文仍由来源路径/SHA可定位；所有候选与不可解析组合回答保留原文/token。
