# SR-F3-CE 外部盲类型审计与版本化定稿

这是独立的数据准备工具。它读取已部署的科学实现，不修改 `root/code`、训练设置、模型、题目、提示、图像选择、评分器版本或生成预算。辅助工具本身的 Git commit 和实际脚本 SHA256 单独进入审计回执。

## 验收边界

计划 §10.4 要求从问题、可见图或明确接口契约确定类型，并由两份独立标注或规则检查核对有歧义的题目，在读取模型输出前冻结。全量 `UNRESOLVED` 且 `reviews=[]` 仅是保守评分接口，不能算完成真实类型审计。

本流程使用两份独立的全量检查，便于明确覆盖。两份检查都必须是实际执行的规则或视觉审查，不能只复制一个检查结果再改 reviewer_id。真实检查后仍然 `UNRESOLVED` 完全允许；没有最低解析率，也不按金标、模型回答或历史错误选择应解析的题目。

输入只包含固定 qid、原问题、图像、图像 SHA256、作者需要的非答案 Paragraph 和明确接口契约。不得提供 `Answer`、`Year` 金标字段、历史模型原文或当前模型输出。作者原生 `Year` 字段不是可供盲注使用的类型证据。

## 类型与证据

| kind | 可接受的证据条件 | comparison |
|---|---|---|
| YEAR | 回答语义明确是一个日历年份，并有时间轴/标签或明确契约支持；时长不自动算年份 | EXACT |
| COUNT_INT | 回答明确是离散实体的整数个数；平均值、连续量或未知尺度不得硬推为整数 | EXACT |
| PERCENT_POINT | 问题和图/契约共同证明答案使用百分数值尺度，例如 2% 对应数值 2 | 事前明确的外部比较规则 |
| RATIO_FRACTION | 问题和图/契约明确要求一个比例标量，例如 2% 对应 0.02；a:b、多值比不得硬推 | 事前明确的外部比较规则 |
| GENERAL_NUMERIC | 明确为单一数值标量且单位、尺度、归一化方式无歧义；不能作为默认类型 | 事前明确的外部比较规则 |
| UNRESOLVED | 单位、日期、尺度、多值、文本或证据不足，或两个独立检查不一致 | null |

其余三种数值类型的 `comparison` 仅允许现有接口支持的 `EXACT` 或 `RELATIVE_5PCT`，且必须有事先确定的外部评分契约依据。不能根据金标或模型答案决定容差。5% 规则仅用于其原本的外部数值用途，不改变合成任务的精确有理数评分。证据不足时继续未解析；只确认 YEAR/COUNT_INT 的保守审核范围可以接受。

每份审核文件使用 JSON 包装：

```json
{
  "experiment_id": "SR-F3-CE-20261016",
  "reviewer_id": "independent-reviewer-A",
  "blind_inputs_only": true,
  "used_gold": false,
  "used_model_output": false,
  "saw_other_review": false,
  "reviews": [
    {
      "qid": "FROZEN_QID",
      "reviewer_id": "independent-reviewer-A",
      "input_identity": "external.blind_item_identity(blind_item) 的 SHA256",
      "method": "independent_contract_rule_check",
      "kind": "UNRESOLVED",
      "comparison": null,
      "evidence": "逐题说明核对了什么，以及为什么仍有单位/尺度/语义歧义。",
      "used_gold": false,
      "used_model_output": false
    }
  ]
}
```

`method` 也可为 `independent_visual_annotation`。两份文件各自恰好覆盖 ChartQA 2500 题和实际已冻结的 ChartQAPro n 题，不能以缺行代替未解析。`input_identity` 绑定 qid、问题、image_sha256、Paragraph、interface_contract，采用已部署的 `blind_item_identity`，不得自行改变字符串或空值规则。

每位审核者还需提供一个独立证据目录，包含实际规则脚本/规则原件、规则 hash、自检输入与输出、候选筛选记录、逐题视觉核查证据、原始输出等。工具会保留指定目录中全部普通文件及其 hash/bytes，拒绝 symlink。操作人仍须核对目录确实包含这些原始材料；“目录中所有文件已复制”不证明没有遗漏应提供的材料。

## 执行

只能在 `EXECUTION_FREEZE.json`、科学 `runs` 和任何 `sealed` 新测试目录产生之前定稿。原始 external/PANELS.json 必须是未审核、全未解析 v0，且尚未被 RUN_IDENTITY 绑定为不可变文件。

只读校验不需要 `--publish`，不写外部面板或审核回执：

```bash
PYTHONPATH="$RUN/code/src" "$PYTHON" finalize_external_types.py \
  --root "$RUN" --review-a A.json --review-b B.json \
  --review-a-evidence-dir A-evidence --review-b-evidence-dir B-evidence
```

完成独立检查、代码提交及审核后，使用已获授权的 CPU Slurm 作业进行发布，作业名称必须以 `-LH` 结尾：

```bash
PYTHONPATH="$RUN/code/src" "$PYTHON" finalize_external_types.py \
  --root "$RUN" --review-a A.json --review-b B.json \
  --review-a-evidence-dir A-evidence --review-b-evidence-dir B-evidence \
  --auxiliary-code-commit FULL_COMMITTED_GIT_SHA --publish
```

`--publish` 会调用原部署代码的 `prepare_chartqa(..., type_reviews=...)` 和 `freeze_chartqapro(..., type_reviews=...)`，产出 `external/chartqa_typed_v1` 与 `external/chartqapro_typed_v1`。它逐字节比较 v0/v1 的 MODEL_INPUTS、TASKS_AUDIT_ONLY、ChartQAPro SELECTION，逐图校验 SHA256；除了类型 registry hash 与未解析数量，其余 FREEZE 字段必须完全相同。

两份审核包装文件在首次读取时各自保留完整字节快照；解析、身份与盲性校验、合并、归档及回执 SHA256 都绑定该快照。后续步骤不重读源审核路径来替换归档内容，因此发布期间源路径变化不会让 registry 与归档审核分离。要改用不同审核内容，必须重新执行验证；已有不同内容的不可变归档不会被覆盖。

不可变原件先写入同目录临时文件并 `fsync`，然后通过不能覆盖目标的原子硬链接发布，再同步父目录。写入中断不会留下占据最终名称的半文件；发布后目录同步中断可以使用相同内容重试。相同完整内容的重复发布允许继续，任何不同内容的覆盖请求（包括并发冲突）会被拒绝。异常退出可能留下未发布的临时文件，它们不作为审核证据或完成标志，也不阻止安全重试。

两份类型/比较规则一致才解析；任何一份未解析或双方分歧，都保留未解析及两份原始证据，不借助金标仲裁。源数据构建函数在 registry 冻结后读取金标，仅用于保持原审计 sidecar，不参与上述规则选择。

## 输出与后续门禁

固定回执目录是 `external/revisions/typed_v1/`：

- `PANELS.v0.json`：原面板索引的精确字节；旧面板目录仍保留。
- `v0/CHARTQA|CHARTQAPRO/{FREEZE,TYPE_REGISTRY,SELECTION}.json`：存在的旧回执原件。
- `REVIEW_A.json`、`REVIEW_B.json`：两位审核者原始包装文件。
- `reviewer_evidence/A|B/` 和 `A_MANIFEST.json`、`B_MANIFEST.json`：规则、自检、视觉等证据及逐文件身份。
- `BLIND_SCOPE.json`、`RECONCILIATION.json`：全量输入身份与一致/分歧/共同未解析结果。
- `TYPE_AUDIT_RECEIPT.json`：审核范围、两位审核者、源码与辅助工具身份、所有输入不变校验。
- `PANELS.v1.json`：经过校验的新面板索引，与新 `external/PANELS.json` 完全一致。
- `PUBLICATION.json`：版本化发布终态与前后索引、审核回执的 SHA256。

只有以上文件与全部被引用材料一致时，调度器才能把类型审计视为完成并继续 `freeze_science`。原始科学 source commit 保持不变。现存未解析项、外部 UNAVAILABLE 状态与全部不利结果继续保留。

## 测试边界

独立测试文件支持 `SRF3_TYPED_FINALIZER=/path/to/typed_finalize.py` 指定辅助工具。测试使用人工构造的盲输入，不生成模型回答、不读取科学金标、不提交作业。覆盖盲性/身份/完整分母拒绝、年份容差拒绝、未解析与分歧保留、可见图 hash、原始证据复制及变更/越界拒绝、默认入口无文件修改。合成发布回归还覆盖审核源路径在验证后变化时的精确字节与合并一致性、实际部分写入中断后的安全重试、并发发布不可覆盖、目录同步中断与相同输入重复发布。真实服务器发布必须另有实际 `PUBLICATION.json` 和下游只读复核；单元测试通过不等于已发布。
