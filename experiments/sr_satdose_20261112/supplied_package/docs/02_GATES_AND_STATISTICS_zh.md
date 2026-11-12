# 门禁、文件schema与统计接口

## 1. 两层冻结，不制造循环哈希

### `DESIGN_FREEZE.json`：模型调用前

最少字段：

```json
{
  "schema": "sr-satdose-design-freeze-v1",
  "experiment_id": "SR-SATDOSE-1-20261112",
  "config_sha256": "真实SHA",
  "artifacts": {
    "calibration_groups": {"path": "frozen/calibration_groups.jsonl", "sha256": "真实SHA"},
    "private_cases": {"path": "frozen/private_cases.jsonl", "sha256": "真实SHA"}
  }
}
```

实际还必须绑定training_slots、所有TRAIN/DEV/PREFIX_CALIB/MAIN manifests、public inputs、图片、源代码与oracle测试报告。记录git commit及冻结时间；第一次模型调用ledger引用此SHA。以上简写不是允许省略其他项目。

### `ADMISSION_INPUTS.json`：干跑后，只汇总证据

```json
{
  "schema": "sr-satdose-registration-v1",
  "mode": "production",
  "experiment_id": "SR-SATDOSE-1-20261112",
  "config_sha256": "真实SHA",
  "artifacts": {
    "design_freeze": {"path": "frozen/DESIGN_FREEZE.json", "sha256": "真实SHA"},
    "calibration_groups": {"path": "frozen/calibration_groups.jsonl", "sha256": "与事前相同的SHA"},
    "private_cases": {"path": "frozen/private_cases.jsonl", "sha256": "与事前相同的SHA"},
    "dry_raw": {"path": "dry/DRY_RAW.jsonl", "sha256": "真实生成文件SHA"}
  }
}
```

GPU报告通过独立CLI参数传入，报告绑定ADMISSION_INPUTS的SHA；ADMISSION_INPUTS不得反向包含GPU报告，否则形成循环。所有证据path相对于`--evidence-root`，禁止绝对路径和`..`，逐文件核对SHA。

## 2. calibration_groups.jsonl

160行：每起点80组。必需字段：

- `group_id`：唯一；`start_seed`：11031或11032；`kind`：CONFLICT/ALIGNED。
- `root_id`,`structure`,`case_id`；private task同时具有root/structure/world/query/gold/boundary_indices/strictly_better_indices/r。
- `request_ids`：8个唯一ID；`draw_seeds`：8个不同种子。

两个起点可共享同一80个校准root；每个起点内部不得重复。每个细分结构×方向×aggregate cell为4个冲突root、1个一致root。私有case的audit必须有真实`list_order`，函数从它重新推导冲突，不能只信一个布尔标记。

## 3. DRY_RAW.jsonl

1,280行完整原始生成，最少字段：

- `request_id,group_id,case_id,seed`；与事前manifest一致。
- `policy_sha256,reference_sha256`；必须匹配配置中的R24/原parent。
- `generation_status="COMPLETE"`、`technical_validation_errors=[]`。
- `assistant_prefill="{"`、`suffix_text`、`full_text`，三者严格拼接一致。
- `finish_reason`：balanced/eos/length；未知终止为技术HOLD。
- `native_score`：必须含真实J及完整native诊断，不是自写奖励。

另存原token、sampler logprob与native capture回执；其索引与hash在GPU验证证据中绑定。门禁工具会按训练语义重新执行native J：finished仅balanced为真。对于评估，原字段局部评分将balanced/eos都视为完成；**两种历史语义必须分别记录，不要悄悄统一改写**。

门禁的c使用该训练语义下的native可测selected，并独立与任务几何授权集合比对。另导出field-local c作敏感性，检查EOS/协议差异是否影响判断。正式主终点按冻结field-local S_hit。

## 4. GPU_VERIFICATION.json

见 `templates/GPU_VERIFICATION.template.json`，初始全PENDING/false。每项`checks.<key>`为：

```json
{"passed": false, "evidence": {"path": "真实证据相对路径", "sha256": "真实SHA"}}
```

必须覆盖模型/reference、分割与泄漏检查、原生夹具、生产loss重放、梯度hook、sampler/teacher、tokens、backend、完整长batch、零优势保留KL、影子Adam重放、评估loader、干跑不变和prefix校准。

每个起点的gradient对象需包含：
`policy_before_sha256,policy_after_sha256,J_vs_source_pg_cosine,J_split_half_cosine,source_split_half_cosine,heldout_effect_half_1,heldout_effect_half_2,effect_repeat_error_floor,evidence`。

所有数值真实、有限，余弦必须在[-1,1]。两个heldout effect来自预定两个witness半集，不得重分半集直到为正。`dry_optimizer_steps=0,dry_parameter_updates=0`；影子optimizer调用单列最多4次。

检查器核对结构、文件SHA、原生评分与数值，但**一个自述JSON不构成执行证明**。Codex还需独立只读重放GPU证据，核对日志/tensor摘要/源SHA；用测试夹具或模拟数值填写生产报告禁止。

## 5. 运行门禁命令（输入文件必须已经真实产生）

```bash
python tools/check_admission.py \
  --config config/experiment.json \
  --registration /new/run/ADMISSION_INPUTS.json \
  --evidence-root /new/run \
  --native-src /verified/legacy/native_src \
  --gpu-report /new/run/GPU_VERIFICATION.json \
  --out /new/run/ADMISSION.json
```

退出码0也只放行四路径各step1–2；退出码2为HOLD。遗漏证据、假production、混入旧checkpoint、hash变化、奖励重放不符、missing/duplicate请求都会阻止放行。

新训练driver须在前两步后独立校验真实`CANARY_RESUME_PASS`才放行剩余预算。本包没有服务器driver，Codex必须实现并给出测试记录，不能直接让旧CLI绕过新门禁。

## 6. 统计输入接口

`tools/analyze_primary.py` 输入：128root × 2start =256行，字段为：

```json
{
 "root_id":"真实root",
 "structure":"2_choose_1",
 "start_seed":11031,
 "draws":2,
 "start_correct":1,
 "T0_correct":0,
 "T1_correct":2
}
```

三个numerator均为该root的list8/conflict两个draw中exact S_hit的整数和。必须先与MAIN manifest、raw、逐回答评分对账，不能用此表把缺失记录“补零”。每个结构32root，两起点完全配对。

```bash
python tools/analyze_primary.py --rows root_contrasts.jsonl --out PRIMARY_ANALYSIS.json
```

程序输出唯一主差、条件root-bootstrap区间、五类互斥解释、每起点差/绝对增益和假设性run噪声敏感性。它不负责生成结论之外的自动后续实验。

A/J/证书/保持、authority、prefix、swap和联合行为表由新分析wrapper逐项报告，不把这些结果合成一个可任意调权重的“成功总分”。

## 7. Prefix的缺失界与authority参数

对于某view严格成功c、unknown u、分母N，真实可观测性不足时至少报告 `[c/N,(c+u)/N]`；比较两view时用两界相减得到保守差界。限定补括号结果是另一个明确测量假设下的诊断，不改变严格结果。

authority的0/1编码：`alpha = logit P(chart | list-rule)`，`gamma = logit P(chart | chart-rule) - alpha`。居中编码另报`alpha_mid`。概率0/1时logit不有限，使用预登记的惩罚估计或只报概率差；不能填极大数再伪装可估计CI。

小样本的主保护性指标是两规则下完整来源分布、授权集合正确率、`P(chart|list-rule)`是否上升。gamma变化为次要描述，不能用不显著来证明能力保持。
