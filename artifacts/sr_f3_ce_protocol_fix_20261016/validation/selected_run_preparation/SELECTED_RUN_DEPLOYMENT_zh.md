# 选定协议新根接续清单（准备件，尚未选 v2 或 C2）

本件只读核对了冻结科学提交 `652e52d530520a276b6420ff0f4ee479dcde6db1` 与本地既有证据。它不是协议选择、服务器执行回执或科学放行。只有独立消融根的九个固定任务均有完整原文、COMPLETE、真实 `COMPLETED/0:0` 终态，且 `ablation.verify_selection` 重算通过，才能执行下文的选中分支。不能凭部分 P2/P3 结果、早期正确率或当前队列状态挑选新根。

本目录的 `EXTERNAL_READONLY_VALIDATION.json` 已核验：2,687 个冻结 source archive 成员、1,785 个本地外部输入文件、2,756 个原盲审问题、1,509 个 ChartQA 独立图像及 256 个 ChartQAPro 图像的字节/像素身份和历史排除。原 4146 `annotation_set` 接受原 REVIEW_A/B，重新合并仍为 85 RESOLVED、2,671 UNRESOLVED。这是既有材料核验，不是新的人工作业或新根发布。

## 1. 固定身份和禁止复用的输出

```sh
export SCIENTIFIC_COMMIT=652e52d530520a276b6420ff0f4ee479dcde6db1
export OPS_COMMIT=652e52d530520a276b6420ff0f4ee479dcde6db1
export EXT_COMMIT=652e52d530520a276b6420ff0f4ee479dcde6db1
export TYPED_COMMIT=4146e156ffa9a476b360daeb67503703faca138f
export ABL=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation
export OLD=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_20261016
export PARENT=/projects/_hdd/varunhdd/louis-ssvc/sr_f2_20261013_r3
export PYTHON=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1
```

`OPS_COMMIT` 和 `EXT_COMMIT` 选 652，是因为这次所需 ops/observer 已在该提交冻结；不要将当前 HEAD 的整树替代 source archive。755 提交的独立消融接续器及其文档不属于 652 科学 archive。任何换用另一已提交 ops commit 都须重新逐文件验证并显式登记，不能改科学 code manifest 的 commit。

新根绝不能预复制 `RUN_IDENTITY.json`、`orchestration/`、`tasks/`、`engine/`、`monitor/`、`accounting/`、`runs/`、`sealed/`、`EXECUTION_FREEZE.json`、`STOP`，也不能复制旧 native masks/PROCESSOR_RECEIPT/SUPPORT_MASK_RECEIPT 作为新证据。原 v1 的 2,880 ENGINE、b1 MONITOR 512、恢复新生成 512，共 3,904 条原文仅保留为历史；2,304 条提示消融另存，不进入新 ENGINE/MONITOR/recovery 分母。

本清单所有服务器动作都应由主执行者实施；本准备任务未运行其中任何一步。

## 2. 选择门禁先于新根创建

从已认证的消融 `code/src` 调用只读 verifier，核实九任务结果和保存的 Slurm 终态，再取 selected_revision。先核消融 source manifest 与实际完整 regular-file 集合，禁止 `__pycache__` 或额外源码；不要使用当前 checkout 代替已部署代码。

```sh
REV=$(PYTHONPATH="$ABL/code/src" "$PYTHON" -B - "$ABL" <<'PY'
import sys
from pathlib import Path
from sr_f3_ce.ablation import verify_selection
root = Path(sys.argv[1]).resolve(strict=True)
v = verify_selection(root)
assert v['status'] == 'PROTOCOL_SELECTED'
assert v['selected_revision'] in {'v2', 'c2'}
assert v['outputs'] == 2304 and len(v['raw_evidence']) == 9
assert v['decision_uses_only'] == 'value_parseable'
print(v['selected_revision'])
PY
)
export REV
export NEW="/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_$REV"
```

若不存在选择文件、状态不是 PROTOCOL_SELECTED 或 verifier 报错，停止此清单；不建立猜测性的 selected_v2/c2，不调用 `select_counts` 自行兜底。确认 NEW 尚不存在、OLD/ABL/NEW canonical 路径互不包含后，以排他方式创建 NEW；同名既有目录必须先读其状态，不能清空后重跑。

候选目录以 `ABLATION_MANIFEST.json`/`PROTOCOL_SELECTION.candidate_identities` 为准，不凭目录名猜来源。复制该候选完整 `plan/`，逐项核 PACKAGE 及 candidate 的 package/config/model_inputs 三个 hash。候选生成时的状态与所有原记录必须原样保留（若其他候选回执有 `selected_for_science:false`，也不能改写）；实际采用记在顶层选择、RUN_IDENTITY、EXECUTION_FREEZE，不能修改候选字节。

## 3. 复制 allowlist 与重建边界

| 内容 | 操作 | 精确约束 |
|---|---|---|
| `code/` | 只展开原 `deployment/SOURCE_CODE.tar` 的 code/ 成员 | 2,687 regular files，成员集合和每项 bytes/SHA 与 SOURCE_MANIFEST 完全相等；无额外 pyc、so、pth 或软链 |
| `deployment/SOURCE_MANIFEST.json` | 原字节复制 | SHA `dae3a506dda4fffee8da0e25a3e0881f4b45fa3259342c4160ecd0c09e03c9e0`，commit 652 |
| `authorization/REQUEST.md` | 原字节复制 | SHA `83f3688707b09d69d84fb4f7ce3a9a7f2706b26c978176d97a6fb9637189c220` |
| 顶层 `PROTOCOL_SELECTION.json` | 从 ABL 原字节复制 | 原根仍须完整可读，每次 selected guard 都在原根重验 |
| `plan/` | 复制所选候选完整文件集合 | 不重建已测试候选，不更改 adopted 标记 |
| `data/`、`parents/`、`prior/` | 见相邻 `selected_data_preparation/` | 严格使用其 COPY_ALLOWLIST 和命令；新 data audit、parent audit、RAW_EXPORT 必须重建 |
| `external/` 及原审核 | 本目录 `EXTERNAL_COPY_ALLOWLIST.json` | 共 1,895 条精确目标记录；其中 parquet 一项仅有已固定服务器源身份，未本地读取；旧 ChartQA FREEZE 只去 provenance |
| `native/` | 对所选 revision 重新 CPU 审计 | 816 masks、PROCESSOR/SUPPORT receipts，不能复用旧 v1 或消融 v1 receipts |
| `NATIVE_AUDIT_SOURCE.json` | 复制 652 SOURCE_MANIFEST 原件 | 用于 P0 复核 inherited native source，与 source manifest 同 SHA |
| 旧全包 | 只复制原 `.tar.xz` 及 `.verification.json` 到 `protocol_revision/history_v1/` | 219,281,700 bytes；SHA `d67af2a3a00f44713c3bcb62d526f90f12e392f5fa0fe2d1872f54a3e1184d60` |
| 消融原件 | 按 selection 中记录逐项复制到 `protocol_revision/ablation/` | manifest、9 COMPLETE、9 terminal、9 allocation、576 raw groups、1,152 REQUEST/pending 原件，再加原 selection；以实际 receipt 路径/bytes/hash 为准 |

复制器必须拒绝绝对 relative_path、`..`、逃出根的软链及重复目标，目标存在时只接受字节完全相同。每项复制前后验 SHA/bytes，写本次 `deployment/INPUT_COPY_RECEIPT.json`，记录原始绝对来源及实际新目标；不能把旧复制回执改 root 后称为新执行。

外部 allowlist 的 `source_root_relative` 是 OLD 下服务器源；`source` 是已验证本地原件的 repository-relative 定位。reviewer evidence 中旧 pyc 是历史原件，应留在独立 provenance 原目录中，绝不加入 PYTHONPATH 或科学 code 树、也不执行旧 reviewer 脚本。原 HISTORY_IMAGES 保留旧 source_root 原文，不伪造本次重新人工审核。

完整消融镜像的记录展开式：

```python
records = [selection['ablation_manifest']]
for cell in selection['raw_evidence']:
    records += [cell['complete'], cell['terminal'], cell['allocation']]
    records += cell['raw_groups'] + cell['raw_artifacts']
# 所有 record.path 均以 ABL 为源根、NEW/protocol_revision/ablation 为目标根。
# 每项只复制精确路径，bytes/sha256 逐项核验；最后复制 selection 原字节。
```

不要把 ablation 原根删除或仅留下这个镜像：selected guard 会对 identity.ablation_root 再运行 verify_selection，副本不是其替代物。原 native protocol_contexts/{v1,v2,c2} 只属于消融证明，可连同消融 native/source/authorization 作为完整 provenance 保留，不能作为新 native/ 的活动审计。

## 4. 新身份之前的 CPU 初始化

这部分需真实 CPU Slurm allocation，不加载模型权重、不采样。在提交前保留全用户 teacher 队列和各相关 ledger 快照；physical GPU 预算计 RUNNING、PENDING、UNKNOWN，teacher MaxSubmit=5 同时包含 CPU job。无法确定的提交/终态占用不能当零。用已验证的 `rose / soujanya-poria-startfund-2026-03 / cluster02 / cpu-2`，作业名以 `-LH` 结尾，`--hold --no-requeue`，先持久化意图、脚本、sbatch stdout/stderr、实际 job_id，再逐字段验证 allocation，最后 release。结果不确定时查现有 job_id，不重交。完成必须 squeue 消失且 sacct 真正 `COMPLETED|0:0`；日志中的 PASS 不替代终态。

在此 allocation 内按顺序执行：

1. data-agent 清单的题目/世界/渲染等价核验并重写 `data/FROZEN_DATA_AUDIT.json`；只复用已校验的 4,096 PNG，避免 Linux 重渲染改变原 macOS 字体像素。
2. 重跑 parent audit，然后 export_prior；不得先复制旧 `parents/PARENT_AUDIT.json` 或 `prior/RAW_EXPORT.json` 到活动目标，因为 write_once 会阻止新绑定。
3. 重建活动外部 v0（下一节）；验证原盲审原件的输入身份；生成 `protocol_revision/EXTERNAL_INPUT_REUSE.json`。
4. `audit_native.py --root NEW` 重做所选 revision 的 processor/masks；此脚本不要求已封存 RUN_IDENTITY，可先完成审计再绑定身份。

```sh
"$PYTHON" -B "$NEW/code/scripts/sr_f3_ce/audit_parents.py" \
  --parent-root "$PARENT" --plan-dir "$NEW/plan" --output-root "$NEW"
"$PYTHON" -B "$NEW/code/scripts/sr_f3_ce/export_prior.py" \
  --parent-root "$PARENT" --output-root "$NEW"
"$PYTHON" -B "$NEW/code/scripts/sr_f3_ce/audit_native.py" --root "$NEW"
```

上述第三条的顺序可在外部 v0 前后，前提是 data/parents/prior 已就绪。PROCESSOR/SUPPORT receipt 须记录选中的 revision、indent_2 serializer、实际 prefill，模型调用/权重加载/CUDA 初始化均为零；816 masks 全部逐 hash 校验，PREFIX/EXEC 目标及 answer mask 完全一致。不可对 selected plan 调 `audit_protocol_contexts`：该函数的基线输入是消融 v1 根，selected native 只需 `audit_native_inputs`。

`deployment/LOCAL_CHECKS.json`、LOCAL_VALIDATION、LOCAL_TEST_OUTPUT 可保留实际 652 本地 687 测试结果及时间/环境原件，因代码 archive 精确相同；不能改成“本次服务器/GPU验证”。新增本次 source/copy/CPU command/terminal 回执，保留真实执行过程。父审计和模型大文件 hash 在 CPU allocation 内做。

## 5. 外部 v0 重建与 4146 原盲审复用

旧 ChartQA FREEZE 的 image_root 是 OLD/external/legacy_images。4146 重新生成 typed_v1 时会把它设为 NEW/external/legacy_images，并严格比较除 type_registry_sha256/typed_unresolved_count 外所有 FREEZE 字段。故不能把旧 ChartQA FREEZE 原件放在 NEW/external/chartqa/FREEZE.json。

在前述 CPU allocation、历史输入复制完成后运行以下已有 API；该代码块不修改科学源、不下载数据、不调用模型或 scorer：

```sh
PYTHONPATH="$NEW/code/src" "$PYTHON" -B - "$NEW" <<'PY'
import json, sys
from pathlib import Path
from sr_f3_ce import external
from sr_f3_ce.evaluation import sha_file
from sr_f12.evaluation import durable_json
r = Path(sys.argv[1]).resolve(strict=True)
history = json.loads((r/'prior/HISTORICAL_IMAGES.json').read_text())
s = r/'external/chartqapro_sources'
qa = external.prepare_chartqa(r/'prior/raw/external/chartqa/TEST_MANIFEST.jsonl',
    r/'external/legacy_images', r/'external/chartqa', history_receipt=history)
pro = external.freeze_chartqapro(s/'test-00000-of-00001.parquet', r/'external/chartqapro',
    history_receipt=history, scorer_source=s/'evaluate_predictions.py',
    prompt_source=s/'vlmeval_chartqapro_utils.py', anls_wheel=s/'anls-0.0.2-py3-none-any.whl')
assert qa['status'] == pro['status'] == 'FROZEN'
assert qa['image_count'] == 2500 and pro['image_count'] == 256
old = r/'protocol_revision/external_v1_original'
for pool in ('chartqa','chartqapro'):
    names = ['MODEL_INPUTS.jsonl','TASKS_AUDIT_ONLY.jsonl','TYPE_REGISTRY.json']
    if pool == 'chartqapro': names += ['SELECTION.json','FREEZE.json']
    for name in names:
        assert sha_file(old/pool/name) == sha_file(r/'external'/pool/name), (pool,name)
old_qa = json.loads((old/'chartqa/FREEZE.json').read_text())
assert {k:v for k,v in old_qa.items() if k != 'image_root'} == {k:v for k,v in qa.items() if k != 'image_root'}
assert qa['image_root'] == str(r/'external/legacy_images')
panels = dict(experiment_id='SR-F3-CE-20261016', status='FROZEN',
    CHARTQA=dict(root='external/chartqa', status='FROZEN', question_count=2500,
        freeze_sha256=sha_file(r/'external/chartqa/FREEZE.json')),
    CHARTQAPRO=dict(root='external/chartqapro', status='FROZEN', image_count=256,
        reason=None, freeze_sha256=sha_file(r/'external/chartqapro/FREEZE.json')))
durable_json(r/'external/PANELS.json', panels)
# 此处应继续记录本次路径迁移/完整可见输入与图像 hash、pixel/history 排除验证到
# protocol_revision/EXTERNAL_INPUT_REUSE.json；不能把本地准备验证冒充本次新根验证。
PY
```

ChartQAPro parquet 为 193,053,989 bytes，SHA `feb03a5579e49114b45350b75aed7f674e285f795a425c83e22f400095c91489`，未在本地旧输入快照内；只能从已固定服务器资产按 hash 复制，先确认存在。其余官方源码/许可证按 EXTERNAL_COPY_ALLOWLIST 原字节保留。任何预期外差异停止，不把差异加入忽略字段。

本次 `EXTERNAL_INPUT_REUSE.json` 在 RUN_IDENTITY 前写，包含旧/新 panel hashes、ChartQA 唯一允许变化 image_root 的旧新值、相同 2,756 qid/model_input/task/selection/registry、1,765 图像 byte/pixel 和历史排除、原 review SHA、selection SHA；不要包含尚未生成的 RUN_IDENTITY SHA，以免循环。

## 6. 一次封存 RUN_IDENTITY

CPU 初始化阶段完成后，按 `RUN_IDENTITY.template.json` 用真实 hash 一次写入身份。最低 revised guard 字段为 experiment_id、root、code_commit、package_manifest_sha256、dirty_diff_sha256、run_stage、protocol_revision、protocol_revision_request_path、protocol_selection_path、ablation_root、bound_files。root/ablation_root 都用 canonical absolute path。

bound_files 逐文件值为文件 bytes 的 SHA256，不是 JSON object hash。至少覆盖授权 REQUEST、选择原件、SOURCE_MANIFEST；本次执行还应绑定完整稳定初始化回执，详细见 `BOUND_FILES_POLICY.json` 及 data-agent guidance。plan 已由 PACKAGE 间接逐文件绑定；仍显式记录 PROTOCOL_REVISION 与关键 config/model_inputs 的 hash 便于审查。

不要将活动 `external/PANELS.json` 加入 bound_files：4146 在后续发布时要切换到 typed_v1。不要绑定尚未存在的 typed receipts、orchestration CONFIG、observability manifest、P0/ENGINE outputs，后者都会引用 RUN_IDENTITY，形成循环或后续修改冲突。原盲审 provenance 可在此时绑定。RUN_IDENTITY 发布之后不再修改，包括 typed 发布之后也不增补 bound_files；由 EXECUTION_FREEZE 绑定新 typed outputs 和 extension。

seal 与 typed publish 可在同一真实 CPU allocation 串行完成。此时只绑定已经写完的 allocation/input 回执，不伪造本 job 的终态，也不绑定仍在写的日志。整个初始化 job 的终态由外部 selected_init ledger 保存并经主执行者核验后，才能提交实际 P0。

随后用新根自身源码运行：

```sh
PYTHONPATH="$NEW/code/src" "$PYTHON" -B - "$NEW" <<'PY'
import json, sys
from pathlib import Path
from sr_f3_ce.protocol import verify_run, verify_run_source_revision
r=Path(sys.argv[1]).resolve(strict=True)
verify_run(r)
print(json.dumps(verify_run_source_revision(r),sort_keys=True))
PY
```

完整返回对象作为后续 `protocol_source_guard` 原值，不能只保留 revision。固定键为 code_commit/protocol_revision/protocol_selection_sha256/request_sha256/ablation_root。

## 7. 原 4146 finalizer 在本次运行中重新发布

将 **实际 4146 文件** 安装到 NEW/operations/4146.../finalize_external_types.py；SHA 必须为 `1a75df6bf5fa1c0ef22a23ee7fd766a8e22f5a1c6c1fe7d579c4ab6135106aaf`。不能使用修改过的同名脚本或改变 auxiliary commit 声明。原 REVIEW_A/B 与完整原 reviewer_evidence 已在 `protocol_revision/type_review_v1/`，保持字节不变。

先运行同一命令不带 `--publish`，要求 VALIDATED_NOT_PUBLISHED 与 85/2671 汇总；随后在真实 CPU allocation、名字 -LH 的一次任务中运行：

```sh
"$PYTHON" -B "$NEW/operations/$TYPED_COMMIT/finalize_external_types.py" \
  --root "$NEW" \
  --review-a "$NEW/protocol_revision/type_review_v1/REVIEW_A.json" \
  --review-b "$NEW/protocol_revision/type_review_v1/REVIEW_B.json" \
  --review-a-evidence-dir "$NEW/protocol_revision/type_review_v1/reviewer_evidence/A" \
  --review-b-evidence-dir "$NEW/protocol_revision/type_review_v1/reviewer_evidence/B" \
  --auxiliary-code-commit "$TYPED_COMMIT" --publish
```

原 review JSON 按每题可见输入 identity 绑定，不含必须改写的新 run 字段；本地原函数已全量验过。新的 TYPE_AUDIT_RECEIPT 必须绑定 **本次** RUN_IDENTITY SHA 与 scientific commit 652；PUBLICATION/live PANELS/typed FREEZE 的 hash 链重新生成。输出仍为 ChartQA 62/2438、Pro 23/233，不得为提高覆盖率补写审核判断。生成 `external/TYPE_REVIEW_REUSE_RECEIPT.json`（本目录模板）明确：这是原两份独立判断的等价输入重发布，`new_human_review:false`，原审核时间/方法/不足/UNRESOLVED 全保留；receipt 指向原审计、本次输入等价证明、本次 RUN_IDENTITY、本次 TYPE_AUDIT/PUBLICATION、CPU allocation，模型调用为零；仍在运行的初始化 job 终态必须明确尚未在本 receipt 证明，主执行者在外部 ledger 核真实 COMPLETED/0:0 后才放行 P0。此新 receipt 由最终 immutable external freeze 绑定，不能追补修改 RUN_IDENTITY。

4146 的历史状态字符串 TWO_INDEPENDENT_REVIEW_PASSES_COMPLETED 不代表本次又做两轮审核；报告必须同时引用 reuse receipt。不复制旧 TYPE_AUDIT/PUBLICATION 到活动路径。发布前不得有 EXECUTION_FREEZE、runs、sealed。

## 8. 实际新 P0 与四个 ENGINE

此阶段显式使用 652 frozen submit.py。新 registered dispatcher 的 `act()` 只会 WAIT_ENGINE，不会替你提交缺失 ENGINE。

```sh
"$PYTHON" -B "$NEW/code/scripts/sr_f3_ce/submit.py" \
 --root "$NEW" --code-root "$NEW/code" --python "$PYTHON" --pool-size 1 \
 --cpu-partition cluster02 --cpu-qos soujanya-poria-startfund-2026-03 --cpu-node cpu-2 --kind p0
```

用同一参数 `--kind status` 读真实 scheduler/receipt，等待 p0 terminal success。P0 核全部 data/native/parent/local checks；native CPU 原件由 P0_REUSE_VERIFICATION 绑定到新 source，不能把本地 P0 替代实际 CPU job。

随后同样参数每次 `--kind engine` 提交一个固定 ready task，依次 engine--b0、engine--b1、engine--b2、engine--resume；每次仍重新计全用户 teacher 的实际+pending+unknown 和所有CPU submitted。pool=1 意味每个GPU job单卡，不是必须串行等待所有GPU；可在实际可用额度内继续提交下一固定任务。任何不确定/失败意图保留并阻断，不自动重试 ENGINE。

新 P0 + 四 ENGINE 必须同时满足任务完整证据和真实 COMPLETED/0:0。其中：

- 三父完整 ENGINE 2,880 条重新生成，包含三协议和三个诊断视图；不能用 P2/P3 的 768 条结构化消融替代。
- 三父 step0 MONITOR 各 512（共 1,536）重新生成并完整评分/commit；新源码 slot/group/cache 带 revision/prefill，采样种子保持原登记值。
- 恢复 ENGINE 512 新生成、256 cached replay，四逻辑更新、六物理更新；2+2 恢复、非零 CE mask/梯度、PREFIX/EXEC 同字节、CE×0 vs RL 均按当前 selected revision 验证。旧缓存一律不移入活动路径。
- C2 实际 request 使用 `{\n  "evidence": [`，raw full_text 恢复这段真实上下文，loss 排除全部预填，不能只把 `{` 拼回结果。
- work 轨迹只从已生成原文描述，不影响 reward、路由、选择或增加采样。

## 9. Observability 扩展与 registered driver

可在 ENGINE 进行时准备静态文件，但 **AMENDMENT 和 EXECUTION_FREEZE 必须等新 P0+四 ENGINE 全部成功、typed 发布核验通过之后**。

从 652 archive 复制 `instrumented_execution.py`、`observe_runtime.py` 到 NEW/native/observability/652.../，复制相关三个测试和两份观测文档作为原件（本目录 OBSERVABILITY_SOURCE_MANIFEST.template.json 列精确来源/目标）；所有 SHA/bytes 对实际新部署复核。两个运行源由 executed_sources 精确声明；tests/docs 不执行。SOURCE_MANIFEST/CPU_VALIDATION/AMENDMENT 不加入自身 sources。

本地 CPU 证明需按实际 source_sha256 覆盖 sources，status PASS、scope CPU_TOY_TRANSPARENCY_AND_ORCHESTRATION、正整数实际 tests_passed、gpu_validation_performed=false、new_model_generations=0、new_optimizer_steps=0。保留真实命令/输出/环境。可引用匹配同源的 687 测试证据及原始日志，不能把上次老 wrapper 的通过结果用于新 wrapper hash，不能写成真实 GPU 等价证明。

扩展 manifest 必须绑定新 RUN_IDENTITY SHA、652 source manifest SHA、完整 selected guard。用真实创建的 CPU_VALIDATION SHA/bytes填写 cpu_validation 字段。先 verify_extension(science=False)；不是手工造 AMENDMENT。

从 652 archive 将 dispatch_registered.py、build_delivery.py、finalize_external_types.py 安装到 NEW/operations/652.../。保持单独 4146 finalizer 的原部署。以下 register 会独占写 OPERATION_SOURCES 和 OPERATION_IDENTITY，重复执行仅可核原件，不能删除后覆盖：

```sh
"$PYTHON" -B "$NEW/operations/$OPS_COMMIT/dispatch_registered.py" \
 --mode register --root "$NEW" --python "$PYTHON" --user varun024 \
 --operation-commit "$OPS_COMMIT" \
 --extension-manifest "$NEW/native/observability/$EXT_COMMIT/SOURCE_MANIFEST.json"
"$PYTHON" -B "$NEW/operations/$OPS_COMMIT/dispatch_registered.py" \
 --mode submit --identity "$NEW/operations/$OPS_COMMIT/OPERATION_IDENTITY.json" --sequence 1
```

第二条是实际 held/no-requeue CPU dispatcher job，同样受 teacher MaxSubmit=5 约束；不能将其 CPU 占用排除。PYTHON 在 register 时 resolve(strict=True)，因此部署前先检查该绝对 interpreter 的 sys.executable/sys.prefix 与 numpy/torch/transformers/pyarrow/PIL 可用性，保留原环境身份，不用系统 python 代替。

OPERATION_IDENTITY 精确 schema 和固定策略见本目录 OPERATIONS_SCHEMA.json。核心字段：scientific652、operation commit、全部helper hash、RUN_IDENTITY/source/config/extension hashes、完整 selected guard、实际 canonical python/user、account rose/qos soujanya-poria-startfund-2026-03/partition cluster02/cpu-2、cpus4/mem12GiB/time3days、pool1、gpu_limit5/submitted_limit5、poll60/handoff900。不能自行写一份伪 registration 绕过 register。

实际 driver 顺序是：真实 status/terminal核验 → verify_typed → `seal_amendment` → `freeze_science` → instrumented Controller 的 train/eval/release。AMENDMENT 的 `repeat_engine:false` 表示扩展不再重跑已成功的本次 ENGINE；不表示可以省略本轮 fresh ENGINE。ENGINE 前没有收集的细分观测仍写 NOT_COLLECTED_DO_NOT_IMPUTE_OR_REPEAT_ENGINE，不能补零。

## 10. 最后放行与交付核验

SCIENCE 前 inspect EXECUTION_FREEZE：status FROZEN_BEFORE_SCIENCE、新 RUN_IDENTITY、code652、实际 chosen revision/prefill/serializer、完整 selection/请求、原 v1 history archive、数据/模型/四 ENGINE/task receipts、typed outputs、extension SOURCE/CPU_VALIDATION/AMENDMENT 均入绑定；operator_has_seen_new_test_results=false；无已有 science job/outputs。科学固定矩阵仍为原 33 tasks，消融9任务不加入；12条训练、64步、三seed、四臂和评分器不变。

最终 build_delivery 通过同一 operation sources + extension manifest 执行，核全矩阵、终态、raw coverage、work轨迹与 selected guard。保留旧3,904、消融2,304、新 ENGINE2,880、新恢复512与replay256、新MONITOR26,112等各自分母。`SERIALIZATION`、C2前缀、TYPE_REVIEW_REUSE、原不利结果和故障均随包保留；完整包仍需实测 <400,000,000 bytes 并逐成员核验后才能交付，不能因包已生成而声称科学成功。

本清单不预估任务何时完成，也不以 P1 已复现或部分 P2 已通过替代完整选择。
