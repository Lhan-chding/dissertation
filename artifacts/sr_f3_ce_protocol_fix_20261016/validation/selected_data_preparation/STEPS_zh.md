# Selected root 的 data / parents / prior 准备步骤

状态：仅做只读准备；未选择 v2 或 c2，未建立 selected root，未在服务器执行。所有动作须等待九项完整消融完成且 `verify_selection(ablation_root)` 返回 `PROTOCOL_SELECTED` 后，由主代理执行。科学源码保持 `652e52d530520a276b6420ff0f4ee479dcde6db1`；辅助调度源码不替代这份科学源码。新作业名称必须以 `-LH` 结束。

## 已核实的等价性

本地用当前源码（逐文件与科学提交 652e52d5 对照相同）分别加载 v2、c2 候选：每份 4096 个任务的原世界、chart/style、问题、query、qid 和顺序不变；每条 render_key 等于旧 IMAGE_MANIFEST；4096 个原 PNG 全量实读哈希通过；新 MODEL_INPUTS 的 4096 条 ModelInputStore 路由全通过。两份 fixed_spotchecks 均与原 48 项完全相同；原 48 项视觉审查逐项绑定原图 SHA。用原 4800 行历史源重做两次 compare_historical_worlds，均得到原 901 个历史世界、4096 个新任务、零交集，返回对象与原 HISTORICAL_WORLD_AUDIT 完全相同。没有重绘、模型调用或 GPU。

| 文件/材料 | 新根中的处理 | 原因 |
|---|---|---|
| plan/ 全部候选文件 | 用所选候选原字节，不再重建 | PROTOCOL_SELECTION 绑定已实际消融的 candidate |
| data/FROZEN_DATA_AUDIT.json | `load_frozen_dataset(new/plan).audit` 重新写入 | package inventory、prompt/teacher 和 protocol_revision 已改变 |
| data/IMAGE_MANIFEST.jsonl 和 4096 个 data/images/*.png | 逐文件 SHA/大小认证后复制原字节 | pixels 只依赖 world/chart/render_style，均已证明不变 |
| RENDER_AUDIT、SEMANTIC_RENDER_SPOTCHECK、VISUAL_SPOTCHECK_REVIEW | 原字节保留，并增加新的等价复用回执 | 是原图与原语义实例的历史审查，不是假称新增视觉审查 |
| HISTORICAL_WORLD_AUDIT、HISTORICAL_COVERAGE_CHAIN | 原字节保留，并绑定全部世界不变证明 | 原历史覆盖链及其内嵌原路径/哈希应保留；不用改路径伪造重审 |
| data/PREPARE_DATA_RECEIPT.json | 针对新模型输入路由重新生成 | 明示图像复用与 native 尚未执行 |
| data/PROTOCOL_REVISION_REUSE.json | 新生成 | 绑定 selection、candidate、旧图像/历史/视觉哈希以及等价检查 |
| parents/*/adapter、source_receipts | 允许相同字节；推荐让 fresh audit_parents 自行认证并复制 | 父政策、checkpoint、adapter 与此次序列化更改无关 |
| parents/PARENT_AUDIT.json | 用新 plan 重跑既有 CPU audit_parents | 原件记录旧 package_manifest_sha256，不能成为新根的活动父审计 |
| prior/source/ 和历史注册/授权/输入元数据 | 允许精确字节复用；fresh audit_parents 自动重新认证 | 历史来源不变 |
| prior/HISTORICAL_IMAGES.json、LEGACY_IMAGE_COPY.json | 原字节保留；export_prior 若重生成须完全相同 | HISTORICAL_IMAGES 的旧 source_root 是来源身份；外部面板绑定其 object hash |
| prior/raw/ 数据及索引 | 只按已认证导出清单复制，或直接 fresh export_prior | 保留所有历史证据与缺失事实；不能把旧结果当作新实验结果 |
| prior/RAW_EXPORT.json | fresh export_prior 重新生成 | 它绑定 parent_audit_sha256，必须指向新父审计 |
| native processor/support masks、P0、ENGINE、训练状态 | 不复制旧活动回执/缓存 | revision_ops 负责新 revision 的 native/P0/ENGINE；本清单不授权科学阶段 |

`COPY_ALLOWLIST.json` 给出 4102 条必需 data 文件（6 个回执/清单 + 4096 PNG）、8 个可选 contact sheets、27 条可选父模型文件和 489 条可选 prior 源文件，均含 SHA 与大小。父 adapter 的 3 个 safetensors 在本地快照中省略；489 条 prior 源中有 2 条省略。相应条目标为 `HASH_METADATA_ONLY`，必须服务器实读核验；不能称为本地已验证权重。原 prior/raw 的 45452 项清单和 234915563-byte 索引亦不在该本地准备快照中，只给严格展开算法，不能任意 `cp -r prior`。

不要在 Linux 直接用 `prepare_data.py` 默认重绘：原图字体为 macOS Arial.ttf，SHA `525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9`，原 Pillow 12.3.0 / FreeType 2.14.3。重绘会引入不必要的像素变化，并使原视觉审查失效。原 RENDER_AUDIT 记载这些历史渲染条件，应原样保留。

## 执行前变量与边界

revision_ops 应先完成完整 selection 验证，建立空的 `selected_v2` 或 `selected_c2` 新根，并放好所选 plan 和冻结 code。下列命令只是待执行文本；这里没有执行它们。须使用其核验过的 CPU Slurm allocation，所有 job name 末尾为 `-LH`。不复用旧 P0/ENGINE/task/orchestration/optimizer/rollout/native mask 回执。

```bash
export SRF3_PYTHON=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
export SRF3_OLD_ROOT=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_20261016
export SRF3_ABLATION_ROOT=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation
export SRF3_PARENT_ROOT=/projects/_hdd/varunhdd/louis-ssvc/sr_f2_20261013_r3
# SRF3_NEW_ROOT 由完整 PROTOCOL_SELECTION 决定，不手填偏好的 revision。
# SRF3_PREP_DIR 是已上传并由整体准备包 SHA 绑定的本目录。
: "${SRF3_NEW_ROOT:?revision_ops must supply the verified fresh selected root}"
: "${SRF3_PREP_DIR:?supply the authenticated preparation directory}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$SRF3_NEW_ROOT/code/src"
```

## data 的精确复制和新审计

下面代码使用现有库函数，不修改源码。它不调用模型，不重绘，只核验与复制原 data，计算新 manifest 审计。`COPY_ALLOWLIST.json` 固定 SHA 为 `761cc753e5c3173511c73a7c245af381cac8956f92f67d5f5e07c4bd3dad2ca1`。如果下面代码失败，保留新根的部分输出和失败日志；不覆盖或修补旧根。

```bash
"$SRF3_PYTHON" -B - <<'PY'
import copy, os
from pathlib import Path
from sr_f3_ce.ablation import verify_selection
from sr_f3_ce.data import ModelInputStore, load_frozen_dataset, object_sha256, read_jsonl
from sr_f3_ce.preflight import bounded_file, copy_verified, file_hash, read_json
from sr_f3_ce.rendering import fixed_spotchecks, write_json

old = Path(os.environ['SRF3_OLD_ROOT']).resolve(strict=True)
new = Path(os.environ['SRF3_NEW_ROOT']).resolve(strict=True)
ablation = Path(os.environ['SRF3_ABLATION_ROOT']).resolve(strict=True)
prep = Path(os.environ['SRF3_PREP_DIR']).resolve(strict=True)
selection = verify_selection(ablation)
if selection.get('status') != 'PROTOCOL_SELECTED':
    raise PermissionError('Complete nine-task protocol selection is required')
revision = selection['selected_revision']
if revision not in ('v2', 'c2') or new != ablation.parent / ('selected_' + revision):
    raise PermissionError('Selected root/revision identity differs')
if new == old or new.is_relative_to(old) or old.is_relative_to(new):
    raise PermissionError('The selected root must be independent')
if (new / 'data').exists():
    raise FileExistsError('Require a fresh data directory; preserve any earlier attempt')
if file_hash(prep / 'COPY_ALLOWLIST.json') != '761cc753e5c3173511c73a7c245af381cac8956f92f67d5f5e07c4bd3dad2ca1':
    raise PermissionError('Preparation allowlist changed')
if file_hash(old / 'RUN_IDENTITY.json') != '3667107b1bb90a90ffa1401067f3d222e513eb2de842648d5dd59d7c3ff8a489':
    raise PermissionError('Original input identity changed')
allow = read_json(prep / 'COPY_ALLOWLIST.json')
original = load_frozen_dataset(old / 'plan')
dataset = load_frozen_dataset(new / 'plan')
variant = {'v2': 'P2', 'c2': 'P3'}[revision]
expected = selection['candidate_identities'][variant]
if dataset.audit['protocol_revision'] != revision or file_hash(new / 'plan/PACKAGE_SHA256.json') != expected['package_sha256']:
    raise PermissionError('Candidate package differs from selection')
images = read_jsonl(old / 'data/IMAGE_MANIFEST.jsonl')
if list(original.tasks) != list(dataset.tasks) or list(dataset.tasks) != [r['qid'] for r in images]:
    raise PermissionError('Task/image order changed')
image_by_qid = {r['qid']: r for r in images}
for qid, task in dataset.tasks.items():
    before = original.tasks[qid]
    conserved = {k: v for k, v in before.items() if k not in ('prompt', 'teacher')}
    current = {k: v for k, v in task.items() if k not in ('prompt', 'teacher')}
    if current != conserved:
        raise PermissionError('Non-protocol task content changed: ' + qid)
    render_key = object_sha256({k: task[k] for k in ('world', 'chart', 'render_style')})
    if render_key != image_by_qid[qid]['render_key']:
        raise PermissionError('Image no longer renders the selected task: ' + qid)
spot = read_json(old / 'data/SEMANTIC_RENDER_SPOTCHECK.json')
visual = read_json(old / 'data/VISUAL_SPOTCHECK_REVIEW.json')
history = read_json(old / 'data/HISTORICAL_WORLD_AUDIT.json')
chain = read_json(old / 'data/HISTORICAL_COVERAGE_CHAIN.json')
if fixed_spotchecks(dataset) != spot['items'] or visual['reviewed_count'] != len(spot['items']) or visual['issues_remaining']:
    raise PermissionError('Semantic/visual review is not equivalent')
if visual['selection_sha256'] != file_hash(old / 'data/SEMANTIC_RENDER_SPOTCHECK.json') or visual['image_manifest_sha256'] != file_hash(old / 'data/IMAGE_MANIFEST.jsonl'):
    raise PermissionError('Visual review source hashes differ')
if {r['qid'] for r in visual['items']} != {r['qid'] for r in spot['items']}:
    raise PermissionError('Visual review qid coverage differs')
for row in visual['items']:
    if row['image_sha256'] != image_by_qid[row['qid']]['image_sha256'] or row['semantic_review'] != 'PASS' or row['source_image_review'] != 'PASS':
        raise PermissionError('Visual review image/semantic result differs')
if history['status'] != 'PASS' or history['overlapping_qids'] or chain['status'] != 'REGISTERED_SR_F1_F12_F2_SYNTHETIC_WORLD_COVERAGE_VERIFIED':
    raise PermissionError('Historical world coverage failed')
if chain['new_tasks_world_audit']['sha256'] != file_hash(old / 'data/HISTORICAL_WORLD_AUDIT.json'):
    raise PermissionError('Historical world chain hash differs')
for item in allow['copy_required_data']:
    source = bounded_file(old, item['path'], allow_internal_symlink=False)
    if file_hash(source) != item['sha256'] or source.stat().st_size != item['bytes']:
        raise PermissionError('Original data changed: ' + item['path'])
for item in allow['copy_required_data']:
    copy_verified(old / item['path'], new, item['path'], item['sha256'])
write_json(new / 'data/FROZEN_DATA_AUDIT.json', dataset.audit)
expected_audit = {'v2': '9715e784490bf81d21674390c3d3cc65990648f0077053403a96d9bd958b6caa', 'c2': 'aeb37e548e551cab09a14419f08e8b38310faae6c790ee1e18414afcd9aa613c'}[revision]
if file_hash(new / 'data/FROZEN_DATA_AUDIT.json') != expected_audit:
    raise PermissionError('Candidate data audit differs from prechecked frozen source')
store = ModelInputStore.from_files(new / 'plan/manifests/MODEL_INPUTS.jsonl', new / 'data/IMAGE_MANIFEST.jsonl', image_root=new / 'data')
for qid in dataset.inputs:
    store.model_input(qid)
receipt = {
    'status': 'CPU_DATA_AND_SOURCE_IMAGES_VERIFIED', 'protocol_revision': revision,
    'tasks': len(dataset.tasks), 'historical_worlds': history['status'],
    'rendering': 'SOURCE_IMAGES_VERIFIED', 'model_input_routes_checked': len(dataset.inputs),
    'model_calls': 0, 'native_processor': 'NOT_RUN',
    'visual_spotchecks': 'REUSED_BY_VERIFIED_IMAGE_AND_SEMANTIC_IDENTITY',
    'scientific_training_authorized_by_this_receipt': False,
}
write_json(new / 'data/PREPARE_DATA_RECEIPT.json', receipt)
write_json(new / 'data/PROTOCOL_REVISION_REUSE.json', {
    'status': 'BYTE_AND_SEMANTIC_EQUIVALENCE_VERIFIED', 'protocol_revision': revision,
    'source_root': str(old), 'source_run_identity_sha256': file_hash(old / 'RUN_IDENTITY.json'),
    'protocol_selection_sha256': file_hash(ablation / 'PROTOCOL_SELECTION.json'),
    'candidate_package_sha256': file_hash(new / 'plan/PACKAGE_SHA256.json'),
    'model_inputs_sha256': file_hash(new / 'plan/manifests/MODEL_INPUTS.jsonl'),
    'frozen_data_audit_sha256': file_hash(new / 'data/FROZEN_DATA_AUDIT.json'),
    'prepare_data_receipt_sha256': file_hash(new / 'data/PREPARE_DATA_RECEIPT.json'),
    'copy_allowlist_sha256': file_hash(prep / 'COPY_ALLOWLIST.json'),
    'reused_receipts': {r['path']: r['sha256'] for r in allow['copy_required_data'] if not r['path'].startswith('data/images/')},
    'task_order_and_nonprotocol_fields_equal': True, 'render_keys_checked': 4096,
    'image_routes_hashed': 4096, 'spotchecks_equal': len(spot['items']),
    'all_task_worlds_equal_to_original': True, 'historical_receipt_preserved': True,
    'historical_source_paths_are_original_provenance': True,
    'new_visual_review_claimed': False, 'new_render_calls': 0, 'model_calls': 0,
    'scientific_training_authorized_by_this_receipt': False,
})
print('DATA_PREPARATION_PASS', revision, expected_audit)
PY
```

可选的 8 张 contact sheets 只用于保留原审查展示证据。复制前同样逐 SHA/大小核验；若原服务器根没有这些展示图，不把它们当数据必要条件，可从本地对应旧 prepared/visual_review 目录按 allowlist 上传。

`dataset.audit.package` 中保留的 `selected_for_science=False` 是不可变候选包的原始状态；不能手改它。新根真正的协议选择由根部 `PROTOCOL_SELECTION.json` 和 RUN_IDENTITY 单独绑定。

## parents 的新审计

下面命令必须在已核验的 CPU Slurm allocation 中执行（入口显式检查 SLURM_JOB_ID）。建议 job name `SRF3CE-SELECTED-PARENTS-LH`。它只在 CPU 读取已认证 checkpoint/adapter tensors 和哈希 base 文件，不实例化 9B 模型、不做推理、不继承 optimizer，不修改 SR-F2 根。运行环境必须满足历史 Python 3.12.14 与包版本；base snapshot 保持原缓存位置。

```bash
: "${SLURM_JOB_ID:?CPU Slurm allocation is required}"
test ! -e "$SRF3_NEW_ROOT/parents/PARENT_AUDIT.json"
"$SRF3_PYTHON" -B "$SRF3_NEW_ROOT/code/scripts/sr_f3_ce/audit_parents.py" \
  --parent-root "$SRF3_PARENT_ROOT" --plan-dir "$SRF3_NEW_ROOT/plan" --output-root "$SRF3_NEW_ROOT"
```

推荐直接让 audit_parents 从原 SR-F2 生成全部 27 个复制件、3 个 TENSOR_INVENTORY 和 prior/source、环境/源审计，省去预复制后再认证的一次重复 I/O。`COPY_ALLOWLIST.json` 中父与 prior/source 条目是精确允许集合，不是必须预复制步骤。新 PARENT_AUDIT 与旧件的唯一预期差异是 package_manifest_sha256；实际结果若还有差异，应保留并调查，不自动改写回执。

## prior 的历史证据和外部图像

建议在上述父审计成功后运行同一既有 exporter，job name `SRF3CE-SELECTED-PRIOR-LH`。其输入是原 SR-F2 released evidence，导出的 historical raw 仍是历史结果。当前已知状态 `MISSING` 是 108 COMPLETE / 1 MISSING；缺失项为原 V0–V4 只有 16 个 numeric roots（32 questions），不足请求的 32 roots。不得补生成或改成 COMPLETE。该缺失原本不阻断新科学实验。

```bash
: "${SLURM_JOB_ID:?CPU Slurm allocation is required}"
test ! -e "$SRF3_NEW_ROOT/prior/RAW_EXPORT.json"
"$SRF3_PYTHON" -B "$SRF3_NEW_ROOT/code/scripts/sr_f3_ce/export_prior.py" \
  --parent-root "$SRF3_PARENT_ROOT" --output-root "$SRF3_NEW_ROOT"
```

fresh export 会复制 prior/raw 的 45452 个源文件、64944 个 raw records 的索引，并复制 ChartQA 的 1509 张 unique images（2500 questions）到 external/legacy_images；它不改分、不采样。`prior/HISTORICAL_IMAGES.json` 的 4400 张历史图像身份与 `prior/LEGACY_IMAGE_COPY.json` 应保持原字节，包括旧 SR-F2 source_root。如果 revision_ops 提前复制了这两个 receipt 和 legacy images，既有 write_once/copy_verified 必须验证完全相同，否则失败保留，不覆盖。外部面板 FREEZE/PANELS 的新根 image_root 重绑定交由 revision_ops，不能沿用旧外部活动路径。

若确需预复制 raw，先认证 `prior/raw/FILE_MANIFEST.json` 的历史 release/registration 链，再逐 entry 的 export_path/sha256/bytes 展开；只接受 `prior/raw/` 下相对路径、无路径穿越或逃逸符号链接。索引只按旧 RAW_EXPORT.record_index 指定 SHA `67fb573dc74a3362cfd5647d89d2e087498708d862f81fe6245372b4b3fb330a` 复制。旧 RAW_EXPORT 本身不得放进新活动路径；新父审计 SHA 变化会造成不可覆盖冲突。更简单且可独立认证的推荐路径仍是直接 fresh export_prior。

## 最终依赖一致性检查与绑定

以下检查只在新的 parent/prior 命令实际成功后执行；它不是把旧 PASS 改字段伪装成重新审计。

```bash
"$SRF3_PYTHON" -B - <<'PY'
import copy, os
from pathlib import Path
from sr_f3_ce.preflight import file_hash, read_json
old = Path(os.environ['SRF3_OLD_ROOT'])
new = Path(os.environ['SRF3_NEW_ROOT'])
expected_parent = copy.deepcopy(read_json(old / 'parents/PARENT_AUDIT.json'))
expected_parent['package_manifest_sha256'] = file_hash(new / 'plan/PACKAGE_SHA256.json')
if read_json(new / 'parents/PARENT_AUDIT.json') != expected_parent:
    raise PermissionError('Unexpected fresh parent audit difference')
expected_prior = copy.deepcopy(read_json(old / 'prior/RAW_EXPORT.json'))
expected_prior['parent_audit_sha256'] = file_hash(new / 'parents/PARENT_AUDIT.json')
if read_json(new / 'prior/RAW_EXPORT.json') != expected_prior:
    raise PermissionError('Unexpected fresh historical export difference')
for relative in ('prior/HISTORICAL_IMAGES.json', 'prior/LEGACY_IMAGE_COPY.json', 'prior/MODEL_ENVIRONMENT_IDENTITY.json'):
    if file_hash(new / relative) != file_hash(old / relative):
        raise PermissionError('Historical identity changed: ' + relative)
print('PARENT_AND_PRIOR_REBINDING_PASS')
PY
```

将 `BOUND_FILES_GUIDANCE.json` 中的 data/parents/prior 必需项按新字节 SHA 加入新 RUN_IDENTITY；revision_ops 同时绑定 selection、request、plan/source、native、external、ops。先完成所有文件再冻结 RUN_IDENTITY，不能冻结后补写 bound file。无需把 234.9 MB raw RECORD_INDEX 直接加入每次 worker verify_run 都重新哈希的清单；RAW_EXPORT 已绑定其 SHA，保留其完整验证链用于最终交付即可。

本准备没有查询服务器现态。服务器源文件仍需执行时验证；没有制造新 PASS、提交 Git、重启实验、选择协议或消耗模型/GPU。
