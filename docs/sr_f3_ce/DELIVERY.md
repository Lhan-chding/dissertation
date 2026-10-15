# SR-F3-CE 最终交付辅助工具

`scripts/sr_f3_ce/build_delivery.py` 是独立的交付辅助源码，不属于已冻结的科学执行源码。它不提交作业、不生成回答、不训练、不改变源文件，也不能释放封存结果。使用运行实验的 Python 环境，在真实 release 已完成之后执行：

```sh
python /path/to/operations/build_delivery.py \
  --root /path/to/sr_f3_ce_20261016 \
  --output /path/to/sr_f3_ce_20261016/delivery/SR_F3_CE_20261016.zip \
  --helper-manifest /path/to/operations/OPERATION_SOURCES.json \
  --extension-manifest /path/to/sr_f3_ce_20261016/native/observability/COMMIT/SOURCE_MANIFEST.json
```

工具从 `--root/code/src` 导入冻结科学实现，首先认证 `deployment/SOURCE_MANIFEST.json` 的每个文件。然后从正式 `EXECUTION_FREEZE.engineering_files` 中唯一解析 `native/observability/<40位commit>/SOURCE_MANIFEST.json`；不搜索 latest，也不允许任意传入文件。上面 `--extension-manifest` 可省略，指定时必须等于该唯一冻结路径。manifest、CPU_VALIDATION、AMENDMENT 三个 JSON 都必须被 freeze 直接绑定，manifest 中所有实际 Python 源码逐文件核对 hash/bytes 后才导入 launcher。

显式扩展 launcher 使用同一次读取的源码字节核对冻结 SHA-256/大小并 `compile` 执行，不调用 loader 重新读取路径或缓存 `.pyc`。`__file__`、`__spec__` 与 `sys.modules` 身份仍指向被认证的实际路径。关闭 bytecode 写入并不足以禁止读取 unchecked-hash 缓存；回归测试单独验证该边界。

原科学包保持普通 Python 导入；首次导入前会拒绝 `root/code` 中已有的任何 `.pyc`/`.pyo`，同时禁止写入新 bytecode，并继续核对原 manifest 和模块实际来源。发现缓存时直接停止并保留原件，不自动删除。这是限定在冻结源码树的静态缓存防护，不声称绝对认证第三方 native 库、已导入对象或所有并发文件系统变化；真实服务器是否无缓存必须以部署检查为据。

交付工具显式调用 `verify_extension(science=True)`、`install(root, manifest)`，并确认冻结 orchestration 模块实际使用返回的 `InstrumentedController`。放行前再次验证扩展身份与 Controller 安装状态，再调用实际 `completed()`：旧 P0/ENGINE 继续原验证，未来科学作业检查冻结扩展 launcher 和真实 LAUNCH 入口。33 个注册任务（P0、4 ENGINE、12 training、15 evaluation、release）仍须经过真实 Slurm 终态、allocation/script、进程退出、任务回执、来源和输出 hash 验证。未满足时不会打开新 TEST 原文。存在 ZIP、文件名、`COMPLETE` 或 `RELEASED` 字符串均不是放行依据。CPU 透明性验证不冒充 GPU 等价性或实验完成。

## 内容和复评分方法

- `ENDPOINT_RAW.jsonl.gz` 原样加入，逐条与原始端点分组核对，并再次调用冻结的生产评分器。`ENDPOINT_SCORES.jsonl`、release 原文重评分回执、投影 hash 和每组评分链一并验证。此流程是全量原文重新评分，**没有声称新独立实现的评分器**。
- `TRAINING_GROUPS.jsonl.gz`、`SUPPORT_MASKS.jsonl.gz`、`MONITOR_RAW.jsonl.gz`、`ENGINE_RAW.jsonl.gz` 直接流式压缩到 ZIP。每行包含 `original_utf8`、原文件相对路径/hash/字节数、记录数及 `task_sidecar`。对 `original_utf8.encode('utf-8')` 求 SHA-256 可检验原文件字节；`json.loads(original_utf8)` 恢复原对象。原件中的 prompt、generated tokens、实际停止、预算与失败回答全部保留。
- `SUPPORT_MASKS` 包括 2,304 个科学支持样本和 16 个 ENGINE 支持样本；原生目标文本、token masks、offset 审计与类型双审原件另外保留。`ENGINE_RAW` 包含 2,880 个协议诊断回答和 512 个唯一工程更新回答；256 条恢复重放不重复计为新生成。
- `TASK_SIDECARS`、图像/root/family/new/legacy/repair/transfer 身份、实际图像、模型轻量配置/tokenizer/processor 源文件、官方外部评分与提示源码、历史原文、技术失败、全部成本账本、任务和调度回执均包含。`IMAGE_SOURCE_MAP` 与 `RAW_SOURCE_MAP` 将原服务器引用映射到包内实际内容。
- `scientific_source/` 与原冻结 manifest 保留科学代码原件；`helper/source/` 和 `HELPER_SOURCE_MANIFEST.json` 单独记录本工具与同次辅助部署的文件。两套来源不得混写。
- 前瞻观测扩展的 manifest、CPU 验证、AMENDMENT 与 `.py` 原件完整在 `evidence/native/observability/<commit>/`；`DELIVERY_PROVENANCE.json` 和外部成功回执的 `observability_extension` 记录该扩展身份。它不替换原科学源码，不重新生成旧 ENGINE 数据。
- `TRAINING_SIGNAL_INDEX.json` 索引全量物理更新回执，保留每个逻辑步及 1/16/32/64 的 loss、分项梯度、clip/参数差、sampler/TF 和 TIS 诊断。每组 J/优势与逐 support answer/work CE 的原始回执完整在包内；重复物理尝试不合并，不冒充新的逻辑更新。
- 权重和完整 `.pt` checkpoint 保留在服务器，逐文件大小、SHA-256、实际位置与省略原因写入 `SERVER_ONLY.json`。字体不分发。原始 ChartQAPro 全量下载 parquet 可留服务器，冻结所选任务的 sidecar、图像、原文与评分全部入包。完全相同的成功 pending journal 是冗余编码；已交付原文承载其全部字段。任何部分、不同或失败 journal 均保留。

中文报告使用实际 `PRIMARY_ANALYSIS`、`EXTERNAL_ANALYSIS` 和 `PREFIX_DIAGNOSTICS` schema。三个主要对比同表展示 b0/b1/b2、95%、98.3333% 及 df=2 t 敏感性；完整呈现修复、新组合/语言/外观/长度迁移、保持和每族退化、无辅助 RL、独立协议、外部与未解析类型。不利结果不筛除。缺失的 GPU 小时、排队、prefill/decode 时间标 `UNKNOWN`，不按申请时间或重复行估算。

成本以一次只读 `sacct` 查询的实际 `ElapsedRaw/Submit/Start/End/AllocTRES` 为依据，严格核对本运行已认证的 INTENT/TERMINAL 身份。成功和失败尝试分别列示，按实际 GPU allocation 计算小时；缺失 GPU TRES 不推断为零。原始输出保存在 `SCHEDULER_COST_RAW.json`。总排队时间为各 allocation 等待之和，可能重叠；总墙钟为最早 Submit 至最晚 End。注册 orchestration attempts 的成本不含单独启动的历史准备 CPU 作业，其原件见 `CPU_PREPARATION_ACCOUNTING.json`（存在时）；尚在运行的 dispatcher/delivery CPU 总成本标 UNKNOWN。每行 `generation_seconds` 属同批累计时间，工具不按回答重复相加。

`accounting/observability/<job_key>/<task_id>/{events.jsonl,INSTALLATION.json,FINAL.json}` 原样全量入包。事件的 execution_binding 必须对应原 INTENT/STATE、run/freeze、扩展和 AMENDMENT；FINAL 还须绑定安装原件、事件原始字节/hash/条数、BEGIN/END 数、最终 observer snapshot 和原任务完成回执。最终 snapshot 必须 `sealed=true`、无活动 scope 或写入错误；seal 不为未闭合 scope 补造 END。每个未来任务须至少有一个严格完成的观测尝试，历史失败尝试仍保留。无 FINAL、无 END、末尾中断事件或显式 UNKNOWN 的 scope 只报告已观察子集，不用零填充未知成本。

`COSTS.prospective_observations` 按物理尝试、kind/purpose/protocol 给实际模型与视觉 hook 计数及主机时间；hook 只归最内层 scope，`model_hook_layers` 是审计信息，不能重复相加。父 scope 墙钟只记范围、不求总和，model/visual 是重叠计时轴，不相加；host inclusive/exclusive 都不是 GPU kernel 时间。FINAL 中的 observer overhead 每尝试只读一次，FINAL 自身写成本仍未知。非重入 checkpoint 的 `CHECKPOINT_EARLY_STOP` 保留为原生控制流，不自动视为科学失败。旧 ENGINE 的 image/prefill/decode 阶段数据依旧 UNKNOWN，未通过回放重建。

类型注册表跟随冻结 `external/PANELS.json` 的 root/freeze hash，旧 v0 和 typed_v1 并存时报告实际活动版本。`OPERATIONS_SNAPSHOT.json` 明确只封存发布前已经存在的不可变辅助来源/运行回执；之后的 driver 结束回执不伪装为当时已存在。活动 PROGRESS/QUEUE/STATE/日志不作冻结来源；相邻已有 TERMINAL.json 的历史 dispatcher 日志完整保留。独立准备作业目录的历史 stdout/stderr、失败回执和当时轻量代码仍完整保存，且不计入新科学矩阵的 33 个任务。

历史 `prior/RAW_EXPORT.json` 可以为 literal `MISSING`，但必须明确 `operation_completed=true`、`critical=false`、`blocks_new_science=false`。此类已完成导出中的非关键旧诊断缺件逐项报告，不阻断完整新实验交付，不生成替代历史样本。ChartQAPro 的 `UNAVAILABLE` 仍表示外部未完成，不能称外部迁移通过。

## 验证和中断恢复

发布前顺序执行 ZIP CRC 全读、safe unique relative names、payload SHA-256、manifest/checksum 闭合、gzip 解压至 EOF、聚合原件 hash/记录数与入口相对链接检查。仅验证通过才原子发布 ZIP，之后写 `OUTPUT.verification.json` 和 `OUTPUT.sha256`。任何源文件或证据链冲突直接拒绝，保留失败回执和临时 ZIP，不覆盖原始实验结果。

若最终 ZIP 已发布、外部回执尚未写完，重跑会再次执行实际 Controller/source 门禁、原始 release 全量重评分、训练/监测/工程覆盖验证，并将 ZIP 内每条原文/sidecar 与实际已认证原件逐项绑定，之后才补回执。已有完整交付重跑同样严格验证；仅自洽的 ZIP manifest 或伪造 coverage 数字不会通过。成本使用归档中的原始查询快照重新计算，不能通过恢复重写既有报告。中断的 `.pending-*` 保留，新尝试使用另一个唯一临时文件；不会在旧 partial 上追加或把它标为完成。发布及外部回执均 fsync；发布后回执出错时失败记录明确 ZIP 已发布及其实际 hash，不声称临时文件仍存在。

成功回执：

```json
{
  "schema_version": 1,
  "status": "VERIFIED_DELIVERY",
  "archive": {"path": "/canonical/output.zip", "sha256": "...", "bytes": 1, "entries": 1},
  "run_identity_sha256": "...",
  "scientific_source_manifest_sha256": "...",
  "helper_manifest_sha256": "...",
  "coverage": {
    "verified_tasks": 33, "training_paths": 12, "evaluated_models": 15,
    "logical_updates": 768, "training_records": 98304, "support_examples": 2304,
    "monitor_records": 26112, "engine_diagnostic_records": 2880,
    "engine_new_rollouts": 512, "engine_replayed_rollouts": 256, "engine_physical_updates": 6
  },
  "checks": {
    "crc": "PASS", "safe_unique_names": "PASS", "payload_hashes": "PASS",
    "manifest_hashes": "PASS", "entry_links": "PASS", "gzip_read_to_eof": "PASS",
    "raw_rescoring_chain": "PASS"
  }
}
```

`coverage.records` 为 `229500 + 15*n`，其中 `n` 是实际冻结的 ChartQAPro 图像根数（0–256）。上述 JSON 仅说明字段，不是实验回执。

辅助 manifest 可采用 `schema_version=1`、`experiment_id`、`operation_commit`、`scientific_code_commit` 和 `files={"build_delivery.py":"sha256", ...}`。所有列出的辅助脚本必须与 manifest 同目录且字节匹配。缺省不指定 manifest 时，工具生成自身源码清单，并把未知 commit 明确写为 `UNKNOWN`。

本地检查：

```sh
PYTHONPATH=src python -m pytest -q -p no:cacheprovider --import-mode=importlib tests/sr_f3_ce/test_delivery.py
python -m ruff check --no-cache scripts/sr_f3_ce/build_delivery.py tests/sr_f3_ce/test_delivery.py
```

这些小型 fixture 验证拒绝缺项/封存/篡改、负结果保留、真实 CRC 损坏、路径/链接和不可覆盖回执，不代表 GPU 实验已完成。交付入口为 ZIP 内 `START_HERE.md`。此包可重评分，但不是包含服务器权重的完整重训包。
