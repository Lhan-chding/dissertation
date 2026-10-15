# SR-F3-CE 前瞻观测修订与真实执行身份

此扩展在正式 science 前补充实际 image encoder、prefill/decode 与 teacher-forcing 观测。原科学部署 `71483fbf31d57213fdc7e6f3c881848397b45329`、`ROOT/code`、`RUN_IDENTITY.json`、原 source manifest、科学配置和已经执行的 P0/ENGINE 全部保留。不会重复 ENGINE，也不会改变训练矩阵、种子、模型、优化步数或采样预算。旧 ENGINE 未收集的细分成本明确为 `NOT_COLLECTED_DO_NOT_IMPUTE_OR_REPEAT_ENGINE`，不补成零或假定值。

`scripts/sr_f3_ce/instrumented_execution.py` 是显式启动入口；观测行为及计量边界见 [OBSERVABILITY.md](OBSERVABILITY.md)。不使用 `sitecustomize`，不向共享环境注入补丁，不将真实 wrapper 命令伪装成原 `run_worker.py`。

## 静态身份与部署

只从 `ROOT/native/observability/<40位扩展commit>/SOURCE_MANIFEST.json` 指定的目录执行。禁止搜索 latest 或任意额外模块。manifest 的相对路径允许安全子目录，例如 `tests/test_observability.py`、`docs/OBSERVABILITY.md`；禁止绝对路径、`..` 和解析后逃出扩展目录。只有 `executed_sources` 中的两个文件作为扩展运行源码加载，测试和文档作为验证原件保留。

```json
{
  "schema_version": 1,
  "experiment_id": "SR-F3-CE-20261016",
  "status": "FROZEN_OBSERVABILITY_EXTENSION",
  "extension_commit": "<40位真实扩展commit>",
  "base_code_commit": "71483fbf31d57213fdc7e6f3c881848397b45329",
  "run_identity_sha256": "<原RUN_IDENTITY文件SHA256>",
  "base_source_manifest_sha256": "<原deployment/SOURCE_MANIFEST文件SHA256>",
  "sources": [
    {"path": "instrumented_execution.py", "sha256": "<SHA256>", "bytes": 0},
    {"path": "observe_runtime.py", "sha256": "<SHA256>", "bytes": 0}
  ],
  "executed_sources": ["instrumented_execution.py", "observe_runtime.py"],
  "cpu_validation": {"path": "CPU_VALIDATION.json", "sha256": "<SHA256>", "bytes": 0}
}
```

上例大小和哈希是占位符，部署必须填写实际值。实际 sources 应同时保留相应测试与文档原件。`SOURCE_MANIFEST.json`、`CPU_VALIDATION.json`、`AMENDMENT.json` 不放入 sources，避免循环引用。

CPU_VALIDATION 至少包含 `status: PASS`、`scope: CPU_TOY_TRANSPARENCY_AND_ORCHESTRATION`、实际正整数 `tests_passed`、`source_sha256` 精确覆盖 sources 的每个路径，并明确 `gpu_validation_performed: false`、`new_model_generations: 0`、`new_optimizer_steps: 0`。这里后两项指本实验新增真实科学模型生成和优化，不把 CPU toy 透明性测试写成真实科学调用。保留实际测试命令、Python/torch 环境、退出码及原始输出证据；本地 CPU 与服务器实际 torch 的 CPU 验证应分别如实记录，不能声明 GPU 等价。

## API 与冻结顺序

- `verify_extension(root, manifest_path, science=False)` 验证静态源、CPU 原件、原 run/source identity；返回路径为绝对路径。`science=True` 还要求 amendment 与最终 freeze 的完整绑定。
- `install(root, manifest_path)` 验证旧源码和扩展后，返回 `InstrumentedController` class，并显式替换该进程内的 `sr_f3_ce.orchestration.Controller`。同一进程只允许相同 root、manifest 与 hash，CPU dispatcher 和交付构建器不安装模型 hooks。
- `seal_amendment(root, manifest_path, typed_verified=...)` 在全部 5 个原工程任务真实终态成功、无 science 目录和科学提交、无最终 freeze 时，封存原 ENGINE/P0 回执哈希、原 CONFIG/科学配置/固定矩阵、双盲发布及扩展身份。`typed_verified` 必须来自 dispatcher 的完整双盲证据验证。

顺序为原 P0/4 ENGINE 真正完成 → 完整 typed_v1 证据核验 → 封存 AMENDMENT → 原 `freeze_science` → 验证扩展静态链被 freeze 绑定 → 固定 train/eval/release。typed 已发布的准备代码保持 `4146e156ffa9a476b360daeb67503703faca138f`，脚本 SHA256 `1a75df6bf5fa1c0ef22a23ee7fd766a8e22f5a1c6c1fe7d579c4ab6135106aaf`；新 helper commit 不重新发布或重做双盲审阅。

原 `immutable_files` 直接纳入 native 下 JSON/Markdown 等，未直接收录 `.py`。因此最终 freeze 必须直接绑定扩展 SOURCE_MANIFEST、CPU_VALIDATION、AMENDMENT 三个文件；manifest 再绑定实际 `.py`、测试和文档原件。每次扩展进入时验证完整链，不把间接哈希称作原科学 source manifest 的变化。

两个扩展模块与原 worker 脚本采用读取源码 bytes → 对同一 bytes 检查已绑定 SHA/大小 → compile 同一 bytes 的显式加载流程，禁止 loader 再读文件或选择未登记 `.pyc`。首次导入原科学包前，driver、wrapper、builder 拒绝 `ROOT/code` 下实际存在的 `.pyc`/`.pyo`，并关闭字节缓存写入；发现缓存只拒绝，不删除原件。原 package 继续使用正常导入及其冻结文件/来源验证，不注入共享环境 importer。该防护针对已复现的静态未绑定缓存和显式加载竞争，不声明能防御恶意进程内存改写、所有文件系统竞争或替代第三方原生依赖的原环境审计。

## 实际提交、恢复和端点验证

所有未来 science 使用单 GPU、已登记任务和原资源上限。INTENT 除原字段外保存 `execution_entry_schema: 1`、绝对 `extension_manifest_path`、manifest/amendment SHA256、真实 `job_entry_command` 和每个 `worker_entry_commands`。claim_id 包含原 run identity、job key、manifest、amendment。batch script 的逐字节正文、sbatch 参数和每个资源字段均核验。

真实 batch 入口为 `instrumented_execution.py --mode job --root ROOT --manifest MANIFEST --job-key KEY`；真实 child LAUNCH 为 `--mode worker --root ROOT --manifest MANIFEST --task-id TASK`。job runner 保留原 held allocation、release intent、真实 GPU 可见性、唯一 task claim、resume chain、子进程 exit 和 task receipt 校验。原 P0/ENGINE verification 完整委托旧 Controller；未来 science 的真实 LAUNCH 单独校验，不修改证据来调用旧验证器。

worker 先验证原 scientific freeze、扩展与绑定，显式安装同一 Controller，使 endpoint 内部 `_completed()` 也验证新入口；随后安装纯 observer 并调用原 worker.main 一次。任何异常都保留失败，先尽力调用幂等 `observer.seal` 保存末态；事件日志缺失或不可读单独记为 `events_sha256: UNKNOWN` 和 `events_hash_error`，不覆盖原科学异常。异常回执本身不能写入时将该原因附在原异常上；不会因观测失败重跑 main。正常完成也 seal，任何未闭合 scope、未知 coverage、写入错误或未 sealed 状态都阻止完整观测回执。

TIMEOUT 仅沿原单 GPU science 精确恢复接口，通过原 CPU checkpoint/cache verifier 后才提交新 attempt。新 attempt 的真实 wrapper 和 amendment 不能丢失，旧目录和未完成观测永久保留。ENGINE 不自动重跑；COMPLETE 科学回执与失败作业并存时仍进入 receipt-only 核对，不重新采样。

## 观测原件与交付

每个 worker 的固定目录为 `accounting/observability/<job_key>/<task_id>/`，不写到 native 冻结树：

- `events.jsonl`：observer 事件加不可替换的 `execution_binding`，逐条 flush/fsync。
- `INSTALLATION.json`：`schema_version: 1`、`status: INSTALLED`、绑定及 observer.install 返回的原始安装身份，含实际原方法来源哈希。
- `FINAL.json`：只有原 worker/task 校验完成、完整 scope 逐项匹配、覆盖 COMPLETE、snapshot 已 sealed 且无 active/write_errors 才写 `status: COMPLETE`；绑定 INSTALLATION、任务回执、事件 SHA/bytes/rows/SCOPE_BEGIN/SCOPE_END 和末态 snapshot。
- `FAILURE.json`：异常类型/原异常、事件哈希、可取得的 sealed snapshot 或 snapshot 错误；缺少 FINAL 的轨迹不是完整观测。

`execution_binding` 固定包含 experiment_id、run_identity_sha256、execution_freeze_sha256、extension_manifest_sha256、amendment_sha256、job_key/job_id/task_id/claim_id。正常处理过的 OOM、`CHECKPOINT_EARLY_STOP` 仍保留实际成本；不能仅因存在 RAISED 就抹掉已恢复的科学路径。缺 END 或 UNKNOWN coverage 仍然拒绝完整观测。耗时是 host perf-counter 时间，不声称 CUDA kernel 时间；父子 scope 不相加冒充成本。

最终 builder 从 freeze 唯一解析扩展 manifest，导入前验证全部静态源/CPU 链，安装同一 Controller 后核验全部任务与原件。成功外部回执新增 `observability_extension`，其 manifest_path 为 run 相对路径，其余绑定 manifest_sha256、extension_commit、base_code_commit、observer_source_sha256、amendment_sha256、cpu_validation_sha256。dispatcher 逐字段核验后才能 DONE。

本地验证使用 `test_instrumented_execution.py`、`test_dispatch_registered.py` 和 `test_observability.py`，均无真实 GPU 或模型采样。这些测试不代替实际 CPU allocation、ENGINE 成功或最终科学交付证据。
