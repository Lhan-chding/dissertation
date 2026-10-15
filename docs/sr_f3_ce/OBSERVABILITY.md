# SR-F3-CE 原生调用观测扩展

本模块属于正式科学执行之前的显式技术修订。科学源码保持 `71483fbf31d57213fdc7e6f3c881848397b45329`；不覆盖 `ROOT/code`，不改已有 RUN_IDENTITY、CONFIG、ENGINE 输出或其验收回执。已经运行的 ENGINE 没有使用本观测器，其未记录的图像编码次数和阶段耗时继续为 `UNKNOWN`。既有科学门禁与新观测层的验收是两组证据。

源码为 `scripts/sr_f3_ce/observe_runtime.py`。它不提交作业、读取模型回答、调用模型、执行反向、更新参数、安装启动注入或修改环境。新 worker 必须通过独立修订明确绑定启动命令，并在导入之前验证部署于 `native/observability/<version>/` 的模块 SHA256。仅新增一个文件或 manifest 不会使旧调度器自动接受新 worker。

## 显式接口与持久化责任

```python
# 此文件必须已由外部 worker 按冻结的扩展 manifest 验 hash。
identity = observe_runtime.install(SRF3CERuntime, emit)
# worker 将 identity 原样保存到 identity.json，然后才构造 runtime。
# emit(event) 由 worker 追加到 events.jsonl，并在返回前 flush + fsync。
runtime = SRF3CERuntime(...)
# 按既有注册程序执行；不为观测额外调用模型。
final_observer_metadata = observe_runtime.seal(SRF3CERuntime)
# worker 在自己的 RECEIPT.json 中保存 final_observer_metadata。
```

`install(runtime_class, emit)` 返回 JSON-safe identity；identity 包含该模块 SHA、实际类、三个原方法的源码/文件 hash、Torch 版本和计数/计时规则。安装本身不调用 emit，也不构造模型。每个类只允许一次安装；既有实例拒绝进入被包装的方法，必须在安装后构造。`snapshot` 是只读元数据查询。worker 在正常/异常终结边界必须调用幂等的 `seal`，它返回同一 snapshot schema 并置 `sealed=true`，之后拒绝新的 runtime/model 入口；不补造 END、不清除未闭合证据、不恢复模型。`uninstall` 仅用于测试或进程清理，活动 scope 在未 seal 时拒绝，不用于生产中途切换版本。

模块只包装 `__init__`、`generate_group`、`_native_forward`。原构造函数返回后注册实际 `runtime.model`、`get_base_model()`、`_visual_module()` 的 forward pre/post hooks；model 与 base 相同时只装一次。原 `model.forward`、native kernel callable 和科学方法的源文件不修改。全部 hooks 返回 `None`；不改参数、输入/输出 tensor、RNG、model mode、cache、checkpointing、autocast、optimizer 或 scheduler。只读取 tensor shape 元数据，不读取 tensor 值，不复制或保留输入、输出、梯度图。

emit 只能持久化 JSON 事件；不能调用模型、改变 RNG、执行科学恢复或重新进入 runtime。观测器阻止 emit 的 runtime 重入。外部 worker负责真实路径和追加规则，当前协作接口为 `accounting/observability/<job_key>/<task_id>/events.jsonl`、`identity.json`、`RECEIPT.json`。外部 worker增加的 run/job/task/freeze/amendment 字段不得覆盖模块事件的同名字段。

## 事件 schema_version = 1

每个 scope 先发送并确认持久化 `SCOPE_BEGIN`，再调用原方法，最后发送 `SCOPE_END`。BEGIN/END 用 `installation_id`、`runtime_id`、`scope_id` 关联；`parent_scope_id` 保留嵌套关系。`kind` 是 `initialization`、`generation`、`teacher_forcing`、`unscoped_model_forward` 或 `unscoped_visual_forward`；另有 `purpose`、`protocol`、`pid`、`time_unix_ns`。标识使用进程内计数和时钟，不消费科学 RNG。

END 包含：

| 字段 | 定义 |
| --- | --- |
| `outcome` | `RETURNED`、`RAISED`、`INTERRUPTED` 或 `CHECKPOINT_EARLY_STOP` |
| `original_call_returned` | 原方法是否实际返回；即便后续观测写入失败，也不能据此重新调用 |
| `exception_type` | 异常类型；不记录异常中可能包含的输入文本 |
| `coverage_status` / `coverage_issues` | `COMPLETE` 表示观测窗口已终结且未发现漏 hook；`UNKNOWN` 表示未终结调用或应有观测缺失 |
| `host_scope_elapsed_ns` | 原方法外侧的 host 单调时钟间隔；包括其内 hooks、嵌套 scope 和同步等待，不是 CUDA kernel 时间 |
| `model_stages` | 实际 model/base 主调用的阶段统计；同一调用只进入一个 stage |
| `visual_stages` | 实际 visual hook 的进入、返回、异常和 host 间隔；独立成本轴 |
| `model_hook_layers` | model/base 每层原始 hook 次数，仅用于核查委托与去重，不能与 `model_stages` 相加 |
| `observer_overhead_ns` | 截至 END 构建前已经完成的观测工作区间；最终累计值取 `snapshot` |
| `visual_expected` | 模型输入或实际 model hook 参数带图像时为 true；不是推算的编码次数 |
| `origin_scope_id` | 常规 scope 为自身；scope 外重入为 `UNKNOWN`，不能误配最近一个 TF |

stage 中字段都是非负整数；未出现的 stage 允许缺键或空字典。`entered`、`returned`、`raised`、`checkpoint_early_stop`、`unresolved` 记录真实 hooks。`entered = returned + raised + checkpoint_early_stop + unresolved`。`input_batch_items`、`input_sequence_slots` 取实际 model hook 的 tensor shape；shape 不可识别时增加 `shape_unknown_calls`，相应计数只是已知部分，不能把 0 当完整零成本。visual 的 `input_element_slots` 来自像素/patch tensor shape；`image_items` 仅在实际 `grid_thw` 的 shape 可识别为 `[images, 3]` 时累计，否则增加 `image_shape_unknown_calls`，绝不将 patch 行数冒充图像数。

成功返回的 generation/TF 若没有实际 primary model hook，或应有图像而完全没有 visual hook，将写 `coverage_status=UNKNOWN` 并抛 `ObservationError`；不能制造一条“真实调用数 0”的合格回执。原科学调用已经异常时保留其实际观测前缀和原异常，未发生的 visual 不能假计为失败编码。

## 阶段、嵌套与时间口径

generation scope 中**实际首次** model/base 主 forward 为 `prefill`，随后实际 forward 为 `incremental_decode`。不使用输出 token 数反推 forward 次数。该分类依据冻结生成器在一个 generate 调用内先全 prompt、再增量 forward 的执行约定；若未来后端执行约定改变，需要重新审核，不能沿用此名称掩盖重算 prompt。TF scope 归 `teacher_forcing`；其余 root/base 重入归 `unscoped_forward`。

PEFT 式 `model.generate → base.generate` 可以绕过顶层 model.forward，因此同时观察 model 与 base。TF 的 `model(**inputs) → base(**inputs)` 在同一 scope 只计一个主调用；两层 hook 的原始次数仍保留在 `model_hook_layers`。scope 内统计只归当前最内层 scope，不重复累加到所有父 scope。

`host_inclusive_ns` 是一对 pre/post hook 之间的 host 间隔。`host_exclusive_ns` 只减去**同一 runtime、同一 kind 的嵌套主调用区间**：model 轴减去嵌套新 scope 的 model 主调用，visual 轴减去嵌套 visual 调用；同 scope 的 model→base 委托本来就只计一次。它不自动扣除视觉阶段、其他 runtime 实例的调用、CPU 调度等待或观测开销。

因此：

- 不将 `model_hook_layers` 与 `model_stages` 加总。
- model/prefill host 时间包含其 visual 时间，**不能将 prefill + visual 当总时间**。
- 同一 scope 各主阶段的互斥计数可加总；父子 `host_scope_elapsed_ns` 不加总。跨实例或并发区间也不能当单一墙钟累加。
- `host_exclusive_ns` 可用于说明已扣除指定嵌套后的 host 区间，不称 decoder-only 时间、GPU 时间或 FLOPs。
- `snapshot` 的 observer overhead 是全安装的累计值，不能再加各 scope 的 overhead 快照。外部最终 RECEIPT 自身的写入时间标为 `UNKNOWN_NOT_MEASURED_BY_OBSERVER`。

观察器只在 scope BEGIN/END 写事件，decode hooks 内仅累计标量，事件数量不随 decode 长度增加。hook、scope bookkeeping、事件构建、emit 的开销以互斥 host 区间计入 `snapshot.observer_overhead_ns`。END 自己尚未完成的构建/写入不可能出现在同一已落盘 END 中；下一次或最终 snapshot 才包含这些实际耗时，不以自引用计时伪造值。

## Checkpoint、异常及写失败

native forward 返回后，backward 可能重新执行 checkpoint 中的 visual/module。此时不再沿用已结束 TF scope，而建立独立 `unscoped_*` BEGIN/END，实际调用计数仍保留。可用时仅查询 Torch 当前 autograd graph task id；这不改变图，也不能证明它属于哪一个旧 TF。`origin_scope_id=UNKNOWN` 是明确归属限制，不是将已观测调用抹成未知。

非 reentrant checkpoint 会通过内部 `_StopRecomputationError` 提前终止重算，整体 backward 仍可能成功。该路径单列 `checkpoint_early_stop` 和 `CHECKPOINT_EARLY_STOP`，不计科学失败，也不伪造完整 visual 返回。合法 `None` 返回通过独立普通 post hook识别，不会仅因 always-call post 中的 `None` 就判异常。若后端不再遵循这些行为，需要独立技术审查。

BEGIN 写失败时原方法尚未执行，抛 `ObservationWriteError`。END 写失败时原调用可能已完整完成，错误携带 `original_call_started`、`original_call_returned`；观测器绝不重试。若科学异常与 END 写失败同时发生，保留原科学异常类型（包括 OOM），附加写失败说明，并在 `snapshot.write_errors` 中记录；即使科学恢复随后完成，外部完整观测门禁也必须拒绝这些缺失事件，不能默默放行。

KeyboardInterrupt/SystemExit 可使 Torch 不运行 always-call post。在被包装的 generation/TF 内，外侧 finally 可将已进入但未返回的 hooks 记 `unresolved`、scope 记 `UNKNOWN`；scope 外 checkpoint/visual 重入没有这个 finally，故留下真实 BEGIN 和 `snapshot.active_scopes`，不会有伪造的 END。仅靠允许的 forward hooks 无法区分“中断后悬挂”与“确实仍在执行的合法重入”，因此不猜测清理。外部 worker 必须在 BaseException 终结边界立即 `seal` 并停止该进程的科学工作，不能捕获中断后继续复用 runtime。seal 后新 wrapper 和直接模块入口均拒绝，post hooks 也不会错误关闭旧悬挂 frame。SIGKILL/机器中断没有机会执行 seal 或写 END，留下的 BEGIN 无 END 同样视为未知，不能据此补 0、覆盖原件或为观测重采样。

外部终结验证应要求身份完全绑定、每个 BEGIN 有对应 END、scope 无未知 coverage、`sealed=true`、`snapshot.active_scopes` 和 `snapshot.write_errors` 为空。记录完整的已注册 OOM 尝试可以有 `RAISED`，不能把所有科学恢复前缀一律判为观测失败。CPU release 若未构造 runtime，安装 identity 与零事件 sealed snapshot 是合法情形；科学 train/eval 的零 scope 是否缺失由外部任务验收判定。

## 本地验证与证明范围

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/sr_f3_ce/test_observability.py
/Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python -B -m ruff check \
  --no-cache scripts/sr_f3_ce/observe_runtime.py tests/sr_f3_ce/test_observability.py
```

测试使用真实本地 CPU Torch toy modules：逐元素验证有/无观测时参数、梯度、输出、采样 RNG、Python/NumPy RNG、mode、cache/checkpointing 标志和 optimizer 状态精确一致；同时覆盖 PEFT 委托、TF 去重、嵌套 scope、model 重入、实际 checkpoint 重算、合法 None、原模型异常、BEGIN/END 写失败、emit 重入、可捕获中断、缺失 hooks、重复安装与清理。没有加载科研模型、提交 GPU 作业、增加 ENGINE 生成或 Adam。

这些 CPU 证据不构成真实 GPU 等价验收；观察器自身增加 host 工作和 I/O，不能声称零时间开销。旧 ENGINE 的成本缺口不会因这份未来观测扩展而消失。
