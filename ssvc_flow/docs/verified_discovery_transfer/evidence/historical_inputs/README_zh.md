# 历史排除集合与训练回放原件

2026-10-06，通过 `ssh eee-cluster` 对 `/projects/varunssd/louis-ssvc` 做只读采集。本目录保存新实验的输入与来源收据，没有修改旧实验，也没有产生模型调用。

## 真值轨道排除

`historical_exclusion.json` 含 3,323 个四整数排序轨道，来自 84 个含真值的历史元数据文件。共枚举并读取 1,552 个候选元数据文件；没有读取或 JSON 解析错误。3,822 个 scene ID 是保守的**已生成任务超集**，并非证明执行过模型推理的场景数。

采集覆盖现存 `train/dev/confirm/calibration/control/ood/natural_pool` 数据，以及 scene、prompt、prepared、cases、audit_labels 命名的元数据。历史训练集和评价集都被排除；尚未调用模型的已生成任务也被排除。包含主数据目录、历史 runs、modeling_v3、modeling_contrast、prospective_selection 与 protocol_state_probes 的元数据。相同原件的多个副本保留各自路径和 SHA-256，轨道最终去重。

`complete=true` 仅表示**声明的现存元数据清单已读完且无读取错误**。它不证明已删除、外部存储、只存在于归档或名字不符合元数据规则的历史任务不存在。目录裁剪规则、符号链接、没有真值的元数据清单均保留在 JSON 中。旧 raw 输出树、模型 checkpoint、缓存和归档不做全量重解析；不能把该字段升级为所有历史模型暴露的绝对完备证明。

补充 `dataset_bindings_audit.json` 从现存小于等于 2 MB 的 runtime/config/manifest/machine/binding JSON 文件追踪显式 `data_root`：找到 29 个带来源 SHA-256 的绑定，指向 6 个现存生成数据根（reviewed-data 与 5 个 Slurm 数据副本）。逐文件记录 SHA-256 后得到各根 3,251 个轨道，全部属于本次排除集合，缺失轨道为 0，读取/解析错误为 0。该审计核实已发现的显式数据根引用，不把只存 data hash 而未写路径的 manifest 自动算成已核实的根引用。

## R 回放来源

`replay_candidates.jsonl` 共 64 条，严格为 duplicate_encoding 32、cross_series 16、trend 16。每条保留历史 prompt/scene 记录、实际 raw_completion、解析后的 vector，以及来源 artifact 路径、完整文件 SHA-256、原始行号与行 SHA-256。

固定采集规则为历史源 lineage 61001，先读取全部 12 个 `COMMIT.json` 指定的 `samples.jsonl`，逐个校验 COMMIT 中保存的 SHA-256，并拒绝未提交或哈希不符的 attempt。按 segment/原始行顺序为每 scene 保留首条符合规则的成功输出；全部读取完成后，按 `base_scene_id` 全局升序和固定 family 配额选择。读取合计 24,268,777 字节；准入候选共 134 个 scene（duplicate 64、cross 39、trend 31），最终固定选择 64 条。没有按后续新实验效果选择回放，没有为缺少成功数据进行新生成。

准入逐项检查：

- prompt 注册表与实际输出行均为 `train`，实际 role 也为 `train`；两端均为 `SYMBOLIC_FRESH`。
- 原始输出 `event=category=X`，完整严格 JSON 是四个 0–99 整数，且与保存的 scene truth 完全相同。
- 注册 O0 的 system/user 原文出现在该次实际推理的 `input_audit.final_prompt` 中。
- 排除历史 D/P 全部 panel 和 protocol probe 全部 cases 的 scene ID，包括 U22 及其变换；合并排除 130 个 scene ID。

这些验证证明来源是实际历史训练输出；公开约束验算和新实验的角色隔离由新数据构造模块再次执行。真实成功的回放来自历史数据，因此并不声称它们是新任务。回放中不使用 teacher 的旧概率作为 PPO 比率。

## 复现和验证

只读采集器为 `ssvc_flow/scripts/verified_discovery_transfer/prepare_historical_inputs.py`，读取服务器历史文件后仅向 stdout 输出结果；外层 SSH 将结果保存到本地，再拆分为本目录 JSON/JSONL。原件尚存时可以通过路径和哈希追查。

十个针对性回归检查覆盖真实训练成功、拒绝评价输出、拒绝被 panel 占用的 scene、拒绝错误 raw 却标 X、拒绝协议 prompt 不一致、元数据损坏时撤销完整性、拒绝未提交 attempt、拒绝 COMMIT 哈希漂移、完整扫描后按 scene ID 选择，以及 runtime 数据根追踪。使用项目 `.venv/bin/python -m pytest -q tests/test_vdt_historical_inputs.py`，10 passed；结果在 `collector_tests.json`。
