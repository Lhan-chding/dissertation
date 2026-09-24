# Confirm 历史暴露审计（2026-09-24）

已完成**已登记本地历史范围内**的审计，状态为 `COMPLETE_FOR_REGISTERED_LOCAL_HISTORY`。数据准备接口的 `status=COMPLETE` 指此明确范围内的完成，不是证明任何时间、任何机器都从未读取过 confirm。

找到两项**真值别名暴露**，必须排除。检查到的 confirm 原池有 288 个唯一场景，每个 family×chart×operation cell 有 16 个。排除后有 286 个场景：trend/line/sum4 剩 14 个，其余 17 个 cell 各 16 个；就这项暴露约束而言，仍可按每 cell 8 个构造 144 场景的 T。源训练/P/D 的真值隔离与最终面板内部真值唯一性仍须由数据构建器验证。本报告没有选择最终面板，没有读取最终模型评估结果。

## 已找到的暴露

| confirm 场景 ID | 真值 SHA-256 | 历史场景别名 | 来源性质 |
|---|---|---|---|
| `7a65b897492b4df107489fab05f7a8a3` | `92e26e4f6a66ff1fc403cc4fb247b727fabd435d553cd8e53f7607f4d6dd49ea` | `ba129d2ae3a502ee03ee794a34d6cd3c` | 历史 CPU 有限动作代理分析 |
| `d5f86d6420552f802a71673f9d002fbd` | `cfc849b8c73f93b8756614fcd2f15370094d32aa48521416753db6ab99f7be8a` | `e8420a1c915d88b0e6c3af448bd23876` | 历史 CPU 有限动作代理分析 |

两项真值哈希出现在 `modeling_v3_20260915/candidate05/Q1_analysis_full/units/*/SUMMARY.json` 的 `prompt_axis_metadata`，共有 384 份 summary 文件包含它们。历史元数据使用 `SYMBOLIC_PROXY` / `EVIDENCE_PROXY`，具有 16 个有限动作。这是已知的历史真值暴露，**不是此前真实 Qwen3.5-9B 在 confirm 上生成结果的证据**。无论旧场景 ID、chart 或 operation 是否不同，都保守排除相同四整数真值。仅检查 base_scene_id 会遗漏这两项。

## 检查范围与方法

检查以下四个服务器根目录中现存的原始运行元数据：

- `/projects/varunssd/louis-ssvc/runs`
- `/projects/varunssd/louis-ssvc/modeling_v3_20260915`
- `/projects/varunssd/louis-ssvc/modeling_v4_20260916`
- `/projects/varunssd/louis-ssvc/decision_modeling_v1_20260919`

元数据文件范围为文件名含 access、manifest、binding、plan、tasks、config、summary、receipt、result、panel 或 progress 的 JSON/JSONL。最终纳入 **2,344 个相关文件，共 1,038,469,890 字节**。初次有尺寸限制的扫描随后补做了逐块扫描，所有超过 5 MB 的相关元数据均已覆盖；最终 **读取错误 0，跳过的相关元数据 0**。重复代码快照、checkout、临时 pytest/validation fixtures 不作为独立运行证据。

逐块扫描检查目标 confirm 的场景 ID 与真值哈希；还对元数据中 `truth_world` / `truth` 的四整数数组计算 canonical JSON SHA-256，避免只保存坐标而未存哈希的别名被遗漏。扫描结果只返回身份命中，不返回坐标数组或性能值。

本次为身份交叉核验，对原 `confirm.jsonl` 做了**元数据投影读取**：解析后仅保留 base_scene_id、truth_structure_hash、family、chart、operation 和 split。没有生成模型输出，没有读取 final endpoint outcome，没有把完整 prompt 或真值坐标提供给选择器。该次审计自身的元数据读取事实记录在机器审计中，不能继续表述成“本次完全没有打开 confirm 文件”。

另外检查了五个历史 Slurm run 的 `data/generated/access.jsonl`（搜索深度不超过 7）和 `reviewed-data-20260908/generated/access.jsonl`。五个 Slurm 日志各有一条 calibration 读取；reviewed-data 日志含两条 dev、四条 calibration 读取。没有找到直接 confirm 读取条目。另有 36 份 Q2 `REFERENCE_ACCESS_RECEIPT.json` 记录有限动作已知事件分数在 prediction freeze 后读取；它们不等同于真实模型 confirm 面板访问。

## 证据与使用边界

- [数据构建器使用的暴露审计](history_evidence/CONFIRM_EXPOSURE.json)：`status=COMPLETE`、明确 scope、两项排除、286 个范围内候选 ID 与 cell 容量。
- [逐文件收据与身份命中](history_evidence/REGISTERED_HISTORY_EXPOSURE_SCAN.json)：2,344 份路径/大小/SHA-256，384 份命中文件与零错误记录。
- [Confirm 身份元数据](history_evidence/CONFIRM_IDENTITY_METADATA.json)：源文件哈希与 288 个场景的身份/分层元数据，无最终结果。
- [首次 P0 历史审计](history_evidence/HISTORY_AUDIT.json)：原 D2 P/E 场景 ID、访问日志和旧 H32 输出核验。其 `confirm_payload_opened_by_audit=false` 专指此前 P0 执行，不覆盖本次新增的 confirm 元数据投影。

本审计不声称存在覆盖所有读取路径的操作系统访问审计。未将原始 completion/sample/bank 大数据树、checkpoint 张量、图片、归档包或未登记外部目录作为额外元数据扫描范围；代码快照和 fixtures 也不作为新的独立暴露来源。没有访问记录只能支持“本范围内未找到记录”，不能支持全局从未读取。后续如发现本范围外的历史暴露，需更新排除并重新冻结相应 T 元数据；不能静默维持旧的未见声明。

286 个候选的含义是“在上述审计范围内，排除已识别身份或真值暴露后的候选”，不是最终规则已经冻结，也不是这些场景具有已验证的任何模型性能。最终运行仍须先冻结选择器、样本量和规则，再写出该原点 decision，随后才开始候选训练及独立评估。
