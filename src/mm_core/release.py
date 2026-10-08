"""Evidence-derived audit release; missing work is explicit and DEV stays unauthorized."""

# ruff: noqa: RUF001 -- Chinese report punctuation is intentional.
from __future__ import annotations

from pathlib import Path

from .execution import (
    BudgetLedger,
    _verify_freeze,
    atomic_json,
    checked_path,
    object_hash,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    verify_common_start,
    verify_format_report,
    verify_stage_plan,
)


def format_not_ready(report):
    """A complete failed independent panel is distinct from missing measurements."""
    if not report or report.get("gate_passed") is not False:
        return False
    if report.get("status") == "FORMAT_NOT_READY":
        return True
    coverage = report.get("coverage", {})
    if coverage.get("complete") is False:
        return False
    return coverage.get("complete") is True or any(
        cohort.get("status") == "FORMAT_NOT_READY" for cohort in report.get("cohorts", [])
    )


def verify_scoring_receipt(root, receipt):
    """Validate all current inputs and scored output against the saved receipt."""
    root = Path(root)
    source = Path(__file__).parent
    required = {
        "question_file_sha256": root / "data/questions.jsonl",
        "scored_outputs_sha256": root / "scoring/MEASUREMENT_AUDIT/SCORED_OUTPUTS.jsonl",
        "implementation_sha256": source / "scoring.py",
        "contracts_sha256": source / "contracts.py",
    }
    for field, path in required.items():
        if receipt.get(field) != sha256_file(path):
            raise PermissionError(f"Stale scoring receipt: {field}")
    for field, pattern in (
        ("raw_file_sha256", "outputs_*.jsonl"),
        ("field_score_file_sha256", "scores_*.jsonl"),
    ):
        records = receipt.get(field)
        if not isinstance(records, list):
            raise PermissionError(f"Scoring receipt omits {field}")
        expected = {
            str(p.relative_to(root)) for p in (root / "raw/MEASUREMENT_AUDIT").glob(pattern)
        }
        names = [item["name"] for item in records]
        if len(names) != len(set(names)) or set(names) != expected:
            raise PermissionError(f"Scoring receipt file set differs: {field}")
        for item in records:
            if sha256_file(checked_path(root, item["name"])) != item["sha256"]:
                raise PermissionError(f"Stale scoring input: {item['name']}")
    scored = read_jsonl(required["scored_outputs_sha256"])
    if receipt.get("scored_record_count") != len(scored):
        raise PermissionError("Scored output count differs from receipt")
    return receipt


def verify_measurement_complete(root, audit_plan, freeze):
    """Require the frozen request set, shard receipts and actual row identities."""
    if not audit_plan:
        return False
    root = Path(root)
    plan = verify_stage_plan(root, "MEASUREMENT_AUDIT", 0, audit_plan["shards"])
    if audit_plan != plan:
        raise PermissionError("Measurement dispatch plan changed")
    expected = {slot["request_id"]: slot for slot in plan["expected_slots"]}
    questions = {q["question_id"]: q for q in read_jsonl(root / "data/questions.jsonl")}
    directory = root / "raw/MEASUREMENT_AUDIT"
    expected_files = {f"outputs_{i}.jsonl" for i in range(plan["shards"])}
    if {p.name for p in directory.glob("outputs_*.jsonl")} - expected_files:
        raise PermissionError("Unregistered measurement shard")
    seen, complete = set(), True
    for shard in range(plan["shards"]):
        output_path = directory / f"outputs_{shard}.jsonl"
        claim_path = directory / f"SHARD_{shard}_STARTED.json"
        receipt_path = directory / f"SHARD_{shard}_COMPLETE.json"
        if not all(path.is_file() for path in (output_path, claim_path, receipt_path)):
            complete = False
        if claim_path.exists():
            claim = read_json(claim_path)
            if (
                claim.get("stage") != "MEASUREMENT_AUDIT"
                or claim.get("shard") != shard
                or claim.get("shards") != plan["shards"]
                or claim.get("freeze_hash") != plan["pre_freeze_hash"]
                or claim.get("stage_plan_hash") != object_hash(plan)
            ):
                raise PermissionError("Measurement shard claim identity mismatch")
        rows = read_jsonl(output_path)
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if (
                not output_path.exists()
                or receipt.get("output_hash") != sha256_file(output_path)
                or receipt.get("stage") != "MEASUREMENT_AUDIT"
                or receipt.get("shard") != shard
                or receipt.get("shards") != plan["shards"]
                or receipt.get("status") != "completed"
                or receipt.get("completed") != sum(row.get("status") == "completed" for row in rows)
            ):
                raise PermissionError("Measurement completion receipt identity mismatch")
        for row in rows:
            slot = expected.get(row.get("request_id"))
            if slot is None or row["request_id"] in seen or slot["shard"] != shard:
                raise PermissionError("Unexpected or duplicate measurement request identity")
            seen.add(row["request_id"])
            q = questions[slot["question_id"]]
            required = dict(
                stage="MEASUREMENT_AUDIT",
                question_id=slot["question_id"],
                sample_index=slot["sample_index"],
                seed=slot["seed"],
                model_hash=plan["model_hash"],
                processor_hash=freeze["processor_hash"],
                image_sha256=q["image_sha256"],
                prompt_hash=q["prompt_sha256"],
            )
            if any(row.get(key) != value for key, value in required.items()):
                raise PermissionError("Measurement raw row differs from frozen request")
            complete = complete and row.get("status") == "completed"
    return complete and len(expected) == 1536 and seen == set(expected)


def release_integrity(root, freeze, common, format_check, measure, audit_plan):
    """Read-only verification; preserve incomplete evidence and fail the READY claim."""
    issues, checked = [], []
    checks = []
    if freeze:
        checks.append(("pre_inference_freeze", lambda: _verify_freeze(root)))
    if common:
        checks.append(("common_start", lambda: verify_common_start(root)))
    if format_check:
        checks.append(("independent_format", lambda: verify_format_report(root, "FORMAT_CHECK")))
    if measure:
        checks.append(("scoring_receipt", lambda: verify_scoring_receipt(root, measure)))
    for name, check in checks:
        try:
            check()
            checked.append(name)
        except (OSError, ValueError, TypeError, KeyError, PermissionError) as error:
            issues.append(dict(artifact=name, error_type=type(error).__name__, detail=str(error)))
    audit_complete = False
    if audit_plan:
        try:
            audit_complete = verify_measurement_complete(root, audit_plan, freeze)
            checked.append("measurement_request_identity")
        except (OSError, ValueError, TypeError, KeyError, PermissionError) as error:
            issues.append(
                dict(artifact="measurement", error_type=type(error).__name__, detail=str(error))
            )
    return dict(status="FAIL" if issues else "PASS", checked=checked, issues=issues), audit_complete


def release_report(run_root, template_root):
    root = Path(run_root)

    def optional(relative):
        path = root / relative
        return read_json(path) if path.is_file() else None

    freeze = optional("manifests/PRE_INFERENCE_FREEZE.json")
    common = optional("manifests/COMMON_START.json")
    stop = optional("manifests/SER_J23_STOP_RECEIPT.json")
    format_check = optional("tables/FORMAT_CHECK_REPORT.json")
    base = optional("tables/FORMAT_BASE_TEST_REPORT.json")
    bridge = optional("engineering/BRIDGE_RECEIPT.json")
    engine = optional("engineering/ENGINE_COMPARE.json")
    measure = optional("tables/MEASUREMENT_AUDIT/SCORING_RECEIPT.json")
    processor = optional("manifests/PROCESSOR_READABILITY_REVIEW.json")
    audit_plan = optional("manifests/STAGE_PLAN_MEASUREMENT_AUDIT.json")
    integrity, audit_complete = release_integrity(
        root, freeze, common, format_check, measure, audit_plan
    )
    if integrity["status"] != "PASS":
        status = "BLOCKED_EVIDENCE_INTEGRITY"
    elif format_not_ready(format_check):
        status = "FORMAT_NOT_READY"
    elif not freeze:
        status = "BLOCKED_PRE_INFERENCE_FREEZE"
    elif not common or not format_check or format_check.get("gate_passed") is not True:
        status = "BLOCKED_FORMAT_CHECK_INCOMPLETE"
    elif (
        not measure
        or measure.get("status") != "SCORED"
        or measure.get("scored_record_count") != 1536
        or not audit_complete
    ):
        status = "BLOCKED_MEASUREMENT_INCOMPLETE"
    elif not engine or engine.get("status") != "PASS":
        status = "ENGINE_REPRO_NOT_READY"
    else:
        status = "READY_FOR_DEV_REVIEW"
    outputs = []
    for path in sorted((root / "raw").glob("*/outputs_*.jsonl")):
        outputs.extend(read_jsonl(path))
    totals = BudgetLedger(root).totals()
    allocations = [read_json(p) for p in sorted((root / "accounting/allocations").glob("*.json"))]
    actual = sum(
        r["terminal"]["elapsed_seconds"] * r["gpus"] / 3600
        for r in allocations
        if r["status"] == "TERMINAL_VERIFIED"
    )
    unresolved = [r["allocation_key"] for r in allocations if r["status"] != "TERMINAL_VERIFIED"]
    proposal = read_json(Path(template_root) / "DEV_FREEZE_PROPOSAL.template.json")
    proposal.update(
        authorized=False,
        status="NOT_RUN_REQUIRES_SEPARATE_REVIEW",
        model_common_start_hash=common.get("model_hash") if common else None,
        processor_hash=freeze.get("processor_hash") if freeze else None,
        protocol_hash=freeze.get("protocol_hash") if freeze else None,
        actual_cost_envelope={
            "physical_counts_and_allocation_estimates": totals,
            "gpu_hours_policy": "ACCOUNTING_ONLY",
            "max_allocated_gpu_hours": None,
            "verified_allocated_gpu_hours": actual,
            "unknown_allocations": unresolved,
            "future_cost_extrapolation": "NOT_CERTIFIED",
        },
    )
    proposal["open_fields"] = [k for k, v in proposal.items() if v is None]
    proposal["open_fields"] += [
        "weights_and_utility.w/v/lambda/epsilon/eta/cost_unit",
        "new_independent_data_roots",
        "independent_user_authorization",
    ]
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(report_dir / "DEV_FREEZE_PROPOSAL.json", proposal)
    summary = {
        "time": utc_now(),
        "status": status,
        "evidence_integrity": integrity,
        "terminal_state": "STOP_FOR_REVIEW",
        "dev_authorized": False,
        "raw_model_responses": len(outputs),
        "completed_responses": sum(r.get("status") == "completed" for r in outputs),
        "failed_responses": sum(r.get("status") == "failed" for r in outputs),
        "budget_reservations": totals,
        "gpu_hours_are_execution_gate": False,
        "verified_allocated_gpu_hours": actual,
        "unknown_allocations": unresolved,
        "base_format_status": base.get("status") if base else "NOT_RUN",
        "independent_format_status": format_check.get("status") if format_check else "NOT_RUN",
        "bridge_status": bridge.get("status") if bridge else "NOT_RUN",
        "engine_status": engine.get("status") if engine else "NOT_RUN",
    }
    atomic_json(report_dir / "FINAL_AUDIT_STATUS.json", summary)
    text = f"""# MM-CORE F1 审计报告

状态：**{status}**。交付边界 `STOP_FOR_REVIEW`，`MM-DEV authorized=false`。
证据完整性：{integrity["status"]}；具体核验项与失败原因见 `FINAL_AUDIT_STATUS.json`。

## 停止核验与来源

SER-J23：{stop.get("status") if stop else "UNKNOWN"}。本轮没有重启旧符号实验。
历史分支、原始来源与缺失见 `../manifests/ASSET_CONTRACTS.csv` 和 `ASSET_MANIFEST.json`。
采用新协议路径 B；旧执行器答案、分步回答与本轮同次双字段输出分开。

## 数据与实际视觉处理

授权原型为 36 个保守根家系、288 张图、864 题；面板分别为 96/96/384/192/96 题。
这些是生成器工程规模，不能当成独立训练状态数或统计功效保证。
真实 processor 视觉审查状态：{processor.get("status") if processor else "NOT_RUN"}。
可构造性、候选拒绝、split 去重及处理后尺寸见 `../manifests/GENERATOR_REPORT.json`、
`ROOT_SPLIT_AUDIT.json`、`PROCESSOR_PREFLIGHT.json`。

## 格式、桥接与测量

基座格式：{summary["base_format_status"]}；独立格式：{summary["independent_format_status"]}；
桥接：{summary["bridge_status"]}。唯一桥接终点若存在，属于使用真值监督的新公共起点。
原始推理完成 {summary["completed_responses"]} 条、技术失败 {summary["failed_responses"]} 条。
未产生回答的样本不填入语义 000 格；缺失、截断、域外和失败保留。

逐回答原文/token 在 `../raw/`，精确评分在 `../scoring/`，带计数/分母的六格、容差八格、
覆盖、误差、S、同提示/跨提示一致性、奖励支持与字段 NLL 在 `../tables/`。
S 的零误差分母记 missing；稳定性差异允许负估计；gold NLL 属于 teacher forcing。
本轮没有检验这些摘要对未见训练收益或决策增益的预测效果。

## 工程恢复与成本

ENGINE：{summary["engine_status"]}。只有真实连续/重启恢复逐字段精确一致才记 PASS。
CPU 测试不是 CUDA 回执；ENGINE 模型不替换公共起点，不进入科学响应表。

最多五张 Pro 6000。GPU 小时仅记账，不设总量上限或阶段门槛。
登记的调度分配时限估计合计 {totals["allocated_gpu_hours"]:.6f} GPU 小时，
当前终态调度记录核验实际 {actual:.6f} GPU 小时；尚未核验 allocations：{unresolved}。
生成尝试 {totals["completion_attempts"]} / 4096，
物理更新 {totals["physical_optimizer_updates"]} / 64，
额外前向 {totals["extra_forward_sequences"]} / 8192。预留包括加载、空闲、失败与未知工作；
保守预留与已核验实际成本分别报告，不以网络超时释放额度或重提交。

## 后续冻结草案及停止

`DEV_FREEZE_PROPOSAL.json` 保留 Hp=H=32、4准备、5状态、30响应、1088逻辑更新的未授权骨架。
尚缺字段保持 null/开放项，不把模板或成本推算当作已执行结果。当前累计科学准备/响应更新为0。
交付后停止，不自动运行 MM-DEV、MM-LOCK、MM-CAL 或在线控制器。
"""
    (report_dir / "FINAL_AUDIT_REPORT_zh.md").write_text(text, encoding="utf-8")
    (report_dir / "DEV_FREEZE_PROPOSAL_zh.md").write_text(
        "# MM-DEV 冻结补充表草案\n\nauthorized=false；本轮未运行 MM-DEV。\n\n"
        "固定骨架：Hp=H=32，4条准备，5个状态，30条响应，1088次逻辑更新。\n\n"
        "审计终态："
        + status
        + "。完整模型身份和成本见 JSON；其余开放字段：\n\n"
        + "\n".join("- " + key for key in proposal["open_fields"])
        + "\n",
        encoding="utf-8",
    )
    atomic_json(
        report_dir / "REPORT_MANIFEST.json",
        {
            str(p.relative_to(root)): sha256_file(p)
            for p in sorted(report_dir.iterdir())
            if p.is_file() and p.name != "REPORT_MANIFEST.json"
        },
    )
    return summary
