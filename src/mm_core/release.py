"""Evidence-derived audit release; missing work is explicit and DEV stays unauthorized."""

# ruff: noqa: RUF001 -- Chinese report punctuation is intentional.
from __future__ import annotations

from pathlib import Path

from .execution import BudgetLedger, atomic_json, read_json, read_jsonl, sha256_file, utc_now


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
    audit_rows = []
    for path in sorted((root / "raw/MEASUREMENT_AUDIT").glob("outputs_*.jsonl")):
        audit_rows.extend(read_jsonl(path))
    audit_complete = (
        bool(audit_plan)
        and len(audit_rows) == 1536
        and all(row.get("status") == "completed" for row in audit_rows)
        and {row["request_id"] for row in audit_rows}
        == {row["request_id"] for row in audit_plan["expected_slots"]}
    )
    if format_check and format_check.get("status") == "FORMAT_NOT_READY":
        status = "FORMAT_NOT_READY"
    elif not freeze:
        status = "BLOCKED_PRE_INFERENCE_FREEZE"
    elif not measure or measure.get("status") == "NO_OUTPUTS" or not audit_complete:
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
            "reserved_caps_consumed": totals,
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
        "terminal_state": "STOP_FOR_REVIEW",
        "dev_authorized": False,
        "raw_model_responses": len(outputs),
        "completed_responses": sum(r.get("status") == "completed" for r in outputs),
        "failed_responses": sum(r.get("status") == "failed" for r in outputs),
        "budget_reservations": totals,
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

最多五张 Pro 6000，累计上限 8 GPU 小时。累计已预留 {totals["allocated_gpu_hours"]:.6f} GPU 小时，
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
