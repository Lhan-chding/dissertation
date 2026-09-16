"""A compact evidence inventory and factual V4 report from completed artifacts.

This reader never treats a completion marker, numerical parity, or a fixture as
evidence that a response model is accurate. It does not read model checkpoints,
raw tokens, or the independent reference arrays again.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import atomic_json


def _read(path):
    return json.loads(Path(path).read_text())


def _relative(path, root):
    return str(Path(path).relative_to(root))


def _number(value):
    return "未提供" if value is None else f"{value:.8g}" if isinstance(value, float) else str(value)


def _text(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def summarize(root, out):
    """Inventory measured evidence, preserving missing and unresolved results.

    Evaluation metrics are accepted only alongside their own completion marker.
    Integrity was checked when those immutable artifacts were imported/written;
    this report does not repeatedly hash their entire raw dependency graph.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    out.mkdir(parents=True, exist_ok=True)
    cpu_path = root / "cpu" / "RUN_SUMMARY.json"
    cpu = (
        _read(cpu_path)
        if cpu_path.is_file() and (cpu_path.parent / "COMPLETE.json").is_file()
        else None
    )
    bridge_path = root / "B_MERGED.json"
    bridge = _read(bridge_path) if bridge_path.is_file() else None
    tasks = []
    for path in sorted((root / "tasks").glob("*/COMPLETE.json")):
        value = _read(path)
        tasks.append(
            {
                "path": _relative(path, root),
                "task_id": path.parent.name,
                "status": value.get("status"),
                "kind": value.get("kind"),
                "task": value.get("task"),
                "execution_kind": value.get("execution_kind"),
                "scientific_status": value.get("scientific_status", "NOT_CERTIFIED"),
            }
        )
    workers = [_read(path) for path in sorted((root / "workers").glob("*/LATEST.json"))]
    evaluations, ignored, metrics = [], [], []
    for path in sorted(root.rglob("METRICS.json")):
        if not (path.parent / "COMPLETE.json").is_file():
            ignored.append({"path": _relative(path, root), "reason": "NO_COMPLETE_MARKER"})
            continue
        value = _read(path)
        if value.get("kind") != "V4_EVALUATION":
            continue
        core = path.parent / value["core_results_file"]
        if core.name != value["core_results_file"] or not core.is_file():
            raise ValueError("Missing or nonlocal evaluation core table")
        binding = value.get("binding", {})
        record = {
            "path": _relative(path, root),
            "core_results": _relative(core, root),
            "binding": binding,
            "bootstrap_seed_order": value.get("bootstrap_seed_order", []),
            "reference_read_scope": value.get("reference_read_scope"),
            "online_ssvc": value.get("online_ssvc", "NOT_CERTIFIED"),
        }
        evaluations.append(record)
        metrics.extend({**row, "source_metrics": record["path"]} for row in value["summary"])
    actual = [
        e
        for e in evaluations
        if e["binding"].get("execution_kind")
        in {"REAL_CUDA_MODEL", "REAL_CUDA_TRAINING", "REAL_CUDA_RESPONSE"}
    ]
    first_maps = [
        t
        for t in tasks
        if t["kind"] == "map"
        and (t["task"] or {}).get("stage") == "C"
        and t["status"] == "COMPLETED"
        and t["execution_kind"] in {"REAL_CUDA_MODEL", "REAL_CUDA_TRAINING", "REAL_CUDA_RESPONSE"}
    ]
    summary = {
        "schema": "ssvc-v4-run-summary-1",
        "status": "EVIDENCE_SNAPSHOT",
        "root": str(root),
        "cpu": cpu,
        "bridge": bridge,
        "tasks": tasks,
        "workers": workers,
        "evaluations": evaluations,
        "incomplete_evaluations": ignored,
        "completed_first_map_collections": len(first_maps),
        "real_cuda_evaluation_count": len(actual),
        "reliability_decision": "REQUIRES_FROZEN_CALIBRATION_AND_INDEPENDENT_TEST",
        "online_ssvc": "NOT_RUN_NOT_CERTIFIED",
        "missing_artifacts": [],
        "integrity_policy": "READ_COMPLETED_DERIVED_ARTIFACTS_NO_RAW_REHASH",
    }
    summary["completed_d_map_collections_by_role"] = {
        role: sum(
            t["kind"] == "map"
            and t["status"] == "COMPLETED"
            and (t["task"] or {}).get("stage") == "D"
            and (t["task"] or {}).get("role") == role
            and t["execution_kind"]
            in {"REAL_CUDA_MODEL", "REAL_CUDA_TRAINING", "REAL_CUDA_RESPONSE"}
            for t in tasks
        )
        for role in ("development", "calibration", "test")
    }
    if cpu is None:
        summary["missing_artifacts"].append("A_SERVER_CPU_DIAGNOSIS")
    if bridge is None:
        summary["missing_artifacts"].append("B_TWO_GPU_BRIDGE")
    if len(first_maps) < 4:
        summary["missing_artifacts"].append("C_FOUR_REAL_ORIGIN_COLLECTIONS")
    if not actual:
        summary["missing_artifacts"].append("REAL_9B_EVALUATION")
    for role, expected in (("development", 6), ("calibration", 6), ("test", 18)):
        if summary["completed_d_map_collections_by_role"][role] < expected:
            summary["missing_artifacts"].append(f"D_{role.upper()}_COLLECTIONS")
    # Merge the existing per-query CSVs without expanding raw arrays or silently
    # mixing away origin, model, observation, reference, or role identifiers.
    target = out / "CORE_RESULTS.csv"
    fields, sources = [], []
    for evaluation in evaluations:
        source = root / evaluation["core_results"]
        with source.open(newline="") as stream:
            header = next(csv.reader(stream))
        fields.extend(name for name in header if name not in fields)
        sources.append((source, evaluation["path"]))
    with target.with_suffix(".csv.pending").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_metrics", *fields])
        writer.writeheader()
        rows = 0
        for source, name in sources:
            with source.open(newline="") as payload:
                for row in csv.DictReader(payload):
                    writer.writerow({**row, "source_metrics": name})
                    rows += 1
    target.with_suffix(".csv.pending").replace(target)
    summary["core_results_rows"] = rows
    summary["core_results_status"] = (
        "DERIVED_FROM_COMPLETED_EVALUATIONS" if rows else "NO_EVALUATION_ROWS"
    )
    atomic_json(out / "RUN_SUMMARY.json", summary)
    _bridge_report(bridge, tasks, workers, root, out)
    _decision_report(summary, metrics, out)
    return {
        "status": summary["status"],
        "out": str(out),
        "core_results_rows": rows,
        "real_cuda_evaluation_count": len(actual),
        "missing_artifacts": summary["missing_artifacts"],
    }


def _bridge_report(bridge, tasks, workers, root, out):
    lines = [
        "# V4 真实 9B 桥接记录",
        "",
        "本文件记录已有桥接输出；技术检查完成状态与概率响应模型误差是不同的记录。",
        "",
        f"合并状态：`{bridge.get('status') if bridge else 'NOT_MEASURED'}`。",
        "",
    ]
    if bridge:
        lines += [
            "## 两张卡的同策略比较",
            "",
            "```json",
            json.dumps(bridge.get("parity"), ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    for task in tasks:
        if task["kind"] != "bridge":
            continue
        value = _read(root / task["path"])
        lines += [
            f"## {task['task_id']}",
            "",
            f"执行类型：`{task['execution_kind']}`。",
            "",
            "```json",
            json.dumps(
                {
                    key: value.get(key)
                    for key in ("gradient", "null", "scoring_path", "timing", "memory")
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    lines += ["## Worker 实测资源记录", ""]
    for worker in workers:
        lines += [
            "```json",
            json.dumps(
                {
                    key: worker.get(key)
                    for key in (
                        "worker_id",
                        "status",
                        "execution_kind",
                        "model_load_seconds",
                        "elapsed_seconds",
                        "phase_measurements",
                        "forward_calls",
                        "generation_calls",
                        "optimizer_and_backward_costs",
                    )
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    lines += [
        "各 phase 的 task_id 区分桥接与后续源训练、响应测量；worker 总耗时不能全部归入桥接。",
        "没有提供的耗时或显存字段表示该记录未包含测量，不能用配置中的工作量代替实测时间。",
        "",
    ]
    _text(out / "GPU_BRIDGE_RESULT.md", "\n".join(lines))


def _decision_report(summary, metrics, out):
    lines = [
        "# V4 概率响应建模记录",
        "",
        "## 当前证据范围",
        "",
        f"已完成的四原点阶段采集：{summary['completed_first_map_collections']}/4。",
        f"带真实 CUDA 执行标识的评估文件：{summary['real_cuda_evaluation_count']}。",
        f"汇总逐 query 行数：{summary['core_results_rows']}。",
        "",
        "本快照不将文件完成、CPU 小型测试、数值路径一致或零和约束满足记为模型可靠性证明。",
        "",
        "在线 SSVC：未执行，未认证。",
        "",
        "## 已记录的误差",
        "",
        "每行保留方法、实验设计、数据角色和观测范围。NRMSE 为 RMSE 除以实际响应信号 RMS；"
        "信号为零时未定义。q95 是残差分位数，不是置信区间。",
        "",
        "|方法|设计|角色|通道|范围|总数|可解析数|RMSE|q95绝对误差|NRMSE|参考SE RMS|",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if row.get("channel") not in {"RAW4", "group_pX", "group_v", "X", "I"} or row.get(
            "scope"
        ) not in {"all", "nonalias"}:
            continue
        keys = (
            "method",
            "design_id",
            "role",
            "channel",
            "scope",
            "all_count",
            "finite_resolved_count",
            "rmse",
            "q95_absolute",
            "nrmse",
            "reference_se_rms",
        )
        lines.append("|" + "|".join(_number(row.get(key)) for key in keys) + "|")
    if not metrics:
        lines += ["", "尚无已完成的 V4 评估数据；误差范围未测得。"]
    lines += [
        "",
        "## 信息、维数和校准记录",
        "",
        "完整 LoRA 参数、实际有效权重、完整方向导数与函数签名分别按其信息来源记录。"
        "模型使用的维数、校准 bank 数、正则化与观测预算以各评估文件的 binding/design 为准；"
        "缺失字段不能从方案配置推定为已经运行。",
        "",
        "两种子结果仅作描述；三校准种子不能提供分布无关的 95% 覆盖保证。"
        "独立测试只评价冻结选择。空间外 query 留在分母内，rho 只作为连续诊断。",
        "",
        "## 未解析与失效记录",
        "",
        "参考精度不足时保留 REFERENCE_UNRESOLVED；未知预测保留 UNKNOWN，不能补零。"
        "同策略别名、零信号与非零响应分开统计。BF16 前向误差不会因 FP64 累计自动消失。",
        "",
        "尚缺证据：" + ("、".join(summary["missing_artifacts"]) or "见阶段任务与独立测试记录"),
        "",
        "完整原件路径、任务状态、评估绑定见 RUN_SUMMARY.json；逐 query 数据见 CORE_RESULTS.csv。",
        "",
    ]
    _text(out / "MODELING_DECISION_V4_zh.md", "\n".join(lines))
