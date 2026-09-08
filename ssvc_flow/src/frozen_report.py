"""Typed exports of frozen observations without manufacturing later phases."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .core import write_json

EXPORT_FILES = (
    "metrics.json",
    "metrics_by_scene.parquet",
    "metrics_by_group.csv",
    "results_report_zh.md",
    "table_status.json",
)


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, list):
            result[name] = json.dumps(item, ensure_ascii=False, allow_nan=False)
        else:
            result[name] = item
    return result


def _records(metrics, key, identity):
    result = []
    for decode_mode, block in (("sample", metrics), ("greedy", metrics["greedy"])):
        for name, values in sorted(block[key].items()):
            result.append({**_flatten(values), identity: name, "decode_mode": decode_mode})
    return result


def write_frozen_report(out, rows, metrics, audit):
    """Export one model/track subtask; caller finalizes its hash manifest/status.

    Raw rows stay in the caller's run store; this function never embeds raw text
    or completions into a report. Missing denominators remain null/empty.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kind = audit.get("execution_kind", "UNKNOWN")
    if kind not in ("REAL_CUDA_MODEL", "CPU_FAKE_ADAPTER_FIXTURE"):
        raise ValueError("frozen report requires an explicit execution_kind")
    # Validate the whole metrics document before writing any result files.
    json.dumps(metrics, allow_nan=False)
    scenes = _records(metrics, "by_prompt", "prompt_id")
    groups = _records(metrics, "by_group", "group")
    if not scenes or not groups:
        raise ValueError("frozen report requires prompt and group metrics")
    fields = sorted({key for row in scenes for key in row})
    table = pa.Table.from_pylist([{key: row.get(key) for key in fields} for row in scenes])
    pq.write_table(table, out / "metrics_by_scene.parquet")
    columns = sorted({key for row in groups for key in row})
    with (out / "metrics_by_group.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(groups)
    write_json(out / "metrics.json", metrics)
    table_status = {
        name: {"status": "OBSERVED_SUBTASK", "execution_kind": kind}
        for name in ("metrics_by_scene.parquet", "metrics_by_group.csv")
    }
    table_status.update(
        {
            name: {
                "status": "NOT_RUN",
                "reason": "P3 frozen evaluation performs no training or intervention forks",
            }
            for name in ("training_steps.parquet", "paired_effects.csv", "flow_forks.parquet")
        }
    )
    write_json(out / "table_status.json", table_status)
    overall = metrics["overall"]
    lines = [
        "# P3 冻结评估子任务结果",
        "",
        f"执行类型：`{kind}`。模型：`{audit.get('model_key', 'UNKNOWN')}`；"
        f"轨道：`{audit.get('track', 'UNKNOWN')}`。",
        "",
        "本报告只覆盖单个模型/轨道子任务，不能据此宣告整个 P3 通过。",
        "CPU_FAKE_ADAPTER_FIXTURE 是程序验证数据，不能作为真实模型科学结果。"
        if kind == "CPU_FAKE_ADAPTER_FIXTURE"
        else "本报告记录真实模型冻结状态下的观测；不包含训练效果。",
        "",
        f"随机采样 prompt 数：{overall.get('prompt_count', 'NA')}；"
        f"输出数：{overall.get('rollout_count', 'NA')}。",
        f"greedy 输出数：{metrics['greedy']['overall'].get('rollout_count', 'NA')}，"
        "独立统计，不混入随机采样概率。",
        "",
        "## 随机采样观测",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
    ]
    for name, value in overall.get("pooled", {}).items():
        lines.append(f"| {name} | {'NA' if value is None else value} |")
    lines.extend(
        [
            "",
            "每个 prompt 和 decode_mode 各占 metrics_by_scene.parquet 一行；"
            "固定分组及空组见 metrics_by_group.csv。",
            "分母为零时 JSON/Parquet 使用 null，CSV 使用空值。"
            "重复采样不是独立场景，区间是条件于 prompt 的逐项区间。",
            "输出长度列表以 JSON 字符串保存于表格字段；完整结构保留在 metrics.json。",
            "",
            "## 尚未执行",
            "",
            "训练更新、配对训练效应和干预 forks 均为 NOT_RUN。",
            "这些冻结观测不证明奖励安全性，也不构成模型规模造成差异的因果证据。",
            "下一步先汇齐并核对计划要求的模型/轨道子任务，再依据验收和预算决定后续实验。",
            "",
        ]
    )
    (out / "results_report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return [out / name for name in EXPORT_FILES]
