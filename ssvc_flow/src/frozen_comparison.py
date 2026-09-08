"""Predeclared support strata for auxiliary, descriptive frozen comparisons."""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .core import canonical_hash, write_json

BIN_EDGES = (0, 0.25, 0.5, 0.75, 1)
MODELS = ("qwen25vl_3b", "qwen35_9b", "qwen25vl_7b")
INTERFACES = {"N": ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH"), "L": ("collision", "separating")}
MEAN_FIELDS = ("pX", "pS", "pW", "pI", "v", "qX", "qS", "pA", "truth_given_answer")


def _bin(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("support coordinates must be finite probabilities")
    return min(int(value * 4), 3)


def _label(index):
    return f"[{BIN_EDGES[index]:g},{BIN_EDGES[index + 1]:g}{']' if index == 3 else ')'}"


def _mean(values):
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def build_support_comparison(subtasks):
    """Consume previously verified metrics, never raw completions or greedy rows.

    Cell membership is model-specific: matching bin labels does not establish
    paired prompt populations. All original overall summaries remain available.
    """
    tasks, cells, overall = {}, defaultdict(list), []
    for task in subtasks:
        model, track, metrics = task["model_key"], task["track"], task["metrics"]
        if model not in MODELS or track not in INTERFACES or metrics.get("track") != track:
            raise ValueError("unknown or inconsistent comparison model/track")
        if (model, track) in tasks:
            raise ValueError("auxiliary comparison requires one verified bank per model/track")
        tasks[model, track] = task
        overall.append(
            {
                "model_key": model,
                "track": track,
                "source": task.get("path"),
                "sampling": copy.deepcopy(metrics["overall"]),
                "greedy": copy.deepcopy(metrics["greedy"]["overall"]),
            }
        )
        for prompt_id, prompt in metrics["by_prompt"].items():
            interface, pooled = prompt["interface"], prompt["pooled"]
            if interface not in INTERFACES[track]:
                raise ValueError("unexpected interface in comparison panel")
            px, valid = _bin(pooled["pX"]), _bin(pooled["v"])
            if pooled["pX"] > pooled["v"]:
                raise ValueError("pX cannot exceed valid support v")
            cells[track, interface, px, valid, model].append((prompt_id, prompt))
    rows = []
    for track in sorted({key[1] for key in tasks}):
        signatures = {
            canonical_hash(task["comparison_identity"])
            for (model, candidate_track), task in tasks.items()
            if candidate_track == track
        }
        comparable = len(signatures) == 1
        for interface in INTERFACES[track]:
            for px in range(4):
                for valid in range(4):
                    members = {model: cells[track, interface, px, valid, model] for model in MODELS}
                    missing = [model for model, entries in members.items() if not entries]
                    common = set.intersection(
                        *[{pid for pid, _ in entries} for entries in members.values()]
                    )
                    coverage = (
                        "INCOMPARABLE_DATA_OR_PROTOCOL"
                        if not comparable
                        else "NONCOMMON_SUPPORT"
                        if missing
                        else "COMMON_SUPPORT"
                    )
                    for model, entries in members.items():
                        prompts = [prompt for _, prompt in entries]
                        undefined = sum(prompt["pooled"].get("qX") is None for prompt in prompts)
                        rows.append(
                            {
                                "track": track,
                                "interface": interface,
                                "model_key": model,
                                "pX_bin": _label(px),
                                "v_bin": _label(valid),
                                "prompt_count": len(prompts),
                                "independent_scene_count": len(
                                    {p["base_scene_id"] for p in prompts}
                                ),
                                "rollout_count": sum(p["rollout_count"] for p in prompts),
                                "coverage_status": coverage,
                                "missing_models": missing,
                                "common_prompt_count": len(common) if comparable else None,
                                "qX_defined_prompt_count": len(prompts) - undefined,
                                "qX_NA_fraction": undefined / len(prompts) if prompts else None,
                                **{
                                    f"mean_prompt_{field}": _mean(
                                        [p["pooled"].get(field) for p in prompts]
                                    )
                                    for field in MEAN_FIELDS
                                },
                            }
                        )
    return {
        "schema_version": 1,
        "status": "AUXILIARY_DESCRIPTIVE",
        "bin_edges": list(BIN_EDGES),
        "bin_rule": "left-closed right-open; final bin includes 1",
        "required_models": list(MODELS),
        "overall": overall,
        "rows": rows,
        "scope": "model-specific conditioning on estimated pX/v; "
        "not paired causal effects or training outcomes",
        "mean_rule": "equal prompt weight; undefined conditional ratios excluded "
        "with explicit NA fraction",
        "greedy_rule": "overall preserved separately; no greedy outputs included in support cells",
    }


def write_support_comparison(out, subtasks):
    result = build_support_comparison(subtasks)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        out / name
        for name in (
            "frozen_support_comparison.json",
            "frozen_support_comparison.csv",
            "frozen_support_comparison_zh.md",
        )
    ]
    write_json(paths[0], result)
    columns = (
        list(result["rows"][0])
        if result["rows"]
        else ["track", "interface", "model_key", "prompt_count"]
    )
    with paths[1].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {**row, "missing_models": json.dumps(row["missing_models"])} for row in result["rows"]
        )
    lines = [
        "# 冻结支持区间辅助比较",
        "",
        "这是辅助描述性分析，不能解释为规模因果效应或训练收益。",
        "固定 pX/v 二维分箱边界为 0、0.25、0.5、0.75、1；左闭右开，末箱包含 1。",
        "按轨道和接口分开，每个模型各自按观测概率分箱；相同箱不保证属于同一组 prompt。",
        "COMMON_SUPPORT 只表示三个模型在该箱均有观测且数据/协议一致；共同 prompt 数单列。",
        "NONCOMMON_SUPPORT 的缺失模型、空箱及数据/协议不可比状态完整保留在 CSV/JSON。",
        "全部原始 overall 统计保留在 JSON，不以区间子集替换。greedy 独立保留且不参与分箱。",
        "均值对 prompt 等权；未定义 qX 不作为零，另报定义数及 NA 比例。",
        "",
        "| 轨道 | 接口 | 模型 | pX区间 | v区间 | prompt数 | 覆盖状态 |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    lines.extend(
        f"| {r['track']} | {r['interface']} | {r['model_key']} | {r['pX_bin']} | "
        f"{r['v_bin']} | {r['prompt_count']} | {r['coverage_status']} |"
        for r in result["rows"]
        if r["prompt_count"]
    )
    paths[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
