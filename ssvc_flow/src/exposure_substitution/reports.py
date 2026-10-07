"""Release-gated SER-J2 evidence reports; missing endpoints stay missing."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .statistics import (
    ARMS,
    PARENTS,
    IncompleteDataError,
    analyze_confirmation,
    safe_ratio,
    summarize_tasks,
)

CELL_FIELDS = (
    "checkpoint_id",
    "parent",
    "block",
    "arm",
    "step",
    "panel",
    "family",
    "center",
    "corrupted_index",
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _json(path: Path, data: Any) -> None:
    _atomic_text(
        path, json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def _jsonl(path: Path, rows) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows
        ),
    )


def _csv(path: Path, rows) -> None:
    import io

    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (list, dict)) or value is None
                else value
                for key, value in row.items()
            }
        )
    _atomic_text(path, buffer.getvalue())


def _validate_release(receipt):
    if (
        not isinstance(receipt, dict)
        or not (
            receipt.get("all_registered_terminal") is True
            or receipt.get("all_registered_models_terminal") is True
        )
        or receipt.get("registered_training_jobs") != 18
    ):
        raise ValueError("E_CONFIRM report requires every registered training endpoint terminal")
    digest = receipt.get("analysis_code_sha256", "")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("E_CONFIRM report requires a frozen analysis SHA256 identity")


def summarize_cells(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in CELL_FIELDS)].append(row)
    cells, values, projections, anchors, operands, joints = [], [], [], [], [], []
    for key, group in groups.items():
        metadata = dict(zip(CELL_FIELDS, key, strict=True))
        metadata.update(
            n_tasks=len({r["task_id"] for r in group}),
            n_roots=len({r["root_id"] for r in group}),
            n_draws=len(group),
        )
        n = len(group)
        counts = {
            name: sum(bool(r[name]) for r in group)
            for name in (
                "X",
                "S",
                "W",
                "I",
                "parse_ok",
                "domain_ok",
                "out_of_domain",
                "truncated",
                "value_present",
                "value_present_but_not_X",
                "projection_success",
                "mixed_anchor_public",
                "mixed_anchor_audit",
                "center_correct",
                "center_correct_but_not_X",
            )
        }
        if counts["X"] != counts["value_present"] - counts["value_present_but_not_X"]:
            raise ValueError("Cell value/extra count decomposition violated")
        cells.append(
            {
                **metadata,
                **{name + "_count": c for name, c in counts.items()},
                **{"p_" + name: c / n for name, c in counts.items()},
                **{
                    name + "_sum_legal": sum(r[name] for r in group if r["domain_ok"])
                    for name in ("M", "B", "L", "F")
                },
                "legal_edit_denominator": counts["domain_ok"],
            }
        )
        value = {
            **metadata,
            "X_count": counts["X"],
            "value_present_count": counts["value_present"],
            "extra_error_count": counts["value_present_but_not_X"],
            "p_X": counts["X"] / n,
            "p_value_present": counts["value_present"] / n,
            "p_extra_error": counts["value_present_but_not_X"] / n,
            "extra_out_of_domain_count": sum(
                r["value_present_but_not_X"] and r["out_of_domain"] for r in group
            ),
            "denominator_all": n,
            "denominator_parsed": counts["parse_ok"],
            "denominator_legal": counts["domain_ok"],
            "extra_given_parsed": safe_ratio(counts["value_present_but_not_X"], counts["parse_ok"]),
            "extra_legal_count": sum(
                r["value_present_but_not_X"] and r["domain_ok"] for r in group
            ),
            "extra_given_legal": safe_ratio(
                sum(r["value_present_but_not_X"] and r["domain_ok"] for r in group),
                counts["domain_ok"],
            ),
        }
        values.append(value)
        projections.append(
            {
                **metadata,
                "raw_X_count": counts["X"],
                "projection_success_count": counts["projection_success"],
                "raw_p_X": counts["X"] / n,
                "projection_p_success": counts["projection_success"] / n,
                "projection_gain": (counts["projection_success"] - counts["X"]) / n,
                "value_present_count": counts["value_present"],
                "verifier_calls": sum(r["projection_verifier_calls"] for r in group),
                "cpu_seconds": sum(r["projection_cpu_seconds"] for r in group),
                "equivalence_violations": sum(
                    r["projection_success"] != r["value_present"] for r in group
                ),
                "uses_audit_truth": False,
                "changes_raw_X": False,
            }
        )
        for k in range(4):
            changed_correct = sum(r["leaf_edited_and_center_correct"][k] for r in group)
            conditional_num = sum(
                r["leaf_edited_and_center_correct"][k] and r["center_correct_but_not_X"]
                for r in group
            )
            den = counts["center_correct_but_not_X"]
            anchors.append(
                {
                    **metadata,
                    "coordinate": k,
                    "public_anchor_count": sum(r["mixed_anchor_public_bits"][k] for r in group),
                    "audit_anchor_count": sum(r["mixed_anchor_audit_bits"][k] for r in group),
                    "leaf_edited_and_center_correct_count": changed_correct,
                    "p_leaf_edited_and_center_correct": changed_correct / n,
                    "conditional_numerator": conditional_num,
                    "conditional_denominator": den,
                    "p_leaf_edited_given_center_correct_but_not_X": safe_ratio(
                        conditional_num, den
                    ),
                    "denominator_all": n,
                }
            )
        signature_counts = Counter()
        signature_denominators = Counter()
        for row in group:
            for signature in row["operand_signatures"]:
                for scope, include in (
                    ("all_proposed_wrong_leaves", signature["wrong_value"]),
                    (
                        "forward_target_right_location_single_edit",
                        signature["forward_target"] and signature["right_location_single_edit"],
                    ),
                    (
                        "forward_target_wrong_right_location_single_edit",
                        signature["forward_target"]
                        and signature["right_location_single_edit"]
                        and signature["wrong_value"],
                    ),
                ):
                    if include:
                        signature_denominators[signature["coordinate"], scope] += 1
                        for name in (
                            "wrong_partner",
                            "other_relation_constant",
                            "off_by_2",
                            "equals_hub",
                        ):
                            signature_counts[signature["coordinate"], scope, name] += bool(
                                signature[name]
                            )
        for k in range(4):
            for scope in (
                "all_proposed_wrong_leaves",
                "forward_target_right_location_single_edit",
                "forward_target_wrong_right_location_single_edit",
            ):
                den = signature_denominators[k, scope]
                for name in ("wrong_partner", "other_relation_constant", "off_by_2", "equals_hub"):
                    num = signature_counts[k, scope, name]
                    operands.append(
                        {
                            **metadata,
                            "coordinate": k,
                            "scope": scope,
                            "signature": name,
                            "numerator": num,
                            "denominator": den,
                            "rate": safe_ratio(num, den),
                            "overlaps_preserved": True,
                        }
                    )
        counter = Counter(
            (
                r["event"],
                tuple(r["edit_mask"]) if r["edit_mask"] is not None else None,
                tuple(r["mixed_anchor_public_bits"]),
                tuple(r["mixed_anchor_audit_bits"]),
                r["value_present"],
                r["parse_ok"],
                r["domain_ok"],
            )
            for r in group
        )
        for (event, mask, public, audit, present, parsed, legal), count in sorted(
            counter.items(), key=repr
        ):
            joints.append(
                {
                    **metadata,
                    "event": event,
                    "edit_mask": mask,
                    "mixed_anchor_public_bits": public,
                    "mixed_anchor_audit_bits": audit,
                    "value_present": present,
                    "parse_ok": parsed,
                    "domain_ok": legal,
                    "count": count,
                    "probability_all": count / n,
                }
            )
    # Same panel/cell/step A comparator and unique parent baseline; no mixing old draws.
    index = {
        (
            r["parent"],
            r["block"],
            r["arm"],
            r["step"],
            r["panel"],
            r["family"],
            r["center"],
            r["corrupted_index"],
        ): r
        for r in values
    }
    parent_index = {
        (r["parent"], r["panel"], r["family"], r["center"], r["corrupted_index"]): r
        for r in values
        if r["arm"] == "PARENT"
    }
    for row in values:
        tail = (row["panel"], row["family"], row["center"], row["corrupted_index"])
        baseline_A = index.get((row["parent"], row["block"], ARMS[0], row["step"], *tail))
        baseline_parent = parent_index.get((row["parent"], *tail))
        for label, reference in (("A", baseline_A), ("parent", baseline_parent)):
            for metric in ("X", "value_present", "extra_error"):
                row[f"delta_{metric}_vs_{label}"] = (
                    row[f"p_{metric}"] - reference[f"p_{metric}"] if reference else None
                )
            row[f"reference_{label}_checkpoint"] = reference["checkpoint_id"] if reference else None
            if (
                reference
                and abs(
                    row[f"delta_X_vs_{label}"]
                    - row[f"delta_value_present_vs_{label}"]
                    + row[f"delta_extra_error_vs_{label}"]
                )
                > 1e-12
            ):
                raise AssertionError("Relative value/extra decomposition failed")
    return {
        "cells": cells,
        "values": values,
        "projections": projections,
        "anchors": anchors,
        "operands": operands,
        "joints": joints,
    }


def _pp(value):
    return "null" if value is None else f"{100 * value:+.3f} pp"


def _interval(values):
    return "null" if values is None else "[" + ", ".join(_pp(v) for v in values) + "]"


def _findings(primary, summaries, macro, scope):
    lines = [
        "# SER-J2 结果",
        "",
        f"状态：`{primary['status']}`。",
        "",
        "本结果限定于固定的两个父模型、32个供体根、三个真实题序、每步11共同题+1供体+4回放、256更新。供体固定j2、同根观察和正确目标，供体权重1/16。新cross真值坐标互异、域0..99；duplicate只作符号保持哨兵。",
        "",
    ]
    if "Gamma" not in primary:
        lines.extend(
            [
                "确认矩阵不完整；没有从已完成子集替代唯一主比较。",
                "",
                str(primary.get("reason", "")),
                "",
                "缺项、技术失败和已观察计数见EXECUTION_SCOPE.json。以下派生表仅保存已获数据，不把技术失败记作模型I。",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## 供体与其他结构",
            "",
            "以下为E_CONFIRM叶单元及其他中心的逐父、逐题序效应；训练哨兵与留出题分开保存在METRICS_BY_CELL.csv。",
            "",
            "| 父 | 题序 | 臂 | 单元 | X | 相对A | 相对父 |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in summaries["values"]:
        if (
            row["panel"] == "E_CONFIRM"
            and row["arm"] in ARMS
            and row["family"] == "cross_series"
            and (row["center"], row["corrupted_index"])
            in ((0, 1), (2, 1), (3, 1), (3, 2), (0, 0), (1, 1))
        ):
            lines.append(
                f"| {row['parent']} | {row['block']} | {row['arm']} | "
                f"c{row['center'] + 1}j{row['corrupted_index'] + 1} | {row['p_X']:.4f} | "
                f"{_pp(row['delta_X_vs_A'])} | {_pp(row['delta_X_vs_parent'])} |"
            )
    lines.extend(
        [
            "",
            "## 保护单元与结构交互",
            "",
            f"Gamma = {_pp(primary['Gamma'])}；双侧95%配对根区间 "
            f"{_interval(primary['ci95_root_conditional'])}。",
            f"Delta4 = {_pp(primary['Delta4'])}，Delta3 = {_pp(primary['Delta3'])}。",
            "",
            "| 父 | 题序 | Gamma | Delta4 | Delta3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for s, parent in enumerate(PARENTS):
        for b in range(3):
            lines.append(
                f"| {parent} | {b} | {_pp(primary['per_parent_schedule_Gamma'][s][b])} | "
                f"{_pp(primary['Delta4_detail']['per_parent_schedule'][s][b])} | "
                f"{_pp(primary['Delta3_detail']['per_parent_schedule'][s][b])} |"
            )
    lines.extend(["", "| 次比较 | 效应 | 描述性95% | 四项Bonferroni98.75% |", "|---|---:|---|---|"])
    for name, item in primary["secondary"].items():
        lines.append(
            f"| {name} | {_pp(item['estimate'])} | "
            f"{_interval(item['ci95_root_conditional'])} | "
            f"{_interval(item['ci98_75_bonferroni_four_secondary'])} |"
        )
    lines.extend(["", "| 相对父模型 | 效应 | 描述性95% |", "|---|---:|---|"])
    for name, item in primary["relative_to_parent"].items():
        lines.append(
            f"| {name} | {_pp(item['estimate'])} | {_interval(item['ci95_root_conditional'])} |"
        )
    lo, hi = primary["ci95_root_conditional"]
    conclusion = (
        "主区间全负，与预登记的负结构交互方向一致。"
        if hi < 0
        else "主区间全正，与预登记的负方向相反。"
        if lo > 0
        else "主区间包含零；本设计未分辨交互方向。"
    )
    lines.extend(
        [
            "",
            conclusion + "交互本身不证明发生绝对损害；必须同时核对相对A、相对父与供体叶变化。",
            "",
            "## 正确值与额外错误",
            "",
            "X = value_present − value_present_but_not_X 对每回答和每计数成立。"
            "正确污染值允许与域外其他坐标共存；域外不计作合法多改。",
            "",
            "| 交互组成 | Gamma |",
            "|---|---:|",
        ]
    )
    for metric, item in primary["value_extra_decomposition"].items():
        lines.append(f"| {metric} | {_pp(item['Gamma'])} |")
    lines.extend(
        [
            "",
            "各单元及A/父对照的值出现与额外错误分解见VALUE_VS_EXTRA_ERRORS.csv；公开/审计混合锚点分开记录，操作数数值签名保留重叠，不解释为内部程序。",
            "",
            "## 投影与总体",
            "",
            "投影最多验证4个已有候选，不读取真值、不发起模型调用。PROJECTION_DIAGNOSTIC.csv单列调用数与CPU时长；原始X未被替换。固定唯一修复契约下，投影成功与value_present逐回答相等。",
            "",
            "MACRO_METRICS.csv与MACRO_CONTRASTS.csv使用16个cross单元等权、4个trend位置等权，二者各0.5；duplicate单列。总分不采用704个cross任务的直接平均。",
            "",
            "## 识别范围",
            "",
            "根bootstrap固定5000次、seed107079，保留同根所有父/题序/臂；条件于实际训练路径。"
            "两个父和三个题序不是任意模型总体。"
            "主MC95%半宽上界约1.25 pp仅限回答采样，不替代根与训练路径不确定性。",
            "",
            "5 pp仅为讨论参考，不是科学PASS门槛。"
            "区间宽、无增益或方向相反均保留，不删臂、不追加剂量。"
            "该替换总效应不能单独识别内部规则、普适干扰核或证明动态控制策略更优。",
            "",
            "实际完成/缺项、技术/正式计数与GPU时长见EXECUTION_SCOPE.json；未提供的耗时或来源不以零补齐。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(
    output_dir, scored_rows, *, release_receipt, execution_scope=None, exposure_rows=None
):
    """Write registered findings only after the evaluator's terminal release.

    Terminal technical failures are explicitly reported without treating a missing
    answer as I or silently estimating the planned contrast on a surviving subset.
    """
    _validate_release(release_receipt)
    output = Path(output_dir)
    rows = list(scored_rows)
    summaries = summarize_cells(rows)
    scope = dict(execution_scope or {})
    scope.setdefault("observed_scored_generations", len(rows))
    scope.setdefault("scope_of_report", "released registered SER-J2 endpoints")
    try:
        analysis = analyze_confirmation(rows)
        primary = analysis["primary"]
        macro = analysis["macro"]
        task_metrics = analysis["task_metrics"]
    except IncompleteDataError as exc:
        primary = {
            "status": "INCOMPLETE_NOT_CERTIFIED",
            "Gamma": None,
            "Delta3": None,
            "Delta4": None,
            "reason": str(exc),
            "no_partial_matrix_substitution": True,
            "scientific_pass_fail_gate": None,
        }
        # Missing primary quantities are absent from findings while remaining explicit JSON nulls.
        macro = {"status": "INCOMPLETE_NOT_CERTIFIED", "metrics": [], "contrasts": []}
        task_metrics = summarize_tasks(rows)
    _json(output / "PRIMARY_EFFECT.json", primary)
    _json(output / "MACRO_STATISTICS.json", macro)
    _json(output / "EXECUTION_SCOPE.json", scope)
    _json(output / "ANALYSIS_RELEASE_RECEIPT.json", release_receipt)
    _csv(output / "METRICS_BY_CELL.csv", summaries["cells"])
    _csv(output / "VALUE_VS_EXTRA_ERRORS.csv", summaries["values"])
    _csv(output / "PROJECTION_DIAGNOSTIC.csv", summaries["projections"])
    _csv(output / "MIXED_ANCHOR_SIGNATURES.csv", summaries["anchors"])
    _csv(output / "OPERAND_SIGNATURES.csv", summaries["operands"])
    _csv(output / "MACRO_METRICS.csv", macro["metrics"])
    _csv(output / "MACRO_CONTRASTS.csv", macro["contrasts"])
    _jsonl(output / "TASK_METRICS.jsonl", task_metrics)
    # Prediction/raw records remain in evaluator-owned immutable files; this
    # separate audit artifact preserves every derived per-answer signature.
    _jsonl(output / "AUDIT_SCORED_OUTPUTS.jsonl", rows)
    _jsonl(output / "SPARSE_JOINT_COUNTS.jsonl", summaries["joints"])
    if exposure_rows is not None:
        _csv(output / "EXPOSURE_EXACT.csv", exposure_rows)
    findings_primary = (
        primary
        if primary.get("Gamma") is not None
        else {k: v for k, v in primary.items() if k != "Gamma"}
    )
    _atomic_text(
        output / "FINAL_FINDINGS_zh.md", _findings(findings_primary, summaries, macro, scope)
    )
    return {
        "status": primary["status"],
        "output_dir": str(output),
        "observed_rows": len(rows),
        "Gamma": primary.get("Gamma"),
        "missing_are_not_model_errors": True,
    }
