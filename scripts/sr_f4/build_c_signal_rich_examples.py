"""Build a small, hand-designed C task book; never a train/evaluation manifest.

CPU arithmetic only. Uses no model, renderer, hashes, scheduler, old data scan,
or production pool. Gold and wrong-rule examples are developer-only sidecars.
Run with PYTHONPATH=src and an explicit output directory.
"""

# Chinese punctuation in the generated reader is intentional.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from sr_f3_ce.prompts import CONTRACT_V2
from sr_f4 import task_contract as oracle
from sr_f4.data import question
from sr_f4.training import native_advantages

STATUS = "DESIGN_PROTOTYPE_NOT_FROZEN_NOT_MODEL_TESTED"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]
TITLES = {
    "BINDING": "排序列与取值列：完整二乘二对照",
    "IMPACT": "固定可读边界，改变错选对答案的影响",
    "TIES": "多元素并列与印刷顺序",
    "SIGNED": "选择与有符号运算分开控制",
    "INTERVAL": "区间端点与条件筛选",
    "K_DIRECTION": "数量与方向：完整二乘二对照",
}


def world(alpha, beta, categories=None):
    return dict(
        categories=list(categories or MONTHS[: len(alpha)]),
        series=dict(Alpha=list(alpha), Beta=list(beta)),
        domain=[10, 90],
    )


def topk(ranking="Alpha", value="Beta", k=3, direction="largest", aggregate="mean"):
    return dict(
        family="TOPK",
        selection_series=ranking,
        value=value,
        k=k,
        direction=direction,
        aggregate=aggregate,
    )


def build():
    tasks = []

    def add(family, label, facts, query, expected, *, world_suffix="0"):
        # Fail closed here: the legacy oracle allows extra fields, which must not
        # silently turn a proposed filter-then-TOPK task into ordinary TOPK.
        common = {"family", "value", "aggregate"}
        specific = (
            {"selection_series", "k", "direction"}
            if query["family"] == "TOPK"
            else {"start", "end", "include_start", "include_end", "positive_only"}
        )
        if set(query) != common | specific:
            raise ValueError("Prototype query must use exactly the supported schema")
        answer = oracle.execute(facts, query)
        independent = oracle.independently_answer(facts, query)
        if answer != independent or answer != Fraction(expected):
            raise ValueError(f"Hand answer / double-oracle mismatch: {family}/{label}")
        root = f"C_DESIGN_20261024_{family}"
        chosen = oracle.selected(facts, query)
        text = question(facts, query)
        tasks.append(
            dict(
                qid=f"{root}_{label}",
                root_id=root,
                world_id=f"{root}_{world_suffix}",
                family=family,
                label=label,
                status=STATUS,
                split="DESIGN_ONLY",
                world=copy.deepcopy(facts),
                query=copy.deepcopy(query),
                question=text,
                prompt=text + "\n" + CONTRACT_V2,
                image_status="NOT_RENDERED_TABLE_IS_DESIGN_SPECIFICATION",
                developer_only=dict(
                    target_indices=chosen,
                    target_set=[facts["categories"][i] for i in chosen],
                    selected_values=[str(oracle.operand(facts, query, i)) for i in chosen],
                    answer_exact=str(answer),
                    independent_answer_exact=str(independent),
                    hand_answer_exact=str(Fraction(expected)),
                ),
            )
        )

    binding = world([84, 68, 55, 43, 29, 16], [19, 47, 68, 83, 72, 54])
    for ranking, value, expected in (
        ("Alpha", "Alpha", "69"),
        ("Alpha", "Beta", "134/3"),
        ("Beta", "Alpha", "127/3"),
        ("Beta", "Beta", "223/3"),
    ):
        add("BINDING", f"{ranking}_to_{value}", binding, topk(ranking, value), expected)

    for label, beta, expected in (
        ("zero", [29, 46, 55, 55, 69, 54], "130/3"),
        ("small", [29, 46, 42, 41, 69, 54], "39"),
        ("medium", [29, 46, 62, 42, 69, 54], "137/3"),
        ("large", [29, 46, 82, 17, 69, 54], "157/3"),
    ):
        add(
            "IMPACT",
            label,
            world([88, 73, 60, 48, 27, 12], beta),
            topk(),
            expected,
            world_suffix=label,
        )

    tied = world(
        [80, 58, 58, 58, 22, 11],
        [31, 47, 68, 83, 72, 54],
        ["May", "June", "April", "January", "February", "March"],
    )
    for label, query, expected in (
        ("top3_mean", topk(), "146/3"),
        ("top3_sum", topk(aggregate="sum"), "146"),
        ("top2_mean", topk(k=2), "39"),
        ("bottom3_mean", topk(direction="smallest"), "173/3"),
    ):
        add("TIES", label, tied, query, expected)

    signed = world([87, 74, 61, 49, 31, 17], [89, 47, 82, 16, 68, 54])
    for label, query, expected in (
        ("diff_mean", topk(value="diff"), "4/3"),
        ("diff_sum", topk(value="diff", aggregate="sum"), "4"),
        ("beta_mean", topk(), "218/3"),
        ("rank_beta_diff_mean", topk(ranking="Beta", value="diff"), "-20"),
    ):
        add("SIGNED", label, signed, query, expected)

    interval = world([70, 67, 58, 41, 66, 28, 81, 20], [20, 26, 22, 41, 37, 50, 30, 65])
    for label, include_start, include_end, positive, expected in (
        ("main", False, True, True, "116/3"),
        ("include_start", True, True, True, "157/4"),
        ("exclude_end", False, False, True, "65/2"),
        ("no_filter", False, True, False, "94/5"),
    ):
        query = dict(
            family="INTERVAL",
            start=1,
            end=6,
            include_start=include_start,
            include_end=include_end,
            positive_only=positive,
            value="diff",
            aggregate="mean",
        )
        add("INTERVAL", label, interval, query, expected)

    quantity = world([42, 86, 57, 21, 74, 35, 63, 12], [66, 24, 78, 43, 51, 87, 19, 60])
    for k, direction, expected in (
        (2, "largest", "75/2"),
        (3, "largest", "94/3"),
        (2, "smallest", "103/2"),
        (3, "smallest", "190/3"),
    ):
        add("K_DIRECTION", f"{direction}_{k}", quantity, topk(k=k, direction=direction), expected)

    if len(tasks) != 24 or Counter(t["family"] for t in tasks) != dict.fromkeys(TITLES, 4):
        raise ValueError("Expected six families with four questions each")
    if len({t["qid"] for t in tasks}) != 24 or len({t["world_id"] for t in tasks}) != 9:
        raise ValueError("Unexpected identities")
    # Answers alone cannot validate selection when wrong sets have equal sums.
    # These independently worked sets deliberately include the collision case.
    hand_sets = {
        "BINDING": {
            "Alpha_to_Alpha": [0, 1, 2],
            "Alpha_to_Beta": [0, 1, 2],
            "Beta_to_Alpha": [2, 3, 4],
            "Beta_to_Beta": [2, 3, 4],
        },
        "IMPACT": {label: [0, 1, 2] for label in ("zero", "small", "medium", "large")},
        "TIES": {
            "top3_mean": [0, 1, 2],
            "top3_sum": [0, 1, 2],
            "top2_mean": [0, 1],
            "bottom3_mean": [1, 4, 5],
        },
        "SIGNED": {
            "diff_mean": [0, 1, 2],
            "diff_sum": [0, 1, 2],
            "beta_mean": [0, 1, 2],
            "rank_beta_diff_mean": [0, 2, 4],
        },
        "INTERVAL": {
            "main": [2, 4, 6],
            "include_start": [1, 2, 4, 6],
            "exclude_end": [2, 4],
            "no_filter": [2, 3, 4, 5, 6],
        },
        "K_DIRECTION": {
            "largest_2": [1, 4],
            "largest_3": [1, 4, 6],
            "smallest_2": [3, 7],
            "smallest_3": [3, 5, 7],
        },
    }
    for task in tasks:
        hand = hand_sets[task["family"]][task["label"]]
        if sorted(task["developer_only"]["target_indices"]) != hand:
            raise ValueError(f"Hand selected-set mismatch: {task['qid']}")
        task["developer_only"]["hand_target_set_indices"] = list(hand)
    return tasks


def audit(tasks):
    """Check worked wrong rules, including a wrong-set/same-answer control."""
    by_key = {(t["family"], t["label"]): t for t in tasks}
    routes = []

    def route(family, label, name, indices, expected):
        task = by_key[family, label]
        values = [oracle.operand(task["world"], task["query"], i) for i in indices]
        value = oracle.aggregate(values, task["query"]["aggregate"])
        if value != Fraction(expected):
            raise ValueError(f"Wrong-rule worked answer mismatch: {name}")
        target = set(task["developer_only"]["target_indices"])
        if set(indices) == target:
            raise ValueError("This audit route must be a different adopted set")
        routes.append(
            dict(
                qid=task["qid"],
                rule=name,
                indices=indices,
                categories=[task["world"]["categories"][i] for i in indices],
                recomputed_answer_exact=str(value),
                same_final_answer=value == Fraction(task["developer_only"]["answer_exact"]),
                is_observed_model_response=False,
            )
        )

    route("BINDING", "Alpha_to_Beta", "rank_by_wrong_column", [2, 3, 4], "223/3")
    for label, expected in (
        ("zero", "130/3"),
        ("small", "116/3"),
        ("medium", "39"),
        ("large", "92/3"),
    ):
        route("IMPACT", label, "swap_boundary_March_for_April", [0, 1, 3], expected)
    route("TIES", "top3_mean", "calendar_order_breaks_tie", [0, 3, 2], "182/3")
    route("TIES", "top3_mean", "include_entire_tie_block", [0, 1, 2, 3], "229/4")
    route("SIGNED", "diff_mean", "rank_by_wrong_column", [0, 2, 4], "-20")
    route("INTERVAL", "main", "include_excluded_start", [1, 2, 4, 6], "157/4")
    route("INTERVAL", "main", "exclude_included_end", [2, 4], "65/2")
    route("INTERVAL", "main", "omit_positive_filter", [2, 3, 4, 5, 6], "94/5")
    bounds = []
    for successes in range(1, 8):
        failure_advantage = native_advantages([1] * successes + [0] * (8 - successes))[-1]
        if failure_advantage + 0.25 >= 0:
            raise ValueError("Mixed-J upper-bound proof does not hold")
        bounds.append(
            dict(
                successes=successes,
                failure_native_advantage=failure_advantage,
                loose_final_upper_bound=failure_advantage + 0.25,
            )
        )
    return dict(
        status="CPU_ARITHMETIC_CHECKED_ONLY",
        design_status=STATUS,
        question_count=len(tasks),
        worlds=9,
        root_lineages=6,
        hand_and_two_oracles_agree=24,
        hand_selected_sets_agree=24,
        wrong_rule_checks=routes,
        mixed_J_failure_bounds=bounds,
        model_calls=0,
        gpu_jobs=0,
        hashes_computed=0,
        images_rendered=0,
        unknown_labels_filled=0,
        natural_signal_density="NOT_MEASURED",
        r6_natural_response_coverage="NOT_MEASURED",
    )


def markdown(tasks, receipt):
    lines = [
        "# C 信号丰富题组：24 道开发原型",
        "",
        f"状态：`{STATUS}`。9 个事实世界、6 个共同构造谱系。",
        "",
        "以下表格与答案供设计审核，不是模型输入，也不是自然生成的回答。"
        "图片尚未生成；每道题有独立的原生 CONTRACT_V2 prompt。"
        "所有同谱系变体必须属于同一数据分区。",
        "",
        "公共语义：并列按图中印刷顺序处理，表格从上到下就是印刷顺序。"
        "均值按实际入选项数计算，接受精确分数；差值指 Alpha 减 Beta。"
        "模型的完整 prompt 另含原生 JSON 证据与答案契约，见 QUESTION_BANK.json。",
        "",
    ]
    for family, title in TITLES.items():
        lines += [f"## {family}：{title}", ""]
        seen = set()
        for task in (t for t in tasks if t["family"] == family):
            if task["world_id"] not in seen:
                lines += [
                    f"事实表 `{task['world_id']}`（从上到下为印刷顺序）：",
                    "",
                    "| 类别 | Alpha | Beta |",
                    "|---|---:|---:|",
                ]
                w = task["world"]
                lines += [
                    f"| {c} | {a} | {b} |"
                    for c, a, b in zip(
                        w["categories"], w["series"]["Alpha"], w["series"]["Beta"], strict=True
                    )
                ]
                lines.append("")
                seen.add(task["world_id"])
            gold = task["developer_only"]
            lines += [
                f"**{task['label']}**：{task['question']}",
                "",
                f"正确采用集合：{', '.join(gold['target_set'])}；"
                f"聚合值：{', '.join(gold['selected_values'])}；"
                f"精确答案：**{gold['answer_exact']}**。",
                "",
            ]
    lines += [
        "## 构造的错误规则对照",
        "",
        "这些是确定性的错误规则演算，不计为模型回答、训练信号或训练数据。",
        "",
        "| 问题 | 错误规则 | 错误集合 | 沿该集合计算 | 与正确答案相同 |",
        "|---|---|---|---:|---|",
    ]
    for r in receipt["wrong_rule_checks"]:
        lines.append(
            f"| {r['qid']} | {r['rule']} | {', '.join(r['categories'])} | "
            f"{r['recomputed_answer_exact']} | {r['same_final_answer']} |"
        )
    lines += [
        "",
        "验证：24 道题均与手工答案及两套 oracle 一致；11 条错误规则演算一致。"
        "这仅证明题目定义与算术正确，尚未证明模型会自然产生足够的组内分歧。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = build()
    receipt = audit(tasks)
    payload = dict(status=STATUS, purpose="DEVELOPER_DESIGN_EXAMPLES_ONLY", tasks=tasks)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("QUESTION_BANK.json", payload), ("CPU_CHECK.json", receipt)):
        (args.output / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    (args.output / "QUESTION_BOOK_ZH.md").write_text(markdown(tasks, receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "status",
                    "question_count",
                    "worlds",
                    "root_lineages",
                    "model_calls",
                    "gpu_jobs",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
