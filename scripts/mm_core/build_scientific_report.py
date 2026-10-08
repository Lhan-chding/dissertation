#!/usr/bin/env python3
"""Render saved MM audit tables; never import a runtime, score answers, or grant a gate.

Usage: python scripts/mm_core/build_scientific_report.py --run-root RUN
Only report/SCIENTIFIC_* files are written. All input tables remain immutable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from pathlib import Path

STAGE = "MEASUREMENT_AUDIT"
TABLES = (
    "COUNTS",
    "COVERAGE",
    "GEOMETRY",
    "ROOT_EQUAL",
    "CONTEXT_AUDIT",
    "REWARD_SUPPORT",
    "FIELD_METRICS",
    "ROOT_BOOTSTRAP",
)
CHARTS = ("grouped_bar", "line")
OPERATIONS = ("sum", "difference", "range")
LEVELS = ("low", "high")
EXACT = ("111", "100", "011", "010", "001", "000")
TOLERANCE = tuple("".join(bits) for bits in product("01", repeat=3))
SOURCES = {
    "self_field_surprisal": ("self 自生成", "generated_tokens_actual_self_prefix"),
    "gold_teacher_forced_field_nll": (
        "gold teacher forcing",
        "gold_completion_answer_prefix_contains_gold_readings",
    ),
}
LABEL_FIELDS = ("chart_type", "operation", "visual_level", "numeric_level")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def fraction(value):
    if value is None:
        return None
    require(isinstance(value, dict), "RATIONAL_OBJECT_REQUIRED")
    n, d = value.get("numerator"), value.get("denominator")
    require(type(n) is int and type(d) is int and d > 0, "INVALID_RATIONAL")
    return Fraction(n, d)


def validate_numbers(value):
    if isinstance(value, dict):
        if {"numerator", "denominator", "rate"} <= value.keys():
            n, d = value["numerator"], value["denominator"]
            require(integer(n) and integer(d) and n <= d, "INVALID_COUNT_RATE")
            expected = Fraction(n, d) if d else None
            require(fraction(value["rate"]) == expected, "COUNT_RATE_MISMATCH")
        elif {"numerator", "denominator"} <= value.keys():
            fraction(value)
        for item in value.values():
            validate_numbers(item)
    elif isinstance(value, list):
        for item in value:
            validate_numbers(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "NONFINITE_INPUT")


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"DUPLICATE_JSON_KEY:{key}")
        result[key] = value
    return result


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}
        self.missing = set()
        self.inventories = {}

    def read_text(self, relative):
        path = self.root / relative
        require(path.resolve().is_relative_to(self.root), "INPUT_OUTSIDE_RUN_ROOT")
        if not path.is_file():
            self.missing.add(relative)
            return None
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(
            relative not in self.hashes or self.hashes[relative] == digest,
            f"INPUT_CHANGED_DURING_REPORT:{relative}",
        )
        self.hashes[relative] = digest
        return data.decode("utf-8")

    def json(self, relative):
        text = self.read_text(relative)
        if text is None:
            return None
        value = json.loads(text, object_pairs_hook=unique_json)
        validate_numbers(value)
        require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{relative}")
        return value

    def jsonl(self, relative):
        text = self.read_text(relative)
        if text is None:
            return None
        rows = [
            json.loads(line, object_pairs_hook=unique_json)
            for line in text.splitlines()
            if line.strip()
        ]
        for row in rows:
            require(isinstance(row, dict), f"JSONL_OBJECT_REQUIRED:{relative}")
            validate_numbers(row)
        return rows

    def paths(self, pattern):
        paths = sorted(
            str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_file()
        )
        self.inventories[pattern] = paths
        return paths

    def stable(self):
        for relative, digest in self.hashes.items():
            require(
                hashlib.sha256((self.root / relative).read_bytes()).hexdigest() == digest,
                f"INPUT_CHANGED_DURING_REPORT:{relative}",
            )
        for relative in self.missing:
            require(
                not (self.root / relative).is_file(), f"INPUT_APPEARED_DURING_REPORT:{relative}"
            )
        for pattern, paths in self.inventories.items():
            require(
                paths
                == sorted(
                    str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_file()
                ),
                f"INPUT_SET_CHANGED_DURING_REPORT:{pattern}",
            )


def cohort(document, stage, name, *, format_report=False):
    if document is None:
        return None
    rows = document.get("cohorts")
    require(isinstance(rows, list), f"COHORTS_REQUIRED:{name}")
    require(len(rows) <= 1, f"MULTIPLE_COHORTS:{name}")
    if not rows:
        return None
    row = rows[0]
    require(isinstance(row, dict) and row.get("stage") == stage, f"STAGE_MISMATCH:{name}")
    if not format_report:
        require(row.get("split") == "AUDIT_MEASURE", f"SPLIT_MISMATCH:{name}")
    require("model_hash" in row, f"MODEL_IDENTITY_MISSING:{name}")
    return row


def groups(rows, factors=False):
    fields = LABEL_FIELDS if factors else LABEL_FIELDS[:2]
    expected = set(
        product(CHARTS, OPERATIONS, LEVELS, LEVELS) if factors else product(CHARTS, OPERATIONS)
    )
    require(isinstance(rows, list), "GROUP_LIST_REQUIRED")
    keyed = {}
    for row in rows:
        key = tuple(row.get(k) for k in fields)
        require(key in expected and key not in keyed, f"INVALID_OR_DUPLICATE_LABEL:{key}")
        keyed[key] = row
    require(set(keyed) == expected, "INCOMPLETE_GROUP_LABELS")
    return keyed


def counted(row):
    n, legal = row["recorded_attempts"], row["joint_scorable_denominator"]
    require(integer(n) and integer(legal) and legal <= n, "INVALID_JOINT_DENOMINATOR")
    for name, cells in (("exact_counts", EXACT), ("tolerance_counts", TOLERANCE)):
        values = row[name]
        require(set(values) == set(cells), f"INCORRECT_CELL_SET:{name}")
        require(all(integer(v) for v in values.values()), "INVALID_CELL_COUNT")
        require(sum(values.values()) == legal, "CELL_DENOMINATOR_MISMATCH")
    require(row["missing_joint_fields_not_000"] == n - legal, "MISSING_FIELD_COUNT_MISMATCH")
    for key in ("P_full", "A_full", "J_full"):
        require(row[key]["denominator"] == n, f"FULL_DENOMINATOR_MISMATCH:{key}")
    cells = row["exact_counts"]
    numerators = {
        "p": cells["111"] + cells["100"],
        "a": cells["111"] + cells["011"] + cells["001"],
        "c": cells["111"] + cells["011"] + cells["010"],
        "j": cells["111"],
        "z": cells["001"],
    }
    for key, numerator in numerators.items():
        value = row["conditional_on_L"][key]
        require(
            value["denominator"] == legal and value["numerator"] == numerator,
            f"CONDITIONAL_DENOMINATOR_OR_COUNT_MISMATCH:{key}",
        )
    for key, full in (("P_given_LP", "P_full"), ("A_given_LA", "A_full")):
        require(
            row[key]["denominator"] <= n and row[key]["numerator"] == row[full]["numerator"],
            "FIELD_RATE_MISMATCH",
        )


def validate_context(context, reward):
    pairs = context["pairs"]
    keys = [
        (p["root_family_id"], p["image_hash"], p["readset_id"], tuple(p["ordered_item_ids"]))
        for p in pairs
    ]
    require(len(set(keys)) == len(keys), "DUPLICATE_CONTEXT_PAIR")
    byq = {q["question_id"]: q for q in reward["questions"]} if reward else {}
    for pair in pairs:
        if byq:
            require(
                all(
                    q in byq and byq[q]["root_family_id"] == pair["root_family_id"]
                    for q in pair["question_ids"]
                ),
                "CONTEXT_QUESTION_IDENTITY_MISMATCH",
            )
        if pair["status"] == "PAIR_UNAVAILABLE":
            continue
        if byq:
            require(
                sorted(byq[q]["operation"] for q in pair["question_ids"]) == ["difference", "sum"],
                "CONTEXT_OPERATION_MISMATCH",
            )
        m1 = pair["sum_coverage"]["L_P"]["numerator"]
        m2 = pair["difference_coverage"]["L_P"]["numerator"]
        for key, den in (("C11", m1 * (m1 - 1)), ("C22", m2 * (m2 - 1)), ("C12", m1 * m2)):
            require(pair[key + "_denominator"] == den, "CONTEXT_DENOMINATOR_MISMATCH")
            value = fraction(pair[key])
            require((den == 0) == (value is None), "CONTEXT_MISSING_VALUE_MISMATCH")
            if value is not None:
                require(
                    0 <= value <= 1 and (value * den).denominator == 1, "CONTEXT_COUNT_MISMATCH"
                )


def validate_measurement(tables, receipt, common):
    present = {name: table for name, table in tables.items() if table is not None}
    identities = {table["model_hash"] for table in present.values()}
    require(len(identities) <= 1, "CROSS_TABLE_MODEL_MISMATCH")
    counts, coverage = tables["COUNTS"], tables["COVERAGE"]
    if counts:
        overall = counts["overall"]
        layers = groups(counts["chart_operation"])
        factors = groups(counts["factor_cells"], True)
        for row in [overall, *layers.values(), *factors.values()]:
            counted(row)
        for parent, children in [
            (overall, list(layers.values())),
            *[
                (row, [f for key, f in factors.items() if key[:2] == label])
                for label, row in layers.items()
            ],
        ]:
            for key in ("recorded_attempts", "joint_scorable_denominator"):
                require(sum(row[key] for row in children) == parent[key], "GROUP_COUNT_MISMATCH")
            for name in ("exact_counts", "tolerance_counts"):
                for cell, n in parent[name].items():
                    require(sum(row[name][cell] for row in children) == n, "GROUP_CELL_MISMATCH")
        if overall["recorded_attempts"]:
            require(
                isinstance(counts["model_hash"], str) and bool(counts["model_hash"]),
                "OBSERVED_MODEL_IDENTITY_REQUIRED",
            )
        if common:
            require(
                counts["model_hash"] in (None, common.get("model_hash")),
                "COMMON_START_MODEL_MISMATCH",
            )
        if receipt:
            require(
                receipt["scored_record_count"] == overall["recorded_attempts"],
                "SCORING_RECEIPT_COUNT_MISMATCH",
            )
    if coverage:
        n = coverage["recorded_attempts"]
        require(integer(n), "INVALID_COVERAGE_COUNT")
        for key in ("L_P", "L_A", "L"):
            require(coverage[key]["denominator"] == n, "COVERAGE_DENOMINATOR_MISMATCH")
        require(sum(coverage["status_counts"].values()) == n, "STATUS_COUNT_MISMATCH")
        if counts:
            require(n == counts["overall"]["recorded_attempts"], "COVERAGE_COUNT_MISMATCH")
            require(
                coverage["L"]["numerator"] == counts["overall"]["joint_scorable_denominator"],
                "COVERAGE_L_MISMATCH",
            )
            require(
                coverage["L_P"]["numerator"] == counts["overall"]["P_given_LP"]["denominator"]
                and coverage["L_A"]["numerator"] == counts["overall"]["A_given_LA"]["denominator"],
                "COVERAGE_FIELD_DENOMINATOR_MISMATCH",
            )
    for name in ("GEOMETRY", "FIELD_METRICS"):
        table = tables[name]
        if not table:
            continue
        groups(table["chart_operation"])
        if name == "FIELD_METRICS":
            groups(table["factor_cells"], True)
        sections = [("overall", [table["overall"]])]
        sections += [
            (key, table[key]) for key in ("chart_operation", "factor_cells") if key in table
        ]
        for scope, rows in sections:
            reference = (
                (
                    {(): counts["overall"]}
                    if scope == "overall"
                    else groups(counts[scope], scope == "factor_cells")
                )
                if counts
                else {}
            )
            for row in rows:
                key = (
                    ()
                    if scope == "overall"
                    else tuple(row[k] for k in LABEL_FIELDS[: 4 if scope == "factor_cells" else 2])
                )
                ref = reference.get(key)
                if name == "GEOMETRY":
                    if ref:
                        require(
                            row["LP_denominator"] == ref["P_given_LP"]["denominator"],
                            "GEOMETRY_LP_MISMATCH",
                        )
                        require(
                            row["residual_identity_checks"] == ref["joint_scorable_denominator"],
                            "GEOMETRY_L_MISMATCH",
                        )
                    require(
                        row["S_denominator"] + row["S_missing_zero_error_count"]
                        == row["LP_denominator"],
                        "GEOMETRY_S_SUPPORT_MISMATCH",
                    )
                else:
                    for source, (_, provenance) in SOURCES.items():
                        packet = row[source]
                        require(
                            packet["expected_prefix_provenance"] == provenance,
                            "FIELD_PREFIX_MISMATCH",
                        )
                        n = packet["response_denominator"]
                        if ref:
                            require(n == ref["recorded_attempts"], "FIELD_RESPONSE_COUNT_MISMATCH")
                        for field in packet["fields"].values():
                            for metric in ("mean_nll", "mean_token_entropy"):
                                value = field[metric]
                                den = value["sequence_denominator"]
                                require(
                                    integer(den)
                                    and den <= n
                                    and value["missing_response_count"] == n - den,
                                    "FIELD_SEQUENCE_DENOMINATOR_MISMATCH",
                                )
                                require(
                                    (den == 0) == (value["sequence_mean"] is None),
                                    "FIELD_MISSING_VALUE_MISMATCH",
                                )
                                require(
                                    (value["field_token_denominator"] == 0)
                                    == (value["token_weighted_mean"] is None),
                                    "FIELD_TOKEN_DENOMINATOR_MISMATCH",
                                )
    reward = tables["REWARD_SUPPORT"]
    if reward:
        groups(reward["chart_operation"])
        qs = reward["questions"]
        require(len({q["question_id"] for q in qs}) == len(qs), "DUPLICATE_REWARD_QUESTION")
        for q in qs:
            n, c, valid = q["K_observed"], q["answer_successes"], q["answer_valid"]
            require(
                all(integer(v) for v in (n, c, valid)) and c <= valid <= n, "REWARD_COUNT_MISMATCH"
            )
            require(
                fraction(q["observed_rate"]) == (Fraction(c, n) if n else None),
                "REWARD_RATE_MISMATCH",
            )
            require(
                (q["chart_type"], q["operation"]) in set(product(CHARTS, OPERATIONS)),
                "REWARD_LABEL_MISMATCH",
            )
        if coverage:
            require(
                len(qs) == coverage["question_count"]
                and sum(q["K_observed"] for q in qs) == coverage["recorded_attempts"],
                "REWARD_COVERAGE_MISMATCH",
            )
        if counts:
            require(
                sum(q["answer_successes"] for q in qs) == counts["overall"]["A_full"]["numerator"],
                "REWARD_SUCCESS_COUNT_MISMATCH",
            )
        for layer in reward["chart_operation"]:
            selected = [q for q in qs if all(q[k] == layer[k] for k in LABEL_FIELDS[:2])]
            require(
                layer["question_count"] == len(selected)
                and layer["observed_question_count"] == sum(q["K_observed"] > 0 for q in selected),
                "REWARD_LAYER_DENOMINATOR_MISMATCH",
            )
    if tables["CONTEXT_AUDIT"]:
        validate_context(tables["CONTEXT_AUDIT"], reward)
    root = tables["ROOT_EQUAL"]
    if root:
        roots = root["roots"]
        require(len({r["root_family_id"] for r in roots}) == len(roots), "DUPLICATE_ROOT")
        if coverage:
            require(len(roots) == coverage["root_family_count"], "ROOT_COUNT_MISMATCH")
        for value in root["aggregate"].values():
            require(
                value["root_count"] == len(roots)
                and 0 <= value["supported_root_count"] <= len(roots),
                "ROOT_SUPPORT_MISMATCH",
            )
        for row in roots:
            for value in row["metrics"].values():
                defined, expected = value["defined_cells"], value["expected_cells"]
                require(
                    integer(defined) and expected == 24 and defined <= expected,
                    "ROOT_CELL_COUNT_MISMATCH",
                )
                require(
                    fraction(value["supported_weight"]) == Fraction(defined, expected),
                    "ROOT_WEIGHT_DENOMINATOR_MISMATCH",
                )
    boot = tables["ROOT_BOOTSTRAP"]
    if boot:
        require(boot["unit"] == "root_family_id", "BOOTSTRAP_UNIT_MISMATCH")
        for value in boot["metrics"].values():
            require(
                value["defined_replicates"] + value["undefined_replicates"]
                == boot["requested_replicates"],
                "BOOTSTRAP_DENOMINATOR_MISMATCH",
            )
        if receipt:
            require(boot["seed"] == receipt["root_bootstrap_seed"], "BOOTSTRAP_SEED_MISMATCH")


def number(value):
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict) and "numerator" in value:
        if "rate" in value:
            n, d = value["numerator"], value["denominator"]
            return f"{n}/{d} (missing)" if not d else f"{n}/{d} ({decimal(Fraction(n, d) * 100)}%)"
        value = fraction(value)
    if isinstance(value, Fraction):
        exact = f"{value.numerator}/{value.denominator}"
        return f"{decimal(value)} [{exact}]" if len(exact) <= 26 else f"≈{decimal(value)}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def decimal(value):
    with localcontext() as context:
        context.prec = 16
        return format(Decimal(value.numerator) / Decimal(value.denominator), ".6g")


def cell(value):
    return (
        number(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("\r", " ")
        .replace("\n", "<br>")
        .replace("`", "&#96;")
    )


def table(headers, rows):
    rows = list(rows)
    require(all(len(row) == len(headers) for row in rows), "REPORT_COLUMN_MISMATCH")
    return (
        "\n".join(
            [
                "| " + " | ".join(map(cell, headers)) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *["| " + " | ".join(map(cell, row)) + " |" for row in rows],
            ]
        )
        + "\n"
    )


def label(row):
    values = [row[k] for k in LABEL_FIELDS if k in row]
    return " / ".join(values) or "总体"


def source_link(name):
    return f"[原始 {name}.json](../tables/{STAGE}/{name}.json)"


def counts_text(rows):
    text = table(
        ["层", "记录回答 N", "联合可评分 L", *EXACT],
        (
            [
                label(r),
                r["recorded_attempts"],
                r["joint_scorable_denominator"],
                *[r["exact_counts"][k] for k in EXACT],
            ]
            for r in rows
        ),
    )
    text += "\n" + table(
        ["层", "p|L", "a|L", "c|L", "j|L", "z|L"],
        ([label(r), *[r["conditional_on_L"][k] for k in ("p", "a", "c", "j", "z")]] for r in rows),
    )
    text += "\n" + table(
        ["层", "P 全记录", "A 全记录", "J 全记录", "P|LP", "A|LA", "缺联合字段"],
        (
            [
                label(r),
                *[
                    r[k]
                    for k in (
                        "P_full",
                        "A_full",
                        "J_full",
                        "P_given_LP",
                        "A_given_LA",
                        "missing_joint_fields_not_000",
                    )
                ],
            ]
            for r in rows
        ),
    )
    return text


def geometry_text(document):
    rows = [document["overall"], *document["chart_operation"]]
    text = table(
        [
            "层",
            "LP",
            "S有效",
            "S零误差missing",
            "S直接条件均值",
            "整向量近失",
            "平均max|e|",
            "域外",
        ],
        (
            [
                label(r),
                *[
                    r[k]
                    for k in (
                        "LP_denominator",
                        "S_denominator",
                        "S_missing_zero_error_count",
                        "S_direct_error_response_mean",
                        "near_miss",
                        "mean_max_abs_error",
                        "outside_axis",
                    )
                ],
            ]
            for r in rows
        ),
    )
    text += "\n" + table(
        ["层", "εP (LP)", "εC (L)", "εO (L)", "精确恒等式核验n", "最大精确残差"],
        (
            [
                label(r),
                *[
                    r[k]
                    for k in (
                        "eps_P_mean",
                        "eps_C_mean",
                        "eps_O_mean",
                        "residual_identity_checks",
                        "max_exact_residual_identity_error",
                    )
                ],
            ]
            for r in rows
        ),
    )
    text += "\n" + table(
        ["层", "操作数个数", "角色索引(0起)", "分母", "偏差", "MAE", "MSE", "RMSE"],
        (
            [
                label(r),
                *[
                    role[k]
                    for k in (
                        "arity",
                        "ordered_role_index",
                        "denominator",
                        "mean_bias",
                        "MAE",
                        "MSE",
                        "RMSE_float",
                    )
                ],
            ]
            for r in rows
            for role in r["roles"]
        ),
    )
    return text


def root_text(root, bootstrap):
    intervals = bootstrap["metrics"] if bootstrap else {}
    text = "区间针对有支持根的条件均值, 不能套在完整固定权重均值上。\n\n"
    text += table(
        [
            "指标",
            "根等权条件均值",
            "完整固定权重均值",
            "支持根/全部根",
            "描述性95%区间",
            "有效/请求重复",
            "undefined",
        ],
        (
            [
                key,
                value["root_equal_conditional_mean"],
                value["complete_fixed_weight_mean"],
                f"{value['supported_root_count']}/{value['root_count']}",
                f"{number(intervals.get(key, {}).get('percentile_95_low'))} ~ "
                f"{number(intervals.get(key, {}).get('percentile_95_high'))}",
                f"{intervals.get(key, {}).get('defined_replicates', 'missing')}/"
                f"{bootstrap['requested_replicates'] if bootstrap else 'missing'}",
                intervals.get(key, {}).get("undefined_replicates"),
            ]
            for key, value in root["aggregate"].items()
        ),
    )
    return text


def field_text(rows):
    text = (
        "self 使用实际生成前缀; gold 答案前缀含 gold readings, "
        "属于 teacher forcing, 不能称作自由生成能力。"
    )
    text += "熵来自记录的模型分布, 不由回答频率估算。浮点模型测量的分数表示不增加测量精度。\n\n"
    text += table(
        [
            "层",
            "来源",
            "字段",
            "量",
            "序列均值",
            "token加权均值",
            "序列分母",
            "字段token分母",
            "缺测回答",
        ],
        (
            [
                label(r),
                SOURCES[source][0],
                field,
                metric,
                *[
                    value[k]
                    for k in (
                        "sequence_mean",
                        "token_weighted_mean",
                        "sequence_denominator",
                        "field_token_denominator",
                        "missing_response_count",
                    )
                ],
            ]
            for r in rows
            for source in SOURCES
            for field in ("readings", "answer")
            for metric in ("mean_nll", "mean_token_entropy")
            for value in [r[source]["fields"][field][metric]]
        ),
    )
    text += "\n" + table(
        ["层", "来源", "响应分母", "已测packet", "状态计数", "前缀权限", "分布", "边界规则"],
        (
            [
                label(r),
                SOURCES[source][0],
                *[
                    r[source][k]
                    for k in (
                        "response_denominator",
                        "measured_packet_count",
                        "source_status_counts",
                        "expected_prefix_provenance",
                        "distribution_counts",
                        "boundary_rule_counts",
                    )
                ],
            ]
            for r in rows
            for source in SOURCES
        ),
    )
    return text


def context_text(context):
    text = (
        "仅匹配同图像、同有序readset、单位、δ与操作数个数的sum/difference; "
        "不截断range向量。C11/C22分母是同提示有序非自配对数, C12分母是跨提示配对数。"
        "差异=(C11+C22)/2-C12, 可为负; K*K配对不增加独立根数。\n\n"
    )
    text += table(
        [
            "根",
            "readset",
            "题ID",
            "状态",
            "C11",
            "C11分母",
            "C22",
            "C22分母",
            "C12",
            "C12分母",
            "差异",
        ],
        (
            [
                p["root_family_id"],
                p["readset_id"],
                p["question_ids"],
                p["status"],
                *[
                    paired_rate(p, k) if k in ("C11", "C22", "C12") else p.get(k)
                    for k in (
                        "C11",
                        "C11_denominator",
                        "C22",
                        "C22_denominator",
                        "C12",
                        "C12_denominator",
                        "collision_contrast",
                    )
                ],
            ]
            for p in context["pairs"]
        ),
    )
    text += "\n" + table(
        ["根 / readset", "提示", "记录/预期", "LP", "L", "P全记录", "平均max|e|", "跨提示差异幅度"],
        (
            [
                f"{p['root_family_id']} / {p['readset_id']}",
                op,
                f"{p.get(op + '_coverage', {}).get('recorded_attempts', 'missing')}/"
                f"{p.get(op + '_coverage', {}).get('expected_completions', 'missing')}",
                p.get(op + "_coverage", {}).get("L_P"),
                p.get(op + "_coverage", {}).get("L"),
                p.get(op + "_P_full"),
                p.get(op + "_mean_max_abs_error"),
                p.get("cross_mean_max_abs_difference_in_delta"),
            ]
            for p in context["pairs"]
            for op in ("sum", "difference")
        ),
    )
    text += "\n" + table(
        ["根", "指标", "均值", "有效配对/预期配对"],
        (
            [
                r["root_family_id"],
                key,
                r[key]["mean"],
                f"{r[key]['defined_pair_count']}/{r[key]['expected_pair_count']}",
            ]
            for r in context["roots"]
            for key in ("C11", "C22", "C12", "collision_contrast")
        ),
    )
    return text


def paired_rate(pair, key):
    value = fraction(pair.get(key))
    if value is None:
        return None
    denominator = pair[key + "_denominator"]
    return {"numerator": int(value * denominator), "denominator": denominator, "rate": pair[key]}


def reward_text(reward, *, details=False):
    text = (
        f"G={reward['group_G']}, {reward['assumption']}; 先验 {reward['prior']}。"
        "逐题G=8混合概率是带iid假设的支持诊断; 后验不是频率学置信区间或训练收益预测。"
        "低对比不触发改奖励、筛题、延长H。\n\n"
    )
    text += table(
        ["层", "有输出题/全部题", "实际非恒定题数", "逐题插件概率均值", "逐题后验预测均值"],
        (
            [
                label(r),
                f"{r['observed_question_count']}/{r['question_count']}",
                r["actual_nonconstant_question_count"],
                r["mean_per_question_plugin_mixed_probability"],
                r["mean_per_question_posterior_predictive"],
            ]
            for r in reward["chart_operation"]
        ),
    )
    if details:
        text += "\n" + table(
            [
                "题ID",
                "根",
                "层",
                "成功c/实际K",
                "有效答案/实际K",
                "实际K/预期K",
                "非恒定奖励",
                "G插件",
                "alpha",
                "beta",
                "G后验预测",
                "状态",
            ],
            (
                [
                    q["question_id"],
                    q["root_family_id"],
                    label(q),
                    f"{q['answer_successes']}/{q['K_observed']}",
                    f"{q['answer_valid']}/{q['K_observed']}",
                    f"{q['K_observed']}/{q['K_expected']}",
                    *[
                        q[k]
                        for k in (
                            "actual_nonconstant_reward",
                            "plugin_mixed_probability",
                            "posterior_alpha",
                            "posterior_beta",
                            "posterior_predictive_mixed_probability",
                            "status",
                        )
                    ],
                ]
                for q in reward["questions"]
            ),
        )
    return text


def open_fields(value, prefix=""):
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key != "open_fields":
                result.extend(open_fields(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(open_fields(item, f"{prefix}[{index}]"))
    elif value is None:
        result.append(prefix)
    return result


def engine_and_time(inputs):
    text = "\n## ENGINE实际更新、奖励与零对比\n\n"
    summaries = []
    for branch in ("continuous", "resumed"):
        rows = inputs.jsonl(f"engineering/engine/{branch}/STEPS.jsonl")
        if rows is None:
            summaries.append([branch, "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"])
            continue
        require(len({r["logical_step"] for r in rows}) == len(rows), "DUPLICATE_ENGINE_STEP")
        contrast = [flag for r in rows for flag in r["zero_contrast_groups"]]
        require(all(type(flag) is bool for flag in contrast), "INVALID_ZERO_CONTRAST_FLAG")
        rewards = Counter(str(value) for row in rows for value in row["rewards"])
        gradients = [r.get("gradient_norm") for r in rows]
        summaries.append(
            [
                branch,
                len(rows),
                f"{sum(contrast)}/{len(contrast)}",
                dict(rewards),
                f"{sum(g != 0 for g in gradients if g is not None)}/"
                f"{sum(g is not None for g in gradients)}",
            ]
        )
        if contrast and all(contrast):
            text += (
                f"{branch} 的已记录组全部零对比; 恢复一致性证据可能不包含非零梯度更新。"
                "不据此挑题重跑, 也不把ENGINE奖励表现并入AUDIT。\n\n"
            )
    text += table(
        [
            "分支",
            "已记录提交步/物理更新",
            "零对比组/全部组",
            "reward计数",
            "非零gradient_norm/已记载",
        ],
        summaries,
    )
    text += (
        "\n提交步数来自STEPS日志, 不将配方预计8步或预算预留冒充已完成; "
        "日志前失败的状态仍需运行回执核验。\n"
    )
    text += "\n## 耗时的不同口径\n\nCPU处理wall time: UNKNOWN(未有统一的实测耗时字段)。\n\n"
    allocations = []
    for path in inputs.paths("accounting/allocations/*.json"):
        row = inputs.json(path)
        terminal = row.get("terminal", {})
        actual = (
            terminal.get("elapsed_seconds") if row.get("status") == "TERMINAL_VERIFIED" else None
        )
        hours = Fraction(actual * row["gpus"], 3600) if actual is not None else None
        allocations.append(
            [
                row.get("allocation_key"),
                row.get("stage"),
                row.get("job_id"),
                row.get("status"),
                row.get("gpus"),
                row.get("seconds"),
                actual,
                hours if hours is not None else "UNKNOWN",
            ]
        )
    text += table(
        ["allocation", "stage", "job", "状态", "GPU数", "预留秒", "终态实际秒", "核验GPU小时"],
        allocations,
    )
    if not allocations:
        text += "\n调度实际GPU小时: UNKNOWN; allocation文件未提供。\n"
    latency_rows = []
    patterns = (
        ("generation_seconds", "raw/*/outputs_*.jsonl"),
        ("scoring_seconds", "raw/*/scores_*.jsonl"),
        ("generation_seconds", "engineering/engine/*/RAW_COMPLETIONS.jsonl"),
    )
    for metric, pattern in patterns:
        for path in inputs.paths(pattern):
            records = inputs.jsonl(path)
            values = [r[metric] for r in records if r.get(metric) is not None]
            require(
                all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in values),
                f"INVALID_LATENCY:{path}",
            )
            total = sum((Decimal(str(v)) for v in values), Decimal(0)) if values else None
            latency_rows.append(
                [
                    path,
                    metric,
                    f"{len(values)}/{len(records)}",
                    str(total) if total is not None else "UNKNOWN",
                ]
            )
    text += "\n" + table(["文件", "计时字段", "已记载/记录数", "记录秒数合计"], latency_rows)
    text += (
        "\n无计时字段的记录为UNKNOWN; 生成/字段评分秒数是已记录调用耗时, "
        "不含未写入的失败尝试, 不能代替含加载、等待、失败的scheduler GPU小时, 彼此不相加。\n"
    )
    return text


def facts(inputs):
    """Render already-recorded engineering evidence without upgrading its authority."""
    text = (
        "# 历史资产、生成器与工程证据\n\n状态均为既有文件记载; "
        "本生成器不重新认证CPU、CUDA或调度状态。\n\n"
    )
    csv_text = inputs.read_text("manifests/ASSET_CONTRACTS.csv")
    if csv_text:
        reader = csv.DictReader(io.StringIO(csv_text))
        text += "## 历史分支(不与新协议合表)\n\n" + table(
            reader.fieldnames, ([row[k] for k in reader.fieldnames] for row in reader)
        )
    else:
        text += "历史六字段契约: missing, 未核验。\n\n"
    files = (
        "manifests/COMMON_START.json",
        "manifests/GENERATOR_REPORT.json",
        "manifests/ROOT_SPLIT_AUDIT.json",
        "manifests/PROCESSOR_PREFLIGHT.json",
        "manifests/PROCESSOR_READABILITY_REVIEW.json",
        "manifests/RESOURCE_OVERRIDE.json",
        "manifests/EXPLICIT_TECHNICAL_RETRY.json",
        "engineering/BRIDGE_RECEIPT.json",
        "manifests/ENGINE_TEST_FREEZE.json",
        "engineering/ENGINE_COMPARE.json",
        "report/FINAL_AUDIT_STATUS.json",
    )
    for relative in files:
        document = inputs.json(relative)
        text += f"\n## {relative}\n\n"
        if document is None:
            text += "missing; 未执行或证据未提供, 不能假定完成。\n"
            continue
        text += f"[原始证据](../{relative})\n\n"
        # Top-level values are transcribed, including nested optimizer/identity/status fields.
        omitted = {
            "candidate_receipts",
            "cell_statistics",
            "artifact_sha256",
            "generation_config_expanded",
            "original_generation_config",
        }
        text += table(
            ["记录字段", "记录值"], ([k, v] for k, v in document.items() if k not in omitted)
        )
        if "cell_statistics" in document:
            text += "\n### 生成格可构造性(标签原样保留)\n\n" + table(
                ["登记格", "构造题数/预期", "constructible_fraction"],
                (
                    [k, f"{v['questions']}/{v['expected']}", v["constructible_fraction"]]
                    for k, v in document["cell_statistics"].items()
                ),
            )
        if "candidate_receipts" in document:
            text += "\n### 候选尝试与拒绝\n\n" + table(
                ["根", "D", "接受/尝试", "拒绝原因", "状态"],
                (
                    [
                        r["root_family_id"],
                        r["numeric_level"],
                        f"{r['accepted']}/{r['attempts']}",
                        r["rejected"],
                        r["status"],
                    ]
                    for r in document["candidate_receipts"]
                ),
            )
    processed = inputs.jsonl("data/processed_images.jsonl")
    text += "\n## 实际processor图像元数据分布\n\n"
    if processed:
        rows = processed
        require(len({r["image_id"] for r in rows}) == len(rows), "DUPLICATE_PROCESSED_IMAGE")
        distribution = Counter(
            (r["split"], tuple(r["processed_size"]), r["image_token_count"]) for r in rows
        )
        text += "单位为图像, 跨操作重复使用同图不增加图像数。\n\n" + table(
            ["split", "处理后尺寸", "图像tokens", "图像数"],
            ([split, size, tokens, n] for (split, size, tokens), n in sorted(distribution.items())),
        )
    else:
        text += "missing; 不从原图尺寸推算实际图像token。\n"
    text += (
        "\nENGINE只检验接口及精确恢复; 工程模型不替换公共起点, 不能进入科学能力响应表。"
        "ENGINE_COMPARE 的 PASS 是原回执声明, 本报告未加载checkpoint复验。"
        "实际成本与保守预留分开, 失败和UNKNOWN不自动释放额度; GPU小时只记账。\n"
    )
    text += engine_and_time(inputs)
    text += (
        "\n最终独立核验在本报告之后生成: "
        "[FINAL_EVIDENCE_VERIFICATION.json](FINAL_EVIDENCE_VERIFICATION.json)、"
        "[FINAL_EVIDENCE_SHA256.json](FINAL_EVIDENCE_SHA256.json)。"
        "本生成器只保留链接, 不读取或哈希这两份后续证据, 避免循环依赖。\n"
    )
    return text


def format_text(inputs):
    text = (
        "## 格式工程面板\n\n门槛: schema∧顺序∧L∧非截断, 总体≥95%, 每图表/运算层≥90%。"
        "BASE通过不用桥接; 独立CHECK失败停止, 不作为第二轮调参集。格式PASS不是科学能力PASS。\n\n"
    )
    for stage in ("FORMAT_BASE_TEST", "FORMAT_TUNE_POST_BRIDGE", "FORMAT_CHECK"):
        document = inputs.json(f"tables/{stage}/FORMAT_REPORT.json")
        bound = inputs.json(f"tables/{stage}_REPORT.json")
        row = cohort(document, stage, stage, format_report=True)
        bound_row = cohort(bound, stage, stage + "_BOUND", format_report=True)
        if row and bound_row:
            for key in ("model_hash", "overall", "chart_operation", "status", "gate_passed"):
                require(row[key] == bound_row[key], f"FORMAT_RECEIPT_TABLE_MISMATCH:{stage}:{key}")
        row = row or bound_row
        text += f"### {stage}\n\n"
        if row is None:
            text += "missing; 未执行或表未提供。\n\n"
            continue
        groups(row["chart_operation"])
        text += f"记录状态: {cell(row['status'])}; model_hash: {cell(row['model_hash'])}。\n\n"
        text += (
            table(
                [
                    "层",
                    "题/根",
                    "记录/预期",
                    "schema",
                    "顺序",
                    "LP",
                    "LA",
                    "L",
                    "格式合格",
                    "截断",
                    "截断未知",
                    "解析失败",
                ],
                (
                    [
                        label(r),
                        f"{r['question_count']}/{r['root_family_count']}",
                        f"{r['recorded_attempts']}/{r['expected_completions']}",
                        *[
                            r[k]
                            for k in (
                                "schema_ok",
                                "field_order_ok",
                                "L_P",
                                "L_A",
                                "L",
                                "format_ok",
                                "truncated",
                                "truncation_unknown_count",
                                "parse_error_counts",
                            )
                        ],
                    ]
                    for r in [row["overall"], *row["chart_operation"]]
                ),
            )
            + "\n"
        )
    return text


def build_report(run_root):
    inputs = Inputs(run_root)
    tables = {
        name: cohort(inputs.json(f"tables/{STAGE}/{name}.json"), STAGE, name) for name in TABLES
    }
    receipt = inputs.json(f"tables/{STAGE}/SCORING_RECEIPT.json")
    common = inputs.json("manifests/COMMON_START.json")
    validate_measurement(tables, receipt, common)
    coverage = tables["COVERAGE"]
    complete = bool(
        all(tables.values())
        and receipt
        and receipt.get("status") == "SCORED"
        and coverage["complete"] is True
        and coverage["question_count"] == 384
        and coverage["root_family_count"] == 16
        and coverage["expected_completions"] == coverage["recorded_attempts"] == 1536
        and coverage["status_counts"] == {"completed": 1536}
        and all(
            coverage[k] == 0
            for k in ("missing_sample_slots", "duplicate_sample_slots", "unexpected_sample_slots")
        )
    )
    status = "MEASUREMENT_TABLES_COMPLETE" if complete else "MEASUREMENT_INCOMPLETE"
    main = [
        "# MM-AUDIT 科学数值附录\n",
        f"表格状态: **{status}**。" + ("" if complete else "测量未完成或证据缺项。"),
        "本报告只呈现保存的表, 未调用模型、未重评分、未重采样。表齐不代表执行门槛或科学假设通过; "
        "不提升release状态, MM-DEV authorized=false。",
        "所有计数比例保留原分子/分母; missing不补0。"
        "较长精确分数的≈值仅供阅读, 精确值保留在链接JSON。"
        "P是报告读数忠实性, C是相对读数的运算一致性, 不认证内部感知或推理机制。",
        "[历史/生成器/ENGINE/资源证据](SCIENTIFIC_EVIDENCE_zh.md)",
    ]
    formats = format_text(inputs)
    main.append(formats)
    main.append("## MEASUREMENT_AUDIT 覆盖与身份\n")
    if coverage:
        main.append(table(["字段", "值"], coverage.items()))
    else:
        main.append("missing; 没有测量覆盖表。BASE/CHECK回答不代替测量回答。")
    documents = {}
    counts = tables["COUNTS"]
    if counts:
        main.append("## 精确计数与成功率\n\n" + source_link("COUNTS"))
        main.append(
            "计数的全样本分母是已记录回答N, 预期但未生成槽位另见覆盖; 缺联合字段不填000。"
            "六格都条件于同一L; 101/110在精确口径不成立。容差δ/2闭区间独立统计八格。"
        )
        main.append(counts_text([counts["overall"], *counts["chart_operation"]]))
        details = "# 24个V/D条件格与容差八格\n\n" + source_link("COUNTS") + "\n\n"
        details += (
            "标签依次为 chart_type / operation / visual_level / numeric_level; "
            "固定格, 不按表现选择。\n\n"
        )
        details += counts_text(counts["factor_cells"])
        details += "\n## 容差八格原始计数\n\n" + table(
            ["层", "L", *TOLERANCE],
            (
                [
                    label(r),
                    r["joint_scorable_denominator"],
                    *[r["tolerance_counts"][k] for k in TOLERANCE],
                ]
                for r in [counts["overall"], *counts["chart_operation"], *counts["factor_cells"]]
            ),
        )
        documents["SCIENTIFIC_COUNTS_zh.md"] = details
        main.append("[24个条件格、总体/六层/24格的容差八格](SCIENTIFIC_COUNTS_zh.md)")
    else:
        main.append("精确计数与容差: missing, 未产生科学数值表。")
    if tables["GEOMETRY"]:
        main.append("## 误差与共享形状\n\n" + source_link("GEOMETRY"))
        main.append(
            "e=(r-v)/δ; 近失为0<max|e|≤1。S只在非零误差向量定义, 正确读数的0/0为missing。"
            "S大不代表偏差小; εO=εP+εC不证明训练修复机制。冻结几何表只含总体与六层, 不另算24格。"
        )
        main.append(geometry_text(tables["GEOMETRY"]))
    if tables["ROOT_EQUAL"]:
        root, boot = tables["ROOT_EQUAL"], tables["ROOT_BOOTSTRAP"]
        main.append(
            "## 根等权与描述性区间\n\n"
            + source_link("ROOT_EQUAL")
            + ("; " + source_link("ROOT_BOOTSTRAP") if boot else "; ROOT_BOOTSTRAP missing")
        )
        main.append(
            "题内K均值→根内固定24格→根等权。缺支持不补零; 条件均值与完整固定权重均值分列。"
            "根簇区间条件于所测模型/面板, 不能解释为训练总体区间或用1536回答代替16根。"
        )
        if boot:
            main.append(
                f"保存的bootstrap seed={boot['seed']}, 请求重复={boot['requested_replicates']}。"
            )
        main.append(root_text(root, boot))
        details = "# 逐根支持与固定权重\n\n" + source_link("ROOT_EQUAL") + "\n\n"
        details += table(
            ["根", "指标", "条件均值", "固定权重分子", "支持权重", "支持格/24", "完整固定权重均值"],
            (
                [
                    r["root_family_id"],
                    key,
                    value["conditional_mean"],
                    value["fixed_weight_numerator"],
                    value["supported_weight"],
                    f"{value['defined_cells']}/{value['expected_cells']}",
                    value["complete_fixed_weight_mean"],
                ]
                for r in root["roots"]
                for key, value in r["metrics"].items()
            ),
        )
        documents["SCIENTIFIC_ROOTS_zh.md"] = details
        main.append("[逐根支持、固定权重分子与缺失](SCIENTIFIC_ROOTS_zh.md)")
    context = tables["CONTEXT_AUDIT"]
    if context:
        documents["SCIENTIFIC_CONTEXT_zh.md"] = (
            "# 匹配提示噪声对照\n\n" + source_link("CONTEXT_AUDIT") + "\n\n" + context_text(context)
        )
        main.append(
            "## 匹配提示\n\n"
            + f"记录状态: {cell(context['status'])}; 配对记录数: {len(context['pairs'])}。"
            "CONTEXT_AUDIT与S分开, 不是内部因果机制证据。"
            "[逐对C11/C22/C12、差异、分母、P与覆盖](SCIENTIFIC_CONTEXT_zh.md)"
        )
        if tables["ROOT_BOOTSTRAP"]:
            main.append(
                table(
                    ["context指标", "95%下界", "95%上界", "有效重复", "undefined"],
                    (
                        [
                            k,
                            *[
                                v[field]
                                for field in (
                                    "percentile_95_low",
                                    "percentile_95_high",
                                    "defined_replicates",
                                    "undefined_replicates",
                                )
                            ],
                        ]
                        for k, v in tables["ROOT_BOOTSTRAP"]["metrics"].items()
                        if k.startswith("context_")
                    ),
                )
            )
    if tables["REWARD_SUPPORT"]:
        reward = tables["REWARD_SUPPORT"]
        main.append(
            "## 答案奖励支持\n\n" + source_link("REWARD_SUPPORT") + "\n\n" + reward_text(reward)
        )
        documents["SCIENTIFIC_REWARD_zh.md"] = (
            "# 逐题奖励支持\n\n"
            + source_link("REWARD_SUPPORT")
            + "\n\n"
            + reward_text(reward, details=True)
        )
        main.append("[逐题c/K、LA、实际非恒定、G插件与后验](SCIENTIFIC_REWARD_zh.md)")
    if tables["FIELD_METRICS"]:
        fields = tables["FIELD_METRICS"]
        main.append(
            "## 字段NLL与token熵\n\n"
            + source_link("FIELD_METRICS")
            + "\n\n"
            + field_text([fields["overall"]])
        )
        documents["SCIENTIFIC_FIELDS_zh.md"] = (
            "# 六层与24条件格的字段测量\n\n"
            + source_link("FIELD_METRICS")
            + "\n\n"
            + field_text([*fields["chart_operation"], *fields["factor_cells"]])
        )
        main.append("[六层/24条件格、来源、token分母及边界规则](SCIENTIFIC_FIELDS_zh.md)")
    documents["SCIENTIFIC_EVIDENCE_zh.md"] = facts(inputs)
    release = inputs.json("report/FINAL_AUDIT_STATUS.json")
    proposal = inputs.json("report/DEV_FREEZE_PROPOSAL.json")
    main.append("## 原release状态与后续开放字段\n")
    main.append(
        f"原release状态: {cell(release.get('status') if release else None)}; "
        "本附录不修改该文件或其权限。"
    )
    delivery = release.get("status") if release else "RELEASE_MISSING"
    if proposal:
        opened = sorted(set(open_fields(proposal) + proposal.get("open_fields", [])))
        main.append(
            f"原提案状态: {cell(proposal.get('status'))}; "
            f"原authorized={cell(proposal.get('authorized'))}。"
        )
        if opened:
            if delivery == "READY_FOR_DEV_REVIEW":
                delivery = "READY_WITH_OPEN_DEV_FIELDS"
            main.append(
                "存在未冻结字段, 仅原release已READY时解释为READY_WITH_OPEN_DEV_FIELDS; "
                "原blocked/ENGINE失败仍保持, 不补默认参数。\n\n"
                + table(["开放字段"], ([k] for k in opened))
            )
    else:
        main.append("DEV补充表missing; 保持未授权, 不生成未来训练参数。")
    main.append(f"交付解释: **{cell(delivery)}**; authorized=false。")
    main.append(
        "本轮描述测量可用性、固定公共起点错误分布与工程成本; "
        "没有检验μ、S或context对未见训练收益/决策增益的预测价值。"
        "保留稀少z、恒定S、低奖励对比、不利结果和技术失败。"
    )
    main.append(
        "## 缺少的输入\n\n"
        + (
            table(["输入路径"], ([k] for k in sorted(inputs.missing)))
            if inputs.missing
            else "无缺少输入路径。"
        )
    )
    main.append(
        "本报告输入/输出SHA256见 "
        "[SCIENTIFIC_REPORT_MANIFEST.json](SCIENTIFIC_REPORT_MANIFEST.json)。"
        "清单证明此次渲染使用的文件字节, 不替代原始运行完整性与checkpoint核验。"
    )
    documents["SCIENTIFIC_RESULTS_zh.md"] = "\n\n".join(main) + "\n"
    inputs.stable()
    report_dir = inputs.root / "report"
    require(report_dir.resolve().is_relative_to(inputs.root), "REPORT_OUTSIDE_RUN_ROOT")
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": status,
        "scientific_stage_authorized": False,
        "dev_authorized": False,
        "release_status_recorded": release.get("status") if release else None,
        "delivery_interpretation": delivery,
        "new_model_calls": 0,
        "new_training_runs": 0,
        "rescored_answers": 0,
        "input_sha256": dict(sorted(inputs.hashes.items())),
        "missing_inputs": sorted(inputs.missing),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {
            f"report/{name}": hashlib.sha256(text.encode()).hexdigest()
            for name, text in sorted(documents.items())
        },
        "scope": "Saved-table presentation only; no runtime or release gate certification",
    }
    documents["SCIENTIFIC_REPORT_MANIFEST.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    for name, text in documents.items():
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=report_dir, delete=False
        ) as out:
            out.write(text)
            temporary = Path(out.name)
        os.replace(temporary, report_dir / name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.run_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "dev_authorized": False,
                "report": str(args.run_root / "report/SCIENTIFIC_RESULTS_zh.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
