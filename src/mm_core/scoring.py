"""Exact scoring and explicit-denominator reporting for the frozen MM audit.

No model access. Missing completions remain missing; missing response fields
count as unsuccessful only in full-response metrics. Statistical units are root
families. Serialization preserves rational numerators and denominators.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any

from .contracts import (
    assert_split_integrity,
    beta_mixed_predictive,
    collision_audit,
    mixed_probability,
    rational,
    reconstruct_six,
    score,
)

SCORER_VERSION = "MM-CORE-F1-exact-v1"
EXACT_CELLS = ("111", "100", "011", "010", "001", "000")
TOLERANCE_CELLS = tuple("".join(map(str, bits)) for bits in product((0, 1), repeat=3))
CHARTS = ("grouped_bar", "line")
OPERATIONS = ("sum", "difference", "range")
LEVELS = ("low", "high")
STAGE_SPLITS = {
    "FORMAT_BASE_TEST": "FORMAT_TUNE",
    "FORMAT_TUNE_POST_BRIDGE": "FORMAT_TUNE",
    "FORMAT_CHECK": "FORMAT_CHECK",
    "MEASUREMENT_AUDIT": "AUDIT_MEASURE",
    "FORMAT_TUNE": "FORMAT_TUNE",
    "AUDIT_MEASURE": "AUDIT_MEASURE",
}
MEASUREMENT_SPLITS = frozenset(STAGE_SPLITS.values())
ALLOWED_SPLITS = MEASUREMENT_SPLITS | {"BRIDGE_TRAIN", "ENGINE_TEST"}
FIELD_METRIC_SOURCES = {
    "self_field_surprisal": "generated_tokens_actual_self_prefix",
    "gold_teacher_forced_field_nll": "gold_completion_answer_prefix_contains_gold_readings",
}
FIELD_SCORE_KEYS = frozenset(FIELD_METRIC_SOURCES) | {"scoring_seconds"}


def discover_raw_files(run_root: Path, stage: str | None = None) -> tuple[list[Path], list[Path]]:
    """Keep immutable completions and optional score sidecars in separate lists."""
    if stage is not None and stage not in STAGE_SPLITS:
        raise ValueError(f"UNREGISTERED_MEASUREMENT_STAGE:{stage}")
    base = Path(run_root) / "raw"
    if stage:
        base = base / stage
    paths = sorted(base.rglob("*.jsonl"))
    outputs = [
        path
        for path in paths
        if path.name.lower().startswith("outputs_")
        or path.name.lower() in {"outputs.jsonl", "completions.jsonl", "raw_completions.jsonl"}
    ]
    scores = [
        path
        for path in paths
        if path.name.lower().startswith("scores_") or path.name.lower() == "scores.jsonl"
    ]
    return outputs, scores


def join_field_scores(
    outputs: Sequence[Mapping[str, Any]], score_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Join optional measurements without changing any original completion field.

    Every shared completion field must match byte-for-byte after canonical JSON
    serialization (including text, tokens and routing); only status may differ.
    Duplicate or orphan score records are errors, even if their values agree.
    """

    def canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    completion_map = {}
    for row in outputs:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("MISSING_REQUEST_ID")
        if request_id in completion_map and canonical(completion_map[request_id]) != canonical(row):
            raise ValueError(f"CONFLICTING_REQUEST_ID:{request_id}")
        completion_map[request_id] = row
    joined = {}
    for row in score_rows:
        request_id = row.get("request_id")
        if request_id in joined:
            raise ValueError(f"DUPLICATE_FIELD_SCORE_REQUEST:{request_id}")
        if request_id not in completion_map:
            raise ValueError(f"ORPHAN_FIELD_SCORE_REQUEST:{request_id}")
        raw = completion_map[request_id]
        if row.get("status") != "scored" or raw.get("status") != "completed":
            raise ValueError(f"INVALID_FIELD_SCORE_STATUS:{request_id}")
        for key in (
            "request_id",
            "question_id",
            "stage",
            "sample_index",
            "model_hash",
            "processor_hash",
            "seed",
        ):
            if key not in raw or key not in row or canonical(row[key]) != canonical(raw[key]):
                raise ValueError(f"FIELD_SCORE_IDENTITY_MISMATCH:{key}:{request_id}")
        for key, value in raw.items():
            if key == "status":
                continue
            if key not in row or canonical(value) != canonical(row[key]):
                raise ValueError(f"FIELD_SCORE_COMPLETION_MISMATCH:{key}:{request_id}")
        unexpected = set(row) - set(raw) - FIELD_SCORE_KEYS - {"status"}
        if unexpected:
            raise ValueError(f"UNREGISTERED_FIELD_SCORE_KEYS:{sorted(unexpected)}")
        joined[request_id] = {key: row[key] for key in FIELD_SCORE_KEYS if key in row}
    return [{**raw, **joined.get(raw["request_id"], {})} for raw in outputs]


def json_exact(value: Any) -> Any:
    """Keep exact values portable; ``float`` is explicitly a display aid only."""
    if isinstance(value, Fraction):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "float": _finite_float(value),
        }
    if is_dataclass(value):
        return json_exact(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_exact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_exact(v) for v in value]
    return value


def _finite_float(value: Any) -> float | None:
    try:
        display = float(value)
    except (OverflowError, ValueError):
        return None
    return display if math.isfinite(display) else None


def _sqrt_float(value: Fraction) -> float | None:
    # The exact MSE can exceed binary float range even when its root does not.
    with localcontext() as context:
        context.prec = 34
        return _finite_float((Decimal(value.numerator) / Decimal(value.denominator)).sqrt())


def _mean(values: Sequence[Any]) -> Any:
    return sum(values, Fraction(0)) / len(values) if values else None


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": Fraction(numerator, denominator) if denominator else None,
    }


def _pick(row: Mapping[str, Any], canonical: str, alias: str) -> Any:
    if canonical in row and alias in row and row[canonical] != row[alias]:
        # Decimal spelling is not a disagreement in numeric aliases.
        a, b = row[canonical], row[alias]
        try:
            equal = (
                tuple(map(rational, a)) == tuple(map(rational, b))
                if isinstance(a, list) and isinstance(b, list)
                else rational(a) == rational(b)
            )
        except (ValueError, TypeError, ArithmeticError):
            equal = False
        if not equal:
            raise ValueError(f"CONFLICTING_QUESTION_ALIASES:{canonical}:{alias}")
    if canonical in row:
        return row[canonical]
    if alias in row:
        return row[alias]
    raise ValueError(f"MISSING_QUESTION_FIELD:{canonical}")


def normalize_question(row: Mapping[str, Any]) -> dict[str, Any]:
    q = dict(row)
    aliases = {
        "true_values_decimal": "true_values",
        "delta_decimal": "delta",
        "image_hash": "image_sha256",
        "visual_level": "V",
        "numeric_level": "D",
    }
    for canonical, alias in aliases.items():
        q[canonical] = _pick(q, canonical, alias)
    for key in ("question_id", "root_family_id", "split", "readset_id", "image_hash"):
        if not isinstance(q.get(key), str) or not q[key]:
            raise ValueError(f"MISSING_QUESTION_FIELD:{key}")
    if q["split"] not in ALLOWED_SPLITS:
        raise ValueError(f"UNAUTHORIZED_QUESTION_SPLIT:{q['split']}")
    if q.get("chart_type") not in CHARTS or q.get("operation") not in OPERATIONS:
        raise ValueError("INVALID_CHART_OR_OPERATION")
    if q["visual_level"] not in LEVELS or q["numeric_level"] not in LEVELS:
        raise ValueError("INVALID_FACTOR_LEVEL")
    truth = tuple(map(rational, q["true_values_decimal"]))
    required = 3 if q["operation"] == "range" else 2
    roles = q.get("ordered_item_ids")
    if (
        len(truth) != required
        or not isinstance(roles, list)
        or len(roles) != required
        or any(not isinstance(role, str) or not role for role in roles)
        or len(set(roles)) != required
    ):
        raise ValueError("INVALID_ORDERED_ROLE_MAPPING")
    if rational(q["delta_decimal"]) <= 0:
        raise ValueError("DELTA_MUST_BE_POSITIVE")
    if not isinstance(q.get("quantity_units"), str) or not q["quantity_units"]:
        raise ValueError("MISSING_QUANTITY_UNITS")
    q.setdefault("axis_min", "0")
    q.setdefault("axis_max", "100")
    if rational(q["axis_min"]) >= rational(q["axis_max"]):
        raise ValueError("INVALID_AXIS")
    return q


def _prepare(
    questions: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    stage: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    qs = [normalize_question(q) for q in questions]
    qmap = {q["question_id"]: q for q in qs}
    if len(qmap) != len(qs):
        raise ValueError("DUPLICATE_QUESTION_ID")
    assert_split_integrity(qs)
    if stage is not None and stage not in STAGE_SPLITS:
        raise ValueError(f"UNREGISTERED_MEASUREMENT_STAGE:{stage}")
    seen: dict[str, dict[str, Any]] = {}
    scored = []
    duplicate_records = 0
    for output in outputs:
        raw = dict(output)
        if not isinstance(raw.get("request_id"), str) or not raw["request_id"]:
            raise ValueError("MISSING_REQUEST_ID")
        request_id = raw["request_id"]
        if request_id in seen:
            if raw != seen[request_id]:
                raise ValueError(f"CONFLICTING_REQUEST_ID:{request_id}")
            duplicate_records += 1
            continue
        seen[request_id] = raw
        if raw.get("stage") not in STAGE_SPLITS:
            raise ValueError(f"UNREGISTERED_MEASUREMENT_STAGE:{raw.get('stage')}")
        if stage is not None and raw["stage"] != stage:
            continue
        qid = raw.get("question_id")
        if qid not in qmap:
            raise ValueError(f"UNKNOWN_QUESTION_ID:{qid}")
        q = qmap[qid]
        if q["split"] != STAGE_SPLITS[raw["stage"]]:
            raise ValueError("OUTPUT_STAGE_SPLIT_MISMATCH")
        index = raw.get("sample_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("INVALID_SAMPLE_INDEX")
        if not isinstance(raw.get("status"), str) or not raw["status"]:
            raise ValueError("MISSING_OUTPUT_STATUS")
        failed_without_text = (
            raw["status"].lower() == "failed" and "raw_text" in raw and raw["raw_text"] is None
        )
        if not isinstance(raw.get("raw_text"), str) and not failed_without_text:
            raise ValueError("MISSING_RAW_TEXT_USE_EMPTY_TEXT_FOR_FAILED_ATTEMPT")
        if not isinstance(raw.get("truncated"), bool) and not (
            failed_without_text and "truncated" in raw and raw["truncated"] is None
        ):
            raise ValueError("MISSING_TRUNCATION_STATUS")
        for source_key, question_key in (
            ("image_hash", "image_hash"),
            ("image_sha256", "image_hash"),
            ("prompt_hash", "actual_model_prompt_hash"),
        ):
            if source_key in raw and question_key in q and raw[source_key] != q[question_key]:
                raise ValueError(f"OUTPUT_PROVENANCE_MISMATCH:{source_key}")
        result = score(
            "" if failed_without_text else raw["raw_text"],
            q["true_values_decimal"],
            q["operation"],
            q["delta_decimal"],
            (q["axis_min"], q["axis_max"]),
        )
        p = result.pop("parsed")
        item = {
            **raw,
            "scorer_version": SCORER_VERSION,
            **{
                k: q[k]
                for k in (
                    "root_family_id",
                    "split",
                    "chart_type",
                    "operation",
                    "visual_level",
                    "numeric_level",
                    "readset_id",
                )
            },
            "image_hash": q["image_hash"],
            "ordered_item_ids": q["ordered_item_ids"],
            "quantity_units": q["quantity_units"],
            "delta": rational(q["delta_decimal"]),
            "parsed_readings": p.readings,
            "parsed_answer": p.answer,
            "schema_ok": p.schema_ok,
            "field_order_ok": p.field_order_ok,
            "parse_errors": ("NO_COMPLETION_TEXT_FAILED_ATTEMPT",)
            if failed_without_text
            else p.errors,
            "completion_text_available": not failed_without_text,
            **result,
        }
        item["format_ok"] = bool(
            p.schema_ok
            and p.field_order_ok
            and p.L
            and not raw["truncated"]
            and raw["status"].lower() in {"completed", "success", "ok"}
        )
        item["residual_identity_error"] = (
            item["eps_O"] - item["eps_P"] - item["eps_C"] if item["L"] else None
        )
        scored.append(item)
    return qs, scored, duplicate_records


def _cohorts(
    qs: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], stage: str | None
) -> list[tuple[str, str | None, list, list]]:
    keys = sorted(
        {(r["stage"], r.get("model_hash")) for r in rows}, key=lambda key: (key[0], str(key[1]))
    )
    if not keys:
        keys = (
            [(stage, None)]
            if stage
            else [
                (s, None)
                for s in ("FORMAT_BASE_TEST", "FORMAT_CHECK", "MEASUREMENT_AUDIT")
                if any(q["split"] == STAGE_SPLITS[s] for q in qs)
            ]
        )
    return [
        (
            s,
            model,
            [q for q in qs if q["split"] == STAGE_SPLITS[s]],
            [r for r in rows if r["stage"] == s and r.get("model_hash") == model],
        )
        for s, model in keys
    ]


def _groups(qs: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], *, factors: bool = False):
    fields = (
        ("chart_type", "operation", "visual_level", "numeric_level")
        if factors
        else ("chart_type", "operation")
    )
    domains = (CHARTS, OPERATIONS, LEVELS, LEVELS) if factors else (CHARTS, OPERATIONS)
    for values in product(*domains):
        label = dict(zip(fields, values, strict=True))
        yield (
            label,
            [q for q in qs if all(q[k] == v for k, v in label.items())],
            [r for r in rows if all(r[k] == v for k, v in label.items())],
        )


def _coverage(
    qs: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], k: int
) -> dict[str, Any]:
    slots = Counter((r["question_id"], r["sample_index"]) for r in rows)
    expected = {(q["question_id"], index) for q in qs for index in range(k)}
    missing = expected - set(slots)
    unexpected = set(slots) - expected
    n = len(rows)
    return {
        "question_count": len(qs),
        "root_family_count": len({q["root_family_id"] for q in qs}),
        "expected_completions": len(expected),
        "recorded_attempts": n,
        "unique_sample_slots": len(slots),
        "missing_sample_slots": len(missing),
        "unexpected_sample_slots": len(unexpected),
        "duplicate_sample_slots": sum(c - 1 for c in slots.values()),
        "complete": bool(expected)
        and not missing
        and not unexpected
        and all(c == 1 for c in slots.values()),
        "L_P": _rate(sum(r["L_P"] for r in rows), n),
        "L_A": _rate(sum(r["L_A"] for r in rows), n),
        "L": _rate(sum(r["L"] for r in rows), n),
        "status_counts": dict(Counter(r["status"] for r in rows)),
    }


def _format_one(qs: list, rows: list, k: int) -> dict[str, Any]:
    report = _coverage(qs, rows, k)
    for field in ("schema_ok", "field_order_ok", "format_ok", "truncated"):
        denominator = sum(r[field] is not None for r in rows)
        report[field] = _rate(sum(bool(r[field]) for r in rows), denominator)
    report["truncation_unknown_count"] = sum(r["truncated"] is None for r in rows)
    report["parse_error_counts"] = dict(Counter(err for r in rows for err in r["parse_errors"]))
    return report


def _format_report(
    qs: list, rows: list, k: int, stage: str | None, strict_panel: bool = True
) -> dict[str, Any]:
    cohorts = []
    for s, model, panel, selected in _cohorts(qs, rows, stage):
        if STAGE_SPLITS[s] not in {"FORMAT_TUNE", "FORMAT_CHECK"}:
            continue
        overall = _format_one(panel, selected, k)
        layers = [
            {**label, **_format_one(subq, subr, k)}
            for label, subq, subr in _groups(panel, selected)
        ]
        expected_size = len(panel) == 96 and len({q["root_family_id"] for q in panel}) == 4
        complete = overall["complete"] and (expected_size or not strict_panel)
        passed = (
            complete
            and overall["format_ok"]["rate"] >= Fraction(95, 100)
            and all(
                layer["format_ok"]["rate"] is not None
                and layer["format_ok"]["rate"] >= Fraction(9, 10)
                for layer in layers
            )
        )
        status = (
            "NO_OUTPUTS"
            if not selected
            else ("INCOMPLETE" if not complete else "PASS" if passed else "FORMAT_NOT_READY")
        )
        cohorts.append(
            {
                "stage": s,
                "model_hash": model,
                "status": status,
                "gate_passed": bool(passed),
                "plan_panel_size_matches": expected_size,
                "overall": overall,
                "chart_operation": layers,
            }
        )
    return {
        "status": (
            "NO_OUTPUTS"
            if not rows
            else "PASS"
            if cohorts and all(c["gate_passed"] for c in cohorts)
            else "NOT_PASSED"
        ),
        "gate_passed": bool(cohorts) and all(c["gate_passed"] for c in cohorts),
        "thresholds": {"overall": Fraction(95, 100), "each_chart_operation": Fraction(9, 10)},
        "engineering_gate_only": True,
        "cohorts": cohorts,
    }


def build_format_report(
    questions: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    expected_k: int = 4,
    stage: str | None = None,
    *,
    strict_panel: bool = True,
) -> dict[str, Any]:
    """Public raw-record format gate. Returns JSON-ready ``gate_passed``/status."""
    _validate_k(expected_k)
    qs, rows, _ = _prepare(questions, outputs, stage)
    return json_exact(_format_report(qs, rows, expected_k, stage, strict_panel))


def _counts(rows: list) -> dict[str, Any]:
    n, lp, la, legal = (
        len(rows),
        sum(r["L_P"] for r in rows),
        sum(r["L_A"] for r in rows),
        sum(r["L"] for r in rows),
    )
    exact = {c: sum(r["cell_exact"] == c for r in rows) for c in EXACT_CELLS}
    tolerance = {c: sum(r["cell_tolerance"] == c for r in rows) for c in TOLERANCE_CELLS}
    assert sum(exact.values()) == legal == sum(tolerance.values())
    marginal_counts = {
        "p": exact["111"] + exact["100"],
        "a": exact["111"] + exact["011"] + exact["001"],
        "c": exact["111"] + exact["011"] + exact["010"],
        "j": exact["111"],
        "z": exact["001"],
    }
    marginals = {key: _rate(value, legal) for key, value in marginal_counts.items()}
    if legal:
        reconstructed = reconstruct_six(
            *(marginals[key]["rate"] for key in ("p", "a", "c", "j", "z"))
        )
        assert reconstructed == {cell: Fraction(count, legal) for cell, count in exact.items()}
    return {
        "recorded_attempts": n,
        "joint_scorable_denominator": legal,
        "exact_counts": exact,
        "tolerance_counts": tolerance,
        "conditional_on_L": marginals,
        "P_full": _rate(sum(r["P_full"] for r in rows), n),
        "A_full": _rate(sum(r["A_full"] for r in rows), n),
        "J_full": _rate(sum(r["J_full"] for r in rows), n),
        "P_given_LP": _rate(sum(r["P_full"] for r in rows), lp),
        "A_given_LA": _rate(sum(r["A_full"] for r in rows), la),
        "six_cell_reconstruction_checked": bool(legal),
        "missing_joint_fields_not_000": n - legal,
    }


def _geometry(rows: list) -> dict[str, Any]:
    valid = [r for r in rows if r["L_P"]]
    shapes = [r["S"] for r in valid if r["S"] is not None]
    roles = []
    for arity in sorted({len(r["e"]) for r in valid}):
        same = [r for r in valid if len(r["e"]) == arity]
        for position in range(arity):
            values = [r["e"][position] for r in same]
            mse = _mean([value * value for value in values])
            roles.append(
                {
                    "arity": arity,
                    "ordered_role_index": position,
                    "denominator": len(values),
                    "mean_bias": _mean(values),
                    "MAE": _mean([abs(v) for v in values]),
                    "MSE": mse,
                    "RMSE_float": _sqrt_float(mse),
                }
            )
    residuals = [abs(r["residual_identity_error"]) for r in rows if r["L"]]
    assert all(error == 0 for error in residuals)
    return {
        "LP_denominator": len(valid),
        "S_denominator": len(shapes),
        "S_missing_zero_error_count": sum(r["P"] for r in valid),
        "S_direct_error_response_mean": _mean(shapes),
        "near_miss": _rate(sum(r["near_miss"] for r in valid), len(valid)),
        "mean_max_abs_error": _mean([max(map(abs, r["e"])) for r in valid]),
        "outside_axis": _rate(sum(bool(r["outside_axis"]) for r in valid), len(valid)),
        "roles": roles,
        "residual_identity_checks": len(residuals),
        "max_exact_residual_identity_error": max(residuals) if residuals else None,
        "eps_P_mean": _mean([r["eps_P"] for r in valid]),
        "eps_C_mean": _mean([r["eps_C"] for r in rows if r["L"]]),
        "eps_O_mean": _mean([r["eps_O"] for r in rows if r["L"]]),
    }


def _question_metrics(rows: list) -> dict[str, Any]:
    values = {
        key: _mean([Fraction(r[key]) for r in rows])
        for key in ("P_full", "A_full", "J_full", "L_P", "L_A", "L")
    }
    legal = [r for r in rows if r["L"]]
    for key, field in (("p", "P"), ("a", "A"), ("c", "C")):
        values[key] = _mean([Fraction(r[field]) for r in legal])
    values["j"] = _mean([Fraction(r["J_full"]) for r in legal])
    values["z"] = _mean([Fraction(r["cell_exact"] == "001") for r in legal])
    values["S"] = _mean([r["S"] for r in rows if r["S"] is not None])
    values["near_miss"] = _mean([Fraction(r["near_miss"]) for r in rows if r["L_P"]])
    values["max_abs_error"] = _mean([max(map(abs, r["e"])) for r in rows if r["L_P"]])
    field_reports = _field_metrics(rows)
    for source in FIELD_METRIC_SOURCES:
        for field in ("readings", "answer"):
            for metric, key in (("nll", "mean_nll"), ("entropy", "mean_token_entropy")):
                values[f"{source}_{field}_{metric}"] = field_reports[source]["fields"][field][key][
                    "sequence_mean"
                ]
    return values


def _root_equal(qs: list, rows: list) -> dict[str, Any]:
    byq: dict[str, list] = defaultdict(list)
    for row in rows:
        byq[row["question_id"]].append(row)
    metrics = {q["question_id"]: _question_metrics(byq[q["question_id"]]) for q in qs}
    root_reports = []
    for root in sorted({q["root_family_id"] for q in qs}):
        rootq = [q for q in qs if q["root_family_id"] == root]
        cell_metrics = []
        for _, cellq, _ in _groups(rootq, [], factors=True):
            cell_metrics.append(
                {
                    key: _mean(
                        [
                            metrics[q["question_id"]][key]
                            for q in cellq
                            if metrics[q["question_id"]][key] is not None
                        ]
                    )
                    for key in _question_metrics([])
                }
            )
        root_metrics = {}
        for key in _question_metrics([]):
            values = [cell[key] for cell in cell_metrics if cell[key] is not None]
            root_metrics[key] = {
                "conditional_mean": _mean(values),
                "fixed_weight_numerator": sum(values, Fraction(0)) / 24,
                "supported_weight": Fraction(len(values), 24),
                "defined_cells": len(values),
                "expected_cells": 24,
                "complete_fixed_weight_mean": _mean(values) if len(values) == 24 else None,
            }
        root_reports.append({"root_family_id": root, "metrics": root_metrics})
    aggregate = {}
    for key in _question_metrics([]):
        values = [
            root["metrics"][key]["conditional_mean"]
            for root in root_reports
            if root["metrics"][key]["conditional_mean"] is not None
        ]
        complete = [root["metrics"][key]["complete_fixed_weight_mean"] for root in root_reports]
        aggregate[key] = {
            "root_equal_conditional_mean": _mean(values),
            "supported_root_count": len(values),
            "root_count": len(root_reports),
            "complete_fixed_weight_mean": _mean(complete)
            if complete and all(v is not None for v in complete)
            else None,
        }
    return {
        "weighting": "question mean, equal 24 chart/operation/V/D cells, equal root families",
        "missing_policy": (
            "No imputation. Conditional means explicitly expose supported cells and roots."
        ),
        "aggregate": aggregate,
        "roots": root_reports,
    }


def _context(qs: list, rows: list, k: int) -> dict[str, Any]:
    groups: dict[tuple, list] = defaultdict(list)
    byq: dict[str, list] = defaultdict(list)
    for row in rows:
        byq[row["question_id"]].append(row)
    for q in qs:
        if q["operation"] not in {"sum", "difference"}:
            continue
        key = (
            q["root_family_id"],
            q["image_hash"],
            q["readset_id"],
            tuple(q["ordered_item_ids"]),
            q["quantity_units"],
            rational(q["delta_decimal"]),
            len(q["true_values_decimal"]),
            q.get("processed_image_hash"),
        )
        groups[key].append(q)
    pairs = []
    for key, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        base = {
            "root_family_id": key[0],
            "image_hash": key[1],
            "readset_id": key[2],
            "ordered_item_ids": key[3],
            "quantity_units": key[4],
            "delta": key[5],
            "question_ids": [q["question_id"] for q in items],
        }
        sums = [q for q in items if q["operation"] == "sum"]
        differences = [q for q in items if q["operation"] == "difference"]
        if len(sums) != 1 or len(differences) != 1:
            pairs.append({**base, "status": "PAIR_UNAVAILABLE"})
            continue
        q1, q2 = sums[0], differences[0]
        if tuple(map(rational, q1["true_values_decimal"])) != tuple(
            map(rational, q2["true_values_decimal"])
        ):
            raise ValueError("MATCHED_PAIR_TRUTH_MISMATCH")
        a, b = byq[q1["question_id"]], byq[q2["question_id"]]
        xs, ys = (
            [r["parsed_readings"] for r in a if r["L_P"]],
            [r["parsed_readings"] for r in b if r["L_P"]],
        )
        collision = collision_audit(xs, ys)
        soft = [max(abs(x - y) for x, y in zip(u, v, strict=True)) / key[5] for u in xs for v in ys]
        pairs.append(
            {
                **base,
                "status": "OBSERVED" if a or b else "NO_OUTPUTS",
                **collision,
                "C11_denominator": len(xs) * (len(xs) - 1),
                "C22_denominator": len(ys) * (len(ys) - 1),
                "C12_denominator": len(xs) * len(ys),
                "sum_coverage": _coverage([q1], a, k),
                "difference_coverage": _coverage([q2], b, k),
                "sum_P_full": _rate(sum(r["P_full"] for r in a), len(a)),
                "difference_P_full": _rate(sum(r["P_full"] for r in b), len(b)),
                "sum_mean_max_abs_error": _question_metrics(a)["max_abs_error"],
                "difference_mean_max_abs_error": _question_metrics(b)["max_abs_error"],
                "cross_mean_max_abs_difference_in_delta": _mean(soft),
            }
        )
    roots = []
    for root in sorted({q["root_family_id"] for q in qs}):
        selected = [p for p in pairs if p["root_family_id"] == root]
        roots.append(
            {
                "root_family_id": root,
                **{
                    key: {
                        "mean": _mean([p[key] for p in selected if p.get(key) is not None]),
                        "defined_pair_count": sum(p.get(key) is not None for p in selected),
                        "expected_pair_count": 8,
                    }
                    for key in ("C11", "C22", "C12", "collision_contrast")
                },
            }
        )
    return {
        "status": "PAIR_UNAVAILABLE" if not pairs else "NO_OUTPUTS" if not rows else "OBSERVED",
        "scope": "CONTEXT_AUDIT_ONLY; not internal causality or primary BGeo-S",
        "cross_pairs_are_not_independent_roots": True,
        "pairs": pairs,
        "roots": roots,
    }


def _reward_support(qs: list, rows: list, k: int, group: int = 8) -> dict[str, Any]:
    byq: dict[str, list] = defaultdict(list)
    for row in rows:
        byq[row["question_id"]].append(row)
    records = []
    for q in qs:
        selected = byq[q["question_id"]]
        n, successes = len(selected), sum(r["A_full"] for r in selected)
        records.append(
            {
                **{
                    key: q[key]
                    for key in ("question_id", "root_family_id", "chart_type", "operation")
                },
                "K_expected": k,
                "K_observed": n,
                "answer_successes": successes,
                "answer_valid": sum(r["L_A"] for r in selected),
                "observed_rate": Fraction(successes, n) if n else None,
                "actual_nonconstant_reward": 0 < successes < n if n >= 2 else None,
                "plugin_mixed_probability": mixed_probability(Fraction(successes, n), group)
                if n
                else None,
                "posterior_alpha": successes + Fraction(1, 2) if n else None,
                "posterior_beta": n - successes + Fraction(1, 2) if n else None,
                "posterior_predictive_mixed_probability": beta_mixed_predictive(successes, n, group)
                if n
                else None,
                "status": "NO_OUTPUTS"
                if not n
                else "LOW_OBSERVED_REWARD_CONTRAST"
                if successes in (0, n)
                else "OBSERVED_REWARD_CONTRAST",
            }
        )
    layers = []
    for chart, op in product(CHARTS, OPERATIONS):
        selected = [r for r in records if r["chart_type"] == chart and r["operation"] == op]
        observed = [r for r in selected if r["K_observed"]]
        layers.append(
            {
                "chart_type": chart,
                "operation": op,
                "question_count": len(selected),
                "observed_question_count": len(observed),
                "actual_nonconstant_question_count": sum(
                    bool(r["actual_nonconstant_reward"]) for r in observed
                ),
                "mean_per_question_plugin_mixed_probability": _mean(
                    [r["plugin_mixed_probability"] for r in observed]
                ),
                "mean_per_question_posterior_predictive": _mean(
                    [r["posterior_predictive_mixed_probability"] for r in observed]
                ),
            }
        )
    return {
        "assumption": "per-question iid Bernoulli; missing answers have zero reward",
        "prior": "Beta(1/2,1/2)",
        "group_G": group,
        "interpretation": (
            "Prior-dependent support diagnostic, not training gain or frequentist confidence."
        ),
        "questions": records,
        "chart_operation": layers,
    }


def _field_metrics(rows: list) -> dict[str, Any]:
    """Report generated-prefix and gold-prefix quantities with separate support.

    Values originated as model floating-point measurements; Fraction preserves
    their recorded decimal spelling for deterministic aggregation, not increased
    measurement precision. No output-frequency proxy is used for entropy.
    """
    reports = {}
    for source, expected_provenance in FIELD_METRIC_SOURCES.items():
        packets = [r.get(source) for r in rows]
        states: Counter = Counter()
        distributions: Counter = Counter()
        boundary_rules: Counter = Counter()
        measured = []
        for packet in packets:
            if packet is None:
                states["MISSING"] += 1
                continue
            if not isinstance(packet, Mapping):
                raise ValueError(f"INVALID_FIELD_METRIC_PACKET:{source}")
            status = packet.get("status")
            if not isinstance(status, str) or not status:
                raise ValueError(f"MISSING_FIELD_METRIC_STATUS:{source}")
            states[status] += 1
            if packet.get("provenance") != expected_provenance:
                raise ValueError(f"FIELD_METRIC_PREFIX_PROVENANCE_MISMATCH:{source}")
            if status != "MEASURED":
                if packet.get("field_nll") is not None:
                    raise ValueError(f"UNVERIFIED_FIELD_BOUNDARIES_WITH_METRICS:{source}")
                continue
            if packet.get("distribution") != "raw_model_before_temperature_or_top_p":
                raise ValueError(f"UNREGISTERED_FIELD_METRIC_DISTRIBUTION:{source}")
            fields = packet.get("field_nll")
            if not isinstance(fields, Mapping) or set(fields) != {"readings", "answer"}:
                raise ValueError(f"INVALID_FIELD_METRIC_FIELDS:{source}")
            if not isinstance(packet.get("boundary_rule"), str) or not packet["boundary_rule"]:
                raise ValueError(f"MISSING_FIELD_BOUNDARY_RULE:{source}")
            distributions[packet["distribution"]] += 1
            boundary_rules[packet["boundary_rule"]] += 1
            measured.append(packet)
        fields_report = {}
        for field in ("readings", "answer"):
            observed = {"mean_nll": [], "mean_token_entropy": []}
            observed_tokens = []
            for packet in measured:
                values = packet["field_nll"][field]
                if not isinstance(values, Mapping):
                    raise ValueError(f"INVALID_FIELD_METRIC_FIELDS:{source}:{field}")
                count = values.get("token_count")
                indices = values.get("token_indices")
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                    or not isinstance(indices, list)
                    or len(indices) != count
                    or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
                    or len(set(indices)) != len(indices)
                ):
                    raise ValueError(f"INVALID_FIELD_TOKEN_DENOMINATOR:{source}:{field}")
                observed_tokens.append(count)
                for key in observed:
                    value = values.get(key)
                    if value is None:
                        continue
                    if count == 0:
                        raise ValueError(f"FIELD_METRIC_WITH_ZERO_TOKENS:{source}:{field}")
                    number = rational(value)
                    if number < 0:
                        raise ValueError(f"NEGATIVE_FIELD_METRIC:{source}:{field}:{key}")
                    observed[key].append((number, count))
            summaries = {}
            for key, pairs in observed.items():
                tokens = sum(count for _, count in pairs)
                summaries[key] = {
                    "sequence_mean": _mean([value for value, _ in pairs]),
                    "token_weighted_mean": sum(
                        (value * count for value, count in pairs), Fraction(0)
                    )
                    / tokens
                    if tokens
                    else None,
                    "sequence_denominator": len(pairs),
                    "field_token_denominator": tokens,
                    "missing_response_count": len(rows) - len(pairs),
                }
            fields_report[field] = {
                "recorded_field_token_count": sum(observed_tokens),
                "positive_token_response_count": sum(count > 0 for count in observed_tokens),
                "zero_token_response_count": sum(count == 0 for count in observed_tokens),
                **summaries,
            }
        reports[source] = {
            "status": "MEASURED" if measured else "NOT_MEASURED",
            "expected_prefix_provenance": expected_provenance,
            "response_denominator": len(rows),
            "measured_packet_count": len(measured),
            "source_status_counts": dict(states),
            "distribution_counts": dict(distributions),
            "boundary_rule_counts": dict(boundary_rules),
            "fields": fields_report,
        }
    return reports


def _bootstrap(root_report: dict, context: dict, replicates: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    roots = root_report["roots"]
    samples: dict[str, list[Fraction]] = {key: [] for key in root_report["aggregate"]}
    samples.update({f"context_{key}": [] for key in ("C11", "C22", "C12", "collision_contrast")})
    context_map = {r["root_family_id"]: r for r in context["roots"]}
    for _ in range(replicates):
        draw = [roots[rng.randrange(len(roots))] for _ in roots]
        for key in samples:
            if key.startswith("context_"):
                metric = key[len("context_") :]
                values = [context_map[r["root_family_id"]][metric]["mean"] for r in draw]
            else:
                values = [r["metrics"][key]["conditional_mean"] for r in draw]
            valid = [v for v in values if v is not None]
            if valid:
                samples[key].append(_mean(valid))

    def quantile(values: list[Fraction], probability: Fraction) -> Fraction | None:
        if not values:
            return None
        values = sorted(values)
        position = (len(values) - 1) * probability
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "seed": seed,
        "requested_replicates": replicates,
        "unit": "root_family_id",
        "interval_scope": (
            "Descriptive conditional on the measured model and panel; "
            "not training-population uncertainty."
        ),
        "metrics": {
            key: {
                "defined_replicates": len(values),
                "undefined_replicates": replicates - len(values),
                "percentile_95_low": quantile(values, Fraction(1, 40)),
                "percentile_95_high": quantile(values, Fraction(39, 40)),
            }
            for key, values in samples.items()
        },
    }


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("INVALID_EXPECTED_K")


def score_records(
    questions: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    expected_k: int = 4,
    stage: str | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261008,
) -> dict[str, Any]:
    """Score immutable records and create tables without inventing unobserved rows."""
    _validate_k(expected_k)
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or not 0 <= bootstrap_replicates <= 10000
    ):
        raise ValueError("INVALID_BOOTSTRAP_REPLICATES")
    qs, rows, duplicate_records = _prepare(questions, outputs, stage)
    tables: dict[str, Any] = {"FORMAT_REPORT": _format_report(qs, rows, expected_k, stage)}
    names = (
        "COUNTS",
        "COVERAGE",
        "GEOMETRY",
        "ROOT_EQUAL",
        "CONTEXT_AUDIT",
        "REWARD_SUPPORT",
        "ROOT_BOOTSTRAP",
        "FIELD_METRICS",
    )
    tables.update({name: {"cohorts": []} for name in names})
    for s, model, panel, selected in _cohorts(qs, rows, stage):
        identity = {"stage": s, "model_hash": model, "split": STAGE_SPLITS[s]}
        cover = _coverage(panel, selected, expected_k)
        counts = {
            "overall": _counts(selected),
            "chart_operation": [
                {**label, **_counts(subr)} for label, _, subr in _groups(panel, selected)
            ],
            "factor_cells": [
                {**label, **_counts(subr)}
                for label, _, subr in _groups(panel, selected, factors=True)
            ],
        }
        geometry = {
            "overall": _geometry(selected),
            "chart_operation": [
                {**label, **_geometry(subr)} for label, _, subr in _groups(panel, selected)
            ],
        }
        root_equal = _root_equal(panel, selected)
        context = _context(panel, selected, expected_k)
        values = {
            "COUNTS": counts,
            "COVERAGE": cover,
            "GEOMETRY": geometry,
            "ROOT_EQUAL": root_equal,
            "CONTEXT_AUDIT": context,
            "REWARD_SUPPORT": _reward_support(panel, selected, expected_k),
            "FIELD_METRICS": {
                "interpretation": (
                    "Self: generated tokens under actual self prefix. "
                    "Gold: teacher-forced answer prefix contains gold readings; "
                    "not free generation performance."
                ),
                "numerical_origin": (
                    "Recorded floating-point model metrics; exact decimal aggregation "
                    "does not imply extra measurement precision."
                ),
                "overall": _field_metrics(selected),
                "chart_operation": [
                    {**label, **_field_metrics(subr)} for label, _, subr in _groups(panel, selected)
                ],
                "factor_cells": [
                    {**label, **_field_metrics(subr)}
                    for label, _, subr in _groups(panel, selected, factors=True)
                ],
            },
        }
        for name, report in values.items():
            tables[name]["cohorts"].append({**identity, **report})
        if STAGE_SPLITS[s] == "AUDIT_MEASURE":
            tables["ROOT_BOOTSTRAP"]["cohorts"].append(
                {
                    **identity,
                    **_bootstrap(root_equal, context, bootstrap_replicates, bootstrap_seed),
                }
            )
    tables["SCORING_RECEIPT"] = {
        "scorer_version": SCORER_VERSION,
        "status": "SCORED" if rows else "NO_OUTPUTS",
        "scored_record_count": len(rows),
        "identical_request_records_deduplicated": duplicate_records,
        "exact_joint_identity_checked_count": sum(r["L"] for r in rows),
        "all_six_and_eight_cells_explicit": True,
        "no_missing_completions_synthesized": True,
        "root_bootstrap_seed": bootstrap_seed,
        "raw_outputs_retain_failed_attempts": True,
        "numerical_serialization": "numerator/denominator exact; float display only",
        "scientific_stage_authorized_by_scoring": False,
    }
    return json_exact({"scored_outputs": rows, "tables": tables})
