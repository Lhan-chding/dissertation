"""CPU release audit with independent Fraction arithmetic and paired root recount.

Only the frozen response parser is shared with the primary scorer. This is not
an independent parser, tensor-equivalence test, or scientific-success decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from mm_core.contracts import parse_response

from . import common
from .contract import CELLS, PLAN_ID, STRATA, digest, run_matrix
from .evaluation import load_panel

STATES = ("S0", "S_A_0", "S_AP_0", "S_A_1", "S_AP_1")
ACTIONS = ("no_train", "a0", "aP", "aC")
METRICS = ("A_full", "P_full", "J_full")
EXACT_CELLS = ("111", "100", "011", "010", "001", "000")
BOOTSTRAP_SEED = 4311636612181726188
REPLICATES = 5000
FLOAT_ATOL = 1e-12
SUMMARY_NAMES = (
    "FINAL_MM_DEV_REPORT_zh.md",
    "RUN_COMPLETENESS.json",
    "PRIMARY_EFFECTS.json",
    "RESPONSE_BY_STATE_ACTION_FUTURE_STRATUM.jsonl",
    "ABSOLUTE_AND_RELATIVE_CHANGES.jsonl",
    "RESPONSE_BY_ROOT.jsonl",
    "SECONDARY_EVENT_AND_NUMERIC_CHANGES.jsonl",
    "PREPARATION_STATE_DIFFERENCES.json",
    "REWARD_AND_ADVANTAGE_SUPPORT.json",
    "JOINT_EVENT_TABLES.json",
    "NUMERIC_AND_GEOMETRY_SUPPORT.json",
    "CONTEXT_AUDIT.json",
    "UTILITY_AND_FEASIBILITY.json",
    "EXPLORATORY_OPPORTUNITY_CROSSFIT.json",
    "ACTUAL_COST_ACCOUNTING.json",
    "NEXT_STAGE_RECOMMENDATION_zh.md",
)
FEATURE_PATH = "features/PRETRAIN_FEATURES_B0_BJ_BTAB_BNUM_BGEO.jsonl"
DERIVED_NAMES = (
    "DATA_AND_IMAGE_HASHES",
    "RUN_MATRIX",
    "TRAINING_STREAM_HASHES",
    "CHECKPOINT_INDEX",
)


class RecountFailure(RuntimeError):
    """A complete release cannot be certified from the supplied evidence."""


def read_json(path):
    return json.loads(
        Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))
    )


def json_rows(path):
    with Path(path).open() as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"BLANK_JSONL_RECORD:{path}:{index}")
            yield json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def path_inside(root, name):
    name = Path(name)
    if name.is_absolute() or not name.parts or ".." in name.parts:
        raise ValueError("UNSAFE_RELEASE_PATH:" + str(name))
    root = Path(root).resolve()
    result = root / name
    if any((root / Path(*name.parts[:i])).is_symlink() for i in range(1, len(name.parts) + 1)):
        raise ValueError("SYMLINK_RELEASE_PATH:" + str(name))
    return result


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def descriptor(root, name):
    path = path_inside(root, name)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(name), "bytes": path.stat().st_size, "sha256": sha256(path)}


def encoded(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def publish(root, name, data):
    """Immutable, atomic, byte-idempotent output. Never replace prior evidence."""
    path = path_inside(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("DIFFERENT_RELEASE_ARTIFACT:" + name)
        return
    fd, temporary = tempfile.mkstemp(prefix=".pending-release-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise
    finally:
        os.unlink(temporary)


def _fraction(value):
    if isinstance(value, bool):
        raise ValueError("BOOLEAN_IS_NOT_A_NUMBER")
    return Fraction(str(value))


def _operation(values, operation):
    if operation == "sum":
        return sum(values, Fraction())
    if operation == "difference" and len(values) == 2:
        return values[0] - values[1]
    if operation == "range":
        return max(values) - min(values)
    raise ValueError("INVALID_REGISTERED_OPERATION")


def recount_response(question, raw):
    """Recalculate scores without importing scorer, operate, shape or score helpers."""
    truth = tuple(map(_fraction, question.get("true_values_decimal", question.get("true_values"))))
    delta = _fraction(question.get("delta_decimal", question.get("delta")))
    if (
        delta <= 0
        or not truth
        or raw.get("status") != "completed"
        or not isinstance(raw.get("raw_text"), str)
    ):
        raise ValueError("NON_COMPLETION_OR_INVALID_TRUTH")
    operation = question["operation"]
    expected_answer = _operation(truth, operation)
    parsed = parse_response(raw["raw_text"], len(truth))
    result = dict.fromkeys(
        (
            "P",
            "C",
            "A",
            "cell_exact",
            "cell_tolerance",
            "e",
            "near_miss",
            "S",
            "eps_P",
            "eps_C",
            "eps_O",
            "outside_axis",
            "residual_identity_error",
        )
    )
    result.update(
        L_P=parsed.L_P,
        L_A=parsed.L_A,
        L=parsed.L,
        delta=delta,
        parsed_readings=parsed.readings,
        parsed_answer=parsed.answer,
        schema_ok=parsed.schema_ok,
        field_order_ok=parsed.field_order_ok,
        parse_errors=list(parsed.errors),
        completion_text_available=True,
    )
    if parsed.L_A:
        result["A"] = parsed.answer == expected_answer
    if parsed.L_P:
        readings = parsed.readings
        result["P"] = readings == truth
        errors = [(a - b) / delta for a, b in zip(readings, truth, strict=True)]
        result["e"] = errors
        result["near_miss"] = 0 < max(map(abs, errors)) <= 1
        square_sum = sum((x * x for x in errors), Fraction())
        if square_sum:
            result["S"] = sum(errors, Fraction()) ** 2 / (len(errors) * square_sum)
        result["eps_P"] = (_operation(readings, operation) - expected_answer) / delta
        lo, hi = _fraction(question.get("axis_min", 0)), _fraction(question.get("axis_max", 100))
        if lo >= hi:
            raise ValueError("INVALID_AXIS")
        result["outside_axis"] = any(x < lo or x > hi for x in readings)
    if parsed.L:
        computed = _operation(parsed.readings, operation)
        result["C"] = parsed.answer == computed
        result["cell_exact"] = "".join(str(int(result[k])) for k in ("P", "C", "A"))
        if result["cell_exact"] not in EXACT_CELLS:
            raise AssertionError("IMPOSSIBLE_EXACT_CELL")
        result["eps_C"] = (parsed.answer - computed) / delta
        result["eps_O"] = (parsed.answer - expected_answer) / delta
        result["residual_identity_error"] = result["eps_O"] - result["eps_P"] - result["eps_C"]
        result["cell_tolerance"] = "".join(
            str(int(x))
            for x in (
                all(abs(a - b) <= delta / 2 for a, b in zip(parsed.readings, truth, strict=True)),
                abs(parsed.answer - computed) <= delta / 2,
                abs(parsed.answer - expected_answer) <= delta / 2,
            )
        )
    result.update(
        P_full=bool(parsed.L_P and result["P"]),
        A_full=bool(parsed.L_A and result["A"]),
        J_full=bool(parsed.L and result["P"] and result["A"]),
        format_ok=bool(
            parsed.schema_ok and parsed.field_order_ok and parsed.L and not raw["truncated"]
        ),
    )
    return result


class Audit:
    """Collect every observed mismatch; never silently impute missing values."""

    def __init__(self):
        self.mismatches = []
        self.comparisons = 0

    def mismatch(self, location, reason):
        self.mismatches.append({"location": location, "reason": reason})

    def check(self, actual, expected, location):
        self.comparisons += 1
        if isinstance(expected, Fraction):
            if not isinstance(actual, dict) or set(actual) != {"numerator", "denominator", "float"}:
                self.mismatch(location, "MISSING_EXACT_RATIONAL")
                return
            n, d = actual["numerator"], actual["denominator"]
            if type(n) is not int or type(d) is not int or d <= 0 or Fraction(n, d) != expected:
                self.mismatch(location, "EXACT_RATIONAL_DIFFERS")
            try:
                approximate = float(expected)
                if not math.isfinite(approximate):
                    approximate = None
            except OverflowError:
                approximate = None
            self.check(actual["float"], approximate, location + ".float")
        elif isinstance(expected, dict):
            if not isinstance(actual, dict):
                self.mismatch(location, "MISSING_MAPPING")
                return
            for key, value in expected.items():
                if key not in actual:
                    self.mismatch(location + "." + key, "MISSING_FIELD")
                else:
                    self.check(actual[key], value, location + "." + key)
        elif isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                self.mismatch(location, "SEQUENCE_LENGTH_DIFFERS")
                return
            for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                self.check(a, b, location + f"[{i}]")
        elif isinstance(expected, float):
            if (
                type(actual) not in (int, float)
                or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=0, abs_tol=FLOAT_ATOL)
            ):
                self.mismatch(location, f"FLOAT_DIFFERS_EXPECTED:{expected!r}")
        elif type(actual) is not type(expected) or actual != expected:
            self.mismatch(location, f"VALUE_DIFFERS_EXPECTED:{expected!r}")

    def require(self, condition, location, reason):
        self.comparisons += 1
        if not condition:
            self.mismatch(location, reason)

    def finish(self):
        if self.mismatches:
            raise RecountFailure(f"INDEPENDENT_RECOUNT_FAILED:{len(self.mismatches)}")


def event_counts(rows):
    legal = sum(r["L"] for r in rows)
    exact = {c: sum(r["cell_exact"] == c for r in rows) for c in EXACT_CELLS}
    tolerance = {f"{i:03b}": sum(r["cell_tolerance"] == f"{i:03b}" for r in rows) for i in range(8)}

    def rate(numerator, denominator):
        return dict(
            numerator=numerator,
            denominator=denominator,
            rate=Fraction(numerator, denominator) if denominator else None,
        )

    marginal = {
        "p": exact["111"] + exact["100"],
        "a": exact["111"] + exact["011"] + exact["001"],
        "c": exact["111"] + exact["011"] + exact["010"],
        "j": exact["111"],
        "z": exact["001"],
    }
    return dict(
        recorded_attempts=len(rows),
        joint_scorable_denominator=legal,
        exact_counts=exact,
        tolerance_counts=tolerance,
        conditional_on_L={k: rate(v, legal) for k, v in marginal.items()},
        **{m: rate(sum(r[m] for r in rows), len(rows)) for m in METRICS},
        P_given_LP=rate(sum(r["P_full"] for r in rows), sum(r["L_P"] for r in rows)),
        A_given_LA=rate(sum(r["A_full"] for r in rows), sum(r["L_A"] for r in rows)),
        six_cell_reconstruction_checked=bool(legal),
        missing_joint_fields_not_000=len(rows) - legal,
    )


def _unique(rows, fields, audit, label):
    result = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in result:
            audit.mismatch(label + repr(key), "DUPLICATE_KEY")
        result[key] = row
    return result


def recount_panel(payload, scored_path, panel, state, roots, audit):
    """Return a cube derived from raw text, never scored summary indicators."""
    if payload.get("complete") is not True:
        raise RecountFailure("INCOMPLETE_RAW_PANEL:" + panel + "/" + state)
    questions = _unique(payload["questions"], ("question_id",), audit, "questions")
    outputs = _unique(payload["outputs"], ("question_id", "sample_index"), audit, "raw")
    scored = _unique(
        json_rows(scored_path), ("question_id", "sample_index"), audit, str(scored_path)
    )
    expected_slots = {(q[0], k) for q in questions for k in range(4)}
    audit.require(
        set(outputs) == set(scored) == expected_slots,
        panel + "/" + state,
        "INCOMPLETE_OR_EXTRA_SLOTS",
    )
    audit.require(len(questions) == roots * 24, panel + "/" + state, "QUESTION_COUNT_DIFFERS")
    sums = np.zeros((roots, 6, 4, 3), dtype=np.int64)
    slots = Counter()
    families = {}
    rows = []
    for key, raw in outputs.items():
        label = f"{panel}/{state}/{key}"
        if (key[0],) not in questions or key not in scored:
            continue
        q, saved = questions[(key[0],)], scored[key]
        values = recount_response(q, raw)
        audit.check(saved, values, label)
        for field, value in raw.items():
            if field not in {"split", "stage"}:
                audit.require(
                    field in saved and saved[field] == value,
                    label + ".raw_binding." + field,
                    "RAW_VALUE_DIFFERS",
                )
        meta = dict(
            state_id=state,
            stage=panel,
            split=panel,
            f2_panel=panel,
            root_index=q["root_index"],
            root_family_id=q["root_family_id"],
            chart_type=q["chart_type"],
            operation=q["operation"],
            visual_level=q.get("visual_level", q.get("V")),
            numeric_level=q.get("numeric_level", q.get("D")),
            image_hash=q.get("image_hash", q.get("image_sha256")),
            readset_id=q["readset_id"],
            ordered_item_ids=q["ordered_item_ids"],
            quantity_units=q["quantity_units"],
        )
        audit.check(saved, meta, label + ".metadata")
        root = q["root_index"]
        cell = meta["visual_level"][0].upper() + meta["numeric_level"][0].upper()
        if (
            type(root) is not int
            or root not in range(roots)
            or cell not in CELLS
            or type(key[1]) is not int
            or key[1] not in range(4)
        ):
            audit.mismatch(label, "INVALID_FIXED_PANEL_CELL")
            continue
        family = q["root_family_id"]
        audit.require(
            root not in families or families[root] == family, label, "ROOT_FAMILY_CONFLICT"
        )
        families[root] = family
        coordinate = (root, STRATA.index((q["chart_type"], q["operation"])), CELLS.index(cell))
        slots[coordinate] += 1
        sums[coordinate] += [int(values[m]) for m in METRICS]
        rows.append({**meta, **values, "question_id": key[0], "sample_index": key[1]})
    audit.require(
        len(slots) == roots * 24 and set(slots.values()) == {4},
        panel + "/" + state,
        "MISSING_FIXED_ROOT_CELLS",
    )
    audit.require(
        len(set(families.values())) == roots, panel + "/" + state, "ROOT_FAMILY_COUNT_DIFFERS"
    )
    return sums.astype(float) / 4, rows, families


def check_joint_table(table, rows, audit, label):
    cohorts = table["counts"]["cohorts"]
    audit.require(len(cohorts) == 1, label, "ONE_MODEL_COHORT_REQUIRED")
    if len(cohorts) != 1:
        return
    cohort = cohorts[0]
    audit.check(cohort.get("overall"), event_counts(rows), label + ".overall")
    coverage = table["coverage"]["cohorts"]
    audit.require(len(coverage) == 1, label + ".coverage", "ONE_MODEL_COHORT_REQUIRED")
    if len(coverage) == 1:
        n = len(rows)
        audit.check(
            coverage[0],
            dict(
                question_count=len({r["question_id"] for r in rows}),
                root_family_count=len({r["root_family_id"] for r in rows}),
                expected_completions=n,
                recorded_attempts=n,
                unique_sample_slots=n,
                missing_sample_slots=0,
                unexpected_sample_slots=0,
                duplicate_sample_slots=0,
                complete=True,
                status_counts={"completed": n},
                **{
                    key: dict(
                        numerator=sum(r[key] for r in rows),
                        denominator=n,
                        rate=Fraction(sum(r[key] for r in rows), n) if n else None,
                    )
                    for key in ("L_P", "L_A", "L")
                },
            ),
            label + ".coverage",
        )
    for field, keys, expected in (
        ("chart_operation", ("chart_type", "operation"), set(STRATA)),
        (
            "factor_cells",
            ("chart_type", "operation", "visual_level", "numeric_level"),
            {(c, o, v, d) for c, o in STRATA for v in ("low", "high") for d in ("low", "high")},
        ),
    ):
        indexed = _unique(cohort[field], keys, audit, label + "." + field)
        audit.require(set(indexed) == expected, label + "." + field, "STRATUM_SET_DIFFERS")
        for key, group in indexed.items():
            selected = [r for r in rows if tuple(r[k] for k in keys) == key]
            audit.check(group, event_counts(selected), label + "." + field + repr(key))


def check_root_equal(table, rows, audit, label):
    """Independent conditional root algebra; missing cells keep explicit support."""
    cohorts = table["root_equal"]["cohorts"]
    audit.require(len(cohorts) == 1, label, "ONE_ROOT_EQUAL_COHORT_REQUIRED")
    if len(cohorts) != 1:
        return
    by_root = {}
    for row in rows:
        cells = by_root.setdefault(row["root_family_id"], {})
        key = (row["chart_type"], row["operation"], row["visual_level"], row["numeric_level"])
        cells.setdefault(key, []).append(row)

    def mean(values):
        return sum(values, Fraction()) / len(values) if values else None

    root_metrics = {}
    for family, cells in by_root.items():
        observations = []
        for responses in cells.values():
            legal = [r for r in responses if r["L"]]
            cell = {
                key: mean([Fraction(r[key]) for r in responses])
                for key in (*METRICS, "L_P", "L_A", "L")
            }
            cell.update(
                {
                    name: mean([Fraction(r[field]) for r in legal])
                    for name, field in (("p", "P"), ("a", "A"), ("c", "C"), ("j", "J_full"))
                }
            )
            cell.update(
                z=mean([Fraction(r["cell_exact"] == "001") for r in legal]),
                S=mean([r["S"] for r in responses if r["S"] is not None]),
                near_miss=mean([Fraction(r["near_miss"]) for r in responses if r["L_P"]]),
                max_abs_error=mean([max(map(abs, r["e"])) for r in responses if r["L_P"]]),
            )
            observations.append(cell)
        root_metrics[family] = {}
        for key in observations[0]:
            values = [cell[key] for cell in observations if cell[key] is not None]
            root_metrics[family][key] = dict(
                conditional_mean=mean(values),
                fixed_weight_numerator=sum(values, Fraction()) / 24,
                supported_weight=Fraction(len(values), 24),
                defined_cells=len(values),
                expected_cells=24,
                complete_fixed_weight_mean=mean(values) if len(values) == 24 else None,
            )
    root_rows = _unique(cohorts[0]["roots"], ("root_family_id",), audit, label)
    audit.require(
        set(root_rows) == {(r,) for r in root_metrics}, label, "ROOT_EQUAL_FAMILIES_DIFFER"
    )
    for family, metrics in root_metrics.items():
        audit.check(root_rows.get((family,)), dict(metrics=metrics), label + "." + family)
    aggregate = {}
    for metric in next(iter(root_metrics.values())):
        values = [
            r[metric]["conditional_mean"]
            for r in root_metrics.values()
            if r[metric]["conditional_mean"] is not None
        ]
        complete = [r[metric]["complete_fixed_weight_mean"] for r in root_metrics.values()]
        aggregate[metric] = dict(
            root_equal_conditional_mean=mean(values),
            supported_root_count=len(values),
            root_count=len(root_metrics),
            complete_fixed_weight_mean=mean(complete)
            if all(x is not None for x in complete)
            else None,
        )
    audit.check(cohorts[0]["aggregate"], aggregate, label + ".aggregate")


def _ci(values, level=0.95):
    # Direct indexing resamples rows. Primary analysis instead multiplies a
    # bincount weight matrix. Shared registered PCG64 draws retain all pairing.
    tail = (1 - level) / 2
    return dict(
        level=level,
        lower=float(np.quantile(values, tail, method="linear")),
        upper=float(np.quantile(values, 1 - tail, method="linear")),
        defined_replicates=len(values),
        undefined_replicates=0,
        quantile_method="linear",
    )


def check_responses(root, cubes, root_families, plan, audit):
    """Independently reconstruct every reported response and registered CI."""
    roots = plan["data"]["root_counts"]["DEV_EVAL"]
    audit.require(
        plan["statistics"]["root_bootstrap_replicates"] == REPLICATES
        and plan["statistics"]["root_bootstrap_seed"] == BOOTSTRAP_SEED,
        "bootstrap",
        "UNREGISTERED_BOOTSTRAP",
    )
    draw = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, roots, (REPLICATES, roots)
    )
    fields = ("start_state", "action", "future_repeat", "chart_type", "operation")
    responses = _unique(
        json_rows(root / "summary/RESPONSE_BY_STATE_ACTION_FUTURE_STRATUM.jsonl"),
        fields,
        audit,
        "responses",
    )
    cells = _unique(
        json_rows(root / "summary/ABSOLUTE_AND_RELATIVE_CHANGES.jsonl"),
        (*fields, "VD_cell"),
        audit,
        "cells",
    )
    root_rows = _unique(
        json_rows(root / "summary/RESPONSE_BY_ROOT.jsonl"),
        (*fields, "evaluation_root_index"),
        audit,
        "root_rows",
    )
    expected_keys = {
        (s, a, f, c, o) for s in STATES for a in ACTIONS for f in (0, 1) for c, o in STRATA
    }
    audit.require(set(responses) == expected_keys, "responses", "RESPONSE_SET_DIFFERS")
    audit.require(
        set(cells) == {(*k, cell) for k in expected_keys for cell in CELLS},
        "cells",
        "CELL_RESPONSE_SET_DIFFERS",
    )
    audit.require(
        set(root_rows) == {(*k, r) for k in expected_keys for r in range(roots)},
        "root_rows",
        "ROOT_RESPONSE_SET_DIFFERS",
    )
    absolute = {}
    relative = {}
    for state in STATES:
        for future in (0, 1):
            baseline = cubes[f"cont_{state}_a0_f{future}"] - cubes[state]
            for action in ACTIONS:
                key = (state, action, future)
                d = (
                    np.zeros_like(baseline)
                    if action == "no_train"
                    else cubes[f"cont_{state}_{action}_f{future}"] - cubes[state]
                )
                delta = d - baseline
                absolute[key], relative[key] = d, delta
                for i, (chart, operation) in enumerate(STRATA):
                    row_key = (*key, chart, operation)
                    label = "responses" + repr(row_key)
                    per_root = d[:, i].mean(axis=1)
                    sampled = per_root[draw].mean(axis=1)
                    audit.check(
                        responses.get(row_key),
                        dict(
                            absolute_D=dict(
                                zip(METRICS, per_root.mean(axis=0).tolist(), strict=True)
                            ),
                            relative_Delta_vs_a0=dict(
                                zip(METRICS, delta[:, i].mean(axis=(0, 1)).tolist(), strict=True)
                            ),
                            root_count=roots,
                            absolute_root_CI95={
                                m: _ci(sampled[:, j]) for j, m in enumerate(METRICS)
                            },
                        ),
                        label,
                    )
                    for c, cell in enumerate(CELLS):
                        audit.check(
                            cells.get((*row_key, cell)),
                            dict(
                                absolute_D=dict(
                                    zip(METRICS, d[:, i, c].mean(axis=0).tolist(), strict=True)
                                ),
                                relative_Delta_vs_a0=dict(
                                    zip(METRICS, delta[:, i, c].mean(axis=0).tolist(), strict=True)
                                ),
                                root_count=roots,
                            ),
                            "cells" + repr((*row_key, cell)),
                        )
                    for r in range(roots):
                        expected = dict(
                            absolute_D_by_VD_cell=d[r, i].tolist(),
                            relative_Delta_by_VD_cell=delta[r, i].tolist(),
                            VD_cell_order=list(CELLS),
                            metric_order=list(METRICS),
                            evaluation_root_family_id=root_families[r],
                        )
                        audit.check(
                            root_rows.get((*row_key, r)),
                            expected,
                            "root_rows" + repr((*row_key, r)),
                        )
    primary = read_json(root / "summary/PRIMARY_EFFECTS.json")
    audit.check(
        primary,
        dict(
            replicates=REPLICATES,
            seed=BOOTSTRAP_SEED,
            rng="numpy.Generator(PCG64)",
            resample_unit="whole_root_all_24_cells_K_models_actions_futures",
            practical_attention_pp=1,
            scientific_pass_threshold=None,
        ),
        "primary",
    )
    contrasts = _unique(primary["contrasts"], ("contrast",), audit, "primary")
    expected_contrasts = {(f"macro_{m[0]}_{a}_minus_a0",) for m in METRICS[:2] for a in ACTIONS[2:]}
    audit.require(set(contrasts) == expected_contrasts, "primary", "PRIMARY_CONTRAST_SET_DIFFERS")
    for j, metric in enumerate(METRICS[:2]):
        for action in ACTIONS[2:]:
            by_future = np.stack(
                [
                    np.stack(
                        [relative[s, action, f][..., j].mean(axis=(1, 2)) for s in STATES]
                    ).mean(axis=0)
                    for f in (0, 1)
                ]
            )
            root_effect = by_future.mean(axis=0)
            point = float(root_effect.mean())
            sampled = root_effect[draw].mean(axis=1)
            future_effects = by_future.mean(axis=1)
            name = f"macro_{metric[0]}_{action}_minus_a0"
            audit.check(
                contrasts.get((name,)),
                dict(
                    effect_probability=point,
                    effect_pp=100 * point,
                    CI95=_ci(sampled),
                    CI98_75=_ci(sampled, 0.9875),
                    future_effects=future_effects.tolist(),
                    future_1_minus_0=float(future_effects[1] - future_effects[0]),
                    future_training_SE=None,
                ),
                name,
            )
    check_utility_and_preparation(root, cubes, absolute, relative, draw, audit)
    return {
        "response_rows": len(responses),
        "cell_rows": len(cells),
        "root_rows": len(root_rows),
        "primary_contrasts": len(contrasts),
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _utility(d):
    gain = float(d[:, 0].sum() / 6)
    loss = float(sum(max(-x, 0) for x in d[:, 1]) / 6)
    return dict(G=gain, I=loss, U=gain - loss, feasible=bool(all(x >= -0.02 for x in d[:, 1])))


def check_utility_and_preparation(root, cubes, absolute, relative, draws, audit):
    utility = read_json(root / "summary/UTILITY_AND_FEASIBILITY.json")
    audit.check(
        utility, {"lambda": 1, "epsilon": 0.02, "eta": 0, "safety_guarantee": False}, "utility"
    )
    rows = _unique(utility["rows"], ("start_state", "action"), audit, "utility")
    audit.require(
        set(rows) == {(s, a) for s in STATES for a in ACTIONS}, "utility", "UTILITY_SET_DIFFERS"
    )
    for state in STATES:
        for action in ACTIONS:
            futures = [absolute[state, action, f] for f in (0, 1)]
            by_root = (futures[0].mean(axis=2) + futures[1].mean(axis=2)) / 2
            d = by_root.mean(axis=0)
            # Loop strata to bound peak memory at 5000 x 128 x 3, not all arms.
            boot = np.stack([by_root[:, s][draws].mean(axis=1) for s in range(6)], axis=1)
            boot_u = boot[..., 0].mean(axis=1) - np.maximum(-boot[..., 1], 0).mean(axis=1)
            protection = [_ci(boot[:, s, 1]) for s in range(6)]
            uncertainty = [
                "above_boundary"
                if x["lower"] >= -0.02
                else "below_boundary"
                if x["upper"] < -0.02
                else "uncertain_crosses_boundary"
                for x in protection
            ]
            audit.check(
                rows.get((state, action)),
                dict(
                    **_utility(d),
                    D_A=d[:, 0].tolist(),
                    D_P=d[:, 1].tolist(),
                    per_future=[_utility(x.mean(axis=(0, 2))) for x in futures],
                    U_CI95=_ci(boot_u),
                    D_P_CI95_by_stratum=protection,
                    protection_uncertainty=uncertainty,
                    all_strata_uncertainty_above_boundary=all(
                        x == "above_boundary" for x in uncertainty
                    ),
                    optimizer_updates=0 if action == "no_train" else 32,
                    shared_probe_cost_is_zero=False,
                    no_train_relative_Delta_vs_a0=(
                        (relative[state, action, 0] + relative[state, action, 1]) / 2
                    )
                    .mean(axis=(0, 2))
                    .tolist()
                    if action == "no_train"
                    else None,
                ),
                "utility" + repr((state, action)),
            )
    prep = read_json(root / "summary/PREPARATION_STATE_DIFFERENCES.json")
    pairs = [(s, "S0") for s in STATES[1:]] + [("S_AP_0", "S_A_0"), ("S_AP_1", "S_A_1")]
    indexed = _unique(prep["comparisons"], ("left", "right"), audit, "preparation")
    audit.require(set(indexed) == set(pairs), "preparation", "PREPARATION_SET_DIFFERS")
    audit.check(
        prep,
        dict(
            metric_order=list(METRICS),
            stratum_order=[list(x) for x in STRATA],
            independent_base_count=1,
        ),
        "preparation",
    )
    for left, right in pairs:
        difference = (cubes[left] - cubes[right]).mean(axis=2)
        macro = difference.mean(axis=1)
        sampled = macro[draws].mean(axis=1)
        audit.check(
            indexed.get((left, right)),
            dict(
                stratum_effects=difference.mean(axis=0).tolist(),
                macro_effects=macro.mean(axis=0).tolist(),
                macro_CI95={m: _ci(sampled[:, j]) for j, m in enumerate(METRICS)},
            ),
            "preparation" + repr((left, right)),
        )


def model_panels():
    endpoints = tuple(r["run_id"] for r in run_matrix() if r["phase"] == "CONTINUE")
    return [("PROBE", s) for s in STATES] + [("DEV_EVAL", s) for s in (*STATES, *endpoints)]


def required_analysis_files():
    return {
        *("summary/" + name for name in SUMMARY_NAMES),
        FEATURE_PATH,
        *(f"scoring/{panel}/{state}/SCORED_OUTPUTS.jsonl" for panel, state in model_panels()),
    }


def verify_analysis_manifest(root, audit):
    manifest = read_json(root / "summary/ANALYSIS_MANIFEST.json")
    audit.check(
        manifest,
        dict(status="ANALYZED", scientific_success_claimed=False, next_stage_authorized=False),
        "analysis_manifest",
    )
    indexed = _unique(manifest["files"], ("path",), audit, "analysis_manifest.files")
    audit.require(
        set(indexed) == {(p,) for p in required_analysis_files()},
        "analysis_manifest.files",
        "MISSING_OR_EXTRA_ANALYSIS_ARTIFACT",
    )
    for key, expected in indexed.items():
        audit.require(
            set(expected) == {"path", "bytes", "sha256"},
            "analysis_manifest.files" + repr(key),
            "INVALID_FILE_DESCRIPTOR",
        )
        audit.check(descriptor(root, key[0]), expected, "analysis_manifest.files" + repr(key))
    return manifest


def derived_manifests(plan_path, root, freeze):
    """Produce only identity projections of actual frozen sources and commits."""
    data = []
    for name, expected in sorted(freeze["input_hashes"].items()):
        item = descriptor(root, name)
        if item["sha256"] != expected:
            raise RecountFailure("FROZEN_INPUT_HASH_DIFFERS:" + name)
        data.append(item)
    freeze_hash = sha256(root / "manifests/F2_FREEZE.json")
    publish(
        root,
        "manifests/DATA_AND_IMAGE_HASHES.json",
        encoded(
            dict(
                plan_id=PLAN_ID,
                freeze_sha256=freeze_hash,
                files=data,
                scope=(
                    "all frozen inputs including original/processed image and token "
                    "routing manifests and QA"
                ),
            )
        ),
    )
    matrix_bytes = (plan_path.parent / "run_matrix.json").read_bytes()
    if json.loads(matrix_bytes) != run_matrix():
        raise RecountFailure("DISPATCHED_RUN_MATRIX_DIFFERS")
    publish(root, "manifests/RUN_MATRIX.json", matrix_bytes)
    streams = []
    for path in sorted((plan_path.parent / "schedules").glob("*.jsonl")):
        rows = list(json_rows(path))
        streams.append(
            dict(
                path="config/schedules/" + path.name,
                bytes=path.stat().st_size,
                sha256=sha256(path),
                parsed_stream_hash=digest(rows),
                records=len(rows),
            )
        )
    if len(streams) != 9:
        raise RecountFailure("NINE_REGISTERED_STREAMS_REQUIRED")
    publish(
        root,
        "manifests/TRAINING_STREAM_HASHES.json",
        encoded(
            dict(
                plan_id=PLAN_ID,
                freeze_sha256=freeze_hash,
                streams=streams,
                runs=[dict(run_id=r["run_id"], schedule_id=r["schedule_id"]) for r in run_matrix()],
            )
        ),
    )
    checkpoints = []
    for run in run_matrix():
        base = f"training/{run['run_id']}/checkpoints/"
        latest = read_json(root / base / "LATEST.json")
        if latest["step"] != 32:
            raise RecountFailure("UNFINISHED_CHECKPOINT_PATH:" + run["run_id"])
        for step in range(33):
            relative = base + f"commit-{step:02d}.json"
            receipt = read_json(root / relative)
            if receipt["step"] != step or (step == 32 and receipt != latest):
                raise RecountFailure("CHECKPOINT_COMMIT_STEP_DIFFERS:" + relative)
            name = base + receipt["path"]
            body = path_inside(root, name)
            exists = body.is_file()
            milestone = step in {0, 8, 16, 24, 32}
            rolling = step in {31, 32}
            if (milestone or rolling) and not exists:
                raise RecountFailure("MISSING_RETAINED_CHECKPOINT:" + name)
            item = descriptor(root, name) if exists else None
            if exists and item["sha256"] != receipt["sha256"]:
                raise RecountFailure("CHECKPOINT_BYTES_DIFFER:" + name)
            checkpoints.append(
                dict(
                    run_id=run["run_id"],
                    step=step,
                    path=name,
                    immutable_milestone=milestone,
                    latest_two_rolling=rolling,
                    exists=exists,
                    retention_status="PRESENT_HASH_VERIFIED"
                    if exists
                    else "PLANNED_ROTATION_COMMIT_RECEIPT_RETAINED",
                    commit_receipt=descriptor(root, relative),
                    committed_sha256=receipt["sha256"],
                    state_hash_from_commit_receipt=receipt["state_hash"],
                    body=item,
                )
            )
    publish(
        root,
        "manifests/CHECKPOINT_INDEX.json",
        encoded(
            dict(
                plan_id=PLAN_ID,
                freeze_sha256=freeze_hash,
                checkpoints=checkpoints,
                immutable_steps=[0, 8, 16, 24, 32],
                rolling_retention="latest two committed steps",
                body_verification=(
                    "byte size and SHA256 only; no independent tensor comparison or deserialization"
                ),
                complete_weights_in_lightweight_package=False,
                server_resident_bodies=True,
            )
        ),
    )
    return ["manifests/" + name + ".json" for name in DERIVED_NAMES]


def derive_training_aggregates(root):
    """Lossless projections of committed records; original failures stay in place."""
    paths = []
    for run in run_matrix():
        base = f"training/{run['run_id']}/"
        for kind in ("ROLLOUTS", "UPDATES"):
            records = []
            for step in range(1, 33):
                names = (
                    [base + f"steps/{step:02d}.json"]
                    if kind == "UPDATES"
                    else [
                        base + f"rollouts/{step:02d}-{slot:02d}-{index}.json"
                        for slot in range(24)
                        for index in range(8)
                    ]
                )
                for name in names:
                    raw = read_json(root / name)
                    if raw.get("logical_step") != step:
                        raise RecountFailure("AGGREGATE_SOURCE_STEP_DIFFERS:" + name)
                    records.append(dict(derived=True, source=descriptor(root, name), record=raw))
            name = base + kind + ".jsonl"
            data = "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                for record in records
            ).encode()
            publish(root, name, data)
            paths.append(name)
    return paths


def evidence_inventory(root):
    """Hash every retained scientific input/output, including failures and bodies.

    Operational leases, live scheduler projections, and this release's own
    future completion marker are explicitly outside the immutable inventory.
    """
    names = set()
    for directory in (
        "data",
        "qa",
        "raw",
        "training",
        "states",
        "summary",
        "scoring",
        "features",
        "accounting",
        "engineering",
        "recovery_checkpoints",
        "verification/failed",
    ):
        for path in (root / directory).rglob("*"):
            if path.is_file():
                names.add(str(path.relative_to(root)))
    for path in (root / "manifests").glob("*.json"):
        if path.name != "ALLOCATIONS.json":
            names.add(str(path.relative_to(root)))
    names.add("orchestration/REGISTRATION.json")
    for directory in (
        "attempts",
        "failures",
        "checkpoints",
        "observations",
        "submissions",
        "completions",
    ):
        for path in (root / "orchestration" / directory).rglob("*"):
            if path.is_file() and path.name != "RELEASE.json":
                names.add(str(path.relative_to(root)))
    return [descriptor(root, name) for name in sorted(names)]


def _release(plan_path, root, audit):
    before = common.verify_analysis_readiness(plan_path, root)
    plan = common.load_plan(plan_path)
    freeze = common.verify_execution(plan_path, root, require_engine=True, full_hashes=True)
    analysis_manifest = verify_analysis_manifest(root, audit)
    audit.check(read_json(root / "summary/RUN_COMPLETENESS.json"), before, "RUN_COMPLETENESS")
    joints = _unique(
        read_json(root / "summary/JOINT_EVENT_TABLES.json"),
        ("panel", "state_id"),
        audit,
        "joint_tables",
    )
    geometries = _unique(
        read_json(root / "summary/NUMERIC_AND_GEOMETRY_SUPPORT.json"),
        ("panel", "state_id"),
        audit,
        "geometry_tables",
    )
    audit.require(
        set(geometries) == set(model_panels()), "geometry_tables", "MISSING_OR_EXTRA_MODEL_PANEL"
    )
    audit.require(
        set(joints) == set(model_panels()), "joint_tables", "MISSING_OR_EXTRA_MODEL_PANEL"
    )
    cubes, counts, panel_reports = {}, Counter(), []
    shared_families = None
    for panel, state in model_panels():
        payload = load_panel(plan_path, root, panel, state)
        cube, rows, families = recount_panel(
            payload,
            root / f"scoring/{panel}/{state}/SCORED_OUTPUTS.jsonl",
            panel,
            state,
            plan["data"]["root_counts"][panel],
            audit,
        )
        counts[panel] += len(rows)
        if (panel, state) in joints:
            audit.check(
                joints[panel, state].get("identity"),
                payload["identity"],
                f"joint_tables/{panel}/{state}.identity",
            )
            check_joint_table(joints[panel, state], rows, audit, f"joint_tables/{panel}/{state}")
        if (panel, state) in geometries:
            check_root_equal(geometries[panel, state], rows, audit, f"root_equal/{panel}/{state}")
        if panel == "DEV_EVAL":
            cubes[state] = cube
            if shared_families is None:
                shared_families = families
            audit.require(families == shared_families, state, "CROSS_MODEL_ROOT_PAIRING_DIFFERS")
        panel_reports.append(
            dict(
                panel=panel,
                state_id=state,
                completions=len(rows),
                roots=len(families),
                full_counts={m: sum(r[m] for r in rows) for m in METRICS},
                exact_counts={c: sum(r["cell_exact"] == c for r in rows) for c in EXACT_CELLS},
                joint_scorable=sum(r["L"] for r in rows),
            )
        )
    audit.check(dict(counts), {"PROBE": 30720, "DEV_EVAL": 430080}, "actual_raw_recount")
    responses = check_responses(root, cubes, shared_families, plan, audit)
    audit.finish()
    derived = derived_manifests(plan_path, root, freeze)
    aggregates = derive_training_aggregates(root)
    inventory = evidence_inventory(root)
    # Frozen sources and the original dispatch package are separate namespaces.
    source_root = Path(__file__).resolve().parents[2]
    sources = []
    for relative, expected in sorted(freeze["source_hashes"].items()):
        item = descriptor(source_root, relative)
        if item["sha256"] != expected:
            raise RecountFailure("FROZEN_SOURCE_CHANGED:" + relative)
        sources.append(item)
    bundle = plan_path.parent.parent
    bundle_files = [
        descriptor(bundle, row["path"])
        for row in read_json(bundle / "BUNDLE_MANIFEST.json")["files"]
    ]
    bundle_files.append(descriptor(bundle, "BUNDLE_MANIFEST.json"))
    after = common.verify_analysis_readiness(plan_path, root)
    if before != after:
        audit.mismatch("readiness", "EVIDENCE_CHANGED_DURING_RECOUNT")
    # The readiness hash covers raw/training; explicitly close publication races
    # for analysis, sources, frozen images, and newly derived identities too.
    verify_analysis_manifest(root, audit)
    audit.require(
        read_json(root / "summary/ANALYSIS_MANIFEST.json") == analysis_manifest,
        "analysis_manifest",
        "ANALYSIS_MANIFEST_CHANGED_DURING_RECOUNT",
    )
    common.verify_execution(plan_path, root, require_engine=True, full_hashes=True)
    audit.finish()
    report = dict(
        status="PASS",
        plan_id=PLAN_ID,
        readiness=before,
        parser="shared frozen mm_core.contracts.parse_response; NOT an independent parser",
        arithmetic=(
            "independent fractions.Fraction implementation; exact P/C/A, missingness, "
            "six/eight cells, residuals and geometry"
        ),
        bootstrap="independent direct-index PCG64 whole-root resampling; 5000 shared draws",
        float_comparison_absolute_tolerance=FLOAT_ATOL,
        float_comparison_relative_tolerance=0,
        exact_fraction_comparisons_have_no_tolerance=True,
        panels=panel_reports,
        actual_completions=dict(counts),
        response_recount=responses,
        comparisons=audit.comparisons,
        mismatches=[],
        analysis_manifest_sha256=sha256(root / "summary/ANALYSIS_MANIFEST.json"),
        independently_recomputed=[
            "raw response score arithmetic",
            "overall/stratum/cell counts and denominators",
            "per-root and root-equal A/P/J response points",
            "primary effects and both registered CI levels",
            "stratum response CI95",
            "utility, protection and preparation points/CI95",
        ],
        integrity_checked_not_independently_recomputed=[
            "teacher-forced field likelihoods",
            "training tensor equivalence",
            "exploratory opportunity/crossfit and secondary feature algebra",
            "natural-language scientific interpretation",
        ],
        scientific_success_claimed=False,
        next_stage_authorized=False,
        new_model_calls=0,
        new_gpu_jobs=0,
    )
    publish(root, "verification/INDEPENDENT_RECOUNT.json", encoded(report))
    inventory.append(descriptor(root, "verification/INDEPENDENT_RECOUNT.json"))
    receipt = dict(
        status="VERIFIED_RELEASE",
        plan_id=PLAN_ID,
        scientific_success_claimed=False,
        next_stage_authorized=False,
        files=sorted(inventory, key=lambda x: x["path"]),
        source_files=sources,
        source_files_root=str(source_root),
        dispatched_plan_files=sorted(bundle_files, key=lambda x: x["path"]),
        dispatched_plan_files_root=str(bundle),
        frozen_model_identity=freeze["model_identity"],
        code_commit=freeze["code_commit"],
        readiness_evidence_set_sha256=before["evidence_set_sha256"],
        inventory_is_not_a_packaged_archive=True,
        lightweight_package_contains_complete_weights=False,
        retained_checkpoint_bodies=(
            "server-only by default; independently checked SHA256, not tensor equivalence"
        ),
        external_base_weights=(
            "external frozen model paths and per-file identities in "
            "MODEL_AND_ENVIRONMENT_IDENTITY; not copied"
        ),
        excluded_from_immutable_inventory=[
            "active orchestration leases and scheduler projections",
            "operational logs outside inventoried trees",
            "this manifest itself and its later orchestration completion marker",
            "externally stored base model bytes",
            "planned-rotated checkpoint bodies; retained commit receipts are indexed",
        ],
        derived_identity_manifests=derived,
        gpu_hours_policy="ACCOUNTING_ONLY",
    )
    receipt["derived_training_aggregates"] = aggregates
    receipt["training_aggregate_scope"] = (
        "lossless committed-record projections with original path/SHA; all retained "
        "failed/uncommitted originals remain inventoried separately"
    )
    publish(root, "verification/RELEASE_MANIFEST.json", encoded(receipt))
    return receipt


def release(plan_path, run_root):
    """Run the real audit; failed attempts are retained without a success marker."""
    root, plan_path = Path(run_root).resolve(strict=True), Path(plan_path).resolve(strict=True)
    audit = Audit()
    try:
        return _release(plan_path, root, audit)
    except Exception as exc:
        failure = dict(
            status="FAIL",
            plan_id=PLAN_ID,
            error_type=type(exc).__name__,
            error=str(exc),
            comparisons=audit.comparisons,
            mismatches=audit.mismatches,
            scientific_success_claimed=False,
            next_stage_authorized=False,
        )
        data = encoded(failure)
        name = (
            "verification/failed/INDEPENDENT_RECOUNT_" + hashlib.sha256(data).hexdigest() + ".json"
        )
        publish(root, name, data)
        raise
