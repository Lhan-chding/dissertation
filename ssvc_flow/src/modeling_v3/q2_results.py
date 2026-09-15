"""Read-only analysis of the complete Q2 development design matrix.

Numerical prediction/reference arrays are held for one design at a time. Cached
error summaries never supply the reported errors. Two development seeds support
descriptive selection tables, not confirmatory intervals or automatic winners.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .cpu_campaign import (
    CONTRAST_NAMES,
    DEV_SEEDS,
    _digest,
    _sources,
    _verify_complete,
    development_designs,
    trajectory_specs,
)
from .cpu_results import ACCEPTED_LABELS, _summarize_pair_set, _unit_key, _write_csv
from .io import atomic_json, finalize_run, sha256_file, source_identity

LABELS = {
    "IDENTICAL_POLICY",
    "NO_CALIBRATION_EXCITATION",
    "OUT_OF_CALIBRATION_SPAN",
    "HIGH_LEVERAGE",
    "MEASUREMENT_UNRESOLVED",
    "PREDICTABLE_AT_VALIDATED_TOLERANCE",
    "STATISTICAL_BASELINE_ONLY",
    "UNKNOWN",
}


def _server(fixture):
    if not fixture and (platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID")):
        raise ValueError("Full Q2 analysis must run on server CPU under Slurm")


def _contract(config, fixture):
    explicit = config.get("q2_analysis_fixture")
    if fixture:
        if not isinstance(explicit, dict) or set(explicit) != {
            "designs",
            "probe_groups",
            "heldout_banks",
            "parameter_dimension",
        }:
            raise ValueError(
                "Tiny fixture design/shape contract must be declared before collecting originals"
            )
        designs = explicit["designs"]
        shape = (3 * explicit["heldout_banks"], len(explicit["probe_groups"]), 4)
        dimension = explicit["parameter_dimension"]
    else:
        if explicit is not None:
            raise ValueError("A fixture design contract cannot authorize production analysis")
        designs = development_designs(config)
        if len(designs) != 1108:
            raise ValueError("The frozen full Q2 protocol requires all 1108 development designs")
        shape, dimension = (36, 72, 4), 737
    if not designs or len({_digest(d) for d in designs}) != len(designs):
        raise ValueError("Distinct complete development design specification required")
    expected = {_digest(design)[:20]: design for design in designs}
    if len(expected) != len(designs) or min(*shape, dimension) < 1:
        raise ValueError("Invalid frozen fixture shapes or duplicate design identity")
    # Fixture mode does not infer, shrink or waive any independent-unit axis.
    if config["cpu"]["arms"] != ["X_BASE", "X_VALID"] or config["cpu"]["anchors"] != [8, 24, 40]:
        raise ValueError("Q2 requires both source arms and all three frozen anchors")
    specs = trajectory_specs(config, "development", pilot=False)
    units = {
        (s["seed"], s["arm"], anchor, 0, s["initialization_seed"])
        for s in specs
        for anchor in config["cpu"]["anchors"]
    }
    if len(specs) != 4 or len(units) != 12 or sorted({s["seed"] for s in specs}) != list(DEV_SEEDS):
        raise ValueError("Q2 requires two development seeds and four complete source trajectories")
    return expected, units, shape, dimension


def _read(path):
    return json.loads(Path(path).read_text())


def _child_complete(root, child, outer):
    """Use the already byte-verified outer inventory for nested CPU originals."""
    child = Path(child).resolve()
    if not child.is_relative_to(root):
        raise ValueError("Nested original must remain inside its complete Q2 root")
    prefix = str(child.relative_to(root)) + "/"
    marker = prefix + "COMPLETE.json"
    if marker not in outer["files"]:
        raise ValueError("Incomplete nested Q2 original")
    receipt = _read(child / "COMPLETE.json")
    inventory = {
        path[len(prefix) :]: digest
        for path, digest in outer["files"].items()
        if path.startswith(prefix) and path != marker
    }
    if receipt.get("status") != "COMPLETE" or receipt.get("files") != inventory:
        raise ValueError(
            "Nested original completion/inventory differs from the verified outer manifest"
        )
    return receipt


def _index(config, inputs, fixture):
    designs, expected_units, shape, dimension = _contract(config, fixture)
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("Pass the distinct complete Q2 root directories as a list")
    roots = [Path(root).resolve() for root in inputs]
    if len(roots) != len(set(roots)):
        raise ValueError("Duplicate complete Q2 input root")
    records, units, originals = defaultdict(list), {}, {}
    groups = None
    current_sources = _sources()
    direct_specs = {(d["estimator"], d["n"]) for d in designs.values()}
    for root in roots:
        outer = _verify_complete(root)
        binding, summary = outer["binding"], outer["summary"]
        if (
            binding.get("stage") != "Q2"
            or binding.get("role") != "development"
            or binding.get("pilot") is not False
            or binding.get("calibration_receipt") is not None
            or summary.get("stage") != "Q2"
            or summary.get("role") != "development"
            or summary.get("pilot") is not False
            or summary.get("scientific_status") != "DEVELOPMENT_ONLY"
        ):
            raise ValueError("Q2 analysis only consumes non-pilot DEVELOPMENT_ONLY originals")
        if (
            binding.get("config_sha256") != _digest(config)
            or binding.get("source_hashes") != current_sources
        ):
            raise ValueError("Q2 source/config identity differs from the frozen CPU dependency set")
        if binding.get("fixture", False) is not fixture:
            raise ValueError("Q2 production and fixture original scopes differ")
        if (
            binding.get("designs") != list(designs.values())
            or _read(root / "DESIGNS.json") != list(designs.values())
            or _read(root / "COVERAGE_SUMMARY.json") != summary
        ):
            raise ValueError("Complete Q2 design/summary original differs from the frozen matrix")
        if groups is not None and groups != summary.get("probe_groups"):
            raise ValueError("Fixed probe groups differ across original Q2 roots")
        groups = summary.get("probe_groups")
        if (
            not isinstance(groups, list)
            or len(groups) != shape[1]
            or any(not isinstance(g, (str, int)) for g in groups)
        ):
            raise ValueError("Original probe grouping must align with the fixed panel")
        if fixture and groups != config["q2_analysis_fixture"]["probe_groups"]:
            raise ValueError("Fixture original grouping differs from its predeclared panel")
        if not fixture and (
            set(groups) != set(range(6)) or any(groups.count(g) != 12 for g in range(6))
        ):
            raise ValueError("Q2 requires the complete 72-probe, six-group finite-action panel")
        declared = {(row["design_id"], *_unit_key(row)): row for row in summary["results"]}
        if len(declared) != len(summary["results"]) or summary.get("fit_count") != len(declared):
            raise ValueError("Duplicate or inconsistent rows in complete Q2 summary")
        observed = set()
        for unit in sorted((root / "units").iterdir()):
            if not unit.is_dir():
                raise ValueError("Unexpected non-unit in Q2 originals")
            body = _child_complete(root, unit, outer)
            unit_binding, body = body["binding"], body["summary"]
            if unit_binding != {**binding, "origin": unit.name} or body.get("origin") != unit.name:
                raise ValueError("Q2 origin binding changed")
            if _read(unit / "RESULTS.json") != body or not body.get("results"):
                raise ValueError("Complete Q2 unit results are missing or changed")
            unit_keys = {_unit_key(row) for row in body["results"]}
            if len(unit_keys) != 1:
                raise ValueError("Q2 origin mixes independent source identities")
            key = unit_keys.pop()
            if key not in expected_units or key in units:
                raise ValueError("Duplicate or unexpected seed/arm/anchor/repeat/init axis")
            prediction_lock = _read(unit / "PREDICTION_LOCK.json")
            if set(prediction_lock) != set(designs):
                raise ValueError("Every origin must freeze the complete Q2 design matrix")
            access = _read(unit / "REFERENCE_ACCESS_RECEIPT.json")
            if (
                access.get("heldout_read_after_prediction_freeze") is not True
                or access.get("reference_status")
                != "KNOWN_EVENT_SCORE_EXACT_FINITE_ACTION_ENUMERATION"
                or access.get("training_seed") != key[0]
                or access.get("prediction_lock_sha256")
                != sha256_file(unit / "PREDICTION_LOCK.json")
                or body.get("prediction_lock_sha256") != access["prediction_lock_sha256"]
            ):
                raise ValueError("Exact reference access lacks its complete prior prediction lock")
            for row in body["results"]:
                design_id = row["design_id"]
                row_key = (design_id, *key)
                if (
                    design_id not in designs
                    or row["design"] != designs[design_id]
                    or row_key in observed
                    or declared.get(row_key) != row
                    or row.get("origin_id") != unit.name
                ):
                    raise ValueError("Origin rows differ from complete frozen Q2 designs")
                observed.add(row_key)
                fit_root = unit / "fits" / design_id
                fitted = _child_complete(root, fit_root, outer)
                if fitted["binding"] != {"design": row["design"], "origin": unit.name} or fitted[
                    "summary"
                ] != {k: v for k, v in row.items() if k != "metrics"}:
                    raise ValueError("Q2 fitted metadata differs from its completed original")
                path = (unit / row["predictions_relative"]).resolve()
                if path != fit_root / "PREDICTIONS.npz":
                    raise ValueError(
                        "Prediction path differs from the actual complete design original"
                    )
                registered = outer["files"].get(str(path.relative_to(root)))
                if (
                    row.get("prediction_sha256") != registered
                    or prediction_lock[design_id] != registered
                ):
                    raise ValueError("Q2 prediction hashes disagree with the frozen original")
                log = _read(fit_root / "QUERY_LOG.json")
                if (
                    log.get("hidden_query_labels_read") is not False
                    or log.get("role") != "development"
                    or log.get("training_seed") != key[0]
                    or log.get("prediction_sha256") != registered
                ):
                    raise ValueError("A Q2 predictor consumed heldout semantics before freezing")
                records[design_id].append(
                    {"row": row, "unit": unit, "path": path, "truth": unit / "QUERY_REFERENCE.npz"}
                )
            declared_direct = [
                (row["estimator"], row["n"])
                for row in body["direct_baselines"]
                if row["model"] == "DIRECT_MEASURE"
            ]
            if (
                set(declared_direct) != direct_specs
                or len(declared_direct) != len(direct_specs)
                or sum(row["model"] == "KNOWN_EVENT_SCORE" for row in body["direct_baselines"]) != 1
            ):
                raise ValueError("Complete strong direct and exact known-event baselines required")
            for _, n in direct_specs:
                direct_receipt = _child_complete(root, unit / f"direct_n{n}", outer)
                if (
                    direct_receipt["binding"].get("n") != n
                    or direct_receipt["summary"].get("n") != n
                ):
                    raise ValueError(
                        "Direct baseline original draw budget differs from its matched design"
                    )
            units[key] = {
                "unit": unit,
                "truth": unit / "QUERY_REFERENCE.npz",
                "origin_id": unit.name,
            }
        if observed != set(declared):
            raise ValueError("Complete Q2 summary has missing original fit rows")
        originals[str(root / "COMPLETE.json")] = sha256_file(root / "COMPLETE.json")
    if set(units) != expected_units or set(records) != set(designs):
        raise ValueError("The full Q2 design/seed/arm/anchor matrix is incomplete")
    for current in records.values():
        if {_unit_key(row["row"]) for row in current} != expected_units or len(current) != 12:
            raise ValueError("Each design must contain all twelve independent-origin cells")
        current.sort(key=lambda row: _unit_key(row["row"]))
    return {
        "records": records,
        "units": units,
        "designs": designs,
        "groups": groups,
        "shape": shape,
        "dimension": dimension,
        "originals": originals,
        "roots": roots,
    }


def _rank(row):
    k, r = row["k"], row["r"]
    if type(k) is not int or type(r) is not int or not 0 <= r <= k:
        raise ValueError("Invalid actual model dimensions")
    if row.get("rank_comparison_eligible") is not (0 < r < k):
        raise ValueError("Rank eligibility disagrees with actual r and k")
    return "R_LESS_K" if 0 < r < k else "R_EQUALS_K" if r == k > 0 else "RANK_ZERO"


def _load(record, data):
    with np.load(record["path"], allow_pickle=False) as arrays:
        prediction = arrays["prediction"].copy()
        labels = arrays["classifications"].copy()
        geometry = {
            key: arrays[key].copy()
            for key in ("rho", "leverage", "e_norm", "e_perp_norm", "singular_values")
            if key in arrays
        }
        if arrays["Q"].shape != (data["dimension"], record["row"]["k"]) or arrays[
            "directions"
        ].shape != (data["dimension"], record["row"]["r"]):
            raise ValueError("Original basis dimensions disagree with actual k/r")
    with np.load(record["truth"], allow_pickle=False) as arrays:
        truth = arrays["truth"].copy()
    if (
        prediction.shape != data["shape"]
        or truth.shape != prediction.shape
        or labels.shape != (len(truth),)
        or not np.isfinite(truth).all()
    ):
        raise ValueError("Prediction, exact reference, labels or complete probe axes are invalid")
    if not set(labels.tolist()) <= LABELS:
        raise ValueError("Unknown original coverage classification")
    if np.any(labels == "PREDICTABLE_AT_VALIDATED_TOLERANCE"):
        raise ValueError(
            "Development-only Q2 cannot already claim calibrated confirmation acceptance"
        )
    for name in ("rho", "leverage", "e_norm", "e_perp_norm"):
        if name in geometry and (
            geometry[name].shape != (len(truth),)
            or not np.isfinite(geometry[name]).all()
            or np.any(geometry[name] < 0)
        ):
            raise ValueError("Original geometry diagnostics have invalid query axes")
    if "rho" in geometry and np.any(geometry["rho"] > 1 + 1e-8):
        raise ValueError("Original rho lies outside its norm-ratio range")
    return prediction, truth, labels, geometry


def _raw4_rows(design_id, kind, entries):
    rows = []
    for target in [None, *range(3)]:
        errors, signals = [], []
        total = accepted_count = 0
        all_energy = accepted_ss = accepted_abs = 0.0
        for _, prediction, truth, labels in entries:
            if target is not None:
                prediction, truth, labels = (
                    prediction[target::3],
                    truth[target::3],
                    labels[target::3],
                )
            finite = np.isfinite(prediction).all(axis=(1, 2))
            accepted = finite & np.isin(labels, list(ACCEPTED_LABELS))
            residual = prediction - truth
            errors.append(residual[finite].reshape(-1))
            signals.append(truth[finite].reshape(-1))
            total += truth.size
            all_energy += float(np.sum(truth**2))
            accepted_count += int(residual[accepted].size)
            accepted_ss += float(np.sum(residual[accepted] ** 2))
            accepted_abs += float(np.abs(residual[accepted]).sum())
        residual, signal = np.concatenate(errors), np.concatenate(signals)
        count, ss, energy = residual.size, float(np.sum(residual**2)), float(np.sum(signal**2))
        is_model = kind == "MODEL"
        row = {
            "design_id": design_id,
            "kind": kind,
            "target_index": target,
            "target": "ALL_TARGETS" if target is None else CONTRAST_NAMES[target],
            "channel": "RAW4",
            "all_cases": total,
            "finite_cases": int(count),
            "unresolved_prediction_cases": total - count,
            "unknown_cases": total - accepted_count if is_model else None,
            "accepted_cases": accepted_count if is_model else None,
            "coverage": accepted_count / total if is_model else None,
            "accepted_mse": accepted_ss / accepted_count if accepted_count else None,
            "accepted_mae": accepted_abs / accepted_count if accepted_count else None,
            "acceptance_is_finiteness": False,
            "acceptance_applicable": is_model,
            "all_case_mse": ss / total if count == total else None,
            "finite_case_mse": ss / count if count else None,
            "mae": float(np.abs(residual).mean()) if count == total else None,
            "bias": float(residual.mean()) if count == total else None,
            "error_ss": ss,
            "signal_ss": energy,
            "all_signal_ss": all_energy,
            "pooled_residual_variance": float(np.var(residual, ddof=1)) if count > 1 else None,
            "pooled_variance_is_measurement_noise_variance": False,
            "nrmse": math.sqrt(ss / energy) if count == total and energy > 0 else None,
            "nrmse_status": "UNKNOWN_PREDICTION"
            if count != total
            else "UNDEFINED_ZERO_SIGNAL"
            if energy == 0
            else "DEFINED",
            "quantile_scope": "ALL_CASES" if count == total else "FINITE_SUBSET_ONLY",
            "quantiles_from_raw_residuals": True,
            "quantiles_are_confidence_intervals": False,
            "maximum_absolute_residual": float(np.max(np.abs(residual))) if count else None,
        }
        row.update(
            {
                f"q{int(q * 100)}_absolute_residual": float(np.quantile(np.abs(residual), q))
                if count
                else None
                for q in (0.5, 0.9, 0.95, 0.99)
            }
        )
        rows.append(row)
    return rows


def _metrics(design_id, kind, entries, groups):
    rows, seed_rows = _summarize_pair_set(design_id, kind, iter(entries), groups)
    return rows + _raw4_rows(design_id, kind, entries), seed_rows


def _distribution(array):
    value = np.asarray(array, dtype=float)
    return (
        {
            "min": float(value.min()),
            "median": float(np.median(value)),
            "q95": float(np.quantile(value, 0.95)),
            "max": float(value.max()),
        }
        if value.size
        else None
    )


def _origin_summary(record, prediction, labels, geometry):
    row = record["row"]
    finite = np.isfinite(prediction).all(axis=(1, 2))
    counts = dict(sorted(Counter(labels.tolist()).items()))
    result = {
        "design_id": row["design_id"],
        "origin_id": row["origin_id"],
        **{
            key: row[key]
            for key in ("seed", "arm", "anchor", "repeat", "initialization_seed", "k", "r")
        },
        "rank_relation": _rank(row),
        "condition_number": row.get("condition_number"),
        "all_queries": len(prediction),
        "finite_queries": int(finite.sum()),
        "unresolved_prediction_queries": int((~finite).sum()),
        "classification_counts": counts,
        "target_excitation": row.get("target_excitation_from_actual_parameter_contrasts"),
        "geometry_alone_validates_tolerance": False,
        "exact_identical_policy_queries": counts.get("IDENTICAL_POLICY", 0),
        "learned_tolerance_accepted_queries": counts.get("PREDICTABLE_AT_VALIDATED_TOLERANCE", 0),
        "model_unknown_queries": len(labels)
        - int(np.count_nonzero(np.isin(labels, list(ACCEPTED_LABELS)) & finite))
        if row["design"]["model"] != "ZERO"
        else None,
        "no_excitation_queries": counts.get("NO_CALIBRATION_EXCITATION", 0),
        "out_of_span_queries": counts.get("OUT_OF_CALIBRATION_SPAN", 0),
        "high_leverage_queries": counts.get("HIGH_LEVERAGE", 0),
        "measurement_unresolved_queries": counts.get("MEASUREMENT_UNRESOLVED", 0),
        "scientific_status": "DEVELOPMENT_ONLY",
        "prediction_sha256": row["prediction_sha256"],
        "regression": row.get("fit_metadata", {}).get("regression"),
        "output_policy": row.get("fit_metadata", {}).get("output_policy"),
        "calibration_contrast_draw_budget": row.get("calibration_contrast_draw_budget"),
        "counterfactual_selected_bank_score_actions": row.get(
            "counterfactual_selected_bank_score_actions"
        ),
        "counterfactual_shared_generation_actions": row.get(
            "counterfactual_shared_generation_actions"
        ),
        "all_pool_geometry_acquisition_fork_updates": row.get(
            "all_pool_geometry_acquisition_fork_updates"
        ),
    }
    for key in ("rho", "leverage", "e_norm", "e_perp_norm"):
        result[key] = _distribution(geometry[key]) if key in geometry else None
    result["singular_values"] = (
        geometry["singular_values"].tolist() if "singular_values" in geometry else None
    )
    if "singular_values" in geometry:
        singular = geometry["singular_values"]
        if singular.ndim != 1 or not np.isfinite(singular).all() or np.any(singular < 0):
            raise ValueError("Original calibration spectrum is invalid")
    return result


class _Table:
    """Stream a single table, exposing its final name only after successful analysis."""

    def __init__(self, root, name):
        self.path = root / name
        self.pending = root / ("." + name + ".pending")
        self.stream = self.pending.open("x", newline="")
        self.writer = None

    def add(self, rows):
        for row in rows:
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )

    def finish(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.link(self.pending, self.path)
        self.pending.unlink()

    def close(self):
        self.stream.close()


def _rank_comparisons(designs, origin_summaries, origin_scores):
    comparisons = []
    for design_id, design in designs.items():
        if design["study"] != "DIMENSION" or design["model"] in {
            "ZERO",
            "FULL_RIDGE",
            "FULL_GLS",
            "RBF_RIDGE",
        }:
            continue
        for full_id, full in designs.items():
            if (
                full["study"] != "DIMENSION"
                or full["model"] not in {"FULL_RIDGE", "FULL_GLS"}
                or full["rank_cap"] != "FULL"
                or any(
                    design.get(key) != full.get(key)
                    for key in ("m", "n", "selector", "design_seed", "estimator", "alpha")
                )
            ):
                continue
            for (current, origin), reduced in origin_summaries.items():
                if current != design_id:
                    continue
                baseline = origin_summaries[full_id, origin]
                if (
                    reduced["regression"] != baseline["regression"]
                    or reduced["output_policy"] != baseline["output_policy"]
                ):
                    continue
                if baseline["r"] != baseline["k"] or reduced["k"] != baseline["k"]:
                    raise ValueError(
                        "Matched full and reduced fits have different calibration ranks"
                    )
                left, right = origin_scores[design_id, origin], origin_scores[full_id, origin]
                comparisons.append(
                    {
                        "design_id": design_id,
                        "full_design_id": full_id,
                        "origin_id": origin,
                        "seed": reduced["seed"],
                        "arm": reduced["arm"],
                        "anchor": reduced["anchor"],
                        "rank_relation": reduced["rank_relation"],
                        "r": reduced["r"],
                        "k": reduced["k"],
                        "same_regression_objective": True,
                        "reduced_error_ss": left["error_ss"],
                        "full_error_ss": right["error_ss"],
                        "reduced_all_case_mse": left["all_case_mse"],
                        "full_all_case_mse": right["all_case_mse"],
                        "paired_mse_difference": left["all_case_mse"] - right["all_case_mse"]
                        if left["all_case_mse"] is not None and right["all_case_mse"] is not None
                        else None,
                        "r_equals_k_is_compression": False,
                        "scientific_status": "DEVELOPMENT_ONLY",
                    }
                )
    return comparisons


def analyze_coverage(config, inputs, out, *, fixture=False):
    """Publish descriptive full-Q2 selection tables without choosing any design."""
    _server(fixture)
    analysis_source = source_identity()
    data = _index(config, inputs, fixture)
    root = Path(out).resolve()
    if any(root.is_relative_to(original) for original in data["roots"]):
        raise ValueError("Analysis output must not modify an immutable input root")
    root.mkdir(parents=True, exist_ok=False)
    origin_table, summary_table, seed_table = [
        _Table(root, name)
        for name in ("ORIGIN_METRICS.csv", "SUMMARY_METRICS.csv", "SEED_METRICS.csv")
    ]
    origin_summaries, origin_scores, design_summaries = {}, {}, []
    summaries = []
    try:
        for design_id, records in data["records"].items():
            design = data["designs"][design_id]
            kind = "ZERO" if design["model"] == "ZERO" else "MODEL"
            entries, geometries, classifications = [], defaultdict(list), Counter()
            current_summaries = []
            for record in records:
                prediction, truth, labels, geometry = _load(record, data)
                identity, origin = _unit_key(record["row"]), record["row"]["origin_id"]
                entry = (identity, prediction, truth, labels)
                entries.append(entry)
                metrics, _ = _metrics(design_id, kind, [entry], data["groups"])
                origin_table.add(
                    [
                        {
                            **metric,
                            "origin_id": origin,
                            "seed": identity[0],
                            "arm": identity[1],
                            "anchor": identity[2],
                        }
                        for metric in metrics
                    ]
                )
                origin_scores[design_id, origin] = next(
                    m for m in metrics if m["target"] == "ALL_TARGETS"
                )
                origin_row = _origin_summary(record, prediction, labels, geometry)
                origin_summaries[design_id, origin] = origin_row
                current_summaries.append(origin_row)
                classifications.update(labels.tolist())
                for key in ("rho", "leverage", "e_norm", "e_perp_norm"):
                    if key in geometry:
                        geometries[key].append(geometry[key])
            metrics, seed_metrics = _metrics(design_id, kind, entries, data["groups"])
            summary_table.add(metrics)
            seed_table.add(seed_metrics)
            summaries.extend(metrics)
            dimensions = Counter(row["rank_relation"] for row in current_summaries)
            design_summaries.append(
                {
                    "design_id": design_id,
                    "design": design,
                    "origins": len(entries),
                    "r_less_k_origins": dimensions["R_LESS_K"],
                    "r_equals_k_origins": dimensions["R_EQUALS_K"],
                    "rank_zero_origins": dimensions["RANK_ZERO"],
                    "classification_counts": dict(sorted(classifications.items())),
                    "k_min": min(row["k"] for row in current_summaries),
                    "k_max": max(row["k"] for row in current_summaries),
                    "r_min": min(row["r"] for row in current_summaries),
                    "r_max": max(row["r"] for row in current_summaries),
                    **{
                        key: _distribution(np.concatenate(geometries[key]))
                        if geometries[key]
                        else None
                        for key in ("rho", "leverage", "e_norm", "e_perp_norm")
                    },
                    "scientific_status": "DEVELOPMENT_ONLY",
                }
            )
            del entries, prediction, truth, labels, geometry, entry
        direct_specs = {(d["estimator"], d["n"]) for d in data["designs"].values()}
        for method, n in [*sorted(direct_specs), ("KNOWN_EVENT_SCORE", None)]:
            kind = "KNOWN_EVENT_SCORE" if n is None else "DIRECT_MEASURE"
            identifier = kind if n is None else f"DIRECT_MEASURE/{method}/n{n}"
            entries = []
            for identity, record in sorted(data["units"].items()):
                with np.load(record["truth"], allow_pickle=False) as arrays:
                    truth = arrays["truth"].copy()
                if n is None:
                    prediction = truth.copy()
                else:
                    with np.load(
                        record["unit"] / f"direct_n{n}/estimates.npz", allow_pickle=False
                    ) as arrays:
                        if method not in arrays:
                            raise ValueError("Strong direct baseline estimator original is missing")
                        base = arrays[method]
                    if base.shape != (data["shape"][0] // 3, 2, data["shape"][1], 4):
                        raise ValueError("Strong direct baseline has incomplete target/prompt axes")
                    prediction = np.stack(
                        (base[:, 0], base[:, 1], base[:, 0] - base[:, 1]), axis=1
                    ).reshape(data["shape"])
                entry = (identity, prediction, truth, np.full(len(truth), "DIRECT_BASELINE"))
                entries.append(entry)
                metrics, _ = _metrics(identifier, kind, [entry], data["groups"])
                origin_table.add(
                    [
                        {
                            **metric,
                            "origin_id": record["origin_id"],
                            "seed": identity[0],
                            "arm": identity[1],
                            "anchor": identity[2],
                        }
                        for metric in metrics
                    ]
                )
            metrics, seed_metrics = _metrics(identifier, kind, entries, data["groups"])
            summary_table.add(metrics)
            seed_table.add(seed_metrics)
            summaries.extend(metrics)
            del entries, prediction, truth, entry
        _write_csv(root / "DESIGN_ORIGIN_SUMMARY.csv", list(origin_summaries.values()))
        _write_csv(root / "DESIGN_SUMMARY.csv", design_summaries)
        comparisons = _rank_comparisons(data["designs"], origin_summaries, origin_scores)
        _write_csv(root / "RANK_COMPARISONS.csv", comparisons)
        for table in (origin_table, summary_table, seed_table):
            table.finish()
    finally:
        for table in (origin_table, summary_table, seed_table):
            table.close()
    report = {
        "schema": "ssvc-v3-q2-development-analysis-1",
        "stage": "Q2_ANALYSIS",
        "status": "DEVELOPMENT_ANALYZED",
        "scientific_status": "DEVELOPMENT_ONLY",
        "fixture": fixture,
        "requires_server_cpu": not fixture,
        "config_sha256": _digest(config),
        "source_sha256": analysis_source["sha256"],
        "cpu_dependency_source_hashes": _sources(),
        "source_manifest_hashes": data["originals"],
        "independent_development_seeds": list(DEV_SEEDS),
        "design_count": len(data["designs"]),
        "completed_fits": sum(len(items) for items in data["records"].values()),
        "unit_matrix": {
            "seeds": list(DEV_SEEDS),
            "arms": config["cpu"]["arms"],
            "anchors": config["cpu"]["anchors"],
            "repeat": [0],
            "initialization_seed": config["cpu"]["fixed_parent_initialization"],
            "trajectory_count": 4,
            "origin_count": 12,
        },
        "primary_v": "-delta_pI",
        "raw_valid_event_sum": "SEPARATE_DIAGNOSTIC_X_PLUS_S_PLUS_W",
        "probe_groups": data["groups"],
        "designs": design_summaries,
        "metrics": summaries,
        "rank_comparison_rows": len(comparisons),
        "confirmatory_inference": "NOT_PERFORMED",
        "confidence_intervals": None,
        "winner_selected": False,
        "thresholds_selected": False,
        "online_ssvc": "NOT_CERTIFIED",
        "quantiles_are_confidence_intervals": False,
        "memory_scope": (
            "One design's prediction/reference arrays across twelve origins; "
            "no full design matrix arrays retained"
        ),
        "baselines": (
            "Strong direct estimators at every matched draw budget; "
            "exact known-event enumeration separately"
        ),
        "table_files": [
            "DESIGN_ORIGIN_SUMMARY.csv",
            "DESIGN_SUMMARY.csv",
            "ORIGIN_METRICS.csv",
            "SUMMARY_METRICS.csv",
            "SEED_METRICS.csv",
            "RANK_COMPARISONS.csv",
        ],
    }
    if source_identity() != analysis_source:
        raise ValueError("Analysis source changed while original metrics were being computed")
    atomic_json(root / "Q2_ANALYSIS.json", report)
    finalize_run(
        root,
        {"config": _digest(config), "source": report["source_sha256"]},
        metadata={"phase": "Q2_ANALYSIS", "scientific_status": "DEVELOPMENT_ONLY"},
    )
    return {
        **{key: value for key, value in report.items() if key not in {"metrics", "designs"}},
        "receipt": {
            "path": str(root / "Q2_ANALYSIS.json"),
            "sha256": sha256_file(root / "Q2_ANALYSIS.json"),
        },
    }
