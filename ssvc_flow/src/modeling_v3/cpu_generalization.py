"""Analyze the prespecified orthogonal CPU initializations independently.

Only completed originals are read. No fit, threshold, radius, or bank selection
is changed here. A source-initialization calibration envelope is an empirical
transfer diagnostic under initialization shift, never a coverage guarantee.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from .cpu_campaign import CONTRAST_NAMES, _verify_complete
from .cpu_results import (
    _collect,
    _direct_id,
    _load_pair,
    _paired_comparisons,
    _server,
    _summarize_pair_set,
    _unit_key,
    _update_geometry_curve,
    _verify_calibration,
    _write_csv,
)
from .io import atomic_json, canonical_hash, finalize_run, sha256_file, source_identity
from .schema import verify_selection_lock
from .statistics import seed_bootstrap

ROLE = "orthogonal_generalization"
SCOPE = "TRANSFERRED_FROZEN_ROUTE_DIAGNOSTIC_NOT_CALIBRATED_UNDER_INITIALIZATION_SHIFT"


def _validate_matrix(config, data, *, fixture):
    cpu = config["cpu"]
    orthogonal = cpu["orthogonal_generalization"]
    axes = [
        sorted(orthogonal["train_rng_seeds"]),
        cpu["arms"],
        cpu["anchors"],
        list(range(cpu["repeat_measurements_test"])),
        sorted(orthogonal["init_seeds"]),
    ]
    if any(not axis or len(axis) != len(set(axis)) for axis in axes):
        raise ValueError("orthogonal frozen matrix axes must be nonempty and unique")
    if set(orthogonal["arms"]) != set(cpu["arms"]):
        raise ValueError("orthogonal frozen matrix arm definitions differ")
    if orthogonal["data_seed"] != cpu["fixed_parent_data_seed"]:
        raise ValueError("orthogonal initialization study must retain the frozen data seed")
    if set(axes[0]) & set(cpu["interval_calibration_seeds"] + cpu["locked_test_seeds"]):
        raise ValueError("orthogonal training seeds overlap calibration or locked test")
    if not fixture and (
        len(axes[0]) != 10
        or set(axes[1]) != {"X_BASE", "X_VALID"}
        or axes[2] != [8, 24, 40]
        or axes[3] != list(range(20))
        or axes[4] != [7101, 7201]
        or cpu["fixed_parent_initialization"] in axes[4]
    ):
        raise ValueError("complete orthogonal 40-trajectory frozen matrix required")
    expected = set(itertools.product(*axes))
    if not data["records"]:
        raise ValueError("empty orthogonal frozen matrix")
    for records in data["records"].values():
        keys = [_unit_key(record["row"]) for record in records]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("orthogonal frozen matrix is incomplete, duplicated, or unexpected")
    return {
        "seeds": axes[0],
        "arms": axes[1],
        "anchors": axes[2],
        "repetitions": axes[3],
        "initialization_seeds": axes[4],
        "trajectory_count": len(axes[0]) * len(axes[1]) * len(axes[4]),
        "unit_count_per_design": len(expected),
    }


def _calibration_chain(config, lock, receipt, *, fixture):
    if not fixture and isinstance(receipt, dict):
        raise ValueError("production calibration requires its immutable receipt path")
    checked = _verify_calibration(config, lock, receipt, fixture)
    expected_status = "TEST_FIXTURE_NOT_AUTHORIZATION" if fixture else "CALIBRATION_ANALYZED"
    if checked.get("status") != expected_status:
        raise ValueError("completed calibration analysis status required")
    roots = [str(Path(path).parent) for path in checked["source_manifest_hashes"]]
    calibration_data = _collect(config, lock, roots, "interval_calibration", fixture)
    if checked.get("calibration_seeds") != calibration_data["seeds"]:
        raise ValueError("calibration seed receipt differs from completed originals")
    if set(checked["designs"]) != set(calibration_data["designs"]):
        raise ValueError("calibration envelope design matrix is incomplete")
    for identifier, envelope in checked["designs"].items():
        if envelope.get("design") != calibration_data["designs"][identifier]:
            raise ValueError("calibration envelope differs from frozen design")
        if envelope.get("nominal") != config["statistics"]["coverage_nominal"]:
            raise ValueError("calibration nominal differs from frozen protocol")
        for name in ("radius", "tolerance"):
            value = envelope.get(name)
            if name == "radius" and value is None:
                continue
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("calibration envelope must be finite nonnegative or unresolved")
    return checked


def _check_prediction_freezes(data):
    expected_by_unit = {}
    for identifier, records in data["records"].items():
        for record in records:
            expected_by_unit.setdefault(record["unit"], {})[identifier] = record["row"][
                "prediction_sha256"
            ]
    for record in data["units"].values():
        unit = record["unit"]
        lock_path = unit / "PREDICTION_LOCK.json"
        frozen = json.loads(lock_path.read_text())
        receipt = json.loads((unit / "REFERENCE_ACCESS_RECEIPT.json").read_text())
        if receipt.get("heldout_read_after_prediction_freeze") is not True or receipt.get(
            "prediction_lock_sha256"
        ) != sha256_file(lock_path):
            raise ValueError("generalization reference lacks a bound prediction freeze")
        if frozen != expected_by_unit[unit]:
            raise ValueError("generalization prediction lock differs from original predictions")


def _stratum(data, initialization):
    return {
        **data,
        "records": {
            identifier: [r for r in records if r["row"]["initialization_seed"] == initialization]
            for identifier, records in data["records"].items()
        },
        "units": {
            identity: r for identity, r in data["units"].items() if identity[-1] == initialization
        },
    }


def _bootstrap_metrics(seed_rows, config):
    grouped = {}
    for row in seed_rows:
        grouped.setdefault(
            (row["design_id"], row["kind"], row["target_index"], row["channel"]), []
        ).append(row)
    results = []
    for (identifier, kind, target, channel), rows in grouped.items():
        if channel not in {"X", "v", "group_pX", "group_v"}:
            continue
        for metric, field in (("mse", "error_ss"), ("mae", "absolute_error_sum")):
            item = {
                "design_id": identifier,
                "kind": kind,
                "target_index": target,
                "target": CONTRAST_NAMES[target],
                "channel": channel,
                "metric": metric,
            }
            if any(row["finite_count"] != row["count"] for row in rows):
                item.update(status="UNKNOWN_INCOMPLETE_PREDICTIONS", estimate=None, interval=None)
            else:
                item.update(
                    seed_bootstrap(
                        [row["seed"] for row in rows],
                        [row[field] / row["count"] for row in rows],
                        reps=config["statistics"]["bootstrap_reps"],
                        seed=config["statistics"]["bootstrap_seed"],
                    )
                )
            item["all_arms_anchors_banks_prompts_repetitions_kept_with_seed"] = True
            results.append(item)
    return results


def _risk_bootstrap(seed_values, config):
    """Resample entire seed numerators and denominators, including rejected seeds."""
    values = np.array(
        [
            [row["all_cases"], row["accepted_cases"], row["accepted_error_ss"]]
            for row in seed_values
        ],
        dtype=float,
    )
    reps, seed = config["statistics"]["bootstrap_reps"], config["statistics"]["bootstrap_seed"]
    rng = np.random.default_rng(seed)
    totals = values[rng.integers(0, len(values), size=(reps, len(values)))].sum(1)
    covered = totals[:, 1] / totals[:, 0]
    resolved = totals[:, 1] > 0
    risk = totals[resolved, 2] / totals[resolved, 1]
    return {
        "unit": "training_seed",
        "independent_seeds": len(values),
        "reps": reps,
        "bootstrap_seed": seed,
        "coverage_interval": np.quantile(covered, [0.025, 0.975]).tolist(),
        "accepted_mse_interval": np.quantile(risk, [0.025, 0.975]).tolist()
        if resolved.all()
        else None,
        "replicates_with_no_accepted_cases": int((~resolved).sum()),
        "status": "EMPIRICAL" if resolved.all() else "UNRESOLVED_ACCEPTED_RISK_BOOTSTRAP",
        "ratio_of_seed_sums": True,
        "rejected_seeds_retained": True,
    }


def _analyze_stratum(config, lock, data, calibration, initialization, *, fixture):
    metrics, seed_metrics, interval_coverage, risk_curves = [], [], [], []
    for identifier, records in data["records"].items():
        envelope = calibration["designs"][identifier]
        seed_maxima = {seed: 0.0 for seed in data["seeds"]}
        unresolved = set()
        curves_by_seed = {seed: {} for seed in data["seeds"]}
        kind = "ZERO" if data["designs"][identifier]["model"] == "ZERO" else "MODEL"

        def entries(
            current=records,
            maxima=seed_maxima,
            missing=unresolved,
            source_envelope=envelope,
            current_kind=kind,
            curves=curves_by_seed,
        ):
            for record in current:
                prediction, truth, labels = _load_pair(record, data["groups"], fixture=fixture)
                seed = record["row"]["seed"]
                if np.isfinite(prediction).all():
                    maxima[seed] = max(maxima[seed], float(np.abs(prediction - truth).max()))
                else:
                    missing.add(seed)
                if np.any(labels == "PREDICTABLE_AT_VALIDATED_TOLERANCE") and (
                    source_envelope["radius"] is None
                    or source_envelope["radius"] > source_envelope["tolerance"]
                ):
                    raise ValueError("transferred route lacks a qualifying frozen source envelope")
                if current_kind == "MODEL":
                    if set(record["geometry"]) != {"rho", "leverage"}:
                        raise ValueError("generalization risk curves require frozen query geometry")
                    _update_geometry_curve(
                        curves[seed],
                        prediction,
                        truth,
                        labels,
                        record["geometry"],
                        data["groups"],
                        lock["selected"],
                        source_envelope,
                    )
                yield _unit_key(record["row"]), prediction, truth, labels

        rows, seed_rows = _summarize_pair_set(identifier, kind, entries(), data["groups"])
        metrics.extend(rows)
        seed_metrics.extend(seed_rows)
        curve_keys = sorted(next(iter(curves_by_seed.values())))
        for target, channel, multiplier in curve_keys:
            per_seed = [curves_by_seed[seed][target, channel, multiplier] for seed in data["seeds"]]
            values = {key: sum(row[key] for row in per_seed) for key in per_seed[0]}
            risk_curves.append(
                {
                    "design_id": identifier,
                    "target_index": target,
                    "target": CONTRAST_NAMES[target],
                    "channel": channel,
                    "geometry_threshold_multiplier": multiplier,
                    "frozen_operating_point": multiplier == 1,
                    "thresholds_chosen_from_generalization_errors": False,
                    "other_points": "PRESPECIFIED_EXPLORATORY_ROUTE_DIAGNOSTIC",
                    **values,
                    "unknown_cases": values["all_cases"] - values["accepted_cases"],
                    "coverage": values["accepted_cases"] / values["all_cases"],
                    "accepted_mse": values["accepted_error_ss"] / values["accepted_cases"]
                    if values["accepted_cases"]
                    else None,
                    "seed_bootstrap": _risk_bootstrap(per_seed, config),
                }
            )
        radius = envelope["radius"]
        covered = {
            str(seed): (seed not in unresolved and seed_maxima[seed] <= radius)
            if radius is not None
            else None
            for seed in data["seeds"]
        }
        interval_coverage.append(
            {
                "design_id": identifier,
                "radius": radius,
                "interval_width": 2 * radius if radius is not None else None,
                "source_nominal": envelope["nominal"],
                "coverage_object": (
                    "seed maximum across both arms, anchors, query contrasts, probes "
                    "and repetitions at this initialization"
                ),
                "all_generalization_seeds": len(data["seeds"]),
                "unresolved_seeds": sorted(unresolved),
                "seed_covered": covered,
                "seed_max_absolute_errors": {
                    str(seed): None if seed in unresolved else seed_maxima[seed]
                    for seed in data["seeds"]
                },
                "empirical_seed_coverage": sum(covered.values()) / len(covered)
                if radius is not None
                else None,
                "coverage_seed_bootstrap": seed_bootstrap(
                    data["seeds"],
                    list(covered.values()),
                    reps=config["statistics"]["bootstrap_reps"],
                    seed=config["statistics"]["bootstrap_seed"],
                )
                if radius is not None
                else None,
                "unknown_seeds_retained_as_uncovered": True,
                "distribution_shift_covered": False,
            }
        )

    direct_specs = {(design["estimator"], design["n"]) for design in data["designs"].values()}
    for estimator, n in sorted(direct_specs):

        def direct_entries(method=estimator, draws=n):
            for identity, record in sorted(data["units"].items()):
                root = record["unit"] / f"direct_n{draws}"
                _verify_complete(root)
                with np.load(root / "estimates.npz", allow_pickle=False) as arrays:
                    base = arrays[method].copy()
                with np.load(record["truth_path"], allow_pickle=False) as arrays:
                    truth = arrays["truth"].copy()
                if base.shape != (len(truth) // 3, 2, len(data["groups"]), 4):
                    raise ValueError("direct baseline shape does not match frozen query targets")
                prediction = np.stack((base[:, 0], base[:, 1], base[:, 0] - base[:, 1]), 1).reshape(
                    truth.shape
                )
                yield identity, prediction, truth, np.full(len(truth), "DIRECT_BASELINE")

        rows, seed_rows = _summarize_pair_set(
            _direct_id(estimator, n), "DIRECT_MEASURE", direct_entries(), data["groups"]
        )
        metrics.extend(rows)
        seed_metrics.extend(seed_rows)

    def exact_entries():
        for identity, record in sorted(data["units"].items()):
            with np.load(record["truth_path"], allow_pickle=False) as arrays:
                truth = arrays["truth"].copy()
            yield identity, truth, truth, np.full(len(truth), "EXACT_FINITE_EVENT_BASELINE")

    rows, seed_rows = _summarize_pair_set(
        "KNOWN_EVENT_SCORE", "KNOWN_EVENT_SCORE", exact_entries(), data["groups"]
    )
    metrics.extend(rows)
    seed_metrics.extend(seed_rows)
    paired = _paired_comparisons(data, seed_metrics, config)
    bootstraps = _bootstrap_metrics(seed_metrics, config)
    for table in (metrics, seed_metrics, paired, bootstraps, risk_curves, interval_coverage):
        for row in table:
            row.update(
                initialization_seed=initialization,
                role=ROLE,
                calibration_guarantee_transfers=False,
                routing_scope=SCOPE,
            )
    return {
        "initialization_seed": initialization,
        "seeds": data["seeds"],
        "metrics": metrics,
        "seed_metrics": seed_metrics,
        "paired_comparisons": paired,
        "seed_bootstrap_metrics": bootstraps,
        "interval_coverage": interval_coverage,
        "risk_coverage_curves": risk_curves,
        "rank_cells": {
            identifier: {
                "total_fits": len(records),
                "r_less_k_fits": sum(0 < r["row"]["r"] < r["row"]["k"] for r in records),
                "r_equals_k_fits": sum(r["row"]["r"] == r["row"]["k"] > 0 for r in records),
                "rank_zero_fits": sum(r["row"]["k"] == 0 for r in records),
            }
            for identifier, records in data["records"].items()
        },
    }


def analyze_generalization(config, lock, inputs, out, *, calibration_receipt, fixture=False):
    """Verify and report both frozen orthogonal initializations without refitting."""
    lock = verify_selection_lock(config, lock)
    calibration = _calibration_chain(config, lock, calibration_receipt, fixture=fixture)
    _server(fixture)
    # _collect's full mode hardcodes cal/test axes. Its fixture flag is used ONLY
    # for shared structural/hash validation. The production orthogonal matrix
    # and real response shapes are enforced below, never inferred from data.
    data = _collect(config, lock, inputs, ROLE, fixture=True)
    matrix = _validate_matrix(config, data, fixture=fixture)
    for root in inputs:
        if _verify_complete(root)["binding"].get("calibration_receipt") != calibration:
            raise ValueError("generalization predictions use a different calibration receipt")
    _check_prediction_freezes(data)
    strata = [
        _analyze_stratum(config, lock, _stratum(data, init), calibration, init, fixture=fixture)
        for init in matrix["initialization_seeds"]
    ]
    report = {
        "schema": "ssvc-v3-cpu-generalization-analysis-1",
        "stage": "Q3_ORTHOGONAL_GENERALIZATION_ANALYSIS",
        "kind": "GENERALIZATION",
        "role": ROLE,
        "fixture": bool(fixture),
        "status": "TEST_FIXTURE_NOT_AUTHORIZATION"
        if fixture
        else "GENERALIZATION_ANALYZED_NOT_CERTIFIED",
        "config_sha256": canonical_hash(config),
        "selection_hash": lock["selection_hash"],
        "source_sha256": source_identity()["sha256"],
        "source_manifest_hashes": data["originals"],
        "calibration_source_manifest_hashes": calibration["source_manifest_hashes"],
        "calibration_receipt_sha256": canonical_hash(calibration),
        "calibration_receipt_file_sha256": sha256_file(calibration_receipt)
        if not isinstance(calibration_receipt, dict)
        else None,
        "calibration_seeds": calibration["calibration_seeds"],
        "unit_matrix": matrix,
        "initialization_strata": strata,
        "pooled_initialization_inference": False,
        "independent_unit": "training_seed within each fixed initialization",
        "initialization_draws_are_population_sample": False,
        "distribution_shift_included": True,
        "calibration_guarantee_transfers": False,
        "source_initialization_seed": config["cpu"]["fixed_parent_initialization"],
        "primary_targets": {"pX": "X", "v": "-I"},
        "routing_scope": SCOPE,
        "thresholds_refitted": False,
        "formal_hypothesis_tests": "NOT_RUN_SECONDARY_GENERALIZATION",
        "online_ssvc": "NOT_CERTIFIED",
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "GENERALIZATION_ANALYSIS.json", report)
    for filename, field in (
        ("METRICS.csv", "metrics"),
        ("SEED_METRICS.csv", "seed_metrics"),
        ("PAIRED_COMPARISONS.csv", "paired_comparisons"),
        ("SEED_BOOTSTRAP_METRICS.csv", "seed_bootstrap_metrics"),
        ("INTERVAL_COVERAGE.csv", "interval_coverage"),
        ("RISK_COVERAGE.csv", "risk_coverage_curves"),
    ):
        _write_csv(out / filename, [row for layer in strata for row in layer[field]])
    finalize_run(
        out,
        {
            "config": report["config_sha256"],
            "source": report["source_sha256"],
            "selection": report["selection_hash"],
        },
        metadata={"stage": report["stage"], "fixture": bool(fixture)},
    )
    return report
