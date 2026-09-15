"""Q3 analysis of completed, immutable finite-action CPU response originals.

The calibration score is a *seed maximum*, not a pooled error quantile. Test
bootstrap resamples whole training seeds and keeps paired arms, origins, banks,
prompts, targets and measurement repetitions together. No analysis launches a
model or a scheduler job. Full analysis runs on the server; tiny fixtures are
explicitly marked unusable as experiment authorization.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cpu_campaign import CONTRAST_NAMES, _digest, _verify_complete, development_designs
from .io import (
    atomic_bytes,
    atomic_json,
    canonical_hash,
    finalize_run,
    sha256_file,
    source_identity,
    verify_manifest,
)
from .schema import verify_selection_lock
from .statistics import conformal_radius, holm, seed_bootstrap

ACCEPTED_LABELS = {"IDENTICAL_POLICY", "PREDICTABLE_AT_VALIDATED_TOLERANCE"}
COVERAGE_OBJECT = (
    "maximum absolute four-channel prediction error across both arms, all three anchors, "
    "all heldout bank contrasts, all probe prompts and all measurement repetitions "
    "for one future exchangeable training seed with the same fixed data and initialization"
)


def _server(fixture):
    if not fixture and platform.system() != "Linux":
        raise ValueError(
            "full CPU result analysis must run on the server; fixture mode is not authorization"
        )


def _unit_key(row):
    return tuple(row[key] for key in ("seed", "arm", "anchor", "repeat", "initialization_seed"))


def _safe_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("response artifact path must stay inside its complete unit")
    return path


def _collect(config, lock, response_roots, role, fixture):
    roots = [Path(root).resolve() for root in response_roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("distinct completed response roots required")
    expected_designs = {
        _digest(d)[:20]: d for d in development_designs(config, selected=lock["selected"])
    }
    records = defaultdict(list)
    originals = {}
    groups = None
    unit_records = {}
    for root in roots:
        manifest = _verify_complete(root)
        summary = manifest["summary"]
        if summary.get("role") != role or summary.get("pilot") is not False:
            raise ValueError("analysis requires the exact non-pilot seed role")
        if manifest["binding"].get("config_sha256") != canonical_hash(config):
            raise ValueError("response config binding differs from frozen protocol")
        bound = manifest["binding"].get("source_hashes", {})
        current = source_identity()["files"]
        if not bound or any(
            current.get(path) != digest for path, digest in bound.items() if path.startswith("src/")
        ):
            raise ValueError("response source differs from frozen implementation")
        declared = summary.get("probe_groups")
        if not isinstance(declared, list) or not declared:
            raise ValueError("response originals require fixed probe_groups metadata")
        if groups is not None and groups != declared:
            raise ValueError("response roots disagree on fixed probe grouping")
        groups = declared
        originals[str(root / "COMPLETE.json")] = sha256_file(root / "COMPLETE.json")
        declared_rows = {(row["design_id"], *_unit_key(row)): row for row in summary["results"]}
        if len(declared_rows) != len(summary["results"]):
            raise ValueError("duplicate rows in completed response summary")
        seen = set()
        for unit in sorted((root / "units").iterdir()):
            if not unit.is_dir():
                continue
            unit_manifest = _verify_complete(unit)
            body = unit_manifest["summary"]
            local = []
            for row in body["results"]:
                design_id = row["design_id"]
                if (
                    design_id not in expected_designs
                    or row["design"] != expected_designs[design_id]
                ):
                    raise ValueError("response design differs from frozen selection")
                key = (design_id, *_unit_key(row))
                if key in seen or key not in declared_rows or declared_rows[key] != row:
                    raise ValueError("unit rows differ from complete response summary")
                seen.add(key)
                if row.get("origin_id", unit.name) != unit.name:
                    raise ValueError("response origin identity does not match its unit")
                prediction_path = _safe_file(unit, row["predictions_relative"])
                if sha256_file(prediction_path) != row["prediction_sha256"]:
                    raise ValueError("frozen prediction hash mismatch")
                record = {
                    "row": row,
                    "unit": unit,
                    "prediction_path": prediction_path,
                    "truth_path": _safe_file(unit, "QUERY_REFERENCE.npz"),
                }
                records[design_id].append(record)
                local.append(record)
            if not local:
                raise ValueError("empty completed response unit")
            unit_key = _unit_key(local[0]["row"])
            if any(_unit_key(record["row"]) != unit_key for record in local):
                raise ValueError("response unit mixes source identities")
            if unit_key in unit_records:
                raise ValueError("duplicate independent-unit identity across response roots")
            unit_records[unit_key] = local[0]
        if seen != set(declared_rows):
            raise ValueError("completed summary has missing raw response units")
    if set(records) != set(expected_designs):
        raise ValueError("frozen design matrix is incomplete")
    first = [record["row"] for record in next(iter(records.values()))]
    seeds = sorted({row["seed"] for row in first})
    if fixture:
        axes = [
            sorted({row[key] for row in first})
            for key in ("seed", "arm", "anchor", "repeat", "initialization_seed")
        ]
    else:
        expected_seeds = config["cpu"][
            "interval_calibration_seeds" if role == "interval_calibration" else "locked_test_seeds"
        ]
        if seeds != sorted(expected_seeds):
            raise ValueError(
                "complete twenty calibration or thirty locked-test seed matrix required"
            )
        axes = [
            sorted(expected_seeds),
            config["cpu"]["arms"],
            config["cpu"]["anchors"],
            range(config["cpu"]["repeat_measurements_test"]),
            [config["cpu"]["fixed_parent_initialization"]],
        ]
        if role == "interval_calibration" and len(seeds) != 20:
            raise ValueError("exactly twenty independent calibration seeds required")
        if role == "locked_test" and len(seeds) != 30:
            raise ValueError("exactly thirty independent locked-test seeds required")
    expected = set(itertools.product(*axes))
    for design_records in records.values():
        keys = [_unit_key(record["row"]) for record in design_records]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(
                "seed/arm/anchor/repetition/initialization matrix is incomplete or duplicated"
            )
    return {
        "records": dict(records),
        "units": unit_records,
        "designs": expected_designs,
        "seeds": seeds,
        "groups": groups,
        "originals": originals,
        "matrix": {
            "seeds": list(axes[0]),
            "arms": list(axes[1]),
            "anchors": list(axes[2]),
            "repetitions": list(axes[3]),
            "initialization_seeds": list(axes[4]),
        },
    }


def _load_pair(record, groups, *, fixture):
    with np.load(record["prediction_path"], allow_pickle=False) as data:
        prediction = data["prediction"].copy()
        classifications = data["classifications"].copy()
        record["geometry"] = {key: data[key].copy() for key in ("rho", "leverage") if key in data}
    with np.load(record["truth_path"], allow_pickle=False) as data:
        truth = data["truth"].copy()
    if (
        prediction.shape != truth.shape
        or truth.ndim != 3
        or truth.shape[1:] != (len(groups), 4)
        or len(truth) == 0
        or len(truth) % 3
        or classifications.shape != (len(truth),)
        or not np.isfinite(truth).all()
    ):
        raise ValueError(
            "raw response shape, target order, classifications or reference is invalid"
        )
    if not fixture and truth.shape != (36, 72, 4):
        raise ValueError("locked CPU response requires twelve banks, three contrasts and 72 probes")
    return prediction, truth, classifications


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    atomic_bytes(path, stream.getvalue().encode())


def analyze_calibration(config, selection_lock, response_roots, out, *, fixture=False):
    """Freeze seed-max conformal envelopes before accessing locked-test labels."""
    lock = verify_selection_lock(config, selection_lock)
    data = _collect(config, lock, response_roots, "interval_calibration", fixture)
    _server(fixture)
    designs = {}
    scores = []
    nominal = float(config["statistics"]["coverage_nominal"])
    tolerance = float(config["statistics"]["old_toy_q95_threshold"])
    for design_id, records in data["records"].items():
        maxima = {seed: 0.0 for seed in data["seeds"]}
        unresolved = set()
        for record in records:
            prediction, truth, _ = _load_pair(record, data["groups"], fixture=fixture)
            seed = record["row"]["seed"]
            if not np.isfinite(prediction).all():
                unresolved.add(seed)
            else:
                maxima[seed] = max(maxima[seed], float(np.abs(prediction - truth).max()))
        for seed in data["seeds"]:
            scores.append(
                {
                    "design_id": design_id,
                    "seed": seed,
                    "max_absolute_four_channel_error": None if seed in unresolved else maxima[seed],
                    "status": "UNKNOWN" if seed in unresolved else "OBSERVED",
                }
            )
        envelope = conformal_radius(list(maxima.values()), nominal=nominal)
        if unresolved:
            envelope.update(radius=None, status="UNRESOLVED_CALIBRATION_PREDICTION")
        designs[design_id] = {
            **envelope,
            "design": data["designs"][design_id],
            "tolerance": tolerance,
            "tolerance_applied_to": "MAX_ABSOLUTE_FOUR_CHANNEL_ERROR",
            "threshold_origin": "old_toy_q95_threshold; applied here to the stricter seed maximum",
            "coverage_object": COVERAGE_OBJECT,
            "unresolved_seeds": sorted(unresolved),
            "all_cases_including_geometry_rejections": True,
            "classification_pending_calibration_is_not_missing_prediction": True,
            "empirical_quantile_is_confidence_interval": False,
            "exchangeability_required": True,
            "online_control_certified": False,
        }
    report = {
        "schema": "ssvc-v3-cpu-calibration-1",
        "stage": "Q3_INTERVAL_CALIBRATION",
        "status": "TEST_FIXTURE_NOT_AUTHORIZATION" if fixture else "CALIBRATION_ANALYZED",
        "fixture": bool(fixture),
        "selection_hash": lock["selection_hash"],
        "config_sha256": canonical_hash(config),
        "source_sha256": source_identity()["sha256"],
        "calibration_seeds": data["seeds"],
        "source_manifest_hashes": data["originals"],
        "unit_matrix": data["matrix"],
        "designs": designs,
        "sequence": "CALIBRATION_BEFORE_LOCKED_TEST",
        "online_ssvc": "NOT_CERTIFIED",
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "CALIBRATION_RECEIPT.json", report)
    _write_csv(out / "SEED_MAX_ERRORS.csv", scores)
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


def _verify_calibration(config, lock, receipt, fixture):
    if not isinstance(receipt, dict):
        receipt_path = Path(receipt)
        verify_manifest(receipt_path.parent)
        receipt = json.loads(receipt_path.read_text())
    if receipt.get("fixture") and not fixture:
        raise ValueError("test fixture calibration cannot authorize locked-test analysis")
    if (
        receipt.get("selection_hash") != lock["selection_hash"]
        or receipt.get("config_sha256") != canonical_hash(config)
        or receipt.get("source_sha256") != source_identity()["sha256"]
    ):
        raise ValueError("calibration receipt source/config/selection binding mismatch")
    if not fixture and receipt.get("calibration_seeds") != sorted(
        config["cpu"]["interval_calibration_seeds"]
    ):
        raise ValueError("completed twenty-seed interval calibration required")
    if not receipt.get("source_manifest_hashes") or not receipt.get("designs"):
        raise ValueError("calibration requires bound completed originals and design envelopes")
    for path, digest in receipt["source_manifest_hashes"].items():
        if sha256_file(path) != digest:
            raise ValueError("calibration original manifest changed")
        _verify_complete(Path(path).parent)
    return receipt


def _channels(prediction, truth, classifications, groups, target):
    pred, actual = prediction[target::3], truth[target::3]
    labels = classifications[target::3]
    finite_query = np.isfinite(pred).all(axis=(1, 2))
    accepted_query = np.isin(labels, list(ACCEPTED_LABELS)) & finite_query
    for index, name in enumerate(("X", "S", "W", "I")):
        yield name, pred[..., index], actual[..., index], finite_query, accepted_query
    yield "v", -pred[..., 3], -actual[..., 3], finite_query, accepted_query
    yield (
        "raw_valid_event_sum",
        pred[..., :3].sum(-1),
        actual[..., :3].sum(-1),
        finite_query,
        accepted_query,
    )
    group_ids = list(dict.fromkeys(groups))
    grouped_prediction = np.stack(
        [pred[:, np.asarray(groups) == group].mean(axis=1) for group in group_ids], axis=1
    )
    grouped_truth = np.stack(
        [actual[:, np.asarray(groups) == group].mean(axis=1) for group in group_ids], axis=1
    )
    yield (
        "group_pX",
        grouped_prediction[..., 0],
        grouped_truth[..., 0],
        finite_query,
        accepted_query,
    )
    yield (
        "group_v",
        -grouped_prediction[..., 3],
        -grouped_truth[..., 3],
        finite_query,
        accepted_query,
    )


def _summarize_pair_set(identifier, kind, entries, groups):
    """Consume one design at a time; pool raw residuals, not per-unit quantiles."""
    chunks = defaultdict(list)
    for identity, prediction, truth, labels in entries:
        for target in range(3):
            for channel, pred, actual, finite, accepted in _channels(
                prediction, truth, labels, groups, target
            ):
                chunks[(target, channel)].append(
                    (identity[0], pred.copy(), actual.copy(), finite.copy(), accepted.copy())
                )
    rows, seed_rows = [], []
    for (target, channel), items in chunks.items():
        residual_parts, signal_parts = [], []
        per_seed = defaultdict(
            lambda: {
                "count": 0,
                "finite_count": 0,
                "error_ss": 0.0,
                "signal_ss": 0.0,
                "absolute_error_sum": 0.0,
            }
        )
        all_count = accepted_count = 0
        all_signal_ss = 0.0
        accepted_ss = accepted_abs = 0.0
        for seed, pred, truth, finite_query, accepted_query in items:
            finite = np.broadcast_to(finite_query[:, None], pred.shape)
            accepted = np.broadcast_to(accepted_query[:, None], pred.shape)
            error = pred - truth
            residual_parts.append(error[finite])
            signal_parts.append(truth[finite])
            all_count += error.size
            all_signal_ss += float(np.square(truth).sum())
            accepted_count += int(accepted.sum())
            accepted_ss += float(np.square(error[accepted]).sum())
            accepted_abs += float(np.abs(error[accepted]).sum())
            values = per_seed[seed]
            values["count"] += error.size
            values["finite_count"] += int(finite.sum())
            values["error_ss"] += float(np.square(error[finite]).sum())
            values["signal_ss"] += float(np.square(truth[finite]).sum())
            values["absolute_error_sum"] += float(np.abs(error[finite]).sum())
        residual = np.concatenate(residual_parts)
        signal = np.concatenate(signal_parts)
        error_ss, signal_ss = float(np.square(residual).sum()), float(np.square(signal).sum())
        finite_count = residual.size
        is_model = kind == "MODEL"
        row = {
            "design_id": identifier,
            "kind": kind,
            "target_index": target,
            "target": CONTRAST_NAMES[target],
            "channel": channel,
            "all_cases": all_count,
            "finite_cases": int(finite_count),
            "unresolved_prediction_cases": all_count - finite_count,
            "unknown_cases": all_count - accepted_count if is_model else None,
            "accepted_cases": accepted_count if is_model else None,
            "coverage": accepted_count / all_count if is_model else None,
            "accepted_mse": accepted_ss / accepted_count if accepted_count else None,
            "accepted_mae": accepted_abs / accepted_count if accepted_count else None,
            "acceptance_is_finiteness": False,
            "acceptance_applicable": is_model,
            "all_case_mse": error_ss / all_count if finite_count == all_count else None,
            "finite_case_mse": error_ss / finite_count if finite_count else None,
            "mae": float(np.abs(residual).mean()) if finite_count == all_count else None,
            "bias": float(residual.mean()) if finite_count == all_count else None,
            "error_ss": error_ss,
            "signal_ss": signal_ss,
            "all_signal_ss": all_signal_ss,
            "pooled_residual_variance": float(np.var(residual, ddof=1))
            if finite_count > 1
            else None,
            "pooled_variance_is_measurement_noise_variance": False,
            "nrmse": math.sqrt(error_ss / signal_ss)
            if signal_ss > 0 and finite_count == all_count
            else None,
            "nrmse_status": "UNKNOWN_PREDICTION"
            if finite_count != all_count
            else "UNDEFINED_ZERO_SIGNAL"
            if signal_ss == 0
            else "DEFINED",
            "quantile_scope": "ALL_CASES" if finite_count == all_count else "FINITE_SUBSET_ONLY",
            "quantiles_from_raw_residuals": True,
            "quantiles_are_confidence_intervals": False,
        }
        for q in (0.5, 0.9, 0.95, 0.99):
            row[f"q{int(q * 100)}_absolute_residual"] = (
                float(np.quantile(np.abs(residual), q)) if finite_count else None
            )
        row["maximum_absolute_residual"] = float(np.abs(residual).max()) if finite_count else None
        rows.append(row)
        seed_rows.extend(
            {
                "design_id": identifier,
                "kind": kind,
                "target_index": target,
                "channel": channel,
                "seed": seed,
                **values,
            }
            for seed, values in sorted(per_seed.items())
        )
    return rows, seed_rows


def _direct_id(estimator, n):
    return f"DIRECT_MEASURE/{estimator}/n{n}"


def _update_geometry_curve(
    accumulator, prediction, truth, labels, geometry, groups, selected, envelope
):
    """Fixed geometry score thresholds; heldout errors never select the route."""
    if set(geometry) != {"rho", "leverage"}:
        return
    rho, leverage = geometry["rho"], geometry["leverage"]
    if rho.shape != (len(prediction),) or leverage.shape != rho.shape:
        raise ValueError("coverage curve geometry must match query rows")
    rho_limit, h_limit = selected["rho_threshold"], selected["leverage_threshold"]
    rho_score = rho / rho_limit if rho_limit > 0 else np.where(rho == 0, 0, np.inf)
    h_score = leverage / h_limit if h_limit > 0 else np.where(leverage == 0, 0, np.inf)
    score = np.maximum(rho_score, h_score)
    radius = envelope.get("radius")
    qualified = radius is not None and radius <= envelope["tolerance"]
    finite = np.isfinite(prediction).all(axis=(1, 2))
    excited = labels != "NO_CALIBRATION_EXCITATION"
    for multiplier in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        accepted = finite & (
            (labels == "IDENTICAL_POLICY") | (qualified & excited & (score <= multiplier))
        )
        for target in range(3):
            for channel, pred, actual, _finite, _accepted in _channels(
                prediction, truth, labels, groups, target
            ):
                if channel not in {"group_pX", "group_v"}:
                    continue
                key = (target, channel, multiplier)
                values = accumulator.setdefault(
                    key, {"all_cases": 0, "accepted_cases": 0, "accepted_error_ss": 0.0}
                )
                mask = np.broadcast_to(accepted[target::3, None], pred.shape)
                values["all_cases"] += pred.size
                values["accepted_cases"] += int(mask.sum())
                values["accepted_error_ss"] += float(np.square((pred - actual)[mask]).sum())


def _paired_comparisons(data, seed_rows, config):
    indexed = {
        (row["design_id"], row["target_index"], row["channel"], row["seed"]): row
        for row in seed_rows
    }
    results = []
    for design_id, design in data["designs"].items():
        comparator = _direct_id(design["estimator"], design["n"])
        for target in range(3):
            for channel in ("X", "v", "group_pX", "group_v"):
                for metric, field in (("mse", "error_ss"), ("mae", "absolute_error_sum")):
                    differences = []
                    for seed in data["seeds"]:
                        left = indexed[(design_id, target, channel, seed)]
                        right = indexed[(comparator, target, channel, seed)]
                        if (
                            left["count"] != right["count"]
                            or left["finite_count"] != left["count"]
                            or right["finite_count"] != right["count"]
                        ):
                            differences = []
                            break
                        differences.append(
                            left[field] / left["count"] - right[field] / right["count"]
                        )
                    result = {
                        "comparison_id": (
                            f"{design_id}__vs__{comparator}__t{target}__{channel}__{metric}"
                        ),
                        "design_id": design_id,
                        "comparator": comparator,
                        "target_index": target,
                        "target": CONTRAST_NAMES[target],
                        "channel": channel,
                        "metric": metric,
                        "paired_unit": "training_seed",
                        "negative_favors_model": True,
                        "arms_anchors_banks_prompts_noise_repetitions_kept_together": True,
                        "formal_pvalue": None,
                    }
                    if differences:
                        result.update(
                            seed_bootstrap(
                                data["seeds"],
                                differences,
                                reps=config["statistics"]["bootstrap_reps"],
                                seed=config["statistics"]["bootstrap_seed"],
                            )
                        )
                        result["paired_seed_differences"] = dict(
                            zip(map(str, data["seeds"]), differences, strict=True)
                        )
                    else:
                        result.update(
                            status="UNKNOWN_INCOMPLETE_PAIRED_PREDICTIONS",
                            estimate=None,
                            interval=None,
                        )
                    results.append(result)
    return results


def _formal_tests(lock, paired, config):
    specs = lock["selected"].get("primary_hypothesis_tests", [])
    if (
        len(specs) != 4
        or len({spec.get("id") for spec in specs}) != 4
        or len({spec.get("comparison_id") for spec in specs}) != 4
    ):
        return {
            "status": "NOT_RUN_NO_COMPLETE_FROZEN_FOUR_TEST_SPECIFICATIONS",
            "pvalues": None,
            "holm_applied": False,
            "exploratory_bootstrap_intervals_are_not_four_RQ_tests": True,
        }
    indexed = {row["comparison_id"]: row for row in paired}
    results = []
    for spec in specs:
        if (
            spec.get("test") != "PAIRED_SEED_SIGN_FLIP_MEAN"
            or spec.get("alternative") != "two-sided"
            or spec.get("null_sign_symmetry_assumed") is not True
            or spec.get("comparison_id") not in indexed
        ):
            return {
                "status": "NOT_RUN_UNSUPPORTED_OR_INCOMPLETE_FROZEN_TEST",
                "pvalues": None,
                "holm_applied": False,
            }
        comparison = indexed[spec["comparison_id"]]
        if "paired_seed_differences" not in comparison:
            return {
                "status": "NOT_RUN_UNKNOWN_PREDICTIONS_IN_PRIMARY_TEST",
                "pvalues": None,
                "holm_applied": False,
            }
        values = np.asarray(list(comparison["paired_seed_differences"].values()))
        rng = np.random.default_rng(int(spec["randomization_seed"]))
        reps = int(spec.get("randomization_reps", config["statistics"]["bootstrap_reps"]))
        if reps < 1:
            raise ValueError("frozen randomization repetitions must be positive")
        null = np.abs((rng.choice([-1.0, 1.0], size=(reps, len(values))) * values).mean(axis=1))
        observed = abs(float(values.mean()))
        pvalue = float((1 + np.sum(null >= observed)) / (reps + 1))
        results.append(
            {
                "hypothesis_id": spec["id"],
                "comparison_id": spec["comparison_id"],
                "test": spec["test"],
                "raw_pvalue": pvalue,
                "repetitions": reps,
                "null_sign_symmetry_assumed": True,
                "randomization_seed": spec["randomization_seed"],
            }
        )
    for row, adjusted in zip(results, holm([row["raw_pvalue"] for row in results]), strict=True):
        row["holm_adjusted_pvalue"] = float(adjusted)
    return {"status": "FOUR_FROZEN_TESTS_EXECUTED", "holm_applied": True, "tests": results}


def _frozen_primary_comparisons(config, lock, data, *, fixture=False):
    """Re-read only declared endpoint originals; select rank scope before losses."""
    from .frozen_comparisons import (
        paired_seed_inference,
        summarize_primary_family,
        validate_primary_comparison_family,
    )

    family = validate_primary_comparison_family(config, lock["selected"])
    if family is None:
        return None
    indexed = {
        did: {_unit_key(r["row"]): r for r in records} for did, records in data["records"].items()
    }
    unit_keys = sorted(data["units"])
    results = []
    groups = np.asarray(data["groups"])
    group_ids = sorted(set(groups.tolist()))

    def endpoint(spec, key):
        if spec["kind"] == "MODEL":
            record = indexed[spec["design_id"]][key]
            with np.load(record["prediction_path"], allow_pickle=False) as a:
                arrays = {
                    name: a[name].copy()
                    for name in a.files
                    if name in {"prediction", "Q", "directions", "query_e", "selected_bank_indices"}
                }
            return arrays["prediction"], arrays, record
        record = data["units"][key]
        with np.load(
            record["unit"] / f"direct_n{spec['n']}" / "estimates.npz", allow_pickle=False
        ) as a:
            base = a[spec["estimator"]].copy()
        pred = np.stack((base[:, 0], base[:, 1], base[:, 0] - base[:, 1]), axis=1).reshape(
            -1, len(groups), 4
        )
        return pred, None, record

    for spec in family["hypotheses"][:3]:
        per_seed = {
            seed: {
                "seed": seed,
                "left_loss_sum": 0.0,
                "right_loss_sum": 0.0,
                "count": 0,
                "eligible_origin_count": 0,
                "total_origin_count": 0,
                "eligible_measurement_units": 0,
                "total_measurement_units": 0,
                "status": "OBSERVED",
                "channel_totals": {
                    c: {"left_loss_sum": 0.0, "right_loss_sum": 0.0, "count": 0}
                    for c in spec["channels"]
                },
            }
            for seed in data["seeds"]
        }
        all_origins, eligible_origins = defaultdict(set), defaultdict(set)
        rank_audit = []
        for key in unit_keys:
            row = per_seed[key[0]]
            row["total_measurement_units"] += 1
            all_origins[key[0]].add((key[1], key[2], key[4]))
            lp, la, lr = endpoint(spec["left"], key)
            rp, ra, rr = endpoint(spec["right"], key)
            eligible = True
            if spec["scope"] == "COMMON_R_LESS_K_ORIGINS":
                for arrays, record in [(la, lr), (ra, rr)]:
                    meta = record["row"]
                    k, r = meta["k"], meta["r"]
                    if (
                        set(arrays)
                        != {"prediction", "Q", "directions", "query_e", "selected_bank_indices"}
                        or arrays["Q"].ndim != 2
                        or arrays["Q"].shape[1] != k
                        or arrays["directions"].shape != (arrays["Q"].shape[0], r)
                        or not np.isfinite(arrays["Q"]).all()
                        or not np.isfinite(arrays["directions"]).all()
                    ):
                        raise ValueError("Original dimension/geometry arrays are incomplete")
                    cap = meta["design"]["rank_cap"]
                    eligible &= 0 < r < k if cap != "FULL" else r == k and k > 0
                if lr["row"]["k"] != rr["row"]["k"] or any(
                    not np.array_equal(la[name], ra[name])
                    for name in ["Q", "query_e", "selected_bank_indices"]
                ):
                    raise ValueError(
                        "RQ3 requires identical calibration geometry, query and bank originals"
                    )
                rank_audit.append(
                    {
                        "unit_key": list(key),
                        "left_r": lr["row"]["r"],
                        "right_r": rr["row"]["r"],
                        "k": lr["row"]["k"],
                        "included": bool(eligible),
                    }
                )
            if not eligible:
                continue
            row["eligible_measurement_units"] += 1
            eligible_origins[key[0]].add((key[1], key[2], key[4]))
            # The rank inclusion decision above uses no reference or residual.
            if lr["truth_path"] != rr["truth_path"]:
                raise ValueError("Paired comparison must use the same original reference")
            with np.load(lr["truth_path"], allow_pickle=False) as a:
                truth = a["truth"].copy()
            if (
                lp.shape != truth.shape
                or rp.shape != truth.shape
                or truth.shape[1:] != (len(groups), 4)
            ):
                raise ValueError("Paired prediction/reference axes do not align")
            target = spec["target_index"]
            left = lp[target::3]
            right = rp[target::3]
            reference = truth[target::3]
            finite = (
                np.isfinite(left).all()
                and np.isfinite(right).all()
                and np.isfinite(reference).all()
            )
            if not finite:
                row["status"] = "UNKNOWN_REQUIRED_PREDICTION_OR_REFERENCE"
            for channel, index, sign in [("group_pX", 0, 1), ("group_v", 3, -1)]:
                left_group_error = np.stack(
                    [
                        sign
                        * (left[:, groups == g, index] - reference[:, groups == g, index]).mean(
                            axis=1
                        )
                        for g in group_ids
                    ],
                    axis=1,
                )
                right_group_error = np.stack(
                    [
                        sign
                        * (right[:, groups == g, index] - reference[:, groups == g, index]).mean(
                            axis=1
                        )
                        for g in group_ids
                    ],
                    axis=1,
                )
                values = row["channel_totals"][channel]
                values["count"] += left_group_error.size
                if finite:
                    values["left_loss_sum"] += float(np.sum(left_group_error**2))
                    values["right_loss_sum"] += float(np.sum(right_group_error**2))
        for seed, row in per_seed.items():
            row["total_origin_count"] = len(all_origins[seed])
            row["eligible_origin_count"] = len(eligible_origins[seed])
            row["left_loss_sum"] = sum(v["left_loss_sum"] for v in row["channel_totals"].values())
            row["right_loss_sum"] = sum(v["right_loss_sum"] for v in row["channel_totals"].values())
            row["count"] = sum(v["count"] for v in row["channel_totals"].values())
        result = paired_seed_inference(
            spec,
            [per_seed[s] for s in sorted(per_seed)],
            bootstrap_seed=config["statistics"]["bootstrap_seed"],
        )
        result.update(
            source_role="locked_test",
            original_bindings=data["originals"],
            selection_hash=lock["selection_hash"],
            accepted_mask_used=False,
            eligibility_selected_before_reference=True,
            origin_definition=(
                "seed/arm/anchor/initialization; observation repeats retained within each origin"
            ),
            rank_audit=rank_audit,
        )
        results.append(result)
    return {
        "family": family,
        "results": results,
        "summary": summarize_primary_family(family, results),
    }


def verify_cpu_primary_comparisons(config, selection_lock, analysis_report, *, fixture=False):
    """Verify complete analysis originals and recompute only frozen CPU RQ pairs."""
    _server(fixture)
    lock = verify_selection_lock(config, selection_lock)
    if isinstance(analysis_report, dict):
        if set(analysis_report) != {"path", "sha256"}:
            raise ValueError(
                "Actual analysis path/hash binding required, not hand-written p-values"
            )
        path = Path(analysis_report["path"])
        if sha256_file(path) != analysis_report["sha256"]:
            raise ValueError("CPU primary analysis binding hash mismatch")
    else:
        path = Path(analysis_report)
    verify_manifest(path.parent)
    report = json.loads(path.read_text())
    expected_status = (
        "TEST_FIXTURE_NOT_AUTHORIZATION" if fixture else "LOCKED_TEST_ANALYZED_NOT_ONLINE_CERTIFIED"
    )
    if (
        report.get("schema") != "ssvc-v3-cpu-test-analysis-1"
        or report.get("stage") != "Q3_LOCKED_TEST_ANALYSIS"
        or report.get("fixture") is not bool(fixture)
        or report.get("status") != expected_status
        or report.get("config_sha256") != canonical_hash(config)
        or report.get("selection_hash") != lock["selection_hash"]
        or report.get("source_sha256") != source_identity()["sha256"]
    ):
        raise ValueError("CPU primary analysis role/source/config/selection identity mismatch")
    originals = report.get("source_manifest_hashes", {})
    if not originals:
        raise ValueError("Complete CPU response original bindings required")
    for name, digest in originals.items():
        if Path(name).name != "COMPLETE.json" or sha256_file(name) != digest:
            raise ValueError("CPU primary response original manifest hash mismatch")
    data = _collect(config, lock, [Path(name).parent for name in originals], "locked_test", fixture)
    if report.get("test_seeds") != data["seeds"] or report.get("unit_matrix") != data["matrix"]:
        raise ValueError("CPU primary test seed/matrix identity mismatch")
    expected = _frozen_primary_comparisons(config, lock, data, fixture=fixture)
    if expected is None or expected != report.get("primary_comparison_family"):
        raise ValueError("CPU primary statistics differ from original frozen comparisons")
    return expected


def analyze_test(
    config, selection_lock, response_roots, out, *, calibration_receipt, fixture=False
):
    """Analyze the complete independent fixed-initialization locked CPU test."""
    lock = verify_selection_lock(config, selection_lock)
    calibration = _verify_calibration(config, lock, calibration_receipt, fixture)
    data = _collect(config, lock, response_roots, "locked_test", fixture)
    _server(fixture)
    metrics, seed_metrics, interval_coverage, risk_curves = [], [], [], []
    if not fixture:
        for root in response_roots:
            binding = _verify_complete(root)["binding"]
            if binding.get("calibration_receipt") != calibration:
                raise ValueError(
                    "test predictions were not generated under this calibration receipt"
                )
    for design_id, records in data["records"].items():
        envelope = calibration["designs"].get(design_id)
        if envelope is None:
            raise ValueError("test design has no frozen calibration envelope")
        seed_maxima = {seed: 0.0 for seed in data["seeds"]}
        unresolved_seeds = set()
        curve = {}

        def entries(
            current=records,
            maxima=seed_maxima,
            unresolved=unresolved_seeds,
            current_envelope=envelope,
            geometry_curve=curve,
        ):
            for record in current:
                prediction, truth, labels = _load_pair(record, data["groups"], fixture=fixture)
                seed = record["row"]["seed"]
                if np.isfinite(prediction).all():
                    maxima[seed] = max(maxima[seed], float(np.abs(prediction - truth).max()))
                else:
                    unresolved.add(seed)
                if not fixture and np.any(labels == "PREDICTABLE_AT_VALIDATED_TOLERANCE"):
                    radius = current_envelope.get("radius")
                    if radius is None or radius > current_envelope["tolerance"]:
                        raise ValueError(
                            "accepted prediction lacks a qualifying frozen error envelope"
                        )
                _update_geometry_curve(
                    geometry_curve,
                    prediction,
                    truth,
                    labels,
                    record["geometry"],
                    data["groups"],
                    lock["selected"],
                    current_envelope,
                )
                yield _unit_key(record["row"]), prediction, truth, labels

        kind = "ZERO" if data["designs"][design_id]["model"] == "ZERO" else "MODEL"
        rows, seed_rows = _summarize_pair_set(design_id, kind, entries(), data["groups"])
        metrics.extend(rows)
        seed_metrics.extend(seed_rows)
        for (target, channel, multiplier), values in sorted(curve.items()):
            if kind == "ZERO":
                continue
            risk_curves.append(
                {
                    "design_id": design_id,
                    "target_index": target,
                    "target": CONTRAST_NAMES[target],
                    "channel": channel,
                    "geometry_threshold_multiplier": multiplier,
                    "frozen_operating_point": multiplier == 1,
                    "thresholds_chosen_from_test_errors": False,
                    "other_points": "EXPLORATORY_ROUTE_DIAGNOSTIC_NOT_NEW_CONTROL_AUTHORIZATION",
                    **values,
                    "unknown_cases": values["all_cases"] - values["accepted_cases"],
                    "coverage": values["accepted_cases"] / values["all_cases"],
                    "accepted_mse": values["accepted_error_ss"] / values["accepted_cases"]
                    if values["accepted_cases"]
                    else None,
                }
            )
        radius = envelope.get("radius")
        covered = {
            str(seed): (seed not in unresolved_seeds and seed_maxima[seed] <= radius)
            if radius is not None
            else None
            for seed in data["seeds"]
        }
        interval_coverage.append(
            {
                "design_id": design_id,
                "radius": radius,
                "interval_width": 2 * radius if radius is not None else None,
                "nominal": envelope["nominal"],
                "coverage_object": COVERAGE_OBJECT,
                "all_test_seeds": len(data["seeds"]),
                "unresolved_seeds": sorted(unresolved_seeds),
                "seed_covered": covered,
                "seed_max_absolute_errors": {
                    str(seed): None if seed in unresolved_seeds else seed_maxima[seed]
                    for seed in data["seeds"]
                },
                "empirical_seed_coverage": sum(covered.values()) / len(covered)
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
                direct_root = record["unit"] / f"direct_n{draws}"
                _verify_complete(direct_root)
                with np.load(direct_root / "estimates.npz", allow_pickle=False) as arrays:
                    base = arrays[method].copy()
                with np.load(record["truth_path"], allow_pickle=False) as arrays:
                    truth = arrays["truth"].copy()
                prediction = np.stack(
                    (base[:, 0], base[:, 1], base[:, 0] - base[:, 1]), axis=1
                ).reshape(truth.shape)
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
    primary_family = _frozen_primary_comparisons(config, lock, data, fixture=fixture)
    formal = primary_family["summary"] if primary_family else _formal_tests(lock, paired, config)
    report = {
        "schema": "ssvc-v3-cpu-test-analysis-1",
        "stage": "Q3_LOCKED_TEST_ANALYSIS",
        "status": "TEST_FIXTURE_NOT_AUTHORIZATION"
        if fixture
        else "LOCKED_TEST_ANALYZED_NOT_ONLINE_CERTIFIED",
        "fixture": bool(fixture),
        "selection_hash": lock["selection_hash"],
        "config_sha256": canonical_hash(config),
        "source_sha256": source_identity()["sha256"],
        "test_seeds": data["seeds"],
        "source_manifest_hashes": data["originals"],
        "calibration_receipt_sha256": canonical_hash(calibration),
        "unit_matrix": data["matrix"],
        "primary_targets": {"pX": "X", "v": "-I"},
        "metrics": metrics,
        "paired_comparisons": paired,
        "formal_tests": formal,
        "primary_comparison_family": primary_family,
        "interval_coverage": interval_coverage,
        "risk_coverage_curves": risk_curves,
        "rank_cells": {
            design_id: {
                "total_fits": len(records),
                "r_less_k_fits": sum(
                    0 < record["row"]["r"] < record["row"]["k"] for record in records
                ),
                "r_equals_k_fits": sum(
                    record["row"]["r"] == record["row"]["k"] > 0 for record in records
                ),
                "rank_zero_fits": sum(record["row"]["k"] == 0 for record in records),
            }
            for design_id, records in data["records"].items()
        },
        "baselines": (
            "DIRECT_MEASURE and KNOWN_EVENT_SCORE reported independently; "
            "baseline losses are not accepted model risk"
        ),
        "distribution_shift_included": False,
        "online_ssvc": "NOT_CERTIFIED",
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "TEST_ANALYSIS.json", report)
    _write_csv(out / "METRICS.csv", metrics)
    _write_csv(out / "SEED_METRICS.csv", seed_metrics)
    _write_csv(out / "PAIRED_COMPARISONS.csv", paired)
    atomic_json(out / "FORMAL_TESTS.json", formal)
    _write_csv(out / "INTERVAL_COVERAGE.csv", interval_coverage)
    _write_csv(out / "RISK_COVERAGE.csv", risk_curves)
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
