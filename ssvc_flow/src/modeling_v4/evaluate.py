"""Evaluation after prediction: explicit scopes, reference noise and seed clusters.

Input is an iterator of method/query blocks (P x 4), or single-prompt rows (4).
Exact quantiles retain numeric residual blocks, not millions of expanded Python
records. Call per design for bounded memory; core_sink streams query summaries.
No geometry threshold changes prediction availability. Reference-derived risk
curves are evaluation output, not selection callbacks or certified intervals.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EVENTS = ("X", "S", "W", "I")
SCOPES = ("all", "nonalias", "alias", "predictable", "unknown", "reference_unresolved")
DIAGNOSTICS = ("rho", "effective_weight_residual", "score_residual", "fisher_residual")
RESOLUTIONS = (0.00025, 0.001, 0.005)


def _matrix(value, count=None):
    a = np.asarray(value, dtype=float)
    if a.shape == (4,):
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 4 or not len(a) or (count is not None and len(a) != count):
        raise ValueError("Expected aligned prompt x X/S/W/I array")
    return a.copy()


def _mask(value, shape):
    a = np.asarray(value)
    if a.dtype.kind != "b":
        raise ValueError("Availability and identity masks must be explicit booleans")
    if a.shape == (shape[0],):
        a = a[:, None]
    try:
        return np.broadcast_to(a, shape).copy()
    except ValueError as exc:
        raise ValueError("Availability mask shape differs") from exc


def _bootstrap_sums(sums, counts, seed_order, indices):
    s = np.array([sums.get(seed, 0.0) for seed in seed_order])
    n = np.array([counts.get(seed, 0) for seed in seed_order])
    denominator = n[indices].sum(1)
    values = np.divide(
        s[indices].sum(1), denominator, out=np.full(len(indices), np.nan), where=denominator > 0
    )
    return values[np.isfinite(values)]


def seed_cluster_bootstrap(values, seed_ids, *, repetitions=2000, random_seed=0):
    """Resample training seeds; retain every arm/origin/query/prompt within seed.

    Estimator is the pooled cell mean after cluster resampling. No p-value or
    distribution-free coverage is claimed. One/two-seed data remain descriptive.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(seed_ids)
    if (
        v.ndim != 1
        or s.shape != v.shape
        or not len(v)
        or not np.isfinite(v).all()
        or s.dtype.kind not in "iu"
        or repetitions != 2000
    ):
        raise ValueError(
            "Finite one-dimensional values, integer seed IDs and 2000 repetitions required"
        )
    order = sorted(set(s.tolist()))
    result = {
        "unit": "training_seed",
        "independent_seed_count": len(order),
        "repetitions": repetitions,
        "point_estimate": float(v.mean()),
        "interval": None,
        "certified_95": False,
        "status": "DESCRIPTIVE_TOO_FEW_SEEDS" if len(order) < 3 else "EMPIRICAL_CLUSTER_BOOTSTRAP",
        "estimator": "POOLED_CELLS_AFTER_CLUSTER_RESAMPLING",
    }
    if len(order) >= 3:
        indices = np.random.default_rng(random_seed).integers(
            len(order), size=(repetitions, len(order))
        )
        draws = _bootstrap_sums(
            {k: float(v[s == k].sum()) for k in order},
            {k: int(np.sum(s == k)) for k in order},
            order,
            indices,
        )
        result["interval"] = np.quantile(draws, [0.025, 0.975]).tolist()
    return result


def _metric(parts):
    err, truth, se, valid, independent = [
        np.concatenate([p[k] for p in parts])
        for k in ("error", "truth", "se", "valid", "independent")
    ]
    all_count, count = len(err), int(valid.sum())
    e, t = err[valid], truth[valid]
    complete = count == all_count and count > 0
    mse = float(np.mean(e * e)) if count else None
    mae = float(np.mean(abs(e))) if count else None
    signal = float(np.mean(t * t)) if count else None
    observed = np.isfinite(err) & np.isfinite(truth)
    oe = err[observed]
    se_available = bool(count and np.isfinite(se[valid]).all())
    refvar = float(np.mean(se[valid] ** 2)) if se_available else None
    corrected = mse - refvar if se_available and independent[valid].all() else None
    row = {
        "all_count": all_count,
        "all_finite_observed_count": int(observed.sum()),
        "all_finite_observed_mse": float(np.mean(oe**2)) if len(oe) else None,
        "all_finite_observed_mae": float(np.mean(abs(oe))) if len(oe) else None,
        "all_finite_observed_q95_absolute": float(np.quantile(abs(oe), 0.95)) if len(oe) else None,
        "all_finite_observed_signed_bias": float(oe.mean()) if len(oe) else None,
        "all_finite_observed_reference_variance_corrected_mse": float(
            np.mean(oe**2) - np.mean(se[observed] ** 2)
        )
        if len(oe) and np.isfinite(se[observed]).all() and independent[observed].all()
        else None,
        "all_finite_observed_scope": "FINITE_POINT_ESTIMATES_INCLUDING_UNRESOLVED_REFERENCE",
        "all_finite_reference_se_rms": float(np.sqrt(np.mean(se[observed] ** 2)))
        if observed.any() and np.isfinite(se[observed]).all()
        else None,
        "normal_approximation_reference_full_width_rms": float(
            3.92 * np.sqrt(np.mean(se[observed] ** 2))
        )
        if observed.any() and np.isfinite(se[observed]).all()
        else None,
        "reference_width_is_coverage_guarantee": False,
        "finite_resolved_count": count,
        "unresolved_count": all_count - count,
        "status": "AVAILABLE" if complete else "EMPTY" if not all_count else "UNRESOLVED",
        "mse": mse if complete else None,
        "mae": mae if complete else None,
        "rmse": float(np.sqrt(mse)) if complete else None,
        "finite_subset_mse": mse,
        "finite_subset_mae": mae,
        "quantile_scope": "ALL_CASES" if complete else "FINITE_RESOLVED_SUBSET_ONLY",
        "q95_absolute": float(np.quantile(abs(e), 0.95)) if count else None,
        "signed_bias": float(e.mean()) if complete else None,
        "signal_rms": float(np.sqrt(signal)) if count else None,
        "signal_scope": "ALL_CASES" if complete else "FINITE_RESOLVED_SUBSET_ONLY",
        "nrmse": float(np.sqrt(mse / signal)) if complete and signal > 0 else None,
        "nrmse_status": "DEFINED"
        if complete and signal > 0
        else "ZERO_SIGNAL"
        if complete
        else "UNRESOLVED",
        "reference_se_rms": float(np.sqrt(refvar)) if se_available else None,
        "reference_se_scope": "FINITE_RESOLVED_SUBSET" if not complete else "ALL_CASES",
        "reference_variance_corrected_mse": corrected if complete else None,
        "finite_subset_reference_variance_corrected_mse": corrected,
        "variance_correction_status": "SECONDARY_INDEPENDENT_REFERENCE"
        if corrected is not None
        else "UNAVAILABLE_OR_DEPENDENT_REFERENCE",
        "variance_correction_clipped": False,
        "resolution_fraction": {
            str(tol): float(np.mean(abs(e) <= tol)) if count else None for tol in RESOLUTIONS
        },
        "quantiles_are_confidence_intervals": False,
    }
    return row


def _risk_curves(entries, metadata):
    result = []
    total, nonalias = len(entries), sum(not r["alias"] for r in entries)
    for name in DIAGNOSTICS:
        observed = [r for r in entries if r["diagnostics"].get(name) is not None]
        observed.sort(key=lambda r: r["diagnostics"][name])
        sse = refvar = cells = selected = predictable = na_selected = unresolved = 0
        na_sse = na_refvar = na_cells = 0
        for i, row in enumerate(observed):
            selected += 1
            predictable += int(row["predictable"])
            na_selected += int(not row["alias"] and row["predictable"])
            unresolved += int(not row["resolved"])
            sse += row["error_ss"]
            cells += row["finite_cells"]
            refvar += row["reference_variance_sum"]
            if not row["alias"]:
                na_sse += row["error_ss"]
                na_refvar += row["reference_variance_sum"]
                na_cells += row["finite_cells"]
            value = row["diagnostics"][name]
            if i + 1 < len(observed) and observed[i + 1]["diagnostics"][name] == value:
                continue
            result.append(
                {
                    **metadata,
                    "diagnostic": name,
                    "threshold": value,
                    "selected_queries": selected,
                    "all_queries": total,
                    "predictable_selected_queries": predictable,
                    "coverage": predictable / total,
                    "nonalias_queries": nonalias,
                    "nonalias_coverage": na_selected / nonalias if nonalias else None,
                    "reference_unresolved_selected_queries": unresolved,
                    "diagnostic_missing_queries": total - len(observed),
                    "finite_resolved_cells": cells,
                    "finite_subset_mse": sse / cells if cells else None,
                    "nonalias_finite_resolved_cells": na_cells,
                    "nonalias_finite_subset_mse": na_sse / na_cells if na_cells else None,
                    "nonalias_reference_se_rms": float(np.sqrt(na_refvar / na_cells))
                    if na_cells and np.isfinite(na_refvar)
                    else None,
                    "reference_se_rms": float(np.sqrt(refvar / cells))
                    if cells and np.isfinite(refvar)
                    else None,
                    "fixed_rho_gate_applied": False,
                    "selection_scope": "EVALUATION_CURVE_NO_THRESHOLD_SELECTED",
                }
            )
    return result


def evaluate_records(records, *, bootstrap_reps=2000, bootstrap_seed=0, core_sink=None):
    """Summarize supplied prediction records, retaining all explicit denominators.

    Required identity: method, seed, role, origin_id, bank_id, target, prompt_ids,
    groups. Required data: prediction/status, reference/kind/SE/resolved, is_alias.
    For exact reference SE must be zero. Noisy-reference variance subtraction is
    only available with reference_independent=True. Group reference uncertainty
    requires explicit group_reference_se or reference_prompt_independent=True.
    """
    if bootstrap_reps != 2000:
        raise ValueError("V4 freezes 2000 shared seed-cluster bootstrap repetitions")
    pools, risks = defaultdict(list), defaultdict(list)
    grouped, query_risk = {}, {}
    metadata, seen, reference_identity, core, seeds = {}, set(), {}, [], set()
    for source in records:
        row = dict(source)
        for name in ("method", "role", "origin_id", "bank_id", "target"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"Missing evaluation identity {name}")
        seed = row.get("seed")
        if type(seed) is not int:
            raise ValueError("Training seed must be an integer")
        seeds.add(seed)
        ref = _matrix(row["reference"])
        n = len(ref)
        prompts = row.get("prompt_ids", [row["prompt_id"]] if "prompt_id" in row else None)
        groups = row.get("groups", [row["semantic_group"]] if "semantic_group" in row else None)
        if (
            not isinstance(prompts, (list, tuple))
            or not isinstance(groups, (list, tuple))
            or len(prompts) != n
            or len(groups) != n
            or len(set(prompts)) != n
            or any(not isinstance(v, str) or not v for v in [*prompts, *groups])
        ):
            raise ValueError("Each prompt needs a unique identity and semantic group")
        status = row.get("prediction_status")
        if status not in ("PREDICTED", "UNKNOWN"):
            raise ValueError("Explicit PREDICTED or UNKNOWN status required")
        known = np.full(ref.shape, status == "PREDICTED", dtype=bool)
        if status == "UNKNOWN":
            if row.get("prediction") is not None:
                raise ValueError("UNKNOWN must not be replaced with a numeric prediction")
            pred = np.full(ref.shape, np.nan)
        else:
            pred = _matrix(row["prediction"], n)
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite predictions must be explicitly UNKNOWN")
        resolved = _mask(row["reference_resolved"], ref.shape)
        if not np.isfinite(ref[resolved]).all():
            raise ValueError("Resolved reference contains missing/nonfinite values")
        se = (
            np.full(ref.shape, float(row["reference_se"]))
            if np.isscalar(row["reference_se"])
            else _matrix(row["reference_se"], n)
        )
        if np.any(se[np.isfinite(se)] < 0):
            raise ValueError("Reference SE cannot be negative")
        kind = row["reference_kind"]
        if kind not in ("EXACT", "MONTE_CARLO") or (kind == "EXACT" and not np.all(se == 0)):
            raise ValueError("Reference kind and actual uncertainty must agree")
        if not np.isfinite(se[resolved]).all():
            raise ValueError("Resolved reference must carry finite actual SE")
        if (
            kind == "MONTE_CARLO"
            and row.get("reference_estimator") == "CROSSFIT_COV_ZERO_SUM"
            and row.get("reference_se_method")
            not in ("INDEPENDENT_PACKET_REPEATS", "FULL_PROCEDURE_BOOTSTRAP")
        ):
            raise ValueError(
                "Crossfit reference SE must refit the full procedure or use independent packets"
            )
        independent = kind == "EXACT" or row.get("reference_independent") is True
        alias = _mask(row["is_alias"], (n, 1))[:, 0]
        if alias.any() and not np.all(ref[alias & resolved.all(1)] == 0):
            raise ValueError("Exact policy alias contradicts resolved nonzero reference")
        meta = {k: row.get(k) for k in ("method", "design_id", "role", "target", "n")}
        meta["information_level"] = row.get("information_level", "UNSPECIFIED")
        meta["evaluation_unit"] = row.get("evaluation_unit", "prompt")
        if meta["evaluation_unit"] not in ("prompt", "semantic_group"):
            raise ValueError("Explicit prompt or semantic_group evaluation unit required")
        experiment = tuple(meta.values())
        metadata[experiment] = meta
        identity = tuple(
            row.get(k)
            for k in (
                "seed",
                "origin_id",
                "bank_id",
                "target",
                "repeat",
                "candidate_id",
                "baseline_id",
            )
        )
        for j, prompt in enumerate(prompts):
            cell_id = (experiment, identity, prompt)
            if cell_id in seen:
                raise ValueError("Duplicate method/query/prompt evaluation row")
            seen.add(cell_id)
            refkey = (row["role"], identity, prompt)
            fingerprint = (
                ref[j].tobytes(),
                se[j].tobytes(),
                resolved[j].tobytes(),
                bool(alias[j]),
                groups[j],
                kind,
            )
            if refkey in reference_identity and reference_identity[refkey] != fingerprint:
                raise ValueError(
                    "Methods must share the same query reference, alias and group identity"
                )
            reference_identity[refkey] = fingerprint
        error, valid = pred - ref, known & resolved
        masks = {
            "all": np.ones(ref.shape, bool),
            "nonalias": np.broadcast_to(~alias[:, None], ref.shape),
            "alias": np.broadcast_to(alias[:, None], ref.shape),
            "predictable": known,
            "unknown": ~known,
            "reference_unresolved": ~resolved,
        }

        def append(
            channel,
            values,
            truth,
            noise,
            available,
            scopes,
            experiment=experiment,
            seed=seed,
            independent=independent,
        ):
            for scope, mask in scopes.items():
                pools[experiment, channel, scope].append(
                    {
                        "seed": seed,
                        "error": values[mask].ravel(),
                        "truth": truth[mask].ravel(),
                        "se": noise[mask].ravel(),
                        "valid": available[mask].ravel(),
                        "independent": np.full(int(mask.sum()), independent, dtype=bool),
                    }
                )

        append("RAW4", error, ref, se, valid, masks)
        for j, name in enumerate(EVENTS):
            append(
                name,
                error[:, j],
                ref[:, j],
                se[:, j],
                valid[:, j],
                {s: m[:, j] for s, m in masks.items()},
            )
        group_records = []
        for group in sorted(set(groups)):
            indices = np.array([g == group for g in groups])
            for channel, j, sign in (("group_pX", 0, 1), ("group_v", 3, -1)):
                gv = bool(valid[indices, j].all())
                ge = sign * error[indices, j].mean()
                if kind == "EXACT" or row.get("reference_prompt_independent") is True:
                    gs = np.sqrt(np.sum(se[indices, j] ** 2)) / indices.sum()
                else:
                    gs = row.get("group_reference_se", {}).get(group, {}).get(channel, np.nan)
                if np.isfinite(gs) and gs < 0:
                    raise ValueError("Group reference SE cannot be negative")
                ga = bool(alias[indices].all())
                group_key = (experiment, identity, group, channel)
                if group_key not in grouped:
                    grouped[group_key] = {
                        "seed": seed,
                        "count": 0,
                        "error_sum": 0.0,
                        "truth_sum": 0.0,
                        "variance_sum": 0.0,
                        "predictable": True,
                        "resolved": True,
                        "alias": ga,
                        "independent": True,
                        "prompt_independent": True,
                        "explicit_se": None,
                    }
                aggregate = grouped[group_key]
                if aggregate["alias"] != ga:
                    raise ValueError("Group policy alias identity changes between prompt blocks")
                aggregate["count"] += int(indices.sum())
                aggregate["error_sum"] += float(sign * error[indices, j].sum())
                aggregate["truth_sum"] += float(sign * ref[indices, j].sum())
                aggregate["variance_sum"] += float(np.sum(se[indices, j] ** 2))
                aggregate["predictable"] &= status == "PREDICTED"
                aggregate["resolved"] &= bool(resolved[indices, j].all())
                aggregate["independent"] &= independent
                aggregate["prompt_independent"] &= (
                    kind == "EXACT" or row.get("reference_prompt_independent") is True
                )
                explicit = row.get("group_reference_se", {}).get(group, {}).get(channel)
                if explicit is not None:
                    if (
                        not np.isfinite(explicit)
                        or explicit < 0
                        or (
                            aggregate["explicit_se"] is not None
                            and aggregate["explicit_se"] != explicit
                        )
                    ):
                        raise ValueError("Explicit whole-group reference uncertainty differs")
                    aggregate["explicit_se"] = float(explicit)
                group_records.append(
                    {
                        "group": group,
                        "channel": channel,
                        "prompt_count": int(indices.sum()),
                        "residual": float(ge) if gv else None,
                        "reference_se": float(gs) if np.isfinite(gs) else None,
                    }
                )
        diagnostic = {}
        for name, value in row.get("diagnostics", {}).items():
            if name in DIAGNOSTICS:
                if value is not None and (
                    not np.isscalar(value) or not np.isfinite(value) or value < 0
                ):
                    raise ValueError(
                        "Geometry diagnostics must be finite nonnegative scalars or null"
                    )
                diagnostic[name] = float(value) if value is not None else None
        if alias.any() and not alias.all():
            raise ValueError("Policy identity alias status must be shared across query prompts")
        risk_key = experiment, identity
        if risk_key not in query_risk:
            query_risk[risk_key] = {
                "diagnostics": diagnostic,
                "alias": bool(alias.all()),
                "predictable": True,
                "resolved": True,
                "finite_cells": 0,
                "error_ss": 0.0,
                "reference_variance_sum": 0.0,
            }
        item = query_risk[risk_key]
        if item["diagnostics"] != diagnostic or item["alias"] != bool(alias.all()):
            raise ValueError("Query geometry or alias identity differs across prompt blocks")
        item["predictable"] &= status == "PREDICTED"
        item["resolved"] &= bool(resolved.all())
        item["finite_cells"] += int(valid.sum())
        item["error_ss"] += float(np.sum(error[valid] ** 2))
        item["reference_variance_sum"] += float(np.sum(se[valid] ** 2))
        query = {
            **meta,
            **{
                k: row.get(k)
                for k in (
                    "seed",
                    "origin_id",
                    "arm",
                    "step",
                    "bank_id",
                    "repeat",
                    "candidate_id",
                    "baseline_id",
                )
            },
            "prompt_count": n,
            "prompt_ids": list(prompts),
            "is_alias": bool(alias.all()),
            "prediction_status": status,
            "reference_kind": kind,
            "reference_resolved_cells": int(resolved.sum()),
            "reference_independent": independent,
            "diagnostics": diagnostic,
            "groups": group_records,
            "group_metrics_scope": "THIS_PROMPT_BLOCK; summary combines all query blocks",
            "event_residuals": [
                [float(v) if ok else None for v, ok in zip(e, mask, strict=True)]
                for e, mask in zip(error, valid, strict=True)
            ],
            "all_finite_observed_event_residuals": [
                [float(v) if np.isfinite(v) else None for v in r] for r in error
            ],
            "reference_se_rms": float(np.sqrt(np.mean(se**2))) if np.isfinite(se).all() else None,
            "cost": row.get("cost"),
            "model_metadata": row.get("model_metadata"),
            "prediction_record_id": row.get("prediction_record_id"),
            "reference_record_id": row.get("reference_record_id"),
            "four_event_mse": float(np.mean(error**2)) if valid.all() else None,
            "four_event_mse_definition": "MEAN_NOT_SUM",
            "raw_mass_residual_max": float(abs(pred.sum(1)).max())
            if status == "PREDICTED"
            else None,
        }
        if core_sink is None:
            core.append(query)
        else:
            core_sink(query)
    if not seen:
        raise ValueError("No actual prediction records supplied")
    for (experiment, _, _, channel), group in grouped.items():
        valid = group["predictable"] and group["resolved"]
        se = group["explicit_se"]
        if se is None:
            se = (
                np.sqrt(group["variance_sum"]) / group["count"]
                if group["prompt_independent"]
                else np.nan
            )
        masks = {
            "all": True,
            "alias": group["alias"],
            "nonalias": not group["alias"],
            "predictable": group["predictable"],
            "unknown": not group["predictable"],
            "reference_unresolved": not group["resolved"],
        }
        for scope, include in masks.items():
            pools[experiment, channel, scope].append(
                {
                    "seed": group["seed"],
                    "error": np.array([group["error_sum"] / group["count"]] if include else []),
                    "truth": np.array([group["truth_sum"] / group["count"]] if include else []),
                    "se": np.array([se] if include else []),
                    "valid": np.array([valid] if include else [], dtype=bool),
                    "independent": np.array([group["independent"]] if include else [], dtype=bool),
                }
            )
    for (experiment, _), item in query_risk.items():
        risks[experiment].append(item)
    order = sorted(seeds)
    draw_cache = {}
    summary, bootstrap, curves = [], [], []
    for (experiment, channel, scope), parts in pools.items():
        metrics = _metric(parts)
        summary.append({**metadata[experiment], "channel": channel, "scope": scope, **metrics})
        sums, counts = defaultdict(float), defaultdict(int)
        for p in parts:
            valid = p["valid"]
            sums[p["seed"]] += float(np.sum(p["error"][valid] ** 2))
            counts[p["seed"]] += int(valid.sum())
        contributing = sum(c > 0 for c in counts.values())
        seed_order = tuple(sorted(counts))
        if seed_order not in draw_cache:
            seed_key = hashlib.sha256(json.dumps([bootstrap_seed, seed_order]).encode()).digest()
            draw_cache[seed_order] = np.random.default_rng(
                int.from_bytes(seed_key[:8], "big")
            ).integers(len(seed_order), size=(bootstrap_reps, len(seed_order)))
        draws = draw_cache[seed_order]
        interval = None
        if contributing >= 3:
            distribution = _bootstrap_sums(sums, counts, seed_order, draws)
            interval = (
                np.quantile(distribution, [0.025, 0.975]).tolist() if len(distribution) else None
            )
        bootstrap.append(
            {
                **metadata[experiment],
                "channel": channel,
                "scope": scope,
                "metric": "mse",
                "statistic_scope": metrics["quantile_scope"],
                "independent_seed_count": contributing,
                "interval": interval,
                "status": "EMPIRICAL_CLUSTER_BOOTSTRAP"
                if interval is not None
                else "DESCRIPTIVE_TOO_FEW_SEEDS",
                "unit": "training_seed",
                "repetitions": bootstrap_reps,
                "certified_95": False,
                "seed_order": list(seed_order),
                "bootstrap_draw_identity": hashlib.sha256(draws.tobytes()).hexdigest(),
                "point_estimate": metrics["finite_subset_mse"],
            }
        )
    for experiment, entries in risks.items():
        curves.extend(_risk_curves(entries, metadata[experiment]))
    return {
        "core_results": core,
        "summary": summary,
        "risk_coverage": curves,
        "bootstrap": bootstrap,
        "bootstrap_seed_order": order,
        "bootstrap_draw_identities": {
            str(k): hashlib.sha256(v.tobytes()).hexdigest() for k, v in draw_cache.items()
        },
        "bootstrap_indices_shared_across_models": True,
        "selection_modified": False,
        "online_ssvc": "NOT_CERTIFIED",
        "reference_read_scope": "EVALUATION_ONLY",
        "fixed_rho_gate_applied": False,
        "meaningful_effect_scales": list(RESOLUTIONS),
    }


def write_evaluation(records, out, *, binding, bootstrap_reps=2000, bootstrap_seed=0):
    """Publish one immutable derived evaluation; completion binds every output.

    Core CSV is streamed per query block. Raw samples/predictions remain owned by
    the acquisition tables; only residuals and their existing IDs are written.
    """
    from ..modeling_v3.io import atomic_json, finalize_run

    if not isinstance(binding, dict) or not binding:
        raise ValueError("Explicit source/config/input binding required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    temporary = out / ".CORE_RESULTS.csv.partial"

    def cell(value):
        return (
            json.dumps(value, allow_nan=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
        )

    with temporary.open("x", newline="") as stream:
        writer = None

        def sink(row):
            nonlocal writer
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow({k: cell(v) for k, v in row.items()})

        result = evaluate_records(
            records, bootstrap_reps=bootstrap_reps, bootstrap_seed=bootstrap_seed, core_sink=sink
        )
    temporary.rename(out / "CORE_RESULTS.csv")
    curve_file = out / "RISK_COVERAGE.csv"
    with curve_file.open("x", newline="") as stream:
        if result["risk_coverage"]:
            writer = csv.DictWriter(stream, fieldnames=list(result["risk_coverage"][0]))
            writer.writeheader()
            writer.writerows(result["risk_coverage"])
        else:
            stream.write("diagnostic,threshold,selection_scope\n")
    atomic_json(
        out / "METRICS.json",
        {
            **result,
            "kind": "V4_EVALUATION",
            "binding": binding,
            "core_results_file": "CORE_RESULTS.csv",
            "risk_coverage_file": "RISK_COVERAGE.csv",
        },
    )
    finalize_run(out, {**binding, "kind": "V4_EVALUATION"})
    return {
        "status": "COMPLETE",
        "out": str(out),
        "metrics": str(out / "METRICS.json"),
        "core_results": str(out / "CORE_RESULTS.csv"),
    }
