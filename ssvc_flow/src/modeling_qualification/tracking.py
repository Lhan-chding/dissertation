"""Fixed-window offline tracking with frozen fits and seed-calibrated envelopes.

The predictor receives paid anchor/fit panels and realized updates only. Dense
trajectory probabilities and Jacobians enter only after prediction serialization.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation import (
    _anchors,
    _dp,
    _dy,
    _entries,
    _groups,
    _manifest,
    _oracle,
    _oracle_j,
    _view,
    aggregate_scores,
    cluster_bootstrap,
    file_hash,
    passing,
    project,
    score_predictions,
    write_csv,
)
from .io import write_json
from .math_contracts import to_helmert
from .models import array_hash, build_subspace, fit_local_model, ridge_map


def fixed_window_prediction(p0, updates, U, C):
    """Keep U/C fixed and return every intermediate Helmert prediction.

    T2 computes projections and path quantities; T3 adds a full cumulative
    displacement. Both have exactly the same point predictions.
    """
    p0, d, u, c = [np.asarray(x, dtype=np.float64) for x in (p0, updates, U, C)]
    if (
        p0.ndim != 2
        or p0.shape[-1] != 3
        or d.ndim != 2
        or u.ndim != 2
        or c.shape != (p0.size, u.shape[1])
        or u.shape[0] != d.shape[1]
        or not all(np.isfinite(x).all() for x in (p0, d, u, c))
    ):
        raise ValueError("finite aligned anchor, updates, basis and response arrays required")
    if not np.allclose(u.T @ u, np.eye(u.shape[1]), rtol=1e-10, atol=1e-10):
        raise ValueError("tracking basis must be orthonormal")
    begin = time.perf_counter()
    projected = d @ u
    per_step_norm = np.linalg.norm(d, axis=1)
    residual_norm = np.sqrt(np.maximum(0.0, per_step_norm**2 - np.sum(projected**2, axis=1)))
    cumulative_x = np.cumsum(projected, axis=0)
    prediction = p0 + (cumulative_x @ c.T).reshape(len(d), *p0.shape)
    telemetry = {
        "update_norm": per_step_norm,
        "path_length": np.cumsum(per_step_norm),
        "projected_path_length": np.cumsum(np.linalg.norm(projected, axis=1)),
        "orthogonal_path_length": np.cumsum(residual_norm),
        "projected_net_displacement": np.linalg.norm(cumulative_x, axis=1),
        "T2_seconds": time.perf_counter() - begin,
    }
    begin = time.perf_counter()
    cumulative_d = np.cumsum(d, axis=0)
    net = np.linalg.norm(cumulative_d, axis=1)
    telemetry["net_displacement"] = net
    telemetry["orthogonal_net_displacement"] = np.sqrt(
        np.maximum(0.0, net**2 - np.sum(cumulative_x**2, axis=1))
    )
    telemetry["T3_additional_seconds"] = time.perf_counter() - begin
    return prediction, telemetry


def empirical_envelope(seed_windows: dict, quantile=0.9):
    """One maximum per training seed, across arms/anchors/noise/groups/steps.

    No rows, prompts or noise replicas are treated as independent seed samples.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between zero and one")
    maxima = {}
    for seed, values in seed_windows.items():
        values = np.asarray(values, dtype=float)
        if not values.size or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("each seed requires finite nonnegative complete-window maxima")
        maxima[str(seed)] = float(values.max())
    return {
        "label": "EMPIRICAL_CALIBRATED_ENVELOPE",
        "width": float(np.quantile(list(maxima.values()), quantile, method="higher"))
        if maxima
        else None,
        "empirical_quantile": quantile,
        "quantile_method": "higher",
        "calibration_clusters": len(maxima),
        "per_seed_complete_window_maximum": maxima,
        "status": "CALIBRATED" if maxima else "UNKNOWN",
        "calibration_unit": "seed maximum over complete 16-step anchor windows and six groups",
        "claim_distribution_free_95_percent": False,
    }


def cost_amortization(fit_banks, horizon, panel_cost=1.0, telemetry_cost=0.0, refresh_cost=1.0):
    """Compare 3 candidate panels per bank plus refresh against dense evaluation."""
    if (
        isinstance(fit_banks, bool)
        or int(fit_banks) != fit_banks
        or fit_banks < 0
        or horizon < 0
        or any(not np.isfinite(v) or v < 0 for v in (panel_cost, telemetry_cost, refresh_cost))
    ):
        raise ValueError("nonnegative finite cost inputs required")
    calibration = 3 * int(fit_banks) * panel_cost
    difference = panel_cost - telemetry_cost
    break_even = math.ceil((calibration + refresh_cost) / difference) if difference > 0 else None
    model = calibration + horizon * telemetry_cost + refresh_cost
    dense = horizon * panel_cost
    return {
        "fit_banks": int(fit_banks),
        "calibration_panels": 3 * int(fit_banks),
        "tested_horizon": horizon,
        "panel_cost": panel_cost,
        "telemetry_cost_per_step": telemetry_cost,
        "refresh_cost": refresh_cost,
        "calibration_cost": calibration,
        "model_cost_at_horizon": model,
        "dense_cost_at_horizon": dense,
        "break_even_horizon": break_even,
        "status": (
            "COST_FEASIBLE_AT_TESTED_HORIZON"
            if break_even is not None and horizon >= break_even
            else "NOT_COST_FEASIBLE_AT_TESTED_HORIZON"
        ),
        "unit": "abstract behavior evaluation panel; real Qwen seconds unmeasured",
    }


def _spec_key(spec):
    return f"{spec['method']}|{spec['rank_cap']}|{spec['n']}"


def _specifications(config, selection, n, noise):
    frozen, selected = selection["frozen_grid"], selection["selected"]
    specs = [
        frozen["B0_PERSISTENCE|0"],
        frozen["B6_FULL_Q_RIDGE|FULL"],
        selected,
        {"method": "NORM_ONLY_LOCAL_RIDGE", "rank_cap": 1, "alpha": selected["alpha"]},
    ]
    exact_selection = selection.get("exact_selection", {})
    if exact_selection.get("rank_cap") is not None:
        specs.append(
            {
                "method": "B5_EXACT_SELECTED_DIAGNOSTIC",
                "rank_cap": exact_selection["rank_cap"],
                "alpha": exact_selection["alpha"],
            }
        )
    # A bounded r/H heatmap; exact and primary n64 replica 0 only. No test tuning.
    if noise == 0 and n in (None, 64):
        specs += [v for v in frozen.values() if v["method"] == selected["method"]]
    result = {}
    for value in specs:
        value = {**value, "n": "exact" if n is None else n, "noise": noise}
        result[_spec_key(value)] = value
    return list(result.values())


def _make_prediction(fit, updates, spec):
    started = time.perf_counter()
    if spec["method"] == "B0_PERSISTENCE":
        # The response panels and Q are not needed to predict persistence.
        # Q/path arrays below serve only post-freeze evaluator diagnostics.
        q, meta = build_subspace(fit["d"])
        u, c = q[:, :0], np.zeros((fit["p0"].shape[0] * 3, 0))
        _, telemetry = fixed_window_prediction(to_helmert(fit["p0"]), updates, u, c)
        telemetry["B0_evaluator_diagnostic_seconds"] = time.perf_counter() - started
        telemetry["T2_seconds"] = telemetry["T3_additional_seconds"] = 0.0
        delta = np.zeros((len(updates), *fit["p0"].shape))
        arrays = {
            "raw": fit["p0"] + delta,
            "delta": delta,
            "anchor": fit["p0"],
            "U": u,
            "C": c,
            "Q": q,
        }
        arrays.update({k: v for k, v in telemetry.items() if isinstance(v, np.ndarray)})
        return arrays, {**meta, "r": 0}, telemetry, 0.0
    y = _dy(fit["p0"], fit["p1"])
    if spec["method"] == "NORM_ONLY_LOCAL_RIDGE":
        norms = np.linalg.norm(fit["d"], axis=1)[:, None]
        c = ridge_map(norms, y, spec["alpha"])
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        per_step = np.linalg.norm(updates, axis=1)
        path = np.cumsum(per_step)
        delta = _dp(path[:, None] @ c.T)
        norm_seconds = time.perf_counter() - started
        # Q is retained for post-freeze evaluator diagnostics, never T1 inputs.
        q, meta = build_subspace(fit["d"])
        model = {**meta, "Q": q, "U": q[:, :0], "C": c, "r": 1}
        telemetry = {
            "update_norm": per_step,
            "path_length": path,
            "T1_seconds": norm_seconds,
            "T2_seconds": 0.0,
            "T3_additional_seconds": 0.0,
        }
    else:
        model = fit_local_model(fit["d"], y, spec["method"], spec["rank_cap"], spec["alpha"])
        fit_seconds = time.perf_counter() - started
        predicted_y, telemetry = fixed_window_prediction(
            to_helmert(fit["p0"]), updates, model["U"], model["C"]
        )
        delta = _dp((predicted_y - to_helmert(fit["p0"])).reshape(len(updates), -1))
    raw = fit["p0"] + delta
    arrays = {
        "raw": raw,
        "delta": delta,
        "anchor": fit["p0"],
        "U": model["U"],
        "C": model["C"],
        "Q": model["Q"],
    }
    arrays.update({key: value for key, value in telemetry.items() if isinstance(value, np.ndarray)})
    return arrays, model, telemetry, fit_seconds


def _error_arrays(raw, delta, truth, truth0):
    projected = project(raw)
    # Correct for the same observed anchor: projection changes the response too.
    observed_anchor = raw[0] - delta[0]
    true_delta = truth - truth0
    return {
        "raw_level": raw - truth,
        "raw_delta": delta - true_delta,
        "projected_level": projected - truth,
        "projected_delta": (projected - observed_anchor) - true_delta,
    }


def _group_errors(error, groups):
    values = np.stack(
        [error[:, groups == g].mean(axis=1) for g in sorted(set(groups.tolist()))], axis=1
    )
    return np.abs(values[..., [0, 3]])


def step_tolerance_status(failed, uninformative):
    """Observable absolute failures take precedence over uninformative NRMSE."""
    if failed:
        return "EXCEEDS_MODEL_TOLERANCE"
    return "UNKNOWN" if uninformative else "WITHIN_MODEL_TOLERANCE"


def _failure_series(raw, delta, truth, truth0, groups, rules):
    errors = _group_errors(delta - (truth - truth0), groups)
    invalid = np.mean(np.any((raw < -1e-12) | (raw > 1 + 1e-12), axis=-1), axis=1)
    mae = np.mean(np.abs(delta - (truth - truth0)), axis=(1, 2))
    energy = np.sum((truth - truth0) ** 2, axis=(1, 2))
    error_ss = np.sum((delta - (truth - truth0)) ** 2, axis=(1, 2))
    nrmse_fail = (energy >= 1e-20) & (error_ss > rules["response_nrmse_max"] ** 2 * energy)
    return (
        nrmse_fail
        | (errors[..., 0].max(axis=1) > rules["group_delta_pX_abs_error_q95_max"])
        | (errors[..., 1].max(axis=1) > rules["group_delta_v_abs_error_q95_max"])
        | (mae > rules["per_prompt_event_mae_max"])
        | (invalid > rules["raw_invalid_simplex_fraction_max"])
    )


def _score_extended(raw, delta, truth, truth0, groups):
    score = score_predictions(raw, delta, truth, truth0, groups)
    projected_delta = project(raw) - (raw[0] - delta[0])
    extra = score_predictions(project(raw), projected_delta, truth, truth0, groups)
    score["projected_delta_mae"] = extra["delta_mae"]
    score["projected_delta_rmse"] = extra["delta_rmse"]
    score["projected_delta_error_ss"] = extra["delta_error_ss"]
    score["projected_response_nrmse"] = extra["response_nrmse"]
    return score


def _aggregate_extended(scores):
    result = aggregate_scores(scores)
    total = sum(s["sample_count"] for s in scores)
    result["projected_delta_mae"] = float(
        sum(s["projected_delta_mae"] * s["sample_count"] for s in scores) / total
    )
    result["projected_delta_rmse"] = float(
        np.sqrt(
            sum(s["projected_delta_error_ss"] for s in scores)
            / sum(s["element_count"] for s in scores)
        )
    )
    error = sum(s["projected_delta_error_ss"] for s in scores)
    result["projected_response_nrmse"] = (
        float(np.sqrt(error / result["delta_truth_ss"]))
        if result["delta_truth_ss"] >= 1e-20
        else None
    )
    return result


def run_track(config: dict, input: Path, models: Path, out: Path) -> dict:
    """Run M4 against frozen M3 choices without online detection or adaptation."""
    started = time.perf_counter()
    input, models, out = Path(input), Path(models), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest, groups = _manifest(input), _groups(input)
    anchors = _anchors(config, manifest)
    calibration, tests = (
        _entries(manifest, "interval_calibration"),
        _entries(manifest, "locked_test"),
    )
    if not calibration or not tests or not (models / "selection.json").is_file():
        summary = {
            "status": "INSUFFICIENT_SPLITS_FOR_FORMAL_M4",
            "calibration_trajectories": len(calibration),
            "test_trajectories": len(tests),
            "selection_present": (models / "selection.json").is_file(),
            "formal_coverage_measured": False,
        }
        write_json(out / "summary.json", summary)
        return summary
    selection = json.loads((models / "selection.json").read_text())
    selection_sha = file_hash(models / "selection.json")
    if selection.get("input_manifest_sha256") != file_hash(input / "manifest.json"):
        raise ValueError("frozen selection was fitted on another collection")
    measurements = [(None, 0)] + [
        (n, noise)
        for n in config["measurement"]["samples_per_prompt_grid"]
        for noise in range(config["measurement"]["noise_replicas"])
    ]
    horizons = config["tracking"]["horizons"]
    max_horizon = max(horizons)
    fit_budget = len(manifest["bank_roles"]["fit"])
    pred_dir = out / "frozen_predictions"
    pred_dir.mkdir()
    subspace_dir = out / "frozen_update_subspaces"
    subspace_dir.mkdir()
    subspaces = {}
    records, timings = [], defaultdict(float)
    # Serialize every calibration and test prediction using only observed views.
    # In particular no test OracleView is opened in this stage.
    for split, entries in (("interval_calibration", calibration), ("locked_test", tests)):
        for entry in entries:
            for n, noise in measurements:
                io_start = time.perf_counter()
                view = _view(input, entry, n, noise)
                timings["observed_view_load_seconds"] += time.perf_counter() - io_start
                for ai, step in enumerate(anchors):
                    updates = view.updates[step : step + max_horizon]
                    if len(updates) == 0:
                        continue
                    fit = view.anchor(ai, role="fit")
                    for spec in _specifications(config, selection, n, noise):
                        arrays, model, telemetry, seconds = _make_prediction(fit, updates, spec)
                        timings["fit_seconds"] += seconds
                        for key in (
                            "T1_seconds",
                            "T2_seconds",
                            "T3_additional_seconds",
                            "B0_evaluator_diagnostic_seconds",
                        ):
                            timings[key] += telemetry.get(key, 0.0)
                        record = len(records)
                        path = pred_dir / f"{record:06d}.npz"
                        io_start = time.perf_counter()
                        subspace_key = (entry["id"], ai)
                        if subspace_key not in subspaces:
                            qpath = subspace_dir / f"{entry['id']}_anchor{step}.npz"
                            np.savez_compressed(qpath, Q=arrays["Q"])
                            subspaces[subspace_key] = (
                                str(qpath.relative_to(out)),
                                file_hash(qpath),
                            )
                        stored = {k: v for k, v in arrays.items() if k not in ("raw", "Q")}
                        basis_is_Q = spec["method"] == "B6_FULL_Q_RIDGE"
                        if basis_is_Q:
                            stored.pop("U")
                        # raw = anchor + delta is reconstructible bit-for-bit; store
                        # its independent hash while avoiding duplicate large arrays.
                        np.savez_compressed(path, **stored)
                        timings["prediction_write_seconds"] += time.perf_counter() - io_start
                        records.append(
                            {
                                "record": record,
                                "split": split,
                                "trajectory_id": entry["id"],
                                "seed": entry["seed"],
                                "arm": entry["arm"],
                                "anchor": step,
                                "anchor_index": ai,
                                "method": spec["method"],
                                "rank_cap": spec["rank_cap"],
                                "alpha": spec["alpha"],
                                "n": spec["n"],
                                "noise": noise,
                                "r": model["r"],
                                "k": model["k"],
                                "fit_budget": fit_budget,
                                "representation": "per_prompt_helmert",
                                "max_available_H": len(updates),
                                "prediction_file": str(path.relative_to(out)),
                                "prediction_file_sha256": file_hash(path),
                                "Q_file": subspaces[subspace_key][0],
                                "Q_file_sha256": subspaces[subspace_key][1],
                                "U_equals_Q": basis_is_Q,
                                "raw_encoding": "anchor_plus_delta_float64",
                                "raw_prediction_hash": array_hash(arrays["raw"]),
                                "delta_prediction_hash": array_hash(arrays["delta"]),
                                "fit_updates_hash": array_hash(fit["d"]),
                                "fit_probabilities_hash": array_hash(fit["p1"]),
                                "anchor_probability_hash": array_hash(fit["p0"]),
                                "input_updates_hash": array_hash(updates),
                                "anchor_measurement_id": fit["anchor_measurement_id"],
                                "selection_sha256": selection_sha,
                                "fit_seconds": seconds,
                                "telemetry_seconds": {
                                    k: v for k, v in telemetry.items() if k.endswith("_seconds")
                                },
                                "basis_fixed_in_window": True,
                                "test_oracle_used_for_prediction": False,
                            }
                        )
    write_json(
        out / "prediction_freeze.json",
        {
            "selection_sha256": selection_sha,
            "input_manifest_sha256": file_hash(input / "manifest.json"),
            "frozen_before_locked_test_scoring": True,
            "records": records,
        },
    )
    freeze_sha = file_hash(out / "prediction_freeze.json")

    # Ten calibration seeds define four separate max-error envelopes per config.
    calibration_maxima = defaultdict(lambda: defaultdict(list))
    oracle_cache = {}
    for rec in records:
        if rec["split"] != "interval_calibration" or rec["max_available_H"] < 16:
            continue
        tid = rec["trajectory_id"]
        if tid not in oracle_cache:
            oracle_cache[tid] = _oracle(input, next(e for e in calibration if e["id"] == tid))
        oracle = oracle_cache[tid]
        step = rec["anchor"]
        with np.load(out / rec["prediction_file"]) as data:
            delta = data["delta"][:16]
            raw = data["anchor"] + delta
        for kind, error in _error_arrays(
            raw, delta, oracle.p[step + 1 : step + 17], oracle.p[step]
        ).items():
            calibration_maxima[(_spec_key(rec), kind)][rec["seed"]].append(
                float(_group_errors(error, groups).max())
            )
    envelopes = {
        f"{key[0]}|{key[1]}": empirical_envelope(values, config["tracking"]["empirical_quantile"])
        for key, values in calibration_maxima.items()
    }
    write_json(
        out / "envelope_freeze.json",
        {
            "envelopes": envelopes,
            "prediction_freeze_sha256": freeze_sha,
            "frozen_before_locked_test_scoring": True,
            "calibration_rule": (
                "maximum within complete 16-step anchor x six groups x pX/v, then "
                "maximum over each seed before higher q.9"
            ),
            "test_tuning": False,
            "claim_distribution_free_95_percent": False,
        },
    )
    envelope_sha = file_hash(out / "envelope_freeze.json")
    oracle_cache.clear()

    # All dense test labels/Jacobians first enter here, after both freezes.
    window_rows, example_rows, decomposition_rows = [], [], []
    exact_selected_examples = []
    buckets, seed_buckets, diagnostic_buckets = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    rank_grid_buckets = defaultdict(list)
    coverage = defaultdict(lambda: defaultdict(list))
    cache_tid, oracle, jacobians = None, None, {}
    scoring_started = time.perf_counter()
    for rec in records:
        if rec["split"] != "locked_test":
            continue
        tid, step = rec["trajectory_id"], rec["anchor"]
        if tid != cache_tid:
            cache_tid = tid
            oracle = _oracle(input, next(e for e in tests if e["id"] == tid))
            jacobians = {}
        if step not in jacobians:
            before = time.perf_counter()
            jacobians[step] = _oracle_j(input, oracle, step)
            timings["oracle_jacobian_diagnostic_seconds"] += time.perf_counter() - before
        j = jacobians[step]
        path = out / rec["prediction_file"]
        if file_hash(path) != rec["prediction_file_sha256"]:
            raise RuntimeError("frozen M4 prediction file changed")
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        delta = arrays["delta"]
        raw = arrays["anchor"] + delta
        if file_hash(out / rec["Q_file"]) != rec["Q_file_sha256"]:
            raise RuntimeError("frozen update subspace changed")
        with np.load(out / rec["Q_file"]) as data:
            arrays["Q"] = data["Q"]
        if rec["U_equals_Q"]:
            arrays["U"] = arrays["Q"]
        if (
            array_hash(raw) != rec["raw_prediction_hash"]
            or array_hash(delta) != rec["delta_prediction_hash"]
        ):
            raise RuntimeError("frozen M4 prediction array changed")
        length = len(raw)
        truth0, truth = oracle.p[step], oracle.p[step + 1 : step + length + 1]
        displacement = oracle.theta[step + 1 : step + length + 1] - oracle.theta[step]
        full_delta = _dp(displacement @ j.T)
        q, u = arrays["Q"], arrays["U"]
        norm = np.linalg.norm(displacement, axis=1)
        qres = np.linalg.norm(displacement - (displacement @ q) @ q.T, axis=1)
        ures = np.linalg.norm(displacement - (displacement @ u) @ u.T, axis=1)
        qratio = np.divide(qres, norm, out=np.zeros_like(norm), where=norm > 1e-14)
        uratio = np.divide(ures, norm, out=np.zeros_like(norm), where=norm > 1e-14)
        response_unknown = np.sum((truth - truth0) ** 2, axis=(1, 2)) < 1e-20
        fail = _failure_series(raw, delta, truth, truth0, groups, config["model_selection"])
        curvature_fail = _failure_series(
            truth0 + full_delta, full_delta, truth, truth0, groups, config["model_selection"]
        )
        rare = (truth[..., :3].sum(axis=-1) <= 0.05) | (truth[..., 1:3].sum(axis=-1) <= 0.05)
        observed = arrays["anchor"]
        unobserved_cond = (observed[:, :3].sum(axis=-1) == 0) | (observed[:, 1:3].sum(axis=-1) == 0)
        all_errors = _error_arrays(raw, delta, truth, truth0)
        group_error = {kind: _group_errors(err, groups) for kind, err in all_errors.items()}
        kinds = list(group_error)
        widths = {
            kind: envelopes.get(f"{_spec_key(rec)}|{kind}", {}).get("width") for kind in kinds
        }
        telemetries = (
            ["NORM_ONLY"]
            if rec["method"] == "NORM_ONLY_LOCAL_RIDGE"
            else ["PROJECTION_AND_PATH", "PROJECTION_AND_NET_DISPLACEMENT"]
        )
        ident = {
            k: rec[k]
            for k in (
                "seed",
                "arm",
                "anchor",
                "method",
                "rank_cap",
                "alpha",
                "n",
                "noise",
                "r",
                "k",
                "fit_budget",
                "representation",
            )
        }
        # Unique intermediate steps are retained; H membership is elapsed_step <= H.
        for elapsed in range(1, length + 1):
            for telemetry in telemetries:
                row = {
                    **ident,
                    "telemetry": telemetry,
                    "elapsed_step": elapsed,
                    "included_in_horizons": [h for h in horizons if elapsed <= h <= length],
                    "tolerance_status": step_tolerance_status(
                        fail[elapsed - 1], response_unknown[elapsed - 1]
                    ),
                    "subspace_Q_exit_ratio": float(qratio[elapsed - 1]),
                    "subspace_U_exit_ratio": float(uratio[elapsed - 1]),
                    "oracle_full_J_exceeds_tolerance": bool(curvature_fail[elapsed - 1]),
                    "rare_condition_prompt_fraction": float(rare[elapsed - 1].mean()),
                    "unobserved_anchor_condition_fraction": float(unobserved_cond.mean()),
                    "path_length": float(arrays["path_length"][elapsed - 1]),
                }
                if telemetry == "PROJECTION_AND_PATH":
                    for key in ("projected_path_length", "orthogonal_path_length"):
                        row[key] = float(arrays[key][elapsed - 1])
                elif telemetry == "PROJECTION_AND_NET_DISPLACEMENT":
                    for key in (
                        "projected_path_length",
                        "orthogonal_path_length",
                        "net_displacement",
                        "projected_net_displacement",
                        "orthogonal_net_displacement",
                    ):
                        row[key] = float(arrays[key][elapsed - 1])
                for kind in kinds:
                    row[f"{kind}_mae"] = float(np.abs(all_errors[kind][elapsed - 1]).mean())
                    row[f"{kind}_worst_group_error"] = float(group_error[kind][elapsed - 1].max())
                    row[f"{kind}_envelope_width"] = widths[kind]
                    row[f"{kind}_envelope_covered"] = (
                        bool(group_error[kind][elapsed - 1].max() <= widths[kind])
                        if widths[kind] is not None
                        else None
                    )
                window_rows.append(row)
        for horizon in horizons:
            if horizon > length:
                continue
            score = _score_extended(raw[:horizon], delta[:horizon], truth[:horizon], truth0, groups)
            for telemetry in telemetries:
                key = (rec["method"], rec["rank_cap"], rec["n"], telemetry, horizon)
                buckets[key].append(score)
                if (
                    rec["method"] == selection["selected"]["method"]
                    and rec["noise"] == 0
                    and rec["n"] in ("exact", 64)
                ):
                    rank_grid_buckets[key].append(score)
                seed_buckets[(*key, rec["seed"])].append(score)
                diagnostic_buckets[key].append(
                    {
                        "r": rec["r"],
                        "k": rec["k"],
                        "failure_rate": float(fail[:horizon].mean()),
                        "uninformative_response_step_fraction": float(
                            response_unknown[:horizon].mean()
                        ),
                        "subspace_exit_failure_rate": float(
                            (fail[:horizon] & (qratio[:horizon] > 0.2)).mean()
                        ),
                        "subspace_U_exit_failure_rate": float(
                            (fail[:horizon] & (uratio[:horizon] > 0.2)).mean()
                        ),
                        "rare_event_failure_rate": float(
                            (fail[:horizon] & rare[:horizon].any(axis=1)).mean()
                        ),
                        "unobserved_condition_failure_rate": float(
                            (fail[:horizon] & unobserved_cond.any()).mean()
                        ),
                        "curvature_failure_rate": float(curvature_fail[:horizon].mean()),
                        "curvature_joint_failure_rate": float(
                            (fail[:horizon] & curvature_fail[:horizon]).mean()
                        ),
                        "Q_exit_gt020_fraction": float((qratio[:horizon] > 0.2).mean()),
                        "U_exit_gt020_fraction": float((uratio[:horizon] > 0.2).mean()),
                    }
                )
                for kind in kinds:
                    if widths[kind] is not None:
                        coverage[(key, kind)][rec["seed"]].append(
                            bool(group_error[kind][:horizon].max() <= widths[kind])
                        )
        # Save vector decomposition for primary exact trajectory windows only.
        if rec["n"] == "exact" and rec["method"] in (
            selection["selected"]["method"],
            "B6_FULL_Q_RIDGE",
        ):
            ju = _dp(((displacement @ u) @ u.T) @ j.T)
            omitted, fitted, nonlinear = ju - full_delta, delta - ju, full_delta - (truth - truth0)
            decomp_dir = out / "oracle_decomposition"
            decomp_dir.mkdir(exist_ok=True)
            dpath = decomp_dir / f"{rec['record']:06d}.npz"
            np.savez_compressed(
                dpath,
                direction_omission=omitted,
                fit_residual=fitted,
                finite_step_nonlinearity=nonlinear,
                total_residual=delta - (truth - truth0),
            )
            for elapsed in range(1, length + 1):
                norms = [
                    float(np.linalg.norm(a[elapsed - 1])) for a in (omitted, fitted, nonlinear)
                ]
                total = float(np.linalg.norm((delta - (truth - truth0))[elapsed - 1]))
                decomposition_rows.append(
                    {
                        **ident,
                        "elapsed_step": elapsed,
                        "direction_omission_norm": norms[0],
                        "fit_residual_norm": norms[1],
                        "finite_step_nonlinearity_norm": norms[2],
                        "total_residual_norm": total,
                        "triangle_upper_bound": sum(norms),
                        "triangle_bound_holds": total <= sum(norms) + 1e-12,
                        "vectors_file": str(dpath.relative_to(out)),
                        "jacobian_fixed_at_anchor": True,
                    }
                )
        if (
            rec["seed"] == 301
            and rec["arm"] == "X_BASE"
            and step == 8
            and rec["noise"] == 0
            and (
                (
                    rec["n"] == 64
                    and rec["method"] == selection["selected"]["method"]
                    and rec["rank_cap"] == selection["selected"]["rank_cap"]
                )
                or rec["method"] == "B5_EXACT_SELECTED_DIAGNOSTIC"
            )
        ):
            for elapsed in range(length):
                for group in sorted(set(groups.tolist())):
                    mask = groups == group
                    for event, col in (("pX", 0), ("v", 3)):
                        actual = float(truth[elapsed, mask, col].mean())
                        prediction = float(raw[elapsed, mask, col].mean())
                        if event == "v":
                            actual, prediction = 1 - actual, 1 - prediction
                        width = widths["raw_level"]
                        destination = (
                            exact_selected_examples
                            if rec["method"] == "B5_EXACT_SELECTED_DIAGNOSTIC"
                            else example_rows
                        )
                        destination.append(
                            {
                                "method": rec["method"],
                                "rank_cap": rec["rank_cap"],
                                "n": rec["n"],
                                "selection_source": "exact model-selection paid observations"
                                if rec["method"] == "B5_EXACT_SELECTED_DIAGNOSTIC"
                                else "n64 model selection",
                                "seed": 301,
                                "arm": "X_BASE",
                                "anchor": 8,
                                "elapsed_step": elapsed + 1,
                                "group": int(group),
                                "event": event,
                                "truth": actual,
                                "raw_prediction": prediction,
                                "envelope_lower": prediction - width if width is not None else None,
                                "envelope_upper": prediction + width if width is not None else None,
                                "envelope_label": "EMPIRICAL_CALIBRATED_ENVELOPE",
                            }
                        )
    timings["locked_test_scoring_seconds"] = time.perf_counter() - scoring_started
    summary_rows, per_seed = [], []
    key_fields = ("method", "rank_cap", "n", "telemetry", "H")
    for key, scores in buckets.items():
        rec = {**dict(zip(key_fields, key, strict=True)), **_aggregate_extended(scores)}
        rec.update(fit_budget=fit_budget, representation="per_prompt_helmert")
        rec["meets_tolerance"] = passing(rec, config["model_selection"])
        rec["tolerance_status"] = (
            "WITHIN_MODEL_TOLERANCE" if rec["meets_tolerance"] else "EXCEEDS_MODEL_TOLERANCE"
        )
        for field in diagnostic_buckets[key][0]:
            values = [r[field] for r in diagnostic_buckets[key]]
            rec[field + ("_mean" if field in ("r", "k") else "")] = float(np.mean(values))
        for kind in ("raw_level", "raw_delta", "projected_level", "projected_delta"):
            seed_coverage = coverage[(key, kind)]
            rec[f"{kind}_seed_cluster_coverage"] = (
                float(np.mean([all(v) for v in seed_coverage.values()])) if seed_coverage else None
            )
            rec[f"{kind}_window_coverage"] = (
                float(np.mean([x for v in seed_coverage.values() for x in v]))
                if seed_coverage
                else None
            )
            rec[f"{kind}_envelope_width"] = envelopes.get(
                f"{key[0]}|{key[1]}|{key[2]}|{kind}", {}
            ).get("width")
        summary_rows.append(rec)
    for key, scores in seed_buckets.items():
        per_seed.append(
            {**dict(zip((*key_fields, "seed"), key, strict=True)), **_aggregate_extended(scores)}
        )
    largest = {}
    for rec in summary_rows:
        key = f"{rec['method']}|{rec['rank_cap']}|{rec['n']}|{rec['telemetry']}"
        largest[key] = max(largest.get(key, 0), rec["H"] if rec["meets_tolerance"] else 0)
    for rec in summary_rows:
        key = f"{rec['method']}|{rec['rank_cap']}|{rec['n']}|{rec['telemetry']}"
        rec["largest_tested_H_meeting_tolerance"] = largest[key] or None
    paired = []
    for key in buckets:
        _method, _cap, n, telemetry, horizon = key
        a = {
            r["seed"]: r["delta_mae"]
            for r in per_seed
            if all(r[k] == v for k, v in zip(key_fields, key, strict=True))
        }
        baseline_telemetry = "PROJECTION_AND_PATH" if telemetry == "NORM_ONLY" else telemetry
        b = {
            r["seed"]: r["delta_mae"]
            for r in per_seed
            if r["method"] == "B0_PERSISTENCE"
            and r["n"] == n
            and r["telemetry"] == baseline_telemetry
            and r["H"] == horizon
        }
        paired.append(
            {
                **dict(zip(key_fields, key, strict=True)),
                **cluster_bootstrap(
                    a,
                    b,
                    config["statistics"]["bootstrap_replicates"],
                    config["statistics"]["bootstrap_seed"],
                ),
            }
        )
    write_csv(out / "tracking_windows.csv", window_rows)
    write_csv(out / "tracking_summary.csv", summary_rows)
    write_csv(
        out / "tracking_rank_horizon_grid.csv",
        [
            {
                **dict(zip(key_fields, key, strict=True)),
                "noise": 0,
                "fit_budget": fit_budget,
                "representation": "per_prompt_helmert",
                **_aggregate_extended(scores),
            }
            for key, scores in rank_grid_buckets.items()
        ],
    )
    write_csv(out / "tracking_by_seed.csv", per_seed)
    write_csv(out / "example_timeseries.csv", example_rows)
    write_csv(out / "example_exact_selected_timeseries.csv", exact_selected_examples)
    write_csv(out / "tracking_error_decomposition.csv", decomposition_rows)
    write_json(out / "paired_cluster_bootstrap.json", paired)
    largest_actually_tested = max((r["H"] for r in summary_rows), default=0)
    costs = [
        cost_amortization(b, largest_actually_tested)
        for b in config["response"]["fit_budget_ablation"]
    ]
    collection_summary = json.loads((input / "summary.json").read_text())
    panel_seconds = (
        collection_summary.get("exact_seconds", 0.0) / collection_summary["exact_panel_calls"]
        if collection_summary.get("exact_panel_calls")
        else None
    )
    branch_count = (
        len(manifest["trajectories"])
        * len(anchors)
        * sum(len(b) for b in manifest["bank_roles"].values())
        * 3
    )
    branch_seconds = (
        collection_summary.get("branch_seconds", 0.0) / branch_count if branch_count else None
    )
    measured_costs = []
    for row in summary_rows:
        matches = [
            r
            for r in records
            if r["split"] == "locked_test"
            and all(r[k] == row[k] for k in ("method", "rank_cap", "n"))
        ]
        fit_cpu = float(np.mean([r["fit_seconds"] for r in matches]))
        telemetry_key = "T1_seconds" if row["telemetry"] == "NORM_ONLY" else "T2_seconds"
        telemetry_cpu = float(
            np.mean(
                [
                    (
                        r["telemetry_seconds"].get(telemetry_key, 0.0)
                        + (
                            r["telemetry_seconds"].get("T3_additional_seconds", 0.0)
                            if row["telemetry"] == "PROJECTION_AND_NET_DISPLACEMENT"
                            else 0.0
                        )
                    )
                    / r["max_available_H"]
                    for r in matches
                ]
            )
        )
        required_panels = 0 if row["method"] == "B0_PERSISTENCE" else fit_budget * 3
        calibration_cpu = (
            required_panels * (panel_seconds + branch_seconds) + fit_cpu
            if panel_seconds is not None and branch_seconds is not None
            else None
        )
        model_cpu = (
            calibration_cpu + row["H"] * telemetry_cpu + panel_seconds
            if calibration_cpu is not None
            else None
        )
        dense_cpu = row["H"] * panel_seconds if panel_seconds is not None else None
        denominator = panel_seconds - telemetry_cpu if panel_seconds is not None else None
        break_even = (
            math.ceil((calibration_cpu + panel_seconds) / denominator)
            if denominator is not None and denominator > 0
            else None
        )
        measured_costs.append(
            {
                **{k: row[k] for k in ("method", "rank_cap", "n", "telemetry", "H", "fit_budget")},
                "mean_fit_cpu_seconds": fit_cpu,
                "mean_telemetry_cpu_seconds_per_step": telemetry_cpu,
                "mean_toy_exact_panel_cpu_seconds": panel_seconds,
                "mean_local_branch_cpu_seconds": branch_seconds,
                "calibration_cpu_seconds": calibration_cpu,
                "calibration_panels_required": required_panels,
                "model_cpu_seconds_at_H": model_cpu,
                "dense_cpu_seconds_at_H": dense_cpu,
                "cpu_break_even_horizon": break_even,
                "scope": (
                    "measured mean toy exact panel and branch allocation; count "
                    "sampling and serialization excluded"
                ),
                "count_sampling_per_configuration_seconds": None,
                "cpu_cost_is_lower_bound_from_measured_components": True,
                "cpu_cost_status": (
                    "NOT_COST_FEASIBLE_AT_TESTED_HORIZON"
                    if break_even is None or break_even > row["H"]
                    else "MEASURED_COMPONENTS_BELOW_DENSE_COST_TOTAL_UNKNOWN"
                ),
            }
        )
    write_csv(out / "tracking_cpu_cost.csv", measured_costs)
    write_json(
        out / "cost_audit.json",
        {
            "abstract_panel_costs": costs,
            "calibrated_model_costs_do_not_apply_to_B0": True,
            "B0_abstract_cost": cost_amortization(0, largest_actually_tested),
            "B0_Q_and_path_construction_is_evaluator_diagnostic_only": True,
            "B5_rank_zero_retains_preregistered_calibration_budget": True,
            "measured_cpu_components_seconds": dict(timings),
            "measured_cpu_comparison_file": "tracking_cpu_cost.csv",
            "measured_cpu_comparison_scope": (
                "mean exact toy panel/branch timings allocated per config; counts "
                "sampling/IO omitted, no real-model inference"
            ),
            "collection_summary_sha256": file_hash(input / "summary.json"),
            "T2_storage_scalars": (
                "P*r basis + 216*r response + r cumulative projected displacement"
            ),
            "T3_additional_storage_bytes_float64": manifest["parameter_count"] * 8,
            "T3_additional_work": "one cumulative P-vector addition and norm per realized step",
            "real_model_seconds": None,
            "real_model_cost_status": "NOT_MEASURED",
            "toy_multinomial_cost_is_not_qwen_generation_cost": True,
            "fit_budget_ablation_accuracy_source": (
                "M3 independent bank evaluation; M4 tracks main fit budget only"
            ),
            "zero_telemetry_abstract_cost_is_optimistic_lower_bound": True,
            "cpu_cost_total_status": "INCOMPLETE_COST_ACCOUNTING_COUNTS_AND_IO_UNALLOCATED",
            "cpu_break_even_is_optimistic_lower_bound": True,
        },
    )
    summary = {
        "status": "COMPLETED",
        "selected": selection["selected"],
        "selection_sha256": selection_sha,
        "prediction_freeze_sha256": freeze_sha,
        "envelope_freeze_sha256": envelope_sha,
        "test_seed_count": len(set(e["seed"] for e in tests)),
        "calibration_seed_count": len(set(e["seed"] for e in calibration)),
        "prediction_blocks": len(records),
        "intermediate_rows": len(window_rows),
        "aggregate_metrics": summary_rows,
        "largest_tested_H_by_configuration": largest,
        "main_cost_status": cost_amortization(fit_budget, largest_actually_tested)["status"],
        "abstract_cost_audit": costs,
        "measured_cpu_components_seconds": dict(timings),
        "walltime_seconds": time.perf_counter() - started,
        "T2_T3_point_predictions_identical_by_construction": True,
        "empirical_coverage_status": "EMPIRICAL_CALIBRATED_ENVELOPE"
        if envelopes
        else "UNKNOWN_NO_COMPLETE_16_STEP_CALIBRATION",
        "online_detector_implemented": False,
        "online_ssvc_started": False,
        "safety_certified": False,
        "qualification_status_is_not_safety_certification": True,
        "failure_rate_definitions": {
            "subspace_exit_failure_rate": (
                "fraction of evaluated steps with tolerance failure AND "
                "cumulative Q residual ratio > .2"
            ),
            "rare_event_failure_rate": (
                "fraction of evaluated steps with tolerance failure AND any exact "
                "valid/error condition mass <= .05"
            ),
            "curvature_failure_rate": (
                "fraction of evaluated steps where fixed anchor full-J linear "
                "prediction fails probability tolerances"
            ),
            "seed_cluster_coverage": (
                "fraction of test seeds for which all arms/anchors/noise windows "
                "and groups are covered"
            ),
        },
        "bounded_grid": (
            "selected/B0/B6/norm-only: exact and n16/64/256 five noises; "
            "selected-method rank caps: exact and n64 noise0 only"
        ),
        "scope": (
            "fixed dataset/initialization; frozen observed local response; "
            "offline realized-update tracking only"
        ),
    }
    write_json(out / "summary.json", summary)
    return summary
