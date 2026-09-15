"""Split-frozen CPU response qualification and seed-clustered evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .models import (
    METHODS,
    array_hash,
    fit_local_model,
    fit_log_predictor,
    log_features,
    prepare_local,
)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("status\nNO_ROWS\n")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )


def freeze_predictions(path: Path, arrays: dict, inputs: dict) -> dict:
    """Persist raw outputs and their input identities before an evaluator sees labels."""
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path / "predictions.npz", **arrays)
    receipt = {
        "prediction_sha256": file_hash(path / "predictions.npz"),
        "inputs": inputs,
        "frozen_before_scoring": True,
        "array_hashes": {k: array_hash(v) for k, v in arrays.items()},
    }
    write_json(path / "freeze.json", receipt)
    return receipt


def project(p):
    from .math_contracts import project_simplex

    return project_simplex(p)


def score_predictions(
    raw: np.ndarray,
    delta: np.ndarray,
    truth: np.ndarray,
    truth0: np.ndarray,
    groups: np.ndarray,
    energy_floor=1e-20,
) -> dict:
    raw, delta, truth, truth0 = [np.asarray(x, dtype=float) for x in (raw, delta, truth, truth0)]
    if raw.shape != truth.shape or delta.shape != raw.shape or truth0.shape != raw.shape[1:]:
        raise ValueError("probability prediction / truth identity shape mismatch")
    if (
        raw.ndim != 3
        or raw.shape[-1] != 4
        or not all(np.all(np.isfinite(x)) for x in (raw, delta, truth, truth0))
    ):
        raise ValueError("finite [candidate,prompt,4] arrays required")
    groups = np.asarray(groups)
    if groups.shape != (raw.shape[1],):
        raise ValueError("group identity mismatch")
    true_delta = truth - truth0
    err, derr = raw - truth, delta - true_delta
    projected = project(raw)
    perr = projected - truth
    projected_delta = projected - (raw - delta)
    pderr = projected_delta - true_delta
    invalid = np.any((raw < -1e-12) | (raw > 1 + 1e-12), axis=-1) | (
        np.abs(raw.sum(-1) - 1) > 1e-10
    )
    energy = float(np.sum(true_delta**2))
    result = {
        "level_mae": float(np.mean(np.abs(err))),
        "level_rmse": float(np.sqrt(np.mean(err**2))),
        "projected_level_mae": float(np.mean(np.abs(perr))),
        "projected_level_rmse": float(np.sqrt(np.mean(perr**2))),
        "delta_mae": float(np.mean(np.abs(derr))),
        "delta_rmse": float(np.sqrt(np.mean(derr**2))),
        "projected_delta_mae": float(np.mean(np.abs(pderr))),
        "projected_delta_rmse": float(np.sqrt(np.mean(pderr**2))),
        "delta_error_ss": float(np.sum(derr**2)),
        "delta_truth_ss": energy,
        "response_nrmse": float(np.sqrt(np.sum(derr**2) / energy))
        if energy >= energy_floor
        else None,
        "response_scale_status": "INFORMATIVE"
        if energy >= energy_floor
        else "UNINFORMATIVE_RESPONSE_SCALE",
        "raw_invalid_fraction": float(np.mean(invalid)),
        "prompt_abs_error_q95": float(np.quantile(np.abs(err), 0.95)),
        "group_rows": [],
        "sample_count": len(raw),
        "element_count": int(raw.size),
        "level_error_ss": float(np.sum(err**2)),
        "projected_level_error_ss": float(np.sum(perr**2)),
        "projected_delta_error_ss": float(np.sum(pderr**2)),
        "negative_probability_mean_magnitude": float(np.mean(np.maximum(-raw, 0.0))),
    }
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        eg, dg = err[:, mask].mean(1), derr[:, mask].mean(1)
        pg = perr[:, mask].mean(1)
        gx, gv = np.abs(dg[:, 0]), np.abs(-dg[:, 3])
        row = {
            "group": int(g),
            "level_pX_mae": float(np.mean(np.abs(eg[:, 0]))),
            "level_v_mae": float(np.mean(np.abs(eg[:, 3]))),
            "projected_level_pX_mae": float(np.mean(np.abs(pg[:, 0]))),
            "projected_level_v_mae": float(np.mean(np.abs(pg[:, 3]))),
            "delta_pX_mae": float(gx.mean()),
            "delta_v_mae": float(gv.mean()),
            "delta_pX_q95": float(np.quantile(gx, 0.95)),
            "delta_v_q95": float(np.quantile(gv, 0.95)),
            "_gx": gx.tolist(),
            "_gv": gv.tolist(),
        }
        result["group_rows"].append(row)
    result["group_delta_pX_q95_max"] = max(row["delta_pX_q95"] for row in result["group_rows"])
    result["group_delta_v_q95_max"] = max(row["delta_v_q95"] for row in result["group_rows"])
    result["direction"] = []
    true_x, pred_x = true_delta[..., 0], delta[..., 0]
    for threshold in (0.002, 0.005, 0.01):
        mask = np.abs(true_x) >= threshold
        count = int(mask.sum())
        result["direction"].append(
            {
                "threshold": threshold,
                "eligible_prompt_branches": count,
                "eligible_branches": int(np.any(mask, axis=1).sum()),
                "correct": int(np.sum(np.sign(pred_x[mask]) == np.sign(true_x[mask]))),
                "accuracy": float(np.mean(np.sign(pred_x[mask]) == np.sign(true_x[mask])))
                if count
                else None,
            }
        )
    result["near_zero_prompt_fraction"] = float(np.mean(np.abs(true_x) < 0.002))
    return result


def cluster_bootstrap(method: dict, baseline: dict, replicates=5000, seed=99001) -> dict:
    keys = sorted(set(method) & set(baseline))
    if not keys:
        return {"status": "NO_PAIRED_CLUSTERS", "cluster_count": 0}
    values = np.array([method[k] - baseline[k] for k in keys], dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(keys), size=(replicates, len(keys)))
    means = values[draws].mean(1)
    return {
        "status": "EXPLORATORY_FIXED_DATASET_FIXED_INITIALIZATION",
        "cluster_count": len(keys),
        "bootstrap_replicates": replicates,
        "paired_mean": float(values.mean()),
        "paired_median": float(np.median(values)),
        "mean_ci025": float(np.quantile(means, 0.025)),
        "mean_ci975": float(np.quantile(means, 0.975)),
        "per_seed_difference": {str(k): float(v) for k, v in zip(keys, values, strict=False)},
    }


def passing(row: dict, rules: dict) -> bool:
    return (
        row.get("response_nrmse") is not None
        and row["group_delta_pX_q95_max"] <= rules.get("group_delta_pX_abs_error_q95_max", 0.01)
        and row["group_delta_v_q95_max"] <= rules.get("group_delta_v_abs_error_q95_max", 0.01)
        and row["delta_mae"] <= rules.get("per_prompt_event_mae_max", 0.02)
        and row["response_nrmse"] <= rules.get("response_nrmse_max", 0.75)
        and row["raw_invalid_fraction"] <= rules.get("raw_invalid_simplex_fraction_max", 0.05)
    )


def choose_smallest_rank(rows: list, rules: dict) -> dict:
    valid = [r for r in rows if passing(r, rules)]
    if not valid:
        return {"status": "NO_ACCEPTABLE_R", "rank_cap": None, "alpha": None}
    row = min(
        valid,
        key=lambda r: (
            100000 if r["rank_cap"] == "FULL" else int(r["rank_cap"]),
            r["group_delta_pX_q95_max"] + r["group_delta_v_q95_max"],
            r["alpha"],
        ),
    )
    return {**row, "status": "ACCEPTABLE_ON_SELECTION_ONLY"}


def aggregate_scores(scores: list) -> dict:
    if not scores:
        return {}
    total = sum(r["sample_count"] for r in scores)
    scalar = (
        "level_mae",
        "projected_level_mae",
        "delta_mae",
        "projected_delta_mae",
        "raw_invalid_fraction",
        "prompt_abs_error_q95",
        "near_zero_prompt_fraction",
    )
    result = {k: float(sum(r[k] * r["sample_count"] for r in scores) / total) for k in scalar}
    ess, tss = sum(r["delta_error_ss"] for r in scores), sum(r["delta_truth_ss"] for r in scores)
    result.update(
        delta_error_ss=ess,
        delta_truth_ss=tss,
        response_nrmse=float(np.sqrt(ess / tss)) if tss >= 1e-20 else None,
        sample_count=total,
    )
    for metric in ("level", "projected_level", "delta", "projected_delta"):
        ss = sum(row[f"{metric}_error_ss"] for row in scores)
        result[f"{metric}_rmse"] = float(np.sqrt(ss / sum(row["element_count"] for row in scores)))
    result["mean_block_prompt_abs_error_q95"] = result.pop("prompt_abs_error_q95")
    gs = defaultdict(lambda: [[], []])
    for score in scores:
        for row in score["group_rows"]:
            gs[row["group"]][0].extend(row["_gx"])
            gs[row["group"]][1].extend(row["_gv"])
    result["group_delta_pX_q95_max"] = max(float(np.quantile(v[0], 0.95)) for v in gs.values())
    result["group_delta_v_q95_max"] = max(float(np.quantile(v[1], 0.95)) for v in gs.values())
    return result


def _manifest(root):
    result = json.loads((root / "manifest.json").read_text())
    return result


def _groups(root):
    meta = json.loads((root / "probe_metadata.json").read_text())
    if isinstance(meta, dict):
        meta = meta.get("prompts", meta.get("probes", meta.get("rows")))
    return np.array(
        [x.get("group", x.get("group_index", x.get("group_id"))) for x in meta], dtype=int
    )


def _entries(manifest, role):
    return [x for x in manifest["trajectories"] if x.get("split", x.get("role")) == role]


def _anchors(config, manifest):
    profile = manifest.get("profile", "core")
    return config["profiles"][profile]["anchors"]


def _dy(p0, p1):
    from .math_contracts import to_helmert

    return (to_helmert(p1) - to_helmert(p0)).reshape(len(p1), -1)


def _dp(y):
    from .math_contracts import helmert

    return np.asarray(y).reshape(len(y), -1, 3) @ helmert().T


def _oracle_j(root, oracle, anchor_step):
    from .math_contracts import helmert
    from .toy import event_jacobian

    theta = oracle if isinstance(oracle, np.ndarray) else oracle.theta
    with np.load(root / "dataset.npz") as ds:
        j = event_jacobian(theta[anchor_step], ds["probe_features"], ds["probe_categories"])
    if hasattr(j, "detach"):
        j = j.detach().cpu().numpy()
    return np.einsum("pcd,ck->pkd", np.asarray(j), helmert()).reshape(-1, theta.shape[-1])


def _view(root, entry, n, noise):
    from .collection import load_observed

    return load_observed(root, entry["id"], n=n, noise=noise)


def _checkpoint(root, entry):
    from .collection import load_checkpoint_parameters

    return load_checkpoint_parameters(root, entry["id"])


def _oracle(root, entry):
    from .collection import load_oracle

    return load_oracle(root, entry["id"])


def _fit_log_globals(config, root, entries, anchors, groups, n, noise):
    xs, ys = [], []
    for entry in entries:
        view = _view(root, entry, n, noise)
        for ai, _ in enumerate(anchors):
            a = view.anchor(ai, role="fit")
            xs.append(log_features(a["p0"], a["logs"], a["d"], a["operation_indices"], groups))
            ys.append(_dy(a["p0"], a["p1"]))
    if not xs:
        return {}
    return {
        str(alpha): fit_log_predictor(np.concatenate(xs), np.concatenate(ys), alpha)
        for alpha in config["response"]["ridge_alpha_grid"]
    }


def _predict_one(
    fit,
    inputs,
    method,
    cap,
    alpha,
    global_log,
    groups,
    jacobian=None,
    fit_budget=None,
    representation="per_prompt",
    input_mode="actual_d",
):
    local_start = time.perf_counter()
    d, p1, logs = fit["d"], fit["p1"], fit["logs"]
    if fit_budget is not None:
        stop = fit_budget * 3
        d, p1, logs = d[:stop], p1[:stop], logs[:stop]
    y = _dy(fit["p0"], p1)
    d_eval = inputs["d"]
    if input_mode == "sgd_proxy":
        d = -0.01 * fit["g"][: len(d)]
        d_eval = -0.01 * inputs["g"]
    elif input_mode == "norm_only":
        d, d_eval = np.linalg.norm(d, axis=1)[:, None], np.linalg.norm(d_eval, axis=1)[:, None]
    if representation != "per_prompt":
        yy = y.reshape(len(y), len(groups), 3)
        y = (
            yy.mean(1)[:, None, :]
            if representation == "pooled"
            else np.stack([yy[:, groups == g].mean(1) for g in sorted(set(groups))], axis=1)
        ).reshape(len(y), -1)
    if method == "B1_LOG_RIDGE":
        x = log_features(
            fit["p0"], inputs["logs"], inputs["d"], inputs["operation_indices"], groups
        )
        global_model = global_log[str(alpha)]
        pred = (x / global_model["scale"]) @ global_model["coef"].T
        q, meta = prepare_local(d, y, alpha)["Q"], prepare_local(d, y, alpha)["meta"]
        fitted = {
            **meta,
            "r": 0,
            "Q": q,
            "U": q[:, :0],
            "C": np.empty((y.shape[1], 0)),
            "response_singular_values": [],
            "status": "GLOBAL_DEVELOPMENT_FIT_LOG_RIDGE",
        }
    else:
        fitted = fit_local_model(d, y, method, cap, alpha, jacobian=jacobian)
        fit_done = time.perf_counter()
        pred = fitted["predict"](d_eval)
        fitted["cpu_fit_seconds"] = fit_done - local_start
        fitted["cpu_inference_seconds"] = time.perf_counter() - fit_done
    fitted.setdefault("cpu_fit_seconds", 0.0)
    fitted.setdefault("cpu_inference_seconds", time.perf_counter() - local_start)
    if representation != "per_prompt":
        yy = pred.reshape(len(pred), -1, 3)
        pred = (
            np.broadcast_to(yy, (len(pred), len(groups), 3))
            if representation == "pooled"
            else yy[:, groups, :]
        ).reshape(len(pred), -1)
    delta = _dp(pred)
    raw = fit["p0"] + delta
    q = fitted["Q"]
    denom = np.linalg.norm(d_eval, axis=1)
    residual = np.linalg.norm(d_eval - (d_eval @ q) @ q.T, axis=1)
    ratio = np.divide(residual, denom, out=np.zeros_like(residual), where=denom > 1e-14)
    return raw, delta, fitted, ratio


def _specs(config):
    for method in config["response"]["models"]:
        caps = (
            [0]
            if method in ("B0_PERSISTENCE", "B1_LOG_RIDGE")
            else ["FULL"]
            if method in ("B6_FULL_Q_RIDGE", "O0_FULL_J_ORACLE")
            else config["response"]["rank_caps"]
        )
        for cap in caps:
            yield method, cap
    yield "B4_ENERGY95", "FULL"
    yield "B2_FULL_PARAMETER_SKETCH", 4


def _key(method, cap):
    return f"{method}|{cap}"


def run_fit_evaluate(config: dict, input: Path, out: Path) -> dict:
    """Execute M3. Selection and prediction freezes precede locked-test labels."""
    started = time.perf_counter()
    input, out = Path(input), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest, groups = _manifest(input), _groups(input)
    anchors = _anchors(config, manifest)
    development = _entries(manifest, "development_fit")
    selection_entries = _entries(manifest, "model_selection")
    tests = _entries(manifest, "locked_test")
    if not development or not selection_entries or not tests:
        summary = {
            "status": "INSUFFICIENT_SPLITS_FOR_FORMAL_M3",
            "development_trajectories": len(development),
            "selection_trajectories": len(selection_entries),
            "test_trajectories": len(tests),
            "selection_performed": False,
        }
        write_json(out / "summary.json", summary)
        return summary
    measurements = [(None, 0)] + [
        (64, noise) for noise in range(config["measurement"]["primary_noise_replicas"])
    ]
    global_start = time.perf_counter()
    global_logs = {
        (n, noise): _fit_log_globals(config, input, development, anchors, groups, n, noise)
        for n, noise in measurements
    }
    global_log_seconds = time.perf_counter() - global_start
    jacobian_seconds = 0.0
    selection_started = time.perf_counter()
    specs = list(_specs(config))
    selection_scores = defaultdict(list)
    jacobians = {}
    for entry in selection_entries:
        oracle = _oracle(input, entry)
        for ai, step in enumerate(anchors):
            truth0 = oracle.p[step]
            truth = oracle.anchor(ai, role="evaluation")["p1"]
            j_start = time.perf_counter()
            j = _oracle_j(input, oracle, step)
            jacobian_seconds += time.perf_counter() - j_start
            jacobians[(entry["id"], ai)] = j
            for n, noise in measurements:
                view = _view(input, entry, n, noise)
                fit, inputs = view.anchor(ai, role="fit"), view.anchor_inputs(ai, role="evaluation")
                for method, cap in specs:
                    alphas = (
                        [0.0]
                        if method in ("B0_PERSISTENCE", "O0_FULL_J_ORACLE", "O1_JQ_SVD_ORACLE")
                        else config["response"]["ridge_alpha_grid"]
                    )
                    for alpha in alphas:
                        raw, delta, _, _ = _predict_one(
                            fit, inputs, method, cap, alpha, global_logs[(n, noise)], groups, j
                        )
                        selection_scores[(method, str(cap), float(alpha), n)].append(
                            score_predictions(raw, delta, truth, truth0, groups)
                        )
    selection_fit_scoring_seconds = time.perf_counter() - selection_started
    selection_rows = []
    frozen = {}
    for method, cap in specs:
        candidates = []
        for key, values in selection_scores.items():
            if key[0] == method and key[1] == str(cap):
                row = {
                    "method": method,
                    "rank_cap": cap,
                    "alpha": key[2],
                    "n": "exact" if key[3] is None else key[3],
                    **aggregate_scores(values),
                }
                selection_rows.append(row)
                if key[3] == 64:
                    candidates.append(row)
        passing_candidates = [r for r in candidates if passing(r, config["model_selection"])]
        choice_pool = (
            passing_candidates
            if method == "B5_WORST_GROUP_RANK" and passing_candidates
            else candidates
        )
        chosen = min(
            choice_pool,
            key=lambda r: (
                max(r["group_delta_pX_q95_max"], r["group_delta_v_q95_max"]),
                r["delta_mae"],
                r["alpha"],
            ),
        )
        frozen[_key(method, cap)] = {"method": method, "rank_cap": cap, "alpha": chosen["alpha"]}
    risk_rows = [
        r
        for r in selection_rows
        if r["method"] == "B5_WORST_GROUP_RANK"
        and r["n"] == 64
        and r["alpha"] == frozen[_key(r["method"], r["rank_cap"])]["alpha"]
    ]
    selected = choose_smallest_rank(risk_rows, config["model_selection"])
    fallback = min(
        risk_rows,
        key=lambda r: (
            max(r["group_delta_pX_q95_max"], r["group_delta_v_q95_max"]),
            r["delta_mae"],
        ),
    )
    diagnostic = {
        "method": "B5_WORST_GROUP_RANK",
        "rank_cap": selected["rank_cap"]
        if selected["rank_cap"] is not None
        else fallback["rank_cap"],
        "alpha": selected["alpha"] if selected["alpha"] is not None else fallback["alpha"],
        "status": selected["status"],
        "qualified": selected["status"] != "NO_ACCEPTABLE_R",
    }
    exact_selection = choose_smallest_rank(
        [
            row
            for row in selection_rows
            if row["method"] == "B5_WORST_GROUP_RANK" and row["n"] == "exact"
        ],
        config["model_selection"],
    )
    selection = {
        "exact_selection": exact_selection,
        "exact_selection_test_method": "B5_EXACT_SELECTED_DIAGNOSTIC",
        "status": selected["status"],
        "selected": diagnostic,
        "frozen_grid": frozen,
        "selection_split": "model_selection",
        "test_tuning": False,
        "representation": "per_prompt_orthonormal_helmert",
        "random_basis_seed": 44001,
        "B1_features": [
            "six_group_anchor_helmert_18",
            "actual_update_norm",
            "grad_norm_preclip",
            "grad_norm_postclip",
            "optimizer_step",
            "train_X_fraction",
            "train_S_fraction",
            "train_W_fraction",
            "train_I_fraction",
            "zero_advantage_fraction",
            "no_x_mixed_fraction",
            "operation_one_hot_3",
        ],
        "alpha_selection_objective": "n64 pooled three-noise worst group delta q95, then delta MAE",
        "input_manifest_sha256": file_hash(input / "manifest.json"),
        "exact_grid_uses_same_n64_selected_alpha": True,
        "no_acceptable_rank_behavior": "Selection-best diagnostic retained as unqualified",
    }
    write_json(out / "selection.json", selection)
    selection_sha = file_hash(out / "selection.json")
    write_csv(out / "selection_grid.csv", selection_rows)
    # All test prediction identities are written before any test evaluation bank is opened.
    pred_dir = out / "frozen_predictions"
    pred_dir.mkdir()
    _identities, dimensions, _pending = [], [], []
    fit_seconds = 0.0
    inference_seconds = 0.0
    prediction_io_seconds = 0.0
    robustness = [
        (n, noise)
        for n in config["measurement"]["samples_per_prompt_grid"]
        for noise in range(config["measurement"]["noise_replicas"])
    ]
    robust_specs = [
        frozen[_key("B0_PERSISTENCE", 0)],
        frozen[_key("B6_FULL_Q_RIDGE", "FULL")],
        diagnostic,
    ]
    if exact_selection["rank_cap"] is not None:
        robust_specs.append(
            {
                "method": "B5_EXACT_SELECTED_DIAGNOSTIC",
                "rank_cap": exact_selection["rank_cap"],
                "alpha": exact_selection["alpha"],
                "selection_source": "EXACT_MODEL_SELECTION_ONLY",
            }
        )
    for n, noise in robustness:
        if (n, noise) not in global_logs:
            # Robustness excludes B1 and needs no additional global response labels.
            global_logs[(n, noise)] = {}
    records = []
    for entry in tests:
        oracle_j_only = _checkpoint(input, entry)
        for ai, step in enumerate(anchors):
            j_start = time.perf_counter()
            j = _oracle_j(input, oracle_j_only, step)
            jacobian_seconds += time.perf_counter() - j_start
            for n, noise in list(dict.fromkeys(measurements + robustness)):
                view = _view(input, entry, n, noise)
                fit, inputs = view.anchor(ai, role="fit"), view.anchor_inputs(ai, role="evaluation")
                current = list(frozen.values()) if (n, noise) in measurements else []
                if n is None and exact_selection["rank_cap"] is not None:
                    current.append(
                        {
                            "method": "B5_EXACT_SELECTED_DIAGNOSTIC",
                            "rank_cap": exact_selection["rank_cap"],
                            "alpha": exact_selection["alpha"],
                        }
                    )
                for spec in robust_specs:
                    if (n, noise) in robustness and not any(
                        _key(x["method"], x["rank_cap"]) == _key(spec["method"], spec["rank_cap"])
                        for x in current
                    ):
                        current.append(spec)
                for spec in current:
                    method, cap, alpha = spec["method"], spec["rank_cap"], spec["alpha"]
                    raw, delta, model, ratio = _predict_one(
                        fit, inputs, method, cap, alpha, global_logs[(n, noise)], groups, j
                    )
                    fit_seconds += model["cpu_fit_seconds"]
                    inference_seconds += model["cpu_inference_seconds"]
                    rid = len(records)
                    path = pred_dir / f"{rid:06d}.npz"
                    io_start = time.perf_counter()
                    np.savez_compressed(path, raw=raw, delta=delta, out_of_span_ratio=ratio)
                    prediction_io_seconds += time.perf_counter() - io_start
                    ident = {
                        "record": rid,
                        "trajectory_id": entry["id"],
                        "seed": entry["seed"],
                        "arm": entry["arm"],
                        "anchor": step,
                        "anchor_index": ai,
                        "method": method,
                        "rank_cap": cap,
                        "alpha": alpha,
                        "n": "exact" if n is None else n,
                        "noise": noise,
                        "fit_budget": 8,
                        "representation": "per_prompt",
                        "input_mode": "actual_d",
                        "r": model["r"],
                        "k": model["k"],
                        "prediction_file": str(path.relative_to(out)),
                        "raw_prediction_hash": array_hash(raw),
                        "delta_prediction_hash": array_hash(delta),
                        "input_updates_hash": array_hash(inputs["d"]),
                        "fit_updates_hash": array_hash(fit["d"]),
                        "fit_probabilities_hash": array_hash(fit["p1"]),
                        "anchor_probability_hash": array_hash(fit["p0"]),
                        "anchor_measurement_id": fit["anchor_measurement_id"],
                        "selection_sha256": selection_sha,
                        "oracle_method": method.startswith("O"),
                        "subspace_exit_gt_020": float(np.mean(ratio > 0.2)),
                    }
                    records.append(ident)
                    dimensions.append(
                        {
                            k: ident[k]
                            for k in (
                                "trajectory_id",
                                "seed",
                                "arm",
                                "anchor",
                                "method",
                                "rank_cap",
                                "r",
                                "k",
                                "alpha",
                                "n",
                                "noise",
                            )
                        }
                        | {
                            "update_singular_values": model["singular_values"],
                            "response_singular_values": model["response_singular_values"],
                            "alias_or_dependent_columns": model["alias_or_dependent_columns"],
                            "condition_number": model["condition_number"],
                            "span_le005": float(np.mean(ratio <= 0.05)),
                            "span_005_to020": float(np.mean((ratio > 0.05) & (ratio <= 0.2))),
                            "span_gt020": float(np.mean(ratio > 0.2)),
                            "status": model["status"],
                        }
                    )
                # Three bounded ablations at frozen diagnostic rule, n64 first 3 noise only.
                if n == 64 and noise < config["measurement"]["primary_noise_replicas"]:
                    ablations = (
                        [(b, "per_prompt", "actual_d") for b in (1, 2, 4)]
                        + [(8, r, "actual_d") for r in ("pooled", "six_group")]
                        + [(8, "per_prompt", r) for r in ("norm_only", "sgd_proxy")]
                    )
                    for budget, rep, inp in ablations:
                        raw, delta, model, ratio = _predict_one(
                            fit,
                            inputs,
                            diagnostic["method"],
                            diagnostic["rank_cap"],
                            diagnostic["alpha"],
                            {},
                            groups,
                            fit_budget=budget,
                            representation=rep,
                            input_mode=inp,
                        )
                        rid = len(records)
                        path = pred_dir / f"{rid:06d}.npz"
                        io_start = time.perf_counter()
                        np.savez_compressed(path, raw=raw, delta=delta, out_of_span_ratio=ratio)
                        prediction_io_seconds += time.perf_counter() - io_start
                        fit_seconds += model["cpu_fit_seconds"]
                        inference_seconds += model["cpu_inference_seconds"]
                        records.append(
                            {
                                "record": rid,
                                "trajectory_id": entry["id"],
                                "seed": entry["seed"],
                                "arm": entry["arm"],
                                "anchor": step,
                                "anchor_index": ai,
                                "method": diagnostic["method"],
                                "rank_cap": diagnostic["rank_cap"],
                                "alpha": diagnostic["alpha"],
                                "n": n,
                                "noise": noise,
                                "fit_budget": budget,
                                "representation": rep,
                                "input_mode": inp,
                                "r": model["r"],
                                "k": model["k"],
                                "prediction_file": str(path.relative_to(out)),
                                "raw_prediction_hash": array_hash(raw),
                                "delta_prediction_hash": array_hash(delta),
                                "input_updates_hash": array_hash(inputs["d"]),
                                "selection_sha256": selection_sha,
                                "oracle_method": False,
                                "subspace_exit_gt_020": float(np.mean(ratio > 0.2)),
                            }
                        )
    write_json(
        out / "prediction_freeze.json",
        {
            "selection_sha256": selection_sha,
            "records": records,
            "frozen_before_locked_test_scoring": True,
            "input_manifest_sha256": file_hash(input / "manifest.json"),
        },
    )
    freeze_sha = file_hash(out / "prediction_freeze.json")
    # Locked-test evaluation begins here; predictions above can no longer be changed.
    scoring_started = time.perf_counter()
    rows, group_rows, scatter, decompositions, direction_rows = [], [], [], [], []
    auxiliary_rows = []
    score_buckets, seed_buckets = defaultdict(list), defaultdict(list)
    oracle_cache = {}
    for rec in records:
        tid = rec["trajectory_id"]
        if tid not in oracle_cache:
            oracle_cache[tid] = _oracle(input, next(e for e in tests if e["id"] == tid))
        oracle = oracle_cache[tid]
        truth0, truth = (
            oracle.p[rec["anchor"]],
            oracle.anchor(rec["anchor_index"], role="evaluation")["p1"],
        )
        with np.load(out / rec["prediction_file"]) as a:
            raw, delta = a["raw"], a["delta"]
        if array_hash(raw) != rec["raw_prediction_hash"]:
            raise RuntimeError("frozen prediction changed before scoring")
        score = score_predictions(raw, delta, truth, truth0, groups)
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
                "fit_budget",
                "representation",
                "input_mode",
                "r",
                "k",
            )
        }
        rows.append(
            {
                **ident,
                **{k: v for k, v in score.items() if k not in ("group_rows", "direction")},
                "subspace_exit_gt_020": rec["subspace_exit_gt_020"],
            }
        )
        for bank in range(len(delta) // 3):
            base = bank * 3
            for operation in (1, 2):
                for g in sorted(set(groups)):
                    mask = groups == g
                    pred_aux = delta[base + operation, mask].mean(0) - delta[base, mask].mean(0)
                    true_aux = truth[base + operation, mask].mean(0) - truth[base, mask].mean(0)
                    auxiliary_rows.append(
                        {
                            **ident,
                            "evaluation_bank_offset": bank,
                            "operation_minus_joint0": operation,
                            "group": int(g),
                            "pred_aux_delta_pX": float(pred_aux[0]),
                            "true_aux_delta_pX": float(true_aux[0]),
                            "pred_aux_delta_v": float(-pred_aux[3]),
                            "true_aux_delta_v": float(-true_aux[3]),
                            "abs_error_pX": float(abs(pred_aux[0] - true_aux[0])),
                            "abs_error_v": float(abs(pred_aux[3] - true_aux[3])),
                        }
                    )
        for direction in score["direction"]:
            direction_rows.append({**ident, **direction})
        for g in score["group_rows"]:
            group_rows.append({**ident, **{k: v for k, v in g.items() if not k.startswith("_")}})
        key = tuple(
            rec[k]
            for k in ("method", "rank_cap", "n", "fit_budget", "representation", "input_mode")
        )
        score_buckets[key].append(score)
        seed_buckets[(*key, rec["seed"])].append(score)
        if (
            rec["n"] == 64
            and rec["noise"] == 0
            and rec["fit_budget"] == 8
            and rec["representation"] == "per_prompt"
            and rec["input_mode"] == "actual_d"
            and rec["method"] == diagnostic["method"]
            and rec["rank_cap"] == diagnostic["rank_cap"]
        ):
            for b in range(len(delta)):
                for g in sorted(set(groups)):
                    mask = groups == g
                    scatter.append(
                        {
                            **ident,
                            "branch": b,
                            "group": int(g),
                            "pred_delta_pX": float(delta[b, mask, 0].mean()),
                            "true_delta_pX": float((truth[b, mask, 0] - truth0[mask, 0]).mean()),
                            "pred_delta_v": float(-delta[b, mask, 3].mean()),
                            "true_delta_v": float(-(truth[b, mask, 3] - truth0[mask, 3]).mean()),
                        }
                    )
        if rec["n"] == "exact" and rec["method"] in (
            "O0_FULL_J_ORACLE",
            "O1_JQ_SVD_ORACLE",
            diagnostic["method"],
            "B6_FULL_Q_RIDGE",
        ):
            decompositions.append(
                {
                    **ident,
                    "delta_rmse": score["delta_rmse"],
                    "delta_truth_ss": score["delta_truth_ss"],
                    "decomposition_level": {
                        "O0_FULL_J_ORACLE": "FULL_J_VS_FINITE_STEP",
                        "O1_JQ_SVD_ORACLE": "LOW_RANK_J_VS_FINITE_STEP",
                    }.get(rec["method"], "FITTED_RESPONSE_VS_FINITE_STEP"),
                }
            )
    locked_scoring_seconds = time.perf_counter() - scoring_started
    aggregate_rows = []
    for key, scores in score_buckets.items():
        ident = dict(
            zip(
                ("method", "rank_cap", "n", "fit_budget", "representation", "input_mode"),
                key,
                strict=False,
            )
        )
        aggregate_rows.append({**ident, **aggregate_scores(scores)})
    per_seed = []
    for key, scores in seed_buckets.items():
        per_seed.append(
            {
                **dict(
                    zip(
                        (
                            "method",
                            "rank_cap",
                            "n",
                            "fit_budget",
                            "representation",
                            "input_mode",
                            "seed",
                        ),
                        key,
                        strict=False,
                    )
                ),
                **aggregate_scores(scores),
            }
        )
    paired = []
    for method, cap in specs:
        for n in ("exact", 64):
            a = {
                r["seed"]: r["delta_mae"]
                for r in per_seed
                if r["method"] == method
                and r["rank_cap"] == cap
                and r["n"] == n
                and r["fit_budget"] == 8
                and r["representation"] == "per_prompt"
                and r["input_mode"] == "actual_d"
            }
            b = {
                r["seed"]: r["delta_mae"]
                for r in per_seed
                if r["method"] == "B0_PERSISTENCE"
                and r["n"] == n
                and r["fit_budget"] == 8
                and r["representation"] == "per_prompt"
                and r["input_mode"] == "actual_d"
            }
            paired.append(
                {
                    "method": method,
                    "rank_cap": cap,
                    "n": n,
                    **cluster_bootstrap(
                        a,
                        b,
                        config["statistics"]["bootstrap_replicates"],
                        config["statistics"]["bootstrap_seed"],
                    ),
                }
            )
    write_csv(out / "auxiliary_minus_baseline.csv", auxiliary_rows)
    diagnostic_started = time.perf_counter()
    diagnostic_summary = run_oracle_diagnostics(
        config, input, out, tests, anchors, groups, diagnostic
    )
    diagnostic_seconds = time.perf_counter() - diagnostic_started
    table_io_started = time.perf_counter()
    write_csv(out / "prediction_by_anchor.csv", rows)
    write_csv(out / "prediction_by_anchor_group.csv", group_rows)
    grouped_seed = defaultdict(list)
    group_fields = (
        "seed",
        "method",
        "rank_cap",
        "alpha",
        "n",
        "fit_budget",
        "representation",
        "input_mode",
        "group",
    )
    for row in group_rows:
        grouped_seed[tuple(row[k] for k in group_fields)].append(row)
    seed_group_rows = []
    for key, group_values in grouped_seed.items():
        row = dict(zip(group_fields, key, strict=False))
        for metric in (
            "level_pX_mae",
            "level_v_mae",
            "projected_level_pX_mae",
            "projected_level_v_mae",
            "delta_pX_mae",
            "delta_v_mae",
            "delta_pX_q95",
            "delta_v_q95",
        ):
            row[metric] = float(np.mean([v[metric] for v in group_values]))
        row["quantile_aggregation"] = (
            "mean of anchor-block q95; aggregate_metrics uses pooled branch q95"
        )
        seed_group_rows.append(row)
    write_csv(out / "prediction_by_seed_group.csv", seed_group_rows)
    write_csv(out / "direction_accuracy.csv", direction_rows)
    write_csv(out / "prediction_by_seed.csv", per_seed)
    write_csv(out / "aggregate_metrics.csv", aggregate_rows)
    write_csv(out / "response_dimensions.csv", dimensions)
    write_csv(out / "response_scatter.csv", scatter)
    write_csv(out / "error_decomposition.csv", decompositions)
    write_json(out / "paired_cluster_bootstrap.json", paired)
    cost_rows = [
        {"component": "development_log_fit", "seconds": global_log_seconds, "overlap": "none"},
        {
            "component": "selection_fits_and_scoring",
            "seconds": selection_fit_scoring_seconds,
            "overlap": "includes selection Jacobians; subtotal",
        },
        {
            "component": "selection_and_test_exact_J",
            "seconds": jacobian_seconds,
            "overlap": "selection subset above",
        },
        {
            "component": "test_basis_and_local_response_fit",
            "seconds": fit_seconds,
            "overlap": "none",
        },
        {"component": "test_linear_inference", "seconds": inference_seconds, "overlap": "none"},
        {"component": "frozen_prediction_IO", "seconds": prediction_io_seconds, "overlap": "none"},
        {
            "component": "locked_scoring_and_read_IO",
            "seconds": locked_scoring_seconds,
            "overlap": "none",
        },
        {
            "component": "post_freeze_oracle_diagnostics",
            "seconds": diagnostic_seconds,
            "overlap": "none",
        },
        {
            "component": "table_output_IO",
            "seconds": time.perf_counter() - table_io_started,
            "overlap": "none",
        },
    ]
    write_csv(out / "cost_ledger.csv", cost_rows)
    summary = {
        "status": "COMPLETED",
        "modeling_status": selected["status"],
        "selected": diagnostic,
        "selection_sha256": selection_sha,
        "prediction_freeze_sha256": freeze_sha,
        "selection_candidates": len(selection_rows),
        "test_prediction_blocks": len(records),
        "test_seed_count": len(set(e["seed"] for e in tests)),
        "all_methods": [*list(METHODS), "B4_ENERGY95", "B2_FULL_PARAMETER_SKETCH"],
        "aggregate_metrics": aggregate_rows,
        "oracle_diagnostics": diagnostic_summary,
        "engineering_fixture_only": bool(manifest.get("engineering_fixture_only", False)),
        "walltime_seconds": time.perf_counter() - started,
        "local_fit_prediction_seconds": fit_seconds + inference_seconds,
        "cost_ledger": cost_rows,
        "qualification_is_not_safety_certification": True,
        "scope": "fixed dataset and model initialization; CPU realized-update tracking",
        "technical_adjustments": [
            "Single n64-selected alpha per method/r reused for exact comparison",
            "All full-grid predictions stored as compressed blocks, seed-noise dependence retained",
            "Unqualified selection-best B5 retained solely as a diagnostic when NO_ACCEPTABLE_R",
        ],
    }
    write_json(out / "summary.json", summary)
    return summary


def _nested_array(p):
    p = np.asarray(p)
    v = 1 - p[..., 3]
    b = p[..., 1] + p[..., 2]
    return np.stack(
        (
            v,
            np.divide(p[..., 0], v, out=np.full_like(v, np.nan), where=v > 0),
            np.divide(p[..., 1], b, out=np.full_like(b, np.nan), where=b > 0),
        ),
        axis=-1,
    )


def _nested_inverse(z):
    v, q, s = np.moveaxis(np.asarray(z), -1, 0)
    return np.stack((v * q, v * (1 - q) * s, v * (1 - q) * (1 - s), 1 - v), axis=-1)


def run_oracle_diagnostics(config, root, out, entries, anchors, groups, selected):
    """Post-freeze diagnostics, with all oracle subset selection explicitly labelled."""
    from .models import ridge_map

    vector_dir = out / "oracle_decomposition_vectors"
    vector_dir.mkdir()
    decomposition, coordinate, conditional, noise_rows = [], [], [], []
    max_equivalence, max_triangle_violation = 0.0, 0.0
    for entry in entries:
        oracle = _oracle(root, entry)
        exact = _view(root, entry, None, 0)
        for ai, step in enumerate(anchors):
            fit_exact = exact.anchor(ai)
            inputs = exact.anchor_inputs(ai)
            j = _oracle_j(root, oracle, step)
            true_delta = _dy(oracle.p[step], oracle.anchor(ai)["p1"])
            full = inputs["d"] @ j.T
            exact_prediction = None
            for n in (None, 16, 64, 256):
                view = exact if n is None else _view(root, entry, n, 0)
                fit = view.anchor(ai)
                _raw, delta, model, _ = _predict_one(
                    fit,
                    inputs,
                    selected["method"],
                    selected["rank_cap"],
                    selected["alpha"],
                    {},
                    groups,
                )
                from .math_contracts import helmert

                fitted = (delta @ helmert()).reshape(len(delta), -1)
                u = model["U"]
                low = (inputs["d"] @ u) @ (j @ u).T
                omission, estimation, nonlinear = low - full, fitted - low, full - true_delta
                total = fitted - true_delta
                triangle = (
                    np.linalg.norm(omission, axis=1)
                    + np.linalg.norm(estimation, axis=1)
                    + np.linalg.norm(nonlinear, axis=1)
                )
                total_norm = np.linalg.norm(total, axis=1)
                violation = float(np.max(total_norm - triangle))
                max_triangle_violation = max(max_triangle_violation, violation)
                if not np.allclose(total, omission + estimation + nonlinear, rtol=1e-9, atol=1e-10):
                    raise AssertionError("oracle vector decomposition identity failed")
                np.savez_compressed(
                    vector_dir / f"{entry['id']}_a{step}_n{n}.npz",
                    full_Jd=full,
                    low_JUUtd=low,
                    fitted_response=fitted,
                    true_response=true_delta,
                    direction_omission=omission,
                    fit_sampling_error=estimation,
                    nonlinearity=nonlinear,
                    total_residual=total,
                    triangle_upper_bound=triangle,
                    U=u,
                )
                base = {
                    "seed": entry["seed"],
                    "arm": entry["arm"],
                    "anchor": step,
                    "n": "exact" if n is None else n,
                    "noise": 0,
                    "r": model["r"],
                    "k": model["k"],
                }
                decomposition.append(
                    {
                        **base,
                        "full_J_rmse": float(np.sqrt(np.mean(nonlinear**2))),
                        "direction_omission_rmse": float(np.sqrt(np.mean(omission**2))),
                        "fit_sampling_rmse": float(np.sqrt(np.mean(estimation**2))),
                        "total_rmse": float(np.sqrt(np.mean(total**2))),
                        "triangle_bound_mean": float(triangle.mean()),
                        "total_norm_mean": float(total_norm.mean()),
                        "max_triangle_violation": violation,
                    }
                )
                if n is None:
                    exact_prediction = delta
                else:
                    noise_rows.append(
                        {
                            **base,
                            "fixed_checkpoint_prediction_noise_rmse": float(
                                np.sqrt(np.mean((delta - exact_prediction) ** 2))
                            ),
                            "anchor_measurement_rmse": float(
                                np.sqrt(np.mean((fit["p0"] - fit_exact["p0"]) ** 2))
                            ),
                            "scope": "fixed path and checkpoint; n changes measurement only",
                        }
                    )
                raw_differences = (fit["p1"] - fit["p0"]).reshape(len(fit["p1"]), -1)
                raw_map = ridge_map(fit["d"] @ u, raw_differences, selected["alpha"])
                raw_prediction = ((inputs["d"] @ u) @ raw_map.T).reshape(delta.shape)
                equivalence = float(np.max(np.abs(raw_prediction - delta)))
                max_equivalence = max(max_equivalence, equivalence)
                coordinate.append(
                    {
                        **base,
                        "raw_vs_helmert_max_abs": equivalence,
                        "status": "PASS" if equivalence < 1e-8 else "FAIL",
                        "same_U_and_ridge_used": True,
                    }
                )
            # Conditional coordinates use only the explicitly oracle-qualified subset.
            window = np.concatenate(
                (oracle.p[step : step + 17], fit_exact["p1"], oracle.anchor(ai)["p1"]), axis=0
            )
            valid = np.all(
                (1 - window[..., 3] > 0.05) & (window[..., 1] + window[..., 2] > 0.05), axis=0
            )
            for n in (None, 64):
                fit = (exact if n is None else _view(root, entry, n, 0)).anchor(ai)
                z0, z1 = _nested_array(fit["p0"]), _nested_array(fit["p1"])
                observed_valid = np.isfinite(z0).all(-1) & np.isfinite(z1).all((0, 2))
                mask = valid & observed_valid
                row = {
                    "seed": entry["seed"],
                    "arm": entry["arm"],
                    "anchor": step,
                    "n": "exact" if n is None else n,
                    "oracle_min_mass": 0.05,
                    "oracle_qualified_prompts": int(valid.sum()),
                    "observed_defined_prompts": int(mask.sum()),
                    "total_prompts": len(groups),
                    "oracle_coverage_fraction": float(valid.mean()),
                    "scored_coverage_fraction": float(mask.mean()),
                    "subset_label": "ORACLE_DIAGNOSTIC_NOT_PRIMARY",
                    "window_includes_fit_eval_and_16_steps": True,
                }
                if np.any(mask):
                    response = (z1[:, mask] - z0[mask]).reshape(len(z1), -1)
                    model = fit_local_model(
                        fit["d"],
                        response,
                        selected["method"],
                        selected["rank_cap"],
                        selected["alpha"],
                    )
                    zpred = z0[mask] + model["predict"](inputs["d"]).reshape(
                        len(inputs["d"]), int(mask.sum()), 3
                    )
                    ppred = _nested_inverse(zpred)
                    ptruth = oracle.anchor(ai)["p1"][:, mask]
                    row["conditional_probability_mae"] = float(np.mean(np.abs(ppred - ptruth)))
                    row["conditional_raw_invalid_fraction"] = float(
                        np.mean(np.any((ppred < 0) | (ppred > 1), axis=-1))
                    )
                    row["status"] = "SCORED_ORACLE_QUALIFIED_SUBSET"
                else:
                    row["conditional_probability_mae"] = None
                    row["conditional_raw_invalid_fraction"] = None
                    row["status"] = "NO_DEFINED_QUALIFIED_PROMPTS"
                conditional.append(row)
    write_csv(out / "four_level_error_decomposition.csv", decomposition)
    write_csv(out / "coordinate_regression_equivalence.csv", coordinate)
    write_csv(out / "conditional_coordinate_diagnostic.csv", conditional)
    write_csv(out / "measurement_noise_decomposition.csv", noise_rows)
    return {
        "four_level_vector_blocks": len(decomposition),
        "raw_helmert_max_abs": max_equivalence,
        "triangle_max_violation": max_triangle_violation,
        "conditional_diagnostic_rows": len(conditional),
        "conditional_primary": False,
    }


def export_shared_noise_tables(config: dict, input: Path, models: Path, out: Path) -> dict:
    """Read-only re-scoring of frozen outputs on separate primary/robust panels.

    No predictor, selection, fitting, sampling, or training API is called. Original
    M3 artifacts are never edited. Re-scoring is needed because block q95 values
    alone cannot reconstruct pooled quantiles for a subset of measurement noise.
    """
    started = time.perf_counter()
    input, models, out = Path(input), Path(models), Path(out)
    if out.resolve().is_relative_to(models.resolve()):
        raise ValueError("postprocessing output must be independent of the original M3 directory")
    out.mkdir(parents=True, exist_ok=False)
    source_files = {
        "selection": models / "selection.json",
        "prediction_freeze": models / "prediction_freeze.json",
        "original_block_metrics": models / "prediction_by_anchor.csv",
        "original_aggregate_metrics": models / "aggregate_metrics.csv",
        "collection_manifest": input / "manifest.json",
    }
    source_before = {key: file_hash(path) for key, path in source_files.items()}
    selection = json.loads(source_files["selection"].read_text())
    freeze = json.loads(source_files["prediction_freeze"].read_text())
    if source_before["selection"] != freeze["selection_sha256"]:
        raise ValueError("frozen selection hash mismatch")
    if source_before["collection_manifest"] != selection["input_manifest_sha256"]:
        raise ValueError("collection differs from the frozen model selection input")
    manifest, groups = _manifest(input), _groups(input)
    entries = {entry["id"]: entry for entry in manifest["trajectories"]}
    identity_fields = (
        "seed",
        "arm",
        "anchor",
        "method",
        "rank_cap",
        "alpha",
        "n",
        "noise",
        "fit_budget",
        "representation",
        "input_mode",
        "r",
        "k",
    )

    def identity(row):
        return tuple(str(row[field]) for field in identity_fields)

    with source_files["original_block_metrics"].open() as stream:
        old_rows = list(csv.DictReader(stream))
    old_metrics = {identity(row): row for row in old_rows}
    if len(old_metrics) != len(old_rows):
        raise ValueError("duplicate original block identity")
    freeze_ids = {identity(row) for row in freeze["records"]}
    if freeze_ids != set(old_metrics):
        raise ValueError("frozen predictions and scored block identities differ")
    selected = selection["selected"]
    robust_specs = {
        ("B0_PERSISTENCE", "0"),
        ("B6_FULL_Q_RIDGE", "FULL"),
        (selected["method"], str(selected["rank_cap"])),
    }
    exact = selection.get("exact_selection", {})
    if exact.get("rank_cap") is not None:
        robust_specs.add(("B5_EXACT_SELECTED_DIAGNOSTIC", str(exact["rank_cap"])))
    bucket_fields = (
        "method",
        "rank_cap",
        "alpha",
        "n",
        "fit_budget",
        "representation",
        "input_mode",
    )
    buckets, seed_buckets = defaultdict(list), defaultdict(list)
    coverage = defaultdict(set)
    original_records = defaultdict(list)
    oracle_cache = {}
    rescored_blocks = 0
    largest_recheck_difference = 0.0
    for rec in freeze["records"]:
        if (
            rec["fit_budget"] != 8
            or rec["representation"] != "per_prompt"
            or rec["input_mode"] != "actual_d"
        ):
            continue
        panels = []
        if rec["n"] == "exact" or (
            rec["n"] == 64 and rec["noise"] < config["measurement"]["primary_noise_replicas"]
        ):
            panels.append("primary")
        if (rec["method"], str(rec["rank_cap"])) in robust_specs and (
            rec["n"] == "exact" or rec["n"] in config["measurement"]["samples_per_prompt_grid"]
        ):
            panels.append("robust")
        if not panels:
            continue
        with np.load(models / rec["prediction_file"], allow_pickle=False) as data:
            raw, delta = data["raw"], data["delta"]
        if (
            array_hash(raw) != rec["raw_prediction_hash"]
            or array_hash(delta) != rec["delta_prediction_hash"]
        ):
            raise RuntimeError("frozen prediction differs from its pre-reveal hash")
        tid = rec["trajectory_id"]
        if tid not in oracle_cache:
            oracle_cache[tid] = _oracle(input, entries[tid])
        oracle = oracle_cache[tid]
        score = score_predictions(
            raw,
            delta,
            oracle.anchor(rec["anchor_index"], role="evaluation")["p1"],
            oracle.p[rec["anchor"]],
            groups,
        )
        old = old_metrics[identity(rec)]
        for metric in (
            "delta_mae",
            "level_mae",
            "delta_error_ss",
            "delta_truth_ss",
            "group_delta_pX_q95_max",
            "group_delta_v_q95_max",
            "raw_invalid_fraction",
        ):
            difference = abs(score[metric] - float(old[metric]))
            largest_recheck_difference = max(largest_recheck_difference, difference)
            if not np.isclose(score[metric], float(old[metric]), rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"frozen re-score differs from original metric: {metric}")
        rescored_blocks += 1
        method_key = tuple(rec[field] for field in bucket_fields)
        block = (rec["seed"], rec["arm"], rec["anchor"], rec["noise"], score["sample_count"])
        for panel in panels:
            key = (panel, *method_key)
            buckets[key].append(score)
            seed_buckets[(*key, rec["seed"])].append(score)
            if block in coverage[key]:
                raise ValueError("duplicate seed/arm/anchor/noise block in shared panel")
            coverage[key].add(block)
            original_records[key].append(rec["record"])
    rows_by_panel, seeds_by_panel, group_by_panel = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    paired_by_panel, coverage_rows = defaultdict(list), []
    for key, scores in buckets.items():
        panel = key[0]
        ident = dict(zip(bucket_fields, key[1:], strict=True))
        expected_noises = (
            {0}
            if ident["n"] == "exact"
            else set(
                range(
                    config["measurement"]["primary_noise_replicas"]
                    if panel == "primary"
                    else config["measurement"]["noise_replicas"]
                )
            )
        )
        observed_noises = {block[3] for block in coverage[key]}
        if observed_noises != expected_noises:
            raise ValueError(f"{panel} noise replica coverage mismatch for {ident}")
        baseline_key = (panel, "B0_PERSISTENCE", 0, 0.0, ident["n"], 8, "per_prompt", "actual_d")
        if coverage[key] != coverage[baseline_key]:
            raise ValueError(f"unpaired baseline block coverage for {ident}")
        extra = {
            "panel": panel,
            "noise_replicas": len(expected_noises),
            "noise_indices": sorted(expected_noises),
            "block_count": len(scores),
            "training_seed_count": len({block[0] for block in coverage[key]}),
            "paired_block_coverage": "MATCHES_B0_EXACTLY",
        }
        rows_by_panel[panel].append({**ident, **aggregate_scores(scores), **extra})
        coverage_rows.append({**ident, **extra, "frozen_record_ids": original_records[key]})
        grouped = defaultdict(list)
        for score in scores:
            for row in score["group_rows"]:
                grouped[row["group"]].append((row, score["sample_count"]))
        for group, values in grouped.items():
            total = sum(weight for _, weight in values)
            metrics = {
                field: sum(row[field] * weight for row, weight in values) / total
                for field in (
                    "level_pX_mae",
                    "level_v_mae",
                    "projected_level_pX_mae",
                    "projected_level_v_mae",
                    "delta_pX_mae",
                    "delta_v_mae",
                )
            }
            metrics["delta_pX_q95"] = float(
                np.quantile([v for row, _ in values for v in row["_gx"]], 0.95)
            )
            metrics["delta_v_q95"] = float(
                np.quantile([v for row, _ in values for v in row["_gv"]], 0.95)
            )
            group_by_panel[panel].append({**ident, "group": group, **metrics, **extra})
    for key, scores in seed_buckets.items():
        panel, method_key, seed = key[0], key[1:-1], key[-1]
        ident = dict(zip(bucket_fields, method_key, strict=True))
        noise_count = len({block[3] for block in coverage[key[:-1]] if block[0] == seed})
        seeds_by_panel[panel].append(
            {
                **ident,
                "seed": seed,
                **aggregate_scores(scores),
                "panel": panel,
                "noise_replicas": noise_count,
            }
        )
    for key in buckets:
        panel, method_key = key[0], key[1:]
        ident = dict(zip(bucket_fields, method_key, strict=True))
        for metric in ("delta_mae", "level_mae"):
            method = {
                row["seed"]: row[metric]
                for row in seeds_by_panel[panel]
                if all(
                    row[field] == value
                    for field, value in zip(bucket_fields, method_key, strict=True)
                )
            }
            baseline = {
                row["seed"]: row[metric]
                for row in seeds_by_panel[panel]
                if row["method"] == "B0_PERSISTENCE" and row["n"] == ident["n"]
            }
            paired_by_panel[panel].append(
                {
                    **ident,
                    "metric": metric,
                    "panel": panel,
                    "paired_block_coverage": "MATCHES_B0_EXACTLY",
                    **cluster_bootstrap(
                        method,
                        baseline,
                        config["statistics"]["bootstrap_replicates"],
                        config["statistics"]["bootstrap_seed"],
                    ),
                }
            )
    for panel in ("primary", "robust"):
        write_csv(out / f"{panel}_aggregate_metrics.csv", rows_by_panel[panel])
        write_csv(out / f"{panel}_by_seed.csv", seeds_by_panel[panel])
        write_csv(out / f"{panel}_by_group.csv", group_by_panel[panel])
        write_json(out / f"{panel}_paired_cluster_bootstrap.json", paired_by_panel[panel])
    write_csv(out / "paired_block_coverage.csv", coverage_rows)
    source_after = {key: file_hash(path) for key, path in source_files.items()}
    if source_before != source_after:
        raise RuntimeError("original data changed during read-only postprocessing")
    summary = {
        "status": "PASS",
        "operation": "READ_ONLY_FROZEN_PREDICTION_RESCORING",
        "new_predictions": 0,
        "new_measurements": 0,
        "training_runs": 0,
        "hyperparameters_changed": False,
        "original_artifacts_modified": False,
        "source_paths": {key: str(path.resolve()) for key, path in source_files.items()},
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "postprocessor_source_sha256": file_hash(Path(__file__)),
        "source_revision_distinction": "Postprocessor and original M3 have separate source hashes",
        "rescored_frozen_blocks": rescored_blocks,
        "largest_original_metric_recheck_difference": largest_recheck_difference,
        "primary_rows": len(rows_by_panel["primary"]),
        "robust_rows": len(rows_by_panel["robust"]),
        "primary_panel": "all main baseline configurations; exact noise0; n64 noise0,1,2",
        "robust_panel": "Selected/B0/B6: exact noise0 and five replicas per finite n",
        "quantiles": "Pooled branch errors from frozen predictions and existing oracle labels",
        "bootstrap": "5000 paired seed clusters; arms, anchors and noise stay inside each seed",
        "walltime_seconds": time.perf_counter() - started,
    }
    write_json(out / "summary.json", summary)
    return summary


def four_level_residuals(
    full: np.ndarray, low: np.ndarray, fitted: np.ndarray, truth: np.ndarray
) -> dict:
    """Vector decomposition; norms are never subtracted as contribution shares."""
    arrays = [np.asarray(value, dtype=float) for value in (full, low, fitted, truth)]
    if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("four response levels require identical matrix shapes")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("four response levels must be finite")
    full, low, fitted, truth = arrays
    omission, fit_noise, nonlinear = low - full, fitted - low, full - truth
    total = fitted - truth
    triangle = sum(np.linalg.norm(value, axis=1) for value in (omission, fit_noise, nonlinear))
    if not np.allclose(total, omission + fit_noise + nonlinear, rtol=1e-10, atol=1e-12):
        raise AssertionError("four-level residual vector identity failed")
    if np.any(np.linalg.norm(total, axis=1) > triangle + 1e-10):
        raise AssertionError("four-level residual triangle bound failed")
    return {
        "direction_omission": omission,
        "fit_sampling_error": fit_noise,
        "nonlinearity": nonlinear,
        "total_residual": total,
        "triangle_upper_bound": triangle,
    }


def export_frozen_exact_rule_diagnostics(
    config: dict, input: Path, models: Path, out: Path
) -> dict:
    """Explain an already frozen nonzero exact-selected rule at each observed n.

    Read frozen responses as the fitted prediction. Reconstruct U/C from unchanged
    paid calibration data solely to audit its direction and reproduce its saved
    output. No model/alpha/r is selected and no new test prediction is introduced.
    """
    from .math_contracts import to_helmert

    started = time.perf_counter()
    input, models, out = Path(input), Path(models), Path(out)
    vector_dir = out / "exact_rule_vectors"
    vector_dir.mkdir(parents=True, exist_ok=False)
    source_paths = {
        "selection": models / "selection.json",
        "prediction_freeze": models / "prediction_freeze.json",
        "collection_manifest": input / "manifest.json",
    }
    before = {key: file_hash(path) for key, path in source_paths.items()}
    selection = json.loads(source_paths["selection"].read_text())
    freeze = json.loads(source_paths["prediction_freeze"].read_text())
    if before["selection"] != freeze["selection_sha256"] or (
        before["collection_manifest"] != selection["input_manifest_sha256"]
    ):
        raise ValueError("frozen selection or collection hash mismatch")
    entries = {entry["id"]: entry for entry in _manifest(input)["trajectories"]}
    groups = _groups(input)
    selected = selection.get("exact_selection", {})
    rows, scatter, measurement = [], [], []
    oracle_cache, jacobian_cache, exact_predictions = {}, {}, {}
    max_reproduction = 0.0
    records = [
        rec
        for rec in freeze["records"]
        if rec["method"] == "B5_EXACT_SELECTED_DIAGNOSTIC"
        and rec["noise"] == 0
        and rec["fit_budget"] == 8
        and rec["representation"] == "per_prompt"
        and rec["input_mode"] == "actual_d"
    ]
    for rec in records:
        if rec["rank_cap"] != selected["rank_cap"] or rec["alpha"] != selected["alpha"]:
            raise ValueError("exact-selected record does not match frozen selection rule")
        entry = entries[rec["trajectory_id"]]
        n = None if rec["n"] == "exact" else rec["n"]
        view = _view(input, entry, n, rec["noise"])
        fit, updates = view.anchor(rec["anchor_index"]), view.anchor_inputs(rec["anchor_index"])
        if array_hash(updates["d"]) != rec["input_updates_hash"]:
            raise ValueError("frozen realized updates changed")
        if array_hash(fit["p1"]) != rec["fit_probabilities_hash"]:
            raise ValueError("frozen paid fit responses changed")
        with np.load(models / rec["prediction_file"], allow_pickle=False) as data:
            raw, delta = data["raw"], data["delta"]
        if (
            array_hash(raw) != rec["raw_prediction_hash"]
            or array_hash(delta) != rec["delta_prediction_hash"]
        ):
            raise ValueError("frozen exact-selected prediction changed")
        model = fit_local_model(
            fit["d"], _dy(fit["p0"], fit["p1"]), rec["method"], rec["rank_cap"], rec["alpha"]
        )
        # This equality check reconstructs a stored value, not a new prediction record.
        reconstructed = _dp((updates["d"] @ model["U"]) @ model["C"].T)
        reproduction = float(np.max(np.abs(reconstructed - delta)))
        max_reproduction = max(max_reproduction, reproduction)
        if reproduction > 1e-10 or model["r"] != rec["r"]:
            raise AssertionError("reconstructed frozen model differs from the saved response")
        tid = rec["trajectory_id"]
        if tid not in oracle_cache:
            oracle_cache[tid] = _oracle(input, entry)
        oracle = oracle_cache[tid]
        anchor_key = (tid, rec["anchor"])
        if anchor_key not in jacobian_cache:
            jacobian_cache[anchor_key] = _oracle_j(input, oracle, rec["anchor"])
        j = jacobian_cache[anchor_key]
        full = updates["d"] @ j.T
        low = (updates["d"] @ model["U"]) @ (j @ model["U"]).T
        fitted = to_helmert(delta, validate=False).reshape(len(delta), -1)
        truth0, truth = oracle.p[rec["anchor"]], oracle.anchor(rec["anchor_index"])["p1"]
        true_response = _dy(truth0, truth)
        decomposition = four_level_residuals(full, low, fitted, true_response)
        ident = {
            key: rec[key]
            for key in (
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
            )
        }
        ident["selection_source"] = "EXACT_MODEL_SELECTION_FROZEN_BEFORE_LOCKED_TEST"
        vector_name = f"{rec['record']:06d}.npz"
        np.savez_compressed(
            vector_dir / vector_name,
            full_Jd=full,
            low_JUUtd=low,
            fitted_response_from_frozen_prediction=fitted,
            true_response=true_response,
            U=model["U"],
            **decomposition,
        )
        row = {
            **ident,
            "metric_space": "concatenated_per_prompt_helmert",
            "full_J_rmse": float(np.sqrt(np.mean((full - true_response) ** 2))),
            "low_J_rmse": float(np.sqrt(np.mean((low - true_response) ** 2))),
            "direction_omission_rmse": float(
                np.sqrt(np.mean(decomposition["direction_omission"] ** 2))
            ),
            "fit_sampling_rmse": float(np.sqrt(np.mean(decomposition["fit_sampling_error"] ** 2))),
            "total_rmse": float(np.sqrt(np.mean(decomposition["total_residual"] ** 2))),
            "triangle_bound_mean": float(decomposition["triangle_upper_bound"].mean()),
            "total_residual_norm_mean": float(
                np.linalg.norm(decomposition["total_residual"], axis=1).mean()
            ),
            "frozen_model_reproduction_max_abs": reproduction,
            "source_frozen_record": rec["record"],
            "vector_file": str(Path("exact_rule_vectors") / vector_name),
        }
        rows.append(row)
        if rec["n"] == "exact":
            exact_predictions[anchor_key] = delta
        else:
            measurement.append(
                {
                    **ident,
                    "fixed_path_prediction_change_from_exact_rmse": float(
                        np.sqrt(np.mean((delta - exact_predictions[anchor_key]) ** 2))
                    ),
                    "anchor_measurement_rmse": float(np.sqrt(np.mean((fit["p0"] - truth0) ** 2))),
                    "scope": "Fixed path/rule; measurement affects fitted U and response",
                }
            )
        if rec["n"] == 64:
            for branch in range(len(delta)):
                for group in sorted(set(groups)):
                    mask = groups == group
                    scatter.append(
                        {
                            **ident,
                            "branch": branch,
                            "group": int(group),
                            "pred_delta_pX": float(delta[branch, mask, 0].mean()),
                            "true_delta_pX": float(
                                (truth[branch, mask, 0] - truth0[mask, 0]).mean()
                            ),
                            "pred_delta_v": float(-delta[branch, mask, 3].mean()),
                            "true_delta_v": float(
                                -(truth[branch, mask, 3] - truth0[mask, 3]).mean()
                            ),
                            "source_frozen_record": rec["record"],
                        }
                    )
    for name, values in (
        ("exact_rule_four_level_decomposition.csv", rows),
        ("response_scatter.csv", scatter),
        ("exact_rule_measurement_noise.csv", measurement),
    ):
        if (out / name).exists():
            raise FileExistsError(out / name)
        write_csv(out / name, values)
    after = {key: file_hash(path) for key, path in source_paths.items()}
    if before != after:
        raise AssertionError("frozen original sources changed during diagnostics")
    summary = {
        "status": "PASS" if rows else "NO_FROZEN_EXACT_SELECTED_RULE",
        "selection_source": "EXACT_MODEL_SELECTION_FROZEN_BEFORE_LOCKED_TEST",
        "rule": selected,
        "vector_blocks": len(rows),
        "scatter_rows": len(scatter),
        "noise_replica": 0,
        "samples": ["exact", 16, 64, 256],
        "fitted_prediction_source": "previously hashed M3 prediction arrays",
        "basis_source": "fixed-rule reconstruction from unchanged paid local-fit data",
        "new_test_prediction_records": 0,
        "new_measurements": 0,
        "max_model_reproduction_difference": max_reproduction,
        "source_hashes_before": before,
        "source_hashes_after": after,
        "postprocessor_source_sha256": file_hash(Path(__file__)),
        "walltime_seconds": time.perf_counter() - started,
    }
    write_json(out / "exact_rule_diagnostics_summary.json", summary)
    return summary
