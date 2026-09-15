"""Finite fit sweeps with pure fit capabilities and sealed predictions."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from .study import (
    TARGETS,
    aggregate_rows,
    configuration_id,
    contrast_inputs,
    digest,
    jwrite,
    load_metadata,
    metric_rows,
)


def _record_stream(path, rows):
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def read_records(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def compact_rows(rows):
    return [{k: v for k, v in row.items() if not k.endswith("_abs_errors")} for row in rows]


def fit_unit(root, entry, ai, configs, packets, out, *, oracle_diagnostics=False, data_role=None):
    """Fit all fixed configs, persist their predictions, then open evaluation truth.

    `packets` is an already materialized paid-observation mapping. Estimators
    receive arrays only, not this runner's root, loader, service, or oracle.
    """
    from src.modeling_qualification.math_contracts import helmert
    from src.modeling_qualification.models import array_hash, fit_local_model

    from .estimators import fit_contrast_model, out_of_subspace, prepare_contrast
    from .joint_packets import fit_packet_covariance, retained_fit_rows
    from .parent import load_parent, load_parent_oracle
    from .targets import group_xv_matrix

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    parent = load_parent(root, entry["id"])
    metadata, groups, _ = load_metadata(root)
    h, gm = helmert(), group_xv_matrix(metadata)
    obs = parent.observations
    step = int(obs["anchors"][ai])
    theta = obs["branch_theta"][ai]
    increments = contrast_inputs(theta)
    query = increments[10:14].reshape(-1, 737)
    predictions, records, fit_ids, model_meta = [], [], [], []
    caches = {}
    exact = load_parent_oracle(root, entry["id"]) if oracle_diagnostics else None
    t0 = time.perf_counter()
    for config in configs:
        c = dict(config)
        n, method, observation, banks = c["n"], c["method"], c["observation"], c["fit_banks"]
        noises = [0] if n == 0 else list(range(8))
        for noise in noises:
            if observation == "EXACT":
                p = exact["branch_p"][ai, :banks]
                response = ((p[:, 1:] - p[:, :1]) @ h).reshape(2 * banks, 72, 3)
                cov = None
            else:
                pkt = packets[(observation, n)]
                response = pkt["helmert"][noise, :banks].reshape(2 * banks, 72, 3)
                cov = fit_packet_covariance(pkt, noise, range(banks))
            e = increments[:banks].reshape(-1, 737)
            active_rows = []
            for bank in range(banks):
                if not np.array_equal(theta[bank, 1], theta[bank, 0]):
                    active_rows.append(2 * bank)
                if not np.array_equal(theta[bank, 2], theta[bank, 0]) and not np.array_equal(
                    theta[bank, 2], theta[bank, 1]
                ):
                    active_rows.append(2 * bank + 1)
            if observation != "EXACT":
                active_rows = retained_fit_rows(pkt, noise, range(banks))
            preparation_key = (observation, n, noise, banks, c["eta"], method in ("C3", "C5", "C6"))
            if method == "C0":
                pred = np.zeros((8, 216))
                model = {"r": 0, "k": 0, "status": "ZERO_BASELINE"}
            elif method.startswith("C1"):
                if n == 0:
                    p0 = exact["p"][step]
                    pp = exact["branch_p"][ai, :banks]
                else:
                    old_packet = packets[("O_IND", n)]
                    p0 = old_packet["legacy_origin_counts"][noise] / n
                    pp = old_packet["legacy_total_levels"][noise, :banks]
                # Original exact-selected r1 rule is preserved as a stronger
                # nonzero baseline alongside the actual finite selected r0 rule.
                rank = 0 if method == "C1_LEGACY_FROZEN" and n else 1
                old = fit_local_model(
                    obs["branch_d"][ai, :banks].reshape(-1, 737),
                    ((pp - p0) @ h).reshape(3 * banks, 216),
                    "B5_WORST_GROUP_RANK",
                    rank,
                    0.01 if rank else 0.0,
                )
                all_pred = old["predict"](obs["branch_d"][ai, 10:14].reshape(-1, 737)).reshape(
                    4, 3, 216
                )
                pred = (all_pred[:, 1:] - all_pred[:, :1]).reshape(8, 216)
                model = old
            else:
                if preparation_key not in caches:
                    caches[preparation_key] = prepare_contrast(
                        e,
                        response,
                        covariance=cov if method in ("C3", "C5", "C6") else None,
                        eta=c["eta"],
                        observation_n=n or None,
                        active_rows=active_rows,
                    )
                model = fit_contrast_model(
                    e,
                    response,
                    method,
                    c["rank"],
                    c["alpha"],
                    prepared=caches[preparation_key],
                    eta=c["eta"],
                    group_map=gm,
                    observation_n=n or None,
                )
                pred = model["predict"](query)
            pred = pred.reshape(4, 2, 72, 3)
            predictions.append(pred)
            ident = dict(
                c,
                configuration_id=configuration_id(c),
                noise_replica=noise,
                seed=entry["seed"],
                arm=entry["arm"],
                anchor=step,
                original_role=entry["split"],
                v2_role=data_role or "legacy_diagnostic",
                actual_rank=int(model["r"]),
                k=int(model["k"]),
                model_status=model["status"],
                prediction_hash=array_hash(pred),
            )
            if "Q" in model:
                oos = out_of_subspace(query, model["Q"])
                ident["out_of_subspace"] = oos
            model_meta.append(ident)
            fit_ids.append(
                dict(
                    configuration_id=ident["configuration_id"],
                    noise=noise,
                    fit_banks=list(range(banks)),
                    evaluation_banks=list(range(10, 14)),
                )
            )
    # A single compressed archive is substantially smaller than thousands of
    # separate files, and includes every method's raw unprojected prediction.
    np.savez_compressed(out / "predictions.npz", prediction=np.asarray(predictions))
    freeze = {
        "frozen_before_scoring": True,
        "oracle_diagnostic": oracle_diagnostics,
        "prediction_sha256": digest(out / "predictions.npz"),
        "fit_inputs": fit_ids,
        "parameters_sha256": entry["sha256"]["observations_file"],
        "packet_hashes": {str(k): v.get("artifact_sha256") for k, v in packets.items()},
        "c1_auxiliary_hashes": {
            str(k): {
                "arrays": v.get("c1_auxiliary_sha256"),
                "receipt": v.get("c1_auxiliary_receipt_sha256"),
                "path": v.get("c1_auxiliary_path"),
            }
            for k, v in packets.items()
            if v.get("c1_auxiliary_sha256")
        },
        "models": model_meta,
        "fit_seconds": time.perf_counter() - t0,
    }
    with gzip.open(out / "freeze_metadata.json.gz", "wt") as stream:
        json.dump(freeze, stream, separators=(",", ":"), allow_nan=False)
    jwrite(
        out / "freeze.json",
        {k: v for k, v in freeze.items() if k not in ("models", "fit_inputs")}
        | {
            "metadata_file": "freeze_metadata.json.gz",
            "metadata_sha256": digest(out / "freeze_metadata.json.gz"),
            "model_count": len(model_meta),
        },
    )
    # The finite branch reaches this loader only after the immutable prediction.
    if exact is None:
        exact = load_parent_oracle(root, entry["id"])
    truth = exact["branch_p"][ai, 10:14, 1:] - exact["branch_p"][ai, 10:14, :1]
    for pred, identity in zip(predictions, model_meta, strict=True):
        fit_e = increments[: identity["fit_banks"]]
        excitation = np.r_[np.any(fit_e != 0, axis=(0, 2)), np.any(fit_e[:, 0] - fit_e[:, 1] != 0)]
        active = np.broadcast_to(excitation, (4, 3))
        records.extend(metric_rows(pred @ h.T, truth, groups, identity, active))
    _record_stream(out / "metrics.jsonl.gz", records)
    return records


def baseline_configs(n=64):
    return [
        dict(
            stage="BASELINE",
            method=m,
            observation="O_IND" if n else "EXACT",
            n=n,
            rank=0 if m in ("C0", "C1_LEGACY_FROZEN") else 1,
            alpha=0.01,
            eta=0.01,
            fit_banks=8,
        )
        for m in ("C0", "C1_LEGACY_FROZEN", "C1_LEGACY_EXACT_RULE")
    ]


def choose_observations(rows, costs):
    eligible = [
        r
        for r in aggregate_rows([r for r in rows if r["original_role"] == "model_selection"])
        if r["population"] == "all" and r["target"] == TARGETS[0] and r["method"] == "C2"
    ]
    for row in eligible:
        row["selection_calibration_cost"] = costs[row["observation"]]
        row["selection_worst_nrmse"] = max(
            row["pX_nrmse"] if row["pX_nrmse"] is not None else float("inf"),
            row["v_nrmse"] if row["v_nrmse"] is not None else float("inf"),
        )
        row["selection_access_regime"] = (
            "SAMPLE_AND_LOGP" if "LR" in row["observation"] else "SAMPLE_ONLY"
        )
    for row in eligible:
        row["pareto_within_access"] = not any(
            other["selection_access_regime"] == row["selection_access_regime"]
            and other["selection_worst_nrmse"] <= row["selection_worst_nrmse"]
            and other["selection_calibration_cost"] <= row["selection_calibration_cost"]
            and (
                other["selection_worst_nrmse"] < row["selection_worst_nrmse"]
                or other["selection_calibration_cost"] < row["selection_calibration_cost"]
            )
            for other in eligible
            if other is not row
        )
    ranked = sorted(
        eligible,
        key=lambda r: (
            r["selection_worst_nrmse"] >= 0.75,
            not r["pareto_within_access"],
            r["selection_worst_nrmse"] * r["selection_calibration_cost"],
            r["selection_worst_nrmse"],
        ),
    )
    for row in ranked:
        row["selection_rule"] = (
            "dual_target_qualified_first_then_within_access_Pareto_"
            "then_NRMSE_times_paid_cost; score=.25,label=.05"
        )
    return [r["observation"] for r in ranked[:2]], ranked


def best_nonzero(rows, observation=None):
    aggregated = aggregate_rows([r for r in rows if r["original_role"] == "model_selection"])
    rr = [
        r
        for r in aggregated
        if r["population"] == "all"
        and r["target"] == TARGETS[0]
        and r["method"] not in ("C0", "C1_LEGACY_FROZEN", "C1_LEGACY_EXACT_RULE")
        and r["actual_rank_max"] > 0
        and (observation is None or r["observation"] == observation)
    ]
    rr.sort(
        key=lambda r: (
            max(r["pX_nrmse"] or float("inf"), r["v_nrmse"] or float("inf")),
            r["actual_rank_max"],
        )
    )
    return rr[0] if rr else None
