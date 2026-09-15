"""Staged, read-only legacy experiments; no optimizer or network entry point.

Finite observations and predictions are persisted before their oracle scores.
New banks use independent RNG packets; legacy shared-count aliases retain
cross-bank covariance. The two contrasts in each bank share a packet.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("O_IND", "O_CRN", "O_LR_ORIGIN", "O_LR_MIX")
TARGETS = ("joint_1_minus_joint_0", "no_x_off_1_minus_joint_0", "joint_1_minus_no_x_off_1")


def jwrite(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fwrite(path, rows):
    rows = list(rows)
    names = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rng_seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16], 16)


def contrast_inputs(theta):
    t = np.asarray(theta, dtype=float)
    if t.ndim < 3 or t.shape[-2] != 3 or not np.isfinite(t).all():
        raise ValueError("bank, three operations, parameter inputs required")
    return np.stack((t[..., 1, :] - t[..., 0, :], t[..., 2, :] - t[..., 0, :]), -2)


def assemble_covariance(blocks):
    b = np.asarray(blocks, dtype=float)
    if b.ndim != 4 or b.shape[-2:] != (6, 6):
        raise ValueError("bank,prompt,6,6 covariance required")
    out = np.zeros((b.shape[1], 6 * len(b), 6 * len(b)))
    for i, block in enumerate(b):
        out[:, 6 * i : 6 * i + 6, 6 * i : 6 * i + 6] = block
    return out


def experiment_grid(stage, observations=METHODS):
    if stage == "N2C" and len(observations) > 2:
        raise ValueError("at most two observation methods may advance")
    rows = []
    if stage == "N2A":
        for banks in (1, 2, 4, 8):
            for method in ("C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6"):
                for rank in ("FULL",) if method in ("C2", "C3") else (1, 2, 4, "FULL"):
                    rows.append(
                        dict(
                            stage=stage,
                            method=method,
                            observation="EXACT",
                            n=0,
                            rank=rank,
                            alpha=1e-5,
                            eta=0.01,
                            fit_banks=banks,
                        )
                    )
    elif stage == "N2B":
        for obs in observations:
            rows.append(
                dict(
                    stage=stage,
                    method="C2",
                    observation=obs,
                    n=64,
                    rank="FULL",
                    alpha=1e-5,
                    eta=0.01,
                    fit_banks=8,
                )
            )
    elif stage == "N2C":
        for obs in observations:
            for method in ("C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6"):
                for rank in ("FULL",) if method in ("C2", "C3") else (1, 2, 4, "FULL"):
                    for alpha in (1e-5, 0.01):
                        for eta in (0.01, 0.1) if method in ("C3", "C5", "C6") else (0.01,):
                            rows.append(
                                dict(
                                    stage=stage,
                                    method=method,
                                    observation=obs,
                                    n=64,
                                    rank=rank,
                                    alpha=alpha,
                                    eta=eta,
                                    fit_banks=8,
                                )
                            )
    else:
        raise ValueError("unknown study stage")
    return rows


def configuration_id(c):
    return "|".join(
        str(c[k]) for k in ("method", "observation", "n", "rank", "alpha", "eta", "fit_banks")
    )


def prediction_first(out, predictor, scorer, inputs):
    from src.modeling_qualification.models import array_hash

    out = Path(out)
    # Empty stage directories are accepted, but completed artifacts never clobber.
    out.mkdir(parents=True, exist_ok=True)
    if (out / "freeze.json").exists() or (out / "predictions.npz").exists():
        raise FileExistsError(out)
    predictions = np.asarray(predictor())
    if not np.isfinite(predictions).all():
        raise FloatingPointError("nonfinite prediction")
    np.savez_compressed(out / "predictions.npz", prediction=predictions)
    jwrite(
        out / "freeze.json",
        dict(
            inputs=inputs,
            prediction_sha256=digest(out / "predictions.npz"),
            array_sha256=array_hash(predictions),
            frozen_before_scoring=True,
        ),
    )
    return scorer(predictions)


def legacy_root():
    root = Path(__file__).resolve().parents[3]
    return (
        root.parent
        / "ssvc-modeling-qualification-20260915/ssvc_flow/runs/modeling_qualification"
        / "qualification_20260915/M2"
    )


def load_metadata(root):
    meta = json.loads((Path(root) / "probe_metadata.json").read_text())
    return meta, np.asarray([m["group"] for m in meta]), np.asarray([m["weight"] for m in meta])


def metric_rows(pred, truth, groups, identity, active=None):
    """Compact sufficient statistics plus error vectors for exact pooled recovery."""
    rows = []
    # [bank,2,prompt,4] includes consistency difference as a prespecified target.
    pp = np.concatenate((pred, (pred[:, 0] - pred[:, 1])[:, None]), axis=1)
    tt = np.concatenate((truth, (truth[:, 0] - truth[:, 1])[:, None]), axis=1)
    for ti, target in enumerate(TARGETS):
        masks = [("all", np.ones(len(pred), dtype=bool))]
        if active is not None:
            mask_array = np.asarray(active, dtype=bool)
            if mask_array.shape != (len(pred), 3):
                raise ValueError(
                    "active requires frozen fit-excitation mask for each of three targets"
                )
            masks.append(("active", mask_array[:, ti]))
        for population, mask in masks:
            a, t = pp[mask, ti], tt[mask, ti]
            if not len(a):
                continue
            err = a - t
            gx = np.stack([err[:, groups == g, 0].mean(1) for g in sorted(set(groups))], 1)
            gv = np.stack([-err[:, groups == g, 3].mean(1) for g in sorted(set(groups))], 1)
            tx = np.stack([t[:, groups == g, 0].mean(1) for g in sorted(set(groups))], 1)
            tv = np.stack([-t[:, groups == g, 3].mean(1) for g in sorted(set(groups))], 1)
            row = dict(
                identity,
                target=target,
                population=population,
                case_count=len(a),
                event_error_ss=float(np.sum(err**2)),
                event_truth_ss=float(np.sum(t**2)),
                event_mae=float(np.abs(err).mean()),
                predicted_difference_norm=float(np.linalg.norm(a)),
                pX_error_ss=float(np.sum(gx**2)),
                pX_truth_ss=float(np.sum(tx**2)),
                v_error_ss=float(np.sum(gv**2)),
                v_truth_ss=float(np.sum(tv**2)),
                pX_abs_errors=np.abs(gx).ravel().tolist(),
                v_abs_errors=np.abs(gv).ravel().tolist(),
            )
            for name in ("pX", "v", "event"):
                energy = row[name + "_truth_ss"]
                row[name + "_nrmse"] = (
                    float(np.sqrt(row[name + "_error_ss"] / energy)) if energy > 0 else None
                )
            rows.append(row)
    return rows


def aggregate_rows(rows, keys=("configuration_id", "target", "population")):
    buckets = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        buckets.setdefault(key, []).append(row)
    result = []
    for key, rr in buckets.items():
        item = {k: v for k, v in zip(keys, key, strict=True)}
        item.update(
            {
                k: rr[0].get(k)
                for k in (
                    "method",
                    "observation",
                    "n",
                    "rank",
                    "alpha",
                    "eta",
                    "fit_banks",
                    "stage",
                )
            }
        )
        item["seed_count"] = len({r["seed"] for r in rr})
        item["actual_rank_min"] = min(r.get("actual_rank", 0) for r in rr)
        item["actual_rank_max"] = max(r.get("actual_rank", 0) for r in rr)
        item["k_min"] = min(r.get("k", 0) for r in rr)
        item["k_max"] = max(r.get("k", 0) for r in rr)
        item["predicted_difference_norm"] = float(
            np.sqrt(sum(r["predicted_difference_norm"] ** 2 for r in rr))
        )
        for field in ("pX", "v", "event"):
            ss = sum(r[field + "_error_ss"] for r in rr)
            energy = sum(r[field + "_truth_ss"] for r in rr)
            item[field + "_error_ss"], item[field + "_truth_ss"] = ss, energy
            item[field + "_nrmse"] = float(np.sqrt(ss / energy)) if energy > 0 else None
            if field != "event":
                errors = np.concatenate([r[field + "_abs_errors"] for r in rr])
                item[field + "_error_q95"] = float(np.quantile(errors, 0.95))
                item[field + "_mae"] = float(errors.mean())
        result.append(item)
    return result
