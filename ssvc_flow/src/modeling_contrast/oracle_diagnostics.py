"""Development-only frozen CPU Jacobians and fit-error isolation.

No optimizer, training-data generation, remote model, or online SSVC entry point
is available here. Oracle truth and covariance are explicitly excluded from the
finite-observation model ranking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .estimators import fit_contrast_model, prepare_contrast, subspace_distance


def autograd_helmert_jacobian(model, theta, features, categories):
    """Differentiate the unchanged parent forward at an existing parameter vector."""
    import torch

    from src.modeling_qualification.math_contracts import helmert
    from src.modeling_qualification.toy import event_probabilities, set_parameters

    started = time.perf_counter()
    torch.set_num_threads(1)
    theta = np.asarray(theta, dtype=np.float64).copy()
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(categories)
    if (
        features.ndim != 3
        or features.shape[1:] != (16, 44)
        or labels.shape != features.shape[:2]
        or not np.isfinite(features).all()
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels > 3))
    ):
        raise ValueError("finite parent prompt/action features and four-event identities required")
    set_parameters(model, theta)
    model.eval()
    probabilities = event_probabilities(model, features, labels)
    response = (probabilities @ torch.tensor(helmert(), dtype=torch.float64)).reshape(-1)
    dimension = response.numel()
    gradients = torch.autograd.grad(
        response,
        tuple(model.parameters()),
        grad_outputs=torch.eye(dimension, dtype=torch.float64),
        is_grads_batched=True,
        create_graph=False,
        retain_graph=False,
    )
    jacobian = (
        torch.cat([g.reshape(dimension, -1) for g in gradients], dim=1).detach().numpy().copy()
    )
    return {
        "jacobian": jacobian,
        "events": probabilities.detach().numpy().copy(),
        "cost": {
            "jacobian_calls": 1,
            "forward_model_calls": 1,
            "scored_prompt_policies": len(features),
            "action_score_values": len(features) * 16,
            "batched_vjp_calls": 1,
            "logical_vjp_directions": dimension,
            "optimizer_updates": 0,
            "cpu_threads": 1,
            "seconds": time.perf_counter() - started,
        },
    }


def common_response_rank_one_witness():
    """Two common response coordinates dominate a small third contrast axis."""
    overall = np.array(
        [[100.0, 1.0, 0.0], [100.0, 1.0, 0.01], [100.0, -1.0, 0.0], [100.0, -1.0, -0.01]]
    )
    u, s, vt = np.linalg.svd(overall, full_matrices=False)
    prediction = (u[:, :1] * s[:1]) @ vt[:1]
    actual_contrast = overall[[1, 3]] - overall[[0, 2]]
    predicted_contrast = prediction[[1, 3]] - prediction[[0, 2]]
    return {
        "overall_response": overall.tolist(),
        "overall_r1_prediction": prediction.tolist(),
        "contrast_truth": actual_contrast.tolist(),
        "contrast_prediction": predicted_contrast.tolist(),
        "overall_r1_energy_fraction": float(s[0] ** 2 / np.sum(s**2)),
        "contrast_truth_energy": float(np.sum(actual_contrast**2)),
        "contrast_nrmse": float(
            np.linalg.norm(predicted_contrast - actual_contrast) / np.linalg.norm(actual_contrast)
        ),
        "diagnostic_only": True,
    }


def isolate_fit_errors(
    e,
    finite_y,
    exact_y,
    cov,
    query,
    group_map,
    n,
    *,
    oracle_covariance=None,
    active_rows=None,
):
    """Return 18 predictions per covariance source for the fixed N2D grid.

    All exact responses are *fit-bank* development data. Records index the
    leading axis of ``predictions``; query updates never select directions.
    Exact covariance, when supplied, is a separate labeled 18-record diagnostic.
    """
    records, predictions = [], []
    covariance_sources = [("FINITE_PAID", cov)]
    if oracle_covariance is not None:
        covariance_sources.append(("EXACT_ORACLE", oracle_covariance))
    for covariance_source, covariance in covariance_sources:
        prepared = prepare_contrast(
            e, finite_y, covariance=covariance, eta=0.01, observation_n=n, active_rows=active_rows
        )
        for method in ("C5", "C6"):
            for rank in (1, 2, "FULL"):
                kwargs = dict(prepared=prepared, eta=0.01, group_map=group_map, observation_n=n)
                exact_u = fit_contrast_model(
                    e, finite_y, method, rank, 1e-5, direction_responses=exact_y, **kwargs
                )
                finite_u_exact_c = fit_contrast_model(
                    e, finite_y, method, rank, 1e-5, coefficient_responses=exact_y, **kwargs
                )
                finite_both = fit_contrast_model(e, finite_y, method, rank, 1e-5, **kwargs)
                for fitted in (exact_u, finite_u_exact_c, finite_both):
                    predictions.append(fitted["predict"](query))
                    records.append(
                        {
                            "prediction_index": len(predictions) - 1,
                            "method": method,
                            "rank_cap": rank,
                            "actual_rank": fitted["r"],
                            "fit_rank": fitted["k"],
                            "status": fitted["status"],
                            "alpha": 1e-5,
                            "eta": 0.01,
                            "n": int(n),
                            "diagnostic": fitted["diagnostic"],
                            "covariance_source": covariance_source,
                            "subspace_to_exact": subspace_distance(fitted["U"], exact_u["U"]),
                            "response_singular_values": fitted["response_singular_values"],
                            "development_only": True,
                            "main_ranking_eligible": False,
                        }
                    )
    return {
        "records": records,
        "predictions": np.asarray(predictions),
        "development_only": True,
        "main_ranking_eligible": False,
        "query_used_for_direction_selection": False,
    }


def run_oracle_diagnostics(root, out):
    """Execute the prespecified 36-origin/504-baseline legacy CPU diagnostic."""
    import torch

    from src.modeling_qualification.math_contracts import helmert
    from src.modeling_qualification.models import array_hash
    from src.modeling_qualification.toy import make_model

    from .parent import load_parent, load_parent_oracle
    from .study import digest, fwrite, jwrite, load_metadata, metric_rows
    from .targets import group_xv_matrix

    torch.set_num_threads(1)
    root, out = Path(root), Path(out)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = sorted(
        (
            e
            for e in manifest["trajectories"]
            if e["seed"] in range(101, 107) and e["arm"] in ("X_BASE", "X_VALID")
        ),
        key=lambda e: (e["seed"], e["arm"]),
    )
    identities = {(e["seed"], e["arm"]) for e in entries}
    if len(entries) != 12 or identities != {
        (s, a) for s in range(101, 107) for a in ("X_BASE", "X_VALID")
    }:
        raise ValueError("exactly the twelve existing development trajectories are required")
    if manifest["anchors"] != [8, 24, 40]:
        raise ValueError("legacy anchors differ")
    dataset_path = root / "dataset.npz"
    if digest(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("legacy dataset hash mismatch")
    with np.load(dataset_path, allow_pickle=False) as ds:
        features, categories = ds["probe_features"].copy(), ds["probe_categories"].copy()
    metadata, groups, _ = load_metadata(root)
    group_map = group_xv_matrix(metadata)
    if features.shape != (72, 16, 44):
        raise ValueError("frozen 72-prompt finite-action dataset required")
    out.mkdir(parents=True, exist_ok=False)
    model = make_model(7001)
    h = helmert()
    rows, receipts, artifacts = [], [], []
    started = time.perf_counter()
    counters = {
        "jacobian_calls": 0,
        "forward_model_calls": 0,
        "scored_prompt_policies": 0,
        "action_score_values": 0,
        "batched_vjp_calls": 0,
        "logical_vjp_directions": 0,
        "optimizer_updates": 0,
    }
    max_probability_disagreement = 0.0
    for entry in entries:
        parent, exact = load_parent(root, entry["id"]), load_parent_oracle(root, entry["id"])
        obs = parent.observations
        increments, truths, origin_predictions, baseline_predictions = [], [], [], []
        for ai, step in enumerate(manifest["anchors"]):
            origin = autograd_helmert_jacobian(model, obs["theta"][step], features, categories)
            origin_j = origin["jacobian"]
            disagreement = float(np.max(np.abs(origin["events"] - exact["p"][step])))
            max_probability_disagreement = max(max_probability_disagreement, disagreement)
            receipts.append(
                {
                    "trajectory": entry["id"],
                    "anchor": step,
                    "location": "ORIGIN",
                    "jacobian_sha256": array_hash(origin_j),
                    "probability_disagreement": disagreement,
                    **origin["cost"],
                }
            )
            for field in counters:
                counters[field] += origin["cost"][field]
            theta = obs["branch_theta"][ai]
            e = theta[:, 1:] - theta[:, :1]
            truth = (exact["branch_p"][ai, :, 1:] - exact["branch_p"][ai, :, :1]) @ h
            predicted_origin = (e.reshape(-1, 737) @ origin_j.T).reshape(14, 2, 72, 3)
            predicted_baseline = np.zeros_like(predicted_origin)
            for bank in range(14):
                result = autograd_helmert_jacobian(model, theta[bank, 0], features, categories)
                predicted_baseline[bank] = (e[bank] @ result["jacobian"].T).reshape(2, 72, 3)
                disagreement = float(
                    np.max(np.abs(result["events"] - exact["branch_p"][ai, bank, 0]))
                )
                max_probability_disagreement = max(max_probability_disagreement, disagreement)
                receipts.append(
                    {
                        "trajectory": entry["id"],
                        "anchor": step,
                        "bank": bank,
                        "location": "BASELINE",
                        "jacobian_sha256": array_hash(result["jacobian"]),
                        "probability_disagreement": disagreement,
                        **result["cost"],
                    }
                )
                for field in counters:
                    counters[field] += result["cost"][field]
                del result
            for method, pred in (
                ("O_J_ORIGIN", predicted_origin),
                ("O_J_BASELINE", predicted_baseline),
            ):
                for role, banks in (
                    ("all_banks", np.arange(14)),
                    ("fit", np.arange(8)),
                    ("diagnostic", np.arange(8, 10)),
                    ("heldout", np.arange(10, 14)),
                ):
                    identity = dict(
                        trajectory_id=entry["id"],
                        seed=entry["seed"],
                        arm=entry["arm"],
                        anchor=step,
                        method=method,
                        bank_role=role,
                        stage="O_J_DIAGNOSTIC",
                        v2_role="legacy_diagnostic",
                        development_only=True,
                        main_ranking_eligible=False,
                    )
                    rows.extend(
                        metric_rows(pred[banks] @ h.T, truth[banks] @ h.T, groups, identity)
                    )
            increments.append(e)
            truths.append(truth)
            origin_predictions.append(predicted_origin)
            baseline_predictions.append(predicted_baseline)
            del origin_j, origin
        arrays = {
            "e": np.asarray(increments),
            "truth_helmert": np.asarray(truths),
            "origin_prediction": np.asarray(origin_predictions),
            "baseline_prediction": np.asarray(baseline_predictions),
            "group_XV_map": group_map,
            "anchors": np.array(manifest["anchors"]),
            "banks": np.arange(14),
        }
        arrays["origin_error"] = arrays["origin_prediction"] - arrays["truth_helmert"]
        arrays["baseline_error"] = arrays["baseline_prediction"] - arrays["truth_helmert"]
        artifact = out / f"{entry['id']}.npz"
        np.savez_compressed(artifact, **arrays)
        artifacts.append(
            {
                "path": artifact.name,
                "sha256": digest(artifact),
                "bytes": artifact.stat().st_size,
                "arrays": {
                    k: {"shape": list(v.shape), "sha256": array_hash(v)} for k, v in arrays.items()
                },
                "source_sha256": entry["sha256"],
            }
        )
        jwrite(
            out / "progress.json",
            {
                "status": "RUNNING",
                "completed_trajectories": len(artifacts),
                "elapsed_seconds": time.perf_counter() - started,
                **counters,
            },
        )
    if counters["jacobian_calls"] != 540 or max_probability_disagreement > 1e-12:
        raise AssertionError("Jacobian call count or parent-forward parity failed")
    pooled = []
    for method in ("O_J_ORIGIN", "O_J_BASELINE"):
        for target in sorted({row["target"] for row in rows}):
            for role in ("all_banks", "fit", "diagnostic", "heldout"):
                for population in ("all", "active"):
                    subset = [
                        r
                        for r in rows
                        if (r["method"], r["target"], r["bank_role"], r["population"])
                        == (method, target, role, population)
                    ]
                    row = dict(
                        method=method,
                        target=target,
                        bank_role=role,
                        population=population,
                        seed_count=len({r["seed"] for r in subset}),
                        main_ranking_eligible=False,
                    )
                    for metric in ("pX", "v", "event"):
                        numerator = sum(r[f"{metric}_error_ss"] for r in subset)
                        denominator = sum(r[f"{metric}_truth_ss"] for r in subset)
                        row[f"{metric}_error_ss"], row[f"{metric}_truth_ss"] = (
                            numerator,
                            denominator,
                        )
                        row[f"{metric}_nrmse"] = (
                            float(np.sqrt(numerator / denominator)) if denominator > 0 else None
                        )
                    pooled.append(row)
    fwrite(out / "ORACLE_JACOBIAN_BY_SEED.csv", rows)
    fwrite(out / "ORACLE_JACOBIAN_POOLED.csv", pooled)
    jwrite(out / "JACOBIAN_RECEIPTS.json", receipts)
    jwrite(out / "COMMON_RESPONSE_R1_WITNESS.json", common_response_rank_one_witness())
    summary = {
        "status": "PASS",
        "scope": "FROZEN_CPU_DEVELOPMENT_ORACLE_ONLY",
        "main_ranking_eligible": False,
        "trajectory_count": 12,
        "seed_count": 6,
        "origin_jacobians": 36,
        "baseline_jacobians": 504,
        "jacobian_coordinates": [216, 737],
        "full_jacobians_saved": False,
        "cpu_threads": 1,
        "model_parameter_count": 737,
        "new_training": False,
        "new_gpu_or_qwen_calls": 0,
        "legacy_root": str(root.resolve()),
        "manifest_sha256": digest(root / "manifest.json"),
        "max_parent_forward_probability_disagreement": max_probability_disagreement,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_bytes": sum(a["bytes"] for a in artifacts),
        "cost": counters,
        "artifacts": artifacts,
    }
    jwrite(out / "SUMMARY.json", summary)
    jwrite(
        out / "progress.json",
        {
            "status": "PASS",
            "completed_trajectories": 12,
            "elapsed_seconds": summary["elapsed_seconds"],
            **counters,
        },
    )
    return summary


def main():
    from .study import legacy_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=legacy_root())
    parser.add_argument(
        "--out", type=Path, default=Path("runs/modeling_contrast_v2/oracle_diagnostics")
    )
    args = parser.parse_args()
    result = run_oracle_diagnostics(args.root, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
