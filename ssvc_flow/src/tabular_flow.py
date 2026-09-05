"""Self-derived finite-group categorical softmax mathematics (CPU only).

Notation is explicitly local because the plan's THEORY_REVIEW is unavailable:
U(n)=(n/K)*A(n), H_K=E[U], Xi_K=E[U U^T], and
B_eta,K=E[softmax(log(p)+eta*U)]-p. Xi is the second moment,
not Cov(U), and the finite step averages probabilities after each update.
These are normalized sequence-score updates; training uses eta/Lnorm.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import gammaln, xlogy


def _probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (4,) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("p must contain four finite nonnegative probabilities")
    if not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    return p / p.sum()


def _rewards(rewards, epsilon):
    reward = np.asarray(rewards, dtype=np.float64)
    if reward.shape != (4,) or not np.isfinite(reward).all():
        raise ValueError("rewards must have four finite entries")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    return reward


@lru_cache(maxsize=8)
def _cached_counts(k):
    return tuple(
        (a, b, c, k - a - b - c)
        for a in range(k + 1)
        for b in range(k - a + 1)
        for c in range(k - a - b + 1)
    )


def count_vectors(k):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("K must be a positive integer")
    return np.asarray(_cached_counts(int(k)), dtype=np.int64)


def multinomial_weights(counts, p):
    p = _probabilities(p)
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 4 or not counts.size:
        raise ValueError("counts must be a nonempty n-by-4 array")
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("counts must be nonnegative integers")
    totals = counts.sum(axis=1)
    if totals[0] < 1 or not np.all(totals == totals[0]):
        raise ValueError("all counts must have the same positive K")
    log_weight = gammaln(totals + 1) - gammaln(counts + 1).sum(axis=1)
    log_weight = log_weight + xlogy(counts, p).sum(axis=1)
    return np.exp(log_weight)


def group_updates(counts, rewards, epsilon=1e-4):
    rewards = _rewards(rewards, epsilon)
    counts = np.asarray(counts, dtype=np.float64)
    if (
        counts.ndim != 2
        or counts.shape[1] != 4
        or np.any(counts < 0)
        or not np.isfinite(counts).all()
    ):
        raise ValueError("invalid counts")
    totals = counts.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("empty groups are invalid")
    frequencies = counts / totals
    # Subtract a supported reward first: exactly tied groups remain exactly zero
    # even if a floating-point weighted mean of that reward rounds differently.
    shifted = rewards - rewards[np.argmax(counts > 0, axis=1)][:, None]
    centered = shifted - (frequencies * shifted).sum(axis=1, keepdims=True)
    variance = (frequencies * centered**2).sum(axis=1)
    scale = np.sqrt(variance + epsilon**2)
    advantages = np.divide(
        centered, scale[:, None], out=np.zeros_like(centered), where=scale[:, None] > 0
    )
    return frequencies * advantages, variance == 0


def _moments(p, rewards, k, epsilon):
    counts = count_vectors(k)
    weight = multinomial_weights(counts, p)
    updates, zero = group_updates(counts, rewards, epsilon)
    first = np.einsum("n,ni->i", weight, updates)
    second = np.einsum("n,ni,nj->ij", weight, updates, updates)
    return weight, updates, first, second, float(weight[zero].sum())


def _finite_step(p, updates, weights, first, second, eta):
    # p*exp(...) remains exact on boundary supports where log(p)=-infinity.
    centered_updates = updates - updates.max(axis=1, keepdims=True)
    unnormalized = p * np.exp(eta * centered_updates)
    successors = unnormalized / unnormalized.sum(axis=1, keepdims=True)
    expected = np.einsum("n,ni->i", weights, successors)
    drift = p * (first - p @ first)
    diagonal = np.diag(second)
    curvature = 0.5 * p * (diagonal - 2 * (second @ p) - p @ diagonal + 2 * (p @ second @ p))
    first_order = p + eta * drift
    second_order = first_order + eta**2 * curvature
    v = float(p[:3].sum())
    q_initial = float(p[0] / v) if v else None
    expected_q = float(weights @ (successors[:, 0] / successors[:, :3].sum(axis=1))) if v else None
    pooled_q = float(expected[0] / expected[:3].sum()) if v else None
    return {
        "exact_p_next": expected.tolist(),
        "delta_p": (expected - p).tolist(),
        "B_eta_K": (expected - p).tolist(),
        "ode_drift": drift.tolist(),
        "first_order_p_next": first_order.tolist(),
        "second_order_p_next": second_order.tolist(),
        "first_order_error": float(np.linalg.norm(expected - first_order)),
        "second_order_error": float(np.linalg.norm(expected - second_order)),
        "q_initial": q_initial,
        "expected_q_next": expected_q,
        "pooled_q_next": pooled_q,
        "expected_delta_q": expected_q - q_initial if v else None,
        "pooled_delta_q": pooled_q - q_initial if v else None,
    }


def finite_group_flow(p, rewards, k, eta, epsilon=1e-4):
    """Exact one-step expectation over every Multinomial(K,p) count vector."""
    p, rewards = _probabilities(p), _rewards(rewards, epsilon)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    weights, updates, first, second, zero = _moments(p, rewards, k, epsilon)
    return {
        "p": p.tolist(),
        "rewards": rewards.tolist(),
        "K": int(k),
        "eta": float(eta),
        "epsilon": float(epsilon),
        "method": "exact_multinomial_enumeration",
        "count_vectors": len(weights),
        "weight_sum": float(weights.sum()),
        "maximum_group_update_sum_error": float(np.max(np.abs(updates.sum(axis=1)))),
        "H_K": first.tolist(),
        "Xi_K": second.tolist(),
        "zero_variance_probability": zero,
        "no_X_probability": float((1 - p[0]) ** k),
        "contains_X_probability": float(1 - (1 - p[0]) ** k),
        "contains_S_probability": float(1 - (1 - p[1]) ** k),
        "contains_I_probability": float(1 - (1 - p[3]) ** k),
        "X_binary_informative_probability": float(1 - p[0] ** k - (1 - p[0]) ** k),
        **_finite_step(p, updates, weights, first, second, eta),
    }


def mean_field_flow(p, rewards, epsilon=1e-4):
    p, rewards = _probabilities(p), _rewards(rewards, epsilon)
    centered = rewards - p @ rewards
    scale = np.sqrt(p @ centered**2 + epsilon**2)
    h = p * centered / scale if scale else np.zeros(4)
    return {
        "H_infinity": h.tolist(),
        "ode_drift": (p * (h - p @ h)).tolist(),
        "method": "population_reward_moments",
    }


def validity_purity_derivative(p, k, epsilon=1e-4):
    """Exact finite-K ODE dqX under reward (1,1,1,0), self-derived.

    H_valid=a*p_valid by exchangeability, hence dqX=a*v*qX*(qX-sum(q²)).
    This is the infinitesimal expected update, not a finite-step guarantee.
    """
    p = _probabilities(p)
    v = float(p[:3].sum())
    if not v:
        return None
    _, _, first, _, _ = _moments(p, [1, 1, 1, 0], k, epsilon)
    q = p[:3] / v
    a = first[:3].sum() / v
    return float(a * v * q[0] * (q[0] - q @ q))


def self_derived_nonclosure_example():
    """Same nonlinear shared policy at +/-1 has unequal probability derivatives."""
    p = np.array([0.15, 0.2, 0.35, 0.3])
    h = np.array(finite_group_flow(p, [3, 1, 1, 0], 8, 0.01)["H_K"])
    jacobian = np.diag(p) - np.outer(p, p)
    derivative_minus, derivative_plus = (
        np.array([-2.0, 2.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 0.0, 0.0]),
    )
    return {
        "source": "self_derived_fixture",
        "theory_review_verified": False,
        "logit_policy": "log(p0)+(theta^2-1)*(1,theta,0,0)",
        "p0": p.tolist(),
        "p_minus": p.tolist(),
        "p_plus": p.tolist(),
        "theta_values": [-1, 1],
        "dp_minus": (jacobian @ derivative_minus * (derivative_minus @ h)).tolist(),
        "dp_plus": (jacobian @ derivative_plus * (derivative_plus @ h)).tolist(),
        "interpretation": "same probabilities and rewards do not close shared-parameter dynamics",
        "missing_acceptance": "exact THEORY_REVIEW counterexample cannot be checked without source",
    }


def _monte_carlo_convergence(p, rewards, k, epsilon, samples, seed):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(k, p, size=samples)
    updates, _ = group_updates(counts, rewards, epsilon)
    estimate = updates.mean(axis=0)
    standard_error = updates.std(axis=0, ddof=1) / math.sqrt(samples)
    target = np.array(mean_field_flow(p, rewards, epsilon)["H_infinity"])
    return {
        "K": k,
        "method": "independent_monte_carlo",
        "samples": samples,
        "sample_seed": seed,
        "H_estimate": estimate.tolist(),
        "H_standard_error": standard_error.tolist(),
        "H_infinity": target.tolist(),
        "l2_error_to_meanfield": float(np.linalg.norm(estimate - target)),
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _acceptance_checks():
    from .decoding_audit import toy_decoder_audit

    p = [0.15, 0.2, 0.35, 0.3]
    ordering_a = finite_group_flow(p, [3, 2, 1, 0], 2, 0.05, 0)
    ordering_b = finite_group_flow(p, [99, 10, 0.1, -1], 2, 0.05, 0)
    tied = finite_group_flow(p, [2, 0, 0, 0], 2, 0.05, 0)
    untied = finite_group_flow(p, [3, 1, 1, 0], 2, 0.05, 0)
    shift_a = finite_group_flow([0.2, 0.3, 0.5, 0], [2, 0, 0, 0], 8, 0.05)
    shift_b = finite_group_flow([0.2, 0.3, 0.5, 0], [3, 1, 1, 0], 8, 0.05)
    smooth_a = finite_group_flow(p, [3, 2, 1, 0], 2, 0.05, 0.2)
    smooth_b = finite_group_flow(p, [99, 10, 0.1, -1], 2, 0.05, 0.2)
    eta_rows = [finite_group_flow(p, [3, 1, 1, 0], 8, eta) for eta in [0.05, 0.01, 0.001]]
    derivative = validity_purity_derivative(p, 8)
    small = finite_group_flow(p, [1, 1, 1, 0], 8, 1e-5)
    numerical = small["expected_delta_q"] / 1e-5
    return {
        "K2_strict_ordering_invariance": bool(np.allclose(ordering_a["H_K"], ordering_b["H_K"])),
        "lambda_zero_tie_boundary_changes_flow": not bool(np.allclose(tied["H_K"], untied["H_K"])),
        "epsilon_positive_spacing_sensitivity": not bool(
            np.allclose(smooth_a["H_K"], smooth_b["H_K"])
        ),
        "all_valid_constant_shift_invariant": bool(
            np.allclose(shift_a["exact_p_next"], shift_b["exact_p_next"])
        ),
        "eta_error_decreases": all(
            a["first_order_error"] > b["first_order_error"]
            and a["second_order_error"] > b["second_order_error"]
            for a, b in itertools.pairwise(eta_rows)
        ),
        "validity_q_derivative_abs_error": abs(derivative - numerical),
        "validity_q_derivative_passed": abs(derivative - numerical) < 1e-6,
        "decoder_toy_passed": toy_decoder_audit()["parser_language_agrees_with_fsa"],
        "exact_theory_review_nonclosure": "PENDING_MISSING_SOURCE",
    }


def _plot_benchmark(out, rows):
    """Every displayed point is selected directly from finite_group_flow.csv."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    selected = [r for r in rows if r["state_id"] == 0 and r["arm"] == "X_VALID"]
    sizes = sorted({r["K"] for r in selected})
    group_size = 8 if 8 in sizes else sizes[-1]
    response = sorted([r for r in selected if r["K"] == group_size], key=lambda r: r["eta"])
    paths = []
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for category, index in zip(["X", "S", "W", "I"], range(4), strict=True):
        axis.plot(
            [r["eta"] for r in response],
            [r["exact_p_next"][index] for r in response],
            marker="o",
            label=category,
        )
    axis.set(
        xlabel="One-step eta (not training time)",
        ylabel="E[p after one update]",
        title=f"CPU toy: exact finite-step probabilities, K={group_size}",
    )
    axis.legend(ncol=4)
    fig.savefig(figures / "probability_trajectories.png", dpi=160)
    plt.close(fig)
    paths.append(figures / "probability_trajectories.png")
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for k in sizes:
        subset = sorted([r for r in selected if r["K"] == k], key=lambda r: r["eta"])
        axis.loglog(
            [r["eta"] for r in subset],
            [r["first_order_error"] for r in subset],
            marker="o",
            label=f"1st order K={k}",
        )
        axis.loglog(
            [r["eta"] for r in subset],
            [r["second_order_error"] for r in subset],
            linestyle="--",
            label=f"2nd order K={k}",
        )
    axis.set(
        xlabel="One-step eta",
        ylabel="L2 error against exact E[p+]",
        title="CPU toy: local ODE / stochastic Taylor errors",
    )
    axis.legend(fontsize=7, ncol=2)
    fig.savefig(figures / "finite_step_error.png", dpi=160)
    plt.close(fig)
    paths.append(figures / "finite_step_error.png")
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    eta = min(r["eta"] for r in rows)
    subset = sorted(
        [r for r in rows if r["state_id"] == 0 and r["arm"] == "X_BASE" and r["eta"] == eta],
        key=lambda r: r["K"],
    )
    for name, key in [
        ("No X", "no_X_probability"),
        ("Zero reward variance", "zero_variance_probability"),
        ("Informative X-binary group", "X_binary_informative_probability"),
    ]:
        axis.plot([r["K"] for r in subset], [r[key] for r in subset], marker="o", label=name)
    axis.set(
        xlabel="Group size K",
        ylabel="Exact group-event probability",
        ylim=(-0.02, 1.02),
        title="CPU toy: support and zero-variance groups, X_BASE",
    )
    axis.legend()
    fig.savefig(figures / "group_support.png", dpi=160)
    plt.close(fig)
    return [*paths, figures / "group_support.png"]


def run_benchmark(config, out):
    from .core import canonical_hash, phase_artifacts, write_json
    from .decoding_audit import toy_decoder_audit
    from .sensitivity import audit_derivative

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    options = config.get("tabular", {})
    probabilities = options.get(
        "probabilities",
        [
            [0.02, 0.18, 0.3, 0.5],
            [0.15, 0.2, 0.35, 0.3],
            [0.6, 0.1, 0.2, 0.1],
            [0.0, 0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
    )
    sizes = options.get("K", [2, 4, 8, 16, 32])
    etas = options.get("eta", [0.001, 0.01, 0.05])
    lambdas = options.get("lambdas", [0, 0.1, 0.25, 0.5, 1, 2, 4])
    if "group_sizes" in options or "etas" in options:
        raise ValueError("tabular config keys are K and eta; legacy aliases are not accepted")
    boundaries = [
        [0.0, 0.2, 0.3, 0.5],
        [0.2, 0.3, 0.5, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    probabilities = [*probabilities, *(p for p in boundaries if p not in probabilities)]
    epsilon = options.get("epsilon", 1e-4)
    arms = {
        "A_BASE": [2, 2, 0, 0],
        "X_BASE": [2, 0, 0, 0],
        "A_VALID": [3, 3, 1, 0],
        "X_VALID": [3, 1, 1, 0],
    }
    rows, convergence = [], []
    for index, p in enumerate(probabilities):
        for arm, rewards in arms.items():
            target = np.array(mean_field_flow(p, rewards, epsilon)["H_infinity"])
            for k in sizes:
                results = [finite_group_flow(p, rewards, k, eta, epsilon) for eta in etas]
                rows.extend({"state_id": index, "arm": arm, **row} for row in results)
                convergence.append(
                    {
                        "state_id": index,
                        "arm": arm,
                        "K": k,
                        "method": "exact_multinomial_enumeration",
                        "H_estimate": results[0]["H_K"],
                        "H_infinity": target.tolist(),
                        "l2_error_to_meanfield": float(
                            np.linalg.norm(np.array(results[0]["H_K"]) - target)
                        ),
                    }
                )
            if index < 3:
                for k in options.get("meanfield_group_sizes", [64, 128]):
                    convergence.append(
                        {
                            "state_id": index,
                            "arm": arm,
                            **_monte_carlo_convergence(
                                p,
                                rewards,
                                k,
                                epsilon,
                                options.get("monte_carlo_samples", 100000),
                                17000 + 100 * index + 10 * list(arms).index(arm) + k,
                            ),
                        }
                    )
    derivatives = [
        audit_derivative([2, 0, 0, 0, 2, 0, 0, 0], [1, 1, 1, 0, 1, 0, 1, 1], lam, epsilon)
        for lam in lambdas
    ]
    outputs = {
        "meanfield_convergence.json": convergence,
        "derivative_audit.json": derivatives,
        "decoder_toy_audit.json": toy_decoder_audit(),
        "nonclosure_self_derived.json": self_derived_nonclosure_example(),
        "acceptance_checks.json": _acceptance_checks(),
    }
    acceptance = {
        **outputs["acceptance_checks.json"],
        "maximum_weight_sum_error": max(abs(r["weight_sum"] - 1) for r in rows),
        "maximum_update_sum_error": max(abs(sum(r["H_K"])) for r in rows),
        "maximum_delta_probability_sum_error": max(abs(sum(r["delta_p"])) for r in rows),
        "autograd_max_abs_error": max(d["autograd_max_abs_error"] for d in derivatives),
        "smallest_fd_step_all_passed": all(
            d["finite_differences"][-1]["passes_mixed_tolerance"] for d in derivatives
        ),
    }
    outputs = {**outputs, "acceptance_checks.json": acceptance}
    for name, value in outputs.items():
        write_json(out / name, value)
    _write_csv(out / "finite_group_flow.csv", rows)
    charts = _plot_benchmark(out, rows)
    write_json(
        out / "figure_sources.json",
        {
            "probability_trajectories": {
                "table": "finite_group_flow.csv",
                "state_id": 0,
                "arm": "X_VALID",
                "K": 8 if 8 in sizes else max(sizes),
                "x": "eta",
                "y": "exact_p_next",
                "scope": "one_step_cpu_toy_not_training_trajectory",
            },
            "finite_step_error": {
                "table": "finite_group_flow.csv",
                "state_id": 0,
                "arm": "X_VALID",
                "x": "eta",
                "series": "K",
                "y": ["first_order_error", "second_order_error"],
            },
            "group_support": {
                "table": "finite_group_flow.csv",
                "state_id": 0,
                "arm": "X_BASE",
                "eta": min(etas),
                "x": "K",
                "y": [
                    "no_X_probability",
                    "zero_variance_probability",
                    "X_binary_informative_probability",
                ],
            },
        },
    )
    details = {
        "scope": "toy_mathematical_validation",
        "rows": len(rows),
        "config_hash": canonical_hash(config),
        "formula_provenance": "self_derived_explicit_definitions_theory_review_source_unavailable",
        "H_K_definition": "E[U], U=(N/K)*group_zscore",
        "Xi_K_definition": "E[U U^T] (not covariance)",
        "B_eta_K_definition": "E[softmax(log p+eta U)]-p",
        "q_aggregation": "expected_q_next=E[q(p_next)]; pooled_q_next=q(E[p_next])",
        "ode_comparison": (
            "first-order local ODE Euler prediction; second-order one-step stochastic Taylor"
        ),
        "meanfield_interpretation": "O(1/K) is asymptotic order, no required finite-scan slope",
        "normalization": (
            "sequence-score group average; training eta must account for fixed Lnorm=64"
        ),
        "pending": [
            "exact THEORY_REVIEW nonclosure construction source",
            "all real-model GPU experiments",
        ],
    }
    phase_artifacts(
        out,
        "P2",
        "PARTIAL",
        details,
        [out / name for name in outputs]
        + [out / "finite_group_flow.csv", out / "figure_sources.json", *charts],
    )
    with (out / "report.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## 数学口径与未完成项\n\n这些结果仅为 CPU toy 数学与实现验证。\n\n"
            "本地自推定义: U=(N/K)A, H_K=E[U], Xi_K=E[UUᵀ]; Xi 不是协方差。"
            "有限步计算 E[softmax(log p+ηU)], 不是 softmax(log p+ηE[U])。"
            "同时报告 E[q(p⁺)] 与 q(E[p⁺]), 两者不能互换。\n\n"
            "一阶比较为 ODE 的局部 Euler 项; 二阶为随机单步更新的 Taylor 项。"
            "K≤32 使用完整枚举; 64/128 使用独立 Monte Carlo 并保存种子、样本数和标准误。"
            "O(1/K) 是渐近量级, 未强制拟合斜率等于 -1。\n\n"
            "THEORY_REVIEW 未提供, H_K/Xi/B 的符号对应与其指定 θ=±1 构造尚未核对。"
            "另附同一非线性共享策略的自推非闭合构造, 不能称为已验证原文反例。"
            "此缺失验收项使 P2 保持 PARTIAL。所有模型实验待 NTU GPU 环境。\n"
        )
    return details


def main(argv=None):
    from .core import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": "P2",
                    "device": "cpu",
                    "prompt_count": 0,
                    "rollout_count": 0,
                    "max_tokens": 0,
                    "forward_count": 0,
                    "backward_count": 0,
                    "budget": "exact enumeration K<=32 plus tagged Monte Carlo K=64/128",
                }
            )
        )
        return
    run_benchmark(config, args.out)


if __name__ == "__main__":
    main()
