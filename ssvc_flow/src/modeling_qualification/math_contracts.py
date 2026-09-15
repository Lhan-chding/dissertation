"""FP64 probability, sampling and local-response contracts for M0.

The four categories describe observable events. Conditional coordinates do not
add information, and unobserved sample categories are not structural zeros.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

CATEGORIES = ("X", "S", "W", "I")


def helmert() -> np.ndarray:
    return np.array(
        [
            [1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
            [-1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
            [0, -2 / math.sqrt(6), 1 / math.sqrt(12)],
            [0, 0, -3 / math.sqrt(12)],
        ],
        dtype=np.float64,
    )


def probability(p, *, categories=CATEGORIES) -> np.ndarray:
    if tuple(categories) != CATEGORIES:
        raise ValueError("Category order must be X,S,W,I")
    p = np.asarray(p, dtype=np.float64)
    if p.ndim < 1 or p.shape[-1] != 4 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Finite nonnegative four-event probabilities required")
    if not np.allclose(p.sum(axis=-1), 1, rtol=0, atol=1e-12):
        raise ValueError("Probability mass must sum to one")
    return p


def to_helmert(p, *, validate=True) -> np.ndarray:
    p = probability(p) if validate else np.asarray(p, np.float64)
    if p.ndim < 1 or p.shape[-1] != 4 or not np.isfinite(p).all():
        raise ValueError("Expected finite final category dimension 4")
    return p @ helmert()


def from_helmert(y) -> np.ndarray:
    """Linear inverse; intentionally preserves out-of-simplex predictions."""
    y = np.asarray(y, np.float64)
    if y.ndim < 1 or y.shape[-1] != 3 or not np.isfinite(y).all():
        raise ValueError("Expected finite final contrast dimension 3")
    return 0.25 + y @ helmert().T


def to_nested(p) -> dict:
    p = probability(p)
    if p.shape != (4,):
        raise ValueError("One probability vector required")
    valid, errors = float(p[:3].sum()), float(p[1:3].sum())
    return {
        "v": valid,
        "q": float(p[0] / valid) if valid > 0 else None,
        "s": float(p[1] / errors) if errors > 0 else None,
        "q_defined": valid > 0,
        "s_defined": errors > 0,
        "q_status": "DEFINED" if valid > 0 else "UNDEFINED_ZERO_MASS",
        "s_status": "DEFINED" if errors > 0 else "UNDEFINED_ZERO_MASS",
    }


def from_nested(v: float, q: float | None, s: float | None) -> np.ndarray:
    if v is None:
        raise ValueError("v must be defined")
    for name, value in (("v", v), ("q", q), ("s", s)):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must lie in [0,1]")
    if v == 0:
        return np.array([0.0, 0.0, 0.0, 1.0], np.float64)
    if q is None:
        raise ValueError("q must be defined for positive valid mass")
    errors = v * (1 - q)
    if errors == 0:
        return np.array([v, 0.0, 0.0, 1 - v], np.float64)
    if s is None:
        raise ValueError("s must be defined for positive valid-error mass")
    return probability([v * q, errors * s, errors * (1 - s), 1 - v])


def nested_jacobian(v, q, s) -> np.ndarray:
    from_nested(v, q, s)
    return np.array(
        [
            [q, v, 0.0],
            [(1 - q) * s, -v * s, v * (1 - q)],
            [(1 - q) * (1 - s), -v * (1 - s), -v * (1 - q)],
            [-1.0, 0.0, 0.0],
        ],
        np.float64,
    )


def nested_fisher(v, q, s) -> np.ndarray:
    if any(not math.isfinite(x) or not 0 < x < 1 for x in (v, q, s)):
        raise ValueError("Interior Fisher formula excludes boundaries")
    return np.diag(
        np.array([1 / (v * (1 - v)), v / (q * (1 - q)), v * (1 - q) / (s * (1 - s))], np.float64)
    )


def _counts(counts, *, allow_empty=False) -> np.ndarray:
    values = list(counts)
    if len(values) != 4 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0
        for v in values
    ):
        raise ValueError("Four nonnegative integer counts required")
    if not allow_empty and sum(values) == 0:
        raise ValueError("Positive count total required")
    return np.array(values, dtype=np.int64)


def mle(counts: Sequence[int]) -> dict:
    counts = _counts(counts)
    p = counts.astype(np.float64) / counts.sum()
    nested = to_nested(p)
    for coordinate in ("q", "s"):
        if nested[coordinate] is None:
            nested[f"{coordinate}_status"] = "UNOBSERVED_CONDITION"
    return {
        "p": p,
        "counts": counts.tolist(),
        "n": int(counts.sum()),
        **nested,
        "category_status": {
            category: "OBSERVED_EVENT" if count else "UNOBSERVED_EVENT"
            for category, count in zip(CATEGORIES, counts, strict=True)
        },
    }


def clopper_pearson(k: int, n: int, alpha=0.05) -> dict:
    if (
        any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (k, n))
        or not 0 <= k <= n
    ):
        raise ValueError("Binomial counts must satisfy 0 <= k <= n")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if n == 0:
        return {
            "k": 0,
            "n": 0,
            "estimate": None,
            "interval": [0.0, 1.0],
            "alpha": float(alpha),
            "status": "UNOBSERVED_CONDITION",
        }
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return {
        "k": int(k),
        "n": int(n),
        "estimate": float(k / n),
        "interval": [lo, hi],
        "alpha": float(alpha),
        "status": "POINTWISE_FIXED_SAMPLE",
    }


def conditional_intervals(counts, alpha=0.05, simultaneous=False) -> dict:
    counts = _counts(counts, allow_empty=True)
    n = int(counts.sum())
    n_valid, n_errors = int(counts[:3].sum()), int(counts[1:3].sum())
    local_alpha = alpha / 7 if simultaneous else alpha
    return {
        "categories": {
            cat: clopper_pearson(int(k), n, local_alpha)
            for cat, k in zip(CATEGORIES, counts, strict=True)
        },
        "v": clopper_pearson(n_valid, n, local_alpha),
        "q": clopper_pearson(int(counts[0]), n_valid, local_alpha),
        "s": clopper_pearson(int(counts[1]), n_errors, local_alpha),
        "coverage_scope": "BONFERRONI_WITHIN_ONE_COUNT_VECTOR"
        if simultaneous
        else "POINTWISE_ONLY",
        "family_alpha": float(alpha),
        "number_of_intervals": 7,
    }


def weighted_group_q(p, weights, prompt_ids, weight_prompt_ids) -> float | None:
    p = probability(p)
    weights = np.asarray(weights, np.float64)
    if list(prompt_ids) != list(weight_prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Prompt/weight identity order must match and be unique")
    if p.ndim != 2 or weights.shape != (len(p),) or len(prompt_ids) != len(p):
        raise ValueError("Aligned per-prompt probabilities and weights required")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Finite nonnegative weights with positive total required")
    denominator = float(weights @ p[:, :3].sum(axis=1))
    return float(weights @ p[:, 0] / denominator) if denominator > 0 else None


def advantages(categories, lam, epsilon=1e-4, no_x_off=False) -> np.ndarray:
    if len(categories) < 2 or any(c not in CATEGORIES for c in categories):
        raise ValueError("Known categories and group size >= 2 required")
    if not math.isfinite(lam) or lam < 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Finite nonnegative lambda and positive epsilon required")
    effective = 0.0 if no_x_off and "X" not in categories else lam
    rewards = np.array([2.0 * (c == "X") + effective * (c != "I") for c in categories], np.float64)
    centered = rewards - rewards.mean()
    return centered / math.sqrt(float(np.mean(centered**2)) + epsilon**2)


def orthogonal_span(deltas, rtol=1e-10, atol=1e-14):
    d = np.asarray(deltas, np.float64)
    if d.ndim != 2 or not np.isfinite(d).all():
        raise ValueError("Finite parameters-by-updates matrix required")
    if any(not math.isfinite(v) or v < 0 for v in (rtol, atol)):
        raise ValueError("Fixed finite nonnegative rank tolerances required")
    if min(d.shape) == 0:
        return np.zeros((d.shape[0], 0), np.float64), np.array([], np.float64)
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    cutoff = max(atol, rtol * float(s[0]))
    return u[:, s > cutoff], s


def truncate_response(r, rank):
    r = np.asarray(r, np.float64)
    if (
        r.ndim != 2
        or not np.isfinite(r).all()
        or type(rank) is not int
        or not 0 <= rank <= min(r.shape)
    ):
        raise ValueError("Invalid response matrix/rank")
    u, s, vt = np.linalg.svd(r, full_matrices=False)
    approx = (u[:, :rank] * s[:rank]) @ vt[:rank]
    return approx, float(s[rank]) if rank < len(s) else 0.0


def project_simplex(p) -> np.ndarray:
    """Euclidean projection; returns a fresh array, never clips source values."""
    p = np.asarray(p, np.float64)
    if p.ndim < 1 or p.shape[-1] != 4 or not np.isfinite(p).all():
        raise ValueError("Finite four-event vectors required")
    values = p.reshape(-1, 4)
    sorted_values = np.sort(values, axis=1)[:, ::-1]
    thresholds = (np.cumsum(sorted_values, axis=1) - 1) / np.arange(1, 5)
    rho = np.sum(sorted_values > thresholds, axis=1) - 1
    theta = thresholds[np.arange(len(values)), rho]
    return np.maximum(values - theta[:, None], 0).reshape(p.shape)


def prediction_metrics(raw, truth) -> dict:
    truth = probability(truth)
    raw = np.asarray(raw, np.float64)
    if raw.shape != truth.shape or not np.isfinite(raw).all():
        raise ValueError("Aligned finite prediction and truth required")
    projected = project_simplex(raw)
    invalid = (
        (raw < -1e-12).any(axis=-1)
        | (raw > 1 + 1e-12).any(axis=-1)
        | (np.abs(raw.sum(axis=-1) - 1) > 1e-12)
    )
    return {
        "raw_event_mae": float(np.mean(np.abs(raw - truth))),
        "projected_event_mae": float(np.mean(np.abs(projected - truth))),
        "raw_invalid_simplex_fraction": float(np.mean(invalid)),
        "raw_negative_probability_max": float(max(0.0, -np.min(raw))),
        "raw_mass_error_max": float(np.max(np.abs(raw.sum(axis=-1) - 1))),
    }


def error_bounds(deltas, basis, *, e=0.0, eps=0.0, kappa=0.0, curvature=0.0) -> dict:
    d, u = np.asarray(deltas, np.float64), np.asarray(basis, np.float64)
    if d.ndim != 2 or u.ndim != 2 or d.shape[1] != u.shape[0]:
        raise ValueError("Expected steps-by-parameters and parameters-by-rank")
    if not np.isfinite(d).all() or not np.isfinite(u).all():
        raise ValueError("Finite arrays required")
    if not np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-12, rtol=1e-12):
        raise ValueError("Basis must be orthonormal")
    if any(not math.isfinite(v) or v < 0 for v in (e, eps, kappa, curvature)):
        raise ValueError("Nonnegative finite error components required")
    projected = d @ u
    perp = d - projected @ u.T
    total = d.sum(axis=0)
    total_proj = total @ u
    total_perp = total - total_proj @ u.T
    net = (
        e
        + eps * np.linalg.norm(total_proj)
        + kappa * np.linalg.norm(total_perp)
        + 0.5 * curvature * float(total @ total)
    )
    path = (
        e
        + eps * np.linalg.norm(projected, axis=1).sum()
        + kappa * np.linalg.norm(perp, axis=1).sum()
        + 0.5 * curvature * np.linalg.norm(d, axis=1).sum() ** 2
    )
    return {"net": float(net), "path": float(path)}


def softmax(logits) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Finite nonempty one-dimensional logits required")
    ex = np.exp(z - z.max())
    return ex / ex.sum()


def event_gradient_affine(matrix, theta, bias, event_weights):
    a, theta, bias, w = (np.asarray(v, np.float64) for v in (matrix, theta, bias, event_weights))
    if (
        a.ndim != 2
        or theta.shape != (a.shape[1],)
        or bias.shape != (a.shape[0],)
        or w.shape != bias.shape
    ):
        raise ValueError("Affine-softmax dimensions must align")
    if any(not np.isfinite(v).all() for v in (a, theta, bias, w)):
        raise ValueError("Finite affine-softmax inputs required")
    p = softmax(a @ theta + bias)
    value = float(p @ w)
    return value, a.T @ (p * (w - value))


def geometry(left, right) -> dict:
    a, b = np.asarray(left, np.float64).reshape(-1), np.asarray(right, np.float64).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparable finite vectors required")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "left_norm": float(na),
        "right_norm": float(nb),
        "distance": float(np.linalg.norm(a - b)),
        "cosine": float(np.clip(a @ b / (na * nb), -1, 1)) if na * nb > 0 else None,
    }


def _save_json(path: Path, data: dict):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_math(config: dict, out: Path) -> dict:
    """Execute M0 FP64 numerical audits and a frozen finite-count fixture panel."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    seed = 202609150
    rng = np.random.default_rng(seed)
    h = helmert()
    reconstruction = max(float(np.max(np.abs(from_helmert(to_helmert(p)) - p))) for p in np.eye(4))
    fisher_errors, fd_errors = [], []
    for x in rng.uniform(0.05, 0.95, size=(100, 3)):
        p, jac = from_nested(*x), nested_jacobian(*x)
        fisher_errors.append(
            float(np.max(np.abs(jac.T @ np.diag(1 / p) @ jac - nested_fisher(*x))))
        )
        fd = np.column_stack(
            [(from_nested(*(x + d)) - from_nested(*(x - d))) / (2e-6) for d in 1e-6 * np.eye(3)]
        )
        fd_errors.append(float(np.max(np.abs(fd - jac))))
    # Delta-probability and contrast regression share the identical design matrix.
    design = rng.normal(size=(30, 5))
    panel = rng.dirichlet(np.ones(4), size=(31, 6))
    delta = panel[1:] - panel[0]
    bp = np.linalg.lstsq(design, delta.reshape(30, -1), rcond=None)[0]
    bh = np.linalg.lstsq(design, to_helmert(delta, validate=False).reshape(30, -1), rcond=None)[0]
    query = rng.normal(size=(4, 5))
    regression_error = float(
        np.max(np.abs((query @ bp).reshape(4, 6, 4) - (query @ bh).reshape(4, 6, 3) @ h.T))
    )
    coverage_min = 1.0
    for n in (1, 8, 32):
        intervals = [clopper_pearson(k, n)["interval"] for k in range(n + 1)]
        for p in np.linspace(0, 1, 101):
            coverage = sum(
                float(binom.pmf(k, n, p))
                for k, (lo, hi) in enumerate(intervals)
                if lo - 1e-14 <= p <= hi + 1e-14
            )
            coverage_min = min(coverage_min, coverage)
    checks = {
        "helmert_vertex_reconstruction_max": reconstruction,
        "fisher_identity_error_max": max(fisher_errors),
        "finite_difference_error_max": max(fd_errors),
        "raw_probability_helmert_regression_error_max": regression_error,
        "cp_exact_binomial_coverage_min_on_fixed_grid": coverage_min,
    }
    passed = (
        reconstruction < 1e-12
        and max(fisher_errors) < 1e-9
        and max(fd_errors) < 1e-9
        and regression_error < 1e-12
        and coverage_min >= 0.95 - 1e-12
    )
    fixtures = {
        "interior": [0.21, 0.13, 0.26, 0.4],
        "rare_X": [0.001, 0.099, 0.8, 0.1],
        "rare_valid": [0.003, 0.003, 0.004, 0.99],
        "rare_valid_error": [0.899, 0.0005, 0.0005, 0.1],
        "all_invalid": [0.0, 0.0, 0.0, 1.0],
        "structural_S_empty": [0.1, 0.0, 0.7, 0.2],
    }
    trials = 512
    rows, raw_counts = [], {}
    for name, p in fixtures.items():
        p = probability(p)
        truth = to_nested(p)
        for n in (16, 64, 256):
            counts = rng.multinomial(n, p, size=trials)
            raw_counts[f"{name}_n{n}"] = counts
            estimated = counts / n
            nested = [mle(c) for c in counts]
            q = np.array([v["q"] if v["q"] is not None else np.nan for v in nested])
            s = np.array([v["s"] if v["s"] is not None else np.nan for v in nested])

            def conditional_rmse(values, actual):
                observed = values[np.isfinite(values)]
                return (
                    float(np.sqrt(np.mean((observed - actual) ** 2)))
                    if actual is not None and len(observed)
                    else None
                )

            rows.append(
                {
                    "fixture": name,
                    "n": n,
                    "replicates": trials,
                    "pX": float(p[0]),
                    "v": truth["v"],
                    "q_oracle": truth["q"],
                    "s_oracle": truth["s"],
                    "frequency_event_mae": float(np.mean(np.abs(estimated - p))),
                    "helmert_roundtrip_max": float(
                        np.max(np.abs(from_helmert(to_helmert(estimated)) - estimated))
                    ),
                    "X_zero_count_fraction": float(np.mean(counts[:, 0] == 0)),
                    "S_zero_count_fraction": float(np.mean(counts[:, 1] == 0)),
                    "q_unobserved_fraction": float(np.mean(~np.isfinite(q))),
                    "s_unobserved_fraction": float(np.mean(~np.isfinite(s))),
                    "q_rmse_when_observed": conditional_rmse(q, truth["q"]),
                    "s_rmse_when_observed": conditional_rmse(s, truth["s"]),
                    "S_structurally_empty_fixture": name == "structural_S_empty",
                }
            )
    with (out / "coordinate_reconstruction.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / "coordinate_counts.npz", **raw_counts)
    counts_hash = hashlib.sha256((out / "coordinate_counts.npz").read_bytes()).hexdigest()
    result = {
        "status": "PASS" if passed else "FAIL",
        "execution_kind": "ANALYTIC_FIXTURE",
        "dtype": "float64",
        "phase": "M0",
        "checks": checks,
        "test_source": "tests/modeling_qualification/test_math_contracts.py",
        "test_names": {
            "helmert_vertex_reconstruction_max": (
                "test_helmert_orthogonality_roundtrip_and_distance"
            ),
            "fisher_identity_error_max": "test_fisher_and_finite_difference_fp64",
            "finite_difference_error_max": "test_fisher_and_finite_difference_fp64",
            "raw_probability_helmert_regression_error_max": (
                "test_linear_regression_probability_helmert_equivalence"
            ),
            "cp_exact_binomial_coverage_min_on_fixed_grid": (
                "test_clopper_pearson_actual_denominators_and_zero"
            ),
        },
        "rng_seed": seed,
        "sampling_fixture_replicates": trials,
        "sampling_fixtures": fixtures,
        "coordinate_counts_sha256": counts_hash,
        "protocol_version": config.get("protocol_version"),
        "interval_scope": (
            "POINTWISE_FIXED_SAMPLE; Bonferroni explicitly available within one count vector"
        ),
        "walltime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Numerical contracts and finite-count fixtures; no real-model or training claim"
        ),
    }
    _save_json(out / "math_results.json", result)
    audit = """# M0 数学审计

事件顺序固定 X/S/W/I。存储原始四计数；Helmert 是正交线性坐标，四概率往返与距离保持；
原概率和 Helmert 对同一更新矩阵的线性回归数值等价。v/q/s 仅作解释。
条件分母为零返回 null：精确概率为零标为 UNDEFINED_ZERO_MASS，
有限样本零分母标为 UNOBSERVED_CONDITION。零计数不证明结构零。

固定策略的多项似然可写为 v^nV (1-v)^nI q^nX (1-q)^nB s^nS (1-s)^nW。
内部点 Fisher 由 D^T diag(1/p) D 得到所给对角式。CP 区间按各自实际分母计算；逐个区间只称点态。
同一四计数向量的四类别加 v/q/s 共七区间使用 alpha/7 的 Bonferroni 分配；
这不是跨 checkpoint 或跨题面板的自动同时覆盖。

对 affine-softmax，令行向量 a_j、均值 μ=Σ p_j a_j、事件对比系数 |w_j|≤1。
单类 Hessian 为 p_j A^T[(e_j-p)(e_j-p)^T−(diag(p)−pp^T)]A。
第一矩阵谱范数≤2，第二矩阵谱范数≤1，因此按 Σ |w_j|p_j≤1 得到 ||∇²f||≤3||A||²。
Taylor 余项≤L||D||²/2，再将起点梯度分成 U 内系数误差及 U 外遗漏，得到 E_net；
应用三角不等式逐项放大得到 E_path。因此每步净位移也能检查中途退化，路径界通常更保守。
M1 用已知 L 同时验证两个界；该全局保证不转移给共享 tanh toy 的经验带。

rank(JQ) 只涉及固定起点、输出表和校准子空间，最优秩 r 的谱误差为下一奇异值。
复制列不增加数值秩。仅给投影和正交残差范数不能识别未覆盖方向的语义响应。
误差向量应分别保存，范数不能相减解释为贡献。

数值结果与固定种子采样原件见 math_results.json、coordinate_reconstruction.csv、
coordinate_counts.npz。
工程测试、Adam 状态恢复与不可用观测访问的测试由完整仓库测试记录共同证明。
"""
    with (out / "MATH_AUDIT_zh.md").open("x", encoding="utf-8") as stream:
        stream.write(audit)
    if not passed:
        raise RuntimeError("M0 numeric contract failed; failure evidence preserved")
    return result
