"""M1 analytic fixtures: identifiable observations, ambiguity and rank controls.

W3 executes small real CPU AdamW steps; all other probability models are
explicit affine-softmax or linear-response fixtures. None load a language model.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from .math_contracts import (
    _save_json,
    error_bounds,
    event_gradient_affine,
    orthogonal_span,
    softmax,
    truncate_response,
)


def witness_w1() -> dict:
    before = np.array([[0.6, 0.1, 0.1, 0.2], [0.1, 0.2, 0.3, 0.4]], np.float64)
    after = before[::-1].copy()
    return {
        "prompt_ids": ["a", "b"],
        "groups": ["g0", "g1"],
        "before": before.tolist(),
        "after": after.tolist(),
        "pooled_change_norm": float(np.linalg.norm(after.mean(axis=0) - before.mean(axis=0))),
        "per_prompt_change_norm": float(np.linalg.norm(after - before)),
        "group_changes": (after - before).tolist(),
        "identity_matching": "original prompt identity; no optimal transport relabeling",
        "conclusion": "Pooled observations do not identify per-prompt or group changes",
    }


def witness_w2() -> dict:
    coefficient = np.array([1.0, 0.0, 0.0, 0.0], np.float64)
    start = softmax(np.zeros(4))
    d = 0.1
    plus, minus = softmax(coefficient * d), softmax(-coefficient * d)
    return {
        "theta": 0.0,
        "d": d,
        "coefficients_plus": coefficient.tolist(),
        "coefficients_minus": (-coefficient).tolist(),
        "p_start_plus": start.tolist(),
        "p_start_minus": start.tolist(),
        "p_after_plus": plus.tolist(),
        "p_after_minus": minus.tolist(),
        "delta_pX_plus": float(plus[0] - start[0]),
        "delta_pX_minus": float(minus[0] - start[0]),
        "same_update": True,
        "conclusion": "Current probabilities alone do not identify local response",
    }


def _adam_history(history):
    """Construct legal Adam moments while holding theta at exactly zero.

    Zero-learning-rate history steps update Adam's first/second moments and
    step counter without changing theta. Set the shared learning rate to .01
    only for the compared future step. This avoids assuming that subtracting
    accumulated floating-point updates exactly inverts their replay, which
    differed across CPU backends in the original fixture.
    """
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64, device="cpu"))
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False
    )
    for gradient in history:
        parameter.grad = torch.tensor([gradient], dtype=torch.float64)
        optimizer.step()
    optimizer.param_groups[0]["lr"] = 0.01
    return parameter, optimizer, 0.0


def witness_w3() -> dict:
    rng_state = torch.random.get_rng_state().clone()
    histories = [[-0.8, -0.8, -0.8], [0.8, 0.8, 0.8]]
    parameters, optimizers, initials = [], [], []
    for history in histories:
        p, opt, initial = _adam_history(history)
        parameters.append(p)
        optimizers.append(opt)
        initials.append(initial)
    current = 0.1
    starts = [float(p.detach()[0]) for p in parameters]
    snapshots = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
    updates, after, linear_errors, p_only_errors, moments = [], [], [], [], []
    for p, opt, state in zip(parameters, optimizers, snapshots, strict=True):
        first = state["state"][0]
        moments.append(
            {
                "step": int(first["step"]),
                "exp_avg": float(first["exp_avg"][0]),
                "exp_avg_sq": float(first["exp_avg_sq"][0]),
            }
        )
        theta = float(p.detach()[0])
        p.grad = torch.tensor([current], dtype=torch.float64)
        opt.step()
        d = float(p.detach()[0]) - theta
        truth = softmax([float(p.detach()[0]), 0.0, 0.0, 0.0])
        prior = softmax([theta, 0.0, 0.0, 0.0])
        derivative = prior * (np.array([1.0, 0.0, 0.0, 0.0]) - prior[0])
        updates.append(d)
        after.append(truth.tolist())
        linear_errors.append(float(np.max(np.abs(truth - prior - derivative * d))))
        p_only_errors.append(float(np.max(np.abs(truth - prior))))
    # Explicit state restoration must replay the same next step.
    p, opt = parameters[0], optimizers[0]
    with torch.no_grad():
        p.fill_(starts[0])
    opt.load_state_dict(copy.deepcopy(snapshots[0]))
    p.grad = torch.tensor([current], dtype=torch.float64)
    opt.step()
    replay_d = float(p.detach()[0]) - starts[0]
    # Zero tensor increments the step and retains momentum; None skips it.
    zero_none = []
    for gradient in (torch.zeros(1, dtype=torch.float64), None):
        with torch.no_grad():
            p.fill_(starts[0])
        opt.load_state_dict(copy.deepcopy(snapshots[0]))
        p.grad = gradient
        opt.step()
        zero_none.append(float(p.detach()[0]) - starts[0])
    return {
        "execution_kind": "ANALYTIC_FIXTURE",
        "optimizer_execution": "REAL_CPU_ADAMW",
        "dtype": "float64",
        "history_gradients": histories,
        "history_initial_parameters": initials,
        "history_learning_rate": 0.0,
        "history_construction": "LEGAL_ADAMW_STEPS_WITH_ZERO_LR_THEN_SHARED_LR_0.01",
        "theta_starts": starts,
        "same_theta": starts[0] == starts[1],
        "same_current_gradient": True,
        "current_gradient": current,
        "optimizer": {
            "name": "AdamW",
            "lr": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "legal_history_states": moments,
        "updates": updates,
        "p_after": after,
        "actual_d_first_order_errors": linear_errors,
        "p_only_errors": p_only_errors,
        "same_history_replay_equal": replay_d == updates[0],
        "explicit_zero_gradient_update": zero_none[0],
        "none_gradient_update": zero_none[1],
        "global_rng_unchanged": bool(torch.equal(rng_state, torch.random.get_rng_state())),
        "conclusion": (
            "Adam history identifies future d; after observing actual d "
            "its momentum effect is already in that input"
        ),
    }


def witness_w4() -> dict:
    u = np.array([[1.0], [0.0]], np.float64)
    rows = []
    for h in (0.01, 0.1, 0.5):
        plus = np.array([0.0, h], np.float64)
        minus = -plus
        pp, pm = softmax([plus.sum(), 0.0, 0.0, 0.0]), softmax([minus.sum(), 0.0, 0.0, 0.0])
        rows.append(
            {
                "h": h,
                "d_plus": plus.tolist(),
                "d_minus": minus.tolist(),
                "projection_plus": (u.T @ plus).tolist(),
                "projection_minus": (u.T @ minus).tolist(),
                "residual_norm_plus": float(np.linalg.norm(plus - u @ (u.T @ plus))),
                "residual_norm_minus": float(np.linalg.norm(minus - u @ (u.T @ minus))),
                "pX_plus": float(pp[0]),
                "pX_minus": float(pm[0]),
                "delta_pX_plus": float(pp[0] - 0.25),
                "delta_pX_minus": float(pm[0] - 0.25),
                "minimax_absolute_error_lower_bound": float((pp[0] - pm[0]) / 2),
            }
        )
    return {
        "A": [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "theta": [0.0, 0.0],
        "U": u.tolist(),
        "rows": rows,
        "conclusion": "Projection and residual norm do not identify unseen semantic direction",
    }


def witness_w5() -> dict:
    # Six groups, one prompt per group; X-I tangent directions have unit norm.
    left = np.zeros((24, 5), np.float64)
    for group in range(5):
        left[4 * group, group] = 1 / np.sqrt(2)
        left[4 * group + 3, group] = -1 / np.sqrt(2)
    calibration = np.eye(5, dtype=np.float64)
    duplicated = np.column_stack([calibration, calibration[:, [0, 1, 0, 3]]])
    singular = np.array([1.0, 0.8, 0.6, 0.4, 0.0], np.float64)
    response = left * singular
    exact_rank = orthogonal_span(response)[0].shape[1]
    rank_errors = []
    for rank in range(6):
        approx, tail = truncate_response(response, rank)
        rank_errors.append(
            {
                "rank": rank,
                "spectral_error": float(np.linalg.norm(response - approx, 2)),
                "next_singular_value": tail,
            }
        )
    low_energy_rows = []
    amplitude, threshold = 0.2, 0.01
    for fifth in (0.02, 0.1):
        singular = np.array([1.0, 0.8, 0.6, 0.4, fifth], np.float64)
        response = left * singular
        energy_rank = int(np.searchsorted(np.cumsum(singular**2) / np.sum(singular**2), 0.95) + 1)
        evaluation = amplitude * np.eye(5, dtype=np.float64)
        exact_changes = (evaluation @ response.T).reshape(5, 6, 4)
        rank_rows = []
        for rank in range(6):
            approx, _ = truncate_response(response, rank)
            residuals = (evaluation @ (response - approx).T).reshape(5, 6, 4)
            rank_rows.append(
                {"rank": rank, "worst_group_pX_error": float(np.max(np.abs(residuals[:, :, 0])))}
            )
        group_rank = next(
            row["rank"] for row in rank_rows if row["worst_group_pX_error"] <= threshold
        )
        energy_error = rank_rows[energy_rank]["worst_group_pX_error"]
        low_energy_rows.append(
            {
                "fifth_singular_value": fifth,
                "singular_values": singular.tolist(),
                "amplitude": amplitude,
                "group_threshold": threshold,
                "affected_group": 4,
                "R": response.tolist(),
                "evaluation_updates": evaluation.tolist(),
                "exact_probability_changes": exact_changes.tolist(),
                "energy_fraction": 0.95,
                "energy_rank": energy_rank,
                "energy_rule_worst_group_pX_error": energy_error,
                "energy_rule_passes_group_threshold": energy_error <= threshold,
                "group_selected_rank": group_rank,
                "group_selected_worst_error": rank_rows[group_rank]["worst_group_pX_error"],
                "rank_rows": rank_rows,
            }
        )
    return {
        "exact_rank4": {
            "singular_values": [1.0, 0.8, 0.6, 0.4, 0.0],
            "response_rank": exact_rank,
            "update_rank_before": orthogonal_span(calibration)[0].shape[1],
            "update_rank_after_duplication": orthogonal_span(duplicated)[0].shape[1],
            "calibration_updates": calibration.tolist(),
            "duplicated_updates": duplicated.tolist(),
            "response_left_singular_vectors": left.tolist(),
            "rank_errors": rank_errors,
        },
        "low_energy_rows": low_energy_rows,
        "baseline_probabilities": np.full((6, 4), 0.25, dtype=np.float64).tolist(),
        "design": "Both fifth-direction amplitudes, group 4, and threshold fixed before evaluation",
        "conclusion": (
            "Energy criteria and worst-group criteria can agree or differ; "
            "no universal failure claim"
        ),
    }


def witness_w6() -> dict:
    matrix = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], np.float64)
    u = np.array([[1.0], [0.0]], np.float64)
    theta, bias, weights = np.zeros(2), np.zeros(4), np.array([1.0, 0.0, 0.0, 0.0])
    start, gradient = event_gradient_affine(matrix, theta, bias, weights)
    coefficient = u.T @ gradient
    kappa = float(np.linalg.norm(gradient - u @ coefficient))
    curvature = float(3 * np.linalg.norm(matrix, 2) ** 2)
    paths = {
        "out_and_return": np.array([[0.0, 0.01], [0.0, -0.01]], np.float64),
        "inside_then_outside": np.array([[0.01, 0.0], [0.0, 0.01], [0.0, 0.01]], np.float64),
    }
    records = {}
    for name, steps in paths.items():
        rows = []
        for t in range(1, len(steps) + 1):
            total = steps[:t].sum(axis=0)
            truth, _ = event_gradient_affine(matrix, theta + total, bias, weights)
            prediction = float(start + coefficient @ (u.T @ total))
            bound = error_bounds(steps[:t], u, kappa=kappa, curvature=curvature)
            wrong = error_bounds(steps[:t], u, kappa=0.0, curvature=curvature)
            error = abs(truth - prediction)
            rows.append(
                {
                    "t": t,
                    "step": steps[t - 1].tolist(),
                    "net_displacement": total.tolist(),
                    "truth_pX": truth,
                    "prediction_pX": prediction,
                    "absolute_error": error,
                    "net_bound": bound["net"],
                    "path_bound": bound["path"],
                    "wrong_zero_omission_net_bound": wrong["net"],
                    "wrong_model_violated": error > wrong["net"] + 1e-14,
                }
            )
        records[name] = rows
    excursion = records["out_and_return"]
    return {
        "A": matrix.tolist(),
        "bias": bias.tolist(),
        "theta": theta.tolist(),
        "U": u.tolist(),
        "event_weights": weights.tolist(),
        "start_pX": start,
        "gradient": gradient.tolist(),
        "subspace_coefficient": coefficient.tolist(),
        "kappa": kappa,
        "global_L": curvature,
        "paths": records,
        "endpoint_only_misses_excursion": excursion[-1]["absolute_error"] == 0
        and excursion[0]["absolute_error"] > 0,
        "wrong_zero_omission_model_broken": any(
            row["wrong_model_violated"] for rows in records.values() for row in rows
        ),
        "conclusion": (
            "Per-step net displacement detects excursions; endpoint-only checking loses them, "
            "and path bounds are conservative"
        ),
    }


def run_witnesses(config: dict, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    witnesses = {
        "W1": witness_w1(),
        "W2": witness_w2(),
        "W3": witness_w3(),
        "W4": witness_w4(),
        "W5": witness_w5(),
        "W6": witness_w6(),
    }
    w = witnesses
    checks = {
        "W1": w["W1"]["pooled_change_norm"] == 0 and w["W1"]["per_prompt_change_norm"] > 0,
        "W2": w["W2"]["delta_pX_plus"] > 0 > w["W2"]["delta_pX_minus"],
        "W3": w["W3"]["same_theta"]
        and w["W3"]["updates"][0] * w["W3"]["updates"][1] < 0
        and w["W3"]["same_history_replay_equal"],
        "W4": all(row["delta_pX_plus"] > 0 > row["delta_pX_minus"] for row in w["W4"]["rows"]),
        "W5": w["W5"]["exact_rank4"]["response_rank"] == 4
        and {row["energy_rule_passes_group_threshold"] for row in w["W5"]["low_energy_rows"]}
        == {True, False},
        "W6": w["W6"]["wrong_zero_omission_model_broken"]
        and all(
            row["absolute_error"] <= row["net_bound"] + 1e-14 <= row["path_bound"] + 2e-14
            for rows in w["W6"]["paths"].values()
            for row in rows
        ),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "phase": "M1",
        "execution_kind": "ANALYTIC_FIXTURE",
        "dtype": "float64",
        "checks": checks,
        "test_source": "tests/modeling_qualification/test_witnesses.py",
        "test_names": {
            f"W{i}": name
            for i, name in enumerate(
                [
                    "test_w1_keeps_prompt_identity",
                    "test_w2_equal_present_probabilities_do_not_identify_response",
                    "test_w3_legal_adam_histories_same_parameter_gradient",
                    "test_w4_same_summary_opposite_semantics_and_minimax_bound",
                    "test_w5_rank_and_predefined_low_energy_positive_negative_controls",
                    "test_w6_intermediate_net_path_and_wrong_omission_model",
                ],
                1,
            )
        },
        "witnesses": witnesses,
        "protocol_version": config.get("protocol_version"),
        "walltime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Constructed qualification fixtures; "
            "not spontaneous discoveries in a real language model"
        ),
    }
    _save_json(out / "identifiability_witnesses.json", result)
    if not all(checks.values()):
        raise RuntimeError("M1 witness self-check failed; failure evidence preserved")
    return result
