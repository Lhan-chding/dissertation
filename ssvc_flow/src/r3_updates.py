"""Actual isolated R3 Adam forks; reuse requires measured direct comparisons."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

from .core import file_hash, frozen_writer, write_json
from .fork_gradients import (
    apply_gradient_update,
    collect_category_scores,
    combine_category_gradients,
    compare_gradient_bundles,
    direct_loss_gradients,
)
from .optimizer_fork import (
    capture_state,
    parameter_hash,
    restore_state,
    save_checkpoint,
    state_hash,
)
from .sensitivity import advantage_derivative

LAMBDAS = (0.0, 0.01, 0.25, 1.0, 2.0)
GRAD_ATOL, GRAD_RTOL = 1e-6, 1e-5
STATE_ATOL, STATE_RTOL = 1e-7, 1e-5


def _scope(meter, label):
    return meter.scope(label) if meter is not None else contextlib.nullcontext()


def _check_optimizer(optimizer, origin):
    import torch

    if not isinstance(optimizer, torch.optim.AdamW) or origin.get("scheduler") is not None:
        raise ValueError("R3 requires AdamW without scheduler or scaler")
    # restore_state loads origin's parameter groups. Check that checkpoint too,
    # before allowing it to replace the otherwise-correct live configuration.
    for group in [*optimizer.param_groups, *origin["optimizer"]["param_groups"]]:
        if (
            group["lr"] != 1e-5
            or tuple(group["betas"]) != (0.9, 0.999)
            or group["eps"] != 1e-8
            or group["weight_decay"] != 0
            or group.get("amsgrad", False)
            or group.get("maximize", False)
        ):
            raise ValueError("R3 optimizer differs from the fixed protocol")


def _reset(adapter, optimizer, origin):
    metadata = restore_state(adapter.model, optimizer, origin)
    adapter.model.zero_grad(set_to_none=True)
    # The certified uncached path creates no retained attention cache. Qwen's
    # mutable positional continuation is reset explicitly between candidates.
    if hasattr(adapter, "_reset_positions"):
        adapter._reset_positions()
    if state_hash(capture_state(adapter.model, optimizer, metadata)) != state_hash(origin):
        raise RuntimeError("R3 failed to restore parameters/Adam/RNG/sampler metadata")
    return metadata


def _tensor_parts(value, prefix="root"):
    import torch

    tensors, structure = {}, []
    if isinstance(value, torch.Tensor):
        tensors[prefix] = value
    elif isinstance(value, dict):
        structure.append((prefix, "dict", sorted(map(str, value))))
        for key, child in value.items():
            ts, ss = _tensor_parts(child, f"{prefix}/{key}")
            tensors.update(ts)
            structure.extend(ss)
    elif isinstance(value, (tuple, list)):
        structure.append((prefix, type(value).__name__, len(value)))
        for index, child in enumerate(value):
            ts, ss = _tensor_parts(child, f"{prefix}/{index}")
            tensors.update(ts)
            structure.extend(ss)
    else:
        structure.append((prefix, type(value).__name__, value))
    return tensors, sorted(structure)


def _compare_states(left, right):
    parameters = compare_gradient_bundles(
        {"gradients": left["parameters"]},
        {"gradients": right["parameters"]},
        atol=STATE_ATOL,
        rtol=STATE_RTOL,
    )
    a, sa = _tensor_parts(left["optimizer"])
    b, sb = _tensor_parts(right["optimizer"])
    optimizer = compare_gradient_bundles(
        {"gradients": a}, {"gradients": b}, atol=STATE_ATOL, rtol=STATE_RTOL
    )
    exact_controls = all(
        state_hash(left[k]) == state_hash(right[k]) for k in ("rng", "scheduler", "metadata")
    )
    return {
        "parameter_comparison": parameters,
        "optimizer_comparison": optimizer,
        "optimizer_structure_equal": sa == sb,
        "rng_scheduler_sampler_equal": exact_controls,
        "bitwise_equal": state_hash(left) == state_hash(right),
        "passed": parameters["allclose"] and optimizer["allclose"] and sa == sb and exact_controls,
    }


def _norm(values):
    return math.sqrt(math.fsum(float(v.detach().double().square().sum()) for v in values.values()))


def _difference(left, right):
    return {name: left[name] - right[name] for name in left}


def validate_reuse_bank(adapter, optimizer, origin, groups, *, meter=None):
    """Measure score/direct preclip gradients AND four actual Adam probe steps.

    An all-zero comparison establishes consistency, not informative permission
    to reuse. The caller must require both preregistered banks to be adoptable.
    """
    _check_optimizer(optimizer, origin)
    training = adapter.model.training
    checks = {}
    try:
        _reset(adapter, optimizer, origin)
        with _scope(meter, "reuse/collect_scores"):
            scores = collect_category_scores(adapter, groups)
        total_backward = scores["audit"]["backward_calls"]
        for lam in (0.0, 1.0):
            _reset(adapter, optimizer, origin)
            with _scope(meter, f"reuse/validate_lambda{lam}/direct"):
                direct = direct_loss_gradients(adapter, groups, arm="X_VALID", auxiliary_weight=lam)
                direct_update = apply_gradient_update(adapter.model, optimizer, direct["gradients"])
                direct_state = capture_state(adapter.model, optimizer, origin["metadata"])
            total_backward += direct["audit"]["backward_calls"]
            _reset(adapter, optimizer, origin)
            with _scope(meter, f"reuse/validate_lambda{lam}/category"):
                reused = combine_category_gradients(
                    scores, groups, arm="X_VALID", auxiliary_weight=lam
                )
                reused_update = apply_gradient_update(adapter.model, optimizer, reused["gradients"])
                reused_state = capture_state(adapter.model, optimizer, origin["metadata"])
            gradients = compare_gradient_bundles(direct, reused, atol=GRAD_ATOL, rtol=GRAD_RTOL)
            state_comparison = _compare_states(direct_state, reused_state)
            checks[str(lam)] = {
                **state_comparison,
                "gradient_comparison": gradients,
                "direct_update": direct_update,
                "reused_update": reused_update,
                "direct_ratio": {
                    k: direct["audit"][k]
                    for k in ("ratio_one_exact", "cannot_adopt_reuse", "clip_fraction")
                },
                "passed": state_comparison["passed"]
                and gradients["passed"]
                and not direct["audit"]["cannot_adopt_reuse"]
                and not reused["audit"]["cannot_adopt_reuse"],
            }
            del direct, reused, direct_state, reused_state
        return {
            "adoptable": all(row["passed"] for row in checks.values()),
            "checks": checks,
            "score_bank": scores,
            "validation_optimizer_updates": 4,
            "backward_calls": total_backward,
            "origin_state_hash": state_hash(origin),
            "scope": "FIXED_CHECKPOINT_SINGLE_STEP_ONLY",
            "gradient_tolerance": {"atol": GRAD_ATOL, "rtol": GRAD_RTOL},
            "state_tolerance": {"atol": STATE_ATOL, "rtol": STATE_RTOL},
        }
    finally:
        _reset(adapter, optimizer, origin)
        adapter.model.train(training)


def _composition(groups):
    rows = []
    for group in groups:
        counts = {c: sum(r["category"] == c for r in group) for c in "XSWI"}
        x, b, i = counts["X"], counts["S"] + counts["W"], counts["I"]
        rows.append(
            {
                "prompt_id": group[0]["prompt_id"],
                "counts": counts,
                "no_X": x == 0,
                "validity_mixed": x + b > 0 and i > 0,
                "aux_only_activation": x == 0 and b > 0 and i > 0,
                "three_level": x > 0 and b > 0 and i > 0,
                "missing_categories": [c for c, n in counts.items() if not n],
                "missing_category_interpretation": (
                    "EMPIRICAL_ZERO_TRUE_PROBABILITY_AND_CONDITIONAL_SCORE_UNESTIMABLE"
                ),
            }
        )
    return {
        "prompts": rows,
        **{
            f"{key}_fraction": sum(r[key] for r in rows) / len(rows)
            for key in ("no_X", "validity_mixed", "aux_only_activation", "three_level")
        },
    }


def _advantage_response(groups, lam, statistics, baseline_statistics):
    differences, derivatives = [], []
    for group, stats, base in zip(groups, statistics, baseline_statistics, strict=True):
        differences.extend(
            a - b for a, b in zip(stats["advantages"], base["advantages"], strict=True)
        )
        derivatives.extend(
            advantage_derivative(
                [2.0 * (r["category"] == "X") for r in group],
                [float(r["category"] != "I") for r in group],
                lam,
                epsilon=1e-4,
            ).tolist()
        )
    return {
        "advantage_delta_l2": math.sqrt(math.fsum(v * v for v in differences)),
        "mean_squared_advantage_delta": math.fsum(v * v for v in differences) / len(differences),
        "mean_squared_dA_dlambda": math.fsum(v * v for v in derivatives) / len(derivatives),
        "epsilon": 1e-4,
    }


def _sgd_reference(gradients, baseline_gradients, delta):
    sgd = {name: -1e-5 * g for name, g in gradients.items()}
    auxiliary = {name: -1e-5 * (g - baseline_gradients[name]) for name, g in gradients.items()}
    norm, adam_norm = _norm(sgd), _norm(delta)
    dot = math.fsum(float((sgd[n].double() * delta[n].double()).sum()) for n in sgd)
    return {
        "definition": "UNCLIPPED_SGD_NO_MOMENTUM_PARAMETER_SPACE_REFERENCE",
        "learning_rate": 1e-5,
        "absolute_delta_l2": norm,
        "auxiliary_delta_l2": _norm(auxiliary),
        "delta_hash": state_hash(sgd),
        "cosine_with_actual_adam": dot / (norm * adam_norm) if norm and adam_norm else None,
        "probability_response": "NOT_MEASURED",
    }


def run_bank_forks(
    adapter,
    optimizer,
    origin,
    groups,
    out,
    identity,
    *,
    reuse_authorized=False,
    score_bank=None,
    meter=None,
):
    """Persist five independent candidates in one fresh immutable bank attempt.

    Each candidate starts from origin; the caller may reuse a completed attempt
    after verifying its manifest, or retain an incomplete one and choose a new
    attempt path. No partial candidate is silently overwritten here.
    """
    _check_optimizer(optimizer, origin)
    out = Path(out)
    training = adapter.model.training
    with frozen_writer(out):
        if any(p.name != ".writer.lock" for p in out.iterdir()):
            raise FileExistsError("R3 bank attempt already has artifacts")
        try:
            _reset(adapter, optimizer, origin)
            scores = score_bank
            collection_backward = 0
            if scores is None and reuse_authorized:
                with _scope(meter, "bank/collect_scores"):
                    scores = collect_category_scores(adapter, groups)
                collection_backward = scores["audit"]["backward_calls"]
            if scores is not None and scores["binding"]["parameter_hash"] != parameter_hash(
                adapter.model, trainable=True
            ):
                raise ValueError("Score bank belongs to a different checkpoint")
            adopted = bool(
                reuse_authorized
                and scores is not None
                and not scores["audit"]["cannot_adopt_reuse"]
            )

            def gradient(arm, lam):
                if adopted:
                    return combine_category_gradients(scores, groups, arm=arm, auxiliary_weight=lam)
                return direct_loss_gradients(adapter, groups, arm=arm, auxiliary_weight=lam)

            candidates = []
            baseline_gradient = baseline_state = baseline_stats = None
            backward = collection_backward
            snapshots = []
            for lam in LAMBDAS:
                _reset(adapter, optimizer, origin)
                prefix = f"bank_{identity['bank_index']:02d}_" if "bank_index" in identity else ""
                candidate_id = f"{prefix}lambda_{lam:g}"
                with _scope(meter, f"candidate/{candidate_id}"):
                    bundle = gradient("X_VALID", lam)
                    update = apply_gradient_update(adapter.model, optimizer, bundle["gradients"])
                    state = capture_state(adapter.model, optimizer, origin["metadata"])
                backward += bundle["audit"]["backward_calls"]
                if lam == 0:
                    baseline_gradient = bundle["gradients"]
                    baseline_state = state
                    baseline_stats = bundle["audit"]["group_statistics"]
                delta = _difference(state["parameters"], origin["parameters"])
                candidate_identity = {
                    **identity,
                    "candidate_id": candidate_id,
                    "lambda": lam,
                    "origin_state_hash": state_hash(origin),
                    "bank_hash": bundle["audit"]["bank_hash"],
                }
                path = (out / f"{candidate_id}.pt").resolve()
                save_checkpoint(path, state, candidate_identity)
                row = {
                    "lambda": lam,
                    "candidate_id": candidate_id,
                    "checkpoint_path": str(path),
                    "checkpoint_identity": candidate_identity,
                    "checkpoint_sha256": file_hash(path),
                    "parameter_hash": parameter_hash(adapter.model, trainable=True),
                    "optimizer_state_hash": state_hash(state["optimizer"]),
                    "state_hash": state_hash(state),
                    "gradient": bundle["audit"],
                    "update": update,
                    "gradient_delta_l2": _norm(_difference(bundle["gradients"], baseline_gradient)),
                    "parameter_auxiliary_delta_l2": _norm(
                        _difference(state["parameters"], baseline_state["parameters"])
                    ),
                    "parameters_vs_lambda0": _compare_states(baseline_state, state)[
                        "parameter_comparison"
                    ],
                    "sgd_reference": _sgd_reference(bundle["gradients"], baseline_gradient, delta),
                    **_advantage_response(
                        groups, lam, bundle["audit"]["group_statistics"], baseline_stats
                    ),
                }
                candidates.append(row)
                write_json(out / f"{candidate_id}.json", row)
                # Candidate pair distances are computed from small LoRA tensors,
                # not from the frozen 9B parameter space.
                snapshots.append((candidate_id, state["parameters"]))
                del bundle, state, delta
            _reset(adapter, optimizer, origin)
            with _scope(meter, "lambda_zero_replay"):
                bundle = gradient("X_VALID", 0.0)
                apply_gradient_update(adapter.model, optimizer, bundle["gradients"])
                repeated = capture_state(adapter.model, optimizer, origin["metadata"])
            backward += bundle["audit"]["backward_calls"]
            replay = _compare_states(baseline_state, repeated)
            replay["passed"] = replay["bitwise_equal"]
            if not replay["passed"]:
                write_json(out / "lambda_zero_replay.json", replay)
                raise RuntimeError("R3 repeated lambda-zero fork differs after restoration")
            answers = {}
            for arm, lam in (("A_BASE", 0.0), ("A_VALID", 1.0)):
                _reset(adapter, optimizer, origin)
                with _scope(meter, f"answer_gradient/{arm}"):
                    bundle = gradient(arm, lam)
                backward += bundle["audit"]["backward_calls"]
                answers[arm] = {
                    "gradient": bundle["audit"],
                    "gradient_delta_from_X_BASE_l2": _norm(
                        _difference(bundle["gradients"], baseline_gradient)
                    ),
                    "optimizer_updates": 0,
                }
            all_valid = all(row["category"] != "I" for group in groups for row in group)
            invariant = {
                "applicable": all_valid,
                "passed": all(c["parameters_vs_lambda0"]["allclose"] for c in candidates)
                if all_valid
                else None,
                "scope": "ACTUAL_CANDIDATE_PARAMETER_COMPARISON_AT_REGISTERED_TOLERANCE",
            }
            if all_valid and not invariant["passed"]:
                write_json(out / "all_valid_invariant.json", invariant)
                raise RuntimeError("R3 all-valid bank actual updates violate invariance")
            pairs = [
                {"left": a, "right": b, "parameter_distance_l2": _norm(_difference(x, y))}
                for i, (a, x) in enumerate(snapshots)
                for b, y in snapshots[i + 1 :]
            ]
            _reset(adapter, optimizer, origin)
            summary = {
                "status": "PASS",
                "passed": True,
                "candidates": candidates,
                "origin_state_hash": state_hash(origin),
                "identity": identity,
                "reuse_adopted": adopted,
                "reuse_requested": bool(reuse_authorized),
                "score_collection_backward_calls": collection_backward,
                "candidate_optimizer_updates": 5,
                "replay_optimizer_updates": 1,
                "validation_optimizer_updates": 0,
                "backward_calls": backward,
                "lambda_zero_replay": replay,
                "all_valid_invariant": invariant,
                "answer_diagnostics": answers,
                "group_composition": _composition(groups),
                "pairwise_distances": pairs,
                "origin_restored": True,
                "zero_gradient_semantics": "EXPLICIT_TENSORS_AND_ALWAYS_ADAM_STEP",
                "cache_policy": "UNCACHED_PRODUCTION_WITH_POSITION_RESET",
                "scheduler": "NOT_CONFIGURED",
                "grad_scaler": "NOT_CONFIGURED",
            }
            write_json(out / "summary.json", summary)
            files = [p for p in sorted(out.iterdir()) if p.is_file() and p.name != ".writer.lock"]
            write_json(
                out / "manifest.json",
                {
                    "files": [
                        {"path": p.name, "sha256": file_hash(p), "bytes": p.stat().st_size}
                        for p in files
                    ]
                },
            )
            return summary
        finally:
            _reset(adapter, optimizer, origin)
            adapter.model.train(training)
