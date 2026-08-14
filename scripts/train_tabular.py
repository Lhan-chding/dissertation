#!/usr/bin/env python3
"""Run config-driven CPU tabular selection and coordination experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAX_OPTIMIZER_STEPS = 10_000
_MAX_OPTIMIZER_BATCH = 100_000
_MAX_SEED_COUNT = 1_000
_MAX_PPO_EPOCHS = 100
_MAX_BOOTSTRAP_RESAMPLES = 100_000
_MAX_ERROR_ACTIONS = 128
_MAX_SELECTION_SAMPLE_UPDATES = 150_000_000
_MAX_BOOTSTRAP_MATRIX_CELLS = 10_000_000
_MAX_COORDINATION_GRID_COUNT = 100
_MAX_COORDINATION_SEEDS = 1_000
_MAX_BIFURCATION_POINTS = 100
_MAX_COORDINATION_HORIZON = 1_000.0


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    import math

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(converted) or (positive and converted <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a {qualifier}finite number")
    return converted


def _positive_int(
    value: Any,
    name: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _float_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    return _finite_float(value[0], f"{name}[0]"), _finite_float(value[1], f"{name}[1]")


def _sensitivity_rates(
    specification: dict[str, Any], name: str, nominal: float
) -> tuple[float, float]:
    low, high = _float_pair(
        specification.get("sensitivity_learning_rates"),
        f"{name}.sensitivity_learning_rates",
    )
    if low <= 0.0 or not low < nominal < high:
        raise ValueError(
            f"{name}.sensitivity_learning_rates must be positive and bracket learning_rate"
        )
    return low, high


def _float_vector(
    value: Any,
    name: str,
    *,
    length: int | None = None,
    maximum_length: int | None = None,
):
    import numpy as np

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector") from error
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite numeric vector")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    if maximum_length is not None and array.size > maximum_length:
        raise ValueError(f"{name} must contain at most {maximum_length} entries")
    return array


def _grid(specification: dict[str, Any], name: str):
    import numpy as np

    start = _finite_float(specification.get("start"), f"{name}.start")
    stop = _finite_float(specification.get("stop"), f"{name}.stop")
    count = _positive_int(
        specification.get("count"),
        f"{name}.count",
        minimum=2,
        maximum=_MAX_COORDINATION_GRID_COUNT,
    )
    if not start < stop:
        raise ValueError(f"{name}.start must be smaller than {name}.stop")
    return np.linspace(start, stop, count, dtype=np.float64)


def _output_path(repository_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    from compbias.io.artifact_paths import validated_artifact_path

    if name.endswith("_metrics"):
        suffix = ".json"
    elif name.endswith("_predictions"):
        suffix = ".csv"
    elif name.endswith("_figure"):
        suffix = ".png"
    else:
        raise ValueError(f"unsupported output field: {name}")
    return validated_artifact_path(
        value,
        repository_root=repository_root,
        label=name,
        suffix=suffix,
    )


def _load_config(path: Path) -> dict[str, Any]:
    from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

    config = load_yaml_mapping(path)
    allowed = {
        "schema_version",
        "experiment",
        "task",
        "seed",
        "errors",
        "selection",
        "scaling",
        "coordination",
        "outputs",
    }
    reject_unknown_fields(config, allowed, label="configuration")
    nested = {
        "errors": {"ids", "severity", "base_probs"},
        "selection": {
            "beta",
            "profiles",
            "mirror_descent",
            "reinforce",
            "ppo_like",
            "grpo_like",
            "bootstrap",
            "gates",
        },
        "scaling": {"beta", "kappa", "finite_difference_step", "gains", "gates"},
        "coordination": {
            "delta",
            "epsilon",
            "horizon",
            "separatrix_tolerance",
            "grid",
            "seeded_initializations",
            "bifurcation",
            "gates",
        },
        "outputs": {
            "selection_metrics",
            "coordination_metrics",
            "bifurcation_metrics",
            "selection_predictions",
            "scaling_metrics",
            "scaling_predictions",
            "coordination_predictions",
            "bifurcation_predictions",
            "selection_figure",
            "coordination_figure",
            "logs",
        },
    }
    for section, fields in nested.items():
        if section in config:
            reject_unknown_fields(config[section], fields, label=section)
    selection = config.get("selection")
    if isinstance(selection, dict):
        selection_sections = {
            "mirror_descent": {"step_size", "steps", "coarse_step_size"},
            "reinforce": {
                "seed_start",
                "num_seeds",
                "learning_rate",
                "sensitivity_learning_rates",
                "steps",
                "batch_size",
            },
            "ppo_like": {
                "seed_start",
                "num_seeds",
                "learning_rate",
                "sensitivity_learning_rates",
                "steps",
                "batch_size",
                "clip_ratio",
                "epochs_per_batch",
            },
            "grpo_like": {
                "seed_start",
                "num_seeds",
                "learning_rate",
                "sensitivity_learning_rates",
                "steps",
                "group_size",
            },
            "bootstrap": {"seed", "resamples"},
            "gates": {
                "exact_l1_max",
                "mirror_l1_max",
                "mirror_pairwise_odds_max_abs_residual",
                "flat_mean_shift_abs_max",
                "directional_mean_shift_min",
                "raw_fixed_reasoner_outcome_sign_accuracy_min",
                "raw_fixed_reasoner_outcome_update_sensitivity_min",
                "raw_fixed_reasoner_conditional_max_abs_error",
                "collapsed_diagnostic_sign_accuracy_min",
                "collapsed_diagnostic_mean_kl_improvement",
                "collapsed_diagnostic_update_sensitivity_min",
                "natural_mirror_trajectory_max_abs_error",
                "natural_mirror_endpoint_l1_max",
            },
        }
        for section, fields in selection_sections.items():
            if section in selection:
                reject_unknown_fields(selection[section], fields, label=f"selection.{section}")
    scaling = config.get("scaling")
    if isinstance(scaling, dict):
        scaling_sections = {
            "gates": {
                "average_gain_spread_max",
                "derivative_finite_difference_max_abs_error",
                "uniform_direction_max_abs",
                "directional_shift_min_abs",
            },
        }
        for section, fields in scaling_sections.items():
            if section in scaling:
                reject_unknown_fields(scaling[section], fields, label=f"scaling.{section}")
    coordination = config.get("coordination")
    if isinstance(coordination, dict):
        coordination_sections = {
            "grid": {"start", "stop", "count"},
            "seeded_initializations": {"seed_start", "num_seeds", "p_range", "gap_range"},
            "bifurcation": {
                "a",
                "beta_over_a",
                "horizon",
                "branch_initial",
                "max_branch_abs_error",
            },
            "gates": {"basin_mismatches_max", "seeded_mismatches_max"},
        }
        for section, fields in coordination_sections.items():
            if section in coordination:
                reject_unknown_fields(
                    coordination[section], fields, label=f"coordination.{section}"
                )
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be exactly 1")
    if config.get("task") not in {"all", "selection", "scaling", "coordination"}:
        raise ValueError("task must be one of: all, selection, scaling, coordination")
    from compbias.io.artifact_paths import validate_experiment_name

    validate_experiment_name(config.get("experiment"))
    _positive_int(config.get("seed"), "seed", minimum=0)
    return config


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": status not in {"", "unavailable"}}


def _run_metadata(
    *,
    config_path: Path,
    repository_root: Path,
    started_at: str,
    git_metadata: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import scipy

    from compbias.io.logging import publishable_command, publishable_path

    command_arguments = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
    ]
    if overwrite:
        command_arguments.append("--overwrite")
    command = publishable_command(
        command_arguments,
        worktree=repository_root,
    )

    return {
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(command),
        "config": publishable_path(config_path, worktree=repository_root),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git": dict(git_metadata),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": "cpu",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty predictions table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _display_output_path(path: Path, repository_root: Path) -> str:
    """Render repository artifacts relatively and approved temp artifacts absolutely."""

    try:
        return str(path.relative_to(repository_root))
    except ValueError:
        return str(path)


def _bootstrap_mean_ci(values, *, seed: int, resamples: int) -> tuple[float, float]:
    import numpy as np

    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size < 2:
        raise ValueError("bootstrap requires at least two scalar observations")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, observations.size, size=(resamples, observations.size))
    means = observations[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def _weighted_covariance(weights, left, right) -> float:
    left_mean = float(weights @ left)
    right_mean = float(weights @ right)
    return float(weights @ ((left - left_mean) * (right - right_mean)))


def _severity_direction_correct(
    observed_shift: float,
    *,
    predicted_shift: float,
    flat_tolerance: float,
) -> bool:
    """Compare one run to the registered severity-shift prediction."""

    if abs(predicted_shift) <= 1e-12:
        return abs(observed_shift) <= flat_tolerance
    return observed_shift * predicted_shift > 0.0


def _pairwise_odds_residual(selected, reference, multiplier) -> float:
    import numpy as np

    support = np.flatnonzero((selected > 0.0) & (reference > 0.0))
    residual = 0.0
    for offset, left in enumerate(support):
        for right in support[offset + 1 :]:
            observed = np.log(selected[left] / selected[right])
            predicted = np.log(reference[left] / reference[right]) + np.log(
                multiplier[left] / multiplier[right]
            )
            residual = max(residual, abs(float(observed - predicted)))
    return residual


def _selection_inputs(config: dict[str, Any]):
    import numpy as np

    errors = _mapping(config.get("errors"), "errors")
    raw_ids = errors.get("ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or not all(isinstance(value, str) and value for value in raw_ids)
    ):
        raise ValueError("errors.ids must be a non-empty list of strings")
    error_ids = tuple(raw_ids)
    if len(error_ids) > _MAX_ERROR_ACTIONS:
        raise ValueError(f"errors.ids must contain at most {_MAX_ERROR_ACTIONS} entries")
    if len(set(error_ids)) != len(error_ids):
        raise ValueError("errors.ids must be unique")
    if any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", error_id) is None
        for error_id in error_ids
    ):
        raise ValueError("errors.ids must be safe ASCII identifiers")
    severity = _float_vector(errors.get("severity"), "errors.severity", length=len(error_ids))
    base = _float_vector(errors.get("base_probs"), "errors.base_probs", length=len(error_ids))
    if np.any(severity < 0.0):
        raise ValueError("errors.severity must be nonnegative")
    if np.any(base <= 0.0) or not np.isclose(base.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("errors.base_probs must be positive and sum to one")

    selection = _mapping(config.get("selection"), "selection")
    profiles_raw = _mapping(selection.get("profiles"), "selection.profiles")
    required_profiles = {"truth_aligned", "flat", "spurious"}
    if set(profiles_raw) != required_profiles:
        raise ValueError(
            "selection.profiles must contain exactly truth_aligned, flat, and spurious"
        )
    profiles = {
        name: _float_vector(value, f"selection.profiles.{name}", length=len(error_ids))
        for name, value in profiles_raw.items()
    }
    if any(np.any((profile < 0.0) | (profile > 1.0)) for profile in profiles.values()):
        raise ValueError("all compensability values must lie in [0, 1]")
    return error_ids, severity, base, selection, profiles


def _run_approximate_reward_mode(
    *,
    reward_mode: str,
    signal,
    target,
    base,
    severity,
    baseline_severity: float,
    predicted_shift: float,
    flat_tolerance: float,
    direction_min: float,
    sign_accuracy_min: float,
    require_kl_improvement: bool,
    sensitivity_min: float,
    fixed_conditional_max_abs_error: float | None,
    profile_name: str,
    profile_index: int,
    error_ids: tuple[str, ...],
    compensability,
    algorithm_specs,
    beta: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[bool]]:
    """Run one explicit reward view without silently substituting another."""

    import numpy as np

    valid_modes = {
        "raw_fixed_reasoner_outcome",
        "unconstrained_joint_trajectory_diagnostic",
        "collapsed_effective_reward_diagnostic",
    }
    if reward_mode not in valid_modes:
        raise ValueError(f"unsupported approximate reward mode: {reward_mode}")
    sampled_outcome_mode = reward_mode != "collapsed_effective_reward_diagnostic"
    fixed_reasoner_mode = reward_mode == "raw_fixed_reasoner_outcome"
    joint_diagnostic_mode = reward_mode == "unconstrained_joint_trajectory_diagnostic"
    signal_keyword = "compensability" if sampled_outcome_mode else "rewards"
    expected_scope = {
        "raw_fixed_reasoner_outcome": "error_policy_fixed_reasoner_outcomes",
        "unconstrained_joint_trajectory_diagnostic": ("unconstrained_joint_trajectory_diagnostic"),
        "collapsed_effective_reward_diagnostic": "collapsed_marginal_diagnostic",
    }[reward_mode]
    diagnostic_arguments = (
        {"unconstrained_joint_trajectory_diagnostic": True} if joint_diagnostic_mode else {}
    )

    def assert_result_contract(result) -> None:
        if result.metrics.get("reward_mode") != reward_mode:
            raise RuntimeError("optimizer returned the wrong registered reward mode")
        if result.metrics.get("training_scope") != expected_scope:
            raise RuntimeError("optimizer returned the wrong registered training scope")
        joint_kl = result.metrics.get("joint_kl_to_theory")
        if fixed_reasoner_mode:
            if hasattr(result, "joint_trajectory") or joint_kl is not None:
                raise RuntimeError("fixed-reasoner optimizer changed the policy action space")
            if result.metrics.get("policy_action_count") != len(base):
                raise RuntimeError("fixed-reasoner optimizer must expose one action per error")
            if result.metrics.get("reasoner_conditional_frozen") is not True:
                raise RuntimeError("fixed-reasoner optimizer did not freeze P(outcome|error)")
            if not np.array_equal(
                np.asarray(result.metrics.get("final_outcome_conditional_by_error")),
                np.asarray(compensability),
            ):
                raise RuntimeError("fixed-reasoner conditional changed during optimization")
        elif joint_diagnostic_mode:
            if not hasattr(result, "joint_trajectory") or joint_kl is None:
                raise RuntimeError("joint diagnostic omitted joint-trajectory evidence")
            if result.metrics.get("formal_gate_eligible") is not False:
                raise RuntimeError("unconstrained joint trajectory may not enter a formal gate")
        elif joint_kl is not None:
            raise RuntimeError("collapsed diagnostic unexpectedly returned joint evidence")

    support = base > 0.0
    initial_kl = float(np.sum(base[support] * np.log(base[support] / target[support])))
    mode_metrics: dict[str, Any] = {}
    means: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    passes: list[bool] = []
    for algorithm_index, (
        algorithm,
        trainer,
        seeds,
        controls,
        sensitivity_rates,
    ) in enumerate(algorithm_specs):
        probabilities = []
        shifts = []
        kls = []
        signs = []
        slopes = []
        update_norms = []
        target_kls = []
        moment_l1s = []
        joint_kls = []
        outcome_rates = []
        final_reasoner_conditional_deviations = []
        aggregate_outcome_draws = np.zeros(len(base), dtype=np.int64)
        aggregate_outcome_successes = np.zeros(len(base), dtype=np.int64)
        final_policy_sensitivities = []
        severity_shift_sensitivities = []
        first_result = None
        for seed in seeds:
            result = trainer(
                base,
                beta=beta,
                seed=seed,
                **{signal_keyword: signal},
                **diagnostic_arguments,
                **controls,
            )
            assert_result_contract(result)
            if first_result is None:
                first_result = result
            probability = np.asarray(result.probabilities)
            probabilities.append(probability)
            shifts.append(float(probability @ severity - baseline_severity))
            kls.append(float(result.metrics["kl_to_theory"]))
            target_kls.append(float(result.metrics["target_kl_from_reference"]))
            moment_l1s.append(float(result.metrics["l1_to_moment_target"]))
            signs.append(
                _severity_direction_correct(
                    shifts[-1],
                    predicted_shift=predicted_shift,
                    flat_tolerance=flat_tolerance,
                )
            )
            slopes.append(float(result.metrics["odds_slope"]))
            update_norms.append(float(result.metrics["mean_update_norm_per_learning_rate"]))
            if joint_diagnostic_mode:
                joint_kls.append(float(result.metrics["joint_kl_to_theory"]))
                final_reasoner_conditional_deviations.append(
                    float(result.metrics["max_abs_final_reasoner_conditional_deviation"])
                )
            if sampled_outcome_mode:
                outcome_rates.append(float(result.metrics["empirical_outcome_rate"]))
                aggregate_outcome_draws += np.asarray(
                    result.metrics["outcome_draws_by_error"], dtype=np.int64
                )
                aggregate_outcome_successes += np.asarray(
                    result.metrics["outcome_successes_by_error"], dtype=np.int64
                )
            low_rate, high_rate = sensitivity_rates
            low_result = trainer(
                base,
                beta=beta,
                seed=seed,
                **{signal_keyword: signal},
                **diagnostic_arguments,
                **{**controls, "learning_rate": low_rate},
            )
            high_result = trainer(
                base,
                beta=beta,
                seed=seed,
                **{signal_keyword: signal},
                **diagnostic_arguments,
                **{**controls, "learning_rate": high_rate},
            )
            assert_result_contract(low_result)
            assert_result_contract(high_result)
            rate_span = high_rate - low_rate
            low_probability = np.asarray(low_result.probabilities)
            high_probability = np.asarray(high_result.probabilities)
            final_policy_sensitivities.append(
                float(np.sum(np.abs(high_probability - low_probability)) / rate_span)
            )
            low_shift = float(low_probability @ severity - baseline_severity)
            high_shift = float(high_probability @ severity - baseline_severity)
            severity_shift_sensitivities.append(abs(high_shift - low_shift) / rate_span)
            for index, error_id in enumerate(error_ids):
                rows.append(
                    {
                        "profile": profile_name,
                        "reward_mode": reward_mode,
                        "algorithm": algorithm,
                        "seed": seed,
                        "error_id": error_id,
                        "severity": float(severity[index]),
                        "base_probability": float(base[index]),
                        "compensability": float(compensability[index]),
                        "predicted_probability": float(target[index]),
                        "observed_probability": float(probability[index]),
                    }
                )
        if first_result is None:
            raise RuntimeError("approximate optimizer requires at least one seed")
        repeated = trainer(
            base,
            beta=beta,
            seed=seeds[0],
            **{signal_keyword: signal},
            **diagnostic_arguments,
            **controls,
        )
        assert_result_contract(repeated)
        seed_reproducible = bool(
            np.array_equal(first_result.probabilities, repeated.probabilities)
            and np.array_equal(first_result.trajectory, repeated.trajectory)
        )
        if joint_diagnostic_mode:
            seed_reproducible = bool(
                seed_reproducible
                and np.array_equal(
                    first_result.joint_trajectory,
                    repeated.joint_trajectory,
                )
            )
        probability_array = np.vstack(probabilities)
        mean_probability = probability_array.mean(axis=0)
        mean_shift = float(np.mean(shifts))
        direction_passed = (
            abs(mean_shift) < flat_tolerance
            if profile_name == "flat"
            else np.sign(mean_shift) == np.sign(predicted_shift) and abs(mean_shift) > direction_min
        )
        mean_kl = float(np.mean(kls))
        sign_accuracy = float(np.mean(signs))
        all_metrics_finite = bool(
            np.all(
                np.isfinite(
                    np.asarray(
                        [
                            *probability_array.ravel(),
                            *shifts,
                            *kls,
                            *slopes,
                            *update_norms,
                            *target_kls,
                            *moment_l1s,
                            *joint_kls,
                            *outcome_rates,
                            *final_reasoner_conditional_deviations,
                            *final_policy_sensitivities,
                            *severity_shift_sensitivities,
                        ]
                    )
                )
            )
        )
        kl_passed = (
            all_metrics_finite
            if sampled_outcome_mode
            else not require_kl_improvement or mean_kl <= initial_kl + 1e-12
        )
        mean_policy_sensitivity = float(np.mean(final_policy_sensitivities))
        sensitivity_expected_nonzero = sampled_outcome_mode or abs(predicted_shift) > 1e-12
        sensitivity_passed = (
            mean_policy_sensitivity >= sensitivity_min
            if sensitivity_expected_nonzero
            else mean_policy_sensitivity <= 1e-12
        )
        empirical_outcome_by_error = None
        conditional_error = None
        fixed_conditional_passed = None
        if sampled_outcome_mode:
            if np.any(aggregate_outcome_draws == 0):
                raise RuntimeError("outcome sampling did not cover every error action")
            empirical_outcome_by_error = aggregate_outcome_successes / aggregate_outcome_draws
        if fixed_reasoner_mode:
            if fixed_conditional_max_abs_error is None:
                raise RuntimeError("fixed-reasoner outcome mode requires a conditional gate")
            conditional_error = float(
                np.max(np.abs(empirical_outcome_by_error - np.asarray(compensability)))
            )
            fixed_conditional_passed = conditional_error <= fixed_conditional_max_abs_error
        base_algorithm_passed = bool(
            direction_passed
            and sign_accuracy >= sign_accuracy_min
            and kl_passed
            and sensitivity_passed
            and seed_reproducible
            and all_metrics_finite
        )
        algorithm_passed = bool(
            base_algorithm_passed and (fixed_conditional_passed if fixed_reasoner_mode else True)
        )
        passes.append(algorithm_passed if not joint_diagnostic_mode else False)
        means[algorithm] = mean_probability
        mode_metrics[algorithm] = {
            "reward_mode": reward_mode,
            "training_scope": expected_scope,
            "num_seeds": len(seeds),
            "seeds": list(seeds),
            "mean_probabilities": mean_probability.tolist(),
            "std_probabilities": probability_array.std(axis=0, ddof=1).tolist(),
            "mean_severity_shift": mean_shift,
            "std_severity_shift": float(np.std(shifts, ddof=1)),
            "bootstrap_95_ci": list(
                _bootstrap_mean_ci(
                    shifts,
                    seed=bootstrap_seed
                    + 100 * fixed_reasoner_mode
                    + 200 * joint_diagnostic_mode
                    + 10 * profile_index
                    + algorithm_index,
                    resamples=bootstrap_resamples,
                )
            ),
            "initial_kl_to_theory": initial_kl,
            "target_kl_from_reference": float(np.mean(target_kls)),
            "mean_kl_to_theory": mean_kl,
            "mean_l1_to_moment_target": float(np.mean(moment_l1s)),
            "mean_joint_kl_to_theory": (float(np.mean(joint_kls)) if joint_kls else None),
            "sign_accuracy": sign_accuracy,
            "mean_odds_slope": float(np.mean(slopes)),
            "mean_update_norm_per_learning_rate": float(np.mean(update_norms)),
            "mean_empirical_outcome_rate": (
                float(np.mean(outcome_rates)) if outcome_rates else None
            ),
            "empirical_outcome_rate_by_error": (
                empirical_outcome_by_error.tolist()
                if empirical_outcome_by_error is not None
                else None
            ),
            "max_abs_empirical_outcome_conditional_error": conditional_error,
            "fixed_reasoner_conditional_max_abs_error": (
                fixed_conditional_max_abs_error if fixed_reasoner_mode else None
            ),
            "fixed_reasoner_conditional_passed": fixed_conditional_passed,
            "mean_final_reasoner_conditional_deviation": (
                float(np.mean(final_reasoner_conditional_deviations))
                if final_reasoner_conditional_deviations
                else 0.0
                if fixed_reasoner_mode
                else None
            ),
            "policy_action_count": int(first_result.metrics["policy_action_count"]),
            "reasoner_conditional_frozen": first_result.metrics["reasoner_conditional_frozen"],
            "formal_gate_eligible": fixed_reasoner_mode,
            "diagnostic_completed": (base_algorithm_passed if joint_diagnostic_mode else None),
            "sensitivity_learning_rates": list(sensitivity_rates),
            "mean_final_policy_l1_per_learning_rate": mean_policy_sensitivity,
            "mean_severity_shift_per_learning_rate": float(np.mean(severity_shift_sensitivities)),
            "sensitivity_expected_nonzero": sensitivity_expected_nonzero,
            "sensitivity_passed": bool(sensitivity_passed),
            "seed_reproducible": seed_reproducible,
            "all_metrics_finite": all_metrics_finite,
            "passed": None if joint_diagnostic_mode else algorithm_passed,
        }
    return mode_metrics, means, rows, passes


def _run_selection(config: dict[str, Any]):
    import numpy as np

    from compbias.rl.exact_kl import exact_kl_projection
    from compbias.rl.grpo_adapter import train_grpo_like
    from compbias.rl.mirror_descent import optimize_mirror_descent
    from compbias.rl.natural_gradient import optimize_natural_policy_gradient
    from compbias.rl.ppo_adapter import train_ppo_like
    from compbias.rl.reinforce import train_reinforce
    from compbias.theory.selection import (
        binary_compensability_multiplier,
        selected_error_distribution,
    )

    error_ids, severity, base, selection, profiles = _selection_inputs(config)
    beta = _finite_float(selection.get("beta"), "selection.beta", positive=True)
    mirror = _mapping(selection.get("mirror_descent"), "selection.mirror_descent")
    mirror_step = _finite_float(
        mirror.get("step_size"), "selection.mirror_descent.step_size", positive=True
    )
    mirror_steps = _positive_int(
        mirror.get("steps"),
        "selection.mirror_descent.steps",
        maximum=_MAX_OPTIMIZER_STEPS,
    )
    coarse_step = _finite_float(
        mirror.get("coarse_step_size"),
        "selection.mirror_descent.coarse_step_size",
        positive=True,
    )
    reinforce = _mapping(selection.get("reinforce"), "selection.reinforce")
    seed_start = _positive_int(
        reinforce.get("seed_start"), "selection.reinforce.seed_start", minimum=0
    )
    seed_count = _positive_int(
        reinforce.get("num_seeds"),
        "selection.reinforce.num_seeds",
        minimum=20,
        maximum=_MAX_SEED_COUNT,
    )
    learning_rate = _finite_float(
        reinforce.get("learning_rate"),
        "selection.reinforce.learning_rate",
        positive=True,
    )
    reinforce_sensitivity_rates = _sensitivity_rates(
        reinforce, "selection.reinforce", learning_rate
    )
    reinforce_steps = _positive_int(
        reinforce.get("steps"),
        "selection.reinforce.steps",
        maximum=_MAX_OPTIMIZER_STEPS,
    )
    batch_size = _positive_int(
        reinforce.get("batch_size"),
        "selection.reinforce.batch_size",
        maximum=_MAX_OPTIMIZER_BATCH,
    )
    ppo = _mapping(selection.get("ppo_like"), "selection.ppo_like")
    ppo_seed_start = _positive_int(
        ppo.get("seed_start"), "selection.ppo_like.seed_start", minimum=0
    )
    ppo_seed_count = _positive_int(
        ppo.get("num_seeds"),
        "selection.ppo_like.num_seeds",
        minimum=20,
        maximum=_MAX_SEED_COUNT,
    )
    ppo_learning_rate = _finite_float(
        ppo.get("learning_rate"), "selection.ppo_like.learning_rate", positive=True
    )
    ppo_sensitivity_rates = _sensitivity_rates(ppo, "selection.ppo_like", ppo_learning_rate)
    ppo_steps = _positive_int(
        ppo.get("steps"),
        "selection.ppo_like.steps",
        maximum=_MAX_OPTIMIZER_STEPS,
    )
    ppo_batch_size = _positive_int(
        ppo.get("batch_size"),
        "selection.ppo_like.batch_size",
        maximum=_MAX_OPTIMIZER_BATCH,
    )
    ppo_clip_ratio = _finite_float(
        ppo.get("clip_ratio"), "selection.ppo_like.clip_ratio", positive=True
    )
    if ppo_clip_ratio >= 1.0:
        raise ValueError("selection.ppo_like.clip_ratio must be smaller than one")
    ppo_epochs = _positive_int(
        ppo.get("epochs_per_batch"),
        "selection.ppo_like.epochs_per_batch",
        maximum=_MAX_PPO_EPOCHS,
    )
    grpo = _mapping(selection.get("grpo_like"), "selection.grpo_like")
    grpo_seed_start = _positive_int(
        grpo.get("seed_start"), "selection.grpo_like.seed_start", minimum=0
    )
    grpo_seed_count = _positive_int(
        grpo.get("num_seeds"),
        "selection.grpo_like.num_seeds",
        minimum=20,
        maximum=_MAX_SEED_COUNT,
    )
    grpo_learning_rate = _finite_float(
        grpo.get("learning_rate"), "selection.grpo_like.learning_rate", positive=True
    )
    grpo_sensitivity_rates = _sensitivity_rates(grpo, "selection.grpo_like", grpo_learning_rate)
    grpo_steps = _positive_int(
        grpo.get("steps"),
        "selection.grpo_like.steps",
        maximum=_MAX_OPTIMIZER_STEPS,
    )
    grpo_group_size = _positive_int(
        grpo.get("group_size"),
        "selection.grpo_like.group_size",
        maximum=_MAX_OPTIMIZER_BATCH,
    )
    bootstrap = _mapping(selection.get("bootstrap"), "selection.bootstrap")
    bootstrap_seed = _positive_int(bootstrap.get("seed"), "selection.bootstrap.seed", minimum=0)
    bootstrap_resamples = _positive_int(
        bootstrap.get("resamples"),
        "selection.bootstrap.resamples",
        minimum=1000,
        maximum=_MAX_BOOTSTRAP_RESAMPLES,
    )
    nominal_workload = (
        seed_count * reinforce_steps * batch_size
        + ppo_seed_count * ppo_steps * ppo_batch_size * ppo_epochs
        + grpo_seed_count * grpo_steps * grpo_group_size
    )
    repeated_workload = (
        reinforce_steps * batch_size
        + ppo_steps * ppo_batch_size * ppo_epochs
        + grpo_steps * grpo_group_size
    )
    workload = len(profiles) * 3 * (3 * nominal_workload + repeated_workload)
    if workload > _MAX_SELECTION_SAMPLE_UPDATES:
        raise ValueError(
            "selection optimizer workload budget exceeds "
            f"{_MAX_SELECTION_SAMPLE_UPDATES} sample-updates"
        )
    if bootstrap_resamples * max(seed_count, ppo_seed_count, grpo_seed_count) > (
        _MAX_BOOTSTRAP_MATRIX_CELLS
    ):
        raise ValueError(
            f"selection bootstrap matrix budget exceeds {_MAX_BOOTSTRAP_MATRIX_CELLS} cells"
        )
    gates = _mapping(selection.get("gates"), "selection.gates")
    exact_l1_max = _finite_float(
        gates.get("exact_l1_max"), "selection.gates.exact_l1_max", positive=True
    )
    mirror_l1_max = _finite_float(
        gates.get("mirror_l1_max"), "selection.gates.mirror_l1_max", positive=True
    )
    mirror_odds_max = _finite_float(
        gates.get("mirror_pairwise_odds_max_abs_residual"),
        "selection.gates.mirror_pairwise_odds_max_abs_residual",
        positive=True,
    )
    flat_max = _finite_float(
        gates.get("flat_mean_shift_abs_max"),
        "selection.gates.flat_mean_shift_abs_max",
        positive=True,
    )
    direction_min = _finite_float(
        gates.get("directional_mean_shift_min"),
        "selection.gates.directional_mean_shift_min",
        positive=True,
    )
    raw_sign_accuracy_min = _finite_float(
        gates.get("raw_fixed_reasoner_outcome_sign_accuracy_min"),
        "selection.gates.raw_fixed_reasoner_outcome_sign_accuracy_min",
        positive=True,
    )
    diagnostic_sign_accuracy_min = _finite_float(
        gates.get("collapsed_diagnostic_sign_accuracy_min"),
        "selection.gates.collapsed_diagnostic_sign_accuracy_min",
        positive=True,
    )
    if raw_sign_accuracy_min > 1.0 or diagnostic_sign_accuracy_min > 1.0:
        raise ValueError("selection sign accuracy gates must not exceed one")
    diagnostic_require_kl_improvement = gates.get("collapsed_diagnostic_mean_kl_improvement")
    if not isinstance(diagnostic_require_kl_improvement, bool):
        raise ValueError("selection.gates.collapsed_diagnostic_mean_kl_improvement must be boolean")
    raw_sensitivity_min = _finite_float(
        gates.get("raw_fixed_reasoner_outcome_update_sensitivity_min"),
        "selection.gates.raw_fixed_reasoner_outcome_update_sensitivity_min",
        positive=True,
    )
    fixed_conditional_max_abs_error = _finite_float(
        gates.get("raw_fixed_reasoner_conditional_max_abs_error"),
        "selection.gates.raw_fixed_reasoner_conditional_max_abs_error",
        positive=True,
    )
    if fixed_conditional_max_abs_error > 1.0:
        raise ValueError("fixed-reasoner conditional error gate must not exceed one")
    diagnostic_sensitivity_min = _finite_float(
        gates.get("collapsed_diagnostic_update_sensitivity_min"),
        "selection.gates.collapsed_diagnostic_update_sensitivity_min",
        positive=True,
    )
    natural_trajectory_max = _finite_float(
        gates.get("natural_mirror_trajectory_max_abs_error"),
        "selection.gates.natural_mirror_trajectory_max_abs_error",
        positive=True,
    )
    natural_endpoint_max = _finite_float(
        gates.get("natural_mirror_endpoint_l1_max"),
        "selection.gates.natural_mirror_endpoint_l1_max",
        positive=True,
    )

    baseline_severity = float(base @ severity)
    profile_metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    plot_records: list[dict[str, Any]] = []
    all_gates: list[bool] = []
    diagnostic_gates: list[bool] = []
    algorithm_specs = (
        (
            "reinforce",
            train_reinforce,
            tuple(range(seed_start, seed_start + seed_count)),
            {
                "learning_rate": learning_rate,
                "steps": reinforce_steps,
                "batch_size": batch_size,
            },
            reinforce_sensitivity_rates,
        ),
        (
            "ppo_like",
            train_ppo_like,
            tuple(range(ppo_seed_start, ppo_seed_start + ppo_seed_count)),
            {
                "learning_rate": ppo_learning_rate,
                "steps": ppo_steps,
                "batch_size": ppo_batch_size,
                "clip_ratio": ppo_clip_ratio,
                "epochs_per_batch": ppo_epochs,
            },
            ppo_sensitivity_rates,
        ),
        (
            "grpo_like",
            train_grpo_like,
            tuple(range(grpo_seed_start, grpo_seed_start + grpo_seed_count)),
            {
                "learning_rate": grpo_learning_rate,
                "steps": grpo_steps,
                "group_size": grpo_group_size,
            },
            grpo_sensitivity_rates,
        ),
    )

    for profile_index, (name, compensability) in enumerate(profiles.items()):
        multiplier = np.asarray(binary_compensability_multiplier(compensability, beta))
        equivalent_rewards = beta * np.log(multiplier)
        predicted = np.asarray(selected_error_distribution(base, multiplier))
        exact = np.asarray(exact_kl_projection(base, equivalent_rewards, beta))
        mirror_result = optimize_mirror_descent(
            base,
            equivalent_rewards,
            beta=beta,
            step_size=mirror_step,
            steps=mirror_steps,
            seed=int(config["seed"]),
        )
        natural_result = optimize_natural_policy_gradient(
            base,
            equivalent_rewards,
            beta=beta,
            step_size=mirror_step,
            steps=mirror_steps,
            seed=int(config["seed"]),
        )
        mirror_observed = np.asarray(mirror_result.probabilities)
        natural_observed = np.asarray(natural_result.probabilities)
        exact_l1 = float(np.sum(np.abs(exact - predicted)))
        mirror_l1 = float(np.sum(np.abs(mirror_observed - predicted)))
        natural_l1 = float(np.sum(np.abs(natural_observed - predicted)))
        natural_trajectory_error = float(
            np.max(
                np.abs(np.asarray(natural_result.trajectory) - np.asarray(mirror_result.trajectory))
            )
        )
        natural_endpoint_error = float(np.sum(np.abs(natural_observed - mirror_observed)))
        natural_passed = bool(
            natural_trajectory_error <= natural_trajectory_max
            and natural_endpoint_error <= natural_endpoint_max
            and natural_l1 < mirror_l1_max
        )
        exact_odds_residual = _pairwise_odds_residual(exact, base, multiplier)
        mirror_odds_residual = _pairwise_odds_residual(
            mirror_observed,
            base,
            multiplier,
        )
        predicted_shift = float(predicted @ severity - baseline_severity)
        covariance = _weighted_covariance(base, severity, compensability)

        fixed_metrics, fixed_means, fixed_rows, fixed_passes = _run_approximate_reward_mode(
            reward_mode="raw_fixed_reasoner_outcome",
            signal=compensability,
            target=predicted,
            base=base,
            severity=severity,
            baseline_severity=baseline_severity,
            predicted_shift=predicted_shift,
            flat_tolerance=flat_max,
            direction_min=direction_min,
            sign_accuracy_min=raw_sign_accuracy_min,
            require_kl_improvement=False,
            sensitivity_min=raw_sensitivity_min,
            fixed_conditional_max_abs_error=fixed_conditional_max_abs_error,
            profile_name=name,
            profile_index=profile_index,
            error_ids=error_ids,
            compensability=compensability,
            algorithm_specs=algorithm_specs,
            beta=beta,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        joint_diagnostic_metrics, joint_diagnostic_means, joint_diagnostic_rows, _ = (
            _run_approximate_reward_mode(
                reward_mode="unconstrained_joint_trajectory_diagnostic",
                signal=compensability,
                target=predicted,
                base=base,
                severity=severity,
                baseline_severity=baseline_severity,
                predicted_shift=predicted_shift,
                flat_tolerance=flat_max,
                direction_min=direction_min,
                sign_accuracy_min=raw_sign_accuracy_min,
                require_kl_improvement=False,
                sensitivity_min=raw_sensitivity_min,
                fixed_conditional_max_abs_error=None,
                profile_name=name,
                profile_index=profile_index,
                error_ids=error_ids,
                compensability=compensability,
                algorithm_specs=algorithm_specs,
                beta=beta,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
        )
        diagnostic_metrics, diagnostic_means, diagnostic_rows, diagnostic_passes = (
            _run_approximate_reward_mode(
                reward_mode="collapsed_effective_reward_diagnostic",
                signal=equivalent_rewards,
                target=predicted,
                base=base,
                severity=severity,
                baseline_severity=baseline_severity,
                predicted_shift=predicted_shift,
                flat_tolerance=flat_max,
                direction_min=direction_min,
                sign_accuracy_min=diagnostic_sign_accuracy_min,
                require_kl_improvement=diagnostic_require_kl_improvement,
                sensitivity_min=diagnostic_sensitivity_min,
                fixed_conditional_max_abs_error=None,
                profile_name=name,
                profile_index=profile_index,
                error_ids=error_ids,
                compensability=compensability,
                algorithm_specs=algorithm_specs,
                beta=beta,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
        )
        rows.extend(fixed_rows)
        rows.extend(joint_diagnostic_rows)
        rows.extend(diagnostic_rows)
        approximation_gap = {
            algorithm: {
                "fixed_reasoner_minus_collapsed_mean_kl_to_theory": (
                    fixed_metrics[algorithm]["mean_kl_to_theory"]
                    - diagnostic_metrics[algorithm]["mean_kl_to_theory"]
                ),
                "fixed_reasoner_minus_collapsed_mean_severity_shift": (
                    fixed_metrics[algorithm]["mean_severity_shift"]
                    - diagnostic_metrics[algorithm]["mean_severity_shift"]
                ),
                "unconstrained_joint_minus_fixed_reasoner_mean_kl_to_theory": (
                    joint_diagnostic_metrics[algorithm]["mean_kl_to_theory"]
                    - fixed_metrics[algorithm]["mean_kl_to_theory"]
                ),
            }
            for algorithm, *_rest in algorithm_specs
        }
        profile_gate = (
            exact_l1 < exact_l1_max
            and mirror_l1 < mirror_l1_max
            and mirror_odds_residual < mirror_odds_max
            and natural_passed
            and all(fixed_passes)
        )
        all_gates.append(bool(profile_gate))
        diagnostic_gates.append(all(diagnostic_passes))

        for reward_mode, algorithm, seed, observed in (
            ("theory_oracle", "exact_kl", "", exact),
            (
                "collapsed_effective_reward_diagnostic",
                "mirror_descent",
                int(config["seed"]),
                mirror_observed,
            ),
            (
                "collapsed_effective_reward_diagnostic",
                "natural_policy_gradient",
                int(config["seed"]),
                natural_observed,
            ),
        ):
            for index, error_id in enumerate(error_ids):
                rows.append(
                    {
                        "profile": name,
                        "reward_mode": reward_mode,
                        "algorithm": algorithm,
                        "seed": seed,
                        "error_id": error_id,
                        "severity": float(severity[index]),
                        "base_probability": float(base[index]),
                        "compensability": float(compensability[index]),
                        "predicted_probability": float(predicted[index]),
                        "observed_probability": float(observed[index]),
                    }
                )
        plot_records.extend(
            [
                {
                    "profile": name,
                    "algorithm": "exact KL",
                    "predicted": predicted,
                    "observed": exact,
                },
                {
                    "profile": name,
                    "algorithm": "mirror descent",
                    "predicted": predicted,
                    "observed": mirror_observed,
                },
                {
                    "profile": name,
                    "algorithm": "natural policy gradient",
                    "predicted": predicted,
                    "observed": natural_observed,
                },
                {
                    "profile": name,
                    "algorithm": "REINFORCE fixed-reasoner outcome mean",
                    "predicted": predicted,
                    "observed": fixed_means["reinforce"],
                },
                {
                    "profile": name,
                    "algorithm": "PPO-like fixed-reasoner outcome mean",
                    "predicted": predicted,
                    "observed": fixed_means["ppo_like"],
                },
                {
                    "profile": name,
                    "algorithm": "GRPO-like fixed-reasoner outcome mean",
                    "predicted": predicted,
                    "observed": fixed_means["grpo_like"],
                },
                {
                    "profile": name,
                    "algorithm": "REINFORCE unconstrained-joint diagnostic mean",
                    "predicted": predicted,
                    "observed": joint_diagnostic_means["reinforce"],
                },
                {
                    "profile": name,
                    "algorithm": "PPO-like unconstrained-joint diagnostic mean",
                    "predicted": predicted,
                    "observed": joint_diagnostic_means["ppo_like"],
                },
                {
                    "profile": name,
                    "algorithm": "GRPO-like unconstrained-joint diagnostic mean",
                    "predicted": predicted,
                    "observed": joint_diagnostic_means["grpo_like"],
                },
                {
                    "profile": name,
                    "algorithm": "REINFORCE collapsed diagnostic mean",
                    "predicted": predicted,
                    "observed": diagnostic_means["reinforce"],
                },
                {
                    "profile": name,
                    "algorithm": "PPO-like collapsed diagnostic mean",
                    "predicted": predicted,
                    "observed": diagnostic_means["ppo_like"],
                },
                {
                    "profile": name,
                    "algorithm": "GRPO-like collapsed diagnostic mean",
                    "predicted": predicted,
                    "observed": diagnostic_means["grpo_like"],
                },
            ]
        )
        profile_metrics[name] = {
            "compensability": compensability.tolist(),
            "severity_compensability_covariance": covariance,
            "predicted_probabilities": predicted.tolist(),
            "predicted_severity_shift": predicted_shift,
            "exact_pairwise_odds_max_abs_residual": exact_odds_residual,
            "mirror_pairwise_odds_max_abs_residual": mirror_odds_residual,
            "exact_kl": {
                "probabilities": exact.tolist(),
                "l1_to_theory": exact_l1,
                "passed": exact_l1 < exact_l1_max,
            },
            "mirror_descent": {
                "probabilities": mirror_observed.tolist(),
                "l1_to_theory": mirror_l1,
                "step_size": mirror_step,
                "steps": mirror_steps,
                "passed": mirror_l1 < mirror_l1_max and mirror_odds_residual < mirror_odds_max,
            },
            "natural_policy_gradient": {
                "probabilities": natural_observed.tolist(),
                "l1_to_theory": natural_l1,
                "step_size": mirror_step,
                "steps": mirror_steps,
                "trajectory_max_abs_difference_to_mirror": (natural_trajectory_error),
                "endpoint_l1_difference_to_mirror": natural_endpoint_error,
                "passed": natural_passed,
            },
            "raw_fixed_reasoner_outcome": fixed_metrics,
            "unconstrained_joint_trajectory_diagnostic": joint_diagnostic_metrics,
            "collapsed_effective_reward_diagnostic": diagnostic_metrics,
            "formal_passed": bool(profile_gate),
            "collapsed_diagnostic_passed": all(diagnostic_passes),
            "approximation_gap": approximation_gap,
            "passed": bool(profile_gate),
        }

    spurious = profiles["spurious"]
    spurious_multiplier = np.asarray(binary_compensability_multiplier(spurious, beta))
    spurious_rewards = beta * np.log(spurious_multiplier)
    coarse = optimize_mirror_descent(
        base,
        spurious_rewards,
        beta=beta,
        step_size=coarse_step,
        steps=mirror_steps,
        seed=int(config["seed"]),
    )
    refined = optimize_mirror_descent(
        base,
        spurious_rewards,
        beta=beta,
        step_size=mirror_step,
        steps=mirror_steps,
        seed=int(config["seed"]),
    )
    coarse_error = float(coarse.metrics["final_l1_to_exact"])
    refined_error = float(refined.metrics["final_l1_to_exact"])
    refinement_passed = refined_error < coarse_error
    all_gates.append(refinement_passed)

    metrics = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "beta": beta,
        "base_probabilities": base.tolist(),
        "severity": severity.tolist(),
        "baseline_severity": baseline_severity,
        "profiles": profile_metrics,
        "mirror_step_refinement": {
            "coarse_step_size": coarse_step,
            "coarse_l1_to_exact": coarse_error,
            "refined_step_size": mirror_step,
            "refined_l1_to_exact": refined_error,
            "passed": refinement_passed,
        },
        "gates": {
            "minimum_reinforce_seeds": 20,
            "configured_reinforce_seeds": seed_count,
            "minimum_ppo_like_seeds": 20,
            "configured_ppo_like_seeds": ppo_seed_count,
            "minimum_grpo_like_seeds": 20,
            "configured_grpo_like_seeds": grpo_seed_count,
            "approximate_algorithms": ["reinforce", "ppo_like", "grpo_like"],
            "formal_reward_mode": "raw_fixed_reasoner_outcome",
            "diagnostic_reward_modes": [
                "unconstrained_joint_trajectory_diagnostic",
                "collapsed_effective_reward_diagnostic",
            ],
            "raw_fixed_reasoner_outcome_sign_accuracy_min": raw_sign_accuracy_min,
            "raw_fixed_reasoner_outcome_update_sensitivity_min": raw_sensitivity_min,
            "raw_fixed_reasoner_conditional_max_abs_error": (fixed_conditional_max_abs_error),
            "raw_fixed_reasoner_outcome_kl_gate": ("finite_only_moment_target_deviation_reported"),
            "unconstrained_joint_trajectory_formal_gate_eligible": False,
            "collapsed_diagnostic_sign_accuracy_min": diagnostic_sign_accuracy_min,
            "collapsed_diagnostic_mean_kl_improvement": (diagnostic_require_kl_improvement),
            "collapsed_diagnostic_update_sensitivity_min": diagnostic_sensitivity_min,
            "exact_l1_max": exact_l1_max,
            "mirror_l1_max": mirror_l1_max,
            "mirror_pairwise_odds_max_abs_residual": mirror_odds_max,
            "natural_mirror_trajectory_max_abs_error": natural_trajectory_max,
            "natural_mirror_endpoint_l1_max": natural_endpoint_max,
            "flat_mean_shift_abs_max": flat_max,
            "directional_mean_shift_min": direction_min,
        },
        "collapsed_diagnostic_calibration_passed": all(diagnostic_gates),
        "passed": all(all_gates),
    }
    return metrics, rows, plot_records, error_ids


def _run_scaling(config: dict[str, Any]):
    import numpy as np

    from compbias.rl.exact_kl import exact_kl_projection
    from compbias.rl.tabular_experiments import run_scaling_paths

    errors = _mapping(config.get("errors"), "errors")
    raw_ids = errors.get("ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or not all(
            isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
            for value in raw_ids
        )
        or len(set(raw_ids)) != len(raw_ids)
    ):
        raise ValueError("errors.ids must be unique safe ASCII identifiers")
    error_ids = tuple(raw_ids)
    if len(error_ids) > _MAX_ERROR_ACTIONS:
        raise ValueError(f"errors.ids must contain at most {_MAX_ERROR_ACTIONS} entries")
    severity = _float_vector(errors.get("severity"), "errors.severity", length=len(error_ids))
    base = _float_vector(errors.get("base_probs"), "errors.base_probs", length=len(error_ids))
    if np.any(severity < 0.0):
        raise ValueError("errors.severity must be nonnegative")
    if np.any(base <= 0.0) or not np.isclose(base.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("errors.base_probs must be positive and sum to one")

    scaling = _mapping(config.get("scaling"), "scaling")
    beta = _finite_float(scaling.get("beta"), "scaling.beta", positive=True)
    kappa = _finite_float(scaling.get("kappa"), "scaling.kappa", positive=True)
    finite_difference_step = _finite_float(
        scaling.get("finite_difference_step"),
        "scaling.finite_difference_step",
        positive=True,
    )
    if finite_difference_step >= kappa:
        raise ValueError("scaling.finite_difference_step must be smaller than kappa")
    raw_gains = _mapping(scaling.get("gains"), "scaling.gains")
    path_names = ("truth_gain", "uniform_gain", "error_gain")
    if set(raw_gains) != set(path_names):
        raise ValueError("scaling.gains must contain exactly truth_gain, uniform_gain, error_gain")
    gains = {
        name: _float_vector(raw_gains[name], f"scaling.gains.{name}", length=len(error_ids))
        for name in path_names
    }
    if any(np.any(gain < 0.0) for gain in gains.values()):
        raise ValueError("scaling gains must be nonnegative")

    results = run_scaling_paths(base, severity, gains, kappa=kappa, beta=beta)
    average_gains = np.asarray([result.average_gain for result in results])
    maximum_gain_difference = float(np.ptp(average_gains))
    derivative_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    path_metrics: dict[str, dict[str, Any]] = {}
    baseline_severity = float(base @ severity)
    for result in results:
        gain = np.asarray(result.gain)
        lower = np.asarray(exact_kl_projection(base, (kappa - finite_difference_step) * gain, beta))
        upper = np.asarray(exact_kl_projection(base, (kappa + finite_difference_step) * gain, beta))
        finite_difference = float(
            ((upper @ severity) - (lower @ severity)) / (2.0 * finite_difference_step)
        )
        derivative_error = abs(result.covariance_derivative - finite_difference)
        derivative_errors.append(derivative_error)
        path_metrics[result.name] = {
            "average_gain": result.average_gain,
            "covariance_derivative": result.covariance_derivative,
            "finite_difference_derivative": finite_difference,
            "derivative_abs_error": derivative_error,
            "severity_shift": result.severity_shift,
            "selected_severity": baseline_severity + result.severity_shift,
        }
        for index, error_id in enumerate(error_ids):
            rows.append(
                {
                    "path": result.name,
                    "error_id": error_id,
                    "severity": float(severity[index]),
                    "base_probability": float(base[index]),
                    "gain": float(result.gain[index]),
                    "selected_probability": float(result.selected[index]),
                }
            )

    gates = _mapping(scaling.get("gates"), "scaling.gates")
    gain_spread_max = _finite_float(
        gates.get("average_gain_spread_max"),
        "scaling.gates.average_gain_spread_max",
    )
    derivative_error_max = _finite_float(
        gates.get("derivative_finite_difference_max_abs_error"),
        "scaling.gates.derivative_finite_difference_max_abs_error",
        positive=True,
    )
    uniform_max = _finite_float(
        gates.get("uniform_direction_max_abs"),
        "scaling.gates.uniform_direction_max_abs",
        positive=True,
    )
    direction_min = _finite_float(
        gates.get("directional_shift_min_abs"),
        "scaling.gates.directional_shift_min_abs",
        positive=True,
    )
    if gain_spread_max < 0.0:
        raise ValueError("scaling.gates.average_gain_spread_max must be nonnegative")
    maximum_derivative_error = max(derivative_errors)
    truth = path_metrics["truth_gain"]
    uniform = path_metrics["uniform_gain"]
    error = path_metrics["error_gain"]
    directions_passed = (
        truth["covariance_derivative"] < 0.0
        and truth["severity_shift"] <= -direction_min
        and abs(uniform["covariance_derivative"]) <= uniform_max
        and abs(uniform["severity_shift"]) <= uniform_max
        and error["covariance_derivative"] > 0.0
        and error["severity_shift"] >= direction_min
    )
    passed = (
        maximum_gain_difference <= gain_spread_max
        and maximum_derivative_error <= derivative_error_max
        and directions_passed
    )
    metrics = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "path_names": list(path_names),
        "beta": beta,
        "kappa": kappa,
        "baseline_severity": baseline_severity,
        "maximum_average_gain_difference": maximum_gain_difference,
        "maximum_derivative_finite_difference_error": maximum_derivative_error,
        "paths": path_metrics,
        "gates": {
            "average_gain_spread_max": gain_spread_max,
            "derivative_finite_difference_max_abs_error": derivative_error_max,
            "uniform_direction_max_abs": uniform_max,
            "directional_shift_min_abs": direction_min,
            "directions_passed": directions_passed,
        },
        "passed": passed,
    }
    return metrics, rows


def _run_coordination(config: dict[str, Any]):
    import numpy as np
    from scipy.integrate import solve_ivp

    from compbias.theory.coordination import (
        CoordinationParams,
        basin_map,
        symmetric_bifurcation_root,
        vector_field,
    )

    coordination = _mapping(config.get("coordination"), "coordination")
    delta = _finite_float(coordination.get("delta"), "coordination.delta", positive=True)
    epsilon = _finite_float(coordination.get("epsilon"), "coordination.epsilon", positive=True)
    if not np.isclose(delta, epsilon, rtol=0.0, atol=1e-14):
        raise ValueError("the registered basin boundary requires delta == epsilon")
    horizon = _finite_float(coordination.get("horizon"), "coordination.horizon", positive=True)
    if horizon > _MAX_COORDINATION_HORIZON:
        raise ValueError(f"coordination.horizon must be at most {_MAX_COORDINATION_HORIZON:g}")
    tolerance = _finite_float(
        coordination.get("separatrix_tolerance"),
        "coordination.separatrix_tolerance",
    )
    if tolerance < 0.0:
        raise ValueError("coordination.separatrix_tolerance must be nonnegative")
    grid = _grid(_mapping(coordination.get("grid"), "coordination.grid"), "coordination.grid")
    params = CoordinationParams(delta=delta, epsilon=epsilon)
    basin = basin_map(
        grid,
        grid,
        params,
        horizon=horizon,
        separatrix_tolerance=tolerance,
    )
    score = basin.p + basin.q - 1.0
    expected_grid = np.where(
        score > tolerance,
        "truthful",
        np.where(score < -tolerance, "compensatory", "separatrix"),
    )
    basin_mismatches = int(np.count_nonzero(basin.labels != expected_grid))
    coordination_rows: list[dict[str, Any]] = []
    for index in np.ndindex(basin.labels.shape):
        coordination_rows.append(
            {
                "kind": "grid",
                "seed": "",
                "p0": float(basin.p[index]),
                "q0": float(basin.q[index]),
                "expected_equilibrium": str(expected_grid[index]),
                "observed_equilibrium": str(basin.labels[index]),
            }
        )

    initialization = _mapping(
        coordination.get("seeded_initializations"),
        "coordination.seeded_initializations",
    )
    seed_start = _positive_int(
        initialization.get("seed_start"),
        "coordination.seeded_initializations.seed_start",
        minimum=0,
    )
    seed_count = _positive_int(
        initialization.get("num_seeds"),
        "coordination.seeded_initializations.num_seeds",
        minimum=20,
        maximum=_MAX_COORDINATION_SEEDS,
    )
    p_range = _float_pair(
        initialization.get("p_range"), "coordination.seeded_initializations.p_range"
    )
    gap_range = _float_pair(
        initialization.get("gap_range"),
        "coordination.seeded_initializations.gap_range",
    )
    if not 0.0 < p_range[0] < p_range[1] < 1.0:
        raise ValueError("seeded p_range must be increasing inside (0, 1)")
    if not 0.0 < gap_range[0] < gap_range[1] < 1.0:
        raise ValueError("seeded gap_range must be increasing inside (0, 1)")
    seeded_mismatches = 0
    seeded_counts = {"truthful": 0, "compensatory": 0, "separatrix": 0}
    for seed in range(seed_start, seed_start + seed_count):
        rng = np.random.default_rng(seed)
        p0 = float(rng.uniform(*p_range))
        gap = float(rng.uniform(*gap_range))
        q0 = 1.0 - p0 + (gap if seed % 2 else -gap)
        if not 0.0 <= q0 <= 1.0:
            raise ValueError("seeded initialization ranges produced q0 outside [0, 1]")
        expected = "truthful" if p0 + q0 > 1.0 else "compensatory"
        observed = str(
            basin_map(
                [p0],
                [q0],
                params,
                horizon=horizon,
                separatrix_tolerance=tolerance,
            ).labels[0, 0]
        )
        seeded_counts[observed] += 1
        seeded_mismatches += int(expected != observed)
        coordination_rows.append(
            {
                "kind": "seeded",
                "seed": seed,
                "p0": p0,
                "q0": q0,
                "expected_equilibrium": expected,
                "observed_equilibrium": observed,
            }
        )

    bifurcation = _mapping(coordination.get("bifurcation"), "coordination.bifurcation")
    a = _finite_float(bifurcation.get("a"), "coordination.bifurcation.a", positive=True)
    ratios = _float_vector(
        bifurcation.get("beta_over_a"),
        "coordination.bifurcation.beta_over_a",
        maximum_length=_MAX_BIFURCATION_POINTS,
    )
    if np.any(ratios <= 0.0) or np.any(np.diff(ratios) <= 0.0):
        raise ValueError("bifurcation beta_over_a must be strictly increasing and positive")
    branch_horizon = _finite_float(
        bifurcation.get("horizon"), "coordination.bifurcation.horizon", positive=True
    )
    if branch_horizon > _MAX_COORDINATION_HORIZON:
        raise ValueError(
            f"coordination.bifurcation.horizon must be at most {_MAX_COORDINATION_HORIZON:g}"
        )
    branch_initial = _float_pair(
        bifurcation.get("branch_initial"),
        "coordination.bifurcation.branch_initial",
    )
    if not 0.0 < branch_initial[0] < 0.5 < branch_initial[1] < 1.0:
        raise ValueError("branch_initial must straddle 0.5 inside (0, 1)")
    max_branch_error = _finite_float(
        bifurcation.get("max_branch_abs_error"),
        "coordination.bifurcation.max_branch_abs_error",
        positive=True,
    )
    branch_rows: list[dict[str, Any]] = []
    observed_negative: list[float] = []
    observed_positive: list[float] = []
    predicted_positive: list[float] = []
    branch_errors: list[float] = []
    symmetry_errors: list[float] = []
    for ratio in ratios:
        predicted = symmetric_bifurcation_root(float(ratio))
        predicted_positive.append(predicted)
        for branch_name, initial, sign in (
            ("negative", branch_initial[0], -1.0),
            ("positive", branch_initial[1], 1.0),
        ):
            effective_initial = 0.5 if np.isclose(ratio, 0.5, atol=1e-14) else initial
            beta = float(ratio * a)
            branch_params = CoordinationParams(
                delta=a,
                epsilon=a,
                beta_p=beta,
                beta_q=beta,
            )
            solution = solve_ivp(
                vector_field,
                (0.0, branch_horizon),
                (effective_initial, effective_initial),
                args=(branch_params,),
                rtol=1e-10,
                atol=1e-12,
            )
            if not solution.success:
                raise RuntimeError(f"bifurcation ODE failed for beta/a={ratio}: {solution.message}")
            p_final, q_final = (float(value) for value in solution.y[:, -1])
            observed_m = p_final + q_final - 1.0
            predicted_m = sign * predicted
            error = abs(observed_m - predicted_m)
            branch_errors.append(error)
            symmetry_errors.append(abs(p_final - q_final))
            if branch_name == "negative":
                observed_negative.append(observed_m)
            else:
                observed_positive.append(observed_m)
            branch_rows.append(
                {
                    "beta_over_a": float(ratio),
                    "branch": branch_name,
                    "initial_probability": effective_initial,
                    "p_final": p_final,
                    "q_final": q_final,
                    "predicted_m": predicted_m,
                    "observed_m": observed_m,
                    "abs_error": error,
                }
            )

    gates = _mapping(coordination.get("gates"), "coordination.gates")
    basin_limit = _positive_int(
        gates.get("basin_mismatches_max"),
        "coordination.gates.basin_mismatches_max",
        minimum=0,
    )
    seeded_limit = _positive_int(
        gates.get("seeded_mismatches_max"),
        "coordination.gates.seeded_mismatches_max",
        minimum=0,
    )
    maximum_observed_branch_error = max(branch_errors)
    coordination_passed = basin_mismatches <= basin_limit and seeded_mismatches <= seeded_limit
    bifurcation_passed = maximum_observed_branch_error < max_branch_error
    coordination_metrics = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "delta": delta,
        "epsilon": epsilon,
        "grid_points": int(basin.labels.size),
        "analytic_basin_mismatches": basin_mismatches,
        "seeded_initializations": seed_count,
        "seeded_equilibrium_counts": seeded_counts,
        "seeded_mismatches": seeded_mismatches,
        "gates": {
            "basin_mismatches_max": basin_limit,
            "seeded_mismatches_max": seeded_limit,
            "minimum_seeds": 20,
        },
        "passed": coordination_passed,
    }
    bifurcation_metrics = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "a": a,
        "beta_over_a": ratios.tolist(),
        "predicted_positive_m": predicted_positive,
        "observed_positive_m": observed_positive,
        "observed_negative_m": observed_negative,
        "max_branch_abs_error": maximum_observed_branch_error,
        "max_symmetry_abs_error": max(symmetry_errors),
        "gate_max_branch_abs_error": max_branch_error,
        "passed": bifurcation_passed,
    }
    plot_data = {
        "ratios": ratios,
        "predicted_positive": np.asarray(predicted_positive),
        "observed_positive": np.asarray(observed_positive),
        "observed_negative": np.asarray(observed_negative),
    }
    return (
        basin,
        coordination_metrics,
        bifurcation_metrics,
        coordination_rows,
        branch_rows,
        plot_data,
    )


def _registered_output_paths(config: dict[str, Any], repository_root: Path) -> dict[str, Path]:
    task = config["task"]
    outputs = _mapping(config.get("outputs"), "outputs")
    keys: list[str] = []
    if task in {"all", "selection"}:
        keys.extend(("selection_metrics", "selection_predictions"))
        if "selection_figure" in outputs:
            keys.append("selection_figure")
    if task in {"all", "scaling"}:
        keys.extend(("scaling_metrics", "scaling_predictions"))
    if task in {"all", "coordination"}:
        keys.extend(
            (
                "coordination_metrics",
                "bifurcation_metrics",
                "coordination_predictions",
                "bifurcation_predictions",
            )
        )
        if "coordination_figure" in outputs:
            keys.append("coordination_figure")
    return {key: _output_path(repository_root, outputs.get(key), f"outputs.{key}") for key in keys}


def _produce_owned_artifacts(
    config: dict[str, Any],
    config_path: Path,
    repository_root: Path,
    started_at: str,
    *,
    run_git_metadata: dict[str, Any],
    output_paths: dict[str, Path],
    published_paths: dict[str, Path],
    overwrite: bool,
) -> tuple[bool, tuple[str, ...], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    from compbias.plots import plot_coordination_summary, plot_selection_comparison

    task = config["task"]
    all_passed: list[bool] = []
    generated: list[str] = []
    metric_payloads: dict[str, dict[str, Any]] = {}
    rollout_rows: dict[str, list[dict[str, Any]]] = {}

    if task in {"all", "selection"}:
        selection_metrics, rows, plot_records, error_ids = _run_selection(config)
        selection_metrics["run"] = _run_metadata(
            config_path=config_path,
            repository_root=repository_root,
            started_at=started_at,
            git_metadata=run_git_metadata,
            overwrite=overwrite,
        )
        metrics_path = output_paths["selection_metrics"]
        predictions_path = output_paths["selection_predictions"]
        _write_json(metrics_path, selection_metrics)
        _write_csv(
            predictions_path,
            rows,
            [
                "profile",
                "reward_mode",
                "algorithm",
                "seed",
                "error_id",
                "severity",
                "base_probability",
                "compensability",
                "predicted_probability",
                "observed_probability",
            ],
        )
        generated.extend(
            _display_output_path(published_paths[name], repository_root)
            for name in ("selection_metrics", "selection_predictions")
        )
        if "selection_figure" in output_paths:
            figure_path = output_paths["selection_figure"]
            selection_plot_data = {
                f"{record['profile']} | {record['algorithm']}": {
                    "predicted": record["predicted"],
                    "observed": record["observed"],
                }
                for record in plot_records
            }
            plot_selection_comparison(
                selection_plot_data,
                figure_path,
                action_labels=error_ids,
            )
            generated.append(
                _display_output_path(published_paths["selection_figure"], repository_root)
            )
        all_passed.append(bool(selection_metrics["passed"]))
        metric_payloads["selection"] = selection_metrics
        rollout_rows["selection"] = rows

    if task in {"all", "scaling"}:
        scaling_metrics, scaling_rows = _run_scaling(config)
        scaling_metrics["run"] = _run_metadata(
            config_path=config_path,
            repository_root=repository_root,
            started_at=started_at,
            git_metadata=run_git_metadata,
            overwrite=overwrite,
        )
        scaling_metrics_path = output_paths["scaling_metrics"]
        scaling_predictions_path = output_paths["scaling_predictions"]
        _write_json(scaling_metrics_path, scaling_metrics)
        _write_csv(
            scaling_predictions_path,
            scaling_rows,
            [
                "path",
                "error_id",
                "severity",
                "base_probability",
                "gain",
                "selected_probability",
            ],
        )
        generated.extend(
            _display_output_path(published_paths[name], repository_root)
            for name in ("scaling_metrics", "scaling_predictions")
        )
        all_passed.append(bool(scaling_metrics["passed"]))
        metric_payloads["scaling"] = scaling_metrics
        rollout_rows["scaling"] = scaling_rows

    if task in {"all", "coordination"}:
        (
            basin,
            coordination_metrics,
            bifurcation_metrics,
            coordination_rows,
            branch_rows,
            plot_data,
        ) = _run_coordination(config)
        run_metadata = _run_metadata(
            config_path=config_path,
            repository_root=repository_root,
            started_at=started_at,
            git_metadata=run_git_metadata,
            overwrite=overwrite,
        )
        coordination_metrics["run"] = run_metadata
        bifurcation_metrics["run"] = run_metadata
        coordination_metrics_path = output_paths["coordination_metrics"]
        bifurcation_metrics_path = output_paths["bifurcation_metrics"]
        coordination_predictions_path = output_paths["coordination_predictions"]
        bifurcation_predictions_path = output_paths["bifurcation_predictions"]
        _write_json(coordination_metrics_path, coordination_metrics)
        _write_json(bifurcation_metrics_path, bifurcation_metrics)
        _write_csv(
            coordination_predictions_path,
            coordination_rows,
            ["kind", "seed", "p0", "q0", "expected_equilibrium", "observed_equilibrium"],
        )
        _write_csv(
            bifurcation_predictions_path,
            branch_rows,
            [
                "beta_over_a",
                "branch",
                "initial_probability",
                "p_final",
                "q_final",
                "predicted_m",
                "observed_m",
                "abs_error",
            ],
        )
        generated.extend(
            _display_output_path(published_paths[name], repository_root)
            for name in (
                "coordination_metrics",
                "bifurcation_metrics",
                "coordination_predictions",
                "bifurcation_predictions",
            )
        )
        if "coordination_figure" in output_paths:
            figure_path = output_paths["coordination_figure"]
            plot_coordination_summary(
                {
                    "basin": basin,
                    "beta_over_a": plot_data["ratios"],
                    "predicted_positive": plot_data["predicted_positive"],
                    "observed_positive": plot_data["observed_positive"],
                    "observed_negative": plot_data["observed_negative"],
                },
                figure_path,
            )
            generated.append(
                _display_output_path(published_paths["coordination_figure"], repository_root)
            )
        all_passed.extend(
            [bool(coordination_metrics["passed"]), bool(bifurcation_metrics["passed"])]
        )
        metric_payloads["coordination"] = coordination_metrics
        metric_payloads["bifurcation"] = bifurcation_metrics
        rollout_rows["coordination"] = coordination_rows
        rollout_rows["bifurcation"] = branch_rows

    return all(all_passed), tuple(generated), metric_payloads, rollout_rows


def _tabular_run_id(start_timestamp: str, config_sha256: str) -> str:
    canonical = f"{start_timestamp}\0{config_sha256}".encode()
    return f"run-{hashlib.sha256(canonical).hexdigest()[:16]}"


def _numeric_prediction_arrays(
    rows_by_kind: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    import numpy as np

    arrays: dict[str, Any] = {}
    for kind, rows in rows_by_kind.items():
        numeric_fields = tuple(
            field
            for field in rows[0]
            if all(
                value is not None
                and not isinstance(value, (bool, str))
                and isinstance(value, (int, float))
                for value in (row.get(field) for row in rows)
            )
        )
        for field in numeric_fields:
            arrays[f"{kind}.{field}"] = np.asarray([row[field] for row in rows])
    if not arrays:
        raise RuntimeError("tabular run produced no numeric prediction arrays")
    return arrays


def _tabular_report(
    *,
    experiment: str,
    task: str,
    passed: bool,
    metric_payloads: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Phase B tabular run",
        "",
        f"- Experiment: `{experiment}`",
        f"- Task: `{task}`",
        f"- Overall gate: **{'PASS' if passed else 'FAIL'}**",
        "",
        "## Component gates",
        "",
    ]
    for kind, payload in metric_payloads.items():
        lines.append(f"- `{kind}`: {'PASS' if payload['passed'] else 'FAIL'}")
    return "\n".join(lines) + "\n"


def _write_run_bundle(
    *,
    config: dict[str, Any],
    repository_root: Path,
    environment: dict[str, object],
    config_sha256: str,
    passed: bool,
    metric_payloads: dict[str, dict[str, Any]],
    rows_by_kind: dict[str, list[dict[str, Any]]],
) -> None:
    from compbias.io.artifact_paths import validated_artifact_path
    from compbias.io.logging import (
        RunLogger,
        publishable_config_snapshot,
    )

    outputs = _mapping(config.get("outputs"), "outputs")
    log_root = validated_artifact_path(
        outputs.get("logs"),
        repository_root=repository_root,
        label="outputs.logs",
        suffix=None,
    )
    log_config = publishable_config_snapshot(
        config,
        path_fields=tuple(("outputs", key) for key in outputs),
        worktree=repository_root,
    )
    report = _tabular_report(
        experiment=config["experiment"],
        task=config["task"],
        passed=passed,
        metric_payloads=metric_payloads,
    )
    with RunLogger(
        root=log_root,
        experiment=config["experiment"],
        run_id=_tabular_run_id(str(environment["start_timestamp"]), config_sha256),
        config=log_config,
        environment=environment,
    ) as logger:
        for kind, payload in metric_payloads.items():
            logger.log_metrics({"kind": kind, **payload})
        for kind, rows in rows_by_kind.items():
            for row in rows:
                logger.log_rollout({"kind": kind, **row})
        logger.save_predictions(_numeric_prediction_arrays(rows_by_kind))
        logger.write_report(report)
        logger.finalize(checkpoint_hash=None)


def _run(
    config_path: Path,
    repository_root: Path,
    started_at: str,
    *,
    overwrite: bool = False,
) -> bool:
    config = _load_config(config_path)
    task = config["task"]
    from compbias.io.artifact_paths import (
        artifact_ownership_transaction,
        ensure_distinct_nonoverlapping,
        prepare_artifact_ownership,
    )

    registered_paths = _registered_output_paths(config, repository_root)
    ensure_distinct_nonoverlapping(registered_paths)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    primary_json = {
        "all": "selection_metrics",
        "selection": "selection_metrics",
        "scaling": "scaling_metrics",
        "coordination": "coordination_metrics",
    }[task]
    ownership = prepare_artifact_ownership(
        registered_paths,
        repository_root=repository_root,
        tool="scripts/train_tabular.py",
        experiment=config["experiment"],
        config_sha256=config_sha256,
        primary_json=primary_json,
        primary_schema_version=1,
        primary_experiment=config["experiment"],
        overwrite=overwrite,
    )
    command_arguments = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
    ]
    if overwrite:
        command_arguments.append("--overwrite")
    from compbias.io.logging import capture_environment

    environment = capture_environment(
        worktree=repository_root,
        dataset_manifest_hash=None,
        seed=config["seed"],
        model_revision=None,
        verl_revision=None,
        command=command_arguments,
    )
    run_git_metadata = _git_metadata(repository_root)
    with artifact_ownership_transaction(
        ownership,
        after_promote=lambda: _write_run_bundle(
            config=config,
            repository_root=repository_root,
            environment=environment,
            config_sha256=config_sha256,
            passed=passed,
            metric_payloads=metric_payloads,
            rows_by_kind=rows_by_kind,
        ),
    ) as staged_paths:
        passed, generated, metric_payloads, rows_by_kind = _produce_owned_artifacts(
            config,
            config_path,
            repository_root,
            started_at,
            run_git_metadata=run_git_metadata,
            output_paths=dict(staged_paths),
            published_paths=registered_paths,
            overwrite=overwrite,
        )
    print(f"Phase B tabular gate: {'PASS' if passed else 'FAIL'}")
    for path in generated:
        print(f"- {path}")
    return passed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Phase-B YAML config")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing validated artifact outputs",
    )
    args = parser.parse_args(argv)

    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        passed = _run(
            args.config.resolve(),
            repository_root,
            started_at,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
