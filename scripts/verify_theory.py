#!/usr/bin/env python3
"""Run the deterministic CPU verification suite for the Phase-A theory gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAX_PROPERTY_CASES = 100_000
_MAX_BASIN_GRID_COUNT = 256
_MAX_BIFURCATION_PLOT_COUNT = 10_000
_MAX_COORDINATION_HORIZON = 1_000.0
_MAX_COORDINATION_FIXTURES = 1_000


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


def _number_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    return _finite_float(value[0], f"{name}[0]"), _finite_float(value[1], f"{name}[1]")


def _output_path(repository_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    from compbias.io.artifact_paths import validated_artifact_path

    suffixes = {
        "outputs.report": ".md",
        "outputs.property_tests": ".json",
        "outputs.bifurcation_figure": ".png",
        "outputs.basin_figure": ".png",
    }
    suffix = suffixes.get(name)
    if suffix is None:
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
        "seed",
        "property_tests",
        "coordination",
        "bifurcation",
        "outputs",
    }
    reject_unknown_fields(config, allowed, label="configuration")
    nested = {
        "property_tests": {"cases", "identity_abs_tolerance", "finite_difference_step"},
        "coordination": {
            "delta",
            "epsilon",
            "horizon",
            "separatrix_tolerance",
            "fixture_endpoint_abs_tolerance",
            "basin_grid",
            "fixtures",
        },
        "bifurcation": {
            "a",
            "critical_search_interval",
            "critical_beta_abs_tolerance",
            "plot_beta_over_a",
        },
        "outputs": {
            "report",
            "property_tests",
            "bifurcation_figure",
            "basin_figure",
            "logs",
        },
    }
    for section, fields in nested.items():
        if section in config:
            reject_unknown_fields(config[section], fields, label=section)
    coordination = config.get("coordination")
    if isinstance(coordination, dict):
        if "basin_grid" in coordination:
            reject_unknown_fields(
                coordination["basin_grid"], {"start", "stop", "count"}, label="basin_grid"
            )
        fixtures = coordination.get("fixtures")
        if isinstance(fixtures, list):
            for index, fixture in enumerate(fixtures):
                reject_unknown_fields(
                    fixture, {"name", "initial", "expected"}, label=f"fixtures[{index}]"
                )
    bifurcation = config.get("bifurcation")
    if isinstance(bifurcation, dict) and "plot_beta_over_a" in bifurcation:
        reject_unknown_fields(
            bifurcation["plot_beta_over_a"],
            {"start", "stop", "count"},
            label="plot_beta_over_a",
        )
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be exactly 1")
    _positive_int(config.get("seed"), "seed", minimum=0)
    from compbias.io.artifact_paths import validate_experiment_name

    validate_experiment_name(config.get("experiment"))
    return config


def _grid(specification: dict[str, Any], name: str, *, maximum_count: int):
    import numpy as np

    start = _finite_float(specification.get("start"), f"{name}.start")
    stop = _finite_float(specification.get("stop"), f"{name}.stop")
    count = _positive_int(
        specification.get("count"),
        f"{name}.count",
        minimum=2,
        maximum=maximum_count,
    )
    if not start < stop:
        raise ValueError(f"{name}.start must be smaller than {name}.stop")
    return np.linspace(start, stop, count, dtype=np.float64)


def _error_summary(errors, tolerance: float) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("a property-test error vector was empty or non-finite")
    maximum = float(np.max(values))
    return {
        "checks": int(values.size),
        "max_abs_error": maximum,
        "mean_abs_error": float(np.mean(values)),
        "p95_abs_error": float(np.quantile(values, 0.95)),
        "tolerance": tolerance,
        "passed": maximum < tolerance,
    }


def _run_random_properties(*, seed: int, cases: int, tolerance: float, fd_step: float):
    import numpy as np

    from compbias.rl.exact_kl import exact_kl_projection
    from compbias.theory.bregman import bregman_three_point
    from compbias.theory.coordination import CoordinationParams, jacobian, vector_field
    from compbias.theory.lockin import repeated_selection, repeated_selection_closed_form
    from compbias.theory.scaling import (
        relative_compensability_gain,
        severity_scaling_derivative,
    )
    from compbias.theory.selection import (
        binary_compensability_multiplier,
        expectation_shift,
        selected_error_distribution,
    )

    rng = np.random.default_rng(seed)
    errors: dict[str, list[float]] = {
        "exact_kl_projection": [],
        "selection_covariance": [],
        "pairwise_odds": [],
        "scaling_finite_difference": [],
        "repeated_selection_closed_form": [],
        "coordination_jacobian": [],
        "bregman_three_point": [],
    }

    for _ in range(cases):
        size = int(rng.integers(2, 10))
        raw = rng.uniform(0.05, 1.0, size=size)
        base = raw / raw.sum()
        rewards = rng.uniform(-1.0, 1.0, size=size)
        beta = float(rng.uniform(0.2, 2.0))

        observed_exact = np.asarray(exact_kl_projection(base, rewards, beta))
        direct_weights = base * np.exp((rewards - np.max(rewards)) / beta)
        direct_exact = direct_weights / direct_weights.sum()
        errors["exact_kl_projection"].append(float(np.sum(np.abs(observed_exact - direct_exact))))

        severity = rng.uniform(-2.0, 3.0, size=size)
        compensability = rng.uniform(0.0, 1.0, size=size)
        multiplier = np.asarray(binary_compensability_multiplier(compensability, beta))
        selected = np.asarray(selected_error_distribution(base, multiplier))
        observed_shift = float(expectation_shift(base, severity, multiplier))
        mean_severity = float(base @ severity)
        mean_multiplier = float(base @ multiplier)
        covariance = float(base @ ((severity - mean_severity) * (multiplier - mean_multiplier)))
        predicted_shift = covariance / mean_multiplier
        errors["selection_covariance"].append(abs(observed_shift - predicted_shift))

        odds_error = 0.0
        for left in range(size):
            for right in range(left + 1, size):
                observed_odds = np.log(selected[left] / selected[right])
                predicted_odds = np.log(base[left] / base[right]) + np.log(
                    multiplier[left] / multiplier[right]
                )
                odds_error = max(odds_error, abs(float(observed_odds - predicted_odds)))
        errors["pairwise_odds"].append(odds_error)

        gain = rng.uniform(-0.5, 0.5, size=size)
        kappa = float(rng.uniform(0.1, 0.9))
        scaling_multiplier = np.exp(kappa * gain)
        scaling_selected = np.asarray(selected_error_distribution(base, scaling_multiplier))
        relative_gain = np.asarray(
            relative_compensability_gain(scaling_multiplier, scaling_multiplier * gain)
        )
        analytic_derivative = float(
            severity_scaling_derivative(scaling_selected, severity, relative_gain)
        )
        upper = np.asarray(selected_error_distribution(base, np.exp((kappa + fd_step) * gain)))
        lower = np.asarray(selected_error_distribution(base, np.exp((kappa - fd_step) * gain)))
        finite_difference = float((upper @ severity - lower @ severity) / (2.0 * fd_step))
        errors["scaling_finite_difference"].append(abs(analytic_derivative - finite_difference))

        lockin_multiplier = np.exp(rng.uniform(-1.0, 1.0, size=size))
        steps = int(rng.integers(0, 20))
        iterated = np.asarray(repeated_selection(base, lockin_multiplier, steps))
        closed_form = np.asarray(repeated_selection_closed_form(base, lockin_multiplier, steps))
        errors["repeated_selection_closed_form"].append(
            float(np.max(np.abs(iterated - closed_form)))
        )

        params = CoordinationParams(
            delta=float(rng.uniform(0.2, 2.0)),
            epsilon=float(rng.uniform(0.2, 2.0)),
        )
        state = rng.uniform(0.05, 0.95, size=2)
        analytic_jacobian = jacobian(state, params)
        numeric_jacobian = np.empty((2, 2), dtype=np.float64)
        for column in range(2):
            direction = np.zeros(2, dtype=np.float64)
            direction[column] = fd_step
            numeric_jacobian[:, column] = (
                vector_field(0.0, state + direction, params)
                - vector_field(0.0, state - direction, params)
            ) / (2.0 * fd_step)
        errors["coordination_jacobian"].append(
            float(np.max(np.abs(analytic_jacobian - numeric_jacobian)))
        )

        dimension = int(rng.integers(1, 9))
        factor = rng.normal(size=(dimension, dimension))
        matrix = factor.T @ factor + 0.25 * np.eye(dimension)
        x, y, z = (rng.normal(size=dimension) for _part in range(3))

        def phi(value, matrix=matrix):
            return 0.5 * float(value @ matrix @ value)

        def grad_phi(value, matrix=matrix):
            return matrix @ value

        total, first, second, interaction = bregman_three_point(x, y, z, phi, grad_phi)
        independent = 0.5 * float((x - z) @ matrix @ (x - z))
        errors["bregman_three_point"].append(
            max(
                abs(float(total - first - second - interaction)),
                abs(float(total - independent)),
            )
        )

    summaries = {name: _error_summary(values, tolerance) for name, values in errors.items()}
    return {
        "random_cases": cases,
        "identity_checks": int(cases * len(summaries)),
        "seed": seed,
        "identities": summaries,
        "passed": all(summary["passed"] for summary in summaries.values()),
    }


def _run_coordination_checks(config: dict[str, Any]):
    import numpy as np
    from scipy.integrate import solve_ivp

    from compbias.theory.coordination import (
        CoordinationParams,
        basin_map,
        vector_field,
    )

    delta = _finite_float(config.get("delta"), "coordination.delta", positive=True)
    epsilon = _finite_float(config.get("epsilon"), "coordination.epsilon", positive=True)
    if not np.isclose(delta, epsilon, rtol=0.0, atol=1e-14):
        raise ValueError("Phase-A analytic basin verification requires delta == epsilon")
    horizon = _finite_float(config.get("horizon"), "coordination.horizon", positive=True)
    if horizon > _MAX_COORDINATION_HORIZON:
        raise ValueError(f"coordination.horizon must be at most {_MAX_COORDINATION_HORIZON:g}")
    tolerance = _finite_float(
        config.get("separatrix_tolerance"),
        "coordination.separatrix_tolerance",
    )
    if tolerance < 0.0:
        raise ValueError("coordination.separatrix_tolerance must be nonnegative")
    fixture_tolerance = _finite_float(
        config.get("fixture_endpoint_abs_tolerance"),
        "coordination.fixture_endpoint_abs_tolerance",
        positive=True,
    )
    params = CoordinationParams(delta=delta, epsilon=epsilon)
    grid = _grid(
        _mapping(config.get("basin_grid"), "coordination.basin_grid"),
        "basin_grid",
        maximum_count=_MAX_BASIN_GRID_COUNT,
    )
    basin = basin_map(
        grid,
        grid,
        params,
        horizon=horizon,
        separatrix_tolerance=tolerance,
    )
    score = basin.p + basin.q - 1.0
    expected = np.where(
        score > tolerance,
        "truthful",
        np.where(score < -tolerance, "compensatory", "separatrix"),
    )
    mismatches = int(np.count_nonzero(basin.labels != expected))

    fixture_config = config.get("fixtures")
    if not isinstance(fixture_config, list) or not fixture_config:
        raise ValueError("coordination.fixtures must be a non-empty list")
    if len(fixture_config) > _MAX_COORDINATION_FIXTURES:
        raise ValueError(
            f"coordination.fixtures must contain at most {_MAX_COORDINATION_FIXTURES} entries"
        )
    fixtures: list[dict[str, Any]] = []
    for index, raw_fixture in enumerate(fixture_config):
        fixture = _mapping(raw_fixture, f"coordination.fixtures[{index}]")
        name = fixture.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"coordination.fixtures[{index}].name must be non-empty")
        initial = _number_pair(fixture.get("initial"), f"fixture {name}.initial")
        expected_endpoint = _number_pair(fixture.get("expected"), f"fixture {name}.expected")
        if any(not 0.0 <= value <= 1.0 for value in (*initial, *expected_endpoint)):
            raise ValueError(f"fixture {name} probabilities must lie in [0, 1]")
        solution = solve_ivp(
            vector_field,
            (0.0, horizon),
            initial,
            args=(params,),
            rtol=1e-10,
            atol=1e-12,
            max_step=min(0.1, horizon),
        )
        if not solution.success:
            raise RuntimeError(f"coordination ODE failed for {name}: {solution.message}")
        endpoint = solution.y[:, -1]
        fixture_error = float(np.max(np.abs(endpoint - np.asarray(expected_endpoint))))
        fixtures.append(
            {
                "name": name,
                "initial": list(initial),
                "expected_endpoint": list(expected_endpoint),
                "observed_endpoint": endpoint.tolist(),
                "max_abs_error": fixture_error,
                "tolerance": fixture_tolerance,
                "passed": fixture_error < fixture_tolerance,
            }
        )

    fixtures_passed = all(fixture["passed"] for fixture in fixtures)
    return basin, {
        "grid_points": int(basin.labels.size),
        "analytic_basin_mismatches": mismatches,
        "fixtures": fixtures,
        "fixture_endpoint_abs_tolerance": fixture_tolerance,
        "passed": mismatches == 0 and fixtures_passed,
    }


def _run_bifurcation_checks(config: dict[str, Any]):
    import numpy as np
    from scipy.optimize import brentq

    from compbias.theory.coordination import (
        CoordinationParams,
        jacobian,
        symmetric_bifurcation_branch,
    )

    a = _finite_float(config.get("a"), "bifurcation.a", positive=True)
    interval = _number_pair(
        config.get("critical_search_interval"),
        "bifurcation.critical_search_interval",
    )
    if not 0.0 < interval[0] < interval[1]:
        raise ValueError("critical_search_interval must be increasing and positive")

    def leading_eigenvalue(beta: float) -> float:
        params = CoordinationParams(
            delta=a,
            epsilon=a,
            beta_p=beta,
            beta_q=beta,
        )
        eigenvalues = np.linalg.eigvalsh(jacobian((0.5, 0.5), params))
        return float(np.max(eigenvalues))

    critical_beta = float(brentq(leading_eigenvalue, interval[0], interval[1]))
    analytic_critical_beta = a / 2.0
    critical_error = abs(critical_beta - analytic_critical_beta)
    tolerance = _finite_float(
        config.get("critical_beta_abs_tolerance"),
        "bifurcation.critical_beta_abs_tolerance",
        positive=True,
    )
    ratios = _grid(
        _mapping(config.get("plot_beta_over_a"), "bifurcation.plot_beta_over_a"),
        "plot_beta_over_a",
        maximum_count=_MAX_BIFURCATION_PLOT_COUNT,
    )
    branch = symmetric_bifurcation_branch(ratios)
    below = (ratios > 0.0) & (ratios < 0.5)
    boundary_margin = float(np.sqrt(np.finfo(np.float64).eps))
    finite_branch = below & (branch.positive < 1.0 - boundary_margin)
    residuals = np.zeros_like(ratios)
    residuals[finite_branch] = np.abs(
        2.0 * ratios[finite_branch] * np.arctanh(branch.positive[finite_branch])
        - branch.positive[finite_branch]
    )
    return branch, {
        "a": a,
        "observed_critical_beta": critical_beta,
        "analytic_critical_beta": analytic_critical_beta,
        "critical_beta_abs_error": critical_error,
        "critical_beta_abs_tolerance": tolerance,
        "max_nonzero_branch_residual": float(np.max(residuals)),
        "float_saturated_branch_points": int(np.count_nonzero(below & ~finite_branch)),
        "branch_residual_boundary_margin": boundary_margin,
        "passed": critical_error < tolerance,
    }


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
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": status not in {"", "unavailable"},
    }


def _render_report(payload: dict[str, Any], config_path: str) -> str:
    properties = payload["property_tests"]
    coordination = payload["coordination"]
    bifurcation = payload["bifurcation"]
    lines = [
        "# Phase A theory verification",
        "",
        f"Overall gate: **{'PASS' if payload['passed'] else 'FAIL'}**.",
        "",
        "## Reproducibility",
        "",
        f"- Config: `{config_path}`",
        f"- Config SHA-256: `{payload['run']['config_sha256']}`",
        f"- Seed: `{properties['seed']}`",
        f"- Random cases per identity: `{properties['random_cases']}`",
        f"- Total random identity checks: `{properties['identity_checks']}`",
        "",
        "## Randomized identities",
        "",
        "| Identity | Checks | Max abs. error | Threshold | Gate |",
        "|---|---:|---:|---:|:---:|",
    ]
    for name, result in properties["identities"].items():
        lines.append(
            f"| `{name}` | {result['checks']} | {result['max_abs_error']:.3e} | "
            f"<{result['tolerance']:.1e} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Coordination ODE and analytic basin",
            "",
            f"- Grid points checked: `{coordination['grid_points']}`",
            f"- Analytic basin mismatches: `{coordination['analytic_basin_mismatches']}`",
        ]
    )
    for fixture in coordination["fixtures"]:
        lines.append(
            f"- `{fixture['name']}` endpoint: `{fixture['observed_endpoint']}` "
            f"(max abs. error `{fixture['max_abs_error']:.3e}`, required "
            f"`<{fixture['tolerance']:.1e}`, "
            f"{'PASS' if fixture['passed'] else 'FAIL'})."
        )
    lines.extend(
        [
            "",
            "## Symmetric KL bifurcation",
            "",
            f"- Analytic critical beta: `{bifurcation['analytic_critical_beta']:.12g}`",
            f"- Numerically observed critical beta: `{bifurcation['observed_critical_beta']:.12g}`",
            f"- Absolute error: `{bifurcation['critical_beta_abs_error']:.3e}` "
            f"(required `<{bifurcation['critical_beta_abs_tolerance']:.1e}`).",
            f"- Maximum nonzero-branch residual: "
            f"`{bifurcation['max_nonzero_branch_residual']:.3e}`.",
            "",
            "## Exit artifacts",
            "",
        ]
    )
    for name, path in payload["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def _run_id(started_at: str, config_sha256: str) -> str:
    canonical = f"{started_at}\0{config_sha256}".encode()
    return f"run-{hashlib.sha256(canonical).hexdigest()[:16]}"


def _write_run_bundle(
    *,
    config: dict[str, Any],
    repository_root: Path,
    environment: dict[str, object],
    payload: dict[str, Any],
    basin: Any,
    branch: Any,
    report: str,
) -> None:
    import numpy as np

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
    with RunLogger(
        root=log_root,
        experiment=config["experiment"],
        run_id=_run_id(str(environment["start_timestamp"]), payload["run"]["config_sha256"]),
        config=log_config,
        environment=environment,
    ) as logger:
        for identity, result in payload["property_tests"]["identities"].items():
            logger.log_metrics({"kind": "identity", "identity": identity, **result})
        logger.log_metrics({"kind": "coordination", **payload["coordination"]})
        logger.log_metrics({"kind": "bifurcation", **payload["bifurcation"]})
        for fixture in payload["coordination"]["fixtures"]:
            logger.log_rollout({"kind": "coordination_fixture", **fixture})
        logger.log_rollout(
            {
                "kind": "random_property_batch",
                "seed": payload["property_tests"]["seed"],
                "random_cases": payload["property_tests"]["random_cases"],
                "identity_checks": payload["property_tests"]["identity_checks"],
                "passed": payload["property_tests"]["passed"],
            }
        )
        logger.save_predictions(
            {
                "basin_p": np.asarray(basin.p),
                "basin_q": np.asarray(basin.q),
                "basin_label_code": np.asarray(
                    np.vectorize(
                        {"compensatory": -1, "separatrix": 0, "truthful": 1}.__getitem__,
                        otypes=[np.int8],
                    )(basin.labels)
                ),
                "beta_over_a": np.asarray(branch.beta_over_a),
                "bifurcation_center": np.asarray(branch.center),
                "bifurcation_positive": np.asarray(branch.positive),
                "bifurcation_negative": np.asarray(branch.negative),
            }
        )
        logger.write_report(report)
        logger.finalize(checkpoint_hash=None)


def _run(
    config_path: Path,
    repository_root: Path,
    started_at: str,
    *,
    overwrite: bool = False,
) -> bool:
    import numpy as np
    import scipy

    from compbias.io.logging import capture_environment, publishable_command, publishable_path
    from compbias.plots import plot_basin_map, plot_bifurcation

    config = _load_config(config_path)
    property_config = _mapping(config.get("property_tests"), "property_tests")
    cases = _positive_int(
        property_config.get("cases"),
        "property_tests.cases",
        minimum=1000,
        maximum=_MAX_PROPERTY_CASES,
    )
    tolerance = _finite_float(
        property_config.get("identity_abs_tolerance"),
        "property_tests.identity_abs_tolerance",
        positive=True,
    )
    if tolerance > 1e-8:
        raise ValueError("identity_abs_tolerance cannot exceed the Phase-A gate of 1e-8")
    fd_step = _finite_float(
        property_config.get("finite_difference_step"),
        "property_tests.finite_difference_step",
        positive=True,
    )
    seed = _positive_int(config.get("seed"), "seed", minimum=0)

    outputs = _mapping(config.get("outputs"), "outputs")
    report_path = _output_path(repository_root, outputs.get("report"), "outputs.report")
    property_path = _output_path(
        repository_root, outputs.get("property_tests"), "outputs.property_tests"
    )
    bifurcation_path = _output_path(
        repository_root,
        outputs.get("bifurcation_figure"),
        "outputs.bifurcation_figure",
    )
    basin_path = _output_path(repository_root, outputs.get("basin_figure"), "outputs.basin_figure")
    from compbias.io.artifact_paths import (
        artifact_ownership_transaction,
        ensure_distinct_nonoverlapping,
        prepare_artifact_ownership,
    )

    output_paths = {
        "report": report_path,
        "property_tests": property_path,
        "bifurcation_figure": bifurcation_path,
        "basin_figure": basin_path,
    }
    ensure_distinct_nonoverlapping(output_paths)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    ownership = prepare_artifact_ownership(
        output_paths,
        repository_root=repository_root,
        tool="scripts/verify_theory.py",
        experiment=config["experiment"],
        config_sha256=config_sha256,
        primary_json="property_tests",
        primary_schema_version=1,
        primary_experiment=config["experiment"],
        overwrite=overwrite,
    )

    property_results = _run_random_properties(
        seed=seed,
        cases=cases,
        tolerance=tolerance,
        fd_step=fd_step,
    )
    basin, coordination_results = _run_coordination_checks(
        _mapping(config.get("coordination"), "coordination")
    )
    branch, bifurcation_results = _run_bifurcation_checks(
        _mapping(config.get("bifurcation"), "bifurcation")
    )

    output_mapping = {
        "property_tests": str(property_path.relative_to(repository_root)),
        "report": str(report_path.relative_to(repository_root)),
        "bifurcation_figure": str(bifurcation_path.relative_to(repository_root)),
        "basin_figure": str(basin_path.relative_to(repository_root)),
    }
    passed = bool(
        property_results["passed"]
        and coordination_results["passed"]
        and bifurcation_results["passed"]
    )
    config_display = publishable_path(config_path, worktree=repository_root)
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
    environment = capture_environment(
        worktree=repository_root,
        dataset_manifest_hash=None,
        seed=seed,
        model_revision=None,
        verl_revision=None,
        command=command_arguments,
    )
    payload = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "passed": passed,
        "property_tests": property_results,
        "coordination": coordination_results,
        "bifurcation": bifurcation_results,
        "outputs": output_mapping,
        "run": {
            "started_at_utc": started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(command),
            "config": config_display,
            "config_sha256": config_sha256,
            "git": _git_metadata(repository_root),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "device": "cpu",
        },
    }
    report = _render_report(payload, config_display)
    with artifact_ownership_transaction(
        ownership,
        after_promote=lambda: _write_run_bundle(
            config=config,
            repository_root=repository_root,
            environment=environment,
            payload=payload,
            basin=basin,
            branch=branch,
            report=report,
        ),
    ) as staged:
        plot_basin_map(basin, staged["basin_figure"])
        plot_bifurcation(branch, staged["bifurcation_figure"])
        staged["property_tests"].write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staged["report"].write_text(report, encoding="utf-8")
    print(report, end="")
    return passed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Phase-A YAML config")
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
    config_path = args.config.resolve()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        passed = _run(
            config_path,
            repository_root,
            started_at,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
