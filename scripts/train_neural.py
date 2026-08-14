#!/usr/bin/env python3
"""Run deterministic CPU small-neural diagnostics and Phase C sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = REPOSITORY_ROOT / "artifacts/metrics/neural_summary.json"
DEFAULT_RUNS = REPOSITORY_ROOT / "artifacts/predictions/neural_runs.csv"
DEFAULT_TRAJECTORIES = REPOSITORY_ROOT / "artifacts/predictions/neural_trajectories.csv"
DEFAULT_FIGURE = REPOSITORY_ROOT / "artifacts/figures/neural_trajectories.png"
DEFAULT_LOG_ROOT = REPOSITORY_ROOT / "artifacts/logs"
VALID_PROFILES = ("truth_aligned", "flat", "spurious")
VALID_MODES = ("perception_only", "reasoning_only", "joint")
BOOTSTRAP_SEED = 20260814
MAX_NEURAL_SEEDS = 128
MAX_NEURAL_WORKLOAD = 1_000_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional YAML configuration")
    parser.add_argument("--profile", choices=VALID_PROFILES)
    parser.add_argument("--profiles", nargs="+", choices=VALID_PROFILES)
    parser.add_argument("--mode", choices=VALID_MODES)
    parser.add_argument("--modes", nargs="+", choices=VALID_MODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=("cpu",))
    parser.add_argument("--joint-profile", choices=VALID_PROFILES)
    parser.add_argument("--output", type=Path, help="legacy result or batch summary JSON")
    parser.add_argument("--runs-output", type=Path)
    parser.add_argument("--trajectories-output", type=Path)
    parser.add_argument("--figure-output", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--experiment", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing validated artifact outputs",
    )
    return parser


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    _ensure_local_package()
    from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

    loaded = load_yaml_mapping(path)
    allowed = {
        "schema_version",
        "experiment",
        "profile",
        "profiles",
        "mode",
        "modes",
        "seed",
        "seeds",
        "steps",
        "learning_rate",
        "device",
        "joint_profile",
        "training",
        "reasoner",
        "outputs",
    }
    reject_unknown_fields(loaded, allowed, label="configuration")
    nested = {
        "training": {
            "profile",
            "profiles",
            "mode",
            "modes",
            "seed",
            "seeds",
            "steps",
            "learning_rate",
            "device",
        },
        "reasoner": {"profile"},
        "outputs": {"metrics", "runs", "trajectories", "figure", "logs"},
    }
    for section, fields in nested.items():
        if section in loaded:
            reject_unknown_fields(loaded[section], fields, label=section)
    return loaded


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _unique_tuple(values: Iterable[Any], name: str) -> tuple[Any, ...]:
    items = tuple(values)
    if not items:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(dict.fromkeys(items))


def _configured_values(
    config: Mapping[str, Any], plural: str, singular: str, default: Any
) -> tuple[Any, ...]:
    raw = config.get(plural, config.get(singular, default))
    values = raw if isinstance(raw, (list, tuple)) else (raw,)
    return _unique_tuple(values, plural)


def _resolve_path(
    raw: object,
    default: Path,
    *,
    label: str,
    suffix: str | None,
) -> Path:
    _ensure_local_package()
    from compbias.io.artifact_paths import validated_artifact_path

    value = default if raw is None else Path(str(raw)).expanduser()
    return validated_artifact_path(
        value,
        repository_root=REPOSITORY_ROOT,
        label=label,
        suffix=suffix,
    )


def _select(cli_plural: object, cli_single: object, configured: tuple[Any, ...]) -> tuple[Any, ...]:
    if cli_plural is not None:
        return _unique_tuple(cli_plural, "command-line values")
    if cli_single is not None:
        return (cli_single,)
    return configured


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_seeds(values: Iterable[object]) -> tuple[int, ...]:
    seeds = _unique_tuple(values, "seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique non-negative integers")
    return tuple(int(seed) for seed in seeds)


def _resolve_settings(args: argparse.Namespace, loaded: Mapping[str, Any]) -> dict[str, Any]:
    training = _mapping(loaded.get("training"), "training")
    outputs = _mapping(loaded.get("outputs"), "outputs")
    reasoner = _mapping(loaded.get("reasoner"), "reasoner")
    configured_profiles = _configured_values(
        loaded, "profiles", "profile", reasoner.get("profile", "spurious")
    )
    configured_modes = _configured_values(loaded, "modes", "mode", training.get("mode", "joint"))
    configured_seeds = _configured_values(
        loaded, "seeds", "seed", training.get("seeds", training.get("seed", 0))
    )
    profiles = _select(args.profiles, args.profile, configured_profiles)
    modes = _select(args.modes, args.mode, configured_modes)
    seeds = _nonnegative_seeds(_select(args.seeds, args.seed, configured_seeds))
    if len(seeds) > MAX_NEURAL_SEEDS:
        raise ValueError(f"seeds must contain at most {MAX_NEURAL_SEEDS} values")
    configured_steps = loaded.get("steps", training.get("steps", 48))
    steps = _positive_int(args.steps if args.steps is not None else configured_steps, "steps")
    _ensure_local_package()
    from compbias.rl.neural_experiment import MAX_NEURAL_STEPS

    if steps > MAX_NEURAL_STEPS:
        raise ValueError(f"steps must be at most {MAX_NEURAL_STEPS}")
    configured_rate = loaded.get("learning_rate", training.get("learning_rate", 0.8))
    learning_rate = args.learning_rate if args.learning_rate is not None else configured_rate
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
    ):
        raise ValueError("learning_rate must be a positive finite number")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be a positive finite number")
    device = args.device or loaded.get("device", training.get("device", "cpu"))
    if device != "cpu":
        raise ValueError("the Phase C diagnostic requires device='cpu'")
    settings = {
        "profiles": tuple(str(value) for value in profiles),
        "modes": tuple(str(value) for value in modes),
        "seeds": seeds,
        "steps": steps,
        "learning_rate": float(learning_rate),
        "device": device,
        "joint_profile": args.joint_profile or loaded.get("joint_profile", "spurious"),
        "experiment": args.experiment or loaded.get("experiment", "neural_phase_c"),
        "metrics_output": _resolve_path(
            args.output or outputs.get("metrics"),
            DEFAULT_METRICS,
            label="outputs.metrics",
            suffix=".json",
        ),
        "runs_output": _resolve_path(
            args.runs_output or outputs.get("runs"),
            DEFAULT_RUNS,
            label="outputs.runs",
            suffix=".csv",
        ),
        "trajectories_output": _resolve_path(
            args.trajectories_output or outputs.get("trajectories"),
            DEFAULT_TRAJECTORIES,
            label="outputs.trajectories",
            suffix=".csv",
        ),
        "figure_output": _resolve_path(
            args.figure_output or outputs.get("figure"),
            DEFAULT_FIGURE,
            label="outputs.figure",
            suffix=".png",
        ),
        "log_root": _resolve_path(
            args.log_root or outputs.get("logs"),
            DEFAULT_LOG_ROOT,
            label="outputs.logs",
            suffix=None,
        ),
    }
    condition_count = sum(
        1 if mode == "joint" else len(settings["profiles"]) for mode in settings["modes"]
    )
    if condition_count * len(seeds) * steps > MAX_NEURAL_WORKLOAD:
        raise ValueError(
            f"scalar neural workload must be at most {MAX_NEURAL_WORKLOAD} training steps"
        )
    from compbias.io.artifact_paths import ensure_distinct_nonoverlapping

    ensure_distinct_nonoverlapping(
        {
            "metrics": settings["metrics_output"],
            "runs": settings["runs_output"],
            "trajectories": settings["trajectories_output"],
            "figure": settings["figure_output"],
            "logs": settings["log_root"],
        }
    )
    return settings


def _validate_choices(settings: Mapping[str, Any]) -> None:
    _ensure_local_package()
    from compbias.io.artifact_paths import validate_experiment_name

    unknown_profiles = sorted(set(settings["profiles"]) - set(VALID_PROFILES))
    unknown_modes = sorted(set(settings["modes"]) - set(VALID_MODES))
    if unknown_profiles:
        raise ValueError(f"unknown profiles: {', '.join(unknown_profiles)}")
    if unknown_modes:
        raise ValueError(f"unknown modes: {', '.join(unknown_modes)}")
    if settings["joint_profile"] not in VALID_PROFILES:
        raise ValueError(f"unknown joint_profile: {settings['joint_profile']!r}")
    if "joint" in settings["modes"] and settings["joint_profile"] != "spurious":
        raise ValueError("joint mode requires joint_profile='spurious'")
    validate_experiment_name(settings["experiment"])


def _conditions(settings: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Deduplicate profile-invariant joint runs under one explicit label."""

    profiles = settings["profiles"]
    conditions = tuple(
        (settings["joint_profile"] if mode == "joint" else profile, mode)
        for mode in settings["modes"]
        for profile in (profiles[:1] if mode == "joint" else profiles)
    )
    return tuple(dict.fromkeys(conditions))


def _protocol_notes(settings: Mapping[str, Any]) -> tuple[str, ...]:
    notes = (
        "perception_shift is final minus initial perception loss; negative means improvement",
        "OOD accuracy executes paired error catalogs while preserving canonical solutions",
    )
    if "joint" in settings["modes"]:
        notes += (
            "joint dynamics are profile-invariant in the current core API; all joint runs are "
            f"deduplicated and labeled profile={settings['joint_profile']}",
        )
    return notes


def _ensure_local_package() -> None:
    source = str(REPOSITORY_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def _run_all(settings: Mapping[str, Any]) -> tuple[Any, ...]:
    _ensure_local_package()
    from compbias.rl.neural_experiment import NeuralExperimentConfig, run_neural_experiment

    return tuple(
        run_neural_experiment(
            NeuralExperimentConfig(
                profile=profile,
                mode=mode,
                seed=seed,
                steps=settings["steps"],
                device=settings["device"],
                learning_rate=settings["learning_rate"],
            )
        )
        for profile, mode in _conditions(settings)
        for seed in settings["seeds"]
    )


def _run_id(result: Any) -> str:
    return f"{result.profile}-{result.mode}-seed{result.seed:04d}"


def _execution_run_id(result: Any, environment: Mapping[str, object]) -> str:
    payload = json.dumps(
        {
            "logical_run_id": _run_id(result),
            "git_commit": environment.get("git_commit"),
            "git_dirty": environment.get("git_dirty"),
            "start_timestamp": environment.get("start_timestamp"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{_run_id(result)}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _run_rows(results: Sequence[Any]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "run_id": _run_id(result),
            "profile": result.profile,
            "mode": result.mode,
            "seed": result.seed,
            "steps": len(result.history) - 1,
            "perception_shift": result.perception_shift,
            "perception_accuracy_shift": -result.perception_shift,
            "equilibrium_mode": result.equilibrium_mode,
            "iid_accuracy": result.iid_accuracy,
            "ood_accuracy": result.ood_accuracy,
            "iid_ood_gap": result.iid_accuracy - result.ood_accuracy,
            "perception_accuracy": result.perception_accuracy,
            "canonical_reasoning_accuracy": result.canonical_reasoning_accuracy,
            "paired_sample_id": result.paired_sample_id,
            "iid_error_mechanism": result.iid_error_mechanism,
            "ood_error_mechanism": result.ood_error_mechanism,
            "iid_correctness_matrix": json.dumps(result.iid_correctness_matrix),
            "ood_correctness_matrix": json.dumps(result.ood_correctness_matrix),
        }
        for result in results
    )


def _trajectory_rows(results: Sequence[Any]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "run_id": _run_id(result),
            "profile": result.profile,
            "mode": result.mode,
            "seed": result.seed,
            "equilibrium_mode": result.equilibrium_mode,
            **asdict(checkpoint),
        }
        for result in results
        for checkpoint in result.history
    )


def _mean_ci(values: Sequence[float], *, salt: int) -> dict[str, object]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED + salt)
    samples = generator.choice(array, size=(10_000, array.size), replace=True).mean(axis=1)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "bootstrap_95_ci": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
    }


def _group_summaries(results: Sequence[Any]) -> tuple[dict[str, object], ...]:
    keys = tuple(dict.fromkeys((result.profile, result.mode) for result in results))
    metrics = (
        "perception_shift",
        "perception_accuracy",
        "canonical_reasoning_accuracy",
        "iid_accuracy",
        "ood_accuracy",
    )
    summaries = []
    for index, (profile, mode) in enumerate(keys):
        group = tuple(
            result for result in results if result.profile == profile and result.mode == mode
        )
        summaries.append(
            {
                "profile": profile,
                "mode": mode,
                "seed_count": len(group),
                "equilibrium_counts": dict(
                    sorted(Counter(r.equilibrium_mode for r in group).items())
                ),
                **{
                    metric: _mean_ci(
                        [float(getattr(result, metric)) for result in group],
                        salt=index * len(metrics) + metric_index,
                    )
                    for metric_index, metric in enumerate(metrics)
                },
                "iid_ood_gap": _mean_ci(
                    [result.iid_accuracy - result.ood_accuracy for result in group],
                    salt=100 + index,
                ),
            }
        )
    return tuple(summaries)


def _direction_gate(results: Sequence[Any]) -> dict[str, object]:
    expected = {"truth_aligned": "negative", "spurious": "positive"}
    observations = {}
    for profile, direction in expected.items():
        values = [
            result.perception_shift
            for result in results
            if result.profile == profile and result.mode == "perception_only"
        ]
        mean = sum(values) / len(values) if values else None
        passed = mean is not None and (mean < 0 if direction == "negative" else mean > 0)
        observations[profile] = {
            "expected_perception_loss_shift": direction,
            "mean_perception_shift": mean,
            "passed": passed,
        }
    return {
        "passed": all(item["passed"] for item in observations.values()),
        "observations": observations,
    }


def _joint_gate(results: Sequence[Any]) -> dict[str, object]:
    joint = tuple(result for result in results if result.mode == "joint")
    counts = Counter(result.equilibrium_mode for result in joint)
    seed_count = len({result.seed for result in joint})
    both = counts["truthful"] > 0 and counts["compensatory"] > 0
    high_iid = bool(joint) and min(result.iid_accuracy for result in joint) >= 0.75
    return {
        "passed": seed_count >= 10 and both and high_iid,
        "seed_count": seed_count,
        "equilibrium_counts": dict(sorted(counts.items())),
        "both_endpoints_observed": both,
        "all_iid_accuracy_at_least_0_75": high_iid,
    }


def _gap_gate(results: Sequence[Any]) -> dict[str, object]:
    import numpy as np

    joint = tuple(result for result in results if result.mode == "joint")
    truthful = np.asarray(
        [r.iid_accuracy - r.ood_accuracy for r in joint if r.equilibrium_mode == "truthful"]
    )
    compensatory = np.asarray(
        [r.iid_accuracy - r.ood_accuracy for r in joint if r.equilibrium_mode == "compensatory"]
    )
    if truthful.size == 0 or compensatory.size == 0:
        return {"passed": False, "reason": "both joint equilibrium groups are required"}
    generator = np.random.default_rng(BOOTSTRAP_SEED + 999)
    truth_samples = generator.choice(truthful, (10_000, truthful.size), replace=True).mean(axis=1)
    comp_samples = generator.choice(compensatory, (10_000, compensatory.size), replace=True).mean(
        axis=1
    )
    differences = comp_samples - truth_samples
    interval = np.quantile(differences, (0.025, 0.975))
    observed = float(compensatory.mean() - truthful.mean())
    return {
        "passed": observed > 0 and float(interval[0]) > 0,
        "truthful_mean_iid_ood_gap": float(truthful.mean()),
        "compensatory_mean_iid_ood_gap": float(compensatory.mean()),
        "compensatory_minus_truthful_gap": observed,
        "bootstrap_95_ci": [float(value) for value in interval],
        "bootstrap_seed": BOOTSTRAP_SEED + 999,
    }


def _training_mode_gate(results: Sequence[Any]) -> dict[str, object]:
    required = {"perception_only", "reasoning_only", "joint"}
    observed = {result.mode for result in results}
    perception_only = tuple(result for result in results if result.mode == "perception_only")
    reasoning_only = tuple(result for result in results if result.mode == "reasoning_only")
    joint = tuple(result for result in results if result.mode == "joint")
    reasoner_frozen = bool(perception_only) and all(
        len({checkpoint.canonical_reasoning_probability for checkpoint in result.history}) == 1
        for result in perception_only
    )
    perception_frozen = bool(reasoning_only) and all(
        len({checkpoint.truthful_perception_probability for checkpoint in result.history}) == 1
        for result in reasoning_only
    )
    joint_updated = bool(joint) and all(
        result.history[-1].truthful_perception_probability
        != result.history[0].truthful_perception_probability
        and result.history[-1].canonical_reasoning_probability
        != result.history[0].canonical_reasoning_probability
        for result in joint
    )
    return {
        "passed": (
            observed >= required and reasoner_frozen and perception_frozen and joint_updated
        ),
        "required": sorted(required),
        "observed": sorted(observed),
        "reasoner_frozen_in_perception_only": reasoner_frozen,
        "perception_frozen_in_reasoning_only": perception_frozen,
        "both_blocks_updated_in_joint": joint_updated,
    }


def _gates(results: Sequence[Any]) -> dict[str, object]:
    seed_count = len({result.seed for result in results})
    direction = _direction_gate(results)
    joint = _joint_gate(results)
    gap = _gap_gate(results)
    training_modes = _training_mode_gate(results)
    seed_gate = {"passed": seed_count >= 10, "unique_seed_count": seed_count, "required": 10}
    return {
        "at_least_10_seeds": seed_gate,
        "two_profile_direction": direction,
        "three_training_modes": training_modes,
        "joint_dual_equilibria": joint,
        "compensatory_ood_gap_larger": gap,
        "all_passed": all(
            gate["passed"] for gate in (seed_gate, direction, training_modes, joint, gap)
        ),
    }


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _summary(settings: Mapping[str, Any], results: Sequence[Any]) -> dict[str, object]:
    artifacts = {
        "metrics": _relative(settings["metrics_output"]),
        "runs": _relative(settings["runs_output"]),
        "trajectories": _relative(settings["trajectories_output"]),
        "figure": _relative(settings["figure_output"]),
        "logs": _relative(settings["log_root"] / settings["experiment"]),
    }
    return {
        "schema_version": 1,
        "experiment": "phase_c_small_neural",
        "protocol": {
            "profiles": list(settings["profiles"]),
            "modes": list(settings["modes"]),
            "joint_profile": settings["joint_profile"],
            "conditions_executed": [list(condition) for condition in _conditions(settings)],
            "seeds": list(settings["seeds"]),
            "steps": settings["steps"],
            "learning_rate": settings["learning_rate"],
            "device": settings["device"],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "notes": list(_protocol_notes(settings)),
        },
        "run_count": len(results),
        "checkpoint_count": sum(len(result.history) for result in results),
        "groups": list(_group_summaries(results)),
        "gates": _gates(results),
        "artifacts": artifacts,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_text(path, payload)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=tuple(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue())


def _plot_group_key(result: Any) -> tuple[str, str, str]:
    equilibrium = result.equilibrium_mode if result.mode == "joint" else "all"
    return result.profile, result.mode, equilibrium


def _plot(results: Sequence[Any], path: Path) -> None:
    matplotlib_cache = Path(tempfile.gettempdir()) / "compbias-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = tuple(dict.fromkeys(_plot_group_key(result) for result in results))
    panels = (
        ("truthful_perception_probability", "Truthful perception probability"),
        ("canonical_reasoning_probability", "Canonical reasoning probability"),
        ("iid_accuracy", "IID accuracy"),
        ("ood_accuracy", "OOD permutation accuracy"),
        ("coupling", "Error coupling term"),
        ("outcome_loss", "Outcome loss"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for key_index, key in enumerate(keys):
        group = tuple(result for result in results if _plot_group_key(result) == key)
        label = "/".join(part for part in key if part != "all")
        steps = np.asarray([checkpoint.step for checkpoint in group[0].history])
        for axis, (field, title) in zip(axes.flat, panels, strict=True):
            values = np.asarray(
                [[getattr(checkpoint, field) for checkpoint in result.history] for result in group]
            )
            for trajectory in values:
                axis.plot(steps, trajectory, color=colors(key_index), alpha=0.13, linewidth=0.6)
            axis.plot(
                steps,
                values.mean(axis=0),
                color=colors(key_index),
                linewidth=1.8,
                label=label,
            )
            axis.set_title(title)
            axis.set_xlabel("Optimization step")
            axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.stem}.", suffix=".png", delete=False
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(
            temporary,
            dpi=180,
            metadata={"Software": "compbias", "Creation Time": "2026-08-14"},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _checkpoint_hash(result: Any) -> str:
    canonical = json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _report(result: Any) -> str:
    gap = result.iid_accuracy - result.ood_accuracy
    return (
        "# Phase C small-neural run\n\n"
        f"- Profile: `{result.profile}`\n"
        f"- Mode: `{result.mode}`\n"
        f"- Seed: `{result.seed}`\n"
        f"- Equilibrium: `{result.equilibrium_mode}`\n"
        f"- Perception loss shift: `{result.perception_shift:.12f}`\n"
        f"- IID accuracy: `{result.iid_accuracy:.12f}`\n"
        f"- OOD accuracy: `{result.ood_accuracy:.12f}`\n"
        f"- IID-OOD gap: `{gap:.12f}`\n"
    )


def _capture_log_environments(
    settings: Mapping[str, Any],
    results: Sequence[Any],
    command: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    _ensure_local_package()
    from compbias.io.logging import capture_environment

    return tuple(
        capture_environment(
            worktree=REPOSITORY_ROOT,
            dataset_manifest_hash=None,
            seed=result.seed,
            model_revision=None,
            verl_revision=None,
            command=command,
        )
        for result in results
    )


def _write_logs(
    settings: Mapping[str, Any],
    results: Sequence[Any],
    environments: Sequence[Mapping[str, object]],
) -> None:
    _ensure_local_package()
    from compbias.io.logging import RunLogger

    if len(environments) != len(results):
        raise ValueError("one captured environment is required per neural run")
    for result, environment in zip(results, environments, strict=True):
        run_config = _expected_log_config(settings, result)
        execution_run_id = _execution_run_id(result, environment)
        with RunLogger(
            root=settings["log_root"],
            experiment=settings["experiment"],
            run_id=execution_run_id,
            config=run_config,
            environment=environment,
        ) as logger:
            for checkpoint in result.history:
                logger.log_metrics(
                    {
                        "profile": result.profile,
                        "mode": result.mode,
                        "seed": result.seed,
                        **asdict(checkpoint),
                    }
                )
            logger.log_rollout(
                {
                    "split": "iid",
                    "accuracy": result.iid_accuracy,
                    "seed": result.seed,
                    "sample_id": result.paired_sample_id,
                    "error_mechanism": result.iid_error_mechanism,
                    "correctness_matrix": result.iid_correctness_matrix,
                }
            )
            logger.log_rollout(
                {
                    "split": "ood_error_permutation",
                    "accuracy": result.ood_accuracy,
                    "seed": result.seed,
                    "sample_id": result.paired_sample_id,
                    "error_mechanism": result.ood_error_mechanism,
                    "correctness_matrix": result.ood_correctness_matrix,
                }
            )
            logger.save_predictions(
                {
                    "step": [checkpoint.step for checkpoint in result.history],
                    "truthful_perception_probability": [
                        checkpoint.truthful_perception_probability for checkpoint in result.history
                    ],
                    "canonical_reasoning_probability": [
                        checkpoint.canonical_reasoning_probability for checkpoint in result.history
                    ],
                    "iid_accuracy": [checkpoint.iid_accuracy for checkpoint in result.history],
                    "ood_accuracy": [checkpoint.ood_accuracy for checkpoint in result.history],
                    "coupling": [checkpoint.coupling for checkpoint in result.history],
                }
            )
            logger.write_report(_report(result))
            logger.finalize(checkpoint_hash=_checkpoint_hash(result))


def _is_batch(
    args: argparse.Namespace,
    loaded: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> bool:
    plural_requested = any(
        value is not None for value in (args.profiles, args.modes, args.seeds)
    ) or any(key in loaded for key in ("profiles", "modes", "seeds", "outputs"))
    return plural_requested or len(_conditions(settings)) * len(settings["seeds"]) > 1


def _legacy_output(result: Any, output: Path | None) -> None:
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "phase_c_small_neural_single",
                **asdict(result),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if output is None:
        print(payload, end="")
    else:
        _atomic_write_text(output, payload)


def _expected_log_config(settings: Mapping[str, Any], result: Any) -> dict[str, object]:
    return {
        "profile": result.profile,
        "mode": result.mode,
        "seed": result.seed,
        "steps": settings["steps"],
        "learning_rate": settings["learning_rate"],
        "device": settings["device"],
    }


def _matching_existing_log(run_dir: Path, settings: Mapping[str, Any], result: Any) -> bool:
    import numpy as np
    import yaml

    required = {
        "checkpoints",
        "config.yaml",
        "environment.json",
        "metrics.jsonl",
        "rollouts.jsonl",
        "predictions.npz",
        "report.md",
    }
    try:
        if {path.name for path in run_dir.iterdir()} != required:
            return False
        if not (run_dir / "checkpoints").is_dir():
            return False
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        metrics = tuple(
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        )
        rollouts = tuple(
            json.loads(line)
            for line in (run_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        )
        expected_metrics = tuple(
            {
                "profile": result.profile,
                "mode": result.mode,
                "seed": result.seed,
                **asdict(checkpoint),
            }
            for checkpoint in result.history
        )
        expected_rollouts = (
            {
                "split": "iid",
                "accuracy": result.iid_accuracy,
                "seed": result.seed,
                "sample_id": result.paired_sample_id,
                "error_mechanism": result.iid_error_mechanism,
                "correctness_matrix": [list(row) for row in result.iid_correctness_matrix],
            },
            {
                "split": "ood_error_permutation",
                "accuracy": result.ood_accuracy,
                "seed": result.seed,
                "sample_id": result.paired_sample_id,
                "error_mechanism": result.ood_error_mechanism,
                "correctness_matrix": [list(row) for row in result.ood_correctness_matrix],
            },
        )
        if config != _expected_log_config(settings, result):
            return False
        if environment.get("checkpoint_hash") != _checkpoint_hash(result):
            return False
        if metrics != expected_metrics or rollouts != expected_rollouts:
            return False
        if (run_dir / "report.md").read_text(encoding="utf-8") != _report(result):
            return False
        expected_predictions = {
            "step": np.asarray([checkpoint.step for checkpoint in result.history]),
            "truthful_perception_probability": np.asarray(
                [checkpoint.truthful_perception_probability for checkpoint in result.history]
            ),
            "canonical_reasoning_probability": np.asarray(
                [checkpoint.canonical_reasoning_probability for checkpoint in result.history]
            ),
            "iid_accuracy": np.asarray([checkpoint.iid_accuracy for checkpoint in result.history]),
            "ood_accuracy": np.asarray([checkpoint.ood_accuracy for checkpoint in result.history]),
            "coupling": np.asarray([checkpoint.coupling for checkpoint in result.history]),
        }
        with np.load(run_dir / "predictions.npz", allow_pickle=False) as predictions:
            if set(predictions.files) != set(expected_predictions):
                return False
            return all(
                np.array_equal(predictions[name], expected)
                for name, expected in expected_predictions.items()
            )
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
        yaml.YAMLError,
        zipfile.BadZipFile,
    ):
        return False


def _preflight_log_directories(settings: Mapping[str, Any], results: Sequence[Any]) -> None:
    experiment_dir = settings["log_root"] / settings["experiment"]
    mismatches = tuple(
        run_dir
        for result in results
        if (run_dir := experiment_dir / _run_id(result)).exists()
        and not _matching_existing_log(run_dir, settings, result)
    )
    if mismatches:
        rendered = ", ".join(str(path) for path in mismatches[:3])
        suffix = " ..." if len(mismatches) > 3 else ""
        raise FileExistsError(
            f"refusing to overwrite non-matching or incomplete run directories: {rendered}{suffix}"
        )


def _ownership_config_hash(
    config_path: Path | None,
    settings: Mapping[str, Any],
) -> str:
    def portable_path(path: Path) -> str:
        try:
            return path.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError:
            return os.fspath(path)

    source_hash = (
        hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path is not None else None
    )
    payload = {
        "source_config_sha256": source_hash,
        "profiles": list(settings["profiles"]),
        "modes": list(settings["modes"]),
        "seeds": list(settings["seeds"]),
        "steps": settings["steps"],
        "learning_rate": settings["learning_rate"],
        "device": settings["device"],
        "joint_profile": settings["joint_profile"],
        "experiment": settings["experiment"],
        "outputs": {
            "metrics": portable_path(settings["metrics_output"]),
            "runs": portable_path(settings["runs_output"]),
            "trajectories": portable_path(settings["trajectories_output"]),
            "figure": portable_path(settings["figure_output"]),
            "logs": portable_path(settings["log_root"]),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        loaded = _load_yaml(args.config)
        settings = _resolve_settings(args, loaded)
        _validate_choices(settings)
        batch = _is_batch(args, loaded, settings)
        artifact_outputs = (
            (
                settings["metrics_output"],
                settings["runs_output"],
                settings["trajectories_output"],
                settings["figure_output"],
            )
            if batch
            else ((settings["metrics_output"],) if args.output is not None else ())
        )
        ownership = None
        if artifact_outputs:
            _ensure_local_package()
            from compbias.io.artifact_paths import prepare_artifact_ownership

            names = ("metrics", "runs", "trajectories", "figure") if batch else ("metrics",)
            ownership = prepare_artifact_ownership(
                dict(zip(names, artifact_outputs, strict=True)),
                repository_root=REPOSITORY_ROOT,
                tool="scripts/train_neural.py",
                experiment=settings["experiment"],
                config_sha256=_ownership_config_hash(args.config, settings),
                primary_json="metrics",
                primary_schema_version=1,
                primary_experiment=(
                    "phase_c_small_neural" if batch else "phase_c_small_neural_single"
                ),
                overwrite=args.overwrite,
            )
        results = _run_all(settings)
        if not batch:
            if ownership is not None:
                from compbias.io.artifact_paths import artifact_ownership_transaction

                with artifact_ownership_transaction(ownership) as staged:
                    _legacy_output(results[0], staged["metrics"])
            else:
                _legacy_output(results[0], None)
            return 0
        arguments = sys.argv[1:] if argv is None else argv
        command = (sys.executable, str(Path(__file__).resolve()), *arguments)
        environments = _capture_log_environments(settings, results, command)
        run_rows = _run_rows(results)
        trajectory_rows = _trajectory_rows(results)
        summary = _summary(settings, results)
        if ownership is None:
            raise RuntimeError("batch artifact ownership was not prepared")
        from compbias.io.artifact_paths import artifact_ownership_transaction

        with artifact_ownership_transaction(
            ownership,
            after_promote=lambda: _write_logs(settings, results, environments),
        ) as staged:
            _atomic_write_csv(staged["runs"], run_rows)
            _atomic_write_csv(staged["trajectories"], trajectory_rows)
            _plot(results, staged["figure"])
            _atomic_write_json(staged["metrics"], summary)
    except (FileExistsError, ModuleNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))
    return 0 if summary["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
