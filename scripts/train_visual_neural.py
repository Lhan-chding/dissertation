#!/usr/bin/env python3
"""Run the config-driven PIL -> CNN -> MLP Phase-C experiment on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "input",
    "model",
    "training",
    "gates",
    "outputs",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="strict Phase-C YAML config")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing validated artifact outputs",
    )
    return parser


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - mapping.keys())
    unknown = sorted(mapping.keys() - expected)
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")


def _literal(value: object, expected: object, name: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{name} must be exactly {expected!r}")


def _string_list(value: object, expected: tuple[str, ...], name: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    if tuple(value) != expected:
        raise ValueError(f"{name} must be exactly {list(expected)!r}")


def _seed_list(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("training.seeds must be a list of integers")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
        raise ValueError("training.seeds must be a list of integers")
    return tuple(value)


def _load_config(path: Path) -> dict[str, Any]:
    source = str(REPOSITORY_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

    config = load_yaml_mapping(path)
    _exact_keys(config, _TOP_LEVEL_KEYS, "configuration")
    nested = {
        "input": {
            "source",
            "image_size",
            "scene_seed",
            "iid_error_mechanism",
            "ood_error_mechanism",
        },
        "model": {"class", "hidden_dim", "perceived_states", "reasoning_actions"},
        "training": {
            "objective",
            "profiles",
            "modes",
            "seeds",
            "steps",
            "perception_learning_rate",
            "reasoning_learning_rate",
            "device",
        },
        "gates": {
            "minimum_profile_shift",
            "minimum_joint_iid_accuracy",
            "minimum_ood_gap_margin",
            "ood_gap_bootstrap_resamples",
            "ood_gap_confidence_level",
            "ood_gap_bootstrap_seed",
        },
        "outputs": {"metrics", "runs", "trajectories", "figure", "logs"},
    }
    for section, fields in nested.items():
        reject_unknown_fields(config[section], fields, label=section)
    _literal(config["schema_version"], 1, "schema_version")
    _literal(config["experiment"], "visual_neural_phase_c", "experiment")
    return config


def _local_imports() -> tuple[Any, Any, Any, Any, Any]:
    source = str(REPOSITORY_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from compbias.io.logging import RunLogger, capture_environment
    from compbias.rl.visual_neural_experiment import (
        VisualNeuralArtifactPaths,
        VisualNeuralSweepConfig,
        run_visual_neural_sweep,
        write_visual_neural_artifacts,
    )

    return (
        RunLogger,
        capture_environment,
        VisualNeuralArtifactPaths,
        VisualNeuralSweepConfig,
        (run_visual_neural_sweep, write_visual_neural_artifacts),
    )


def _sweep_config(config: dict[str, Any], config_class: Any) -> Any:
    input_config = _mapping(config["input"], "input")
    _exact_keys(
        input_config,
        {
            "source",
            "image_size",
            "scene_seed",
            "iid_error_mechanism",
            "ood_error_mechanism",
        },
        "input",
    )
    _literal(input_config["source"], "cva_renderer_pil", "input.source")
    model = _mapping(config["model"], "model")
    _exact_keys(
        model,
        {"class", "hidden_dim", "perceived_states", "reasoning_actions"},
        "model",
    )
    _literal(model["class"], "ModularPerceiverReasoner", "model.class")
    _literal(model["perceived_states"], 2, "model.perceived_states")
    _literal(model["reasoning_actions"], 2, "model.reasoning_actions")
    training = _mapping(config["training"], "training")
    _exact_keys(
        training,
        {
            "objective",
            "profiles",
            "modes",
            "seeds",
            "steps",
            "perception_learning_rate",
            "reasoning_learning_rate",
            "device",
        },
        "training",
    )
    _literal(
        training["objective"],
        "outcome_only_expected_reward",
        "training.objective",
    )
    _string_list(training["profiles"], ("truth_aligned", "spurious"), "training.profiles")
    _string_list(
        training["modes"],
        ("perception_only", "reasoning_only", "joint"),
        "training.modes",
    )
    gates = _mapping(config["gates"], "gates")
    _exact_keys(
        gates,
        {
            "minimum_profile_shift",
            "minimum_joint_iid_accuracy",
            "minimum_ood_gap_margin",
            "ood_gap_bootstrap_resamples",
            "ood_gap_confidence_level",
            "ood_gap_bootstrap_seed",
        },
        "gates",
    )
    return config_class(
        seeds=_seed_list(training["seeds"]),
        steps=training["steps"],
        image_size=input_config["image_size"],
        hidden_dim=model["hidden_dim"],
        perception_learning_rate=training["perception_learning_rate"],
        reasoning_learning_rate=training["reasoning_learning_rate"],
        device=training["device"],
        scene_seed=input_config["scene_seed"],
        iid_error_mechanism=input_config["iid_error_mechanism"],
        ood_error_mechanism=input_config["ood_error_mechanism"],
        minimum_profile_shift=gates["minimum_profile_shift"],
        minimum_joint_iid_accuracy=gates["minimum_joint_iid_accuracy"],
        minimum_ood_gap_margin=gates["minimum_ood_gap_margin"],
        ood_gap_bootstrap_resamples=gates["ood_gap_bootstrap_resamples"],
        ood_gap_confidence_level=gates["ood_gap_confidence_level"],
        ood_gap_bootstrap_seed=gates["ood_gap_bootstrap_seed"],
    )


def _path(value: object, name: str, *, suffix: str | None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    from compbias.io.artifact_paths import validated_artifact_path

    return validated_artifact_path(
        value,
        repository_root=REPOSITORY_ROOT,
        label=name,
        suffix=suffix,
    )


def _output_paths(config: dict[str, Any], path_class: Any) -> tuple[Any, Path]:
    outputs = _mapping(config["outputs"], "outputs")
    _exact_keys(outputs, {"metrics", "runs", "trajectories", "figure", "logs"}, "outputs")
    paths = path_class(
        metrics_json=_path(outputs["metrics"], "outputs.metrics", suffix=".json"),
        runs_csv=_path(outputs["runs"], "outputs.runs", suffix=".csv"),
        trajectories_csv=_path(outputs["trajectories"], "outputs.trajectories", suffix=".csv"),
        figure_png=_path(outputs["figure"], "outputs.figure", suffix=".png"),
    )
    log_root = _path(outputs["logs"], "outputs.logs", suffix=None)
    from compbias.io.artifact_paths import ensure_distinct_nonoverlapping

    ensure_distinct_nonoverlapping(
        {**{name: Path(value) for name, value in asdict(paths).items()}, "logs": log_root}
    )
    return paths, log_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _published_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return f"<external>/{path.name}"


def _run_id(start_timestamp: str, config_hash: str) -> str:
    timestamp = re.sub(r"[^0-9TZ]", "", start_timestamp)
    return f"sweep-{timestamp}-{config_hash[:12]}"


def _report(sweep: Any, paths: Any) -> str:
    gates = sweep.gates
    return (
        "# Phase C visual-neural sweep\n\n"
        f"- Passed: `{gates.passed}`\n"
        f"- Joint equilibria: `{gates.truthful_joint_runs}` truthful, "
        f"`{gates.compensatory_joint_runs}` compensatory\n"
        f"- Minimum joint IID accuracy: `{gates.minimum_joint_iid_accuracy:.12f}`\n"
        f"- Metrics: `{_published_path(paths.metrics_json)}`\n"
    )


def _write_run_bundle(
    *,
    run_logger: Any,
    sweep: Any,
    paths: Any,
    log_root: Path,
    experiment: str,
    run_id: str,
    log_config: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    with run_logger(
        root=log_root,
        experiment=experiment,
        run_id=run_id,
        config=log_config,
        environment=environment,
    ) as logger:
        for result in sweep.runs:
            logger.log_metrics(
                {
                    "profile": result.profile,
                    "mode": result.mode,
                    "seed": result.seed,
                    "equilibrium_mode": result.equilibrium_mode,
                    "iid_accuracy": result.iid_accuracy,
                    "ood_accuracy": result.ood_accuracy,
                    "conv_parameter_delta": result.conv_parameter_delta,
                }
            )
            logger.log_rollout(
                {
                    "sample_id": result.paired_sample_id,
                    "profile": result.profile,
                    "mode": result.mode,
                    "seed": result.seed,
                    "iid_error_mechanism": result.iid_error_mechanism,
                    "ood_error_mechanism": result.ood_error_mechanism,
                    "iid_accuracy": result.iid_accuracy,
                    "ood_accuracy": result.ood_accuracy,
                    "equilibrium_mode": result.equilibrium_mode,
                }
            )
        logger.save_predictions(
            {
                "seed": [result.seed for result in sweep.runs],
                "iid_accuracy": [result.iid_accuracy for result in sweep.runs],
                "ood_accuracy": [result.ood_accuracy for result in sweep.runs],
                "final_p": [
                    result.history[-1].truthful_perception_probability for result in sweep.runs
                ],
            }
        )
        logger.write_report(_report(sweep, paths))
        artifact_hash = "sha256:" + hashlib.sha256(paths.metrics_json.read_bytes()).hexdigest()
        logger.finalize(checkpoint_hash=artifact_hash)


def _run(
    config_path: Path,
    raw_config: dict[str, Any],
    command: list[str],
    *,
    overwrite: bool = False,
) -> tuple[Any, Any]:
    RunLogger, capture_environment, path_class, config_class, functions = _local_imports()
    run_sweep, write_artifacts = functions
    sweep_config = _sweep_config(raw_config, config_class)
    paths, log_root = _output_paths(raw_config, path_class)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    from compbias.io.artifact_paths import (
        artifact_ownership_transaction,
        prepare_artifact_ownership,
    )

    ownership = prepare_artifact_ownership(
        {name: Path(value) for name, value in asdict(paths).items()},
        repository_root=REPOSITORY_ROOT,
        tool="scripts/train_visual_neural.py",
        experiment=raw_config["experiment"],
        config_sha256=config_hash,
        primary_json="metrics_json",
        primary_schema_version=1,
        primary_experiment="visual_neural_phase_c",
        overwrite=overwrite,
    )
    environment = capture_environment(
        worktree=REPOSITORY_ROOT,
        dataset_manifest_hash=None,
        seed=sweep_config.seeds[0],
        model_revision=None,
        verl_revision=None,
        command=command,
    )
    environment = {
        **environment,
        "seeds": list(sweep_config.seeds),
        "config_path": _published_path(config_path),
        "config_sha256": config_hash,
    }
    run_id = _run_id(str(environment["start_timestamp"]), config_hash)
    log_config = {
        **raw_config,
        "outputs": {
            name: _published_path(Path(value))
            for name, value in _mapping(raw_config["outputs"], "outputs").items()
        },
    }
    sweep = run_sweep(sweep_config)
    provenance = {
        **environment,
        "end_timestamp": _utc_now(),
        "run_directory": _published_path(log_root / raw_config["experiment"] / run_id),
    }
    with artifact_ownership_transaction(
        ownership,
        after_promote=lambda: _write_run_bundle(
            run_logger=RunLogger,
            sweep=sweep,
            paths=paths,
            log_root=log_root,
            experiment=raw_config["experiment"],
            run_id=run_id,
            log_config=log_config,
            environment=environment,
        ),
    ) as staged:
        staged_paths = path_class(
            metrics_json=staged["metrics_json"],
            runs_csv=staged["runs_csv"],
            trajectories_csv=staged["trajectories_csv"],
            figure_png=staged["figure_png"],
        )
        write_artifacts(
            sweep,
            paths=staged_paths,
            provenance=provenance,
            overwrite=False,
        )
    return sweep, paths


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        command = [
            "python",
            "scripts/train_visual_neural.py",
            "--config",
            _published_path(config_path),
        ]
        config = _load_config(config_path)
        if args.overwrite:
            command.append("--overwrite")
        sweep, paths = _run(
            config_path,
            config,
            command,
            overwrite=args.overwrite,
        )
    except (
        FileExistsError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(sweep.gates), indent=2, sort_keys=True))
    print(f"metrics: {_published_path(paths.metrics_json)}")
    return 0 if sweep.gates.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
