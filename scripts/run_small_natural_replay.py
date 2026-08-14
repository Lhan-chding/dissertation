#!/usr/bin/env python3
"""Run the v2 PIL-CNN natural-mediator replay experiment on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TOP_LEVEL = {
    "schema_version",
    "experiment",
    "data",
    "replay",
    "training",
    "synthetic",
    "confirmatory",
    "outputs",
}
_OUTPUT_NAMES = ("summary", "compensabilities", "crossed_risks", "selection", "manifest")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="strict v2 YAML configuration")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a complete authenticated output bundle for the same config",
    )
    return parser


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _exact(mapping: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{name} has invalid fields ({'; '.join(details)})")


def _local_imports() -> tuple[Any, ...]:
    source = str(REPOSITORY_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from compbias.envs.cva_world.schema import SemanticSplit, TaskFamily
    from compbias.io.artifact_paths import (
        artifact_ownership_transaction,
        ensure_distinct_nonoverlapping,
        prepare_artifact_ownership,
        validated_artifact_path,
    )
    from compbias.io.logging import capture_environment, publishable_path
    from compbias.io.yaml_config import load_yaml_mapping
    from compbias.rl.small_natural_replay import SmallNaturalReplayConfig, run_small_natural_replay

    return (
        SemanticSplit,
        TaskFamily,
        artifact_ownership_transaction,
        ensure_distinct_nonoverlapping,
        prepare_artifact_ownership,
        validated_artifact_path,
        capture_environment,
        publishable_path,
        load_yaml_mapping,
        SmallNaturalReplayConfig,
        run_small_natural_replay,
    )


def _load_config(path: Path, loader: Any) -> dict[str, Any]:
    config = loader(path, label="small natural replay configuration")
    _exact(config, _TOP_LEVEL, "configuration")
    if config["schema_version"] != 2 or type(config["schema_version"]) is not int:
        raise ValueError("schema_version must be exactly 2")
    if config["experiment"] != "small_neural_natural_replay_v2":
        raise ValueError("experiment must be small_neural_natural_replay_v2")
    _exact(
        _mapping(config["data"], "data"),
        {
            "samples_per_family_per_split",
            "realizations_per_semantic",
            "data_seed",
            "splits",
            "task_families",
        },
        "data",
    )
    _exact(
        _mapping(config["replay"], "replay"),
        {"n_mediators", "n_forks", "bootstrap_draws"},
        "replay",
    )
    _exact(
        _mapping(config["training"], "training"),
        {
            "seeds",
            "steps",
            "batch_size",
            "image_size",
            "hidden_dim",
            "learning_rate",
            "device",
        },
        "training",
    )
    _exact(
        _mapping(config["synthetic"], "synthetic"),
        {"error_mass", "role"},
        "synthetic",
    )
    _exact(_mapping(config["outputs"], "outputs"), set(_OUTPUT_NAMES), "outputs")
    if config["training"]["device"] != "cpu":
        raise ValueError("training.device must be cpu")
    if config["synthetic"]["role"] != "off_support_stress_test":
        raise ValueError("synthetic.role must be off_support_stress_test")
    if not isinstance(config["confirmatory"], bool):
        raise TypeError("confirmatory must be boolean")
    return config


def _experiment_config(config: dict[str, Any], cls: Any, split_cls: Any, family_cls: Any) -> Any:
    data = config["data"]
    replay = config["replay"]
    training = config["training"]
    synthetic = config["synthetic"]
    if not isinstance(data["splits"], list) or not isinstance(data["task_families"], list):
        raise ValueError("data.splits and data.task_families must be lists")
    if not isinstance(training["seeds"], list):
        raise ValueError("training.seeds must be a list")
    return cls(
        samples_per_family_per_split=data["samples_per_family_per_split"],
        realizations_per_semantic=data["realizations_per_semantic"],
        n_mediators=replay["n_mediators"],
        n_forks=replay["n_forks"],
        training_seeds=tuple(training["seeds"]),
        training_steps=training["steps"],
        batch_size=training["batch_size"],
        image_size=training["image_size"],
        hidden_dim=training["hidden_dim"],
        learning_rate=training["learning_rate"],
        bootstrap_draws=replay["bootstrap_draws"],
        data_seed=data["data_seed"],
        synthetic_error_mass=synthetic["error_mass"],
        confirmatory=config["confirmatory"],
        splits=tuple(split_cls(value) for value in data["splits"]),
        task_families=tuple(family_cls(value) for value in data["task_families"]),
    )


def _paths(config: dict[str, Any], validator: Any, distinct: Any) -> dict[str, Path]:
    suffixes = {
        "summary": ".json",
        "compensabilities": ".csv",
        "crossed_risks": ".csv",
        "selection": ".csv",
        "manifest": ".json",
    }
    paths = {
        name: validator(
            config["outputs"][name],
            repository_root=REPOSITORY_ROOT,
            label=f"outputs.{name}",
            suffix=suffixes[name],
        )
        for name in _OUTPUT_NAMES
    }
    distinct(paths)
    return paths


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("CSV output rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_estimates(
    rows: list[dict[str, object]],
    *,
    bootstrap_draws: int,
    seed: int,
) -> list[dict[str, object]]:
    metrics = ("c_sel", "c_fork", "c_syn", "mediator_gap", "transport_gap")
    families = sorted({str(row["error_family"]) for row in rows})
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, object]] = []
    for family in families:
        family_rows = [row for row in rows if row["error_family"] == family]
        summary: dict[str, object] = {"error_family": family, "n_seeds": len(family_rows)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in family_rows], dtype=np.float64)
            if len(values) == 1:
                low = high = mean = float(values[0])
            else:
                indices = rng.integers(0, len(values), size=(bootstrap_draws, len(values)))
                draws = values[indices].mean(axis=1)
                low, high = np.quantile(draws, (0.025, 0.975), method="linear")
                mean = float(values.mean())
            summary[metric] = {
                "mean": mean,
                "ci_low": float(low),
                "ci_high": float(high),
            }
        summaries.append(summary)
    return summaries


def _payloads(
    result: Any,
    *,
    config_sha256: str,
    environment: dict[str, object],
) -> tuple[
    dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    compensabilities: list[dict[str, object]] = []
    crossed: list[dict[str, object]] = []
    selection: list[dict[str, object]] = []
    for seed_result in result.seed_results:
        for row in seed_result.by_error_family:
            compensabilities.append({"seed": seed_result.seed, **asdict(row)})
        crossed.append({"seed": seed_result.seed, **asdict(seed_result.crossed_risk)})
        selection.append(
            {
                "seed": seed_result.seed,
                "initial_error_probability": seed_result.initial_error_probability,
                "final_error_probability": seed_result.final_error_probability,
                "selection_error_ratio": seed_result.selection_error_ratio,
                "iid_accuracy": seed_result.iid_accuracy,
                "ood_accuracy": seed_result.ood_accuracy,
                "model_sha256": seed_result.model_sha256,
            }
        )
    summary = {
        "schema_version": 2,
        "experiment": "small_neural_natural_replay_v2",
        "status": "CONFIRMATORY_COMPLETE" if result.config.confirmatory else "PILOT_COMPLETE",
        "claim_scope": "small_neural_operational_mechanism_only",
        "config_sha256": config_sha256,
        "provenance": environment,
        "input_source": result.input_source,
        "model_path": result.model_path,
        "counts": {
            "semantic_states": result.semantic_state_count,
            "visual_realizations": result.visual_realization_count,
            "natural_mediators": result.natural_mediator_count,
            "forked_continuations": result.forked_continuation_count,
            "synthetic_mediators": result.synthetic_mediator_count,
            "training_seeds": len(result.seed_results),
        },
        "error_families": list(result.error_families),
        "family_estimates": _family_estimates(
            compensabilities,
            bootstrap_draws=result.config.bootstrap_draws,
            seed=result.config.data_seed,
        ),
        "all_crossed_risk_identities_pass": all(
            seed.crossed_risk.identity_residual < 1e-10 for seed in result.seed_results
        ),
        "synthetic_is_natural_evidence": False,
    }
    return summary, compensabilities, crossed, selection


def _run(config_path: Path, *, overwrite: bool) -> dict[str, object]:
    (
        split_cls,
        family_cls,
        transaction,
        distinct,
        prepare,
        validator,
        capture_environment,
        publishable_path,
        loader,
        config_cls,
        run_experiment,
    ) = _local_imports()
    config = _load_config(config_path, loader)
    experiment_config = _experiment_config(config, config_cls, split_cls, family_cls)
    paths = _paths(config, validator, distinct)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    environment = capture_environment(
        worktree=REPOSITORY_ROOT,
        dataset_manifest_hash=None,
        seed=experiment_config.data_seed,
        model_revision="local-small-cnn-v2",
        verl_revision=None,
        command=(sys.executable, str(Path(__file__)), "--config", str(config_path)),
    )
    ownership = prepare(
        paths,
        repository_root=REPOSITORY_ROOT,
        tool="run_small_natural_replay",
        experiment=config["experiment"],
        config_sha256=config_sha256,
        primary_json="summary",
        primary_schema_version=2,
        primary_experiment=config["experiment"],
        overwrite=overwrite,
    )
    result = run_experiment(experiment_config)
    summary, compensabilities, crossed, selection = _payloads(
        result,
        config_sha256=config_sha256,
        environment=environment,
    )
    with transaction(ownership) as staged:
        _write_json(staged["summary"], summary)
        _write_csv(staged["compensabilities"], compensabilities)
        _write_csv(staged["crossed_risks"], crossed)
        _write_csv(staged["selection"], selection)
        files = [
            {
                "name": name,
                "path": publishable_path(paths[name], worktree=REPOSITORY_ROOT),
                "sha256": _sha256(staged[name]),
            }
            for name in ("summary", "compensabilities", "crossed_risks", "selection")
        ]
        manifest = {
            "schema_version": 2,
            "experiment": config["experiment"],
            "config_sha256": config_sha256,
            "natural_mediators": {
                "count": result.natural_mediator_count,
                "materialization": "cluster_aggregated",
                "independence_unit": "sample_id",
            },
            "forked_continuations": {
                "count": result.forked_continuation_count,
                "materialization": "cluster_aggregated",
                "independence_unit": "sample_id",
            },
            "synthetic_mediators": {
                "count": result.synthetic_mediator_count,
                "evidence_role": "off_support_stress_test",
            },
            "files": files,
        }
        _write_json(staged["manifest"], manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = _run(arguments.config, overwrite=arguments.overwrite)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
