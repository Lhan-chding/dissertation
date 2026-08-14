"""Run-directory and provenance contracts required by the experiment plan."""

from __future__ import annotations

import json
import stat
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from compbias.io.artifact_paths import (
    ensure_distinct_nonoverlapping,
    validated_artifact_path,
)
from compbias.io.logging import RunLogger, capture_environment

REQUIRED_ENVIRONMENT_KEYS = {
    "git_commit",
    "git_dirty",
    "python_version",
    "package_versions",
    "cuda_available",
    "gpu_devices",
    "dataset_manifest_hash",
    "seed",
    "model_revision",
    "verl_revision",
    "command",
    "start_timestamp",
}


def _environment(worktree: Path) -> dict[str, object]:
    return capture_environment(
        worktree=worktree,
        dataset_manifest_hash="sha256:dataset-fixture",
        seed=7,
        model_revision="model-revision-fixture",
        verl_revision=None,
        command=("python", "scripts/train_neural.py", "--config", "fixture.yaml"),
    )


def test_capture_environment_is_complete_even_without_torch_or_a_git_commit(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)

    assert environment.keys() >= REQUIRED_ENVIRONMENT_KEYS
    assert environment["dataset_manifest_hash"] == "sha256:dataset-fixture"
    assert environment["seed"] == 7
    assert environment["command"] == [
        "python",
        "scripts/train_neural.py",
        "--config",
        "fixture.yaml",
    ]
    assert isinstance(environment["package_versions"], dict)
    assert isinstance(environment["gpu_devices"], list)
    assert isinstance(environment["cuda_available"], bool)
    datetime.fromisoformat(str(environment["start_timestamp"]).replace("Z", "+00:00"))


def test_capture_environment_redacts_machine_specific_absolute_paths(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "train.py"
    config = tmp_path / "configs" / "run.yaml"
    outside = tmp_path.parent / "private-output.json"

    environment = capture_environment(
        worktree=tmp_path,
        dataset_manifest_hash=None,
        seed=0,
        model_revision=None,
        verl_revision=None,
        command=(
            "/private/runtime/bin/python",
            str(script),
            "--config",
            str(config),
            "--output",
            str(outside),
        ),
    )

    assert environment["command"] == [
        "python",
        "scripts/train.py",
        "--config",
        "configs/run.yaml",
        "--output",
        "<external>/private-output.json",
    ]
    assert str(tmp_path) not in json.dumps(environment)


def test_run_logger_writes_complete_atomic_run_bundle(tmp_path: Path) -> None:
    config = {
        "experiment": "neural_smoke",
        "seed": 7,
        "training": {"mode": "joint", "steps": 1},
    }
    original_config = {
        "experiment": "neural_smoke",
        "seed": 7,
        "training": {"mode": "joint", "steps": 1},
    }
    logger = RunLogger(
        root=tmp_path,
        experiment="neural_smoke",
        run_id="run-000",
        config=config,
        environment=_environment(tmp_path),
    )

    with logger:
        logger.log_metrics({"step": 0, "reward": 1.0})
        logger.log_rollout(
            {
                "sample_id": "sample-0",
                "error_id": "truth",
                "seed": 7,
                "raw_output": "answer=3",
                "parsed": {"answer": 3},
            }
        )
        logger.save_predictions({"answer": np.asarray([3]), "reward": np.asarray([1.0])})
        logger.write_report("# Neural smoke\n\nComplete.\n")
        logger.finalize(checkpoint_hash="sha256:checkpoint-fixture")

    run_dir = tmp_path / "neural_smoke" / "run-000"
    assert config == original_config, "logging must not mutate caller-owned configuration"
    assert (run_dir / "checkpoints").is_dir()
    assert {
        "config.yaml",
        "environment.json",
        "metrics.jsonl",
        "rollouts.jsonl",
        "predictions.npz",
        "report.md",
    } <= {path.name for path in run_dir.iterdir()}

    assert yaml.safe_load((run_dir / "config.yaml").read_text()) == original_config
    environment = json.loads((run_dir / "environment.json").read_text())
    assert environment.keys() >= REQUIRED_ENVIRONMENT_KEYS
    assert environment["checkpoint_hash"] == "sha256:checkpoint-fixture"
    datetime.fromisoformat(environment["end_timestamp"].replace("Z", "+00:00"))

    metric = json.loads((run_dir / "metrics.jsonl").read_text().strip())
    rollout = json.loads((run_dir / "rollouts.jsonl").read_text().strip())
    assert metric == {"step": 0, "reward": 1.0}
    assert rollout["sample_id"] == "sample-0"
    assert stat.S_IMODE((run_dir / "metrics.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((run_dir / "rollouts.jsonl").stat().st_mode) == 0o600
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as predictions:
        np.testing.assert_array_equal(predictions["answer"], np.asarray([3]))


def test_existing_run_directory_is_never_silently_overwritten(tmp_path: Path) -> None:
    arguments = {
        "root": tmp_path,
        "experiment": "neural_smoke",
        "run_id": "duplicate",
        "config": {"seed": 7},
        "environment": _environment(tmp_path),
    }

    RunLogger(**arguments)

    try:
        RunLogger(**arguments)
    except FileExistsError:
        pass
    else:
        raise AssertionError("RunLogger must reject an existing run_id")


def test_successful_run_context_requires_every_canonical_bundle_member(
    tmp_path: Path,
) -> None:
    logger = RunLogger(
        root=tmp_path,
        experiment="incomplete",
        run_id="run-000",
        config={"seed": 7},
        environment=_environment(tmp_path),
    )

    with pytest.raises(RuntimeError, match="missing required run artifacts"), logger:
        logger.log_metrics({"passed": True})


def test_failed_run_context_may_finalize_an_incomplete_failure_record(
    tmp_path: Path,
) -> None:
    logger = RunLogger(
        root=tmp_path,
        experiment="failed",
        run_id="run-000",
        config={"seed": 7},
        environment=_environment(tmp_path),
    )

    with pytest.raises(ValueError, match="deliberate failure"), logger:
        raise ValueError("deliberate failure")

    environment = json.loads((logger.run_dir / "environment.json").read_text())
    assert environment["status"] == "failed"
    assert environment["error_type"] == "ValueError"


@pytest.mark.parametrize("method_name", ("log_metrics", "log_rollout"))
def test_run_logger_rejects_nonfinite_jsonl_before_creating_member(
    tmp_path: Path,
    method_name: str,
) -> None:
    logger = RunLogger(
        root=tmp_path,
        experiment="nonfinite",
        run_id=method_name,
        config={"seed": 7},
        environment=_environment(tmp_path),
    )

    method = getattr(logger, method_name)
    with pytest.raises(ValueError, match=r"JSON compliant|finite"):
        method({"invalid": float("nan")})

    filename = "metrics.jsonl" if method_name == "log_metrics" else "rollouts.jsonl"
    assert not (logger.run_dir / filename).exists()


def test_run_logger_rejects_nonfinite_environment_before_publishing_json(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["invalid"] = float("inf")

    with pytest.raises(ValueError, match=r"JSON compliant|finite"):
        RunLogger(
            root=tmp_path,
            experiment="nonfinite",
            run_id="environment",
            config={"seed": 7},
            environment=environment,
        )

    assert not (tmp_path / "nonfinite" / "environment" / "environment.json").exists()


def test_run_logger_rejects_symlinked_experiment_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    (log_root / "neural_smoke").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        RunLogger(
            root=log_root,
            experiment="neural_smoke",
            run_id="run-000",
            config={"seed": 7},
            environment=_environment(tmp_path),
        )

    assert tuple(outside.iterdir()) == ()


@pytest.mark.parametrize("unsafe_name", ("../escape", "/absolute", "nested/member"))
def test_prediction_archive_rejects_unsafe_member_names(tmp_path: Path, unsafe_name: str) -> None:
    logger = RunLogger(
        root=tmp_path,
        experiment="neural_smoke",
        run_id="unsafe-member",
        config={"seed": 7},
        environment=_environment(tmp_path),
    )

    with pytest.raises(ValueError, match="prediction name"), logger:
        logger.save_predictions({unsafe_name: np.asarray([1.0])})

    assert not (logger.run_dir / "predictions.npz").exists()


def test_artifact_paths_allow_only_repo_artifacts_or_temporary_workspace(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    allowed = validated_artifact_path(
        "artifacts/metrics/result.json",
        repository_root=repository,
        label="metrics",
        suffix=".json",
    )
    temporary = validated_artifact_path(
        tmp_path / "external.json",
        repository_root=repository,
        label="metrics",
        suffix=".json",
    )

    assert allowed == repository / "artifacts/metrics/result.json"
    assert temporary == tmp_path / "external.json"
    with pytest.raises(ValueError, match="artifacts"):
        validated_artifact_path(
            repository / "src/overwrite.py",
            repository_root=repository,
            label="metrics",
            suffix=".json",
        )
    with pytest.raises(ValueError, match="suffix"):
        validated_artifact_path(
            repository / "artifacts/metrics/result.py",
            repository_root=repository,
            label="metrics",
            suffix=".json",
        )


def test_artifact_paths_reject_symlinks_and_overlapping_targets(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    artifacts = repository / "artifacts"
    real = artifacts / "real"
    real.mkdir(parents=True)
    linked = artifacts / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        validated_artifact_path(
            linked / "result.json",
            repository_root=repository,
            label="metrics",
            suffix=".json",
        )
    with pytest.raises(ValueError, match="overlap"):
        ensure_distinct_nonoverlapping(
            {
                "logs": artifacts / "logs",
                "metrics": artifacts / "logs/metrics.json",
            }
        )
