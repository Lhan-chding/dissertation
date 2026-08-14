"""Successful end-to-end contracts for the deterministic CPU phase runners."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml

from scripts import train_tabular, verify_theory
from scripts.train_tabular import _run as run_tabular
from scripts.verify_theory import _run as run_theory

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_TIMESTAMP = "2026-08-14T00:00:00+00:00"


def _copy_config(source: Path, destination: Path) -> dict[str, object]:
    configuration = yaml.safe_load(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    return configuration


def _assert_complete_run_bundle(root: Path, experiment: str) -> Path:
    run_directories = tuple((root / "artifacts/logs" / experiment).iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    assert run_directory.is_dir()
    expected = {
        "config.yaml",
        "environment.json",
        "metrics.jsonl",
        "rollouts.jsonl",
        "predictions.npz",
        "checkpoints",
        "report.md",
    }
    assert {path.name for path in run_directory.iterdir()} == expected
    assert (run_directory / "checkpoints").is_dir()
    assert (run_directory / "metrics.jsonl").read_text(encoding="utf-8").strip()
    assert (run_directory / "rollouts.jsonl").read_text(encoding="utf-8").strip()
    assert (run_directory / "report.md").read_text(encoding="utf-8").startswith("# ")
    with np.load(run_directory / "predictions.npz", allow_pickle=False) as predictions:
        assert predictions.files
    environment = json.loads((run_directory / "environment.json").read_text(encoding="utf-8"))
    assert environment["seed"] == 20260814
    assert environment["dataset_manifest_hash"] is None
    assert environment["model_revision"] is None
    assert environment["verl_revision"] is None
    assert environment["checkpoint_hash"] is None
    assert environment["command"]
    assert environment["start_timestamp"].endswith("Z")
    assert environment["end_timestamp"].endswith("Z")
    return run_directory


def test_phase_a_runner_writes_a_complete_owned_verification_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    config_path = tmp_path / "configs/theory/all.yaml"
    _copy_config(REPOSITORY_ROOT / "configs/theory/all.yaml", config_path)

    assert run_theory(config_path, tmp_path, FROZEN_TIMESTAMP) is True

    evidence = json.loads(
        (tmp_path / "artifacts/theory_verification/random_property_tests.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["passed"] is True
    assert evidence["property_tests"]["identity_checks"] == 7_000
    assert evidence["coordination"]["passed"] is True
    assert evidence["bifurcation"]["passed"] is True
    assert (tmp_path / "artifacts/theory_verification/report.md").is_file()
    assert (tmp_path / "artifacts/figures/bifurcation.png").is_file()
    assert (tmp_path / "artifacts/figures/basin_map.png").is_file()
    _assert_complete_run_bundle(tmp_path, "phase_a_theory_verification")


def test_phase_b_coordination_runner_writes_metrics_predictions_and_figure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    source = REPOSITORY_ROOT / "configs/tabular/all.yaml"
    config_path = tmp_path / "configs/tabular/coordination.yaml"
    configuration = _copy_config(source, config_path)
    configuration["experiment"] = "phase_b_tabular_coordination_integration"
    configuration["task"] = "coordination"
    config_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")

    assert run_tabular(config_path, tmp_path, FROZEN_TIMESTAMP) is True

    coordination = json.loads(
        (tmp_path / "artifacts/metrics/tabular_coordination.json").read_text(encoding="utf-8")
    )
    bifurcation = json.loads(
        (tmp_path / "artifacts/metrics/tabular_bifurcation.json").read_text(encoding="utf-8")
    )
    assert coordination["passed"] is True
    assert coordination["grid_points"] == 19 * 19
    assert coordination["seeded_initializations"] == 20
    assert bifurcation["passed"] is True
    assert (tmp_path / "artifacts/predictions/tabular_coordination.csv").is_file()
    assert (tmp_path / "artifacts/predictions/tabular_bifurcation.csv").is_file()
    assert (tmp_path / "artifacts/figures/tabular_coordination.png").is_file()
    _assert_complete_run_bundle(tmp_path, "phase_b_tabular_coordination_integration")


@pytest.mark.parametrize("phase", ("a", "b"), ids=("phase-a", "phase-b"))
def test_cpu_runner_captures_provenance_before_opening_artifact_transaction(
    phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import compbias.io.logging as run_logging

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    if phase == "a":
        config_path = tmp_path / "configs/theory/all.yaml"
        _copy_config(REPOSITORY_ROOT / "configs/theory/all.yaml", config_path)

        def runner() -> bool:
            return run_theory(config_path, tmp_path, FROZEN_TIMESTAMP)

    else:
        config_path = tmp_path / "configs/tabular/coordination.yaml"
        configuration = _copy_config(REPOSITORY_ROOT / "configs/tabular/all.yaml", config_path)
        configuration["experiment"] = "phase_b_provenance_order"
        configuration["task"] = "coordination"
        config_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
        monkeypatch.setattr(
            train_tabular,
            "_git_metadata",
            lambda _root: {
                "commit": "fixture-commit",
                "dirty": bool(tuple(tmp_path.rglob(".*.transaction-*"))),
            },
        )

        def runner() -> bool:
            return run_tabular(config_path, tmp_path, FROZEN_TIMESTAMP)

    real_capture_environment = run_logging.capture_environment
    captures = 0

    def capture_before_transaction(**kwargs):
        nonlocal captures
        assert not tuple(tmp_path.rglob(".*.transaction-*"))
        captures += 1
        return real_capture_environment(**kwargs)

    monkeypatch.setattr(run_logging, "capture_environment", capture_before_transaction)

    assert runner() is True
    assert captures == 1
    if phase == "b":
        for name in ("tabular_coordination.json", "tabular_bifurcation.json"):
            metrics = json.loads((tmp_path / "artifacts/metrics" / name).read_text())
            assert metrics["run"]["git"] == {
                "commit": "fixture-commit",
                "dirty": False,
            }


@pytest.mark.parametrize("module", (verify_theory, train_tabular), ids=("phase-a", "phase-b"))
@pytest.mark.parametrize(("passed", "expected_status"), ((True, 0), (False, 1)))
def test_cpu_runner_main_forwards_validated_arguments_and_maps_gate_status(
    module: ModuleType,
    passed: bool,
    expected_status: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    received: dict[str, object] = {}

    def fake_run(config, repository_root, started_at, *, overwrite=False):
        received.update(
            config=config,
            repository_root=repository_root,
            started_at=started_at,
            overwrite=overwrite,
        )
        return passed

    monkeypatch.setattr(module, "_run", fake_run)

    assert module.main(["--config", str(config_path), "--overwrite"]) == expected_status
    assert received["config"] == config_path.resolve()
    assert received["repository_root"] == REPOSITORY_ROOT
    assert str(received["started_at"]).endswith("+00:00")
    assert received["overwrite"] is True


@pytest.mark.parametrize("module", (verify_theory, train_tabular), ids=("phase-a", "phase-b"))
def test_cpu_runner_main_turns_validation_failures_into_cli_errors(
    module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")

    def fail_run(*_args, **_kwargs):
        raise ValueError("invalid integration fixture")

    monkeypatch.setattr(module, "_run", fail_run)
    with pytest.raises(SystemExit) as error:
        module.main(["--config", str(config_path)])

    assert error.value.code == 2
    assert "invalid integration fixture" in capsys.readouterr().err
