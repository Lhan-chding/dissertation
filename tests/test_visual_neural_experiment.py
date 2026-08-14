"""Acceptance tests for the image-backed Phase-C neural experiment."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import yaml

from compbias.rl.visual_neural_experiment import (
    VisualNeuralConfig,
    VisualNeuralSweepConfig,
    run_visual_neural_experiment,
    run_visual_neural_sweep,
    write_visual_neural_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_config(*, profile: str, mode: str, seed: int = 0) -> VisualNeuralConfig:
    return VisualNeuralConfig(
        profile=profile,
        mode=mode,
        seed=seed,
        steps=32,
        image_size=16,
        hidden_dim=4,
        perception_learning_rate=0.35,
        reasoning_learning_rate=0.35,
        device="cpu",
    )


@pytest.fixture(scope="module")
def sweep():
    pytest.importorskip("torch")
    return run_visual_neural_sweep(
        VisualNeuralSweepConfig(
            seeds=tuple(range(10)),
            steps=32,
            image_size=16,
            hidden_dim=4,
            perception_learning_rate=0.35,
            reasoning_learning_rate=0.35,
            device="cpu",
        )
    )


def test_config_is_frozen_and_strictly_validated() -> None:
    config = _run_config(profile="truth_aligned", mode="perception_only")

    with pytest.raises((FrozenInstanceError, AttributeError)):
        config.seed = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="profile"):
        _run_config(profile="invented", mode="perception_only")
    with pytest.raises(ValueError, match="mode"):
        _run_config(profile="truth_aligned", mode="invented")
    with pytest.raises(ValueError, match=r"joint.*spurious"):
        _run_config(profile="truth_aligned", mode="joint")
    with pytest.raises(ValueError, match="device"):
        VisualNeuralConfig(profile="spurious", mode="joint", seed=0, device="cuda")
    with pytest.raises(ValueError, match="at least 10"):
        VisualNeuralSweepConfig(seeds=tuple(range(9)))


@pytest.mark.neural
def test_real_pil_pixels_flow_through_cnn_and_paired_ood_changes_only_mechanism() -> None:
    pytest.importorskip("torch")

    result = run_visual_neural_experiment(
        _run_config(profile="truth_aligned", mode="perception_only")
    )

    assert result.input_source == "cva_renderer_pil"
    assert result.image_tensor_shape == (1, 3, 16, 16)
    assert result.image_sha256 == result.ood_image_sha256
    assert result.iid_error_mechanism != result.ood_error_mechanism
    assert result.shifted_factors == ("error_mechanism",)
    assert result.conv_parameter_delta > 0.0
    assert result.reasoner_parameter_delta == pytest.approx(0.0, abs=1e-12)


@pytest.mark.neural
def test_fixed_reasoner_profiles_move_cnn_perception_in_opposite_directions() -> None:
    pytest.importorskip("torch")

    truthful = run_visual_neural_experiment(
        _run_config(profile="truth_aligned", mode="perception_only")
    )
    spurious = run_visual_neural_experiment(_run_config(profile="spurious", mode="perception_only"))

    assert truthful.perception_shift > 0.02
    assert spurious.perception_shift < -0.02
    assert truthful.history[0].truthful_state_success_probability > (
        truthful.history[0].erroneous_state_success_probability
    )
    assert spurious.history[0].truthful_state_success_probability < (
        spurious.history[0].erroneous_state_success_probability
    )


@pytest.mark.neural
def test_reasoning_only_uses_the_cnn_but_freezes_its_parameters() -> None:
    pytest.importorskip("torch")
    result = run_visual_neural_experiment(_run_config(profile="spurious", mode="reasoning_only"))

    assert result.conv_parameter_delta == pytest.approx(0.0, abs=1e-12)
    assert result.reasoner_parameter_delta > 0.0
    assert len({checkpoint.truthful_perception_probability for checkpoint in result.history}) == 1


@pytest.mark.neural
def test_visual_ood_metric_executes_paired_catalogs_and_model_interventions() -> None:
    pytest.importorskip("torch")
    result = run_visual_neural_experiment(_run_config(profile="spurious", mode="joint", seed=1))
    final = result.history[-1]

    assert result.paired_sample_id
    assert result.iid_correctness_matrix == ((True, False), (False, True))
    assert result.ood_correctness_matrix == ((True, False), (False, False))
    assert final.iid_accuracy == pytest.approx(
        final.truthful_perception_probability * final.truthful_state_success_probability
        + (1.0 - final.truthful_perception_probability) * final.erroneous_state_success_probability,
        abs=1e-7,
    )
    assert final.ood_accuracy == pytest.approx(
        final.truthful_perception_probability * final.truthful_state_success_probability,
        abs=1e-7,
    )


@pytest.mark.neural
def test_every_checkpoint_records_injected_reasoner_rates_and_exact_decomposition(
    sweep,
) -> None:
    for result in sweep.runs:
        assert tuple(checkpoint.step for checkpoint in result.history) == tuple(
            range(len(result.history))
        )
        for checkpoint in result.history:
            probabilities = (
                checkpoint.truthful_perception_probability,
                checkpoint.truthful_state_success_probability,
                checkpoint.erroneous_state_success_probability,
                checkpoint.iid_accuracy,
                checkpoint.ood_accuracy,
            )
            assert all(0.0 <= value <= 1.0 for value in probabilities)
            assert checkpoint.l_o == pytest.approx(
                checkpoint.l_p + checkpoint.l_r + 2.0 * checkpoint.coupling,
                abs=1e-7,
            )
            assert checkpoint.iid_accuracy == pytest.approx(1.0 - checkpoint.l_o, abs=1e-7)


@pytest.mark.neural
def test_joint_ten_seed_sweep_reaches_both_learned_equilibria_and_ood_separates_them(
    sweep,
) -> None:
    joint = tuple(result for result in sweep.runs if result.mode == "joint")

    assert len(joint) == 10
    assert {result.seed for result in joint} == set(range(10))
    assert {result.equilibrium_mode for result in joint} == {
        "truthful",
        "compensatory",
    }
    initial_reasoner_rates = {
        (
            result.history[0].truthful_state_success_probability,
            result.history[0].erroneous_state_success_probability,
        )
        for result in joint
    }
    assert len(initial_reasoner_rates) >= 2
    assert len({result.history for result in joint}) >= 2
    assert all(result.conv_parameter_delta > 0.0 for result in joint)
    truthful_gaps = tuple(
        result.iid_accuracy - result.ood_accuracy
        for result in joint
        if result.equilibrium_mode == "truthful"
    )
    compensatory_gaps = tuple(
        result.iid_accuracy - result.ood_accuracy
        for result in joint
        if result.equilibrium_mode == "compensatory"
    )
    assert np.mean(compensatory_gaps) > np.mean(truthful_gaps) + 0.20
    assert sweep.gates.fixed_profiles_opposite is True
    assert sweep.gates.two_joint_equilibria is True
    assert sweep.gates.convolution_updated is True
    assert sweep.gates.three_training_modes is True
    assert sweep.gates.perception_frozen_in_reasoning_only is True
    assert sweep.gates.reasoner_frozen_in_perception_only is True
    assert sweep.gates.compensatory_ood_gap_larger is True
    assert sweep.gates.ood_gap_bootstrap_resamples == 10_000
    assert sweep.gates.ood_gap_confidence_level == pytest.approx(0.95)
    assert sweep.gates.ood_gap_difference_ci_low > sweep.config.minimum_ood_gap_margin
    assert sweep.gates.ood_gap_difference_ci_high >= sweep.gates.ood_gap_difference_ci_low
    assert sweep.gates.passed is True


def test_visual_ood_gate_requires_significant_bootstrap_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    import compbias.rl.visual_neural_experiment as visual_experiment

    monkeypatch.setattr(
        visual_experiment,
        "_bootstrap_gap_difference_interval",
        lambda *_args, **_kwargs: (-0.01, 0.90),
    )

    sweep = run_visual_neural_sweep(VisualNeuralSweepConfig())

    assert sweep.gates.mean_compensatory_ood_gap > sweep.gates.mean_truthful_ood_gap + 0.20
    assert sweep.gates.ood_gap_difference_ci_low == pytest.approx(-0.01)
    assert sweep.gates.compensatory_ood_gap_larger is False
    assert sweep.gates.passed is False


@pytest.mark.neural
def test_fixed_seed_is_reproducible_without_polluting_process_rng_state() -> None:
    torch = pytest.importorskip("torch")
    config = _run_config(profile="spurious", mode="joint", seed=4)
    random.seed(9182)
    np.random.seed(9182)
    torch.manual_seed(9182)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    first = run_visual_neural_experiment(config)
    second = run_visual_neural_experiment(config)

    assert first is not second
    assert first == second
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)


@pytest.mark.neural
def test_sweep_artifacts_are_machine_readable_and_use_visual_neural_names(tmp_path, sweep) -> None:
    paths = write_visual_neural_artifacts(sweep, output_root=tmp_path)

    assert paths.metrics_json.name.startswith("visual_neural_")
    assert paths.metrics_json.suffix == ".json"
    assert paths.runs_csv.name.startswith("visual_neural_")
    assert paths.trajectories_csv.name.startswith("visual_neural_")
    assert paths.figure_png.name.startswith("visual_neural_")
    assert paths.figure_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    payload = json.loads(paths.metrics_json.read_text(encoding="utf-8"))
    assert payload["gates"]["passed"] is True
    assert payload["provenance"]["input_source"] == "cva_renderer_pil"
    assert len(payload["joint_runs"]) == 10
    assert "conv_parameter_delta" in paths.runs_csv.read_text(encoding="utf-8")
    trajectory_header = paths.trajectories_csv.read_text(encoding="utf-8").splitlines()[0]
    assert all(
        field in trajectory_header
        for field in (
            "truthful_perception_probability",
            "truthful_state_success_probability",
            "erroneous_state_success_probability",
            "l_p",
            "l_r",
            "coupling",
            "l_o",
            "iid_accuracy",
            "ood_accuracy",
        )
    )


def _cli_config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/neural/visual_modular.yaml").read_text(encoding="utf-8")
    )
    source["outputs"] = {
        "metrics": str(tmp_path / "custom_metrics.json"),
        "runs": str(tmp_path / "custom_runs.csv"),
        "trajectories": str(tmp_path / "custom_trajectories.csv"),
        "figure": str(tmp_path / "custom_figure.png"),
        "logs": str(tmp_path / "custom_logs"),
    }
    config_path = tmp_path / "visual_cli.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return config_path, source


@pytest.mark.neural
def test_visual_cli_strictly_consumes_yaml_outputs_and_writes_run_provenance(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    config_path, config = _cli_config(tmp_path)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/train_visual_neural.py"),
        "--config",
        str(config_path),
    ]

    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "/Users/" not in completed.stdout
    assert "metrics: <external>/custom_metrics.json" in completed.stdout
    outputs = config["outputs"]
    assert isinstance(outputs, dict)
    expected_outputs = tuple(
        Path(outputs[name]) for name in ("metrics", "runs", "trajectories", "figure")
    )
    assert all(path.is_file() for path in expected_outputs)
    payload = json.loads(expected_outputs[0].read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    published_command = [
        "python",
        "scripts/train_visual_neural.py",
        "--config",
        f"<external>/{config_path.name}",
    ]
    assert provenance["config_path"] == f"<external>/{config_path.name}"
    assert provenance["config_sha256"] == sha256(config_path.read_bytes()).hexdigest()
    assert provenance["command"] == published_command
    assert provenance["seeds"] == list(range(10))
    assert provenance["python_version"]
    assert provenance["package_versions"]["torch"]
    assert provenance["package_versions"]["Pillow"]
    assert provenance["git_commit"]
    assert isinstance(provenance["git_dirty"], bool)
    assert provenance["start_timestamp"].endswith("Z")
    assert provenance["end_timestamp"].endswith("Z")
    run_directories = tuple((Path(outputs["logs"]) / "visual_neural_phase_c").iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    assert {
        "config.yaml",
        "environment.json",
        "metrics.jsonl",
        "rollouts.jsonl",
        "predictions.npz",
        "report.md",
    } <= {path.name for path in run_directory.iterdir()}
    rollouts = [
        json.loads(line)
        for line in (run_directory / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rollouts) == 13
    assert {record["mode"] for record in rollouts} == {
        "perception_only",
        "reasoning_only",
        "joint",
    }
    environment = json.loads((run_directory / "environment.json").read_text())
    assert environment["command"] == published_command
    assert environment["seeds"] == list(range(10))
    assert environment["config_sha256"] == provenance["config_sha256"]
    assert "/Users/" not in expected_outputs[0].read_text(encoding="utf-8")


@pytest.mark.neural
@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda config: config["training"].update({"answer_lookup": "forbidden"}), "unknown"),
        (lambda config: config["training"].update({"seeds": "0..9"}), "training.seeds"),
        (lambda config: config.update({"experiment": "different_experiment"}), "experiment"),
    ],
)
def test_visual_cli_rejects_unknown_and_wrong_typed_yaml_fields(
    tmp_path: Path, mutation, expected_error: str
) -> None:
    pytest.importorskip("torch")
    config_path, config = _cli_config(tmp_path)
    mutation(config)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/train_visual_neural.py"),
            "--config",
            str(config_path),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not Path(config["outputs"]["metrics"]).exists()


def test_visual_cli_rejects_repository_source_as_output(tmp_path: Path) -> None:
    config_path, config = _cli_config(tmp_path)
    forbidden = REPOSITORY_ROOT / "src/compbias/forbidden-visual-output.json"
    config["outputs"]["metrics"] = str(forbidden)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/train_visual_neural.py"),
            "--config",
            str(config_path),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode != 0
    assert "artifacts" in completed.stderr
    assert not forbidden.exists()
