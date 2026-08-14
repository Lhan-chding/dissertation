"""Deterministic, CPU-sized acceptance tests for the small-neural gate."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest
import yaml

from compbias.rl.neural_experiment import NeuralExperimentConfig, run_neural_experiment
from scripts.train_neural import _load_yaml, _parser, _resolve_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _config(*, profile: str, mode: str, seed: int) -> NeuralExperimentConfig:
    return NeuralExperimentConfig(
        profile=profile,
        mode=mode,
        seed=seed,
        steps=48,
        device="cpu",
    )


def test_neural_experiment_config_is_an_immutable_validated_value() -> None:
    config = _config(profile="truth_aligned", mode="perception_only", seed=0)

    assert is_dataclass(config)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        config.seed = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="profile"):
        _config(profile="invented", mode="joint", seed=0)
    with pytest.raises(ValueError, match="mode"):
        _config(profile="truth_aligned", mode="invented", seed=0)
    with pytest.raises(ValueError, match=r"joint.*spurious"):
        _config(profile="truth_aligned", mode="joint", seed=0)
    with pytest.raises((TypeError, ValueError), match="learning_rate"):
        NeuralExperimentConfig(
            profile="spurious",
            mode="perception_only",
            seed=0,
            learning_rate=True,
        )
    with pytest.raises(ValueError, match="at most"):
        NeuralExperimentConfig(
            profile="spurious",
            mode="perception_only",
            seed=0,
            steps=513,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("steps", 513),
        ("seeds", list(range(129))),
    ),
)
def test_scalar_neural_runner_rejects_oversized_workload_controls(
    field: str, value: object
) -> None:
    config = _load_yaml(REPOSITORY_ROOT / "configs/neural/all.yaml")
    config[field] = value

    with pytest.raises(ValueError, match="at most"):
        _resolve_settings(_parser().parse_args([]), config)


@pytest.mark.neural
@pytest.mark.parametrize(
    ("profile", "expected_sign"),
    [("truth_aligned", -1), ("spurious", 1)],
)
def test_fixed_reasoner_profiles_move_perception_in_opposite_predicted_directions(
    profile: str,
    expected_sign: int,
) -> None:
    pytest.importorskip("torch")

    result = run_neural_experiment(_config(profile=profile, mode="perception_only", seed=0))

    assert result.profile == profile
    assert result.mode == "perception_only"
    assert result.seed == 0
    assert expected_sign * result.perception_shift > 0.0


@pytest.mark.neural
def test_same_seed_is_reproducible_but_each_run_returns_a_new_result() -> None:
    pytest.importorskip("torch")
    config = _config(profile="spurious", mode="joint", seed=2)

    first = run_neural_experiment(config)
    second = run_neural_experiment(config)

    assert first is not second
    assert first == second
    assert isinstance(first.history, tuple)
    assert first.history


@pytest.mark.neural
def test_reasoning_only_freezes_perception_while_updating_reasoning() -> None:
    pytest.importorskip("torch")
    result = run_neural_experiment(_config(profile="spurious", mode="reasoning_only", seed=0))

    perception = tuple(checkpoint.truthful_perception_probability for checkpoint in result.history)
    reasoning = tuple(checkpoint.canonical_reasoning_probability for checkpoint in result.history)

    assert len(set(perception)) == 1
    assert reasoning[-1] != pytest.approx(reasoning[0])


@pytest.mark.neural
def test_scalar_ood_accuracy_is_executed_against_a_paired_error_catalog() -> None:
    pytest.importorskip("torch")
    result = run_neural_experiment(_config(profile="spurious", mode="joint", seed=1))

    assert result.shifted_factors == ("error_mechanism",)
    assert result.iid_correctness_matrix == ((True, False), (False, True))
    assert result.ood_correctness_matrix == ((True, False), (False, False))
    for checkpoint in result.history:
        state_probabilities = (
            checkpoint.truthful_perception_probability,
            1.0 - checkpoint.truthful_perception_probability,
        )
        action_probabilities = (
            checkpoint.canonical_reasoning_probability,
            1.0 - checkpoint.canonical_reasoning_probability,
        )

        def expected_accuracy(
            matrix: tuple[tuple[bool, bool], ...],
            states: tuple[float, float] = state_probabilities,
            actions: tuple[float, float] = action_probabilities,
        ) -> float:
            return sum(
                state_probability * action_probability * float(matrix[state][action])
                for state, state_probability in enumerate(states)
                for action, action_probability in enumerate(actions)
            )

        assert checkpoint.iid_accuracy == pytest.approx(
            expected_accuracy(result.iid_correctness_matrix), abs=1e-12
        )
        assert checkpoint.ood_accuracy == pytest.approx(
            expected_accuracy(result.ood_correctness_matrix), abs=1e-12
        )


@pytest.mark.neural
def test_joint_training_has_seed_dependent_truthful_and_compensatory_attractors() -> None:
    pytest.importorskip("torch")

    results = tuple(
        run_neural_experiment(_config(profile="spurious", mode="joint", seed=seed))
        for seed in range(6)
    )

    assert {result.equilibrium_mode for result in results} >= {
        "truthful",
        "compensatory",
    }
    assert all(result.iid_accuracy >= 0.75 for result in results)


@pytest.mark.neural
def test_error_permutation_hurts_compensatory_attractor_more_than_truthful_attractor() -> None:
    pytest.importorskip("torch")
    results = tuple(
        run_neural_experiment(_config(profile="spurious", mode="joint", seed=seed))
        for seed in range(6)
    )
    truthful = next(result for result in results if result.equilibrium_mode == "truthful")
    compensatory = next(result for result in results if result.equilibrium_mode == "compensatory")

    truthful_gap = truthful.iid_accuracy - truthful.ood_accuracy
    compensatory_gap = compensatory.iid_accuracy - compensatory.ood_accuracy

    assert compensatory_gap > truthful_gap
    assert compensatory.ood_accuracy < compensatory.iid_accuracy


@pytest.mark.neural
def test_history_persists_the_error_coupling_decomposition() -> None:
    pytest.importorskip("torch")
    result = run_neural_experiment(_config(profile="spurious", mode="joint", seed=0))

    for checkpoint in result.history:
        assert checkpoint.outcome_loss == pytest.approx(
            checkpoint.perception_loss + checkpoint.reasoning_loss + 2.0 * checkpoint.coupling,
            abs=1e-8,
        )


@pytest.mark.neural
def test_batch_cli_exits_nonzero_when_any_phase_c_gate_fails(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    metrics = tmp_path / "metrics.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/train_neural.py"),
            "--profiles",
            "truth_aligned",
            "spurious",
            "--modes",
            "perception_only",
            "reasoning_only",
            "joint",
            "--seeds",
            "0",
            "--steps",
            "1",
            "--output",
            str(metrics),
            "--runs-output",
            str(tmp_path / "runs.csv"),
            "--trajectories-output",
            str(tmp_path / "trajectories.csv"),
            "--figure-output",
            str(tmp_path / "figure.png"),
            "--log-root",
            str(tmp_path / "logs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode != 0
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["gates"]["three_training_modes"]["passed"] is True
    assert payload["gates"]["all_passed"] is False


def test_scalar_neural_cli_rejects_repository_source_as_output(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/neural/all.yaml").read_text(encoding="utf-8")
    )
    forbidden = REPOSITORY_ROOT / "src/compbias/forbidden-output.json"
    config["outputs"]["metrics"] = str(forbidden)
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/train_neural.py"),
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
