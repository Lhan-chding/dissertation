"""Behavioral tests for the public tabular experiment runners."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import scripts.train_neural as train_neural_script
from compbias.rl.exact_kl import exact_kl_projection
from compbias.rl.tabular_experiments import (
    run_coordination_grid,
    run_scaling_paths,
    run_selection_profiles,
)
from compbias.theory.coordination import CoordinationParams, basin_map
from scripts.train_neural import _load_yaml as load_neural_config
from scripts.train_neural import main as neural_main
from scripts.train_tabular import (
    _load_config,
    _run_coordination,
    _run_scaling,
    _run_selection,
    _severity_direction_correct,
)
from scripts.train_tabular import _output_path as tabular_output_path
from scripts.train_tabular import _run as run_tabular
from scripts.train_visual_neural import _load_config as load_visual_config
from scripts.train_visual_neural import _run as run_visual_neural
from scripts.verify_theory import _load_config as load_theory_config
from scripts.verify_theory import _output_path as theory_output_path
from scripts.verify_theory import _run as run_theory

BASE = np.array([0.45, 0.35, 0.20], dtype=np.float64)
SEVERITY = np.array([0.0, 1.0, 3.0], dtype=np.float64)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "loader",
    [load_theory_config, _load_config, load_neural_config, load_visual_config],
    ids=("theory", "tabular", "neural", "visual-neural"),
)
def test_cpu_experiment_configs_reject_duplicate_yaml_keys_at_any_depth(
    loader, tmp_path: Path
) -> None:
    config = tmp_path / "ambiguous.yaml"
    config.write_text(
        "schema_version: 1\nexperiment: test\ntraining:\n  seed: 1\n  seed: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate YAML key"):
        loader(config)


@pytest.mark.parametrize(
    ("loader", "source"),
    [
        (load_theory_config, REPOSITORY_ROOT / "configs/theory/all.yaml"),
        (_load_config, REPOSITORY_ROOT / "configs/tabular/all.yaml"),
        (load_neural_config, REPOSITORY_ROOT / "configs/neural/all.yaml"),
        (load_visual_config, REPOSITORY_ROOT / "configs/neural/visual_modular.yaml"),
    ],
    ids=("theory", "tabular", "neural", "visual-neural"),
)
def test_cpu_experiment_configs_reject_unknown_top_level_fields(
    loader, source: Path, tmp_path: Path
) -> None:
    config = tmp_path / "unknown.yaml"
    config.write_text(
        source.read_text(encoding="utf-8") + "\nmisspelled_gate: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown fields"):
        loader(config)


@pytest.mark.parametrize(
    ("loader", "source", "section", "field"),
    [
        (
            load_theory_config,
            REPOSITORY_ROOT / "configs/theory/all.yaml",
            "property_tests",
            "misspelled_cases",
        ),
        (
            _load_config,
            REPOSITORY_ROOT / "configs/tabular/all.yaml",
            "selection",
            "misspelled_gate",
        ),
        (
            load_neural_config,
            REPOSITORY_ROOT / "configs/neural/all.yaml",
            "outputs",
            "misspelled_metrics",
        ),
        (
            load_visual_config,
            REPOSITORY_ROOT / "configs/neural/visual_modular.yaml",
            "training",
            "misspelled_steps",
        ),
    ],
    ids=("theory", "tabular", "neural", "visual-neural"),
)
def test_cpu_experiment_configs_reject_unknown_nested_fields(
    loader, source: Path, section: str, field: str, tmp_path: Path
) -> None:
    import yaml

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw[section][field] = True
    config = tmp_path / "unknown-nested.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        loader(config)


@pytest.mark.parametrize("unsafe_identifier", ["=1+1", "+cmd", "-2+3", "@formula", "\tcell"])
def test_tabular_runner_rejects_spreadsheet_formula_error_identifiers(
    unsafe_identifier: str,
) -> None:
    config = deepcopy(_load_config(REPOSITORY_ROOT / "configs/tabular/all.yaml"))
    config["errors"]["ids"][1] = unsafe_identifier

    with pytest.raises(ValueError, match="safe ASCII identifiers"):
        _run_selection(config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("selection", "mirror_descent", "steps"), 100_001),
        (("selection", "reinforce", "num_seeds"), 1_001),
        (("selection", "reinforce", "steps"), 10_001),
        (("selection", "reinforce", "batch_size"), 100_001),
        (("selection", "ppo_like", "epochs_per_batch"), 101),
        (("selection", "grpo_like", "group_size"), 100_001),
        (("selection", "bootstrap", "resamples"), 100_001),
        (("errors", "ids"), [f"error-{index}" for index in range(129)]),
    ],
    ids=(
        "mirror-steps",
        "seed-count",
        "optimizer-steps",
        "batch-size",
        "ppo-epochs",
        "group-size",
        "bootstrap-resamples",
        "error-actions",
    ),
)
def test_tabular_selection_rejects_oversized_workload_controls(
    path: tuple[str, ...], value: int | list[str]
) -> None:
    config = deepcopy(_load_config(REPOSITORY_ROOT / "configs/tabular/all.yaml"))
    target = config
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="at most"):
        _run_selection(config)


def test_tabular_selection_rejects_oversized_combined_workload_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import compbias.rl.reinforce as reinforce_module

    config = deepcopy(_load_config(REPOSITORY_ROOT / "configs/tabular/all.yaml"))
    for algorithm, batch_field in (
        ("reinforce", "batch_size"),
        ("ppo_like", "batch_size"),
        ("grpo_like", "group_size"),
    ):
        config["selection"][algorithm]["steps"] = 10_000
        config["selection"][algorithm][batch_field] = 1_000
    monkeypatch.setattr(
        reinforce_module,
        "train_reinforce",
        lambda *_args, **_kwargs: pytest.fail("training started before workload rejection"),
    )

    with pytest.raises(ValueError, match="workload budget"):
        _run_selection(config)


def test_tabular_selection_rejects_oversized_bootstrap_matrix() -> None:
    config = deepcopy(_load_config(REPOSITORY_ROOT / "configs/tabular/all.yaml"))
    for algorithm, batch_field in (
        ("reinforce", "batch_size"),
        ("ppo_like", "batch_size"),
        ("grpo_like", "group_size"),
    ):
        config["selection"][algorithm]["num_seeds"] = 1_000
        config["selection"][algorithm]["steps"] = 1
        config["selection"][algorithm][batch_field] = 1
    config["selection"]["ppo_like"]["epochs_per_batch"] = 1
    config["selection"]["bootstrap"]["resamples"] = 10_001

    with pytest.raises(ValueError, match="bootstrap matrix budget"):
        _run_selection(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grid", 101),
        ("num_seeds", 10_001),
        ("bifurcation_points", 101),
        ("horizon", 10_001.0),
    ],
)
def test_tabular_coordination_rejects_oversized_workload_controls(
    field: str, value: int | float
) -> None:
    config = deepcopy(_load_config(REPOSITORY_ROOT / "configs/tabular/all.yaml"))
    if field == "grid":
        config["coordination"]["grid"]["count"] = value
    elif field == "num_seeds":
        config["coordination"]["seeded_initializations"]["num_seeds"] = value
    elif field == "bifurcation_points":
        config["coordination"]["bifurcation"]["beta_over_a"] = [
            (index + 1) / (value + 1) for index in range(int(value))
        ]
    else:
        config["coordination"]["horizon"] = value

    with pytest.raises(ValueError, match="at most"):
        _run_coordination(config)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("property_tests", "cases", 100_001),
        ("coordination", "horizon", 1_001.0),
        ("coordination.basin_grid", "count", 257),
        ("bifurcation.plot_beta_over_a", "count", 10_001),
    ),
)
def test_phase_a_rejects_oversized_workload_controls(
    section: str, field: str, value: int | float
) -> None:
    config = deepcopy(load_theory_config(REPOSITORY_ROOT / "configs/theory/all.yaml"))
    target = config
    for component in section.split("."):
        target = target[component]
    target[field] = value

    if section == "property_tests":
        with pytest.raises(ValueError, match="at most"):
            from scripts.verify_theory import _positive_int as theory_positive_int

            theory_positive_int(value, "property_tests.cases", minimum=1, maximum=100_000)
    elif section == "coordination.basin_grid":
        from scripts.verify_theory import _run_coordination_checks

        with pytest.raises(ValueError, match="at most"):
            _run_coordination_checks(config["coordination"])
    elif section == "bifurcation.plot_beta_over_a":
        from scripts.verify_theory import _run_bifurcation_checks

        with pytest.raises(ValueError, match="at most"):
            _run_bifurcation_checks(config["bifurcation"])
    else:
        from scripts.verify_theory import _run_coordination_checks

        with pytest.raises(ValueError, match="at most"):
            _run_coordination_checks(target)


def test_formal_phase_b_runner_executes_all_registered_approximate_optimizers() -> None:
    config = _load_config(REPOSITORY_ROOT / "configs/tabular/all.yaml")

    metrics, rows, _plots, _identifiers = _run_selection(config)

    assert metrics["passed"] is True
    assert metrics["gates"]["approximate_algorithms"] == [
        "reinforce",
        "ppo_like",
        "grpo_like",
    ]
    assert metrics["gates"]["formal_reward_mode"] == "raw_fixed_reasoner_outcome"
    assert metrics["gates"]["diagnostic_reward_modes"] == [
        "unconstrained_joint_trajectory_diagnostic",
        "collapsed_effective_reward_diagnostic",
    ]
    assert metrics["gates"]["raw_fixed_reasoner_outcome_kl_gate"] == (
        "finite_only_moment_target_deviation_reported"
    )
    for profile_name, profile in metrics["profiles"].items():
        natural = profile["natural_policy_gradient"]
        assert natural["step_size"] == profile["mirror_descent"]["step_size"]
        assert natural["steps"] == profile["mirror_descent"]["steps"]
        assert natural["trajectory_max_abs_difference_to_mirror"] <= 1e-12
        assert natural["endpoint_l1_difference_to_mirror"] <= 1e-12
        assert natural["passed"] is True
        assert profile["formal_passed"] is True
        assert profile["collapsed_diagnostic_passed"] is True
        for algorithm in ("reinforce", "ppo_like", "grpo_like"):
            raw = profile["raw_fixed_reasoner_outcome"][algorithm]
            joint_diagnostic = profile["unconstrained_joint_trajectory_diagnostic"][algorithm]
            diagnostic = profile["collapsed_effective_reward_diagnostic"][algorithm]
            assert raw["reward_mode"] == "raw_fixed_reasoner_outcome"
            assert raw["training_scope"] == "error_policy_fixed_reasoner_outcomes"
            assert raw["num_seeds"] >= 20
            assert raw["sign_accuracy"] >= 0.90
            assert np.isfinite(raw["mean_kl_to_theory"])
            assert np.isfinite(raw["mean_l1_to_moment_target"])
            assert raw["mean_joint_kl_to_theory"] is None
            assert raw["policy_action_count"] == len(config["errors"]["ids"])
            assert raw["reasoner_conditional_frozen"] is True
            assert raw["fixed_reasoner_conditional_passed"] is True
            assert raw["max_abs_empirical_outcome_conditional_error"] <= 0.03
            assert np.isfinite(raw["mean_odds_slope"])
            assert len(raw["bootstrap_95_ci"]) == 2
            assert np.all(np.isfinite(raw["bootstrap_95_ci"]))
            assert raw["target_kl_from_reference"] >= 0.0
            assert raw["sensitivity_learning_rates"] == [0.10, 0.30]
            assert raw["seed_reproducible"] is True
            assert raw["all_metrics_finite"] is True
            assert joint_diagnostic["reward_mode"] == ("unconstrained_joint_trajectory_diagnostic")
            assert joint_diagnostic["training_scope"] == (
                "unconstrained_joint_trajectory_diagnostic"
            )
            assert joint_diagnostic["formal_gate_eligible"] is False
            assert joint_diagnostic["passed"] is None
            assert isinstance(joint_diagnostic["diagnostic_completed"], bool)
            assert np.isfinite(joint_diagnostic["mean_joint_kl_to_theory"])
            assert diagnostic["reward_mode"] == "collapsed_effective_reward_diagnostic"
            assert diagnostic["training_scope"] == "collapsed_marginal_diagnostic"
            assert diagnostic["mean_joint_kl_to_theory"] is None
            assert diagnostic["num_seeds"] >= 20
            assert diagnostic["mean_kl_to_theory"] <= (diagnostic["initial_kl_to_theory"] + 1e-12)
            assert diagnostic["sensitivity_learning_rates"] == [0.10, 0.30]
            if profile_name == "flat":
                assert raw["mean_final_policy_l1_per_learning_rate"] > 0.0
                assert raw["sensitivity_expected_nonzero"] is True
                assert diagnostic["mean_final_policy_l1_per_learning_rate"] == pytest.approx(0.0)
                assert diagnostic["sensitivity_expected_nonzero"] is False
            else:
                assert raw["mean_final_policy_l1_per_learning_rate"] > 0.0
                assert raw["mean_severity_shift_per_learning_rate"] > 0.0
                assert raw["sensitivity_expected_nonzero"] is True
            assert raw["passed"] is True
            assert diagnostic["passed"] is True
            gap = profile["approximation_gap"][algorithm]
            assert gap["fixed_reasoner_minus_collapsed_mean_kl_to_theory"] == pytest.approx(
                raw["mean_kl_to_theory"] - diagnostic["mean_kl_to_theory"]
            )
    assert {row["algorithm"] for row in rows} >= {
        "exact_kl",
        "mirror_descent",
        "natural_policy_gradient",
        "reinforce",
        "ppo_like",
        "grpo_like",
    }
    assert {row["reward_mode"] for row in rows} == {
        "theory_oracle",
        "raw_fixed_reasoner_outcome",
        "unconstrained_joint_trajectory_diagnostic",
        "collapsed_effective_reward_diagnostic",
    }


def test_formal_phase_b_scaling_runner_registers_matched_gain_opposite_directions() -> None:
    config = _load_config(REPOSITORY_ROOT / "configs/tabular/all.yaml")

    metrics, rows = _run_scaling(config)

    assert metrics["passed"] is True
    assert metrics["path_names"] == ["truth_gain", "uniform_gain", "error_gain"]
    assert metrics["maximum_average_gain_difference"] <= 1e-12
    assert metrics["maximum_derivative_finite_difference_error"] <= 1e-6
    assert metrics["paths"]["truth_gain"]["covariance_derivative"] < 0.0
    assert metrics["paths"]["uniform_gain"]["covariance_derivative"] == pytest.approx(
        0.0, abs=1e-14
    )
    assert metrics["paths"]["error_gain"]["covariance_derivative"] > 0.0
    assert len(rows) == 3 * len(config["errors"]["ids"])


def test_severity_sign_gate_cannot_be_waived_by_reward_improvement() -> None:
    """Reward can improve while the registered deviation moves the wrong way."""

    reference = np.array([0.5, 0.5])
    observed = np.array([0.8, 0.2])
    rewards = np.array([1.0, 0.0])
    severity = np.array([0.0, 1.0])

    assert float(observed @ rewards) > float(reference @ rewards)
    assert (
        _severity_direction_correct(
            float(observed @ severity - reference @ severity),
            predicted_shift=0.2,
            flat_tolerance=0.03,
        )
        is False
    )


@pytest.mark.parametrize(
    ("resolver", "name", "value"),
    [
        (tabular_output_path, "outputs.selection_metrics", "src/compbias/result.json"),
        (theory_output_path, "outputs.report", "README.md"),
    ],
)
def test_cpu_experiment_outputs_cannot_overwrite_repository_sources(
    resolver, name: str, value: str
) -> None:
    with pytest.raises(ValueError, match="artifacts"):
        resolver(REPOSITORY_ROOT, value, name)


def _owned_output_fixture(tmp_path: Path):
    from compbias.io.artifact_paths import prepare_artifact_ownership

    targets = {
        "metrics": tmp_path / "metrics.json",
        "predictions": tmp_path / "predictions.csv",
        "figure": tmp_path / "figure.png",
    }
    ownership = prepare_artifact_ownership(
        targets,
        repository_root=REPOSITORY_ROOT,
        tool="scripts/example_runner.py",
        experiment="example_experiment",
        config_sha256="a" * 64,
        primary_json="metrics",
        primary_schema_version=1,
        primary_experiment="example_experiment",
        overwrite=False,
    )
    targets["metrics"].write_text(
        json.dumps({"schema_version": 1, "experiment": "example_experiment"}),
        encoding="utf-8",
    )
    targets["predictions"].write_text("value\n1\n", encoding="utf-8")
    targets["figure"].write_bytes(b"png fixture")
    return targets, ownership


def test_artifact_ownership_allows_first_run_and_same_config_rerun(tmp_path: Path) -> None:
    from compbias.io.artifact_paths import (
        finalize_artifact_ownership,
        prepare_artifact_ownership,
    )

    targets, ownership = _owned_output_fixture(tmp_path)
    finalize_artifact_ownership(ownership)

    assert ownership.marker_path.is_file()
    marker = json.loads(ownership.marker_path.read_text(encoding="utf-8"))
    assert marker["tool"] == "scripts/example_runner.py"
    assert marker["experiment"] == "example_experiment"
    assert marker["config_sha256"] == "a" * 64
    assert {target["name"] for target in marker["targets"]} == set(targets)
    assert all(len(target["sha256"]) == 64 for target in marker["targets"])

    repeated = prepare_artifact_ownership(
        targets,
        repository_root=REPOSITORY_ROOT,
        tool="scripts/example_runner.py",
        experiment="example_experiment",
        config_sha256="a" * 64,
        primary_json="metrics",
        primary_schema_version=1,
        primary_experiment="example_experiment",
        overwrite=True,
    )
    assert repeated.marker_path == ownership.marker_path


@pytest.mark.parametrize("existing", (False, True), ids=("first-run", "overwrite"))
@pytest.mark.parametrize("failure_index", (1, 2, 4), ids=("first", "middle", "marker"))
def test_artifact_transaction_rolls_back_every_byte_when_nth_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    failure_index: int,
) -> None:
    import compbias.io.artifact_paths as artifact_paths

    targets = {
        "metrics": tmp_path / "metrics.json",
        "predictions": tmp_path / "predictions.csv",
        "figure": tmp_path / "figure.png",
    }

    def prepare(*, overwrite: bool):
        return artifact_paths.prepare_artifact_ownership(
            targets,
            repository_root=REPOSITORY_ROOT,
            tool="scripts/example_runner.py",
            experiment="example_experiment",
            config_sha256="a" * 64,
            primary_json="metrics",
            primary_schema_version=1,
            primary_experiment="example_experiment",
            overwrite=overwrite,
        )

    if existing:
        accepted = prepare(overwrite=False)
        with artifact_paths.artifact_ownership_transaction(accepted) as staged:
            staged["metrics"].write_text(
                json.dumps({"schema_version": 1, "experiment": "example_experiment"}),
                encoding="utf-8",
            )
            staged["predictions"].write_text("old predictions\n", encoding="utf-8")
            staged["figure"].write_bytes(b"old figure")

    ownership = prepare(overwrite=existing)
    old_bytes = {
        path: path.read_bytes()
        for path in (*targets.values(), ownership.marker_path)
        if path.exists()
    }
    promotion_count = 0
    real_promote = artifact_paths._promote_staged_file

    def fail_nth_promotion(source: Path, destination: Path) -> None:
        nonlocal promotion_count
        promotion_count += 1
        if promotion_count == failure_index:
            raise OSError("injected promotion failure")
        real_promote(source, destination)

    monkeypatch.setattr(artifact_paths, "_promote_staged_file", fail_nth_promotion)

    with (
        pytest.raises(OSError, match="injected promotion"),
        artifact_paths.artifact_ownership_transaction(ownership) as staged,
    ):
        staged["metrics"].write_text(
            json.dumps({"schema_version": 1, "experiment": "example_experiment"}),
            encoding="utf-8",
        )
        staged["predictions"].write_text("new predictions\n", encoding="utf-8")
        staged["figure"].write_bytes(b"new figure")

    if existing:
        assert {path: path.read_bytes() for path in old_bytes} == old_bytes
    else:
        assert all(not path.exists() for path in (*targets.values(), ownership.marker_path))
    assert not tuple(tmp_path.glob(".*.transaction-*"))


def test_artifact_transaction_restores_accepted_bundle_when_backup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import compbias.io.artifact_paths as artifact_paths

    targets = {
        "metrics": tmp_path / "metrics.json",
        "predictions": tmp_path / "predictions.csv",
        "figure": tmp_path / "figure.png",
    }
    common = dict(
        repository_root=REPOSITORY_ROOT,
        tool="scripts/example_runner.py",
        experiment="example_experiment",
        config_sha256="a" * 64,
        primary_json="metrics",
        primary_schema_version=1,
        primary_experiment="example_experiment",
    )
    accepted = artifact_paths.prepare_artifact_ownership(targets, overwrite=False, **common)
    with artifact_paths.artifact_ownership_transaction(accepted) as staged:
        staged["metrics"].write_text(
            json.dumps({"schema_version": 1, "experiment": "example_experiment"}),
            encoding="utf-8",
        )
        staged["predictions"].write_text("accepted\n", encoding="utf-8")
        staged["figure"].write_bytes(b"accepted")
    ownership = artifact_paths.prepare_artifact_ownership(targets, overwrite=True, **common)
    accepted_bytes = {
        path: path.read_bytes() for path in (*targets.values(), ownership.marker_path)
    }
    real_backup = artifact_paths._backup_accepted_file
    backup_count = 0

    def fail_third_backup(source: Path, backup: Path) -> None:
        nonlocal backup_count
        backup_count += 1
        if backup_count == 3:
            raise OSError("injected backup failure")
        real_backup(source, backup)

    monkeypatch.setattr(artifact_paths, "_backup_accepted_file", fail_third_backup)
    with (
        pytest.raises(OSError, match="injected backup"),
        artifact_paths.artifact_ownership_transaction(ownership) as staged,
    ):
        staged["metrics"].write_text(
            json.dumps({"schema_version": 1, "experiment": "example_experiment"}),
            encoding="utf-8",
        )
        staged["predictions"].write_text("new\n", encoding="utf-8")
        staged["figure"].write_bytes(b"new")

    assert {path: path.read_bytes() for path in accepted_bytes} == accepted_bytes
    assert not tuple(tmp_path.glob(".*.transaction-*"))


@pytest.mark.parametrize("existing", (False, True), ids=("first-run", "overwrite"))
def test_artifact_transaction_discards_staging_when_writer_fails(
    tmp_path: Path,
    existing: bool,
) -> None:
    import compbias.io.artifact_paths as artifact_paths

    targets = {"metrics": tmp_path / "metrics.json", "figure": tmp_path / "figure.png"}
    common = dict(
        repository_root=REPOSITORY_ROOT,
        tool="scripts/example_runner.py",
        experiment="example_experiment",
        config_sha256="a" * 64,
        primary_json="metrics",
        primary_schema_version=1,
        primary_experiment="example_experiment",
    )
    if existing:
        accepted = artifact_paths.prepare_artifact_ownership(targets, overwrite=False, **common)
        with artifact_paths.artifact_ownership_transaction(accepted) as staged:
            staged["metrics"].write_text(
                json.dumps({"schema_version": 1, "experiment": "example_experiment"}),
                encoding="utf-8",
            )
            staged["figure"].write_bytes(b"accepted")
    ownership = artifact_paths.prepare_artifact_ownership(targets, overwrite=existing, **common)
    accepted_bytes = {
        path: path.read_bytes()
        for path in (*targets.values(), ownership.marker_path)
        if path.exists()
    }

    with (
        pytest.raises(OSError, match="injected writer"),
        artifact_paths.artifact_ownership_transaction(ownership) as staged,
    ):
        staged["metrics"].write_text("partial", encoding="utf-8")
        raise OSError("injected writer failure")

    assert {path: path.read_bytes() for path in accepted_bytes} == accepted_bytes
    if not existing:
        assert all(not path.exists() for path in (*targets.values(), ownership.marker_path))
    assert not tuple(tmp_path.glob(".*.transaction-*"))


@pytest.mark.parametrize("existing", (False, True), ids=("first-run", "overwrite"))
def test_artifact_transaction_rolls_back_when_post_promotion_evidence_fails(
    tmp_path: Path,
    existing: bool,
) -> None:
    import compbias.io.artifact_paths as artifact_paths

    targets = {"metrics": tmp_path / "metrics.json", "figure": tmp_path / "figure.png"}
    common = dict(
        repository_root=REPOSITORY_ROOT,
        tool="scripts/example_runner.py",
        experiment="example_experiment",
        config_sha256="a" * 64,
        primary_json="metrics",
        primary_schema_version=1,
        primary_experiment="example_experiment",
    )
    if existing:
        accepted = artifact_paths.prepare_artifact_ownership(targets, overwrite=False, **common)
        with artifact_paths.artifact_ownership_transaction(accepted) as staged:
            staged["metrics"].write_text(
                json.dumps({"schema_version": 1, "experiment": "example_experiment"}),
                encoding="utf-8",
            )
            staged["figure"].write_bytes(b"accepted")
    ownership = artifact_paths.prepare_artifact_ownership(targets, overwrite=existing, **common)
    accepted_bytes = {
        path: path.read_bytes()
        for path in (*targets.values(), ownership.marker_path)
        if path.exists()
    }

    def fail_evidence_commit() -> None:
        raise OSError("injected evidence failure")

    with (
        pytest.raises(OSError, match="injected evidence"),
        artifact_paths.artifact_ownership_transaction(
            ownership,
            after_promote=fail_evidence_commit,
        ) as staged,
    ):
        staged["metrics"].write_text(
            json.dumps({"schema_version": 1, "experiment": "example_experiment"}),
            encoding="utf-8",
        )
        staged["figure"].write_bytes(b"new")

    assert {path: path.read_bytes() for path in accepted_bytes} == accepted_bytes
    if not existing:
        assert all(not path.exists() for path in (*targets.values(), ownership.marker_path))
    assert not tuple(tmp_path.glob(".*.transaction-*"))


@pytest.mark.parametrize("payload", (b'{"value": NaN}', b'{"value": 1e999}'))
def test_artifact_ownership_reader_rejects_non_finite_json_numbers(
    tmp_path: Path,
    payload: bytes,
) -> None:
    from compbias.io.artifact_paths import _strict_json_mapping

    path = tmp_path / "ownership.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="non-finite"):
        _strict_json_mapping(path, "artifact ownership marker")


def test_artifact_ownership_reader_enforces_size_depth_and_node_limits(tmp_path: Path) -> None:
    from compbias.io.artifact_paths import _strict_json_mapping

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="16 MiB"):
        _strict_json_mapping(oversized, "artifact ownership marker")

    too_deep = tmp_path / "deep.json"
    too_deep.write_text('{"x":' * 65 + "{}" + "}" * 65, encoding="utf-8")
    with pytest.raises(ValueError, match=r"depth|complexity"):
        _strict_json_mapping(too_deep, "artifact ownership marker")

    too_many = tmp_path / "nodes.json"
    too_many.write_text(json.dumps({"x": [0] * 100_001}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"depth|complexity"):
        _strict_json_mapping(too_many, "artifact ownership marker")


@pytest.mark.parametrize(
    "value",
    ("../escape", "line\nbreak", "api_token_backup", "client-secret"),
)
def test_experiment_names_reject_paths_controls_and_secret_like_values(value: str) -> None:
    from compbias.io.artifact_paths import validate_experiment_name

    with pytest.raises(ValueError, match="experiment"):
        validate_experiment_name(value)


@pytest.mark.parametrize(
    ("loader", "source"),
    (
        (load_theory_config, REPOSITORY_ROOT / "configs/theory/all.yaml"),
        (_load_config, REPOSITORY_ROOT / "configs/tabular/all.yaml"),
    ),
    ids=("theory", "tabular"),
)
def test_formal_cpu_yaml_loaders_enforce_safe_experiment_names(
    loader,
    source: Path,
    tmp_path: Path,
) -> None:
    import yaml

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["experiment"] = "../credential-token"
    config = tmp_path / source.name
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment"):
        loader(config)


def test_scalar_neural_formal_config_enforces_safe_experiment_name(tmp_path: Path) -> None:
    import yaml

    payload = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/neural/all.yaml").read_text(encoding="utf-8")
    )
    payload["experiment"] = "api-secret-run"
    config = tmp_path / "neural.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        neural_main(["--config", str(config)])

    assert error.value.code == 2


def test_artifact_ownership_marker_name_is_checkout_portable(tmp_path: Path) -> None:
    from compbias.io.artifact_paths import prepare_artifact_ownership

    marker_names = []
    for checkout in (tmp_path / "checkout-a", tmp_path / "checkout-b"):
        targets = {
            "metrics": checkout / "metrics/summary.json",
            "figure": checkout / "figures/result.png",
        }
        ownership = prepare_artifact_ownership(
            targets,
            repository_root=REPOSITORY_ROOT,
            tool="scripts/example_runner.py",
            experiment="example_experiment",
            config_sha256="a" * 64,
            primary_json="metrics",
            primary_schema_version=1,
            primary_experiment="example_experiment",
            overwrite=False,
        )
        marker_names.append(ownership.marker_path.name)

    assert marker_names[0] == marker_names[1]


def test_scalar_neural_ownership_hash_is_checkout_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def settings(root: Path) -> dict[str, object]:
        return {
            "profiles": ("truth_aligned", "spurious"),
            "modes": ("perception_only", "reasoning_only", "joint"),
            "seeds": (0,),
            "steps": 1,
            "learning_rate": 0.8,
            "device": "cpu",
            "joint_profile": "spurious",
            "experiment": "neural_phase_c",
            "metrics_output": root / "artifacts/metrics/neural.json",
            "runs_output": root / "artifacts/predictions/runs.csv",
            "trajectories_output": root / "artifacts/predictions/trajectories.csv",
            "figure_output": root / "artifacts/figures/figure.png",
            "log_root": root / "artifacts/logs",
        }

    first_root = tmp_path / "checkout-a"
    second_root = tmp_path / "checkout-b"
    monkeypatch.setattr(train_neural_script, "REPOSITORY_ROOT", first_root)
    first = train_neural_script._ownership_config_hash(None, settings(first_root))
    monkeypatch.setattr(train_neural_script, "REPOSITORY_ROOT", second_root)
    second = train_neural_script._ownership_config_hash(None, settings(second_root))

    assert first == second


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("config", "config"),
        ("experiment", "experiment"),
        ("tamper", "hash"),
        ("partial", "complete"),
    ],
)
def test_artifact_ownership_rejects_mismatched_or_incomplete_reruns(
    tmp_path: Path, mutation: str, message: str
) -> None:
    from compbias.io.artifact_paths import (
        finalize_artifact_ownership,
        prepare_artifact_ownership,
    )

    targets, ownership = _owned_output_fixture(tmp_path)
    finalize_artifact_ownership(ownership)
    config_hash = "a" * 64
    experiment = "example_experiment"
    if mutation == "config":
        config_hash = "b" * 64
    elif mutation == "experiment":
        experiment = "other_experiment"
    elif mutation == "tamper":
        targets["predictions"].write_text("changed\n", encoding="utf-8")
    elif mutation == "partial":
        targets["figure"].unlink()

    with pytest.raises(FileExistsError, match=message):
        prepare_artifact_ownership(
            targets,
            repository_root=REPOSITORY_ROOT,
            tool="scripts/example_runner.py",
            experiment=experiment,
            config_sha256=config_hash,
            primary_json="metrics",
            primary_schema_version=1,
            primary_experiment=experiment,
            overwrite=True,
        )


def test_artifact_ownership_rejects_legacy_outputs_without_marker(tmp_path: Path) -> None:
    from compbias.io.artifact_paths import prepare_artifact_ownership

    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"schema_version": 1, "experiment": "same"}', encoding="utf-8")

    with pytest.raises(FileExistsError, match="ownership marker"):
        prepare_artifact_ownership(
            {"metrics": metrics},
            repository_root=REPOSITORY_ROOT,
            tool="scripts/example_runner.py",
            experiment="same",
            config_sha256="a" * 64,
            primary_json="metrics",
            primary_schema_version=1,
            primary_experiment="same",
            overwrite=True,
        )


def test_artifact_ownership_rejects_symlinked_target_and_marker(tmp_path: Path) -> None:
    from compbias.io.artifact_paths import (
        finalize_artifact_ownership,
        prepare_artifact_ownership,
    )

    targets, ownership = _owned_output_fixture(tmp_path)
    finalize_artifact_ownership(ownership)
    real_predictions = tmp_path / "real-predictions.csv"
    targets["predictions"].replace(real_predictions)
    os.symlink(real_predictions, targets["predictions"])

    with pytest.raises(ValueError, match="symlink"):
        prepare_artifact_ownership(
            targets,
            repository_root=REPOSITORY_ROOT,
            tool="scripts/example_runner.py",
            experiment="example_experiment",
            config_sha256="a" * 64,
            primary_json="metrics",
            primary_schema_version=1,
            primary_experiment="example_experiment",
            overwrite=True,
        )
    targets["predictions"].unlink()
    real_predictions.replace(targets["predictions"])
    marker_copy = tmp_path / "marker-copy.json"
    ownership.marker_path.replace(marker_copy)
    os.symlink(marker_copy, ownership.marker_path)
    with pytest.raises(ValueError, match="symlink"):
        prepare_artifact_ownership(
            targets,
            repository_root=REPOSITORY_ROOT,
            tool="scripts/example_runner.py",
            experiment="example_experiment",
            config_sha256="a" * 64,
            primary_json="metrics",
            primary_schema_version=1,
            primary_experiment="example_experiment",
            overwrite=True,
        )


def _config_with_external_outputs(source: Path, tmp_path: Path) -> Path:
    import yaml

    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    for name, value in tuple(outputs.items()):
        source_path = Path(value)
        outputs[name] = str(tmp_path / f"{name}{source_path.suffix}")
    config_path = tmp_path / source.name
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _inject_first_artifact_promotion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import compbias.io.artifact_paths as artifact_paths

    def fail_promotion(_source: Path, _destination: Path) -> None:
        raise OSError("injected runner artifact promotion failure")

    monkeypatch.setattr(artifact_paths, "_promote_staged_file", fail_promotion)


def test_scalar_neural_runner_does_not_publish_logs_when_artifact_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    config_path = _config_with_external_outputs(
        REPOSITORY_ROOT / "configs/neural/all.yaml", tmp_path
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    log_root = Path(config["outputs"]["logs"])
    _inject_first_artifact_promotion_failure(monkeypatch)

    with pytest.raises(SystemExit) as error:
        neural_main(["--config", str(config_path)])

    assert error.value.code == 2
    assert not tuple(log_root.rglob("environment.json"))


def test_scalar_neural_captures_provenance_before_opening_artifact_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    import compbias.io.logging as run_logging

    config_path = _config_with_external_outputs(
        REPOSITORY_ROOT / "configs/neural/all.yaml", tmp_path
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["seeds"] = [0]
    config["steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    real_capture_environment = run_logging.capture_environment
    captures = 0

    def capture_before_transaction(**kwargs):
        nonlocal captures
        assert not tuple(tmp_path.rglob(".*.transaction-*"))
        captures += 1
        environment = real_capture_environment(**kwargs)
        return {**environment, "git_dirty": False}

    monkeypatch.setattr(run_logging, "capture_environment", capture_before_transaction)

    assert neural_main(["--config", str(config_path)]) == 1
    assert captures == 5


@pytest.mark.neural
def test_visual_neural_runner_does_not_publish_logs_when_artifact_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    import yaml

    config_path = _config_with_external_outputs(
        REPOSITORY_ROOT / "configs/neural/visual_modular.yaml", tmp_path
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    log_root = Path(config["outputs"]["logs"])
    _inject_first_artifact_promotion_failure(monkeypatch)

    with pytest.raises(OSError, match="injected runner artifact promotion"):
        run_visual_neural(
            config_path,
            load_visual_config(config_path),
            ["python", "scripts/train_visual_neural.py", "--config", str(config_path)],
        )

    assert not tuple(log_root.rglob("environment.json"))


@pytest.mark.parametrize(
    ("source", "runner"),
    [
        (
            REPOSITORY_ROOT / "configs/theory/all.yaml",
            lambda path: run_theory(
                path, REPOSITORY_ROOT, "2026-08-14T00:00:00+00:00", overwrite=True
            ),
        ),
        (
            REPOSITORY_ROOT / "configs/tabular/all.yaml",
            lambda path: run_tabular(
                path, REPOSITORY_ROOT, "2026-08-14T00:00:00+00:00", overwrite=True
            ),
        ),
        (
            REPOSITORY_ROOT / "configs/neural/visual_modular.yaml",
            lambda path: run_visual_neural(
                path,
                load_visual_config(path),
                ["python", "scripts/train_visual_neural.py", "--config", str(path)],
                overwrite=True,
            ),
        ),
    ],
    ids=("theory", "tabular", "visual-neural"),
)
def test_phase_runners_reject_overwrite_of_complete_legacy_target_set(
    tmp_path: Path, source: Path, runner
) -> None:
    import yaml

    config_path = _config_with_external_outputs(source, tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for name, value in config["outputs"].items():
        if name != "logs":
            Path(value).write_bytes(b"legacy artifact")

    with pytest.raises(FileExistsError, match="ownership marker"):
        runner(config_path)


def test_scalar_neural_runner_rejects_overwrite_of_complete_legacy_target_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    config_path = _config_with_external_outputs(
        REPOSITORY_ROOT / "configs/neural/all.yaml", tmp_path
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for name, value in config["outputs"].items():
        if name != "logs":
            Path(value).write_bytes(b"legacy artifact")

    with pytest.raises(SystemExit) as error:
        neural_main(["--config", str(config_path), "--overwrite"])

    assert error.value.code == 2
    assert "ownership marker" in capsys.readouterr().err


def test_selection_profile_runner_matches_exact_projection_and_freezes_results() -> None:
    profiles = {
        "truth_aligned": np.array([0.9, 0.6, 0.2]),
        "spurious": np.array([0.2, 0.6, 0.9]),
    }
    original_profiles = {name: values.copy() for name, values in profiles.items()}

    results = run_selection_profiles(
        BASE,
        SEVERITY,
        profiles,
        beta=0.5,
        step_size=0.5,
        steps=80,
        seed=7,
    )
    for name, original in original_profiles.items():
        np.testing.assert_array_equal(profiles[name], original)
    profiles["truth_aligned"][0] = 0.1

    assert tuple(result.name for result in results) == ("truth_aligned", "spurious")
    assert results[0].severity_shift < 0.0 < results[1].severity_shift
    for result in results:
        assert result.l1_error < 1e-8
        assert result.pairwise_odds_residual < 1e-8
        assert not result.predicted.flags.writeable
        assert not result.observed.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            result.predicted[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        results[0].name = "changed"  # type: ignore[misc]


def test_selection_profile_runner_accepts_rng_and_zero_support() -> None:
    base = np.array([0.5, 0.5, 0.0])

    (result,) = run_selection_profiles(
        base,
        np.array([0.0, 1.0, 10.0]),
        {"supported": np.array([0.8, 0.3, 0.9])},
        beta=0.7,
        steps=60,
        rng=np.random.default_rng(9),
    )

    assert result.predicted[2] == 0.0
    assert result.observed[2] == 0.0
    assert result.pairwise_odds_residual < 1e-8


def test_selection_profile_runner_reports_observed_residual_before_convergence() -> None:
    (result,) = run_selection_profiles(
        BASE,
        SEVERITY,
        {"spurious": np.array([0.2, 0.6, 0.9])},
        beta=0.5,
        step_size=0.01,
        steps=1,
        seed=7,
    )

    assert result.l1_error > 0.3
    assert result.pairwise_odds_residual > 0.5


@pytest.mark.parametrize("profiles", [{}, (), []])
def test_selection_profile_runner_requires_a_named_profile_mapping(profiles: object) -> None:
    with pytest.raises(ValueError, match="profiles must be a non-empty mapping"):
        run_selection_profiles(
            BASE,
            SEVERITY,
            profiles,  # type: ignore[arg-type]
            beta=0.5,
            steps=2,
            seed=0,
        )


@pytest.mark.parametrize("name", ["", None, 1])
def test_selection_profile_runner_rejects_invalid_profile_names(name: object) -> None:
    with pytest.raises(ValueError, match="profile names must be non-empty strings"):
        run_selection_profiles(
            BASE,
            SEVERITY,
            {name: np.array([0.7, 0.5, 0.3])},  # type: ignore[dict-item]
            beta=0.5,
            steps=2,
            seed=0,
        )


def test_selection_profile_runner_rejects_profile_shape_and_missing_randomness() -> None:
    with pytest.raises(ValueError, match=r"profile 'short' must have shape \(3,\)"):
        run_selection_profiles(
            BASE,
            SEVERITY,
            {"short": np.array([0.7, 0.3])},
            beta=0.5,
            steps=2,
            seed=0,
        )

    with pytest.raises(ValueError, match="provide exactly one randomness source"):
        run_selection_profiles(
            BASE,
            SEVERITY,
            {"valid": np.array([0.7, 0.5, 0.3])},
            beta=0.5,
            steps=2,
        )


def test_scaling_path_runner_matches_projection_and_covariance_directions() -> None:
    gains = {
        "truth_gain": np.array([2.0, 2.0 / 7.0, 0.0]),
        "uniform_gain": np.ones(3),
        "error_gain": np.array([0.0, 0.0, 5.0]),
    }
    original_gains = {name: values.copy() for name, values in gains.items()}

    results = run_scaling_paths(BASE, SEVERITY, gains, kappa=0.2, beta=0.8)
    gains["truth_gain"][0] = 99.0

    assert tuple(result.name for result in results) == (
        "truth_gain",
        "uniform_gain",
        "error_gain",
    )
    assert results[0].severity_shift < 0.0
    assert results[1].severity_shift == pytest.approx(0.0, abs=1e-14)
    assert results[2].severity_shift > 0.0
    assert results[0].covariance_derivative < 0.0
    assert results[1].covariance_derivative == pytest.approx(0.0, abs=1e-14)
    assert results[2].covariance_derivative > 0.0
    for result, original_gain in zip(results, original_gains.values(), strict=True):
        expected = exact_kl_projection(BASE, 0.2 * original_gain, beta=0.8)
        np.testing.assert_allclose(result.selected, expected, atol=1e-14, rtol=0.0)
        assert not result.gain.flags.writeable
        assert not result.selected.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            result.selected[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        results[0].name = "changed"  # type: ignore[misc]


def test_scaling_path_zero_step_leaves_reference_unchanged() -> None:
    (result,) = run_scaling_paths(
        BASE,
        SEVERITY,
        {"no_step": np.array([2.0, 1.0, 0.0])},
        kappa=0.0,
    )

    np.testing.assert_allclose(result.selected, BASE, atol=1e-14, rtol=0.0)
    assert result.severity_shift == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    ("kappa", "exception"),
    [
        (True, TypeError),
        ("bad", TypeError),
        (-0.1, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
    ],
)
def test_scaling_path_runner_rejects_invalid_kappa(
    kappa: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception, match="kappa must be"):
        run_scaling_paths(
            BASE,
            SEVERITY,
            {"gain": np.ones(3)},
            kappa=kappa,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("gains", [{}, (), []])
def test_scaling_path_runner_requires_a_named_gain_mapping(gains: object) -> None:
    with pytest.raises(ValueError, match="gains must be a non-empty mapping"):
        run_scaling_paths(
            BASE,
            SEVERITY,
            gains,  # type: ignore[arg-type]
            kappa=0.2,
        )


@pytest.mark.parametrize("name", ["", None, 1])
def test_scaling_path_runner_rejects_invalid_gain_names(name: object) -> None:
    with pytest.raises(ValueError, match="gain names must be non-empty strings"):
        run_scaling_paths(
            BASE,
            SEVERITY,
            {name: np.ones(3)},  # type: ignore[dict-item]
            kappa=0.2,
        )


def test_scaling_path_runner_rejects_gain_shape() -> None:
    with pytest.raises(ValueError, match=r"gain 'short' must have shape \(3,\)"):
        run_scaling_paths(
            BASE,
            SEVERITY,
            {"short": np.ones(2)},
            kappa=0.2,
        )


def test_coordination_grid_runner_matches_public_basin_map_and_is_immutable() -> None:
    p_values = np.array([0.2, 0.8])
    q_values = np.array([0.2, 0.8])
    params = CoordinationParams(delta=1.0, epsilon=1.0)

    actual = run_coordination_grid(
        p_values,
        q_values,
        params,
        horizon=20.0,
        separatrix_tolerance=1e-10,
    )
    expected = basin_map(
        p_values,
        q_values,
        params,
        horizon=20.0,
        separatrix_tolerance=1e-10,
    )

    np.testing.assert_array_equal(actual.p, expected.p)
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.labels, expected.labels)
    assert not actual.labels.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        actual.labels[0, 0] = "changed"


def test_coordination_grid_runner_preserves_input_validation() -> None:
    with pytest.raises(ValueError, match="p_values probabilities"):
        run_coordination_grid(
            np.array([-0.1]),
            np.array([0.5]),
            CoordinationParams(delta=1.0, epsilon=1.0),
        )
