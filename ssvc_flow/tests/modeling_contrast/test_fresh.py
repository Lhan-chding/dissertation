"""Gate refusal tests never invoke training or generate new trajectories."""

import json
from unittest.mock import Mock

import pytest

from src.modeling_contrast.collect_fresh import (
    build_fresh_parent_config,
    run_collect_fresh,
    scan_seed_collisions,
    validate_fresh_gate,
)
from src.modeling_contrast.io import RunWriter, canonical_hash, sha256_file
from src.modeling_contrast.protocol import load_config
from src.modeling_contrast.selection import freeze_selection


def frozen_fixture(tmp_path, mutate=None):
    config = load_config()
    source = tmp_path / "source.py"
    source.write_text("# fixture source, no model\n")
    data = tmp_path / "old_data.txt"
    data.write_text("old fixture\n")
    packet = tmp_path / "packet.txt"
    packet.write_text("paid fixture\n")
    candidate = {
        "configuration_id": "finite",
        "model": "C3_CONTRAST_GLS_FULL",
        "method": "O_LR_ORIGIN",
        "access_regime": "SAMPLE_AND_LOGP",
        "n": 64,
        "actual_rank": 1,
        "nonalias_eval_units": 12,
        "predicted_difference_norm": 0.1,
        "pX_nrmse": 0.4,
        "v_nrmse": 0.3,
        "stable_improvement_C0": True,
        "stable_improvement_C1": True,
    }
    cost = {
        "configuration_id": "finite",
        "access_regime": "SAMPLE_AND_LOGP",
        "includes_fit_score_label": True,
        "score_to_generation_ratio": 0.25,
        "within_domain": True,
        "cost_feasible": True,
        "comparison_baseline": "D_DIRECT",
    }
    lock = freeze_selection(
        [candidate],
        protocol=config,
        input_hashes={str(data): sha256_file(data)},
        source_hashes={str(source): sha256_file(source)},
        packet_hashes={str(packet): sha256_file(packet)},
        resource_forecast={
            "total_walltime_hours": 1,
            "peak_ram_gib": 1,
            "added_output_gib": 0.1,
            "temporary_gib": 0.1,
        },
        data_gate={"status": "PASS", "legacy_complete": True, "fresh_seed_collision": False},
        cost_frontier=[cost],
    )
    if mutate:
        mutate(lock)
        lock["lock_sha256"] = canonical_hash(
            {key: value for key, value in lock.items() if key != "lock_sha256"}
        )
    out = tmp_path / "runs/modeling_contrast_v2/N3"
    with RunWriter(out, lock["binding"], project_root=tmp_path) as writer:
        writer.write_json("MODEL_SELECTION_LOCK.json", lock)
    kwargs = dict(
        existing_run_roots=[tmp_path / "runs"],
        project_root=tmp_path,
        resource_forecast=dict(
            wall_seconds=3600, peak_ram_gib=1, added_output_bytes=100, temporary_bytes=100
        ),
    )
    return config, out / "MODEL_SELECTION_LOCK.json", lock, kwargs


def test_fresh_parent_adapter_preserves_optimizer_and_fixed_budget():
    adapted = build_fresh_parent_config(load_config())
    profile = adapted["profiles"]["contrast_fresh"]
    assert profile["seeds_by_role"] == {
        "fresh_calibration": list(range(501, 507)),
        "fresh_locked_test": list(range(601, 611)),
    }
    assert profile["steps"] == 64 and profile["B"] == 4 and profile["K"] == 8
    assert profile["arms"] == ["X_BASE", "X_VALID"]
    assert adapted["budget_derived"]["total_optimizer_steps"] == 6080
    assert adapted["dataset"]["seed"] == 20260915
    assert adapted["toy_model"]["model_initialization_seed"] == 7001


def test_gate_failure_does_not_call_collector_or_create_output(tmp_path):
    config = load_config()
    path = tmp_path / "runs/modeling_contrast_v2/N3"
    binding = {
        key: canonical_hash(key) for key in ("source", "config", "data", "selector", "packet")
    }
    with RunWriter(path, binding, project_root=tmp_path) as writer:
        writer.write_json("MODEL_SELECTION_LOCK.json", {"allow_fresh_cpu": False})
    collector = Mock(side_effect=AssertionError("new CPU training was entered"))
    output = tmp_path / "runs/modeling_contrast_v2/N4"
    with pytest.raises(ValueError, match="fresh gate"):
        run_collect_fresh(
            config,
            path / "MODEL_SELECTION_LOCK.json",
            output,
            binding=binding,
            existing_run_roots=[tmp_path / "runs"],
            resource_forecast={},
            collector=collector,
            project_root=tmp_path,
        )
    collector.assert_not_called()
    assert not output.exists()


def test_missing_lock_and_changed_outputs_are_rejected(tmp_path):
    config = load_config()
    with pytest.raises(ValueError, match="fresh gate"):
        validate_fresh_gate(
            config,
            tmp_path / "missing.json",
            existing_run_roots=[tmp_path],
            resource_forecast={},
            project_root=tmp_path,
        )


def test_existing_training_seeds_are_not_silently_renumbered(tmp_path):
    root = tmp_path / "runs/legacy"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"trajectories": [{"seed": 601, "id": "seed601_X_BASE"}]})
    )
    collisions = scan_seed_collisions([tmp_path / "runs"], set(range(601, 611)))
    assert collisions[0]["seed"] == 601
    assert "manifest.json" in collisions[0]["evidence"]


def test_seed_inventory_rejects_missing_scan_roots(tmp_path):
    with pytest.raises(ValueError, match="inventory"):
        scan_seed_collisions([tmp_path / "missing"], {601})


def test_valid_frozen_gate_can_be_checked_without_training(tmp_path):
    config, path, lock, kwargs = frozen_fixture(tmp_path)
    receipt = validate_fresh_gate(config, path, **kwargs)
    assert receipt["status"] == "PASS"
    assert receipt["binding"] == lock["binding"]


@pytest.mark.parametrize("kind", ["source", "data", "packet", "selector"])
def test_mutated_bound_files_refuse_before_collector(tmp_path, kind):
    config, path, lock, kwargs = frozen_fixture(tmp_path)
    if kind == "selector":
        path.write_text("{}")
    else:
        field = {"source": "source_hashes", "data": "input_hashes", "packet": "packet_hashes"}[kind]
        from pathlib import Path

        Path(next(iter(lock[field]))).write_text("changed")
    collector = Mock(side_effect=AssertionError("new training entered"))
    with pytest.raises(ValueError, match="fresh gate"):
        run_collect_fresh(
            config,
            path,
            tmp_path / "runs/modeling_contrast_v2/N4",
            binding=lock["binding"],
            collector=collector,
            **kwargs,
        )
    collector.assert_not_called()


@pytest.mark.parametrize("kind", ["science_gate", "cost_gate", "resource_gate", "data_gate"])
def test_one_failed_gate_refuses_new_training(tmp_path, kind):
    config, path, lock, kwargs = frozen_fixture(
        tmp_path, lambda lock: lock[kind].update(status="FAIL")
    )
    collector = Mock(side_effect=AssertionError("new training entered"))
    with pytest.raises(ValueError, match="fresh gate"):
        run_collect_fresh(
            config,
            path,
            tmp_path / "runs/modeling_contrast_v2/N4",
            binding=lock["binding"],
            collector=collector,
            **kwargs,
        )
    collector.assert_not_called()


def test_seed_scan_ignores_explicit_test_fixtures_but_checks_actual_runs(tmp_path):
    fixture = tmp_path / "fixtures/example"
    fixture.mkdir(parents=True)
    (fixture / "manifest.json").write_text("{broken fixture JSON")
    (fixture / "seed601_fake.npz").write_bytes(b"fixture")
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "manifest.json").write_text(json.dumps({"trajectories": [{"seed": 602}]}))
    hits = scan_seed_collisions([tmp_path], {601, 602})
    assert len(hits) == 1 and hits[0]["seed"] == 602
