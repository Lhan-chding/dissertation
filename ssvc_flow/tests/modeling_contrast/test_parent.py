"""Readonly historical-input boundaries; these fixtures do not train a model."""

import hashlib
import json

import numpy as np
import pytest

from src.modeling_contrast.parent import (
    BANK_ROLES,
    audit_aliases,
    audit_frozen_predictions,
    audit_parent_collection,
    audit_seed_collisions,
    complete_state_fingerprint,
    load_parent,
    parameter_hash,
    resolve_member,
)


def test_parameter_equality_is_not_norm_equality():
    assert parameter_hash(np.array([1.0, 0.0])) != parameter_hash(np.array([-1.0, 0.0]))
    report = audit_aliases(np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]), [0, 0, 2])
    assert report["alias_count"] == 1
    with pytest.raises(ValueError, match="alias parameters"):
        audit_aliases(np.array([[1.0, 0.0], [-1.0, 0.0]]), [0, 0])


def test_state_hash_needs_full_state_and_never_merges_adam_histories():
    incomplete = complete_state_fingerprint({"parameters": np.ones(2)})
    assert incomplete["complete_state_sha256"] is None
    assert "adam_m" in incomplete["missing_components"]
    state = dict(
        parameters=np.ones(2),
        adam_m=np.zeros(2),
        adam_v=np.zeros(2),
        adam_step=1,
        gradients=np.zeros(2),
        gradient_is_none=[False],
        optimizer_config={"lr": 0.001},
        numpy_rng={"state": 1},
        torch_rng=[1, 2],
    )
    other = dict(state, adam_m=np.ones(2))
    assert (
        complete_state_fingerprint(state)["complete_state_sha256"]
        != complete_state_fingerprint(other)["complete_state_sha256"]
    )
    assert parameter_hash(state["parameters"]) == parameter_hash(other["parameters"])


def test_member_paths_cannot_escape_source(tmp_path):
    assert resolve_member(tmp_path, "observations/a.npz") == tmp_path / "observations/a.npz"
    for path in ("../outside", str(tmp_path / "absolute")):
        with pytest.raises(ValueError, match=r"relative|escape"):
            resolve_member(tmp_path, path)
    (tmp_path / "linked").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escape"):
        resolve_member(tmp_path, "linked/outside")


def test_seed_audit_checks_used_trajectory_seeds_not_planned_lists(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"trajectories": [{"seed": 501, "id": "seed501_X_BASE"}]})
    )
    (root / "protocol.json").write_text(json.dumps({"fresh_locked_test_seeds": [601]}))
    receipt = audit_seed_collisions([tmp_path])
    assert receipt["status"] == "STOP_FOR_REVIEW"
    assert receipt["colliding_seeds"] == [501]
    assert receipt["checked_candidate_seeds"] == list(range(501, 507)) + list(range(601, 611))


def test_seed_audit_excludes_intentionally_corrupt_test_fixtures(tmp_path):
    fixture = tmp_path / "fixtures" / "tampering"
    fixture.mkdir(parents=True)
    (fixture / "manifest.json").write_text("intentional tampering test bytes")
    report = audit_seed_collisions([tmp_path])
    assert report["status"] == "PASS"
    assert report["excluded_test_fixture_manifest_count"] == 1


@pytest.mark.parametrize("relative_root", ["fixtures", "fixtures/preflight_155484"])
def test_seed_audit_does_not_exclude_fixture_named_root_or_ancestor(tmp_path, relative_root):
    root = tmp_path / relative_root
    nested = root / "fixtures/tampering"
    nested.mkdir(parents=True)
    (nested / "manifest.json").write_text("intentional nested fixture corruption")
    (root / "manifest.json").write_text(
        json.dumps({"trajectories": [{"seed": 601, "id": "seed601_X_BASE"}]})
    )
    report = audit_seed_collisions([root])
    assert report["status"] == "STOP_FOR_REVIEW"
    assert report["colliding_seeds"] == [601]
    assert report["manifests_scanned"] == 1
    assert report["excluded_test_fixture_manifest_count"] == 1
    assert report["scan_errors"] == []


def test_old_prediction_audit_reports_missing_and_corrupt_arrays(tmp_path):
    raw, delta = np.ones((2, 4)), np.zeros((2, 4))
    np.savez(tmp_path / "present.npz", raw=raw, delta=delta)
    (tmp_path / "selection.json").write_text("{}")
    row = {
        "record": 0,
        "prediction_file": "present.npz",
        "raw_prediction_hash": parameter_hash(raw),
        "delta_prediction_hash": parameter_hash(delta),
    }
    freeze = {
        "selection_sha256": hashlib.sha256(b"{}").hexdigest(),
        "records": [row, dict(row, record=1, prediction_file="missing.npz")],
    }
    (tmp_path / "prediction_freeze.json").write_text(json.dumps(freeze))
    assert audit_frozen_predictions(tmp_path)["status"] == "MISSING_INPUTS"
    np.savez(tmp_path / "present.npz", raw=raw + 1, delta=delta)
    report = audit_frozen_predictions(tmp_path)
    assert report["status"] == "HASH_MISMATCH"
    assert len(report["mismatches"]) == 1 and len(report["missing_inputs"]) == 1


@pytest.fixture
def parent_fixture(tmp_path):
    root = tmp_path / "parent"
    root.mkdir()
    theta = np.zeros((65, 737))
    branches = np.zeros((3, 14, 3, 737))
    branches[:, :, 1, 0] = 0.1
    branches[:, :, 2, 1] = 0.2
    # Canonical aliases remain in each parameter equivalence class.
    alias = np.tile(np.arange(3), 42).reshape(3, 14, 3)
    observation = root / "observations.npz"
    np.savez(
        observation,
        theta=theta,
        branch_theta=branches,
        branch_d=branches.copy(),
        branch_alias=alias,
        anchors=[8, 24, 40],
    )
    entry = {
        "id": "seed101_X_BASE",
        "seed": 101,
        "arm": "X_BASE",
        "split": "fit",
        "observations_file": observation.name,
        "sha256": {"observations_file": hashlib.sha256(observation.read_bytes()).hexdigest()},
    }
    manifest = dict(
        trajectories=[entry],
        bank_roles=BANK_ROLES,
        operation_names=["joint_0", "joint_1", "no_x_off_1"],
        anchors=[8, 24, 40],
        events=["X", "S", "W", "I"],
        parameter_count=737,
        probe_count=72,
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "probe_metadata.json").write_text("[]")
    return root


def test_bridge_preserves_split_and_bounds_parameter_only_inputs(parent_fixture):
    parent = load_parent(parent_fixture, "seed101_X_BASE")
    assert parent.entry["split"] == "fit"
    assert parent.entry["v2_role"] == "legacy_diagnostic"
    assert not hasattr(parent, "counts") and not hasattr(parent, "oracle")
    fit = parent.anchor_inputs(0)
    evaluation = parent.anchor_inputs(0, "evaluation")
    assert fit["e"].shape == (8, 2, 737)
    assert evaluation["bank_indices"].tolist() == [10, 11, 12, 13]
    assert set(fit["bank_indices"]).isdisjoint(evaluation["bank_indices"])
    assert not fit["e"].flags.writeable
    assert not parent.observations["theta"].flags.writeable
    with pytest.raises(ValueError):
        parent.observations["theta"][0, 0] = 1
    with pytest.raises(ValueError, match="role"):
        parent.anchor_inputs(0, "test")


def test_changed_historical_roles_fail_closed(parent_fixture):
    path = parent_fixture / "manifest.json"
    value = json.loads(path.read_text())
    value["bank_roles"]["fit"] = list(range(9))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="bank"):
        load_parent(parent_fixture, "seed101_X_BASE")


def test_hash_corruption_refused(parent_fixture):
    with (parent_fixture / "observations.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_parent(parent_fixture, "seed101_X_BASE")


def test_missing_historical_inputs_are_listed_and_never_regenerated(parent_fixture):
    before = set(parent_fixture.rglob("*"))
    report = audit_parent_collection(parent_fixture)
    assert report["status"] == "MISSING_INPUTS"
    assert report["complete_trajectories"] == 0
    assert any(row.get("field") == "counts_file" for row in report["missing_inputs"])
    assert report["expected_trajectories"] == 60
    assert set(parent_fixture.rglob("*")) == before
