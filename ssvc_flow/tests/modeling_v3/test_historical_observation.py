import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v3.cpu_campaign import _digest, _hash, _verify_complete, observe_toy
from src.modeling_v3.historical_observation import (
    prepare_historical_observation_collection,
    select_historical_origins,
)


def _write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def original(tmp_path):
    """Synthetic engineering originals; no trajectory training or server work."""
    root = tmp_path / "original"
    root.mkdir()
    rng = np.random.default_rng(133)
    features = rng.normal(0, 0.1, (72, 16, 44))
    categories = np.tile([0, 1, *([2] * 12), 3, 3], (72, 1)).astype(np.int8)
    np.savez(root / "dataset.npz", probe_features=features, probe_categories=categories)
    metadata = [{"prompt_id": f"probe-{i}", "group": i // 12, "weight": 1 / 72} for i in range(72)]
    _write(root / "probe_metadata.json", metadata)
    _write(root / "train_metadata.json", [{"prompt_id": f"train-{i}"} for i in range(72)])
    _write(root / "parameter_layout.json", {"fixture": True})
    source_config = {
        "toy_model": {
            "parameter_count": 737,
            "dtype": "float64",
            "model_initialization_seed": 7001,
        },
        "dataset": {"candidate_actions": 16},
    }
    _write(root / "resolved_config.json", source_config)
    theta = rng.normal(0, 0.05, (9, 737))
    branch = np.broadcast_to(theta[8], (1, 2, 3, 737)).copy()
    branch[:, :, 1, 0] += 0.005
    branch[:, :, 2, 2] += 0.004
    arrays = {"theta": theta, "branch_theta": branch, "anchors": np.array([8])}
    for key in (
        "adam_m",
        "adam_v",
        "adam_step",
        "branch_adam_m",
        "branch_adam_v",
        "branch_adam_step",
        "checkpoint_grad",
        "checkpoint_grad_is_none",
    ):
        arrays[key] = np.zeros(9)
    np.savez(root / "observations.npz", **arrays)
    _write(root / "rng.json", {"fixture_rng_original": rng.bit_generator.state})
    entry = {
        "id": "seed501_X_BASE",
        "seed": 501,
        "arm": "X_BASE",
        "split": "fresh_calibration",
        "observations_file": "observations.npz",
        "rng_file": "rng.json",
        "sha256": {
            "observations_file": _hash(root / "observations.npz"),
            "rng_file": _hash(root / "rng.json"),
        },
    }
    toy = Path(__file__).resolve().parents[2] / "src/modeling_qualification/toy.py"
    manifest = {
        "events": ["X", "S", "W", "I"],
        "parameter_count": 737,
        "probe_count": 72,
        "operation_names": ["joint_0", "joint_1", "no_x_off_1"],
        "anchors": [8],
        "bank_roles": {"fit": [0], "evaluation": [1]},
        "trajectories": [entry],
        "config_sha256": _digest(source_config),
        "source_hashes": {"toy.py": _hash(toy)},
    }
    for name, field in (
        ("dataset.npz", "dataset_sha256"),
        ("probe_metadata.json", "probe_identity_sha256"),
        ("train_metadata.json", "train_identity_sha256"),
        ("parameter_layout.json", "parameter_layout_sha256"),
    ):
        manifest[field] = _hash(root / name)
    _write(root / "manifest.json", manifest)
    return root


def test_historical_selection_is_identity_only_and_balanced():
    manifest = {
        "anchors": [8, 24, 40],
        "bank_roles": {"fit": list(range(8))},
        "trajectories": [
            {
                "id": f"seed{seed}_{arm}",
                "seed": seed,
                "arm": arm,
                "split": "old_public",
                "error": seed,
            }
            for seed in range(101, 107)
            for arm in ("X_BASE", "X_VALID")
        ],
    }
    first = select_historical_origins(manifest, 12)
    manifest["trajectories"].reverse()
    for row in manifest["trajectories"]:
        row["error"] = -10000
    assert select_historical_origins(manifest, 12) == first
    assert all(
        sum(row["arm"] == arm and row["step"] == step for row in first) == 2
        for arm in ("X_BASE", "X_VALID")
        for step in (8, 24, 40)
    )


def test_original_theta_scoring_is_metered_and_consumable_by_observe_toy(original, tmp_path):
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/modeling_v3/protocol.json").read_text()
    )
    config["historical_observation"] = {"max_origins": 1}
    config["observation"]["methods"] = ["RAW4", "PRESERVE_XI"]
    before = {str(path): _hash(path) for path in original.iterdir()}
    out = tmp_path / "prepared"
    result = prepare_historical_observation_collection(original, out, config)
    assert result["source_updates"] == result["fork_updates"] == 0
    assert result["scoring_cost"]["new_batched_forward_calls"] == 3  # origin aliases joint_0
    assert result["scoring_cost"]["exact_parameter_cache_hits"] == 1
    assert result["scoring_cost"]["new_scored_action_values"] == 3 * 72 * 16
    assert result["provenance"] == "REUSED_HISTORICAL_FIXED_POLICIES_NEW_SCORING"
    assert _verify_complete(out)["summary"] == result
    trajectory = result["trajectories"][0]
    unit = out / trajectory["path"]
    identity = json.loads((unit / "identity.json").read_text())
    assert identity["bank_original_ids"] == [[trajectory["original_bank"]]]
    assert identity["original_adam_references"]["adam_m"]["indices"] == [8]
    with np.load(unit / "calibration_action_p.npz", allow_pickle=False) as data:
        assert data["action_p"].shape == (1, 1, 3, 72, 16)
        np.testing.assert_allclose(data["action_p"].sum(-1), 1, atol=1e-14)
    observed = observe_toy(config, out, tmp_path / "observation_pilot", pilot=True)
    assert observed["unit_count"] == 4
    assert all(_hash(path) == digest for path, digest in before.items())


def test_missing_original_never_scores_or_publishes_complete(original, tmp_path, monkeypatch):
    import src.modeling_qualification.toy as toy

    (original / "observations.npz").unlink()
    monkeypatch.setattr(toy, "_numpy_forward", lambda *args: pytest.fail("missing input scored"))
    out = tmp_path / "missing"
    result = prepare_historical_observation_collection(original, out, {})
    assert result["status"] == "MISSING_INPUTS"
    assert result["new_forward_calls"] == 0
    assert not (out / "COMPLETE.json").exists()


def test_tampered_original_refused_before_scoring(original, tmp_path, monkeypatch):
    import src.modeling_qualification.toy as toy

    (original / "rng.json").write_text("{}")
    monkeypatch.setattr(toy, "_numpy_forward", lambda *args: pytest.fail("bad hash scored"))
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_historical_observation_collection(original, tmp_path / "tampered", {})


def test_cannot_publish_inside_historical_originals(original):
    with pytest.raises(ValueError, match="outside"):
        prepare_historical_observation_collection(original, original / "new", {})
