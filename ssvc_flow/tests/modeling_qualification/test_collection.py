import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.modeling_qualification.collection import (
    load_observed,
    load_oracle,
    run_collect,
    validate_collection,
)


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    config = json.loads(Path("configs/modeling_qualification/protocol.json").read_text())
    # A test fixture is explicitly separate from the locked experiment profiles.
    config["profiles"]["unit"] = {
        "seeds": [101],
        "arms": ["X_BASE", "X_VALID"],
        "steps": 2,
        "B": 4,
        "K": 8,
        "anchors": [1],
        "fit_banks": 1,
        "diagnostic_banks": 1,
        "evaluation_banks": 1,
    }
    out = tmp_path_factory.mktemp("collect") / "run"
    result = run_collect(config, "unit", out)
    return out, config, result


def test_collection_counts_cost_and_same_initialization(collected):
    out, _config, result = collected
    assert result["optimizer_steps"] == 22
    assert result["sampled_finite_actions"] == 320
    manifest = json.loads((out / "manifest.json").read_text())
    a, b = [load_oracle(out, x["id"]) for x in manifest["trajectories"]]
    np.testing.assert_array_equal(a.theta[0], b.theta[0])
    for entry in manifest["trajectories"]:
        o = load_observed(out, entry["id"], n=64, noise=0)
        anchor = o.anchor(0)
        assert anchor["p_fit"].shape == (3, 72, 4)
        np.testing.assert_allclose(anchor["p_fit"].sum(-1), 1)
        assert anchor["d_fit"].shape == (3, 737)
        assert anchor["d_eval"].shape == (3, 737)
        assert "p_eval" not in anchor
        assert not hasattr(o, "p") and not hasattr(o, "theta")
        assert not hasattr(o, "root") and not hasattr(o, "oracle_file")
        assert not hasattr(o, "trajectory_counts")
        with pytest.raises(PermissionError):
            o.anchor(0, role="evaluation")
        np.testing.assert_array_equal(anchor["anchor_counts"].sum(-1), 64)


def test_collection_rejects_clobber(collected):
    out, config, _ = collected
    with pytest.raises(FileExistsError):
        run_collect(config, "unit", out)


def test_alias_outputs_share_counts_and_anchor_identity(collected):
    out, _, _ = collected
    manifest = json.loads((out / "manifest.json").read_text())
    entry = manifest["trajectories"][0]
    obs = np.load(out / entry["observations_file"])
    counts = np.load(out / entry["counts_file"])
    oracle = load_oracle(out, entry["id"])
    aliases = obs["branch_alias"].reshape(-1)
    for index, alias in enumerate(aliases):
        if index != alias:
            np.testing.assert_array_equal(
                oracle.branch_p.reshape(-1, 72, 4)[index], oracle.branch_p.reshape(-1, 72, 4)[alias]
            )
            np.testing.assert_array_equal(
                counts["branch_n64"][0].reshape(-1, 72, 4)[index],
                counts["branch_n64"][0].reshape(-1, 72, 4)[alias],
            )


def test_validation_checks_all_shapes_counts_and_cost(collected):
    result = validate_collection(collected[0])
    assert result["status"] == "PASS"
    assert result["main_optimizer_steps"] == 4
    assert result["fork_optimizer_steps"] == 18
    assert result["total_sampled_finite_actions"] == 320


@pytest.mark.parametrize("filename", ["dataset.npz", "parameter_layout.json"])
def test_validation_rejects_tampered_shared_identity(collected, tmp_path, filename):
    copy = tmp_path / "corrupted"
    shutil.copytree(collected[0], copy)
    path = copy / filename
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_collection(copy)
    entry = json.loads((copy / "manifest.json").read_text())["trajectories"][0]
    with pytest.raises(ValueError, match="identity mismatch"):
        load_oracle(copy, entry["id"])


def test_counts_predictor_never_opens_oracle_labels(collected, monkeypatch):
    original = np.load
    opened = []

    def guarded(path, *args, **kwargs):
        if Path(path).parent.name == "oracle":
            raise AssertionError("count predictor attempted to open oracle labels")
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    out = collected[0]
    entry = json.loads((out / "manifest.json").read_text())["trajectories"][0]
    view = load_observed(out, entry["id"], n=64, noise=0)
    assert view.meta["access_level"] == "UPDATE_AWARE"
    assert opened


def test_validation_ignores_trajectory_listing_order_but_preserves_membership(collected, tmp_path):
    copied = tmp_path / "reordered"
    shutil.copytree(collected[0], copied)
    path = copied / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["trajectories"] = manifest["trajectories"][::-1]
    path.write_text(json.dumps(manifest, sort_keys=True))
    assert validate_collection(copied)["status"] == "PASS"
    manifest["trajectories"][0]["split"] = "wrong_global_split"
    path.write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="identities/splits"):
        validate_collection(copied)
