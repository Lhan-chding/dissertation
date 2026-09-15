import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def helper():
    path = Path(__file__).parents[2] / "scripts/initialize_modeling_v3_snapshot_git.py"
    spec = importlib.util.spec_from_file_location("snapshot_git_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.initialize_snapshot


def fixture_snapshot(tmp_path):
    root = tmp_path / "checkout"
    source = root / "ssvc_flow/src/legacy.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 3\n")
    (root / "private_data.json").write_text("{}")
    manifest = root / "CODE_SNAPSHOT_MANIFEST.json"
    manifest.write_text(
        json.dumps({"ssvc_flow/src/legacy.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    )
    return root, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_snapshot_supplies_real_git_blobs_without_staging_data(tmp_path):
    root, digest = fixture_snapshot(tmp_path)
    receipt = helper()(root, digest, tmp_path / "receipt.json")
    blob = subprocess.check_output(
        ["git", "show", receipt["snapshot_git_commit"] + ":ssvc_flow/src/legacy.py"], cwd=root
    )
    assert blob == b"value = 3\n"
    tracked = (
        subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=root)
        .decode()
        .splitlines()
    )
    assert tracked == ["CODE_SNAPSHOT_MANIFEST.json", "ssvc_flow/src/legacy.py"]
    assert receipt["commit_scope"] == "NEW_LOCAL_SNAPSHOT_COMMIT; NOT_UPSTREAM_HISTORY"


def test_changed_original_refuses_before_creating_git(tmp_path):
    root, digest = fixture_snapshot(tmp_path)
    (root / "ssvc_flow/src/legacy.py").write_text("value = 4\n")
    with pytest.raises(ValueError, match="original changed"):
        helper()(root, digest, tmp_path / "receipt.json")
    assert not (root / ".git").exists()
