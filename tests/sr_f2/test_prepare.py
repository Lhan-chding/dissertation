import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sr_f2/prepare.py"
SPEC = importlib.util.spec_from_file_location("sr_f2_prepare_tested", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_snapshot_blob_links_allowed_but_unrelated_paths_rejected(tmp_path):
    model = tmp_path / "cache/snapshots/revision"
    blobs = tmp_path / "cache/blobs"
    model.mkdir(parents=True)
    blobs.mkdir()
    payload = blobs / "content-hash"
    payload.write_bytes(b"verified separately by sha256")
    link = model / "model.safetensors"
    link.symlink_to(payload)
    assert MODULE.model_file(model, link.name) == link
    outside = tmp_path / "unrelated"
    outside.write_bytes(b"outside")
    (model / "escape").symlink_to(outside)
    with pytest.raises(PermissionError, match="escapes"):
        MODULE.model_file(model, "escape")
    with pytest.raises(PermissionError, match="Unsafe"):
        MODULE.model_file(model, "../anything")


def test_normal_materialized_model_directory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    assert MODULE.model_file(model, "config.json") == model / "config.json"
